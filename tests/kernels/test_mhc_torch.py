# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.mhc.torch import (
    hc_head_torch,
    mhc_fused_post_pre_torch,
    mhc_post_torch,
    mhc_pre_torch,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mhc_torch_preserves_activation_dtype(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    num_tokens, hc_mult, hidden_size = 3, 4, 16
    hc_mix = (2 + hc_mult) * hc_mult

    residual = torch.randn(num_tokens, hc_mult, hidden_size).to(dtype)
    x = torch.randn(num_tokens, hidden_size).to(dtype)
    fn = torch.randn(hc_mix, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(hc_mix, dtype=torch.float32)
    norm_weight = torch.randn(hidden_size).to(dtype)

    post_mix, comb_mix, layer_input = mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=3,
        norm_weight=norm_weight,
        norm_eps=1e-5,
    )

    assert post_mix.dtype == torch.float32
    assert comb_mix.dtype == torch.float32
    assert layer_input.dtype == dtype
    _, _, layer_input_raw = mhc_pre_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=3,
    )
    expected_layer_input = F.rms_norm(
        layer_input_raw.float(),
        (hidden_size,),
        weight=norm_weight.float(),
        eps=1e-5,
    ).to(dtype)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(
        layer_input,
        expected_layer_input,
        rtol=tolerance,
        atol=tolerance,
    )

    expected_residual = mhc_post_torch(x, residual, post_mix, comb_mix)
    expected = mhc_pre_torch(
        expected_residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=3,
        norm_weight=norm_weight,
        norm_eps=1e-5,
    )
    actual = mhc_fused_post_pre_torch(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=3,
        norm_weight=norm_weight,
        norm_eps=1e-5,
    )

    torch.testing.assert_close(actual[0], expected_residual)
    for actual_tensor, expected_tensor in zip(actual[1:], expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hc_head_torch_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(1)
    num_tokens, hc_mult, hidden_size = 2, 4, 32
    x = torch.randn(num_tokens, hc_mult, hidden_size).to(dtype)
    fn = torch.randn(hc_mult, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = torch.randn(1, dtype=torch.float32)
    hc_base = torch.randn(hc_mult, dtype=torch.float32)

    actual = hc_head_torch(x, fn, hc_scale, hc_base, 1e-6, 1e-6)

    x_flat = x.flatten(-2).float()
    x_normed = F.rms_norm(x_flat, (hc_mult * hidden_size,), eps=1e-6)
    pre_mix = torch.sigmoid(F.linear(x_normed, fn) * hc_scale + hc_base) + 1e-6
    expected = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2).to(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
