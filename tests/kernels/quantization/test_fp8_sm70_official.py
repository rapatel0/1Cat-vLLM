# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.models.deepseek_v4.nvidia.sm70 import sm70_inv_rope_einsum
from vllm.platforms import current_platform

MODEL_ENV = "VLLM_DEEPSEEK_V4_MODEL"


def _load_tensor(model: Path, name: str) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][name]
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(name)


def _sm70_fp8_method() -> Fp8LinearMethod:
    method = object.__new__(Fp8LinearMethod)
    method.use_marlin = False
    method.use_sm70_fp8_turbomind = True
    method.use_sm70_dequant_fallback = False
    method.weight_block_size = [128, 128]
    return method


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires CUDA SM70",
)
def test_sm70_fp8_grouped_bmm_uses_per_group_turbomind_layout() -> None:
    class GroupedBMM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight_bytes = torch.cat(
                (
                    torch.full((128, 256), 0x38, dtype=torch.uint8),
                    torch.full((128, 256), 0x40, dtype=torch.uint8),
                )
            ).cuda()
            self.weight = torch.nn.Parameter(
                weight_bytes.view(torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = torch.nn.Parameter(
                torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
                .cuda()
                .view(torch.float8_e8m0fnu),
                requires_grad=False,
            )
            self.input_scale = None
            self.orig_dtype = torch.float16
            self.is_bmm = True
            self.bmm_batch_size = 2

    layer = GroupedBMM()
    expected_weight = layer.weight.detach().clone()
    expected_scale = layer.weight_scale_inv.detach().clone()
    _sm70_fp8_method().process_weights_after_loading(layer)

    assert layer.weight.shape == (2, 256, 128)
    assert layer.weight_scale_inv.shape == (2, 2, 128)
    assert layer.sm70_fp8_grouped_bmm

    torch.manual_seed(7)
    o = torch.randn(3, 2, 256, dtype=torch.float16, device="cuda")
    rotary = torch.nn.Module()
    rotary.register_buffer("cos_sin_cache", torch.empty(0, device="cuda"))
    actual = sm70_inv_rope_einsum(
        rotary,
        o,
        torch.zeros(3, dtype=torch.int64, device="cuda"),
        0,
        2,
        128,
        layer,
    )
    scale = expected_scale.to(torch.float32).view(2, 1, 2)
    expanded_scale = scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    ref_weight = expected_weight.view(2, 128, 256).to(torch.float16)
    ref_weight = ref_weight * expanded_scale.to(torch.float16)
    expected = torch.einsum("tgd,grd->tgr", o, ref_weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires CUDA SM70",
)
@pytest.mark.parametrize("num_tokens", [1, 16])
def test_official_deepseek_v4_fp8_dense_matches_reference(num_tokens: int) -> None:
    model_value = os.getenv(MODEL_ENV)
    if model_value is None:
        pytest.skip(f"set {MODEL_ENV} to an official DeepSeek V4 checkpoint")
    assert model_value is not None
    model = Path(model_value)

    prefix = "layers.0.attn.wq_a"
    weight = _load_tensor(model, f"{prefix}.weight").cuda()
    scales = _load_tensor(model, f"{prefix}.scale").to(torch.float32).cuda()
    weight, scales = process_fp8_weight_block_strategy(weight, scales)

    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
        weight,
        scales,
        128,
        False,
    )
    torch.manual_seed(7)
    hidden = torch.randn(
        num_tokens,
        weight.shape[1],
        dtype=torch.float16,
        device="cuda",
    )
    actual = torch.empty(
        num_tokens,
        weight.shape[0],
        dtype=torch.float16,
        device="cuda",
    )
    sm70_ops.fp8_gemm_sm70_out(
        actual,
        hidden,
        tm_weight,
        tm_scales,
        128,
        int(meta[0].item()),
        int(meta[1].item()),
        False,
    )

    expanded_scales = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    ref_weight = weight.to(torch.float16) * expanded_scales[
        : weight.shape[0], : weight.shape[1]
    ].to(torch.float16)
    expected = F.linear(hidden, ref_weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires CUDA SM70",
)
@pytest.mark.parametrize("fused_gated_silu", [False, True])
def test_official_deepseek_v4_fp8_shared_expert_matches_reference(
    fused_gated_silu: bool,
) -> None:
    model_value = os.getenv(MODEL_ENV)
    if model_value is None:
        pytest.skip(f"set {MODEL_ENV} to an official DeepSeek V4 checkpoint")
    assert model_value is not None
    model = Path(model_value)

    prefix = "layers.0.ffn.shared_experts"
    weight = torch.cat(
        [_load_tensor(model, f"{prefix}.{proj}.weight") for proj in ("w1", "w3")]
    ).cuda()
    scales = (
        torch.cat(
            [_load_tensor(model, f"{prefix}.{proj}.scale") for proj in ("w1", "w3")]
        )
        .to(torch.float32)
        .cuda()
    )
    weight, scales = process_fp8_weight_block_strategy(weight, scales)

    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
        weight,
        scales,
        128,
        True,
    )
    torch.manual_seed(7)
    hidden = torch.randn(
        16,
        weight.shape[1],
        dtype=torch.float16,
        device="cuda",
    )
    output_size = weight.shape[0] // 2 if fused_gated_silu else weight.shape[0]
    actual = torch.empty(
        hidden.shape[0],
        output_size,
        dtype=torch.float16,
        device="cuda",
    )
    sm70_ops.fp8_gemm_sm70_out(
        actual,
        hidden,
        tm_weight,
        tm_scales,
        128,
        int(meta[0].item()),
        int(meta[1].item()),
        fused_gated_silu,
    )

    expanded_scales = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    ref_weight = weight.to(torch.float16) * expanded_scales[
        : weight.shape[0], : weight.shape[1]
    ].to(torch.float16)
    expected = F.linear(hidden, ref_weight)
    if fused_gated_silu:
        gate, up = expected.chunk(2, dim=-1)
        expected = F.silu(gate) * up
    else:
        expected = (
            expected.reshape(hidden.shape[0], 2, output_size // 2)
            .transpose(1, 2)
            .reshape(hidden.shape[0], output_size)
        )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires CUDA SM70",
)
def test_official_deepseek_v4_wo_a_matches_reference() -> None:
    model_value = os.getenv(MODEL_ENV)
    if model_value is None:
        pytest.skip(f"set {MODEL_ENV} to an official DeepSeek V4 checkpoint")
    assert model_value is not None
    model = Path(model_value)

    prefix = "layers.0.attn.wo_a"
    weight = _load_tensor(model, f"{prefix}.weight").cuda()
    scales = _load_tensor(model, f"{prefix}.scale").cuda()

    class Rotary(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cos_sin_cache = torch.cat(
                (
                    torch.ones(32, device="cuda"),
                    torch.zeros(32, device="cuda"),
                )
            ).view(1, -1)

    class Projection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(scales, requires_grad=False)
            self.input_scale = None
            self.orig_dtype = torch.float16
            self.is_bmm = True
            self.bmm_batch_size = 8

    torch.manual_seed(7)
    o = torch.randn(1, 64, 512, dtype=torch.float16, device="cuda")
    positions = torch.zeros(1, dtype=torch.int64, device="cuda")
    projection = Projection()
    _sm70_fp8_method().process_weights_after_loading(projection)
    actual = sm70_inv_rope_einsum(
        Rotary(),
        o,
        positions,
        64,
        8,
        1024,
        projection,
    )

    scale = torch.exp2(scales.view(torch.uint8).float() - 127.0)
    expanded_scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    ref_weight = weight.to(torch.float16) * expanded_scale.to(torch.float16)
    expected = torch.einsum(
        "tgd,grd->tgr",
        o.view(1, 8, 4096),
        ref_weight.view(8, 1024, 4096),
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)
