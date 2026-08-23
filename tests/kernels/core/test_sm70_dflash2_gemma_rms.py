# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    _sm70_dflash2_gemma_fused_add_rms_norm,
    _use_sm70_dflash2_gemma_fused_add_rms,
)


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 137, 512])
@pytest.mark.parametrize("weight_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sm70_dflash2_gemma_fused_add_rms_is_within_one_fp16_ulp(
    num_tokens: int,
    weight_dtype: torch.dtype,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    torch.manual_seed(20260823)
    x = torch.randn((num_tokens, 5120), dtype=torch.float16, device="cuda").mul_(0.125)
    residual = torch.randn((num_tokens, 5120), dtype=torch.float32, device="cuda").mul_(
        0.125
    )
    weight = torch.randn(5120, dtype=torch.float32, device="cuda").mul_(0.05)
    weight = weight.to(weight_dtype)

    expected_normalized, expected_residual = GemmaRMSNorm._forward_static_with_residual(
        weight, 1e-6, x, residual
    )
    actual_normalized, actual_residual = _sm70_dflash2_gemma_fused_add_rms_norm(
        x, residual, weight, 1e-6
    )

    lower = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, -float("inf")),
    )
    upper = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, float("inf")),
    )
    assert torch.all((actual_normalized >= lower) & (actual_normalized <= upper))
    assert torch.equal(actual_residual, expected_residual)


def test_sm70_dflash2_gemma_fused_add_rms_graph_replay_reads_current_inputs():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    torch.manual_seed(20260823)
    x = torch.randn((8, 5120), dtype=torch.float16, device="cuda").mul_(0.125)
    residual = torch.randn((8, 5120), dtype=torch.float32, device="cuda").mul_(0.125)
    weight = torch.randn(5120, dtype=torch.float32, device="cuda").mul_(0.05)
    _sm70_dflash2_gemma_fused_add_rms_norm(x, residual, weight, 1e-6)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_normalized, actual_residual = _sm70_dflash2_gemma_fused_add_rms_norm(
            x, residual, weight, 1e-6
        )

    x.copy_(torch.randn_like(x).mul_(0.125))
    residual.copy_(torch.randn_like(residual).mul_(0.125))
    weight.copy_(torch.randn_like(weight).mul_(0.05))
    graph.replay()
    torch.cuda.synchronize()

    expected_normalized, expected_residual = GemmaRMSNorm._forward_static_with_residual(
        weight, 1e-6, x, residual
    )
    lower = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, -float("inf")),
    )
    upper = torch.nextafter(
        expected_normalized,
        torch.full_like(expected_normalized, float("inf")),
    )
    assert torch.all((actual_normalized >= lower) & (actual_normalized <= upper))
    assert torch.equal(actual_residual, expected_residual)


def test_sm70_dflash2_gemma_fused_add_rms_allows_aot_warmup_shape(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS", "1")
    monkeypatch.setenv("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH", "1")
    x = torch.empty((512, 5120), dtype=torch.float16, device="cuda")
    residual = torch.empty_like(x, dtype=torch.float32)
    weight = torch.empty(5120, dtype=torch.float16, device="cuda")

    assert _use_sm70_dflash2_gemma_fused_add_rms(x, residual, weight)
