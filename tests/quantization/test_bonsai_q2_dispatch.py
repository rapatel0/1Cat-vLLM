# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization import gguf as gguf_quant


@pytest.fixture
def sm70_dispatch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gguf_quant.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        gguf_quant.current_platform,
        "is_device_capability",
        lambda capability: capability == 70,
    )


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (1, "dp4a"),
        (2, "dp4a"),
        (4, "mma"),
        (8, "mma"),
        (16, "dp4a"),
        (63, "dp4a"),
        (64, "mma"),
        (65, "mma"),
    ],
)
def test_bonsai_q2_sm70_shape_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    sm70_dispatch,
    num_tokens: int,
    expected: str,
):
    calls: list[str] = []
    qweight = torch.empty((3, 34), dtype=torch.uint8)
    x = torch.empty((num_tokens, 128), dtype=torch.half)

    def mma(weight: torch.Tensor, inputs: torch.Tensor, row: int):
        calls.append("mma")
        return torch.empty((inputs.shape[0], row), dtype=inputs.dtype)

    def dp4a(
        weight: torch.Tensor,
        inputs: torch.Tensor,
        quant_type: int,
        row: int,
    ):
        calls.append("dp4a")
        return torch.empty((inputs.shape[0], row), dtype=inputs.dtype)

    monkeypatch.setattr(gguf_quant.ops, "ggml_mul_mat_q2_0_sm70", mma)
    monkeypatch.setattr(gguf_quant.ops, "ggml_mul_mat_vec_a8", dp4a)

    output = gguf_quant._fused_mul_mat_gguf(x, qweight, gguf_quant.BONSAI_Q2_0)

    assert output.shape == (num_tokens, qweight.shape[0])
    assert calls == [expected]


def test_bonsai_q2_non_sm70_keeps_dequant_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    qweight = torch.empty((3, 34), dtype=torch.uint8)
    x = torch.ones((8, 128), dtype=torch.half)

    monkeypatch.setattr(gguf_quant.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        gguf_quant.current_platform,
        "is_device_capability",
        lambda capability: False,
    )

    def dequantize(
        weight: torch.Tensor,
        quant_type: int,
        m: int,
        n: int,
        dtype: torch.dtype,
    ):
        calls.append("dequant")
        return torch.zeros((m, n), dtype=dtype)

    monkeypatch.setattr(gguf_quant.ops, "ggml_dequantize", dequantize)

    output = gguf_quant._fused_mul_mat_gguf(x, qweight, gguf_quant.BONSAI_Q2_0)

    assert output.shape == (x.shape[0], qweight.shape[0])
    assert calls == ["dequant"]


def test_bonsai_q2_sm70_non_fp16_keeps_dp4a(
    monkeypatch: pytest.MonkeyPatch,
    sm70_dispatch,
):
    calls: list[str] = []
    qweight = torch.empty((3, 34), dtype=torch.uint8)
    x = torch.empty((8, 128), dtype=torch.bfloat16)

    def dp4a(
        weight: torch.Tensor,
        inputs: torch.Tensor,
        quant_type: int,
        row: int,
    ):
        calls.append("dp4a")
        return torch.empty((inputs.shape[0], row), dtype=inputs.dtype)

    monkeypatch.setattr(gguf_quant.ops, "ggml_mul_mat_vec_a8", dp4a)

    gguf_quant._fused_mul_mat_gguf(x, qweight, gguf_quant.BONSAI_Q2_0)

    assert calls == ["dp4a"]
