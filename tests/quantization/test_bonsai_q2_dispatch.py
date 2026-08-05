# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization import gguf as gguf_quant
from vllm.transformers_utils.gguf_utils import (
    BONSAI_Q2_0_BLOCK_SIZE,
    BONSAI_Q2_0_TYPE,
    BONSAI_Q2_0_TYPE_SIZE,
    ensure_bonsai_q2_0_gguf_compat,
)


def test_bonsai_q2_0_gguf_reader_compatibility():
    q2_0 = ensure_bonsai_q2_0_gguf_compat()

    assert q2_0.value == BONSAI_Q2_0_TYPE
    assert q2_0.name == "Q2_0"
    assert gguf_quant.gguf.GGMLQuantizationType(BONSAI_Q2_0_TYPE) is q2_0
    assert gguf_quant.gguf.GGML_QUANT_SIZES[q2_0] == (
        BONSAI_Q2_0_BLOCK_SIZE,
        BONSAI_Q2_0_TYPE_SIZE,
    )


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


def test_gguf_fused_numeric_shards_use_canonical_output_order(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeQWeight:
        def __init__(self, order: list[int]) -> None:
            self.data = torch.tensor(order, dtype=torch.uint8).view(-1, 1)
            self.shard_id = order
            self.shard_offset_map = {
                shard_id: (index, index + 1, 1)
                for index, shard_id in enumerate(order)
            }

        def __getitem__(self, index):
            return self.data[index]

    physical_order = [3, 0, 1, 2, 5, 4]
    layer = type("Layer", (), {})()
    layer.qweight = FakeQWeight(physical_order)
    layer.qweight_type = type("WeightType", (), {})()
    layer.qweight_type.shard_weight_type = {
        shard_id: gguf_quant.BONSAI_Q2_0 for shard_id in physical_order
    }

    def fake_matmul(
        x: torch.Tensor,
        weight: torch.Tensor,
        qweight_type: int,
    ) -> torch.Tensor:
        del qweight_type
        return weight[:, 0].to(x.dtype).view(1, -1).expand(x.shape[0], -1)

    monkeypatch.setattr(gguf_quant, "fused_mul_mat_gguf", fake_matmul)

    method = object.__new__(gguf_quant.GGUFLinearMethod)
    output = method.apply(layer, torch.empty((2, 4)))

    assert output.tolist() == [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
