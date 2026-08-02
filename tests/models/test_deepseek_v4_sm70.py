# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.nvidia.sm70 import (
    _apply_gptj_rope,
    _e4m3fn_fp16_lut,
    _indexer_logits_sm70,
    _reference_sparse_attention,
)
from vllm.platforms.interface import DeviceCapability


def test_sm70_e4m3fn_software_decode() -> None:
    lut = _e4m3fn_fp16_lut(torch.device("cpu"))
    encoded = torch.tensor([0x00, 0x01, 0x08, 0x7E, 0x80, 0x81, 0xFE])
    expected = torch.tensor(
        [0.0, 2.0**-9, 2.0**-6, 448.0, -0.0, -(2.0**-9), -448.0],
        dtype=torch.float16,
    )
    torch.testing.assert_close(lut[encoded], expected, rtol=0, atol=0)


def test_sm70_gptj_rope_inverse_round_trip() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 3, 16, dtype=torch.float16)
    positions = torch.tensor([0, 1, 3, 7])
    frequencies = torch.randn(8, 4)
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)

    rotated = _apply_gptj_rope(x, positions, cache, rope_dim=8)
    recovered = _apply_gptj_rope(rotated, positions, cache, rope_dim=8, inverse=True)

    torch.testing.assert_close(recovered, x, rtol=2e-3, atol=2e-3)


def test_sm70_sparse_attention_matches_reference_with_sink() -> None:
    torch.manual_seed(1)
    q = torch.randn(2, 3, 8, dtype=torch.float16)
    kv = torch.randn(2, 5, 8, dtype=torch.float16)
    invalid = torch.tensor(
        [[False, False, True, True, True], [False, False, False, True, True]]
    )
    sink = torch.tensor([-0.2, 0.1, 0.4])
    scale = 0.25

    actual = _reference_sparse_attention(q, kv, invalid, scale, sink)

    scores = torch.bmm(q.float(), kv.float().transpose(1, 2)) * scale
    scores.masked_fill_(invalid.unsqueeze(1), -torch.inf)
    sink_scores = sink.view(1, -1, 1).expand(q.shape[0], -1, -1)
    probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)
    expected = torch.bmm(probabilities[..., :-1], kv.float()).to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_sm70_indexer_logits_masks_each_query_range() -> None:
    torch.manual_seed(2)
    q = torch.randn(2, 3, 4, dtype=torch.float16)
    k = torch.randn(6, 4, dtype=torch.float16)
    weights = torch.randn(2, 3, dtype=torch.float32)
    starts = torch.tensor([0, 2])
    ends = torch.tensor([3, 6])

    actual = _indexer_logits_sm70(q, k, weights, starts, ends)

    scores = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    expected = torch.sum(torch.relu(scores) * weights.unsqueeze(-1), dim=1)
    expected[0, 3:] = -torch.inf
    expected[1, :2] = -torch.inf
    torch.testing.assert_close(actual, expected)


def test_sm70_decode_metadata_does_not_require_flashmla(monkeypatch) -> None:
    from vllm.v1.attention.backends.mla import sparse_swa

    class SM70Platform:
        @staticmethod
        def is_cuda() -> bool:
            return True

        @staticmethod
        def is_rocm() -> bool:
            return False

        @staticmethod
        def is_xpu() -> bool:
            return False

        @staticmethod
        def get_device_capability() -> DeviceCapability:
            return DeviceCapability(major=7, minor=0)

    builder = object.__new__(sparse_swa.DeepseekSparseSWAMetadataBuilder)
    builder._layer_types = {
        sparse_swa._LAYER_TYPE_SWAONLY,
        sparse_swa._LAYER_TYPE_C4A,
        sparse_swa._LAYER_TYPE_C128A,
    }
    monkeypatch.setattr(sparse_swa, "current_platform", SM70Platform())

    def fail_if_called():
        raise AssertionError("SM70 must not initialize FlashMLA metadata")

    monkeypatch.setattr(sparse_swa, "get_mla_metadata", fail_if_called)

    assert builder.build_tile_scheduler(num_decode_tokens=1) == {
        sparse_swa._LAYER_TYPE_SWAONLY: None,
        sparse_swa._LAYER_TYPE_C4A: None,
        sparse_swa._LAYER_TYPE_C128A: None,
    }
