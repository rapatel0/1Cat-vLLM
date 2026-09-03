# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and contract tests for QSA over signed int8_block32 KV pages.

Source under test:
  vllm/models/qwen4_exp/nvidia/ops/qsa.py
    qsa_sparse_paged_attention_int8_block32
  vllm/models/qwen4_exp/nvidia/qsa.py
    Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import (
    qsa_sparse_paged_attention,
    qsa_sparse_paged_attention_int8_block32,
)
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
from vllm.v1.kv_cache_interface import (
    INT8_BLOCK32_CHANNEL_SIZE,
    make_int8_block32_kv_cache_views,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _quantize_block32(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [pages, tokens, heads, dim] FP16 to signed int8 with block scales.

    One scale per page, per head, per 32-channel block, matching the layout that
    ``make_int8_block32_kv_cache_views`` exposes.
    """
    pages, tokens, heads, dim = dense.shape
    blocks = dim // INT8_BLOCK32_CHANNEL_SIZE
    grouped = dense.reshape(pages, tokens, heads, blocks, INT8_BLOCK32_CHANNEL_SIZE)
    # Scales are shared across the whole page, so reduce over the token axis.
    amax = grouped.abs().amax(dim=(1, 4)).clamp(min=1e-4)
    scale = (amax / 127.0).to(torch.float16)
    payload = torch.clamp(
        torch.round(grouped / scale[:, None, :, :, None].to(torch.float32)),
        -127,
        127,
    ).to(torch.int8)
    return payload.reshape(pages, tokens, heads, dim), scale


def _build_int8_cache(
    key_dense: torch.Tensor,
    value_dense: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    pages, tokens, heads, dim = key_dense.shape
    blocks = dim // INT8_BLOCK32_CHANNEL_SIZE
    payload_bytes = tokens * heads * dim
    page_bytes = 2 * payload_bytes + 2 * heads * blocks * 2 + 4
    raw = torch.zeros(pages * page_bytes, dtype=torch.int8, device=key_dense.device)
    views = make_int8_block32_kv_cache_views(
        raw,
        num_blocks=pages,
        block_size=tokens,
        num_kv_heads=heads,
        head_size=dim,
    )
    k_cache, v_cache, k_scales, v_scales, _owners = views
    k_payload, k_scale = _quantize_block32(key_dense)
    v_payload, v_scale = _quantize_block32(value_dense)
    k_cache.copy_(k_payload)
    v_cache.copy_(v_payload)
    k_scales.copy_(k_scale)
    v_scales.copy_(v_scale)
    return k_cache, v_cache, k_scales, v_scales, k_scale, v_scale


def test_backend_admits_int8_block32():
    """The QSA backend must advertise int8_block32 alongside FP16 and BF16."""
    supported = Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
    assert "int8_block32" in supported
    # The pre-existing routes must remain advertised.
    assert "auto" in supported
    assert "float16" in supported
    assert "bfloat16" in supported


@CUDA
@pytest.mark.parametrize("num_queries", [1, 5])
def test_int8_block32_matches_dequantized_reference(num_queries):
    """INT8 QSA must match FP16 QSA run on the dequantized same values.

    This pins the dequantization contract: payload times its block scale. Any
    silent reinterpretation of INT8 bytes as FP16 would diverge grossly.
    """
    torch.manual_seed(0)
    device = "cuda"
    pages, tokens, kv_heads, dim = 6, 32, 1, 256
    query_heads, topk = 6, 64

    key_dense = torch.randn(pages, tokens, kv_heads, dim, device=device) * 0.5
    value_dense = torch.randn(pages, tokens, kv_heads, dim, device=device) * 0.5
    k_cache, v_cache, k_scales, v_scales, k_scale, v_scale = _build_int8_cache(
        key_dense.to(torch.float16), value_dense.to(torch.float16)
    )

    # Exact dequantization of the stored payload, used as the control input.
    blocks = dim // INT8_BLOCK32_CHANNEL_SIZE
    k_ref = (
        k_cache.reshape(pages, tokens, kv_heads, blocks, INT8_BLOCK32_CHANNEL_SIZE).to(
            torch.float16
        )
        * k_scale[:, None, :, :, None]
    ).reshape(pages, tokens, kv_heads, dim)
    v_ref = (
        v_cache.reshape(pages, tokens, kv_heads, blocks, INT8_BLOCK32_CHANNEL_SIZE).to(
            torch.float16
        )
        * v_scale[:, None, :, :, None]
    ).reshape(pages, tokens, kv_heads, dim)

    query = torch.randn(
        num_queries, query_heads, dim, device=device, dtype=torch.float16
    )
    indices = torch.arange(topk, device=device, dtype=torch.int32)
    indices = indices.repeat(num_queries, 1).contiguous()
    block_table = torch.arange(pages, device=device, dtype=torch.int32)
    block_table = block_table.repeat(num_queries, 1).contiguous()
    token_to_req = torch.arange(num_queries, device=device, dtype=torch.int32)

    int8_out = qsa_sparse_paged_attention_int8_block32(
        query,
        k_cache,
        v_cache,
        k_scales,
        v_scales,
        indices,
        block_table,
        token_to_req,
    )
    reference = qsa_sparse_paged_attention(
        query,
        k_ref.contiguous(),
        v_ref.contiguous(),
        indices,
        block_table,
        token_to_req,
    )
    torch.testing.assert_close(int8_out, reference, rtol=2e-2, atol=2e-2)


@CUDA
def test_int8_block32_rejects_fp16_payload():
    """FP16 payloads must be refused, never reinterpreted as INT8."""
    device = "cuda"
    pages, tokens, kv_heads, dim = 2, 32, 1, 256
    blocks = dim // INT8_BLOCK32_CHANNEL_SIZE
    query = torch.randn(1, 2, dim, device=device, dtype=torch.float16)
    fp16_cache = torch.zeros(
        pages, tokens, kv_heads, dim, device=device, dtype=torch.float16
    )
    scales = torch.ones(pages, kv_heads, blocks, device=device, dtype=torch.float16)
    indices = torch.zeros(1, 8, device=device, dtype=torch.int32)
    block_table = torch.zeros(1, pages, device=device, dtype=torch.int32)
    token_to_req = torch.zeros(1, device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="signed int8"):
        qsa_sparse_paged_attention_int8_block32(
            query,
            fp16_cache,
            fp16_cache,
            scales,
            scales,
            indices,
            block_table,
            token_to_req,
        )


@CUDA
def test_int8_block32_rejects_mismatched_scale_shape():
    """A scale grid that does not match its payload must fail fast."""
    device = "cuda"
    pages, tokens, kv_heads, dim = 2, 32, 1, 256
    query = torch.randn(1, 2, dim, device=device, dtype=torch.float16)
    payload = torch.zeros(pages, tokens, kv_heads, dim, device=device, dtype=torch.int8)
    wrong = torch.ones(pages, kv_heads, 3, device=device, dtype=torch.float16)
    indices = torch.zeros(1, 8, device=device, dtype=torch.int32)
    block_table = torch.zeros(1, pages, device=device, dtype=torch.int32)
    token_to_req = torch.zeros(1, device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="scale shape"):
        qsa_sparse_paged_attention_int8_block32(
            query,
            payload,
            payload,
            wrong,
            wrong,
            indices,
            block_table,
            token_to_req,
        )
