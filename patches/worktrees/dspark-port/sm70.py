# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional DeepSeek-V4 sparse-MLA fallbacks for SM70 GPUs."""

from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch.nn.functional as F

from vllm import _sm70_ops as sm70_ops
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )


@lru_cache(maxsize=16)
def _e4m3fn_fp16_lut(device: torch.device) -> torch.Tensor:
    """Return all finite E4M3FN byte values without using CUDA FP8 casts."""
    byte = torch.arange(256, dtype=torch.int16)
    sign = torch.where((byte & 0x80) != 0, -1.0, 1.0)
    exponent = (byte >> 3) & 0x0F
    mantissa = byte & 0x07
    subnormal = mantissa.to(torch.float32) * (2.0**-9)
    normal = (1.0 + mantissa.to(torch.float32) / 8.0) * torch.exp2(
        exponent.to(torch.float32) - 7.0
    )
    value = torch.where(exponent == 0, subnormal, normal)
    # E4M3FN reserves exponent=15,mantissa=7 for NaN. Saturating writers
    # never emit it; map it to the largest finite magnitude defensively.
    value = torch.where((exponent == 15) & (mantissa == 7), 448.0, value)
    return (sign * value).to(torch.float16).to(device)


@triton.jit
def _gather_fp8_ds_mla_rows_sm70_kernel(
    output_ptr,
    cache_ptr,
    indices_ptr,
    fp8_lut_ptr,
    num_cache_tokens,
    block_stride: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_bytes: tl.constexpr,
    scale_bytes: tl.constexpr,
    nope_dim: tl.constexpr,
    head_dim: tl.constexpr,
) -> None:
    """Gather one packed cache row per program without hardware FP8 casts."""
    row = tl.program_id(0)
    index = tl.load(indices_ptr + row).to(tl.int64)
    valid = (index >= 0) & (index < num_cache_tokens)
    safe_index = tl.where(valid, index, 0)
    block = safe_index // cache_block_size
    position = safe_index % cache_block_size
    cache_block_ptr = cache_ptr + block * block_stride
    token_ptr = cache_block_ptr + position * token_data_bytes
    scale_ptr = (
        cache_block_ptr + cache_block_size * token_data_bytes + position * scale_bytes
    )

    offsets = tl.arange(0, head_dim)
    is_nope = offsets < nope_dim
    fp8_bytes = tl.load(token_ptr + offsets, mask=is_nope, other=0)
    fp8 = tl.load(fp8_lut_ptr + fp8_bytes.to(tl.int32), mask=is_nope, other=0.0)
    encoded_scale = tl.load(
        scale_ptr + offsets // 64,
        mask=is_nope,
        other=127,
    )
    scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
    nope = fp8 * scale

    rope_ptr = (token_ptr + nope_dim).to(tl.pointer_type(tl.float16))
    rope = tl.load(
        rope_ptr + offsets - nope_dim,
        mask=~is_nope,
        other=0.0,
    )
    value = tl.where(is_nope, nope, rope)
    value = tl.where(valid, value, 0.0)
    tl.store(output_ptr + row * head_dim + offsets, value)


@triton.jit
def _store_fp16_paged_rows_sm70_kernel(
    cache_ptr,
    rows_ptr,
    slots_ptr,
    num_cache_tokens,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_block_size: tl.constexpr,
    row_width: tl.constexpr,
) -> None:
    """Store fixed-shape compressor rows while skipping inactive slots."""
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    valid = (slot >= 0) & (slot < num_cache_tokens)
    safe_slot = tl.where(valid, slot, 0)
    block = safe_slot // cache_block_size
    position = safe_slot % cache_block_size
    offsets = tl.arange(0, row_width)
    values = tl.load(rows_ptr + row * row_width + offsets)
    output = (
        cache_ptr + block * cache_stride_block + position * cache_stride_token + offsets
    )
    tl.store(output, values, mask=valid)


@triton.jit
def _paged_indexer_logits_sm70_kernel(
    logits_ptr,
    q_ptr,
    cache_ptr,
    weights_ptr,
    seq_lens_ptr,
    block_table_ptr,
    q_stride_token,
    q_stride_head,
    cache_stride_block,
    cache_stride_token,
    block_table_stride,
    seq_lens_stride,
    cache_block_size: tl.constexpr,
    max_context: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute paged FP16 indexer logits without host-side cache gathers."""
    query = tl.program_id(0)
    column_block = tl.program_id(1)
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    context_len = tl.load(seq_lens_ptr + query * seq_lens_stride)
    if column_block * BLOCK_N < context_len:
        valid_columns = (columns < context_len) & (columns < max_context)
        logical_blocks = columns // cache_block_size
        physical_blocks = tl.load(
            block_table_ptr + query * block_table_stride + logical_blocks,
            mask=valid_columns,
            other=0,
        ).to(tl.int64)
        cache_positions = columns % cache_block_size

        heads = tl.arange(0, num_heads)
        features = tl.arange(0, head_dim)
        q = tl.load(
            q_ptr
            + query * q_stride_token
            + heads[:, None] * q_stride_head
            + features[None, :]
        ).to(tl.float16)
        k = tl.load(
            cache_ptr
            + physical_blocks[None, :] * cache_stride_block
            + cache_positions[None, :] * cache_stride_token
            + features[:, None],
            mask=valid_columns[None, :],
            other=0.0,
        ).to(tl.float16)
        scores = tl.dot(q, k, out_dtype=tl.float32)
        head_weights = tl.load(weights_ptr + query * num_heads + heads)
        logits = tl.sum(tl.maximum(scores, 0.0) * head_weights[:, None], axis=0)
        tl.store(
            logits_ptr + query * max_context + columns,
            logits,
            mask=valid_columns,
        )


def _gather_fp8_ds_mla_rows_reference(
    kv_cache: torch.Tensor,
    flat_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Framework reference for packed FP8-DS-MLA row gathering."""
    assert kv_cache.dtype == torch.uint8
    nope_dim = 448
    rope_dim = 64
    token_data_bytes = nope_dim + rope_dim * 2
    scale_bytes = 8

    indices = flat_indices.to(torch.int64).reshape(-1)
    max_index = kv_cache.shape[0] * block_size
    valid = (indices >= 0) & (indices < max_index)
    safe = indices.clamp(0, max(max_index - 1, 0))
    block_indices = safe // block_size
    positions = safe % block_size

    cache = kv_cache.view(kv_cache.shape[0], -1)
    token_offsets = positions * token_data_bytes
    scale_offsets = block_size * token_data_bytes + positions * scale_bytes

    nope_offsets = torch.arange(nope_dim, device=kv_cache.device)
    scale_index = torch.arange(nope_dim // 64, device=kv_cache.device)
    rope_offsets = torch.arange(rope_dim * 2, device=kv_cache.device)

    fp8_bytes = cache[block_indices[:, None], token_offsets[:, None] + nope_offsets]
    encoded_scales = cache[block_indices[:, None], scale_offsets[:, None] + scale_index]
    fp8 = _e4m3fn_fp16_lut(kv_cache.device)[fp8_bytes.to(torch.long)]
    scales = torch.exp2(encoded_scales.to(torch.float32) - 127.0).to(torch.float16)
    nope = (fp8.view(-1, nope_dim // 64, 64) * scales.unsqueeze(-1)).view(-1, nope_dim)

    rope_bytes = cache[
        block_indices[:, None],
        token_offsets[:, None] + nope_dim + rope_offsets,
    ].contiguous()
    rope = rope_bytes.view(torch.float16).view(-1, rope_dim)
    rows = torch.cat((nope, rope), dim=-1)
    rows.masked_fill_(~valid.unsqueeze(-1), 0)
    return rows


def _gather_fp8_ds_mla_rows(
    kv_cache: torch.Tensor,
    flat_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather packed FP8-DS-MLA rows and dequantize them to FP16."""
    assert kv_cache.dtype == torch.uint8
    if not kv_cache.is_cuda:
        return _gather_fp8_ds_mla_rows_reference(kv_cache, flat_indices, block_size)

    indices = flat_indices.contiguous().reshape(-1)
    output = torch.empty(
        (indices.numel(), 512),
        dtype=torch.float16,
        device=kv_cache.device,
    )
    _gather_fp8_ds_mla_rows_sm70_kernel[(indices.numel(),)](
        output,
        kv_cache,
        indices,
        _e4m3fn_fp16_lut(kv_cache.device),
        kv_cache.shape[0] * block_size,
        block_stride=kv_cache.stride(0),
        cache_block_size=block_size,
        token_data_bytes=576,
        scale_bytes=8,
        nope_dim=448,
        head_dim=512,
        num_warps=8,
    )
    return output


def _reference_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    invalid: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    """FP32 softmax reference over already-gathered FP16 KV rows."""
    if q.is_cuda:
        scores = torch.bmm(
            q,
            kv.transpose(1, 2),
            out_dtype=torch.float32,
        )
    else:
        scores = torch.bmm(q.to(torch.float32), kv.to(torch.float32).transpose(1, 2))
    scores *= scale
    scores.masked_fill_(invalid.unsqueeze(1), -torch.inf)

    if attn_sink is not None:
        sink = (
            attn_sink[: q.shape[1]]
            .to(torch.float32)
            .view(1, -1, 1)
            .expand(q.shape[0], -1, -1)
        )
        probabilities = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)
        probabilities = probabilities[..., :-1]
    else:
        probabilities = torch.softmax(scores, dim=-1)
        probabilities.nan_to_num_(nan=0.0)
    if q.is_cuda:
        return torch.bmm(probabilities.to(q.dtype), kv)
    return torch.bmm(probabilities, kv.to(torch.float32)).to(q.dtype)


def _gather_scope(
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_indices = indices.reshape(indices.shape[0], -1)
    gathered = _gather_fp8_ds_mla_rows(
        kv_cache, dense_indices.reshape(-1), block_size
    ).view(*dense_indices.shape, 512)
    invalid = dense_indices < 0
    if lengths is not None:
        width = dense_indices.shape[-1]
        offsets = torch.arange(width, device=indices.device)
        invalid |= offsets.unsqueeze(0) >= lengths.reshape(-1, 1)
    return gathered, invalid


def _gather_sequence_cache(
    output: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    for req_idx in range(seq_lens.shape[0]):
        seq_len = int(seq_lens[req_idx].item())
        gather_len = (
            seq_len if gather_lens is None else int(gather_lens[req_idx].item())
        )
        if gather_len <= 0:
            continue
        positions = torch.arange(
            seq_len - gather_len,
            seq_len,
            dtype=torch.int64,
            device=kv_cache.device,
        )
        physical_blocks = block_table[req_idx, positions // block_size].to(torch.int64)
        flat_indices = physical_blocks * block_size + positions % block_size
        output[req_idx, offset : offset + gather_len].copy_(
            _gather_fp8_ds_mla_rows(kv_cache, flat_indices, block_size)
        )


class DeepseekV4SM70SparseBackend(DeepseekV4FlashMLASparseBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]

    @staticmethod
    def get_name() -> str:
        return "V4_SM70_SPARSE_REFERENCE"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4SparseMLAAttentionImpl"]:
        return DeepseekV4SM70SparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 7


class DeepseekV4SM70SparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """Correctness-first sparse MLA implementation for FP16 Volta serving."""

    backend_cls = DeepseekV4SM70SparseBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        for padded in (8, 16, 32, 64, 128):
            if num_heads <= padded:
                return padded
        raise ValueError(
            f"DeepSeek V4 SM70 supports at most 128 heads, got {num_heads}"
        )

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del kv
        assert q.dtype == torch.float16
        assert output.shape == q.shape and output.dtype == q.dtype
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            swa_only = layer.compress_ratio <= 1
            compressed_len = (
                0
                if swa_only
                else (layer.max_model_len + layer.compress_ratio - 1)
                // layer.compress_ratio
            )
            workspace_len = (
                compressed_len + layer.window_size + layer.max_num_batched_tokens
            )
            current_workspace_manager().get_simultaneous(
                ((cls.PREFILL_CHUNK_SIZE, workspace_len, q.shape[-1]), q.dtype),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        sparse_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            cls._forward_prefill(
                layer,
                q[num_decode_tokens:],
                positions[num_decode_tokens:],
                layer.kv_cache if layer.compress_ratio > 1 else None,
                layer.swa_cache_layer.kv_cache,
                output[num_decode_tokens:],
                sparse_metadata,
                swa_metadata,
            )
        if swa_metadata.num_decodes > 0:
            cls._forward_decode(
                layer,
                q[:num_decode_tokens],
                layer.kv_cache if layer.compress_ratio > 1 else None,
                swa_metadata,
                sparse_metadata,
                layer.compress_ratio <= 1,
                output[:num_decode_tokens],
            )

    @classmethod
    def _forward_decode(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        gathered, invalid = _gather_scope(
            layer.swa_cache_layer.kv_cache,
            swa_metadata.decode_swa_indices,
            swa_metadata.decode_swa_lens,
            swa_metadata.block_size,
        )

        if not swa_only:
            assert kv_cache is not None and attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                topk_indices, topk_lens = compute_global_topk_indices_and_lens(
                    layer.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    attn_metadata.block_size // layer.compress_ratio,
                    swa_metadata.is_valid_token[:num_decode_tokens],
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
            assert topk_indices is not None
            extra, extra_invalid = _gather_scope(
                kv_cache,
                topk_indices,
                topk_lens,
                attn_metadata.block_size // layer.compress_ratio,
            )
            gathered = torch.cat((gathered, extra), dim=1)
            invalid = torch.cat((invalid, extra_invalid), dim=1)

        result = _reference_sparse_attention(
            q, gathered, invalid, layer.scale, layer.attn_sink
        )
        output.copy_(result)

    @classmethod
    def _forward_prefill(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        del positions
        swa_only = attn_metadata is None
        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc = swa_metadata.query_start_loc
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc is not None and query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            assert attn_metadata is not None
            if layer.compress_ratio == 4:
                assert layer.topk_indices_buffer is not None
                topk_indices = layer.topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
            else:
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            compressed_len = (
                layer.max_model_len + layer.compress_ratio - 1
            ) // layer.compress_ratio
        else:
            assert layer.topk_indices_buffer is not None
            topk_indices = layer.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            compressed_len = 0

        workspace_len = (
            compressed_len + layer.window_size + layer.max_num_batched_tokens
        )
        workspace = current_workspace_manager().get_simultaneous(
            ((cls.PREFILL_CHUNK_SIZE, workspace_len, q.shape[-1]), q.dtype),
        )[0]
        num_chunks = (num_prefills + cls.PREFILL_CHUNK_SIZE - 1) // (
            cls.PREFILL_CHUNK_SIZE
        )
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * cls.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + cls.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None and compressed_k_cache is not None
                _gather_sequence_cache(
                    workspace[:chunk_size],
                    compressed_k_cache,
                    seq_lens[chunk_start:chunk_end] // layer.compress_ratio,
                    None,
                    attn_metadata.block_table[num_decodes:][chunk_start:chunk_end],
                    attn_metadata.block_size // layer.compress_ratio,
                    0,
                )
            _gather_sequence_cache(
                workspace[:chunk_size],
                swa_k_cache,
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                swa_metadata.block_table[num_decodes:][chunk_start:chunk_end],
                swa_metadata.block_size,
                compressed_len,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                layer.window_size,
                layer.compress_ratio,
                top_k,
                workspace_len,
                compressed_len,
            )
            gathered, invalid = _gather_scope_from_dense(
                workspace[:chunk_size].reshape(-1, q.shape[-1]),
                combined_indices,
                combined_lens,
            )
            result = _reference_sparse_attention(
                q[query_start:query_end],
                gathered,
                invalid,
                layer.scale,
                layer.attn_sink,
            )
            output[query_start:query_end].copy_(result)


def _gather_scope_from_dense(
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense_indices = indices.reshape(indices.shape[0], -1)
    invalid = (dense_indices < 0) | (dense_indices >= kv.shape[0])
    if lengths is not None:
        offsets = torch.arange(dense_indices.shape[-1], device=indices.device)
        invalid |= offsets.unsqueeze(0) >= lengths.reshape(-1, 1)
    safe = dense_indices.clamp(0, max(kv.shape[0] - 1, 0))
    gathered = kv.index_select(0, safe.reshape(-1)).view(
        *dense_indices.shape, kv.shape[-1]
    )
    gathered.masked_fill_(invalid.unsqueeze(-1), 0)
    return gathered, invalid


def _apply_gptj_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply interleaved GPT-J RoPE to the final ``rope_dim`` features."""
    if rope_dim == 0 or x.numel() == 0:
        return x
    half = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    cos = cache[:, :half].to(torch.float32)
    sin = cache[:, half : 2 * half].to(torch.float32)
    view_shape = (positions.shape[0],) + (1,) * (x.ndim - 2) + (half,)
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)

    x_float = x.to(torch.float32)
    rope = x_float[..., nope_dim:]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    if inverse:
        rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
    else:
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    result = x_float.clone()
    result[..., nope_dim:] = rotated.flatten(-2)
    return result.to(x.dtype)


def compress_norm_rope_store_sm70(
    *,
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
) -> None:
    """Reference compressor used to establish a correct FP16 SM70 boot."""
    if num_actual == 0:
        return
    positions = positions[:num_actual]
    requests = token_to_req_indices[:num_actual].to(torch.long)
    state_slots = state_slot_mapping[:num_actual]
    kv_slots = kv_slot_mapping[:num_actual]
    boundary = (
        ((positions + 1) % compress_ratio == 0) & (state_slots >= 0) & (kv_slots >= 0)
    )
    window = (1 + int(overlap)) * compress_ratio
    # CUDA graphs require fixed-shape intermediates. Process every scheduled
    # row, but direct non-boundary writes to slot -1 so the store kernels skip
    # them in-device. Dummy rows use a valid request/window to avoid an
    # all-masked softmax while leaving the cache untouched.
    active_positions = torch.where(boundary, positions, window - 1).to(torch.long)
    active_requests = torch.where(boundary, requests, 0).clamp(
        0, block_table.shape[0] - 1
    )
    active_slots = torch.where(boundary, kv_slots, -1).to(torch.long)
    offsets = torch.arange(window, device=state_cache.device)
    source_positions = active_positions[:, None] - window + 1 + offsets
    valid = source_positions >= 0
    safe_positions = source_positions.clamp_min(0)
    physical_blocks = block_table[
        active_requests[:, None], safe_positions // block_size
    ].to(torch.long)
    states = state_cache[
        physical_blocks,
        safe_positions % block_size,
    ]

    feature_offsets = torch.arange(head_dim, device=state_cache.device)
    head_offsets = (offsets >= compress_ratio).to(torch.long) * head_dim
    feature_indices = head_offsets[None, :, None] + feature_offsets
    feature_indices = feature_indices.expand(num_actual, -1, -1)
    kv = torch.gather(states, 2, feature_indices).to(torch.float32)
    score = torch.gather(states, 2, feature_indices + state_width).to(torch.float32)
    score.masked_fill_(~valid.unsqueeze(-1), -torch.inf)
    kv.masked_fill_(~valid.unsqueeze(-1), 0.0)
    compressed = torch.sum(kv * torch.softmax(score, dim=1), dim=1)
    compressed = F.rms_norm(
        compressed,
        (head_dim,),
        weight=rms_norm_weight.to(torch.float32),
        eps=rms_norm_eps,
    ).to(torch.float16)
    compressed_positions = (active_positions // compress_ratio) * compress_ratio

    if head_dim == 512:
        # The existing fused op has a valid FP16 specialization for SM70. Its
        # Q output is discarded here; the KV branch applies RoPE, quantizes the
        # 448 NoPE values, and writes the packed fp8_ds_mla cache row.
        q_scratch = torch.zeros(
            compressed.shape[0],
            8,
            head_dim,
            dtype=torch.float16,
            device=compressed.device,
        )
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q_scratch,
            compressed,
            kv_cache.view(kv_cache.shape[0], -1),
            active_slots,
            compressed_positions,
            cos_sin_cache,
            8,
            rms_norm_eps,
            kv_cache.shape[1],
        )
        return

    assert head_dim == 128 and kv_cache.dtype == torch.float16
    rotated = _apply_gptj_rope(
        compressed,
        compressed_positions,
        cos_sin_cache,
        rope_head_dim,
    )
    cache_block_size = kv_cache.shape[1]
    _store_fp16_paged_rows_sm70_kernel[(num_actual,)](
        kv_cache,
        rotated,
        active_slots,
        kv_cache.shape[0] * cache_block_size,
        cache_stride_block=kv_cache.stride(0),
        cache_stride_token=kv_cache.stride(1),
        cache_block_size=cache_block_size,
        row_width=head_dim,
        num_warps=4,
    )


def _gather_fp16_paged_sequence(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    assert kv_cache.dtype == torch.float16
    if seq_len <= 0:
        return kv_cache.new_empty((0, kv_cache.shape[-1]))
    block_size = kv_cache.shape[1]
    positions = torch.arange(seq_len, dtype=torch.long, device=kv_cache.device)
    physical_blocks = block_table[positions // block_size].to(torch.long)
    return kv_cache[physical_blocks, positions % block_size]


def _indexer_logits_sm70(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("mhd,nd->mhn", q.to(torch.float32), k.to(torch.float32))
    logits = torch.sum(torch.relu(scores) * weights.to(torch.float32).unsqueeze(-1), 1)
    offsets = torch.arange(k.shape[0], device=k.device)
    valid = (offsets.unsqueeze(0) >= starts.reshape(-1, 1)) & (
        offsets.unsqueeze(0) < ends.reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -torch.inf)


def _write_indexer_topk(
    logits: torch.Tensor,
    output: torch.Tensor,
    topk_tokens: int,
) -> None:
    width = min(topk_tokens, logits.shape[-1])
    if width == 0:
        return
    values, indices = torch.topk(logits, width, dim=-1, sorted=False)
    indices = indices.to(torch.int32)
    indices.masked_fill_(~torch.isfinite(values), -1)
    output[:, :width].copy_(indices)


def _paged_indexer_logits_sm70(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    """Batched graph-safe decode logits over the FP16 paged indexer cache."""
    assert q.dtype == torch.float16 and kv_cache.dtype == torch.float16
    assert q.shape[1:] == (64, 128)
    num_queries = q.shape[0]
    cache_block_size = kv_cache.shape[1]
    max_context = block_table.shape[1] * cache_block_size
    logits = torch.full(
        (num_queries, max_context),
        -torch.inf,
        dtype=torch.float32,
        device=q.device,
    )
    # Volta favors narrow N tiles here: 16 columns with eight warps was the
    # fastest measured geometry from 32 through 8,192 context tokens at both
    # batch 1 and batch 4.
    block_n = 16
    _paged_indexer_logits_sm70_kernel[(num_queries, triton.cdiv(max_context, block_n))](
        logits,
        q,
        kv_cache,
        weights,
        seq_lens,
        block_table,
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_table.stride(0),
        seq_lens.stride(0),
        cache_block_size=cache_block_size,
        max_context=max_context,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        BLOCK_N=block_n,
        num_warps=8,
    )
    return logits


def sparse_attn_indexer_sm70(
    *,
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    """Reference Lightning Indexer over a plain FP16 paged K cache."""
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

    attn_metadata = get_forward_context().attn_metadata
    topk_indices_buffer[: hidden_states.shape[0]].fill_(-1)
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer
    metadata = attn_metadata[k_cache_prefix]
    assert isinstance(metadata, DeepseekV32IndexerMetadata)

    if metadata.num_prefills > 0:
        assert metadata.prefill is not None
        for chunk in metadata.prefill.chunks:
            sequence_rows: list[torch.Tensor] = []
            for request_idx in range(chunk.num_reqs):
                seq_start = int(chunk.cu_seq_lens[request_idx].item())
                seq_end = int(chunk.cu_seq_lens[request_idx + 1].item())
                sequence_rows.append(
                    _gather_fp16_paged_sequence(
                        kv_cache,
                        chunk.block_table[request_idx],
                        seq_end - seq_start,
                    )
                )
            k = torch.cat(sequence_rows, dim=0)
            q_slice = q[chunk.token_start : chunk.token_end]
            weight_slice = weights[chunk.token_start : chunk.token_end]
            logits = _indexer_logits_sm70(
                q_slice,
                k,
                weight_slice,
                chunk.cu_seqlen_ks.to(q.device),
                chunk.cu_seqlen_ke.to(q.device),
            )
            _write_indexer_topk(
                logits,
                topk_indices_buffer[chunk.token_start : chunk.token_end],
                topk_tokens,
            )

    if metadata.num_decodes > 0:
        assert metadata.decode is not None
        decode = metadata.decode
        num_decode_tokens = metadata.num_decode_tokens
        assert decode.decode_lens.shape[0] == num_decode_tokens, (
            "SM70 graph decode requires one token per indexer request"
        )
        seq_lens = decode.seq_lens.reshape(num_decode_tokens, -1)[:, :1]
        logits = _paged_indexer_logits_sm70(
            q[:num_decode_tokens],
            kv_cache,
            weights[:num_decode_tokens],
            seq_lens,
            decode.block_table,
        )
        topk_output = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        if topk_tokens in (512, 1024, 2048):
            (topk_workspace,) = current_workspace_manager().get_simultaneous(
                ((1024 * 1024,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_output,
                topk_workspace,
                topk_tokens,
                logits.shape[1],
            )
        else:
            _write_indexer_topk(logits, topk_output, topk_tokens)

    return topk_indices_buffer


def sm70_inv_rope_einsum(
    rotary_emb: torch.nn.Module,
    o: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int,
    n_local_groups: int,
    o_lora_rank: int,
    wo_a: torch.nn.Module,
) -> torch.Tensor:
    """FP16 inverse-RoPE and block-FP8 WO_A reference for Volta."""
    o_ref = _apply_gptj_rope(
        o,
        positions,
        rotary_emb.cos_sin_cache,
        rope_head_dim,
        inverse=True,
    ).view(o.shape[0], n_local_groups, -1)
    hidden_dim = o_ref.shape[-1]

    if getattr(wo_a, "sm70_fp8_grouped_bmm", False):
        output = torch.empty(
            o.shape[0],
            n_local_groups,
            o_lora_rank,
            dtype=o.dtype,
            device=o.device,
        )
        meta = wo_a.sm70_fp8_grouped_meta
        for group in range(n_local_groups):
            sm70_ops.fp8_gemm_sm70_out(
                output[:, group, :],
                o_ref[:, group, :],
                wo_a.weight[group],
                wo_a.weight_scale_inv[group],
                128,
                int(meta[group, 0].item()),
                int(meta[group, 1].item()),
                False,
            )
        return output

    weight = wo_a.weight.view(n_local_groups, o_lora_rank, hidden_dim)
    if weight.dtype in (torch.float8_e4m3fn, torch.uint8):
        weight = _e4m3fn_fp16_lut(weight.device)[
            weight.view(torch.uint8).to(torch.long)
        ]
    else:
        weight = weight.to(torch.float16)

    if hasattr(wo_a, "weight_scale_inv"):
        scale = wo_a.weight_scale_inv.view(
            n_local_groups,
            -1,
            wo_a.weight_scale_inv.shape[-1],
        )
        if scale.dtype == torch.float8_e8m0fnu:
            scale = torch.exp2(scale.view(torch.uint8).to(torch.float32) - 127.0)
        else:
            scale = scale.to(torch.float32)
        row_blocks, col_blocks = scale.shape[-2:]
        row_block = (o_lora_rank + row_blocks - 1) // row_blocks
        col_block = (hidden_dim + col_blocks - 1) // col_blocks
        scale = torch.repeat_interleave(scale, row_block, dim=-2)[:, :o_lora_rank, :]
        scale = torch.repeat_interleave(scale, col_block, dim=-1)[:, :, :hidden_dim]
        weight = (weight.to(torch.float32) * scale).to(torch.float16)

    return torch.einsum("tgd,grd->tgr", o_ref, weight)
