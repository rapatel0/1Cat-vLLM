# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional DeepSeek-V4 sparse-MLA fallbacks for SM70 GPUs."""

from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch.nn.functional as F

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


def _gather_fp8_ds_mla_rows(
    kv_cache: torch.Tensor,
    flat_indices: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather packed FP8-DS-MLA rows and dequantize them to FP16."""
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


def _reference_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    invalid: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None,
) -> torch.Tensor:
    """FP32 softmax reference over already-gathered FP16 KV rows."""
    q_float = q.to(torch.float32)
    kv_float = kv.to(torch.float32)
    scores = torch.bmm(q_float, kv_float.transpose(1, 2)) * scale
    scores.masked_fill_(invalid.unsqueeze(1), -torch.inf)

    row_max = scores.amax(dim=-1)
    if attn_sink is not None:
        sink = attn_sink[: q.shape[1]].to(torch.float32).view(1, -1)
        row_max = torch.maximum(row_max, sink)
    else:
        sink = None
    finite_max = torch.where(torch.isfinite(row_max), row_max, 0.0)
    probabilities = torch.exp(scores - finite_max.unsqueeze(-1))
    probabilities.masked_fill_(invalid.unsqueeze(1), 0.0)
    denominator = probabilities.sum(dim=-1)
    if sink is not None:
        denominator = denominator + torch.exp(sink - finite_max)
    probabilities = probabilities / denominator.clamp_min(1e-30).unsqueeze(-1)
    return torch.bmm(probabilities, kv_float).to(q.dtype)


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
    positions = positions[:num_actual]
    requests = token_to_req_indices[:num_actual].to(torch.long)
    state_slots = state_slot_mapping[:num_actual]
    kv_slots = kv_slot_mapping[:num_actual]
    boundary = (
        ((positions + 1) % compress_ratio == 0) & (state_slots >= 0) & (kv_slots >= 0)
    )
    active = torch.nonzero(boundary, as_tuple=False).flatten()
    if active.numel() == 0:
        return

    active_positions = positions.index_select(0, active).to(torch.long)
    active_requests = requests.index_select(0, active)
    active_slots = kv_slots.index_select(0, active).to(torch.long)
    window = (1 + int(overlap)) * compress_ratio
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
    feature_indices = feature_indices.expand(active.shape[0], -1, -1)
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
    kv_cache[
        active_slots // cache_block_size,
        active_slots % cache_block_size,
    ] = rotated


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
        cursor = 0
        for request_idx in range(decode.decode_lens.shape[0]):
            query_len = int(decode.decode_lens[request_idx].item())
            for query_offset in range(query_len):
                if decode.seq_lens.ndim == 2:
                    context_len = int(decode.seq_lens[request_idx, query_offset].item())
                else:
                    context_len = int(decode.seq_lens[request_idx].item())
                k = _gather_fp16_paged_sequence(
                    kv_cache,
                    decode.block_table[request_idx],
                    context_len,
                )
                logits = _indexer_logits_sm70(
                    q[cursor : cursor + 1],
                    k,
                    weights[cursor : cursor + 1],
                    torch.zeros(1, dtype=torch.int32, device=q.device),
                    torch.full((1,), context_len, dtype=torch.int32, device=q.device),
                )
                _write_indexer_topk(
                    logits,
                    topk_indices_buffer[cursor : cursor + 1],
                    topk_tokens,
                )
                cursor += 1
        assert cursor == metadata.num_decode_tokens

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
