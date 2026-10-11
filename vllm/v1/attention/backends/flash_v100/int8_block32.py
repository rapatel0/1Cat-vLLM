# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signed INT8 block-32 KV-cache routes for Flash-V100.

The cache is a ``[pages, page_bytes]`` INT8 tensor holding signed payloads,
separate FP16 page/head/32-channel scales and a page owner word. Every route in
this module reads or writes that layout through the native INT8 operators:

- cache write (``reshape_and_cache``),
- single-token decode (XQA with block scales, or the register decoder),
- prefix prefill, either as small-query decode rows, as a native INT8 prefill,
  or by expanding the active pages to an FP16 workspace for large chunks.

No-prefix prefill never reads the cache: it uses the fresh FP16 K/V.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.attention.backends.flash_v100 import dense_prefill as _dense_prefill
from vllm.v1.attention.backends.flash_v100 import kv_layout as _kv_layout
from vllm.v1.attention.backends.flash_v100 import masks as _masks
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

# Chunks shorter than this stay on the small-query INT8 kernels; longer chunks
# amortize the page expansion into the FP16 prefill kernel.
BRIDGE_MIN_QUERY_TOKENS = 32


@dataclass(frozen=True)
class Int8Block32Ops:
    """Native callables and layer geometry; no backend receiver."""

    scale: float
    num_kv_heads: int
    head_size: int
    decode_paged: Any
    prefill_paged: Any
    reshape_and_cache: Any
    paged_kv_to_fp16: Any
    xqa: Any = None
    fp16_prefill_paged: Any = None
    use_prefill_paged: bool = False
    window_size: Callable[[bool], tuple[int, int]] = lambda causal: (-1, -1)

    @property
    def available(self) -> bool:
        return (
            self.decode_paged is not None
            and self.prefill_paged is not None
            and self.reshape_and_cache is not None
        )


class Int8Block32Attention:
    def __init__(self, ops: Int8Block32Ops):
        self.ops = ops

    def _split(self, kv_cache: torch.Tensor):
        return _kv_layout.split_int8_block32_kv_cache(
            kv_cache,
            num_kv_heads=self.ops.num_kv_heads,
            head_size=self.ops.head_size,
        )

    def update_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.ops.reshape_and_cache is None:
            raise RuntimeError("FLASH_ATTN_V100 INT8 block cache writer is unavailable")
        key_cache, value_cache, key_scales, value_scales, page_owners = self._split(
            kv_cache
        )
        self.ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            page_owners,
            slot_mapping,
        )

    def decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        window_size = self.ops.window_size(True)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        out_view = output[:num_actual_tokens]
        if query.shape[0] == 0:
            return output
        if window_size != (-1, -1):
            raise RuntimeError(
                "FLASH_ATTN_V100 INT8 block cache does not support "
                "sliding-window decode"
            )
        if query.shape[0] != attn_metadata.seq_lens.shape[0]:
            raise RuntimeError(
                "FLASH_ATTN_V100 INT8 block cache supports one query token per request"
            )
        if self.ops.decode_paged is None:
            raise RuntimeError("FLASH_ATTN_V100 INT8 block decoder is unavailable")
        key_cache, value_cache, key_scales, value_scales, _ = self._split(kv_cache)
        if self.ops.xqa is not None:
            self.ops.xqa(
                query,
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                softmax_scale=self.ops.scale,
                out=out_view,
                kv_cache_dtype="int8_block32",
                k_scale=1.0,
                v_scale=1.0,
                window_size=window_size,
                max_seq_len_hint=getattr(
                    attn_metadata, "flash_v100_decode_max_seq_len_hint", None
                ),
                workspace_seq_capacity_hint=getattr(
                    attn_metadata,
                    "flash_v100_decode_workspace_seq_capacity_hint",
                    None,
                ),
                active_num_partitions=getattr(
                    attn_metadata, "flash_v100_decode_active_num_partitions", None
                ),
                partition_size_hint=256,
                batch_context_routing=False,
                key_scales=key_scales,
                value_scales=value_scales,
            )
            _routing.record_route(_routing.ROUTE_SPECS["decode_int8_block32_xqa"].name)
            return output
        self.ops.decode_paged(
            query,
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            out_view,
            self.ops.scale,
        )
        _routing.record_route(
            _routing.ROUTE_SPECS["decode_int8_block32_register_fallback"].name
        )
        return output

    def _bridge_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        capacity_hint: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Expand active signed INT8 pages into a graph-stable FP16 workspace."""
        if (
            self.ops.paged_kv_to_fp16 is None
            or block_table.ndim != 2
            or block_table.shape[0] != 1
            or seq_lens.ndim != 1
            or seq_lens.shape[0] != 1
        ):
            return None

        input_block_size = int(key_cache.shape[1])
        input_capacity = int(block_table.shape[1]) * input_block_size
        if capacity_hint is None or capacity_hint <= 0:
            if torch.cuda.is_current_stream_capturing():
                return None
            capacity_hint = int(seq_lens.max().item())
        active_capacity = min(input_capacity, max(1, capacity_hint))
        active_input_blocks = min(
            int(block_table.shape[1]),
            _masks.cdiv_int(active_capacity, input_block_size),
        )
        if active_input_blocks <= 0:
            return None
        active_block_table = block_table[:, :active_input_blocks]
        required_blocks = _masks.cdiv_int(
            active_input_blocks * input_block_size,
            _dense_prefill.FP8_PREFILL_BRIDGE_PAGE_SIZE,
        )
        workspace = _dense_prefill.get_fp8_prefill_bridge_workspace(
            key_cache, required_blocks
        )
        if workspace is None:
            return None
        key_out, value_out, output_block_table = workspace
        self.ops.paged_kv_to_fp16(
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            active_block_table,
            seq_lens,
            key_out,
            value_out,
        )
        return key_out, value_out, output_block_table

    def _run_prefill_bridge(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        seq_len: int,
        out: torch.Tensor,
    ) -> bool:
        if (
            not self.ops.use_prefill_paged
            or self.ops.fp16_prefill_paged is None
            or query.shape[0] < BRIDGE_MIN_QUERY_TOKENS
        ):
            return False
        bridged = self._bridge_cache(
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            block_table,
            seq_lens,
            capacity_hint=seq_len,
        )
        if bridged is None:
            return False
        key_out, value_out, output_block_table = bridged
        query_batched = query.unsqueeze(0)
        out_batched = out.unsqueeze(0)
        result = self.ops.fp16_prefill_paged(
            query_batched,
            key_out,
            value_out,
            output_block_table,
            seq_lens,
            softmax_scale=self.ops.scale,
            out=out_batched,
            kv_cache_dtype="auto",
            k_scale=1.0,
            v_scale=1.0,
            causal=True,
            window_size=(-1, -1),
        )
        if result.data_ptr() != out_batched.data_ptr():
            out_batched.copy_(result)
        _routing.record_route(
            _routing.ROUTE_SPECS["prefill_prefix_int8_block32_bridge_fp16"].name
        )
        return True

    def _run_multi_request_prefill_bridge(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        out: torch.Tensor,
    ) -> bool:
        """Split mixed bulk-prefill batches into bounded exact request routes.

        Only batches whose every request is a bulk chunk qualify; any short
        request returns False so the caller uses the small-query INT8 kernels.
        """
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
        block_table = attn_metadata.block_table
        if (
            query_start_loc_cpu is None
            or seq_lens_cpu is None
            or query_start_loc_cpu.ndim != 1
            or seq_lens_cpu.ndim != 1
            or block_table.ndim != 2
        ):
            return False

        query_starts = [int(value) for value in query_start_loc_cpu.tolist()]
        sequence_lengths = [int(value) for value in seq_lens_cpu.tolist()]
        num_requests = len(query_starts) - 1
        if (
            num_requests <= 1
            or num_requests > int(block_table.shape[0])
            or num_requests > len(sequence_lengths)
            or query_starts[0] != 0
            or query_starts[-1] != int(query.shape[0])
        ):
            return False

        input_block_size = int(key_cache.shape[1])
        max_num_blocks = int(block_table.shape[1])
        max_required_blocks = 0
        for request_idx in range(num_requests):
            query_len = query_starts[request_idx + 1] - query_starts[request_idx]
            seq_len = sequence_lengths[request_idx]
            if (
                query_starts[request_idx] < 0
                or query_len < BRIDGE_MIN_QUERY_TOKENS
                or seq_len < query_len
            ):
                return False
            active_input_blocks = min(
                max_num_blocks, _masks.cdiv_int(seq_len, input_block_size)
            )
            required_blocks = _masks.cdiv_int(
                active_input_blocks * input_block_size,
                _dense_prefill.FP8_PREFILL_BRIDGE_PAGE_SIZE,
            )
            if required_blocks <= 0:
                return False
            max_required_blocks = max(max_required_blocks, required_blocks)

        if (
            _dense_prefill.get_fp8_prefill_bridge_workspace(
                key_cache, max_required_blocks
            )
            is None
        ):
            return False

        for request_idx in range(num_requests):
            query_start = query_starts[request_idx]
            query_end = query_starts[request_idx + 1]
            if not self._run_prefill_bridge(
                query[query_start:query_end],
                key_cache,
                value_cache,
                key_scales,
                value_scales,
                block_table[request_idx : request_idx + 1],
                attn_metadata.seq_lens[request_idx : request_idx + 1],
                seq_len=sequence_lengths[request_idx],
                out=out[query_start:query_end],
            ):
                return False

        _routing.record_route(
            _routing.ROUTE_SPECS[
                "prefill_prefix_int8_block32_bridge_fp16_multi_request"
            ].name
        )
        return True

    def small_query_prefill_as_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run causal prefix queries (speculative verification, chunks) over the cache."""
        if not bool(getattr(attn_metadata, "causal", True)):
            raise RuntimeError(
                "FLASH_ATTN_V100 INT8 block cache small-query prefill "
                "requires causal attention"
            )
        if self.ops.decode_paged is None or self.ops.prefill_paged is None:
            raise RuntimeError(
                "FLASH_ATTN_V100 INT8 block cache attention ops are unavailable"
            )

        key_cache, value_cache, key_scales, value_scales, _ = self._split(kv_cache)
        num_query_tokens = min(
            int(attn_metadata.num_actual_tokens),
            int(query.shape[0]),
            int(output.shape[0]),
        )
        query = query[:num_query_tokens]
        out_view = output[:num_query_tokens]

        if (
            num_query_tokens >= BRIDGE_MIN_QUERY_TOKENS
            and not torch.cuda.is_current_stream_capturing()
        ):
            if attn_metadata.block_table.shape[0] == 1:
                seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
                seq_len = (
                    int(seq_lens_cpu[0].item())
                    if seq_lens_cpu is not None
                    else int(attn_metadata.seq_lens[0].item())
                )
                if self._run_prefill_bridge(
                    query,
                    key_cache,
                    value_cache,
                    key_scales,
                    value_scales,
                    attn_metadata.block_table[:1],
                    attn_metadata.seq_lens[:1],
                    seq_len=seq_len,
                    out=out_view,
                ):
                    return output
            elif self._run_multi_request_prefill_bridge(
                query,
                key_cache,
                value_cache,
                key_scales,
                value_scales,
                attn_metadata,
                out=out_view,
            ):
                return output

        persistent_block_table = getattr(
            attn_metadata, "smallq_decode_block_table", None
        )
        persistent_seq_lens = getattr(attn_metadata, "smallq_decode_seq_lens", None)
        if (
            persistent_block_table is not None
            and persistent_seq_lens is not None
            and int(persistent_block_table.shape[0]) >= num_query_tokens
            and int(persistent_seq_lens.shape[0]) >= num_query_tokens
            and not _kv_layout.metadata_expects_more_query_tokens_than_available(
                attn_metadata, num_query_tokens
            )
        ):
            self.ops.decode_paged(
                query,
                key_cache,
                value_cache,
                key_scales,
                value_scales,
                persistent_block_table[:num_query_tokens],
                persistent_seq_lens[:num_query_tokens],
                out_view,
                self.ops.scale,
            )
            _routing.record_route(
                _routing.ROUTE_SPECS["prefill_smallq_int8_block32_register"].name
            )
            return output

        if _routing.is_cuda_graph_capturing(query):
            raise RuntimeError(
                "FLASH_ATTN_V100 INT8 small-query prefill entered CUDA graph "
                "capture without persistent decode metadata"
            )

        query_start_loc = _kv_layout.normalize_query_start_loc_for_available_tokens(
            attn_metadata.query_start_loc, num_query_tokens
        ).to(
            device=attn_metadata.seq_lens.device,
            dtype=attn_metadata.query_start_loc.dtype,
        )
        self.ops.prefill_paged(
            query,
            key_cache,
            value_cache,
            key_scales,
            value_scales,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            query_start_loc,
            out_view,
            self.ops.scale,
        )
        _routing.record_route(
            _routing.ROUTE_SPECS["prefill_prefix_int8_block32_register"].name
        )
        return output
