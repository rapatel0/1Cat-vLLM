# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact packed DFlash2 verifier over the signed INT8 block-32 KV cache.

The native operator verifies one request's q8/q16 rows, or up to
``max_requests`` packed q8 requests, reusing each paged-KV scan across a GQA
group. The INT8 executor asks this object whether a batch qualifies and runs
it; everything else falls back to the generic INT8 kernels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_v100 import routing as _routing
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

logger = init_logger("vllm.v1.attention.backends.flash_attn_v100")

# Persistent per-(requests, device) partitions. A captured CUDA graph replays the
# tensor's address, so it must outlive every graph that used it.
_packed_verify_offsets: dict[tuple[int, int], torch.Tensor] = {}


def packed_verify_offsets(num_requests: int, device: torch.device) -> torch.Tensor:
    """The constant [0, 8, ..., 8N] partition for N packed q8 verifier requests.

    vLLM pads query_start_loc by repeating the real token count, so a graph
    captured for more requests than are live would replay a partition whose
    terminal offset no longer equals the packed query rows. The packed route
    always carries exactly eight query tokens per request, and padded request
    slots have zero sequence length, so this partition is always valid.
    """
    key = (num_requests, device.index if device.index is not None else -1)
    offsets = _packed_verify_offsets.get(key)
    if offsets is None:
        offsets = torch.arange(
            0, (num_requests + 1) * 8, 8, dtype=torch.int32, device=device
        )
        _packed_verify_offsets[key] = offsets
    return offsets


def query_partition_is_valid(
    query_start_loc_cpu: torch.Tensor | None,
    *,
    num_requests: int,
    num_query_tokens: int,
    max_query_tokens: int,
) -> bool:
    if (
        num_requests <= 0
        or num_query_tokens <= 0
        or max_query_tokens <= 0
        or query_start_loc_cpu is None
        or query_start_loc_cpu.device.type != "cpu"
        or query_start_loc_cpu.dtype != torch.int32
        or query_start_loc_cpu.ndim != 1
        or query_start_loc_cpu.numel() != num_requests + 1
    ):
        return False
    offsets: list[int] = query_start_loc_cpu.tolist()
    if offsets[0] != 0 or offsets[-1] != num_query_tokens:
        return False
    return all(
        0 < end - start <= max_query_tokens
        for start, end in zip(offsets, offsets[1:], strict=False)
    )


@dataclass(frozen=True)
class Int8GroupedVerify:
    """Policy, native entry and layer geometry of the packed INT8 verifier."""

    op: Any = None
    scale: float = 1.0
    window_size: Callable[[bool], tuple[int, int]] = lambda causal: (-1, -1)
    enabled: bool = False
    batched_enabled: bool = False
    request_major_abi: int = 0
    max_query_tokens: int = 8
    max_requests: int = 1
    min_model_len: int = 0
    # The FP16 page-expansion op must exist for the INT8 source to be admitted.
    bridge_available: bool = False

    def allowed(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        num_query_tokens: int,
    ) -> bool:
        """Gate the verifier on its hardware and tensor-layout contract."""
        block_table = getattr(attn_metadata, "block_table", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        num_requests = (
            int(query_start_loc.numel()) - 1 if query_start_loc is not None else 0
        )
        query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
        if (
            query_start_loc_cpu is None
            and query_start_loc is not None
            and query_start_loc.device.type == "cpu"
        ):
            query_start_loc_cpu = query_start_loc
        max_query_len = int(getattr(attn_metadata, "max_query_len", 0))
        query_partition_valid = query_partition_is_valid(
            query_start_loc_cpu,
            num_requests=num_requests,
            num_query_tokens=num_query_tokens,
            max_query_tokens=16 if num_requests == 1 and num_query_tokens > 8 else 8,
        )
        source = bool(
            key_cache.dtype == torch.int8
            and value_cache.dtype == torch.int8
            and self.bridge_available
        )
        single_request_shape = bool(
            num_requests == 1
            and num_query_tokens in (8, 16)
            and num_query_tokens <= self.max_query_tokens
        )
        packed_shape = bool(max_query_len == 8 and num_query_tokens == num_requests * 8)
        multi_request_shape = bool(
            source and 1 < num_requests <= self.max_requests and packed_shape
        )
        batched_request_shape = bool(
            self.batched_enabled
            and self.request_major_abi >= 1
            and num_requests in (2, 4, 8)
            and packed_shape
        )
        allowed = bool(
            self.enabled
            and (single_request_shape or batched_request_shape)
            and self.op is not None
            and source
            and getattr(attn_metadata, "is_dflash_selector_target", False)
            and getattr(attn_metadata, "max_model_len", 0) >= self.min_model_len
            and getattr(attn_metadata, "causal", True)
            and self.window_size(True) == (-1, -1)
            and (single_request_shape or multi_request_shape or batched_request_shape)
            and tuple(query.shape) == (num_query_tokens, 6, 256)
            and query.dtype == torch.float16
            and query.is_contiguous()
            and key_cache.ndim == 4
            and value_cache.ndim == 4
            and key_cache.device == query.device
            and value_cache.device == query.device
            # The runtime-stride operator is exact for the aligned hybrid-cache
            # pages of both the block-8 (1648/3296) and q15 (1728/3456) layouts.
            and key_cache.shape[1] in (1648, 1728, 3296, 3456)
            and tuple(key_cache.shape[2:]) == (1, 256)
            and tuple(value_cache.shape) == tuple(key_cache.shape)
            and key_cache.stride(-1) == 1
            and value_cache.stride(-1) == 1
            and block_table is not None
            and block_table.ndim == 2
            and block_table.shape[0] >= num_requests
            and block_table.device == query.device
            and block_table.dtype == torch.int32
            and block_table.is_contiguous()
            and seq_lens is not None
            and seq_lens.ndim == 1
            and seq_lens.shape[0] >= num_requests
            and seq_lens.device == query.device
            and seq_lens.dtype == torch.int32
            and seq_lens.is_contiguous()
            and query_start_loc is not None
            and query_start_loc.ndim == 1
            and query_start_loc.device == query.device
            and query_start_loc.dtype == torch.int32
            and query_start_loc.is_contiguous()
            and query_partition_valid
        )
        if self.enabled and not allowed:
            logger.info_once(
                "FLASH_ATTN_V100 DFlash2 INT8 grouped verifier gate rejected: "
                "op=%s marker=%s max_model_len=%s min_model_len=%s "
                "causal=%s actual=%d max_query=%d native_max_q=%d requests=%d "
                "native_max_requests=%d q=%s/%s k=%s/%s block_table=%s seq_lens=%s.",
                self.op is not None,
                getattr(attn_metadata, "is_dflash_selector_target", False),
                getattr(attn_metadata, "max_model_len", None),
                self.min_model_len,
                getattr(attn_metadata, "causal", True),
                num_query_tokens,
                max_query_len,
                self.max_query_tokens,
                num_requests,
                self.max_requests,
                tuple(query.shape),
                query.dtype,
                tuple(key_cache.shape),
                key_cache.dtype,
                None if block_table is None else tuple(block_table.shape),
                None if seq_lens is None else tuple(seq_lens.shape),
                scope="process",
                key="flash_v100.int8_block32_grouped_verify_gate",
            )
        return allowed

    def call(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        *,
        out: torch.Tensor,
    ) -> None:
        num_requests = int(attn_metadata.query_start_loc.numel()) - 1
        logger.info_once(
            "FLASH_ATTN_V100 DFlash2 direct signed INT8 grouped verifier active "
            "for up to %d requests.",
            self.max_requests,
            scope="process",
            key="flash_v100.int8_block32_grouped_verify",
        )
        self.op(
            query,
            key_cache,
            value_cache,
            attn_metadata.block_table[:num_requests],
            attn_metadata.seq_lens[:num_requests],
            softmax_scale=self.scale,
            out=out,
            kv_cache_dtype="int8_block32",
            k_scale=1.0,
            v_scale=1.0,
            one_pass=True,
            query_start_loc=(
                packed_verify_offsets(num_requests, query.device)
                if num_requests > 1
                else None
            ),
            _query_partition_validated=True,
        )
        _routing.record_route(
            _routing.ROUTE_SPECS[
                "prefill_smallq_dflash2_int8_block32_grouped_verify"
            ].name
        )

    def request_allowed(
        self, attn_metadata: TritonAttentionMetadata, query_len: int
    ) -> bool:
        """Whether one request of a mixed batch can use the single-request form."""
        return bool(
            query_len in (8, 16)
            and query_len <= self.max_query_tokens
            and self.enabled
            and self.op is not None
            and getattr(attn_metadata, "is_dflash_selector_target", False)
            and getattr(attn_metadata, "max_model_len", 0) >= self.min_model_len
            and getattr(attn_metadata, "causal", True)
            and self.window_size(True) == (-1, -1)
        )

    def call_request(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        self.op(
            query,
            key_cache,
            value_cache,
            block_table,
            seq_lens,
            softmax_scale=self.scale,
            out=out,
            kv_cache_dtype="int8_block32",
            k_scale=1.0,
            v_scale=1.0,
            one_pass=True,
        )
        _routing.record_route(
            _routing.ROUTE_SPECS[
                "prefill_smallq_dflash2_int8_block32_grouped_verify"
            ].name
        )


def from_attention(
    attention: Any, *, grouped_op_count: int, bridge_available: bool
) -> Int8GroupedVerify:
    """Read the captured verifier policy of an attention implementation."""
    get = partial(getattr, attention)
    return Int8GroupedVerify(
        op=get("flash_attn_grouped_verify_paged", None),
        scale=attention.scale,
        window_size=attention._flash_v100_window_size,
        enabled=bool(get("use_dflash2_grouped_verify", False)),
        batched_enabled=bool(get("use_dflash2_batched_grouped_verify", False)),
        request_major_abi=int(
            get("dflash2_grouped_verify_request_major_abi_version", 0)
        ),
        max_query_tokens=int(get("dflash2_grouped_verify_max_query_tokens", 8)),
        max_requests=grouped_op_count,
        min_model_len=int(get("dflash2_grouped_verify_min_model_len", 0)),
        bridge_available=bridge_available,
    )
