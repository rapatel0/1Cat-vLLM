# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable query-head gathering for decode context parallel attention."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

try:
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:  # pragma: no cover - older PyTorch only.
    FakeTensor = None  # type: ignore[misc,assignment]

_DCPQueryGatherBuffers = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_dcp_query_gather_buffer_cache: dict[tuple[object, ...], _DCPQueryGatherBuffers] = {}


def _direct_query_gather_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_DCP_QUERY_GATHER_DIRECT", "1") != "0"


def _is_fake_tensor(tensor: torch.Tensor) -> bool:
    return FakeTensor is not None and isinstance(tensor, FakeTensor)


def _dcp_query_gather_buffers(
    query: torch.Tensor,
    world_size: int,
    group_name: str,
) -> _DCPQueryGatherBuffers:
    """Return non-aliasing storage keyed by group, stream, shape, and stride."""
    if query.ndim != 3:
        raise ValueError(
            "DCP query gather requires [tokens, heads, head_dim], got "
            f"{tuple(query.shape)}."
        )
    stream_id = (
        int(torch.cuda.current_stream(query.device).cuda_stream)
        if query.device.type == "cuda"
        else -1
    )
    key = (
        group_name,
        query.device,
        stream_id,
        world_size,
        query.dtype,
        tuple(query.shape),
        tuple(query.stride()),
    )
    buffers = _dcp_query_gather_buffer_cache.get(key)
    if buffers is None:
        tokens, local_heads, head_dim = query.shape
        buffers = (
            torch.empty(query.shape, dtype=query.dtype, device=query.device),
            torch.empty(
                (world_size, tokens, local_heads, head_dim),
                dtype=query.dtype,
                device=query.device,
            ),
            torch.empty(
                (tokens, world_size * local_heads, head_dim),
                dtype=query.dtype,
                device=query.device,
            ),
        )
        _dcp_query_gather_buffer_cache[key] = buffers
    return buffers


def _reformat_rank_major_query(
    rank_major: torch.Tensor,
    head_major: torch.Tensor,
) -> torch.Tensor:
    """Copy [rank, token, head, dim] into [token, rank*head, dim]."""
    if rank_major.ndim != 4 or head_major.ndim != 3:
        raise ValueError("Unexpected DCP query gather workspace rank.")
    world_size, tokens, local_heads, head_dim = rank_major.shape
    expected = (tokens, world_size * local_heads, head_dim)
    if tuple(head_major.shape) != expected:
        raise ValueError(
            f"DCP query gather output shape {tuple(head_major.shape)} != {expected}."
        )
    head_major.view(tokens, world_size, local_heads, head_dim).copy_(
        rank_major.permute(1, 0, 2, 3)
    )
    return head_major


def _direct_pynccl_communicator(
    query: torch.Tensor,
    cp_group: GroupCoordinator,
):
    if (
        not _direct_query_gather_enabled()
        or torch.compiler.is_compiling()
        or query.is_meta
        or _is_fake_tensor(query)
        or not query.is_cuda
        or query.ndim != 3
        or cp_group.world_size <= 1
    ):
        return None

    device_communicator = getattr(cp_group, "device_communicator", None)
    pynccl = getattr(device_communicator, "pynccl_comm", None)
    if pynccl is None or getattr(pynccl, "disabled", True):
        return None
    if getattr(pynccl, "world_size", None) != cp_group.world_size:
        return None
    if getattr(pynccl, "rank", None) != cp_group.rank_in_group:
        return None
    if getattr(pynccl, "device", None) != query.device:
        return None
    return pynccl


def dcp_query_all_gather(
    query: torch.Tensor,
    cp_group: GroupCoordinator,
    trace_range: Callable[[str], AbstractContextManager[None]] | None = None,
) -> tuple[torch.Tensor, bool]:
    """Gather DCP query heads with persistent direct buffers when supported.

    Returns the head-major gathered query and whether the direct PyNCCL path
    executed. Unsupported communicators and compiler/fake execution retain the
    existing GroupCoordinator operation.
    """
    pynccl = _direct_pynccl_communicator(query, cp_group)
    query_bytes = query.numel() * query.element_size()
    if pynccl is None:
        if trace_range is None:
            prepared = query.contiguous()
            return cp_group.all_gather(prepared, dim=1), False
        with trace_range(f"query_gather_prepare_fallback bytes={query_bytes}"):
            prepared = query.contiguous()
        with trace_range(f"query_all_gather_fallback bytes={query_bytes}"):
            gathered = cp_group.all_gather(prepared, dim=1)
        return gathered, False

    group_name = str(getattr(cp_group, "unique_name", "dcp"))
    if trace_range is None:
        local_input, rank_major, head_major = _dcp_query_gather_buffers(
            query, cp_group.world_size, group_name
        )
    else:
        with trace_range("query_gather_cache_acquire"):
            local_input, rank_major, head_major = _dcp_query_gather_buffers(
                query, cp_group.world_size, group_name
            )

    if query.is_contiguous():
        prepared = query
    elif trace_range is None:
        local_input.copy_(query)
        prepared = local_input
    else:
        with trace_range(f"query_gather_prepare_direct bytes={query_bytes}"):
            local_input.copy_(query)
        prepared = local_input

    if trace_range is None:
        pynccl.all_gather(rank_major, prepared)
        return _reformat_rank_major_query(rank_major, head_major), True

    with trace_range(f"query_all_gather_direct bytes={query_bytes}"):
        pynccl.all_gather(rank_major, prepared)
    reformat_bytes = rank_major.numel() * rank_major.element_size()
    with trace_range(f"query_gather_reformat bytes={reformat_bytes}"):
        gathered = _reformat_rank_major_query(rank_major, head_major)
    return gathered, True
