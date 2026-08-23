# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA golden tests for PP-MTP align-mode recurrent-state preprocessing."""

import pytest
import torch

from vllm.v1.spec_decode.utils import update_num_computed_tokens_for_batch_change
from vllm.v1.worker.mamba_utils import (
    preprocess_mamba_fused_kernel,
    reset_mamba_preprocess_counts_kernel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pp_mtp_num_computed_correction_preserves_state_selector_count() -> None:
    device = torch.device("cuda")
    num_computed = torch.tensor([100], dtype=torch.int32, device=device)
    state_selector_count = torch.tensor([1], dtype=torch.int32, device=device)
    update_num_computed_tokens_for_batch_change(
        num_computed,
        state_selector_count,
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.tensor([3], dtype=torch.int64, device=device),
        torch.tensor([3], dtype=torch.int32, device=device),
        torch.tensor([103], dtype=torch.int32, device=device),
        update_num_accepted_tokens=False,
    )

    assert num_computed.item() == 103
    assert state_selector_count.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("accepted_count", "selector", "prev_idx", "curr_idx"),
    [
        (1, 1, 1, 2),
        (3, 3, 1, 2),
        (3, 3, 1, 1),
    ],
)
def test_pp_mtp_fused_preprocess_matches_copy_specs(
    accepted_count: int,
    selector: int,
    prev_idx: int,
    curr_idx: int,
) -> None:
    device = torch.device("cuda")
    block_table = torch.arange(8, dtype=torch.int32, device=device).reshape(1, 8)
    block_ptrs = torch.tensor(
        [block_table.data_ptr()], dtype=torch.int64, device=device
    )
    conv = torch.full((8, 4, 2), -9.0, dtype=torch.float32, device=device)
    temporal = torch.full((8, 2, 2), -7.0, dtype=torch.float32, device=device)
    conv[prev_idx].copy_(
        torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
    )
    temporal_src_idx = prev_idx + selector - 1
    temporal[temporal_src_idx].copy_(
        torch.tensor([[31.0, 32.0], [33.0, 34.0]], device=device)
    )
    original_conv_dest = conv[curr_idx].clone()
    original_temporal_dest = temporal[curr_idx].clone()

    state_base = torch.tensor(
        [conv.data_ptr(), temporal.data_ptr()], dtype=torch.int64, device=device
    )
    state_stride = torch.tensor(
        [
            conv.stride(0) * conv.element_size(),
            temporal.stride(0) * temporal.element_size(),
        ],
        dtype=torch.int64,
        device=device,
    )
    element_size = torch.tensor(
        [conv.element_size(), temporal.element_size()],
        dtype=torch.int32,
        device=device,
    )
    inner_size = torch.tensor(
        [conv.stride(1), temporal[0].numel()], dtype=torch.int64, device=device
    )
    conv_width = torch.tensor([conv.size(1), 0], dtype=torch.int32, device=device)
    group_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)
    accepted = torch.tensor([accepted_count], dtype=torch.int32, device=device)
    selectors = torch.tensor([selector], dtype=torch.int32, device=device)
    prev = torch.tensor([prev_idx], dtype=torch.int32, device=device)
    curr = torch.tensor([curr_idx], dtype=torch.int32, device=device)

    preprocess_mamba_fused_kernel[(1, 2)](
        accepted,
        selectors,
        prev,
        curr,
        block_ptrs,
        block_table.stride(0),
        state_base,
        state_stride,
        element_size,
        inner_size,
        conv_width,
        group_idx,
        1,
        COPY_BLOCK_SIZE=32,
    )
    reset_mamba_preprocess_counts_kernel[(1,)](
        accepted,
        selectors,
        prev,
        curr,
        1,
        BLOCK_SIZE=1,
    )
    torch.cuda.synchronize()

    if prev_idx == curr_idx:
        assert torch.equal(conv[curr_idx], original_conv_dest)
        assert torch.equal(temporal[curr_idx], original_temporal_dest)
        assert accepted.item() == accepted_count
        assert selectors.item() == selector
        return

    bias = accepted_count - 1
    assert torch.equal(conv[curr_idx, : 4 - bias], conv[prev_idx, bias:])
    assert torch.equal(conv[curr_idx, 4 - bias :], original_conv_dest[4 - bias :])
    assert torch.equal(temporal[curr_idx], temporal[temporal_src_idx])
    assert accepted.item() == 1
    assert selectors.item() == 1
