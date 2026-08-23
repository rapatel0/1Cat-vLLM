# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.flash_attn_v100 import (
    _sm70_prepare_smallq_decode_metadata,
)


def _reference_smallq_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    num_query_tokens: int,
    real_num_query_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    repeat_query_lens = query_lens.clone()
    repeat_query_lens[-1] += num_query_tokens - real_num_query_tokens
    effective_seq_lens = torch.maximum(seq_lens, query_lens)
    decode_block_table = torch.repeat_interleave(
        block_table.clamp_min(0),
        repeat_query_lens,
        dim=0,
        output_size=num_query_tokens,
    ).contiguous()
    seq_lens_rep = torch.repeat_interleave(
        effective_seq_lens,
        repeat_query_lens,
        output_size=num_query_tokens,
    )
    query_lens_rep = torch.repeat_interleave(
        query_lens,
        repeat_query_lens,
        output_size=num_query_tokens,
    )
    start_locs_rep = torch.repeat_interleave(
        query_start_loc[:-1],
        repeat_query_lens,
        output_size=num_query_tokens,
    )
    token_indices = torch.arange(
        num_query_tokens, dtype=seq_lens.dtype, device=seq_lens.device
    )
    decode_seq_lens = (
        seq_lens_rep - query_lens_rep + token_indices - start_locs_rep + 1
    ).contiguous()
    if real_num_query_tokens < num_query_tokens:
        padding_mask = token_indices >= real_num_query_tokens
        decode_seq_lens = decode_seq_lens.masked_fill(padding_mask, 0)
        decode_block_table = decode_block_table.masked_fill(padding_mask[:, None], 0)
    return decode_block_table, decode_seq_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("query_lens", "seq_lens", "num_query_tokens"),
    [
        ([8], [128], 8),
        ([3, 2, 0], [8, 7, 0], 6),
        ([1, 4, 2, 0], [1, 10, 8, 0], 8),
    ],
)
def test_sm70_fused_smallq_metadata_matches_repeat_interleave(
    query_lens: list[int],
    seq_lens: list[int],
    num_query_tokens: int,
):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 CUDA device required")

    device = torch.device("cuda")
    num_reqs = len(query_lens)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    real_num_query_tokens = sum(query_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    block_table = torch.arange(
        num_reqs * 7,
        dtype=torch.int32,
        device=device,
    ).view(num_reqs, 7)
    block_table[:, 0] = -1
    out_block_table = torch.empty(
        (num_query_tokens, block_table.shape[1]),
        dtype=torch.int32,
        device=device,
    )
    out_seq_lens = torch.empty(num_query_tokens, dtype=torch.int32, device=device)
    out_query_start_loc = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)

    _sm70_prepare_smallq_decode_metadata(
        out_block_table,
        out_seq_lens,
        out_query_start_loc,
        block_table,
        seq_lens_tensor,
        query_start_loc,
        num_reqs=num_reqs,
        num_query_tokens=num_query_tokens,
        real_num_query_tokens=real_num_query_tokens,
    )
    expected_block_table, expected_seq_lens = _reference_smallq_metadata(
        block_table,
        seq_lens_tensor,
        query_start_loc,
        num_query_tokens=num_query_tokens,
        real_num_query_tokens=real_num_query_tokens,
    )
    torch.cuda.synchronize()

    assert torch.equal(out_block_table, expected_block_table)
    assert torch.equal(out_seq_lens, expected_seq_lens)
    assert torch.equal(out_query_start_loc, query_start_loc)
