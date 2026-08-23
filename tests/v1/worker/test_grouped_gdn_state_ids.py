# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.attention.backends.gdn_attn import (
    _build_grouped_single_request_gdn_spec_metadata_kernel,
    _build_single_request_gdn_spec_metadata_kernel,
)
from vllm.v1.attention.backends.flash_attn_v100 import (
    _expand_single_request_smallq_metadata_kernel,
    _prebuild_single_spec_q4_smallq_metadata_kernel,
)
from vllm.v1.worker.gpu_model_runner import (
    _populate_grouped_gdn_state_block_ids,
)


def test_grouped_gdn_state_ids_match_per_group_reference_4096_transitions() -> None:
    rng = random.Random(1701)
    num_groups = 5
    gdn_group_ids = (0, 2, 4)
    max_reqs = 8
    max_state_slots = 4
    destination = torch.full(
        (num_groups, max_reqs, max_state_slots),
        123456,
        dtype=torch.int32,
    )

    for transition in range(4096):
        num_reqs = rng.randint(1, 4)
        num_reqs_padded = rng.randint(num_reqs, max_reqs)
        req_ids = [f"req-{transition}-{req_idx}" for req_idx in range(num_reqs)]
        requests = {}
        mamba_state_idx = {}
        for req_idx, req_id in enumerate(req_ids):
            group_block_ids = []
            for group_id in range(num_groups):
                length = rng.randint(1, 12)
                base = transition * 100_000 + group_id * 10_000 + req_idx * 100
                group_block_ids.append([base + offset for offset in range(length)])
            requests[req_id] = SimpleNamespace(block_ids=group_block_ids)
            if rng.random() >= 0.1:
                mamba_state_idx[req_id] = rng.randint(0, 11)

        expected = destination.clone()
        expected[:, :num_reqs_padded].fill_(PAD_SLOT_ID)
        for group_id in gdn_group_ids:
            for req_idx, req_id in enumerate(req_ids):
                state_idx = mamba_state_idx.get(req_id)
                if state_idx is None:
                    continue
                block_ids = requests[req_id].block_ids[group_id]
                for offset in range(max_state_slots):
                    block_idx = state_idx + offset
                    if block_idx >= len(block_ids):
                        break
                    expected[group_id, req_idx, offset] = block_ids[block_idx]

        _populate_grouped_gdn_state_block_ids(
            destination,
            gdn_group_ids,
            req_ids,
            requests,
            mamba_state_idx,
            max_state_slots,
            num_reqs_padded,
        )

        assert torch.equal(destination, expected)


def test_grouped_gdn_metadata_kernel_matches_three_legacy_launches() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    num_groups = 3
    batch_size = 4
    state_width = 4
    output_rows = 64

    for accepted_value in range(4):
        for selector_value in range(4):
            for variant in range(4):
                source_state_ids = (
                    torch.arange(
                        num_groups * output_rows * state_width,
                        device=device,
                        dtype=torch.int32,
                    ).reshape(num_groups, output_rows, state_width)
                    + variant * 10_000
                )
                source_accepted = torch.tensor(
                    [accepted_value], device=device, dtype=torch.int32
                )
                source_selectors = torch.tensor(
                    [selector_value], device=device, dtype=torch.int32
                )

                grouped_state = torch.full_like(source_state_ids, -777)
                grouped_masks = torch.zeros(
                    (num_groups, output_rows), device=device, dtype=torch.bool
                )
                grouped_tokens = torch.full(
                    (num_groups, output_rows * state_width),
                    -777,
                    device=device,
                    dtype=torch.int32,
                )
                grouped_query = torch.full(
                    (num_groups, output_rows + 1),
                    -777,
                    device=device,
                    dtype=torch.int32,
                )
                grouped_accepted = torch.full(
                    (num_groups, output_rows),
                    -777,
                    device=device,
                    dtype=torch.int32,
                )
                grouped_selectors = torch.full_like(grouped_accepted, -777)

                reference_state = grouped_state.clone()
                reference_masks = grouped_masks.clone()
                reference_tokens = grouped_tokens.clone()
                reference_query = grouped_query.clone()
                reference_accepted = grouped_accepted.clone()
                reference_selectors = grouped_selectors.clone()

                _build_grouped_single_request_gdn_spec_metadata_kernel[
                    (batch_size, num_groups)
                ](
                    source_state_ids,
                    source_state_ids.stride(0),
                    source_accepted,
                    source_selectors,
                    grouped_state,
                    grouped_state.stride(0),
                    grouped_state.stride(1),
                    grouped_masks,
                    grouped_masks.stride(0),
                    grouped_tokens,
                    grouped_tokens.stride(0),
                    grouped_query,
                    grouped_query.stride(0),
                    grouped_accepted,
                    grouped_accepted.stride(0),
                    grouped_selectors,
                    grouped_selectors.stride(0),
                    batch_size=batch_size,
                    real_query_tokens=batch_size,
                    state_width=state_width,
                    state_block_size=state_width,
                    capture_only=False,
                    pad_slot_id=PAD_SLOT_ID,
                )
                for group_id in range(num_groups):
                    _build_single_request_gdn_spec_metadata_kernel[
                        (batch_size,)
                    ](
                        source_state_ids[group_id],
                        source_accepted,
                        source_selectors,
                        reference_state[group_id],
                        reference_state[group_id].stride(0),
                        reference_masks[group_id],
                        reference_tokens[group_id],
                        reference_query[group_id],
                        reference_accepted[group_id],
                        reference_selectors[group_id],
                        batch_size=batch_size,
                        real_query_tokens=batch_size,
                        state_width=state_width,
                        state_block_size=state_width,
                        capture_only=False,
                        pad_slot_id=PAD_SLOT_ID,
                    )

                torch.cuda.synchronize()
                assert torch.equal(grouped_state, reference_state)
                assert torch.equal(grouped_masks, reference_masks)
                assert torch.equal(grouped_tokens, reference_tokens)
                assert torch.equal(grouped_query, reference_query)
                assert torch.equal(grouped_accepted, reference_accepted)
                assert torch.equal(grouped_selectors, reference_selectors)


def test_prebuilt_q4_flash_metadata_matches_legacy_expansion() -> None:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    block_cols_per_program = 256

    for seq_len in (3, 4, 5, 128):
        for block_cols in (256, 300, 2048):
            source_block_table = torch.arange(
                -7,
                block_cols - 7,
                device=device,
                dtype=torch.int32,
            ).reshape(1, block_cols)
            source_seq_lens = torch.tensor(
                [seq_len],
                device=device,
                dtype=torch.int32,
            )
            reference_block_table = torch.full(
                (4, block_cols),
                -777,
                device=device,
                dtype=torch.int32,
            )
            reference_seq_lens = torch.full(
                (4,),
                -777,
                device=device,
                dtype=torch.int32,
            )
            reference_query_start_loc = torch.tensor(
                [0, 4],
                device=device,
                dtype=torch.int32,
            )
            candidate_block_table = torch.full_like(reference_block_table, -888)
            candidate_seq_lens = torch.full_like(reference_seq_lens, -888)
            candidate_query_start_loc = torch.full_like(
                reference_query_start_loc,
                -888,
            )
            grid = (
                4,
                (block_cols + block_cols_per_program - 1)
                // block_cols_per_program,
            )

            _expand_single_request_smallq_metadata_kernel[grid](
                source_block_table,
                source_seq_lens,
                reference_block_table,
                reference_seq_lens,
                reference_block_table.stride(0),
                num_block_cols=block_cols,
                real_query_tokens=4,
                block_cols_per_program=block_cols_per_program,
            )
            _prebuild_single_spec_q4_smallq_metadata_kernel[grid](
                source_block_table,
                source_seq_lens,
                candidate_block_table,
                candidate_seq_lens,
                candidate_query_start_loc,
                candidate_block_table.stride(0),
                num_block_cols=block_cols,
                block_cols_per_program=block_cols_per_program,
            )

            torch.cuda.synchronize()
            assert torch.equal(candidate_block_table, reference_block_table)
            assert torch.equal(candidate_seq_lens, reference_seq_lens)
            assert torch.equal(
                candidate_query_start_loc,
                reference_query_start_loc,
            )
