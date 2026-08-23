# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.pp_mtp_payload import (
    correct_optimistic_num_computed,
    pack_pp_mtp_payload,
    pp_mtp_payload_width,
    reconcile_pp_num_tokens_no_spec,
    request_fingerprint,
    unpack_pp_mtp_payload,
    validate_pp_mtp_payload,
)


def test_pp_mtp_reconciles_nonfinal_token_end_after_accepted_drafts() -> None:
    # PP0 has staged only the mandatory token (end=11), while the scheduler
    # reports a rejection-corrected computed position after two accepted drafts.
    assert reconcile_pp_num_tokens_no_spec(11, 12) == 13


def test_pp_mtp_reconcile_does_not_use_optimistic_rejected_drafts() -> None:
    # With all three drafts rejected, PP0 already staged the mandatory token.
    # The corrected computed position remains at 10 even if the scheduler's
    # optimistic output count temporarily describes an end of 14.
    assert reconcile_pp_num_tokens_no_spec(11, 10) == 11


def test_pp_mtp_cpu_correction_handles_state_block_boundary() -> None:
    corrected = correct_optimistic_num_computed(
        optimistic_num_computed=61,
        previous_draft_count=3,
        valid_sampled_count=1,
    )

    assert corrected == 58
    assert (61 + 4 + 63) // 64 - 1 == 1
    assert (corrected + 4 + 63) // 64 - 1 == 0


def test_pp_mtp_payload_preserves_count_meanings_and_latest_token() -> None:
    sampled = torch.tensor([[10, 11, 12, -1], [20, -1, -1, -1]], dtype=torch.int64)
    fingerprints = torch.tensor(
        [request_fingerprint("req-a"), request_fingerprint("req-b")],
        dtype=torch.int64,
    )
    payload = torch.empty((8, pp_mtp_payload_width(3)), dtype=torch.int64)
    next_drafts = torch.tensor([[30, 31, 32], [40, 41, 42]], dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        sampled,
        fingerprints,
        torch.tensor([True, True]),
        step_epoch=17,
        next_draft_token_ids=next_drafts,
    )

    assert validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([True, True]), 17, 2, 3
    )
    committed, accepted, valid, next_count, unpacked_drafts = unpack_pp_mtp_payload(
        payload, 2
    )
    assert committed.tolist() == [12, 20]
    assert accepted.tolist() == [2, 0]
    assert valid.tolist() == [3, 1]
    assert next_count.tolist() == [3, 3]
    assert unpacked_drafts.tolist() == next_drafts.tolist()


def test_pp_mtp_unpacked_drafts_alias_reusable_payload() -> None:
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((2, pp_mtp_payload_width(1)), dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        torch.tensor([[10, -1]], dtype=torch.int64),
        fingerprints,
        torch.tensor([True]),
        step_epoch=4,
        next_draft_token_ids=torch.tensor([[20]], dtype=torch.int64),
    )

    *_, drafts = unpack_pp_mtp_payload(payload, 1)
    payload[0, -1] = 21
    assert drafts.tolist() == [[21]]


def test_pp_mtp_payload_rejects_stale_identity_epoch_and_count() -> None:
    sampled = torch.tensor([[10, 11, -1, -1]], dtype=torch.int64)
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((4, pp_mtp_payload_width(3)), dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        sampled,
        fingerprints,
        torch.tensor([True]),
        step_epoch=4,
    )

    assert not validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([True]), 5, 1, 3
    )
    assert not validate_pp_mtp_payload(
        payload,
        torch.tensor([request_fingerprint("req-b")]),
        torch.tensor([True]),
        4,
        1,
        3,
    )
    payload[0, 5] = 4
    assert not validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([True]), 4, 1, 3
    )


def test_pp_mtp_payload_rejects_num_reqs_mismatch() -> None:
    sampled = torch.tensor([[10, -1], [20, -1]], dtype=torch.int64)
    fingerprints = torch.tensor(
        [request_fingerprint("req-a"), request_fingerprint("req-b")],
        dtype=torch.int64,
    )
    payload = torch.empty((4, pp_mtp_payload_width(1)), dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        sampled,
        fingerprints,
        torch.tensor([True, True]),
        step_epoch=4,
    )

    assert not validate_pp_mtp_payload(
        payload,
        fingerprints[:1],
        torch.tensor([True]),
        4,
        1,
        1,
    )

    overread_fingerprints = torch.tensor(
        [request_fingerprint("req-a"), request_fingerprint("req-b")],
        dtype=torch.int64,
    )
    one_row_payload = torch.empty((4, pp_mtp_payload_width(1)), dtype=torch.int64)
    pack_pp_mtp_payload(
        one_row_payload,
        torch.tensor([[10, -1]], dtype=torch.int64),
        overread_fingerprints,
        torch.tensor([True]),
        step_epoch=4,
    )
    assert not validate_pp_mtp_payload(
        one_row_payload,
        overread_fingerprints,
        torch.tensor([True, True]),
        4,
        2,
        1,
    )


def test_pp_mtp_payload_rejects_mtp_width_mismatch() -> None:
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    mtp3_payload = torch.empty((2, pp_mtp_payload_width(3)), dtype=torch.int64)
    pack_pp_mtp_payload(
        mtp3_payload,
        torch.tensor([[10, -1]], dtype=torch.int64),
        fingerprints,
        torch.tensor([True]),
        step_epoch=4,
        next_draft_token_ids=torch.tensor([[20, 21, 22]], dtype=torch.int64),
    )

    assert not validate_pp_mtp_payload(
        mtp3_payload, fingerprints, torch.tensor([True]), 4, 1, 1
    )


def test_pp_mtp1_active_draft_round_trip() -> None:
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((2, pp_mtp_payload_width(1)), dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        torch.tensor([[10, -1]], dtype=torch.int64),
        fingerprints,
        torch.tensor([True]),
        step_epoch=4,
        next_draft_token_ids=torch.tensor([[20]], dtype=torch.int64),
    )

    assert validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([True]), 4, 1, 1
    )
    *_, next_count, drafts = unpack_pp_mtp_payload(payload, 1)
    assert next_count.tolist() == [1]
    assert drafts.tolist() == [[20]]


def test_pp_mtp_payload_rejects_valid_request_without_sample() -> None:
    sampled = torch.full((1, 2), -1, dtype=torch.int64)
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((2, pp_mtp_payload_width(1)), dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        sampled,
        fingerprints,
        torch.tensor([True]),
        step_epoch=4,
    )

    assert not validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([True]), 4, 1, 1
    )


def test_pp_mtp_payload_accepts_inactive_and_empty_batches() -> None:
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((2, pp_mtp_payload_width(1)), dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        torch.full((1, 2), -1, dtype=torch.int64),
        fingerprints,
        torch.tensor([False]),
        step_epoch=4,
    )
    assert validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([False]), 4, 1, 1
    )

    empty_payload = torch.empty((2, pp_mtp_payload_width(1)), dtype=torch.int64)
    pack_pp_mtp_payload(
        empty_payload,
        torch.empty((0, 2), dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.bool),
        step_epoch=5,
    )
    assert validate_pp_mtp_payload(
        empty_payload,
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.bool),
        5,
        0,
        1,
    )


def test_pp_mtp_payload_rejects_zero_width_and_acceptance_overflow() -> None:
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((2, pp_mtp_payload_width(1)), dtype=torch.int64)
    with pytest.raises(ValueError, match="width >= 1"):
        pack_pp_mtp_payload(
            payload,
            torch.empty((1, 0), dtype=torch.int64),
            fingerprints,
            torch.tensor([True]),
            step_epoch=4,
        )

    pack_pp_mtp_payload(
        payload,
        torch.tensor([[10, 11, 12]], dtype=torch.int64),
        fingerprints,
        torch.tensor([True]),
        step_epoch=4,
    )
    assert not validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([True]), 4, 1, 1
    )


def test_pp_mtp_payload_rejects_missing_or_noncontiguous_next_drafts() -> None:
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((2, pp_mtp_payload_width(3)), dtype=torch.int64)
    sampled = torch.tensor([[10, -1]], dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        sampled,
        fingerprints,
        torch.tensor([True]),
        step_epoch=4,
        next_draft_token_ids=torch.tensor([[20, -1, 22]], dtype=torch.int64),
    )

    assert not validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([True]), 4, 1, 3
    )


def test_pp_mtp_payload_masks_inactive_next_drafts() -> None:
    fingerprints = torch.tensor([request_fingerprint("req-a")], dtype=torch.int64)
    payload = torch.empty((2, pp_mtp_payload_width(2)), dtype=torch.int64)
    pack_pp_mtp_payload(
        payload,
        torch.full((1, 2), -1, dtype=torch.int64),
        fingerprints,
        torch.tensor([False]),
        step_epoch=4,
        next_draft_token_ids=torch.tensor([[20, 21]], dtype=torch.int64),
    )

    assert validate_pp_mtp_payload(
        payload, fingerprints, torch.tensor([False]), 4, 1, 2
    )
    *_, next_count, drafts = unpack_pp_mtp_payload(payload, 1)
    assert next_count.tolist() == [0]
    assert drafts.tolist() == [[-1, -1]]
