# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape GPU payload for native MTP under pipeline parallelism."""

from __future__ import annotations

import hashlib
from enum import IntEnum

import torch


class PPMTPPayloadField(IntEnum):
    REQUEST_VALID = 0
    REQUEST_FINGERPRINT = 1
    STEP_EPOCH = 2
    COMMITTED_TOKEN = 3
    ACCEPTED_DRAFT_COUNT = 4
    VALID_SAMPLED_COUNT = 5
    NEXT_DRAFT_COUNT = 6


PP_MTP_NEXT_DRAFT_TOKEN_OFFSET = len(PPMTPPayloadField)


def pp_mtp_payload_width(max_draft_tokens: int) -> int:
    if max_draft_tokens < 0:
        raise ValueError("max_draft_tokens must be non-negative")
    return PP_MTP_NEXT_DRAFT_TOKEN_OFFSET + max_draft_tokens


def request_fingerprint(req_id: str) -> int:
    """Return a stable positive int64 fingerprint for a scheduler request ID."""
    digest = hashlib.blake2b(
        req_id.encode("utf-8"),
        digest_size=8,
        person=b"vllm-mtp",
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def count_contiguous_valid_tokens(sampled_token_ids: torch.Tensor) -> torch.Tensor:
    """Count the valid prefix before the first ``-1`` in every sampled row."""
    valid = sampled_token_ids.ne(-1).to(torch.int64)
    return valid.cumprod(dim=1).sum(dim=1)


def correct_optimistic_num_computed(
    optimistic_num_computed: int,
    previous_draft_count: int,
    valid_sampled_count: int,
) -> int:
    """Apply the scheduler's rejected-draft correction without GPU state."""
    accepted_drafts = valid_sampled_count - 1
    if not 0 <= accepted_drafts <= previous_draft_count:
        raise ValueError(
            "valid sampled count is inconsistent with the previous draft count"
        )
    return optimistic_num_computed - (previous_draft_count - accepted_drafts)


def reconcile_pp_num_tokens_no_spec(
    current_end: int,
    num_computed_tokens: int,
) -> int:
    """Advance a non-final PP token mirror to the rejection-corrected end."""
    if min(current_end, num_computed_tokens) < 0:
        raise ValueError("PP token bookkeeping values must be non-negative")
    return max(current_end, num_computed_tokens + 1)


def pack_pp_mtp_payload(
    payload: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    request_fingerprints: torch.Tensor,
    request_valid: torch.Tensor,
    step_epoch: int,
    next_draft_token_ids: torch.Tensor | None = None,
) -> None:
    """Pack sampler output and the next drafts into a fixed-address payload."""
    if sampled_token_ids.ndim != 2 or sampled_token_ids.shape[1] == 0:
        raise ValueError("sampled_token_ids must have shape [num_reqs, width >= 1]")
    num_reqs = sampled_token_ids.shape[0]
    if payload.ndim != 2 or payload.shape[1] < PP_MTP_NEXT_DRAFT_TOKEN_OFFSET:
        raise ValueError("invalid PP MTP payload shape")
    if request_fingerprints.shape[0] < num_reqs:
        raise ValueError("request fingerprint buffer is too small")
    if request_valid.shape != (num_reqs,):
        raise ValueError("request_valid must have shape [num_reqs]")
    for name, tensor in (
        ("sampled_token_ids", sampled_token_ids),
        ("request_fingerprints", request_fingerprints),
        ("request_valid", request_valid),
    ):
        if tensor.device != payload.device:
            raise ValueError(f"{name} must be on the payload device")
    max_draft_tokens = payload.shape[1] - PP_MTP_NEXT_DRAFT_TOKEN_OFFSET
    if next_draft_token_ids is not None:
        if next_draft_token_ids.shape != (num_reqs, max_draft_tokens):
            raise ValueError(
                "next_draft_token_ids must have shape "
                f"[{num_reqs}, {max_draft_tokens}]"
            )
        if next_draft_token_ids.device != payload.device:
            raise ValueError("next draft tokens must be on the payload device")

    payload.fill_(-1)
    if num_reqs == 0:
        return

    valid_count = count_contiguous_valid_tokens(sampled_token_ids)
    safe_last_index = (valid_count - 1).clamp_min(0)
    committed_token = sampled_token_ids.gather(1, safe_last_index.unsqueeze(1)).squeeze(
        1
    )
    active = request_valid & valid_count.gt(0)
    accepted_draft_count = valid_count - 1
    rows = payload[:num_reqs]
    # Preserve the caller's scheduler-derived validity bit. A valid request
    # whose sampler produced no token must fail validation rather than being
    # silently reclassified as an inactive/chunked-prefill row.
    rows[:, PPMTPPayloadField.REQUEST_VALID] = request_valid
    rows[:, PPMTPPayloadField.REQUEST_FINGERPRINT].copy_(
        request_fingerprints[:num_reqs]
    )
    rows[:, PPMTPPayloadField.STEP_EPOCH] = step_epoch
    rows[:, PPMTPPayloadField.COMMITTED_TOKEN] = torch.where(
        active, committed_token, -1
    )
    rows[:, PPMTPPayloadField.ACCEPTED_DRAFT_COUNT] = torch.where(
        active, accepted_draft_count, 0
    )
    rows[:, PPMTPPayloadField.VALID_SAMPLED_COUNT] = torch.where(active, valid_count, 1)
    if next_draft_token_ids is None or max_draft_tokens == 0:
        rows[:, PPMTPPayloadField.NEXT_DRAFT_COUNT] = 0
        return

    next_valid = next_draft_token_ids.ge(0).to(torch.int64)
    next_count = next_valid.cumprod(dim=1).sum(dim=1)
    next_active = request_valid & next_count.gt(0)
    rows[:, PPMTPPayloadField.NEXT_DRAFT_COUNT] = torch.where(
        next_active, next_count, 0
    )
    draft_rows = rows[:, PP_MTP_NEXT_DRAFT_TOKEN_OFFSET:]
    draft_rows.copy_(
        torch.where(request_valid.unsqueeze(1), next_draft_token_ids, -1)
    )


def validate_pp_mtp_payload(
    payload: torch.Tensor,
    expected_fingerprints: torch.Tensor,
    expected_request_valid: torch.Tensor,
    expected_epoch: int,
    num_reqs: int,
    max_draft_tokens: int,
) -> torch.Tensor:
    """Return a scalar device predicate for identity, epoch, and count invariants."""
    invalid_shape = (
        payload.ndim != 2
        or payload.shape[1] != pp_mtp_payload_width(max_draft_tokens)
        or not 0 <= num_reqs <= payload.shape[0]
        or expected_fingerprints.ndim != 1
        or expected_fingerprints.shape[0] < num_reqs
        or expected_request_valid.shape != (num_reqs,)
        or expected_fingerprints.device != payload.device
        or expected_request_valid.device != payload.device
    )
    if invalid_shape:
        return torch.zeros((), dtype=torch.bool, device=payload.device)
    rows = payload[:num_reqs]
    request_valid = rows[:, PPMTPPayloadField.REQUEST_VALID]
    accepted = rows[:, PPMTPPayloadField.ACCEPTED_DRAFT_COUNT]
    valid_count = rows[:, PPMTPPayloadField.VALID_SAMPLED_COUNT]
    committed = rows[:, PPMTPPayloadField.COMMITTED_TOKEN]
    next_count = rows[:, PPMTPPayloadField.NEXT_DRAFT_COUNT]
    valid_flag = (request_valid == 0) | (request_valid == 1)
    request_valid_ok = request_valid.eq(expected_request_valid.to(torch.int64))
    identity_ok = rows[:, PPMTPPayloadField.REQUEST_FINGERPRINT].eq(
        expected_fingerprints[:num_reqs]
    )
    epoch_ok = rows[:, PPMTPPayloadField.STEP_EPOCH].eq(expected_epoch)
    active_ok = (
        accepted.ge(0)
        & accepted.le(max_draft_tokens)
        & valid_count.eq(accepted + 1)
        & committed.ge(0)
    )
    inactive_ok = accepted.eq(0) & valid_count.eq(1) & committed.eq(-1)
    counts_ok = torch.where(request_valid.bool(), active_ok, inactive_ok)
    next_count_ok = next_count.ge(0) & next_count.le(max_draft_tokens)
    if max_draft_tokens:
        next_tokens = rows[:, PP_MTP_NEXT_DRAFT_TOKEN_OFFSET:]
        next_positions = torch.arange(
            max_draft_tokens, device=payload.device, dtype=torch.int64
        )
        next_token_ok = torch.where(
            next_positions.unsqueeze(0) < next_count.unsqueeze(1),
            next_tokens.ge(0),
            next_tokens.eq(-1),
        ).all(dim=1)
    else:
        next_token_ok = torch.ones(num_reqs, dtype=torch.bool, device=payload.device)
    next_drafts_ok = next_count_ok & next_token_ok
    next_drafts_ok &= request_valid.bool() | next_count.eq(0)
    # The fixed-size collective has no host-visible header. Requiring every
    # unused row to retain the sentinel also proves that all stages agree on
    # num_reqs without introducing a device-to-host synchronization.
    padding_ok = payload[num_reqs:, PPMTPPayloadField.REQUEST_VALID].eq(-1).all()
    return (
        valid_flag
        & request_valid_ok
        & identity_ok
        & epoch_ok
        & counts_ok
        & next_drafts_ok
    ).all() & padding_ok


def unpack_pp_mtp_payload(
    payload: torch.Tensor,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return views of committed metadata and the next real draft tokens.

    Every returned tensor aliases ``payload`` and is valid only until the next
    pack or collective overwrites that storage. PP0 intentionally retains the
    draft-token view for exactly the following execute-model step.
    """
    rows = payload[:num_reqs]
    return (
        rows[:, PPMTPPayloadField.COMMITTED_TOKEN],
        rows[:, PPMTPPayloadField.ACCEPTED_DRAFT_COUNT],
        rows[:, PPMTPPayloadField.VALID_SAMPLED_COUNT],
        rows[:, PPMTPPayloadField.NEXT_DRAFT_COUNT],
        rows[:, PP_MTP_NEXT_DRAFT_TOKEN_OFFSET:],
    )
