# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.sample.rejection_sampler import rejection_random_sample_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("reject_at", "expected_tokens", "expected_count", "expected_next"),
    [
        (None, [1, 2, 3, 7], 4, 7),
        (0, [9, -1, -1, -1], 1, 9),
        (1, [1, 9, -1, -1], 2, 9),
        (2, [1, 2, 9, -1], 3, 9),
    ],
)
def test_sm70_fused_rejection_aux_truth_table(
    reject_at: int | None,
    expected_tokens: list[int],
    expected_count: int,
    expected_next: int,
) -> None:
    device = torch.device("cuda")
    draft = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    target = torch.zeros((3, 16), dtype=torch.float32, device=device)
    target[torch.arange(3, device=device), draft.long()] = 1.0
    if reject_at is not None:
        target[reject_at, draft[reject_at].long()] = 0.0

    output = torch.full((1, 4), -1, dtype=torch.int32, device=device)
    valid = torch.empty((1,), dtype=torch.int32, device=device)
    next_token = torch.empty((1,), dtype=torch.int32, device=device)
    rejection_random_sample_kernel[(1,)](
        output,
        valid,
        next_token,
        torch.tensor([3], dtype=torch.int32, device=device),
        draft,
        None,
        target,
        torch.tensor([7], dtype=torch.int32, device=device),
        torch.tensor([9, 9, 9], dtype=torch.int32, device=device),
        torch.full((3,), 0.5, dtype=torch.float32, device=device),
        torch.tensor([False], dtype=torch.bool, device=device),
        3,
        16,
        None,
        NO_DRAFT_PROBS=True,
        SYNTHETIC_MODE=False,
        WRITE_AUX=True,
    )
    torch.cuda.synchronize()

    assert output[0].tolist() == expected_tokens
    assert valid.item() == expected_count
    assert next_token.item() == expected_next
