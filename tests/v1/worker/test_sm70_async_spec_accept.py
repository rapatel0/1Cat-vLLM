# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _brute_force_stays_in_state_block(
    prev_state_idx: int,
    optimistic_computed: int,
    scheduled_tokens: int,
    block_size: int,
    max_rejected_tokens: int,
) -> bool:
    min_computed = max(0, optimistic_computed - max_rejected_tokens)
    for corrected_computed in range(min_computed, optimistic_computed + 1):
        state_idx = cdiv(corrected_computed + scheduled_tokens, block_size) - 1
        if state_idx != prev_state_idx:
            return False
    return True


def test_sm70_async_spec_state_block_guard_boundary_cases() -> None:
    guard = GPUModelRunner._sm70_async_spec_stays_in_state_block

    # All possible rejection corrections remain in state block 0.
    assert guard(0, 100, 4, 400, 3)

    # A correction can move the request back across the state-block boundary.
    assert not guard(0, 399, 4, 400, 3)

    # Even with no rejection correction, the scheduled step enters a new block.
    assert not guard(0, 400, 4, 400, 0)

    # The same rules hold beyond the first block.
    assert guard(2, 1001, 4, 400, 3)
    assert not guard(2, 1199, 4, 400, 3)


def test_sm70_async_spec_state_block_guard_matches_brute_force() -> None:
    rng = random.Random(38003)
    guard = GPUModelRunner._sm70_async_spec_stays_in_state_block

    for _ in range(4096):
        block_size = rng.choice((16, 32, 64, 128, 400))
        optimistic_computed = rng.randint(0, block_size * 32)
        scheduled_tokens = rng.randint(1, 8)
        max_rejected_tokens = rng.randint(0, 7)
        prev_state_idx = rng.randint(-1, 31)

        expected = _brute_force_stays_in_state_block(
            prev_state_idx,
            optimistic_computed,
            scheduled_tokens,
            block_size,
            max_rejected_tokens,
        )
        actual = guard(
            prev_state_idx,
            optimistic_computed,
            scheduled_tokens,
            block_size,
            max_rejected_tokens,
        )
        assert actual == expected
