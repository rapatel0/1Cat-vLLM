# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    _sm70_compact_topk20_probs,
    _sm70_compact_topk20_rejection_enabled,
    _sm70_compact_topk20_rejection_kernel,
    apply_sampling_constraints,
    rejection_sample,
    sm70_compact_topk20_rejection_sample,
)
from vllm.v1.worker.gpu_input_batch import _sm70_compact_topk20_eligible

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SM70 compact rejection requires CUDA",
)


def _sampling_metadata(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.tensor([1.0], device=device),
        top_k=torch.tensor([20], dtype=torch.int32, device=device),
        top_p=torch.tensor([0.95], device=device),
        generators={},
    )


def test_compact_topk20_cpu_eligibility_contract() -> None:
    top_k = np.array([20], dtype=np.int32)
    top_p = np.array([0.95], dtype=np.float32)
    temperature = np.array([1.0], dtype=np.float32)

    assert _sm70_compact_topk20_eligible(
        1, True, False, top_k, False, top_p, temperature
    )
    assert not _sm70_compact_topk20_eligible(
        2, True, False, top_k, False, top_p, temperature
    )
    assert not _sm70_compact_topk20_eligible(
        1, False, False, top_k, False, top_p, temperature
    )
    assert not _sm70_compact_topk20_eligible(
        1,
        True,
        False,
        np.array([19], dtype=np.int32),
        False,
        top_p,
        temperature,
    )
    assert not _sm70_compact_topk20_eligible(
        1,
        True,
        False,
        top_k,
        False,
        np.array([0.90], dtype=np.float32),
        temperature,
    )
    assert not _sm70_compact_topk20_eligible(
        1,
        True,
        False,
        top_k,
        False,
        top_p,
        np.array([0.8], dtype=np.float32),
    )


def test_compact_topk20_runtime_gate_requires_exact_contract() -> None:
    device = torch.device(current_platform.device_type)
    metadata = SimpleNamespace(num_draft_tokens=[3], max_spec_len=3)
    sampling = SimpleNamespace(
        all_random=True,
        sm70_compact_topk20_eligible=True,
        top_k=torch.tensor([20], dtype=torch.int32, device=device),
        top_p=torch.tensor([0.95], device=device),
        max_num_logprobs=None,
    )
    target_logits = torch.empty((3, 32_768), device=device)

    assert _sm70_compact_topk20_rejection_enabled(
        metadata, None, target_logits, sampling, False, True
    )
    sampling.sm70_compact_topk20_eligible = False
    assert not _sm70_compact_topk20_rejection_enabled(
        metadata, None, target_logits, sampling, False, True
    )
    sampling.sm70_compact_topk20_eligible = True
    assert not _sm70_compact_topk20_rejection_enabled(
        metadata, None, target_logits, sampling, False, False
    )


def test_compact_topk20_probabilities_match_dense_reference_4096_rows() -> None:
    device = torch.device(current_platform.device_type)
    generator = torch.Generator(device=device).manual_seed(38020)
    logits = torch.randn(
        (4096, 257),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    dense_logits = logits.clone()
    top_k = torch.full((4096,), 20, dtype=torch.int32, device=device)
    top_p = torch.full((4096,), 0.95, dtype=torch.float32, device=device)
    apply_top_k_top_p_pytorch(dense_logits, top_k, top_p)
    expected = dense_logits.softmax(dim=-1, dtype=torch.float32)

    compact_ids, compact_probs = _sm70_compact_topk20_probs(
        logits.clone(),
        torch.tensor([1.0], device=device),
        torch.tensor([0.95], device=device),
    )
    actual = torch.zeros_like(expected).scatter_(1, compact_ids, compact_probs)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("relative_prob", [None, 0.05])
def test_compact_topk20_rejection_matches_dense_reference_128_draws(
    monkeypatch: pytest.MonkeyPatch,
    relative_prob: float | None,
) -> None:
    if relative_prob is None:
        monkeypatch.delenv("VLLM_MTP_FORCE_TARGET_RELATIVE_PROB", raising=False)
    else:
        monkeypatch.setenv(
            "VLLM_MTP_FORCE_TARGET_RELATIVE_PROB",
            str(relative_prob),
        )
    device = torch.device(current_platform.device_type)
    sampling = _sampling_metadata(device)
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    bonus_token_ids = torch.tensor([[17]], dtype=torch.int32, device=device)
    vocab_size = 248_320

    for seed in range(128):
        logits_generator = torch.Generator(device=device).manual_seed(91_000 + seed)
        request_generator = None
        if seed % 2:
            request_generator = torch.Generator(device=device).manual_seed(
                191_000 + seed
            )
            sampling.generators = {0: request_generator}
        else:
            sampling.generators = {}
        logits = torch.randn(
            (3, vocab_size),
            generator=logits_generator,
            dtype=torch.float32,
            device=device,
        )
        top_ids = logits.topk(20, dim=-1).indices
        draft_token_ids = torch.stack(
            (
                top_ids[0, seed % 20],
                top_ids[1, (seed * 7) % 20],
                torch.tensor(seed % vocab_size, device=device),
            )
        ).to(torch.int32)
        metadata = SimpleNamespace(
            num_draft_tokens=[3],
            max_spec_len=3,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_token_ids=draft_token_ids,
        )

        dense_logits = apply_sampling_constraints(
            logits.clone(),
            cu_num_draft_tokens,
            sampling,
        )
        rng_state = torch.cuda.get_rng_state(device)
        request_rng_state = (
            request_generator.get_state() if request_generator is not None else None
        )
        expected = rejection_sample(
            draft_token_ids,
            [3],
            3,
            cu_num_draft_tokens,
            None,
            dense_logits,
            bonus_token_ids,
            sampling,
        )
        expected_rng_state = torch.cuda.get_rng_state(device)
        expected_request_rng_state = (
            request_generator.get_state() if request_generator is not None else None
        )
        torch.cuda.set_rng_state(rng_state, device)
        if request_generator is not None:
            assert request_rng_state is not None
            request_generator.set_state(request_rng_state)
        actual = sm70_compact_topk20_rejection_sample(
            metadata,
            logits.clone(),
            bonus_token_ids,
            sampling,
            relative_prob,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.cuda.get_rng_state(device), expected_rng_state, rtol=0, atol=0
        )
        if request_generator is not None:
            assert expected_request_rng_state is not None
            torch.testing.assert_close(
                request_generator.get_state(),
                expected_request_rng_state,
                rtol=0,
                atol=0,
            )


def test_compact_topk20_recovery_excludes_rejected_draft_token() -> None:
    """Recovery for a greedy draft must sample max(p - delta_d, 0)."""
    device = torch.device(current_platform.device_type)
    max_spec_len = 3
    vocab_size = 32
    support_size = 20
    draft_token_id = 5

    output_token_ids = torch.full(
        (1, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor(
        [draft_token_id, 6, 7], dtype=torch.int32, device=device
    )
    target_topk_ids = torch.arange(
        support_size, dtype=torch.int64, device=device
    ).repeat(max_spec_len, 1)
    target_topk_probs = torch.zeros(
        (max_spec_len, support_size), dtype=torch.float32, device=device
    )
    target_topk_probs[:, 0] = 0.90
    target_topk_probs[:, draft_token_id] = 0.01
    target_topk_probs[:, 1] = 0.09
    bonus_token_ids = torch.tensor([[9]], dtype=torch.int32, device=device)
    uniform_probs = torch.full(
        (max_spec_len,), 0.5, dtype=torch.float64, device=device
    )

    # If the rejected draft token incorrectly retains target probability, its
    # deliberately dominant exponential score wins recovery. Correct standard
    # rejection assigns it zero mass and recovers token 0 instead.
    inv_q = torch.ones((1, vocab_size), dtype=torch.float32, device=device)
    inv_q[0, draft_token_id] = 10_000.0

    _sm70_compact_topk20_rejection_kernel[(1,)](
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_topk_ids,
        target_topk_probs,
        bonus_token_ids,
        uniform_probs,
        inv_q,
        max_spec_len,
        vocab_size=vocab_size,
        relative_prob=0.05,
        force_target_relative=True,
        support_size=support_size,
        support_block=32,
    )

    assert output_token_ids[0].tolist() == [
        0,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
        PLACEHOLDER_TOKEN_ID,
    ]
