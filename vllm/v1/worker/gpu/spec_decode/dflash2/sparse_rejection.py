# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strictly gated compact target rejection for DFlash2 on SM70."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)

_TARGET_TOP_K = 20


def _supports_sparse_sampling_contract(
    rejection_sampler: RejectionSampler,
    input_batch: InputBatch,
) -> bool:
    """Whether compact logits preserve every requested sampling transform."""
    if rejection_sampler.rejection_sample_method != "standard":
        return False
    # Start with the single-request path used by the latency target. The
    # kernel supports batches, but mixed-request graph validation is a
    # separate promotion gate.
    if input_batch.num_reqs != 1 or np.any(input_batch.is_prefilling_np):
        return False

    sampler = rejection_sampler.sampler
    idx = input_batch.idx_mapping_np
    states = sampler.sampling_states
    temperatures = states.temperature.np[idx]
    top_k = states.top_k.np[idx]
    top_p = states.top_p.np[idx]
    if np.any(temperatures <= 0.0):
        return False
    if np.any(top_k != _TARGET_TOP_K):
        return False
    if np.any((top_p <= 0.0) | (top_p > 1.0)):
        return False
    if np.any(states.min_p.np[idx] != 0.0):
        return False

    if np.any(sampler.penalties_state.use_penalty[idx]):
        return False
    if np.any(sampler.logit_bias_state.use_logit_bias[idx]):
        return False
    if np.any(sampler.bad_words_state.num_bad_words.np[idx] != 0):
        return False
    if states.max_num_logprobs(idx) != NO_LOGPROBS:
        return False
    if sampler.logprob_token_ids_state.max_num_token_ids(idx) != 0:
        return False
    return not sampler.compute_nans


def try_dflash2_sparse_target_rejection(
    model: Any,
    speculator: Any,
    rejection_sampler: RejectionSampler,
    sample_hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
) -> SamplerOutput | None:
    """Sample from compact target/draft supports, or return ``None`` safely."""
    if not envs.VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION:
        return None
    if not isinstance(speculator, DFlash2Speculator):
        return None
    if grammar_output is not None or input_batch.has_structured_output_reqs:
        return None
    if sample_hidden_states.device.type != "cuda":
        return None
    if torch.cuda.get_device_capability(sample_hidden_states.device) != (7, 0):
        return None
    if not hasattr(model, "get_topk_tokens_and_logits"):
        return None
    if not _supports_sparse_sampling_contract(rejection_sampler, input_batch):
        return None

    sparse_draft_logits = speculator.get_sparse_draft_logits()
    if sparse_draft_logits is None:
        return None
    draft_topk_ids, draft_topk_logits = sparse_draft_logits
    target_topk_ids, target_topk_logits = model.get_topk_tokens_and_logits(
        sample_hidden_states,
        _TARGET_TOP_K,
    )
    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    pos = input_batch.positions[input_batch.logits_indices]
    sampled, num_sampled = dflash2_sparse_topk_rejection_sample(
        target_topk_ids,
        target_topk_logits,
        draft_topk_ids,
        draft_topk_logits,
        draft_sampled,
        input_batch.cu_num_logits,
        pos,
        input_batch.idx_mapping,
        rejection_sampler.sampler.sampling_states.temperature.gpu,
        rejection_sampler.sampler.sampling_states.top_p.gpu,
        rejection_sampler.sampler.sampling_states.seeds.gpu,
        rejection_sampler.num_speculative_steps,
        use_fp64=rejection_sampler.sampler.use_fp64_gumbel,
    )
    logger.info_once("Using SM70 DFlash2 compact target top-k rejection sampling.")
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
    )
