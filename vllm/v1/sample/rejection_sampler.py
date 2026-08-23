# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, SamplerOutput
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words_with_drafts
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)

_SPEC_ALIGNMENT_DUMP_COUNT = 0
_SPEC_ALIGNMENT_STEP_COUNTER = 0
_QWEN38_TARGET_REFERENCE_DUMP_COUNTER = 0

PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
GREEDY_TEMPERATURE: tl.constexpr = 0
# Maximum number of speculative draft tokens allowed per request in a single
# step. This value is chosen to be large enough to handle typical use cases.
MAX_SPEC_LEN = 128


def _maybe_dump_qwen38_target_reference(
    raw_target_logits: torch.Tensor,
    metadata: SpecDecodeMetadata,
) -> None:
    """Dump the target row conditioned only on the committed request prefix."""
    dump_dir = os.environ.get("VLLM_SM70_DUMP_SAMPLER_LOGITS_DIR")
    if not dump_dir:
        return
    enable_file = os.environ.get("VLLM_SM70_DUMP_SAMPLER_LOGITS_ENABLE_FILE")
    if enable_file and not os.path.exists(enable_file):
        return
    if metadata.num_draft_tokens != [3] or raw_target_logits.shape[0] != 3:
        return

    global _QWEN38_TARGET_REFERENCE_DUMP_COUNTER
    _QWEN38_TARGET_REFERENCE_DUMP_COUNTER += 1
    max_steps = int(os.environ.get("VLLM_SM70_DUMP_SAMPLER_LOGITS_MAX_STEPS", "0"))
    if max_steps > 0 and max_steps < _QWEN38_TARGET_REFERENCE_DUMP_COUNTER:
        return

    os.makedirs(dump_dir, exist_ok=True)
    reference_logits = raw_target_logits[:1]
    torch.save(
        {
            "logits": reference_logits.detach().cpu(),
            "stage": "target_committed_prefix",
            "step": _QWEN38_TARGET_REFERENCE_DUMP_COUNTER,
            "pid": os.getpid(),
            "shape": tuple(reference_logits.shape),
            "dtype": str(reference_logits.dtype),
            "draft_token_ids": metadata.draft_token_ids.detach().cpu(),
        },
        os.path.join(
            dump_dir,
            (
                f"target_reference_pid{os.getpid()}"
                f"_step{_QWEN38_TARGET_REFERENCE_DUMP_COUNTER:04d}.pt"
            ),
        ),
    )


_SM70_COMPACT_TOP_K = 20


@triton.jit(do_not_specialize=["max_spec_len"])
def _sm70_compact_topk20_rejection_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_topk_ids_ptr,
    target_topk_probs_ptr,
    bonus_token_ids_ptr,
    uniform_probs_ptr,
    inv_q_ptr,
    max_spec_len,
    vocab_size: tl.constexpr,
    relative_prob: tl.constexpr,
    force_target_relative: tl.constexpr,
    support_size: tl.constexpr,
    support_block: tl.constexpr,
):
    """Batch-row rejection over compact target support.

    The production drafter is greedy, so its proposal probability is one.
    Recovery therefore samples from ``max(p - delta_draft, 0)``. The full-
    vocabulary exponential-noise tensor is retained outside this kernel to
    preserve the generic sampler's RNG consumption and recovery draw.
    """
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(
        cu_num_draft_tokens_ptr + req_idx - 1
    )
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    support_offsets = tl.arange(0, support_block)
    support_mask = support_offsets < support_size

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            flat_pos = start_idx + pos
            support_base = flat_pos * support_size
            target_ids = tl.load(
                target_topk_ids_ptr + support_base + support_offsets,
                mask=support_mask,
                other=0,
            )
            target_probs = tl.load(
                target_topk_probs_ptr + support_base + support_offsets,
                mask=support_mask,
                other=0.0,
            )
            draft_token_id = tl.load(draft_token_ids_ptr + flat_pos).to(tl.int64)
            draft_prob = tl.max(
                tl.where(target_ids == draft_token_id, target_probs, 0.0)
            )
            mode_prob = tl.max(target_probs)
            uniform_prob = tl.load(uniform_probs_ptr + flat_pos)
            forced = False
            if force_target_relative:
                forced = draft_prob >= relative_prob * mode_prob
            accepted = forced or (draft_prob > 0.0 and draft_prob >= uniform_prob)

            if accepted:
                token_id = draft_token_id
            else:
                inv_q = tl.load(
                    inv_q_ptr
                    + req_idx * vocab_size
                    + target_ids,
                    mask=support_mask,
                    other=0.0,
                )
                # The production drafter is greedy, so q is a delta at the
                # proposed token. Standard rejection recovery samples from
                # max(p - q, 0), which gives the rejected draft token exactly
                # zero recovery mass. Leaving it in the compact support changes
                # the distribution and can immediately resample the token that
                # was just rejected.
                recovery_scores = tl.where(
                    target_ids == draft_token_id,
                    0.0,
                    target_probs * inv_q,
                )
                recovered_offset = tl.argmax(recovery_scores, axis=0)
                token_id = tl.load(
                    target_topk_ids_ptr + support_base + recovered_offset
                )
                rejected = True

            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr
            + req_idx * (max_spec_len + 1)
            + num_draft_tokens,
            bonus_token_id,
        )


@triton.jit
def _force_target_relative_prob_kernel(
    target_probs_ptr,
    draft_token_ids_ptr,
    uniform_probs_ptr,
    vocab_size: tl.constexpr,
    relative_prob: tl.constexpr,
    block_size: tl.constexpr,
):
    """Force plausible draft tokens without a separate max/gather/mask stack."""
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    row_start = token_idx * vocab_size
    mode_prob = 0.0
    for block_start in range(0, vocab_size, block_size):
        vocab_offsets = block_start + offsets
        values = tl.load(
            target_probs_ptr + row_start + vocab_offsets,
            mask=vocab_offsets < vocab_size,
            other=0.0,
        )
        mode_prob = tl.maximum(mode_prob, tl.max(values))

    draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
    draft_prob = tl.load(target_probs_ptr + row_start + draft_token_id)
    uniform_prob = tl.load(uniform_probs_ptr + token_idx)
    tl.store(
        uniform_probs_ptr + token_idx,
        tl.where(draft_prob >= relative_prob * mode_prob, 0.0, uniform_prob),
    )


def _parse_step_filter(raw_steps: str | None) -> set[int] | None:
    if not raw_steps:
        return None
    steps: set[int] = set()
    try:
        for item in raw_steps.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start_text, end_text = item.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if start < 0 or end < start:
                    return set()
                steps.update(range(start, end + 1))
                continue
            step = int(item)
            if step < 0:
                return set()
            steps.add(step)
        return steps
    except ValueError:
        return set()


def _token_matching_processor_safe(processor: object) -> bool:
    if isinstance(processor, MinTokensLogitsProcessor):
        return not processor.min_toks
    return False


def _token_matching_sampling_enabled(
    sampling_metadata: SamplingMetadata,
) -> bool:
    """Use 0.0.3-style token matching for MTP stochastic sampling when safe."""
    if os.getenv("VLLM_MTP_STOCHASTIC_TOKEN_MATCHING", "0") != "1":
        return False
    if sampling_metadata.max_num_logprobs is not None:
        return False
    if not sampling_metadata.no_penalties:
        return False
    if sampling_metadata.allowed_token_ids_mask is not None:
        return False
    if sampling_metadata.bad_words_token_ids:
        return False
    if sampling_metadata.generators:
        return False
    if any(
        not _token_matching_processor_safe(processor)
        for processor in sampling_metadata.logitsprocs.argmax_invariant
    ):
        return False
    if any(
        not _token_matching_processor_safe(processor)
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant
    ):
        return False
    return True


class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens:
        Tokens are finally generated with the rejection sampler.
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.sampler = sampler
        logprobs_mode = self.sampler.logprobs_mode
        self.is_processed_logprobs_mode = logprobs_mode.startswith("processed")
        self.is_logits_logprobs_mode = logprobs_mode.endswith("logits")

        self.synthetic_conditional_rates: torch.Tensor | None = None
        if (
            spec_config is not None
            and spec_config.rejection_sample_method == "synthetic"
        ):
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        self.synthetic_mode = self.synthetic_conditional_rates is not None
        self.sm70_compact_topk20_enabled = (
            os.getenv("VLLM_SM70_MTP_COMPACT_TOPK20_REJECTION", "0") == "1"
        )
        self.sm70_fused_rejection_aux_enabled = (
            os.getenv("VLLM_SM70_FUSED_REJECTION_AUX", "0") == "1"
        )
        self.sm70_compact_topk20_relative_prob: float | None = None
        if self.sm70_compact_topk20_enabled:
            relative_prob_raw = os.getenv("VLLM_MTP_FORCE_TARGET_RELATIVE_PROB")
            if relative_prob_raw is not None:
                relative_prob = float(relative_prob_raw)
                if not 0.0 < relative_prob <= 1.0:
                    raise ValueError(
                        "VLLM_MTP_FORCE_TARGET_RELATIVE_PROB must be in (0, 1]"
                    )
                self.sm70_compact_topk20_relative_prob = relative_prob

    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        draft_probs: torch.Tensor | None,
        # [num_tokens + batch_size, vocab_size]
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """
        Args:
            metadata:
                Metadata for spec decoding.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [num_tokens, vocab_size]. Can be None if probabilities are
                not provided, which is the case for ngram spec decode.
            logits (torch.Tensor):
                Target model's logits probability distribution.
                Shape is [num_tokens + batch_size, vocab_size]. Here,
                probabilities from different requests are flattened into a
                single tensor because this is the shape of the output logits.
                NOTE: `logits` can be updated in place to save memory.
            sampling_metadata (vllm.v1.sample.metadata.SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            SamplerOutput:
                Contains the final output token IDs and their logprobs if
                requested.
        """
        assert metadata.max_spec_len <= MAX_SPEC_LEN

        if (
            draft_probs is None
            and not sampling_metadata.all_greedy
            and _token_matching_sampling_enabled(sampling_metadata)
        ):
            return self._sample_by_token_matching(metadata, logits, sampling_metadata)

        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices

        # When indexing with a tensor (bonus_logits_indices), PyTorch
        # creates a new tensor with separate storage from the original
        # logits tensor. This means any in-place operations on bonus_logits
        # won't affect the original logits tensor.
        assert logits is not None
        bonus_logits = logits[bonus_logits_indices]
        bonus_sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=replace(
                sampling_metadata,
                max_num_logprobs=-1,
            ),
            predict_bonus_token=True,
            # Override the logprobs mode to return logits because they are
            # needed later to compute the accepted token logprobs.
            logprobs_mode_override="processed_logits"
            if self.is_processed_logprobs_mode
            else "raw_logits",
        )
        bonus_token_ids = bonus_sampler_output.sampled_token_ids

        # Just like `bonus_logits`, `target_logits` is a new tensor with
        # separate storage from the original `logits` tensor. Therefore,
        # it is safe to update `target_logits` in place.
        raw_target_logits = logits[target_logits_indices]
        # Use float32 for the target_logits.
        raw_target_logits = raw_target_logits.to(torch.float32)
        _maybe_dump_qwen38_target_reference(raw_target_logits, metadata)
        target_logits = raw_target_logits
        if not self.is_processed_logprobs_mode:
            # Clone raw_target_logits before applying processors to preserve
            # the original raw logits for logprobs computation, since
            # apply_logits_processors modifies the tensor in-place.
            target_logits = target_logits.clone()
        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        if _sm70_compact_topk20_rejection_enabled(
            metadata,
            draft_probs,
            target_logits,
            sampling_metadata,
            self.synthetic_mode,
            self.sm70_compact_topk20_enabled,
        ):
            output_token_ids = sm70_compact_topk20_rejection_sample(
                metadata,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
                self.sm70_compact_topk20_relative_prob,
            )
            return SamplerOutput(
                sampled_token_ids=output_token_ids,
                logprobs_tensors=None,
            )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )
        fused_rejection_aux = (
            self.sm70_fused_rejection_aux_enabled
            and target_logits.is_cuda
            and sampling_metadata.all_random
            and sampling_metadata.max_num_logprobs is None
            and metadata.max_spec_len == 3
            and bool(metadata.num_draft_tokens)
            and all(num_draft == 3 for num_draft in metadata.num_draft_tokens)
        )
        if self.sm70_fused_rejection_aux_enabled:
            if fused_rejection_aux:
                logger.info_once("Using fused rejection auxiliary outputs")
            else:
                logger.info_once(
                    "Fused rejection auxiliary outputs gated off: "
                    f"cuda={target_logits.is_cuda} "
                    f"all_random={sampling_metadata.all_random} "
                    f"max_num_logprobs={sampling_metadata.max_num_logprobs} "
                    f"num_draft_tokens={metadata.num_draft_tokens} "
                    f"max_spec_len={metadata.max_spec_len}"
                )
        rejection_result = rejection_sample(
            metadata.draft_token_ids,
            metadata.num_draft_tokens,
            metadata.max_spec_len,
            metadata.cu_num_draft_tokens,
            draft_probs,
            target_logits,
            bonus_token_ids,
            sampling_metadata,
            synthetic_mode=self.synthetic_mode,
            synthetic_conditional_rates=self.synthetic_conditional_rates,
            return_aux=fused_rejection_aux,
        )
        valid_sampled_token_count = None
        next_token_ids = None
        if fused_rejection_aux:
            assert isinstance(rejection_result, tuple)
            (
                output_token_ids,
                valid_sampled_token_count,
                next_token_ids,
            ) = rejection_result
        else:
            assert isinstance(rejection_result, torch.Tensor)
            output_token_ids = rejection_result
        self._maybe_dump_alignment(
            metadata=metadata,
            draft_probs=draft_probs,
            target_logits=target_logits,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            bonus_token_ids=bonus_token_ids,
            output_token_ids=output_token_ids,
            sampling_metadata=sampling_metadata,
        )

        logprobs_tensors = None
        if sampling_metadata.max_num_logprobs is not None:
            logprobs_tensors = self._get_logprobs_tensors(
                sampling_metadata.max_num_logprobs,
                metadata,
                logits,
                target_logits if self.is_processed_logprobs_mode else raw_target_logits,
                bonus_sampler_output.logprobs_tensors.logprobs,
                output_token_ids,
            )

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=logprobs_tensors,
            valid_sampled_token_count=valid_sampled_token_count,
            next_token_ids=next_token_ids,
        )

    def _sample_by_token_matching(
        self,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        num_sampled_tokens = logits.shape[0]
        temperature = expand_batch_to_tokens(
            sampling_metadata.temperature,
            metadata.cu_num_sampled_tokens,
            num_sampled_tokens,
            replace_from=GREEDY_TEMPERATURE,
            replace_to=1,
        )
        top_k = None
        if sampling_metadata.top_k is not None:
            top_k = expand_batch_to_tokens(
                sampling_metadata.top_k,
                metadata.cu_num_sampled_tokens,
                num_sampled_tokens,
            )
        top_p = None
        if sampling_metadata.top_p is not None:
            top_p = expand_batch_to_tokens(
                sampling_metadata.top_p,
                metadata.cu_num_sampled_tokens,
                num_sampled_tokens,
            )
        expanded_sampling_metadata = replace(
            sampling_metadata,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_num_logprobs=None,
            output_token_ids=[],
            spec_token_ids=None,
        )
        sampled_token_ids = self.sampler(
            logits=logits,
            sampling_metadata=expanded_sampling_metadata,
        ).sampled_token_ids.view(-1)
        output_token_ids = token_match_sample(metadata, sampled_token_ids)
        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

    @staticmethod
    def _maybe_dump_alignment(
        *,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        target_logits: torch.Tensor,
        target_logits_indices: torch.Tensor,
        bonus_logits_indices: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        output_token_ids: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> None:
        global _SPEC_ALIGNMENT_DUMP_COUNT, _SPEC_ALIGNMENT_STEP_COUNTER
        _SPEC_ALIGNMENT_STEP_COUNTER += 1
        dump_limit = envs.VLLM_SPEC_DUMP_ALIGNMENT_LIMIT
        target_steps = _parse_step_filter(envs.VLLM_SPEC_DUMP_ALIGNMENT_STEPS)
        step_matches = (
            target_steps is None or _SPEC_ALIGNMENT_STEP_COUNTER in target_steps
        )
        should_dump_alignment = (
            envs.VLLM_SPEC_DUMP_ALIGNMENT
            and step_matches
            and dump_limit > _SPEC_ALIGNMENT_DUMP_COUNT
            and sum(metadata.num_draft_tokens) >= min(4, metadata.max_spec_len)
        )
        if not should_dump_alignment:
            return

        with torch.no_grad():
            k = min(10, target_logits.shape[-1])
            target_topk = torch.topk(target_logits, k=k, dim=-1)
            target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
            draft_token_ids_i64 = metadata.draft_token_ids.to(torch.int64)
            # CUDA graph profiling can call the sampler with zero-filled dummy
            # draft ids. Skipping those rows keeps the diagnostic focused on
            # real request traffic instead of warmup placeholders.
            if bool(torch.all(draft_token_ids_i64 == 0).item()):
                return
            draft_target_logits = target_logits.gather(
                1, draft_token_ids_i64.view(-1, 1)
            ).squeeze(1)
            draft_target_probs = target_probs.gather(
                1, draft_token_ids_i64.view(-1, 1)
            ).squeeze(1)
            target_topk_matches = target_topk.indices == draft_token_ids_i64.view(
                -1, 1
            )
            target_topk_rank = torch.where(
                target_topk_matches.any(dim=-1),
                target_topk_matches.to(torch.int32).argmax(dim=-1) + 1,
                torch.full(
                    (target_topk_matches.shape[0],),
                    -1,
                    dtype=torch.int32,
                    device=target_topk_matches.device,
                ),
            )
            payload = {
                "rank": int(os.getenv("RANK", "-1")),
                "step": _SPEC_ALIGNMENT_STEP_COUNTER,
                "draft_token_ids": metadata.draft_token_ids.detach().cpu(),
                "num_draft_tokens": list(metadata.num_draft_tokens),
                "cu_num_draft_tokens": metadata.cu_num_draft_tokens.detach().cpu(),
                "output_token_ids": output_token_ids.detach().cpu(),
                "output_valid_counts": (
                    (output_token_ids != PLACEHOLDER_TOKEN_ID)
                    .sum(dim=-1)
                    .detach()
                    .cpu()
                ),
                "target_argmax": target_logits.argmax(dim=-1).detach().cpu(),
                "target_topk_ids": target_topk.indices.detach().cpu(),
                "target_topk_values": target_topk.values.detach().cpu(),
                "draft_token_target_topk_rank": target_topk_rank.detach().cpu(),
                "draft_target_logits": draft_target_logits.detach().cpu(),
                "draft_target_probs": draft_target_probs.detach().cpu(),
                "target_logits_indices": target_logits_indices.detach().cpu(),
                "bonus_logits_indices": bonus_logits_indices.detach().cpu(),
                "bonus_token_ids": bonus_token_ids.detach().cpu(),
                "all_greedy": sampling_metadata.all_greedy,
                "all_random": sampling_metadata.all_random,
                "sampling_output_token_ids_tail": [
                    list(ids[-32:]) for ids in sampling_metadata.output_token_ids
                ],
                "sampling_spec_token_ids": [
                    list(ids) for ids in (sampling_metadata.spec_token_ids or [])
                ],
            }
            if draft_probs is not None:
                draft_token_probs = draft_probs.gather(
                    1, draft_token_ids_i64.view(-1, 1)
                ).squeeze(1)
                residual_probs = (target_probs - draft_probs).clamp_min_(0.0)
                residual_mass = residual_probs.sum(dim=-1)
                payload["draft_token_probs"] = draft_token_probs.detach().cpu()
                payload["draft_acceptance_caps"] = (
                    (
                        draft_target_probs
                        / draft_token_probs.clamp_min(torch.finfo(torch.float32).tiny)
                    )
                    .clamp_max(1.0)
                    .detach()
                    .cpu()
                )
                payload["recovered_residual_mass"] = residual_mass.detach().cpu()
                draft_topk = torch.topk(draft_probs, k=k, dim=-1)
                payload["draft_topk_ids"] = draft_topk.indices.detach().cpu()
                payload["draft_topk_values"] = draft_topk.values.detach().cpu()
            _SPEC_ALIGNMENT_DUMP_COUNT += 1
            dump_path = (
                f"/tmp/spec_alignment_pid{os.getpid()}_"
                f"step{_SPEC_ALIGNMENT_STEP_COUNTER:06d}_"
                f"{_SPEC_ALIGNMENT_DUMP_COUNT}.pt"
            )
            torch.save(payload, dump_path)
            logger.warning(
                "Dumped speculative alignment diagnostics to %s", dump_path
            )

    def _get_logprobs_tensors(
        self,
        max_num_logprobs: int,
        metadata: SpecDecodeMetadata,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        bonus_logits: torch.Tensor,
        sampled_token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        cu_num_sampled_tokens = torch.zeros_like(metadata.cu_num_sampled_tokens)
        cu_num_sampled_tokens[1:] = metadata.cu_num_sampled_tokens[:-1]

        # Collect target and bonus logits.
        bonus_logits_indices = metadata.bonus_logits_indices
        target_logits_indices = metadata.target_logits_indices
        final_logits = torch.zeros_like(logits, dtype=torch.float32)
        final_logits[target_logits_indices] = target_logits.to(torch.float32)
        final_logits[bonus_logits_indices] = bonus_logits.to(torch.float32)

        # NOTE: To avoid cpu-gpu synchronization, we now simply compute indices for
        # all draft tokens, including the rejected ones. The rejected tokens will
        # be filtered out in the `parse_output`.
        logit_start_indices = cu_num_sampled_tokens
        offsets = torch.arange(
            sampled_token_ids.shape[-1],
            device=logit_start_indices.device,
            dtype=logit_start_indices.dtype,
        )
        accepted_logit_indices = (
            logit_start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        accepted_logit_indices.clamp_(max=final_logits.shape[0] - 1)
        accepted_tokens = sampled_token_ids.clone().flatten()
        # we replace rejected token ids with 0 to avoid gather_logprobs error
        accepted_tokens[accepted_tokens == PLACEHOLDER_TOKEN_ID] = 0

        # Compute logprobs for accepted tokens.
        accepted_logits = final_logits[accepted_logit_indices]
        accepted_logprobs = (
            accepted_logits
            if self.is_logits_logprobs_mode
            else self.sampler.compute_logprobs(accepted_logits)
        )
        return self.sampler.gather_logprobs(
            accepted_logprobs,
            max_num_logprobs,
            accepted_tokens.to(torch.int64),
        )

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Parse the output of the rejection sampler.
        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID` by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.
            discard_req_indices: Optional row indices to discard tokens in.
            logprobs_tensors: Optional logprobs tensors to filter.
        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != PLACEHOLDER_TOKEN_ID) & (
            output_token_ids_np < vocab_size
        )
        output_logprobs = None
        if logprobs_tensors is not None:
            cu_num_tokens = [0] + valid_mask.sum(axis=1).cumsum().tolist()
            filtered_tensors = logprobs_tensors.filter(valid_mask.flatten())
            output_logprobs = filtered_tensors.tolists(cu_num_tokens)

        if len(discard_req_indices) > 0:
            valid_mask[discard_req_indices] = False
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs, output_logprobs

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
            sampling_metadata.bad_words_token_ids or has_penalties
        )
        holder = sampling_metadata.thinking_budget_state_holder
        needs_thinking = holder is not None and holder.has_tracked_requests()

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words or needs_thinking:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        repeat_indices: torch.Tensor | None = None
        need_repeat_indices = (
            sampling_metadata.allowed_token_ids_mask is not None
            or has_penalties
            or needs_thinking
        )
        if need_repeat_indices:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            logits = self.apply_penalties(
                logits, sampling_metadata, metadata, repeat_indices, output_token_ids
            )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            if isinstance(processor, MinTokensLogitsProcessor):
                logits = processor.apply_with_spec_decode(
                    logits, metadata.num_draft_tokens
                )
        if holder is not None and holder.has_tracked_requests():
            assert repeat_indices is not None
            holder.update_state(
                output_token_ids,
                sampling_metadata.spec_token_ids,
                repeat_indices,
            )
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token=False,
                spec_token_ids=sampling_metadata.spec_token_ids,
            )
        return logits

    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if sampling_metadata.no_penalties:
            return logits

        assert sampling_metadata.prompt_token_ids is not None

        prompt_token_ids = sampling_metadata.prompt_token_ids[repeat_indices]
        presence_penalties = sampling_metadata.presence_penalties[repeat_indices]
        frequency_penalties = sampling_metadata.frequency_penalties[repeat_indices]
        repetition_penalties = sampling_metadata.repetition_penalties[repeat_indices]

        logits = apply_all_penalties(
            logits,
            prompt_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            output_token_ids,
        )
        return logits

    @staticmethod
    def _combine_outputs_with_spec_tokens(
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None = None,
    ) -> list[list[int]]:
        if spec_token_ids is None:
            return output_token_ids

        result = []
        for out, spec in zip(output_token_ids, spec_token_ids):
            if len(spec) == 0:
                continue
            result.append(out)
            for i in range(len(spec) - 1):
                result.append([*result[-1], spec[i]])
        return result


def _sm70_compact_topk20_rejection_enabled(
    metadata: SpecDecodeMetadata,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool,
    compact_enabled: bool,
) -> bool:
    """Gate the exact production-shape compact rejection path.

    InputBatch validates top-k, top-p, temperature, and batch shape from its
    CPU-side mirrors when request metadata changes. Consuming that boolean here
    avoids a device synchronization on every speculative decode step.
    """
    enabled = (
        compact_enabled
        and target_logits.is_cuda
        and draft_probs is None
        and not synthetic_mode
        and sampling_metadata.all_random
        and sampling_metadata.sm70_compact_topk20_eligible
        and sampling_metadata.top_k is not None
        and sampling_metadata.top_p is not None
        and sampling_metadata.max_num_logprobs is None
        and len(metadata.num_draft_tokens) == 1
        and metadata.max_spec_len == 3
        and metadata.num_draft_tokens == [3]
    )
    if enabled:
        logger.info_once(
            "Using compact top-k-20 rejection/recovery for batch-1 SM70 MTP3"
        )
    return enabled


def _sm70_compact_topk20_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return token IDs and normalized probabilities on top-k-20 support."""
    logits.div_(temperature[0])
    topk_values, topk_ids = torch.topk(
        logits,
        _SM70_COMPACT_TOP_K,
        dim=-1,
        sorted=True,
    )
    topk_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    exclusive_mass = topk_probs.cumsum(dim=-1) - topk_probs
    topk_values.masked_fill_(exclusive_mass >= top_p[0], -float("inf"))
    constrained_probs = topk_values.softmax(dim=-1, dtype=torch.float32)
    return topk_ids, constrained_probs


def sm70_compact_topk20_rejection_sample(
    metadata: SpecDecodeMetadata,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    relative_prob: float | None,
) -> torch.Tensor:
    """Production top-k-20 rejection without dense target probabilities.

    ``relative_prob=None`` preserves standard rejection-sampling semantics.
    A configured value applies the same optional forced-acceptance heuristic
    as the dense fallback path.
    """
    batch_size = len(metadata.num_draft_tokens)
    num_tokens, vocab_size = target_logits.shape
    device = target_logits.device
    assert batch_size == 1
    assert num_tokens == sum(metadata.num_draft_tokens)
    assert sampling_metadata.temperature is not None
    assert sampling_metadata.top_p is not None
    assert _SM70_COMPACT_TOP_K <= 32

    target_topk_ids, target_topk_probs = _sm70_compact_topk20_probs(
        target_logits,
        sampling_metadata.temperature,
        sampling_metadata.top_p,
    )
    uniform_probs = generate_uniform_probs(
        num_tokens,
        metadata.num_draft_tokens,
        sampling_metadata.generators,
        device,
    )

    # Match sample_recovered_tokens' one full-vocabulary exponential-noise row
    # per request. Only the 20 selected entries are read by the fused kernel,
    # but generating the full row preserves RNG advancement and recovery draws.
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for req_idx, generator in sampling_metadata.generators.items():
        if metadata.num_draft_tokens[req_idx] > 0:
            q[req_idx].exponential_(generator=generator)
    inv_q = q.reciprocal()

    output_token_ids = torch.full(
        (batch_size, metadata.max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )
    _sm70_compact_topk20_rejection_kernel[(batch_size,)](
        output_token_ids,
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        target_topk_ids,
        target_topk_probs,
        bonus_token_ids,
        uniform_probs,
        inv_q,
        metadata.max_spec_len,
        vocab_size=vocab_size,
        relative_prob=relative_prob if relative_prob is not None else 1.0,
        force_target_relative=relative_prob is not None,
        support_size=_SM70_COMPACT_TOP_K,
        support_block=32,
    )
    return output_token_ids


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [batch_size]
    num_draft_tokens: list[int],
    max_spec_len: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_logits: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    synthetic_mode: bool = False,
    synthetic_conditional_rates: torch.Tensor | None = None,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_logits.ndim == 2

    batch_size = len(num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_logits.shape[-1]
    device = target_logits.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_logits.shape == (num_tokens, vocab_size)
    if return_aux:
        # The first specialized consumer is the production random-sampling
        # q=4 path. Keep every other rejection mode on its established
        # contract; in particular this excludes zero-draft and mixed-greedy
        # rows whose auxiliary outputs would otherwise be undefined.
        assert sampling_metadata.all_random
        assert num_draft_tokens
        assert all(num_draft == 3 for num_draft in num_draft_tokens)
        assert max_spec_len == 3

    # Create output buffer.
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    valid_sampled_token_count = (
        torch.empty((batch_size,), dtype=torch.int32, device=device)
        if return_aux
        else None
    )
    next_token_ids = (
        torch.empty((batch_size,), dtype=torch.int32, device=device)
        if return_aux
        else None
    )

    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE

    # Generate uniform probabilities before either kernel because synthetic
    # mode needs them in the greedy kernel too.  Skip only when all requests
    # are greedy *and* synthetic mode is off (the standard fast-path).
    # [num_tokens]
    uniform_probs: torch.Tensor | None = None
    if synthetic_mode or not sampling_metadata.all_greedy:
        uniform_probs = generate_uniform_probs(
            num_tokens,
            num_draft_tokens,
            sampling_metadata.generators,
            device,
        )

    if not sampling_metadata.all_random:
        # Rejection sampling for greedy sampling requests.
        target_argmax = target_logits.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            is_greedy,
            max_spec_len,
            uniform_probs,
            synthetic_conditional_rates,
            SYNTHETIC_MODE=synthetic_mode,
        )
        if sampling_metadata.all_greedy:
            return output_token_ids

    # Compute probability distribution from target logits.
    target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    assert target_probs.is_contiguous()

    relative_prob_raw = os.getenv("VLLM_MTP_FORCE_TARGET_RELATIVE_PROB")
    if relative_prob_raw is not None:
        assert uniform_probs is not None
        relative_prob = float(relative_prob_raw)
        if not 0.0 < relative_prob <= 1.0:
            raise ValueError(
                "VLLM_MTP_FORCE_TARGET_RELATIVE_PROB must be in (0, 1]"
            )
        _force_target_relative_prob_kernel[(target_probs.shape[0],)](
            target_probs,
            draft_token_ids,
            uniform_probs,
            vocab_size=target_probs.shape[1],
            relative_prob=relative_prob,
            block_size=8192,
        )

    # Sample recovered tokens for each position.
    # [num_tokens]
    recovered_token_ids = sample_recovered_tokens(
        max_spec_len,
        num_draft_tokens,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        sampling_metadata,
        device,
    )

    # Rejection sampling for random sampling requests.
    assert uniform_probs is not None
    rejection_random_sample_kernel[(batch_size,)](
        output_token_ids,
        valid_sampled_token_count,
        next_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
        vocab_size,
        synthetic_conditional_rates,
        NO_DRAFT_PROBS=draft_probs is None,
        SYNTHETIC_MODE=synthetic_mode,
        WRITE_AUX=return_aux,
    )
    if return_aux:
        assert valid_sampled_token_count is not None
        assert next_token_ids is not None
        return output_token_ids, valid_sampled_token_count, next_token_ids
    return output_token_ids


def token_match_sample(
    metadata: SpecDecodeMetadata,
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size = len(metadata.num_draft_tokens)
    output_token_ids = torch.full(
        (batch_size, metadata.max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=sampled_token_ids.device,
    )
    token_match_sample_kernel[(batch_size,)](
        output_token_ids,
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        sampled_token_ids.to(torch.int32),
        metadata.max_spec_len,
    )
    return output_token_ids


def apply_sampling_constraints(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Process logits based on sampling metadata.

    This function applies temperature scaling to the logits,
    as well as top-k and top-p. For greedy decoding, it returns
    the original logits.

    Args:
        logits: Input logits tensor to be processed.
        cu_num_draft_tokens: Cumulative number of draft tokens.
        sampling_metadata: Metadata containing sampling parameters such as
            temperature and whether greedy sampling is used.

    Returns:
        torch.Tensor: Processed logits if non-greedy sampling is used,
        otherwise returns the original logits.
    """
    assert logits.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    if sampling_metadata.all_greedy:
        return logits

    num_tokens = logits.shape[0]
    temperature = expand_batch_to_tokens(
        sampling_metadata.temperature,
        cu_num_draft_tokens,
        num_tokens,
        replace_from=GREEDY_TEMPERATURE,
        replace_to=1,
    )
    # NOTE(woosuk): Update `logits` in place to avoid allocating a new tensor.
    logits.div_(temperature.unsqueeze(-1))

    # Get expanded top_k and top_p tensors.
    top_k = None
    if sampling_metadata.top_k is not None:
        top_k = expand_batch_to_tokens(
            sampling_metadata.top_k,
            cu_num_draft_tokens,
            num_tokens,
        )
    top_p = None
    if sampling_metadata.top_p is not None:
        top_p = expand_batch_to_tokens(
            sampling_metadata.top_p,
            cu_num_draft_tokens,
            num_tokens,
        )

    # NOTE(woosuk): `apply_top_k_top_p` uses sorting to calculate the mask,
    # which is slow for large vocab sizes. This may cause performance issues.
    return apply_top_k_top_p(logits, top_k, top_p)


def expand_batch_to_tokens(
    x: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor:
    """Expand [batch_size] tensor to [num_tokens] tensor based on the number of
    tokens per batch in cu_num_tokens.

    For example, if x = [a, b, c] and cu_num_tokens = [2, 5, 6], then
    num_tokens = 6, and expanded_x = [a, a, b, b, b, c].

    Args:
        x: [batch_size] tensor to expand.
        cu_num_tokens: [batch_size] tensor containing the cumulative number of
            tokens per batch. Each element represents the total number of
            tokens up to and including that batch.
        num_tokens: Total number of tokens.
        replace_from: int = 0
            Value to be replaced if it is found in x.
        replace_to: int = 0
            Value to replace with when replace_from is found.
    Returns:
        expanded_x: [num_tokens] tensor.
    """
    batch_size = x.shape[0]
    assert cu_num_tokens.shape[0] == batch_size
    expanded_x = x.new_empty(num_tokens)
    expand_kernel[(batch_size,)](
        expanded_x,
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.
    )
    return expanded_x


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor:
    """
    Generates a batch of uniform random samples, with optional seeding
    if available.

    This method creates a tensor of shape `(num_tokens, )` filled
    with uniform random values in the range [0, 1). If `generators` is provided,
    the requests with their own seeds will use the provided `torch.Generator`
    for reproducibility. The samples for the other requests will be generated
    without a seed.

    Args:
        num_tokens: int
            Total number of tokens.
        num_draft_tokens: List[List[int]]
            Number of draft tokens per request.
        generators: Optional[Dict[int, torch.Generator]]
            A dictionary mapping indices in the batch to
            `torch.Generator` objects.
        device: torch.device
            The device on which to allocate the tensor.
    Returns:
        uniform_rand: torch.Tensor
            A tensor of shape `(num_tokens, )` containing uniform
            random values in the range [0, 1).
    """
    # NOTE(woosuk): We deliberately use float64 instead of float32 here
    # because when using float32, there's a non-negligible chance that
    # uniform_prob is sampled to be exact 0.0 as reported in
    # https://github.com/pytorch/pytorch/issues/16706. Using float64
    # mitigates the issue.
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float64,
        device=device,
    )
    start_idx = 0
    for req_idx, n in enumerate(num_draft_tokens):
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if n == 0:
            continue
        end_idx = start_idx + n
        generator = generators.get(req_idx)
        if generator is not None:
            uniform_probs[start_idx:end_idx].uniform_(generator=generator)
        start_idx = end_idx
    return uniform_probs


def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: torch.Tensor | None,
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor:
    # NOTE(woosuk): Create only one distribution for each request.
    batch_size = len(num_draft_tokens)
    vocab_size = target_probs.shape[-1]
    q = torch.empty(
        (batch_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    q.exponential_()
    for i, generator in sampling_metadata.generators.items():
        # Do not generate random numbers for requests with no draft tokens.
        # This can be important for reproducibility.
        if num_draft_tokens[i] > 0:
            q[i].exponential_(generator=generator)

    inv_q = q.reciprocal()

    recovered_token_ids = torch.empty_like(draft_token_ids)
    BLOCK_SIZE = 8192
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        recovered_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=draft_probs is None,
    )
    return recovered_token_ids


@triton.jit(do_not_specialize=["max_spec_len"])
def token_match_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_logits_indices_ptr,  # [num_tokens]
    bonus_logits_indices_ptr,  # [batch_size]
    sampled_token_ids_ptr,  # [num_tokens + batch_size]
    max_spec_len,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            flat_pos = start_idx + pos
            target_idx = tl.load(target_logits_indices_ptr + flat_pos)
            target_token_id = tl.load(sampled_token_ids_ptr + target_idx)
            draft_token_id = tl.load(draft_token_ids_ptr + flat_pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_token_id,
            )
            if target_token_id != draft_token_id:
                rejected = True

    if not rejected:
        bonus_idx = tl.load(bonus_logits_indices_ptr + req_idx)
        bonus_token_id = tl.load(sampled_token_ids_ptr + bonus_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
    uniform_probs_ptr,  # [num_tokens] or None (synthetic mode only)
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    SYNTHETIC_MODE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos).to(tl.int32)
            if SYNTHETIC_MODE:
                uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
                token_id = draft_token_id if accepted else target_argmax_id
                rejected = not accepted
            else:
                token_id = target_argmax_id
                rejected = draft_token_id != target_argmax_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    valid_sampled_token_count_ptr,  # [batch_size] or None
    next_token_ids_ptr,  # [batch_size] or None
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    synthetic_conditional_rates_ptr,  # [num_speculative_tokens] or None
    NO_DRAFT_PROBS: tl.constexpr,
    SYNTHETIC_MODE: tl.constexpr,
    WRITE_AUX: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    valid_count = tl.full((), 0, tl.int32)
    next_token_id = tl.full((), -1, tl.int32)
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            if SYNTHETIC_MODE:
                rate = tl.load(synthetic_conditional_rates_ptr + pos)
                accepted = uniform_prob < rate
            else:
                if NO_DRAFT_PROBS:
                    draft_prob = 1
                else:
                    draft_prob = tl.load(
                        draft_probs_ptr
                        + (start_idx + pos) * vocab_size
                        + draft_token_id
                    )
                target_prob = tl.load(
                    target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
                # NOTE(woosuk): While the draft probability should never be 0,
                # we check it to avoid NaNs. If it happens to be 0, we reject.
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
            if accepted:
                token_id = draft_token_id
            else:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )
            valid_count = pos + 1
            next_token_id = token_id

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
        valid_count = num_draft_tokens + 1
        next_token_id = bonus_token_id

    if WRITE_AUX:
        tl.store(valid_sampled_token_count_ptr + req_idx, valid_count)
        tl.store(next_token_ids_ptr + req_idx, next_token_id)


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["replace_from", "replace_to"])
def expand_kernel(
    output_ptr,  # [num_tokens]
    input_ptr,  # [batch_size]
    cu_num_tokens_ptr,  # [batch_size]
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:  # noqa: SIM108
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_tokens_ptr + req_idx)
    num_tokens = end_idx - start_idx

    src_val = tl.load(input_ptr + req_idx)
    src_val = tl.where(src_val == replace_from, replace_to, src_val)
    offset = tl.arange(0, MAX_NUM_TOKENS)
    tl.store(output_ptr + start_idx + offset, src_val, mask=offset < num_tokens)


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    inv_q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    token_idx = start_idx + pos

    if NO_DRAFT_PROBS:
        draft_token_id = tl.load(draft_token_ids_ptr + token_idx)

    max_val = float("-inf")
    recovered_id = 0
    for v in range(0, vocab_size, BLOCK_SIZE):
        vocab_offset = v + tl.arange(0, BLOCK_SIZE)
        vocab_mask = vocab_offset < vocab_size

        if NO_DRAFT_PROBS:
            prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=(vocab_mask & (vocab_offset != draft_token_id)),
                other=0.0,
            )
        else:
            draft_prob = tl.load(
                draft_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            target_prob = tl.load(
                target_probs_ptr + token_idx * vocab_size + vocab_offset,
                mask=vocab_mask,
                other=0.0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0.0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        inv_q = tl.load(
            inv_q_ptr + req_idx * vocab_size + vocab_offset,
            mask=vocab_mask,
            other=0.0,
        )

        # Local tile reduction
        score = prob * inv_q
        local_max, local_id = tl.max(score, axis=0, return_indices=True)

        if local_max > max_val:
            max_val = local_max
            recovered_id = v + local_id

    tl.store(output_token_ids_ptr + token_idx, recovered_id)
