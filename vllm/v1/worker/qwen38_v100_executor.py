# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape executor boundary for Qwen3.8 EXL3 on V100.

The first milestone deliberately owns no model math. It provides a fail-closed
support predicate and stable device buffers for the future captured MTP3
verification cycle. Keeping the boundary inert makes it possible to validate
real-checkpoint parity before replacing any generic vLLM operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class Qwen38V100ExecutorMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ON = "on"

    @classmethod
    def parse(cls, value: str) -> Qwen38V100ExecutorMode | None:
        try:
            return cls(value.strip().lower())
        except ValueError:
            return None


class Qwen38V100SupportReason(str, Enum):
    ELIGIBLE = "eligible"
    MODE_OFF = "mode_off"
    INVALID_MODE = "invalid_mode"
    DEVICE = "device"
    CHECKPOINT = "checkpoint"
    ARCHITECTURE = "architecture"
    GEOMETRY = "geometry"
    QUANTIZATION = "quantization"
    DTYPE = "dtype"
    TOPOLOGY = "topology"
    SPECULATION = "speculation"


class Qwen38V100StepReason(str, Enum):
    ELIGIBLE = "eligible"
    EXECUTOR_DISABLED = "executor_disabled"
    BUFFERS_UNINITIALIZED = "buffers_uninitialized"
    BATCH_SIZE = "batch_size"
    VERIFY_WIDTH = "verify_width"
    SPEC_METADATA = "spec_metadata"
    EXECUTION_NOT_IMPLEMENTED = "execution_not_implemented"


@dataclass(frozen=True)
class Qwen38V100SupportDecision:
    eligible: bool
    reason: Qwen38V100SupportReason
    detail: str


@dataclass(frozen=True)
class Qwen38V100StepDecision:
    shape_matched: bool
    should_execute: bool
    reason: Qwen38V100StepReason
    detail: str


class Qwen38V100ControlField(IntEnum):
    ABI_VERSION = 0
    ACTIVE_REQUESTS = 1
    VERIFY_WIDTH = 2
    DRAFT_WIDTH = 3
    ACCEPTED_COUNT = 4
    STATE_SLOT = 5
    OUTPUT_COUNT = 6
    STATUS = 7


@dataclass
class Qwen38V100PersistentBuffers:
    """Stable-address buffers forming the versioned executor ABI."""

    control_host: torch.Tensor
    control: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    draft_token_ids: torch.Tensor
    sampled_token_ids: torch.Tensor
    accepted_count: torch.Tensor
    state_slot_selector: torch.Tensor
    rng_seed_offset: torch.Tensor
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor

    ABI_VERSION = 1
    CONTROL_WORDS = 16
    VERIFY_WIDTH = 4
    DRAFT_WIDTH = 3

    @classmethod
    def allocate(cls, device: torch.device) -> Qwen38V100PersistentBuffers:
        if device.type != "cuda":
            raise ValueError("Qwen3.8 V100 executor buffers require a CUDA device")
        control_host = torch.zeros(
            cls.CONTROL_WORDS,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        control_host[Qwen38V100ControlField.ABI_VERSION] = cls.ABI_VERSION
        control_host[Qwen38V100ControlField.VERIFY_WIDTH] = cls.VERIFY_WIDTH
        control_host[Qwen38V100ControlField.DRAFT_WIDTH] = cls.DRAFT_WIDTH
        return cls(
            control_host=control_host,
            control=control_host.to(device, non_blocking=True),
            input_ids=torch.empty(cls.VERIFY_WIDTH, dtype=torch.int32, device=device),
            positions=torch.empty(cls.VERIFY_WIDTH, dtype=torch.int64, device=device),
            slot_mapping=torch.empty(
                cls.VERIFY_WIDTH, dtype=torch.int64, device=device
            ),
            draft_token_ids=torch.empty(
                cls.DRAFT_WIDTH, dtype=torch.int32, device=device
            ),
            sampled_token_ids=torch.empty(
                cls.VERIFY_WIDTH, dtype=torch.int32, device=device
            ),
            accepted_count=torch.zeros(1, dtype=torch.int32, device=device),
            state_slot_selector=torch.zeros(1, dtype=torch.int32, device=device),
            rng_seed_offset=torch.zeros(2, dtype=torch.int64, device=device),
            cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32, device=device),
            cu_num_sampled_tokens=torch.tensor([4], dtype=torch.int32, device=device),
            target_logits_indices=torch.arange(3, dtype=torch.int32, device=device),
            bonus_logits_indices=torch.tensor([3], dtype=torch.int32, device=device),
            logits_indices=torch.arange(4, dtype=torch.int32, device=device),
        )

    def data_ptrs(self) -> tuple[int, ...]:
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.control,
                self.input_ids,
                self.positions,
                self.slot_mapping,
                self.draft_token_ids,
                self.sampled_token_ids,
                self.accepted_count,
                self.state_slot_selector,
                self.rng_seed_offset,
                self.cu_num_draft_tokens,
                self.cu_num_sampled_tokens,
                self.target_logits_indices,
                self.bonus_logits_indices,
                self.logits_indices,
            )
        )

    def commit_control(self) -> None:
        """Publish host control-word changes to the stable device buffer."""
        self.control.copy_(self.control_host, non_blocking=True)


class Qwen38V100Executor:
    """Opt-in shell for the fixed Qwen3.8/EXL3/SM70/TP4 executor."""

    EXPECTED_ARCHITECTURES = frozenset(
        {
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5EXL3TextForCausalLM",
        }
    )
    EXPECTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
    EXPECTED_MODEL_TYPE = "qwen3_5_text"
    EXPECTED_CHECKPOINT_REVISION = "c45c273b0d6ef2859cb2d85b36dd52253c80d878"
    EXPECTED_LAYER_TYPES = tuple(
        "full_attention" if layer_idx % 4 == 3 else "linear_attention"
        for layer_idx in range(64)
    )

    def __init__(
        self,
        mode: Qwen38V100ExecutorMode | None,
        support: Qwen38V100SupportDecision,
        metadata_compare: bool,
    ) -> None:
        self.mode = mode
        self.support = support
        self.buffers: Qwen38V100PersistentBuffers | None = None
        self.observed_steps = 0
        self.eligible_steps = 0
        self.static_metadata_steps = 0
        self.fast_preprocess_steps = 0
        self.metadata_compare = metadata_compare
        self._shadow_metadata: SpecDecodeMetadata | None = None
        self._prepare_packet_hosts: list[torch.Tensor] = []
        self._prepare_packet_events: list[torch.cuda.Event | None] = []
        self._prepare_packet_index = 0
        self._prepare_packet_device: torch.Tensor | None = None
        self._prepare_packet_words: int | None = None
        self._prepare_num_computed_cpu_values: torch.Tensor | None = None
        self.last_q4_gdn_metadata_fused = False

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        mode_value: str,
        is_sm70: bool,
        nvlink_island_verified: bool,
        checkpoint_revision: str | None,
        metadata_compare: bool = False,
    ) -> Qwen38V100Executor:
        mode = Qwen38V100ExecutorMode.parse(mode_value)
        support = cls._evaluate_support(
            vllm_config,
            device,
            mode=mode,
            mode_value=mode_value,
            is_sm70=is_sm70,
            nvlink_island_verified=nvlink_island_verified,
            checkpoint_revision=checkpoint_revision,
        )
        return cls(mode, support, metadata_compare)

    @classmethod
    def _evaluate_support(
        cls,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        mode: Qwen38V100ExecutorMode | None,
        mode_value: str,
        is_sm70: bool,
        nvlink_island_verified: bool,
        checkpoint_revision: str | None,
    ) -> Qwen38V100SupportDecision:
        if mode is None:
            return cls._unsupported(
                Qwen38V100SupportReason.INVALID_MODE,
                f"unknown executor mode {mode_value!r}",
            )
        if mode is Qwen38V100ExecutorMode.OFF:
            return cls._unsupported(Qwen38V100SupportReason.MODE_OFF, "mode is off")
        if device.type != "cuda" or not is_sm70:
            return cls._unsupported(
                Qwen38V100SupportReason.DEVICE, "requires CUDA capability 7.0"
            )
        if checkpoint_revision != cls.EXPECTED_CHECKPOINT_REVISION:
            return cls._unsupported(
                Qwen38V100SupportReason.CHECKPOINT,
                "deployment did not attest the qualified checkpoint revision",
            )

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        text_config = model_config.hf_text_config
        architectures = tuple(getattr(hf_config, "architectures", ()) or ())
        if cls.EXPECTED_ARCHITECTURES.isdisjoint(architectures):
            return cls._unsupported(
                Qwen38V100SupportReason.ARCHITECTURE,
                f"unsupported architectures {architectures!r}",
            )

        expected_geometry: dict[str, Any] = {
            "model_type": cls.EXPECTED_MODEL_TYPE,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "mtp_num_hidden_layers": 1,
            "vocab_size": 248320,
        }
        mismatches = {
            name: getattr(text_config, name, None)
            for name, expected in expected_geometry.items()
            if getattr(text_config, name, None) != expected
        }
        layer_types = tuple(getattr(text_config, "layer_types", ()) or ())
        if layer_types != cls.EXPECTED_LAYER_TYPES:
            mismatches["layer_types"] = layer_types
        if mismatches:
            return cls._unsupported(
                Qwen38V100SupportReason.GEOMETRY,
                f"checkpoint geometry mismatch: {mismatches!r}",
            )
        if model_config.quantization != "exl3":
            return cls._unsupported(
                Qwen38V100SupportReason.QUANTIZATION,
                f"requires exl3, got {model_config.quantization!r}",
            )
        quant_config = getattr(hf_config, "quantization_config", {}) or {}
        expected_quant_config: dict[str, Any] = {
            "quant_method": "exl3",
            "version": "1.4.2",
            "codebook": "mcg",
            "bits": 4.0,
            "head_bits": 6,
            "mtp_bits": 4,
            "out_scales": "always",
        }
        quant_mismatches = {
            name: quant_config.get(name)
            for name, expected in expected_quant_config.items()
            if quant_config.get(name) != expected
        }
        if quant_mismatches:
            return cls._unsupported(
                Qwen38V100SupportReason.QUANTIZATION,
                f"EXL3 layout mismatch: {quant_mismatches!r}",
            )
        dtype_name = str(model_config.dtype).removeprefix("torch.")
        if dtype_name not in {"float16", "half"}:
            return cls._unsupported(
                Qwen38V100SupportReason.DTYPE,
                f"requires float16 execution, got {dtype_name!r}",
            )

        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.tensor_parallel_size != 4
            or parallel_config.pipeline_parallel_size != 1
            or not nvlink_island_verified
        ):
            return cls._unsupported(
                Qwen38V100SupportReason.TOPOLOGY,
                "requires attested single-island TP4 with PP1",
            )
        spec_config = vllm_config.speculative_config
        if (
            spec_config is None
            or spec_config.method != "mtp"
            or spec_config.num_speculative_tokens != 3
        ):
            return cls._unsupported(
                Qwen38V100SupportReason.SPECULATION,
                "requires MTP with exactly three speculative tokens",
            )
        return Qwen38V100SupportDecision(
            True,
            Qwen38V100SupportReason.ELIGIBLE,
            "exact Qwen3.8 EXL3 SM70 TP4 MTP3 configuration",
        )

    @staticmethod
    def _unsupported(
        reason: Qwen38V100SupportReason, detail: str
    ) -> Qwen38V100SupportDecision:
        return Qwen38V100SupportDecision(False, reason, detail)

    def initialize_persistent_buffers(self, device: torch.device) -> None:
        if not self.support.eligible or self.buffers is not None:
            return
        self.buffers = Qwen38V100PersistentBuffers.allocate(device)

    def can_fuse_q4_input_ids(
        self,
        *,
        prev_index: int,
        input_ids: Any,
        prev_sampled_token_ids: torch.Tensor | None,
        draft_token_ids: torch.Tensor | None,
    ) -> bool:
        """Check the device-resident exact-q4 input-token contract."""
        return bool(
            self.support.eligible
            and prev_index >= 0
            and prev_sampled_token_ids is not None
            and draft_token_ids is not None
            and input_ids.gpu.dtype is torch.int32
            and input_ids.gpu.numel() >= 4
            and input_ids.gpu.is_contiguous()
            and prev_sampled_token_ids.dtype is torch.int32
            and prev_sampled_token_ids.ndim == 2
            and prev_sampled_token_ids.is_contiguous()
            and prev_index < prev_sampled_token_ids.shape[0]
            and draft_token_ids.dtype in {torch.int32, torch.int64}
            and draft_token_ids.is_contiguous()
            and draft_token_ids.numel() >= (prev_index + 1) * 3
        )

    def commit_q4_prepare_packet(
        self,
        *,
        block_tables: Any,
        query_start_loc: Any,
        discard_request_mask: Any,
        num_accepted_tokens: Any,
        state_slot_selectors: Any,
        prev_positions: Any,
        prev_num_draft_tokens: Any,
        num_computed_tokens_cpu: torch.Tensor,
        num_computed_tokens_gpu: torch.Tensor,
        req_indices: Any,
        query_pos: Any,
        num_scheduled_tokens: Any,
        input_ids: Any,
        prev_sampled_token_ids: torch.Tensor | None,
        draft_token_ids: torch.Tensor | None,
        grouped_state_ids: Any | None = None,
        grouped_gdn_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        self.last_q4_gdn_metadata_fused = False
        prev_index = int(prev_positions.np[0])
        if (
            not self.support.eligible
            or len(block_tables.block_tables) != 4
            or not hasattr(torch.ops.qwen38_control, "scatter_prepare_inputs")
            or not self.can_fuse_q4_input_ids(
                prev_index=prev_index,
                input_ids=input_ids,
                prev_sampled_token_ids=prev_sampled_token_ids,
                draft_token_ids=draft_token_ids,
            )
        ):
            return None
        assert prev_sampled_token_ids is not None
        assert draft_token_ids is not None
        widths = tuple(
            int(table.block_table.cpu.shape[1]) for table in block_tables.block_tables
        )
        if widths != (44, 44, 44, 41):
            return None
        metadata_keys = (
            "spec_state_indices_tensor",
            "spec_sequence_masks",
            "spec_token_indx",
            "spec_query_start_loc",
            "num_accepted_tokens",
            "spec_state_slot_selectors",
        )
        fuse_gdn_metadata = bool(
            grouped_state_ids is not None
            and grouped_gdn_metadata is not None
            and all(key in grouped_gdn_metadata for key in metadata_keys)
            and hasattr(
                torch.ops.qwen38_control,
                "scatter_prepare_inputs_and_gdn_metadata",
            )
            and grouped_state_ids.cpu.dtype is torch.int32
            and grouped_state_ids.cpu.ndim == 3
            and grouped_state_ids.cpu.shape[0] >= 3
            and grouped_state_ids.cpu.shape[1] >= 1
            and grouped_state_ids.cpu.shape[2] >= 4
        )
        packet_words = 226 if fuse_gdn_metadata else 214
        if not self._prepare_packet_hosts:
            self._prepare_packet_hosts = [
                torch.empty(
                    packet_words,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(2)
            ]
            self._prepare_packet_events = [None, None]
            self._prepare_packet_device = torch.empty(
                packet_words,
                dtype=torch.int64,
                device=num_computed_tokens_gpu.device,
            )
            self._prepare_packet_words = packet_words
            # Keep the CPU-authoritative fallback values separate from the
            # prior GPU state. Async speculative drift correction gathers the
            # old GPU values before choosing between corrected and CPU values.
            self._prepare_num_computed_cpu_values = torch.empty(
                4, dtype=torch.int32, device=num_computed_tokens_gpu.device
            )
        elif self._prepare_packet_words != packet_words:
            # The specialized ABI is fixed for the lifetime of the engine.
            # If initialization/runtime metadata disagree, fail closed rather
            # than reallocate buffers whose addresses may already be captured.
            return None
        slot = self._prepare_packet_index
        event = self._prepare_packet_events[slot]
        if event is not None and not event.query():
            event.synchronize()
        host = self._prepare_packet_hosts[slot]
        packet = host.numpy()
        offset = 0
        for width, table in zip(widths, block_tables.block_tables):
            packet[offset : offset + width] = table.block_table.np[0]
            offset += width
        sources = (
            (query_start_loc.np[:5], 5),
            (discard_request_mask.np[:4], 4),
            (num_accepted_tokens.np[:4], 4),
            (state_slot_selectors.np[:4], 4),
            (prev_positions.np[:4], 4),
            (prev_num_draft_tokens.np[:4], 4),
            (num_computed_tokens_cpu.numpy()[:4], 4),
            (req_indices.np[:4], 4),
            (query_pos.np[:4], 4),
            (num_scheduled_tokens.np[:4], 4),
        )
        for source, length in sources:
            packet[offset : offset + length] = np.asarray(source)
            offset += length
        assert offset == 214
        if fuse_gdn_metadata:
            packet[offset : offset + 12] = grouped_state_ids.np[:3, 0, :4].reshape(
                -1
            )
            offset += 12
        assert offset == packet_words
        device = self._prepare_packet_device
        assert device is not None
        cpu_num_computed_values = self._prepare_num_computed_cpu_values
        assert cpu_num_computed_values is not None
        device.copy_(host, non_blocking=True)
        destinations = [table.block_table.gpu[0] for table in block_tables.block_tables]
        prepare_args = (
            device,
            *destinations,
            query_start_loc.gpu,
            discard_request_mask.gpu,
            num_accepted_tokens.gpu,
            state_slot_selectors.gpu,
            prev_positions.gpu,
            prev_num_draft_tokens.gpu,
            cpu_num_computed_values,
            req_indices.gpu,
            query_pos.gpu,
            num_scheduled_tokens.gpu,
            input_ids.gpu,
            prev_sampled_token_ids,
            prev_sampled_token_ids.stride(0),
            draft_token_ids,
        )
        if fuse_gdn_metadata:
            assert grouped_gdn_metadata is not None
            torch.ops.qwen38_control.scatter_prepare_inputs_and_gdn_metadata(
                *prepare_args,
                grouped_state_ids.gpu,
                *(grouped_gdn_metadata[key] for key in metadata_keys),
            )
            self.last_q4_gdn_metadata_fused = True
        else:
            torch.ops.qwen38_control.scatter_prepare_inputs(*prepare_args)
        if event is None:
            event = torch.cuda.Event(blocking=False, interprocess=False)
            self._prepare_packet_events[slot] = event
        event.record(torch.cuda.current_stream(device.device))
        self._prepare_packet_index = (slot + 1) % len(self._prepare_packet_hosts)
        return cpu_num_computed_values

    def prepare_q4_text_model_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Bind the exact q=4 text-only model inputs without generic dispatch.

        This deliberately performs no copies and launches no kernels. The
        returned views are identical to the generic runner's text-only path;
        the runner owns the stricter model/step predicate and falls back before
        calling this method for every other shape.
        """
        buffers = self.buffers
        if (
            not self.support.eligible
            or buffers is None
            or input_ids.dtype is not torch.int32
            or positions.dtype is not torch.int64
            or input_ids.device != buffers.logits_indices.device
            or positions.device != input_ids.device
            or input_ids.ndim != 1
            or positions.ndim != 1
            or input_ids.numel() < buffers.VERIFY_WIDTH
            or positions.numel() < buffers.VERIFY_WIDTH
            or input_ids.stride(0) != 1
            or positions.stride(0) != 1
        ):
            return None

        self.fast_preprocess_steps += 1
        width = buffers.VERIFY_WIDTH
        return input_ids[:width], positions[:width]

    def make_q4_spec_decode_metadata(
        self,
        *,
        num_draft_tokens: Any,
        cu_num_scheduled_tokens: Any,
        input_ids: torch.Tensor,
    ) -> SpecDecodeMetadata | None:
        """Build fixed batch-one MTP3 metadata without CPU/NumPy index work."""
        if (
            self.mode not in {Qwen38V100ExecutorMode.ON, Qwen38V100ExecutorMode.SHADOW}
            or (
                self.mode is Qwen38V100ExecutorMode.SHADOW and not self.metadata_compare
            )
            or not self.support.eligible
            or self.buffers is None
            or tuple(num_draft_tokens.shape) != (1,)
            or tuple(cu_num_scheduled_tokens.shape) != (1,)
            or int(num_draft_tokens[0]) != self.buffers.DRAFT_WIDTH
            or int(cu_num_scheduled_tokens[0]) != self.buffers.VERIFY_WIDTH
            or input_ids.numel() < self.buffers.VERIFY_WIDTH
            or input_ids.dtype is not torch.int32
            or input_ids.device != self.buffers.logits_indices.device
            or input_ids.stride(0) != 1
        ):
            return None

        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

        self.static_metadata_steps += 1
        self.buffers.draft_token_ids.copy_(
            input_ids[1 : self.buffers.VERIFY_WIDTH], non_blocking=True
        )
        return SpecDecodeMetadata(
            draft_token_ids=self.buffers.draft_token_ids,
            num_draft_tokens=[self.buffers.DRAFT_WIDTH],
            cu_num_draft_tokens=self.buffers.cu_num_draft_tokens,
            cu_num_sampled_tokens=self.buffers.cu_num_sampled_tokens,
            target_logits_indices=self.buffers.target_logits_indices,
            bonus_logits_indices=self.buffers.bonus_logits_indices,
            logits_indices=self.buffers.logits_indices,
        )

    @property
    def static_metadata_enabled(self) -> bool:
        return self.mode is Qwen38V100ExecutorMode.ON

    @staticmethod
    def _metadata_mismatches(
        candidate: SpecDecodeMetadata,
        reference: SpecDecodeMetadata,
    ) -> tuple[str, ...]:
        mismatches = []
        if candidate.num_draft_tokens != reference.num_draft_tokens:
            mismatches.append("num_draft_tokens")
        for name in (
            "draft_token_ids",
            "cu_num_draft_tokens",
            "cu_num_sampled_tokens",
            "target_logits_indices",
            "bonus_logits_indices",
            "logits_indices",
        ):
            candidate_tensor = getattr(candidate, name)
            reference_tensor = getattr(reference, name)
            if (
                candidate_tensor.dtype != reference_tensor.dtype
                or candidate_tensor.device != reference_tensor.device
                or candidate_tensor.shape != reference_tensor.shape
                or candidate_tensor.stride() != reference_tensor.stride()
                or not torch.equal(candidate_tensor, reference_tensor)
            ):
                mismatches.append(name)
        return tuple(mismatches)

    def compare_and_store_shadow_metadata(
        self,
        candidate: SpecDecodeMetadata,
        reference: SpecDecodeMetadata,
    ) -> tuple[str, ...]:
        mismatches = self._metadata_mismatches(candidate, reference)
        self._shadow_metadata = candidate
        return mismatches

    def compare_late_shadow_metadata(
        self,
        reference: SpecDecodeMetadata,
    ) -> tuple[str, ...] | None:
        candidate = self._shadow_metadata
        self._shadow_metadata = None
        if candidate is None:
            return None
        return self._metadata_mismatches(candidate, reference)

    def observe_verify_q4(
        self,
        *,
        num_reqs: int,
        num_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> Qwen38V100StepDecision:
        self.observed_steps += 1
        if not self.support.eligible:
            return Qwen38V100StepDecision(
                False,
                False,
                Qwen38V100StepReason.EXECUTOR_DISABLED,
                self.support.detail,
            )
        if self.buffers is None:
            return Qwen38V100StepDecision(
                False,
                False,
                Qwen38V100StepReason.BUFFERS_UNINITIALIZED,
                "persistent ABI buffers have not been allocated",
            )
        if num_reqs != 1:
            return Qwen38V100StepDecision(
                False,
                False,
                Qwen38V100StepReason.BATCH_SIZE,
                f"fixed executor requires batch size 1, got {num_reqs}",
            )
        if num_tokens != self.buffers.VERIFY_WIDTH:
            return Qwen38V100StepDecision(
                False,
                False,
                Qwen38V100StepReason.VERIFY_WIDTH,
                f"fixed executor requires q=4, got q={num_tokens}",
            )
        if (
            spec_decode_metadata is None
            or spec_decode_metadata.max_spec_len != self.buffers.DRAFT_WIDTH
            or spec_decode_metadata.num_draft_tokens != [self.buffers.DRAFT_WIDTH]
            or spec_decode_metadata.draft_token_ids.numel() != self.buffers.DRAFT_WIDTH
        ):
            return Qwen38V100StepDecision(
                False,
                False,
                Qwen38V100StepReason.SPEC_METADATA,
                "requires one request with exactly three draft tokens",
            )
        self.eligible_steps += 1
        if self.mode is Qwen38V100ExecutorMode.ON:
            return Qwen38V100StepDecision(
                True,
                False,
                Qwen38V100StepReason.EXECUTION_NOT_IMPLEMENTED,
                "executor math is not installed; using generic verifier",
            )
        return Qwen38V100StepDecision(
            True,
            False,
            Qwen38V100StepReason.ELIGIBLE,
            "shadow-observed fixed q=4 verification step",
        )
