# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.model_executor.models.qwen3_5 import Qwen3_5EXL3TextForCausalLM
from vllm.v1.worker.qwen38_v100_executor import (
    Qwen38V100Executor,
    Qwen38V100ExecutorMode,
    Qwen38V100StepReason,
    Qwen38V100SupportReason,
)


def make_config() -> SimpleNamespace:
    text_config = SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        mtp_num_hidden_layers=1,
        vocab_size=248320,
        layer_types=list(Qwen38V100Executor.EXPECTED_LAYER_TYPES),
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[Qwen38V100Executor.EXPECTED_ARCHITECTURE],
            quantization_config={
                "quant_method": "exl3",
                "version": "1.4.2",
                "codebook": "mcg",
                "bits": 4.0,
                "head_bits": 6,
                "mtp_bits": 4,
                "out_scales": "always",
            },
        ),
        hf_text_config=text_config,
        quantization="exl3",
        dtype=torch.float16,
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
        ),
    )


def make_executor(
    config: SimpleNamespace | None = None,
    *,
    mode: str = "shadow",
    device: str = "cuda",
    is_sm70: bool = True,
    nvlink_verified: bool = True,
    metadata_compare: bool = False,
) -> Qwen38V100Executor:
    return Qwen38V100Executor.from_vllm_config(
        config or make_config(),
        torch.device(device),
        mode_value=mode,
        is_sm70=is_sm70,
        nvlink_island_verified=nvlink_verified,
        checkpoint_revision=Qwen38V100Executor.EXPECTED_CHECKPOINT_REVISION,
        metadata_compare=metadata_compare,
    )


def test_exact_checkpoint_is_eligible() -> None:
    executor = make_executor()

    assert executor.mode is Qwen38V100ExecutorMode.SHADOW
    assert executor.support.eligible
    assert executor.support.reason is Qwen38V100SupportReason.ELIGIBLE


def test_text_only_architecture_is_eligible() -> None:
    config = make_config()
    config.model_config.hf_config.architectures = [
        "Qwen3_5EXL3TextForCausalLM"
    ]

    executor = make_executor(config)

    assert executor.support.eligible
    assert executor.support.reason is Qwen38V100SupportReason.ELIGIBLE


def test_text_only_checkpoint_weight_mapper_is_exact_and_fail_closed() -> None:
    mapper = Qwen3_5EXL3TextForCausalLM.conditional_to_text_mapper

    assert mapper.apply_list(
        [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "lm_head.weight",
            "model.visual.blocks.0.norm1.weight",
            "mtp.fc.weight",
        ]
    ) == [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "lm_head.weight",
        "mtp.fc.weight",
    ]


def test_text_only_checkpoint_module_prefix_resolves_exl3_storage() -> None:
    module_prefix = Qwen3_5EXL3TextForCausalLM.checkpoint_module_prefix()
    linear_prefix = f"{module_prefix}.model.layers.0.self_attn.q_proj"
    storage_name = "model.language_model.layers.0.self_attn.q_proj"
    storage_entry = {"quant_format": "exl3"}
    quant_config = Exl3Config(tensor_storage={storage_name: storage_entry})

    assert module_prefix == "model.language_model"
    assert quant_config._storage_entry(linear_prefix) is storage_entry


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("off", Qwen38V100SupportReason.MODE_OFF),
        ("invalid", Qwen38V100SupportReason.INVALID_MODE),
    ],
)
def test_mode_gate_is_fail_closed(mode: str, reason: Qwen38V100SupportReason) -> None:
    executor = make_executor(mode=mode)

    assert not executor.support.eligible
    assert executor.support.reason is reason


def test_geometry_gate_rejects_stale_48_layer_assumption() -> None:
    config = make_config()
    config.model_config.hf_text_config.num_hidden_layers = 48

    executor = make_executor(config)

    assert not executor.support.eligible
    assert executor.support.reason is Qwen38V100SupportReason.GEOMETRY
    assert "num_hidden_layers" in executor.support.detail


def test_layer_pattern_is_independent_and_exact() -> None:
    expected = tuple(
        "full_attention" if layer_idx in range(3, 64, 4) else "linear_attention"
        for layer_idx in range(64)
    )
    assert expected == Qwen38V100Executor.EXPECTED_LAYER_TYPES

    config = make_config()
    config.model_config.hf_text_config.layer_types[7] = "linear_attention"
    executor = make_executor(config)

    assert not executor.support.eligible
    assert executor.support.reason is Qwen38V100SupportReason.GEOMETRY


def test_checkpoint_revision_requires_deployment_attestation() -> None:
    config = make_config()
    executor = Qwen38V100Executor.from_vllm_config(
        config,
        torch.device("cuda"),
        mode_value="shadow",
        is_sm70=True,
        nvlink_island_verified=True,
        checkpoint_revision=None,
    )

    assert not executor.support.eligible
    assert executor.support.reason is Qwen38V100SupportReason.CHECKPOINT


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda config: setattr(
                config.model_config.hf_config,
                "architectures",
                ["Qwen3ForCausalLM"],
            ),
            Qwen38V100SupportReason.ARCHITECTURE,
        ),
        (
            lambda config: setattr(config.model_config, "quantization", "fp8"),
            Qwen38V100SupportReason.QUANTIZATION,
        ),
        (
            lambda config: setattr(config.model_config, "dtype", torch.bfloat16),
            Qwen38V100SupportReason.DTYPE,
        ),
        (
            lambda config: setattr(config.parallel_config, "tensor_parallel_size", 8),
            Qwen38V100SupportReason.TOPOLOGY,
        ),
        (
            lambda config: setattr(
                config.speculative_config, "num_speculative_tokens", 1
            ),
            Qwen38V100SupportReason.SPECULATION,
        ),
    ],
)
def test_configuration_mismatches_are_reason_coded(
    mutate: object, reason: Qwen38V100SupportReason
) -> None:
    config = make_config()
    mutate(config)  # type: ignore[operator]

    executor = make_executor(config)

    assert not executor.support.eligible
    assert executor.support.reason is reason


def test_topology_requires_deployment_attestation() -> None:
    executor = make_executor(nvlink_verified=False)

    assert not executor.support.eligible
    assert executor.support.reason is Qwen38V100SupportReason.TOPOLOGY


def test_device_requires_exact_sm70_cuda() -> None:
    executor = make_executor(device="cpu", is_sm70=False)

    assert not executor.support.eligible
    assert executor.support.reason is Qwen38V100SupportReason.DEVICE


@pytest.mark.parametrize(
    ("num_reqs", "num_tokens", "draft_width", "reason"),
    [
        (2, 4, 3, Qwen38V100StepReason.BATCH_SIZE),
        (1, 1, 3, Qwen38V100StepReason.VERIFY_WIDTH),
        (1, 4, 2, Qwen38V100StepReason.SPEC_METADATA),
    ],
)
def test_runtime_step_gate_falls_back_before_execution(
    num_reqs: int,
    num_tokens: int,
    draft_width: int,
    reason: Qwen38V100StepReason,
) -> None:
    executor = make_executor()
    executor.buffers = SimpleNamespace(VERIFY_WIDTH=4, DRAFT_WIDTH=3)
    metadata = SimpleNamespace(
        max_spec_len=draft_width,
        num_draft_tokens=[draft_width],
        draft_token_ids=torch.zeros(draft_width, dtype=torch.int32),
    )

    decision = executor.observe_verify_q4(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        spec_decode_metadata=metadata,
    )

    assert not decision.shape_matched
    assert not decision.should_execute
    assert decision.reason is reason


def test_shadow_observes_q4_without_changing_execution() -> None:
    executor = make_executor()
    executor.buffers = SimpleNamespace(VERIFY_WIDTH=4, DRAFT_WIDTH=3)
    metadata = SimpleNamespace(
        max_spec_len=3,
        num_draft_tokens=[3],
        draft_token_ids=torch.zeros(3, dtype=torch.int32),
    )

    decision = executor.observe_verify_q4(
        num_reqs=1,
        num_tokens=4,
        spec_decode_metadata=metadata,
    )

    assert decision.shape_matched
    assert not decision.should_execute
    assert decision.reason is Qwen38V100StepReason.ELIGIBLE
    assert executor.eligible_steps == 1


def test_on_mode_still_falls_back_until_math_is_installed() -> None:
    executor = make_executor(mode="on")
    executor.buffers = SimpleNamespace(VERIFY_WIDTH=4, DRAFT_WIDTH=3)
    metadata = SimpleNamespace(
        max_spec_len=3,
        num_draft_tokens=[3],
        draft_token_ids=torch.zeros(3, dtype=torch.int32),
    )

    decision = executor.observe_verify_q4(
        num_reqs=1,
        num_tokens=4,
        spec_decode_metadata=metadata,
    )

    assert decision.shape_matched
    assert not decision.should_execute
    assert decision.reason is Qwen38V100StepReason.EXECUTION_NOT_IMPLEMENTED


def test_disabled_and_uninitialized_step_gates() -> None:
    disabled = make_executor(mode="off")
    decision = disabled.observe_verify_q4(
        num_reqs=1,
        num_tokens=4,
        spec_decode_metadata=None,
    )
    assert not decision.shape_matched
    assert decision.reason is Qwen38V100StepReason.EXECUTOR_DISABLED

    uninitialized = make_executor()
    decision = uninitialized.observe_verify_q4(
        num_reqs=1,
        num_tokens=4,
        spec_decode_metadata=None,
    )
    assert not decision.shape_matched
    assert decision.reason is Qwen38V100StepReason.BUFFERS_UNINITIALIZED


def test_buffer_initialization_is_noop_when_ineligible() -> None:
    executor = make_executor(mode="off")

    executor.initialize_persistent_buffers(torch.device("cpu"))

    assert executor.buffers is None


def test_q4_text_model_inputs_are_exact_aliases() -> None:
    executor = make_executor()
    executor.buffers = SimpleNamespace(
        VERIFY_WIDTH=4,
        logits_indices=torch.empty(4, dtype=torch.int32),
    )
    input_ids = torch.tensor([10, 11, 12, 13, 99], dtype=torch.int32)
    positions = torch.tensor([20, 21, 22, 23, 99], dtype=torch.int64)

    prepared = executor.prepare_q4_text_model_inputs(
        input_ids=input_ids,
        positions=positions,
    )

    assert prepared is not None
    prepared_ids, prepared_positions = prepared
    assert torch.equal(prepared_ids, input_ids[:4])
    assert torch.equal(prepared_positions, positions[:4])
    assert prepared_ids.data_ptr() == input_ids.data_ptr()
    assert prepared_positions.data_ptr() == positions.data_ptr()
    assert executor.fast_preprocess_steps == 1


@pytest.mark.parametrize(
    ("input_ids", "positions"),
    [
        (
            torch.tensor([10, 11, 12], dtype=torch.int32),
            torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        ),
        (
            torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        ),
    ],
)
def test_q4_text_model_inputs_fail_closed(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    executor = make_executor()
    executor.buffers = SimpleNamespace(
        VERIFY_WIDTH=4,
        logits_indices=torch.empty(4, dtype=torch.int32),
    )

    assert (
        executor.prepare_q4_text_model_inputs(
            input_ids=input_ids,
            positions=positions,
        )
        is None
    )
    assert executor.fast_preprocess_steps == 0


@pytest.mark.parametrize("draft_dtype", [torch.int32, torch.int64])
def test_q4_input_fusion_accepts_native_mtp_token_dtypes(
    draft_dtype: torch.dtype,
) -> None:
    executor = make_executor()
    input_ids = SimpleNamespace(gpu=torch.empty(8, dtype=torch.int32))
    prev_sampled = torch.tensor([[17], [23]], dtype=torch.int32)
    drafts = torch.tensor([[31, 32, 33], [41, 42, 43]], dtype=draft_dtype)

    assert executor.can_fuse_q4_input_ids(
        prev_index=1,
        input_ids=input_ids,
        prev_sampled_token_ids=prev_sampled,
        draft_token_ids=drafts,
    )


@pytest.mark.parametrize(
    ("prev_index", "prev_dtype", "draft_width"),
    [
        (-1, torch.int32, 3),
        (2, torch.int32, 3),
        (0, torch.int64, 3),
        (0, torch.int32, 2),
    ],
)
def test_q4_input_fusion_contract_fails_closed(
    prev_index: int,
    prev_dtype: torch.dtype,
    draft_width: int,
) -> None:
    executor = make_executor()
    input_ids = SimpleNamespace(gpu=torch.empty(8, dtype=torch.int32))
    prev_sampled = torch.tensor([[17]], dtype=prev_dtype)
    drafts = torch.arange(draft_width, dtype=torch.int64).view(1, -1)

    assert not executor.can_fuse_q4_input_ids(
        prev_index=prev_index,
        input_ids=input_ids,
        prev_sampled_token_ids=prev_sampled,
        draft_token_ids=drafts,
    )


def test_on_mode_builds_exact_q4_metadata_without_gathers() -> None:
    executor = make_executor(mode="on")
    executor.buffers = SimpleNamespace(
        VERIFY_WIDTH=4,
        DRAFT_WIDTH=3,
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([4], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        draft_token_ids=torch.empty(3, dtype=torch.int32),
    )
    input_ids = torch.tensor([10, 11, 12, 13], dtype=torch.int32)

    metadata = executor.make_q4_spec_decode_metadata(
        num_draft_tokens=torch.tensor([3], dtype=torch.int32).numpy(),
        cu_num_scheduled_tokens=torch.tensor([4], dtype=torch.int32).numpy(),
        input_ids=input_ids,
    )

    assert metadata is not None
    assert metadata.num_draft_tokens == [3]
    assert torch.equal(metadata.draft_token_ids, torch.tensor([11, 12, 13]))
    assert torch.equal(metadata.cu_num_draft_tokens, torch.tensor([3]))
    assert torch.equal(metadata.cu_num_sampled_tokens, torch.tensor([4]))
    assert torch.equal(metadata.target_logits_indices, torch.tensor([0, 1, 2]))
    assert torch.equal(metadata.bonus_logits_indices, torch.tensor([3]))
    assert torch.equal(metadata.logits_indices, torch.tensor([0, 1, 2, 3]))
    assert (
        metadata.draft_token_ids.data_ptr()
        == executor.buffers.draft_token_ids.data_ptr()
    )
    assert metadata.draft_token_ids.data_ptr() != input_ids[1:].data_ptr()
    assert executor.static_metadata_steps == 1


def test_static_metadata_path_falls_back_for_wrong_shape() -> None:
    executor = make_executor(mode="on")
    executor.buffers = SimpleNamespace(VERIFY_WIDTH=4, DRAFT_WIDTH=3)
    result = executor.make_q4_spec_decode_metadata(
        num_draft_tokens=torch.tensor([2], dtype=torch.int32).numpy(),
        cu_num_scheduled_tokens=torch.tensor([3], dtype=torch.int32).numpy(),
        input_ids=torch.tensor([10, 11, 12], dtype=torch.int32),
    )
    assert result is None


def test_shadow_metadata_compare_detects_late_buffer_reuse() -> None:
    executor = make_executor(metadata_compare=True)
    executor.buffers = SimpleNamespace(
        VERIFY_WIDTH=4,
        DRAFT_WIDTH=3,
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([4], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        draft_token_ids=torch.empty(3, dtype=torch.int32),
    )
    candidate = executor.make_q4_spec_decode_metadata(
        num_draft_tokens=torch.tensor([3], dtype=torch.int32).numpy(),
        cu_num_scheduled_tokens=torch.tensor([4], dtype=torch.int32).numpy(),
        input_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
    )
    assert candidate is not None
    assert not executor.static_metadata_enabled

    reference = SimpleNamespace(
        draft_token_ids=torch.tensor([11, 12, 13], dtype=torch.int32),
        num_draft_tokens=[3],
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([4], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert executor.compare_and_store_shadow_metadata(candidate, reference) == ()

    executor.buffers.draft_token_ids[0] = 99
    assert executor.compare_late_shadow_metadata(reference) == ("draft_token_ids",)
    assert executor.compare_late_shadow_metadata(reference) is None


def test_shadow_metadata_compare_requires_identical_tensor_contract() -> None:
    executor = make_executor(metadata_compare=True)
    candidate = SimpleNamespace(
        draft_token_ids=torch.tensor([11, 12, 13], dtype=torch.int32),
        num_draft_tokens=[3],
        cu_num_draft_tokens=torch.tensor([3], dtype=torch.int32),
        cu_num_sampled_tokens=torch.tensor([4], dtype=torch.int32),
        target_logits_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([3], dtype=torch.int32),
        logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    reference = SimpleNamespace(**vars(candidate))
    reference.logits_indices = candidate.logits_indices.to(torch.int64)

    assert executor.compare_and_store_shadow_metadata(candidate, reference) == (
        "logits_indices",
    )
