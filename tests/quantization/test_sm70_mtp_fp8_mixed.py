# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed NVFP4/FP8 MTP dispatch, scale layout, and serialization tests."""

from __future__ import annotations

from unittest.mock import Mock, patch

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.fp8_sm70_moe import (
    Sm70OnlineBlockFp8MoEMethod,
    Sm70SerializedBlockFp8MoEMethod,
    block_fp8_quantize,
)
from vllm.model_executor.layers.quantization.mixed_module_format import (
    FORMAT_FP8_BLOCK128,
    lookup_mixed_format,
    validate_runtime_amax_experts,
    validate_serialized_fp8_experts,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
)

_FLASH_NEXT_HF_QUANT = {
    "producer": {"name": "modelopt", "version": "0.46.0"},
    "quantization": {
        "exclude_modules": [
            "model.embed_tokens",
            "mtp.*",
            "model.mtp.*",
            "*.self_attn.*",
            "*.linear_attn.*",
            "*.mlp.gate*",
            "*.mlp.shared_expert.*",
            "*.mlp.shared_expert_gate*",
            "*hyper_connection*",
            "*.ple.*",
            "model.visual.*",
            "model.language_model.embed_tokens",
            "lm_head",
        ],
        "group_size": 16,
        "quant_algo": "NVFP4",
    },
}
_ROUTED_EXPERT_PREFIX = "model.language_model.layers.0.mlp.experts"


def _tp4_qwen38_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=512,
        experts_per_token=10,
        hidden_dim=2560,
        intermediate_size_per_partition=160,
        num_local_experts=512,
        num_logical_experts=512,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=4,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        in_dtype=torch.float16,
    )

_MTP_PREFIX = "model.layers.0.mlp.experts"
_MIXED = {
    "producer": {"name": "modelopt", "version": "0.46.0"},
    "quantization": {
        **_FLASH_NEXT_HF_QUANT["quantization"],
        "mixed_modules": {
            "mtp.layers.*.mlp.experts": "fp8_block128",
            "model.layers.*.mlp.experts": "fp8_block128",
        },
        "fp8_serialized": True,
        "fp8_scale_convention": "amax_div_448",
        "fp8_block_size": [128, 128],
    },
}


def _layer() -> Mock:
    layer = Mock(spec=RoutedExperts)
    layer.__class__ = RoutedExperts
    layer.moe_config = _tp4_qwen38_moe_config()
    return layer


def _sm70_patches():
    return (
        patch(
            "vllm.model_executor.layers.quantization.modelopt.sm70_tm."
            "is_exact_sm70_cuda_platform",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.quantization.modelopt.sm70_tm."
            "should_use_nvfp4_moe_turbomind",
            return_value=True,
        ),
    )


def test_lookup_mixed_format_unique_and_ambiguous():
    mixed = {
        "model.layers.*.mlp.experts": FORMAT_FP8_BLOCK128,
        "model.language_model.layers.*.mlp.experts": "nvfp4",
    }
    assert lookup_mixed_format(mixed, _MTP_PREFIX) == FORMAT_FP8_BLOCK128
    assert (
        lookup_mixed_format(mixed, "model.language_model.layers.0.mlp.experts")
        == "nvfp4"
    )
    try:
        lookup_mixed_format(
            {
                "model.layers.*": "fp8_block128",
                "model.layers.0.mlp.experts": "nvfp4",
            },
            _MTP_PREFIX,
        )
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("expected ambiguous mixed_modules error")


def test_scale_broadcast_and_partial_blocks():
    torch.manual_seed(0)
    full = torch.randn(1, 640, 2560, dtype=torch.bfloat16)
    q, s = block_fp8_quantize(full)
    assert s.shape == (1, 5, 20)
    recon = q.float() * s.repeat_interleave(128, 1)[:, :640].repeat_interleave(
        128, 2
    )[:, :, :2560]
    assert (recon - full.float()).norm() / full.float().norm() < 0.05

    partial = torch.randn(1, 160, 2560, dtype=torch.bfloat16)
    pq, ps = block_fp8_quantize(partial)
    assert pq.shape == (1, 160, 2560)
    assert ps.shape == (1, 2, 20)


def test_wrong_orientation_scale_shape():
    weight = torch.randn(1, 640, 2560, dtype=torch.bfloat16)
    _, s = block_fp8_quantize(weight)
    _, s_t = block_fp8_quantize(weight.transpose(1, 2).contiguous())
    assert s.shape == (1, 5, 20)
    assert s_t.shape == (1, 20, 5)
    assert s.shape != s_t.shape


def test_serialization_roundtrip(tmp_path):
    torch.manual_seed(1)
    weight = torch.randn(2, 128, 256, dtype=torch.bfloat16)
    q, s = block_fp8_quantize(weight)
    path = tmp_path / "mtp.pt"
    torch.save({"w": q, "s": s}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["w"].dtype == torch.float8_e4m3fn
    assert loaded["s"].dtype == torch.bfloat16
    assert torch.equal(loaded["w"].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(loaded["s"], s)
    validate_serialized_fp8_experts(loaded["w"], loaded["w"])


def test_validate_rejects_nvfp4_and_bf16_as_serialized():
    bf = torch.zeros(1, 64, 64, dtype=torch.bfloat16)
    try:
        validate_serialized_fp8_experts(bf, bf)
    except ValueError as exc:
        assert "float8" in str(exc)
    else:
        raise AssertionError("expected float8 dtype error")
    u8 = torch.zeros(1, 64, 32, dtype=torch.uint8)
    try:
        validate_serialized_fp8_experts(u8, u8)
    except ValueError as exc:
        assert "float8" in str(exc)
    else:
        raise AssertionError("expected float8 dtype error")
    fp8 = torch.zeros(1, 64, 64, dtype=torch.float8_e4m3fn)
    validate_runtime_amax_experts(bf, bf)
    try:
        validate_runtime_amax_experts(fp8, fp8)
    except ValueError as exc:
        assert "BF16" in str(exc)
    else:
        raise AssertionError("expected BF16 dtype error")


def test_mixed_checkpoint_dispatches_serialized_fp8_not_nvfp4():
    config = ModelOptNvFp4Config.from_config(_MIXED)
    assert config.fp8_serialized is True
    assert config.mixed_modules["model.layers.*.mlp.experts"] == "fp8_block128"
    p1, p2 = _sm70_patches()
    with p1, p2, patch(
        "vllm.model_executor.layers.quantization.modelopt._is_qwen4_mtp_draft",
        return_value=True,
    ):
        mtp = config.get_quant_method(_layer(), _MTP_PREFIX)
        main = config.get_quant_method(_layer(), _ROUTED_EXPERT_PREFIX)
    assert isinstance(mtp, Sm70SerializedBlockFp8MoEMethod)
    assert isinstance(main, ModelOptNvFp4SM70MoEMethod)


def test_missing_mixed_mtp_format_fails_closed():
    cfg = {
        "producer": {"name": "modelopt", "version": "0.46.0"},
        "quantization": {
            **_FLASH_NEXT_HF_QUANT["quantization"],
            "mixed_modules": {
                "never.matches.*": "fp8_block128",
            },
            "fp8_serialized": True,
        },
    }
    config = ModelOptNvFp4Config.from_config(cfg)
    p1, p2 = _sm70_patches()
    with p1, p2, patch(
        "vllm.model_executor.layers.quantization.modelopt._is_qwen4_mtp_draft",
        return_value=True,
    ):
        try:
            config.get_quant_method(_layer(), _MTP_PREFIX)
        except ValueError as exc:
            assert "Fail closed" in str(exc)
        else:
            raise AssertionError("expected fail-closed MTP metadata error")


def test_arch_fallback_runtime_amax_when_no_mixed_modules():
    config = ModelOptNvFp4Config.from_config(_FLASH_NEXT_HF_QUANT)
    p1, p2 = _sm70_patches()
    with p1, p2, patch(
        "vllm.model_executor.layers.quantization.modelopt._is_qwen4_mtp_draft",
        return_value=True,
    ):
        method = config.get_quant_method(_layer(), _MTP_PREFIX)
    assert isinstance(method, Sm70OnlineBlockFp8MoEMethod)


def test_unsupported_scale_convention_rejected():
    cfg = {
        "producer": {"name": "modelopt", "version": "0.46.0"},
        "quantization": {
            **_FLASH_NEXT_HF_QUANT["quantization"],
            "fp8_scale_convention": "mse",
        },
    }
    try:
        ModelOptNvFp4Config.from_config(cfg)
    except ValueError as exc:
        assert "amax_div_448" in str(exc)
    else:
        raise AssertionError("expected scale-convention error")
