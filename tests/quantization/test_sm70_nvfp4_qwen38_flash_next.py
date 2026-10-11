# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 route gates for Qwen3.8 Flash Next ModelOpt NVFP4 checkpoints.

Pins the dealignai/RadixArk Flash Next NVFP4 recipe: static W4A4 metadata,
group size 16, routed experts only. On V100 the experts run weight-only
(FP16 activations, packed E2M1 tiles, TurboMind HMMA).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutedExperts,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
    validate_nvfp4_sm70_moe_contract,
)

# Frozen dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4
# revision be794b990578ef3031eccf9f28e675a289a09ee9. Byte-identical to
# RadixArk/Qwen3.8-Flash-Next-NVFP4 hf_quant_config.json.
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

_EXCLUDED_PREFIXES = (
    "model.language_model.layers.0.self_attn.qkv_proj",
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.mlp.gate",
    "model.language_model.layers.0.mlp.shared_expert.gate_up_proj",
    "model.language_model.layers.0.mlp.shared_expert_gate",
    "model.language_model.layers.0.hyper_connection.stream_mix",
    "model.language_model.layers.2.ple.embedding",
    "model.visual.patch_embed.proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "lm_head",
)

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


def _flash_next_config() -> ModelOptNvFp4Config:
    config = ModelOptNvFp4Config.from_config(_FLASH_NEXT_HF_QUANT)
    assert isinstance(config, ModelOptNvFp4Config)
    return config


def test_flash_next_hf_quant_is_static_nvfp4_group16():
    config = _flash_next_config()
    assert config.quant_method == "NVFP4"
    assert config.group_size == 16
    assert config.is_checkpoint_nvfp4_serialized
    assert config.kv_cache_quant_algo is None


def test_flash_next_ignore_list_keeps_non_expert_layers_unquantized():
    config = _flash_next_config()
    for prefix in _EXCLUDED_PREFIXES:
        assert config.is_layer_excluded(prefix), prefix
        assert config.get_quant_method(Mock(), prefix) is None


def test_qwen38_tp4_contract_is_validated():
    validate_nvfp4_sm70_moe_contract(_tp4_qwen38_moe_config())


def test_sm70_selects_weight_only_nvfp4_moe():
    config = _flash_next_config()
    layer = Mock(spec=RoutedExperts)
    layer.__class__ = RoutedExperts
    layer.moe_config = _tp4_qwen38_moe_config()
    with (
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
    ):
        method = config.get_quant_method(layer, _ROUTED_EXPERT_PREFIX)
    assert isinstance(method, ModelOptNvFp4SM70MoEMethod)
    assert method.use_a16 is True


def test_sm70_nvfp4_min_capability_is_70_with_turbomind():
    with patch(
        "vllm.model_executor.layers.quantization.modelopt.sm70_tm.use_turbomind",
        return_value=True,
    ):
        assert ModelOptNvFp4Config.get_min_capability() == 70


def test_block_fp8_matches_official_mtp_scale_layout():
    from vllm.model_executor.layers.quantization.fp8_sm70_moe import (
        block_fp8_quantize,
    )

    torch.manual_seed(0)
    weight = torch.randn(2, 640, 2560, dtype=torch.float16)
    quant, scale = block_fp8_quantize(weight)
    assert quant.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.bfloat16
    assert quant.shape == (2, 640, 2560)
    assert scale.shape == (2, 5, 20)
    recon = (
        quant.float()
        * scale.repeat_interleave(128, dim=1)[:, :640]
        .repeat_interleave(128, dim=2)[:, :, :2560]
    )
    rel = (recon - weight.float()).norm() / weight.float().norm()
    assert rel < 0.05


def test_smoke_script_uses_native_sm70_path_not_emulation():
    source = Path("tools/qwen38_flash_next_nvfp4_smoke.py").read_text()
    assert 'moe_backend="emulation"' not in source
    assert 'os.environ.pop("VLLM_USE_V2_MODEL_RUNNER"' in source
    assert "VLLM_SM70_NVFP4_TURBOMIND" in source
    assert "VLLM_PLE_CPU_OFFLOAD" in source
    assert "language_model_only=True" in source
