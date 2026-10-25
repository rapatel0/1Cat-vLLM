# SPDX-License-Identifier: Apache-2.0
"""CPU checks that do not import the CUDA model stack."""

import importlib.util
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[3]
_GGUF = _ROOT / "vllm/models/flint/gguf_nvfp4.py"
_CONFIG = _ROOT / "vllm/models/flint/configs.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_flint_geometry():
    configs = _load(_CONFIG, "flint_configs_under_test")
    full = [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48]
    layer_types = []
    for idx in range(49):
        layer_types.append("full_attention" if idx in full else "sliding_attention")
    config = configs.FlintConfig(
        layer_types=layer_types,
        no_rope_layer_indices=full,
        mlp_only_layers=[0, 1],
    )
    assert config.hidden_size == 2048
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 4
    assert config.head_dim == 128
    assert config.num_experts == 256
    assert config.num_experts_per_tok == 6
    assert config.moe_intermediate_size == 512
    assert config.shared_expert_intermediate_size == 1024
    assert config.sliding_window == 513
    assert config.router_score_function == "sigmoid"
    assert config.norm_topk_prob is True
    assert layer_types.count("sliding_attention") == 35
    assert layer_types.count("full_attention") == 14


def test_nvfp4_repack_round_trip_one_block():
    gguf = _load(_GGUF, "flint_gguf_under_test")
    blocks = torch.zeros(36, dtype=torch.uint8)
    blocks[0] = 0x38
    blocks[4] = 0x21
    weight, scale = gguf._repack_nvfp4(blocks, 64)
    assert tuple(weight.shape) == (32,)
    assert int(weight[0]) == 0x12
    values = gguf.dequantize_nvfp4(weight, scale)
    assert values.shape[-1] == 64
    assert torch.isfinite(values).all()
    name, stacked = gguf._map_name("blk.2.ffn_gate_exps")
    assert stacked
    assert name == "model.layers.2.mlp.experts.{}.gate_proj"
    name, stacked = gguf._map_name("blk.0.attn_sinks.weight")
    assert not stacked
    assert name.endswith("softmax_offset")
