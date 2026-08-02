# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from tests.quantization.reference_mxfp4 import dq_mxfp4_torch
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    mxfp4_marlin_process_scales,
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

MODEL_ENV = "VLLM_DEEPSEEK_V4_MODEL"


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires CUDA SM70",
)
def test_mxfp4_sm70_preserves_logical_scale_order() -> None:
    raw_scales = torch.tensor(
        [[119, 120, 121, 122, 122, 121, 120, 119]],
        dtype=torch.uint8,
        device="cuda",
    )
    logical_scales = raw_scales.view(torch.float8_e8m0fnu).to(torch.float16)

    processed = mxfp4_marlin_process_scales(logical_scales)

    torch.testing.assert_close(processed.view(torch.uint8), raw_scales)


def _load_tensor(model: Path, name: str) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shard = model / index["weight_map"][name]
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(name)


def _official_expert(model: Path, layer: int, expert: int):
    prefix = f"layers.{layer}.ffn.experts.{expert}"

    def get(proj: str, kind: str) -> torch.Tensor:
        tensor = _load_tensor(model, f"{prefix}.{proj}.{kind}")
        return tensor.view(torch.uint8)

    w1 = get("w1", "weight")
    w3 = get("w3", "weight")
    w2 = get("w2", "weight")
    s1 = get("w1", "scale")
    s3 = get("w3", "scale")
    s2 = get("w2", "scale")
    return (
        torch.cat((w1, w3)).unsqueeze(0).cuda(),
        w2.unsqueeze(0).cuda(),
        torch.cat((s1, s3)).unsqueeze(0).cuda(),
        s2.unsqueeze(0).cuda(),
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(70),
    reason="requires CUDA SM70",
)
@pytest.mark.parametrize("num_tokens", [1, 16])
def test_official_deepseek_v4_mxfp4_expert_matches_reference(
    num_tokens: int,
) -> None:
    model_value = os.getenv(MODEL_ENV)
    if model_value is None:
        pytest.skip(f"set {MODEL_ENV} to an official DeepSeek V4 checkpoint")
    assert model_value is not None
    model = Path(model_value)

    w13, w2, s13, s2 = _official_expert(model, layer=0, expert=0)
    ref_w13 = dq_mxfp4_torch(w13[0], s13[0], torch.float16)
    ref_w2 = dq_mxfp4_torch(w2[0], s2[0], torch.float16)

    layer = SimpleNamespace(params_dtype=torch.float16)
    marlin_w13, marlin_w2, marlin_s13, marlin_s2, _, _ = (
        prepare_moe_mxfp4_layer_for_marlin(
            layer,
            w13,
            w2,
            s13,
            s2,
            None,
            None,
        )
    )

    torch.manual_seed(7)
    hidden = torch.randn(
        num_tokens,
        ref_w13.shape[1],
        device="cuda",
        dtype=torch.float16,
    )
    topk_ids = torch.zeros((num_tokens, 1), device="cuda", dtype=torch.int32)
    topk_weights = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)

    actual = fused_marlin_moe(
        hidden_states=hidden,
        w1=marlin_w13,
        w2=marlin_w2,
        bias1=None,
        bias2=None,
        w1_scale=marlin_s13,
        w2_scale=marlin_s2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=scalar_types.float4_e2m1f.id,
        global_num_experts=1,
        clamp_limit=10.0,
    )

    expected_w13 = F.linear(hidden, ref_w13)
    gate, up = expected_w13.chunk(2, dim=-1)
    activated = F.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
        up, min=-10.0, max=10.0
    )
    expected = F.linear(activated, ref_w2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)
