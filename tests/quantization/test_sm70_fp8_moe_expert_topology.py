# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.fp8_sm70_moe import (
    Fp8SM70MoEMethod,
    _single_token_shortcuts_support_expert_topology,
)


@pytest.mark.parametrize(
    ("expert_map", "local_num_experts", "global_num_experts", "expected"),
    [
        (None, 64, 64, True),
        (None, 64, 512, False),
        (torch.arange(64, dtype=torch.int32), 64, 64, False),
        (torch.arange(512, dtype=torch.int32), 64, 512, False),
    ],
)
def test_single_token_shortcuts_require_fully_replicated_experts(
    expert_map: torch.Tensor | None,
    local_num_experts: int,
    global_num_experts: int,
    expected: bool,
) -> None:
    layer = cast(
        RoutedExperts,
        SimpleNamespace(
            expert_map=expert_map,
            local_num_experts=local_num_experts,
            global_num_experts=global_num_experts,
        ),
    )

    assert _single_token_shortcuts_support_expert_topology(layer) is expected


def test_legacy_single_token_shortcut_rejects_expert_parallel_topology() -> None:
    layer = cast(
        RoutedExperts,
        SimpleNamespace(
            expert_map=torch.arange(512, dtype=torch.int32),
            local_num_experts=64,
            global_num_experts=512,
        ),
    )
    method = object.__new__(Fp8SM70MoEMethod)

    with pytest.raises(RuntimeError, match="fully replicated experts"):
        method._apply_legacy_single_token_compact(
            layer,
            x=None,
            topk_weights=None,
            topk_ids_i32=None,
            buffers={},
            top_k=8,
            output=None,
        )
