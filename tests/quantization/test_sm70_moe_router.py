# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute,
    select_sm70_quantized_moe_route,
)


def test_grouped_route_when_per_expert_dispatch_is_disabled():
    route = select_sm70_quantized_moe_route(
        batched_enabled=True,
        num_tokens=4,
        total_slots=40,
        w13_per_expert_dispatch=False,
        w2_per_expert_dispatch=False,
    )

    assert route.use_batched_moe_gemm
    assert route.w13 == Sm70MoeStageRoute.BATCHED
    assert route.w2 == Sm70MoeStageRoute.BATCHED
