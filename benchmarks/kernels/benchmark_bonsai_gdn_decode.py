# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep SM70 mixed-QKV GDN decode schedules for Bonsai's live shape."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as gdn
from vllm.platforms import current_platform

# Extracted from Ternary-Bonsai-27B-Q2_0.gguf on 2026-07-17:
# ssm.inner_size=6144, ssm.group_count=16, state_size=128. With V=128 this
# yields 48 value heads; each four-slot decode invokes this kernel per GDN
# layer (48 layers total).
TOKENS = 4
QK_HEADS = 16
VALUE_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128


@dataclass(frozen=True)
class Schedule:
    bv: int
    warps: int
    stages: int

    @property
    def label(self) -> str:
        return f"bv{self.bv}-w{self.warps}-s{self.stages}"


def _parse_schedules(value: str) -> list[Schedule]:
    schedules: list[Schedule] = []
    for item in value.split(","):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid schedule {item!r}; expected BV:WARPS:STAGES")
        schedule = Schedule(*(int(part) for part in parts))
        if min(schedule.bv, schedule.warps, schedule.stages) <= 0:
            raise ValueError(f"schedule values must be positive: {item!r}")
        schedules.append(schedule)
    if not schedules:
        raise ValueError("at least one schedule is required")
    return schedules


def _set_schedule(schedule: Schedule) -> None:
    # The launch configuration is passed as Triton constexpr metadata. Updating
    # these module-level test controls exercises the same public launch path as
    # the environment overrides without restarting a process per candidate.
    gdn._SM70_FUSED_SIGMOID_BV_OVERRIDE = schedule.bv
    gdn._SM70_FUSED_SIGMOID_WARPS_OVERRIDE = schedule.warps
    gdn._SM70_FUSED_SIGMOID_STAGES_OVERRIDE = schedule.stages
    gdn._SM70_FUSED_SIGMOID_SCHEDULE = True


def _time_ms(fn: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _make_inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qkv_width = 2 * QK_HEADS * HEAD_K_DIM + VALUE_HEADS * HEAD_V_DIM
    return {
        "A_log": torch.zeros(VALUE_HEADS, device="cuda", dtype=torch.float16),
        "a": torch.full(
            (TOKENS, VALUE_HEADS), -3.0, device="cuda", dtype=torch.float16
        ),
        "b": torch.zeros((TOKENS, VALUE_HEADS), device="cuda", dtype=torch.float16),
        "dt_bias": torch.zeros(
            VALUE_HEADS, device="cuda", dtype=torch.float16
        ),
        "mixed_qkv": torch.randn(
            (TOKENS, qkv_width),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.125),
        "state": torch.randn(
            (TOKENS, VALUE_HEADS, HEAD_V_DIM, HEAD_K_DIM),
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        ).mul_(0.125),
        "state_indices": torch.arange(TOKENS, device="cuda", dtype=torch.int32),
        "cu_seqlens": torch.arange(TOKENS + 1, device="cuda", dtype=torch.int32),
    }


def _run(
    inputs: dict[str, torch.Tensor], state: torch.Tensor, out: torch.Tensor
) -> None:
    gdn.fused_sigmoid_gating_delta_rule_update_mixed_qkv_out(
        A_log=inputs["A_log"],
        a=inputs["a"],
        b=inputs["b"],
        dt_bias=inputs["dt_bias"],
        mixed_qkv=inputs["mixed_qkv"],
        num_q_heads=QK_HEADS,
        num_v_heads=VALUE_HEADS,
        head_k_dim=HEAD_K_DIM,
        head_v_dim=HEAD_V_DIM,
        scale=HEAD_K_DIM**-0.5,
        initial_state=state,
        out=out,
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["state_indices"],
        use_qk_l2norm_in_kernel=True,
        gqa_tiled=True,
    )


def _fp32_reference(
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one mixed-QKV update in FP32 for schedule-independent error."""
    mixed_qkv = inputs["mixed_qkv"].float()
    state = inputs["state"].float().clone()
    q_width = QK_HEADS * HEAD_K_DIM
    v_width = VALUE_HEADS * HEAD_V_DIM
    q, k, v = torch.split(mixed_qkv, (q_width, q_width, v_width), dim=-1)
    q = q.view(TOKENS, QK_HEADS, HEAD_K_DIM)
    k = k.view(TOKENS, QK_HEADS, HEAD_K_DIM)
    v = v.view(TOKENS, VALUE_HEADS, HEAD_V_DIM)
    q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    q = q * HEAD_K_DIM**-0.5
    g = -inputs["A_log"].float().exp() * torch.nn.functional.softplus(
        inputs["a"].float() + inputs["dt_bias"].float()
    )
    beta = inputs["b"].float().sigmoid()
    out = torch.empty(
        (TOKENS, 1, VALUE_HEADS, HEAD_V_DIM), device="cuda", dtype=torch.float32
    )
    for token in range(TOKENS):
        state_index = int(inputs["state_indices"][token])
        h = state[state_index]
        # Qwen3.5 maps value head h to q/k head h % QK_HEADS, rather than
        # contiguous grouped-GQA expansion.  Keep the benchmark's scalar
        # reference aligned with the GGUF fused GDN implementation.
        q_token = q[token].repeat((VALUE_HEADS // QK_HEADS, 1))
        k_token = k[token].repeat((VALUE_HEADS // QK_HEADS, 1))
        h.mul_(g[token, :, None, None].exp())
        value = v[token] - (h * k_token[:, None, :]).sum(dim=-1)
        value.mul_(beta[token, :, None])
        h.add_(value[:, :, None] * k_token[:, None, :])
        out[token, 0].copy_((h * q_token[:, None, :]).sum(dim=-1))
    return out.half(), state.half()


def _measure_schedule(
    schedule: Schedule,
    inputs: dict[str, torch.Tensor],
    warmup: int,
    iterations: int,
    trials: int,
    reference_out: torch.Tensor,
    reference_state: torch.Tensor,
) -> dict[str, Any]:
    _set_schedule(schedule)
    state_template = inputs["state"]
    state = state_template.clone()
    out = torch.empty(
        (TOKENS, 1, VALUE_HEADS, HEAD_V_DIM),
        device="cuda",
        dtype=torch.float16,
    )

    def run() -> None:
        _run(inputs, state, out)

    # The candidate is compared from an identical state before timing mutates it.
    state.copy_(state_template)
    run()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(state).all()):
        raise RuntimeError(f"non-finite output for {schedule.label}")
    output_error = (out.float() - reference_out.float()).abs()
    state_error = (state.float() - reference_state.float()).abs()

    samples: list[float] = []
    for _ in range(trials):
        state.copy_(state_template)
        samples.append(_time_ms(run, warmup, iterations))
    median = statistics.median(samples)
    record = {
        "schedule": schedule.label,
        "bv": schedule.bv,
        "warps": schedule.warps,
        "stages": schedule.stages,
        "median_ms": median,
        "samples_ms": samples,
        "four_slot_48_layer_ms": median * 48,
        "max_abs_output_error_vs_fp32": float(output_error.max().item()),
        "mean_abs_output_error_vs_fp32": float(output_error.mean().item()),
        "max_abs_state_error_vs_fp32": float(state_error.max().item()),
        "mean_abs_state_error_vs_fp32": float(state_error.mean().item()),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedules",
        default=(
            "16:1:1,16:1:3,16:2:1,16:2:3,16:4:1,16:4:3,16:8:1,16:8:3,"
            "32:1:1,32:1:3,32:2:1,32:2:3,32:4:1,32:4:3,32:8:1,32:8:3,"
            "64:1:1,64:1:3,64:2:1,64:2:3,64:4:1,64:4:3,64:8:1,64:8:3"
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not current_platform.is_device_capability(70):
        raise RuntimeError("this benchmark requires an exact SM70 CUDA device")
    if args.warmup < 0 or args.iterations <= 0 or args.trials <= 0:
        raise ValueError("warmup, iterations, and trials must be valid")

    inputs = _make_inputs(args.seed)
    results: list[dict[str, Any]] = []
    reference_out, reference_state = _fp32_reference(inputs)
    for schedule in _parse_schedules(args.schedules):
        result = _measure_schedule(
            schedule,
            inputs,
            args.warmup,
            args.iterations,
            args.trials,
            reference_out,
            reference_state,
        )
        results.append(result)

    baseline = next(
        result for result in results if result["schedule"] == "bv32-w4-s3"
    )
    for result in results:
        result["speedup_vs_baseline"] = baseline["median_ms"] / result["median_ms"]
        result["output_error_ratio_vs_baseline"] = (
            result["mean_abs_output_error_vs_fp32"]
            / baseline["mean_abs_output_error_vs_fp32"]
        )
        result["state_error_ratio_vs_baseline"] = (
            result["mean_abs_state_error_vs_fp32"]
            / baseline["mean_abs_state_error_vs_fp32"]
        )
    payload = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {
            "tokens": TOKENS,
            "qk_heads": QK_HEADS,
            "value_heads": VALUE_HEADS,
            "head_k_dim": HEAD_K_DIM,
            "head_v_dim": HEAD_V_DIM,
        },
        "baseline": baseline["schedule"],
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
