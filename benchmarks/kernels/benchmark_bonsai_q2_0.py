# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Prism Q2_0 SM70 MMA kernels against the DP4A path."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

Q2_TYPE = 42
QK2_0 = 128
Q2_0_BLOCK_BYTES = 34


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _make_qweight(n: int, k: int, seed: int) -> torch.Tensor:
    if k <= 0 or k % QK2_0:
        raise ValueError(f"K must be a positive multiple of {QK2_0}, got {k}")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    blocks = k // QK2_0
    raw = torch.empty(
        (n, blocks, Q2_0_BLOCK_BYTES),
        dtype=torch.uint8,
        device="cuda",
    )
    raw[..., 2:] = torch.randint(
        0,
        256,
        (n, blocks, Q2_0_BLOCK_BYTES - 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scales = (
        torch.rand(
            (n, blocks),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.125
        - 0.0625
    ).half()
    raw[..., :2] = scales.view(torch.uint8).reshape(n, blocks, 2)
    return raw.reshape(n, blocks * Q2_0_BLOCK_BYTES)


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _measure_shape(
    qweight: torch.Tensor,
    n: int,
    k: int,
    tokens: int,
    warmup: int,
    iterations: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed + tokens)
    x = torch.randn(
        (tokens, k),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )

    def mma() -> torch.Tensor:
        return ops.ggml_mul_mat_q2_0_sm70(qweight, x, n)

    def dp4a() -> torch.Tensor:
        return ops.ggml_mul_mat_vec_a8(qweight, x, Q2_TYPE, n)

    candidate = mma()
    baseline = dp4a()
    torch.cuda.synchronize()
    delta = (candidate.float() - baseline.float()).abs()
    if not bool(torch.isfinite(candidate).all()):
        raise RuntimeError(f"non-finite MMA output for M={tokens}, N={n}, K={k}")

    samples: dict[str, list[float]] = {"mma": [], "dp4a": []}
    functions = {"mma": mma, "dp4a": dp4a}
    for trial in range(trials):
        order = ("mma", "dp4a") if trial % 2 == 0 else ("dp4a", "mma")
        for name in order:
            samples[name].append(_time_ms(functions[name], warmup, iterations))

    mma_median = statistics.median(samples["mma"])
    dp4a_median = statistics.median(samples["dp4a"])
    return {
        "m": tokens,
        "n": n,
        "k": k,
        "warmup": warmup,
        "iterations": iterations,
        "trials": trials,
        "mma_ms": mma_median,
        "mma_p20_ms": _percentile(samples["mma"], 0.2),
        "mma_p80_ms": _percentile(samples["mma"], 0.8),
        "dp4a_ms": dp4a_median,
        "dp4a_p20_ms": _percentile(samples["dp4a"], 0.2),
        "dp4a_p80_ms": _percentile(samples["dp4a"], 0.8),
        "speedup": dp4a_median / mma_median,
        "max_abs_delta_vs_dp4a": float(delta.max().item()),
        "mean_abs_delta_vs_dp4a": float(delta.mean().item()),
        "mma_samples_ms": samples["mma"],
        "dp4a_samples_ms": samples["dp4a"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--tokens", default="4,8,64")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not current_platform.is_device_capability(70):
        raise RuntimeError(
            f"exact SM70 is required, got {torch.cuda.get_device_capability()}"
        )
    if args.n <= 0:
        raise ValueError("N must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.trials <= 0:
        raise ValueError("warmup, iterations, and trials must be valid")

    tokens = _parse_csv_ints(args.tokens)
    if any(value not in (4, 8) and value < 64 for value in tokens):
        raise ValueError("MMA tokens must be 4, 8, or at least 64")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    qweight = _make_qweight(args.n, args.k, args.seed)
    results = [
        _measure_shape(
            qweight,
            args.n,
            args.k,
            value,
            args.warmup,
            args.iterations,
            args.trials,
            args.seed,
        )
        for value in tokens
    ]
    payload = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
