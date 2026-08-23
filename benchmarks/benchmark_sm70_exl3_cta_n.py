#!/usr/bin/env python3
"""Compare exact TurboMind-derived EXL3 N128 and N256 CTA kernels."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class Shape:
    name: str
    source: str
    k: int
    n: int
    bits: int
    frequency: int


SHAPES = (
    Shape("gdn_q", "model.language_model.layers.0.mlp.gate_proj", 5120, 512, 5, 48),
    Shape("gdn_v", "model.language_model.layers.0.mlp.gate_proj", 5120, 1536, 5, 48),
    Shape("gdn_out", "model.language_model.layers.0.linear_attn.out_proj", 1536, 5120, 5, 48),
    Shape("self_q", "model.language_model.layers.0.mlp.gate_proj", 5120, 3072, 5, 16),
    Shape("self_k", "model.language_model.layers.0.mlp.gate_proj", 5120, 256, 5, 16),
    Shape("mlp_gate", "model.language_model.layers.0.mlp.gate_proj", 5120, 4352, 5, 64),
    Shape("mlp_down", "model.language_model.layers.0.mlp.down_proj", 4352, 5120, 6, 64),
)


M1_POLICIES = {
    (5, 5120, 1536): (6, 2),
    (5, 1536, 5120): (5, 0),
    (5, 5120, 4352): (7, 0),
    (5, 5120, 3072): (6, 2),
    (6, 4352, 5120): (4, 0),
}

M4_POLICIES = {
    (5, 5120, 512): (11, 0),
    (5, 5120, 1536): (7, 0),
    (5, 1536, 5120): (5, 0),
    (5, 5120, 3072): (6, 0),
    (5, 5120, 256): (11, 0),
    (5, 5120, 4352): (11, 1),
    (6, 4352, 5120): (6, 2),
}


def policy(shape: Shape, rows: int) -> tuple[int, int]:
    table = M1_POLICIES if rows == 1 else M4_POLICIES
    return table.get((shape.bits, shape.k, shape.n), (8, 0))


def load_tensor(root: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def load_trellis(root: Path, index: dict[str, str], shape: Shape) -> torch.Tensor:
    trellis = load_tensor(root, index, f"{shape.source}.trellis")
    if trellis.shape[0] * 16 == shape.k:
        trellis = trellis[:, : shape.n // 16, :]
    else:
        trellis = trellis[: shape.k // 16, : shape.n // 16, :]
    expected = (shape.k // 16, shape.n // 16, shape.bits * 16)
    if tuple(trellis.shape) != expected:
        raise ValueError(f"{shape.name}: got {tuple(trellis.shape)}, expected {expected}")
    return trellis.contiguous().cuda()


def call(
    name: str,
    out: torch.Tensor,
    x_had: torch.Tensor,
    state: torch.Tensor,
    partials: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    selected: tuple[int, int],
) -> None:
    getattr(torch.ops._C, name)(
        out, x_had, state, partials, locks, bits, selected[0], selected[1]
    )


def timed_us(
    name: str,
    out: torch.Tensor,
    x_had: torch.Tensor,
    state: torch.Tensor,
    partials: torch.Tensor,
    locks: torch.Tensor,
    bits: int,
    selected: tuple[int, int],
    iterations: int,
    repeats: int,
) -> float:
    for _ in range(10):
        call(name, out, x_had, state, partials, locks, bits, selected)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            call(name, out, x_had, state, partials, locks, bits, selected)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", default="1,4")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.ops.load_library(str(args.library))
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)["weight_map"]

    records: list[dict[str, object]] = []
    for rows in (int(item) for item in args.rows.split(",")):
        for shape in SHAPES:
            trellis = load_trellis(args.checkpoint, index, shape)
            state = torch.ops._C.exl3_sm70_tm_state_repack(trellis)
            generator = torch.Generator(device="cuda").manual_seed(
                380000 + rows * 10000 + shape.k + shape.n
            )
            x_had = torch.randn(
                (rows, shape.k), generator=generator, device="cuda", dtype=torch.float16
            ).mul_(0.125)
            out128 = torch.empty((rows, shape.n), device="cuda", dtype=torch.float16)
            out256 = torch.empty_like(out128)
            outk32 = torch.empty_like(out128)
            partials = torch.empty((8, shape.n), device="cuda", dtype=torch.float32)
            locks = torch.zeros(shape.n // 128, device="cuda", dtype=torch.int32)
            selected = policy(shape, rows)
            call(
                "exl3_sm70_tm_state_core_out",
                out128,
                x_had,
                state,
                partials,
                locks,
                shape.bits,
                selected,
            )
            baseline = out128.clone()
            call(
                "exl3_sm70_tm_state_core_out_n256",
                out256,
                x_had,
                state,
                partials,
                locks,
                shape.bits,
                selected,
            )
            candidate = out256.clone()
            call(
                "exl3_sm70_tm_state_core_out_k32",
                outk32,
                x_had,
                state,
                partials,
                locks,
                shape.bits,
                selected,
            )
            candidate_k32 = outk32.clone()
            torch.cuda.synchronize()
            mismatch = int(
                torch.count_nonzero(
                    baseline.view(torch.int16) != candidate.view(torch.int16)
                ).item()
            )
            max_abs = float((baseline.float() - candidate.float()).abs().max().item())
            mismatch_k32 = int(
                torch.count_nonzero(
                    baseline.view(torch.int16) != candidate_k32.view(torch.int16)
                ).item()
            )
            max_abs_k32 = float(
                (baseline.float() - candidate_k32.float()).abs().max().item()
            )
            # Bracket the candidate with the baseline and use the slower
            # baseline bracket, which biases against declaring an N256 win.
            baseline_before = timed_us(
                "exl3_sm70_tm_state_core_out",
                out128,
                x_had,
                state,
                partials,
                locks,
                shape.bits,
                selected,
                args.iterations,
                args.repeats,
            )
            candidate_us = timed_us(
                "exl3_sm70_tm_state_core_out_n256",
                out256,
                x_had,
                state,
                partials,
                locks,
                shape.bits,
                selected,
                args.iterations,
                args.repeats,
            )
            candidate_k32_us = timed_us(
                "exl3_sm70_tm_state_core_out_k32",
                outk32,
                x_had,
                state,
                partials,
                locks,
                shape.bits,
                selected,
                args.iterations,
                args.repeats,
            )
            baseline_after = timed_us(
                "exl3_sm70_tm_state_core_out",
                out128,
                x_had,
                state,
                partials,
                locks,
                shape.bits,
                selected,
                args.iterations,
                args.repeats,
            )
            baseline_us = max(baseline_before, baseline_after)
            record = {
                "shape": shape.name,
                "rows": rows,
                "bits": shape.bits,
                "k": shape.k,
                "n": shape.n,
                "frequency": shape.frequency,
                "policy": list(selected),
                "mismatch": mismatch,
                "max_abs": max_abs,
                "mismatch_k32": mismatch_k32,
                "max_abs_k32": max_abs_k32,
                "n128_us": baseline_us,
                "n256_us": candidate_us,
                "speedup": baseline_us / candidate_us,
                "k32_us": candidate_k32_us,
                "speedup_k32": baseline_us / candidate_k32_us,
            }
            records.append(record)
            print(
                f"{shape.name:10s} M={rows} p={selected} exact={mismatch == 0} "
                f"N128={baseline_us:8.3f}us N256={candidate_us:8.3f}us "
                f"speedup={baseline_us / candidate_us:.3f}x "
                f"K32={candidate_k32_us:8.3f}us "
                f"exact={mismatch_k32 == 0} speedup={baseline_us / candidate_k32_us:.3f}x"
            )
            del trellis, state, x_had, out128, out256, outk32, partials, locks
            torch.cuda.empty_cache()

    if any(
        int(record["mismatch"]) or int(record["mismatch_k32"])
        for record in records
    ):
        raise RuntimeError("CTA geometry exactness gate failed")
    weighted_128 = sum(
        float(record["n128_us"]) * int(record["frequency"]) for record in records
    )
    weighted_256 = sum(
        float(record["n256_us"]) * int(record["frequency"]) for record in records
    )
    weighted_k32 = sum(
        float(record["k32_us"]) * int(record["frequency"]) for record in records
    )
    result = {
        "library": str(args.library),
        "checkpoint": str(args.checkpoint),
        "weighted_n128_ms": weighted_128 / 1000.0,
        "weighted_n256_ms": weighted_256 / 1000.0,
        "weighted_speedup": weighted_128 / weighted_256,
        "weighted_k32_ms": weighted_k32 / 1000.0,
        "weighted_k32_speedup": weighted_128 / weighted_k32,
        "records": records,
    }
    print(
        f"WEIGHTED N128={weighted_128 / 1000.0:.4f}ms "
        f"N256={weighted_256 / 1000.0:.4f}ms "
        f"speedup={weighted_128 / weighted_256:.4f}x "
        f"K32={weighted_k32 / 1000.0:.4f}ms "
        f"speedup={weighted_128 / weighted_k32:.4f}x"
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
