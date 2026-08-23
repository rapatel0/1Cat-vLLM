#!/usr/bin/env python3
"""Sweep split-K/swizzle policies for real Qwen3.8 TP4 EXL3 shapes.

This deliberately evaluates policies that may change FP32 accumulation order.
It reports their numerical distance from the currently selected policy instead
of treating bitwise inequality as an automatic failure.  Any policy promoted
from this benchmark still requires the production-sampling quality gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


TP_SIZE = 4
GATE = "model.language_model.layers.0.mlp.gate_proj"
GDN_OUT = "model.language_model.layers.0.linear_attn.out_proj"
DOWN = "model.language_model.layers.0.mlp.down_proj"
LM_HEAD = "lm_head"


@dataclass(frozen=True)
class Shape:
    name: str
    source: str
    k: int
    n: int
    bits: int
    fused_output: bool
    frequency: int


SHAPES = (
    # V22 executes each equal-shape pair in one grouped scheduler grid.  The
    # single-matrix operator below is sufficient to rank split/swizzle policy;
    # the live gate must retime the winning policy in the grouped op.
    Shape("gdn_qk_group", GATE, 5120, 512, 5, True, 48),
    Shape("gdn_vz_group", GATE, 5120, 1536, 5, True, 48),
    Shape("gdn_out", GDN_OUT, 1536, 5120, 5, True, 48),
    Shape("self_q", GATE, 5120, 3072, 5, False, 16),
    Shape("self_kv_group", GATE, 5120, 256, 5, True, 16),
    Shape("self_out", GDN_OUT, 1536, 5120, 5, True, 16),
    Shape("mlp_gate_up_group", GATE, 5120, 4352, 5, True, 64),
    Shape("mlp_down", DOWN, 4352, 5120, 6, True, 64),
    Shape("lm_head", LM_HEAD, 5120, 62080, 6, False, 1),
)

M4_CANDIDATE_POLICIES = {
    (5, 5120, 512): (11, 0),
    (5, 5120, 1536): (7, 0),
    (5, 1536, 5120): (5, 0),
    (5, 5120, 3072): (6, 0),
    (5, 5120, 256): (11, 0),
    (5, 5120, 4352): (11, 1),
    (6, 4352, 5120): (6, 2),
    (6, 5120, 62080): (3, 2),
}


def default_policy(bits: int, k: int, n: int, rows: int) -> tuple[int, int]:
    """Mirror sm70_exl3_tm_state_policy for reproducible comparisons."""
    if rows == 1:
        policies = {
            (5, 5120, 2560): (8, 2),
            (5, 5120, 1536): (6, 2),
            (5, 1536, 5120): (5, 0),
            (5, 5120, 4352): (7, 0),
            (5, 5120, 3072): (6, 2),
            (5, 5120, 1024): (9, 3),
            (6, 4352, 5120): (4, 0),
            (6, 5120, 62080): (3, 0),
        }
        if (bits, k, n) in policies:
            return policies[bits, k, n]
    if (bits, k, n) == (5, 5120, 2560):
        return 11, 0
    if (bits, k, n) == (6, 5120, 62080):
        return 1, 0
    policies = {
        (5, 5120, 1536): (15, 0),
        (5, 5120, 3072): (11, 0),
        (5, 5120, 4352): (11, 1),
        (5, 5120, 1024): (9, 0),
        (6, 4352, 5120): (9, 2),
    }
    splits, swizzle = policies.get((bits, k, n), (8, 0))
    return min(splits, max(1, k // 128)), swizzle


def load_tensor(root: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def load_projection(
    root: Path, index: dict[str, str], shape: Shape
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trellis = load_tensor(root, index, f"{shape.source}.trellis")
    suh = load_tensor(root, index, f"{shape.source}.suh")
    svh = load_tensor(root, index, f"{shape.source}.svh")

    # The representative K5 gate matrix supplies all K=5120 column-parallel
    # timings.  Weight values do not affect the kernel schedule; cropping N
    # yields the exact production geometry without loading every layer.
    if trellis.shape[0] * 16 == shape.k:
        trellis = trellis[:, : shape.n // 16, :]
        svh = svh[: shape.n]
    else:
        # Row-parallel projections take rank zero's K shard.  The LM head is
        # column parallel and therefore enters the branch above.
        trellis = trellis[: shape.k // 16, : shape.n // 16, :]
        suh = suh[: shape.k]
        svh = svh[: shape.n]
    if trellis.shape != (shape.k // 16, shape.n // 16, shape.bits * 16):
        raise ValueError(
            f"{shape.name}: got trellis {tuple(trellis.shape)}, expected "
            f"{(shape.k // 16, shape.n // 16, shape.bits * 16)}"
        )
    return trellis.contiguous().cuda(), suh.contiguous().cuda(), svh.contiguous().cuda()


class ProjectionRunner:
    def __init__(
        self,
        shape: Shape,
        rows: int,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
    ) -> None:
        self.shape = shape
        self.rows = rows
        self.trellis = trellis
        self.state = torch.ops._C.exl3_sm70_tm_state_repack(trellis)
        generator = torch.Generator(device="cuda").manual_seed(8107 + shape.n + rows)
        self.x = torch.randn(
            (rows, shape.k), generator=generator, device="cuda", dtype=torch.float16
        ).mul_(0.125)
        self.suh = suh
        self.svh = svh
        self.out = torch.empty((rows, shape.n), device="cuda", dtype=torch.float16)
        self.x_had = torch.empty_like(self.x)
        self.accum = torch.empty((rows, shape.n), device="cuda", dtype=torch.float32)
        self.partials = torch.empty((8, shape.n), device="cuda", dtype=torch.float32)
        self.locks = torch.zeros(
            ((rows + 7) // 8) * (shape.n // 128),
            device="cuda",
            dtype=torch.int32,
        )

    def run(self, splits: int, swizzle: int) -> torch.Tensor:
        if self.shape.fused_output:
            torch.ops._C.exl3_sm70_tm_state_gemm_hadamard_out(
                self.out,
                self.x,
                self.state,
                self.suh,
                self.svh,
                self.x_had,
                self.partials,
                self.locks,
                self.shape.bits,
                splits,
                swizzle,
            )
        else:
            torch.ops._C.exl3_sm70_tm_state_gemm_out(
                self.out,
                self.x,
                self.state,
                self.suh,
                self.svh,
                self.x_had,
                self.accum,
                self.partials,
                self.locks,
                self.shape.bits,
                splits,
                swizzle,
            )
        return self.out

    def run_dispatch(self) -> torch.Tensor:
        return torch.ops._C.exl3_sm70_tm_dispatch_gemm_persistent_locks(
            self.x,
            self.trellis,
            self.state,
            self.suh,
            self.svh,
            self.locks,
            self.shape.bits,
            True,
            False,
        )


def timed_us(
    runner: ProjectionRunner,
    policy: tuple[int, int],
    warmup: int,
    repeats: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        runner.run(*policy)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            runner.run(*policy)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def distance(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    denom = torch.linalg.vector_norm(ref)
    cosine = torch.nn.functional.cosine_similarity(ref.flatten(), cand.flatten(), dim=0)
    return {
        "mismatch": int(torch.count_nonzero(reference.view(torch.int16) != candidate.view(torch.int16)).item()),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(delta) / denom).item()) if denom else 0.0,
        "cosine": float(cosine.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", default="1")
    parser.add_argument("--max-splits", type=int, default=16)
    parser.add_argument("--quick-iterations", type=int, default=20)
    parser.add_argument("--final-iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--graph-replays", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("this benchmark requires an SM70 GPU")
    torch.ops.load_library(str(args.library))
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)["weight_map"]

    all_results: list[dict[str, object]] = []
    for rows in (int(value) for value in args.rows.split(",")):
        for shape in SHAPES:
            trellis, suh, svh = load_projection(args.checkpoint, index, shape)
            runner = ProjectionRunner(shape, rows, trellis, suh, svh)
            baseline_policy = default_policy(shape.bits, shape.k, shape.n, rows)
            baseline = runner.run(*baseline_policy).clone()
            torch.cuda.synchronize()
            # Bring clocks and caches to a stable state before the quick
            # screen.  The authoritative baseline is retimed alongside the
            # finalists below; using this first number would bias the result
            # toward later, warmer candidates.
            timed_us(
                runner, baseline_policy, 5, args.repeats, args.final_iterations
            )

            quick: list[tuple[float, tuple[int, int]]] = []
            max_splits = min(args.max_splits, shape.k // 128)
            for splits in range(1, max_splits + 1):
                for swizzle in range(6):
                    policy = (splits, swizzle)
                    elapsed = timed_us(runner, policy, 2, 2, args.quick_iterations)
                    quick.append((elapsed, policy))
            finalists = [policy for _, policy in sorted(quick)[: args.top]]
            if baseline_policy not in finalists:
                finalists.append(baseline_policy)

            candidates: list[dict[str, object]] = []
            for policy in finalists:
                elapsed = timed_us(
                    runner, policy, 5, args.repeats, args.final_iterations
                )
                candidate = runner.run(*policy).clone()
                torch.cuda.synchronize()
                record: dict[str, object] = {
                    "splits": policy[0],
                    "swizzle": policy[1],
                    "time_us": elapsed,
                    **distance(baseline, candidate),
                }
                candidates.append(record)
            candidates.sort(key=lambda item: float(item["time_us"]))
            baseline_us = next(
                float(item["time_us"])
                for item in candidates
                if (int(item["splits"]), int(item["swizzle"]))
                == baseline_policy
            )
            for item in candidates:
                item["speedup"] = baseline_us / float(item["time_us"])
            dispatch_validation: dict[str, object] | None = None
            if rows == 4 and args.graph_replays:
                expected_policy = M4_CANDIDATE_POLICIES[
                    (shape.bits, shape.k, shape.n)
                ]
                expected = runner.run(*expected_policy).clone()
                dispatched = runner.run_dispatch().clone()
                torch.cuda.synchronize()
                initial_distance = distance(expected, dispatched)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = runner.run_dispatch()
                for _ in range(args.graph_replays):
                    graph.replay()
                torch.cuda.synchronize()
                replayed = captured.clone()
                torch.cuda.synchronize()
                dispatch_validation = {
                    "expected_policy": list(expected_policy),
                    "initial": initial_distance,
                    "after_replay": distance(expected, replayed),
                    "graph_replays": args.graph_replays,
                    "nonzero_locks": int(torch.count_nonzero(runner.locks).item()),
                }
                if initial_distance["mismatch"] or dispatch_validation[
                    "after_replay"
                ]["mismatch"] or dispatch_validation["nonzero_locks"]:
                    raise RuntimeError(
                        f"{shape.name}: dispatch/graph validation failed: "
                        f"{dispatch_validation}"
                    )
            result: dict[str, object] = {
                "shape": shape.name,
                "rows": rows,
                "k": shape.k,
                "n": shape.n,
                "bits": shape.bits,
                "frequency": shape.frequency,
                "fused_output": shape.fused_output,
                "baseline_policy": list(baseline_policy),
                "baseline_us": baseline_us,
                "candidates": candidates,
                "dispatch_validation": dispatch_validation,
            }
            all_results.append(result)
            winner = candidates[0]
            print(
                f"{shape.name:12s} M={rows} base={baseline_policy} "
                f"{baseline_us:8.3f} us best=({winner['splits']},"
                f"{winner['swizzle']}) {float(winner['time_us']):8.3f} us "
                f"{float(winner['speedup']):.3f}x max_abs="
                f"{float(winner['max_abs']):.6g} rel_l2="
                f"{float(winner['relative_l2']):.3g}"
            )
            del runner, trellis, suh, svh
            torch.cuda.empty_cache()

    weighted_baseline_us = sum(
        float(item["baseline_us"]) * int(item["frequency"])
        for item in all_results
    )
    weighted_best_us = sum(
        float(item["candidates"][0]["time_us"]) * int(item["frequency"])
        for item in all_results
    )
    summary = {
        "library": str(args.library),
        "checkpoint": str(args.checkpoint),
        "weighted_baseline_ms": weighted_baseline_us / 1000.0,
        "weighted_best_ms": weighted_best_us / 1000.0,
        "weighted_speedup": weighted_baseline_us / weighted_best_us,
        "results": all_results,
    }
    print(
        f"WEIGHTED baseline={weighted_baseline_us / 1000.0:.4f} ms "
        f"best={weighted_best_us / 1000.0:.4f} ms "
        f"speedup={weighted_baseline_us / weighted_best_us:.4f}x"
    )
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
