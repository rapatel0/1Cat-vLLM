#!/usr/bin/env python3
"""Qualify the research-only SM70 EXL3 -> tile-scaled INT8 path.

The INT8 representation is produced once from the exact EXL3 trellis.  This
benchmark excludes that startup cost, sweeps the INT8 split-K policy, and
compares complete input-Hadamard/GEMM/output-Hadamard latency and output error
against the current exact-state kernel on real Qwen3.8 TP4 projection shapes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


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
    Shape("gdn_qk", GATE, 5120, 512, 5, True, 48),
    Shape("gdn_vz", GATE, 5120, 1536, 5, True, 48),
    Shape("gdn_out", GDN_OUT, 1536, 5120, 5, True, 48),
    Shape("self_q", GATE, 5120, 3072, 5, False, 16),
    Shape("self_kv", GATE, 5120, 256, 5, True, 16),
    Shape("self_out", GDN_OUT, 1536, 5120, 5, True, 16),
    # Gate and up share one grouped launch in production.  Count both matrix
    # products here because this first INT8 path intentionally runs one matrix
    # per launch; grouped INT8 is a follow-up only if the core wins.
    Shape("mlp_gate_up", GATE, 5120, 4352, 5, True, 128),
    Shape("mlp_down", DOWN, 4352, 5120, 6, True, 64),
    Shape("lm_head", LM_HEAD, 5120, 62080, 6, False, 1),
)


def default_policy(bits: int, k: int, n: int, rows: int) -> tuple[int, int]:
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
    if rows == 4:
        policies = {
            (5, 5120, 512): (11, 0),
            (5, 5120, 1536): (7, 0),
            (5, 1536, 5120): (5, 0),
            (5, 5120, 3072): (6, 0),
            (5, 5120, 256): (11, 0),
            (5, 5120, 4352): (11, 1),
            (6, 4352, 5120): (6, 2),
            (6, 5120, 62080): (3, 2),
        }
        if (bits, k, n) in policies:
            return policies[bits, k, n]
    return min(8, max(1, k // 128)), 0


def load_tensor(root: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def load_projection(
    root: Path, index: dict[str, str], shape: Shape
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trellis = load_tensor(root, index, f"{shape.source}.trellis")
    suh = load_tensor(root, index, f"{shape.source}.suh")
    svh = load_tensor(root, index, f"{shape.source}.svh")
    if trellis.shape[0] * 16 == shape.k:
        trellis = trellis[:, : shape.n // 16, :]
        svh = svh[: shape.n]
    else:
        trellis = trellis[: shape.k // 16, : shape.n // 16, :]
        suh = suh[: shape.k]
        svh = svh[: shape.n]
    expected = (shape.k // 16, shape.n // 16, shape.bits * 16)
    if trellis.shape != expected:
        raise ValueError(f"{shape.name}: trellis={tuple(trellis.shape)}, expected={expected}")
    return (
        trellis.contiguous().cuda(),
        suh.contiguous().cuda(),
        svh.contiguous().cuda(),
    )


class Runner:
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
        self.packed_lane, self.tile_scales = (
            torch.ops._C.exl3_sm70_tm_int8_repack(trellis)
        )
        self.int6_words, self.int6_scales = (
            torch.ops._C.exl3_sm70_tm_int6_repack(trellis)
        )
        self.int10_words, self.int10_scales = (
            torch.ops._C.exl3_sm70_tm_int10_repack(trellis)
        )
        self.e4m3_lane, self.e4m3_scales = (
            torch.ops._C.exl3_sm70_tm_e4m3_repack(trellis)
        )
        generator = torch.Generator(device="cuda").manual_seed(
            11731 + shape.k + shape.n + rows
        )
        self.x = torch.randn(
            (rows, shape.k), generator=generator, device="cuda", dtype=torch.float16
        ).mul_(0.125)
        self.suh = suh
        self.svh = svh
        self.exact_out = torch.empty(
            (rows, shape.n), device="cuda", dtype=torch.float16
        )
        self.int8_out = torch.empty_like(self.exact_out)
        self.e4m3_out = torch.empty_like(self.exact_out)
        self.int6_out = torch.empty_like(self.exact_out)
        self.int10_out = torch.empty_like(self.exact_out)
        self.exact_x_had = torch.empty_like(self.x)
        self.int8_x_had = torch.empty_like(self.x)
        self.e4m3_x_had = torch.empty_like(self.x)
        self.int6_x_had = torch.empty_like(self.x)
        self.int10_x_had = torch.empty_like(self.x)
        self.exact_accum = torch.empty(
            (rows, shape.n), device="cuda", dtype=torch.float32
        )
        self.int8_accum = torch.empty_like(self.exact_accum)
        self.exact_partials = torch.empty(
            (8, shape.n), device="cuda", dtype=torch.float32
        )
        self.int8_partials = torch.empty_like(self.exact_partials)
        self.e4m3_partials = torch.empty_like(self.exact_partials)
        self.int6_partials = torch.empty_like(self.exact_partials)
        self.int10_partials = torch.empty_like(self.exact_partials)
        lock_count = ((rows + 7) // 8) * (shape.n // 128)
        self.exact_locks = torch.zeros(lock_count, device="cuda", dtype=torch.int32)
        self.int8_locks = torch.zeros_like(self.exact_locks)
        self.e4m3_locks = torch.zeros_like(self.exact_locks)
        self.int6_locks = torch.zeros_like(self.exact_locks)
        self.int10_locks = torch.zeros_like(self.exact_locks)

    def exact(self, policy: tuple[int, int], fused: bool) -> torch.Tensor:
        op = (
            torch.ops._C.exl3_sm70_tm_state_gemm_hadamard_out
            if fused
            else torch.ops._C.exl3_sm70_tm_state_gemm_out
        )
        workspace = (
            (
                self.exact_out,
                self.x,
                self.state,
                self.suh,
                self.svh,
                self.exact_x_had,
                self.exact_partials,
                self.exact_locks,
            )
            if fused
            else (
                self.exact_out,
                self.x,
                self.state,
                self.suh,
                self.svh,
                self.exact_x_had,
                self.exact_accum,
                self.exact_partials,
                self.exact_locks,
            )
        )
        op(*workspace, self.shape.bits, *policy)
        return self.exact_out

    def int8(self, policy: tuple[int, int], fused: bool) -> torch.Tensor:
        if fused:
            torch.ops._C.exl3_sm70_tm_int8_gemm_hadamard_out(
                self.int8_out,
                self.x,
                self.packed_lane,
                self.tile_scales,
                self.suh,
                self.svh,
                self.int8_x_had,
                self.int8_partials,
                self.int8_locks,
                self.shape.bits,
                *policy,
            )
        else:
            torch.ops._C.exl3_sm70_tm_int8_gemm_out(
                self.int8_out,
                self.x,
                self.packed_lane,
                self.tile_scales,
                self.suh,
                self.svh,
                self.int8_x_had,
                self.int8_accum,
                self.int8_partials,
                self.int8_locks,
                self.shape.bits,
                *policy,
            )
        return self.int8_out

    def int8_dispatch(self) -> torch.Tensor:
        return torch.ops._C.exl3_sm70_tm_int8_dispatch_gemm_persistent_locks(
            self.x,
            self.trellis,
            self.packed_lane,
            self.tile_scales,
            self.suh,
            self.svh,
            self.int8_locks,
            self.shape.bits,
            True,
            False,
        )

    def e4m3(self, policy: tuple[int, int]) -> torch.Tensor:
        torch.ops._C.exl3_sm70_tm_e4m3_gemm_hadamard_out(
            self.e4m3_out,
            self.x,
            self.e4m3_lane,
            self.e4m3_scales,
            self.suh,
            self.svh,
            self.e4m3_x_had,
            self.e4m3_partials,
            self.e4m3_locks,
            self.shape.bits,
            *policy,
        )
        return self.e4m3_out

    def int6(self, policy: tuple[int, int]) -> torch.Tensor:
        torch.ops._C.exl3_sm70_tm_int6_gemm_hadamard_out(
            self.int6_out,
            self.x,
            self.int6_words,
            self.int6_scales,
            self.suh,
            self.svh,
            self.int6_x_had,
            self.int6_partials,
            self.int6_locks,
            self.shape.bits,
            *policy,
        )
        return self.int6_out

    def int10(self, policy: tuple[int, int]) -> torch.Tensor:
        torch.ops._C.exl3_sm70_tm_int10_gemm_hadamard_out(
            self.int10_out,
            self.x,
            self.int10_words,
            self.int10_scales,
            self.suh,
            self.svh,
            self.int10_x_had,
            self.int10_partials,
            self.int10_locks,
            self.shape.bits,
            *policy,
        )
        return self.int10_out


def timed_us(function, warmup: int, repeats: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def distance(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    norm = torch.linalg.vector_norm(ref)
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float((torch.linalg.vector_norm(delta) / norm).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                ref.flatten(), cand.flatten(), dim=0
            ).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", default="1,4")
    parser.add_argument("--shapes", default="all")
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
    for name in (
        "exl3_sm70_tm_int8_repack",
        "exl3_sm70_tm_int8_gemm_out",
        "exl3_sm70_tm_int8_gemm_hadamard_out",
        "exl3_sm70_tm_int8_dispatch_gemm_persistent_locks",
        "exl3_sm70_tm_e4m3_repack",
        "exl3_sm70_tm_e4m3_gemm_hadamard_out",
        "exl3_sm70_tm_int6_repack",
        "exl3_sm70_tm_int6_gemm_hadamard_out",
        "exl3_sm70_tm_int10_repack",
        "exl3_sm70_tm_int10_gemm_hadamard_out",
    ):
        if not hasattr(torch.ops._C, name):
            raise RuntimeError(f"sidecar does not register _C::{name}")
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)["weight_map"]

    requested = set(args.shapes.split(","))
    shapes = SHAPES if args.shapes == "all" else tuple(
        shape for shape in SHAPES if shape.name in requested
    )
    if not shapes:
        raise ValueError(f"no matching shapes for {args.shapes}")

    results: list[dict[str, object]] = []
    for rows in (int(value) for value in args.rows.split(",")):
        for shape in shapes:
            trellis, suh, svh = load_projection(args.checkpoint, index, shape)
            runner = Runner(shape, rows, trellis, suh, svh)
            exact_policy = default_policy(shape.bits, shape.k, shape.n, rows)
            exact_fused_us = timed_us(
                lambda: runner.exact(exact_policy, shape.fused_output),
                5,
                args.repeats,
                args.final_iterations,
            )
            exact_unfused_us = timed_us(
                lambda: runner.exact(exact_policy, False),
                5,
                args.repeats,
                args.final_iterations,
            )
            reference = runner.exact(exact_policy, shape.fused_output).clone()

            quick: list[tuple[float, tuple[int, int]]] = []
            for splits in range(1, min(args.max_splits, shape.k // 128) + 1):
                for swizzle in range(6):
                    policy = (splits, swizzle)
                    elapsed = timed_us(
                        lambda policy=policy: runner.int8(
                            policy, shape.fused_output
                        ),
                        2,
                        2,
                        args.quick_iterations,
                    )
                    quick.append((elapsed, policy))
            finalists = [policy for _, policy in sorted(quick)[: args.top]]
            candidates: list[dict[str, object]] = []
            for policy in finalists:
                elapsed = timed_us(
                    lambda policy=policy: runner.int8(
                        policy, shape.fused_output
                    ),
                    5,
                    args.repeats,
                    args.final_iterations,
                )
                candidate = runner.int8(policy, shape.fused_output).clone()
                torch.cuda.synchronize()
                candidates.append(
                    {
                        "policy": list(policy),
                        "time_us": elapsed,
                        "speedup_vs_exact": exact_fused_us / elapsed,
                        **distance(reference, candidate),
                    }
                )
            candidates.sort(key=lambda item: float(item["time_us"]))
            best = candidates[0]
            int6_candidates: list[dict[str, object]] = []
            int10_candidates: list[dict[str, object]] = []
            int6_graph_validation: dict[str, object] | None = None
            if shape.fused_output:
                int6_quick: list[tuple[float, tuple[int, int]]] = []
                for splits in range(1, min(args.max_splits, shape.k // 128) + 1):
                    for swizzle in range(6):
                        policy = (splits, swizzle)
                        elapsed = timed_us(
                            lambda policy=policy: runner.int6(policy),
                            2,
                            2,
                            args.quick_iterations,
                        )
                        int6_quick.append((elapsed, policy))
                int6_finalists = [
                    policy for _, policy in sorted(int6_quick)[: args.top]
                ]
                for policy in int6_finalists:
                    elapsed = timed_us(
                        lambda policy=policy: runner.int6(policy),
                        5,
                        args.repeats,
                        args.final_iterations,
                    )
                    candidate = runner.int6(policy).clone()
                    torch.cuda.synchronize()
                    int6_candidates.append(
                        {
                            "policy": list(policy),
                            "time_us": elapsed,
                            "speedup_vs_exact": exact_fused_us / elapsed,
                            "speedup_vs_int8": float(best["time_us"]) / elapsed,
                            **distance(reference, candidate),
                        }
                    )
                int6_candidates.sort(key=lambda item: float(item["time_us"]))
                int10_quick: list[tuple[float, tuple[int, int]]] = []
                for splits in range(1, min(args.max_splits, shape.k // 128) + 1):
                    for swizzle in range(6):
                        policy = (splits, swizzle)
                        elapsed = timed_us(
                            lambda policy=policy: runner.int10(policy),
                            2,
                            2,
                            args.quick_iterations,
                        )
                        int10_quick.append((elapsed, policy))
                int10_finalists = [
                    policy for _, policy in sorted(int10_quick)[: args.top]
                ]
                for policy in int10_finalists:
                    elapsed = timed_us(
                        lambda policy=policy: runner.int10(policy),
                        5,
                        args.repeats,
                        args.final_iterations,
                    )
                    candidate = runner.int10(policy).clone()
                    torch.cuda.synchronize()
                    int10_candidates.append(
                        {
                            "policy": list(policy),
                            "time_us": elapsed,
                            "speedup_vs_exact": exact_fused_us / elapsed,
                            "speedup_vs_int8": float(best["time_us"]) / elapsed,
                            **distance(reference, candidate),
                        }
                    )
                int10_candidates.sort(key=lambda item: float(item["time_us"]))
                if args.graph_replays:
                    int6_best_policy = tuple(int6_candidates[0]["policy"])
                    initial = runner.int6(int6_best_policy).clone()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = runner.int6(int6_best_policy)
                    for _ in range(args.graph_replays):
                        graph.replay()
                    torch.cuda.synchronize()
                    replayed = captured.clone()
                    int6_graph_validation = {
                        "replayed_vs_initial": distance(initial, replayed),
                        "graph_replays": args.graph_replays,
                        "nonzero_locks": int(
                            torch.count_nonzero(runner.int6_locks).item()
                        ),
                    }
                    if (
                        int6_graph_validation["replayed_vs_initial"]["max_abs"]
                        != 0.0
                        or int6_graph_validation["nonzero_locks"] != 0
                    ):
                        raise RuntimeError(
                            f"{shape.name}: INT6 graph replay failed: "
                            f"{int6_graph_validation}"
                        )
            e4m3_candidates: list[dict[str, object]] = []
            e4m3_graph_validation: dict[str, object] | None = None
            if shape.fused_output:
                e4m3_quick: list[tuple[float, tuple[int, int]]] = []
                for splits in range(1, min(args.max_splits, shape.k // 128) + 1):
                    for swizzle in range(6):
                        policy = (splits, swizzle)
                        elapsed = timed_us(
                            lambda policy=policy: runner.e4m3(policy),
                            2,
                            2,
                            args.quick_iterations,
                        )
                        e4m3_quick.append((elapsed, policy))
                e4m3_finalists = [
                    policy for _, policy in sorted(e4m3_quick)[: args.top]
                ]
                for policy in e4m3_finalists:
                    elapsed = timed_us(
                        lambda policy=policy: runner.e4m3(policy),
                        5,
                        args.repeats,
                        args.final_iterations,
                    )
                    candidate = runner.e4m3(policy).clone()
                    torch.cuda.synchronize()
                    e4m3_candidates.append(
                        {
                            "policy": list(policy),
                            "time_us": elapsed,
                            "speedup_vs_exact": exact_fused_us / elapsed,
                            "speedup_vs_int8": float(best["time_us"]) / elapsed,
                            **distance(reference, candidate),
                        }
                    )
                e4m3_candidates.sort(key=lambda item: float(item["time_us"]))
                if args.graph_replays:
                    e4m3_best_policy = tuple(e4m3_candidates[0]["policy"])
                    initial = runner.e4m3(e4m3_best_policy).clone()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = runner.e4m3(e4m3_best_policy)
                    for _ in range(args.graph_replays):
                        graph.replay()
                    torch.cuda.synchronize()
                    replayed = captured.clone()
                    e4m3_graph_validation = {
                        "replayed_vs_initial": distance(initial, replayed),
                        "graph_replays": args.graph_replays,
                        "nonzero_locks": int(
                            torch.count_nonzero(runner.e4m3_locks).item()
                        ),
                    }
                    if (
                        e4m3_graph_validation["replayed_vs_initial"]["max_abs"]
                        != 0.0
                        or e4m3_graph_validation["nonzero_locks"] != 0
                    ):
                        raise RuntimeError(
                            f"{shape.name}: E4M3 graph replay failed: "
                            f"{e4m3_graph_validation}"
                        )
            graph_validation: dict[str, object] | None = None
            if args.graph_replays:
                dispatched = runner.int8_dispatch().clone()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = runner.int8_dispatch()
                for _ in range(args.graph_replays):
                    graph.replay()
                torch.cuda.synchronize()
                replayed = captured.clone()
                torch.cuda.synchronize()
                graph_validation = {
                    "initial_vs_exact": distance(reference, dispatched),
                    "replayed_vs_initial": distance(dispatched, replayed),
                    "graph_replays": args.graph_replays,
                    "nonzero_locks": int(
                        torch.count_nonzero(runner.int8_locks).item()
                    ),
                }
                if (
                    graph_validation["replayed_vs_initial"]["max_abs"] != 0.0
                    or graph_validation["nonzero_locks"] != 0
                ):
                    raise RuntimeError(
                        f"{shape.name}: INT8 graph replay failed: "
                        f"{graph_validation}"
                    )
            record: dict[str, object] = {
                "shape": shape.name,
                "rows": rows,
                "k": shape.k,
                "n": shape.n,
                "bits": shape.bits,
                "frequency": shape.frequency,
                "exact_policy": list(exact_policy),
                "exact_fused_output": shape.fused_output,
                "int8_fused_output": shape.fused_output,
                "exact_us": exact_fused_us,
                "exact_unfused_us": exact_unfused_us,
                "trellis_bytes": trellis.numel() * trellis.element_size(),
                "int8_bytes": (
                    runner.packed_lane.numel() * runner.packed_lane.element_size()
                    + runner.tile_scales.numel() * runner.tile_scales.element_size()
                ),
                "int6_bytes": (
                    runner.int6_words.numel() * runner.int6_words.element_size()
                    + runner.int6_scales.numel()
                    * runner.int6_scales.element_size()
                ),
                "candidates": candidates,
                "graph_validation": graph_validation,
                "int6_candidates": int6_candidates,
                "int6_graph_validation": int6_graph_validation,
                "int10_candidates": int10_candidates,
                "int10_bytes": (
                    runner.int10_words.numel() * runner.int10_words.element_size()
                    + runner.int10_scales.numel()
                    * runner.int10_scales.element_size()
                ),
                "e4m3_candidates": e4m3_candidates,
                "e4m3_graph_validation": e4m3_graph_validation,
            }
            results.append(record)
            print(
                f"{shape.name:12s} M={rows} exact={exact_fused_us:8.3f} us "
                f"int8={float(best['time_us']):8.3f} us "
                f"speedup={float(best['speedup_vs_exact']):.3f}x "
                f"policy={tuple(best['policy'])} "
                f"rel_l2={float(best['relative_l2']):.4g} "
                f"cos={float(best['cosine']):.7f}"
            )
            if e4m3_candidates:
                e4m3_best = e4m3_candidates[0]
                print(
                    f"{'':12s} M={rows} e4m3={float(e4m3_best['time_us']):8.3f} us "
                    f"vs_int8={float(e4m3_best['speedup_vs_int8']):.3f}x "
                    f"policy={tuple(e4m3_best['policy'])} "
                    f"rel_l2={float(e4m3_best['relative_l2']):.4g} "
                    f"cos={float(e4m3_best['cosine']):.7f}"
                )
            if int6_candidates:
                int6_best = int6_candidates[0]
                print(
                    f"{'':12s} M={rows} int6={float(int6_best['time_us']):8.3f} us "
                    f"vs_int8={float(int6_best['speedup_vs_int8']):.3f}x "
                    f"policy={tuple(int6_best['policy'])} "
                    f"rel_l2={float(int6_best['relative_l2']):.4g} "
                    f"cos={float(int6_best['cosine']):.7f}"
                )
            if int10_candidates:
                int10_best = int10_candidates[0]
                print(
                    f"{'':12s} M={rows} int10={float(int10_best['time_us']):8.3f} us "
                    f"vs_int8={float(int10_best['speedup_vs_int8']):.3f}x "
                    f"policy={tuple(int10_best['policy'])} "
                    f"rel_l2={float(int10_best['relative_l2']):.4g} "
                    f"cos={float(int10_best['cosine']):.7f}"
                )
            del runner, trellis, suh, svh
            torch.cuda.empty_cache()

    weighted_exact = sum(
        float(item["exact_us"]) * int(item["frequency"]) for item in results
    )
    weighted_int8 = sum(
        float(item["candidates"][0]["time_us"]) * int(item["frequency"])
        for item in results
    )
    int6_records = [item for item in results if item["int6_candidates"]]
    weighted_int6_int8 = sum(
        float(item["candidates"][0]["time_us"]) * int(item["frequency"])
        for item in int6_records
    )
    weighted_int6 = sum(
        float(item["int6_candidates"][0]["time_us"])
        * int(item["frequency"])
        for item in int6_records
    )
    e4m3_records = [item for item in results if item["e4m3_candidates"]]
    weighted_e4m3_exact = sum(
        float(item["exact_us"]) * int(item["frequency"])
        for item in e4m3_records
    )
    weighted_e4m3_int8 = sum(
        float(item["candidates"][0]["time_us"]) * int(item["frequency"])
        for item in e4m3_records
    )
    weighted_e4m3 = sum(
        float(item["e4m3_candidates"][0]["time_us"])
        * int(item["frequency"])
        for item in e4m3_records
    )
    summary = {
        "library": str(args.library),
        "checkpoint": str(args.checkpoint),
        "weighted_exact_ms": weighted_exact / 1000.0,
        "weighted_int8_ms": weighted_int8 / 1000.0,
        "weighted_speedup": weighted_exact / weighted_int8,
        "weighted_int6_ms": weighted_int6 / 1000.0,
        "weighted_int6_speedup_vs_int8": (
            weighted_int6_int8 / weighted_int6 if weighted_int6 else 0.0
        ),
        "weighted_e4m3_exact_ms": weighted_e4m3_exact / 1000.0,
        "weighted_e4m3_int8_ms": weighted_e4m3_int8 / 1000.0,
        "weighted_e4m3_ms": weighted_e4m3 / 1000.0,
        "weighted_e4m3_speedup_vs_int8": (
            weighted_e4m3_int8 / weighted_e4m3 if weighted_e4m3 else 0.0
        ),
        "results": results,
    }
    print(
        f"WEIGHTED exact={weighted_exact / 1000.0:.4f} ms "
        f"int8={weighted_int8 / 1000.0:.4f} ms "
        f"speedup={weighted_exact / weighted_int8:.4f}x"
    )
    if weighted_e4m3:
        print(
            f"E4M3 WEIGHTED exact={weighted_e4m3_exact / 1000.0:.4f} ms "
            f"int8={weighted_e4m3_int8 / 1000.0:.4f} ms "
            f"e4m3={weighted_e4m3 / 1000.0:.4f} ms "
            f"vs_int8={weighted_e4m3_int8 / weighted_e4m3:.4f}x"
        )
    if weighted_int6:
        print(
            f"INT6 WEIGHTED int8={weighted_int6_int8 / 1000.0:.4f} ms "
            f"int6={weighted_int6 / 1000.0:.4f} ms "
            f"vs_int8={weighted_int6_int8 / weighted_int6:.4f}x"
        )
    int10_records = [item for item in results if item.get("int10_candidates")]
    weighted_int10_int8 = sum(
        float(item["candidates"][0]["time_us"]) * int(item["frequency"])
        for item in int10_records
    )
    weighted_int10 = sum(
        float(item["int10_candidates"][0]["time_us"])
        * int(item["frequency"])
        for item in int10_records
    )
    summary["weighted_int10_ms"] = weighted_int10 / 1000.0
    summary["weighted_int10_speedup_vs_int8"] = (
        weighted_int10_int8 / weighted_int10 if weighted_int10 else 0.0
    )
    if weighted_int10:
        print(
            f"INT10 WEIGHTED int8={weighted_int10_int8 / 1000.0:.4f} ms "
            f"int10={weighted_int10 / 1000.0:.4f} ms "
            f"vs_int8={weighted_int10_int8 / weighted_int10:.4f}x"
        )
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
