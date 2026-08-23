#!/usr/bin/env python3
"""Validate and time the exact TP4 Qwen3.8 EXL3 fused MLP boundary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open


TP_SIZE = 4
GATE = "model.language_model.layers.0.mlp.gate_proj"
UP = "model.language_model.layers.0.mlp.up_proj"
DOWN = "model.language_model.layers.0.mlp.down_proj"


def load_tensor(root: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def load_column_rank0(
    root: Path, index: dict[str, str], base: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trellis = load_tensor(root, index, f"{base}.trellis")
    suh = load_tensor(root, index, f"{base}.suh")
    svh = load_tensor(root, index, f"{base}.svh")
    trellis = trellis[:, : trellis.shape[1] // TP_SIZE, :].contiguous()
    svh = svh[: svh.numel() // TP_SIZE].contiguous()
    return trellis.cuda(), suh.cuda(), svh.cuda()


def load_row_rank0(
    root: Path, index: dict[str, str], base: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    trellis = load_tensor(root, index, f"{base}.trellis")
    suh = load_tensor(root, index, f"{base}.suh")
    svh = load_tensor(root, index, f"{base}.svh")
    trellis = trellis[: trellis.shape[0] // TP_SIZE, :, :].contiguous()
    suh = suh[: suh.numel() // TP_SIZE].contiguous()
    return trellis.cuda(), suh.cuda(), svh.cuda()


def timed_us(fn: object, warmup: int, repeats: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", default="1,2,3,4,5,6,7,8,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--graph-replays", type=int, default=10001)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("this benchmark requires an SM70 GPU")
    torch.ops.load_library(str(args.library))
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)["weight_map"]

    gate_trellis, gate_suh, gate_svh = load_column_rank0(
        args.checkpoint, index, GATE
    )
    up_trellis, up_suh, up_svh = load_column_rank0(args.checkpoint, index, UP)
    down_trellis, down_suh, down_svh = load_row_rank0(
        args.checkpoint, index, DOWN
    )
    gate_state = torch.ops._C.exl3_sm70_tm_state_repack(gate_trellis)
    up_state = torch.ops._C.exl3_sm70_tm_state_repack(up_trellis)
    down_state = torch.ops._C.exl3_sm70_tm_state_repack(down_trellis)
    down_packed_lane, down_tile_scales = (
        torch.ops._C.exl3_sm70_tm_int8_repack(down_trellis)
    )
    gate_packed_lane, gate_tile_scales = (
        torch.ops._C.exl3_sm70_tm_int8_repack(gate_trellis)
    )
    up_packed_lane, up_tile_scales = (
        torch.ops._C.exl3_sm70_tm_int8_repack(up_trellis)
    )
    gate_metadata = torch.ops._C.exl3_sm70_tm_gate_up_metadata(
        gate_state, up_state, gate_svh, up_svh
    )
    gate_int8_metadata = torch.ops._C.exl3_sm70_tm_int8_pair_metadata(
        gate_packed_lane,
        gate_tile_scales,
        up_packed_lane,
        up_tile_scales,
        gate_svh,
        up_svh,
    )
    gate_bits = gate_trellis.shape[2] // 16
    down_bits = down_trellis.shape[2] // 16
    gate_k = gate_state.shape[0] * 16
    intermediate_n = gate_state.shape[1] * 32
    down_n = down_state.shape[1] * 32
    gate_offsets = torch.tensor(
        [0, intermediate_n, 2 * intermediate_n, 0, 8, 16],
        dtype=torch.int32,
        device="cuda",
    )
    empty_state = torch.empty((0,), dtype=torch.int32, device="cuda")

    results = []
    for rows in (int(value) for value in args.rows.split(",")):
        generator = torch.Generator(device="cuda").manual_seed(3821 + rows)
        x = torch.randn(
            (rows, gate_k), generator=generator, device="cuda", dtype=torch.float16
        ).mul_(0.125)
        baseline_gate_locks = torch.zeros(
            2 * (intermediate_n // 128 + 4), dtype=torch.int32, device="cuda"
        )
        baseline_down_locks = torch.zeros(
            down_n // 128, dtype=torch.int32, device="cuda"
        )
        fused_gate_locks = torch.zeros_like(baseline_gate_locks)
        fused_down_locks = torch.zeros_like(baseline_down_locks)
        int8_gate_locks = torch.zeros(
            2 * (intermediate_n // 128 + 64), dtype=torch.int32, device="cuda"
        )
        int8_down_locks = torch.zeros_like(baseline_down_locks)
        memory_neutral_gate_locks = torch.zeros_like(int8_gate_locks)
        memory_neutral_down_locks = torch.zeros_like(int8_down_locks)

        def baseline_call() -> torch.Tensor:
            activated = torch.ops._C.exl3_sm70_tm_state_gate_up_silu_mul(
                x,
                gate_trellis,
                up_trellis,
                gate_state,
                up_state,
                gate_suh,
                up_suh,
                gate_svh,
                up_svh,
                gate_metadata,
                gate_offsets,
                baseline_gate_locks,
                gate_bits,
            )
            return torch.ops._C.exl3_sm70_tm_dispatch_gemm_persistent_locks(
                activated,
                down_trellis,
                down_state,
                down_suh,
                down_svh,
                baseline_down_locks,
                down_bits,
                True,
                False,
            )

        def fused_call() -> torch.Tensor:
            return torch.ops._C.exl3_sm70_tm_state_mlp(
                x,
                gate_trellis,
                up_trellis,
                down_trellis,
                gate_state,
                up_state,
                down_state,
                gate_suh,
                up_suh,
                down_suh,
                gate_svh,
                up_svh,
                down_svh,
                down_state,
                down_svh,
                gate_metadata,
                gate_offsets,
                fused_gate_locks,
                fused_down_locks,
                gate_bits,
                down_bits,
                False,
                False,
            )

        def int8_down_fused_call() -> torch.Tensor:
            return torch.ops._C.exl3_sm70_tm_state_mlp(
                x,
                gate_trellis,
                up_trellis,
                down_trellis,
                gate_state,
                up_state,
                down_state,
                gate_suh,
                up_suh,
                down_suh,
                gate_svh,
                up_svh,
                down_svh,
                down_packed_lane,
                down_tile_scales,
                gate_metadata,
                gate_offsets,
                int8_gate_locks,
                int8_down_locks,
                gate_bits,
                down_bits,
                False,
                True,
            )

        def int8_fused_call() -> torch.Tensor:
            return torch.ops._C.exl3_sm70_tm_state_mlp(
                x,
                gate_trellis,
                up_trellis,
                down_trellis,
                gate_state,
                up_state,
                down_state,
                gate_suh,
                up_suh,
                down_suh,
                gate_svh,
                up_svh,
                down_svh,
                down_packed_lane,
                down_tile_scales,
                gate_int8_metadata,
                gate_offsets,
                int8_gate_locks,
                int8_down_locks,
                gate_bits,
                down_bits,
                True,
                True,
            )

        def memory_neutral_int8_call() -> torch.Tensor:
            return torch.ops._C.exl3_sm70_tm_state_mlp(
                x,
                gate_trellis,
                up_trellis,
                down_trellis,
                empty_state,
                empty_state,
                empty_state,
                gate_suh,
                up_suh,
                down_suh,
                gate_svh,
                up_svh,
                down_svh,
                down_packed_lane,
                down_tile_scales,
                gate_int8_metadata,
                gate_offsets,
                memory_neutral_gate_locks,
                memory_neutral_down_locks,
                gate_bits,
                down_bits,
                True,
                True,
            )

        reference = baseline_call()
        candidate = fused_call()
        int8_down_candidate = int8_down_fused_call()
        int8_candidate = int8_fused_call()
        memory_neutral_candidate = memory_neutral_int8_call()
        torch.cuda.synchronize()
        mismatch = torch.count_nonzero(reference.view(torch.int16) != candidate.view(torch.int16)).item()
        result: dict[str, float | int | bool] = {
            "rows": rows,
            "bitwise_equal": mismatch == 0,
            "mismatch": mismatch,
            "max_abs": (reference.float() - candidate.float()).abs().max().item(),
            "int8_rel_l2": (
                (reference.float() - int8_candidate.float()).norm()
                / reference.float().norm()
            ).item(),
            "int8_max_abs": (
                reference.float() - int8_candidate.float()
            ).abs().max().item(),
            "int8_gate_incremental_rel_l2": (
                (int8_down_candidate.float() - int8_candidate.float()).norm()
                / int8_down_candidate.float().norm()
            ).item(),
            "memory_neutral_equal": torch.equal(
                int8_candidate, memory_neutral_candidate
            ),
        }
        if rows <= 8:
            baseline_us = timed_us(
                baseline_call, args.warmup, args.repeats, args.iterations
            )
            fused_us = timed_us(fused_call, args.warmup, args.repeats, args.iterations)
            int8_fused_us = timed_us(
                int8_fused_call, args.warmup, args.repeats, args.iterations
            )
            int8_down_fused_us = timed_us(
                int8_down_fused_call,
                args.warmup,
                args.repeats,
                args.iterations,
            )
            result.update(
                {
                    "baseline_us": baseline_us,
                    "fused_us": fused_us,
                    "speedup": baseline_us / fused_us,
                    "int8_fused_us": int8_fused_us,
                    "int8_vs_exact_speedup": fused_us / int8_fused_us,
                    "int8_down_fused_us": int8_down_fused_us,
                    "int8_gate_incremental_speedup": (
                        int8_down_fused_us / int8_fused_us
                    ),
                }
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = int8_fused_call()
            for _ in range(args.graph_replays):
                graph.replay()
            torch.cuda.synchronize()
            result["int8_graph_stable"] = bool(
                torch.equal(graph_output, int8_candidate)
            )
        result["locks_zero"] = bool(
            baseline_gate_locks.count_nonzero().item() == 0
            and baseline_down_locks.count_nonzero().item() == 0
            and fused_gate_locks.count_nonzero().item() == 0
            and fused_down_locks.count_nonzero().item() == 0
            and int8_gate_locks.count_nonzero().item() == 0
            and int8_down_locks.count_nonzero().item() == 0
            and memory_neutral_gate_locks.count_nonzero().item() == 0
            and memory_neutral_down_locks.count_nonzero().item() == 0
        )
        results.append(result)

    print(json.dumps(results, indent=2))
    if not all(
        result["bitwise_equal"]
        and result["memory_neutral_equal"]
        and result["locks_zero"]
        and result.get("int8_graph_stable", True)
        for result in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
