#!/usr/bin/env python3
"""Compare state, raw-trellis, and INT8 grouped EXL3 projections."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


SOURCE = "model.language_model.layers.0.mlp.gate_proj"


@dataclass(frozen=True)
class Shape:
    name: str
    k: int
    n: int
    bits: int
    frequency: int
    shared_suh: bool = True


SHAPES = (
    Shape("gdn_qk", 5120, 512, 5, 48),
    Shape("gdn_vz", 5120, 1536, 5, 48),
    Shape("self_kv", 5120, 256, 5, 16),
    Shape("gate_up", 5120, 4352, 5, 64, False),
)


def load_tensor(root: Path, index: dict[str, str], suffix: str) -> torch.Tensor:
    key = f"{SOURCE}.{suffix}"
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def timed_us(function, iterations: int, repeats: int) -> float:
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples)


def distance(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref)).item()
        ),
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
    parser.add_argument("--rows", default="1,2,4,8")
    parser.add_argument(
        "--shapes",
        default=",".join(shape.name for shape in SHAPES),
        help="comma-separated subset of projection shapes",
    )
    parser.add_argument(
        "--profile-once",
        choices=("exact", "raw", "raw-unfused", "int8"),
        help="run the selected projection once and exit for Nsight Compute",
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--graph-replays", type=int, default=10001)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.ops.load_library(str(args.library))
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)["weight_map"]
    source_trellis = load_tensor(args.checkpoint, index, "trellis")
    source_suh = load_tensor(args.checkpoint, index, "suh")
    source_svh = load_tensor(args.checkpoint, index, "svh")
    selected_shapes = set(args.shapes.split(","))
    unknown_shapes = selected_shapes.difference(shape.name for shape in SHAPES)
    if unknown_shapes:
        parser.error(f"unknown shapes: {','.join(sorted(unknown_shapes))}")

    records: list[dict[str, object]] = []
    for rows in (int(value) for value in args.rows.split(",")):
        for shape in SHAPES:
            if shape.name not in selected_shapes:
                continue
            trellis0 = source_trellis[: shape.k // 16, : shape.n // 16].cuda()
            trellis1 = trellis0.clone()
            suh0 = source_suh[: shape.k].cuda()
            suh1 = suh0 if shape.shared_suh else suh0.clone()
            svh0 = source_svh[: shape.n].cuda()
            svh1 = svh0.clone()
            state0 = torch.ops._C.exl3_sm70_tm_state_repack(trellis0)
            state1 = state0.clone()
            packed0, scales0 = torch.ops._C.exl3_sm70_tm_int8_repack(trellis0)
            packed1, scales1 = torch.ops._C.exl3_sm70_tm_int8_repack(trellis1)
            exact_metadata = torch.ops._C.exl3_sm70_tm_gate_up_metadata(
                state0, state1, svh0, svh1
            )
            raw_metadata = torch.ops._C.exl3_sm70_tm_raw_pair_metadata(
                trellis0, trellis1, svh0, svh1
            )
            int8_metadata = torch.ops._C.exl3_sm70_tm_int8_pair_metadata(
                packed0, scales0, packed1, scales1, svh0, svh1
            )
            second_input_row = 0 if shape.shared_suh else 8
            offsets = torch.tensor(
                [
                    0,
                    shape.n,
                    2 * shape.n,
                    0,
                    second_input_row,
                    16,
                    0,
                    8,
                    16,
                ],
                dtype=torch.int32,
                device="cuda",
            )
            locks = torch.zeros(
                2 * (shape.n // 128 + 64), dtype=torch.int32, device="cuda"
            )
            generator = torch.Generator(device="cuda").manual_seed(
                38271 + rows * 10000 + shape.n
            )
            x = torch.randn(
                (rows, shape.k),
                generator=generator,
                dtype=torch.float16,
                device="cuda",
            ).mul_(0.125)

            def exact() -> torch.Tensor:
                return torch.ops._C.exl3_sm70_tm_state_pair_gemm(
                    x,
                    trellis0,
                    trellis1,
                    state0,
                    state1,
                    suh0,
                    suh1,
                    svh0,
                    svh1,
                    exact_metadata,
                    offsets,
                    locks,
                    shape.bits,
                )

            def raw(splits: int = -1, swizzle: int = -1) -> torch.Tensor:
                return torch.ops._C.exl3_sm70_tm_raw_pair_gemm(
                    x,
                    trellis0,
                    trellis1,
                    suh0,
                    suh1,
                    svh0,
                    svh1,
                    raw_metadata,
                    offsets,
                    locks,
                    shape.bits,
                    splits,
                    swizzle,
                    True,
                )

            def raw_unfused(
                splits: int = -1, swizzle: int = -1
            ) -> torch.Tensor:
                return torch.ops._C.exl3_sm70_tm_raw_pair_gemm(
                    x,
                    trellis0,
                    trellis1,
                    suh0,
                    suh1,
                    svh0,
                    svh1,
                    raw_metadata,
                    offsets,
                    locks,
                    shape.bits,
                    splits,
                    swizzle,
                    False,
                )

            def int8() -> torch.Tensor:
                return torch.ops._C.exl3_sm70_tm_int8_pair_gemm(
                    x,
                    trellis0,
                    trellis1,
                    packed0,
                    scales0,
                    packed1,
                    scales1,
                    suh0,
                    suh1,
                    svh0,
                    svh1,
                    int8_metadata,
                    offsets,
                    locks,
                    shape.bits,
                )

            if args.profile_once:
                selected = {
                    "exact": exact,
                    "raw": raw,
                    "raw-unfused": raw_unfused,
                    "int8": int8,
                }[args.profile_once]
                selected()
                torch.cuda.synchronize()
                print(
                    f"profiled {args.profile_once} {shape.name} M={rows}"
                )
                return

            reference = exact().clone()
            raw_candidate = raw().clone()
            raw_unfused_candidate = raw_unfused().clone()
            candidate = int8().clone()
            torch.cuda.synchronize()
            exact_us = timed_us(exact, args.iterations, args.repeats)
            raw_us = timed_us(raw, args.iterations, args.repeats)
            raw_unfused_us = timed_us(
                raw_unfused, args.iterations, args.repeats
            )
            int8_us = timed_us(int8, args.iterations, args.repeats)
            initial = raw_unfused().clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = raw_unfused()
            for _ in range(args.graph_replays):
                graph.replay()
            torch.cuda.synchronize()
            replayed = captured.clone()
            graph_distance = distance(initial, replayed)
            nonzero_locks = int(torch.count_nonzero(locks).item())
            record = {
                "shape": shape.name,
                "rows": rows,
                "k": shape.k,
                "n": shape.n,
                "bits": shape.bits,
                "frequency": shape.frequency,
                "exact_us": exact_us,
                "raw_us": raw_us,
                "raw_unfused_us": raw_unfused_us,
                "int8_us": int8_us,
                "raw_vs_int8_speedup": int8_us / raw_us,
                "raw_unfused_vs_int8_speedup": int8_us / raw_unfused_us,
                "raw_distance": distance(reference, raw_candidate),
                "raw_unfused_distance": distance(
                    reference, raw_unfused_candidate
                ),
                "int8_distance": distance(reference, candidate),
                "graph_distance": graph_distance,
                "graph_replays": args.graph_replays,
                "nonzero_locks": nonzero_locks,
            }
            records.append(record)
            print(
                f"{shape.name:8s} M={rows} exact={exact_us:8.3f}us "
                f"raw={raw_us:8.3f}us unfused={raw_unfused_us:8.3f}us "
                f"int8={int8_us:8.3f}us "
                f"raw/int8={int8_us / raw_us:.3f}x "
                f"unfused/int8={int8_us / raw_unfused_us:.3f}x "
                f"raw_rel={record['raw_distance']['relative_l2']:.2e} "
                f"unfused_rel="
                f"{record['raw_unfused_distance']['relative_l2']:.2e} "
                f"int8_rel={record['int8_distance']['relative_l2']:.5f} "
                f"graph_max={graph_distance['max_abs']:.1f} locks={nonzero_locks}"
            )
            if graph_distance["max_abs"] != 0.0 or nonzero_locks != 0:
                raise RuntimeError(f"graph replay failed: {record}")
            del (
                trellis0,
                trellis1,
                suh0,
                suh1,
                svh0,
                svh1,
                state0,
                state1,
                packed0,
                packed1,
                scales0,
                scales1,
                exact_metadata,
                raw_metadata,
                int8_metadata,
                offsets,
                locks,
                x,
                reference,
                raw_candidate,
                raw_unfused_candidate,
                candidate,
                initial,
                captured,
                replayed,
                graph,
            )
            torch.cuda.empty_cache()

    m1 = [record for record in records if record["rows"] == 1]
    weighted_exact = sum(
        float(record["exact_us"]) * int(record["frequency"]) for record in m1
    )
    weighted_int8 = sum(
        float(record["int8_us"]) * int(record["frequency"]) for record in m1
    )
    weighted_raw = sum(
        float(record["raw_us"]) * int(record["frequency"]) for record in m1
    )
    weighted_raw_unfused = sum(
        float(record["raw_unfused_us"]) * int(record["frequency"])
        for record in m1
    )
    result = {
        "library": str(args.library),
        "checkpoint": str(args.checkpoint),
        "weighted_m1_exact_ms": weighted_exact / 1000.0,
        "weighted_m1_raw_ms": weighted_raw / 1000.0,
        "weighted_m1_raw_unfused_ms": weighted_raw_unfused / 1000.0,
        "weighted_m1_int8_ms": weighted_int8 / 1000.0,
        "weighted_m1_raw_vs_int8_speedup": weighted_int8 / weighted_raw,
        "weighted_m1_raw_unfused_vs_int8_speedup": (
            weighted_int8 / weighted_raw_unfused
        ),
        "records": records,
    }
    print(
        f"WEIGHTED M1 exact={weighted_exact / 1000.0:.4f}ms "
        f"raw={weighted_raw / 1000.0:.4f}ms "
        f"unfused={weighted_raw_unfused / 1000.0:.4f}ms "
        f"int8={weighted_int8 / 1000.0:.4f}ms "
        f"raw_vs_int8={weighted_int8 / weighted_raw:.4f}x "
        f"unfused_vs_int8={weighted_int8 / weighted_raw_unfused:.4f}x"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
