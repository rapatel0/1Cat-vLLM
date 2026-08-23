#!/usr/bin/env python3
"""Compare the full SM70 raw-trellis projection with state and INT8 paths."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from benchmark_sm70_exl3_int8 import SHAPES, distance, load_projection


def timed_us(function, repeats: int, iterations: int) -> float:
    for _ in range(10):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1, choices=(1, 4))
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("this benchmark requires an SM70 GPU")
    torch.ops.load_library(str(args.library))
    required = (
        "exl3_sm70_gemm",
        "exl3_sm70_tm_state_repack",
        "exl3_sm70_tm_int8_repack",
        "exl3_sm70_tm_dispatch_gemm_persistent_locks",
        "exl3_sm70_tm_raw_dispatch_gemm_persistent_locks",
        "exl3_sm70_tm_int8_dispatch_gemm_persistent_locks",
    )
    for name in required:
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

    totals = {"raw": 0.0, "state": 0.0, "int8": 0.0}
    for shape in shapes:
        trellis, suh, svh = load_projection(args.checkpoint, index, shape)
        state = torch.ops._C.exl3_sm70_tm_state_repack(trellis)
        packed_lane, tile_scales = torch.ops._C.exl3_sm70_tm_int8_repack(
            trellis
        )
        generator = torch.Generator(device="cuda").manual_seed(
            91021 + shape.k + shape.n + args.rows
        )
        x = torch.randn(
            (args.rows, shape.k),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        ).mul_(0.125)
        lock_count = ((args.rows + 7) // 8) * (shape.n // 128)
        raw_locks = torch.zeros(lock_count, device="cuda", dtype=torch.int32)
        state_locks = torch.zeros_like(raw_locks)
        int8_locks = torch.zeros_like(raw_locks)

        def raw() -> torch.Tensor:
            return torch.ops._C.exl3_sm70_tm_raw_dispatch_gemm_persistent_locks(
                x, trellis, suh, svh, raw_locks, shape.bits, True, False
            )

        def exact_state() -> torch.Tensor:
            return torch.ops._C.exl3_sm70_tm_dispatch_gemm_persistent_locks(
                x,
                trellis,
                state,
                suh,
                svh,
                state_locks,
                shape.bits,
                True,
                False,
            )

        def int8() -> torch.Tensor:
            return torch.ops._C.exl3_sm70_tm_int8_dispatch_gemm_persistent_locks(
                x,
                trellis,
                packed_lane,
                tile_scales,
                suh,
                svh,
                int8_locks,
                shape.bits,
                True,
                False,
            )

        reference = torch.ops._C.exl3_sm70_gemm(
            x, trellis, suh, svh, True, False
        ).clone()
        raw_output = raw().clone()
        state_output = exact_state().clone()
        int8_output = int8().clone()
        torch.cuda.synchronize()
        raw_us = timed_us(raw, args.repeats, args.iterations)
        state_us = timed_us(exact_state, args.repeats, args.iterations)
        int8_us = timed_us(int8, args.repeats, args.iterations)
        lock_residue = (
            int(torch.count_nonzero(raw_locks).item()),
            int(torch.count_nonzero(state_locks).item()),
            int(torch.count_nonzero(int8_locks).item()),
        )
        if lock_residue != (0, 0, 0):
            raise RuntimeError(f"{shape.name}: nonzero split-K locks {lock_residue}")
        raw_error = distance(reference, raw_output)
        state_error = distance(reference, state_output)
        int8_error = distance(reference, int8_output)
        totals["raw"] += shape.frequency * raw_us
        totals["state"] += shape.frequency * state_us
        totals["int8"] += shape.frequency * int8_us
        print(
            f"{shape.name:12s} M={args.rows} "
            f"raw={raw_us:8.3f} state={state_us:8.3f} int8={int8_us:8.3f} us "
            f"raw/int8={raw_us / int8_us:.3f} "
            f"raw_rel={raw_error['relative_l2']:.4g} "
            f"state_rel={state_error['relative_l2']:.4g} "
            f"int8_rel={int8_error['relative_l2']:.4g}"
        )

    print(
        "WEIGHTED "
        + " ".join(f"{name}={value / 1000.0:.4f} ms" for name, value in totals.items())
        + f" raw/int8={totals['raw'] / totals['int8']:.4f}"
    )


if __name__ == "__main__":
    main()
