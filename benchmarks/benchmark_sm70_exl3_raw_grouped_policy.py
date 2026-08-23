#!/usr/bin/env python3
"""Sweep split-K/swizzle for the grouped raw-trellis SM70 projection."""

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
    shared_suh: bool = True


SHAPES = (
    Shape("gdn_qk", 5120, 512, 5),
    Shape("gdn_vz", 5120, 1536, 5),
    Shape("self_kv", 5120, 256, 5),
    Shape("gate_up", 5120, 4352, 5, False),
)


def load_tensor(root: Path, index: dict[str, str], suffix: str) -> torch.Tensor:
    key = f"{SOURCE}.{suffix}"
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def timed_us(function, iterations: int, repeats: int) -> float:
    for _ in range(8):
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
    delta = candidate.float() - ref
    return {
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref)).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", default="1,4")
    parser.add_argument("--splits", default="2,3,4,5,6,7,8,9,10,11,12,15,16")
    parser.add_argument("--swizzles", default="0,1,2,3")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.ops.load_library(str(args.library))
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)["weight_map"]
    source_trellis = load_tensor(args.checkpoint, index, "trellis")
    source_suh = load_tensor(args.checkpoint, index, "suh")
    source_svh = load_tensor(args.checkpoint, index, "svh")
    splits = [int(value) for value in args.splits.split(",")]
    swizzles = [int(value) for value in args.swizzles.split(",")]
    records: list[dict[str, object]] = []

    for rows in (int(value) for value in args.rows.split(",")):
        for shape in SHAPES:
            trellis0 = source_trellis[: shape.k // 16, : shape.n // 16].cuda()
            trellis1 = trellis0.clone()
            suh0 = source_suh[: shape.k].cuda()
            suh1 = suh0 if shape.shared_suh else suh0.clone()
            svh0 = source_svh[: shape.n].cuda()
            svh1 = svh0.clone()
            metadata = torch.ops._C.exl3_sm70_tm_raw_pair_metadata(
                trellis0, trellis1, svh0, svh1
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
                9817 + rows * 10000 + shape.n
            )
            x = torch.randn(
                (rows, shape.k),
                generator=generator,
                dtype=torch.float16,
                device="cuda",
            ).mul_(0.125)

            def call(split: int = -1, swizzle: int = -1) -> torch.Tensor:
                return torch.ops._C.exl3_sm70_tm_raw_pair_gemm(
                    x,
                    trellis0,
                    trellis1,
                    suh0,
                    suh1,
                    svh0,
                    svh1,
                    metadata,
                    offsets,
                    locks,
                    shape.bits,
                    split,
                    swizzle,
                )

            reference = call().clone()
            best: dict[str, object] | None = None
            for split in splits:
                if split > shape.k // 128:
                    continue
                for swizzle in swizzles:
                    candidate = call(split, swizzle).clone()
                    torch.cuda.synchronize()
                    elapsed = timed_us(
                        lambda split=split, swizzle=swizzle: call(split, swizzle),
                        args.iterations,
                        args.repeats,
                    )
                    metric = distance(reference, candidate)
                    record: dict[str, object] = {
                        "shape": shape.name,
                        "rows": rows,
                        "split": split,
                        "swizzle": swizzle,
                        "microseconds": elapsed,
                        **metric,
                    }
                    records.append(record)
                    if best is None or elapsed < float(best["microseconds"]):
                        best = record
            assert best is not None
            nonzero_locks = int(torch.count_nonzero(locks).item())
            print(
                f"{shape.name:8s} M={rows} best={best['microseconds']:.3f}us "
                f"split={best['split']} swizzle={best['swizzle']} "
                f"max_abs={best['max_abs']:.3g} rel={best['relative_l2']:.3g} "
                f"locks={nonzero_locks}"
            )
            if nonzero_locks:
                raise RuntimeError(f"split-K lock leak for {shape.name} M={rows}")
            del (
                trellis0,
                trellis1,
                suh0,
                suh1,
                svh0,
                svh1,
                metadata,
                offsets,
                locks,
                x,
                reference,
            )
            torch.cuda.empty_cache()

    result = {"library": str(args.library), "records": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
