#!/usr/bin/env python3
"""Phase 0 census: INT8 vs exact-state vs raw on leftover no-MTP shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

SHAPES = (
    ("gdn_out", "model.language_model.layers.0.linear_attn.out_proj", 1536, 5120, 5),
    ("self_q", "model.language_model.layers.0.mlp.gate_proj", 5120, 3072, 5),
    ("self_out", "model.language_model.layers.0.linear_attn.out_proj", 1536, 5120, 5),
)


def load_tensor(root: Path, index: dict[str, str], key: str) -> torch.Tensor:
    with safe_open(root / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def load_proj(root: Path, index: dict[str, str], source: str, k: int, n: int, bits: int):
    trellis = load_tensor(root, index, f"{source}.trellis")
    suh = load_tensor(root, index, f"{source}.suh")
    svh = load_tensor(root, index, f"{source}.svh")
    if trellis.shape[0] * 16 == k:
        trellis = trellis[:, : n // 16, :]
        svh = svh[:n]
    else:
        trellis = trellis[: k // 16, : n // 16, :]
        suh = suh[:k]
        svh = svh[:n]
    expected = (k // 16, n // 16, bits * 16)
    if trellis.shape != expected:
        raise ValueError(f"trellis={tuple(trellis.shape)} expected={expected}")
    return trellis.contiguous().cuda(), suh.contiguous().cuda(), svh.contiguous().cuda()


def distance(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    delta = cand.float() - ref.float()
    norm = torch.linalg.vector_norm(ref.float())
    return {
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(delta) / norm).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                ref.float().flatten(), cand.float().flatten(), dim=0
            ).item()
        ),
    }


def timed_us(fn, warmup=5, repeats=3, iters=40) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("SM70 required")
    torch.ops.load_library(str(args.library))
    needed = (
        "exl3_sm70_tm_state_repack",
        "exl3_sm70_tm_int8_repack",
        "exl3_sm70_tm_state_gemm_hadamard_out",
        "exl3_sm70_tm_state_gemm_out",
        "exl3_sm70_tm_int8_gemm_hadamard_out",
        "exl3_sm70_tm_int8_gemm_out",
        "exl3_sm70_tm_raw_dispatch_gemm_persistent_locks",
    )
    for name in needed:
        if not hasattr(torch.ops._C, name):
            raise RuntimeError(f"missing {name}")
    with (args.checkpoint / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)["weight_map"]

    results = []
    for name, source, k, n, bits in SHAPES:
        trellis, suh, svh = load_proj(args.checkpoint, index, source, k, n, bits)
        state = torch.ops._C.exl3_sm70_tm_state_repack(trellis)
        packed, scales = torch.ops._C.exl3_sm70_tm_int8_repack(trellis)
        for rows in (1, 4):
            x = torch.randn((rows, k), device="cuda", dtype=torch.float16).mul_(0.125)
            exact_out = torch.empty((rows, n), device="cuda", dtype=torch.float16)
            int8_out = torch.empty_like(exact_out)
            x_had_e = torch.empty_like(x)
            x_had_i = torch.empty_like(x)
            partials_e = torch.empty((8, n), device="cuda", dtype=torch.float32)
            partials_i = torch.empty_like(partials_e)
            accum = torch.empty((rows, n), device="cuda", dtype=torch.float32)
            locks_e = torch.zeros(((rows + 7) // 8) * (n // 128), device="cuda", dtype=torch.int32)
            locks_i = torch.zeros_like(locks_e)
            locks_r = torch.zeros_like(locks_e)

            def exact_fused():
                torch.ops._C.exl3_sm70_tm_state_gemm_hadamard_out(
                    exact_out, x, state, suh, svh, x_had_e, partials_e, locks_e,
                    bits, 5 if n == 5120 else 6, 0,
                )
                return exact_out

            def exact_unfused():
                torch.ops._C.exl3_sm70_tm_state_gemm_out(
                    exact_out, x, state, suh, svh, x_had_e, accum, partials_e, locks_e,
                    bits, 5 if n == 5120 else 6, 0,
                )
                return exact_out

            def int8_fused():
                torch.ops._C.exl3_sm70_tm_int8_gemm_hadamard_out(
                    int8_out, x, packed, scales, suh, svh, x_had_i, partials_i, locks_i,
                    bits, 5 if n == 5120 else 6, 0,
                )
                return int8_out

            def int8_unfused():
                torch.ops._C.exl3_sm70_tm_int8_gemm_out(
                    int8_out, x, packed, scales, suh, svh, x_had_i, accum, partials_i, locks_i,
                    bits, 5 if n == 5120 else 6, 0,
                )
                return int8_out

            def raw():
                return torch.ops._C.exl3_sm70_tm_raw_dispatch_gemm_persistent_locks(
                    x, trellis, suh, svh, locks_r, bits, True, False,
                )

            ref = exact_fused().clone()
            torch.cuda.synchronize()
            row = {
                "shape": name,
                "rows": rows,
                "k": k,
                "n": n,
                "bits": bits,
                "exact_fused_us": timed_us(exact_fused),
                "exact_unfused_us": timed_us(exact_unfused),
                "int8_fused_us": timed_us(int8_fused),
                "int8_unfused_us": timed_us(int8_unfused),
                "raw_us": timed_us(raw),
                "int8_vs_exact": distance(ref, int8_fused().clone()),
                "raw_vs_exact": distance(ref, raw().clone()),
            }
            results.append(row)
            print(
                f"{name:10} M={rows} exact_f={row['exact_fused_us']:7.2f} "
                f"exact_u={row['exact_unfused_us']:7.2f} "
                f"int8_f={row['int8_fused_us']:7.2f} "
                f"int8_u={row['int8_unfused_us']:7.2f} "
                f"raw={row['raw_us']:7.2f} "
                f"int8_l2={row['int8_vs_exact']['relative_l2']:.4g} "
                f"raw_l2={row['raw_vs_exact']['relative_l2']:.4g}"
            )
        del trellis, suh, svh, state, packed, scales
        torch.cuda.empty_cache()

    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
