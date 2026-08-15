# SPDX-License-Identifier: Apache-2.0
"""Benchmark a draft-only block-FP8 shadow of an FP16 LM head on SM70."""

import argparse
import json

import torch

from vllm import _sm70_ops as sm70_ops


def bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def block_quantize(weight: torch.Tensor, group_size: int = 128):
    n, k = weight.shape
    assert n % group_size == 0 and k % group_size == 0
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (n // group_size, k // group_size),
        dtype=torch.float32,
        device=weight.device,
    )
    # Chunk by output rows to bound peak memory beside a live serving engine.
    row_chunk = group_size * 16
    for start in range(0, n, row_chunk):
        end = min(start + row_chunk, n)
        blocks = weight[start:end].reshape(
            (end - start) // group_size,
            group_size,
            k // group_size,
            group_size,
        )
        chunk_scales = blocks.abs().amax(dim=(1, 3)).float() / 448.0
        chunk_scales.clamp_min_(torch.finfo(torch.float32).tiny)
        scales[start // group_size : end // group_size].copy_(chunk_scales)
        expanded = (
            chunk_scales[:, None, :, None]
            .expand_as(blocks)
            .reshape(end - start, k)
        )
        qweight[start:end].copy_((weight[start:end] / expanded).to(qweight.dtype))
    return qweight, scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=62080)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16-baseline-ms", type=float, default=0.7713773)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    x = torch.randn((1, args.k), device=device, dtype=torch.float16)
    weight = torch.randn((args.n, args.k), device=device, dtype=torch.float16)
    ref_out = torch.mm(x, weight.t())
    ref_values, ref_indices = ref_out.max(dim=-1)
    qweight, scales = block_quantize(weight)
    del weight, ref_out
    torch.cuda.empty_cache()
    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
        qweight, scales, 128, False
    )
    del qweight, scales
    torch.cuda.empty_cache()
    k_ld, q_ld = int(meta[0].item()), int(meta[1].item())

    fp8_out = torch.empty((1, args.n), device=device, dtype=torch.float16)

    def fp8_gemm_max():
        sm70_ops.fp8_gemm_sm70_out(
            fp8_out, x, tm_weight, tm_scales, 128, k_ld, q_ld, False
        )
        return fp8_out.max(dim=-1)

    fp8_values, fp8_indices = fp8_gemm_max()
    fp8_ms = bench_ms(fp8_gemm_max, args.warmup, args.iters)
    print(
        json.dumps(
            {
                "shape": [1, args.n, args.k],
                "fp8_ms": fp8_ms,
                "fp16_ms": args.fp16_baseline_ms,
                "speedup": args.fp16_baseline_ms / fp8_ms,
                "argmax_equal": bool(torch.equal(fp8_indices, ref_indices)),
                "top1_value_abs_diff": float(
                    (fp8_values.float() - ref_values.float()).abs().max().item()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
