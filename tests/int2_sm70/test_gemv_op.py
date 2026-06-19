#!/usr/bin/env python3
"""Numerical validation of the int2 SM70 GEMV op vs a torch dequant-matmul ref.

Quantizes random fp16 weights to the per-group affine 2-bit format, runs the
custom op, and checks the output matches `A @ dequant(W).T` to fp16 tolerance.
This is the kernel-level numerical gate (independent of any model / text).

  /opt/venv/bin/python tests/int2_sm70/test_gemv_op.py
"""
import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_OP = os.path.join(_HERE, "..", "..", "csrc", "sm70_int2", "int2_gemv_op.cu")

ext = load(
    name="int2_sm70_gemv",
    sources=[_OP],
    extra_cuda_cflags=["-O3", "-arch=sm_70", "-allow-unsupported-compiler"],
    verbose=True,
)


def quantize_affine_2bit(W: torch.Tensor, group: int):
    """W [N,K] fp16 -> (qweight uint8 [N,K/4], scales/bias half [N,K/group], Wdq)."""
    N, K = W.shape
    ng = K // group
    Wg = W.view(N, ng, group).float()
    mn = Wg.amin(dim=2, keepdim=True)
    mx = Wg.amax(dim=2, keepdim=True)
    scale = (mx - mn) / 3.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round((Wg - mn) / scale).clamp_(0, 3).to(torch.uint8)  # [N,ng,group]
    Wdq = (q.float() * scale + mn).view(N, K)
    qf = q.view(N, K)
    qweight = torch.zeros(N, K // 4, dtype=torch.uint8, device=W.device)
    for t in range(4):                       # byte b holds values 4b+0..3 at bits 0,2,4,6
        qweight |= (qf[:, t::4] << (t * 2))
    scales = scale.squeeze(-1).to(torch.float16).contiguous()
    bias = mn.squeeze(-1).to(torch.float16).contiguous()
    return qweight.contiguous(), scales, bias, Wdq


def run(N, K, group, seed=0):
    torch.manual_seed(seed)
    W = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float16)
    qweight, scales, bias, Wdq = quantize_affine_2bit(W, group)
    A = (torch.randn(1, K, device="cuda")).to(torch.float16)
    out = ext.int2_gemv_m1(A, qweight, scales, bias, group)
    ref = (A.float() @ Wdq.float().t()).to(torch.float16)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)).item()
    maxabs = (out.float() - ref.float()).abs().max().item()
    print(f"  N={N:5d} K={K:6d} group={group:4d}  rel={rel:.3e}  maxabs={maxabs:.3e}")
    return rel


def run_n(M, N, K, group, seed=0):
    torch.manual_seed(seed)
    W = (torch.randn(N, K, device="cuda") * 0.05).to(torch.float16)
    qweight, scales, bias, Wdq = quantize_affine_2bit(W, group)
    wt = ext.int2_repack_nmajor(qweight, K)
    A = (torch.randn(M, K, device="cuda")).to(torch.float16)
    out = ext.int2_gemv_n(A, wt, scales, bias, group)
    ref = (A.float() @ Wdq.float().t()).to(torch.float16)
    rel = ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)).item()
    print(f"  M={M} N={N:5d} K={K:6d} group={group:4d}  rel={rel:.3e}")
    return rel


def main():
    print("[int2 GEMV op] numerical validation vs torch dequant-matmul")
    worst = 0.0
    print(" M=1 (gemv_m1):")
    for (N, K, group) in [(256, 2048, 128), (512, 4096, 128), (1024, 8192, 64),
                          (256, 2048, 512), (128, 16384, 128)]:
        worst = max(worst, run(N, K, group))
    print(" M=2..8 (gemv_n, n-split + repack):")
    for M in (2, 4, 8):
        worst = max(worst, run_n(M, 256, 2048, 128))
        worst = max(worst, run_n(M, 512, 8192, 128))
    ok = worst < 5e-3
    print(f"worst rel={worst:.3e}  ->  {'[GEMV OP OK]' if ok else '[GEMV OP FAIL]'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
