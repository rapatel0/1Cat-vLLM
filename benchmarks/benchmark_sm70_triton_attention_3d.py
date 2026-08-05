# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep Volta Triton 3D-attention schedules at Hy3's live decode shape.

The production Hy3 TP8 profile has two 64K requests, eight query heads and
one KV head per rank (head dimension 128).  With fewer than the regular 2D
launch threshold, Triton uses its segmented 3D attention path.  This harness
times exactly that path, including the segment reduction, and compares every
candidate against the production 16-segment schedule.

It is deliberately not a generic benchmark: the default geometry mirrors the
64K/two-slot deployment.  A candidate that changes the segment count is
reported as non-bitwise-identical even when its FP16 error is small; it must
therefore pass end-to-end deterministic checks before deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


@dataclass(frozen=True)
class Candidate:
    name: str
    segments: int
    warps: int


@dataclass
class Result:
    candidate: str
    path: str
    segments: int
    warps: int
    equal: bool
    max_diff: float
    mean_diff: float
    num_different: int
    median_ms: float
    speedup_vs_baseline: float
    fp32_max_diff: float | None
    fp32_mean_diff: float | None


@contextmanager
def _decode_warps(warps: int):
    key = "VLLM_SM70_TRITON_ATTN_DECODE_NUM_WARPS"
    previous = os.environ.get(key)
    try:
        os.environ[key] = str(warps)
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _make_inputs(
    *,
    seq_len: int,
    num_seqs: int,
    query_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (num_seqs * query_len, q_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    num_blocks = (seq_len + block_size - 1) // block_size
    k = torch.randn(
        (num_seqs * num_blocks, block_size, kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    v = torch.randn(
        (num_seqs * num_blocks, block_size, kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    block_table = torch.arange(
        num_seqs * num_blocks, device="cuda", dtype=torch.int32
    ).view(num_seqs, num_blocks)
    seq_lens = torch.full((num_seqs,), seq_len, device="cuda", dtype=torch.int32)
    cu_seqlens_q = (
        torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32) * query_len
    )
    return q, k, v, block_table, seq_lens, cu_seqlens_q


def _scratch(
    *,
    num_seqs: int,
    q_heads: int,
    head_dim: int,
    segments: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            (num_seqs, q_heads, segments, head_dim),
            device="cuda",
            dtype=torch.float32,
        ),
        torch.empty(
            (num_seqs, q_heads, segments),
            device="cuda",
            dtype=torch.float32,
        ),
        torch.empty(
            (num_seqs, q_heads, segments),
            device="cuda",
            dtype=torch.float32,
        ),
    )


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    segments: int,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    force_3d: bool,
) -> torch.Tensor:
    segm_output, segm_max, segm_expsum = scratch
    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        # The wrapper normally sends multi-token speculative verification to
        # 2D attention. Passing 1 only to its dispatch decision forces the
        # otherwise identical 3D implementation; cu_seqlens_q still carries
        # the real per-sequence query length into the kernel.
        max_seqlen_q=1 if force_3d else query_len,
        seqused_k=seq_lens,
        max_seqlen_k=seq_len,
        softmax_scale=q.shape[2] ** -0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        # The segmented scratch is indexed by query token, not by request.
        # A speculative q=3 verification batch therefore needs 3x as many
        # rows as two single-token decode requests.
        seq_threshold_3D=q.shape[0],
        num_par_softmax_segments=segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        kv_quant_mode=KVQuantMode.NONE,
    )
    return out


def _time(
    candidate: Candidate,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    warmup: int,
    repeats: int,
    force_3d: bool,
) -> tuple[torch.Tensor, float]:
    out = torch.empty_like(q)
    scratch = _scratch(
        num_seqs=q.shape[0],
        q_heads=q.shape[1],
        head_dim=q.shape[2],
        segments=candidate.segments,
    )
    with _decode_warps(candidate.warps):
        for _ in range(warmup):
            _run(
                q,
                k,
                v,
                out,
                block_table,
                seq_lens,
                cu_seqlens_q,
                seq_len=seq_len,
                query_len=query_len,
                segments=candidate.segments,
                scratch=scratch,
                force_3d=force_3d,
            )
        torch.cuda.synchronize()
        samples: list[float] = []
        result = None
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = _run(
                q,
                k,
                v,
                out,
                block_table,
                seq_lens,
                cu_seqlens_q,
                seq_len=seq_len,
                query_len=query_len,
                segments=candidate.segments,
                scratch=scratch,
                force_3d=force_3d,
            )
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    assert result is not None
    return result, float(statistics.median(samples))


def _fp32_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_len: int,
    query_len: int,
    block_size: int,
    kv_heads: int,
) -> torch.Tensor:
    """Reference the last-token causal attention result in FP32."""
    num_seqs = k.shape[0] // (seq_len // block_size)
    q_heads = q.shape[1]
    head_dim = q.shape[2]
    num_blocks = seq_len // block_size
    k_tokens = k.view(num_seqs, num_blocks, block_size, kv_heads, head_dim)
    v_tokens = v.view(num_seqs, num_blocks, block_size, kv_heads, head_dim)
    k_heads = k_tokens.flatten(1, 2).permute(0, 2, 1, 3).float()
    v_heads = v_tokens.flatten(1, 2).permute(0, 2, 1, 3).float()
    repeats = q_heads // kv_heads
    q_grouped = q.float().view(
        num_seqs, query_len, kv_heads, repeats, head_dim
    ).permute(0, 2, 3, 1, 4)
    scores = torch.einsum("bgrqd,bgld->bgrql", q_grouped, k_heads)
    positions = torch.arange(seq_len, device=q.device)
    query_positions = seq_len - query_len + torch.arange(
        query_len, device=q.device
    )
    scores = scores.masked_fill(
        positions[None, None, None, None, :]
        > query_positions[None, None, None, :, None],
        float("-inf"),
    )
    probabilities = torch.softmax(scores * (head_dim**-0.5), dim=-1)
    return (
        torch.einsum("bgrql,bgld->bgrqd", probabilities, v_heads)
        .permute(0, 3, 1, 2, 4)
        .reshape(num_seqs * query_len, q_heads, head_dim)
        .to(q.dtype)
    )


def _parse_candidates(raw: str) -> list[Candidate]:
    candidates = []
    for spec in raw.split(","):
        segments_raw, warps_raw = spec.split("x", 1)
        segments = int(segments_raw)
        warps = int(warps_raw)
        if segments <= 0 or warps not in (1, 2, 4, 8):
            raise ValueError(f"Invalid candidate {spec!r}; expected SEGMENTSxWARPS")
        candidates.append(Candidate(f"s{segments}_w{warps}", segments, warps))
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--baseline", default="16x8")
    parser.add_argument(
        "--candidates", default="8x8,16x4,16x8,32x4,32x8,64x4,64x8"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--fp32-reference", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    baseline = _parse_candidates(args.baseline)[0]
    candidates = _parse_candidates(args.candidates)
    if baseline not in candidates:
        candidates.insert(0, baseline)
    q, k, v, block_table, seq_lens, cu_seqlens_q = _make_inputs(
        seq_len=args.seq_len,
        num_seqs=args.num_seqs,
        query_len=args.query_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        seed=args.seed,
    )
    reference_force_3d = args.query_len == 1
    reference, baseline_ms = _time(
        baseline,
        q,
        k,
        v,
        block_table,
        seq_lens,
        cu_seqlens_q,
        seq_len=args.seq_len,
        query_len=args.query_len,
        warmup=args.warmup,
        repeats=args.repeats,
        force_3d=reference_force_3d,
    )
    fp32_reference = None
    if args.fp32_reference:
        fp32_reference = _fp32_reference(
            q,
            k,
            v,
            seq_len=args.seq_len,
            query_len=args.query_len,
            block_size=args.block_size,
            kv_heads=args.kv_heads,
        )
        torch.cuda.synchronize()
    results: list[Result] = [
        Result(
            candidate=("s16_w8" if reference_force_3d else "2d_default"),
            path="3d" if reference_force_3d else "2d",
            segments=baseline.segments if reference_force_3d else 0,
            warps=baseline.warps if reference_force_3d else 0,
            equal=True,
            max_diff=0.0,
            mean_diff=0.0,
            num_different=0,
            median_ms=baseline_ms,
            speedup_vs_baseline=1.0,
            fp32_max_diff=(
                float((reference.float() - fp32_reference.float()).abs().max().item())
                if fp32_reference is not None
                else None
            ),
            fp32_mean_diff=(
                float((reference.float() - fp32_reference.float()).abs().mean().item())
                if fp32_reference is not None
                else None
            ),
        )
    ]
    for candidate in candidates:
        actual, median_ms = _time(
            candidate,
            q,
            k,
            v,
            block_table,
            seq_lens,
            cu_seqlens_q,
            seq_len=args.seq_len,
            query_len=args.query_len,
            warmup=args.warmup,
            repeats=args.repeats,
            force_3d=True,
        )
        diff = (actual.float() - reference.float()).abs()
        fp32_diff = (
            (actual.float() - fp32_reference.float()).abs()
            if fp32_reference is not None
            else None
        )
        results.append(
            Result(
                candidate=candidate.name,
                path="3d",
                segments=candidate.segments,
                warps=candidate.warps,
                equal=bool(torch.equal(actual, reference)),
                max_diff=float(diff.max().item()),
                mean_diff=float(diff.mean().item()),
                num_different=int((actual != reference).sum().item()),
                median_ms=median_ms,
                speedup_vs_baseline=baseline_ms / median_ms,
                fp32_max_diff=(
                    float(fp32_diff.max().item()) if fp32_diff is not None else None
                ),
                fp32_mean_diff=(
                    float(fp32_diff.mean().item()) if fp32_diff is not None else None
                ),
            )
        )
    payload = {
        "shape": {
            "seq_len": args.seq_len,
            "num_seqs": args.num_seqs,
            "query_len": args.query_len,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
        },
        "baseline": asdict(baseline),
        "results": [asdict(result) for result in results],
    }
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
