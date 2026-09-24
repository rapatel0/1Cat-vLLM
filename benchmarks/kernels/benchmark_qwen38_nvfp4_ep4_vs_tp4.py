#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B023
"""Compare per-rank Qwen3.8 Flash-Next MoE cost under TP4 and EP4 on SM70.

TP4 gives every rank all 512 experts with a quarter of the intermediate
dimension (W13 N=320, W2 K=160). EP4 gives every rank 128 whole experts
(W13 N=1280, W2 K=640). Both read the same NVFP4 bytes on average. This
benchmark tests whether the larger EP4 GEMMs run more efficiently.

Timing is cold-L2 per layer, which matches decode: one layer's active expert
weights do not survive in the 6 MiB L2 until the next step. The EP4 step
time is the busiest rank, because the MoE output all-reduce waits for it.

Paths:
  tp4_generic  nvfp4_moe_dense_stage (W13) + silu_and_mul + dense_stage (W2)
  tp4_tuned    production qpn m1 (M=1) or mtp5 (M=5) kernels, W2 split 1
  ep4_generic  the tp4_generic sequence on EP4 shapes and local routes only

A correctness check runs first: the sum over four TP4 slices and the sum
over four EP4 ranks must both equal the full MoE output.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops
from vllm.model_executor.layers.quantization.sm70_turbomind import (
    unpack_mxfp4_weight,
)

MODEL = Path("/models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4")
PREFIX = "model.language_model.layers.0.mlp.experts"
HIDDEN = 2560
FULL_INTER = 640
TP = 4
EXPERTS = 512
TOP_K = 10
SOURCE_EXPERTS = 32
GROUP = 16
# dense_stage uses its compact grouped launch for at most 80 one-row groups.
COMPACT_MAX_ROUTES = 80


def cold_us(fn: Callable[[], None], flush: torch.Tensor, trials: int) -> float:
    for _ in range(5):
        flush.zero_()
        fn()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        flush.zero_()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


class Loader:
    def __init__(self, model: Path) -> None:
        self.model = model
        index = json.loads((model / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]

    def get(self, expert: int, suffix: str) -> torch.Tensor:
        key = f"{PREFIX}.{expert}.{suffix}"
        with safe_open(self.model / self.weight_map[key], "pt", device="cpu") as f:
            return f.get_tensor(key).cuda().contiguous()


def prepare(loader: Loader, expert: int, inter_lo: int, inter_hi: int):
    """Prepare W13 and W2 for intermediate rows [inter_lo, inter_hi)."""
    g = loader.get(expert, "gate_proj.weight")[inter_lo:inter_hi]
    u = loader.get(expert, "up_proj.weight")[inter_lo:inter_hi]
    gs = (
        loader.get(expert, "gate_proj.weight_scale")[inter_lo:inter_hi].float()
        * loader.get(expert, "gate_proj.weight_scale_2").float()
    )
    us = (
        loader.get(expert, "up_proj.weight_scale")[inter_lo:inter_hi].float()
        * loader.get(expert, "up_proj.weight_scale_2").float()
    )
    w13 = sm70_ops.nvfp4_sm70_prepare(
        unpack_mxfp4_weight(torch.cat((g, u)).contiguous()),
        torch.cat((gs, us)).half().t().contiguous(),
        GROUP,
    )
    d = loader.get(expert, "down_proj.weight")[:, inter_lo // 2 : inter_hi // 2]
    ds = (
        loader.get(expert, "down_proj.weight_scale")[
            :, inter_lo // GROUP : inter_hi // GROUP
        ].float()
        * loader.get(expert, "down_proj.weight_scale_2").float()
    )
    w2 = sm70_ops.nvfp4_sm70_prepare(
        unpack_mxfp4_weight(d.contiguous()), ds.half().t().contiguous(), GROUP
    )
    return w13, w2


class ExpertBank:
    """Prepared experts for one rank, tiled from the real source experts."""

    def __init__(self, items: list, count: int, inter: int) -> None:
        reps = (count + len(items) - 1) // len(items)
        self.inter = inter
        for name, idx in (("w13", 0), ("w2", 1)):
            w = torch.stack([it[idx][0] for it in items]).repeat(reps, 1, 1)[:count]
            s = torch.stack([it[idx][1] for it in items]).repeat(reps, 1, 1)[:count]
            meta = items[0][idx][2]
            ptrs = sm70_ops.awq_moe_build_strided_ptrs(
                w.contiguous(), s.contiguous(), int(meta[0]), int(meta[1]), count
            )
            setattr(self, name, (w.contiguous(), s.contiguous(), ptrs))


class GenericMoE:
    """dense_stage W13 -> silu_and_mul -> dense_stage W2 over compact groups."""

    def __init__(self, bank: ExpertBank, x: torch.Tensor, routes) -> None:
        # routes: list of (token, local_expert)
        routes = sorted(routes, key=lambda r: (r[1], r[0]))
        self.bank = bank
        self.rows = len(routes)
        experts = [e for _, e in routes]
        if self.rows <= COMPACT_MAX_ROUTES:
            # Production decode keeps one row per route, which selects the
            # compact grouped launch. Merged duplicates fall to a per-expert
            # launch loop inside dense_stage.
            uniq = experts
            offsets = list(range(self.rows + 1))
        else:
            uniq = sorted(set(experts))
            offsets = [0]
            for e in uniq:
                offsets.append(offsets[-1] + experts.count(e))
        dev = x.device
        self.tokens = torch.tensor([t for t, _ in routes], device=dev)
        self.inp = x.index_select(0, self.tokens).contiguous()
        self.offsets = torch.tensor(offsets, device=dev, dtype=torch.int32)
        self.ids = torch.tensor(uniq, device=dev, dtype=torch.int32)
        self.groups = len(uniq)
        self.h13 = torch.empty(self.rows, 2 * bank.inter, device=dev, dtype=torch.half)
        self.act = torch.empty(self.rows, bank.inter, device=dev, dtype=torch.half)
        self.out = torch.empty(self.rows, HIDDEN, device=dev, dtype=torch.half)

    def __call__(self) -> None:
        if self.rows == 0:
            return
        _, _, p13 = self.bank.w13
        _, _, p2 = self.bank.w2
        sm70_ops.nvfp4_moe_dense_stage_sm70_out(
            self.h13,
            self.inp,
            self.offsets,
            self.ids,
            p13[0],
            p13[1],
            self.groups,
            HIDDEN,
            2 * self.bank.inter,
            GROUP,
        )
        torch.ops._C.silu_and_mul(self.act, self.h13)
        sm70_ops.nvfp4_moe_dense_stage_sm70_out(
            self.out,
            self.act,
            self.offsets,
            self.ids,
            p2[0],
            p2[1],
            self.groups,
            self.bank.inter,
            HIDDEN,
            GROUP,
        )

    def combine(self, weights: dict, tokens: int) -> torch.Tensor:
        """Weighted sum of route outputs per token, FP32."""
        result = torch.zeros(tokens, HIDDEN, device=self.out.device)
        if self.rows:
            w = torch.tensor(
                [
                    weights[(int(t), int(e))]
                    for t, e in zip(self.tokens.tolist(), self._expert_per_row())
                ],
                device=self.out.device,
            )
            result.index_add_(0, self.tokens, self.out.float() * w[:, None])
        return result

    def _expert_per_row(self) -> list[int]:
        out = []
        offs = self.offsets.tolist()
        for g, e in enumerate(self.ids.tolist()):
            out += [e] * (offs[g + 1] - offs[g])
        return out


def sample_routing(tokens: int, gen: torch.Generator) -> list[list[int]]:
    return [
        torch.randperm(EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)
    ]


def correctness(loader: Loader, x_tokens: int, dev) -> dict:
    """Full MoE on 32 experts: sum of TP4 slices vs sum of EP4 ranks."""
    n = SOURCE_EXPERTS
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(x_tokens, HIDDEN, device=dev, dtype=torch.half).mul_(0.1)
    route = [torch.randperm(n, generator=gen)[:TOP_K].tolist() for _ in range(x_tokens)]
    wts = {
        (t, e): float(w)
        for t in range(x_tokens)
        for e, w in zip(route[t], torch.softmax(torch.randn(TOP_K, generator=gen), 0))
    }
    step = FULL_INTER // TP
    tp_sum = torch.zeros(x_tokens, HIDDEN, device=dev)
    for r in range(TP):
        items = [prepare(loader, e, r * step, (r + 1) * step) for e in range(n)]
        moe = GenericMoE(
            ExpertBank(items, n, step),
            x,
            [(t, e) for t in range(x_tokens) for e in route[t]],
        )
        moe()
        tp_sum += moe.combine(wts, x_tokens)
    full_items = [prepare(loader, e, 0, FULL_INTER) for e in range(n)]
    ep_sum = torch.zeros(x_tokens, HIDDEN, device=dev)
    per = n // TP
    for r in range(TP):
        local = [
            (t, e - r * per)
            for t in range(x_tokens)
            for e in route[t]
            if r * per <= e < (r + 1) * per
        ]
        bank = ExpertBank(full_items[r * per : (r + 1) * per], per, FULL_INTER)
        moe = GenericMoE(bank, x, local)
        moe()
        # combine() keys use local ids; remap weights to local ids for this rank.
        local_w = {
            (t, e - r * per): w
            for (t, e), w in wts.items()
            if r * per <= e < (r + 1) * per
        }
        ep_sum += moe.combine(local_w, x_tokens)
    torch.accelerator.synchronize()
    delta = (tp_sum - ep_sum).abs()
    return {
        "tokens": x_tokens,
        "max_abs": float(delta.max()),
        "relative_l2": float((tp_sum - ep_sum).norm() / tp_sum.norm()),
        "output_rms": float(tp_sum.pow(2).mean().sqrt()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 2, 4, 5, 8, 10, 16, 32]
    )
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--trials", type=int, default=51)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("requires SM70")
    dev = torch.device("cuda")
    loader = Loader(args.model)
    result: dict = {"gpu": torch.cuda.get_device_name(), "correctness": []}

    for m in (1, 5):
        result["correctness"].append(correctness(loader, m, dev))
        print(json.dumps(result["correctness"][-1]), flush=True)

    step = FULL_INTER // TP
    tp_items = [prepare(loader, e, 0, step) for e in range(SOURCE_EXPERTS)]
    ep_items = [prepare(loader, e, 0, FULL_INTER) for e in range(SOURCE_EXPERTS)]
    tp_bank = ExpertBank(tp_items, EXPERTS, step)
    ep_bank = ExpertBank(ep_items, EXPERTS // TP, FULL_INTER)
    del tp_items, ep_items
    flush = torch.empty(64 * 1024 * 1024, device=dev, dtype=torch.uint8)
    have_qpn = sm70_ops.has_nvfp4_qpn_mtp5_dispatch()
    gen = torch.Generator().manual_seed(7)
    result["cases"] = []

    for m in args.tokens:
        x = torch.randn(m, HIDDEN, device=dev, dtype=torch.half).mul_(0.1)
        tp_g, tp_t, ep_max, ep_mean, ep_rows = [], [], [], [], []
        for _ in range(args.draws):
            route = sample_routing(m, gen)
            tp_routes = [(t, e) for t in range(m) for e in route[t]]
            tp_g.append(cold_us(GenericMoE(tp_bank, x, tp_routes), flush, args.trials))
            if not have_qpn:
                pass
            elif m in (1, 5):
                op = (
                    sm70_ops.nvfp4_moe_qpn_m1_sm70_out
                    if m == 1
                    else sm70_ops.nvfp4_moe_qpn_mtp5_sm70_out
                )
                split = 16 if m == 1 else 4
                ids = torch.tensor(
                    [e for t in range(m) for e in route[t]],
                    device=dev,
                    dtype=torch.int32,
                )
                h13 = torch.empty(m * TOP_K, 2 * step, device=dev, dtype=torch.half)
                act = torch.empty(m * TOP_K, step, device=dev, dtype=torch.half)
                o2 = torch.empty(m * TOP_K, HIDDEN, device=dev, dtype=torch.half)
                w13, s13, _ = tp_bank.w13
                w2, s2, _ = tp_bank.w2

                def tuned() -> None:
                    op(h13, x, w13, s13, ids, True, split)
                    torch.ops._C.silu_and_mul(act, h13)
                    op(o2, act, w2, s2, ids, False, 1)

                tp_t.append(cold_us(tuned, flush, args.trials))
            elif m in (4, 8, 16):
                # Production batch decode: fused W13+SwiGLU, fused W2+reduce.
                ids = torch.tensor(
                    [e for t in range(m) for e in route[t]],
                    device=dev,
                    dtype=torch.int32,
                )
                act = torch.empty(m * TOP_K, step, device=dev, dtype=torch.half)
                out = torch.empty(m, HIDDEN, device=dev, dtype=torch.half)
                topw = torch.softmax(torch.randn(m, TOP_K, device=dev), -1)
                w13, s13, _ = tp_bank.w13
                w2, s2, _ = tp_bank.w2

                def tuned_batch() -> None:
                    sm70_ops.nvfp4_moe_qpn_w13_swiglu_batch_sm70_out(
                        act, x, w13, s13, ids, False
                    )
                    sm70_ops.nvfp4_moe_qpn_w2_reduce_sm70_out(
                        out, act, w2, s2, ids, topw
                    )

                tp_t.append(cold_us(tuned_batch, flush, args.trials))
            rank_us, rank_rows = [], []
            per = EXPERTS // TP
            for r in range(TP):
                local = [
                    (t, e - r * per)
                    for t, e in tp_routes
                    if r * per <= e < (r + 1) * per
                ]
                rank_rows.append(len(local))
                rank_us.append(
                    cold_us(GenericMoE(ep_bank, x, local), flush, args.trials)
                    if local
                    else 0.0
                )
            ep_max.append(max(rank_us))
            ep_mean.append(statistics.mean(rank_us))
            ep_rows.append(rank_rows)
        case = {
            "tokens": m,
            "routes": m * TOP_K,
            "tp4_generic_us": statistics.median(tp_g),
            "tp4_tuned_us": statistics.median(tp_t) if tp_t else None,
            "ep4_generic_busiest_rank_us": statistics.median(ep_max),
            "ep4_generic_mean_rank_us": statistics.median(ep_mean),
            "ep4_rank_rows": ep_rows,
        }
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
