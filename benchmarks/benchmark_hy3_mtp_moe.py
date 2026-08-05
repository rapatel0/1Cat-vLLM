"""Strict microbenchmark for Hy3 MTP's SM70 unquantized MoE fallback.

The Hy3 MTP drafter uses a 192-expert, 4096-wide MoE whose per-TP-rank
intermediate width is 192.  This harness times the relevant small-token
Triton ``fused_experts`` path and rejects a launch configuration unless its
output is bitwise equal to the established SM70 0.0.3 configuration.

Use a smaller ``--num-experts`` value when benchmarking beside a loaded model:
only the selected experts contribute work for the small-token path, while the
matrix geometry and routed-expert behavior remain representative.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

import torch

from vllm.model_executor.layers.fused_moe import override_config
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


@dataclass(frozen=True)
class Candidate:
    name: str
    config: dict[str, int]


@dataclass
class Result:
    tokens: int
    candidate: str
    config: dict[str, int]
    equal: bool
    max_diff: float
    median_us: float
    speedup_vs_baseline: float
    error: str | None = None


BASELINE = Candidate(
    "sm70_0dot3",
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
    },
)

CANDIDATES = (
    BASELINE,
    Candidate(
        "n64",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        },
    ),
    Candidate(
        "k128",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        },
    ),
    Candidate(
        "n64_k128",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        },
    ),
    Candidate(
        "n128",
        {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        },
    ),
)


def _run(
    candidate: Candidate,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    with override_config(candidate.config):
        return fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            global_num_experts=num_experts,
        )


def _time(
    candidate: Candidate,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        output = _run(
            candidate,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            num_experts,
        )
    torch.cuda.synchronize()

    samples = []
    output = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = _run(
            candidate,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            num_experts,
        )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    assert output is not None
    return output, float(statistics.median(samples))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="2,4")
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=41)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.topk > args.num_experts:
        raise ValueError("topk cannot exceed num-experts")

    torch.manual_seed(20260718)
    device = torch.device("cuda")
    dtype = torch.float16
    w1 = torch.randn(
        args.num_experts,
        2 * args.intermediate_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    w2 = torch.randn(
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        device=device,
        dtype=dtype,
    )

    results: list[Result] = []
    for tokens in (int(value) for value in args.tokens.split(",") if value):
        hidden_states = torch.randn(
            tokens, args.hidden_size, device=device, dtype=dtype
        )
        topk_ids = (
            torch.arange(tokens * args.topk, device=device, dtype=torch.int64)
            .reshape(tokens, args.topk)
            .remainder(args.num_experts)
        )
        topk_weights = torch.full(
            (tokens, args.topk),
            1.0 / args.topk,
            device=device,
            dtype=torch.float32,
        )
        reference, baseline_us = _time(
            BASELINE,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            args.num_experts,
            args.warmup,
            args.repeats,
        )
        for candidate in CANDIDATES:
            try:
                output, median_us = _time(
                    candidate,
                    hidden_states,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    args.num_experts,
                    args.warmup,
                    args.repeats,
                )
                diff = (output.float() - reference.float()).abs()
                results.append(
                    Result(
                        tokens=tokens,
                        candidate=candidate.name,
                        config=candidate.config,
                        equal=bool(torch.equal(output, reference)),
                        max_diff=float(diff.max().item()),
                        median_us=median_us,
                        speedup_vs_baseline=baseline_us / median_us,
                    )
                )
            except Exception as error:  # Candidate compilation can be invalid.
                results.append(
                    Result(
                        tokens=tokens,
                        candidate=candidate.name,
                        config=candidate.config,
                        equal=False,
                        max_diff=float("inf"),
                        median_us=float("inf"),
                        speedup_vs_baseline=0.0,
                        error=str(error),
                    )
                )

    payload = {
        "device": torch.cuda.get_device_name(),
        "num_experts": args.num_experts,
        "topk": args.topk,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
