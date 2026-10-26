#!/usr/bin/env python3
"""Benchmark Flint decode throughput against a running OpenAI-compatible server.

Records single-stream tokens/s, aggregate tokens/s for 2-8 concurrent slots,
throughput per GPU, and the longest context that still fits at a given slot
count. Greedy decoding rejects ``n>1``, so concurrency is driven by one
request per slot in its own thread.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def complete(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            payload = json.loads(response.read().decode())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"request to {base} failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    usage = payload.get("usage", {})
    return {
        "elapsed_s": elapsed,
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def run_slots(base: str, model: str, prompt: str, max_tokens: int, slots: int) -> dict:
    with ThreadPoolExecutor(max_workers=slots) as pool:
        started = time.perf_counter()
        results = list(
            pool.map(lambda _: complete(base, model, prompt, max_tokens), range(slots))
        )
        elapsed = time.perf_counter() - started
    tokens = sum(row["completion_tokens"] for row in results)
    slowest = max(row["elapsed_s"] for row in results)
    return {
        "slots": slots,
        "wall_s": round(elapsed, 3),
        "completion_tokens": tokens,
        "aggregate_tok_s": round(tokens / elapsed, 2) if elapsed else 0.0,
        "slowest_slot_tok_s": round(
            max(r["completion_tokens"] for r in results) / slowest, 2
        )
        if slowest
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="flint")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--slots", default="1,2,4,8")
    parser.add_argument(
        "--context-words",
        type=int,
        default=0,
        help="Repeat this many filler words into the prompt to grow the context.",
    )
    parser.add_argument("--gpus", type=int, default=1, help="GPUs in the layout.")
    args = parser.parse_args()

    prompt = "Write a short Python function that adds two integers.\n"
    if args.context_words:
        filler = "The quick brown fox jumps over the lazy dog. "
        prompt = filler * args.context_words + "\n" + prompt

    complete(args.base, args.model, prompt, 8)  # warmup, JIT and cache warm

    rows = []
    for slot in [int(item) for item in args.slots.split(",") if item]:
        row = run_slots(args.base, args.model, prompt, args.max_tokens, slot)
        row["tok_s_per_gpu"] = round(row["aggregate_tok_s"] / args.gpus, 2)
        row["prompt_words"] = args.context_words
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
