#!/usr/bin/env python3
"""Three-way MTP bench: BF16 vs runtime-amax FP8 vs serialized ModelOpt-parity FP8."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

PROMPTS = {
    "coding": "Write a Python function that merges two sorted lists. Return only code.",
    "factual": "Name the capital of France and the year of the French Revolution.",
    "reasoning": "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
    "tool_use": "Call a weather API for Tokyo, then summarize the forecast in one sentence.",
}


def chat(url: str, prompt: str, max_tokens: int, extra: dict | None = None) -> dict:
    body = {
        "model": "qwen38",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if extra:
        body.update(extra)
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=1800)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage") or {}
    text = data["choices"][0]["message"]["content"]
    spec = data.get("spec_decode") or data.get("metrics") or {}
    out_tok = usage.get("completion_tokens") or 0
    return {
        "elapsed_s": elapsed,
        "out_tokens": out_tok,
        "tok_s": (out_tok / elapsed) if elapsed else 0,
        "text": text[:200],
        "usage": usage,
        "spec": spec,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()
    for _ in range(args.warmup):
        chat(args.url, "Say hi.", 8)
    rows = []
    for name, prompt in PROMPTS.items():
        row = chat(args.url, prompt, args.max_tokens)
        row["prompt"] = name
        rows.append(row)
        print(args.label, name, f"{row['tok_s']:.2f} tok/s", row["out_tokens"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"label": args.label, "url": args.url, "rows": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
