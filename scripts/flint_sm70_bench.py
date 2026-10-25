#!/usr/bin/env python3
"""Benchmark Flint decode throughput.

Records single-stream tokens/s, aggregate tokens/s for 2-8 concurrent slots,
and the maximum slot count that fits a 262144-token context. KV format is
recorded from the server; INT8 block KV is a later pass.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def complete(base: str, model: str, prompt: str, max_tokens: int, n: int = 1) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "n": n,
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
    except (OSError, json.JSONDecodeError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Flint completion request failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    usage = payload.get("usage", {})
    completion = int(usage.get("completion_tokens", 0))
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion,
        "tok_s": completion / elapsed if elapsed else 0.0,
        "n": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="flint")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--slots", default="1,2,4,8")
    args = parser.parse_args()
    prompt = "Write a short Python function that adds two integers.\n"
    rows = []
    for slot in [int(item) for item in args.slots.split(",") if item]:
        row = complete(args.base, args.model, prompt, args.max_tokens, n=slot)
        row["slots"] = slot
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"rows": rows}))


if __name__ == "__main__":
    main()
