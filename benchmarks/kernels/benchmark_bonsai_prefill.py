# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure uncached fixed-token Bonsai prefill through a vLLM HTTP server."""

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def request_json(
    base_url: str, path: str, payload: dict[str, object], timeout_s: int
) -> dict[str, object]:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def build_prompt(prompt_tokens: int) -> str:
    return ("decode " * prompt_tokens).rstrip()


def confirm_prompt_tokens(args: argparse.Namespace, prompt: str) -> None:
    tokenized = request_json(
        args.base_url,
        "/tokenize",
        {"model": args.model, "prompt": prompt},
        args.timeout_s,
    )
    count = tokenized.get("count")
    if count != args.prompt_tokens:
        raise RuntimeError(
            f"Expected {args.prompt_tokens} prompt tokens, tokenizer returned {count}"
        )


def run_once(args: argparse.Namespace, prompt: str) -> dict[str, object]:
    start = time.perf_counter()
    response = request_json(
        args.base_url,
        "/v1/completions",
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "ignore_eos": True,
            "stream": False,
        },
        args.timeout_s,
    )
    elapsed_s = time.perf_counter() - start
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"Server did not return usage: {response}")
    if usage.get("prompt_tokens") != args.prompt_tokens:
        raise RuntimeError(f"Wrong prompt usage: {response}")
    if usage.get("completion_tokens") != 1:
        raise RuntimeError(f"Wrong completion usage: {response}")
    return {
        "elapsed_s": elapsed_s,
        "prefill_tokens_per_s": args.prompt_tokens / elapsed_s,
    }


def main() -> None:
    args = parse_args()
    if args.prompt_tokens < 1:
        raise ValueError("--prompt-tokens must be positive")
    prompt = build_prompt(args.prompt_tokens)
    confirm_prompt_tokens(args, prompt)
    # Establish model and allocator state outside the reported samples.
    run_once(args, prompt)
    iterations = [run_once(args, prompt) for _ in range(args.iterations)]
    elapsed_s = [float(iteration["elapsed_s"]) for iteration in iterations]
    result = {
        "prompt_tokens": args.prompt_tokens,
        "iterations": iterations,
        "median_elapsed_s": statistics.median(elapsed_s),
        "median_prefill_tokens_per_s": statistics.median(
            float(iteration["prefill_tokens_per_s"]) for iteration in iterations
        ),
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.json_out is not None:
        args.json_out.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
