# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure steady four-slot decode throughput through a vLLM HTTP server."""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-words", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def request_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "ignore_eos": True,
            "stream": False,
        }
    ).encode()
    request = Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def build_prompt(slot: int, prompt_words: int) -> str:
    context = "decode " * prompt_words
    return (
        f"Slot {slot}: {context}\n"
        "Complete this sentence with a short factual answer: The V100 is a"
    )


def completion_tokens(response: dict[str, object]) -> int:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"Server did not return usage: {response}")
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens < 1:
        raise RuntimeError(f"Invalid completion token count: {response}")
    return tokens


def run_group(args: argparse.Namespace) -> dict[str, object]:
    prompts = [build_prompt(slot, args.prompt_words) for slot in range(4)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(
                lambda prompt: request_completion(
                    args.base_url,
                    args.model,
                    prompt,
                    args.max_tokens,
                ),
                prompts,
            )
        )
    elapsed_s = time.perf_counter() - start
    tokens = sum(completion_tokens(response) for response in responses)
    expected_tokens = len(prompts) * args.max_tokens
    if tokens != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} completion tokens with ignore_eos, got "
            f"{tokens}: {responses}"
        )
    return {
        "elapsed_s": elapsed_s,
        "completion_tokens": tokens,
        "aggregate_tokens_per_s": tokens / elapsed_s,
        "samples": [
            response["choices"][0]["text"]
            for response in responses
            if isinstance(response.get("choices"), list) and response["choices"]
        ],
    }


def main() -> None:
    args = parse_args()
    # Compile/JIT and cache allocation do not represent steady decode.
    run_group(args)
    iterations = [run_group(args) for _ in range(args.iterations)]
    throughputs = [
        float(iteration["aggregate_tokens_per_s"]) for iteration in iterations
    ]
    result = {
        "slots": 4,
        "max_tokens": args.max_tokens,
        "prompt_words": args.prompt_words,
        "iterations": iterations,
        "median_aggregate_tokens_per_s": statistics.median(throughputs),
    }
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.json_out is not None:
        args.json_out.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
