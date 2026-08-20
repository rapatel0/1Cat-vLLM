#!/usr/bin/env python3
"""Final correctness and throughput gate for Qwen3.8 TP8+DCP2+MTP4.

The harness talks directly to the vLLM service. It never uses a response-cache
router, gives every measured request a unique prompt prefix, and calculates
throughput from the server-reported ``usage.completion_tokens`` rather than SSE
chunk counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def tokenize_chat(base_url: str, model: str, content: str) -> list[int]:
    result = post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return result["tokens"]


def render_fixed_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    nonce: str,
) -> list[int]:
    unit = "Compiler scheduling and distributed systems benchmark context. "
    suffix = "\nContinue with detailed technical prose and do not stop early."

    def render(repetitions: int) -> list[int]:
        return tokenize_chat(
            base_url,
            model,
            f"Unique request {nonce}. " + unit * repetitions + suffix,
        )

    low, high = 0, max(2, target_tokens)
    while low + 1 < high:
        middle = (low + high) // 2
        if len(render(middle)) <= target_tokens:
            low = middle
        else:
            high = middle
    return render(low)


def render_needle_prompt(
    base_url: str,
    model: str,
    target_tokens: int,
    secret: str,
) -> tuple[list[dict[str, str]], int]:
    unit = (
        "Historical ledger filler: copper amber cedar quartz delta seven. "
        "This sentence is irrelevant except for extending the archive. "
    )

    def content(repetitions: int) -> str:
        split = max(1, repetitions // 8)
        return (
            "Read the entire archive and retain the secret code.\n"
            + unit * split
            + f"\nThe secret code is {secret}.\n"
            + unit * (repetitions - split)
            + "\nReply with the secret code only, with no punctuation or explanation."
        )

    def token_count(repetitions: int) -> int:
        return len(tokenize_chat(base_url, model, content(repetitions)))

    low, high = 1, max(2, target_tokens)
    while low + 1 < high:
        middle = (low + high) // 2
        if token_count(middle) <= target_tokens:
            low = middle
        else:
            high = middle
    final_content = content(low)
    final_tokens = len(tokenize_chat(base_url, model, final_content))
    return [{"role": "user", "content": final_content}], final_tokens


@dataclass
class RequestResult:
    elapsed_s: float
    completion_tokens: int
    prompt_tokens: int
    tok_s: float
    finish_reason: str | None
    response_sha256: str


def run_completion(
    base_url: str,
    model: str,
    prompt_tokens: list[int],
    output_tokens: int,
) -> RequestResult:
    started = time.perf_counter()
    result = post_json(
        base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": prompt_tokens,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        },
    )
    elapsed = time.perf_counter() - started
    usage = result["usage"]
    completion_tokens = int(usage["completion_tokens"])
    text = result["choices"][0].get("text", "")
    return RequestResult(
        elapsed_s=elapsed,
        completion_tokens=completion_tokens,
        prompt_tokens=int(usage["prompt_tokens"]),
        tok_s=completion_tokens / elapsed,
        finish_reason=result["choices"][0].get("finish_reason"),
        response_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def run_needle_gate(
    base_url: str,
    model: str,
    target_tokens: int,
) -> dict[str, Any]:
    secret = f"DCP2_MTP4_128K_{uuid.uuid4().hex[:10].upper()}"
    messages, rendered_tokens = render_needle_prompt(
        base_url, model, target_tokens, secret
    )
    started = time.perf_counter()
    result = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    elapsed = time.perf_counter() - started
    choice = result["choices"][0]
    actual = (choice["message"].get("content") or "").strip()
    passed = actual == secret
    return {
        "passed": passed,
        "secret": secret,
        "actual": actual,
        "rendered_prompt_tokens": rendered_tokens,
        "server_prompt_tokens": int(result["usage"]["prompt_tokens"]),
        "completion_tokens": int(result["usage"]["completion_tokens"]),
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
    }


def run_single(
    base_url: str,
    model: str,
    prompt_tokens: int,
    output_tokens: int,
    runs: int,
) -> dict[str, Any]:
    # A distinct warmup pays any first-request/JIT cost without sharing the
    # measured request's content prefix.
    warmup_prompt = render_fixed_prompt(
        base_url, model, prompt_tokens, f"warmup-{uuid.uuid4().hex}"
    )
    warmup = run_completion(base_url, model, warmup_prompt, 32)

    results = []
    for run_index in range(runs):
        prompt = render_fixed_prompt(
            base_url,
            model,
            prompt_tokens,
            f"single-{run_index}-{uuid.uuid4().hex}",
        )
        results.append(run_completion(base_url, model, prompt, output_tokens))
    rates = [result.tok_s for result in results]
    return {
        "settings": {
            "concurrency": 1,
            "target_prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "runs": runs,
            "temperature": 0.0,
            "ignore_eos": True,
            "unique_prompts": True,
        },
        "warmup": asdict(warmup),
        "runs": [asdict(result) for result in results],
        "median_tok_s": statistics.median(rates),
        "mean_tok_s": statistics.mean(rates),
        "min_tok_s": min(rates),
        "max_tok_s": max(rates),
    }


def run_aggregate(
    base_url: str,
    model: str,
    concurrency: int,
    prompt_tokens: int,
    output_tokens: int,
    runs: int,
) -> dict[str, Any]:
    cohorts = []
    for cohort_index in range(runs):
        prompts = [
            render_fixed_prompt(
                base_url,
                model,
                prompt_tokens,
                f"c{concurrency}-{cohort_index}-{i}-{uuid.uuid4().hex}",
            )
            for i in range(concurrency)
        ]
        started = time.perf_counter()
        results: list[RequestResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    run_completion,
                    base_url,
                    model,
                    prompt,
                    output_tokens,
                )
                for prompt in prompts
            ]
            for future in as_completed(futures):
                results.append(future.result())
        wall_s = time.perf_counter() - started
        total_tokens = sum(result.completion_tokens for result in results)
        cohorts.append(
            {
                "wall_s": wall_s,
                "completion_tokens": total_tokens,
                "aggregate_tok_s": total_tokens / wall_s,
                "request_tok_s_mean": statistics.mean(
                    result.tok_s for result in results
                ),
                "request_elapsed_s_min": min(result.elapsed_s for result in results),
                "request_elapsed_s_max": max(result.elapsed_s for result in results),
                "requests": [asdict(result) for result in results],
            }
        )
    rates = [cohort["aggregate_tok_s"] for cohort in cohorts]
    return {
        "settings": {
            "concurrency": concurrency,
            "target_prompt_tokens": prompt_tokens,
            "output_tokens_per_request": output_tokens,
            "runs": runs,
            "temperature": 0.0,
            "ignore_eos": True,
            "unique_prompts": True,
        },
        "cohorts": cohorts,
        "median_aggregate_tok_s": statistics.median(rates),
        "mean_aggregate_tok_s": statistics.mean(rates),
        "min_aggregate_tok_s": min(rates),
        "max_aggregate_tok_s": max(rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen38-27b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--needle-target-tokens", type=int, default=127_500)
    parser.add_argument("--single-prompt-tokens", type=int, default=1_024)
    parser.add_argument("--single-output-tokens", type=int, default=512)
    parser.add_argument("--single-runs", type=int, default=3)
    parser.add_argument("--aggregate-concurrency", type=int, default=32)
    parser.add_argument("--aggregate-prompt-tokens", type=int, default=256)
    parser.add_argument("--aggregate-output-tokens", type=int, default=256)
    parser.add_argument("--aggregate-runs", type=int, default=3)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "methodology": {
            "direct_service": True,
            "streaming": False,
            "token_count_source": "server usage.completion_tokens",
            "response_cache": False,
            "unique_prompts": True,
        },
    }
    try:
        report["needle_128k"] = run_needle_gate(
            args.base_url, args.model, args.needle_target_tokens
        )
        if not report["needle_128k"]["passed"]:
            raise RuntimeError("128K exact needle gate failed")
        report["single_stream"] = run_single(
            args.base_url,
            args.model,
            args.single_prompt_tokens,
            args.single_output_tokens,
            args.single_runs,
        )
        report["aggregate"] = run_aggregate(
            args.base_url,
            args.model,
            args.aggregate_concurrency,
            args.aggregate_prompt_tokens,
            args.aggregate_output_tokens,
            args.aggregate_runs,
        )
        report["passed"] = True
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
