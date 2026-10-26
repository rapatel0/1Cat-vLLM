"""Score Flint output quality across KV cache dtypes on long contexts.

Throughput alone cannot choose a KV format: an 8-bit payload with coarse scales
can still be the fastest cache while losing recall over a 20K-token context.
Each check has one expected answer, so a dtype comparison is a pass count rather
than a impressionistic read of the text.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def complete(base: str, model: str, messages: list, max_tokens: int) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode())
            return payload["choices"][0]["message"]["content"] or ""
        except (OSError, ValueError, KeyError):
            time.sleep(2)
    return ""


def fillers(count: int) -> str:
    return " ".join(
        f"The {name} sensor reported a nominal reading at index {index}."
        for index, name in enumerate(
            ["boiler", "pump", "valve", "filter", "compressor", "tank"] * (count // 6 + 1)
        )
    )


def checks() -> list:
    """Each entry: name, messages, expected substrings, max_tokens."""
    mid = fillers(1400)
    long_prompt = (
        "Maintenance log. "
        + mid
        + " The maintenance code for pump 7 is QR-4417. "
        + mid
        + " Answer with the code only."
    )
    hop_a = fillers(400)
    hop_prompt = (
        "Notes. "
        + hop_a
        + " Ada owns the billing service. "
        + hop_a
        + " The billing service restarts at 04:30 UTC. "
        + hop_a
        + " Who owns the service that restarts at 04:30 UTC, and at what time? "
        "Answer in one short sentence."
    )
    return [
        (
            "long_context_retrieval",
            [{"role": "user", "content": long_prompt}],
            ["QR-4417"],
            48,
        ),
        (
            "two_hop_ownership",
            [{"role": "user", "content": hop_prompt}],
            ["Ada", "04:30"],
            64,
        ),
        (
            "instruction_following",
            [
                {
                    "role": "system",
                    "content": "Reply with exactly three comma separated numbers.",
                },
                {
                    "role": "user",
                    "content": "List the even numbers from 1 to 6.",
                },
            ],
            ["2, 4, 6"],
            32,
        ),
    ]


def tool_check(base: str, model: str) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "max_tokens": 64,
        "temperature": 0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    payload: dict = {}
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read().decode())
            break
        except (OSError, ValueError):
            time.sleep(2)
    elapsed = time.perf_counter() - started
    choices = payload.get("choices") or [{}]
    call = choices[0].get("message", {}).get("tool_calls") or []
    arguments = (call[0].get("function", {}).get("arguments") or "") if call else ""
    return {
        "check": "tool_call",
        "passed": bool(call) and "Paris" in arguments,
        "detail": arguments[:120],
        "wall_s": round(elapsed, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18002")
    parser.add_argument("--model", default="flint")
    parser.add_argument("--kv-dtype", default="unknown")
    args = parser.parse_args()

    results = []
    for name, messages, expected, max_tokens in checks():
        started = time.perf_counter()
        text = complete(args.base, args.model, messages, max_tokens)
        elapsed = time.perf_counter() - started
        missing = [token for token in expected if token not in text]
        results.append(
            {
                "check": name,
                "passed": not missing,
                "missing": missing,
                "detail": text[:160].replace("\n", " ").strip(),
                "wall_s": round(elapsed, 3),
            }
        )
        print(json.dumps(results[-1]), flush=True)

    results.append(tool_check(args.base, args.model))
    print(json.dumps(results[-1]), flush=True)

    passed = sum(1 for row in results if row["passed"])
    print(
        json.dumps(
            {
                "kv_dtype": args.kv_dtype,
                "passed": passed,
                "total": len(results),
                "rows": results,
            }
        )
    )


if __name__ == "__main__":
    main()
