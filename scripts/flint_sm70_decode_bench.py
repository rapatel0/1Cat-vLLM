"""Split prefill and decode cost for the Flint SM70 service.

Reports time-to-first-token and steady decode rate separately, so a long
context does not look slow just because its prompt had to be processed.
"""

import argparse
import json
import time
import urllib.request


def timed_chat(base, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft = None
    tokens = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    tokens += 1
    total = time.perf_counter() - start
    return ttft or total, total, tokens


def wave(base, model, slots, prompt, max_tokens):
    # Same prompt for every slot so prefix caching is exercised the way a
    # coding-agent session with a shared preamble behaves.
    start = time.perf_counter()
    results = [timed_chat(base, model, prompt, max_tokens) for _ in range(slots)]
    wall = time.perf_counter() - start
    completed = sum(tokens for _, _, tokens in results)
    return {
        "slots": slots,
        "ttft_s": round(min(ttft for ttft, _, _ in results), 3),
        "decode_tok_s_per_stream": round(
            completed / max(sum(t - f for f, t, _ in results), 1e-6), 2
        ),
        "aggregate_tok_s": round(completed / wall, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18002")
    parser.add_argument("--model", default="flint")
    parser.add_argument("--slots", default="1,2,4,8")
    parser.add_argument("--context-words", default="0,2000,20000")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    for words in [int(value) for value in args.context_words.split(",")]:
        prompt = " ".join(f"word{index}" for index in range(words)) if words else "Ready."
        # Warm the prefix and the graphs once per context length.
        timed_chat(args.base, args.model, prompt, 8)
        for slots in [int(value) for value in args.slots.split(",")]:
            print(json.dumps({"prompt_words": words, **wave(
                args.base, args.model, slots, prompt, args.max_tokens
            )}))


if __name__ == "__main__":
    main()
