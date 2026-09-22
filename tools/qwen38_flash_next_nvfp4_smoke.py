#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Qwen3.8-Flash-Next NVFP4 smoke test for 4x V100.

Uses the native SM70 TurboMind NVFP4 MoE path. Packed E2M1 expert weights stay
packed. The kernel unpacks tiles to FP16 for HMMA. The script does not
materialize a persistent FP16 expert copy and does not use MoE emulation.
"""

from __future__ import annotations

import argparse
import os
import time


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_PLE_CPU_OFFLOAD", "1")
    os.environ.setdefault("VLLM_SM70_NVFP4_TURBOMIND", "1")
    os.environ.setdefault("FLASH_ATTN_V100", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)

    from vllm import LLM, SamplingParams

    log("constructing TP4 engine")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="half",
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        kv_cache_memory_bytes=256 << 20,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
    )
    log(f"engine ready; generating {args.max_tokens} token(s)")
    outputs = llm.generate(
        [args.prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens)
    )
    text = outputs[0].outputs[0].text
    token_ids = outputs[0].outputs[0].token_ids
    log(f"SMOKE_OK token_ids={token_ids!r} text={text!r}")


if __name__ == "__main__":
    main()
