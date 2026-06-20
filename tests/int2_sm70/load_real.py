#!/usr/bin/env python3
"""Load a real converted GLM-5.2 (packed-2bit) checkpoint and generate.

Validates the real-weight path: the int2_sm70 packed-MoE weight loader +
mixed-precision (fp16 attn/embed, 2-bit experts) on an actual converted
checkpoint. A truncated (few-layer) checkpoint won't be coherent, but proves
loading + the full forward on real weights.

  MODEL_DIR=/workspace/tmp/glm52-int2-test TP=2 \
    INT2_QUANT=int2_sm70 INT2_PACKED_MOE=1 python tests/int2_sm70/load_real.py
"""
import os
import sys


def main() -> int:
    model = os.environ["MODEL_DIR"]
    tp = int(os.environ.get("TP", "2"))
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        runner="generate",
        tensor_parallel_size=tp,
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("GMU", "0.85")),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "512")),
        skip_tokenizer_init=True,
        trust_remote_code=True,
        quantization=os.environ.get("INT2_QUANT", "int2_sm70"),
        cpu_offload_gb=float(os.environ.get("CPU_OFFLOAD_GB", "0")),
        max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "1")),
        max_num_batched_tokens=int(os.environ.get("MAX_BATCHED_TOKENS", "2048")),
    )
    nseq = int(os.environ.get("NUM_SEQS", "1"))
    maxtok = int(os.environ.get("MAX_TOKENS", "8"))
    prompts = [{"prompt_token_ids": [1 + (i % 7), 2, 3, 4, 5, 6, 7, 8]}
               for i in range(nseq)]
    sp = SamplingParams(max_tokens=maxtok, temperature=0.0, ignore_eos=True)
    out = llm.generate(prompts, sp)
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    ok = all(len(o.outputs[0].token_ids) > 0 for o in out)
    print(f"[REAL-LOAD] tp={tp} tokens={toks} "
          f"sample={list(out[0].outputs[0].token_ids)[:6]}")
    print("[REAL-LOAD OK]" if ok else "[REAL-LOAD FAIL]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
