#!/usr/bin/env python3
"""Tiny GLM4-MoE end-to-end smoke on 2x V100 (TP=2), dummy weights.

The dev-loop target for the int2 SM70 integration (docs/INT2_SM70_INTEGRATION.md
Rung 2): exercises the *whole* runtime path — config -> glm4_moe.py -> router ->
FusedMoE -> attention -> sampling — on a toy model that fits 2 GPUs, with random
weights. Proves the scaffolding without the real GLM-5.2 or 8 GPUs.

Phases:
  INT2_QUANT unset    -> fp16 baseline (proves env + harness)
  INT2_QUANT=int2_sm70 -> our quant method (once P1/P2 land)

Run on the dev pod:
  CUDA_VISIBLE_DEVICES=4,7 TP=2 /opt/venv/bin/python tests/int2_sm70/smoke_tp2.py
"""
import os
import sys
import tempfile


def build_tiny_config(path: str) -> str:
    from transformers import Glm4MoeConfig

    E = int(os.environ.get("EXPERTS", "8"))
    H = int(os.environ.get("HIDDEN", "512"))
    L = int(os.environ.get("LAYERS", "4"))
    NH = int(os.environ.get("HEADS", "8"))
    KV = int(os.environ.get("KV_HEADS", "2"))
    cfg = Glm4MoeConfig(
        vocab_size=int(os.environ.get("VOCAB", "2048")),
        hidden_size=H,
        intermediate_size=int(os.environ.get("DENSE_INT", str(H * 2))),
        moe_intermediate_size=int(os.environ.get("MOE_INT", str(H))),
        num_hidden_layers=L,
        first_k_dense_replace=int(os.environ.get("DENSE_REPLACE", "1")),
        n_routed_experts=E,
        num_experts_per_tok=min(E, int(os.environ.get("TOPK", "2"))),
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_attention_heads=NH,
        num_key_value_heads=KV,      # GQA
        head_dim=H // NH,
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        num_nextn_predict_layers=0,  # no MTP for the smoke
        tie_word_embeddings=False,
    )
    cfg.architectures = ["Glm4MoeForCausalLM"]   # else vLLM picks the embed runner
    cfg.save_pretrained(path)
    return path


def main() -> int:
    quant = os.environ.get("INT2_QUANT") or None
    tp = int(os.environ.get("TP", "2"))
    model_dir = tempfile.mkdtemp(prefix="tiny_glm_moe_")
    build_tiny_config(model_dir)
    print(f"[smoke] tiny GLM4-MoE config at {model_dir}  tp={tp}  quant={quant}")

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model_dir,
        runner="generate",              # else toy config auto-detects as embed
        load_format="dummy",            # random weights, no checkpoint
        tensor_parallel_size=tp,
        enforce_eager=True,             # skip CUDA-graph capture for the smoke
        gpu_memory_utilization=0.5,
        max_model_len=512,
        skip_tokenizer_init=True,       # toy config has no tokenizer
        trust_remote_code=True,
    )
    if quant:
        kwargs["quantization"] = quant

    llm = LLM(**kwargs)
    nseq = int(os.environ.get("NUM_SEQS", "1"))
    maxtok = int(os.environ.get("MAX_TOKENS", "8"))
    prompts = [{"prompt_token_ids": [1 + (i % 7), 2, 3, 4, 5, 6, 7, 8]}
               for i in range(nseq)]
    sp = SamplingParams(max_tokens=maxtok, temperature=0.0, ignore_eos=True)

    import time
    # warmup (compile/capture), then timed
    llm.generate(prompts, SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True))
    t0 = time.perf_counter()
    out = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    ok = all(len(o.outputs[0].token_ids) > 0 for o in out)
    tag = os.environ.get("BENCH_TAG", "")
    print(f"[BENCH]{tag} nseq={nseq} maxtok={maxtok} tp={tp} "
          f"tokens={toks} time={dt:.3f}s tok/s={toks/dt:.1f} "
          f"sample={list(out[0].outputs[0].token_ids)[:4]}")
    print("[SMOKE OK]" if ok else "[SMOKE FAIL]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
