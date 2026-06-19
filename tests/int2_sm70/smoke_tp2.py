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

    cfg = Glm4MoeConfig(
        vocab_size=2048,
        hidden_size=512,
        intermediate_size=1024,      # dense FFN (first_k_dense_replace layers)
        moe_intermediate_size=512,   # per-expert FFN
        num_hidden_layers=4,
        first_k_dense_replace=1,     # layer 0 dense, 1..3 MoE
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        num_attention_heads=8,
        num_key_value_heads=2,       # GQA
        head_dim=64,
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
    out = llm.generate(
        {"prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8]},
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    toks = out[0].outputs[0].token_ids
    ok = len(toks) > 0 and all(isinstance(t, int) for t in toks)
    print(f"[smoke] generated token_ids: {list(toks)}")
    print("[SMOKE OK]" if ok else "[SMOKE FAIL]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
