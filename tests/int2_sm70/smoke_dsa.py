#!/usr/bin/env python3
"""Tiny GLM-5.2 (glm_moe_dsa) end-to-end smoke on V100, dummy weights.

Unlike smoke_tp2.py (which approximates GLM-5.2 with the Glm4MoeForCausalLM /
GQA arch), this exercises the REAL GLM-5.2 architecture: GlmMoeDsaForCausalLM ->
deepseek_v2.py (MLA latent attention + DeepSeek Sparse Attention + MoE). It
builds a *scaled-down* config from the real GLM-5.2 config.json — same MLA/index
dims and model_type, fewer layers + pruned experts + small vocab — so it fits a
couple of V100s with random weights.

Purpose: prove the dense-MLA sm70 fallback path (the DSA indexer needs
DeepGEMM/sm90; on V100 we run full dense MLA — _dsa_runtime_enabled() returns
False, no indexer built, is_sparse=False). Gibberish output is expected (dummy
weights); we validate that the architecture instantiates, loads, and generates.

  CONFIG_JSON=/models/GLM-5.2-FP8/config.json TP=2 \
    /opt/venv/bin/python tests/int2_sm70/smoke_dsa.py
"""
import json
import os
import sys
import tempfile


def build_tiny_dsa_config(path: str) -> str:
    src = os.environ.get("CONFIG_JSON", "/models/GLM-5.2-FP8/config.json")
    with open(src) as f:
        cfg = json.load(f)

    L = int(os.environ.get("LAYERS", "4"))
    E = int(os.environ.get("EXPERTS", "8"))
    VOCAB = int(os.environ.get("VOCAB", "4096"))

    # Shrink the *count* dims; keep the real MLA / indexer / MoE shape dims so
    # the sm70 dense-MLA path is exercised faithfully.
    cfg["num_hidden_layers"] = L
    cfg["n_routed_experts"] = E
    cfg["num_experts_per_tok"] = min(E, int(os.environ.get("TOPK", "4")))
    cfg["first_k_dense_replace"] = int(os.environ.get("DENSE_REPLACE", "1"))
    cfg["vocab_size"] = VOCAB
    cfg["num_nextn_predict_layers"] = 0  # no MTP for the smoke
    # per-layer type arrays must match num_hidden_layers
    for key in ("indexer_types", "mlp_layer_types"):
        if key in cfg and isinstance(cfg[key], list):
            base = cfg[key]
            cfg[key] = [base[i] if i < len(base) else base[-1] for i in range(L)]
    # quantization is applied via vLLM --quantization, not the checkpoint
    cfg.pop("quantization_config", None)
    cfg["tie_word_embeddings"] = False
    cfg["architectures"] = ["GlmMoeDsaForCausalLM"]

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main() -> int:
    quant = os.environ.get("INT2_QUANT") or None
    tp = int(os.environ.get("TP", "2"))
    model_dir = tempfile.mkdtemp(prefix="tiny_glm_dsa_")
    build_tiny_dsa_config(model_dir)
    print(f"[smoke-dsa] tiny glm_moe_dsa config at {model_dir}  tp={tp}  quant={quant}")

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=model_dir,
        runner="generate",
        load_format="dummy",
        tensor_parallel_size=tp,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "512")),
        skip_tokenizer_init=True,
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

    out = llm.generate(prompts, sp)
    toks = sum(len(o.outputs[0].token_ids) for o in out)
    ok = all(len(o.outputs[0].token_ids) > 0 for o in out)
    print(f"[DSA] nseq={nseq} maxtok={maxtok} tp={tp} tokens={toks} "
          f"sample={list(out[0].outputs[0].token_ids)[:6]}")
    print("[DSA SMOKE OK]" if ok else "[DSA SMOKE FAIL]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
