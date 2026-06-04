# Plan: Gemma-4-31B Dense (AWQ) on 1Cat-vLLM / V100

**Goal:** serve **Gemma-4-31B-it Dense** on the homelab V100s via this fork (1Cat-vLLM), 4-bit
**AWQ** so it uses the SM70 TurboMind `awq_sm70` kernels. "Q4" = 4-bit AWQ (TurboMind-accelerated);
"Q5"-equivalent = the 6-bit AWQ checkpoint (quality option, no uint4 kernel). Dense only — **no MTP /
coupled assistant** in scope.

**Branch/PR hygiene:** this work lives on **`gemma4-31b-awq-v100`, forked from `main`** (NOT
`rotorquant-mtp`) so the PR is clean of the MTP/rotorquant work. The fork's own base features (Qwen
MTP, V100 TurboMind kernels) are part of `main` and fine to build on.

## Feasibility summary (researched 2026-06-04)

**De-risked / positive:**
- **AWQ checkpoints already exist — no minting needed.** `QuantTrio/gemma-4-31B-it-AWQ` (4-bit,
  data-free), `QuantTrio/gemma-4-31B-it-AWQ-6Bit`, `cyankiwi/gemma-4-31B-it-AWQ-4bit`,
  `nvidia/Gemma-4-31B-IT-NVFP4`. QuantTrio's AWQ format already A/B-tested on this fork (the Qwen
  `quanttrio6` variant) → known-compatible with the sm70 path.
- **vLLM models Gemma-4 upstream** (vLLM blog 2026-04-02; `gemma4`/`gemma4_mm` in docs). → a **port**
  into the fork, not a write-from-scratch. People run Gemma-4-31B INT4 on vLLM already (vLLM #39133).
- **The 50/60 sliding layers (head_dim 256) are supported** on V100 (flash-v100 head_dim switch has a
  256 case; `gemma3.py` already implements the ISWA machinery: `layer_types`, `is_sliding`,
  per-layer `sliding_window`, `query_pre_attn_scalar`).

**Gating risks (the real work):**
1. **head_dim 512 (global layers) NOT supported on V100.** Gemma-4-31B's 10 global layers use
   **head_dim 512** (4 KV heads, "unified K/V", Proportional RoPE). `flash_decode_paged.cu`'s head_dim
   switch is `{64,80,96,112,128,256, default→"Unsupported"}` — **512 errors out.** Triton fallbacks
   also generally cap ≤256. → must extend the V100 attention to HDIM 512, or find a path that runs it.
2. **transformers 4.57.6 → 5.5.0.** Gemma-4 (`model_type: gemma4`) needs **transformers ≥5.5.0**; the
   fork ships **4.57.6** (major-version jump, breaking changes). Risk of destabilizing other models /
   the vLLM↔transformers integration. Open Q: does upstream vLLM's `gemma4` bundle its own config to
   avoid the hard 5.5 requirement?

**Secondary risks:** Gemma-4 features absent from `gemma3.py` (variable head_dim per layer-type,
unified-K/V global layers, Proportional RoPE, possible PLE/KV-sharing — `gemma3n.py` has PLE to
borrow); reconciling upstream `gemma4.py` against the **diverged fork base**; sm70 AWQ kernel fit for
the unusual global-layer proj shapes.

## Architecture facts (Gemma-4-31B dense)
60 layers, 5 local : 1 global (50 sliding + 10 global, final layer global). Sliding: head_dim 256,
16 KV heads, sliding_window 1024. Global: head_dim **512**, 4 KV heads, unified K/V, Proportional
RoPE. max_position_embeddings 262144. Reasoning model (output → `reasoning_content` first; clients
need `max_tokens ≥ 256`). vocab/hidden/intermediate: TBD from config (Phase 0).

## Plan — phased, de-risk first

### Phase 0 — de-risk spikes (DO FIRST; either failure reshapes/blocks the effort)
- **0a. head_dim 512 on V100.** Instantiate a HDIM=512 path of the flash-v100 decode kernel (and/or
  check the Triton base) and confirm it **compiles within Volta's register/smem budget** and runs a
  correctness round-trip — mirror the TurboMind-256 spike (which passed at 157 regs; 512 is 2× → the
  open question). If neither flash-v100 nor Triton can do 512 on V100, the dense model can't run here
  as-is → stop / rethink (e.g. CPU-offload the 10 global layers, or a custom kernel).
- **0b. transformers 5.5 blast radius.** In a throwaway env, bump `transformers==5.5.0` against the
  fork and inventory what breaks (imports, config classes, the model registry, tokenizer). Decide:
  contained bump vs. bundle-the-gemma4-config-locally vs. rebase-the-fork.
- **0c. Pull the real config.** Get `QuantTrio/gemma-4-31B-it-AWQ` `config.json` + weight index:
  confirm head_dim per layer-type, AWQ group_size (expect 128), `modules_to_not_convert`, whether the
  dense 31B uses PLE, and the exact `architectures`/`model_type`.

### Phase 1 — model port
- Bring `gemma4.py` (+ any `gemma4` config) from upstream vLLM into the fork; register
  `Gemma4ForCausalLM`. Wire variable head_dim per layer-type, unified-K/V global layers, Proportional
  RoPE. Reuse `gemma3.py`/`gemma3n.py` patterns (ISWA, PLE) where they match.
- Resolve the transformers dependency per Phase 0b.

### Phase 2 — AWQ + sm70 wiring
- Load `QuantTrio/gemma-4-31B-it-AWQ` (4-bit); confirm the `awq_sm70` (TurboMind uint4) kernel binds
  to Gemma's q/k/v/o + MLP projections (group_size 128; the gemm is shape-agnostic). Keep the 6-bit
  checkpoint as the quality variant (non-uint4 path).

### Phase 3 — head_dim 512 attention (depends on Phase 0a)
- Implement the chosen 512 path (extend flash-v100 HDIM 512, or route global layers to a working
  fallback). The 10 global layers are the minority but mandatory + the final layer.

### Phase 4 — validate
- Coherence (raw + chat, reasoning_content-first, `max_tokens ≥ 256`). PPL vs the GGUF/fp16 reference
  via the offline harness (reuse the rotorquant PPL harness pattern). Confirm sliding+global layers
  both produce correct attention on V100.

### Phase 5 — deploy + benchmark
- TP4 on the 4 free **NUMA0** V100s; manifest mirrors `14-deployment-vision-nospec.yaml`
  (auto-NUMA-pin, compile-cache hostPath, reasoning parser, `VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100`),
  served as `gemma4-31b`. Benchmark vs the 4090 baseline (25.7 tok/s) and check the VRAM headroom
  (128 GB across 4 cards → big context / the 6-bit variant fit comfortably).

## Open questions to resolve in Phase 0
- Does the V100 flash/Triton attention run head_dim 512 at all? (0a — the make-or-break)
- Is transformers 5.5 a contained bump or a fork rebase? (0b)
- Does dense 31B use PLE / unified-KV in a way `gemma3n.py` already models? (0c)
- Does upstream `gemma4.py` need transformers 5.5 at runtime, or just for config (bundle-able)?

## Sources
- vLLM Gemma-4: https://vllm.ai/blog/2026-04-02-gemma4 ; https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html ; transformers-version issue https://github.com/vllm-project/vllm/issues/38868 ; INT4 run https://github.com/vllm-project/vllm/issues/39133
- AWQ checkpoints: https://huggingface.co/QuantTrio/gemma-4-31B-it-AWQ ; https://huggingface.co/QuantTrio/gemma-4-31B-it-AWQ-6Bit ; https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-4bit
- Architecture: https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture ; https://huggingface.co/docs/transformers/model_doc/gemma4
