# GLM-5.2 on 8×V100 — final summary

Goal: serve **GLM-5.2** (753B MoE, `glm_moe_dsa` = MLA + DeepSeek Sparse Attention
+ MoE) on 8× Tesla V100-SXM2-32GB. Two paths were built; the **llama.cpp + Unsloth
GGUF** path is the one that produced a *usable* model. The custom **vLLM + int2**
path is a validated-but-incomplete kernel bet.

---

## TL;DR — what's deployed and works

**GLM-5.2 via llama.cpp + Unsloth `UD-IQ1_S` (dynamic ~1.8-bit GGUF), 8×V100:**

| property | value |
|---|---|
| quality | ~76% (Unsloth single-choice benchmark) — generates correct reasoning |
| decode | **16.1 tok/s** single-stream |
| concurrency | 2 slots → **24.8 tok/s aggregate** (12.6 ea), each at **272K context** |
| max context | **384K** single-slot (f16 KV) · 272K×2 (q8_0 KV) · ~768K single-slot ceiling at q8_0 |
| prefill | ~80 tok/s (ubatch 128) |
| weights | ~210 GB across 8 GPUs (~26 GB/GPU, balanced) |
| engine | llama.cpp (sm70 build), `--split-mode layer` (pipeline parallel) |

Manifest: `deploy/glm52-gguf-llamacpp.yaml`. GGUF on PVC at `/models/GLM-5.2-GGUF/UD-IQ1_S`.

---

## Path A — llama.cpp + Unsloth GGUF (the pragmatic winner)

**Why it won:** off-the-shelf 76% quality, mature optimized i-quant MoE kernels,
runs GLM-5.2 as **dense MLA** (DSA indexer loaded-but-ignored, llama.cpp PR
#19460 — same dense-attention situation as our vLLM path, no DSA-kernel wall).

### Key findings / tuning
- **Unsloth "1-bit" UD-IQ1_S is really ~2.3-bit avg** (217 GB) — dynamic
  allocation keeps salient layers (attn/router/output) at 4-6 bit, which is
  *why* it holds 76%. Same footprint as our uniform 2-bit, but *measured* quality.
- **Image pull:** ghcr.io llama.cpp image won't pull (blob CDN IPv6-unreachable);
  **build from docker.io `nvidia/cuda` devel for sm70**, cached on /localpool.
- **Tokenizer:** GLM-5.2 ships a transformers-5.x tokenizer (`TokenizersBackend`);
  rewrite config to `PreTrainedTokenizerFast` so it loads.
- **Context is cheap (MLA latent KV ~90 KB/token):** 16K→128K cost almost no
  memory and zero decode speed. The limiter was a GPU **imbalance**, not KV.
- **Rebalance:** `--tensor-split 12,10,10,10,8,10,10,10` cut per-GPU spread
  8.5GB→3.5GB → freed the hot GPU → enabled 384K.
- **384K needs `--ubatch-size 128`** (shrinks the attention compute buffer to fit
  alongside near-full KV).
- **q8_0 KV + `-fa on` WORK on V100/MLA** → halve KV → 2-way concurrency at 272K.
  (q8_0 KV is *better* than fp8 KV here: more precision/byte, no fp8 HW needed.)
- **MTP (`--spec-type draft-mtp`) works (93% draft accept) but is NET SLOWER**
  (9.8 vs 16.5 tok/s): GLM-5.2's nextn head is a full MoE layer (~2× per-step
  work) and MoE verification activates more experts (less weight-read
  amortization) + multi-GPU sync. Reverted.

### Performance analysis (why decode is ~16 tok/s)
- **Decode is overhead-bound, not bandwidth-bound.** Token = ~62 ms; weight-read
  floor (9 GB active ÷ 900 GB/s) = ~10 ms → **~52 ms is overhead** (≈1000+ kernel
  launches/token across 79 MoE layers + multi-GPU sequential handoffs).
- **Layer-split = pipeline parallel, sequential** for single-stream (1 GPU active
  at a time). NOT tensor parallel.
- **NCCL:** the build links libnccl; llama.cpp now has `--split-mode {layer,row,
  tensor}` — `tensor` is NCCL-backed TP. **Untested** here. TP could use aggregate
  bandwidth (vs sequential layer-split) but adds all-reduce over the slow
  inter-island NVLink link 78×/token — net uncertain. The one real decode-speed
  experiment left.
- **q4 KV won't help decode** (~0.8%; KV is ~1.6% of decode bandwidth).

### Levers map
| want | lever | status |
|---|---|---|
| more concurrency | `--parallel N` | 2-way done; 3+ pushes <256K/slot |
| faster decode | TP (`--split-mode tensor`) | **untested, the one real experiment** |
| faster prefill/TTFT | HMMA fused-dequant kernel fork | scoped, not built |
| more context | q8_0 KV single-slot | up to ~768K available |

---

## Path B — custom vLLM + int2 (validated kernel bet, incomplete)

Built the full path; it runs but never produced a *usable* model (single-digit
tok/s, untested quality). Stands as the "fast affine tensor-core kernel" bet.

### What was validated
- Real `GlmMoeDsaForCausalLM` on V100 at TP=8 (dense-MLA sm70 fallback — DSA
  indexer is DeepGEMM/sm90-only; `_dsa_runtime_enabled()` gates it; dense MLA is
  a correct superset).
- `glm_moe_dsa` → `DeepseekV3Config` registration; `use_mla` + `kv_b_proj`-fp16
  fixes → **TRITON_MLA latent KV** (the ~7× KV win; materialized FLASH_ATTN_V100
  was the wrong path, ~3.6K ceiling → ~19K with latent).
- Info-preserving FP8→2-bit quantizer (MSE-opt affine, real experts 0.50→0.33 rel).
- Packed-2-bit MoE storage + compiled int2 GEMV kernel (per-expert), bit-exact.
- `[REAL-LOAD OK]` — real weights load + generate at TP=8 (gibberish-free vocab,
  but the MoE compute is a Python per-expert loop → single-digit tok/s).
- fp8 e5m2 latent KV on V100 (6-fix overlay; e4m3 is Hopper-only).

### Why it's incomplete
- MoE = **Python per-expert loop** (no grouped int2 kernel) → single-digit tok/s.
- The "fast" kernels (grouped int2 MoE, turbomind `Config_U2`) were never built.
- Quality never measured (uniform 2-bit, no salient protection — likely worse
  than Unsloth's dynamic).

### The kernel-fork opportunity (carries to llama.cpp)
The **dequant-in-register → V100 FP16 HMMA tensor cores** approach is *proven*
(turbomind `Config_E4M3` fp8 + tc-grid int8/int2 kernels exist & use HMMA). To
accelerate **prefill** in llama.cpp: write a codebook(IQ1_S)→fp16 unpack front-end
feeding the existing HMMA mainloop + grouped-MoE GEMM, integrated into ggml-cuda.
Helps prefill/TTFT (compute-bound), NOT decode (bandwidth/overhead-bound). Real
but tractable (machinery exists); the new work is codebook front-end + ggml
integration + grouped-MoE.

---

## Bottom line
- **Usable GLM-5.2 today = llama.cpp + Unsloth IQ1_S:** 76% quality, 16 tok/s
  (24.8 concurrent), 272K–384K context, on 8×V100. Off-the-shelf, dense-MLA.
- **Custom vLLM+int2** = the faster-in-principle path, blocked on the grouped int2
  MoE kernel; validated end-to-end with real weights but single-digit tok/s.
- **Biggest open levers:** TP test for decode; HMMA kernel fork for prefill;
  measure quality vs Qwen-27B before committing more (the unvalidated core
  assumption: 753B@~2-bit > 27B@4-bit, especially for reasoning).
