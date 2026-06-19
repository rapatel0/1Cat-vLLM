# INT2 SM70 (V100) GEMV/GEMM integration — design doc

**Branch:** `int2-v100-gemv` (worktree `~/repos/1Cat-vLLM-int2`, off clean `origin/main` e64d39aa7).
**Status:** design — no production code wired yet. Kernels staged in
`csrc/sm70_int2/kernels_staged/`.
**Goal:** serve **GLM 5.2** (large MoE) on V100s at **4–8 concurrency** with
2-bit weights, using custom int2 kernels that beat cuBLAS in the small-M regime
where decode / MoE-per-expert / spec-decode actually live.

---

## 1. Why this exists (the finding)

Benchmarking on V100 (sm_70, `~/repos/deepseek/tools/tc-grid`) showed the tile /
tensor-core int2 path is **memory-parallelism-starved at small M**: ~125 GB/s
(14% of the ~900 GB/s peak), wall-clock *flat* from M=1→16, while cuBLAS
saturates DRAM at 565–740 GB/s. Net: at M=1, NK=8192 our tile kernel read **6×
fewer bytes than cuBLAS yet finished in the same time** — the entire value of
2-bit was unrealized at small M.

A GEMV-shaped kernel fixes it. Results at **NK=16384** (model-scale, exceeds L2):

| M | kernel | wall-clock | GB/s | vs cuBLAS | rel_err |
|---|---|---|---|---|---|
| 1 | `gemv_m1` (register-A, no smem) | 0.42 ms | 200 | **2.25×** | 3e-7 |
| 2 | `gemv_n` (n-split + n-major repack) | 0.67 ms | 126 | **1.52×** | 1e-6 |
| 4 | `gemv_n` | 0.72 ms | 118 | **1.44×** | 1e-6 |
| 8 | `gemv_n` | 1.32 ms | 65 | 0.79× | 1e-6 |

Two structural properties matter for serving:
- **Crossover at ~M=6–8**: above it, tensor cores (cuBLAS / sm70 Marlin) reclaim
  the lead. So the dispatch must be M-adaptive (below).
- **f32 accumulation ⇒ bit-accurate AND robust**: rel_err ~1e-7–1e-6 (vs the
  f16-acc tile path's ~2e-3) and **no `inf` overflow** on heavy-tailed inputs,
  where the f16-acc kernels saturate. A correctness win, not just speed.

GLM 5.2 is a large MoE; its decode-time cost centers — **per-expert GEMMs**
(small per-expert M even at large global batch), **single-stream/low-batch
decode**, **spec-decode verification** (M = draft length) — all live at M=1–8,
the band we win.

---

## 1.5 Memory budget — why 2-bit is *mandatory* (not optional)

Hardware: 8 × V100-SXM2-32GB = **256 GB** total.

| precision | GLM-5.2 size | fits 256 GB? |
|---|---|---|
| 4-bit | **372–475 GB** | ❌ no |
| 2-bit (uniform) | ~186–238 GB | ✅ yes (tight) |

(4-bit at 372–475 GB ⇒ the model is ~**750–950B params**; 2-bit ≈ half.) So this
is the **"fits only at 2-bit"** regime: 4-bit is impossible here, 2-bit is the
*price of admission*, and these kernels are the enabling path — there is no fast
sm70 2-bit MoE alternative.

### The tension: mixed precision vs the memory budget vs KV cache
The quality-protection plan (§3d — keep router/attention/shared/embed at 4-bit)
**costs extra memory**: keeping ~10% of a 950B model at 4-bit instead of 2-bit
adds **~24 GB** over uniform 2-bit → ~210–262 GB for the weights alone. At the
high end that **leaves almost nothing** for KV cache (4–8 concurrent, long
context) + activations + overhead, and may not fit at all.

So the mixed-precision strategy is **budget-constrained**, and there's a dial:
- More 4-bit layers → better quality, less KV/concurrency headroom (maybe can't
  fit 8-way or long context).
- More 2-bit → fits more KV/concurrency, lower quality.

Concrete consequences to plan for: (a) the "keep at 4-bit" set must be *minimized*
to the highest-leverage layers (router + a few attention/embed) — we likely
**cannot** afford 4-bit on shared experts at the large size; (b) **KV-cache
quantization** (GQA KV → int8/fp8) is probably needed to fit 4–8 concurrency; (c)
max context / concurrency may be budget-limited. **Add the per-layer bit
assignment + KV budget as an explicit fitting problem (P2/P3), not an
afterthought.**

## 1.6 Quality — the honest tradeoff

- **2-bit materially loses quality vs 4-bit** (bit-width is the first-order lever;
  4-bit ≈ lossless, 2-bit is the cliff). But the relevant comparison here is **not
  2-bit-vs-4-bit** (4-bit can't run) — it's **2-bit GLM-5.2 (~850B) vs the best
  model that *does* fit at 4-bit (~450B-class)**. Larger-model-lower-bit usually
  wins down to ~2–3 bit, but 2-bit is near the edge → **must be measured, not
  assumed**.
- Mitigations make "2-bit experts + higher-bit sensitive layers" *much* milder
  than uniform 2-bit: MoE experts are quantization-robust (routing redundancy);
  protecting the router/attention confines the damage; GPTQ calibration ≫ RTN.
- **IQ2 caveat**: codebook i-quants (ds4/llama.cpp) likely have better
  quality-per-bit, but are kernel-incompatible (table lookup, no fast sm70 path).
  We trade some quality-per-bit for a fast linear kernel — a deliberate choice.
- **Gate**: quantize one GLM-MoE variant to 2-bit-experts early and eval
  (perplexity + a couple of task/reasoning evals) vs an fp16/4-bit reference
  *before* building the full pipeline. Risk is highest for long reasoning chains.

## 2. The kernels — turbomind-leveraged split

**Key realization: we already have the turbomind kernels** (vendored in
`lmdeploy/src/turbomind/`, wired into `csrc/sm70_turbomind/`). Turbomind's sm70
GEMM is a mature, format-agnostic tensor-core framework — `MainloopSm70` +
**SplitK** + epilogue + smem swizzle — with per-format `Transform`s for
`Config_E4M3` (fp8), `Config_U4_d/g` (int4), `MXF4`, `F16`, **and a MoE
grouped-GEMM path (`moe_utils_v2.cu`)**. It has **no `Config_U2`** and **no
small-M GEMV**. That defines a clean division of labor:

### 2a. Inherit from turbomind (tensor-core, large-M)
- **Add `Config_U2`** to `arch/config_sm70_s884.h` + a 2-bit `Transform` in
  `transform.h` (our `dequant_2bit` *is* that convert), following the `U4_d/g`
  pattern. We inherit mainloop / **SplitK** / epilogue / swizzle for free.
- Used for: **prefill, dense M≥8, and the larger-M MoE experts** (via
  `moe_utils_v2` + `Config_U2`).
- Accumulation is **f16** (the V100 tensor core's 2× path) — fine for prefill
  (bounded post-norm activations), see §2c.

### 2b. Our novel kernels (small-M, the gap turbomind can't fill)
`csrc/sm70_int2/kernels_staged/` — turbomind's GEMM is tensor-core, i.e.
structurally wasteful at M=1–8; these own that regime:

| entry point | regime | notes |
|---|---|---|
| `mm_int2_gemv_m1<COLS_PER_BLOCK,WARPS>` | **M=1** | lanes-split-K, per-lane activation in registers (no smem/sync) |
| `mm_int2_gemv_n<WARPS,KK,MAXM>` | **M=2–~5** | lanes-split-N, activation broadcast, needs `repack_nmajor` |
| `repack_nmajor` | load-time | row-major uint32 → `[N/32][K/16][32]` |
| **grouped int2 GEMV** *(to build)* | **MoE experts, tiny M (0–2)** | one launch over experts w/ ragged offsets; avoids per-expert launch storm. Crossover vs turbomind `moe_utils_v2`+U2 to be measured |
| `repack_gptq_int2` *(to build)* | load-time | GPTQ int32-packed → our layouts |

Accumulation is **f32** (the decode-accuracy / no-overflow path, §2c).

### 2c. Accumulation: f32 in the GEMV, f16 in turbomind — *measured*
The accumulator choice is **opposite** between the two execution units, and an
A/B test (tc-grid v23 f32-acc vs v25 f16-acc, M=1, NK=16384) settles it:

| | f32-acc | f16-acc |
|---|---|---|
| throughput | **200 GB/s** | 131 GB/s (**1.5× slower**) |
| rel_err (U(-1,1)) | **2.4e-7** | 1.7e-3 |
| LogNormal | finite | **inf (overflow)** |

- **GEMV runs on CUDA cores**, where scalar `half` has *no* throughput edge and
  the `float→half` conversions add instructions that **break the
  memory-boundedness** (200→131 GB/s). So **f32-acc is both faster *and*
  bit-accurate *and* overflow-safe** — strictly better. **Keep f32 in the GEMV.**
- **turbomind GEMM runs on tensor cores**, where f16-accumulate *is* the 2× path
  (HMMA f16-acc 125 vs f32-acc 62 TF) — *mandatory* for speed, and safe because
  prefill activations are bounded. (The f16 "2× FMA" advantage that justifies it
  there simply doesn't exist on the CUDA cores the GEMV uses.)

So **decode→GEMV (f32-acc), prefill→turbomind (f16-acc)** is not a compromise —
each accumulator is the *faster* choice on its own unit.

### 2d. `int2_rf` tile — demoted to reference/fallback
Our hand-rolled tensor-core tile (`int2_rf_kernels.cuh`, ~40 TF prefill) is
**superseded by turbomind+`Config_U2`** on the critical path (turbomind is
integrated, maintained, and we found our pipelining couldn't beat its
structure). Keep it as a correctness/perf reference and a fallback if `Config_U2`
slips. `int2_v3_kernels.cuh` stays as the wmma-f32 correctness oracle;
`mma_sm70.cuh` is the HMMA wrapper (only needed if we use `int2_rf`).

---

## 3. 2-bit weight format

### 3a. Research format (what the kernels assume *today*)
Matches the tc-grid quantizer (`data_gen.cu::k_quantize_int2`):
- **Packed weights `qs`**: per output row `n` (N rows = output channels), K values
  packed **4 per byte**, signed two's-complement 2-bit; value at K index `k` lives
  in bits `[2*(k%4) : 2*(k%4)+1]` of byte `k/4`. Row stride = `K/4` bytes.
- **Scales**: fp16, one per `QK_INT2 = 32` consecutive K values:
  `scales[n * (K/32) + k/32]`. **Symmetric (no zero-point)**, per-32-block absmax.
- Dequant in-kernel = `sign_extend2(field) * scale` (clamped to ternary `{-1,0,1}`).
- **`gemv_n` repack**: regroups the uint32 stream into `[N/32][K/16][32]`
  (transpose only; payload untouched), done once at load time.

This is fine for **kernel benchmarking** but the naive per-32 absmax + ternary
clamp would **destroy GLM 5.2 quality** — it is *not* the production format.

### 3b. Production format: **GPTQ `bits=2`** (the decision — resolves old §6.1)
There is **no 2-bit path in this fork today** (AWQ rejects ≠4-bit; `moe_wna16`
asserts 4/8; sm70 Marlin has `u4/u8/...` but **no `u2`**), and **no off-the-shelf
2-bit GLM 5.2 checkpoint exists** (GLM ships fp16; community quants are 4-bit). So
**we quantize GLM 5.2 ourselves**, and we target **GPTQ `bits=2`** as both the
algorithm (Hessian-based, real 2-bit quality via GPTQModel / llm-compressor) and
the on-disk format:
- **Linear, group-wise, asymmetric**: `w = (q − zero) * scale`, `q ∈ [0,3]`,
  per-group fp16 `scale` + packed `zero`, **group_size 128** (typ), packed int32
  along K, `pack_factor = 32/bits = 16`.
- **Why GPTQ-2bit**: it's a *standard linear format* our kernels can decode, it
  reuses vLLM's already-bit-generic GPTQ/`moe_wna16` **loading** machinery (only
  the kernels/asserts cap bits at 4/8 — the loader parameterizes `weight_bits`),
  and it lets us ingest third-party 2-bit GPTQ checkpoints later via the same path.
- **Incompatible formats (do NOT target)**: GGUF `IQ2_*`/`Q2_K` (codebook /
  k-quant — table lookup, not a multiply), AQLM/QuIP# (vector quant). These need a
  fundamentally different kernel; out of scope.

### 3c. Required kernel change — generalize the dequant (P1, no-regret)
Make the in-kernel dequant **`(q − zero) * scale`** with a **configurable group
size** (`GROUP` template/param, default 128), instead of the hard-wired symmetric
`sign_extend2 * scale` at group 32. Cost: one extra subtract per value + a
zero-point load + a scale/zero stride parameter. Applies to **both** our GEMV
(`gemv_m1`, `gemv_n`) and the turbomind `Config_U2` `Transform` — makes them
GPTQ-2bit-compatible and unlocks the "quantize-ourselves" and "ingest-3rd-party"
paths. **First P1 task (§7).**

### 3d. Mixed precision — 2-bit only the MoE experts
**Do not** uniformly 2-bit the model. Quantize **only the MoE expert FFN weights**
to 2-bit (the memory bulk, and exactly where our kernels win at small per-expert
M); keep **attention, router/gate, shared experts, embeddings, lm_head** at
**4-bit (sm70 Marlin u4) or fp16**. Standard sub-4-bit practice, and it aligns
perfectly: 2-bit lands where we're fast *and* where the bytes are, while the
quality-sensitive layers stay higher-bit. The quant method must therefore be
*per-module* (a `modules_to_not_convert` / expert-only selector, like
`awq_sm70_moe.py`).

---

## 4. Integration architecture

```
 GLM 5.2 (glm4_moe.py)  ──uses──▶  Int2Sm70LinearMethod / Int2Sm70MoEMethod
                                          │  (Python quant method;
                                          │   patterns: awq_sm70_moe.py / moe_wna16.py)
                                          ▼
              M-adaptive dispatch (per GEMM, by token count M):
                 ┌─────────────────────────────┬──────────────────────────────┐
                 │  small-M (decode/expert)    │  large-M (prefill/dense)      │
                 │  OUR kernels (f32-acc)      │  TURBOMIND (f16-acc)          │
                 ├─────────────────────────────┼──────────────────────────────┤
   dense/attn    │ M==1 → gemv_m1              │ M≥~6 → turbomind Config_U2    │
                 │ 2≤M≤~5 → gemv_n             │                              │
   MoE experts   │ tiny M(0–2) → grouped gemv  │ larger M → moe_utils_v2 + U2  │
                 └─────────────────────────────┴──────────────────────────────┘
   (crossover ~M=6, shape-dependent → tunable threshold)
```

**Where each piece lands (all precedented on this branch):**
- **Turbomind `Config_U2`**: extend `lmdeploy/src/turbomind/kernels/gemm/`
  (`arch/config_sm70_s884.h`, `transform.h`); register via
  `csrc/sm70_turbomind/`. Inherits SplitK / epilogue / swizzle / `moe_utils_v2`.
- **Our GEMV op**: new module `csrc/sm70_int2/`, registered like
  `csrc/sm70_turbomind/ops/tm_registry_sm70.cu`. Owns small-M only.
- **Quant method**: `vllm/model_executor/layers/quantization/int2_sm70_moe.py`,
  patterned on `awq_sm70_moe.py` + `moe_wna16.py`. Registers a
  `QuantizationConfig` + `LinearMethodBase` + a `FusedMoEMethodBase`.
- **MoE hook**: `vllm/model_executor/layers/fused_moe/` (`fused_moe_method_base.py`)
  — the per-expert GEMM is where our small-M win is largest.
- **Model**: `glm4_moe.py` (+ `_mtp` for spec-decode). GLM 5.2 = config delta.

**Why vLLM and not ds4 or llama.cpp** (recorded for posterity): ds4 is
compile-time-welded to DeepSeek-V4 compressed attention (`#define
DS4_V100_HEAD_DIM 512`, indexer/hc KV) — adapting it to GLM's GQA means gutting
its core. vLLM already ships `glm4_moe(+mtp)`, mature TP / paged-KV / **CUDA
graphs** (which amortize the per-launch overhead that small-M kernels are
sensitive to), and an sm70 low-bit MoE family to extend. `~/repos/dsv4-vllm`
`PORT_PLAN.md` already concluded "inherit the vLLM runtime; ds4 stays the
read-only oracle." ds4 / vLLM-fp16 = correctness oracles, not targets.

---

## 5. Kernel adaptation needed (Phase 1 work)

1. **De-harness**: drop `#include "tc_grid.h"`; define `constexpr int QK_INT2 =
   32;` locally (or template it for future group sizes).
2. **Entry points**: wrap each kernel in a host launcher taking raw pointers +
   `(M, N, K)` and a stream; expose `int2_sm70_gemm(out, A, qs, scales, M,N,K)`,
   `int2_sm70_repack_nmajor(dst, qs, N, K)` as torch ops.
3. **Dtype**: harness used `float` activations + f32 accumulate. vLLM activations
   are fp16/bf16 → either cast on load into the kernel or add fp16-input variants
   (accumulate stays f32 — that's the robustness win, keep it).
4. **Adaptive dispatch** in the host launcher keyed on M (and a per-shape
   threshold; the crossover is ~M=6 but depends on N/K — make it tunable).
5. **Load-time repack**: `gemv_n`'s n-major layout built in
   `process_weights_after_loading`; scales kept as-is.
6. **Build**: add `csrc/sm70_int2/*.cu` to `CMakeLists.txt` under the existing
   sm70 gating; verify the two-wheel V100 build (per fork README, pod-only).

---

## 6. Open questions (resolve before/with Phase 2)

1. **✅ RESOLVED — Quantization format.** Decision (see §3b): quantize GLM 5.2
   ourselves with **GPTQ `bits=2`** (linear, asymmetric, group 128), 2-bit on MoE
   experts only (§3d). Drives the P1 dequant generalization (§3c). Remaining
   sub-task: pick the tool (GPTQModel vs llm-compressor) + the calibration set.
2. **M≥8 dense path**: reuse the existing `sm70_marlin` (u4 today; would need a u2
   or run dense fp16) or ship our `int2_rf` tile? Marlin is more battle-tested;
   our tile is int2-native. Lean Marlin/existing first, our tile as an option.
3. **Per-expert M distribution** in GLM 5.2 MoE at concurrency 4–8 — measure to
   confirm experts land at M=1–4 (where we win biggest). Drives the threshold.
4. **TP weight sharding**: the n-major repack must compose with column/row TP
   sharding of the expert weights. Repack *after* sharding, per shard.

---

## 7. Phases

- **P0 (done)**: kernels built + validated in tc-grid; this branch + worktree;
  kernels staged.
- **P1 — kernel module**:
  1. **Generalize the dequant to `(q − zero)·scale` + configurable group size**
     (§3c) — the first concrete task; makes the kernels GPTQ-2bit-compatible.
  2. De-harness (drop `tc_grid.h`), host launchers, torch op, build wiring.
  3. Standalone numerical test (vs `int2_v3` oracle, then vs a GPTQ-2bit dequant
     reference) on the pod.
- **P2 — quant method**: `int2_sm70_moe.py` (Config + Linear + MoE methods);
  load-time repack; resolve the format question (§6.1).
- **P3 — GLM 5.2 bring-up**: run `glm4_moe` with int2 weights; validate logits
  token-for-token vs fp16 (vLLM) and/or ds4 oracle.
- **P4 — measure**: end-to-end token latency / throughput at concurrency 4–8;
  confirm the per-GEMM small-M wins convert to serving wins; tune the M-threshold.

---

## 8. Risks

- **Format mismatch (§6.1)** — highest. Could turn "wire in kernels" into
  "re-quantize GLM 5.2 or rework dequant." Resolve first.
- **CUDA-graph capture** of an M-adaptive (data-dependent) dispatch — graphs want
  fixed shapes. May need per-M captured graphs or a fixed-M expert path. Verify
  early.
- **Repack × TP sharding** correctness (§6.4).
- **M=8 boundary**: if real per-expert M clusters at 6–8, our edge shrinks; the
  dense Marlin path may dominate. Measure (§6.3).
