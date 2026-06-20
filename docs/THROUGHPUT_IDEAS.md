# GLM-5.2 / 8×V100 throughput optimization — plan

Side-quest brainstorm → plan for raising cluster tok/s once the real model is up.
Grounded in this stack: 8×V100-SXM2-32GB (sm70; HMMA.884 f16-acc, ~900 GB/s HBM2,
NVLink), GLM-5.2 743B MoE (256 experts / 8 active), **2-bit experts + fp8
attention/MLA** (block-fp8 on the fork's SM70 FP8 TurboMind W8A16 kernels),
decode memory-bandwidth-bound. Current MoE compute = per-expert Python loop +
`enforce_eager` (correct, serial, launch-overhead-bound). **Validated deployment
parallelism is TP4×PP2** (TP within each NVLink island, PP across the slow
inter-island link) — see `GLM52_PARALLELISM_ARCH.md` for the topology rationale.

## Current bottlenecks (where time/bandwidth goes)
1. **MoE per-expert Python loop** — 256-iter Python loop, one kernel launch per
   active expert, *zero* cross-expert batching. #1 decode hotspot. The int2 GEMV
   kernel is fast per call; Python dispatch + launch latency + no batching dominate.
2. **`enforce_eager`** — every tiny kernel pays full launch latency every token.
3. **Dense-MLA fallback** — materializes K/V for FLASH_ATTN_V100; O(seq²), and the
   materialization wastes the bandwidth MLA's latent compression is meant to save.
4. **Inter-island all-reduce** — *largely mitigated by TP4×PP2.* Flat TP8's
   per-layer all-reduce crossed the slow inter-island link 78×/token; under
   TP4×PP2 the all-reduce is confined to the fast NVLink island (TP=4, on-island)
   and only the cheap PP hidden-state handoff (~12 KB/token, point-to-point)
   crosses the slow link. Remaining comm cost = per-stage TP4 all-reduce (cheap)
   + PP bubbles. Items 1–3 are parallelism-independent and remain the real
   bottlenecks.

---

### Parallelism is now fixed: TP4×PP2

The deployment-parallelism question is **settled and validated** on the 8×V100
node: **TP4×PP2** (TP within each NVLink island, PP across). It is topology-matched
(all-reduce stays on-island), gives ~2× context headroom (KV sharded by stage, not
replicated — 4.25 vs 2.24 GiB KV/GPU), and lowers per-GPU weights (25.91 vs 27.72
GiB; embed/lm_head split across stages) and load time (437 vs 941 s). The
throughput levers below are **orthogonal to the parallelism map** — they apply
per-stage identically (each stage runs the MoE/attention for its ~39 layers). The
*one* PP-specific throughput requirement is keeping the pipeline fed with
concurrent microbatches (see Tier-2 D+E).

---

## TIER 1 — definite next steps (do these first)

### A. Grouped / batched int2 MoE kernel  ⟵ largest ceiling
Replace the Python per-expert loop with **one kernel over all experts**:
1. Route → `topk_ids`; **sort tokens by expert** (`moe_align_block_size` pattern)
   so each expert owns a contiguous token block.
2. Single grouped-GEMM launch with **ragged per-expert offsets**; weights stay
   packed 2-bit, **dequant inside the kernel** (reuse the validated
   `int2_gemv`/`int2_v3` dequant in a batched mainloop). Fuse gate/up + SiLU +
   down to cut HBM round-trips.
- **Why it matters:** experts are 97% of weight traffic; only here does 2-bit's
  8× bandwidth advantage actually convert to tok/s. Kills Python dispatch *and*
  per-expert launch overhead.
- **Build options:** (i) extend tc-grid `int2_gemv` into a grouped variant
  (`mm_int2_gemv` already templated on `MAXM`); (ii) add **`Config_U2`** to the
  turbomind `moe_utils_v2` grouped-GEMM path (design §2a) — inherits its mature
  tile/SplitK/swizzle machinery; best for the larger-M experts.
- **Effort:** high (kernel project). **Impact:** highest. This is *the* item that
  turns our validated-but-serial MoE into a bandwidth-saturating one.
- **Under TP4×PP2:** each stage runs the MoE for its ~39 layers; the grouped kernel
  applies per-stage identically — orthogonal to the parallelism map, no rework.
  Still the largest ceiling.

### B. CUDA graphs for decode (drop `enforce_eager` in prod)
Capture decode as CUDA graphs (per batch size). Amortizes exactly the per-launch
overhead small-M MoE kernels are most sensitive to (design doc calls this out).
- **Constraint:** graphs want fixed shapes → the M-adaptive dispatch (G) and the
  grouped-MoE offsets must be **graph-stable** (fixed capture buckets, padded
  token counts). Pairs naturally with A (one stable launch vs 256 variable ones).
- **Effort:** low–med once shapes are stable. **Impact:** large, near-free.
- **Under TP4×PP2:** graphs are captured per-stage; the pipeline microbatch shapes
  must be graph-stable. PP *needs* multiple in-flight microbatches to keep both
  stages full — which pairs naturally with capturing one graph per batch size.

### C. MTP speculative decoding  ⟵ highest model-level leverage
GLM-5.2 **ships MTP** (`num_nextn_predict_layers`; card cites 5 draft tokens).
Draft k tokens, verify in one forward → **~2–3× decode tok/s**. The fork already
has MTP plumbing (`glm4_moe_mtp`, Qwen MTP, `vllm/v1/spec_decode/*`).
- **Work:** wire the GLM-5.2 MTP head (the checkpoint has the nextn layer; we
  currently drop it). Verify draft/verify alignment on sm70; reuse the existing
  rejection sampler + MTP proposer.
- **Effort:** med (mostly integration, designed-in). **Impact:** 2–3×.
- **Synergy:** MTP verify is a small-M batch (M = draft len) → exactly the regime
  our int2 GEMV + (A) grouped MoE win; and graphs (B) cover the fixed draft shape.
- **Under TP4×PP2:** the MTP head sits at the **last stage (PP1)**, alongside
  lm_head. Draft tokens flow through the full pipeline; verify is a small-M batch
  that fits our int2 GEMV + grouped MoE regime. No conflict with PP.

---

## TIER 2 — strong, do after Tier 1

### D + E. Bigger batches via fp8/fp4 KV cache  (synergistic — treat as one)
2-bit experts free ~150 GB vs fp16; that headroom → **more KV → more concurrent
sequences**. Decode is weight-bandwidth-bound, so batching amortizes the (now
2-bit) weight reads across sequences → throughput scales with batch until
bandwidth re-saturates. The lever to *grow* the batch is a smaller KV cache.

**Use fp8 (e4m3) / fp4 (e2m1) KV, NOT int8** — the decisive detail:
- The turbomind sm70 attention kernels already ship
  **`ConvertKvCache<fp8_e4m3_t>` and `ConvertKvCache<fp4_e2m1_t>`**
  (`csrc/sm70_turbomind/.../attention/quantization.h`): they unpack fp8/fp4 → f16
  **inside the attention kernel**. So the read path is already accelerated.
- **fp8/fp4 are floating-point** → dynamic range is in the exponent, so they work
  **scale-free** (or with a single global scale), avoiding int8's per-group/
  per-token **scale tensors** — less memory, less bookkeeping, no scale-load in
  the hot loop, no calibration. (fp4_e2m1 microscaling carries block scales, but
  the turbomind converter handles that layout.)
- **fp4 KV** ≈ 4× smaller than fp16 → ~4× the batch / context for the same VRAM;
  **fp8 KV** ≈ 2× and higher fidelity. Start fp8 (quality-safe), try fp4 for
  long-context / max-batch.
- Pairs with MLA: KV is already latent-compressed (`kv_lora_rank=512`); fp8/fp4 on
  top compounds it.
- **Work:** wire `kv_cache_dtype=fp8_e4m3 / fp4_e2m1` through the MLA/FLASH_ATTN_V100
  path to the turbomind converters. **Effort:** med. **Impact:** large (batch ↑).
- **Under TP4×PP2 — this is where PP matters MOST for throughput.** The 2-stage
  pipeline must be **fed** with concurrent microbatches (≥2, ideally 16–32) to keep
  both islands busy, so batching isn't just a throughput nicety — it's **required**
  to amortize the pipeline (idle stages = bubbles). PP also shards KV by stage, so
  each concurrent request can be **long-context**: the TP8-replicated-KV ceiling is
  removed. fp8 KV (`kv_cache_dtype=fp8_e4m3`) is validated/plumbed via the fork's
  `flash_attn_v100` FP8-KV routes, targeting ~128 K context.

### F. Absorbed-MLA path (stop materializing K/V)
Our dense fallback materializes full per-head K/V for FLASH_ATTN_V100. The
**absorbed** MLA formulation keeps attention in the 512-dim latent space (fold the
up-projection into Q/O), so it reads/writes far less and needs no full-head KV.
- **Why:** cuts both attention HBM traffic *and* KV footprint (compounds D/E), and
  reduces the O(seq²) constant at long context.
- **Work:** check whether the fork's MLA can run absorbed on sm70 (DeepSeek-V2
  absorbed path) rather than the materialized FLASH_ATTN_V100 route; if not, add
  the absorbed decode path. **Effort:** med–high. **Impact:** med–large
  (long-context + batch).

### G. M-adaptive expert dispatch (small-M → int2 GEMV, large-M → tensor cores)
At higher batch some experts receive M≫8 tokens; there tensor cores beat our
memory-bound GEMV (crossover ~M=6–8, design §1). Dispatch per expert by its token
count:
- M=1 → `int2_gemv_m1`; 2–8 → `int2_gemv_n`; **M>8 → tensor-core grouped GEMM**.
- For the large-M path, leverage the **existing `fp8_sm70_moe.py`** /
  turbomind `Config_U2`+`moe_utils_v2` rather than building anew — fp8 experts via
  the turbomind kernels are already fast on this stack, and (per D/E rationale)
  fp8 avoids int-scale bookkeeping. Mixed expert precision (2-bit for the
  bandwidth-bound small-M majority, fp8 for the compute-bound large-M tail) is a
  clean fit.
- **Effort:** med (dispatch + threshold tuning; threshold already in design).
  **Impact:** med, grows with batch — and directly complements D/E (bigger batch
  pushes more experts into the large-M regime).
- **Under TP4×PP2:** experts shard **EP=4 within each stage** (vs EP=8 flat), so
  each GPU now owns **more experts per stage** (256/4 = 64 vs 256/8 = 32). Per-expert
  token counts (M) may shift accordingly — **re-measure the M-threshold under the
  TP4×PP2 expert distribution** before tuning the GEMV↔tensor-core crossover.

---

## TIER 3 — micro-opts / later
- **Compute/comm overlap** — overlap TP all-reduce with expert compute (fork has
  SM70 compact-allreduce to build on).
- **Persistent decode kernels**, split-K GEMV at large N, vectorized loads.
- **int4 attention** (sm70 Marlin u4) — more decode speed if quality holds
  (tension with §3d "keep attention fp16" — measure).
- **EP-vs-TP tuning** for the MoE specifically (EP avoids per-expert all-reduce).

---

## Sequencing (impact-per-effort)

Parallelism is settled — **TP4×PP2** (validated, topology-matched). The throughput
sequence below is **unchanged and orthogonal** to the parallelism map (it applies
per-stage identically); the only PP-specific dependency is that D+E batching keeps
the pipeline fed.

1. **C — MTP** (designed-in, 2–3×, mostly wiring)
2. **B — CUDA graphs** (near-free once shapes stable; enables A/C/G to be graph-safe)
3. **A — grouped int2 MoE kernel** (largest ceiling; the real kernel project)
4. **D+E — fp8/fp4 KV → bigger batch** (turbomind converters already exist)
5. **F — absorbed MLA**, **G — M-adaptive dispatch** (compound the batch gains)

Cross-cutting theme: **prefer the fp8/fp4 turbomind kernels already in 1Cat-vLLM
over inventing int8 paths** — they're accelerated on sm70 and avoid explicit scale
factors. Our 2-bit work owns the *weight-memory* problem (experts); fp8/fp4 owns
the *KV/activation* problem and the *compute-bound large-M* tail.
