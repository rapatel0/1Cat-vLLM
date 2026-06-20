# GLM-5.2 / 8×V100 — parallelism & context architecture

How to map GLM-5.2 onto the 8×V100 node for **long context + serving throughput**.
Companion to `INT2_SM70_INTEGRATION.md` (weights) and `THROUGHPUT_IDEAS.md` (tok/s).
Conclusion up front: **TP4 × PP2 matches the NVLink topology and reaches ~120 K
context with fp8 KV** — likely faster *and* longer-context than flat TP8.

---

## 1. The hardware topology (the constraint that drives everything)

The 8×V100-SXM2 node is **two 4-GPU all-to-all NVLink islands** with a slower link
between the islands:

```
 island A (all-to-all NVLink)        island B (all-to-all NVLink)
   GPU0 ─ GPU1                          GPU4 ─ GPU5
    │  ╳  │                              │  ╳  │
   GPU2 ─ GPU3      ── slow link ──      GPU6 ─ GPU7
```

- **Within an island (4 GPUs):** full-bandwidth, low-latency NVLink all-to-all.
- **Across islands:** slower (fewer NVLink hops / PCIe-class). Any collective that
  spans both islands is gated by this link.

The single most important scheduling fact: **keep high-volume collectives inside an
island; let only cheap point-to-point traffic cross between islands.**

---

## 2. Why flat TP8 is the wrong map

Tensor parallel slices every layer across all 8 GPUs and does an **all-reduce after
every layer** (78×/token). With TP8 spanning both islands, **every one of those 78
all-reduces crosses the slow inter-island link.** That is almost certainly the
dominant decode comm cost today, and it also replicates the MLA KV latent on all 8
GPUs (KV capacity = one GPU's free memory).

| problem | flat TP8 |
|---|---|
| per-layer all-reduce | crosses slow link 78×/token |
| MLA KV | replicated on all 8 → ~32 K context ceiling |

---

## 3. The map: TP4 × PP2 (TP inside islands, PP across)

**Rule:** TP within a fast NVLink island, PP across the slow link.

```
 stage 0 = layers 0..38            stage 1 = layers 39..77
 island A: TP=4 (GPU0-3)   ──PP handoff──▶  island B: TP=4 (GPU4-7)
   all-reduce on fast NVLink          all-reduce on fast NVLink
            └──────── only hidden state crosses slow link ────────┘
```

- **TP=4** within each island → per-layer all-reduce stays **entirely on fast
  all-to-all NVLink** (4 GPUs, no slow-link crossing). Much cheaper than TP8.
- **PP=2** across islands → the **only** inter-island traffic is the pipeline
  hand-off: the hidden state at the stage boundary, ~`hidden_size(6144) × 2B ×
  batch` ≈ **12 KB/token**, point-to-point. The slow link no longer cares.

**Net:** removes the slow-link all-reduce from the critical path → **likely faster
than flat TP8**, not merely equal. Weights/GPU are unchanged (28.4 GB): PP splits by
layer, TP slices within — each GPU still holds 1/8 of the model.

---

## 4. Context: PP shards the KV cache

MLA KV is **replicated across TP** but **partitioned across PP**. PP=2 → each GPU
holds KV for ~39 layers instead of 78 → **KV/token/GPU halves (88 → 44 KB) → 2×
context.**

KV/token/GPU and resulting context at ~2.9 GB/GPU KV budget (g256 expert scales):

| layout | KV dtype | KV/tok/GPU | context |
|---|---|---|---|
| TP8 | fp16 | 88 KB | ~32 K |
| TP8 | fp8 | 44 KB | ~64 K |
| **TP4×PP2** | fp16 | 44 KB | ~64 K |
| **TP4×PP2** | **fp8** | **22 KB** | **~128 K** ✅ |
| TP4×PP2 | fp4 | 11 KB | ~256 K |

**TP4×PP2 + fp8 KV reaches the 120 K target** — without needing fp4, so KV stays
higher-fidelity. (fp4 is the lever for 256 K+ later.)

KV math: per token per layer = `kv_lora_rank(512) + qk_rope_head_dim(64) = 576`
values; × dtype bytes × layers-per-stage(39). fp16 = 44 KB/tok/GPU.

---

## 5. Throughput characteristics

- **Single stream:** PP adds one pipeline crossing → marginally worse *in
  isolation*, but the faster intra-island all-reduce likely nets **positive** vs
  TP8. tok/s ceiling is the same bandwidth bound either way (~280 tok/s theoretical;
  see THROUGHPUT_IDEAS.md — gated by the Python MoE loop until Tier-1 lands).
- **Batched serving:** fill the 2-stage pipeline with ≥2 microbatches (ideally
  16–32) → both islands busy, requests interleaved. Aggregate tok/s ≈ TP8's, with
  **less slow-link comm** and the ability to run **concurrent long-context**
  requests (KV sharded by stage, not replicated).
- **Why PP for *us*:** not raw tok/s (≈ TP8) — it's (a) topology-matched comm and
  (b) long-context-×-concurrency that replicated-KV TP8 physically can't hold.

---

## 6. Higher PP (future, if context must exceed ~128 K)
- **PP=8** (pure pipeline) → KV/token/GPU = 11 KB → ~270 K @ fp16, ~1 M @ fp4 — the
  model's native 1 M context becomes memory-feasible. Cost: 8-stage pipeline needs
  many microbatches to stay full; single-stream latency worse; **off-island stages**
  → more slow-link hops. Only worth it for very-long-context serving.
- **TP2×PP4**: smaller NVLink TP domains, more KV sharding — viable but TP2 may not
  saturate the per-layer compute as well as TP4. TP4×PP2 is the sweet spot for this
  topology.

---

## 7. Risks / unknowns (must verify before committing)
1. **Our custom path under PP is untested.** vLLM `deepseek_v2` supports PP via
   `make_layers`, but the `Int2Sm70PackedMoEMethod` + DSA-dense MLA have only run
   under flat TP. Confirm they pipeline-split cleanly.
2. **EP × PP composition.** Experts currently shard EP=8; under TP4×PP2 they shard
   **EP=4 within each stage** — the `expert_map` / packed-loader sharding must
   compose with the PP layer partition. This is the most likely break point.
3. **fp8 KV through MLA/FLASH_ATTN_V100.** The turbomind `ConvertKvCache<fp8_e4m3_t>`
   exists; confirm the MLA decode backend accepts `kv_cache_dtype=fp8` on sm70.
4. **Pipeline scheduling** in vLLM v1 with our quant method (microbatch feeding).

---

## 8. Recommended sequence
1. **Baseline:** get flat **TP8** to load + generate (proves weights/fp8/MoE
   end-to-end; read the real KV ceiling + tok/s). ← current step.
2. **Re-launch TP4×PP2** (`--tensor-parallel-size 4 --pipeline-parallel-size 2`),
   pinning ranks to islands (CUDA_VISIBLE_DEVICES ordering 0-3 | 4-7). Compare
   tok/s + verify our MoE composes with PP.
3. **Add fp8 KV** (`kv_cache_dtype=fp8`) → target ~128 K context.
4. **g256 expert scales** if more KV headroom needed (+1.4 GB/GPU).
5. Later: Tier-1 throughput (grouped MoE kernel, CUDA graphs, MTP) — orthogonal to
   the parallelism map; multiplies tok/s in either layout.

**Bottom line:** flat TP8 is a topology mismatch (slow-link all-reduce ×78/token +
replicated KV). **TP4×PP2 confines all-reduce to the fast NVLink islands and shards
KV across the slow link** → faster comm and, with fp8 KV, ~120–128 K context. Adopt
it once we confirm the custom MoE/MLA path pipelines cleanly.
