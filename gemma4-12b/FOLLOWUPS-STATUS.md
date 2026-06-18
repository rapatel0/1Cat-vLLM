# Follow-up tasks — stock-taking (updated 2026-06-18, second pass)

Status after working the gemma-12B follow-up list to resolution. Two buckets: **V100-fork-general**
(apply to whatever gemma4 model is live) and **31B-MTP-specific** (the 31B + spec decode is currently
scaled to 0, replaced by the live `gemma4-12b`).

## Live state
- **`gemma4-12b`** (omni `gemma4_unified`, fp16, TP2, 262K context) is the active deploy on 2 NUMA0
  V100s. **Single-stream 33.5 tok/s** with PIECEWISE CUDA graphs, **and now concurrency-stable**
  (#20 fixed). 31B + 31B-mtp at 0.

## Done

### #16 — RoPE null-key primitive fix ✅ (committed `d160bac1c`)
`forward_cuda` now routes `key=None` → `forward_native` (correct query rotation), mirroring
`forward_xpu`. Removes the need for the per-model dummy-key workaround for future Q-only callers.
*Validation pending a 31B-MTP redeploy* (forward-looking hardening; the gemma4_mtp dummy-key
workaround stays until revalidated).

### #17 — throughput / single-slot tuning ✅
Context maxed (262K fits via sliding-window KV), TP swept (TP1/2/4 all ~21 eager → not TP-bound,
it's launch-overhead bound), **PIECEWISE CUDA graphs lifted single-stream 21 → 33.5 tok/s (1.6×)**.

### #20 — PIECEWISE concurrency hang ✅ (committed in homelab `2fbea01`)
**Root cause:** captured PIECEWISE graphs read stale input-buffer pointers when batch composition
varies across concurrent requests → conc≥2 hung (requests timed out, engine survived).
**Fix:** `cudagraph_copy_inputs: true` in the compilation-config (copies inputs into internally-
managed buffers before replay; only effective under PIECEWISE). Verified on the live deploy:
- conc=1: **33.5 tok/s** sustained single-stream (unchanged from 33.7 — copy adds *no* single-stream cost)
- conc=2: 36.6 / conc=4: 105.7 / conc=8: 108.5 tok/s aggregate — **all 0 errors, 0 pod restarts**

The PIECEWISE config is now both single-slot-optimal **and** multi-user-safe; no need to fall back
to `--enforce-eager` for concurrency. (Eager still wins raw aggregate at high conc — ~159 @ conc8 —
but loses single-stream 21 vs 33.5; PIECEWISE+copy is the right default for the single-slot priority.)

### #18 — bake overlay into image ✅ (built, validated, LIVE)
The deploy was doing three things at boot (transformers 5.5 install, gemma4_unified python overlay
extract, hd512 `.so` copy). Folded into a self-contained image so the wrapper isn't needed:
- `Dockerfile.110-gemma12b` — base 1.1.0 sm_70 recipe + the three gemma layers.
- `00-build-110-gemma12b.yaml` — buildkit Job; initContainer stages the **byte-exact validated
  artifacts from the PVC** (overlay tarball + 2 `.so` kernels) into the build context. New tag
  `onecat-vllm:v1-v100-110-gemma12b` (doesn't disturb the existing `:v1-v100-110`).
- `28-deployment-gemma4-12b-baked.yaml` — same serving config, command is just NUMA-pin + launch.
- **Why bake from PVC artifacts, not git:** the 13 overlay `.py` files span multiple working
  branches (base gemma4 port + gemma4_unified + flash_attn_v100 mods), so they're not cleanly
  reproducible from one git ref; the PVC artifacts are exactly what's validated-running.
- **Gotcha found + fixed:** a fresh build re-resolved fastapi/starlette to today's latest
  (0.137.1 / 1.3.1), which break vLLM 1.1.0's router (`'_IncludedRouter' object has no attribute
  'path'` on every request). Pinned to the prod-validated pair `fastapi==0.136.3 / starlette==1.2.1`
  (the versions baked into the working `v1-v100-110`). Deploy uses `imagePullPolicy: Always` (the
  tag is mutable).
- **Validated LIVE on the baked image:** boots clean (0 restarts, health 200), sustained
  single-stream **33.5 tok/s**, conc 1/2/4 = 33.3 / 35.9 / 105.5 tok/s, **0 errors** — identical to
  the overlay deploy, with no boot-time install/extract. The live `gemma4-12b` deploy now runs the
  baked image; the overlay manifest (`28-deployment-gemma4-12b.yaml`) stays as the fallback.

### #19 — FULL CUDA graphs (more single-slot) ✅ SOLVED via `FULL_AND_PIECEWISE` (no rewrite)
Initial read was that the whole V100 attention is a host-driven per-seq dispatch loop needing a
kernel rewrite. **That's only true of the PREFILL path** (`_flash_v100_prefill*`, host `.item()`
loops at lines 593-603/807-852/925). The **DECODE path** (`_flash_v100_decode`, lines 631-666) is
already device-resident — one batched `flash_attn_decode_paged` launch over `block_table`/`seq_lens`
device tensors, **zero `.item()`** (it even logs "CUDA-graph safe"); `_flash_v100_small_query_prefill
_as_decode` is likewise built for FULL replay. Pure `FULL` crashed earlier because it tried to capture
*prefill*. **`cudagraph_mode: FULL_AND_PIECEWISE`** captures FULL graphs for **decode** batches (the
graph-safe path) and PIECEWISE for mixed/prefill → no crash, no rewrite. Verified live (decode-FULL
capture succeeds, 0.68 GiB, 0 restarts):
| metric | PIECEWISE (#20) | FULL_AND_PIECEWISE (#19) |
|---|---|---|
| single-stream | 33.5 | **34.1** (+1.8%) |
| conc=2 agg | 36.6 | **47.0** (+28%) |
| conc=4 agg | 105.7 | 107.3 |
| errors / restarts | 0 / 0 | **0 / 0** |
Strictly better than PIECEWISE; now the committed config in both 12B manifests. The modest single-
stream delta confirms decode-attention launch overhead is small — which means the once-feared
**prefill** device-resident rewrite would be *even lower* ROI (prefill isn't the single-stream
bottleneck), so it stays unbuilt by choice, not blocked.

## Gated on 31B-MTP redeploy (not 12B tasks)

### #14 — hd512 small-query verify kernel (31B-MTP) — gated on 31B redeploy
Small-query-as-decode verify path is numerically imprecise at hd512 (rare wrong-accepts), currently
env-disabled (`VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q=0`) → lossless but slower verify. Fix = debug-
compare (`VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE=1`) + a `flash_decode_paged.cu` D=512 fix. Deep CUDA
work, **optimization-only (MTP already lossless)**, requires the now-offline 31B MTP redeployed. Not
a 12B task.

### #15 — raise MTP draft acceptance (16% → 50-70%) — gated on 31B redeploy
Needs a plain-transformers `assistant_model=` reference (E2B pair) to localize the gap.
Optimization-only (MTP is lossless, just a modest 1.19×); requires the 31B MTP up + a reference
harness. Not a 12B task.

## Bottom line
- **Every live-12B follow-up is resolved:** single-slot 21 → 34.1 tok/s (#17 PIECEWISE → #19
  FULL_AND_PIECEWISE), concurrency unblocked with no single-stream cost (#20), decode attention now
  in FULL graphs (#19), RoPE primitive hardened (#16), overlay baked into a self-contained image and
  validated live (#18).
- **Live config:** `gemma4-12b` (omni, fp16, TP2, 262K) on the baked image
  `onecat-vllm:v1-v100-110-gemma12b`, `cudagraph_mode: FULL_AND_PIECEWISE` + `cudagraph_copy_inputs`.
  34.1 tok/s single-stream, 0 errors / 0 restarts through conc=4.
- **#14/#15** are the only open follow-ups and are **out of 12B scope** — 31B-MTP optimizations
  (already lossless), gated on redeploying the replaced 31B; worth doing only if the 31B-MTP returns.
