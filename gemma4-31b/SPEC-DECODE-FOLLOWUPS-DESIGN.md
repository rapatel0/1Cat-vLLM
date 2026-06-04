# Design: Gemma-4-31B MTP Speculative Decoding — Follow-up Optimizations

**Status:** spec decode is **working + lossless** on 4× V100 (TP4) as of 2026-06-04 (see
`SPEC-DECODE-PLAN.md` for the build + the two root-cause fixes). This document scopes the follow-up
work to (a) remove the correctness workaround and (b) raise the speedup from the current modest 1.19×.

## 1. Current state (baseline)

| Property | Value |
|---|---|
| Model / quant | Gemma-4-31B-it Dense, AWQ 4-bit (gs64), fp16 compute |
| Drafter | `gemma-4-31B-it-assistant` (~0.5B, native MTP, KV-shared), `num_speculative_tokens=4` |
| Hardware | 4× V100 32 GB (sm_70), NUMA0, TP4, enforce-eager |
| Context | 8,192 (native 262,144); KV 84,448 tok; 11.2× conc headroom; `max_num_seqs=4` |
| VRAM | ~28.3 GB / 32 GB per GPU (~86%) |
| Throughput | 21.9 tok/s single-stream (vs 18.4 no-spec = **1.19×**); 82.9 tok/s @ conc 4 |
| Acceptance | 15.8% per-token, mean accept length **1.63** tok/step (decay 56/23/13/7) |
| Correctness | **Lossless** (byte-identical to no-spec, verified) |

**Two known compromises shipped to get to lossless:**
- The draft RoPE fix uses a **dummy key** workaround in `gemma4_mtp.py` (the V100 in-place rotary op
  leaves the query un-rotated when `key=None`).
- Verify is forced onto the **paged-prefill** kernel via `VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q=0`
  because the faster **small-query-as-decode** verify path is numerically imprecise at hd512.

Manifests: `homelab.ds4/manifests/qwen36-27b-vllm-sm70/26-deployment-gemma4-31b-mtp.yaml` (serving) +
`25-hydrate-gemma4-assistant.yaml` (drafter download). Code on branch `gemma4-31b-awq-v100`; delivered
as the PVC overlay tarball `/models/gemma-overlay.tar.gz`.

## 2. Goals / non-goals

**Goals:** keep losslessness; remove the env/kernel workarounds; raise effective speedup toward the
~3× that MTP targets upstream; improve operational reproducibility.
**Non-goals:** changing the target model/quant; multimodal (text-only by design); 256K-context tuning
(separate effort).

## 3. Follow-up tasks (prioritized)

### P0 — Fix the hd512 small-query verify kernel
**Problem.** `_flash_v100_small_query_prefill_as_decode` (the MTP-verify path: a tiny multi-token
query span over the KV prefix) is numerically imprecise at head_dim 512, so the target's verify
greedy occasionally differs from the true greedy → a wrong draft token is accepted → lossy. Currently
disabled via env, which routes verify to the **paged-prefill** kernel — correct but slower (the
small-q path exists precisely because paged-prefill hits the SM70 96 KB smem limit at hd512 long
context, and is heavier per-call for the small-q shape).
**Why it matters.** (1) removes the env workaround → losslessness without config foot-guns; (2) the
small-q path is the *fast* verify path → directly lifts spec throughput; (3) unblocks longer context
(paged-prefill smem ceiling).
**Approach.** Enable `VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE=1` and diff small-q-as-decode vs the dense
reference at hd512 to find the first divergent row/value. Likely suspects: the per-row
`decode_seq_lens`/`block_table` construction interacting with the D=512 partition kernel in
`flash_decode_paged.cu`, or the unified-K/V (V=K) global-layer cache read. Candidate fixes: correct
the partition/reduce for the multi-row verify shape at D=512, or give it a small-tile D=512 config.
**Risk.** Volta CUDA-kernel debugging + nvcc rebuild cycles (the runtime image needs nvcc wired — see
memory `onecat-image-nvcc-build`). **Effort:** M–H.

### P1 — Raise draft acceptance (the speedup limiter)
**Problem.** 15.8% per-token / mean 1.63 is low for a 0.5B MTP assistant (upstream reports up to ~3×,
implying ~50–70% pos-0 acceptance). The mechanism is correct (lossless), so this is a *quality* gap.
**Hypotheses.** (a) the dummy-key RoPE rotates q correctly but perhaps not bit-identically to how the
assistant was trained; (b) the constant-position drafting / KV-share read is subtly off in a way that
hurts draft quality without breaking losslessness; (c) target AWQ vs draft fp16 distribution mismatch;
(d) sampling/threshold details (`num_speculative_tokens=4` may be past the useful depth — try 2–3).
**Approach.** Stand up a `gemma-4-E2B-it` + its assistant pair (fits in memory) and compare the vLLM
drafter's per-step logits against a plain-transformers `assistant_model=` reference to quantify the
gap and localize it; sweep `num_speculative_tokens` ∈ {2,3,4,6}. **Risk.** May surface another subtle
forward bug. **Effort:** M.

### P1 — Fix the RoPE null-key at the primitive level (remove the workaround)
**Problem.** The dummy-key workaround lives in `gemma4_mtp.py`. The real defect is the V100 fork's
in-place `torch.ops._C.rotary_embedding` (and/or `forward_cuda`) not rotating `query` when `key=None`.
Any future KV-shared/Q-only drafter will hit the same trap.
**Approach.** Make `forward_cuda` rotate the query regardless of key presence (fix the wrapper to
handle `key=None`, or patch the C++ op). Then revert the dummy-key in `gemma4_mtp.py`. **Risk.** Core
primitive used by *every* model → needs a regression pass (a few non-MTP models, q/k parity).
**Effort:** L–M.

### P2 — Throughput tuning (config-level)
- **Raise `max_num_seqs`** 4 → 8–11 (KV supports 11.2×): more batched aggregate. Validate VRAM.
  Effort: trivial.
- **CUDA graphs at hd512**: currently `enforce-eager`. Validate cudagraph capture with spec decode at
  hd512 (untested) → lifts per-stream. Effort: M (capture validation).
- **Context window**: 8,192 → higher (native 262,144) at a KV/VRAM trade-off; gated by the P0 smem
  ceiling for long-context verify. Effort: trivial config once P0 lands.

### P3 — Operational hardening
- **Bake the overlay into the image.** Currently 16-file PVC tarball extracted at startup. Bake the
  `gemma4-31b-awq-v100` changes + transformers 5.5 + the hd512 kernel into `onecat-vllm` for a
  rebuild-free, reproducible deploy. Effort: L.
- **Decide active service.** spec (`gemma4-31b-mtp`, 1.19× single-stream) vs no-spec
  (`gemma4-31b`, higher batched aggregate at conc 8). Pick per workload; both manifests committed.

## 4. Sequencing & expected payoff

1. **P0 (small-q hd512)** — biggest single lever: lossless without the workaround **and** faster
   verify. Do first; it also unblocks long-context.
2. **P1 (acceptance)** in parallel — the E2B reference harness is reusable and quantifies how much
   headroom exists; pair with `num_speculative_tokens` sweep.
3. **P1 (RoPE primitive)** — small, removes a latent foot-gun; can land anytime.
4. **P2 config tuning** — quick wins after P0 (graphs, max-num-seqs, context).
5. **P3** — once the above stabilize, bake an image and choose the default service.

**Target:** P0 + P1 together should move the single-stream speedup from 1.19× toward the 1.5–2.5×
range typical of a working MTP drafter on this target, while keeping losslessness.

## 5. Risks / open questions
- The hd512 small-q kernel may have a structural Volta limitation (smem/partition) that resists a
  clean fix — fallback is to keep paged-prefill verify and instead push acceptance (P1) for speedup.
- If P1 reveals the draft quality is inherently limited on V100 (quant/precision), the realistic
  ceiling may be ~1.5×; that's still a lossless win and worth shipping.
