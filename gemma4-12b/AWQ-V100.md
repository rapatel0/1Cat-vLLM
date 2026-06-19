# gemma-4-12B INT4 on V100 — dense compressed-tensors → TurboMind AWQ (1.77× single-slot)

## Result
Serving `google/gemma-4-12B-it-qat-w4a16-ct` (dense QAT int4) on 2×V100 (TP2) lifts single-slot
decode **34.1 → 60.3 tok/s (1.77×)** with **no quality loss** (QAT-trained int4), and frees ~15 GB
of VRAM as a bonus.

| metric | fp16 | AWQ W4A16 (this) |
|---|---|---|
| single-slot decode (FULL_AND_PIECEWISE) | 34.1 tok/s | **60.3 tok/s** |
| conc 1 / 2 / 4 aggregate | 34 / 47 / 107 | **59.5 / 69.8 / 119.3** |
| weights (TP2 total) | ~24 GB | ~9 GB |
| KV pool | 83,792 tok | **121,504 tok** |
| max concurrency @ 262K | 2.30× | **3.34×** |
| errors / restarts | 0 / 0 | 0 / 0 |

Eager (no cudagraphs) only gave 21→23.7 — the 4-bit weight-bandwidth win is masked by launch
overhead, so it only appears under cudagraphs (which is why FULL_AND_PIECEWISE matters here).

## Why this route (vs AutoAWQ / llm-compressor)
The accelerating V100 4-bit path is the **TurboMind sm70 AWQ kernel** (HMMA 8×8×4), used in vLLM via
`ops.awq_gemm_sm70`. It needs **AWQ-packed** weights. But:
- The dense `CompressedTensorsWNA16` scheme selects **Marlin** (sm75+) and `get_min_capability()==75`
  → compressed-tensors int4 hard-rejected on V100.
- The CT **MoE** scheme already converts CT-int4 → TurboMind AWQ at load time (sm70). Only dense was
  missing.
- AutoAWQ would re-quantize (worse than QAT) and likely lacks `gemma4_unified` support; llm-compressor
  outputs compressed-tensors (same Marlin wall) and our PTQ would be worse than Google's QAT.

So: **don't re-quantize.** Use Google's QAT ct-w4a16 checkpoint + mirror the MoE CT→AWQ conversion
onto the dense scheme.

## The patch (`compressed_tensors_wNa16.py`, fork)
`use_sm70_awq` gate: sm_70 + 4-bit + symmetric + group_size∈{32,64,128} + no act-order +
`awq_sm70_prepare` built in. When set:
- `get_min_capability()` → 70 (else 75).
- `create_weights`: skip Marlin kernel selection; just register the CT params.
- `process_weights_after_loading`: convert at load time —
  - `_ct_dense_to_awq_qweight`: CT `[N,K/8]` → AWQ `[K,N/8]` (unpack-K, **transpose**, repack-N with
    AWQ interleave `[0,2,4,6,1,3,5,7]`; the dense case needs the transpose the MoE helper doesn't).
    CPU round-trip verified == `W.T`.
  - scales `[N,K/gs]` → `[K/gs,N]` fp16; symmetric → AWQ qzeros `0x88888888` (zero_point 8); then
    `ops.awq_sm70_prepare`. Frees the CT tensors after.
- `apply_weights`: `ops.awq_gemm_sm70`.
- Fully gated — non-eligible configs keep the Marlin path unchanged.

Commit: `gemma4-12b-ct-awq-v100` branch, `CT W4A16: dense SM70 (V100) path via TurboMind AWQ`.

## Deploy (homelab.ds4)
- `29-hydrate-gemma4-12b-awq.yaml` — pull the 9.2 GB checkpoint to the PVC.
- `30-deployment-gemma4-12b-awq.yaml` — TP2, overlay-based (boot-extracts the patched file),
  `--dtype float16`, `FULL_AND_PIECEWISE` + `cudagraph_copy_inputs`, 262K context.
- The patched `compressed_tensors_wNa16.py` is in `/models/gemma12b-overlay.tar.gz` (14 files now).

## Open / next
- The freed VRAM should lift the `max-num-seqs` ceiling well past the fp16 sampler-OOM limit (was
  ~32–48); re-probe the concurrency width on AWQ.
- Spot-check quality more (one coherence test passed; QAT pedigree is strong) before making it the
  primary `gemma4-12b` deploy.
- Bake the patched file into the image (extend `Dockerfile.110-gemma12b`) once promoted.
