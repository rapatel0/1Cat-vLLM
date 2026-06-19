# gemma-4-12B omni (image) on V100 — WORKING

**Image input works** on the AWQ `gemma4-12b-awq` deploy (V100, TP2, fp16 LM + fp32 vision):
```
red 336×336        -> "Red"
blue circle/white  -> "A blue circle is centered on a white background."
```
Text decode stays at **58 tok/s** (cudagraphs intact). Validated end-to-end through the OpenAI
`image_url` chat API. Video enabled (frame path reuses image); audio vendored but unvalidated.

## The fix chain (what it took)
Enabling vision required six independent fixes. Clean sources in `omni-port/`.

1. **Vendor the HF processors** (transformers 5.5 lacks `gemma4_unified`). The 5.13 processing files
   vendor cleanly (all base classes exist in 5.5). Image preprocessing verified: 336×336 →
   `pixel_values (1,280,6912)` + `image_position_ids`.
2. **mm-path port bugs** — `self.ctx.get_merged_mm_kwargs()` (4 sites, gemma4_unified.py +
   gemma4_mm.py) doesn't exist in vLLM 1.1.0; it was the optional `max_soft_tokens` override →
   neutralized.
3. **hd512 prefill smem crash** — the image-extended prefill exceeds V100's 96 KB smem in the hd512
   global-layer kernel (`KernelConfig<512>::TOTAL_SMEM`=118 KB). Worked around with
   `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` (gather+fallback path).
4. **`is_mm_prefix_lm`** (vllm/config/model.py) hardcoded `("gemma3","molmo2","paligemma")` — gemma4
   wasn't recognised, so the bidirectional-vision mask (`mm_prefix_range`) was never built. Patched
   to also accept any model whose text config sets `use_bidirectional_attention`. Plus a worker-side
   recompute in gpu_model_runner (the cached_property didn't survive multiproc serialization).
5. **Bidirectional vision attention** — gemma4 needs the image span to attend bidirectionally in
   *sliding* layers, causal in *full-attn* layers, driven by `mm_prefix_range`. The flash_attn_v100
   kernels are causal-only and ignored it. Two parts: this fork shares **one** attn_metadata object
   across both layer types, so `_clear_mm_prefix_for_full_attn_layers` is now a **no-op** (clearing
   would wipe the shared object for sliding layers too); the per-layer decision moved into the
   flash_attn_v100 impl (`_triton_with_bidi_gate`), which routes masked calls to the Triton impl
   (the only one honouring `mm_prefix_range`) and temporarily nulls the mask for full-attn layers.
   Text-only requests (empty ranges) stay on the fast flash path.
6. **THE root cause — fp16 overflow → NaN.** Even with 1–5, output was empty/garbage. Direct
   instrumentation showed `pixel_values` healthy but the **vision embedder output was NaN**:
   `patch_dense.weight` has std ~15.6, and the encoder-free projection over 6912 dims **overflows
   fp16** (gemma is bf16-native). Fix: run the (unquantized, tiny) `vision_embedder` in **fp32**
   (`gemma4_unified.load_weights` casts it; the fp32 output is downcast before the projection).

## Live config (homelab `30-deployment-gemma4-12b-awq.yaml`)
TP2, `--dtype float16`, AWQ W4A16 (CT→sm70), `FULL_AND_PIECEWISE` cudagraphs,
`VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1`, `--limit-mm-per-prompt {image:2,video:1,audio:0}`,
gpu-mem 0.90, max-num-seqs 4. Overlay (`/models/gemma12b-overlay.tar.gz`) carries the patched
vLLM files + vendored processors (21 files).

## Video — WORKING
Validated with a synthetic mp4 (red/blue/green frames): the model lists "Red, Dark Blue, Dark Green".
Reuses the vision path (frames → the fp32 vision embedder). Needs `librosa`/`soundfile` + `cv2`
(opencv, already in the image) for decode. No extra fixes beyond the image work.

## Audio — pipeline functional, perception NOT working
Deps: add `librosa soundfile` to the boot install (vLLM raises "Please install vllm[audio]" otherwise).
After that the pipeline runs end-to-end: the vendored `Gemma4UnifiedAudioFeatureExtractor` produces raw
waveform frames `(N,640)`, `embed_audio` projects them (matches the HF `get_audio_features` exactly:
RMSNorm→Linear), embeds are **healthy** (std ~1.3, no NaN), ~25 tokens/sec are injected (prompt tokens
grow), and audio is excluded from the vision bidi span (runner modality filter: image/video only, since
`use_bidirectional_attention='vision'`). **But the model does not perceive the audio** — it returns
"SILENT" for both a full-scale tone and true silence, i.e. cannot distinguish them. The merge path is
the same one image uses (which works), and the embedder matches HF, so this looks like the **QAT
12B checkpoint's audio capability being weak/untrained** (or perception only emerging for real speech,
which couldn't be synthesized here) rather than a code defect. Needs the full HF model as a reference
(or a known audio-strong checkpoint) to settle.

## Notes
- The fp32-vision, bidi-gate, is_mm_prefix_lm and audio-modality-filter patches are V100-fork-specific;
  candidates to upstream into the fork.
