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

## Open
- **Audio**: feature extractor + `embed_audio` vendored/constructed but not run end-to-end (likely
  needs the same fp32 treatment for any large-weight audio projection + validation).
- **Video**: enabled (reuses the image path) but not yet validated with a real clip.
- The fp32-vision and bidi-gate patches are V100-fork-specific; candidates to upstream into the fork.
