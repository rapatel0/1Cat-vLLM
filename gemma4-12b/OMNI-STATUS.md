# gemma-4-12B omni (image) on V100 — WORKING

**Image input works** on the AWQ `gemma4-12b-awq` deploy (V100, TP2, fp16 LM + fp32 vision):
```
red 336×336        -> "Red"
blue circle/white  -> "A blue circle is centered on a white background."
```
Text decode stays at **58 tok/s** (cudagraphs intact). Validated end-to-end through the OpenAI
`image_url` chat API. Video enabled (frame path reuses image). **Audio enabled and validated** — the
pipeline is byte-identical to HF; perception requires a multimodal system prompt (see Audio section).

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
`VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1`, `--limit-mm-per-prompt {image:2,video:1,audio:1}`,
`--trust-request-chat-template` (lets audio clients pass a multimodal system prompt / Gemma-format
template per request), gpu-mem 0.90, max-num-seqs 4. Overlay (`/models/gemma12b-overlay.tar.gz`)
carries the patched vLLM files + vendored processors (21 files).

## Video — WORKING
Validated with a synthetic mp4 (red/blue/green frames): the model lists "Red, Dark Blue, Dark Green".
Reuses the vision path (frames → the fp32 vision embedder). Needs `librosa`/`soundfile` + `cv2`
(opencv, already in the image) for decode. No extra fixes beyond the image work.

## Audio — pipeline CORRECT; "blindness" was a chat-template refusal persona, not a vLLM bug
Deps: add `librosa soundfile` to the boot install (vLLM raises "Please install vllm[audio]" otherwise).
Pipeline runs end-to-end: the vendored `Gemma4UnifiedAudioFeatureExtractor` chunks the raw waveform into
frames `(N,640)` (640 samples/token), `embed_audio` projects them (RMSNorm→Linear, matches HF
`get_audio_features` exactly), padding stripped via the mask, embeds merged like image, audio kept causal
(runner modality filter — `use_bidirectional_attention='vision'` excludes audio).

**The vLLM port is faithful — proven byte-identical to HF.** An HF reference harness (transformers 5.5 +
vendored 5.13 modeling, same w4a16 weights) and an offline vLLM run on the **same** kernels/env both give
`embed_audio` output **std 1.3352, absmax 16.30** for a 440 Hz tone (HF: 1.3351 / 16.25). Audio token
count scales correctly with duration (text 15 → +27 for 1 s → +102 for 4 s). Audio decode (librosa) is
identical to the offline raw-array path (std 0.636, absmax 0.9). So feature-extraction, embedding, scale,
merge, masking and token count are all correct.

**Root cause of the earlier "returns SILENT / I-cannot-hear" result: a learned refusal persona under the
checkpoint's default chat template, *not* non-perception.** Findings:
- This checkpoint's `chat_template.jinja` uses **harmony** turn delimiters (`<|turn>`=105, `<turn|>`=106,
  `<|channel>thought`); `<start_of_turn>`/`<end_of_turn>` are **not** special tokens here (split into 7).
- Under the harmony template with no system message the model replies *"I am a text-based AI, I cannot
  hear"* — even though the audio embeds are present and identical to HF.
- A **system prompt** ("you can hear audio; listen and describe") **unlocks perception**: 440 Hz →
  *"high-pitched, electronic … resembles a synthesized alarm / siren"*, matching the HF reference ceiling.
  Text and image quality are unaffected by the system prompt (TEXT→"Hello!", IMAGE→"Green").
- The Gemma `<start_of_turn>` template also perceives without a system prompt but **degrades text-only**
  quality, so it is not used as the default; instead the deploy adds `--trust-request-chat-template` so a
  client may override per request, and audio clients pass a multimodal system prompt.

**Perception quality ceiling (checkpoint, not code):** on *synthetic* tones the model is weakly
discriminative — 80 Hz / 3000 Hz / white-noise all map toward "high-pitched electronic alarm/whine". This
matches the HF reference (same limited behaviour) and is expected for a model trained on speech/music fed
pure sine tones; it is a checkpoint-capability ceiling, not a port defect. Real-speech/music perception
was not testable here (no sample, no pod network).

**Boot fix:** the audio dummy-input builder referenced `feature_extractor.fft_length` (a conformer-era
attr the encoder-free unified FE lacks) → crashed dummy profiling whenever `--limit-mm-per-prompt`
included `audio>0`. Fixed to `audio_seq_length * audio_samples_per_token` (gemma4_mm.py). The deploy now
boots with `audio:1`.

## Notes
- The fp32-vision, bidi-gate, is_mm_prefix_lm and audio-modality-filter patches are V100-fork-specific;
  candidates to upstream into the fork.
