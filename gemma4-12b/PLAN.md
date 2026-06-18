# Plan: Gemma-4-12B Omni (text + vision + audio) on 1Cat-vLLM / V100

**Goal:** serve **`google/gemma-4-12B-it`** (the omni `gemma4_unified` model) on the homelab V100s,
replacing the gemma4-31b deployment. **fp16 (no quant needed — 12B fits)**, **text+vision first,
audio as a follow-up** (per operator). Branch: **`gemma4-12b-omni-v100`** (off `gemma4-31b-awq-v100`
to inherit the gemma4 base + hd512 flash-v100 kernels).

## Why this is tractable (de-risked 2026-06-04)
- `Gemma4UnifiedForConditionalGeneration` (model_type `gemma4_unified`) **subclasses our already-
  ported `gemma4_mm.Gemma4ForConditionalGeneration`** and uses `gemma4.Gemma4ForCausalLM` as the LM.
  Upstream `gemma4_unified.py` is **469 lines** and imports 12 symbols from our `gemma4_mm.py` — all
  present (verified). So the port is mostly *reuse*.
- **Vision is encoder-free**: `Gemma4UnifiedVisionEmbedder` = LayerNorm → dense → factorized 2D
  pos-emb (no SigLIP ViT). Audio is likewise encoder-free (raw frame features → projection). So **no
  heavy mm encoder runs on V100** — the historical "vision tower on sm70" risk is gone.
- Same **hd512 global layers** → the flash-v100 head_dim-512 kernel work carries over unchanged.
- **fp16 ≈ 24 GB** → fits comfortably on the V100s; **no AWQ checkpoint needed** (none exists anyway).
- Not gated; 1 safetensors file.

## Port (done — code complete, loads)
- Copy `gemma4_unified.py` → `vllm/model_executor/models/`; register
  `Gemma4UnifiedForConditionalGeneration` in registry.py. All imports resolve
  (`init_vllm_registered_model`, `MultiModelKeys.from_string_field`,
  `AutoWeightsLoader(ignore_unexpected_prefixes=)`, `VideoDummyOptions`).
- New config module `vllm/transformers_utils/configs/gemma4_unified.py`: `Gemma4UnifiedConfig` +
  `Gemma4UnifiedTextConfig` (subclasses the known `gemma4_text`) + `Gemma4UnifiedVisionConfig` +
  `Gemma4UnifiedAudioConfig` (transformers 5.5 doesn't know these; the checkpoint targets tf 5.10).
  Registered in configs/__init__ + config.py `_CONFIG_REGISTRY`. **Verified**: instantiates from the
  real config.json (text 48L/3840/hd256+512, vision patch-48/mm_embed-3840, audio-640, tokens).

## Deploy + validate (in progress)
- Hydrate `google/gemma-4-12B-it` → PVC (`27-hydrate-gemma4-12b.yaml`, ~24 GB, fp16).
- Scale down gemma4-31b + gemma4-31b-mtp (free NUMA0 V100s); deploy `gemma4-12b` TP4, fp16,
  `--limit-mm-per-prompt '{"image":N,"video":0,"audio":0}'` (audio off for phase 1). Overlay tarball
  = the gemma4 base files + `gemma4_unified.py` (model + config).
- **Validate:** (1) text chat coherence; (2) image understanding (send an image_url, check the
  description); (3) benchmark. Phase 2 (follow-up): enable audio (`audio:N`), validate ASR/audio QA.

## Open risks
- The mm **processor** (`Gemma4MultiModalProcessor` / `Gemma4UnifiedProcessingInfo`) running text+image
  end-to-end on this transformers/vLLM combo is the main untested path (config + model load are
  verified; the image-token expansion + patchification at inference is not yet).
- Audio (phase 2): the audio processor + `audio_samples_per_token` framing on V100 — untested.
- TP4 may be overkill for a 12B (TP2 would free 2 GPUs + cut comms) — a follow-up optimization.
