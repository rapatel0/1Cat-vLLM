# gemma-4-12B omni (image/audio/video) enablement on V100 — status

Goal: enable image + audio + video on the gemma4-12b deploy (currently text-only). The model is
omni and the AWQ checkpoint keeps the vision/audio embedders unquantized (fp16), so the weights
are present. This is the status after a deep enablement pass. **Image is ~90% there but blocked on
a vision-numerics correctness bug.**

## What works (validated)
1. **HF processor vendoring.** transformers 5.5 has the *base* classes the gemma4_unified processors
   need (TorchvisionBackend, SequenceFeatureExtractor, BaseImageProcessorFast, MultiModalData,
   VideosKwargs, VideoInput), so the upstream (transformers 5.13) gemma4_unified processing files
   vendor cleanly. Vendored into the overlay at `transformers/models/gemma4_unified/`
   (image_processing, feature_extraction, video_processing, processing + trimmed __init__; **no
   modeling** — vLLM uses its own). Source saved in `omni-port/transformers_gemma4_unified/`.
   - `Gemma4UnifiedProcessor.from_pretrained(ckpt)` constructs cleanly (image_processor +
     audio feature_extractor resolved).
   - **Image preprocessing is correct**: a 336×336 image →
     `pixel_values (1, 280, 6912)` (280 soft tokens × 48·48·3 raw patches — the encoder-free design)
     + `image_position_ids (1, 280, 2)`. Matches what the port's `_process_image_input` consumes.
2. **Port mm-path bugs fixed.** The never-exercised mm path called `self.ctx.get_merged_mm_kwargs()`
   (4 sites: gemma4_unified.py ×1, gemma4_mm.py ×3) — a method the deployed vLLM 1.1.0
   `InputProcessingContext` lacks. It was only the optional `max_soft_tokens` override; neutralized
   (use `dict(kwargs)` / config default). Fixed files in `omni-port/{gemma4_unified,gemma4_mm}.py`.
   With this, the engine **boots with image enabled**, profiles vision ("4 image items"), captures
   graphs, and serves.
3. **End-to-end pipeline runs.** An `image_url` request is processed correctly: placeholders expand
   (256 image embeds + boundaries), `pixel_values` reach the model, no Python errors.

## The blocker: vision output is wrong (empty / special-token gibberish)
With image enabled, an image request returns **40 generated tokens that decode to empty string**
(all special tokens) — `finish=length`, `completion_tokens=40`, `content=''`. The model "sees" the
image but produces garbage.

Isolation done (what it is NOT):
- **Not the prefill fallback.** A 422-token *text* prefill (same dense + paged-KV-gather chunked
  paths) returns a correct, coherent answer. So the long-prefill numerics are fine.
- **Not weight loading.** The checkpoint has all vision weights
  (`vision_embedder.{patch_dense,patch_ln1,patch_ln2,pos_embedding}`,
  `embed_vision.embedding_projection`) with real values (non-random); the WeightsMapper maps
  `model.vision_embedder.` → `vision_embedder.` correctly; no load warnings.
- ⇒ The bug is in the **vision forward / embed-merge / hd512 vision-token attention numerics**.

Also hit + worked around on the way: the **V100 hd512 prefill shared-memory crash**
(`Shared memory exceeds 96KB: 118656 bytes`, `fused_mha_forward*.cu`). `KernelConfig<512>::TOTAL_SMEM`
is a compile-time 118 KB > V100's 96 KB, so the hd512 prefill kernel can't run. Text dodges it
(short prefills / Triton fallback); the image's longer prefill hit it. Worked around with
`VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` (routes to the gather+fallback path → no crash). This is
the same hd512 hardness family as #14/#19.

## To resume (what fixing image needs)
A **reference comparison**: run the same image through HF `transformers` `gemma4_unified` (5.13) to
get golden `embed_vision` outputs + first-layer hidden states, then diff against the vLLM port's
`_process_image_input` / embed-merge / the hd512 attention over the 256 image tokens. The garbage
output means one of: (a) the encoder-free vision projection is computed differently than HF, (b) the
embeds are merged at the wrong positions, or (c) the hd512 attention over vision tokens (with the
paged-prefill workaround) is numerically off. The reference harness needs the full HF model loadable
(transformers 5.13 + the unquantized `google/gemma-4-12B-it-qat-q4_0-unquantized`).

## Audio / video
Gated behind image. The audio feature extractor + video processor vendored and construct, but
untested end-to-end (and would hit the same hd512 prefill workaround + whatever the vision fix is).

## Where the work lives
- Overlay (`/models/gemma12b-overlay.tar.gz`, PVC): contains the vendored processors + fixed port —
  deployable as-is for continued debugging (enable image + `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1`).
- Repo: `omni-port/` (vendored processors + fixed gemma4_unified.py / gemma4_mm.py).
- Live `gemma4-12b-awq` deploy: **restored to validated stable text-only** (AWQ W4A16, 60 tok/s,
  mns4, image off) — not left in the broken mm state.
