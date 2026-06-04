# Plan: Gemma-4-31B MTP speculative decoding on 1Cat-vLLM / V100

**Goal:** add Gemma-4's native MTP (multi-token prediction) speculative decoding to the working
`gemma4-31b` AWQ deployment, via vLLM's `--speculative-config`. Target serve line:
```
--speculative-config '{"model":"/models/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
```
Expected ~up to 3x decode speedup (lossless). Branch: continue on `gemma4-31b-awq-v100`.

## Feasibility (de-risked 2026-06-04) — GREEN on the model, AMBER on the integration

Gemma-4 MTP is **activation-coupled, two-checkpoint** spec decode (see memory `gemma4-mtp-coupled-head`).
The drafter `google/gemma-4-31B-it-assistant` (arch `Gemma4AssistantForCausalLM`, model_type
`gemma4_assistant`):
- **KV-shared**: Q-only attention layers, reads the *target's* KV cache (no own K/V).
- **Dimension bridge**: draft `hidden_size=1024` ≠ `backbone_hidden_size=5376`; `pre_projection`
  (2·5376→1024) and `post_projection` (1024→5376) bridge them. Forward returns **two** hidden states
  per step (draft-dim for logits, backbone-dim for the feedback buffer).
- 4 MTP layers, vocab 262144, same hybrid attention (sliding hd256 + global hd512 p-RoPE).
- Checkpoint downloaded to PVC: **`/models/gemma-4-31B-it-assistant`** (277 MB, BF16, NOT gated).
  Hydrate job: `homelab.ds4/.../25-hydrate-gemma4-assistant.yaml`.

### What's clean (low risk)
- **Model module**: upstream `gemma4_mtp.py` (627 lines, class `Gemma4MTP`) ports cleanly — only
  depends on `Gemma4MLP` + `_get_text_config` from our already-ported `gemma4.py`. Saved for
  reference at `gemma4-31b/upstream-ref/gemma4_mtp.py`. Has centroids-masking embedder (E2B/E4B
  feature, gated on `use_ordered_embeddings` — likely off for 31B).
- **Registry**: add `"Gemma4AssistantForCausalLM": ("gemma4_mtp", "Gemma4MTP")` to
  `_SPECULATIVE_DECODING_MODELS`.
- **Method detect**: add `"gemma4_assistant"` to `MTPModelTypes` in `vllm/config/speculative.py`
  → auto-detects `method="mtp"`.
- **V100 attention**: the KV-shared Q-only attention uses the standard `Attention` layer; the V100
  backend `FlashAttnV100Impl(TritonAttentionImpl)` inherits `kv_sharing_target_layer_name` support
  via `*args/**kwargs`. Global layers are hd512 → our rebuilt flash-v100 512 kernel covers it.
  (Unvalidated in a KV-shared decode.)

### What's the real work (AMBER / high risk)
1. **Config class** — transformers 5.5.0 does **NOT** recognize `gemma4_assistant`
   (`AutoConfig.from_pretrained` raises "Transformers does not recognize this architecture"). The
   main gemma4 model works on 5.5.0, but the *assistant* arch is newer. Options: (a) write a local
   `vllm/transformers_utils/configs/gemma4_assistant.py` + register it (the `qwen3_5` pattern) —
   preferred, no blast radius; (b) bump transformers further (risks the working main model). →
   **go with (a).**
2. **Proposer + KV-sharing setup + dual-dim buffer** — the gemma4 integration (`Gemma4Proposer`,
   `_setup_gemma4_kv_sharing`, the backbone-dim feedback buffer) lives in upstream
   **`vllm/v1/worker/gpu_model_runner.py`** (32 gemma4/kv-share hits there; **0** in eagle.py) and is
   **absent from the fork**. `gpu_model_runner.py` is the most-diverged file in this V100 fork
   (V100 attention paths, qwen35-MTP defaults, etc.), so this port must be hand-reconciled, not
   copied. The fork's `SpecDecodeBaseProposer.propose()` already unpacks a 2-tuple return (Qwen MTP
   path) and `EagleProposer` passes hidden states to the model, but the fork allocates a **single**
   draft-dim hidden buffer — gemma4 needs the backbone-dim (5376) feedback path + the KV-sharing
   layer wiring. This is the crux: ~2–3 days + 1–2 days V100 validation.

## Phased plan (de-risk the runner port last)
- **Phase A — DONE ✅ (2026-06-04, commit a63d357e5).** config class
  (`vllm/transformers_utils/configs/gemma4_assistant.py` + registered in configs/__init__ +
  config.py), ported `gemma4_mtp.py`, registry + MTPModelTypes entries. **Validated** on a no-GPU
  pod: config loads as Gemma4AssistantConfig (backbone 5376; text hidden 1024 / 4 layers / vocab
  262144 / head_dim 256+512 / layer_types 3×sliding+1×full), registry resolves
  Gemma4AssistantForCausalLM, drafter module imports (Gemma4MTP / MultiTokenPredictor /
  MTPAttention), method auto-detects mtp. Note: the drafter `from .gemma4 import ...` means the
  spec-decode deploy must overlay the main-port gemma4 files too (already in the deploy ConfigMap).
- **Phase B (the crux) — MAPPED & RE-SCOPED SMALLER (2026-06-04).** The fork turns out to already
  have most of the machinery, so this is "port one file + 4 small wirings," not a runner rewrite:
  - **Already in the fork (no work):** KV-sharing kv-cache-group infra
    (`add_kv_sharing_layers_to_kv_cache_groups`, `maybe_add_kv_sharing_layers_to_kv_cache_groups`,
    fast-prefill), the base `build_per_group_and_layer_attn_metadata` (calls `build_for_drafting`,
    the same V100 path Qwen MTP already uses), `_maybe_share_lm_head`, `validate_same_kv_cache_group`,
    `_draft_attn_layer_names`/`draft_attn_groups`, `pass_hidden_states_to_model`, and every import the
    proposer needs (`replace`, `AttentionLayerBase`, `CommonAttentionMetadata`, `KVCacheSpec`,
    `UniformTypeKVCacheSpecs`, `AttentionGroup`). The base `model_returns_tuple()` already unpacks the
    2-tuple return (Qwen MTP path).
  - **To port (saved in `upstream-ref/`):**
    1. `vllm/v1/spec_decode/gemma4.py` — `Gemma4Proposer` (340 lines). Only reconciliation: upstream
       imports `SpecDecodeBaseProposer` from `llm_base_proposer`; the fork keeps it in `eagle.py` →
       change that import. It overrides `model_returns_tuple`, `build_per_group_and_layer_attn_metadata`
       (per-group block-table swap), `initialize_attn_backend` (per-head-dim groups),
       `_setup_gemma4_kv_sharing` (maps each draft layer to the last non-shared target layer of the
       same sliding/full type), `_create_draft_vllm_config` (carry the TRITON/V100 backend through),
       `_maybe_share_lm_head` (keep draft-dim lm_head), `validate_same_kv_cache_group` (skip — multi
       group). Centroids paths are dead code for 31B (`use_ordered_embeddings=False`).
    2. **`constant_draft_positions` — the one genuine base gap.** Upstream's base proposer defines it
       (default False) and consumes it at 3 points in `propose()` (ref:
       `upstream-ref/llm_base_proposer.py` lines 107, 523, 576, 588): seed `self.positions[:bs]`,
       skip `_update_positions_dependent_metadata`, and build attn metadata once (reuse). Port these
       3 conditionals into the fork's base `SpecDecodeBaseProposer.propose()` (eagle.py). Default
       False = no behavior change for existing proposers. **This is the delicate surgery.**
    3. `use_gemma4_mtp()` on `SpeculativeConfig` (mirror `use_eagle()`), gated on
       `method=="mtp" and draft model_type=="gemma4_assistant"`.
    4. `gpu_model_runner.py`: add the `elif self.speculative_config.use_gemma4_mtp(): self.drafter =
       Gemma4Proposer(...)` branch (+ import + the `isinstance(drafter, Gemma4Proposer)` handling for
       per-group block tables / bidi-sliding, and call `set_per_group_block_table` in `_prepare_inputs`).
  - **V100 unit check:** KV-shared Q-only attention at hd512 + hd256 via the Triton/flash-v100 backend
    (the kernels exist; KV-shared decode reading the target's cache is the unproven bit).
- **Phase C (validate + benchmark, ~1 day):** acceptance length via `/metrics`, lossless-output
  check vs the no-spec deploy, decode speedup vs the current 18.4 tok/s single-stream. enforce-eager
  (cudagraph + spec at hd512 untested).

## Phase C status (2026-06-04) — LOADS + FAST, but a correctness bug remains

End-to-end spec decode **runs on V100**: the target + `gemma4-31b-it-assistant` drafter load, the
KV-sharing wires correctly on all 4 TP workers (draft sliding layers 0–2 → target layer 58, full
layer 3 → target layer 59), the profile run passes, and serving is **3.3× faster** (61.6 vs 18.4
tok/s single-stream). Delivered via the PVC tarball overlay (`/models/gemma-overlay.tar.gz`, 16
files) + `26-deployment-gemma4-31b-mtp.yaml`.

Fixes needed to get there (committed): add `Gemma4Proposer` to every drafter `isinstance` guard in
`gpu_model_runner.py` (5 asserts + the type annotation + the cudagraph assert — `use_eagle()` is True
for gemma4 so it enters those eagle blocks); add a `gemma4_assistant` model-arch-config convertor so
`get_hidden_size()` returns `backbone_hidden_size` (5376) — the proposer's hidden-state feedback
buffer holds the *target's* last-layer hidden, not the draft's 1024.

**BUG (open): output is garbage** ("Paris" then degenerates), so spec decode is NOT lossless yet.
Diagnostic: acceptance is **all-or-nothing** — `num_accepted_per_pos` is uniform (453 at every
position 0–3; 453/526 drafts accept all 4, 73 reject at pos 0). Genuine rejection sampling decays
monotonically with position; uniform = draft and target are in lockstep on garbage, i.e. the
*target's* spec path is producing the garbage and the draft mirrors it.
- Ruled out: V100 KV-cache corruption by the draft — none of the flash_attn_v100 paths write the
  paged cache (the common attention layer does the reshape_and_cache, and it honors
  `kv_sharing_target_layer_name`).
- **LOCALIZED (num_spec=1 diagnostic):** with `num_speculative_tokens=1` the draft loop never runs
  (`for ... in range(num_spec-1)` is empty), so `constant_draft_positions` is NOT the cause — yet
  output is still garbage AND acceptance collapses to ~0% (1/141). At ~0% acceptance almost every
  emitted token is the **target's** bonus token, so **the target's spec-verify forward is itself
  producing garbage** — the drafter and `constant_draft_positions` are exonerated.
- **Refined prime suspect: cross-model KV sharing on V100.** The gemma4-specific new ingredient vs
  the working Qwen MTP is that the draft is *KV-shared* — it reads the target's KV cache
  (`_setup_gemma4_kv_sharing` + `add_kv_sharing_layers_to_kv_cache_groups` add the draft layers to
  the target's KV cache groups for layers 58/59). Qwen MTP has its *own* KV (not shared), so this
  path is exercised for the first time here. The shared KV-cache-group / block handling appears to
  corrupt the target's own attention at the shared layers (garbage from layer 58 onward), or the
  multi-KV-group (hd256 sliding + hd512 global → 2 groups) target verify is mis-metadata'd on V100.
  This is a deeper, kernel-adjacent fix than the loop.
### ROOT CAUSE FOUND (2026-06-04): the hd512 multi-token verify kernel

Ruled out and fixed along the way (both correct, committed): the backbone-dim feedback buffer
(model-arch convertor → `get_hidden_size`=5376) and the stale `get_supported_head_sizes`
(`[64,128,256]` → add `512`). Neither fixed correctness, but the head-size fix changed acceptance
86%→30% (the draft's hd512 global layer was being mis-routed).

**The bug is `_flash_v100_small_query_prefill_as_decode` at head_dim 512.** This is the V100 path
for MTP *verification* (a tiny multi-token query span over the KV prefix). It was only ever
validated for single-query decode; the multi-row verify at hd512 produces wrong results. Proof:
setting `VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q=0` (route verify to the direct paged-prefill kernel
instead) makes short output **correct** — "capital of France" → **Paris**, clean — and the
acceptance metric finally **decays monotonically** (per-pos 13/11/11/11 instead of the broken
uniform all-or-nothing). So the spec-decode *mechanism* (proposer, KV sharing, constant-position
drafting, rejection sampling) is **correct**; only the hd512 verify kernel is wrong.

Residual: with small-q disabled, *longer* output still degrades ("1, 2, 1000000…") because the
direct paged-prefill kernel hits the SM70 96 KB smem limit at hd512 long contexts — which is the
very reason the small-q path exists. So **both** V100 verify paths are flawed at 512:
small-q-as-decode = incorrect, paged-prefill = smem-limited.

**Remaining work (bounded CUDA-kernel fix):** make one hd512 verify path correct on SM70 — either
debug `flash_decode_paged.cu`'s partition path for the multi-row/long-seq verify shape at D=512
(the kernel was validated only single-row), or give the paged-prefill kernel an even-smaller-tile
D=512 config so it fits 96 KB at long context. Then drop the `VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q`
workaround. The non-kernel port (Phases A/B + the runner/config wiring) is complete and correct.

### CORRECTION (2026-06-04, later): it is NOT the verify kernel — it's the cross-model KV read

Broader testing overturned the "verify kernel" diagnosis. All THREE V100 verify paths were forced
and tested (env-only, no rebuild): small-q-as-decode (`SMALLQ_DECODE_MAX_Q` default), the dedicated
paged-prefill kernel (`SMALLQ=0`), and the dense paged-KV-gather + FA2 path (`SMALLQ=0` +
`DISABLE_PAGED_PREFILL=1`). **All three produce the same broken signature:** all-or-nothing
acceptance (per-pos uniform, e.g. 140/140/140/140 — a draft is either fully accepted or rejected at
pos 0) and garbage output. The earlier "13/11/11/11 looks like decay" was a misread — it was 11
full-accepts + 2 pos-0-only, i.e. still dominantly all-or-nothing. So "mechanism proven correct" was
**overstated**; only the first (prefill) token is reliably correct.

Decisive new clue: **output varies across identical temperature-0 runs** ("Paris" / "ParisB" /
"ParisBC" / "ParisB orLP"). Determinism at temp 0 is mandatory, so the spec path is reading
**uninitialized / wrong memory**. The drafter passes `kv_dummy = torch.empty(...)` (uninitialized)
as K/V (gemma4_mtp.py:251) and relies entirely on cross-model KV sharing to read the *target's* KV
blocks. All-or-nothing-on-garbage = draft and target agreeing on corrupted state; random variation =
uninitialized read. Conclusion: **the cross-model KV-share block-table / cache read for the draft's
layers is wrong on the fork's V100 multi-group cache** (the draft reads its own unallocated/garbage
blocks instead of the target's), which is the one factor common to all verify paths and is the
gemma4-specific thing Qwen MTP (own-KV) never exercises.

Next debug step (needs instrumentation, not config): in `_setup_gemma4_kv_sharing` / the runner,
dump the draft attention layers' resolved block_table + `kv_sharing_target_layer_name` vs the
target's, and confirm the draft's attn actually reads the target's physical blocks at run time
(add a one-shot tensor dump in `_flash_v100_decode` for a kv-shared layer). Likely fix is in
`add_kv_sharing_layers_to_kv_cache_groups` / per-group block-table wiring for the multi-group case,
or making the drafter pass real (cache-resident) K/V rather than `kv_dummy` on the V100 path.

---
Kernel read (2026-06-04): `flash_attention_decode_partition_kernel<D,PART,KV>` processes each query
row as an independent `(batch_idx, head, partition)` block — `q_shared[D]` is 1 KB at D=512, smem is
fine, and the per-row block_table/seq_len indexing is independent — so a multi-row verify should
behave like N independent (validated) single-row decodes. That means the D=512 defect is subtle
(candidates: `dot_qk_cache<512>` warp-reduction over 16 elems/lane, or the K/V cache read at the
verify's per-row seq_lens) and won't yield to static reading. Next debug step is numerical:
deploy with `VLLM_FLASH_V100_DEBUG_PREFILL_COMPARE=1` (compares small-q-as-decode vs paged-prefill
outputs in-kernel) to catch the first divergent layer/row — a deliberate active-session loop, not an
autonomous one (each iteration is an nvcc rebuild + GPU redeploy).

The no-spec deploy (`24-deployment`, 18.4/147 tok/s) is restored as the active service; the mtp
deploy is scaled to 0 (manifest + overlay preserved for resuming).

## Open risks to watch
- gpu_model_runner divergence (Phase B) is the dominant risk — could balloon if the fork's runner
  has refactored the spec-decode hooks the upstream gemma4 path depends on.
- V100 KV-shared decode at hd512 unproven (kernel reads target's cache through the Triton-derived
  V100 backend).
- Centroids-masking path (if 31B ships `use_ordered_embeddings`) adds a sparse-logits path to validate.

## Sources
- https://ai.google.dev/gemma/docs/mtp/overview ; vLLM recipe (Gemma4) spec-config example.
- Upstream `gemma4_mtp.py` @ vllm-project/vllm main; gemma4 runner integration in
  `vllm/v1/worker/gpu_model_runner.py` (main).
