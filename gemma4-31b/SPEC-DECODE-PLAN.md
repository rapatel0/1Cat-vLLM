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
- **Phase A (clean wins, ~½ day):** add the config class + registry + MTPModelTypes; port
  `gemma4_mtp.py`. Goal: vLLM *loads* the assistant as a draft model (even if propose() is stubbed)
  — proves config/model/registry without touching the runner.
- **Phase B (the crux, ~2–3 days):** port the `gpu_model_runner.py` gemma4 proposer + KV-sharing
  setup + backbone-dim feedback buffer, reconciling against the fork's divergence. Unit-validate the
  KV-shared Q-only attention on V100 (hd512 + hd256) in isolation first.
- **Phase C (validate + benchmark, ~1 day):** acceptance length via `/metrics`, lossless-output
  check vs the no-spec deploy, decode speedup vs the current 18.4 tok/s single-stream. enforce-eager
  (cudagraph + spec at hd512 untested).

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
