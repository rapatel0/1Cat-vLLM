# Integrated DFlash2 configuration options

## Purpose and evidence boundary

This ledger describes the integrated DFlash2 stack at commit `3b37dea1a` or a later documentation-only descendant.

The integrated stack contains these inputs:

- 1Cat-vLLM PR #252: MRV2 DFlash routing and model-runner foundation.
- 1Cat-vLLM PR #253: Qwen3.8 DFlash2, SM70 correctness, hybrid cache, prefix reuse, and non-causal Flash-V100 support.
- 1Cat-vLLM PR #254: the experimental SM70 fused selector.
- 1Cat-vLLM PR #257: shared and fused GDN verifier metadata.
- 1Cat-vLLM PR #266: target-graph fusions, sparse rejection, QPN8, and TP4 push all-reduce.
- 1Cat-vLLM PR #267: FP32 speculative GDN beta.
- 1Cat-vLLM PR #268: batched E5M2 XQA wide loads.
- vLLM PR #53017: stride-correct probabilistic DFlash2 score caching.

The source locations and defaults in this file come from the integrated tree. Performance evidence comes only from committed design records or retained qualification results.

Recommendations are provisional. They do not replace paired V100 tests of the complete stack.

The recommendation values are:

- **retain-on**: keep the current default enabled unless a regression appears.
- **retain-off**: keep the current default disabled.
- **needs paired qualification**: do not select a default from component evidence alone.

## Transfer-first mapping from pinned SGLang-V100

The reference repository is `haohervchb/sglang-V100` at commit `845b9fdf7a7eff7064df82b82c32587dd1a5137a`.

The pinned README defines DFlash2 block eight as one anchor plus seven proposed tokens. It also separates selector top-K 16 from proposal length.

The pinned README uses E5M2 target KV, one running request, and a batch-one CUDA graph. It does not supply the eight-slot FP16 target-KV profile.

Eight slots, FP16 target KV, and c8 graph coverage come from retained operator experiments. The table labels each evidence source explicitly.

| SGLang-V100 setting | Source or evidence classification | 1Cat-vLLM equivalent | Transfer note |
| --- | --- | --- | --- |
| TP4 | pinned: `--tensor-parallel-size 4` | `--tensor-parallel-size 4` | Use one fully connected four-V100 NVLink island. |
| DCP1 | pinned: no separate DCP split | `--decode-context-parallel-size 1` | Required by `SlidingWindowSpec.max_memory_usage_bytes` in `vllm/v1/kv_cache_interface.py:485`. |
| Context 262,144 | pinned: `--context-length 262144` | `--max-model-len 262144` | This is a per-request context ceiling. |
| Eight slots | retained operator profile: `--max-running-requests 8` | `--max-num-seqs 8` | The pinned README uses one request. Capture target and draft graph shapes for all eight slots. |
| DFlash block eight | pinned: `--speculative-dflash-block-size 8` | `num_speculative_tokens=7` | MRV2 adds one anchor query, so the verifier query length is eight. |
| Draft window 2,048 | pinned: `--speculative-draft-window-size 2048` | Checkpoint `dflash_config.use_swa=true` and `swa_window_size=2048` | The integrated checkpoint owns the window. There is no equivalent runtime CLI override. |
| Draft checkpoint | pinned: `--speculative-draft-model-path ...DFlash2` | `speculative_config.model` | Use the pinned Qwen3.8 DFlash2 checkpoint. |
| Draft quantization | pinned: `--speculative-draft-model-quantization unquant` | `speculative_config.quantization=null` | The draft remains unquantized. |
| Target E5M2 KV | pinned: `--kv-cache-dtype fp8_e5m2` | `--kv-cache-dtype fp8_e5m2` | This is the published FP8-target DFlash2 precision. |
| Target FP16 KV arm | retained operator transfer arm | `--kv-cache-dtype auto` | This is not a pinned README result for the FP8 target. It requires a paired vLLM test. |
| Draft FP16 KV | pinned: separate draft allocation uses FP16 | `speculative_config.kv_cache_dtype="auto"` | PR #253 resolves SM70 DFlash draft KV to model dtype instead of inheriting target E5M2. |
| Mamba convolution cache FP16 | pinned: `SGLANG_MAMBA_CONV_DTYPE=float16` | `--mamba-cache-dtype float16` | The live vLLM arm left this at `auto`. Transfer requires exact state and output gates. |
| Mamba SSM state FP16 | pinned: `SGLANG_MAMBA_SSM_DTYPE=float16` | `--mamba-ssm-cache-dtype float16` | The live vLLM arm resolved SSM state to FP32. This is a material precision change. |
| Target Flash-V100 | pinned: `--attention-backend flash_attn_v100` | `--attention-backend FLASH_ATTN_V100` | The target uses the SM70 paged backend. |
| Draft non-causal Flash-V100 | pinned: DFlash worker selects the V100 backend | `speculative_config.attention_backend="FLASH_ATTN_V100"` | The draft needs non-causal D256 support. |
| CUDA graphs | pinned: `--cuda-graph-bs 1` | `--compilation-config ...` | c8 coverage is a retained operator profile, not a pinned source result. Include verifier sizes through 64. |
| Prefill chunk 8,192 | pinned: `--chunked-prefill-size 8192` | `--max-num-batched-tokens 8192` | This controls scheduler work per iteration, not total KV capacity. |
| SGLang token pool | pinned: `--max-total-tokens 262144` | KV profiling plus `--gpu-memory-utilization` | Do not map this to `max_num_batched_tokens`. vLLM profiles blocks from available memory. |
| SGLang memory fraction | pinned: `--mem-fraction-static 0.75` | `--gpu-memory-utilization` | Values are not numerically equivalent because allocation models differ. |
| NVLink P2P hint | pinned: `NCCL_P2P_LEVEL=NVL` | same NCCL environment variable | Direct transport hint. It does not select vLLM custom all-reduce. |
| NCCL NVLS flag | pinned: `--enable-nccl-nvls` | no direct 1Cat CLI equivalent | Keep NCCL automatic unless a matched topology test proves a mapping. |
| Speculative V2 | pinned: `SGLANG_ENABLE_SPEC_V2=1` | automatic V2 model-runner selection | Explicit `VLLM_USE_V2_MODEL_RUNNER=0` is rejected for flat DFlash. Implementations and graph ownership differ. |
| Plan-stream overlap | pinned: `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` | no direct equivalent | Review streams, events, and graph capture before any port. |
| Mamba memory ratio | pinned: `--mamba-full-memory-ratio 0.1` | no direct ratio equivalent | vLLM profiles unified cache groups from `gpu_memory_utilization`. |
| Mamba scheduler strategy | pinned: `--mamba-scheduler-strategy extra_buffer` | no direct strategy equivalent | vLLM uses `mamba_cache_mode` and GPU state migration. Semantics differ. |
| One-stage collective | pinned: `SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage` | existing custom all-reduce or `VLLM_SM70_TP4_PUSH_ALLREDUCE=1` | The push route remains default-off pending a complete paired result. |
| Warmup | pinned: graph capture plus first-request JIT | excluded graph, JIT, and node warmups | Sustained c8 node warmup is a retained operator requirement. Exclude it from official timing. |

`max_total_tokens` is a shared physical token-pool limit in SGLang. vLLM has no direct serving CLI equivalent in this stack.

vLLM derives KV blocks from `gpu_memory_utilization`, model memory, graph memory, and cache geometry. `max_num_batched_tokens` limits one scheduler iteration.

## Three distinct configurations

### Source defaults

These values exist before deployment overrides:

| Setting | Source default | Location |
| --- | ---: | --- |
| TP size | `1` | `vllm/config/parallel.py:114` |
| DCP size | `1` | `vllm/config/parallel.py:321` |
| GPU memory utilization | `0.92` | `vllm/config/cache.py:67` |
| Target KV dtype | `auto` | `vllm/config/cache.py:75` |
| Prefix caching | `true` | `vllm/config/cache.py:95` |
| Max sequences | `128` | `vllm/config/scheduler.py:44` |
| Max batched tokens | `2048` | `vllm/config/scheduler.py:42` |
| Mamba cache mode | `none` | `vllm/config/cache.py:136` |
| Mamba convolution cache dtype | `auto` | `vllm/config/cache.py:128` |
| Mamba SSM cache dtype | `auto` | `vllm/config/cache.py:132` |
| Draft sample method | `greedy` | `vllm/config/speculative.py:288` |
| Draft KV dtype | `null` | `vllm/config/speculative.py:125` |
| Flash-V100 dynamic decode partitions | `true` | `vllm/envs.py:2365` |
| SM70 compiled graph policy | source env `false`. SM70 policy can auto-set `true` | `vllm/envs.py:2877`, `vllm/config/vllm.py:1335` |

Prefix caching with linear attention changes an unspecified Mamba cache mode from `none` to `align`. This occurs in `vllm/engine/arg_utils.py:1773`.

### Exact live E5M2 qualification arm

This arm qualified the corrected base before the complete PR #254/#257/#266 stack deployment.

```text
TP=4
DCP=1
max_model_len=262144
max_num_seqs=8
max_num_batched_tokens=8192
target kv_cache_dtype=fp8_e5m2
draft kv_cache_dtype=auto  # FP16 for this draft
model dtype=float16
gpu_memory_utilization=0.80
enable_prefix_caching=true
mamba_cache_dtype=auto
mamba_ssm_cache_dtype=auto  # resolved persistent SSM state is FP32
VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS=0
draft_sample_method=greedy
num_speculative_tokens=7
method=dflash
target attention backend=FLASH_ATTN_V100
draft attention backend=FLASH_ATTN_V100
```

The final c8 graph override was:

```json
{
  "cudagraph_mode": "FULL_AND_PIECEWISE",
  "cudagraph_capture_sizes": [1, 2, 4, 8, 9, 16, 18, 24, 32, 40, 48, 56, 64],
  "max_cudagraph_capture_size": 64
}
```

The eight-slot verifier uses 64 target tokens because every request contributes eight query tokens.

The base arm retained 1,047,948 KV tokens and used 0.94 GiB for target and draft graphs. The warmed c8 result was 232.536 aggregate tok/s.

That result does not qualify the later combined performance gates. It only records the deployment configuration and the c8 graph-shape effect.

### Proposed FP16 transferable arm

This arm matches the requested SGLang precision transfer while retaining vLLM allocation semantics.

```text
TP=4
DCP=1
max_model_len=262144
max_num_seqs=8
max_num_batched_tokens=8192
target kv_cache_dtype=auto  # FP16 with model dtype float16
draft kv_cache_dtype=auto   # FP16
gpu_memory_utilization=0.80 initial bound
enable_prefix_caching=true
mamba_cache_dtype=float16 candidate
mamba_ssm_cache_dtype=float16 candidate
VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS=0
draft_sample_method=greedy for deterministic bring-up
num_speculative_tokens=7
method=dflash
cudagraph verifier sizes=[8,16,24,32,40,48,56,64]
```

The `0.80` memory value is an initial safety bound, not a transferred SGLang value. Record profiled KV tokens and free memory before any increase.

The two explicit FP16 Mamba dtypes are unqualified transfer candidates. They change the live vLLM persistent SSM state from FP32 to FP16.

Require exact recurrent-state, output-trajectory, retrieval, quality, memory, and throughput gates before retaining either FP16 Mamba dtype.

Run excluded graph capture, first-request JIT, c8 admission, and sustained node warmup before official timing.

## DFlash2 speculative configuration fields

The public entry is `--speculative-config`. Field definitions are in `vllm/config/speculative.py`.

| Exact field | Current default | DFlash2 scope and control | Provenance | Dependencies and evidence | Risk | Provisional recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| `method` | `null`, inferred when possible | Set `"dflash"`. Flat DFlash requires MRV2. | #252 | `vllm/config/vllm.py:537` forces Model Runner V2. `dflash_ddtree` is a different V1 route. | A wrong method selects different scheduler semantics. | retain-on as `dflash` |
| `model` | `null` | Draft checkpoint path or model ID. | #252/#253 | The checkpoint architecture must resolve to `DFlash2DraftModel`. | A mismatched draft changes weights and selector geometry. | retain-on with pinned revision |
| `num_speculative_tokens` | checkpoint value or required | Set `7`. MRV2 executes one anchor plus seven masks. | #252/#253 | `num_lookahead_tokens()` reserves seven additional slots. Block eight is not KV block size. | Other values depart from the published checkpoint contract. | retain-on at `7` |
| `draft_tensor_parallel_size` | target TP size | Set `4` for TP4. Valid values are one or target TP. | #252/#253 | Candidate TopK performs TP-aware local and global reduction. | TP1 replicates the draft and changes memory and communication. | retain-on at `4` |
| `tensor_parallel_size` | `null` | Deprecated speculative field. | upstream | The validator directs users to `draft_tensor_parallel_size`. | Wrong spelling can fail validation. | retain-off |
| `enforce_eager` | `null` | The integrated MRV2 DFlash2 path does not honor draft-only `true`. It still captures DFlash2 graphs. | #252 | Current eager parity requires global `--enforce-eager`, which disables target and draft compilation. Add a route test before claiming draft-only support. | Eager throughput is not representative. A silent graph capture invalidates the intended arm. | retain-off |
| `quantization` | `null` | `null` means unquantized draft. | #252/#253 | The Qwen3.8 DFlash2 checkpoint is unquantized. | Draft quantization can change candidates and acceptance. | retain-off |
| `moe_backend` | `null`, inherits target | Dense Qwen3.8 DFlash2 does not need a draft MoE override. | upstream | No measured DFlash2 evidence supports an override. | Unneeded backend changes widen scope. | retain-off |
| `attention_backend` | `null`, automatic | Set `FLASH_ATTN_V100` for the non-causal V100 draft. | #253 | PR #253 validates non-causal paged attention and graph metadata. | A causal-only backend changes draft attention semantics. | retain-on |
| `kv_cache_dtype` | `null` | `null` normally inherits target KV. SM70 quantized-target DFlash resolves it to `auto`. | #253 | Explicit `auto` documents FP16 draft KV for E5M2 target KV. | Inheriting E5M2 into the draft can change acceptance and kernels. | retain-on as explicit `auto` |
| `max_model_len` | `null` | Optional draft speculation ceiling. | upstream/#253 | Leave unset for the 262K service unless testing speculation bypass. | A lower value silently disables speculation beyond the limit. | retain-off |
| `revision` | `null` | Pins draft weights. | upstream/#253 | Correctness evidence pins checkpoint revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`. | Floating revisions can change selector weights. | retain-on with a pin |
| `code_revision` | `null` | Pins remote draft code. | upstream | The integrated native class reduces the need for remote code. | Remote code can diverge from the audited model class. | retain-off unless required |
| `draft_load_config` | `null`, inherits target load config | Optional draft-specific loader settings. | upstream | No retained DFlash2 evidence uses an override. | Different loading rules can change dtype or tensor layout. | retain-off |
| `draft_sample_method` | `greedy` | `greedy` uses one-hot draft probabilities. `probabilistic` stores FP32 sparse proposal scores. | #252/#253 | PR #253 reports greedy and probabilistic acceptance. The live arm used greedy. | Sampling mode changes acceptance, memory, and rejection semantics. | needs paired qualification |
| `rejection_sample_method` | `standard` | `standard` is the real rejection path. | upstream/#253 | Sparse target rejection also requires `standard`. | `synthetic` is not a quality or throughput result. | retain-on as `standard` |
| `synthetic_acceptance_rates` | `null` | Test-only synthetic rejection input. | upstream | Requires `rejection_sample_method="synthetic"`. | It manufactures acceptance and cannot qualify production. | retain-off |
| `synthetic_acceptance_length` | `null` | Test-only synthetic mean acceptance. | upstream | Mutually exclusive with synthetic rates. | It manufactures acceptance and cannot qualify production. | retain-off |
| `use_local_argmax_reduction` | `false` | Generic non-tree greedy shortcut. DFlash2 has a separate TP candidate selector. | upstream/#253 | No DFlash2 source consumer uses this as the selector control. | Assuming this controls DFlash2 can hide full selector cost. | retain-off |
| `parallel_drafting` | `false`, then auto-set `true` for DFlash | Internal parallel-draft classification. | #252 | `SpeculativeConfig` sets it for the DFlash family. | A manual override does not change block-diffusion semantics. | retain-on through automatic resolution |

DDTree fields are not flat DFlash2 options. `method="dflash_ddtree"` retains a separate scheduler, state, and tree-verification contract.

## Deployment CLI ledger

| Exact CLI or config | Source default | Exact live value | Scope and interaction | Evidence and risk | Provisional recommendation |
| --- | --- | --- | --- | --- | --- |
| `--tensor-parallel-size` | `1` | `4` | The target and draft shard across one NVLink island. | TP4 is the pinned V100 reference topology. | retain-on at `4` |
| `--decode-context-parallel-size` | `1` | `1` | Draft sliding-window KV rejects DCP greater than one. | `SlidingWindowSpec` asserts DCP1. | retain-on at `1` |
| `--max-model-len` | model-derived | `262144` | Per-request context ceiling. | Long-context retrieval passed in the base arm. | retain-on at `262144` |
| `--max-num-seqs` | `128` | `8` | Admission cap and QPN8 eligibility cap. | QPN8 rejects values above eight. c8 graph sizes require this value. | retain-on at `8` |
| `--max-num-batched-tokens` | `2048` | `8192` | Scheduler budget and physical draft SWA admission term. | Draft held KV is bounded by `2047 + max_num_batched_tokens`, then by context. | retain-on at `8192` |
| `--dtype` | model-derived | `float16` | Runtime target and draft arithmetic dtype. | SM70 adaptations use FP16 projections with explicit FP32 and BF16 points. | retain-on at `float16` |
| `--kv-cache-dtype` | `auto` | `fp8_e5m2` | Target KV precision. | The E5M2 arm has native SM70 routes. The FP16 transfer arm uses `auto`. | needs paired qualification |
| `--gpu-memory-utilization` | `0.92` | `0.80` | vLLM KV profiling bound. | The c8 base retained about 3.0 GiB free after 64-token graphs. | retain-on at `0.80` for bring-up |
| `--enable-prefix-caching` | source cache default `true` | enabled | Enables target and draft prefix reuse. It auto-selects Mamba `align`. | PR #253 measured exact prefix hits and lower TTFT. | retain-on |
| `--mamba-cache-mode` | `none` | automatic `align` | Prefix caching with linear attention changes the unresolved default to `align`. | Fused PR #257 metadata requires `none`, creating a direct conflict. | needs paired qualification |
| `--mamba-cache-dtype` | `auto` | `auto` | Controls convolution-state cache precision. | SGLang pins FP16. The live vLLM arm did not. Precision changes can alter recurrent trajectories. | needs paired qualification |
| `--mamba-ssm-cache-dtype` | `auto` | `auto`, resolved SSM state FP32 | Controls persistent recurrent-state precision. | SGLang pins FP16. The live vLLM state is FP32. This is not a safe mechanical transfer. | needs paired qualification |
| `--attention-backend` | automatic | `FLASH_ATTN_V100` | Target paged attention backend. | PR #253 validates the non-causal draft and target E5M2 paths. | retain-on |
| `--compilation-config` | automatic | explicit graph sizes through `64` | c8 needs eight requests times eight verifier tokens. | The c8 full-graph arm improved aggregate throughput from 161.229 to 232.536 tok/s. | retain-on for c8 |
| `--enforce-eager` | `false` | omitted | Global control that disables target compilation and DFlash2 graph capture. | This is the current reliable eager parity control. | retain-off |
| `--disable-custom-all-reduce` | `false` | omitted | Must remain false for the optional TP4 push route. | Push requires custom all-reduce and full NVLink connectivity. | retain-off |
| `--block-size` | platform default, normally 16 | omitted | Physical KV cache page size. | DFlash block eight is unrelated to this field. | retain-off |

## Direct DFlash2 and integrated performance environment variables

Boolean values use `1` for enabled and `0` for disabled.

| Exact variable | Location and default | Scope and eligibility | Provenance | Dependencies and measured evidence | Correctness or performance risk | Provisional recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| `VLLM_USE_V2_MODEL_RUNNER` | `vllm/envs.py:3827`, `null` | Flat `method=dflash` forces V2. Explicit `0` fails closed. | #252 | Triton and V2-supported features are required. | V1 flat DFlash semantics are removed. | retain-on through automatic selection |
| `VLLM_SM70_FLASH_ATTN_V100` | `vllm/envs.py:2080`, `1` | Enables the SM70 Flash-V100 backend. | pre-existing, required by #253 | Target and draft attention evidence uses this backend. | Disabling it changes attention kernels and graph policy. | retain-on |
| `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` | `vllm/envs.py:2877`, env default `0` | The SM70 policy auto-sets `1` unless compilation is disabled. | pre-existing, affected by #253/#254/#266 | Selects `VLLM_COMPILE` and full-plus-piecewise graphs. | It normalizes the fused selector off and changes graph gates. | retain-on for production graphs |
| `VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS` | `vllm/envs.py:2365`, `1` | Controls dynamic Flash-V100 decode workspace partitioning. | pre-existing deployment setting | The live arm used `0` to avoid dynamic workspace growth. | Dynamic workspaces previously caused TP4 OOMs in related services. | retain-off for this arm |
| `VLLM_FLASH_V100_DECODE_PARTITION_SIZE` | `vllm/envs.py:2368`, unset | Forces one partition size. | pre-existing | An explicit value disables some automatic batch-context routing. | One fixed value can regress other lengths. | retain-off |
| `VLLM_FLASH_V100_SMALLQ_DECODE_MAX_Q` | `vllm/envs.py:2359`, `16` | Small-query Flash-V100 eligibility. | pre-existing, affected by #253/#266 | DFlash block eight needs Q=8, which is within the default. | Values below eight force fallback. | retain-on at `16` |
| `VLLM_FLASH_V100_SMALLQ_DECODE_MAX_MODEL_LEN` | `vllm/envs.py:2362`, `0` | Optional small-query context cap. Zero means no configured cap. | pre-existing | No retained DFlash2 result needs a cap. | A low cap creates long-context route changes. | retain-off at `0` |
| `VLLM_SM70_ENABLE_LM_HEAD_FASTPATH` | `vllm/envs.py:1738`, `0` | Enables the full SM70 dense FP16 LM-head route and prepares its alternate layout. | pre-existing, interacts with #254 | The fused selector shares LM-head layout preparation logic in `vocab_parallel_embedding.py`. Full logits are not bitwise equal to Torch. | Enabling it changes target candidate logits and adds memory. | retain-off |
| `VLLM_SM70_LM_HEAD_TOP1` | `vllm/envs.py:1748`, normally `1`. Compiled SM70 policy resolves `0` | Prepares the LM-head top-one layout for eligible greedy paths. | pre-existing, interacts with #254 | The compiled Flash-V100 policy auto-sets zero. The fused selector also enters this layout-selection boundary. | A route or layout change can alter selector memory and candidate ordering. | retain-off for compiled DFlash2 |
| `VLLM_SM70_LM_HEAD_TOP1_TC` | `vllm/envs.py:1764`, `0` | Enables the Tensor Core top-one path and enters the same alternate LM-head layout boundary. | pre-existing, interacts with #254 | `_is_sm70_lm_head_fastpath_eligible()` evaluates it beside full LM-head, top-one, and fused-selector controls. | It can prepare extra layout and alter the candidate path used beside the selector. | retain-off |
| `VLLM_SM70_DFLASH2_FUSED_SELECTOR` | `vllm/envs.py:1769`, `0` | Eager-only dense FP16 selector candidate. Compiled graph startup resets `1` to `0`. | #254 | M=7 was 5.78% slower and used about 1.19 GiB more memory per rank. | Duplicate weights reduce KV capacity. Tie order must match `torch.topk`. | retain-off |
| `VLLM_SM70_DFLASH2_VERIFY_FASTPATH` | `vllm/envs.py:1775`, `0` | Umbrella for shared DFlash2 target GDN metadata. | #257 | Required by the fused metadata gate. Shared B1 graph evidence exists. | Eager regressed, and mixed c8 evidence is incomplete. | needs paired qualification |
| `VLLM_SM70_DFLASH2_FUSED_GDN_METADATA` | `vllm/envs.py:1781`, `0` | Fused persistent GDN metadata. Requires umbrella, CUDA SM70, DFlash2, and Mamba cache mode `none`. | #257 | B1 paired result reduced rounds from 46.8045 to 35.0308 ms and kept 430 tokens exact. | Prefix caching auto-selects `align`, which makes this gate ineligible. B2/B4/c8 remain incomplete. | needs paired qualification |
| `VLLM_SM70_DFLASH2_GDN_METADATA_SHADOW` | `vllm/envs.py:1786`, `0` | Debug oracle for fused-versus-legacy metadata. | #257 | Use poison and mixed-length qualification only. | Adds legacy materialization and comparison overhead. | retain-off |
| `VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY` | `vllm/envs.py:1792`, `0` | Packed DFlash2 target GDN recurrent verifier. Independent of the metadata umbrella. | #257 | Production-shape component parity passed in PR evidence. | Full endpoint and broad quality evidence remain incomplete. | needs paired qualification |
| `VLLM_SM70_DFLASH2_FUSED_GDN_NORM` | `vllm/envs.py:1798`, `1` | SM70 DFlash2 target GDN one-pass output RMSNorm. | #266 | Part of the exact target-graph reduction from 24.740 to 19.317 ms. | Route scope and eager/prefill behavior still need combined-stack checks. | retain-on |
| `VLLM_SM70_DFLASH2_FUSED_GDN_SPLIT` | `vllm/envs.py:1804`, `1` | Fuses nonzero-offset Qwen3.8 z/b/a materialization. | #266 | Part of the exact 1,355-node target-graph reduction. | Plain views remain unsafe under full graphs for nonzero offsets. | retain-on |
| `VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA` | `vllm/envs.py:1811`, `1` | Target Flash-V100 metadata only, SM70 DFlash2, not the draft model. | #266 | Draft-to-target fell from 5.720 to 1.911 ms. Full round fell 12.1%. Tokens and acceptance matched. | Mixed c8 replay still needs the integrated build. | retain-on |
| `VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS` | `vllm/envs.py:1816`, `0` | Fuses FP16 projection, FP32 residual, and Gemma RMS suffix under compile graphs. | #266 | It completed the 19.317 ms target graph. Preliminary numeric evidence classifies it Type B. | Perplexity and broader quality gates remain pending. | retain-off |
| `VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION` | `vllm/envs.py:1823`, `0` | SM70 DFlash2, standard rejection, one non-prefill request, temperature above zero, target `top_k=20`, no penalties, grammar, logprobs, bias, or bad words. | #266 | Component path was 0.0901 ms. End-to-end round fell from 27.951 to 27.166 ms. | The live greedy arm is ineligible. Distribution and dataset gates remain pending. | retain-off |
| `VLLM_SM70_TP4_PUSH_ALLREDUCE` | `vllm/envs.py:1829`, `0` | Fully connected SM70 TP4 custom all-reduce for exact FP16 `[8,5120]`. | #266 | The 128-call chain was 0.850-0.856 ms versus 1.854 ms. | Full-model paired rerun was preempted. Buffer lifetime and changed-input replay remain risks. | retain-off |
| `VLLM_SM70_TP4_M5_AR_THREADS` | `vllm/envs.py:1870`, unset. C++ default `512` | Adjacent custom all-reduce thread override for TP4 M5 bytes. Accepted values are 128, 256, or 512. | pre-existing | This M5 control does not configure the PR #266 `[8,5120]` push route. | Confusing the shapes invalidates an A/B and can regress the unrelated M5 path. | retain-off at unset |
| `VLLM_SM70_FP8_QPN8` | `vllm/envs.py:1627`, `1` | Qwen3.8-27B-FP8 TP4, `max_num_seqs<=8`, target-only or MRV2 DFlash2. M=1-8 uses QPN8. | #266 | Complete-source c1 improved 3.752%. WikiText, GSM8K, MMLU, and C-Eval gates passed. | Combined DFlash c8 and all-rank graph replay remain required. Missing automatic ops fall back. | retain-on |
| `VLLM_SM70_FP8_QPN8_LIBRARY` | `vllm/envs.py:1630`, unset | Optional source-built QPN8-only shared library. | #266 | Production links operators into `vllm._C`. | A wrong library can mix incompatible operators. | retain-off |
| `VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING` | `vllm/envs.py:2431`, `1` | Enables target-only XQA batch/context route selection when `spec_config is None` and E5M2 geometry matches. | pre-existing, prerequisite for #268 | PR #268 wide loads sit inside this route. Current DFlash target config is ineligible. | Treating it as active under DFlash misattributes performance. | retain-on for target-only |
| `VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING_TRACE` | `vllm/envs.py:2434`, `0` | Emits XQA batch/context route diagnostics. | pre-existing, diagnostic for #268 | Use it only to prove route hits. | Trace output can perturb logs but not route semantics. | retain-off |
| `VLLM_FLASH_V100_XQA_E5M2_BATCH_WIDE_LOAD` | `vllm/envs.py:2452`, `1` | Batched E5M2 XQA route with p256, page size multiple of 16, and accepted batch-context routing. | #268 | Reuses page IDs and paired 128-bit loads. | Current Python batch-context routing requires `spec_config is None`. Do not claim a DFlash speedup. | retain-on for target-only |

An explicit `VLLM_SM70_FP8_QPN8=1` fails if required operators are missing. An unset variable allows automatic fallback to TurboMind.

## DFlash diagnostics and registered compatibility controls

These controls are default-off unless stated otherwise. None belongs in a performance result unless the result records it.

| Exact variable | Default | Current consumer and scope | Provenance | Risk and recommendation |
| --- | ---: | --- | --- | --- |
| `VLLM_DFLASH_PROFILE` | `0` | V1 DFlash proposer profiling in `vllm/v1/spec_decode/dflash.py` | pre-existing, affected by #252 | Adds timing work. Retain-off. |
| `VLLM_DFLASH_PROFILE_LOG_INTERVAL` | `32` | Profile log interval | pre-existing | Meaningless when profile is off. Retain default. |
| `VLLM_DFLASH_DUMP_FIRST_PASS` | `0` | V1 first-pass tensor dump | pre-existing | Writes large tensor artifacts. Retain-off. |
| `VLLM_DFLASH_DISABLE_AUX_OUTPUTS` | `0` | Legacy GPU runner auxiliary-output diagnostic | pre-existing, affected by #252 | Can remove hidden states required by DFlash. Retain-off. |
| `VLLM_DFLASH_DEBUG_STATE_TABLE` | `0` | GDN speculative metadata dump | pre-existing/#252 | Adds host copies and files. Retain-off. |
| `VLLM_DFLASH_DEBUG_CORRUPTION` | `0` | DFlash corruption checks in `llm_base_proposer.py` | pre-existing/#252 | Adds debug synchronization. Retain-off. |
| `VLLM_DFLASH_DUMP_DRAFT_LOGITS` | `0` | Draft-logit dumps in `llm_base_proposer.py` | pre-existing/#252 | Large outputs and synchronization. Retain-off. |
| `VLLM_FLASH_V100_DFLASH_PREFIX_DUMP` | `0` | Flash-V100 prefix metadata dump | pre-existing/#253 | Diagnostic only. Retain-off. |
| `VLLM_DFLASH_SYNC_CONTEXT_KV` | `0` | Registered. No current MRV2 consumer found | legacy compatibility | Do not assume it synchronizes DFlash2. Retain-off. |
| `VLLM_DFLASH_SKIP_CONTEXT_KV_PRECOMPUTE` | `0` | Registered. No current MRV2 consumer found | legacy compatibility | Unsafe if reconnected because draft context KV is required. Retain-off. |
| `VLLM_DFLASH_DEBUG_CONTEXT_KV` | `0` | Registered. No current MRV2 consumer found | legacy compatibility | Diagnostic only. Retain-off. |
| `VLLM_DFLASH_DUMP_LAYER_HIDDENS` | `0` | Registered. No current MRV2 consumer found | legacy compatibility | Diagnostic only. Retain-off. |
| `VLLM_DFLASH_DUMP_LAYER0_COMPONENTS` | `0` | Registered. No current MRV2 consumer found | legacy compatibility | Diagnostic only. Retain-off. |
| `VLLM_DFLASH_DUMP_ATTN_COMPONENTS` | `0` | Registered. No current MRV2 consumer found | legacy compatibility | Diagnostic only. Retain-off. |

## Checkpoint and source-level gates

These controls are not serving CLI fields.

| Exact gate or checkpoint key | Location and current behavior | Provenance | Dependencies and evidence | Risk | Provisional recommendation |
| --- | --- | --- | --- | --- | --- |
| Architecture `DFlash2DraftModel` | Registry maps it to `DFlash2Qwen3ForCausalLM` in `vllm/model_executor/models/registry.py:601`. | #253 | Required for DFlash2 speculator, GDN gates, and QPN8 target eligibility. | Another architecture bypasses these paths. | retain-on |
| Target LM head is unquantized | `qwen3_dflash2.py:397` rejects other quant methods. | #253 | Candidate generation needs exact target shard logits. | Quantized LM head changes candidate ranking. | retain-on |
| `dflash_config.selector_top_k` | Required integer. Pinned checkpoint value is 16. | #253 | Distinct from seven draft tokens and target sampling top-K 20. | A changed K changes selector tensors and candidates. | retain-on at checkpoint value |
| `dflash_config.selector_rank` | Required integer for selector codebooks. | #253 | Defines candidate-selector low-rank geometry. | A mismatch prevents weight compatibility. | retain-on at checkpoint value |
| `dflash_config.input_embedding_scale` | Default `1.0` in `qwen3_dflash2.py:358`. | #253 | Checkpoint-controlled numeric transform. | A wrong value shifts every draft layer. | retain-on at checkpoint value |
| `dflash_config.output_multiplier` | Default `1.0` in `qwen3_dflash2.py:390`. | #253 | Applies before candidate scoring. | A wrong value changes proposal distribution. | retain-on at checkpoint value |
| `dflash_config.final_logit_softcapping` | Disabled when absent or non-positive. | #253 | Checkpoint-controlled selector logits. | Enabling it changes candidates. | retain-on at checkpoint value |
| `dflash_config.sample_from_anchor` | Default `false`. `true` raises for MRV2 DFlash. | #252/#253 | The anchor is the bonus token. Mask rows predict proposals. | Anchor sampling changes the 1+N layout. | retain-off |
| `dflash_config.use_swa` | Default `false`. Pinned DFlash2 uses sliding attention. | #253 | `swa_window_size` or top-level window must exist. | Missing window fails. Wrong window changes draft KV semantics. | retain-on from checkpoint |
| `dflash_config.swa_window_size` | No generic default. Pinned value is 2048. | #253 | Creates `SlidingWindowSpec` and physical vLLM block reclamation. | DCP greater than one is unsupported. | retain-on at `2048` |
| `dflash_config.causal` | Optional override. Pinned DFlash2 is non-causal. | #253 | Draft attention backend must support non-causal D256. | A causal override changes the diffusion block. | retain-off for this checkpoint |
| `dflash_config.attention_sink_bias` | Falls back to top-level `false`. | #253 | Checkpoint-controlled SWA behavior. | A mismatched sink bias changes attention. | retain-on from checkpoint |
| Five draft layer IDs and target boundaries | Checkpoint layer IDs `[5,19,33,47,61]` map to target boundaries `[6,20,34,48,62]`. | #253 | Hidden-state extraction and grouped K normalization depend on this mapping. | Off-by-one mapping silently reduces acceptance. | retain-on |
| Grouped RMSNorm outer-row indexing | 2D weights select the outer input row in `csrc/libtorch_stable/layernorm_kernels.cu:36`. | #253 dependency repair | Raised pooled acceptance from 3.9725 to 4.5644 and aggregate throughput 12.10%. | Row-zero reuse corrupts four draft layers. No opt-out exists. | retain-on |
| DFlash score-cache stride | Uses `draft_logits.stride(1)` at `dflash2/speculator.py:306`. | vLLM #53017 | Supports nonstandard vocabulary strides. | Using vocabulary width corrupts cached score columns. No opt-out exists. | retain-on |
| Speculative beta dtype | Every DFlash speculative GDN gate requests `torch.float32`. | #267 | Reduces recurrent-state error and preserves the repaired trajectory. | FP16 beta can change accepted-state evolution. No opt-out exists. | retain-on |
| SM70 BF16 emulation | Automatic for the Qwen3.8 draft on SM70. | #253 | Keeps FP32 residuals, explicit BF16 RNE points, and scaled FP16 projections. | Removing one conversion boundary changes candidates. | retain-on |
| FlashInfer TopK capability gate | SM70 DFlash2 uses `torch.topk`, not FlashInfer TopK. | #253 | FlashInfer TopK is unavailable below SM80. | Forcing it can fail or change ordering. | retain-on |
| Physical draft SWA admission bound | `min(window-1 + max_num_batched_tokens, max_model_len)` in `kv_cache_interface.py:468`. | #253 | With window 2048 and scheduler 8192, one request needs at most 10,239 logical positions plus block alignment. | This is not a separate global draft allocator. | retain-on |
| Prefix reuse without Eagle block drop | `SpeculativeConfig.use_eagle_kv_cache()` excludes flat DFlash. | #253 | MRV2 stores projected draft context KV with the target cache. | Restoring Eagle recompute can eliminate hybrid prefix hits. | retain-on |
| Target c8 verifier graph size | Eight requests times eight queries equals 64. | #253 plus deployment correction | Automatic TP4 policy captures only through four requests unless explicitly extended. | Missing 64 falls back and reduces utilization. | retain-on at `64` for c8 |
| E5M2 batched XQA routing and wide-load gates | `VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING=1` selects eligible target-only routing. The kernel then checks p256 and page geometry. | #268 plus existing route policy | The current DFlash target does not satisfy Python `spec_config is None` batch-context routing. The trace gate proves future hits. | Treat it as retained target-only behavior, not DFlash evidence. | retain-on for target-only |

## Final retained defaults

The FP16-KV TP4 qualification at `a39f22ed5` supersedes the provisional recommendations in this matrix.

Two official c8 windows measured 259.68 and 280.77 aggregate tok/s. The median was 270.23 tok/s.

Exact changed-input, 8K, 32K, 128K, repeated-prefix, and repetition gates passed.

| Area | Proposed value | Recommendation | Reason |
| --- | --- | --- | --- |
| Topology | TP4, DCP1 | retain-on | Matches the pinned V100 topology and sliding-window restriction. |
| Context and slots | 262,144 context, eight slots | retain-on | Matches the transfer target. |
| Scheduler | 8,192 batched tokens | retain-on | Supports prefill chunks and bounded draft SWA allocation. |
| Draft geometry | seven drafts, block eight, selector K16, window 2048 | retain-on | Checkpoint-owned contract. |
| Target backend | Flash-V100 | retain-on | Required SM70 route. |
| Draft backend | Flash-V100 non-causal | retain-on | Required diffusion attention semantics. |
| Graphs | target and draft through c8, verifier max 64 | retain-on | Direct c8 throughput evidence supports the missing shapes. |
| Prefix caching | enabled with Mamba align | retain-on | Exact prefix evidence exists. |
| Mamba convolution cache dtype | `auto` | retain-on | The final arm preserves the model configuration. |
| Mamba SSM cache dtype | resolved FP32 | retain-on | The final arm preserves recurrent-state precision. |
| Target KV | `auto`, resolved FP16 | retain-on | The final objective excludes KV quantization. Exact long-context gates passed. |
| Draft KV | explicit `auto` or FP16 | retain-on | Avoid target E5M2 inheritance. |
| Draft sampling | greedy bring-up, probabilistic official-sampling A/B | needs paired qualification | Modes have different acceptance and memory contracts. |
| Fused selector | off | retain-off | It was slower and used more memory. |
| Shared GDN umbrella | off initially | needs paired qualification | Prefix align conflicts with fused metadata eligibility. |
| Fused GDN metadata | off initially | needs paired qualification | Strong B1 evidence exists, but c8 and align remain unresolved. |
| Fused GDN verifier | off | needs paired qualification | Component parity exists without complete service evidence. |
| Fused GDN norm | on | retain-on | Exact target-graph reduction evidence exists. |
| Fused GDN split | on | retain-on | Exact graph reduction evidence exists. |
| Fused small-query metadata | on | retain-on | Exact route and 12.1% round reduction evidence exists. |
| Fused Gemma RMS | on | retain-on | Exact hashes and quality gates passed. Matched c8 improved 11.58%. |
| Sparse rejection | off | retain-off | The live greedy arm is ineligible. |
| QPN8 | off | retain-off for c8 | It improved c1 but reduced matched c8 throughput by 17.17%. |
| TP4 push all-reduce | off | retain-off | Full-model paired evidence remains incomplete. |
| Dynamic Flash partitions | off for this TP4 arm | retain-off | Static partitions avoid prior workspace pressure. |
| Diagnostic dumps and traces | off | retain-off | They add synchronization, files, or route changes. |

## Interaction matrix

| Combination | Current behavior | Required check |
| --- | --- | --- |
| Prefix caching `on` + unspecified Mamba mode | Resolves to `align`. | Confirm exact repeated-prefix retrieval after every combined-stack change. |
| Mamba `align` + fused GDN metadata `1` | Fused metadata is ineligible because it requires `none`. | Compare prefix-on align against prefix-off or explicit none. Do not combine results. |
| Compiled SM70 graphs + fused selector `1` | Startup normalizes selector to `0`. | Treat selector tests as eager-only. |
| Draft-only `speculative_config.enforce_eager=true` + DFlash2 | Commit `a39f22ed5` disables only draft CUDA graphs. | Use it for draft parity tests. Keep it false for the retained service. |
| LM-head fastpath, top-one, or Tensor Core top-one layout + fused selector | The controls share LM-head layout preparation. | Hold all three LM-head controls fixed during selector tests and record memory. |
| Eight slots + automatic TP4 verifier shapes | Automatic source policy stops at four requests and 32 verifier tokens. | Supply 40, 48, 56, and 64 explicitly. |
| `max_num_seqs>8` + QPN8 | QPN8 falls back to TurboMind. | Keep eight slots for the accepted QPN8 contract. |
| Explicit QPN8 `1` + missing operators | Startup fails closed. | Build and load the matching `_C` extension. |
| Unset QPN8 + missing operators | Automatic route warns and falls back. | Record the route-hit log before any benchmark. |
| E5M2 target KV + draft KV `null` | PR #253 resolves SM70 DFlash draft KV to `auto`. | Keep explicit `auto` in manifests for audit clarity. |
| Mamba cache dtype `auto` + SSM dtype `auto` | The live Qwen arm resolves persistent SSM state to FP32. | Record resolved dtypes, not only CLI strings. |
| Mamba FP16 convolution and SSM dtypes | Matches pinned SGLang knobs but changes live vLLM state precision. | Require exact trajectory, retrieval, quality, memory, and throughput gates. |
| Draft window 2048 + DCP greater than one | Sliding-window memory sizing asserts. | Keep DCP1. |
| Draft window 2048 + scheduler budget 8192 | Physical draft admission uses about 10,239 positions per request plus alignment. | Report actual draft and target block counts. |
| Sparse rejection + greedy temperature zero | Sparse route falls back to dense. | Use official probabilistic top-K20 sampling for its A/B. |
| Sparse rejection + penalties, grammar, bias, bad words, or logprobs | Sparse route falls back to dense. | Add fallback route tests for every unsupported feature. |
| Push all-reduce + non-fully-connected topology | Push storage is not selected. | Verify physical GPU mapping and peer access first. |
| Push all-reduce + wrong tensor shape | Generic custom all-reduce remains active. | Record route-hit counts for `[8,5120]`. |
| `VLLM_SM70_TP4_M5_AR_THREADS` + PR #266 push route | The M5 thread override does not configure `[8,5120]` push. | Keep it unset and separate both route counters. |
| Fused Gemma RMS + non-FP16 input or non-FP32 residual | Fused route is ineligible. | Verify dtype logs and eager fallback. |
| Dynamic partitions + low memory headroom | Workspace growth can OOM under admission. | Keep static partitions for initial combined tests. |
| #268 E5M2 wide load + DFlash speculative target | `VLLM_FLASH_V100_XQA_BATCH_CONTEXT_ROUTING` requires `spec_config is None`. | Keep the wide-load and route controls classified as target-only. Use the trace control to prove any future route hit. |
| Greedy draft + probabilistic baseline evidence | Acceptance numbers are not comparable. | Hold draft and target sampling constant in every A/B. |
| Full c8 graphs + larger graph memory | Graph memory rose from 0.44 to 0.94 GiB in the base arm. | Record minimum free memory during warmup and load. |

## Unresolved paired tests

Run each pair with identical weights, prompt order, seeds, graph sizes, and sampling.

1. Do not add an E5M2 target-KV arm to this retained profile. The final objective requires FP16 KV.
2. Compare automatic draft KV resolution against explicit `auto`. Require identical route logs and output hashes.
3. Keep automatic Mamba convolution dtype and FP32 SSM state. The final objective excludes a precision reduction.
4. Require exact recurrent states before any future Mamba dtype experiment.
5. Compare prefix-on Mamba align against a no-prefix Mamba-none arm before enabling fused GDN metadata.
6. Draft-only `speculative_config.enforce_eager=true` passed its route test and now disables draft graphs only.
7. Test shared GDN metadata at c1, c2, c4, and c8 with mixed lengths and poisoned persistent buffers.
8. Test fused GDN verifier off versus on with FP16 and FP32 recurrent state.
9. Test fused GDN norm and split together, then disable each leaf independently.
10. Fused Gemma RMS passed exact greedy hashes, long retrieval, repeated-prefix, and repetition gates. Logprob and perplexity expansion remains optional.
11. Test sparse rejection under standard probabilistic sampling, top-K20, and every dense fallback feature.
12. QPN8 off versus on completed. Retain it off because c8 regressed 17.17%.
13. TP4 push all-reduce completed its service A/B. Retain it off because c8 regressed 3.01%.
14. Test static versus dynamic decode partitions with identical admission and free-memory telemetry.
15. Test automatic graph sizes against explicit c8 64-token coverage. Record full-graph route hits.
16. FP16 KV passed exact 8K, 32K, and 128K retrieval. Near-262K remains a future stress gate.
17. Test repeated-prefix retrieval after every metadata, rejection, and collective combination.
18. Run excluded graph capture, JIT, and sustained node warmup before every official 2-4 minute window.

## Pinned SGLang-V100 code-reference plan

Use SGLang-V100 as a design reference, not as a patch source. Review semantics and tensor layouts before every port.

| Area | Pinned SGLang-V100 references | 1Cat-vLLM destination | Required semantics and layout review |
| --- | --- | --- | --- |
| Selector placement | `python/sglang/srt/speculative/triton_ops/dflash_selector.py`, `python/sglang/srt/speculative/triton_ops/dflash_sampling.py`, `python/sglang/srt/models/dflash.py` | `qwen3_dflash2.py`, `dflash2/speculator.py`, `vocab_parallel_embedding.py` | Compare anchor/mask rows, selector K16, TP local/global TopK, equal-score tie order, FP32 scores, and sampled path state. |
| GDN metadata | `python/sglang/srt/speculative/dflash_worker.py`, `python/sglang/srt/speculative/dflash_info.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/jit_kernel/tests/test_fused_verify_triton_gdn.py` | `gdn_attn.py`, `mamba_hybrid.py`, `attn_utils.py`, `qwen_gdn_linear_attn.py` | Compare request classification, padded rows, state-slot ownership, accepted counts, per-step FP16 reload, and commit order. |
| Non-causal D256 attention | `python/sglang/srt/layers/attention/flash_attn_v100_backend.py`, `patches/flash-attention-v100-sglang.patch` | `flash_attn_v100.py`, `flash-attention-v100/kernel` | Compare causality, scale convention, D256 layout, page tables, KV dtype, K/V scales, sequence lengths, and output stride. |
| Graph node reduction | `python/sglang/srt/speculative/dflash_worker.py`, `python/sglang/srt/speculative/dflash_worker_v2.py`, `python/sglang/srt/speculative/triton_ops/dflash_prepare_block.py`, `python/sglang/srt/speculative/triton_ops/dflash_accept_bonus.py` | target and draft graph managers, metadata builders, rejection path | Compare graph boundaries, persistent buffer ownership, complete overwrite rules, changed-input replay, and object lifetime. |
| KV windowing and allocation | `python/sglang/srt/speculative/dflash_worker.py`, `python/sglang/srt/speculative/dflash_worker_v2.py` | `SlidingWindowSpec`, scheduler block managers, draft cache groups | The pinned worker uses logical suffix indexing with target-global allocation. Do not infer a physical ring allocator. Compare absolute RoPE and local KV addresses. |
| Synchronization | `python/sglang/srt/speculative/dflash_worker.py`, `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` paths | graph capture, context-KV precompute, metadata transitions | Compare stream acquisition inside capture, event ownership, side-stream joins, host fences, and buffer lifetimes. |
| TP collectives | `python/sglang/jit_kernel/csrc/distributed/custom_all_reduce_push.cuh`, `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`, `python/sglang/jit_kernel/tests/test_custom_all_reduce.py` | `custom_all_reduce.cuh`, `custom_all_reduce.py` | Compare rank order, NaN sentinel protocol, epoch storage, topology eligibility, tensor shape, graph registration, and changed-input replay. |

Before a port, capture these source facts from both implementations:

- Tensor shapes, strides, dtypes, and ownership.
- Absolute and local token indices.
- Causal and non-causal mask definitions.
- State selection and accepted-token conventions.
- TP rank and candidate-ID offsets.
- Graph capture stream and event lifetime.
- Fallback behavior for unsupported requests.
- Numeric classification as Type A, Type B, or pending.

Do not copy a kernel because its name or shape appears similar. First prove that semantics, layout, and state lifetime match.

## Provenance summary

| Change | Configuration effect |
| --- | --- |
| PR #252 | Makes `method=dflash` MRV2-only and establishes parallel block drafting. |
| PR #253 | Adds DFlash2 model fields, draft KV dtype control, prefix reuse, sliding-window allocation, non-causal Flash-V100, grouped RMSNorm, and SM70 numeric adaptation. |
| PR #254 | Adds `VLLM_SM70_DFLASH2_FUSED_SELECTOR`, default-off and eager-only. |
| PR #257 | Adds the shared metadata umbrella, fused metadata, shadow oracle, and packed GDN verifier gates. |
| PR #266 | Adds GDN norm, split, small-query metadata, Gemma RMS, sparse rejection, QPN8, and push all-reduce controls. |
| PR #267 | Makes speculative beta FP32 without a user option. |
| PR #268 | Adds default-on batched E5M2 XQA wide loads for its accepted target-only route. |
| vLLM #53017 | Uses the true draft-logit column stride without a user option. |

## Review rule

A sane default needs a source route-hit, exact configuration record, correctness gate, memory record, and paired performance window.

If any item is absent, classify the option as **needs paired qualification** or **retain-off**.
