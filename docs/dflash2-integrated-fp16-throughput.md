# Integrated DFlash2 TP4 FP16-KV result

## Result

The retained service runs the complete DFlash2 stack on one four-V100 NVLink island.

The target is `Qwen3.8-27B-FP8`. The draft is the existing unquantized `Qwen3.8-27B-DFlash2` checkpoint.

Target KV and draft KV use `auto`, which resolves to FP16 for these model configurations. No KV quantization is active.

The persistent Mamba SSM state remains FP32. The service does not reduce recurrent-state precision.

| Metric | Final retained result |
| --- | ---: |
| c1 throughput | 75.76 tok/s |
| c8 aggregate median | 270.23 tok/s |
| c8 per-slot median | 33.78 tok/s |
| c8 verifier round latency | 80.78 ms |
| c8 accepted draft tokens per round | 1.73 |
| KV capacity | 634,281 tokens |
| Minimum free memory | 5,427 MiB per GPU |
| Final restarts | 0 |

Two official c8 windows lasted 157.73 and 145.89 seconds. Both windows used eight active requests and zero waiting requests.

The two windows produced 259.68 and 280.77 aggregate tok/s. Their median was 270.23 aggregate tok/s.

All four GPUs reached 90% median SM use. Mean SM use remained between 86.5% and 87.8% across ranks.

## Retained configuration

```text
tensor_parallel_size=4
decode_context_parallel_size=1
max_model_len=262144
max_num_seqs=8
max_num_batched_tokens=8192
gpu_memory_utilization=0.80
target kv_cache_dtype=auto  # FP16
draft kv_cache_dtype=auto   # FP16
mamba_cache_dtype=auto
mamba_ssm_cache_dtype=float32
mamba_cache_mode=align
enable_prefix_caching=true
num_speculative_tokens=7
draft_sample_method=greedy
target attention backend=FLASH_ATTN_V100
draft attention backend=FLASH_ATTN_V100
VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS=0
VLLM_SM70_FP8_QPN8=0
VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA=1
VLLM_SM70_DFLASH2_FUSED_GDN_NORM=1
VLLM_SM70_DFLASH2_FUSED_GDN_SPLIT=1
VLLM_SM70_DFLASH2_FUSED_GEMMA_RMS=1
VLLM_SM70_DFLASH2_FUSED_SELECTOR=0
VLLM_SM70_DFLASH2_VERIFY_FASTPATH=0
VLLM_SM70_DFLASH2_FUSED_GDN_METADATA=0
VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY=0
VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION=0
VLLM_SM70_TP4_PUSH_ALLREDUCE=0
```

The target graph sizes are `1,2,4,8,9,16,18,24,32,40,48,56,64`.

The verifier graph sizes are `8,16,24,32,40,48,56,64`. The draft graph covers request counts one through eight.

## Source and build integrity

The retained source is commit `a39f22ed5a0be62cd6d6d9f9936bdbf0aa17fab6` on `experiment/dflash2-pr-integration-20260823`.

The immutable projection is `/workspace/dflash2-integrated-a39f22ed5a`.

The build used a fresh FetchContent directory after a stale shared dependency tree rejected its patch set.

The build compiled these artifacts for SM70:

- `vllm/_C.abi3.so`
- `vllm/_C_stable_libtorch.abi3.so`
- `flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so`
- `paged_kv_utils.cpython-312-x86_64-linux-gnu.so`

The projection checks every core and Flash-V100 artifact checksum before server startup.

The full `_C` build contains the QPN8 and TP4 push-all-reduce operators. The retained runtime leaves both routes disabled.

## Correctness

The final qualification passed all required service gates.

- Eight changed-input prompts produced eight distinct hashes.
- Two replays matched all eight hashes.
- Final hashes matched the earlier control hashes for all eight prompts.
- The 8K, 32K, and 128K retrieval checks returned exact secrets.
- Two repeated 32K prefix requests returned identical exact secrets.
- The second repeated prefix request used 29,952 cached tokens.
- The 512-token repetition audit had a 1.0 unique-line ratio.
- Every request returned HTTP 200.
- The service reported zero restarts.

The final TP4 smoke returned HTTP 200 and exactly 512 tokens. All four GPUs reached 75-76% SM use during that request.

## Mixed-prefill crash fix

Cold c8 admission exposed an existing DFlash2 failure. Chunked-prefill rows can temporarily use parallel anchor ID `248320`.

That ID equals the vocabulary size. The compiled selector embedding rejected it and terminated all TP ranks.

Commit `a39f22ed5` adds an opaque Triton sanitizer before selector embedding. It covers candidate IDs and anchor IDs.

Valid TopK and anchor IDs remain unchanged. Invalid temporary rows map to vocabulary boundaries and remain subject to exact target verification.

This guard removed the c8 device assertion. The exact control hashes and long-context retrieval checks still matched.

The same commit makes draft-only `speculative_config.enforce_eager=true` functional. The final service keeps draft CUDA graphs active.

## Throughput decisions

| Arm | c1 tok/s | c8 aggregate tok/s | Decision |
| --- | ---: | ---: | --- |
| All optional routes off | 58.55 | 226.85 | control |
| GDN and small-query fusions | 62.28 | 231.45 | retain |
| Gemma fusion added | 76.00 | 258.26 | retain |
| QPN8 default stack | 78.67 | 191.73 | reject |
| Gemma plus push all-reduce | 71.32 | 250.49 | reject |
| Final Gemma repeats | 75.76 | 259.68 / 280.77 | retain |

Gemma fusion improved the matched c8 arm by 11.58%. It also improved c1 by 22.02%.

QPN8 improved c1 but reduced c8 by 17.17% against the matched fusion arm. The final service disables QPN8.

TP4 push all-reduce reduced c8 by 3.01% against Gemma without push. It also reduced c1 by 6.15%.

The fused selector remains off because it is eager-only, slower, and uses more memory.

Shared GDN metadata remains off because the validated route requires Mamba mode `none`. The retained prefix configuration requires `align`.

Sparse target rejection remains off because the retained greedy draft mode is ineligible.

Dynamic Flash-V100 partitions remain off because static partitions avoid prior workspace pressure.

NCCL and PyNCCL retain automatic transport selection. The service verified a fully connected four-GPU NVLink island.

## Rejected runtime change

A follow-up draft-state change skipped writes from chunked-prefill rows. It passed throughput tests but failed all long retrieval checks.

The 8K, 32K, and 128K results returned `duct` instead of their exact secrets. That change was removed before the final commit.

## Live state

The final pod is `onecat-dflash2-integrated-fp16` in namespace `llm` on `gpu-01`.

The pod reports ready, running, and zero restarts. Its worker processes are `TP0`, `TP1`, `TP2`, and `TP3`.

The separate SGLang DFlash2 pod retained its original UID and zero restarts throughout this work.

Machine-readable evidence is in `docs/dflash2-integrated-fp16-throughput.json`.
