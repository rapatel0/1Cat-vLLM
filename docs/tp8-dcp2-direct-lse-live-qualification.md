# TP8+DCP2 direct-LSE live qualification

Date: 2026-08-20

Canonical source: `53893bfb47c847aa0e4fe5d01ff77b560516223d`

Hardware: `gpu-01`, 8 x Tesla V100-SXM2-32GB

Machine-readable summary:
[`tp8-dcp2-direct-lse-live-qualification.json`](tp8-dcp2-direct-lse-live-qualification.json).

## Verdict

The direct local decode-LSE implementation is **correctness-qualified** on
V100 for scalar and XQA kernels, empty/uneven shards, strided output, MTP4
CUDA graphs, 8K/32K/128K retrieval, and repeated-prefix use. The specific
profiled component is also qualified: the old standalone
`lse_reconstruction` range is absent on all ranks.

It does **not yet earn an overall throughput promotion claim**. Against the
prior three-run direct-service artifact, MTP4 c1 was within the plan's 2%
regression bound (`-1.49%`), but c32 median was `-4.49%` with a wide
310–379 tok/s cohort range. The last cohort matched the prior range, so this
is not enough evidence to revert the direct-LSE change, but it does not prove
an end-to-end speedup.

The next bottleneck is the three-collective chain: query all-gather, LSE
all-gather, and output reduce-scatter total **470.15 us/call** in the host
ranges. After communication-path work, the next planned target remains q>1
MTP4 correction batching.

Per the gpu-01 operating policy, the qualified TP8+DCP2+MTP4 experiment remains
active for the next iteration. The old TP8 control is scaled to zero. The
gateway was not retargeted.

## Exact build and runtime projection

This commit changes the Flash-V100 pybind ABI. No Python-only overlay was used.
A full extension was built with the serving image's PyTorch and CUDA toolchain:

- PyTorch: `2.10.0+cu128`;
- CUDA toolkit: `12.8.93`;
- architecture: `sm_70`;
- source: `/localpool/onecat-vllm-hy3-sm70/direct-lse-53893bfb47-src`;
- immutable projection:
  `/localpool/onecat-vllm-hy3-sm70/dcp2-branch-53893bfb47`;
- container projection: `/workspace/dcp2-branch-53893bfb47`.

The build used the full `flash-attention-v100/setup.py build_ext --inplace`
path, compiling all six Flash-V100 extension translation units and the paged
KV utility. The extension imported against the runtime PyTorch before any
all-GPU swap. The shared venv was never edited.

### Source hashes

| File | SHA256 |
|---|---|
| `flash-attention-v100/flash_attn_v100/flash_attn_interface.py` | `c639d26bdb7b9a09c2369e2b654796f623d89974f8b8ff9fe96d41084ff0f93b` |
| `flash-attention-v100/kernel/flash_decode_paged.cu` | `08b7eee89903d25172b47bd9565717718bfd42124d2c03fd46a4a23a0795efcc` |
| `flash-attention-v100/include/fused_mha.h` | `ff0d3449958fc5c15586523c9c7a07f3e796bb2323b49be3140d3b529d6701b9` |
| `flash-attention-v100/kernel/fused_mha_api.cpp` | `60ad8e3d99abcbd84db30c50c555738d37739383d64af3c7fd08256801143222` |
| `vllm/v1/attention/backends/flash_attn_v100.py` | `8b9a1c34e73a85d0ef8d334e386f5cd9e84ea8136b00cc39edfa2c40242d6789` |
| `vllm/v1/attention/ops/common.py` | `0607a65b87041b5f70ab244af681bca8d19e945a8816a3cbda2d6cce48ddf56d` |

### Binary hashes

| Binary | SHA256 |
|---|---|
| `flash_attn_v100_cuda.cpython-312-x86_64-linux-gnu.so` | `ad5a8d31cab8a6ebfecd13bd5b13d0ff2e5d0ddddf0a00685ae69f6f7cee42b0` |
| `paged_kv_utils.cpython-312-x86_64-linux-gnu.so` | `fab66ab5d8235c42563684e478b120dc231d6693a7f9e37249c8fc5228e55f08` |

Every canary startup checked the projection's `SHA256SUMS` before importing
vLLM.

## Direct-LSE GPU and ABI tests

The rebuilt extension's ABI/import check passed against runtime PyTorch. The
new SM70 tests then ran on V100 GPU 0:

```text
scalar direct LSE versus workspace and dense reference      PASS
XQA direct LSE versus workspace and dense reference         PASS
return_lse=false leaves final-LSE workspace untouched       PASS
extension writes non-contiguous/strided FP32 final LSE      PASS
```

Result: **4 passed**, including sequence lengths `[0, 257, 513]`. This covers
an empty local shard, uneven multi-partition rows, scalar decode, XQA decode,
and actual-stride writes by the extension.

## MTP4 live qualification

Configuration:

- TP8, DCP2, native MTP4;
- `FLASH_ATTN_V100` for target and drafter;
- max model length 262,144;
- `max-num-seqs=32`, `max-num-batched-tokens=16384`;
- `gpu-memory-utilization=0.78`;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `dcp_comm_backend=ag_rs`.

Startup and full CUDA graph capture passed. The first benchmark startup
reported 2,090,088 KV tokens. The final active non-profiler startup reported
2,097,152 KV tokens. Both pass the 2.0M MTP4 floor. Available KV memory was
18.02 GiB and graph capture used **4.88 GiB**.

### Correctness

| Gate | Prompt tokens | Result | Wall |
|---|---:|---|---:|
| 8K needle | 7,984 | Pass | 5.8597 s |
| 32K needle | 31,988 | Pass | 9.2299 s |
| 128K needle | 127,493 | Pass | 66.2423 s |
| repeated 32K prefix, first | 31,987 | Pass | 9.0466 s |
| repeated 32K prefix, second | 31,987 | Pass | 9.0392 s |

### Matched direct-service benchmark

The existing harness was used with unique prompts, direct service access,
server completion-token counts, separate exact-shape warmups, and three
measured runs/cohorts.

| Metric | Direct LSE | Prior artifact | Change |
|---|---:|---:|---:|
| c1 median, ~1K prompt / 512 output | **50.6951 tok/s** | 51.4641 | -1.49% |
| c32 median, ~256 prompt / 256 output | **360.0948 tok/s** | 377.0046 | -4.49% |
| c1 accepted drafts/verifier step | 1.4144 | 1.4413 | diagnostic |
| c32 accepted drafts/verifier step | 1.5200 | 1.4955 | diagnostic |

Measured c1 runs: 52.6451, 47.5225, 50.6951 tok/s.

Measured c32 cohorts: 310.0407, 360.0948, 378.9249 tok/s. The wide first-cohort
outlier prevents a confident regression or improvement claim. NVTX and Nsight
were disabled for this benchmark.

Raw result SHA256:
`7d36882f4f1232667870e801a019d4413459494f971b950f8543173b3bd721ae`.

## No-speculation shared-path check

The same TP8+DCP2 commit and rebuilt extension were restarted without a
speculative config. DCP NVTX remained disabled.

- 8K exact needle: pass, 7,985 prompt tokens, 3.3032 s;
- c1 median: **55.1923 tok/s**;
- c1 runs: 55.1762, 55.1923, 55.3049 tok/s;
- c32 median: **430.5089 tok/s**;
- c32 cohorts: 415.8893, 430.5089, 435.0101 tok/s;
- graph footprint: **0.46 GiB**;
- available KV memory: 18.09 GiB;
- KV tokens: **2,326,331**.

This proves the shared direct-LSE path remains correct and operational without
MTP. The steady c1 result is above the prior final-correction 45.5–50.4 tok/s
range. There is no retained post-correction c32 baseline with identical
methodology. The old 539 tok/s c32 result predates final LSE correction and is
not a valid regression comparator.

At utilization 0.78, no-MTP did **not** meet the plan's 2.48M capacity floor.
That capacity result should be reconciled before a no-MTP capacity claim; it is
not attributable to the tiny `[B,H]` FP32 final-LSE workspace by itself.

Raw result SHA256:
`a3f0855af229aa4c5945a67dbd5fed710d1b0e9797233daec7a47f2a151e47c2`.

## NVTX / Nsight result

`VLLM_FLASH_V100_DCP_DECODE_NVTX=1` was enabled only for a separate MTP4
profile restart. The warmed capture included c1 and c32 work. Every worker had
819 calls per remaining DCP stage.

| Stage | Mean host span | Median host span |
|---|---:|---:|
| query all-gather | 174.703 us | 167.407 us |
| LSE all-gather | 160.600 us | 155.611 us |
| output reduce-scatter | 134.852 us | 122.154 us |
| output correction | 78.546 us | 76.081 us |
| local attention | 132.726 us | **22.536 us** |
| **LSE reconstruction** | **absent: 0 calls** | — |

The old standalone LSE reconstruction was 306.227 us/call. It is completely
absent now. Its time did not visibly move into normal local-attention host
dispatch: the new local-attention median is 22.536 us versus the old 22.867 us
mean. The new local-attention mean is distorted by a small number of multi-ms
outliers and rank skew; the p90 is only 26.667 us.

As before, Nsight exported NVTX data but no CUDA API/kernel tables. These are
host spans, not device-kernel attribution. No device-timing claim is made.

Persistent gpu-01 artifacts:

| Artifact | SHA256 |
|---|---|
| `/srv/dev/dcp2-direct-lse-profile-53893bfb47/direct-lse-53893bfb47-all-ranks.nsys-rep` | `ab974500c6013c41d1fd7402fdbc8a5fde5a98eeaaf817f2d00eb786ff5b6bfa` |
| `/srv/dev/dcp2-direct-lse-profile-53893bfb47/direct-lse-53893bfb47-all-ranks.sqlite` | `2f5ef368fd40fec958690e78f06ef70343b4c3713e143c132be781b324c0d483` |

## Commands run

Representative commands:

```bash
# Full ABI-compatible build in an isolated build pod.
CUDA_HOME=/workspace/bench-hy3/forge-001/toolchain/cuda-12.8.1-a99a1860/cuda-12.8 \
TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 \
  /opt/venv/bin/python setup.py build_ext --inplace

# Direct-LSE GPU tests.
/opt/venv/bin/python -m pytest \
  /tmp/test_sm70_flash_v100_decode_planner.py -q \
  -k 'direct_lse or without_lse or strided_final_lse' -vv

# Existing service harness, run once for MTP4 and once without speculation.
/opt/venv/bin/python /tmp/benchmark_qwen38_dcp2_service.py \
  --base-url http://127.0.0.1:8000 --needle-target-tokens 8000 \
  --single-prompt-tokens 1024 --single-output-tokens 512 --single-runs 3 \
  --aggregate-concurrency 32 --aggregate-prompt-tokens 256 \
  --aggregate-output-tokens 256 --aggregate-runs 3 --output RESULT.json

# Delayed all-rank host profile.
nsys profile --session-new=directlse538profile --start-later=true --kill=none \
  --sample=none --cpuctxsw=none --trace=cuda,nvtx,nccl --nccl-trace=all \
  --cuda-graph-trace=node --trace-fork-before-exec=true \
  --output=/profiles/direct-lse-53893bfb47-all-ranks \
  taskset -c 0-79 /opt/venv/bin/python -m vllm.entrypoints.openai.api_server ...
nsys start --session=directlse538profile
# warmed c1 + c32 workload
nsys stop --session=directlse538profile
```

## Final gpu-01 and gateway state

- `llm/qwen38-27b-fp8-tp8`: `0/0`;
- `llm/qwen38-27b-fp8-tp8-dcp2`: desired/ready/available `1/1/1`;
- active DCP2 mode: TP8+DCP2+native MTP4, direct-LSE commit `53893bfb47`;
- DCP NVTX: disabled;
- direct DCP2 `/health`: passed;
- gateway config before/after is byte-identical, SHA256
  `adedf22babdc6478f33ab1d6b0d74e37991aeac517abe50e8f89808f82f146b7`;
- gateway target remains
  `http://qwen38-27b-fp8-tp8.llm.svc.cluster.local/v1` by operator policy.

The gateway target is intentionally unchanged even though the old TP8 control
is scaled to zero; use the DCP2 service directly for continued experiments
unless the operator explicitly requests a gateway retarget.
