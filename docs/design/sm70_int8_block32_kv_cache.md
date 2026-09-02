# SM70 signed INT8 block32 KV-cache validation

## Scope

This report covers the signed block32 INT8 KV-cache implementation for the
SM70 `FLASH_ATTN_V100` backend, Qwen3.8-27B-FP8 plus DFlash2, hybrid Mamba
cache alignment, and CUDA Graph replay. The source base is
`94f64714263e3af4399bbc0dd78069e5ce3a84b9`; the reviewed implementation diff
before this report has SHA256
`d80f2c26c17f0feca04525340ec4fa452eead4fd8cdc3e50f55ed1b09a866baf`.

The validated runtime used Python 3.12, CUDA 12.8, PyTorch 2.10.0+cu128, and
four Tesla V100-SXM2-32GB GPUs at compute capability 7.0.

## Representation and update contract

Each physical page contains signed INT8 K and V payloads, separate FP16 K and V
scales per KV head and 32-channel block, and one INT32 publication owner.
Page-scale growth computes the batch-final scale first and re-quantizes
historical values at most once before publishing incoming values. Page reset is
triggered by a write to offset zero, so reused pages do not retain stale payload
or scale state. One elected CTA owns each physical page during an update,
including reordered and noncontiguous slot mappings.

Hybrid caches retain logical INT8 page width while using the unified physical
page stride. Both the legacy and V2 GPU model runners create strided views, and
the backend preserves physical stride and nonzero base storage offsets when it
splits payload, scale, and owner views.

## Build and package evidence

The final synchronized source built this wheel:

- path: `/workspace/int8-block32-build-final/wheels/1cat_vllm-1.5.0.dev0+sm70int8-cp312-cp312-linux_x86_64.whl`
- size: 151,242,314 bytes
- SHA256: `ae2ea6340750798230a09e6971511b4b46f6b09379ac553e18602f3320539cf8`
- independent install: `/workspace/int8-block32-build-final/install-final`

The independent install imported `vllm._C` and the bundled Flash-V100 CUDA
extension. The extension exposed the INT8 reshape, paged decode, and paged
prefill operators. Packaged Python files compared byte-for-byte equal to the
synchronized source.

## Test evidence

The final wheel install passed 19 focused tests. Coverage includes:

- signed payloads and independent K/V scales;
- batch-final scale growth and historical re-quantization;
- reordered block tables and page reuse/reset;
- padded physical-page stride and nonzero storage offsets;
- FP16-draft hybrid page growth behavior;
- CUDA Graph replay across parameterized decode shapes;
- persistent DFlash2 small-query metadata, graph padding masks, and overflow
  rejection.

Ruff, Ruff format, `git diff --check`, pi-lens blocking diagnostics, and an
independent read-only diff review passed. The reviewer returned "OK with
notes"; its only note was the missing nonzero-storage-offset regression test,
which was added and passed.

## Final-wheel graph smoke

The independent final wheel completed TP4 DFlash2 generation with
`int8_block32`, Flash-V100, TurboMind FP8 weights, and Mamba align. PIECEWISE,
FULL target, and FULL DFlash2 graphs all captured and replayed. The request
returned 16/16 tokens without corruption.

- aggregate output: 36.849 tok/s
- steady decode: 48.458 tok/s
- TTFT: 0.1234 s
- acceptance length: 1.600
- result SHA256: `35d16add5d3296102f1448e438b29c75eb1475689c72402809b650858cd9f907`

## FP16 versus INT8 performance

The matched short-context matrix used TP4, DFlash2 q7, Flash-V100, FULL CUDA
Graphs, prefix caching, Mamba align, three repeats, 128 output tokens, and the
same prompt/seeds. `auto` resolves to the FP16 target cache in this runtime.
Cold TTFT includes the first full prompt; warm TTFT includes prefix-cache hits.

| Target KV | Batch / input | Aggregate mean / median | Mean steady decode | Cold / warm TTFT |
| --- | --- | ---: | ---: | ---: |
| FP16 (`auto`) | 1 / 8,192 | 78.457 / 95.171 tok/s | 119.979 tok/s | 2.095 / 0.282 s |
| INT8 block32 | 1 / 8,192 | 13.402 / 15.786 tok/s | 22.802 tok/s | 8.582 / 2.955 s |
| FP16 (`auto`) | 8 / 4,096 | 80.889 / 97.510 tok/s | 15.016 tok/s/request | 3.829 / 1.747 s |
| INT8 block32 | 8 / 4,096 | 46.035 / 51.836 tok/s | 9.088 tok/s/request | 11.081 / 5.706 s |

INT8 is therefore functional but not a performance win: aggregate throughput
is 82.9% lower at batch 1 and 43.1% lower at batch 8 in this matrix.

Result SHA256 values:

- FP16 B1: `7d5e7f78457b24adc602ef08387bfd0db7f59904bdd1b4e1157fb0b7e29e67ab`
- INT8 B1: `0873fa20d6fdaa04a746479db3ca906db84b161f348f6a1b0673bcbb5dc0bdf1`
- FP16 B8: `7010f190217df3d3b835e69748673a1b165b84c52df6ad4e584f7c2a178279d2`
- INT8 B8: `ba99fc66416c7d9fb4990831c71a3b4a9e3b34d14851343f0d9982e4e9beb5ff`

## Quality evidence

A deterministic 16-question GSM8K comparison used the same shuffled rows,
greedy DFlash2 drafting, temperature zero, 512 output tokens, and graph mode.
Both cache formats passed and failed the exact same questions.

| Target KV | Accuracy | Invalid answers | Aggregate output | Acceptance length |
| --- | ---: | ---: | ---: | ---: |
| FP16 (`auto`) | 13/16 (81.25%) | 0/16 | 194.714 tok/s | 4.3463 |
| INT8 block32 | 13/16 (81.25%) | 0/16 | 184.627 tok/s | 4.3141 |

- dataset SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
- FP16 result SHA256: `e5b3d1cac7076600baef6365436f3aa0d62e44663c42b5cabf370f904385595c`
- INT8 result SHA256: `825a1e7b687096f938fe4cf75e3e0ddd7143253dcc52b91ec92a85f8ae192bc3`

## Long-context result

The FP16 control completed 261,632 input plus 512 output tokens at batch 1:

- cold TTFT: 376.593 s;
- warm steady decode: 16.81--18.68 tok/s;
- aggregate mean: 11.770 tok/s;
- result SHA256: `b251d1e28cffb27b35f81a68a6ae8a5d9970cdf65456130f2f6dcf582440a819`.

The corresponding INT8 engine initialized, captured graphs, and reported a
927,111-token cache capacity (3.54x concurrency at 262,144 tokens), but the
first measured request did not complete after approximately 1 hour 47 minutes.
The enclosing matrix reached its two-hour bound and was terminated. The long
batch-8 cases were not reached. This is a measured performance timeout, not an
OOM or cache-capacity rejection.

## Flash-Next and TP8 disposition

The requested `unsloth/Qwen3.8-Flash-Next-FP8` checkpoint contains 185.5 GB of
weight files. It cannot fit a TP4 deployment on the available 128 GB aggregate
VRAM. The staging job failed without creating a destination, and its Kubernetes
job was deleted. The only local Flash-Next checkpoint is NVFP4, not FP8. TP8
was not attempted because only four GPUs are exposed. No image or checkpoint
was promoted.

## Reproduction and retained artifacts

The local evidence bundle is under
`.pi/benchmarks/results/int8-block32/` in the enclosing workspace. It includes
the wheel checksum, final-wheel smoke, short matrix, GSM8K pair and dataset,
long-context control, timed-out INT8 log, and raw logs. The benchmark driver is
`.pi/benchmarks/benchmark_qwen38_dflash_kv.py`.

The essential runtime settings were:

```bash
export PYTHONPATH=/workspace/int8-block32-build-final/install-final
export FLASH_ATTN_V100=1
export VLLM_SM70_FP8_TURBOMIND=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_LEVEL=NVL
```

## Promotion decision

Correctness, graph replay, packaging, and the bounded GSM8K quality comparison
pass. Performance does not: the short matrix regresses materially and the 261K
INT8 request exceeds the two-hour campaign bound. This implementation must not
be promoted as a faster or production-default KV-cache path. It remains an
explicit experimental cache format pending kernel performance work and a
completed long-context and batch-8 matrix.
