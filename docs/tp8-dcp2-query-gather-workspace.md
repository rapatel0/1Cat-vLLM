# TP8+DCP2 direct query-gather qualification

Date: 2026-08-20

Runtime source: `e276623a9238ffd29a7b8258ebceae3384b4d101`

Projection tip: `601f574e99d88bb1ca3274d689287e291738ee4c`

Machine record:
[`tp8-dcp2-query-gather-workspace.json`](tp8-dcp2-query-gather-workspace.json).

## Verdict

The direct PyNCCL query-gather path is exact, graph-safe, and faster. It remains active on experimental `gpu-01`.

The c32 median increased from 595.8024 to **637.5508 tok/s**. This result is a **7.01% gain** against the qualified A2A batch baseline.

The c1 three-run median increased from 49.4195 to **51.0174 tok/s**. The prior seven-run median was 52.9340 tok/s.

The direct path reduced the median annotated query-gather envelope from 200.630 to 124.767 microseconds. This is a **37.81% component reduction**.

Exact 8K, 32K, and 128K retrieval passed. Repeated 32K prefix retrieval passed twice.

MTP4 retained 2,097,152 KV tokens and 2.04 GiB of graph memory.

## Implementation

The old path called `GroupCoordinator.all_gather(query.contiguous(), dim=1)`. Its base communicator allocated rank-major output and reformatted it for head order.

The new path uses the established DCP `PyNcclCommunicator` directly. It uses three separate persistent tensors:

- a contiguous local-input tensor for strided input;
- a rank-major collective tensor with `[rank, token, local_head, dim]` order;
- a head-major output tensor with `[token, rank * local_head, dim]` order.

The final copy explicitly changes rank-major order to token/head-major order. No reshape operation hides an allocation.

The cache key includes:

- the DCP group name;
- the device;
- the CUDA stream;
- the DCP world size;
- the dtype;
- the shape;
- the input stride.

Buffers from one key do not alias. Separate streams and shapes receive separate storage.

Contiguous input uses its original graph-stable address. Strided input copies into the persistent local-input tensor.

The path verifies communicator world size, local rank, and device before use. A mismatch uses the existing coordinator path.

Compiler, meta, fake-tensor, CPU, disabled-feature, and unavailable-PyNCCL cases also use the existing path.

The direct code does not catch collective failures. This fail-closed behavior prevents one rank from entering a different collective sequence.

## NVTX and route evidence

The new stages are:

- `query_gather_cache_acquire`;
- `query_gather_prepare_direct`;
- `query_all_gather_direct`;
- `query_gather_reformat`;
- `query_gather_prepare_fallback`;
- `query_all_gather_fallback`.

Route counters distinguish `dcp_query_gather_direct` and `dcp_query_gather_fallback`.

Every worker emitted 1,232 direct calls in the captured workload. The total was 9,856 direct calls and zero fallback calls.

| Stage | Calls | Median host span |
|---|---:|---:|
| cache acquire | 9,856 | 25.5155 us |
| direct collective | 9,856 | 61.0155 us |
| rank/head reformat | 9,856 | 38.2355 us |
| direct input copy | 0 | 0 us |
| **component sum** | — | **124.7665 us** |
| prior coordinator envelope | 3,648 | 200.6300 us |

The service query input was contiguous, so the direct input-copy stage did not run.

Nsight again exported no CUDA kernel tables. These values are host NVTX spans, not device kernel times.

## Tests

The focused CPU/source DCP suite passed **19 tests**. It covers rank order, shape keys, stride keys, pointer reuse, non-aliasing, and fallback labels.

A two-rank V100 test used 160 tokens, three local heads, and head dimension 256. This matches the MTP4 c32 query-gather shape.

The V100 test passed these checks:

- direct PyNCCL rank order;
- exact head order;
- strided-input equivalence;
- same-stream pointer reuse;
- shape separation;
- stream separation;
- non-aliasing storage;
- CUDA graph capture and replay after input changes.

Result: **1 passed, 37 deselected**.

## MTP4 performance

The direct-service harness used unique prompts, server token counts, separate warmups, and three measured runs or cohorts.

| Metric | Direct gather | Qualified A2A batch baseline | Change |
|---|---:|---:|---:|
| c1 median | **51.0174** | 49.4195 | +3.23% |
| c32 median | **637.5508** | 595.8024 | **+7.01%** |
| c32 accepted drafts/step | 1.5258 | 1.5050 | diagnostic |

Direct c1 runs were 54.5565, 51.0174, and 50.7116 tok/s.

Direct c32 cohorts were 637.5508, 626.3068, and 677.9984 tok/s.

The previous seven-run c1 median was 52.9340 tok/s. The direct three-run result is 3.62% lower than that longer sample.

The c1 route does not use uniform batch correction. Acceptance variance remains a major source of c1 variance.

## Correctness and capacity

| Gate | Result |
|---|---:|
| 8K exact needle | Pass |
| 32K exact needle | Pass |
| 128K exact needle | Pass |
| repeated 32K prefix | Pass / Pass |
| two-rank direct gather graph | Pass |
| MTP4 graph capture | Pass |
| no-MTP 8K smoke | Pass |

The 32K test used 31,988 prompt tokens. The 128K test used 127,494 prompt tokens.

The final MTP4 startup reported 2,097,152 KV tokens and 2.04 GiB of graph memory.

The no-MTP smoke reported 2,326,331 KV tokens and 0.46 GiB of graph memory.

## Projection and artifacts

The checksum-recorded projection is:

```text
/localpool/onecat-vllm-hy3-sm70/dcp2-branch-e276623a92
```

Its manifest SHA256 is:

```text
bdf0ba5c098558158623d3ab7bd38beb3c0b5a3c78200e1e0bddac51db45af21
```

Profile artifacts remain on `gpu-01`:

- `/srv/dev/dcp2-direct-lse-profile-53893bfb47/query-gather-direct-601f574e99-all-ranks.nsys-rep`
- `/srv/dev/dcp2-direct-lse-profile-53893bfb47/query-gather-direct-601f574e99-all-ranks.sqlite`

Their SHA256 values are in the machine record. The shared venv was not changed.

## Active state

- old TP8: `0/0`;
- TP8+DCP2: `1/1`, healthy;
- active mode: TP8+DCP2+A2A+MTP4+uniform correction+direct query gather;
- direct query gather: enabled;
- DCP NVTX: disabled;
- gateway: unchanged.

## Residual risks

The c32 result still trails the historical 1,730 tok/s DCP1 anchor.

The persistent cache retains one small buffer set for each observed group, stream, shape, dtype, and stride.

Large or irregular prefix shapes can add cache entries. Current graph shapes and long-context tests did not cause an operational memory regression.

Nsight device timing remains unavailable. Host spans and end-to-end throughput provide the current evidence.
