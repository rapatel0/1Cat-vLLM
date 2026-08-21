# TP8+DCP2 native MTP3 batch scaling

## Verdict

Larger batches increase aggregate throughput, but they do not raise sustained SM utilization.

The stable c64 test reached 902.68 tok/s, 16.01% above c32 at 778.10 tok/s.

C64 increased median request latency by 69.93%. It also left about 321 MiB free on the most constrained GPU.

Max-seqs 96 failed during sampler warmup. Max-seqs 128 failed during CUDA graph capture.

The final service uses max-seqs 32. It is healthy at `1/1`.

The evidence identifies device dependencies and graph replay gaps as the low-utilization cause. Scheduler capacity is not the primary cause.

## Fixed policy

The sweep changed only `max_num_seqs`.

| Policy | Value |
|---|---|
| Runtime source | `d299e4acbc`, byte-equivalent to `59a5fa11f0` |
| MTP depth | 3 |
| TP / DCP | TP8 / DCP2 |
| Attention | `FLASH_ATTN_V100` |
| DCP route | packed A2A plus direct query gather |
| TP route | automatic PyNCCL |
| Draft sample method | probabilistic |
| Maximum batch tokens | 16,384 |

The batch-token limit stayed at 16,384. A c64 prompt batch uses exactly 64 × 256 = 16,384 prompt tokens.

A c64 q=4 verifier uses only 256 decode tokens. A larger batch-token limit does not increase decode concurrency.

## Short fixed-corpus matrix

Each cell used two warmups and five measured cohorts. Each request used a fixed 256-token prompt and 256 completion tokens.

| Max sequences | c32 tok/s | c40 tok/s | c48 tok/s |
|---:|---:|---:|---:|
| 32 | **719.97** | 529.74 | 555.73 |
| 40 | 669.63 | **833.86** | 560.08 |
| 48 | 673.49 | 785.41 | **902.92** |

Requests above the configured limit entered the scheduler queue.

- Max-seqs 32 reported eight waiting requests at c40 and sixteen at c48.
- Max-seqs 40 reported eight waiting requests at c48.
- Max-seqs 48 reported zero waiting requests at c48.

The c40 profile increased saturated throughput by 15.82%. Its c32 result decreased by 6.99%.

The c48 profile increased saturated throughput by 25.41%. Its c32 result decreased by 6.46%.

Median request latency increased from 11.00 seconds at c32 to 11.97 seconds at c40.

Median request latency increased to 13.18 seconds at c48.

The short test defines a Pareto trade-off. Max-seqs 32 protects c32, while max-seqs 48 favors saturated aggregate throughput.

## Closed-loop steady state

The closed-loop client replaced each completed request immediately. Each request generated 2,048 tokens.

Each measured interval lasted 600 seconds. Scheduler and GPU telemetry used one-second command intervals.

| Metric | c32, max-seqs 32 | c64, max-seqs 64 | Change |
|---|---:|---:|---:|
| Aggregate throughput | 778.10 tok/s | **902.68 tok/s** | **+16.01%** |
| Median request latency | **84.25 s** | 143.16 s | +69.93% |
| Verifier batch step | **100.73 ms** | 175.18 ms | +73.91% |
| Completion tokens/request step | 2.4493 | 2.4708 | +0.88% |
| Accepted drafts/request step | 1.4492 | 1.4705 | +1.47% |
| Active requests, mean | 31.90 | 63.67 | — |
| Active requests, p50 | 32 | 64 | — |
| Full-active samples | 91.83% | 77.33% | — |
| Waiting requests, maximum | 0 | 0 | — |
| SM use, mean | **76.39%** | 69.33% | -7.06 points |
| SM use, p50 | **83%** | 79% | -4 points |
| SM use, p95 | 86% | **89%** | +3 points |
| GPU idle samples | 0% | 0% | — |

The one-second sampler cannot see submillisecond graph gaps. The GPU trace provides that attribution.

The client kept the p50 active count at the configured limit. Immediate replacements caused brief active-count dips.

C32 per-GPU mean SM use ranged from 73.61% to 79.75%.

C64 per-GPU mean SM use ranged from 64.94% to 75.89%.

C32 mean PCIe RX ranged from 1,631.61 to 1,796.71 MiB/s per GPU. Mean TX ranged from 218.84 to 233.86 MiB/s.

C64 mean PCIe RX ranged from 1,345.44 to 1,484.74 MiB/s per GPU. Mean TX ranged from 186.81 to 200.69 MiB/s.

NVLink throughput counters returned `N/A` on all V100 links. Device-level GPM NVLink metrics were unavailable.

No thermal limit occurred. The maximum temperature was 57 C.

## Admission and memory

| Max sequences | q=4 graph shapes | Result | Graph memory | Exact limiter or headroom |
|---:|---|---|---:|---|
| 32 | 1 through 32 | Pass | 1.99 GiB | 2,123,901 KV tokens |
| 40 | 1 through 40 | Pass | 2.48–2.95 GiB | Stable short c40 load |
| 48 | 1 through 48 | Pass | 3.43 GiB | 1,785 MiB minimum free during short load |
| 64 | 1 through 64 | Pass | 4.98 GiB | About 321 MiB free at the steady peak |
| 96 | 1 through 96 | Fail | 6.93 GiB before failure | Sampler warmup OOM with 96 dummy requests |
| 128 | 1 through 128 | Fail | Incomplete | CUDA graph capture memory allocation failed |

The max-seqs 96 attempt had 56.5–220.5 MiB free before failed allocations of 92–274 MiB.

The max-seqs 128 attempt failed during `capture_end` with `cudaErrorMemoryAllocation`.

Neither attempt used eager fallback. MTP3 and all service policy remained unchanged.

## All-rank c64 trace

A bounded Torch trace captured 36.49 seconds across all eight ranks.

The trace contained active `generation_64` graph annotations. This independently confirms the expanded runtime shape.

The union of GPU kernels occupied 70.63% of the trace wall. Device idle gaps occupied 29.37%.

The gap heuristic attributed 29.06% of wall time to gaps inside graph replay. It attributed only 0.31% to host or scheduler gaps.

Serialized CUDA kernel duration had this distribution:

| Category | Share |
|---|---:|
| FP8 GEMM | **38.79%** |
| Other linear GEMM | 18.54% |
| TP all-reduce | **17.71%** |
| Attention and GDN | 12.08% |
| DCP query all-gather | 2.27% |
| DCP packed A2A | 2.13% |
| Other | 8.49% |

Serialized shares can include stream overlap. The device-busy value uses the union of all kernel intervals.

The trace confirms real device gaps after c64 fills the scheduler. More active requests do not remove those gaps.

The bounded profiler produced all eight trace files. Its final flush exceeded 900 seconds and timed out the instrumented engine.

The clean non-profiled restart passed after this profiler-only failure.

## Correctness and final state

Exact 8K retrieval passed for max-seqs 32, 40, 48, and 64.

All short saturated outputs completed. The 2,048-token steady outputs passed repeated-line and digit-run checks.

Final state:

- TP8+DCP2: `1/1`, healthy.
- Native MTP3: active.
- Maximum sequences: 32.
- Maximum batch tokens: 16,384.
- KV capacity: 2,123,901 tokens.
- CUDA graph memory: 1.99 GiB.
- Profiler: disabled.
- Final exact 8K retrieval: pass.

## Decision

Batch capacity limits aggregate throughput at c40 and above. It does not explain the low sustained SM percentage.

Max-seqs 48 is the safer throughput-oriented profile. It reaches 902.92 tok/s in short c48 tests and retains more memory than max-seqs 64.

Max-seqs 32 remains the balanced profile. It protects c32 performance and retains about 4 GiB of device headroom.

The next performance work must reduce GEMM, TP reduction, and graph-internal dependency gaps.

## Artifacts

Persistent root:

```text
/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp3-batch-scaling
```

Manifest SHA256:

```text
195537ef3b62bcac7de3da2db3c324b9a083845eccddac021950479846a85a5b
```

The root contains 70 files and eight compressed all-rank trace files.

## Residual risks

- The dmon command used a one-second delay, but eight-GPU collection produced fewer than 570 steady samples per rank.
- NVLink traffic counters were unavailable on this driver.
- Short cohort variance remained material.
- Max-seqs 64 passed ten minutes, but its 321 MiB peak margin is not safe for retention.
