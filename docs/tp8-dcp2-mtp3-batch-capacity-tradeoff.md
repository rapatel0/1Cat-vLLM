# TP8+DCP2 MTP3 batch-capacity tradeoff

## Outcome

The KV trade made c96 safe and faster.

The separate c96 profile reached 972.46 tok/s, 25.79% above the fresh c32 control.

The profile uses:

- `max_num_seqs=96`
- `gpu_memory_utilization=0.67`
- `max_num_batched_tokens=16384`

It passed 600 seconds, raw-exact 8K and 128K retrieval, and the memory gate.

C128 did not pass the combined admission and memory gate.

The balanced c32 profile remains active at `1/1`.

## Fixed policy

Only these configuration limits changed:

- maximum sequences.
- GPU memory utilization.
- maximum batch tokens.

The fixed runtime policy was:

| Policy | Value |
|---|---|
| Runtime source | `d299e4acbc`, byte-equivalent to `59a5fa11f0` |
| MTP depth | 3 |
| TP / DCP | TP8 / DCP2 |
| Attention | `FLASH_ATTN_V100` |
| DCP route | packed A2A plus direct query gather |
| TP route | automatic PyNCCL |
| Draft sample method | probabilistic |
| Image digest | `sha256:253e98bfd4a3f9e89187321b37dae01dd27642b3dc11546be881ce188df96c72` |

No source, gateway, shared environment, or model policy changed.

## Baseline memory budget

The baseline per-GPU budget was:

| Item | Value |
|---|---:|
| Physical memory | 32.00 GiB |
| Model memory | 4.53 GiB |
| KV reservation | 18.02 GiB |
| CUDA graph memory | 1.99 GiB |
| Calculated non-reclaimable footprint | 3.13-3.20 GiB |
| Idle physical free memory | 4.00-4.06 GiB |
| KV capacity | 2,123,901 tokens |

The non-reclaimable value is physical use minus the model, KV, and graph values.

The final clean restart reported 2,116,258 KV tokens. This 7,643-token variance occurred in earlier clean restarts.

## Protocol

Each request used a fixed 256-token source-neutral prompt and 2,048 completion tokens.

The client replaced each completed request immediately.

The first c32 and qualified c96 windows lasted 600 seconds.

Later boundary windows lasted 120-240 seconds after supervisor direction reduced test duration.

GPU and scheduler telemetry used one-second intervals.

A deep backlog used 128 clients for c96 and 160 clients for c128.

Each valid profile passed raw-exact 8K and 128K c1 retrieval.

The memory gate required at least 1,024 MiB physical free memory under load.

## C32 control

| Metric | Result |
|---|---:|
| Aggregate throughput | 773.11 tok/s |
| Median request latency | 85.31 s |
| P95 request latency | 93.58 s |
| Active p50 | 32 |
| Waiting maximum | 0 |
| SM mean / p50 / p95 | 76.79% / 83% / 86% |
| Minimum physical free memory | 2,778 MiB |
| 8K / 128K retrieval | raw-exact pass |

## C96 result

The first point, utilization 0.74, captured all q=4 shapes through c96.

It left only 545 MiB idle free memory, so it did not enter a load test.

Utilization 0.69 passed load but left only 492 MiB at peak.

Utilization 0.67 passed every gate with the 16,384-token budget.

| Metric | C32 control | C96 throughput profile |
|---|---:|---:|
| Aggregate throughput | 773.11 tok/s | **972.46 tok/s** |
| Change | control | **+25.79%** |
| Median request latency | 85.31 s | 198.43 s |
| P95 request latency | 93.58 s | 243.55 s |
| Active p50 | 32 | **96** |
| Active mean | 31.88 | 95.42 |
| Waiting maximum | 0 | 0 |
| SM mean / p50 / p95 | 76.79% / 83% / 86% | 65.06% / 66.5% / 88% |
| Minimum physical free memory | 2,778 MiB | **1,138 MiB** |
| KV tokens | 2,123,901 | 1,711,960 |
| Graph memory | 1.99 GiB | 6.52 GiB |
| 8K / 128K retrieval | pass | raw-exact pass |

The 16,384-token budget needs two token passes for a `96 x 256` prompt batch.

The active p50 of 96 proves that this limit did not restrict steady decode admission.

A corrected one-pass test used `max_num_batched_tokens=32768` and 128 clients.

That point ended with 96 active requests and 32 waiting requests.

It reached 899.61 tok/s, but the minimum free memory fell to 358 MiB.

The larger token budget also reduced KV capacity to 1,489,558 tokens.

It increased graph memory to 6.93 GiB.

The c96 one-pass point failed the memory gate.

## C128 result

A `128 x 256` prompt batch needs exactly 32,768 scheduler tokens.

All c128 tests used `max_num_batched_tokens=32768`.

Every tested point captured all q=4 shapes through c128.

Every tested point passed raw-exact 8K and 128K retrieval.

| GPU utilization | KV tokens | Graph | Active p50 / max | Throughput | Free minimum | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 0.58 | 1,152,516 | 8.84 GiB | 107 / 115 | 814.03 tok/s | 1,886 MiB | reject, KV limit |
| 0.60 | 1,227,414 | 8.62 GiB | 114 / 123 | 724.83 tok/s | 1,478 MiB | reject, KV limit |
| 0.61 | 1,250,342 | 8.62 GiB | 117 / 125 | 877.68 tok/s | 1,292 MiB | reject, KV limit |
| 0.62 | 1,287,791 | 8.62 GiB | 120 / 128 | 935.27 tok/s | 950 MiB | reject, memory limit |

At utilization 0.61, KV use reached 100% before c128 admission.

At utilization 0.62, the scheduler reached 128 only for 32.14% of measured samples.

The p50 stayed at 120, and physical free memory fell below 1 GiB.

No tested c128 point satisfies both admission and memory safety.

The c128 throughput values are not saturated c128 results.

## Decision

The active balanced profile remains:

```text
max_num_seqs=32
gpu_memory_utilization=0.78
max_num_batched_tokens=16384
```

The separate qualified throughput profile is:

```text
max_num_seqs=96
gpu_memory_utilization=0.67
max_num_batched_tokens=16384
```

The c96 profile trades KV capacity and latency for 25.79% more aggregate throughput.

Do not use c128 with this 32 GiB memory budget.

## Configuration fingerprints

The hash input contains the sorted image, command, environment, resources, node selector, volumes, and security context.

| Profile | SHA256 |
|---|---|
| Baseline c32, initial and final | `79b427cb9c7243485a940a33983792115919a2cec768211bde7ba59a3b7d4bc0` |
| C96, utilization 0.67, tokens 16,384 | `fc5bbd8c0618fc689be6c385412f16373fcd2ce1a1abf35f3ccca287af79a409` |
| C96, utilization 0.67, tokens 32,768 | `79e1be9654297e25b5a3685cf2d9591c4d2cd09b2f12ef4b8601262c6ff6b1ee` |
| C128, utilization 0.58 | `d7d9333de08580917189339a4243ac93c0e7448bba3d9a6933010dab5e291d48` |
| C128, utilization 0.60 | `93ab7ec2e93bdd2357066214b4dc4c6a0372266ee58e8cfbb285e95944304238` |
| C128, utilization 0.61 | `54ca46921ed771c8868027a35c70a0492e0c5b503799ed3eb61b625d5e7ff66f` |
| C128, utilization 0.62 | `5240933eebf8df7ca4c68ec019dd8280b647778bf95a5a14b317bb840e2f52ca` |

The initial and final c32 hashes match.

## Final state

- TP8+DCP2 is healthy at `1/1`.
- Native MTP3 is active.
- Maximum sequences is 32.
- GPU memory utilization is 0.78.
- Maximum batch tokens is 16,384.
- Graph memory is 1.99 GiB.
- KV capacity is 2,116,258 tokens.
- Final 8K and 128K retrieval passed raw-exact checks.

## Artifacts

Persistent evidence is under:

```text
/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp3-batch-capacity-tradeoff
```

The manifest contains 88 files.

Its SHA256 is:

```text
9db700f44452e5781503e833ec0796cca07be81b585e2e3109617aea546cf02e
```

## Review findings

- **High:** C128 has no safe point with active p50 at 128.
- **High:** C96 at utilization 0.67 is the fastest safe tested profile.
- **Medium:** One-pass 32,768-token prompt admission removes too much KV and physical margin at c96.
- **Correct:** All valid c96 and c128 points passed raw-exact 8K and 128K retrieval.

## Residual risks

- The c96 throughput profile uses two initial prompt token passes.
- The c128 boundary windows lasted two to four minutes.
- C128 throughput values do not represent saturated c128 work.
- The baseline KV report varies by 7,643 tokens across clean starts.
