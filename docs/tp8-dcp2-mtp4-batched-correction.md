# TP8+DCP2 MTP4 batch correction qualification

Date: 2026-08-20

Runtime source: `9bc01fd4a3691632dfeeb63f8875a757f84d2214`

Final source and test tip before this record:
`1d71b50cae53158753e04a2cf157b121dde23fc0`

Machine record:
[`tp8-dcp2-mtp4-batched-correction.json`](tp8-dcp2-mtp4-batched-correction.json).

## Verdict

The uniform MTP4 DCP correction route is correct and substantially improves
aggregate throughput. It remains active on experimental `gpu-01`.

The c32 median increased from 356.4301 to **595.8024 tok/s**, a 67.16% gain
against the qualified A2A control. It is 58.04% above the original final DCP2
artifact at 377.0046 tok/s.

The official three-run c1 median was 49.4195 tok/s. A seven-run follow-up had a
52.9340 tok/s median, 4.30% above the A2A control. Acceptance variance caused
large c1 variance, but the uniform route does not run for one request.

The batch route reduced the q=5 correction call count from 2,438 to 157 per
rank in matched profile envelopes. This is a 15.53x reduction. Total A2A calls
across all profile shapes fell 59.77%.

Graph memory fell from 4.13 GiB to **2.04 GiB**. KV capacity stayed at
**2,090,088 tokens**.

## Implementation

`FlashAttnV100Impl._forward_with_dcp` now selects one narrow batch route when
all live requests have the same small causal query length. Native MTP4 uses
five query tokens per request.

The route performs these exact operations:

1. Gather query heads once across each DCP pair.
2. Reshape tokens into request-major `[B,Q,H,D]` order.
3. Compute causal suffix attention once for the batch.
4. Compute local paged-prefix attention once for the batch.
5. Flatten request tokens without cross-request attention.
6. Apply one exact packed A2A LSE correction.
7. Merge each prefix state with its causal suffix state.

Paged-prefix attention retains each request's block table and local sequence
length. Empty local shards use zero output and negative-infinity LSE. Uneven
local context lengths remain valid within one uniform query batch.

The route supports explicit `a2a` and `ag_rs` backends. A2A remains the active
backend.

The old per-request path remains explicit. It handles these cases:

- one request;
- unequal query lengths;
- noncausal attention;
- query lengths outside the configured small-query limit;
- mismatched token counts;
- invalid metadata structure.

New route counters distinguish batch execution from fallback execution. The
counters are:

- `prefill_prefix_dcp_uniform_batch`;
- `prefill_prefix_dcp_uniform_batch_a2a`;
- `prefill_prefix_dcp_uniform_batch_ag_rs`;
- `prefill_prefix_dcp_per_request_fallback`.

## Correctness

### Source and CPU tests

The focused Flash-V100 DCP test file passed **16 tests**. Tests cover:

- four-request MTP4-like q=5 input;
- empty and uneven prefix contexts;
- exact dense-attention parity;
- one combine per rank for A2A and AG+RS;
- noncontiguous verifier query views;
- irregular query-length fallback;
- fallback combine counts and route counters.

### Distributed CUDA graph test

A two-rank NCCL test used the exact c32 verifier combine shape:

```text
input tokens:       160 = 32 requests x 5 tokens
heads before A2A:     6
heads after A2A:      3
head dimension:     256
```

The test compiled Triton, reserved persistent buffers on the capture stream,
captured A2A in a CUDA graph, replayed it, and compared output and LSE against
the dense reference. Result: **1 passed, 36 deselected**.

### Service gates

| Gate | Result |
|---|---:|
| 8K exact needle | Pass |
| 32K exact needle | Pass |
| 128K exact needle | Pass |
| repeated 32K prefix | Pass / Pass |
| no-MTP 8K smoke | Pass |
| MTP4 CUDA graphs | Pass |
| MTP4 KV capacity | 2,090,088 tokens |
| MTP4 graph memory | 2.04 GiB |

The 32K gate used 31,988 prompt tokens. The 128K gate used 127,494 prompt
tokens. Both exact secrets matched.

## Performance

### Native MTP4 direct-service result

The existing direct-service harness used unique prompts, server token counts,
separate warmups, and three measured runs or cohorts.

| Metric | Batch route | Prior A2A | Change |
|---|---:|---:|---:|
| c1 median, official three runs | 49.4195 | 50.7535 | -2.63% |
| c1 median, seven-run follow-up | **52.9340** | 50.7535 | **+4.30%** |
| c32 median | **595.8024** | 356.4301 | **+67.16%** |
| c32 accepted drafts/step | 1.5050 | 1.5018 | diagnostic |

Official c1 runs were 49.4195, 49.9476, and 46.8676 tok/s.

The seven follow-up c1 runs were:

```text
52.9611, 51.1229, 52.9340, 53.7277, 49.0402, 53.0012, 51.9053
```

The c32 cohorts were:

```text
507.2152, 595.8024, 649.5790 tok/s
```

The c32 gain does not come from a higher acceptance rate. The accepted draft
count per step stayed near the A2A control.

### No-MTP smoke

The no-MTP route passed an 8K exact needle. A short one-run smoke measured
51.1393 tok/s at c1 and 344.0542 tok/s at c32.

Those smoke requests used 128 and 64 output tokens. They are not comparable
with the retained 512 and 256 output-token no-MTP benchmark.

## All-rank NVTX evidence

The matched control commit did not select the batch route because verifier
query tensors were strided views. Commit `9bc01fd4a3` uses an exact
request-order reshape, which supports those views under graph capture.

Every rank recorded 88 `uniform_prefix_attention` and 88
`uniform_suffix_attention` ranges across capture and the profiled workload.
The full c32 A2A payload was 495,360 bytes.

| Full c32 stage | Calls/rank | Median host span |
|---|---:|---:|
| pack | 63 | 76.9895 us |
| all-to-all | 63 | 146.2310 us |
| unpack/combine | 63 | 72.7515 us |
| total | — | **295.9720 us** |

One old q=5 per-request combine had a 254.774 us median envelope. The new
32-request payload is 32 times larger, but its median envelope is only 16.17%
larger than one old combine.

Matched profile counts per rank were:

| Counter | Per-request control | Batch candidate |
|---|---:|---:|
| q=5 per-request combines | 2,438 | 69 fallback calls |
| uniform combines | 0 | 88 |
| total q=5 correction calls | 2,438 | **157** |

Total A2A collective events across all shapes fell from 30,368 to 12,216.

Nsight again exported no CUDA kernel tables. All component times are NVTX host
spans, not device-kernel times.

## Projection and artifacts

The final checksum-recorded projection is:

```text
/localpool/onecat-vllm-hy3-sm70/dcp2-branch-1d71b50cae
```

Projection manifest SHA256:
`46e12c24245babf5637cf7aac18be502d229840df02f88748bd20160748ff39c`.

Profile artifacts:

- `/srv/dev/dcp2-direct-lse-profile-53893bfb47/batched-9bc01fd4a3-all-ranks.nsys-rep`
- `/srv/dev/dcp2-direct-lse-profile-53893bfb47/batched-9bc01fd4a3-all-ranks.sqlite`

Their SHA256 values are in the machine record. The shared venv was not
modified.

## Final experimental state

- `qwen38-27b-fp8-tp8`: `0/0`;
- `qwen38-27b-fp8-tp8-dcp2`: `1/1`, healthy;
- active mode: TP8+DCP2+A2A+native MTP4+uniform batch correction;
- source projection: `1d71b50cae`;
- runtime backend source: `9bc01fd4a3`;
- DCP NVTX: disabled;
- gateway: unchanged.

## Residual risks

The c32 result still trails the historical 1,730 tok/s DCP1 anchor. The batch
route closes a material part of the gap but does not finish the campaign.

Nsight device timing remains unavailable. Host spans and end-to-end throughput
provide the current evidence.

The batch route is restricted to small uniform causal queries. Mixed query
lengths still use the slower per-request fallback by design.
