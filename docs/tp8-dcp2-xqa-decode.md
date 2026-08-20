# TP8+DCP2 q=1 XQA decode qualification

Date: 2026-08-20

Candidate source: `deb0fa7da1e1d6fc2a7ce69c274108b659ffcf7e`

Machine record:
[`tp8-dcp2-xqa-decode.json`](tp8-dcp2-xqa-decode.json).

## Verdict

The narrow DCP q=1 XQA route is exact and graph-safe, but it is slower than
scalar paged decode. The candidate is rejected for performance.

The qualified scalar configuration remains active on experimental `gpu-01`:

```text
TP8 + DCP2 + packed A2A + native MTP4
+ uniform q=5 correction batching
+ direct query gather
+ scalar q=1 paged decode
```

XQA passed all correctness gates. However, MTP4 c1 and c32 regressed 6.17%
and 6.50%. No-MTP c32 regressed 12.02%.

The annotated XQA local-attention host span also showed no component gain. Its
median was 22.7845 microseconds versus 22.5900 microseconds for scalar decode.
Nsight did not provide device-kernel timing.

## Implementation

The candidate added one narrow selector inside
`FlashAttnV100Impl._flash_v100_decode`. It required all existing XQA
conditions plus these DCP constraints:

- one query token per live request;
- a direct-LSE-capable XQA wrapper;
- six gathered Q heads over one local KV head;
- FP16 Q/K/V execution with head dimension 256;
- full causal attention with no sliding window;
- compatible decode metadata and partition hints.

The selected XQA kernel emitted local FP32 LSE from its existing split-KV
reduction. Packed A2A then applied the unchanged exact cross-rank correction.

The scalar DCP route remained an explicit fallback. Route counters identified
XQA, scalar fallback, and each communication backend independently.

The XQA wrapper emitted a `local_attention_xqa` NVTX range. The MTP4 q=5
prefix/verifier route remained separate and unchanged.

## Source and GPU tests

The focused CPU/source suite passed **22 tests**. Coverage included:

- q-per-KV 6 selection;
- exact output and LSE parity;
- scalar fallback;
- empty and uneven local contexts;
- unsupported shape and window rejection;
- q>1 XQA exclusion;
- A2A route counters;
- XQA-specific NVTX evidence.

The new two-rank V100 test passed:

```text
test_distributed_dcp2_q1_xqa_q_per_kv6_a2a_cuda_graph
1 passed, 38 deselected
```

It captured XQA plus packed A2A in one CUDA graph. It covered empty and uneven
local sequence lengths and compared output/LSE with dense global attention.

## Correctness and capacity

| Gate | Result |
|---|---:|
| 8K exact needle | Pass |
| 32K exact needle | Pass |
| 128K exact needle | Pass |
| repeated 32K prefix | Pass / Pass |
| two-rank XQA+A2A graph replay | Pass |
| MTP4 graph capture | Pass |
| no-MTP 8K smoke | Pass |

MTP4 retained **2,097,152 KV tokens** and **2.04 GiB** of CUDA graph memory.
No-MTP reported 2,326,331 KV tokens and 0.46 GiB of graph memory.

## Performance

The direct-service harness used unique prompts, server token counts, separate
warmups, and three measured runs or cohorts.

### Native MTP4

| Metric | XQA candidate | Qualified scalar | Change |
|---|---:|---:|---:|
| c1 median | **47.8691 tok/s** | 51.0174 | **-6.17%** |
| c32 median | **596.0862 tok/s** | 637.5508 | **-6.50%** |

Candidate c1 runs:

```text
47.8691, 47.8324, 50.0576 tok/s
```

Candidate c32 cohorts:

```text
505.4918, 596.0862, 676.0370 tok/s
```

Accepted drafts per verifier step were 1.3568 at c1 and 1.5084 at c32. The
c32 acceptance result matched the scalar range, so acceptance did not explain
the aggregate loss.

### No MTP

| Metric | XQA candidate | Qualified A2A scalar | Change |
|---|---:|---:|---:|
| c1 median | **55.4541 tok/s** | 55.8002 | -0.62% |
| c32 median | **388.2960 tok/s** | 441.3470 | **-12.02%** |

The no-MTP result independently confirmed that q=1 XQA did not improve this
DCP shape.

## All-rank route evidence

The all-rank capture recorded **6,744 XQA calls**, with exactly 843 calls on
each of eight workers. It recorded zero scalar `local_attention` calls in the
candidate profile.

| Route | Calls | Median host span | p90 host span |
|---|---:|---:|---:|
| XQA local attention | 6,744 | 22.7845 us | 27.6140 us |
| qualified scalar local attention | 6,624 | 22.5900 us | 27.3895 us |

The candidate profile also recorded 720 uniform-prefix and 720 uniform-suffix
ranges. This confirms that q>1 MTP4 verification stayed on the qualified batch
route rather than XQA.

Nsight again exported NVTX ranges but no CUDA kernel tables. The host spans
only prove route execution and dispatch timing.

Persistent profile artifacts:

- `/srv/dev/dcp2-direct-lse-profile-53893bfb47/xqa-deb0fa7da1-all-ranks.nsys-rep`
  - SHA256 `345817f8b9ad32df340e796da2e0db01a5a7dabb5e6b055e1ee491432f40645c`
- `/srv/dev/dcp2-direct-lse-profile-53893bfb47/xqa-deb0fa7da1-all-ranks.sqlite`
  - SHA256 `01187fce6e9977c332608ce599c1c79f2c1ba1ee9ab65a4a75ab8d079e38729f`

## Projection and final state

The candidate projection is:

```text
/localpool/onecat-vllm-hy3-sm70/dcp2-branch-deb0fa7da1
```

Its manifest SHA256 is
`594ae367f62cbb44a9f63ce3231965b419580348b9d77c85d77061e6d606cf82`.
The shared venv was unchanged.

The rejected candidate remains in Git for source evidence. The live service
uses the prior qualified scalar projection:

- `qwen38-27b-fp8-tp8`: `0/0`;
- `qwen38-27b-fp8-tp8-dcp2`: `1/1`, healthy;
- runtime projection: `dcp2-branch-e276623a92`;
- DCP NVTX: disabled;
- gateway: unchanged.

## Next action

Do not invest further in q-per-KV 6 XQA for DCP q=1 on this V100 shape. The
next planned experiment is graph-safe overlap of query all-gather with the
independent replicated suffix work.
