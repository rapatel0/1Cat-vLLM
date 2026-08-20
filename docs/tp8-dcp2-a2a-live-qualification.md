# TP8+DCP2 packed-A2A live qualification

Date: 2026-08-20

Canonical source: `b996cc4a393459075317d2d8fa24227612c003d9`

Machine-readable record:
[`tp8-dcp2-a2a-live-qualification.json`](tp8-dcp2-a2a-live-qualification.json).

## Verdict

The packed A2A backend is **correctness- and graph-qualified** for the
Flash-V100 TP8+DCP2 path and remains active on experimental `gpu-01`.

A2A replaces the LSE all-gather plus corrected-output reduce-scatter with one
packed output+FP32-LSE all-to-all. Per full-attention decode call, the DCP
collective count therefore changes from:

```text
AG+RS: query AG + LSE AG + output RS = 3
A2A:   query AG + packed output/LSE A2A = 2
```

The all-rank profile observed 6,384 instances of each A2A pack/collective/unpack
stage and **zero** LSE-all-gather or output-reduce-scatter ranges. Median host
spans were 61.507 us pack + 118.917 us A2A + 74.350 us unpack = **254.774 us**.
The prior direct-LSE AG+RS combine envelope was 356.678 us including LSE prep,
LSE AG, correction, and RS. This is a **28.57% component reduction**.

End-to-end MTP4 throughput is effectively neutral rather than faster: c1 is
+0.12% and c32 is -1.02% versus the matched direct-LSE AG+RS artifact. Both
stay inside the plan's 2% regression bound. No-MTP improved +1.10% c1 and
+2.52% c32. Because the named component improved, one collective was removed,
correctness passed, no-MTP improved, and MTP4 remained within the bound, A2A
is retained for the next iteration.

## Implementation

Flash-V100 reads `parallel_config.dcp_comm_backend` at construction and:

- uses packed A2A only for explicit `a2a`;
- preserves `ag_rs` as the default and exact existing behavior;
- rejects unknown backends instead of silently selecting another route;
- records backend-specific route counters;
- emits `a2a_pack`, `a2a_all_to_all`, and `a2a_unpack` NVTX ranges with payload
  bytes;
- uses A2A for DCP prefix-context combination as well, so an explicit A2A
  service never silently switches backends on prefix hits;
- preserves direct FP32 LSE, base-e weighting, empty-shard `-inf`, head scatter,
  output/LSE strides, and `return_lse` semantics.

The existing A2A buffer behavior was not graph-safe enough for this server.
The final adaptation uses per-device/per-stream/per-shape persistent buffers
for decode. Large prefix-prefill combines use request-scoped buffers outside
CUDA capture after the global workspace manager is locked. A captured prefix
route that would require unmanaged storage fails explicitly.

Changed source/test commits:

1. `fd35ee3ebc` — select/instrument packed A2A in Flash-V100;
2. `c871231ce0` / `b90526dda6` — add DCP2 distributed/config coverage;
3. `afd63b96f5` — bound large prefix-prefill buffer lifetime;
4. `b996cc4a39` — retain graph-stable A2A decode buffers.

No MTP3, DFlash, q>1 batching, approximate sampling, gateway, or shared-venv
change was made.

## Problems found during qualification

Two real workspace assumptions failed closed and were fixed before qualification:

1. The first 8K prefix request needed 58.34 MiB after the workspace manager was
   locked at 0.04 MiB. Prefix A2A now explicitly uses eager request-scoped
   buffers when locked and outside capture.
2. The first c32 MTP draft needed 0.24 MiB while the manager remained locked at
   0.04 MiB. Decode A2A now uses a dedicated persistent per-stream/shape cache,
   providing stable CUDA-graph addresses without attempting manager growth.

Both failures produced clear assertions; neither silently fell back to AG+RS.
The final candidate passed the same workloads that exposed them.

## Source projection and binary

The immutable runtime projection is:

- host: `/localpool/onecat-vllm-hy3-sm70/dcp2-branch-b996cc4a39`;
- container: `/workspace/dcp2-branch-b996cc4a39`;
- projection `SHA256SUMS` SHA256:
  `275acce27affb78cd772a7a846f612634d6fedcdd200657fe80966ca3bae5c64`.

The direct-LSE extension ABI did not change in this iteration, so the qualified
binary from source commit `53893bfb47` was reused:

- `flash_attn_v100_cuda...so` SHA256:
  `ad5a8d31cab8a6ebfecd13bd5b13d0ff2e5d0ddddf0a00685ae69f6f7cee42b0`.

The shared venv was not modified. Every startup checks the projection hashes.

## Tests and correctness

Local focused source/CPU DCP tests: **13 passed**.

V100 A2A tests: **32 passed, 4 skipped** (the skips require four visible GPUs;
the deployed DCP2 two-rank test ran). Coverage includes:

- fp16/bf16/fp32 pack/unpack;
- base-e and base-2 LSE;
- exact empty-shard zero output and `-inf` LSE;
- stable/non-aliasing workspace and persistent buffers;
- actual two-rank NCCL `all_to_all_single` with workspace.

Service gates:

| Gate | Result |
|---|---:|
| MTP4 CUDA graph capture | Pass |
| 8K needle | Pass |
| 32K needle | Pass |
| 128K needle | Pass |
| repeated 32K prefix | Pass / Pass |
| no-MTP 8K smoke | Pass |

Final active MTP4 startup reports 2,090,088 KV tokens and 4.13 GiB CUDA graph
capture.

## Matched performance

### Native MTP4

| Metric | A2A | Direct-LSE AG+RS | Change |
|---|---:|---:|---:|
| c1 median | **50.7535 tok/s** | 50.6951 | +0.12% |
| c32 median | **356.4301 tok/s** | 360.0948 | -1.02% |

A2A c1 runs: 50.2817, 52.2476, 50.7535 tok/s.

A2A c32 cohorts: 356.3245, 356.4301, 391.0371 tok/s.

Accepted drafts/verifier step: 1.3528 c1, 1.5018 c32.

### No MTP

| Metric | A2A | Direct-LSE AG+RS | Change |
|---|---:|---:|---:|
| c1 median | **55.8002 tok/s** | 55.1923 | +1.10% |
| c32 median | **441.3470 tok/s** | 430.5089 | +2.52% |

A2A c1 runs: 55.8644, 55.7915, 55.8002 tok/s.

A2A c32 cohorts: 441.3470, 443.9813, 435.8287 tok/s.

## Profile artifacts

- Nsight report:
  `/srv/dev/dcp2-direct-lse-profile-53893bfb47/a2a-b996cc4a39-all-ranks.nsys-rep`
  - SHA256 `7c816e474c690d383ebb81a880c7831dcba8c98b50b5ba4491ef62d214a3872d`
- SQLite:
  `/srv/dev/dcp2-direct-lse-profile-53893bfb47/a2a-b996cc4a39-all-ranks.sqlite`
  - SHA256 `de223333b5e1ba675f22907ab9666042b5170724b0791969f86c9812e3fd11cc`

As in prior runs, Nsight exported NVTX but no CUDA kernel tables. Timings are
host-range spans, not device-kernel timings. The unpack mean is contaminated by
host scheduling outliers; medians/p90 are reported for component comparison.

Raw result SHA256s:

- MTP4 benchmark:
  `d3ca0dfb08ccd25f3f96b162867816489d94378cb801a99f4d922f1e6710b815`;
- long correctness:
  `faf68035453271569acc7e1162b16d38322fad74970608aa041b0c13623c380d`;
- no-MTP benchmark:
  `d4524558293453248f7d38c391a9f05cea5824fca57e3a063e29054d844b1d0b`.

## Final live state

- `qwen38-27b-fp8-tp8-dcp2`: `1/1`, healthy;
- mode: TP8+DCP2+A2A+native MTP4;
- canonical runtime: `b996cc4a39`;
- old TP8: `0/0`;
- NVTX/Nsight wrapper: disabled;
- gateway: untouched and still targets the old TP8 service, as requested.

The next performance target remains MTP4 q>1 correction batching; A2A removes
a collective but does not close the aggregate throughput gap by itself.
