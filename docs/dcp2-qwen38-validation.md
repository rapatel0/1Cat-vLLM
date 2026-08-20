# Qwen3.8 TP8 + DCP2 validation

Date: 2026-08-20
Branch: `experiment/qwen38-mtp-dcp2`
Implementation tip tested: `18dd13ac6ec916df355b23210ec62ae8e064fea7`
Hardware: gpu-01, 8 x V100-SXM2-32GB

## Source commits

- `6a182f4496` — persist hybrid DCP cache support
- `43e1d74ab1` — implement DCP prefill LSE merge
- `bf188f87c5` — return and reduce decode LSE for DCP
- `cdab77f1ac` — scale hybrid prefix hits for DCP
- `80e9a9c4d1` — make rank-specific DCP lengths CUDA-graph safe
- `98f0f821d8` — expose the dense Flash-V100 LSE already computed by the kernel
- `8bf34a1bf8` — fail closed on invalid DCP prefix block tables
- `14ffa39680` — return paged-prefill LSE and use paged KV directly for DCP prefix attention
- `18dd13ac6e` — make the DCP prefill route CUDA-graph safe for MTP

All commits were pushed to `rapatel0/1Cat-vLLM` on the branch above.

## Focused validation

- CPU numerical DCP prefill merge: no prefix, uneven 5-token prefix, and a
  1-token prefix for which one rank owns no context — pass.
- CPU numerical DCP decode merge: uneven and empty-local-shard cases — pass.
- Decode partition-workspace LSE reconstruction — pass.
- Actual V100 scalar paged decode against dense PyTorch reference:
  output max abs error `0.000244140625`, LSE max abs error
  `4.76837158203125e-07` — pass.
- Actual V100 dense prefill LSE against dense PyTorch reference:
  output max abs error `0.0009765625`, LSE max abs error
  `9.5367431640625e-07` — pass.
- Actual V100 paged prefill LSE (q=83, kv=1000, D=256) against dense
  reference: output max abs error `0.0001220703125`, LSE max abs error
  `9.5367431640625e-07` — pass.
- Rank-specific local sequence-length calculation captured and replayed in a
  CUDA graph on V100 — pass.
- Hybrid DCP prefix-cache unit test — pass. Related non-DCP hybrid matrix:
  17 tests passed.

## End-to-end canary

Canary deployment: `llm/qwen38-27b-fp8-tp8-dcp2`. The canary loaded a
commit-labelled overlay generated from this branch; production's shared venv
was not modified.

### DCP2 without MTP (`gpu-memory-utilization=0.85`)

- KV cache: `2,612,021` tokens; maximum 262K concurrency `9.96x`.
- Exact-response smoke matched the TP8 production response byte-for-byte after
  normalizing the response envelope (`DCP2_BASELINE_OK`).
- Arithmetic: `137 * 29` returned `3973`.
- Needle retrieval with thinking disabled:
  - 7,647 prompt tokens: exact code returned, 1.85 s.
  - 31,648 prompt tokens: exact code returned, 8.29 s.
  - 127,649 prompt tokens: exact code returned, 62.26 s.
- 31,683-token prefix-cache request repeated twice: both returned the exact
  code; elapsed 10.64 s then 5.92 s.
- Short greedy requests measured about 45.5–50.4 completion tok/s including
  request latency. This is not a steady-state decode-only benchmark.

A correctness defect was found during this gate: dense gathering of a large
cached prefix could issue an illegal access at the final chunk. The final
implementation uses the paged-prefill kernel directly and returns its existing
LSE; the 32K repeated-prefix and 128K tests above passed after that fix.

### DCP2 + MTP4

MTP4 required additional runtime headroom for graph capture and prefill
workspaces. At `gpu-memory-utilization=0.83`, startup passed but an 8K request
could OOM. At `0.78`:

- KV cache: `2,097,152` tokens; maximum 262K concurrency `8.00x`.
- Startup and MTP CUDA-graph capture passed (`4.88 GiB` graph footprint).
- Exact smoke returned `DCP2_MTP4_OK`.
- 7,641-token needle retrieval returned `MTP8_DCP2_6A2F` exactly.
- 31,683-token prefix request repeated twice returned
  `PREFIX32_DCP2_4C91` exactly both times (10.10 s and 9.79 s).

MTP4 at 128K and aggregate-concurrency throughput were not rerun before the
maintenance window ended. Consequently the canary was not promoted.

## Final live state

- `llm/qwen38-27b-fp8-tp8`: `1/1` ready (TP8 + MTP4 production baseline).
- `llm/qwen38-27b-fp8-tp8-dcp2`: scaled to `0/0`.
- Gateway remained unchanged and points to
  `http://qwen38-27b-fp8-tp8.llm.svc.cluster.local/v1`.
