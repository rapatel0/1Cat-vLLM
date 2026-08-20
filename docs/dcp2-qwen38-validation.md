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

### Final MTP4 promotion gate (2026-08-20)

The remaining gate was run from the direct canary service with the
commit-labelled `18dd13ac6e` implementation overlay. The deployment retained
TP8, DCP2, MTP4, `max-num-seqs=32`, `max-num-batched-tokens=16384`, and
`gpu-memory-utilization=0.78`. This start reported 2,090,088 KV-cache tokens
(`7.97x` 262K concurrency) and the same 4.88 GiB CUDA-graph footprint.

The durable machine-readable summary is
[`dcp2-qwen38-final-benchmark.json`](dcp2-qwen38-final-benchmark.json). The
full temporary result was
`/tmp/dcp2-mtp4-final-artifacts/dcp2-mtp4-final-v2.json`, SHA256
`344250da2b3ca3483a0de6ff1274b8e4b7f1e32ba6663b5fa54e8789aa08a831`.

#### Correctness

The exact needle gate passed at 127,496 server-reported prompt tokens. The
secret `DCP2_MTP4_128K_7C653486A1` was returned exactly in 66.5687 seconds
(23 completion tokens, stop finish reason). This closes the remaining MTP4
128K correctness item.

#### Cache-safe throughput methodology

All measured requests went directly to vLLM, not through the LiteLLM/Redis
router. They were non-streaming, counted from server
`usage.completion_tokens`, used an early unique nonce in every prompt, and
therefore could not be response-cache hits or long shared-prefix hits. Greedy
sampling (`temperature=0`, `top_p=1`, `ignore_eos=true`) made every measured
request return its full token budget. A separate warmup preceded both shapes.

Single stream used approximately 1,024 prompt tokens and 512 completion tokens:

| Run | Completion tokens | API wall (s) | tok/s |
|---:|---:|---:|---:|
| 1 | 512 | 9.9487 | 51.4641 |
| 2 | 512 | 9.2901 | 55.1125 |
| 3 | 512 | 9.9513 | 51.4507 |
| **median** | — | — | **51.4641** |

The measured single-stream MTP draft acceptance was 36.03% (908 accepted of
2,520 drafted tokens; 1.441 accepted tokens per verifier step).

Aggregate throughput used 32 concurrent requests, each with approximately 256
prompt tokens and exactly 256 completion tokens. A separate c32 warmup produced
2,048 tokens in 7.0769 seconds and was excluded:

| Cohort | Requests | Completion tokens | Cohort wall (s) | Aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 32 | 8,192 | 22.1466 | 369.8982 |
| 2 | 32 | 8,192 | 21.1593 | 387.1580 |
| 3 | 32 | 8,192 | 21.7292 | 377.0046 |
| **median** | — | — | — | **377.0046** |

Measured c32 draft acceptance was 37.39% (14,726 accepted of 39,388 drafted
tokens; 1.495 accepted tokens per verifier step).

#### Baseline comparison and decision

The historical TP8+MTP4 anchors are 126 tok/s single and 1,730 tok/s at c32.
Their original prompt lengths, output lengths, exact warmup policy, and whether
prefill was excluded were not retained, so this is not a perfectly matched
A/B. The final gate includes API wall time and uses explicitly unique prompts;
those choices make it stricter than a decode-only or repeated-prefix test.
However, the gaps are too large to attribute to that methodological difference:
51.4641 is 40.84% of the single anchor (-59.16%), and 377.0046 is 21.79% of the
c32 anchor (-78.21%). The DCP cross-rank query/LSE output correction remains a
large throughput tax, and low MTP acceptance on this workload compounds it.

**Promotion gate: failed on performance.** Correctness and memory capacity
passed, but traffic was not moved to DCP2. The canary was scaled back to zero
and the TP8+MTP4 baseline was restored.

Exact benchmark command inside the canary pod:

```bash
/opt/venv/bin/python /tmp/benchmark_qwen38_dcp2_service.py \
  --base-url http://127.0.0.1:8000 \
  --output /tmp/dcp2-mtp4-final-v2.json \
  --needle-target-tokens 127500 \
  --single-prompt-tokens 1024 --single-output-tokens 512 --single-runs 3 \
  --aggregate-concurrency 32 --aggregate-prompt-tokens 256 \
  --aggregate-output-tokens 256 --aggregate-runs 3
```

## Final live state

- `llm/qwen38-27b-fp8-tp8`: `1/1` ready (TP8 + MTP4 production baseline).
- `llm/qwen38-27b-fp8-tp8-dcp2`: scaled to `0/0`.
- Gateway remained unchanged and points to
  `http://qwen38-27b-fp8-tp8.llm.svc.cluster.local/v1`.
