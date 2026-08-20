# TP8+DCP2 Phase-1 live profile

Date: 2026-08-20

Canonical source: `a6c7d817816cf6fc9289a1d00c117e19afbcb421`

Hardware: `gpu-01`, 8 x Tesla V100-SXM2-32GB

## Result

The first workspace-reuse/NVTX slice passed live TP8+DCP2+MTP4 startup,
CUDA-graph capture, 8K/32K/128K exact retrieval, and repeated-prefix
correctness. It retained 2,097,152 KV tokens.

The all-rank profile succeeded for NVTX and proves that every worker executed
the instrumented DCP path. Nsight Systems did **not** record CUDA APIs or GPU
kernels, even in a standalone Torch CUDA self-test. The timings below are
therefore host range spans, not device kernel times. This run does not satisfy
the performance plan's 90% device-time-attribution exit gate and makes no
workspace-reuse speedup claim.

The useful directional finding is that workspace acquisition is only 3.11% of
the annotated DCP envelope, while the three collectives account for 50.28% and
Python/Torch LSE reconstruction accounts for 32.37%. The next implementation
should remove/materially reduce LSE reconstruction before broadening workspace
caching.

Machine-readable summary:
[`tp8-dcp2-phase1-live-profile.json`](tp8-dcp2-phase1-live-profile.json).

## Runtime and source proof

The temporary canary used the final qualified shape:

- TP8, DCP2, native MTP4;
- `FLASH_ATTN_V100` for target and drafter;
- 262,144 maximum model length;
- `max-num-seqs=32`, `max-num-batched-tokens=16384`;
- `gpu-memory-utilization=0.78`;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `dcp_comm_backend=ag_rs`;
- `VLLM_FLASH_V100_DCP_DECODE_NVTX=1` only for this profiler run.

An immutable, read-only projection was created at
`/localpool/onecat-vllm-hy3-sm70/dcp2-branch-a6c7d81781`. It copied the last
qualified `18dd13ac6e` runtime overlay, then replaced the only three runtime
Python files changed through `a6c7d81781` directly from Git objects. Commits
between those points otherwise changed only tests, benchmarks, and docs.
`GIT_COMMIT` contained the full canonical SHA. Startup printed:

```text
[dcp2-profile] commit=a6c7d817816cf6fc9289a1d00c117e19afbcb421
8b9a1c34...  vllm/v1/attention/backends/flash_attn_v100.py
0607a65b...  vllm/v1/attention/ops/common.py
95d399a0...  flash_attn_v100/flash_attn_interface.py
```

The server process environment had the overlay first on `PYTHONPATH`; all
worker PIDs carried the canonical commit and NVTX environment values. Startup
also logged NCCL 2.27.5, PYNCCL for both `tp:0` and `dcp:0`, TP8/DCP2,
`Qwen3_5MTP`, MTP depth four, and Flash-V100 on all eight ranks.

## Correctness and capacity

| Gate | Prompt tokens | Result | Wall |
|---|---:|---|---:|
| 8K exact needle | 7,983 | Pass | 13.1864 s |
| 32K exact needle | 31,986 | Pass | 9.2350 s |
| 128K exact needle | 127,496 | Pass | 66.5068 s |
| repeated 32K prefix, run 1 | 31,983 | Pass | 9.0093 s |
| repeated 32K prefix, run 2 | 31,983 | Pass | 8.9694 s |

Startup/memory inventory:

- KV cache: **2,097,152 tokens**;
- available KV-cache memory: 18.02 GiB;
- CUDA graph footprint: **5.10 GiB**;
- post-c32 GPU memory: 31,880-31,958 MiB used per card.

The 5.10 GiB graph footprint passes the 5.12 GiB plan ceiling narrowly, but is
0.22 GiB above the prior 4.88 GiB qualified run. The run does not isolate
whether the difference comes from retained workspaces, profiler launch state,
or ordinary capture variance. Treat this as an important memory watch item.

## Small direct-service sanity benchmark

This was one candidate run/cohort, not a baseline matrix:

| Shape | Result | Accepted draft tokens / verifier step |
|---|---:|---:|
| c1, ~1K prompt / 512 output | 49.4826 tok/s | 1.4218 |
| c32, ~256 prompt / 256 output each | 317.2760 aggregate tok/s | 1.5398 |

These are below the earlier `18dd13ac6e` three-run medians (51.4641 and
377.0046), but the comparison is not matched: this run had NVTX enabled, an
idle Nsight wrapper, and one measurement. It neither proves a regression nor a
speedup. No performance claim is made for workspace reuse.

## All-rank NVTX profile

Profiler command wrapped the server from launch, delayed collection until the
workloads were warm, and left the target alive after collection:

```bash
nsys profile \
  --session-new=dcp2a6profile --start-later=true --kill=none \
  --sample=none --cpuctxsw=none \
  --trace=cuda,nvtx,nccl --nccl-trace=all \
  --cuda-graph-trace=node --trace-fork-before-exec=true \
  --output=/profiles/dcp2-a6c7d81781-all-ranks \
  python -m vllm.entrypoints.openai.api_server ...

nsys start --session=dcp2a6profile
# warmed c1: 384 output; warmed c32: 32 x 128 output
nsys stop --session=dcp2a6profile
```

All eight worker PIDs were present. Each emitted exactly 714 instances of each
DCP stage (7,140 custom ranges per rank), so query AG, LSE AG, and output RS
counts were 1:1:1 on every rank.

| DCP stage | Mean host span/call | Share of summed annotated spans |
|---|---:|---:|
| LSE reconstruction | **306.227 us** | **32.37%** |
| query all-gather | 174.583 us | 18.45% |
| LSE all-gather | 169.379 us | 17.90% |
| output reduce-scatter | 131.803 us | 13.93% |
| output correction | 79.147 us | 8.37% |
| output workspace acquire | 29.420 us | 3.11% |
| output copy | 25.338 us | 2.68% |
| local attention dispatch span | 22.867 us | 2.42% |
| query prepare | 4.420 us | 0.47% |
| LSE prepare | 2.832 us | 0.30% |

The three collective ranges total 475.765 us/call (50.28%). The summed
annotated host envelope is 946.016 us/call. These spans can overlap and include
dispatch/enqueue behavior; they must not be reported as additive GPU latency.

TP1/DCP1 (PID 649) was the slowest rank across most stages. LSE reconstruction
ranged from 285.485 to 403.219 us (1.412x max/min); query AG ranged from
164.867 to 206.281 us (1.251x). This rank skew is worth checking after device
tracing is repaired.

### Nsight limitation

The 89 MiB report and its exported SQLite contain all-rank NVTX/NCCL ranges but
no CUDA API or CUDA kernel tables. A separate `torch.arange` + add + synchronize
profile produced the same `does not contain CUDA kernel data` result. This
localizes the limitation to CUDA tracing in this Nsight/container environment,
not the vLLM graph path.

Persistent artifacts on `gpu-01`:

| Artifact | SHA256 |
|---|---|
| `/srv/dev/dcp2-phase1-profile-a6c7d81781/dcp2-a6c7d81781-all-ranks.nsys-rep` | `8bccdeb0fb297f33bc8e039cf6ed7a8fb40acda0e04b01414dadb59c21fb171b` |
| `/srv/dev/dcp2-phase1-profile-a6c7d81781/dcp2-a6c7d81781-all-ranks.sqlite` | `2352ca1ea84a26cab6f4e19be5512f69d475a366746750561a024f03c102f358` |
| `/srv/dev/dcp2-phase1-profile-a6c7d81781/nsys-cuda-selftest.nsys-rep` | `8aca7efcec5b72af6eb0c47b5f852967563a3e3cefa3b433c4e72401380a89a9` |

## Findings and next action

1. **Important — LSE reconstruction is the largest non-collective host span.**
   Implement the planned direct final-LSE write/reuse before spending more
   effort on output-workspace caching.
2. **Important — collectives remain half of the annotated DCP envelope.** Query
   AG + LSE AG + output RS should remain the primary communication target.
3. **Important — Nsight CUDA tracing is blocked.** Repair/replace the profiler
   environment before claiming device attribution or the plan's 90% exit gate.
4. **Watch — graph memory is close to its ceiling.** Inventory retained
   workspace shapes and avoid expanding the cache until the 5.10 GiB result is
   explained.
5. **No source blocker/correctness regression.** The slice is safe enough to
   retain for the next source optimization, but has no demonstrated throughput
   win.

## Final live state

- `llm/qwen38-27b-fp8-tp8`: `1/1`, direct health passed;
- `llm/qwen38-27b-fp8-tp8-dcp2`: `0/0`, dormant deployment spec restored;
- gateway config SHA before/after:
  `55af22359c276978d00015be45310e09156c8c0195a9a798ef4ffac209a74cb6`;
- gateway target unchanged:
  `http://qwen38-27b-fp8-tp8.llm.svc.cluster.local/v1`.
