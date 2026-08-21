# TP8+DCP2 Performance Plan

## Decision

Keep TP8+DCP2 as the architecture. Optimize the correctness-qualified DCP
path for native MTP3 and no-speculation operation. TP8+DCP1 is a historical
control, not a mandatory live rollback target on experimental gpu-01.

Native **MTP3 is the active speculative configuration** for this campaign.
A matched live gate found that q=4 cost fell more than the useful-token rate.
MTP3 increased seven-run c32 from 734.934 to 746.271 tok/s versus MTP4.

`incoai/Qwen3.8-27B-DFlash2` is a separate, gated alternative.  Its published
results use H200 + FlashAttention 3 and do not establish V100 performance.  It
must not delay DCP work or be canaried until its DFlash2/V2-runner/SM70 path is
explicitly qualified.

## Existing evidence — do not rerun as a Phase-0 prerequisite

| Mode | Existing throughput evidence | KV tokens | Provenance |
|---|---:|---:|---|
| TP8+DCP1, no speculation | about 120 tok/s single; 1,700 tok/s c32 | 1,209,336 | operator-confirmed reference; the final artifact previously mislabeled it as MTP4 |
| TP8+DCP1, no speculation | 70.9 tok/s at ~13K context | — | different long-context shape in `homelab/docs/qwen38-27b-bringup.md` |
| TP8+DCP2, no speculation | 45.5–50.4 short-request tok/s | 2,612,021 | `docs/dcp2-qwen38-validation.md` |
| TP8+DCP2+MTP4, original | 51.464 tok/s single; 377.005 tok/s c32 | 2,090,088 | `docs/dcp2-qwen38-final-benchmark.json` |
| TP8+DCP2+MTP4, final | 54.251 tok/s single; 734.934 tok/s c32 | 2,097,152 | seven-run fused-ZBA qualification |
| TP8+DCP2+MTP3, historical | 54.052 tok/s single; 746.768 tok/s c32 | 2,123,901 | fused-ZBA harness with 249-256-token prompts; not comparable to later metadata campaigns |
| TP8+DCP2+MTP3, current fixed-corpus | 724.816 tok/s c32 | 2,123,901 | fresh six-start reconciliation; active source `59a5fa11f0` |

The early 55.4/539 DCP2 figures in the implementation report predate the final
cross-rank LSE correction and are deliberately excluded. The old 126/1,730
artifact values were operator-supplied and mislabeled as TP8+MTP4; the corrected
reference is about 120/1,700 for TP8 without MTP. The retained benchmarks use
different prompt and warmup shapes, so they are capacity/performance anchors
rather than a promotion comparison by themselves.
They are sufficient to start code work. No baseline-recreation matrix is a
Phase-0 prerequisite.
New canary runs are required only to validate a changed candidate, not to
re-create every baseline variant.

For speculation, compare **accepted draft tokens per verifier step** and
**completion tokens per verifier step**, not `accepted / drafted`.  A 50% MTP3
rate means 1.5 accepted draft tokens/step; the measured MTP4 c32 result is
1.495 accepted draft tokens/step.  The MTP4 fourth proposal currently has low
marginal acceptance, but that is not evidence that the metric collection is
wrong.

## Scope

### In scope

- TP8+DCP2 Flash-V100 decode and q>1 verifier performance.
- No-MTP and native-MTP3 regression/correctness validation after each accepted
  optimization.
- CUDA-graph-safe allocation, copy, LSE, and collective improvements.
- DCP profiler instrumentation sufficient to attribute time and collective
  counts on all ranks.
- DFlash2 source/SM70 compatibility gate and a separate branch only if it
  passes.

### Out of scope for this campaign

- TP4×CP2 external context parallelism.
- Further MTP-depth sweeps beyond the qualified MTP3/MTP4 comparison.
- Approximate acceptance policies or any target-distribution change.
- Gateway changes or traffic retargeting without explicit approval.

## Execution sequence

### 1. Instrument without changing behavior

Add graph-safe, opt-in DCP profiling/counters to
`vllm/v1/attention/backends/flash_attn_v100.py` and the DCP combine helpers.
The profiler must expose, per rank and per steady decode/verifier step:

1. query all-gather;
2. local paged/prefix attention;
3. LSE reconstruction;
4. LSE all-gather;
5. output-correction kernel;
6. output reduce-scatter / selected communication path;
7. packing, `contiguous`, allocation, and copy work;
8. target graph replay and MTP-draft timing.

Use CUDA events/NVTX without per-step synchronization. Set
`VLLM_FLASH_V100_DCP_DECODE_NVTX=1` only for profiler runs; the emitted
`vllm.flash_v100.dcp_decode/*` ranges include collective byte counts and are
disabled by default. Profile a bounded post-warmup S1 and c32 MTP3 run, plus
no-MTP only when diagnosing a regression.
Observed collective counts and bytes must match the persisted model topology:
16 full-attention layers, DCP world size 2, and MTP verifier query length.

**Exit:** at least 90% of steady-step device time is attributed; all-rank skew
and collective counts are recorded; instrumentation is disabled by default.

### 2. Remove universal DCP overhead

Implement and validate one focused change per commit, in this order:

1. Reuse graph-stable query-gather, LSE, correction, output, and collective
   work buffers. Remove avoidable `contiguous`, temporary allocation, and
   Python-side workspace reconstruction in the hot decode path.
   **Query gather completed:** runtime source `e276623a92` uses persistent
   rank-major and head-major buffers with direct PyNCCL output. It preserves an
   explicit coordinator fallback. The median annotated query-gather component
   fell 37.81%, and c32 increased 7.01% to 637.5508 tok/s. See
   `docs/tp8-dcp2-query-gather-workspace.md`.
2. Return/write final local LSE directly from the decode reduction workspace,
   avoiding redundant tensor materialization or reduction kernels.
3. Honor and benchmark the configured DCP communication backend at the real
   V100 DCP2 shapes; compare AG+RS and packed A2A only where both are exact.
   **Completed:** packed A2A is correctness/graph-qualified at `b996cc4a39`,
   removes one collective and 28.57% of the median annotated combine envelope,
   improves no-MTP, and keeps MTP4 within the 2% regression bound. See
   `docs/tp8-dcp2-a2a-live-qualification.md`.
4. Restore an eligible small-query/XQA route under DCP only if exact LSE,
   uneven local lengths, prefix cache, and graph replay are preserved.
   **Tested and rejected:** source `deb0fa7da1` selected XQA only for exact DCP
   q=1 shapes and passed all correctness/graph gates. However, MTP4 c1/c32
   regressed 6.17%/6.50%, no-MTP c32 regressed 12.02%, and the annotated local
   attention span did not improve. The qualified scalar route remains active.
   See `docs/tp8-dcp2-xqa-decode.md`.
5. Overlap query all-gather with independent replicated causal-suffix work,
   using graph-safe streams/events, only if the trace shows useful overlap.
6. **Completed:** native MTP3 and MTP4 on SM70 now use the exact fused Qwen3.5
   GDN z/b/a projection-slice copy by default. MTP3 component, graph, state,
   retrieval, no-MTP, and performance gates passed at source `59a5fa11f0`.
   The MTP3 seven-run c32 result was 746.768 tok/s with 2,123,901 KV tokens.
   See `docs/tp8-dcp2-mtp3-fused-zba.md`.
7. **Audited, no candidate retained:** TP8 q=5 all-reduce uses PyNCCL because
   the physical eight-GPU group lacks direct all-to-all NVLink. The custom
   all-reduce route rejects this topology. NCCL already selects twelve
   topology-aware ring channels. A graph microbenchmark favored Ring/LL with
   24 channels, but the global setting reduced live c32 throughput to 614.622
   tok/s. The qualified automatic PyNCCL policy remains active. See
   `docs/tp8-dcp2-mtp4-tp-collective-campaign.md`.
8. **Corrected and rejected:** the q=4 producer-overlap prototype kept all
   graph buffers alive and used the actual capture stream. Five routes passed
   100 changed-input replays on all eight ranks. Real overlap still regressed
   6.31% because it doubled GEMMs and reductions.
9. **Attributed, no candidate retained:** q=4 attention metadata measured
   4.194 ms at c1 and 6.579 ms at c32. Three GDN groups dominate the c32 cost.
   Removing one GPU scalar assertion did not improve the full step. See
   `docs/tp8-dcp2-mtp3-attention-metadata.md`.
10. **Reconciled and rejected:** homogeneous c32 q=4 GDN groups shared
    persistent invariant metadata while each group retained its state contract.
    The original 4.88% result used source-specific prompt strings. A fresh
    fixed-corpus sequence measured -8.39% cohort median and -3.99% pooled
    throughput. Source `59a5fa11f0` is active. See
    `docs/tp8-dcp2-mtp3-common-gdn-reconciliation.md`.

Each change first passes targeted unit/kernel tests and no-MTP/MTP3 graph and
numerical tests, then a direct canary S1/c32 candidate gate.  Keep a change
only when its named component improves by >=10% and no tested workload loses
more than 2%.

### 3. Fix q>1 MTP scaling

**Completed:** runtime source `9bc01fd4a3` batches compatible uniform small
queries across causal suffix attention, local paged-prefix attention, exact LSE
correction, and state merge. The per-request route remains explicit for
irregular layouts. See `docs/tp8-dcp2-mtp4-batched-correction.md`.

The qualified route reduced matched q=5 correction calls from 2,438 to 157 per
rank, increased c32 from 356.4301 to 595.8024 tok/s, and reduced graph memory
from 4.13 to 2.04 GiB. Exact 8K/32K/128K retrieval, repeated-prefix use,
no-MTP smoke, and a two-rank c32-shape CUDA graph test passed.

**Exit met:** MTP4 c32 verifier correction no longer scales once per request
for uniform q>1 batches. Exact outputs, fallback routing, graph replay, prefix
cache use, and hybrid GDN-state service behavior remain correct.

### 4. Qualification after accepted code changes

For every candidate image, run in order:

1. startup/topology/health and 8K smoke;
2. no-MTP and MTP3 graph/eager parity, DCP numerical tests, q=1 and q=4;
3. DCP2 repeated-prefix and exact 32K/128K needle retrieval;
4. direct-service MTP3 S1 and c32 performance; no-MTP direct service only as
   a regression diagnostic or after a no-MTP-specific change;
5. capacity inventory and a 30-minute MTP3 soak.

Capacity floors: no-MTP >=2.48M KV tokens; MTP3 >=2.0M KV tokens; graph
footprint <=5.12 GiB.  The final goal is >=85% of the appropriate same-commit
DCP1 control.  If the published anchors are reproduced under the same harness,
use 107 tok/s single and 1,470 tok/s c32 as absolute promotion floors.

gpu-01 is an experimentation node. Keep the current TP8+DCP2 candidate active
between successful iterations so the next focused fix can use it directly;
leave the old TP8 control scaled to zero unless an experiment specifically
needs it. The gateway remains unchanged unless the operator explicitly asks to
retarget it. A technically broken candidate must be stopped, but there is no
mandatory end-of-run restoration to TP8+DCP1.

## DFlash2 decision gate

DFlash2 is currently **no-go for a V100 canary**.  A separate branch may be
opened only after all of the following are demonstrated:

- a pinned, reviewed port of vLLM PR #52816 registers `DFlash2DraftModel` and
  fails closed instead of routing the checkpoint through DFlash1/V1;
- the FP16 draft, dynamic convolution, candidate selector/top-k path, and
  Triton walk compile and execute on SM70;
- the target LM-head quantization meets selector requirements;
- non-causal draft attention is exact on Flash-V100 or a separately-qualified
  SM70 backend;
- TP8+DCP1 greedy/eager/graph parity, rejection-heavy cases, and a soak pass.

Only then compare DFlash2 to MTP3 using the same target commit and direct
service workload. Promote DFlash2 only if DCP2 beats the best qualified MTP3
mode by >=10% at c1/c8, does not regress c32 by >5%, and retains >=1.8M target
KV tokens.

## First implementation task

Start with opt-in DCP timing/counter instrumentation and graph-safe workspace
reuse in the Flash-V100 DCP decode path.  This gives an attributable baseline
without delaying code work, and it is shared by no-MTP and MTP3.
