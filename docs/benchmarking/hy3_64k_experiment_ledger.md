# Hy3 64K Experiment Ledger

Workload: `cyankiwi/Hy3-AWQ-INT4` pinned at
`c8b08e2c23dd45cb1b277d1290800e40c3dd8eec`, greedy MTP decoding, a
64,001-token repeated-`the` prompt, and 256 completion tokens.  The rate is
the streamed steady decode rate excluding time to first token.

| ID | Eight-V100 configuration | Result | Decision |
| --- | --- | --- | --- |
| B0 | TP8, 64K, Triton attention safe default (decode: 8 warps), MTP 2 | 1.919 tok/s single slot | Baseline / retain |
| B1 | TP8, 64K, Triton attention safe default (decode: 8 warps), MTP 2, two concurrent streams | Slot 1: 1.949 tok/s; slot 2: 1.905 tok/s; 3.809 aggregate tok/s (510 post-first-token completions over the slower slot's 133.878 s decode window) | Retain: 1.985x B0 aggregate scaling; each slot within 0.7% of B0 |
| E1 | TP4xPP2, 64K, MTP 2, Marlin AWQ-MoE | The real two-slot run passed the repaired PP sampled-token broadcast, but produced only 1.4-2.0 aggregate tok/s in engine metrics with low draft acceptance. It was stopped before scoring. | Reject |
| E2 | TP8, 64K, MTP 2, decode: 4 warps | 1.896 tok/s single slot (`256` completion tokens; `134.512` steady seconds; `0.218651` seconds to first token) | Reject: 1.2% below B0 |
| E3 | TP8, 64K, MTP 4, decode: 8 warps | 1.873 tok/s single slot (`256` completion tokens; `136.123` steady seconds; `0.225831` seconds to first token) | Reject: 2.4% below B0 |

The four-warp run was preceded by an identical 64K one-token warm-up. That
warm-up compiled the long-context unified-attention and fused-MoE decode
kernels; the scored request had a 50% prefix-cache hit rate and 100% MTP
acceptance. The result is therefore not attributable to cold kernel JIT or
speculative-token rejection.

E3 had a 50% draft acceptance rate: the first two MTP positions were accepted
and positions three and four were rejected. Reusing Hy3's single native MTP
layer beyond two proposals therefore did not amortize the verifier enough to
outperform B0.

At the full 65,536-token context limit, the deployed TP8 engine reports
158,016 usable KV-cache tokens, or 2.41 concurrent maximum-context requests.
Therefore batch 1 and batch 2 are the only valid simultaneous decode points
for this workload. A batch-4 run would queue or exceed the full-context KV
budget and is deliberately excluded from the throughput curve.

The B1 requests used a shared 64,001-token prefix and streamed 256 tokens each.
The first-token times were excluded; the aggregate score uses the two streams'
shared wall-clock decode interval, rather than summing their request durations.
It was repeated after correcting only a reporting-parentheses bug in the
benchmark job; both runs were consistent (first run: 3.677 aggregate tok/s;
clean run: 3.809 aggregate tok/s).

## Starting-fork and dispatch audit

`experiment/bonsai-q2-volta-mma` shares merge-base `66232f91` with the older
`hy3-sm70-marlin` fork. Its committed divergence in the relevant serving path
is only GGUF speculator detection in `vllm/engine/arg_utils.py`, which is not
selected by this non-GGUF Hy3 invocation. The additional local Hy3 changes are
pipeline-parallel MTP support, guarded by `pipeline_world_size > 1`, and an
opt-in exact-shape SM70 MTP-MoE Triton configuration. The live TP8 run has
pipeline parallelism 1, so the PP machinery is inactive; no TP8 attention or
batch dispatch regression was found relative to the starting fork. The live
engine kept two requests running concurrently throughout B1, and its coarse
10-second metrics reported 3.6-4.2 aggregate generation tok/s, corroborating
the harness result.

The 50 tok/s records under `reference/ds4` belong to a different DS4 engine
and model family. They are not comparable Hy3 measurements and must not be
used as a Hy3 baseline.
