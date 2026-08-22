<!-- forge-controller-brief:v2:start -->

# Forge 001: Exact TP8 DCP2 Native MTP3 Kernel Campaign

## North star

Deliver the strongest exact TP8/DCP2/native-MTP3 verifier result from the remaining kernel and communication families. Preserve current output semantics, DCP behavior, recurrent state, and generic fallbacks.

## Inherited evidence and decisions

The retained implementation already removes row-parallel fallback copies and specializes two FP8 producer shapes. Earlier GDN conversion-only fusion, simple attention CTA tuning, FP8 tuner selection changes, NCCL setting changes, and two-slab overlap did not qualify. The active verifier still contains a fused GDN recurrent boundary, paged multi-query-head attention, TP reductions, DCP transfers, and FP8 gated-MLP input work.

The campaign must treat historical measurements as evidence. Reproduce every material claim on the active exact graph before it affects a decision.

## Delivery horizon

Complete the following portfolio before terminal judgment:

1. Fuse GDN causal convolution and recurrence without an intermediate graph allocation.
2. Build an exact multi-query-head paged-attention route that reuses KV data while preserving DCP LSE behavior.
3. Remove or overlap TP and DCP communication dependencies without configuration-only tuning.
4. Replace or improve FP8 gated-MLP input work only when exact baseline arithmetic remains intact.
5. Combine compatible improvements, including earlier sub-threshold improvements, when the integrated graph gains materially.

A failed route is evidence, not a stopping point. Revisit a rejected micro-optimization only when a combined candidate changes its allocation, synchronization, or arithmetic context.

## Acceptance

A retained result must pass all applicable gates:

- all-rank deterministic output and recurrent-state parity;
- changed-input CUDA-graph capture and replay;
- DCP LSE, exact retrieval, and capacity checks;
- matched c1 and c32 service cohorts with a material cumulative gain;
- graph-memory safety and unchanged generic, eager, irregular, non-DCP, and unsupported-shape fallbacks;
- focused static and kernel tests;
- final independent red-team attacks against numerical semantics, graph lifetime, DCP behavior, collective ordering, profiler attribution, and benchmark methodology.

The controller can retain compatible sub-threshold changes only when their integrated result clears the material performance gate. A microbenchmark alone never qualifies a result.

## Macro graph

Observe the complete q=4 verifier graph. Select the highest-value unresolved boundary. Implement and integrate one mechanism. Attack its arithmetic, graph, and distributed behavior. Reobserve the whole graph. Continue through the portfolio, then combine passing mechanisms and run final verification and red-team repair loops.

## Initial micro frontier

Map the GDN convolution-to-recurrence contract, buffer ownership, state update order, and graph allocations. Build one narrowly gated no-extra-buffer fusion. Qualify it before introducing attention or collective changes.

## Authority and constraints

The controller owns kernel design, code organization, worker fanout, and experiment order. It can reserve the authorized V100 test environment and use reversible local commits. It must not change model weights, tokenizer behavior, output semantics, MTP acceptance policy, or service policy. Do not use approximate arithmetic or configuration-only sweeps. Do not re-run rejected candidates without new discriminating evidence. Do not publish a product branch or alter unrelated source.

Use the existing repository-native environment and the verified V100 test boundary. Keep raw experiment artifacts outside Forge state. Use only declared product paths. Add a narrow architectural lint before writer fanout. Preserve unrelated work.

## Cadence

Use implement -> verify -> redteam as a single adaptive item. Start with the GDN fusion because it is the largest remaining kernel-only boundary. Continue through the other portfolio families and all required repairs. Terminal judgment requires a fixed point across the complete portfolio, not one local patch.

## Stop and resume

Continue while any safe in-scope implementation, test, profile, falsification, repair, or integration action remains. Stop only for explicit pause, credible harm, inaccessible required authority, or an external resource boundary. On interruption, return the earliest unverified action and exact recovery evidence in the same lane.


<!-- forge-controller-brief:v2:end -->

## Amendments and decisions

## Outcome

## Continuity
