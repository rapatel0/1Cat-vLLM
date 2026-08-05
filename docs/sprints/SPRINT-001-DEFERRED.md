# Sprint 001 Deferred Work

## Explicitly deferred

- Real Hy3 weight download, production deployment, and throughput tuning.
  Sprint 001 exits on functional loader/reference proof plus TP4xPP2 dummy
  dispatch, not a performance target.
- `tc-grid` and `kernels/v100/dense.cuh` integration. They remain available
  fallback options, not parallel work.
- Broad kernel rewrites, a new kernel family, or generalized asymmetric MoE
  support beyond the Hy3 W4A16/group-32/no-act-order contract.
- Any changes to `rapatel0/1Cat-vLLM` legacy `main`,
  `gemma4-12b-ct-awq-v100`, or `int2-v100-gemv`.

## Revisit triggers

1. The tested loader bridge reaches the SM70 op with correct zero-point layout
   and the V100 reference test exposes a Marlin-specific functional defect
   that a bounded binding/C++ fix cannot resolve.
2. Sprint 001 succeeds and a later production benchmark establishes a
   performance shortfall against a correct alternative on the same placement.
3. The actual runtime topology cannot supply eight V100s as two valid TP4
   cliques or lacks the needed PP2 transport; this is an environment decision,
   not evidence to replace Marlin.
