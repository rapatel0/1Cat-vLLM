# Sprint 002: Q2 Volta Tensor Core Kernel Campaign

Status: validation blocked
Track: SM70 performance
Branch: `experiment/bonsai-q2-volta-mma`
Effort: high

## Outcome

Fuse Prism Q2_0 weight expansion into SM70 Tensor Core GEMMs for wide prefill
and four-/eight-slot decode. Keep the packed GGUF weights resident, avoid a
global FP16 weight materialization, and preserve the DP4A route for batch-one
decode and unsupported target shapes.

The sprint exits only after numerical, instruction-level, integrated
performance, and serving-quality gates pass. A single-GPU canary is the final
deployment action. The eight-replica Bonsai service, Q4 KV layout, attention
kernels, GGUF format, and production manifest remain unchanged.

## Implementation

1. Add a shared Q2_0 packed-byte expansion and per-128-value scale primitive
   that creates FP16 SM70 MMA fragments in registers.
2. Add a fused wide-token prefill GEMM for token counts of at least 64.
3. Add a separate fused decode GEMM for batch 4 and batch 8.
4. Dispatch the new kernels only for FP16 on exact SM70. Retain DP4A for other
   SM70 shapes and retain the existing dequantization fallback elsewhere.
5. Build only in the gpu-01 localpool-backed workspace and canary on one V100
   only after every gate below passes.

## Acceptance Gates

- [x] Q2 unpack matches an independent scalar reference for all 256 packed
  byte patterns and FP16 scale edge cases.
- [x] Candidate error is no worse than the existing DP4A path against an FP32
  reference.
- [x] Runtime traces confirm both dispatched kernels and final-binary SASS
  confirms HMMA/Tensor Core instructions.
- [ ] Integrated 4K/32K prefill and four-slot decode improve by at least 20%
  at the median. The 4K prefill gate passes, but four-slot decode improves by
  only 2.75%; the 32K baseline median was not run after the terminal gates
  failed.
- [x] Single-slot decode regresses by no more than 3%.
- [ ] Deterministic greedy known-answer, long-context needle, and four-slot
  concurrency checks remain valid. Transport-level four-slot concurrency
  passes, but the native vLLM Bonsai loader fails known-answer and needle
  quality on both DP4A and MMA configurations.
- [ ] A single-GPU canary passes without changing the reference eight-replica
  deployment. The canary served requests, but the quality and integrated
  four-slot performance gates block promotion.

## Decision Rules

- Unsupported shapes fail back to DP4A on SM70; do not broaden the MMA route
  without numerical and performance evidence.
- Any failed numerical or serving-quality gate blocks the canary regardless of
  microbenchmark speed.
- Missing profiler, GPU-allocation, or integrated benchmark evidence is a
  measured blocker, not grounds for promotion.
- Do not commit, push, open a PR, or promote the full service automatically.

## Validation Evidence

- A clean editable CUDA build for `sm_70` completed in the verified
  localpool-backed workspace. The final optimized GGUF translation unit and
  stable libtorch extension also rebuilt successfully in the persistent
  focused build directory.
- Dispatch tests: `10 passed`. Exact-SM70 FP16 shapes 4, 8, 64, and 65 select
  MMA; batch 1 and unsupported SM70 shapes select DP4A; non-SM70 retains the
  dequantization fallback. Integrated A/B used a temporary canary-only DP4A
  control and does not add a production environment switch.
- V100 numeric tests: `6 passed`. The exhaustive case covers all 256 packed
  byte values and FP16 scale edges. Candidate max and mean-square errors are
  no worse than DP4A for token counts 4, 8, 64, and 65.
- Runtime profiling captured both final decode and prefill dispatches.
  `cuobjdump` confirms `HMMA.884.F32.F32` instructions in the final binary.
  Nsight Compute identified low short-prefill occupancy and L1TEX scoreboard
  stalls; DCGM was paused only for the bounded profile and restored afterward.
- The locally converged variant uses split-K4 and four accumulator groups for
  decode, caches each Q2 block scale, vector-loads eight FP16 activations per
  row, and adaptively selects a 16-row prefill tile below 128 tokens or a
  64-row tile otherwise. Sweeps rejected split-K2/8, accumulator-group 2/8,
  prefill tiles 8/32/128, prefill warp counts 2/8, packed-value lookup, and
  half2 expansion because each regressed at least one serving shape.
- Final kernel medians on actual Bonsai matrix shapes measured the following
  MMA speedups over DP4A for batch 4 / batch 8 / tokens 64:
    - 10,240 x 5,120: 2.50x / 3.93x / 6.06x
    - 17,408 x 5,120: 2.47x / 3.77x / 6.70x
    - 5,120 x 17,408: 2.16x / 3.90x / 3.50x
    - 6,144 x 5,120: 2.10x / 3.28x / 3.43x
    - 5,120 x 6,144: 2.16x / 3.70x / 3.52x
    - 248,320 x 5,120 LM head: 2.67x / 4.33x / 4.18x
- For 10,240 x 5,120 long-prefill kernels, 4K measured 10.189 ms versus
  145.697 ms (14.30x), and 32K measured 86.544 ms versus 1,172.329 ms
  (13.55x).
- On the single-GPU native-vLLM canary, uncached 4K prefill improved from a
  71.894-second median to 8.384 seconds (8.58x). One 32K candidate request
  completed in 177.965 seconds; a DP4A median was not run after the terminal
  quality and decode gates had already failed.
- Four-slot, 128-token decode improved from 41.025 to 42.154 aggregate
  tokens/second (2.75%), below the required 20%. Single-slot decode changed
  from 10.823 to 10.778 tokens/second, a 0.42% regression that passes its 3%
  limit.
- Greedy known-answer output was deterministic but invalid and byte-identical
  with MMA enabled or disabled. A 4K `COBALT-7319` needle check also failed.
  This isolates the quality blocker to the existing native vLLM Bonsai
  loading/serving path rather than the new Tensor Core dispatch.

## Operational Update

- On 2026-07-17, the reference `bonsai-27b-ternary` benchmark deployment was
  scaled from eight replicas to zero. It is not routed through the LiteLLM
  gateway. Its Service, manifests, model PVC, and localpool workspace remain
  intact for recovery.
- Kubernetes now reports zero requested GPUs on `gpu-01`; all eight V100s
  report 0 MiB used, 32,495 MiB free, 0% utilization, and no compute
  processes when the canary is stopped.
- A one-V100 `q2-volta-canary` performed all numeric, profiling, real-shape,
  and integrated A/B work. It did not modify the reference Deployment,
  Service, Q4 KV arguments, attention kernels, model PVC, or production
  routing. The temporary Pod was deleted after the failed terminal gates; its
  source, build, and result artifacts remain on the localpool-backed PVC.

## Remaining Gates

- Raising four-slot integrated decode from 2.75% to the required 20% would
  require work outside the Q2 GEMM campaign: unchanged attention/GDN and
  scheduling dominate the measured decode step.
- The native vLLM Bonsai load must produce the reference known answer and
  recover the long-context needle before this kernel can be promoted through
  that serving stack. The temporary DP4A control reproduces the same invalid
  output.
- A controlled DP4A 32K prefill median remains uncollected because the two
  terminal gates above already prevent canary success. The declarative
  reference deployment manifest remains unchanged.

## Expected Write Set

- `csrc/quantization/gguf/q2_0_mma.cuh`
- `csrc/libtorch_stable/quantization/gguf/gguf_kernel.cu`
- `csrc/libtorch_stable/ops.h`
- `csrc/libtorch_stable/torch_bindings.cpp`
- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/quantization/gguf.py`
- `benchmarks/kernels/benchmark_bonsai_q2_0.py`
- focused Q2 dispatch and CUDA tests under `tests/`
- this sprint record and `docs/sprints/ledger.tsv`
