# Sprint 001 Intent: Validate Hy3 asymmetric W4A16 Marlin on V100

effort: high

## Seed

Establish whether `cyankiwi/Hy3-AWQ-INT4` can be served on eight Tesla V100
32 GB GPUs using the existing SM70 Marlin WNA16 implementation. Start from
the 1Cat-vLLM `v1.2.1` release, apply upstream PR #100 by immutable commit
SHA, build a dedicated `TORCH_CUDA_ARCH_LIST=7.0` artifact, then validate the
compressed-tensors MoE path before attempting a real checkpoint deployment
and TP4-clique/PP2 topology experiment. Hold the vendored `tc-grid` and
`kernels/v100/dense.cuh` paths in reserve; they are not the first approach.

## Context

- A clean fork branch, `rapatel0/1Cat-vLLM:hy3-sm70-marlin`, now exactly
  matches upstream `v1.2.1` (`4e9fdbc807178baa3bc98a1a59af7af7d3b63131`). It
  deliberately does not use the fork's legacy `main`, which is 43 commits
  ahead and 3,492 commits behind upstream, nor either legacy feature branch.
- Upstream PR #100 is based on that exact release commit. Its ordered commits
  are `9afb0434975a13c0632a7e2221da6c5bb8951328` and
  `054e2fd223a6eb389dfdf5605716e7b28a60afbe`; its current GitHub merge status
  is `UNSTABLE`, so this sprint must pin SHAs rather than rely on a merge.
- The current CMake build makes SM70 Marlin mutually exclusive with the
  sm_75+ implementation because both sets register the same torch operators.
  The dedicated SM70 source contains dense and MoE Marlin GEMMs, including
  `moe_wna16_marlin_gemm`; a mixed-architecture wheel can otherwise reach
  `gptq_marlin_repack` and fail at CUDA launch.
- Hy3's public configuration is `HYV3ForCausalLM`, 80 layers, 192 experts,
  top-8 routing, 4-bit grouped compressed-tensors WNA16 (`group_size=32`,
  `actorder=null`, `symmetric=false`, `format=pack-quantized`). The current
  `CompressedTensorsWNA16MarlinMoEMethod` explicitly asserts
  `weight_quant.symmetric`; Hy3 is therefore rejected before any CUDA Marlin
  dispatch. PR #100 is necessary for SM70 kernels but not sufficient for this
  model.
- The root repository has no prior `docs/sprints/` material or vision tier.
  Planning therefore starts from this operational goal; the untracked
  `reference/` and `logs/` trees are user-owned and out of scope.

## Recent Sprint Context

Upstream `v1.2.1` is the release baseline. Subsequent upstream commits after
the tag are release-documentation changes and are intentionally excluded from
this reproducibility branch. PR #100 closes #94 and makes #87 actionable;
issue #87 independently reproduces the same MoE WNA16 Marlin repack launch
failure on four V100s.

## Vision Context

No vision document — planning from scratch. The operational direction is to
make a modern asymmetric AWQ MoE model deployable on existing V100 hardware
without prematurely replacing a compatible, already-vendored Marlin path.

## Relevant Codebase Areas

- `CMakeLists.txt` — selects the dedicated SM70 Marlin build and binds
  `ENABLE_SM70_MARLIN`.
- `csrc/quantization/marlin/{gptq,awq}_marlin_repack.cu` — shared repack
  operator registration and error behavior.
- `csrc/moe/marlin_moe_wna16/` and `csrc/moe/torch_bindings.cpp` — Volta MoE
  WNA16 Marlin cubins and torch operation bindings.
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
  — dense WNA16 capability selection.
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py`
  — Hy3's compressed-tensors MoE repack and dispatch path.
- `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` and
  `vllm/_custom_ops.py` — MoE Marlin invocation surface.
- `docs/design/sm70_v100_migration_control.md` and existing SM70 benchmark
  tooling — prior build and runtime evidence conventions.

## Constraints

- Preserve the fork's legacy `main`, `gemma4-12b-ct-awq-v100`, and
  `int2-v100-gemv` branches; implementation work is confined to
  `hy3-sm70-marlin`.
- Begin at the exact `v1.2.1` commit and cherry-pick both PR #100 SHAs in
  order. Do not depend on the PR becoming mergeable.
- Build only a dedicated SM70 artifact: `TORCH_CUDA_ARCH_LIST=7.0`; never
  certify a mixed-architecture artifact for this workload.
- Follow repository policy: Python commands use `uv` or `.venv/bin/python`,
  not system `python3` or bare `pip`.
- The target runtime is eight V100 32 GB GPUs arranged as two local
  NVLink-capable four-GPU cliques: TP4 within each clique, PP2 between them.
  Do not use a four-GPU dummy load as evidence for the TP4×PP2 deployment.
- No tc-grid or `dense.cuh` integration is in the primary scope. Escalate to
  it only after a recorded Marlin-specific incompatibility or performance
  result.

## Success Criteria

1. The fork branch contains the exact `v1.2.1` base plus the two audited PR
   #100 commits, with the resulting SHAs recorded in the deployment evidence.
2. A reproducible dedicated-SM70 build completes and import-time checks show
   `sm70_marlin_available()` is true.
3. Artifact inspection proves that the shared repack and MoE Marlin code is
   present for SM70; no launch reaches an sm_75+ only implementation on V100.
4. A bounded, tested asymmetric W4A16 adapter either makes the Hy3 MoE loader
   select `CompressedTensorsWNA16MarlinMoEMethod` with non-null zero points,
   or proves why the existing SM70 Marlin dispatch cannot consume that layout.
5. Only after criterion 4, a TP4×PP2 Hy3 dummy load reaches a bounded
   engine-start or first-token smoke test without a kernel-image error or
   hidden capability override.
6. A real-weight deployment and topology experiment records correctness,
   capacity, and throughput against a documented, clique-aligned GPU mapping.
7. If the asymmetric adapter cannot pass its reference test, the failure is
   classified with a minimal reproducer and a decision memo explicitly triggers
   the tc-grid/dense.cuh fallback investigation rather than an unbounded
   kernel rewrite.

## Verification Strategy

- **Build provenance:** record commit SHAs, CUDA/toolchain versions,
  `TORCH_CUDA_ARCH_LIST`, environment, `vllm.collect_env`, and build logs.
- **Binary evidence:** import `_C` and `_moe_C`; require
  `torch.ops._C.sm70_marlin_available()` and inspect cubins/symbols for SM70
  dense repack plus `moe_wna16_marlin_gemm` before attempting Hy3.
- **Targeted kernel smoke:** run shape-appropriate W4A16 dense and MoE
  repack/GEMM checks on a V100, including a nonzero-zero-point asymmetric
  fixture. First prove the loader carries zero points through the currently
  symmetric-only `CompressedTensorsWNA16MarlinMoEMethod` boundary, then compare
  the Marlin result with the existing torch reference.
- **Model config audit:** pin Hy3 revision
  `c8b08e2c23dd45cb1b277d1290800e40c3dd8eec`; assert the public quantization
  fields before model construction and preserve the resulting selected scheme
  in logs.
- **Dummy load:** run on the complete eight-GPU TP4×PP2 allocation with
  minimal sequence/KV settings. It intentionally allocates model-shaped
  quantized tensors, so capacity success is part of the result; it must not
  download the 182-GB checkpoint.
- **Real deployment:** use a pinned local snapshot, short deterministic
  prompts, fixed sampling, and a warmup/steady-state measurement. Compare
  repeated token sequences and check output finiteness.
- **Topology experiment:** capture `nvidia-smi topo -m`, GPU UUID ordering,
  and NCCL topology logs. Compare only placements that keep each TP4 group in
  one four-GPU clique, with PP2 spanning the two cliques.

## Uncertainty Assessment

- Correctness uncertainty: **High** — Hy3 currently hits a deliberate
  symmetric-only MoE loader assertion. The C++ Marlin MoE schema accepts an
  optional zero-point tensor, but the required loader conversion is unproven.
- Scope uncertainty: **Medium** — the primary route is bounded, but a
  fallback kernel integration could expand scope substantially.
- Architecture uncertainty: **Medium** — Marlin is existing architecture,
  but correct TP4×PP2 placement and the exact model loader behavior must be
  confirmed on the target cluster.

## Open Questions

1. Can the current SM70 Marlin MoE implementation correctly consume Hy3's
   group-32 asymmetric zero points after a bounded loader conversion, and what
   exact zero-point tensor layout/dtype does the SM70 dispatch require?
2. What exact target-host CUDA, PyTorch, and driver versions are available,
   and do they match the published successful build envelope?
3. Is the two-clique TP4×PP2 runtime already provisioned with a stable
   inter-node transport and a known GPU UUID-to-clique mapping?
4. What performance or startup threshold justifies investigating the tc-grid
   fallback after functional correctness succeeds?
