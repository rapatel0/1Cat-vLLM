# Sprint 001: Hy3 Asymmetric W4A16 Marlin on SM70

Status: planned
Track: SM70 compatibility
Branch: `rapatel0/hy3-sm70-marlin`
Effort: high

## Outcome

Make the compressed-tensors MoE Marlin path correctly handle Hy3's asymmetric
W4A16 metadata on V100, then prove it with a dedicated-SM70 artifact, an
adversarial V100 reference test, and an eight-GPU TP4xPP2 dummy-load smoke.

Sprint exit is functional, not performance-based: the loader must preserve
asymmetric zero points and the dummy run must select and enter the Marlin path
without a kernel-image or capability failure. A real-weight deployment and
throughput tuning are deferred. `tc-grid` and `kernels/v100/dense.cuh` remain
fallback-only.

## Provenance and Non-Negotiables

- The implementation branch starts at upstream `v1.2.1`, commit
  `4e9fdbc807178baa3bc98a1a59af7af7d3b63131`. The fork's legacy `main` and
  legacy feature branches are out of scope.
- Cherry-pick PR #100 by immutable source SHA, in order:
  `9afb0434975a13c0632a7e2221da6c5bb8951328`, then
  `054e2fd223a6eb389dfdf5605716e7b28a60afbe`. Record the resulting local
  commit SHAs; do not depend on the open PR's merge state.
- Build a native, dedicated SM70 artifact only:
  `TORCH_CUDA_ARCH_LIST=7.0`. A mixed-architecture Marlin build is invalid
  evidence because the SM70 and sm75+ paths register overlapping torch ops.
- Use CUDA 12.8-compatible tooling; CUDA 13 is outside the source branch's
  supported SM70 architecture list. All Python commands use `uv` or
  `.venv/bin/python`.
- Hy3 is pinned for configuration checks at
  `c8b08e2c23dd45cb1b277d1290800e40c3dd8eec`: 4-bit, group size 32,
  `symmetric=false`, `actorder=null`, 192 experts, top-k 8.
- No real checkpoint download occurs during this sprint's dummy-load path.

## Why Source Work Is Required

PR #100 makes the dedicated SM70 kernels buildable and removes the dense
capability gate. It does not make Hy3's MoE loadable. The current
`CompressedTensorsWNA16MarlinMoEMethod` rejects asymmetric weights, registers
no MoE zero-point parameters, never repacks them, passes `w1_zp=None` and
`w2_zp=None`, and selects symmetric `uint4b8`. Hy3 needs `uint4` with real
zero-point metadata. The SM70 C++ op already supports that contract.

The bridge must preserve these semantics:

- asymmetric `uint4` requires zero points; symmetric `uint4b8` rejects them;
- logical zero points are `zero_point_int * scale`, not raw integer zeros;
- zero-point N ordering must match Marlin's scale permutation and any required
  nibble interleave;
- both `w13` and `w2` must carry non-null zero points.

## Plan

### 1. Freeze the source and build provenance

1. On `hy3-sm70-marlin`, cherry-pick the two PR #100 commits in order and
   record source-to-result SHA mapping, `git diff v1.2.1..HEAD`, CUDA/driver,
   PyTorch, NCCL, and `vllm.collect_env` output in sanitized external evidence.
2. Build from source with `TORCH_CUDA_ARCH_LIST=7.0`. Reject a build log that
   shows mixed Marlin arches or the SM70-skip CMake warning.
3. Retain the build command and toolchain versions, but not credentials,
   hostnames, tokens, or private paths.

### 2. Prove the artifact and route before loading a model

1. On a V100, require capability `(7, 0)`,
   `ops.sm70_marlin_available()`, dense repack ops, and
   `torch.ops._moe_C.moe_wna16_marlin_gemm`.
2. Inspect the extension with `cuobjdump` or `nvdisasm` and save sanitized
   evidence that SM70 dense repack and SM70 MoE Marlin code are present.
3. Add a static routing/alignment check for the Hy3 TP4 partition: evaluate
   `check_moe_marlin_supports_layer` and the kernel's group-32/scale alignment
   conditions using the actual post-TP dimensions. Stop before loader work if
   the layer cannot route to Marlin; the non-Marlin fallback shares the same
   asymmetric blocker and is not success evidence.

### 3. Implement the bounded asymmetric loader bridge

1. In `compressed_tensors_moe_wna16_marlin.py`, replace the symmetric-only
   assertion with explicit supported constraints: W4A16, group size accepted by
   Marlin, no act-order, and no unsupported bias. Preserve the old symmetric
   path unchanged.
2. Register checkpoint-loadable zero-point parameters for both `w13` and `w2`.
   Select `uint4` when the quantization config is asymmetric and retain
   `uint4b8` only for symmetric weights.
3. Derive the conversion from the existing Marlin/AWQ helpers and prove the
   exact checkpoint layout. The preferred representation is fp16 logical zeros
   (`zero_point_int * scale`) in Marlin-permuted N order. Use the existing
   repack helper where it establishes the needed permutation/interleave.
4. Feed non-null `w1_zp` and `w2_zp` into `FusedMoEQuantConfig` and the
   selected experts path. Confirm whether monolithic `apply()` is unreachable;
   otherwise make it propagate zero points or fail closed rather than silently
   taking a symmetric path.
5. If the existing C++ interface cannot correctly express the proven layout,
   a small C++ change is allowed only when paired with a focused regression
   test and a clear op-contract explanation. A new kernel family or tc-grid
   integration is outside this sprint.

### 4. Prove asymmetric correctness on a V100

1. Add a compressed-tensors loader test using checkpoint-style asymmetric
   metadata. Assert Marlin method selection, zero points for `w13` and `w2`,
   and `uint4` rather than `uint4b8`.
2. Add a V100-only reduced Hy3-shaped MoE reference case: group size 32,
   top-k 8, fp16 activations, no act-order, and deliberately nonzero zero
   points. Compare Marlin output with the existing torch reference under a
   declared fp16 tolerance.
3. Make the fixture fail if zero points are dropped, left unscaled, or left in
   unpermuted N order. Record whether the modular experts or monolithic path
   executes.
4. Run targeted tests with `.venv/bin/python -m pytest`. A V100 is a scheduled
   dependency for the CUDA reference case; a CPU-only pass cannot certify it.

### 5. Run the target-neutral TP4xPP2 dummy smoke

1. Preflight an eight-V100 allocation: two local NVLink-capable four-GPU
   cliques, sufficient model/KV headroom, and a measured UUID-to-clique map.
   If PP2 spans hosts, additionally prove the multi-node transport before
   treating it as topology evidence.
2. Use a pinned, config-only/local dummy snapshot and `--load-format dummy`.
   Launch with `--tensor-parallel-size 4`, `--pipeline-parallel-size 2`, and
   `--dtype float16`; SM70 Marlin MoE rejects BF16.
3. Capture topology, GPU UUID order, NCCL diagnostics, selected quant method,
   zero-point shapes, and engine-ready or first-token outcome. A four-GPU run
   is diagnostic only and cannot satisfy this sprint.

## Definition of Done

- [ ] The branch records v1.2.1 and both PR #100 source/result SHAs.
- [ ] A native `TORCH_CUDA_ARCH_LIST=7.0` build completes with SM70 Marlin
  evidence and no mixed-arch certification.
- [ ] V100 import and binary checks prove the dense repack and MoE Marlin ops.
- [ ] Hy3's TP4 dimensions statically route to Marlin and meet group-32/kernel
  alignment requirements.
- [ ] The loader carries checkpoint-style non-null `w13`/`w2` zero points,
  selects `uint4`, and has no silent symmetric fallback.
- [ ] The adversarial asymmetric V100 reference test passes at the declared
  tolerance and records the zero-point semantic/layout it validates.
- [ ] A clique-aligned eight-V100 TP4xPP2 dummy smoke reaches engine-ready or
  a first-token attempt without the prior kernel-image/capability failure.
- [ ] Tests, lint, skips, and sanitized evidence locations are recorded.

## Decision Rules

`tc-grid` and `dense.cuh` remain deferred unless a minimal reproducer proves a
Marlin-specific functional incompatibility after the loader bridge and V100
reference test are complete. Any such escalation requires a short decision
memo with tensor shapes, observed contract mismatch, and why a bounded C++
adjustment cannot resolve it. Performance is deliberately not an exit metric
for this sprint.

## Expected Write Set

- `CMakeLists.txt`
- `csrc/quantization/marlin/awq_marlin_repack.cu`
- `csrc/quantization/marlin/gptq_marlin_repack.cu`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils.py` only if an
  existing helper must be extended for the proven zero-point layout
- `tests/kernels/moe/test_moe.py`
- `tests/quantization/test_compressed_tensors.py`
- narrowly-scoped C++/binding files only if Phase 3 proves the existing op
  cannot represent the correct layout

No changes to the fork's legacy branches, `reference/`, or `logs/` are in
scope.
