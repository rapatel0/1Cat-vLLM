# Sprint 001 Codex Critique

## Scope

This critique treats `SPRINT-001-INTENT.md` as the baseline and reviews the
Claude and Antigravity drafts against it. I read only the named draft files and
repository source needed to resolve the WNA16/zero-point question. I did not
edit source code and did not read or write `reference/` or `logs/`.

## Executive Assessment

- **Claude draft:** Strongest draft. It has the right gated shape, catches the
  dedicated-SM70 build requirement, records useful binary evidence, and places
  an asymmetric reference check before an expensive model load. Its main flaw is
  material: it says the sprint is mostly verification and that no production
  source changes are authored beyond PR #100, but the current Python
  compressed-tensors MoE path still needs a loader/packing bridge before Hy3 can
  even reach the SM70 zero-point adapters.
- **Antigravity draft:** Useful high-level skeleton, but not sufficient for this
  sprint. It repeats the intent without resolving the central loader barrier,
  puts asymmetric validation at the dummy-load stage instead of before model
  construction, and lacks enough risk, evidence, and DoD detail for a high-risk
  CUDA/model-loading sprint.

## Source-Backed Zero-Point Assessment

The intent is correct to call out the loader barrier: Hy3 is `symmetric=false`
and the current method asserts `weight_quant.symmetric`, rejecting the model
before CUDA dispatch (`SPRINT-001-INTENT.md:30-35`,
`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py:56-69`).

The C++ SM70 MoE implementation does have explicit zero-point plumbing:
`moe_wna16_marlin_gemm` accepts `b_zeros_or_none`
(`csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu:224-240`), converts
packed int32 zero points to fp16 zero-point metadata
(`csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu:68-159`), validates
zero-point shape and dtype (`.../sm70_marlin_moe_dispatch.cu:497-540`), and
requires zero points for the uint4 path (`.../sm70_marlin_moe_dispatch.cu:570-576`).

That does not make Hy3 loadable today. The current compressed-tensors Marlin MoE
Python path does not create `w13/w2` zero-point parameters, does not pass zero
points to `int4_w4a16_moe_quant_config`, and does not pass them into
`fused_marlin_moe` (`.../compressed_tensors_moe_wna16_marlin.py:466-470`,
`.../compressed_tensors_moe_wna16_marlin.py:552-575`). It also hardwires
`WNA16_SUPPORTED_TYPES_MAP`, which selects symmetric `uint4b8`, while the dense
WNA16 scheme uses `WNA16_ZP_SUPPORTED_TYPES_MAP` for asymmetric weights
(`.../compressed_tensors_wNa16.py:37-38`, `.../compressed_tensors_wNa16.py:74-78`).

Conclusion: the correct plan must include a bounded Python loader/packing bridge
and reference tests before any dummy or real model load. Existing raw kernel
tests are useful precedent because they exercise `scalar_types.uint4` with zero
points and pass `w1_zeros/w2_zeros` into `fused_marlin_moe`
(`tests/kernels/moe/test_moe.py:123-125`, `tests/kernels/moe/test_moe.py:721`,
`tests/kernels/moe/test_moe.py:982-1009`), but they do not prove that
`CompressedTensorsWNA16MarlinMoEMethod` can load Hy3 checkpoint metadata.

## Claude Draft

### Strengths

- Correctly builds the plan around gates: branch provenance, dedicated SM70
  artifact, binary proof, pre-load asymmetric fixture, eight-GPU dummy load, real
  deployment, and disciplined fallback (`SPRINT-001-CLAUDE-DRAFT.md:29-45`).
- Correctly treats mixed-architecture wheels as a first-order risk. The source
  supports this: SM70 MoE sources are compiled only for SM70-only Marlin MoE
  builds, otherwise CMake emits the mixed-arch skip path
  (`CMakeLists.txt:1235-1257`).
- Correctly raises the fp16 runtime constraint. SM70 MoE rejects non-fp16
  activations and outputs for this path
  (`csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu:304-320`;
  `SPRINT-001-CLAUDE-DRAFT.md:110-124`).
- Strong evidence discipline: resulting cherry-pick SHAs, build log, cubins,
  selected scheme, topology, GPU UUID mapping, and throughput are all called out
  in the DoD (`SPRINT-001-CLAUDE-DRAFT.md:377-407`).
- The fallback policy is bounded and avoids prematurely moving to `tc-grid` /
  `dense.cuh` (`SPRINT-001-CLAUDE-DRAFT.md:333-351`, `SPRINT-001-CLAUDE-DRAFT.md:423-441`).

### Weaknesses

- The draft under-scopes source work. It says no production source changes are
  authored beyond PR #100 (`SPRINT-001-CLAUDE-DRAFT.md:357-359`), but the
  current loader path cannot carry asymmetric zero points. At minimum the plan
  needs explicit implementation tasks in
  `compressed_tensors_moe_wna16_marlin.py` and likely a focused loader test.
- The Phase 3 fixture is described mainly as a raw kernel-level fixture
  (`SPRINT-001-CLAUDE-DRAFT.md:239-260`). That is necessary but not sufficient:
  it must also prove that the compressed-tensors MoE loader registers non-null
  zero-point tensors, selects `uint4`, repacks them correctly, and passes them
  through `FusedMoEQuantConfig`.
- It claims Open Question #1 is retired by Phase 3
  (`SPRINT-001-CLAUDE-DRAFT.md:489-493`) before specifying how checkpoint-format
  compressed-tensors zero points are bridged into the SM70 expected layout.
- It uses `logs_ledger/` in example commands while the intent explicitly keeps
  `logs/` out of scope and says evidence should be recorded in repository sprint
  material or a design ledger. That is fixable, but the final plan should choose
  one evidence location and avoid ambiguous local log directories.

### Risk Gaps

- Loader parameter naming and FusedMoE mapping are not spelled out. The FusedMoE
  loader handles `"zero"` names generically (`vllm/model_executor/layers/fused_moe/layer.py:1082-1122`),
  but only if the quant method actually registers the corresponding parameters.
- The plan should explicitly guard against accidental FlashInfer selection for
  `group_size=32` W4 (`compressed_tensors_moe_wna16_marlin.py:80-87`). The
  evidence gate should require the Marlin backend on SM70, not only the class
  name.
- The raw kernel fixture needs a checkpoint-layout case, not only synthetic
  already-Marlin-shaped tensors. Otherwise it can pass while the loader bridge
  still mispacks Hy3 zero points.
- The performance threshold is useful, but proposed numbers should be labeled
  as operator-ratified before they become a hard fallback trigger.

### Missing Edge Cases

- Nonzero zero points for both `w13` and `w2`, including the combined gate/up
  `w13` layout and the down-projection sharding path.
- `top_k=8` and 192-expert metadata, not just smaller generic MoE shapes.
- TP4 sharding effects on scales and zero points, especially `w2` loading.
- Explicit assertion that `actorder=null` results in no `g_idx`/`perm` path,
  because SM70 MoE rejects act-order metadata (`sm70_marlin_moe_dispatch.cu:560-568`).
- A negative test showing Hy3 currently fails at the symmetric assertion before
  the bridge, then passes the selection/metadata test after the bridge.

### DoD Completeness

Mostly complete, but not final as written. Add DoD items for:

- Removing or narrowing the symmetric-only assertion for the supported
  `symmetric=false`, group-32 W4 compressed-tensors MoE case.
- Selecting asymmetric `uint4`/`uint8` scalar types, registering/loading
  `w13/w2` zero-point tensors, and passing non-null zero points through
  `int4_w4a16_moe_quant_config` and `fused_marlin_moe`.
- A CPU/reference or CUDA reference test that starts from checkpoint-style
  packed zero points and proves the loader bridge before any model load.

With those additions, the Claude draft is a strong basis for the final sprint.

## Antigravity Draft

### Strengths

- It preserves the main sequencing from the intent: immutable `v1.2.1` base,
  PR #100 SHAs, dedicated `TORCH_CUDA_ARCH_LIST=7.0`, binary validation, dummy
  load, real deployment, and fallback memo (`SPRINT-001-ANTIGRAVITY-DRAFT.md:16-23`).
- It correctly names the core correctness unknown: whether SM70 Marlin MoE
  handles Hy3 group-32 asymmetric zero points or needs layout/scale conversion
  (`SPRINT-001-ANTIGRAVITY-DRAFT.md:41-44`, `SPRINT-001-ANTIGRAVITY-DRAFT.md:55-59`).
- It is concise enough to serve as an outline.

### Weaknesses

- It does not resolve the central blocker. Current source asserts symmetric
  weights in `CompressedTensorsWNA16MarlinMoEMethod.__init__`, so the dummy load
  DoD cannot succeed without a prior bridge (`compressed_tensors_moe_wna16_marlin.py:56-69`;
  `SPRINT-001-ANTIGRAVITY-DRAFT.md:34-39`).
- It places asymmetric validation at the TP4xPP2 dummy-load stage
  (`SPRINT-001-ANTIGRAVITY-DRAFT.md:21`, `SPRINT-001-ANTIGRAVITY-DRAFT.md:37`),
  while the intent requires a bounded adapter/reference proof before model
  construction (`SPRINT-001-INTENT.md:98-103`, `SPRINT-001-INTENT.md:118-122`).
- It lacks source-backed detail for the CMake mutual-exclusion gate, the C++
  zero-point adapter, the Python quant-type mismatch, and the missing
  zero-point propagation.
- It does not distinguish raw kernel support from compressed-tensors loader
  support, which is the main decision point of the sprint.

### Risk Gaps

- No explicit CUDA version bound or "full native build, not precompiled" guard.
- No proof that mixed-arch builds did not skip SM70 MoE sources.
- No dtype risk for SM70 fp16-only activations/outputs.
- No checkpoint revision pin in the DoD, no source-to-result cherry-pick SHA
  mapping, and no PR #100 diff audit.
- No capacity headroom, `nvidia-smi topo -m`, GPU UUID, or NCCL evidence
  requirements.
- No explicit rule that a four-GPU run is not evidence for TP4xPP2.

### Missing Edge Cases

- Loader-level zero-point parameters and exact packed layout/dtype.
- Nonzero asymmetric fixture before dummy load.
- `top_k=8`, 192 experts, and group-size-32 coverage.
- FlashInfer-vs-Marlin backend assertion.
- Negative tests for current failure mode and positive tests for bridge
  behavior.
- Failure classification between loader layout, zero-point math, build/cubin
  routing, dtype, capacity, and topology.

### DoD Completeness

Incomplete for this sprint. The DoD has branch/build/dummy/deploy/fallback
headings, but it does not prove the asymmetric loader bridge, does not require
non-null zero points, and does not include pre-load reference tests. It should
not be used as the final plan without importing the Claude draft's gates and
adding the loader/packing bridge explicitly.

## Recommended Merge Direction

Use the Claude draft as the base, but revise it so the final sprint is not only
"verification." The final plan should state that PR #100 plus existing SM70 C++
adapters are necessary but not sufficient: the sprint must first implement or
prove a compressed-tensors MoE loader bridge for asymmetric W4A16 zero points,
then run source-backed reference tests, then proceed to dummy load and real
deployment.
