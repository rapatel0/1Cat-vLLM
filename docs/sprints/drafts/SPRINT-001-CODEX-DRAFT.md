# Sprint 001: Validate Hy3 asymmetric W4A16 Marlin on V100

Status: draft
Track: SM70 Runtime Validation
Blocked By: none
Blocks: Hy3 production deployment on eight V100 32 GB GPUs; any tc-grid or `kernels/v100/dense.cuh` fallback sprint
Parallelizable With: none identified; the build artifact, MoE loader, and topology validation are coupled
Shared Files: `CMakeLists.txt`, `csrc/torch_bindings.cpp`, `csrc/moe/torch_bindings.cpp`, `csrc/quantization/marlin/*`, `csrc/moe/marlin_moe_wna16/*`, `vllm/_custom_ops.py`, `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py`, `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py`, `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`, `vllm/model_executor/layers/fused_moe/config.py`, `tests/kernels/moe/test_moe.py`, `tests/quantization/test_compressed_tensors.py`, `docs/sprints/SPRINT-001.md`, `docs/sprints/SPRINT-001-DEFERRED.md`
Cross-Track Risk: any mixed-architecture Marlin build or unrecorded fallback contaminates the evidence for V100 certification

## Overview

This sprint determines whether `cyankiwi/Hy3-AWQ-INT4` can run on eight Tesla V100 32 GB GPUs through the existing SM70 Marlin WNA16 path, rather than starting with a new kernel port. The branch must start from upstream `v1.2.1` at `4e9fdbc807178baa3bc98a1a59af7af7d3b63131` and apply upstream PR #100 by immutable commit SHA, in order: `9afb0434975a13c0632a7e2221da6c5bb8951328`, then `054e2fd223a6eb389dfdf5605716e7b28a60afbe`.

The sprint is evidence-first. A dedicated `TORCH_CUDA_ARCH_LIST=7.0` artifact must prove that SM70 dense Marlin repack and SM70 MoE `moe_wna16_marlin_gemm` are present before any Hy3 model load is treated as signal. Hy3 then has to prove the asymmetric W4A16 compressed-tensors MoE route: group size 32, 4-bit packed weights, `symmetric=false`, `actorder=null`, 192 experts, top-8 routing, and `HYV3ForCausalLM`.

The fallback is deliberately narrow. `tc-grid` and `kernels/v100/dense.cuh` are not touched unless the SM70 Marlin path either fails a minimal asymmetric zero-point reproducer after loader/kernel boundaries are isolated, or passes correctness but misses the defined throughput threshold on the target TP4×PP2 topology.

## Use Cases

1. **Build reproducibility**: An engineer can rebuild the exact V100 artifact from the `v1.2.1` base plus the two PR #100 SHAs without relying on PR merge state or branch movement.
2. **Artifact certification**: Before loading Hy3, the operator can show `_C::sm70_marlin_available() == true`, SM70 cubins exist, and `_moe_C::moe_wna16_marlin_gemm` is registered from the SM70 MoE source set.
3. **Asymmetric MoE validation**: The compressed-tensors MoE loader preserves nonzero zero-point metadata and reaches `fused_marlin_moe` with `uint4` plus `w1_zp` and `w2_zp`, not the symmetric `uint4b8` path.
4. **Eight-GPU topology run**: Hy3 is tested on two four-GPU V100 cliques with tensor parallelism inside each clique and pipeline parallelism across cliques: TP4×PP2, not a four-GPU dummy substitute.
5. **Disciplined fallback decision**: Failures produce a reproducer and decision memo; performance fallback is triggered only by a measured threshold, not by guesswork.

## Architecture

The primary route is:

```text
Hy3 compressed-tensors config
  weights: num_bits=4, group_size=32, symmetric=false, actorder=null
        |
        v
CompressedTensorsMoEMethod.get_moe_method(...)
        |
        v
CompressedTensorsWNA16MarlinMoEMethod
  - registers packed expert weights
  - registers scales
  - must register and repack zero points
        |
        v
FusedMoEQuantConfig
  - weight_dtype="int4"
  - w1_zp/w2_zp are non-None
        |
        v
MarlinExperts or BatchedMarlinExperts
        |
        v
ops.moe_wna16_marlin_gemm(...)
  b_q_type = scalar_types.uint4
  b_qzeros != None
  is_zp_float = true after SM70 conversion
        |
        v
csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu
```

The sprint should treat the current symmetric assertion in `compressed_tensors_moe_wna16_marlin.py` as a known blocker to validate, not as an acceptable product limitation. The SM70 C++ dispatch already has a zero-point contract for `kU4` and `kU8`: it requires zero points, converts int packed zero metadata to fp16 logical zero points when needed, and rejects `uint4b8` when zero points are expected. The loader work is therefore to preserve Hy3's asymmetric metadata into the existing MoE expert path and prove the kernel consumes it correctly.

The build architecture must remain single-arch for Marlin. `CMakeLists.txt` includes SM70 Marlin sources only when `MARLIN_SM70_ARCHS` exists and `MARLIN_OTHER_ARCHS` does not; the MoE source set follows the same rule. A mixed `7.0;7.5` or broader wheel must be classified as invalid evidence because it can skip SM70 Marlin or register incompatible operator implementations.

Runtime topology is fixed for certification:

```text
Clique A: GPU0 GPU1 GPU2 GPU3 -> TP group 0
Clique B: GPU4 GPU5 GPU6 GPU7 -> TP group 1
Pipeline: stage 0 on Clique A, stage 1 on Clique B
Parallelism: tensor_parallel_size=4, pipeline_parallel_size=2
```

The actual GPU IDs must come from `nvidia-smi topo -m` and UUID order on the target host. If the physical cliques are not `0-3` and `4-7`, the launch map must use the measured clique grouping instead of these example IDs.

## Implementation

### Phase 1: Reproducible PR #100 Branch And Artifact (~15% of effort)

**Files:**
- `docs/sprints/SPRINT-001.md` — record exact base, cherry-pick order, and resulting evidence paths.
- Build outputs outside git — wheel, logs, environment capture.

**Tasks:**
- [ ] Start from upstream `v1.2.1` commit `4e9fdbc807178baa3bc98a1a59af7af7d3b63131`.
- [ ] Cherry-pick PR #100 commits in this exact order: `9afb0434975a13c0632a7e2221da6c5bb8951328`, `054e2fd223a6eb389dfdf5605716e7b28a60afbe`.
- [ ] Record `git rev-parse HEAD`, `git show --no-patch --format=fuller HEAD`, CUDA version, driver version, PyTorch version, and `vllm.collect_env`.
- [ ] Build with repository policy-compliant Python commands only:

```bash
uv venv --python 3.12
uv pip install -r requirements/lint.txt
TORCH_CUDA_ARCH_LIST=7.0 uv pip install -e . --torch-backend=auto
```

- [ ] Reject any artifact whose build log does not show only SM70 for Marlin or whose CMake log prints `Skipping SM70 Marlin kernels in mixed Marlin arch build`.

### Phase 2: SM70 Artifact Proof Before Model Load (~15% of effort)

**Files:**
- `CMakeLists.txt` — inspect only unless PR #100 requires correction.
- `csrc/torch_bindings.cpp` — confirm `_C::sm70_marlin_available`.
- `csrc/moe/torch_bindings.cpp` and `csrc/moe/marlin_moe_wna16/*` — confirm MoE op registration and SM70 dispatch.

**Tasks:**
- [ ] Run an import proof on a V100 host:

```bash
.venv/bin/python - <<'PY'
import torch
import vllm._custom_ops as ops

print("cuda_capability", torch.cuda.get_device_capability())
print("sm70_marlin_available", ops.sm70_marlin_available())
print("has_gptq_repack", hasattr(torch.ops._C, "gptq_marlin_repack"))
print("has_awq_repack", hasattr(torch.ops._C, "awq_marlin_repack"))
print("has_moe_marlin", hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"))
assert torch.cuda.get_device_capability()[0:2] == (7, 0)
assert ops.sm70_marlin_available()
assert hasattr(torch.ops._C, "gptq_marlin_repack")
assert hasattr(torch.ops._C, "awq_marlin_repack")
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")
PY
```

- [ ] Inspect extension binaries with `cuobjdump` or `nvdisasm`; archive proof that `_C` contains SM70 dense Marlin repack code and `_moe_C` contains SM70 `moe_wna16_marlin_gemm` code.
- [ ] Confirm no sm_75+ only Marlin cubin is part of the certified path. If a mixed artifact exists, discard it and rebuild.

### Phase 3: Asymmetric W4A16 MoE Loader Support (~25% of effort)

**Files:**
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py`
- `vllm/model_executor/layers/fused_moe/config.py`
- `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`
- `vllm/_custom_ops.py`

**Tasks:**
- [ ] Replace the symmetric-only MoE assertion with explicit support detection:
  - symmetric W4A16 continues to use `scalar_types.uint4b8`;
  - asymmetric W4A16 uses `scalar_types.uint4`;
  - actorder remains rejected for SM70 Marlin MoE because the C++ dispatch requires no act-order.
- [ ] Register zero-point parameters for both expert matrices when `weight_quant.symmetric is False`, with shapes matching the loaded compressed-tensors format and the existing `w13_weight_packed`/`w2_weight_packed` partitioning.
- [ ] Repack zero points into the SM70 MoE logical shape expected by `sm70_marlin_moe_dispatch.cu`: rank 3, fp16 by kernel entry, dimensions `[num_experts, num_groups, size_n]`.
- [ ] Construct `FusedMoEQuantConfig` with `w1_zp` and `w2_zp` instead of the current `None` values.
- [ ] Add logging that records `CompressedTensorsWNA16MarlinMoEMethod`, `symmetric=false`, `group_size=32`, `quant_type=uint4`, and zero-point tensor shapes once per process.
- [ ] Preserve symmetric behavior and existing non-SM70 paths.

### Phase 4: Targeted Asymmetric Kernel Validation (~20% of effort)

**Files:**
- `tests/kernels/moe/test_moe.py`
- `tests/quantization/test_compressed_tensors.py`
- Optional new helper under `tests/kernels/moe/` if the fixture becomes too large.

**Tasks:**
- [ ] Add a Hy3-shaped reduced test derived from `test_fused_marlin_moe`: `b_type=scalar_types.uint4`, `group_size=32`, `topk=8`, no act-order, fp16 activations, nonzero zero points, and expert count scaled down for CI while preserving the same metadata layout.
- [ ] Compare SM70 Marlin MoE output against the existing torch reference with a predeclared fp16 tolerance. Classify any nonzero diff using the project SM70 policy: Type A layout errors require exact root cause; Type B reduction-order noise requires bounded evidence.
- [ ] Add a compressed-tensors MoE loader test that creates or loads a tiny asymmetric W4A16 MoE fixture and asserts:
  - selected method is `CompressedTensorsWNA16MarlinMoEMethod`;
  - `w13_weight_zero_point` and `w2_weight_zero_point` survive loading;
  - `w1_zp` and `w2_zp` in the fused MoE quant config are non-None;
  - `quant_type_id` resolves to asymmetric `uint4`, not symmetric `uint4b8`.
- [ ] Run the relevant targeted tests with `.venv/bin/python -m pytest`, not system Python.

### Phase 5: Hy3 Dummy Load And TP4×PP2 Smoke (~15% of effort)

**Files:**
- No source files required unless Phase 3 exposes loader gaps.
- Runtime evidence files outside git or in an agreed evidence directory.

**Tasks:**
- [ ] Pin Hy3 revision `c8b08e2c23dd45cb1b277d1290800e40c3dd8eec` and record the public config fields before model construction.
- [ ] Run a dummy-weight load on all eight V100s with model-shaped quantized tensors and no 182 GB checkpoint download.
- [ ] Use TP4×PP2, not TP4 alone. Suggested launch shape:

```bash
CUDA_VISIBLE_DEVICES=<cliqueA0>,<cliqueA1>,<cliqueA2>,<cliqueA3>,<cliqueB0>,<cliqueB1>,<cliqueB2>,<cliqueB3> \
VLLM_LOGGING_LEVEL=DEBUG \
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model <hy3-config-or-local-dummy-snapshot> \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --dtype float16 \
  --max-model-len <minimal-smoke-len> \
  --enforce-eager
```

- [ ] Capture `nvidia-smi topo -m`, GPU UUID ordering, NCCL logs, selected quant method logs, zero-point shape logs, and whether startup reaches engine-ready or first-token smoke.
- [ ] Treat a four-GPU load as diagnostic only. It cannot satisfy this sprint's topology requirement.

### Phase 6: Real Checkpoint Deployment And Fallback Decision (~10% of effort)

**Files:**
- Runtime evidence files and sprint notes.
- `docs/sprints/SPRINT-001-DEFERRED.md` — record fallback items not taken.

**Tasks:**
- [ ] Use a pinned local Hy3 snapshot and deterministic short prompts with fixed sampling.
- [ ] Run warmup plus steady-state measurements; record TTFT, decode tokens/sec, memory, startup time, token hashes, output finiteness, and repeated prompt stability.
- [ ] Compare only clique-aligned TP4×PP2 placements. Do not compare placements that split a TP group across cliques.
- [ ] Apply the fallback threshold:
  - Functional fallback threshold: trigger fallback only if a minimal asymmetric W4A16 MoE reproducer still fails after the loader has proven correct zero-point shapes and the failure is isolated to SM70 Marlin dispatch or GEMM.
  - Performance fallback threshold: after functional correctness passes, trigger fallback only if steady-state decode throughput is below 70% of the best available correct non-tc-grid fallback on the same TP4×PP2 placement, or startup remains above 30 minutes after one documented low-risk tuning pass.
  - Scope threshold: stop before tc-grid or `kernels/v100/dense.cuh` if the needed change is broader than a bounded adapter/conversion layer plus targeted tests.
- [ ] If fallback triggers, write a decision memo with the reproducer, failing tensor shapes, logs, and why the threshold was crossed. Otherwise defer fallback work explicitly.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `CMakeLists.txt` | Inspect / possibly modify | Keep SM70 Marlin and SM70 MoE Marlin single-arch source selection correct. |
| `csrc/torch_bindings.cpp` | Inspect / possibly modify | Prove `_C::sm70_marlin_available()` accurately reflects the certified artifact. |
| `csrc/moe/torch_bindings.cpp` | Inspect | Confirm `_moe_C::moe_wna16_marlin_gemm` schema is available. |
| `csrc/quantization/marlin/*` | Inspect / possibly modify | Ensure dense repack and Marlin GEMM are built for SM70 only. |
| `csrc/moe/marlin_moe_wna16/*` | Inspect / possibly modify | Validate asymmetric zero-point dispatch and SM70 MoE kernels. |
| `vllm/_custom_ops.py` | Inspect / possibly modify | Keep Python op wrappers and fake registrations aligned with C++ schemas. |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | Inspect | Ensure dense WNA16 SM70 capability and asymmetric type selection remain correct. |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py` | Modify if needed | Route Hy3 W4A16 MoE to Marlin only when its constraints are met. |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py` | Modify | Add asymmetric W4A16 zero-point loading, repack, quant config, and logging. |
| `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` | Inspect / possibly modify | Verify `w1_zp` and `w2_zp` reach `ops.moe_wna16_marlin_gemm`. |
| `vllm/model_executor/layers/fused_moe/config.py` | Inspect / possibly modify | Confirm W4A16 MoE quant configs preserve zero-point tensors. |
| `tests/kernels/moe/test_moe.py` | Modify | Add asymmetric SM70 Marlin MoE kernel coverage with nonzero zero points. |
| `tests/quantization/test_compressed_tensors.py` | Modify | Add compressed-tensors asymmetric MoE loader coverage. |
| `docs/sprints/SPRINT-001.md` | Create later | Final sprint/evidence summary after merge. |
| `docs/sprints/SPRINT-001-DEFERRED.md` | Create later | Record tc-grid and dense.cuh fallback deferral unless threshold is met. |

## Definition of Done

- [ ] Branch provenance records base `4e9fdbc807178baa3bc98a1a59af7af7d3b63131` plus PR #100 commits `9afb0434975a13c0632a7e2221da6c5bb8951328` and `054e2fd223a6eb389dfdf5605716e7b28a60afbe` in order.
- [ ] Dedicated `TORCH_CUDA_ARCH_LIST=7.0` build completes through `uv` / `.venv/bin/python` tooling and the build log proves SM70 Marlin was built without mixed Marlin arch contamination.
- [ ] Import proof on a V100 shows `ops.sm70_marlin_available()`, dense repack ops, and `_moe_C.moe_wna16_marlin_gemm`.
- [ ] Binary inspection records SM70 cubin or symbol evidence for dense Marlin repack and MoE Marlin GEMM.
- [ ] Asymmetric W4A16 compressed-tensors MoE loader reaches `CompressedTensorsWNA16MarlinMoEMethod` with non-None zero points and `uint4` quant type.
- [ ] Targeted asymmetric MoE tests pass on V100 with nonzero zero points and predeclared tolerances.
- [ ] TP4×PP2 dummy Hy3 load runs across eight V100s using measured clique mapping and records selected method, zero-point metadata, NCCL topology, and startup or first-token result.
- [ ] Real Hy3 checkpoint smoke uses pinned revision `c8b08e2c23dd45cb1b277d1290800e40c3dd8eec`, deterministic prompts, fixed sampling, and recorded throughput/correctness evidence.
- [ ] Any fallback decision cites the explicit functional or performance threshold; otherwise tc-grid and `kernels/v100/dense.cuh` remain deferred.
- [ ] Relevant tests and lint checks are run and recorded; skipped GPU tests are explicitly labeled with host reason.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hy3 asymmetric MoE zero points are dropped by the compressed-tensors loader. | High | High | Add loader-level assertions and tests proving `w1_zp`/`w2_zp` reach `FusedMoEQuantConfig`. |
| SM70 Marlin MoE accepts zero-point metadata but produces incorrect output for group-32 U4. | Medium | High | Add reduced Hy3-shaped kernel reference test with nonzero zero points before real checkpoint load. |
| Mixed-arch artifact masks missing SM70 code or registers an incompatible implementation. | Medium | High | Certify only `TORCH_CUDA_ARCH_LIST=7.0`; discard mixed builds. |
| TP groups are accidentally split across V100 cliques. | Medium | Medium | Gate topology evidence on `nvidia-smi topo -m`, UUID order, and explicit `CUDA_VISIBLE_DEVICES`. |
| Real checkpoint load exceeds V100 memory even with TP4×PP2. | Medium | High | Run dummy model-shaped allocation first and record KV/cache settings before downloading or loading full weights. |
| Fallback scope expands into a broad kernel rewrite. | Medium | High | Enforce the functional, performance, and scope thresholds before touching tc-grid or `dense.cuh`. |
| Upstream PR #100 changes after planning. | Low | Medium | Use immutable commit SHAs only; do not rely on PR mergeability. |

## Security

- Do not contact external services during implementation evidence capture except for explicitly approved model snapshot retrieval; prefer pinned local snapshots for real-weight validation.
- Do not write secrets, tokens, private model paths, or hostnames into committed logs. Evidence should redact credentials and use GPU UUIDs instead of infrastructure identifiers where possible.
- The dummy load must not silently download the 182 GB checkpoint. Any real checkpoint access must be explicit, pinned, and recorded separately from loader validation.
- Generated artifacts, benchmark logs, and build logs should remain outside source control unless a maintainer explicitly approves a sanitized evidence path.

## Dependencies

- Target host with eight Tesla V100 32 GB GPUs arranged as two NVLink-capable four-GPU cliques.
- CUDA toolkit and driver capable of offline SM70 compilation; CUDA 12.8 is acceptable per current CMake comments, while CUDA 13 removes pre-7.5 offline support.
- Python environment managed through `uv` and `.venv/bin/python`; no system `python3` or bare `pip`.
- Upstream `v1.2.1` base and immutable PR #100 commits:
  - `4e9fdbc807178baa3bc98a1a59af7af7d3b63131`
  - `9afb0434975a13c0632a7e2221da6c5bb8951328`
  - `054e2fd223a6eb389dfdf5605716e7b28a60afbe`
- Hy3 model revision `c8b08e2c23dd45cb1b277d1290800e40c3dd8eec` for config and real-weight validation.
- Existing vLLM MoE reference tests and SM70 migration policy in `docs/design/sm70_v100_migration_control.md`.

## Open Questions

1. What are the exact CUDA, driver, PyTorch, and NCCL versions on the target V100 host, and do they match a known successful SM70 build envelope?
2. What are Hy3's exact expert hidden/intermediate dimensions after TP4 partitioning, and do they satisfy every SM70 Marlin MoE alignment check for group size 32?
3. Does the compressed-tensors checkpoint store MoE zero points under names that map cleanly to `w13` and `w2`, or is an expert-specific name remap required?
4. Should the dummy load use a synthetic local config directory or a sanitized pinned snapshot containing config files only?
5. What non-tc-grid correct fallback should be used for the 70% throughput comparison if SM70 Marlin is functionally correct but slow?
6. Where should sanitized build and runtime evidence live so it is durable but does not commit oversized or sensitive logs?
