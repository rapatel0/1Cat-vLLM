# Sprint 001: Validate Hy3 asymmetric W4A16 Marlin on V100

## Overview
Establish whether the `cyankiwi/Hy3-AWQ-INT4` model can be served on eight Tesla V100 32 GB GPUs using the existing SM70 Marlin WNA16 implementation. We will start from the `1Cat-vLLM` `v1.2.1` release (`4e9fdbc807178baa3bc98a1a59af7af7d3b63131`), apply upstream PR #100 by specific immutable commit SHAs, build a dedicated `TORCH_CUDA_ARCH_LIST=7.0` artifact, and validate the compressed-tensors MoE path. A real checkpoint deployment will then be performed under a TP4×PP2 topology.

## Use Cases
- **Asymmetric W4A16 MoE Validation:** Verify that Hy3's public configuration (`HYV3ForCausalLM`, 80 layers, 192 experts, top-8 routing, 4-bit grouped compressed-tensors WNA16 with `group_size=32`, `actorder=null`, `symmetric=false`, `format=pack-quantized`) can successfully execute the asymmetric zero-point path on Volta architecture.
- **V100 32GB Deployment:** Enable modern MoE models on existing older V100 hardware without prematurely replacing a compatible, already-vendored Marlin path.

## Architecture
- **Base:** Fork branch `hy3-sm70-marlin` matching upstream `v1.2.1` release.
- **Topology:** Eight V100 32 GB GPUs arranged as two local NVLink-capable four-GPU cliques: TP4 within each clique, and PP2 between them.
- **Execution Path:** The runtime will use the dedicated SM70 Marlin build (`ENABLE_SM70_MARLIN`). The build must remain mutually exclusive with the sm_75+ implementation.
- **Kernel Fallback:** The vendored `tc-grid` and `kernels/v100/dense.cuh` paths are held in reserve and are NOT the primary approach.

## Implementation
1. **Branch Preparation:** Checkout `rapatel0/1Cat-vLLM:hy3-sm70-marlin` at `v1.2.1` commit `4e9fdbc807178baa3bc98a1a59af7af7d3b63131`.
2. **Cherry-Pick PR #100:** Apply upstream PR #100 commits exactly by their immutable SHAs: `9afb0434975a13c0632a7e2221da6c5bb8951328` followed by `054e2fd223a6eb389dfdf5605716e7b28a60afbe`.
3. **Build Artifact:** Use `uv` to construct a Python environment and compile a dedicated SM70 artifact using `TORCH_CUDA_ARCH_LIST=7.0`. Ensure system `python3` or bare `pip` are not used.
4. **Binary Validation:** Assert `torch.ops._C.sm70_marlin_available()` and ensure SM70 dense repack plus `moe_wna16_marlin_gemm` are present. No sm_75+ fallback should be hit.
5. **Dummy Load Test:** Execute a TP4×PP2 Hy3 dummy load with model-shaped quantized tensors to test `CompressedTensorsWNA16MarlinMoEMethod` selection and asymmetric W4A16 repack without downloading the full 182-GB checkpoint.
6. **Real Deployment:** Conduct a real-weight deployment using the specified clique-aligned GPU mapping. Use deterministic prompts to record correctness, capacity, and throughput.
7. **Fallback Path:** If the asymmetric WNA16 path fails, classify the failure with a minimal reproducer and generate a decision memo to explicitly trigger the `tc-grid` and `dense.cuh` fallback investigation. Do not rewrite kernels boundlessly.

## Files Summary
- `CMakeLists.txt`: SM70 Marlin build bindings.
- `csrc/quantization/marlin/{gptq,awq}_marlin_repack.cu`: Repack operator registration.
- `csrc/moe/marlin_moe_wna16/` & `csrc/moe/torch_bindings.cpp`: Volta MoE WNA16 Marlin cubins and bindings.
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py`
- `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`
- `vllm/_custom_ops.py`

## Definition of Done
1. **Branch Integrity:** Fork branch contains the exact `v1.2.1` base and the two PR #100 commits (`9afb0434975a13c0632a7e2221da6c5bb8951328`, `054e2fd223a6eb389dfdf5605716e7b28a60afbe`).
2. **SM70 Artifact Proof:** A reproducible build finishes and import checks confirm `sm70_marlin_available()` is true, and the correct cubins for SM70 are included.
3. **Asymmetric W4A16 MoE Validation:** A TP4×PP2 Hy3 dummy load successfully selects `CompressedTensorsWNA16MarlinMoEMethod`, completes its asymmetric W4A16 repack, and executes a bounded smoke test without kernel-image errors.
4. **Topology Verification:** Real deployment completes correctness, capacity, and throughput tests using a validated clique-aligned TP4×PP2 mapping.
5. **Disciplined Fallback:** Any asymmetric WNA16 failure triggers a minimal reproducer and a decision memo, halting work before engaging `tc-grid` or `dense.cuh` integrations.

## Risks
- **Correctness:** No direct evidence yet that the SM70 Marlin MoE implementation handles Hy3's group-32 asymmetric zero points properly on Volta without a layout/scale conversion.
- **Topology Setup:** Incorrect TP4×PP2 placement or model loader behavior on the cluster could lead to OOM or NVLink inefficient configurations.
- **Scope Creep:** Engaging `tc-grid/dense.cuh` fallback integrations without a decision memo will lead to unbounded scope creep.

## Security
- No external unverified binaries will be executed.
- Ensure isolation in the test cluster environment, avoiding downloads of malicious checkpoint files if testing outside verified cyankiwi repositories.

## Dependencies
- Existing NVLink-capable eight V100 32GB GPUs.
- Exact PyTorch and CUDA versions compatible with the published successful SM70 build envelope.
- `uv` package manager for virtual environment and build requirements (following `AGENTS.md`).

## Open Questions
- Does the SM70 Marlin MoE implementation correctly consume Hy3's group-32 asymmetric zero points, or does it require a layout/scale conversion?
- What exact target-host CUDA, PyTorch, and driver versions are available, and do they match the published successful build envelope?
- Is the two-clique TP4×PP2 runtime already provisioned with a stable inter-node transport and a known GPU UUID-to-clique mapping?
- What performance or startup threshold justifies investigating the `tc-grid` fallback after functional correctness succeeds?
