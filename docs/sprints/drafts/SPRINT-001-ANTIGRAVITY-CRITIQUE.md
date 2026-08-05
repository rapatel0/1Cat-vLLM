# Sprint 001 Critique: Claude vs. Codex Drafts

## 1. Claude's Draft Critique

### Strengths
- **Rigorous Operational Discipline:** Excellent gating mechanism (Phases 0-6) ensuring strict evidence gathering, provenance tracking, and artifact verification.
- **Specific Verification Commands:** Provides concrete `cuobjdump` and `vllm._moe_C` checks to prove that the SM70 code is built and SM75+ code is excluded, directly mitigating the mixed-architecture wheel risk.
- **Clear Topology Strategy:** Strictly enforces the TP4xPP2 topology with specific UUID mapping to ensure NVLink cliques are respected.

### Weaknesses & Risk Gaps
- **Critical Technical Failure:** Claude explicitly states, "No production source changes are authored by this sprint beyond the two cherry-picked PR #100 commits." It fundamentally misunderstands the intent's warning about the Python loader. Because `CompressedTensorsWNA16MarlinMoEMethod.__init__` asserts `weight_quant.symmetric`, the dummy load in Phase 4 will immediately fail with an `AssertionError`.
- **Missing Loader Bridge:** Claude relies heavily on testing the kernel, assuming the python loader will seamlessly pass the zero-points if `dtype=float16` is used. It misses the need to bridge the checkpoint's zero-point metadata to the C++ kernel.

### Missing Edge Cases
- Fails to account for zero-point shape and layout conversions (e.g., from packed `uint4`/`uint4b8` in the checkpoint to the fp16 format expected by the kernel).

### Definition of Done Completeness
- **Incomplete.** The DoD completely misses the necessary Python loader and packing modifications, dooming the sprint to fail at the first model load.

---

## 2. Codex's Draft Critique

### Strengths
- **Accurate Scope & Insight:** Correctly identifies that the `symmetric-only` assertion in Python is a blocker that must be removed.
- **Proper Bridge Scoping:** Explicitly scopes modifying `CompressedTensorsWNA16MarlinMoEMethod` to extract, repack, and register `w1_zp` and `w2_zp` for the SM70 MoE logical shape (Phase 3).
- **Staged Validation:** Accurately phases the work: first modifying the loader (Phase 3), then adding targeted asymmetric kernel reference tests (Phase 4), before attempting the model load.

### Weaknesses & Risk Gaps
- **Weaker Operational Verification:** Lacks the concrete, copy-pasteable artifact verification commands (like `cuobjdump`) that Claude provides. The artifact proof phase relies more on prose than precise execution.
- **Slightly Vague on Fallback:** The fallback thresholds are well-defined in principle, but the draft could be stricter about how to roll back the Python loader changes if the SM70 kernel ultimately fails the asymmetric tests.

### Missing Edge Cases
- Does not extensively cover the exact bit-unpacking logic if the checkpoint zero-points are stored in an incompatible packed format that the SM70 kernel cannot ingest natively.
- Less emphasis on ensuring the `float16` dtype constraint is strictly enforced throughout the stack.

### Definition of Done Completeness
- **Complete.** The DoD covers the end-to-end requirement: PR provenance, single-arch artifact proof, Python loader modifications, targeted tests, dummy load, and real checkpoint deployment.

---

## 3. Source-Backed Point Resolution

**The Issue:** `CompressedTensorsWNA16MarlinMoEMethod.__init__` asserts `weight_quant.symmetric`, which currently rejects the asymmetric Hy3 model (`symmetric=false`). However, the SM70 C++ MoE implementation already contains zero-point adapters, meaning the underlying kernel supports asymmetric W4A16.

**Resolution:** A Python loader/packing bridge is mandatory. The `assert weight_quant.symmetric` must be removed or gated. The loader must extract the zero-point tensors from the checkpoint, repack them into the rank-3 fp16 logical shape expected by the SM70 MoE dispatch (`[num_experts, num_groups, size_n]`), and pass them down via `FusedMoEQuantConfig` (as `w1_zp` and `w2_zp`). Furthermore, reference tests must be implemented to validate this specific bridge and kernel execution before a full model load is attempted.

**Assessment of Plans:**
- **Claude's Plan** incorrectly assumes the existing Python code works out-of-the-box and explicitly forbids source code changes outside the PR. It completely misses the necessary loader bridge and will fail.
- **Codex's Plan** correctly scopes the problem. It explicitly designs Phase 3 to build the Python loader/packing bridge and Phase 4 to implement reference tests for the asymmetric execution before any model load is attempted. Codex is the technically accurate plan.
