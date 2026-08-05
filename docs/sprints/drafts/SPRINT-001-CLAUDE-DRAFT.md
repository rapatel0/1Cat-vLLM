# Sprint 001 (Claude Draft): Validate Hy3 asymmetric W4A16 Marlin on V100

- **Effort:** high
- **Branch:** `rapatel0/1Cat-vLLM:hy3-sm70-marlin` (implementation confined here)
- **Base commit (immutable):** `4e9fdbc807178baa3bc98a1a59af7af7d3b63131` (`1Cat-vLLM v1.2.1`)
- **Model under test:** `cyankiwi/Hy3-AWQ-INT4`, pinned revision
  `c8b08e2c23dd45cb1b277d1290800e40c3dd8eec`
- **Status:** planning draft only — no code, branches, or deployments produced by this document.

> This is an independent draft written from the intent, the repository, and
> `AGENTS.md`/`CLAUDE.md`. It parts from the intent on one material point of
> emphasis: the SM70 Marlin dense **and** MoE kernels — including the
> asymmetric zero-point adapters — are already present in the `v1.2.1` tree.
> The sprint is therefore primarily a **verification and correctness-proof**
> exercise on an existing code path, not a kernel-authoring exercise. PR #100
> is treated as an immutable, audited input that hardens that path; its two
> commits are pinned by SHA and never depended on for merge state.

---

## Overview

Determine, with recorded evidence, whether `cyankiwi/Hy3-AWQ-INT4`
(`HYV3ForCausalLM`; 80 layers; 192 experts; top-8 routing; 4-bit grouped
compressed-tensors WNA16 with `group_size=32`, `actorder=null`,
`symmetric=false`, `format=pack-quantized`) can be served on eight Tesla V100
32 GB GPUs through the existing SM70 Marlin WNA16 path.

The work proceeds in gated phases:

1. Reconstruct the branch from `v1.2.1` plus the two audited PR #100 commits,
   recording both the *source* SHAs and the *resulting* cherry-pick SHAs.
2. Build one dedicated `TORCH_CUDA_ARCH_LIST=7.0` artifact and prove, by binary
   inspection, that the SM70 dense repack and MoE `moe_wna16_marlin_gemm` are
   compiled in and that **no** sm_75+-only Marlin MoE cubin is present.
3. Prove the asymmetric (`symmetric=false`, `group_size=32`) W4A16 MoE path is
   *numerically correct* on a V100 with a cheap kernel-level fixture, before
   spending money on a full model.
4. Run a TP4×PP2 dummy load on the complete eight-GPU allocation to prove
   scheme selection, repack, capacity, and a bounded first-token smoke test.
5. Deploy the real pinned checkpoint on the clique-aligned TP4×PP2 topology and
   record correctness, capacity, and throughput.
6. Only on a *classified* failure or a *ratified* performance-floor breach,
   emit a decision memo that authorizes a bounded tc-grid / `dense.cuh`
   investigation — never an open-ended kernel rewrite.

The controlling insight — established from the tree, not assumed — is that the
correct kernels already exist and are already guarded; the risk is (a) that a
**mixed-architecture** wheel silently routes the MoE GEMM to an sm_75+ kernel
that dies at launch on Volta, and (b) that the asymmetric zero-point math,
though implemented, has never been validated for Hy3's `group_size=32` layout.
This sprint retires both risks with evidence.

---

## Use Cases

- **Asymmetric W4A16 MoE inference on Volta.** A modern grouped
  compressed-tensors MoE (explicit, nonzero zero points) executes correctly
  through `CompressedTensorsWNA16MarlinMoEMethod` →
  `torch.ops._moe_C.moe_wna16_marlin_gemm` on sm_70, producing finite,
  reference-consistent logits.
- **Reuse existing V100 fleet for a large MoE.** Serve a checkpoint whose
  weights (~182 GB packed) exceed any single-node dense-GPU budget by sharding
  it TP4×PP2 across two four-GPU NVLink cliques, without buying newer hardware
  and without prematurely replacing a compatible, already-vendored Marlin path.
- **Reproducible provenance for a fork release.** Produce a single, auditable
  build+deploy ledger (SHAs, toolchain, cubin listing, selected scheme,
  topology, benchmarks) that a human can defend end-to-end, consistent with the
  existing `docs/design/sm70_v100_migration_control.md` evidence convention.

---

## Architecture

### Existing code path (verified at `v1.2.1`)

| Layer | Component | Location |
| --- | --- | --- |
| Capability probe | `sm70_marlin_available()` op + Python wrapper | `csrc/torch_bindings.cpp:13,137-138`; `vllm/_custom_ops.py:1233` |
| Dense SM70 Marlin | repack + GEMM sources; `ENABLE_SM70_MARLIN=1` | `csrc/quantization/marlin/sm70_*.cu`; `CMakeLists.txt:447-467` |
| MoE SM70 Marlin | `moe_wna16_marlin_gemm` + asymmetric zp adapters | `csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu:224,122,98,68` |
| MoE op binding | `moe_wna16_marlin_gemm(...)` → `torch.ops._moe_C` | `csrc/moe/torch_bindings.cpp:81` |
| MoE method select | `CompressedTensorsWNA16MarlinMoEMethod` | `.../compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py:56` |
| Dense scheme select | `CompressedTensorsWNA16` + SM70 gating | `.../schemes/compressed_tensors_wNa16.py:42,81-84,239` |

### The mutual-exclusivity gate (the crux)

`CMakeLists.txt:1235-1257` compiles the SM70 MoE dispatch **only** when the
build targets sm_70 *and no other Marlin-MoE arch*:

```
cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_SM70_ARCHS  "7.0"          "${CUDA_ARCHS}")
if (MARLIN_MOE_SM70_ARCHS AND NOT MARLIN_MOE_OTHER_ARCHS)
    #   -> compiles sm70_marlin_moe_dispatch.cu (provides moe_wna16_marlin_gemm)
elseif (MARLIN_MOE_SM70_ARCHS)
    #   -> "Skipping SM70 Marlin MOE kernels in mixed Marlin arch build"
```

Consequently a **mixed** wheel (e.g. `TORCH_CUDA_ARCH_LIST=7.0;8.0`) drops the
SM70 MoE dispatch and lets the sm_75+ `ops.cu` implementation
(`CMakeLists.txt:1335`) claim the same `moe_wna16_marlin_gemm` operator — which
then fails at CUDA launch on a V100 (the "reach `gptq_marlin_repack` and fail"
symptom in the intent, and the four-V100 repack launch failure of issue #87).
A **dedicated `TORCH_CUDA_ARCH_LIST=7.0` artifact is therefore mandatory, not a
preference**: it is the only configuration in which the working Volta MoE
kernel wins operator registration.

### Runtime dtype constraint (verified, surfaced early)

`sm70_marlin_moe_dispatch.cu:304-320` hard-checks that activations, outputs,
and (non-FP4) scales are **float16**:

```
TORCH_CHECK(a_type == kFloat16, "SM70 Marlin MoE supports only float16 activations.");
TORCH_CHECK(c_type == kFloat16, "SM70 Marlin MoE supports only float16 outputs.");
TORCH_CHECK(b_type == kU4 || kU4B8 || kU8 || kU8B128 || kFE4M3fn || kFE2M1f, ...);
```

Hy3's asymmetric grouped W4 maps to `b_type = kU4` with explicit `b_zeros`.
The deployment **must** load in `dtype=float16`; a `bfloat16` default trips a
hard `TORCH_CHECK` at the first MoE GEMM. This constraint is asserted before
any load (Phase 3/4), not discovered at runtime.

### Target topology

Eight V100 32 GB GPUs as two NVLink-capable four-GPU cliques: **TP4 within each
clique, PP2 between them.** Each TP4 group must live entirely inside one
four-GPU clique so tensor-parallel all-reduces stay on NVLink; PP2 spans the
two cliques over the slower inter-clique link. A four-GPU run is a *unit test
of the TP4 shard only* and is explicitly not accepted as evidence for the
TP4×PP2 deployment.

### Reserved fallback (not primary scope)

The vendored `tc-grid` and `kernels/v100/dense.cuh` paths are held in reserve.
They are engaged only through the disciplined-fallback gate (see Implementation
Phase 6 and the threshold in Risks), after a recorded, classified Marlin
incompatibility or a ratified performance-floor breach.

---

## Implementation

Each phase ends in a **gate** with a named evidence artifact. A gate that fails
halts the sprint at that phase; downstream phases do not begin. All Python runs
use `uv` / `.venv/bin/python` — never system `python3` or bare `pip`
(`AGENTS.md §2`). Because this build includes C/C++/CUDA changes, it must be a
**full** build: do **not** set `VLLM_USE_PRECOMPILED=1` (that flag skips the
native build and would ship stock kernels).

All evidence lands in a single ledger,
`docs/design/hy3_sm70_marlin_validation.md`, following the format of the
existing `sm70_v100_migration_control.md`.

### Phase 0 — Environment & provenance baseline

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/lint.txt && pre-commit install
.venv/bin/python -c "import torch, subprocess; print(torch.version.cuda)"
.venv/bin/python -m vllm.collect_env > logs_ledger/collect_env.txt   # ledger copy
```

Record CUDA toolchain version, driver, PyTorch, and confirm the toolchain is in
`[12.8, 13.0)` — CUDA ≥ 13.0 removes `7.0` from `CUDA_SUPPORTED_ARCHS`
(`CMakeLists.txt:102-116`) and would silently produce **no** SM70 code.

**Gate G0:** `collect_env` captured; CUDA in `[12.8, 13.0)` confirmed on the
build host.

### Phase 1 — Branch reconstruction from immutable SHAs

Confirm `HEAD == 4e9fdbc807178baa3bc98a1a59af7af7d3b63131`, then cherry-pick the
two PR #100 commits **in order**:

```bash
git rev-parse HEAD          # must equal the v1.2.1 base SHA
git cherry-pick 9afb0434975a13c0632a7e2221da6c5bb8951328
git cherry-pick 054e2fd223a6eb389dfdf5605716e7b28a60afbe
git log --oneline 4e9fdbc807178baa3bc98a1a59af7af7d3b63131..HEAD   # exactly 2 commits
git rev-parse HEAD~1 HEAD    # RESULTING SHAs — record both
```

Cherry-pick creates **new** commits, so the resulting SHAs differ from the
source PR #100 SHAs. The ledger records the mapping explicitly:

| Role | SHA |
| --- | --- |
| Base (`v1.2.1`) | `4e9fdbc807178baa3bc98a1a59af7af7d3b63131` |
| PR #100 source commit 1 | `9afb0434975a13c0632a7e2221da6c5bb8951328` |
| PR #100 source commit 2 | `054e2fd223a6eb389dfdf5605716e7b28a60afbe` |
| Resulting branch commit 1 | *(record `HEAD~1` after cherry-pick)* |
| Resulting branch commit 2 | *(record `HEAD` after cherry-pick)* |

If either cherry-pick conflicts, stop and record the conflict; do **not**
improvise a resolution that diverges from PR #100's intent, and do not rely on
the PR's `UNSTABLE` merge status. Legacy `main`, `gemma4-12b-ct-awq-v100`, and
`int2-v100-gemv` branches are left untouched.

**Gate G1:** `4e9fdbc80..HEAD` is exactly two commits; source→resulting SHA map
recorded.

### Phase 2 — Dedicated SM70 artifact + binary proof

```bash
# Full native build (AGENTS.md §2, C/C++ changes). Do NOT set VLLM_USE_PRECOMPILED.
TORCH_CUDA_ARCH_LIST=7.0 uv pip install -e . --torch-backend=auto \
  2>&1 | tee logs_ledger/build_sm70.log
```

Watch the build log for `Building SM70 Marlin MOE kernels for archs: 7.0` and
the **absence** of `Skipping SM70 Marlin MOE kernels in mixed Marlin arch
build` (`CMakeLists.txt:1254,1256`). Then prove the artifact:

```bash
.venv/bin/python - <<'PY'
from vllm import _custom_ops as ops
import torch, vllm._C, vllm._moe_C
assert ops.sm70_marlin_available() is True, "sm70_marlin_available() must be True"
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"), "MoE Marlin op missing"
print("OK: sm70 available and moe_wna16_marlin_gemm bound")
PY

# Cubin-level proof: SM70 present, no sm_75+ Marlin-MoE cubin shipped.
SO=$(.venv/bin/python -c "import vllm._moe_C, os; print(vllm._moe_C.__file__)")
cuobjdump --list-elf "$SO" | tee logs_ledger/moe_cubins.txt | grep -E "sm_70" | head
! cuobjdump --list-elf "$SO" | grep -E "sm_(75|80|86|89|90)" && echo "OK: no sm_75+ cubin"
```

`cuobjdump --list-elf` enumerates the embedded cubins by arch; the dedicated
build must show `sm_70` and no higher Marlin-MoE arch. This directly discharges
Success Criterion #3 ("no launch reaches an sm_75+ only implementation").

**Gate G2:** `sm70_marlin_available()` True; `moe_wna16_marlin_gemm` bound; cubin
listing shows sm_70 and no sm_75+ Marlin-MoE arch. Log + listing in ledger.

### Phase 3 — Asymmetric-correctness micro-fixture (cheap, decisive)

Before any model load, retire the material unknown (Open Question #1) with a
kernel-level fixture on **one V100**. Construct a synthetic
`format=pack-quantized`, `group_size=32`, `symmetric=false` expert weight with
**nonzero** zero points and known fp16 reference, then compare
`moe_wna16_marlin_gemm` output against an fp16 dequant→matmul reference:

- **Asymmetric case (primary):** random weights, nonzero per-group zero points,
  `b_type=kU4`, fp16 activations/scales. Relative L2 error vs. reference
  **≤ 2e-2**; no NaN/Inf.
- **Symmetric control:** identical shapes with zero-valued zero points. If the
  symmetric control passes but the asymmetric case fails, the defect is
  localized to the zero-point adapter (`sm70_moe_zp_as_half` /
  `sm70_unpack_moe_zp_to_half_kernel`, `sm70_marlin_moe_dispatch.cu:122,68`),
  which is a bounded, in-file fix — not a fallback trigger.
- **Shape coverage:** at least the Hy3-relevant `size_k` divisibility for
  `group_size=32` and a `top_k=8` routing shape.

**Gate G3:** asymmetric fixture within tolerance and finite on a V100; symmetric
control passes; both saved to the ledger. A failure here routes to the
Phase-6 classification path, *not* onward to an expensive load.

### Phase 4 — TP4×PP2 dummy load (full eight-GPU allocation)

Run vLLM in dummy-weights mode on **all eight** GPUs, TP4×PP2, with minimal
sequence/KV settings. This allocates model-shaped quantized tensors (so
capacity success is a real result) but must **not** download the 182 GB
checkpoint.

```bash
VLLM_LOGGING_LEVEL=DEBUG \
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model cyankiwi/Hy3-AWQ-INT4 --revision c8b08e2c23dd45cb1b277d1290800e40c3dd8eec \
  --load-format dummy --dtype float16 \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.90 \
  --enforce-eager
```

Assertions from logs/telemetry:

- Config audit: public quant fields (`group_size=32`, `actorder=null`,
  `symmetric=false`, `format=pack-quantized`) asserted **before** construction;
  `dtype=float16` enforced.
- Scheme selection: `CompressedTensorsWNA16MarlinMoEMethod` chosen; **no**
  hidden capability override to a non-Marlin scheme. Selected scheme captured
  from logs.
- Repack: asymmetric W4A16 repack completes with no kernel-image error.
- Capacity: ~182 GB / 8 ≈ **22.75 GB weights/GPU**, leaving ~9 GB/GPU for KV,
  activations, Marlin workspace, and CUDA context; require ≥ 2 GB free per GPU
  after KV allocation at this smoke config. Record `nvidia-smi` per-GPU memory.
- Smoke: bounded engine-start or single first-token generation succeeds.

**Gate G4:** correct scheme selected, repack clean, capacity headroom met,
first-token smoke passes — all on the eight-GPU TP4×PP2 allocation.

### Phase 5 — Real-weight deployment + topology experiment

Use a **pinned local snapshot** of revision
`c8b08e2c23dd45cb1b277d1290800e40c3dd8eec` (pre-staged; no live 182 GB pull
during the run). Capture the physical topology first:

```bash
nvidia-smi topo -m > logs_ledger/topo.txt
nvidia-smi --query-gpu=index,uuid --format=csv > logs_ledger/gpu_uuids.txt
```

Derive the clique-aligned mapping and pin GPU order via
`CUDA_VISIBLE_DEVICES` so each TP4 group is one NVLink clique:

| Pipeline stage | Clique | Physical GPUs (by UUID) | Parallel role |
| --- | --- | --- | --- |
| PP stage 0 | Clique A | GPUs 0–3 (record UUIDs) | TP4 |
| PP stage 1 | Clique B | GPUs 4–7 (record UUIDs) | TP4 |

Serve with `--dtype float16 --tensor-parallel-size 4 --pipeline-parallel-size 2`,
short deterministic prompts, fixed sampling (`temperature=0`, fixed seed),
warmup then steady-state measurement (NUMA-pin workers before benchmarking, per
established V100 bench practice). Record:

- **Correctness:** repeated identical prompts yield identical token sequences;
  all logits finite; spot-check against the model card's expected continuations.
- **Capacity:** achieved `max-model-len`, KV blocks, per-GPU memory.
- **Throughput:** single-stream steady-state decode tok/s and aggregate tok/s
  at defined concurrency, plus an estimated memory-bandwidth-utilization (MBU).
- **Topology validity:** confirm from NCCL topology logs that TP4 all-reduces
  stay intra-clique. Only compare placements that keep each TP4 group in one
  clique with PP2 spanning cliques; a four-GPU run is not accepted as TP4×PP2
  evidence.

**Gate G5:** clique-aligned TP4×PP2 deployment records correctness, capacity,
and throughput against the documented GPU→clique mapping.

### Phase 6 — Disciplined fallback gate (conditional)

Engaged only on a G3/G5 correctness failure **or** a ratified performance-floor
breach (see Risks → Fallback threshold). Steps, in order:

1. **Classify** the failure with a **minimal reproducer** (smallest shape /
   fewest experts that reproduces it) and label it: `layout/scale-adapter`,
   `zero-point-math`, `capacity`, `topology/NCCL`, or `performance`.
2. If the class is adapter-local, spend a **bounded** debugging spike
   (≤ 2 iterations, changes confined to
   `csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu`) before escalating.
3. Only if that spike is exhausted, or the class is fundamental, write a
   **decision memo** that explicitly authorizes a **time-boxed** tc-grid /
   `kernels/v100/dense.cuh` investigation (proposed cap: 3 engineer-days,
   re-decision at the cap). The memo cites the reproducer and states the exact
   trigger. No unbounded kernel rewrite is ever authorized by this gate.

**Gate G6:** either the sprint completed without needing this gate, or a
classified reproducer + decision memo exists before any fallback code is touched.

---

## Files Summary

No production source changes are authored by this sprint beyond the two
cherry-picked PR #100 commits; the deliverables are the artifact and the
evidence ledger. Files touched or read:

| Path | Role in this sprint |
| --- | --- |
| *(git history)* PR #100 commits `9afb043…`, `054e2fd…` | Cherry-picked onto `v1.2.1` (Phase 1) |
| `CMakeLists.txt:102-116,447-467,1235-1257` | Arch envelope, `ENABLE_SM70_MARLIN`, MoE mutual-exclusivity gate (read/verify) |
| `csrc/torch_bindings.cpp:13,137-138` | `sm70_marlin_available()` proof (Phase 2) |
| `csrc/quantization/marlin/sm70_*.cu` | SM70 dense repack/GEMM (build + cubin proof) |
| `csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu:224,122,98,68,304-320` | `moe_wna16_marlin_gemm`, asymmetric zp adapters, fp16 checks (fixture + deploy) |
| `csrc/moe/torch_bindings.cpp:81` | MoE op binding (`torch.ops._moe_C`) |
| `.../compressed_tensors_moe/compressed_tensors_moe_wna16_marlin.py:56` | `CompressedTensorsWNA16MarlinMoEMethod` selection (Phase 4/5) |
| `.../schemes/compressed_tensors_wNa16.py:42,81-84,239` | Dense WNA16 scheme + SM70 gating |
| `vllm/_custom_ops.py:1233` | Python `sm70_marlin_available()` wrapper |
| `docs/design/hy3_sm70_marlin_validation.md` | **New** evidence ledger (SHAs, toolchain, cubins, scheme, topo, benchmarks) |
| `docs/design/sm70_v100_migration_control.md` | Prior evidence-convention reference (read only) |

---

## Definition of Done

1. **Branch integrity.** `hy3-sm70-marlin` is exactly the `v1.2.1` base
   (`4e9fdbc80…`) plus the two PR #100 commits, applied in order; the source
   SHAs (`9afb043…`, `054e2fd…`) and the resulting cherry-pick SHAs are both
   recorded in the ledger. Legacy branches untouched. *(G1)*
2. **Reproducible SM70 build.** A dedicated `TORCH_CUDA_ARCH_LIST=7.0` build
   completes from a full (non-precompiled) install, and import-time
   `sm70_marlin_available()` returns True. *(G2)*
3. **Artifact proof.** Cubin/symbol inspection shows SM70 dense repack and
   `moe_wna16_marlin_gemm` present, and **no** sm_75+-only Marlin-MoE cubin is
   shipped; the build log shows the SM70 MoE kernels built (not skipped). *(G2)*
4. **Asymmetric correctness.** The `group_size=32`, `symmetric=false` MoE GEMM
   is within tolerance and finite on a V100 fixture, with a passing symmetric
   control isolating the zero-point path. *(G3)*
5. **TP4×PP2 dummy load.** On the full eight-GPU allocation, the load selects
   `CompressedTensorsWNA16MarlinMoEMethod`, completes the asymmetric W4A16
   repack, meets the per-GPU capacity headroom, and passes a bounded
   first-token smoke with no kernel-image error or hidden capability
   override. *(G4)*
6. **Real deployment + topology.** A pinned-snapshot deployment on the
   clique-aligned TP4×PP2 mapping records correctness (deterministic repeatable
   tokens, finite logits), capacity, and throughput; TP4 all-reduces are
   confirmed intra-clique. *(G5)*
7. **Disciplined fallback (if reached).** Any asymmetric-WNA16 failure or
   ratified performance breach produces a minimal reproducer and a decision
   memo authorizing a time-boxed fallback investigation — not an unbounded
   rewrite. *(G6)*
8. **Ledger complete.** `hy3_sm70_marlin_validation.md` contains SHAs,
   toolchain/`collect_env`, build log, cubin listing, fixture results, selected
   scheme, `nvidia-smi topo -m`, GPU UUID→clique mapping, and benchmark numbers.

---

## Risks

| # | Risk | Likelihood / Impact | Mitigation |
| --- | --- | --- | --- |
| R1 | **Asymmetric zp math wrong on Volta.** The group-32, `symmetric=false` adapter mis-handles zero points, giving wrong logits. (Open Q #1) | Med / High | Phase-3 fixture with symmetric control *before* any model load; failure is localized in-file, not escalated blindly. Correctness uncertainty is **High** per intent. |
| R2 | **Mixed-arch wheel routes MoE GEMM to sm_75+ kernel** → CUDA launch failure on V100 (issue #87 symptom). | Med / High | Mandatory dedicated `TORCH_CUDA_ARCH_LIST=7.0`; Phase-2 cubin proof asserts no sm_75+ Marlin-MoE cubin and that SM70 MoE was built, not skipped. |
| R3 | **Wrong compute dtype.** Loading Hy3 in `bfloat16` trips the hard fp16 `TORCH_CHECK` (`sm70_marlin_moe_dispatch.cu:304-320`) at the first MoE GEMM. | Med / Med | Enforce and assert `--dtype float16` in Phases 4–5; documented as a first-class constraint. |
| R4 | **Capacity.** ~22.75 GB weights/GPU leaves ~9 GB for KV + activations + workspace + context; a too-large `max-model-len`/`max-num-seqs` OOMs. | Med / Med | Capacity is an explicit G4 gate with ≥ 2 GB/GPU headroom; tune `max-model-len`/`max-num-seqs`/`gpu-memory-utilization` from the dummy-load evidence. |
| R5 | **Topology mis-placement.** A TP4 group straddles two cliques, forcing tensor-parallel all-reduce across the slow inter-clique link (or a 4-GPU run is mistaken for TP4×PP2 proof). | Med / Med | Pin GPU order by UUID→clique; verify intra-clique all-reduce from NCCL logs; reject non-clique-aligned or 4-GPU results as evidence. |
| R6 | **Scope creep into kernel authoring.** tc-grid/`dense.cuh` engaged without discipline. | Low / High | Phase-6 gate requires a classified reproducer + decision memo + time box before any fallback code is touched. |
| R7 | **PR #100 conflict / UNSTABLE reliance.** Cherry-pick conflicts or a temptation to use merge state. | Low / Med | Pin SHAs, apply in order, stop and record on conflict; never depend on merge status. |

### Fallback threshold (resolves Open Question #4)

The tc-grid / `dense.cuh` investigation is authorized only by one of two
explicit, recorded triggers:

- **Correctness trigger:** the asymmetric W4A16 MoE path cannot be made
  numerically correct on sm_70 (Phase-3 fixture or Phase-5 deployment) after a
  bounded in-file adapter spike (≤ 2 iterations). This is a *classification*,
  backed by a minimal reproducer — not a guess.
- **Performance trigger (proposed defaults, ratify with operator):** functional
  correctness passes but steady-state performance is below the floor —
  **single-stream decode < 8 tok/s**, **or** aggregate **< 40 tok/s** at 8
  concurrent streams, **or** **MBU < 25 %**. These numbers are proposed for
  ratification (Open Q #4); the sprint records achieved values regardless, so
  the threshold can be set from evidence.

In both cases the trigger produces a decision memo with a hard time box
(proposed 3 engineer-days, re-decision at the cap). Absent a trigger, the
Marlin path stands and the fallback is not touched.

---

## Security

- **Supply-chain / model provenance.** Pin the Hy3 checkpoint to revision
  `c8b08e2c23dd45cb1b277d1290800e40c3dd8eec` and stage it as a local snapshot;
  verify the snapshot hash before the real-weight run. Do not execute or trust
  arbitrary checkpoint-side code; load only through vLLM's standard loader.
- **Build integrity.** All native code is the audited `v1.2.1` base plus the two
  pinned PR #100 SHAs — no out-of-band patches. The cubin listing in the ledger
  is the tamper-evident record of what actually shipped in the artifact.
- **Environment hygiene.** Python only via `uv` / `.venv/bin/python`
  (`AGENTS.md §2`); no system `python3`/`pip`. No secrets or checkpoint paths
  committed to the repo; the untracked `reference/` and `logs/` trees are
  user-owned and out of scope and are not read or modified.
- **Contribution policy.** This is internal fork validation. If it later yields
  an upstream PR, `AGENTS.md §1` applies: duplicate-work checks first, a human
  who understands and defends every changed line, recorded test commands and
  results, and explicit disclosure of AI assistance.

---

## Dependencies

- **Hardware:** eight Tesla V100 32 GB GPUs as two NVLink-capable four-GPU
  cliques with a stable inter-clique transport for PP2. (Open Q #3 — confirm the
  runtime is provisioned and the UUID→clique map is known.)
- **Toolchain (hard constraint):** CUDA in **`[12.8, 13.0)`**. CUDA ≥ 13.0 drops
  `7.0` from `CUDA_SUPPORTED_ARCHS` (`CMakeLists.txt:102-116`) and yields no
  SM70 code; the build host must match the published successful SM70 envelope.
  (Open Q #2.)
- **Build inputs:** PR #100 commits `9afb043…` and `054e2fd…` cleanly
  cherry-pickable onto `4e9fdbc80…`.
- **Python env:** `uv`, Python 3.12, `requirements/lint.txt`, `pre-commit`; a
  full (non-precompiled) editable/build install with `--torch-backend=auto`.
- **Model:** pre-staged local snapshot of `cyankiwi/Hy3-AWQ-INT4` at the pinned
  revision (~182 GB) reachable from the deployment host without a live pull.
- **Tooling:** `cuobjdump`/`nm` for cubin/symbol inspection; `nvidia-smi topo -m`
  and NCCL topology logging for the topology experiment.
- **Precedent:** `docs/design/sm70_v100_migration_control.md` as the evidence
  convention the new ledger follows.

---

## Open Questions

1. **Asymmetric zp consumption (correctness).** Does
   `sm70_marlin_moe_dispatch.cu` correctly consume Hy3's `group_size=32`,
   `symmetric=false` zero points as-is, or does it need a layout/scale
   conversion? → **Retired in Phase 3** by the fixture + symmetric control
   before any expensive load.
2. **Build envelope.** Exact target-host CUDA/PyTorch/driver? Confirm it lands
   in `[12.8, 13.0)` and matches the published successful SM70 build. → Gate G0.
3. **Cluster provisioning.** Is the two-clique TP4×PP2 runtime provisioned with
   a stable inter-clique transport and a known GPU UUID→clique map? → Confirmed
   in Phase 5 setup; blocks G5 if unknown.
4. **Performance floor.** What throughput/latency justifies the tc-grid
   investigation? → **Proposed defaults** in Risks → Fallback threshold
   (< 8 tok/s single-stream, < 40 tok/s @8, or MBU < 25 %); ratify with the
   operator once Phase-5 numbers exist.
5. **PR #100 semantics (to confirm during audit).** Given the SM70 dense+MoE
   kernels already exist at `v1.2.1`, exactly what do the two PR #100 commits
   change — operator-registration dedup, CMake gating, or the asymmetric
   adapter? The audit records the actual diff so the artifact proof targets the
   right surface.
```