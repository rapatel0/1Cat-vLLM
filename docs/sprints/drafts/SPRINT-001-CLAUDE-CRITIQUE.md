# Sprint 001 Critique: Validate Hy3 asymmetric W4A16 Marlin on V100

Reviewer: Claude (Opus 4.8)
Inputs: `SPRINT-001-INTENT.md`, `SPRINT-001-CODEX-DRAFT.md`, `SPRINT-001-ANTIGRAVITY-DRAFT.md`
Method: source-verified against the branch at HEAD. No source was modified; `reference/` and `logs/` were not touched.

---

## 1. The source-backed point, resolved

The intent's central technical claim is **correct, and materially deeper than either draft
represents.** I verified every link in the chain.

### 1.1 The blocker is real, and it is not a single assertion

`CompressedTensorsWNA16MarlinMoEMethod.__init__` asserts symmetry:

```
assert weight_quant.symmetric, "Only symmetric quantization is supported for MoE"
```
— `compressed_tensors_moe_wna16_marlin.py:67-69`

Hy3 is `symmetric=false`, so it is rejected before any CUDA dispatch. But the assertion is only
the **first of five** places the symmetric assumption is baked into the Python method. A correct
bridge must address all five:

1. **`__init__:67-69`** — the hard assertion (the visible blocker).
2. **`create_weights:161-322`** — registers `w13_weight_packed`, `w2_weight_packed`, scales,
   `g_idx`, and sort indices, but **registers no zero-point parameter at all.** There is no
   `w13_weight_zero_point` / `w2_weight_zero_point` for the checkpoint loader to populate. Even
   past the assert, Hy3's `weight_zero_point` tensors have nowhere to land.
3. **`process_weights_after_loading:324-459`** — repacks weights (`gptq_marlin_moe_repack`) and
   permutes scales (`marlin_moe_permute_scales`); does nothing with zero points.
4. **`get_fused_moe_quant_config:461-472`** — hardcodes `w1_zp=None, w2_zp=None`. The builder
   `int4_w4a16_moe_quant_config` already accepts both (`config.py:870-885`), so this is a
   two-line omission, but it is load-bearing: the modular path derives its quant type from zp
   presence (see below).
5. **`self.quant_type = WNA16_SUPPORTED_TYPES_MAP[num_bits]`** resolves to
   `scalar_types.uint4b8` (`compressed_tensors_wNa16.py:37`). The asymmetric type
   `scalar_types.uint4` lives in a **separate** map, `WNA16_ZP_SUPPORTED_TYPES_MAP`
   (`compressed_tensors_wNa16.py:38`), which this MoE method never consults.

Neither draft enumerates points 2–5. Codex's Phase 3 gestures at all of them collectively
("register zero-point parameters… construct FusedMoEQuantConfig with w1_zp/w2_zp"); Antigravity
does not mention any of them and its file list omits the loader-adjacent files entirely.

### 1.2 The SM70 C++ side is ready, but its contract is specific

The intent is right that the SM70 dispatch already has zero-point adapters. The contract in
`csrc/moe/marlin_moe_wna16/sm70_marlin_moe_dispatch.cu` is precise, and **neither draft states any
of the three properties that actually make a bridge correct**:

- **Type coupling.** For `b_type == kU4`, the kernel *requires* `has_zp && zp_is_float`
  (`:570-572`). For `kU4B8` it *rejects* zeros (`:588-591`). So asymmetric Hy3 must dispatch as
  `uint4` **and** carry zeros — the two are inseparable. The modular experts path already encodes
  exactly this coupling: `MarlinExpertsBase.quant_type_id` returns `uint4` iff `w1_zp`/`w2_zp` is
  non-None, else `uint4b8` (`marlin_moe.py:637-642`). This is the strongest argument that the
  correct fix is "make the quant config carry zeros" and let the type follow — a point Codex
  approaches but does not name.
- **Zero-point *semantic*.** The logical zero point the kernel consumes is `zp_int * scale`, not
  the raw integer: `out[idx] = __float2half(zp * scales[idx])` (`:95`). Any Python-side
  pre-conversion to fp16 must pre-multiply by scale. Neither draft mentions this.
- **Zero-point *layout*.** Two encodings are accepted (`:497-522`): (a) fp16 already-scaled
  logical zeros, rank-3 `[num_experts, num_groups, size_n]`, `is_zp_float=true` (`:126-127`,
  `:528-540`); or (b) packed `int32` that the device kernel `sm70_moe_zp_as_half` unpacks
  (`:122-160`). Path (b) is **not** raw checkpoint layout — the unpack kernel applies
  `sm70_inverse_scale_perm` and a 4-bit nibble interleave `{0,4,1,5,2,6,3,7}`
  (`sm70_inverse_zp_interleave`, `:36-89`), i.e. it expects zeros already in the **same
  Marlin-permuted N-order that `marlin_moe_permute_scales` produces for scales**, plus the AWQ
  nibble interleave. The existing `awq_marlin_moe_repack` (`_custom_ops.py:1327`) is the natural
  tool to produce this and should be named in the plan.

The consequence: "repack zero points into rank 3, fp16, `[E, num_groups, size_n]`" (Codex Phase 3)
is directionally correct but under-specifies the single riskiest operation in the sprint. The
choice between encoding (a) and (b), the `zp*scale` semantic, and the permutation/interleave
requirement are exactly where a Type-A layout bug will hide and silently produce plausible-looking
garbage.

### 1.3 A routing fork neither draft addresses

Whether Hy3 reaches the Marlin method at all depends on `check_moe_marlin_supports_layer`
(`marlin_utils.py:233`), which requires `hidden_size % 128 == 0`,
`intermediate_size_per_partition % max(64, group_size) == 0`, and
`group_size in [-1, 32, 64, 128]`. Notably there is **no device-capability gate** here — group_size
32 is allowed — so on a V100 with the SM70 build the layer routes to Marlin *iff its post-TP4
shapes divide cleanly*. If `intermediate_size_per_partition % 64 != 0` after TP4, Hy3 silently
falls back to `CompressedTensorsWNA16MoEMethod` — which **also asserts `symmetric`**
(`compressed_tensors_moe_wna16.py:50-52`) and **also hardcodes `w1_zp=None, w2_zp=None`**
(`:212-213`), and dispatches a *different, non-Marlin* kernel (`moe_wna16_gemm`) whose SM70
asymmetric support is unproven.

This means the plan should prove, as an early static gate, (a) that Hy3's TP4-partitioned expert
dims satisfy `check_moe_marlin_supports_layer`, and (b) that routing actually lands on the Marlin
method for *this layer*, not merely that `moe_wna16_marlin_gemm` is registered in the binary.
Codex's Open Question 2 circles this; neither draft makes it a gating task, and neither notes that
the fallback method shares the same symmetric blocker.

### 1.4 Verdict on the resolved point

**Does the plan correctly scope a loader/packing bridge and reference tests before a model load?**

- **Codex: yes, structurally — with real depth gaps.** It sequences a dedicated loader phase
  (Phase 3) → a bounded reference/loader test phase (Phase 4) → dummy load (Phase 5) → real
  checkpoint (Phase 6). Phase 4 explicitly compares the SM70 Marlin MoE result against the torch
  reference with a predeclared tolerance *and* asserts the loader carries `w1_zp`/`w2_zp` non-None
  and resolves `uint4` not `uint4b8`, both **before** any Hy3 load. This is the correct
  test-before-model-load discipline and directly satisfies the intent's Success Criterion 4 and
  its verification note. What it lacks is the *content* of the bridge (§1.2) and the routing gate
  (§1.3).
- **Antigravity: no.** It has no loader-code phase and no reference-test phase. Implementation
  Step 5 jumps from a compiled artifact straight to a TP4×PP2 dummy model load "to test
  `CompressedTensorsWNA16MarlinMoEMethod` selection and asymmetric W4A16 repack," and DoD item 3
  makes that model load the *primary* proof of asymmetric correctness. That inverts the intent's
  explicit ordering ("First prove the loader carries zero points… then compare the Marlin result
  with the existing torch reference"). It never acknowledges the assertion, the missing zp
  parameters, or the C++ contract. As written, this plan hits `:67-69` on first load with no test
  scaffold to diagnose it.

---

## 2. Codex draft

### 2.1 Strengths

- **Evidence-first gating is genuinely well designed.** Build provenance (SHAs, `collect_env`,
  toolchain), an import-time proof (`sm70_marlin_available()`, repack ops, `moe_wna16_marlin_gemm`),
  and a *binary* inspection step (`cuobjdump`/`nvdisasm`) before any model load. The insistence
  that a mixed-arch wheel is *invalid evidence* (not merely suboptimal) is correct and matches the
  CMake mutual-exclusivity the intent describes.
- **Correct framing of the blocker** as "a known blocker to validate, not an acceptable product
  limitation," and correct identification that symmetric→`uint4b8` / asymmetric→`uint4`, with
  act-order remaining rejected for SM70 (matches `:560-561`).
- **Fallback is operationalized.** Quantified thresholds (decode < 70% of best correct non-tc-grid
  fallback; startup > 30 min after one tuning pass; scope wider than a bounded adapter) each gated
  behind a decision memo. This is exactly what the intent's Criterion 7 asks for.
- **Correct file surface.** It reaches `config.py`, `marlin_moe.py`, and `_custom_ops.py` — the
  three files that actually carry zeros into the op — plus both test files.
- **Security is concrete:** the 182 GB silent-download trap, secret/hostname redaction, evidence
  kept out of git, UUIDs over infra identifiers.
- **Open questions are sharp and mostly the right ones** (expert dims after TP4 vs group-32
  alignment; zp checkpoint-name remapping; dummy config source).

### 2.2 Weaknesses / risk gaps

- **[High] The packing bridge is under-specified — the highest-risk code treated as mechanical.**
  Phase 3 does not choose between the fp16-prescaled path and the packed-int32 device path, does
  not state the `zp*scale` semantic (`:95`), does not state the Marlin-permutation + nibble-
  interleave requirement (`:36-89`), and does not name `awq_marlin_moe_repack` as the likely tool.
  These are the exact spots a silent Type-A error hides. The plan hedges correctly ("prove the
  kernel consumes it"), so this is a depth gap, not a wrong direction — but it should be closed
  before implementation starts.
- **[High] The monolithic `apply()` path is a latent symmetric fallback and is unaddressed.**
  `CompressedTensorsWNA16MarlinMoEMethod.apply` (`:542-576`) calls `fused_marlin_moe(...)` with
  `quant_type_id=self.quant_type.id` (= `uint4b8`) and passes **no** `w1_zeros`/`w2_zeros`
  (`fused_marlin_moe` defaults them to None, `marlin_moe.py:251-252`). The two `None`s it does
  pass are `bias1, bias2` (`marlin_moe.py:228-229`), not zeros. If this path ever executes for the
  Marlin backend, it computes a symmetric result and **silently drops the asymmetric offset** — no
  crash, wrong numbers. `is_monolithic` is False for Marlin (`:512-514`), so the modular
  `select_gemm_impl`→`MarlinExperts` path is expected to run and `apply()` is likely dead here —
  but the plan should *confirm* which path executes and either fix `apply()` or guard it, and the
  reference test must be constructed to catch a symmetric-fallback regression. This is exactly the
  failure a "loader carries zeros" assertion alone will not catch.
- **[Medium] Routing to the Marlin method is assumed, not proven (§1.3).** Criterion/Phase-2
  proves the *op* exists in the binary; nothing proves *this layer* routes to it rather than to the
  equally-symmetric non-Marlin fallback. Add an early static check of Hy3's TP4 expert dims
  against `check_moe_marlin_supports_layer`.
- **[Medium] Group-32 / Marlin alignment is parked as an open question, not gated.** The kernel
  enforces `size_n % min_thread_n == 0`, `size_k % tile_size == 0`, and scale numel divisible by
  the perm length (32 or 64) (`:340-354`, `:546-548`, `:106-108`). If Hy3's dims fail these after
  TP4, the whole zp effort is moot. This belongs as a Phase-2.5 static gate, ahead of loader work.
- **[Medium] The reference test needs a V100 *early*, not just at deployment.** `MarlinExperts`
  gates on `has_device_capability((7,0))` (`marlin_moe.py:586-589`) and `moe_wna16_marlin_gemm` is
  CUDA-only; the "reduced Hy3-shaped" reference test cannot run on CPU/CI. The plan's
  "before a model load" sequencing is sound *only if* a V100 is available at Phase 4. Surface this
  as a scheduling dependency, not just a "label skipped GPU tests" footnote.
- **[Low] PP2 transport ambiguity.** "Eight V100" blurs single-host-8-GPU vs two hosts. PP2 across
  two cliques on separate hosts needs a working multi-node runtime (Ray / network transport), not
  just NVLink. Codex's Open Question 1 touches versions but not this topology assumption.

### 2.3 Definition-of-Done completeness

Strong: maps ~1:1 to the intent's 7 success criteria (provenance, single-arch build, import proof,
binary evidence, loader reaches `uint4`+non-None zeros, kernel tests pass with tolerances, TP4×PP2
dummy, real checkpoint, fallback-threshold-cited, tests/lint recorded). Gaps to close:

- No DoD line asserting the **monolithic `apply()` path is fixed or proven unused** (§2.2).
- No DoD line for the **group-32 / `check_moe_marlin_supports_layer` static gate** and the routing
  confirmation (§1.3).
- The reference-comparison DoD line should name the **`zp*scale` semantic and the permuted/
  interleaved layout** as acceptance conditions, so "tests pass" cannot be satisfied by an
  accidentally-symmetric fixture.

Net: Codex is a strong, executable plan whose weaknesses are all *depth* (the bridge internals) and
*sequencing gates* (routing/alignment before loader), not direction.

---

## 3. Antigravity draft

### 3.1 Strengths

- Captures the correct high-level frame and the non-negotiables: exact base + PR #100 SHAs in
  order, single-arch SM70 build, `uv`-only tooling, TP4×PP2 topology, tc-grid/`dense.cuh` held in
  reserve behind a decision memo.
- Concise and readable; the top-level risk list names the right three themes (asymmetric-zp
  correctness on Volta, topology OOM / inefficiency, scope creep).

### 3.2 Weaknesses / risk gaps

- **[Critical] No loader/code-change phase and no reference-test phase.** It never acknowledges the
  `:67-69` assertion, the missing zp parameters, the hardcoded `w1_zp=None/w2_zp=None`, the
  `uint4` vs `uint4b8` distinction, or that `compressed_tensors_moe_wna16_marlin.py` must be
  edited at all. Its file list omits `compressed_tensors_moe.py`, `config.py`, and every test
  file. The plan as written cannot pass its own DoD.
- **[Critical] Correctness is validated by a model load, not a bounded reference.** DoD item 3
  makes a TP4×PP2 dummy load the proof of method selection + repack. There is no torch-reference
  comparison anywhere, no numerical tolerance, and no Type-A/Type-B error classification. This
  directly contradicts the intent's verification strategy.
- **[High] The C++ zero-point contract — the task's central point — is absent.** No mention of
  `uint4`/`is_zp_float`, the `zp*scale` semantic, rank-3 layout, or the permutation/interleave.
  The one gesture ("layout/scale conversion" under Risks) has no corresponding plan to produce it.
- **[High] Reproducibility gaps.** It omits the pinned Hy3 revision `c8b08e2c…` that the intent
  fixes; it has no provenance-capture list (`collect_env`, CUDA/driver/PyTorch/NCCL versions); and
  it has no binary/cubin inspection step. Success is asserted rather than evidenced.
- **[Medium] Fallback is not operationalized.** "If the asymmetric path fails → decision memo,"
  with no functional-vs-performance threshold, no 70%/30-min criteria, no scope bound — the exact
  discipline the intent's Criterion 7 and Open Question 4 demand.
- **[Medium] Topology proof is thin.** "Validated clique-aligned mapping" with no
  `nvidia-smi topo -m` / UUID / NCCL capture and no explicit rejection of a 4-GPU substitute as
  evidence (which the intent forbids).
- **[Medium] Dummy vs real separation is under-guarded.** It mentions both but does not forbid the
  182 GB silent download in the dummy step with the intent's firmness.

### 3.3 Definition-of-Done completeness

Materially incomplete against the intent. Its 5 checkboxes cover provenance, artifact proof,
(load-based) selection, topology, and fallback — but every substantive gate is load-gated, so there
are **no cheap early gates** and no partial-credit evidence path if load fails. Missing entirely:
the loader-boundary proof (non-None zeros, `uint4` not `uint4b8`), the torch-reference numerical
comparison, binary cubin evidence, the pinned Hy3 revision, and a tests/lint-recorded gate.

Net: a correct executive summary, not an executable plan. It would need Codex's Phases 3–4 grafted
in wholesale to be safe to run.

---

## 4. Comparative verdict and recommendation

**Adopt the Codex draft as the base.** It is the only one of the two that satisfies the intent's
core discipline — a bounded loader/packing bridge with reference *and* loader tests before any Hy3
model load — and its provenance/build/security rigor is production-grade. Antigravity is a useful
one-page framing but is not safe to execute: it would hit the symmetric assertion on first load
with no diagnostic scaffold.

Before executing Codex, close these gaps (ordered):

1. **Add a Phase-2.5 static gate** proving Hy3's TP4-partitioned expert dims satisfy
   `check_moe_marlin_supports_layer` (`marlin_utils.py:233`) and the kernel alignment checks
   (`:340-354`, `:546-548`), and confirming the layer routes to the *Marlin* method — not the
   equally-symmetric non-Marlin fallback (`compressed_tensors_moe_wna16.py:50-52`). Cheap, and it
   de-risks the entire rest of the sprint. (§1.3)
2. **Specify the packing bridge concretely** in Phase 3: pick the encoding (recommend producing
   fp16 `zp*scale` logical zeros via the existing scale-permute + `awq_marlin_moe_repack` path so
   `is_zp_float=true`, avoiding the packed-int32 interleave subtleties), and state the `zp*scale`
   semantic and permuted-N layout as explicit acceptance conditions. (§1.2)
3. **Resolve the `apply()` vs modular path** and make the reference test able to catch a
   symmetric-fallback regression (a fixture whose asymmetric offset is non-zero enough that a
   dropped zp fails the tolerance). Fix or guard `apply()` (`:542-576`). (§2.2)
4. **Add the two missing DoD lines** (monolithic-path status; group-32/routing gate) and enrich the
   reference-comparison DoD line with the `zp*scale` + layout acceptance conditions.

### Edge cases both plans miss

- **Silent symmetric fallback** through `apply()` (uint4b8 + no zeros) — passes the kernel's
  `kU4B8` checks and returns wrong-but-finite output. The reference test must be adversarial to it.
- **`zp * scale` vs raw-integer zp** — a bridge that forwards raw integer zeros as fp16, or forgets
  the scale multiply, is off by exactly the scale factor per group; may still look plausible.
- **Marlin scale-permutation on zeros** — un-permuted zeros pass all shape checks (`[E, ng, n]`)
  yet index the wrong columns; classic Type-A.
- **Routing to the non-Marlin `moe_wna16_gemm`** on a dim misalignment — a *different* symmetric
  assertion, easy to misattribute to the Marlin path.
- **`num_groups` from group_size 32** must satisfy `size_k % num_groups == 0` (`:448-456`) *and*
  the scale-numel-divisible-by-perm-len check (`:106-108`) simultaneously; TP4 sharding can break
  one without the other.
- **Two-host PP2** — if the cliques are on separate hosts, this is a multi-node bring-up, not a
  placement choice.
