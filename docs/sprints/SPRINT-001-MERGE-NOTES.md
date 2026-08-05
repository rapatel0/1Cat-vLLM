# Sprint 001 Merge Notes

## Inputs merged

- Intent: `docs/sprints/drafts/SPRINT-001-INTENT.md`
- Drafts: Codex, Claude, and Antigravity
- Cross-critiques: Codex, Claude, and Antigravity

## Decisions ratified by the operator

- Include the bounded asymmetric MoE loader/zero-point bridge.
- Make functional correctness and TP4xPP2 dummy dispatch the sprint exit.
- Permit narrowly-scoped C++ changes when the existing op cannot represent the
  proven layout; require a focused regression test.
- Keep deployment planning target-neutral behind a topology preflight.
- Keep tc-grid and dense.cuh strictly deferred until a classified functional
  incompatibility or later ratified performance threshold.

## Changes from the draft baseline

- Added the early post-TP4 Marlin routing/alignment gate.
- Made zero-point semantic (`zp * scale`) and Marlin ordering testable
  acceptance conditions.
- Required a guard or proof for the monolithic symmetric-fallback path.
- Moved real-weight and performance work out of Sprint 001's exit criteria.
