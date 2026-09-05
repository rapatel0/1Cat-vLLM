# Iron Increment: Bonsai route cleanup and dirty-worktree audit

## Sealed contract

### Outcome

Remove the stale public Bonsai route and unused Hermes Bonsai provider, while preserving Bonsai storage and source history. Inspect the remaining dirty files in the original homelab worktree and report their purpose, source status, and safe next action without altering them.

### Deliverable

A pushed update to `feat/fp8-default-4090-gaming`, matching live routing cleanup, and a concise classified inventory of dirty original-worktree files.

### Delivery floor

- Public LiteLLM no longer advertises `bonsai-27b`.
- Live Hermes configuration has no `bonsai-27b` provider or route dependency.
- `qwen38-flash-next` public serving remains healthy.
- Bonsai Deployment, Service, ConfigMaps, PVCs, models, Jobs, and benchmark/history content remain untouched.
- The dirty-worktree inventory distinguishes direct active-FP8 material, experiments, reports, agent state, and unknown material.

### Results that do not count

- Removing a manifest route without removing the live route.
- Removing Bonsai storage or history.
- Resetting, staging, committing, moving, or deleting pre-existing dirty files during the audit.
- Declaring a dirty file obsolete based only on its name.

### Acceptance evidence

- Gateway model list and public Flash request succeed after the update.
- Live Hermes and LiteLLM config text contains no active Bonsai route/provider.
- Source parses, passes a server-side ConfigMap dry run, and contains no active Bonsai serving route/provider.
- The feature branch is pushed.
- The audit reports each dirty-path class and any dynamic/reference uncertainty.

### Preserve

- FP8 EP8 public route and all current runtime constraints.
- Bonsai data, resources, and historical evidence.
- Original dirty worktree contents and staging state.

### Non-goals

- Deleting Bonsai workloads/data, merging the feature branch, modifying the original dirty worktree, or broad dormant-profile cleanup.

### Material effects

- Public `bonsai-27b` requests stop routing after the gateway restart.
- Hermes loses an unused provider declaration.

### Autonomous decisions

- Use a targeted live config update, choose validation, and classify the dirty paths from Git/reference evidence.

### User authority

- Any change to an original dirty file, Bonsai resource/storage deletion, broad cleanup, feature-branch merge, reset, or force-push.

## Confirmed amendments

None.

## Current orchestration

### Lead lens and contract fields

Operations is the lead lens. The target state contains only the FP8 public route and provider. The known-good state is the current FP8 public route, which returned HTTP 200. The Bonsai route is stale because its model Deployment is zero-scale and its Hermes provider has no remaining consumer. The allowed effect is targeted removal of the live and desired-state route/provider. A public FP8 request or missing FP8 route triggers restoration from saved config. The dirty original worktree remains read-only.

### Active constraint lenses and required checks

Production readiness requires a pre-change config snapshot, ConfigMap API dry run, rollout checks, and public FP8 route validation. Cleanup requires direct checks that no active Bonsai route/provider remains and requires retention of Bonsai resources/history.

### Cadence and evidence policy

Execute → validate. Use the clean feature branch for source changes and do not modify the original worktree.

### Validation requirements

Check live LiteLLM models, public FP8 request, live Hermes config, source YAML parsing, ConfigMap server dry run, no active source route/provider references, and pushed feature-branch status. The dirty-file audit needs path/status/reference evidence only.

### Current approach and material invalidated approaches

Remove the exact Bonsai `model_list` item in LiteLLM and the exact Hermes provider item. Do not delete the zero-scale Bonsai workload, service, storage, ConfigMaps, jobs, or history. Remove the stale README topology entry. Do not stage, reset, or change original worktree files.

### Open material defects and repair evidence

None.

### Explicit assumptions and non-blocking unknowns

External clients that explicitly request `bonsai-27b` will receive an unsupported-model response after cleanup. No current Hermes path uses it.

## Current checkpoint

None.

## Completion

### Delivered artifact

The stale `bonsai-27b` public LiteLLM route and unused Hermes provider were removed from the live cluster and from `feat/fp8-default-4090-gaming`. Source commit `09216a0` is pushed after the FP8 consolidation commits.

A read-only audit classified the original dirty homelab worktree without changing its contents or staging state.

### Acceptance evidence and criterion status

- Full — Public `https://llm.tail7cd5.ts.net/v1/models` listed only `qwen38-flash-next`; a public FP8 completion returned HTTP 200 with `FP8-CLEAN` and the FP8 EP8 fingerprint.
- Full — Live Hermes configuration and LiteLLM payload contain no `bonsai-27b`; the FP8 Deployment is ready 1/1.
- Full — Clean source Hermes/gateway manifests contain no active Bonsai provider or model route. YAML parsing, ConfigMap server dry run, diff check, and LSP diagnostics passed.
- Full — Independent review reported no blocker and confirmed the clean source branch matches pushed `09216a0`.
- Full — The original worktree stayed read-only during audit.

### Preserved behavior evidence

Bonsai Deployment/Service, manifests, model storage, ConfigMaps, Jobs, benchmarks, and history were not changed. The public FP8 route remains available.

### Reproduction or run commands

- `curl -k https://llm.tail7cd5.ts.net/v1/models`
- `git -C /Users/ravi/repos/homelab-fp8-consolidation log --oneline -3`
- `git -C /Users/ravi/repos/homelab status --short`

### Material residuals

The original worktree retains a mixed stale staged index, DCP experiments, and research notes. It must not be blindly committed or applied.

### Unverified behavior

The repository baseline cannot prevent future Hermes PVC runtime drift; current live configuration was verified.

### Dirty-worktree audit

- Already represented on the clean branch: the active FP8 `sglang-v100-dflash2` Deployment, Service, and README. Their staged index contains older NVFP4/SGLang content, so retain the clean worktree version and do not commit the index.
- Active FP8 candidates not represented: `18-tp4-pp2-fp8-130.yaml` and `19-tp8-fp8-130.yaml`. They are isolated eight-GPU candidate manifests with no live deployment/gateway route; retain pending separate qualification.
- DCP experiments: the attention patch, DCP2 manifest, test, patch README, and old SGLang staging Job. Decode-LSE correctness remains incomplete; do not deploy or merge.
- Research documentation: DCP investigation/implementation reports and parked integration plans. Retain as notes and reconcile claims before publication.
- Stale alternate routing: original dirty Hermes/gateway changes would reintroduce the retired EXL3 4090 route. Do not merge them. Live routing no longer has that EXL route.
- `.pi/` consists of local agent state and must not be committed.
