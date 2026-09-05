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
