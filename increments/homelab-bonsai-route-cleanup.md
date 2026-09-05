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

Pending controller alignment.

## Current checkpoint

None.
