# Iron Increment: Fork main upstream mirror

## Sealed contract

### Outcome

Preserve the fork’s pre-sync deployed source on `deployed`, then make `rapatel0/1Cat-vLLM` main exactly match current `1CatAI/1Cat-vLLM` main.

### Deliverable

A pushed `deployed` branch at the former fork tip and a lease-protected force update of fork main to current upstream main.

### Delivery floor

- `deployed` retains `94f647142` and its block-scaled INT8 commit.
- Fork main equals upstream main at the update point.
- No unrelated ref is changed.

### Results that do not count

- Dropping the deployed branch, force-pushing without a lease, or leaving main on a merge commit that diverges from upstream.

### Acceptance evidence

- Git ancestry proves `6833b8529` is retained on `deployed`.
- Fork and upstream main resolve to the same commit after push.
- The clean clone contains no uncommitted changes.

### Preserve

- `deployed`, all existing non-main branches, cluster deployment state, and unrelated worktrees.

### Non-goals

- Rebuilding/deploying code, rewriting upstream, or merging/deleting other branches.

### Material effects

- Force-updates public fork main under the user-confirmed lease-protected reset.

### Autonomous decisions

- Fetch upstream, use the current upstream main SHA, and verify refs before/after the push.

### User authority

Confirmed: reset fork main to upstream and force-push with lease after `deployed` preserves the fork state.

## Confirmed amendments

None.

## Current orchestration

Main mirror operation: validate `deployed`, fetch upstream, reset only the clean clone main ref, force-push only with a lease for current fork main, and verify ref equality.

## Current checkpoint

None.
