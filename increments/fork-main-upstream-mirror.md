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

## Completion

### Delivered artifact

`origin/deployed` preserves the fork source at `94f64714263e3af4399bbc0dd78069e5ce3a84b9`. It contains block-scaled INT8 commit `6833b8529`.

`origin/main` now exactly equals `upstream/main` at `24ff99d157bd8fc34755fd5d44849ab434010473`.

### Acceptance evidence and criterion status

- Full — `origin/deployed` resolved to `94f647142` before and after the reset.
- Full — `6833b8529` is an ancestor of `origin/deployed`.
- Full — Fork main and upstream main resolved to the same SHA after the lease-protected force push.
- Full — The clean clone was clean before reset and remained clean afterward.

### Preserved behavior evidence

Only fork main changed. The deployed branch and cluster state were not changed.

### Reproduction or run commands

- `git ls-remote origin refs/heads/main refs/heads/deployed`
- `git merge-base --is-ancestor 6833b8529 origin/deployed`

### Material residuals

The deployed branch is a source-history checkpoint. The live overlay has no `.git` directory, so it cannot prove byte-for-byte identity with this commit.

### Unverified behavior

No source build or cluster deployment occurred.
