# Iron Increment: 1Cat-vLLM upstream sync

## Sealed contract

### Outcome

Update the `rapatel0/1Cat-vLLM` fork main branch from `1CatAI/1Cat-vLLM` main while preserving the fork-only commits and resolving merge conflicts with evidence.

### Deliverable

A clean clone-based merge result on the fork main branch, pushed to `origin/main`, with a concise conflict and validation report.

### Delivery floor

- Fork `main` contains current upstream `main` plus its two fork-only commits.
- No unresolved conflict markers exist.
- The merge preserves the fork’s intended custom work.
- The result is pushed without force-push.

### Results that do not count

- Resetting fork main to upstream and losing fork commits.
- Force-pushing, merging unreviewed conflicts, or changing a live deployment.
- Reporting a fast-forward when the divergent histories require a merge or rebase.

### Acceptance evidence

- Git ancestry proves that both the pre-merge fork main and upstream main are ancestors of the final fork main.
- Conflict files receive focused source/build validation.
- The clean clone has no uncommitted changes or conflict markers after the merge.
- `origin/main` points to the final merge commit.

### Preserve

- Fork-only commits and public source history.
- Existing operational deployments and separate local worktrees.
- Remote branch protection and normal Git history.

### Non-goals

- Rebuilding/deploying the cluster image, changing runtime configuration, rewriting upstream history, or merging unrelated feature branches.

### Material effects

- Pushes a merge commit to the public fork main branch.

### Autonomous decisions

- Use a fresh clone, choose merge rather than unsafe fast-forward, resolve mechanical conflicts, run focused validation, and create a normal merge commit.

### User authority

- Force-push, discarding fork commits, changing the target branch, accepting a conflict that changes intended fork behavior without evidence, or deployment/rebuild work.

## Confirmed amendments

None.

## Current orchestration

### Lead lens and contract fields

Migration is the lead lens. The old state is fork main `94f64714263e3af4399bbc0dd78069e5ce3a84b9`, two commits ahead and 246 commits behind upstream main. The target is a normal merge with upstream main `24ff99d157bd8fc34755fd5d44849ab434010473` as an ancestor and fork work retained. Mapping invariants are both parent ancestries, no unmerged files, and no public-history rewrite. The consumer is the fork main branch; no deployment consumer is part of this increment. Rollback is `git merge --abort` before commit or normal revert after push.

### Active constraint lenses and required checks

Compatibility requires focused inspection of every conflict and preservation of fork-only INT8 KV cache behavior. Production readiness applies only to the public Git push: use a clean clone, no force push, verify ancestry, and confirm remote ref.

### Cadence and evidence policy

Execute → validate. Merge upstream main with `--no-commit`; resolve only actual conflicts after inspecting both sides; validate affected source and run focused checks before a normal merge commit.

### Validation requirements

Record fork/upstream starting commits and divergence. Verify no conflict markers or unmerged index entries. Verify both starting commits are ancestors of the result. Run the nearest available focused checks for conflict files and the repository's relevant source checks. Confirm pushed `origin/main` equals the merge commit.

### Current approach and material invalidated approaches

Use `/Users/ravi/repos/1cat-vllm-upstream-sync`, a fresh clone. The fork-only commits are `6833b8529` and its merge commit `94f647142`. A direct fast-forward is invalid because the fork diverges; use a normal merge and retain both histories.

### Open material defects and repair evidence

None before the merge.

### Explicit assumptions and non-blocking unknowns

Upstream main is the intended source at fetch time. A large upstream delta can create source conflicts or broad test cost; focused checks will cover the resolved surface and any unrun scope will be stated.

## Current checkpoint

None.
