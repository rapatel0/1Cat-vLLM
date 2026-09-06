# Iron Increment: Port deployed INT8 KV cache to upstream

## Sealed contract

### Outcome

Merge current upstream-aligned `main` into fork `deployed`, retain and repair block-scaled INT8 KV cache behavior, build an image from that branch, and deploy it through the homelab manifest.

### Deliverable

A pushed `deployed` branch with upstream history and INT8 behavior, an immutable V100 image tag, and a committed homelab manifest that deploys the tag.

### Delivery floor

- `main` remains an exact upstream mirror.
- `deployed` incorporates current main and retains `int8_block32` behavior.
- The deployed manifest references the built tag.
- The rollout reaches ready state and passes a representative model request.

### Results that do not count

- Building upstream main without the INT8 patch, changing main, using an uncommitted deployment image, or promoting a tag without manifest changes.

### Acceptance evidence

- Git ancestry and focused INT8 tests show the port survives the merge.
- Build logs show a successful immutable image push.
- Git evidence shows a committed manifest tag update.
- Deployment readiness, server logs, and an OpenAI-compatible model request succeed.

### Preserve

- Fork `main` as upstream mirror, the old live image until readiness passes, existing public model identity, and unrelated homelab worktree changes.

### Non-goals

- Changing model architecture, public route identity, request defaults, or deploying upstream main without the custom patch.

### Material effects

- Pushes `deployed`, pushes a registry image, updates a committed production manifest, and rolls the `llm` deployment.

### Autonomous decisions

- Resolve concrete source conflicts, select a monotonically newer immutable image tag, and use the existing model deployment path.

### User authority

Confirmed: merge main into deployed, adapt the block-scaled INT8 changes, build the image, deploy to the cluster, and update the homelab configuration.

## Confirmed amendments

None.

## Current orchestration

### Lead lens and contract fields

Migration is the lead lens. Main at `24ff99d15` is the old upstream state, and deployed at `94f647142` is the customized state. The target is deployed plus upstream history, with main unchanged. Mapping invariants are both parent ancestries, `int8_block32` routes/validation retained, and immutable deployment tag consistency.

### Active constraint lenses and required checks

Compatibility requires the existing public OpenAI model identity and GPU V100 support. Production readiness requires a manifest-backed rollout and rollback path. Performance requires recording a representative warm model request metric only after correctness passes; no unverified performance claim is accepted.

### Cadence and evidence policy

Merge → inspect → focused source validation → build → manifest update → rollout → readiness/request check. Keep the old tag in manifest history for rollback.

### Validation requirements

Verify no merge markers or unmerged files; compile/execute relevant INT8 checks where environment permits; compare built and deployment image tags; verify deployment readiness, logs, `/v1/models`, and one representative request.

### Current approach and material invalidated approaches

Merge `main` into `deployed` in the clean clone. If the merge conflicts, repair the fork INT8 semantics rather than preferring either side without inspection. Do not build plain main because it omits the deployed patch.

### Open material defects and repair evidence

None before the merge.

### Explicit assumptions and non-blocking unknowns

The current `sglang-dflash2-fp16-tp4-final` deployment is the desired target. The manifest may use a custom source overlay or build path. Discovery will establish the actual target and tag before changing it.

## Current checkpoint

`deployed` was merged and pushed at `2f9ec861cce8329c9204cbf1d6d70cf4470ce524`. It contains upstream main `24ff99d15`, the original `int8_block32` commit, and the additional live-overlay INT8 bridge/QSA/output-gate port. Static source checks, Python compilation, binding-presence checks, and no-conflict checks passed. The next action is an immutable SM70 image build from this exact public commit, then a manifest-backed rollout.
