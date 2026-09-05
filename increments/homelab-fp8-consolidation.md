# Iron Increment: Homelab FP8 consolidation

## Sealed contract

### Outcome

Consolidate homelab LLM serving around the already-live Qwen3.8 Flash-Next FP8 EP8 deployment, release the RTX 4090 from the active EXL3 server for the existing Steam Headless/Sunshine workload, and remove confirmed obsolete zero-scale LLM resources and deprecated EXL3 serving material.

### Deliverable

A reviewed homelab Git branch with the reconciled active FP8 route/default and cleanup changes, applied live operational changes in Kubernetes, and a concise report of final cluster state, verification, rollback boundaries, and removed scope.

### Delivery floor

- `qwen38-flash-next` remains the default through Hermes and works through LiteLLM.
- No active EXL3 model or Hermes/gateway reference requires the RTX 4090.
- Steam Headless is running on the 4090 with Sunshine listening.
- The four confirmed orphan zero-scale LLM services/deployments are deleted without shared storage deletion.
- Homelab source no longer retains the deprecated EXL3 4090 deployment as an active serving path.
- Commits are pushed without including agent state, experimental DCP work, or unrelated reports.

### Results that do not count

- Merely scaling an already-zero resource or deleting a Service while stale gateway/Hermes references remain.
- Starting a gaming pod without GPU, Sunshine, or service reachability evidence.
- Declaring the FP8 model default from a manifest while the live Hermes PVC configuration disagrees.
- A broad delete, reset, force-push, or deletion of shared PVCs, ConfigMaps, benchmark evidence, or retained zero-scale model manifests without direct evidence.

### Acceptance evidence

- Direct LiteLLM/Hermes model request succeeds with `qwen38-flash-next` after EXL removal.
- Gateway model list no longer advertises the EXL3 route; live Hermes configuration has no EXL3 provider/fallback/vision dependency.
- The EXL deployment is zero/absent and its pod no longer uses the 4090.
- `steam-headless-0` is ready, owns the 4090, and has Sunshine listeners; its KasmVNC Tailnet endpoint remains usable.
- Confirmed orphan deployments/services are absent, with no shared PVC or ConfigMap deletion.
- Homelab repository change is based on current `origin/main`, excludes unrelated changes, validates manifests, and is pushed.

### Preserve

- FP8 deployment `sglang-dflash2-fp16-tp4-final`, its service, model revision, TP8/EP8/MTP4 FP8 runtime contract, and public `qwen38-flash-next` alias.
- `llm-gateway`, Redis cache, ingress, trace compaction, `superqwen-nfs-backup-helper`, Lift OCR artifacts, gaming host-path data, Steam credentials, and existing game image.
- Shared model/workspace PVCs, ConfigMaps, completed-job logs, research/benchmark artifacts, and zero-scale model manifests not directly confirmed obsolete.
- Existing user worktree changes, including DCP experiments and documentation, unless their file is directly required by this outcome.

### Non-goals

- Testing a 512-token scheduler limit, NVFP4 work, source/kernel/model changes, routine historical job deletion, PVC reclamation, ConfigMap evidence deletion, or a comprehensive retirement of every dormant LLM profile.
- Adding Tailnet UDP exposure for Sunshine; Sunshine remains LAN-only unless separately authorized.
- Altering unrelated models, storage, cluster node services, or GPU-01 workloads.

### Material effects

- Removes public EXL3 availability and changes Hermes fallback/vision routing to FP8.
- Scales down EXL3 and starts one GPU-backed gaming pod on GPU-02.
- Deletes four confirmed unused zero-scale deployment/service pairs and one orphan Service.
- Produces and pushes a homelab source branch; direct pushes to `main`, force pushes, resets, and broad staging are prohibited.

### Autonomous decisions

- Reconcile the clean source branch with current `origin/main`; choose a feature branch; select targeted manifest/ref/doc edits; sequence route removal, fallback replacement, GPU handoff, smoke tests, and reversible rollbacks.
- Keep a directly affected zero-scale model manifest when direct reference evidence is uncertain.

### User authority

- Any deletion of PVCs, shared ConfigMaps, completed Jobs or their logs, non-confirmed dormant model profiles, benchmark evidence, or game/model data.
- Any permanent change to Sunshine network exposure or credentials.
- Any force push, direct push to `main`, or step requiring reset/discard of pre-existing user changes.

## Confirmed amendments

None.

## Current orchestration

### Lead lens and contract fields

Operations is the lead lens.

- Target and intended state: public FP8 EP8 serving remains the Hermes default; the 4090 is released from EXL3 and runs Steam Headless with LAN-only Sunshine; confirmed orphan zero-scale objects are absent; a clean homelab branch contains matching desired state.
- Authority and allowed effects: the sealed contract authorizes route/ref replacement, EXL scale-down, gaming scale-up, targeted orphan-object deletion, feature-branch push, and direct live verification. It excludes destructive storage/history cleanup and broad dormant-profile deletion.
- Known-good state: `sglang-dflash2-fp16-tp4-final` and gateway are 1/1; `qwen38-flash-next` is the live Hermes default and works publicly; EXL3 4090 is 1/1; Steam Headless is 0/0; orphan candidates are already zero with no endpoints.
- Rollback criteria: any failure of public FP8 or Hermes post-change check restores the saved live Hermes config and gateway ConfigMap; gaming start failure scales Steam back to zero and EXL3 to one. No orphan resource is restored unless a post-delete reference is found.
- Health and user signals: public LiteLLM model-list/chat checks, Hermes configuration inspection, backend health, endpoints, GPU ownership, Steam readiness, Sunshine listeners, and KasmVNC Tailnet probe.
- Stop condition and handoff owner: stop before unapproved destructive deletion or network exposure. The user owns later decisions about historical jobs/PVCs/other dormant models.

### Active constraint lenses and required checks

Production readiness: preserve known-good snapshots before modifying live ConfigMaps/Deployments; test the user-facing LiteLLM and gaming access surfaces; ensure rollback follows a failure.

Compatibility: remove all live and desired-state EXL3 references before decommissioning it; preserve `qwen38-flash-next` alias and backend `qwen` model mapping.

### Cadence and evidence policy

Execute → validate. Use direct live observations as runtime evidence and isolated source-branch diffs/server-side dry runs as source evidence. Avoid broad applies and never apply an unreconciled user worktree.

### Validation requirements

- Save live target manifests/config before effectful changes.
- Validate config changes against the API server before apply; validate the source branch with focused text/reference checks.
- Check public FP8 health/model list/chat after route/ref updates.
- Check absence of EXL3 model listing/provider references and absence of EXL pod/endpoints.
- Check Steam readiness, `nvidia-smi`, Sunshine listening ports, and Tailnet KasmVNC route after start.
- Verify targeted orphan objects are absent and named shared storage/config resources persist.
- Push only the clean feature branch based on current `origin/main`.
- Obtain fresh independent review before completion.

### Current approach and material invalidated approaches

The FP8 deployment is already the live Hermes default, so no default-model promotion is needed. Replace EXL3 fallback/vision dependencies with FP8, remove the EXL3 public route, then release the GPU and start Steam. Delete only the four manifest-free confirmed orphan deployment/service groups and the orphan Gemma Service. Create a clean feature branch from updated `origin/main`, selectively carry the desired FP8 routing/service manifests, add source cleanup, and leave the dirty local `main` unchanged.

Invalidated approaches: applying the existing mixed staged index; deleting all zero-scale deployments; deleting EXL storage or historical jobs; adding a Sunshine Service/UDP Tailnet exposure; or treating a LiteLLM alias as a global gateway default.

### Open material defects and repair evidence

None. The local homelab `main` is behind remote and has mixed staged/unstaged work; a separate clean branch avoids data loss and non-fast-forward push risk.

### Explicit assumptions and non-blocking unknowns

Steam credentials and the existing image are valid. Sunshine is expected to remain LAN-only at host `192.168.102.6`; direct remote/UDP validation is not available. The 4090 is single-GPU, so EXL3 must fully release before Steam starts.

## Current checkpoint

None.
