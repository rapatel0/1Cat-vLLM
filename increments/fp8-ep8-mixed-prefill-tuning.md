# Iron Increment: FP8 EP8 Mixed-Prefill Tuning

## Sealed brief

### Outcome

Compare the current 2K scheduler limit against 1K during concurrent 64K prefill and decode traffic.

Keep the official FP8 public route and restore the known-good configuration after the experiment.

### Delivery floor

- Measure 2K and 1K with the same control and mixed-load protocols.
- Test one decode alone and beside one 64K prefill.
- Test three decodes alone and beside one 64K prefill.
- Use three measured repeats after matched warmups.
- Record decode TTFT, TPOT, throughput, inter-token histogram bounds, prefill latency, capacity, and failures.
- Restore the known-good 2K configuration after the experiment.

### Results that do not count

- NVFP4 results do not count.
- A cached long prompt does not count.
- A run without exactly 65,536 prompt tokens does not count.
- Four decodes plus one prefill do not count because the service has four active slots.
- Cold-JIT results do not count as steady-state results.
- A candidate that changes the model, MTP4, or full-context contract does not count.

### Acceptance evidence

- Raw JSON for all 24 measured runs.
- Exact token usage for every long prefill.
- Three-repeat medians and ranges for every scenario.
- Runtime identity and KV capacity for both limits.
- Exact restoration of the frozen deployment command.
- Health, text, tool, image, and video checks after restoration.

### Preserve

- Official FP8 revision `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- TP8 with EP8, MTP4, `int8_block32`, and CUDA graphs.
- Four active 262,144-token slots.
- Asynchronous scheduling, prefix caching, and Mamba alignment.
- Text, tools, image, video, service names, and LiteLLM route names.
- No eager enforcement.

### Non-goals

- Do not test 512 unless a later user decision authorizes it.
- Do not test NVFP4 or decode context parallelism.
- Do not change model weights, kernels, or source code.
- Do not raise `max_num_seqs` above four.
- Do not permanently promote the 1K candidate.

### Material effects

- The benchmark applies controlled load to the public service.
- The 1K candidate and final restoration require two Recreate restarts.
- Each restart causes about six minutes of public route unavailability.

### Autonomous decisions

- Use one 64K synthetic token prompt with a unique leading block per run.
- Use three decode streams plus one prefill for the four-slot mixed scenario.
- Stop after startup, health, correctness, or capacity failure.
- Restore the known-good deployment after every terminal path.

### User authority

- The user authorized this 2K versus 1K mixed-load experiment now.
- Any 512 test or permanent promotion requires a later decision.
- Any source or model change requires new authority.

## Confirmed amendments

## Current orchestration

### Lead lens and contract fields

The evaluation lens leads this increment.

- Artifact: the public FP8 EP8 service at the sealed revision.
- Criteria: matched mixed-load performance, complete request success, preserved capacity, and exact restoration.
- Test set: synthetic 65,536-token prefills and deterministic 512-token decode streams.
- Protocol: decode-only controls and mixed scenarios at one and three decode streams, with three repeats.
- Metrics: decode TTFT, average TPOT, aggregate decode rate, histogram P99 bounds, prefill latency, and KV capacity.
- Baseline: the implicit 2,048-token batch limit.
- Required decisions: classify the 1K candidate and state whether a 512 test has evidence-based value.

### Active constraint lenses and required checks

The performance lens requires identical token prompts, workload order, warmups, and metric calculations across both limits.

The operations lens requires a frozen command, a rollback trap, candidate identity checks, health checks, and exact restoration.

### Cadence and evidence policy

Use execute, validate, and deliver. Store raw evidence under `/workspace/iron-002/tuning-fp8-ep8` and copy it to the local result directory.

Use unique leading token blocks so prefix caching cannot reuse long-prefill blocks across measured runs.

### Validation requirements

- Verify exactly 65,536 prompt tokens from server usage for each long request.
- Verify 512 output tokens and numeric content for every decode request.
- Capture server histogram deltas for inter-token latency.
- Verify FP8, TP8, EP8, MTP4, `int8_block32`, graph mode, and four active sequences.
- Verify that the 1K candidate retains at least 1,048,576 KV tokens.
- Restore the original command and repeat all public interface checks.

### Current approach and material invalidated approaches

Use four scenarios per scheduler limit: D1, D1 plus P1, D3, and D3 plus P1.

D3 plus P1 is the maximum mixed active batch. D4 plus P1 is invalid because one request must queue behind the four-slot limit.

Do not use client chunk gaps as token-level P99. MTP can return several accepted tokens in one SSE event. Use server histogram deltas instead.

### Open material defects and repair evidence

No material defect is open before execution. The restored 2K baseline is healthy.

### Explicit assumptions and non-blocking unknowns

The synthetic token prompt represents scheduler pressure, not natural-language prompt composition.

Server histogram buckets provide an upper bound for P99 rather than an exact percentile.

### Dependency map and initial frontier

1. Freeze the restored 2K deployment and benchmark-client identity.
2. Validate one exact 64K prompt request.
3. Measure all four 2K scenarios.
4. Restart with 1K and verify runtime identity and capacity.
5. Measure the same four scenarios.
6. Restore 2K and verify all public interfaces.
7. Calculate comparisons, inspect failures, and run independent review.

The initial frontier is benchmark-client construction and a 64K prompt probe.

## Current checkpoint
