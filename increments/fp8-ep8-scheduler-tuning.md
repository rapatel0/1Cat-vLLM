# Iron Increment: FP8 EP8 Scheduler Tuning

## Sealed brief

### Outcome

Run a controlled scheduler tuning experiment on the official Qwen3.8 Flash-Next FP8 EP8 service.

Measure the 2K, 4K, and 8K token batch limits. Keep the public route on FP8.

### Delivery floor

- Measure each token batch limit with the same B1, B2, and B4 protocol.
- Use three measured repeats after a matching warmup.
- Record TTFT, decode throughput, end-to-end throughput, MTP acceptance, and failures.
- Restore the known-good FP8 configuration after the experiment.
- Report the best measured candidate without permanent promotion.

### Results that do not count

- NVFP4 results do not count.
- Cached gateway responses do not count.
- Token counts without content checks do not count.
- Cold-JIT results do not count as steady-state results.
- A single successful request does not count as a concurrency result.
- A candidate that reduces the four-slot 262K contract does not count.

### Acceptance evidence

- Exact model revision and runtime arguments from the live engine log.
- Raw JSON for every measured repeat.
- Median metrics for every configuration and concurrency.
- HTTP success and non-empty output for every request.
- Direct health and text checks after restoration.
- Restored arguments that match the known-good configuration.

### Preserve

- Official FP8 checkpoint revision `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- TP8 with EP8.
- Four active sequences with a 262,144-token limit each.
- `int8_block32`, MTP4, CUDA graphs, asynchronous scheduling, prefix caching, and Mamba alignment.
- Text, tools, image, and video interfaces.
- The public service and LiteLLM route names.
- No eager enforcement.

### Non-goals

- Do not test NVFP4.
- Do not implement decode context parallelism.
- Do not change model weights, kernels, or source code.
- Do not raise `max_num_seqs` above four.
- Do not permanently promote a candidate in this increment.

### Material effects

- The benchmark applies controlled load to the public service.
- Candidate settings require three Recreate restarts.
- Each restart causes about six minutes of public route unavailability.

### Autonomous decisions

- Select unique deterministic prompts and a fixed output size.
- Stop a candidate after startup, health, correctness, or capacity failure.
- Restore the known-good deployment after any failure.
- Select the smallest sufficient benchmark protocol within the delivery floor.

### User authority

- The user authorized the three restart test window now.
- Any permanent configuration promotion requires a later decision.
- Any source change or model change requires new authority.

## Confirmed amendments

## Current orchestration

### Lead lens and contract fields

The evaluation lens leads this increment.

- Artifact and version: the public Qwen3.8 Flash-Next FP8 service at revision `236dfdf285828023ca3bcd3f37366c58a3469b13`.
- Criteria: identical protocol, successful outputs, preserved four-slot capacity, and relative scheduler performance.
- Test set: deterministic unique count prompts at B1, B2, and B4 with 512 output tokens.
- Protocol: one matching warmup and three measured repeats for each configuration and concurrency.
- Metrics: TTFT, aggregate decode rate, aggregate end-to-end rate, per-slot rate, MTP acceptance, and failures.
- Baseline: the live 2,048-token batch limit with MTP4.
- Required decisions: classify each configuration as full, degraded, or failed, then rank measured candidates.

### Active constraint lenses and required checks

The performance lens controls workload equivalence, repeatability, cache isolation, median calculations, and variance disclosure.

The operations lens supports safe live execution. It requires a frozen deployment baseline, rollback traps, health checks, and final restoration.

### Cadence and evidence policy

Use execute, validate, and deliver. Store raw JSON and logs outside Git under `/workspace/iron-002/tuning-fp8-ep8`.

Use direct service requests rather than LiteLLM. Use unique prompt nonces to prevent response-cache and prefix-cache contamination.

### Validation requirements

- Verify the model revision, FP8 quantization, EP8, MTP4, `int8_block32`, graph mode, and four sequence slots from logs.
- Verify HTTP 200, reported token usage, and non-empty count output for every request.
- Use three measured repeats and median summaries.
- Restore the exact original command after every terminal path.
- Verify health, one text response, and the restored scheduler argument after rollback.
- Record multimodal and tool checks as unchanged interface checks if existing fixtures remain available.

### Current approach and material invalidated approaches

Patch only the deployment command in the live Kubernetes object. Add one explicit token batch limit per candidate.

Keep the checked-in manifest unchanged. Do not use the idle development pod as a second server because the public process already occupies all GPUs.

Do not test more sequence slots. Prior evidence already rejects eight slots, and the accepted capacity requires four full contexts.

### Open material defects and repair evidence

No material defect is open before execution. The public service is healthy on the known-good baseline.

### Explicit assumptions and non-blocking unknowns

The experiment assumes that no unrelated operator changes the deployment during the test window.

Production traffic can add variance. The report will expose repeat dispersion and any concurrent request evidence.

### Dependency map and initial frontier

1. Freeze the live deployment object and benchmark client hash.
2. Measure the current 2K baseline at B1, B2, and B4.
3. Restart with 4K, verify runtime identity, and repeat the matrix.
4. Restart with 8K, verify runtime identity, and repeat the matrix.
5. Restore the exact baseline, then verify health and public behavior.
6. Calculate medians, inspect failures, and write the evaluation report.

The initial frontier is baseline capture and benchmark-client construction.

## Current checkpoint
