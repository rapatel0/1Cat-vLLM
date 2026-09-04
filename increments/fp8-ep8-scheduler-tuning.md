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

Pending controller alignment.

## Current checkpoint
