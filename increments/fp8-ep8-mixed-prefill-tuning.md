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

Pending controller alignment.

## Current checkpoint
