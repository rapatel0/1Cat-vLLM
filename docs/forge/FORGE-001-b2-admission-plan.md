# FORGE-001 B2 Admission and Service A/B Plan

## Purpose

Establish, before expensive service scoring, that an isolated TP8/PP1 Hy3
baseline can hold two independent 64,001-token prompts concurrently. A valid
admission result is a scheduler fact: during the two-client-request overlap,
every 50 ms scheduler sample must show `running=2` and `waiting=0`. A merely
Ready Pod, a two-request submission, or a completed pair without this proof is
not B2 admission.

The admission probe is
`benchmarks/benchmark_hy3_max_context_admission.py`. It uses two fixed,
independently sourced realistic prompt-token JSON files, one completion token
per request, separate cache salts, streamed token IDs, and an exact prompt
echo. It writes a new, exclusive PVC result directory containing the invocation
plan, raw scheduler samples, request records, success/failure summary, and a
SHA-256 manifest. Inputs and outputs must all be under the same mounted
localpool-backed PVC; host-root storage is prohibited.

## Invariants

- The original Deployment remains untouched at zero replicas.
- Each trial uses the original immutable baseline image digest, Hy3 model and
  tokenizer revision, TP8/PP1, FP16 KV, maximum model length 65,536, two-slot
  contract, model clocks, and prompt inputs. The candidate later changes only
  image/source/cache/provenance.
- The Pod must be Ready with zero restarts and report physical KV capacity at
  least 128,514 tokens before the probe starts.
- `max_num_batched_tokens=2048` is retained rejected-control evidence and is
  never re-scored. `131584` is preserved OOM/crash-loop evidence and is never
  retried. B4/B8, MTP4, short-prompt proxies, TurboQuant q3, and alternate
  TP/PP layouts are outside this branch.
- CUDA graphs are preferred. An eager/no-graph result is eligible only under
  the narrow fallback below and must win all later end-to-end gates on merit.

## Isolated scheduler funnel

Every row gets a new `runtime_config_id`, Pod name, cache directory, output
directory, config/argv/source/server/GPU/JIT provenance files, and manifest.
Do not reuse an output directory after any result, including a failure.

| Order | `max_num_batched_tokens` | `max_num_partial_prefills` | `max_long_partial_prefills` | `long_prefill_token_threshold` | Graph mode |
| --- | ---: | ---: | ---: | ---: | --- |
| A1 | 4096 | 2 | 2 | 2048 | normal graphs |
| A2 | 8192 | 2 | 2 | 4096 | normal graphs |
| A3 | 16384 | 2 | 2 | 8192 | normal graphs |

Run rows in the listed order. Stop at the first passing row. Only if the first
booting partial-prefill row cannot pass admission with normal graphs, repeat
that row once with identical scheduler values and `compilation_config.mode=0`.
Do not treat eager mode as a fallback winner without the complete matched
baseline/candidate service evaluation.

Before each admission run, capture and hash the exact deployment argv,
environment/config, source tree, server image/pod identity, GPU UUID list, and
post-warm log evidence that the one-token shape has no JIT. Verify the API is
idle (`running=0`, `waiting=0`). The probe then requires both 64,001-token
requests to succeed with exactly one output token and a `length` finish. It
fails on any missing metric response, queue sample, non-`2` running sample in
the client overlap interval, missing exact prompt echo, salt collision,
unexpected JIT, timeout, OOM, restart, or artifact collision.

## After the first passing row

1. Warm both exact 64K shapes outside scoring until logs show no JIT.
2. Use the existing fixed five-repeat B1/B2 service harness on the baseline.
   Require no queued B2 interval, deterministic output hashes, complete
   manifests, and no JIT in scored windows.
3. Deploy the candidate using byte-identical selected scheduler/graph/topology,
   model, KV format, prompt, clock, and maximum-context configuration. Gather
   native q3 grouped-XQA MTP2 route proof and Nsight instruction/counter
   evidence before interpreting service numbers.
4. Execute matched baseline -> candidate -> baseline-confirm ->
   candidate-confirm service runs and score them only with the offline paired
   analyzer.
5. Do not start MTP-depth, quality, or canary experiments until the candidate
   clears realistic B1/B2 service throughput and integrated-dispatch gates.

Candidate promotion still requires the existing paired-improvement, 2x
realistic B2, per-slot balance, numerical, KAT/needle, short-regression,
profiler, and final falsification gates. If one fails, retain the stable
baseline and report the measured blocker with its manifest rather than
reframing a queued or partial run as a success.
