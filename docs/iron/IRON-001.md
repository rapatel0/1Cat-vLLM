<!-- iron-controller-brief:v2:start -->

# SM70 INT8 block32 KV-cache production repair

## Outcome

Deliver a merge-ready SM70/V100 `int8_block32` KV-cache implementation for Qwen3.8 with DFlash2 that preserves the signed block32 representation and correctness contract while removing the demonstrated decode, prefill, and hybrid-capacity regressions.

## Required behavior

- Preserve signed INT8 K/V payloads.
- Preserve separate FP16 K/V scales per KV head and 32-channel block.
- Preserve batch-final scale selection, historical page re-quantization when scale grows, page reset and reuse semantics, arbitrary block tables, nonzero storage offsets, prefix caching, Mamba alignment, and CUDA Graph replay.
- Keep unsupported shapes and contracts on a safe exact fallback.
- Keep non-INT8 cache formats and Flash-Next behavior unchanged.

## Acceptance

- Replace the measured scalar DFlash2 q8 bottleneck with a production-shaped grouped INT8 verifier or another evidence-backed route that achieves matched q8 round latency no worse than 1.25 times FP8 E5M2 on TP4 V100 at batch one and 8K context.
- Remove the hybrid-page allocation discontinuity and expose at least 95 percent of the E5M2 cache-token capacity under the matched TP4 Qwen3.8 hybrid configuration.
- Validate batch sizes one and eight, relevant context lengths, CUDA Graph replay, page reuse and scale growth, and the exact route selected at runtime.
- Preserve output quality under the established focused correctness suite and bounded GSM8K comparison.
- Produce reproducible benchmark artifacts with model, weight format, target and draft KV dtypes, backend, graph policy, tensor parallelism, lengths, environment, module provenance, and content hashes.
- Pass affected tests, formatting and static checks, diff checks, independent red-team assessment, and final review on the frozen candidate.

## Exclusions

- SGLang integration.
- TP8 claims while only four V100 GPUs are available.
- FP8 KV support for Flash-Next QSA, whose accepted reference contract requires FP16/BF16 main KV.
- Image promotion or default enablement before all acceptance gates pass.

## Authority and adaptation

The controller owns internal architecture, kernel organization, test design, benchmark ordering, and removal of superseded experimental code within this outcome. Renew user approval before changing representation semantics, enabling INT8 by default, weakening the capacity or latency gates, using additional paid hardware, or promoting deployment artifacts.


<!-- iron-controller-brief:v2:end -->

## Amendments and decisions

## Outcome

Delivered the merge-ready SM70/V100 `int8_block32` KV-cache repair for Qwen3.8 with DFlash2. The grouped INT8 verifier preserves signed K/V payloads, separate FP16 block32 scales, batch-final scale selection, historical page re-quantization, page reset and reuse, arbitrary paging, storage offsets, prefix caching, Mamba alignment, CUDA Graph replay, exact fallbacks, non-INT8 formats, and Flash-Next behavior.

Matched TP4 V100 evidence passed the sealed latency and capacity gates at batch sizes one and eight. The 261,632-token run, focused correctness suites, bounded GSM8K comparison, reproducible package probes, independent red team, and distinct adequate assessment passed. No blocking semantic residual remains.

## Continuity

The `int8_block32` route remains explicit and non-default. The accepted evidence covers TP4 on four V100 GPUs and a fixed 16-case visible GSM8K comparison. SGLang integration, TP8 claims, Flash-Next INT8 KV support, default INT8 enablement, and image or deployment promotion remain excluded.

Future changes to representation semantics, latency or capacity thresholds, hardware scope, default enablement, or promoted artifacts require separate authorization and fresh evidence. The two reusable probe filenames retain their `call2-` prefix; their call-3 executions and hashes remain current.

