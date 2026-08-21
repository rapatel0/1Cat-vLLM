# TP8+DCP2 native MTP3 attention metadata result

## Decision

Keep the qualified native MTP3 runtime at source `59a5fa11f0`.

Actual q=4 metadata preparation is material, but no narrow candidate passed the promotion gate.

The tested GPU scalar-read removal did not reduce metadata time. No runtime source remains changed.

## Configuration

The profile used the active experimental `gpu-01` service:

- eight Tesla V100-SXM2-32GB GPUs.
- tensor parallel size 8.
- decode context parallel size 2.
- packed A2A and direct query gather.
- automatic PyNCCL.
- native MTP3 with three proposals and q=4 verification.
- fused GDN `z/b/a` extraction.
- maximum sequence count 32.
- 0.78 GPU memory utilization.

The profile enabled the existing async worker trace for one bounded baseline window.

It did not change runtime source.

Each attribution cohort used unique direct requests, greedy sampling, and ignored EOS tokens.

The c1 cohort used a 304-token prompt and 256 output tokens.

The c32 cohort used 306-308-token prompts and 256 output tokens.

## Actual q=4 attribution

The values below are medians of the slowest rank for each complete all-rank step.

| Phase | c1, 111 steps | c32, 93 steps |
|---|---:|---:|
| full preparation | 7.103 ms | 10.291 ms |
| `_prepare_inputs` | 2.139 ms | 2.733 ms |
| Mamba preprocessing | 0.082 ms | 0.127 ms |
| slot mapping | 0.061 ms | 0.057 ms |
| attention metadata | **4.194 ms** | **6.579 ms** |
| model preprocessing | 0.415 ms | 0.449 ms |

The earlier 6.1-7.6 ms citation came from MTP4 q=5. The new c32 q=4 result is 6.579 ms.

All four cache groups used the `build` path. No group used the cached metadata path.

| Builder and group | c1 | c32 |
|---|---:|---:|
| GDN, KV group 0 | 1.121 ms | 1.899 ms |
| GDN, KV group 1 | 1.094 ms | 1.748 ms |
| GDN, KV group 2 | 1.088 ms | 1.710 ms |
| Flash-V100, KV group 3 | 0.307 ms | 0.586 ms |

The three GDN groups dominate c32 metadata preparation.

They serve 48 GDN layers. The Flash-V100 group serves 16 full-attention layers.

## Existing GDN subphase profile

One bounded second window enabled the existing GDN subphase profiler. It did not change runtime source.

The table contains all-rank c32 medians for exact 32-request, 128-token q=4 steps.

| GDN group | total | state contract | graph buffers | remaining work |
|---|---:|---:|---:|---:|
| KV group 0 | 0.978 ms | 0.368 ms | 0.175 ms | 0.435 ms |
| KV group 1 | 0.954 ms | 0.357 ms | 0.176 ms | 0.419 ms |
| KV group 2 | 0.938 ms | 0.352 ms | 0.175 ms | 0.410 ms |

Registration measured 0.000 ms. The remaining bucket combines several operations and is not one replaceable boundary.

The runner timer starts before current-state ID materialization.

The internal builder timer starts after that runner work.

These timers came from separate bounded windows. Their difference does not isolate the state-ID path.

The state-ID source handles at most 128 integer assignments and one 512-byte asynchronous copy at c32 q=4.

No measured state-ID-only boundary exceeded 0.4 ms.

The proposed vectorization therefore lacked promotion evidence and was not implemented.

## Rejected scalar-read candidate

GDN metadata contained one unconditional GPU scalar read:

```python
assert spec_query_start_loc[-1].item() == num_spec_decode_tokens
```

A candidate removed only that assertion. It preserved every model, state, sampling, collective, and graph decision.

Nine focused metadata tests passed. The exact q=4 test covered full-graph padding and MTP3 state indices.

A 100-replay V100 component comparison changed every state-ID input. Output hashes matched across baseline and candidate runs.

The live trace failed the promotion gate:

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| c1 attention metadata | 4.194 ms | 4.251 ms | +0.057 ms |
| c32 attention metadata | 6.579 ms | 6.677 ms | +0.098 ms |
| c32 full preparation | 10.291 ms | 10.464 ms | +0.173 ms |

The result showed no full-step saving. The candidate was removed before correctness and throughput qualification.

## Corrected producer-overlap result

The earlier stale replay came from two prototype defects:

- The graph-local producer buffer lost its Python lifetime reference.
- Dependency events used a stream cached before CUDA graph capture selected its stream.

The repaired matrix passed 100 changed-input replays on all eight ranks. All five copy, GEMM, and PyNCCL routes were bit exact.

The corrected trace proved real GEMM and PyNCCL overlap. The valid split route still regressed from 18.9154 to 20.1086 ms.

The 6.31% regression came from 254 GEMMs and 254 reductions. No overlap source entered the runtime.

## Next measurable target

The next target is common uniform-q4 GDN metadata across KV groups.

A valid design must compute request classification and common graph fields once.

It must keep group-specific state IDs and block tables exact.

Do not implement this design before a component test identifies one repeated boundary worth at least 0.4 ms.

The fallback must preserve irregular batches, c1, no-MTP, MTP4, cache ownership changes, and generic builders.

## Final state

- Old TP8 service: `0/0`.
- TP8+DCP2 service: `1/1`, healthy.
- Active source: `59a5fa11f0`.
- Repository tip before this record: `d738a8d28e`.
- KV capacity: 2,123,901 tokens.
- CUDA graph memory: 1.99 GiB.
- Profiler variables: disabled.
- Gateway and shared virtual environment: unchanged.

Raw artifacts are under `/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp3-attention-metadata`.

The artifact manifest SHA256 is `1f89391ccf36a7acd66cd5c7b8cf4c1eeec724afaf35ac13629d8a322b919d8b`.
