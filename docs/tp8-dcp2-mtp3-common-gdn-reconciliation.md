# TP8+DCP2 native MTP3 common-GDN reconciliation

## Outcome

The fresh matched comparison rejected the common-GDN candidate at source `2b5d45abb0`.

Baseline source `59a5fa11f0` won the cohort median by 8.39% and pooled throughput by 3.99%.

The baseline is active on `gpu-01` at healthy `1/1`.

## Matched method

The sequence alternated baseline and candidate for six clean starts.

Each source received three starts and nine measured c32 cohorts.

Every start used the same:

- local client and port-forward path;
- image digest;
- normalized deployment configuration;
- fixed prompt corpus and order;
- two c32 warmups;
- 256 output tokens per request;
- temperature 0.0, top-p 1.0, and `ignore_eos=True`;
- exact 8K needle before measurement.

Each measured prompt contained 311 tokens.

The three corpus hashes matched across all six starts.

Startup verified each read-only projection with `sha256sum -c`.

## Fresh direct comparison

| Metric | `59a5fa11f0` | `2b5d45abb0` | Candidate change |
|---|---:|---:|---:|
| c32 cohort median | **724.8159 tok/s** | 664.0300 tok/s | **-8.39%** |
| Pooled throughput | **697.4629 tok/s** | 669.6403 tok/s | **-3.99%** |
| Median verifier step | **112.9450 ms** | 118.7300 ms | **+5.12%** |
| Median completion tokens/step | 2.5672 | 2.5632 | -0.16% |
| Median accepted drafts/step | 1.5697 | 1.5682 | -0.10% |

Baseline cohort results were:

```text
599.2963, 737.6947, 759.3566,
630.3393, 739.3396, 658.2195,
707.6186, 724.8159, 762.3684 tok/s
```

Candidate cohort results were:

```text
581.3608, 656.5048, 713.3910,
664.0300, 655.8454, 764.4202,
582.2408, 732.4007, 728.2189 tok/s
```

Baseline won the median for each fixed corpus by 7.63%, 11.01%, and 4.10%.

Pooled useful-token yield was equal within 0.03%.

The candidate regression came from verifier-step latency, not acceptance.

## Historical discrepancy

The historical 746.7680 and 694.8579 results are not directly comparable.

The 746.7680 result used `benchmark_qwen38_dcp2_service.py` with 249-256-token prompts and one 64-token c32 warmup.

The 694.8579 result used the metadata profile harness with 310-312-token prompts and one 32-token c32 warmup.

That profile harness also ran c1 work before c32.

Its source label appeared inside each prompt, so baseline and candidate did not receive identical corpora.

The canonical benchmark source did not change between `59a5fa11f0` and `2b5d45abb0`.

Historical accepted drafts were 1.38885 per step.

Fresh pooled values were 1.53801 for baseline and 1.53826 for candidate.

These acceptance values confirm a different prompt and acceptance mix.

Fresh cohort ranges remained wide, so run variance also contributed.

Only the fresh fixed-corpus comparison supports the current source decision.

## Correctness and capacity

All six start needles passed with the same 8,178-token prompt hash.

The final restored baseline passed the same needle again.

All starts used image digest:

```text
localhost:32000/onecat-vllm@sha256:253e98bfd4a3f9e89187321b37dae01dd27642b3dc11546be881ce188df96c72
```

Baseline reported 2,123,901 KV tokens and 1.99 GiB graphs for all three starts.

Candidate reported 2,123,901 KV tokens twice and 2,116,258 tokens once.

Candidate graph memory stayed at 1.99 GiB.

Probabilistic draft sampling changed some c32 response hashes under temperature zero.

Acceptance counters and exact needles supplied the correctness evidence.

## Decision

Source `2b5d45abb0` is not retained.

Source `59a5fa11f0` remains active with:

- TP8;
- DCP2 packed A2A and direct query gather;
- automatic PyNCCL;
- native MTP3;
- fused GDN ZBA extraction;
- 2,123,901 KV tokens;
- 1.99 GiB CUDA graphs.

No TP, NCCL, DCP, MTP, sampling, gateway, image, or virtual-environment policy changed.

## Artifacts

The structured record is `docs/tp8-dcp2-mtp3-common-gdn-reconciliation.json`.

Raw artifacts are under `/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp3-common-gdn-reconciliation`.

The initial artifact manifest SHA256 is `883782b3b9be4be1c00388c6fb87e965142819650e6e1a7dc99cfa1adbd1acf1`.

## Residual risks

Both sources retained wide cohort variance.

Historical artifacts do not attest their client host or image digest.

One candidate start showed a 7,643-token capacity decrease.
