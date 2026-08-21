# TP8+DCP2 native MTP3 common GDN metadata

## Outcome

The common uniform q=4 GDN metadata route passed its component, graph, retrieval, and no-MTP gates.

A fresh fixed-corpus comparison rejected its performance claim and restored source `59a5fa11f0`.

Source commits `dd432ec253` and `2b5d45abb0` contain the inactive candidate.

The candidate makes the first GDN group write invariant buffers once. Each GDN group retains its distinct state contract.

## Phase-1 measurement

The probe excluded the state-contract call, the state-index copy, and the state-index tail fill.

It measured mask work, query-length work, token-index work, and invariant graph-buffer writes. The scalar assertion was also excluded from the conservative result.

| GDN group | Common boundary, critical-rank median |
|---|---:|
| KV group 0 | 0.4085 ms |
| KV group 1 | 0.4005 ms |
| KV group 2 | 0.3870 ms |
| Redundant groups 1 and 2 | **0.7875 ms** |

The probe compared 40 warmed c32 q=4 steps per rank and group.

All seven proposed invariant tensor fields matched byte for byte across the three groups on all eight ranks:

- `spec_query_start_loc`
- `non_spec_query_start_loc`
- `spec_sequence_masks`
- `spec_token_indx`
- `non_spec_token_indx`
- `num_accepted_tokens`
- `spec_state_slot_selectors`

The live tensors had these q=4 graph shapes:

| Tensor | Shape |
|---|---:|
| `spec_query_start_loc` | `[129]` |
| `spec_sequence_masks` | `[128]` |
| `spec_token_indx` | `[128]` |
| `num_accepted_tokens` | `[128]` |
| `spec_state_slot_selectors` | `[128]` |
| `non_spec_token_indx` | `[0]` |
| `non_spec_query_start_loc` | absent |

The baseline used distinct invariant pointers for each group. Each group registered 16 GDN layer names, for 48 total GDN layers.

## Candidate route

The runner enables the route only when all conditions match:

- SM70
- native MTP3
- three homogeneous GDN KV groups
- 32 live requests
- four query tokens per request
- align-mode Mamba cache
- unchanged request ownership
- full CUDA graphs
- no DDTree metadata
- no debug state-table dump

The state width comes from the active speculative configuration. The route does not use a fixed state-width constant.

The first eligible group writes the shared persistent buffers for one runner epoch. The next two groups reuse those buffers.

Each group keeps a separate persistent `spec_state_indices_tensor`. Each group also uses its own state IDs and block table.

The regular builder remains active for c1, irregular batches, MTP4, no-MTP, ownership changes, graph capture, debug dumps, and other builders.

## Correctness and graph evidence

Focused tests covered different state IDs for all three groups, shared invariant pointers, PAD tails, changed accepted counts, and changed selectors.

The tests also covered c1, MTP4, no-MTP, and graph-state fallback behavior.

A V100 component gate changed all state IDs, accepted counts, and selectors for 100 graph replays.

Every replay matched exactly. State pointers stayed distinct, and invariant pointers stayed shared.

Three matched c1 responses were byte-identical between the baseline and candidate. This supplied the target top-1 parity gate.

## Performance

The original campaign reported 694.8579 tok/s for baseline and 728.7711 tok/s for candidate.

That comparison used source labels inside each prompt, so the two prompt corpora differed.

It also used a different corpus and warmup from the historical 746.7680 result.

A fresh sequence used six alternating clean starts and nine identical fixed-corpus cohorts per source.

| Metric | `59a5fa11f0` | `2b5d45abb0` | Candidate change |
|---|---:|---:|---:|
| c32 cohort median | **724.8159 tok/s** | 664.0300 tok/s | **-8.39%** |
| Pooled throughput | **697.4629 tok/s** | 669.6403 tok/s | **-3.99%** |
| Median verifier step | **112.9450 ms** | 118.7300 ms | **+5.12%** |
| Median completion tokens/step | 2.5672 | 2.5632 | -0.16% |
| Median accepted drafts/step | 1.5697 | 1.5682 | -0.10% |

Baseline won all three corpus-specific medians.

The candidate failed the fresh promotion gate and is inactive.

See `docs/tp8-dcp2-mtp3-common-gdn-reconciliation.md`.

## Historical candidate gates

Before reconciliation, the candidate passed these retrieval gates:

- 8K needle at 8,785 prompt tokens
- 32K needle at 34,970 prompt tokens
- 128K needle at 139,695 prompt tokens
- repeated 32K prefix twice

The repeated prefix request fell from 7.03 seconds to 2.05 seconds. Both responses returned the exact secret.

MTP3 retained 2,123,901 KV tokens and 1.99 GiB of CUDA graph memory.

The no-MTP restart passed an exact 8K retrieval request. It reported no speculative metrics, 2,347,416 KV tokens, and 0.46 GiB graph memory.

## Artifacts

Original artifacts are under `/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp3-common-gdn-metadata`.

The original artifact manifest SHA256 is `97ba306328e2d5b3ad1cb74601dc5a8caa60e504c795b344bd9da5c22149bb98`.

Reconciliation artifacts are under `/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp3-common-gdn-reconciliation`.

## Residual risks

Both fresh sources retained wide cohort variance.

Probabilistic draft sampling changed some c32 response hashes under temperature zero.

One candidate start reported 7,643 fewer KV tokens.

The full metadata test file has known environment-sensitive failures under automatic SM70 graph defaults. The focused candidate tests passed.
