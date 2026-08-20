# TP8+DCP2 MTP4 gap campaign

## Outcome

The campaign corrected the reference and moved MTP4 materially toward it.
The TP8 no-MTP reference is about 120 tok/s single and 1,700 tok/s c32.
Earlier artifacts incorrectly labeled the 126/1,730 anchors as TP8+MTP4.

The qualified candidate uses exact fused extraction for Qwen3.5 GDN projection
slices. It retains 2,097,152 KV tokens and all prior DCP optimizations.

| Metric | Matched baseline | Candidate | Change |
|---|---:|---:|---:|
| c1, seven-run median | 54.1710 | **54.2509** | +0.15% |
| c32, seven-run median | 678.6384 | **734.9341** | **+8.30%** |
| c1 verifier-step latency | 45.6855 ms | **44.6519 ms** | -2.26% |
| c32 batch-step latency | 118.9269 ms | **109.0993 ms** | **-8.26%** |
| MTP4 KV capacity | 2,097,152 | **2,097,152** | unchanged |
| CUDA graph memory | 2.04 GiB | **2.03 GiB** | unchanged |

The candidate reaches 45.2% of the 120 tok/s single reference.
It reaches 43.2% of the 1,700 tok/s c32 reference.
Against the original DCP2+MTP4 gate, c32 improved from 377.0 to 734.9 tok/s.
That is a 94.9% aggregate gain from the original qualified implementation.

## Benchmark interpretation

The official three-run candidate produced 53.4249 tok/s c1 and 663.5078 tok/s
c32. Its first c32 cohort was cold and measured 523.4129 tok/s.

The seven-run follow-up stabilized at 725.9166–748.8999 tok/s c32.
Its median was 734.9341 tok/s.
The matched seven-run baseline median was 678.6384 tok/s.

Raw acceptance percentage was not used as a performance decision.
The candidate produced these useful metrics:

- c1: 1.3988 accepted drafts and 2.4021 completion tokens per verifier step.
- c32: 1.5030 accepted drafts and 2.5030 completion tokens per verifier step.

The c32 acceptance length stayed effectively unchanged.
The throughput gain came from lower verifier-step cost.

## Bounded attribution

The diagnostic service enabled existing CUDA-event and async CPU instruments.
The profile synchronized for attribution, so its throughput was not a gate.

A delta over the last 64 steady c32 calls measured:

| Stage | Diagnostic time |
|---|---:|
| target q=5 forward | 59.45 ms |
| native MTP4 draft total | 13.26 ms |
| target logits | 2.621 ms |
| rejection sampler | 0.818 ms |

The proposer profile split its 13.051 ms GPU total into:

- first forward: 4.597 ms;
- first sample: 0.587 ms;
- three loop forwards: 1.569, 1.526, and 1.502 ms;
- three loop samples: 0.586, 0.594, and 0.590 ms;
- loop metadata on the CPU: 2.191 ms.

The async CPU trace measured 9.3–11.5 ms for target preprocessing.
Attention metadata consumed 6.1–7.6 ms of that range.
Sampling submission consumed 0.6–1.1 ms, and state updates consumed 0.4–0.7 ms.

A bounded eight-iteration Torch profile captured CUDA kernels on all ranks.
The rank-zero serialized kernel-duration census was:

| Category | Share |
|---|---:|
| linear GEMMs | **48.48%** |
| TP all-reduce | **32.73%** |
| GDN core | 3.72% |
| packed DCP A2A | 2.98% |
| full attention | 2.30% |
| DCP query all-gather | 1.60% |
| DCP pack/merge | 0.34% |
| sampler/recovery | 0.08% |
| other | 7.76% |

Kernel-duration shares are serialized sums.
NCCL kernels can include device wait, and independent streams can overlap.
The census still shows that DCP communication is no longer the primary gap.

## Implemented change

Commit `8b760b5f65` enables the existing exact fused z/b/a extraction for native
SM70 MTP4. One Triton kernel replaces three projection-slice copy launches in
each Qwen3.5 GDN layer.

The default is narrow:

- device capability SM70;
- speculative method `mtp`;
- exactly four speculative tokens.

`VLLM_SM70_GDN_FUSED_ZBA_EXTRACT` remains an explicit override.
No-MTP keeps its previous default path.
Sampling, rejection, and target probabilities are unchanged.

Commit `708d115dbd` also reverted the rejected q=1 DCP XQA source.
This made the canonical branch match the fastest scalar live route again.

## Qualification

The candidate passed:

- three new fused-route unit tests;
- exact 8K, 32K, and 128K needle retrieval;
- repeated 32K prefix retrieval twice;
- MTP4 CUDA graphs at 2.03 GiB;
- 2,097,152 MTP4 KV tokens;
- a no-MTP 8K smoke with zero speculative counters;
- a checksum-recorded immutable projection.

The no-MTP smoke reported 2,326,331 KV tokens and 0.46 GiB of graph memory.
Its short throughput shape was diagnostic only and was not compared with the
120/1,700 reference.

## Next bottleneck

The q=5 target verifier trunk is now the main bottleneck.
Linear GEMMs and TP all-reduce account for 81.2% of serialized kernel duration.
The next design must reduce q=5 TP arrival skew or verifier GEMM cost.
More sampler or DCP-combine work cannot close the remaining gap by itself.

The fastest exact candidate remains active on experimental `gpu-01`.
The old TP8 deployment remains scaled to zero, and the gateway remains unchanged.

## Artifacts

- Projection: `/localpool/onecat-vllm-hy3-sm70/dcp2-branch-8b760b5f6519`
- Projection manifest SHA256: `94935eb74dc9505f8dbc22a04f875ade52fdc0cf075e8a4ce725b1ac0c26ea1b`
- Torch profiles: `/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp-gap-torch`
- Rank-zero trace SHA256: `385b2ae5931eadd4c3c5809d86b8ddd29d057acec55936b41caa954fb1dc8f5a`
- Machine record: `docs/tp8-dcp2-mtp4-gap-campaign.json`
