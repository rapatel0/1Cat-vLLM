# TP8+DCP2 MTP3 fused GDN extraction

## Verdict

The exact fused Qwen3.5 GDN `z/b/a` extraction now supports native MTP3 by default on SM70.

The candidate passed component, graph, target-output, retrieval, prefix-cache, capacity, and no-MTP gates.

The seven-run c32 median changed from 746.2713 to 746.7680 tok/s. This 0.07% difference is within run variance.

The exact fusion reduces launches without an end-to-end performance claim. The candidate remains active on experimental `gpu-01`.

| Metric | Unfused MTP3 | Fused MTP3 | Change |
|---|---:|---:|---:|
| c1, three-run median | 53.3336 tok/s | **54.0520 tok/s** | +1.35% |
| c32, seven-run median | 746.2713 tok/s | **746.7680 tok/s** | +0.07% |
| c32 completion tokens per step | 2.3894 | 2.3888 | -0.02% |
| c32 batch-step cost | 103.4912 ms | **103.2699 ms** | -0.21% |
| KV tokens | 2,123,901 | **2,123,901** | unchanged |
| CUDA graph memory | 2.00 GiB | **1.99 GiB** | unchanged |

The official candidate c32 median was 657.6331 tok/s across three runs. Its cohorts increased from 627.2634 to 694.1931 tok/s.

The longer c32 gate resolved this cold-run variance. Its seven cohorts spanned 712.7048 through 759.0724 tok/s.

## Gate audit

The old default required exactly four speculative tokens. That condition recorded the only qualified live depth.

The extraction kernel contains no MTP-depth constant. It uses the row count, slice widths, offsets, and row strides.

MTP3 changes the target row count from q=5 to q=4. It does not change the GDN projection widths or state contract.

The new default requires:

- SM70
- native `mtp`
- three or four speculative tokens

`VLLM_SM70_GDN_FUSED_ZBA_EXTRACT` remains the explicit override. No-MTP keeps the unfused default.

## Implementation

Source commit `59a5fa11f0` extends the narrow default gate from MTP4 to MTP3/4.

One Triton kernel copies `z`, `b`, and `a`. The old path uses three separate contiguous-copy launches.

The change does not alter:

- sampling
- MTP depth selection
- recurrent-state selection
- DCP or TP collectives
- graph topology
- the gateway
- the shared virtual environment

## Component and state checks

Ten focused route and CUDA tests passed on a V100.

The CUDA tests covered four and 128 q=4 rows. They covered contiguous FP16 rows and padded row strides.

Changed-input CUDA graph replay matched the unfused slice reference exactly for all `z`, `b`, and `a` values.

A live all-rank A/B compared 32 sample records and 288 q=4 graph buffers.

All extracted state inputs matched exactly. Layer-zero output differed by at most 0.00006104 across separate clean starts.

Target logits differed by at most 0.0625. Target top-1 tokens matched at every compared row.

Four sequential target steps produced the same top-1 sequence and the same 16-token response. This check exercises recurrent-state transitions after exact state inputs.

## Live qualification

The candidate passed:

- exact 8K retrieval
- exact 32K retrieval
- exact 128K retrieval
- repeated 32K prefix retrieval twice
- q=4 CUDA graph capture and replay
- a no-MTP exact 8K smoke
- a final clean MTP3 restart and exact 8K check

The repeated-prefix requests completed in 8.9703 and 5.4846 seconds. Both returned the exact secret.

The no-MTP smoke reported 2,326,331 KV tokens and 0.46 GiB of graph memory. All speculative counter deltas were zero.

The final MTP3 service reports 2,123,901 KV tokens and 1.99 GiB of graph memory.

## Artifacts

Projection:

`/localpool/onecat-vllm-hy3-sm70/dcp2-branch-59a5fa11f0a1`

Projection manifest SHA256:

`3dd61333fa9786e888e554e45f1bcbcf3a77762b2225106646e102ea6a8ae7a1`

Persistent evidence root:

`/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp3-fused-zba`

Key hashes:

- parity report: `ac529bd3b866d449efe4523e466173d25593349e1116f13edee045e3982d27e7`
- official three-run gate: `3aa0dffb8a859a09911c8fb03b80cf9ac23d3d0eba749c0ccdc238166c1daaf0`
- c32 seven-run gate: `84df8d4eef9db2e1345441d7d4e02fcffbced9fdfa26417b032298ab976fa535`
- short needles and prefix: `746874682b3b87a0d7bdaf8cd23737ddad4a68e9a462a68f89810e471f7a2bfb`
- no-MTP smoke: `78d426fde53a82d857a49751a2898d8dfd2789f66502325767107dab91d3d560`
- final 8K check: `94d09d2f3bf1eb69b45c0db288acaf4ba568ec7ccd636fc7f1e2fc164bf188b0`

## Final state

- old TP8: `0/0`
- TP8+DCP2: `1/1`, healthy
- active speculation: native MTP3
- active projection: `59a5fa11f0`
- DCP backend: packed A2A
- TP backend: automatic PyNCCL
- fused GDN extraction: MTP3 default
- profiler: disabled
- gateway: unchanged
- shared virtual environment: unchanged

## Residual risks

The end-to-end c32 difference is within run variance. The result proves no loss, not a throughput gain.

The live A/B used separate clean starts. FP16 target outputs were tolerance-equal, not bitwise equal after downstream accumulation.

The historical 120/1,700 no-MTP reference lacks its original workload record.

The unchanged no-MTP mode reported 2,326,331 KV tokens. This remains below the plan's historical 2.48M floor.
