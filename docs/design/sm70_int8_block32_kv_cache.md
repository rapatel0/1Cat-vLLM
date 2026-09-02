# SM70 signed INT8 block32 KV-cache production repair

## Status

This report covers the SM70 `int8_block32` repair for Qwen3.8-27B-FP8 with DFlash2.

The repair passes the B1 performance threshold and restores E5M2 cache capacity. It does not enable INT8 by default.

The controller did not promote an image. B8 still uses the exact scalar fallback and needs a grouped multi-request route.

## Representation contract

Each physical page keeps these fields:

- signed INT8 K and V payloads;
- separate FP16 K and V scales for each KV head and 32-channel block;
- one INT32 publication owner.

One CTA owns each page scale during a publication. Eight warps divide reset, maximum collection, historical re-quantization, and incoming writes.

The writer selects the batch-final scale before it changes historical payloads. It re-quantizes historical values once when a scale grows.

A write to page offset zero resets the page. Arbitrary block tables, reordered slots, and nonzero storage offsets remain valid.

## Read architecture

DFlash2 B1 q8 and q16 verification uses the existing grouped SM70 verifier. A signed INT8 loader expands each eight-channel vector into the verifier shared-memory tile.

The loader reads scales from the sealed page layout. It uses two aligned 32-bit payload loads because valid page strides require only four-byte alignment.

Large B1 prefix prefill uses a bounded FP16 bridge and the existing paged FP16 kernel. Unsupported shapes retain the exact scalar INT8 kernels.

Non-INT8 routes remain unchanged. Flash-Next behavior remains unchanged.

## Hybrid capacity repair

The old unifier padded a 1,648-token INT8 page toward the larger draft page. Fixed scale and owner bytes prevented exact division.

The repair grows the INT8 and aligned Mamba block size to 3,296 tokens. It then aligns the common physical page stride to 16 bytes.

The candidate reports 1,595,299 cache tokens. The matched E5M2 control reports 1,595,299 cache tokens.

The first unaligned candidate reported 1,561,055 tokens but caused a CUDA alignment fault. The final 16-byte stride removed that fault.

## TP4 V100 performance

The matched runs used four V100-SXM2-32GB GPUs, B1, 8,192 input tokens, 128 output tokens, and DFlash2 q7.

| Route | Warm prefill | Decode rounds | Decode time | Time per round |
| --- | ---: | ---: | ---: | ---: |
| E5M2 grouped control | 0.409023 s | 42 | 0.875713 s | 20.850 ms |
| INT8 direct grouped, steady slice 1 | 0.435971 s | 44 | 1.027303 s | 23.348 ms |
| INT8 direct grouped, steady slice 2 | 0.434932 s | 48 | 1.118698 s | 23.306 ms |

The INT8 round ratios are 1.120 and 1.118. Both values pass the required 1.25 limit.

The INT8 warm-prefill ratios are 1.066 and 1.063. The old INT8 warm prefill was 2.948006 seconds.

The candidate B1 aggregate throughput reached 86.637 tokens per second. The old INT8 maximum was 15.551 tokens per second.

The B8, 4,096-token validation completed three repeats. Its aggregate rates were 30.138, 53.760, and 55.433 tokens per second.

B8 remains on the scalar exact route. This route is safe but does not remove the existing B8 performance gap.

## Kernel probes

The grouped q8 8K operator used 100 synchronized calls.

- INT8 direct grouped: 0.066386 ms per call.
- E5M2 grouped: 0.055009 ms per call.
- Ratio: 1.207.

The forced scale-growth writer improved from 1.355069 ms to 0.202947 ms per call. This is a 6.68-fold improvement.

## Correctness and graph evidence

Seven CUDA tests passed on SM70. They cover these contracts:

- signed payloads and separate K/V scales;
- batch-final scale growth and historical re-quantization;
- page reset and reuse;
- reordered and arbitrary block tables;
- padded page strides and nonzero storage offsets;
- direct grouped INT8 verification;
- bridge and attention CUDA Graph replay.

Seven cache-interface tests passed. They cover view geometry, alignment, hybrid unification, and rejected layouts.

The B1 and B8 engines captured PIECEWISE, target FULL, and DFlash2 FULL graphs. Runtime route counters confirmed direct grouped INT8 verification for B1.

## Rejected alternatives

A global INT8-to-FP16 bridge was correct and graph-safe. Its complete round time was about 35.5 ms, so the controller rejected it for verification.

An unaligned metadata-aware common page recovered capacity but broke vector alignment. The controller rejected that stride.

Packed 64-bit INT8 loads were unsafe for valid four-byte-aligned pages. The final loader uses 32-bit loads.

## Provenance

The owner workspace is `/workspace/iron-001-6d92c8c5` in pod `qwen38-int8-bench`.

The final focused extension SHA256 is `245bd9dadb472623e29d1dc3af4fc0d0e97ef8d29b18df654c50ec985782d134`.

Material result SHA256 values:

- B1 INT8 JSON: `fbd79b7bc38feb8420c05ce41167089978e2e70b0e5d41a4ec426be7137468d3`;
- B1 INT8 log: `b6e343bd4e485938d8c3860ded5607b46379a4c1334180073588e9c62a15b2fb`;
- B1 E5M2 JSON: `9b79727a116c89f3c781cc93f44a3f5d5b1c561b9d088f6565c57736956db765`;
- B1 E5M2 log: `d0ec8e73e97ec509b2412c4bb8f3f36c91773db8f783e0a7aa9ffc48ba21398b`;
- B8 INT8 JSON: `7f1f6ba1375f25042086dcae0489f649e3ce24ebacdbdfcd2a59533794341011`;
- B8 INT8 log: `858e51d3502a21f7557fa65d3b67b5ce88c90edb65a3067f18d87afbe15b8972`;
- final focused tests: `bd121539ac658411b0ba51eea34430c102882d333e6e34b30f9fd61e00c30891`;
- final policy tests: `10e74799a26f450173a7f6ae6eb88dda32d8a1554e248b717a5c9cde2ad6059d`;
- writer probe: `19792cb97f6ad1ad3c5beee9fbed3d8c6ddf77129b89be2f9f3cd8d52242fb7d`.

The pinned QUASAR manifest has SHA256 `818a675a075f38a4f1f5917c77fe644e6a2a489a775f2ce121bcb92af0e6e1c3`.

The pinned revision is `d8e6fbfa3e3a78899b440222b827430045a05b44`. The validated download contains 17 files and 20,582,483,351 bytes.

## Quality evidence

The final provenance-bearing GSM8K run scored 13/16, or 81.25%. It reported zero invalid answers across 16 questions.

The prior FP16 control also scored 13/16. The final INT8 result SHA256 is `5272d991b5fd19341b7ebb3e2207d8346920f9615beadf385800d4ceca31f18e`.

The final quality log SHA256 is `af6a35cb1a3159ad2e982df00e162e89e05a5f050d1c7c70088860284c9838e4`.

## Remaining gate

B8 needs a grouped multi-request verifier before production promotion. Until that gate passes, this route remains explicit and non-default.
