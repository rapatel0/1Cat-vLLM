# SM70 signed INT8 block32 KV-cache production repair

## Status

This report covers the SM70 `int8_block32` repair for Qwen3.8-27B-FP8 with DFlash2.

The repair passes the B1 and B8 performance thresholds. It also restores E5M2 cache capacity.

The controller did not enable INT8 by default or promote an image.

## Representation contract

Each physical page keeps these fields:

- signed INT8 K and V payloads;
- separate FP16 K and V scales for each KV head and 32-channel block;
- one INT32 publication owner.

One CTA owns each page scale during a publication. Eight warps divide reset, maximum collection, historical re-quantization, and incoming writes.

The writer selects the batch-final scale before it changes historical payloads. It re-quantizes historical values once when a scale grows.

A write to page offset zero resets the page. Arbitrary block tables, reordered slots, and nonzero storage offsets remain valid.

## Read architecture

DFlash2 q8 verification groups up to 16 requests in one native SM70 launch. B1 q16 verification uses the same verifier with one request.

The kernel reads each request's query range, sequence length, and block-table row. A signed INT8 loader expands each eight-channel vector into shared memory.

The loader reads scales from the sealed page layout. It uses two aligned 32-bit payload loads because valid page strides require four-byte alignment.

Large prefix prefill uses a bounded FP16 bridge and the existing paged FP16 kernel. Mixed prefill batches split into exact per-request routes.

Unsupported shapes retain the exact scalar INT8 kernels.

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

The matched B8 runs used 8,192 input tokens and 128 output tokens for each request.

| Route | Cold elapsed | Steady elapsed | Steady aggregate rate |
| --- | ---: | ---: | ---: |
| E5M2 grouped control | 30.733 s | 12.636 s | 81.038 tokens/s |
| INT8 grouped candidate | 23.910 s | 11.728 s | 87.315 tokens/s |
| INT8 grouped candidate | - | 10.775 s | 95.035 tokens/s |

The median candidate steady elapsed time is 11.251 seconds. Its ratio to the E5M2 steady time is 0.890.

Each TP worker recorded 4,896 direct grouped verifier calls. The scalar INT8 control reached only 25.184 aggregate tokens per second.

## Kernel probes

The grouped q8 8K operator used 100 synchronized calls.

- INT8 direct grouped: 0.066386 ms per call.
- E5M2 grouped: 0.055009 ms per call.
- Ratio: 1.207.

The forced scale-growth writer improved from 1.355069 ms to 0.202947 ms per call. This is a 6.68-fold improvement.

## Correctness and graph evidence

Eight CUDA tests passed on SM70. They cover these contracts:

- signed payloads and separate K/V scales;
- batch-final scale growth and historical re-quantization;
- page reset and reuse;
- reordered and arbitrary block tables;
- padded page strides and nonzero storage offsets;
- direct grouped INT8 verification;
- bridge and attention CUDA Graph replay.

Seven cache-interface tests passed. They cover view geometry, alignment, hybrid unification, and rejected layouts.

The B1 and B8 engines captured PIECEWISE, target FULL, and DFlash2 FULL graphs. Runtime counters confirmed direct grouped INT8 verification for both batch sizes.

The complete SM70 routing policy file passed 120 tests. It includes the 16-request capability boundary and the exact fallback boundary.

## Rejected alternatives

A global INT8-to-FP16 bridge was correct and graph-safe. Its complete round time was about 35.5 ms, so the controller rejected it for verification.

An unaligned metadata-aware common page recovered capacity but broke vector alignment. The controller rejected that stride.

Packed 64-bit INT8 loads were unsafe for valid four-byte-aligned pages. The final loader uses 32-bit loads.

## Provenance

The owner workspace is `/workspace/iron-001-6d92c8c5` in pod `qwen38-int8-bench`.

The call 2 extension SHA256 is `63918518b8646f485fc9db0eb1abe8a9baa8991f69a3d35e7fe9fda8f7823168`.

Material call 2 result SHA256 values:

- B1 INT8 JSON: `27924c6555a9b86e6c8b5b9d24909dfa7ec01a910d3b1122a2e65559737ac8ba`;
- B1 INT8 log: `1023c31436d4ea2f49ed01655fc2a58ad016d9a3c4139aa5873b0bef9a68d729`;
- B8 E5M2 JSON: `829f2eb3f13012eb7e69680a483861fa9100d0ea238bde207b64880cc9912e91`;
- B8 E5M2 log: `4eb01a09fb54e9b17c4b62879655afdd38de91c3cf16a51a275bbe3c0a7b618f`;
- B8 INT8 JSON: `80dc5bfb34238d32851a4803c823156a563fb4fe334f1c1ec2cab5a07ce2a8f3`;
- B8 INT8 log: `2bf40522eabf269cbb095a9c295d743b2e070293ab56201d8e9486e23b0c637c`;
- focused tests: `f309fe435ec3e9b2f54cee19f5ae313ad0cbfe947fb1da4c25bc4dc81d0cb3fa`;
- policy tests: `0e36f0357704b0bb029ae0e2027e42d79b052d95e7f36ce9495ea138d0c3307a`.

The pinned QUASAR manifest has SHA256 `818a675a075f38a4f1f5917c77fe644e6a2a489a775f2ce121bcb92af0e6e1c3`.

The pinned revision is `d8e6fbfa3e3a78899b440222b827430045a05b44`. The validated download contains 17 files and 20,582,483,351 bytes.

## Quality evidence

The final provenance-bearing GSM8K run scored 13/16, or 81.25%. It reported zero invalid answers across 16 questions.

The prior FP16 control also scored 13/16. The final INT8 result SHA256 is `5272d991b5fd19341b7ebb3e2207d8346920f9615beadf385800d4ceca31f18e`.

The final quality log SHA256 is `af6a35cb1a3159ad2e982df00e162e89e05a5f050d1c7c70088860284c9838e4`.

## Production posture

The grouped verifier removes the B8 scalar bottleneck. The route remains explicit and non-default until release promotion finishes.
