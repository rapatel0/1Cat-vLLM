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

The grouped dispatch validates the packed query partition before computation. Offsets must start at zero, end at the query count, and increase strictly.

Each multi-request query range must contain at most eight tokens. A device assertion preserves this validation during CUDA Graph replay.

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

The sealed B8 comparison uses 8,192 input tokens and 128 output tokens for each request.

The controller acceptance report records the replacement-candidate latency, throughput, route counts, and artifact hashes.

## Kernel probes

The grouped q8 8K operator used 100 synchronized calls.

- INT8 direct grouped: 0.066386 ms per call.
- E5M2 grouped: 0.055009 ms per call.
- Ratio: 1.207.

The forced scale-growth writer improved from 1.355069 ms to 0.202947 ms per call. This is a 6.68-fold improvement.

## Correctness and graph evidence

The SM70 CUDA regression suite covers these contracts:

- signed payloads and separate K/V scales;
- batch-final scale growth and historical re-quantization;
- page reset and reuse;
- reordered and arbitrary block tables;
- padded page strides and nonzero storage offsets;
- direct grouped INT8 verification;
- malformed and variable grouped query partitions;
- independent query ranges for grouped sparse page4 attention;
- bridge and attention CUDA Graph replay.

Seven cache-interface tests passed. They cover view geometry, alignment, hybrid unification, and rejected layouts.

The B1 and B8 engines captured PIECEWISE, target FULL, and DFlash2 FULL graphs. Runtime counters confirmed direct grouped INT8 verification for both batch sizes.

The complete SM70 routing policy suite covers query-partition bounds, the 16-request limit, and exact fallback.

The sparse page4 regression matches the accepted parent output hashes before and after CUDA Graph replay.

## Rejected alternatives

A global INT8-to-FP16 bridge was correct and graph-safe. Its complete round time was about 35.5 ms, so the controller rejected it for verification.

An unaligned metadata-aware common page recovered capacity but broke vector alignment. The controller rejected that stride.

Packed 64-bit INT8 loads were unsafe for valid four-byte-aligned pages. The final loader uses 32-bit loads.

## Provenance

The owner workspace is `/workspace/iron-001-6d92c8c5` in pod `qwen38-int8-bench`.

The controller acceptance report records the replacement extension SHA256.

The accepted parent and replacement sparse page4 outputs use these SHA256 values:

- direct output: `9f79c318c6f1537ea1ddaeb5f8dbca992d2e498783599f28ae4a9f78c46d94dd`;
- graph replay output: `7f3ae5e4f37ff45b9fc7eb1e9bd6a02c2a354f68f6b61c6db0d020defbfedce8`.

The controller acceptance report records all replacement result hashes.

The pinned QUASAR manifest has SHA256 `818a675a075f38a4f1f5917c77fe644e6a2a489a775f2ce121bcb92af0e6e1c3`.

The pinned revision is `d8e6fbfa3e3a78899b440222b827430045a05b44`. The validated download contains 17 files and 20,582,483,351 bytes.

## Quality evidence

The controller acceptance report records the frozen-candidate GSM8K comparison and content hashes.

## Production posture

The grouped verifier removes the B8 scalar bottleneck. The route remains explicit and non-default until release promotion finishes.
