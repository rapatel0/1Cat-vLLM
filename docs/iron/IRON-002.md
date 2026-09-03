<!-- iron-controller-brief:v2:start -->

# Qwen3.8 Flash Next INT8 KV with native MTP4

## Outcome

Deliver a merge-ready SM70/V100 implementation that runs `RadixArk/Qwen3.8-Flash-Next-NVFP4` with the signed `int8_block32` KV cache and native Qwen4Exp MTP4 speculation enabled together. Preserve the accepted Qwen3.8 Flash Next hybrid-cache behavior and the landed INT8 block32 representation.

## Fixed runtime contract

- Model: `RadixArk/Qwen3.8-Flash-Next-NVFP4` at Hugging Face revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Hardware: four NVIDIA V100-SXM2-32GB GPUs at SM70.
- Runtime: TP4, PP1, V2 model runner, FP16 activations, ModelOpt NVFP4 weights, language-model-only, `FLASH_ATTN_V100`, native Qwen4Exp MTP with four speculative tokens, CUDA graphs, prefix caching, chunked prefill, and Mamba alignment.
- Cache: signed `int8_block32` K/V payloads with separate FP16 K and V scales for each KV head and 32-channel block.
- Storage: place every new model download, Hugging Face cache, build artifact, benchmark artifact, and workspace on localpool. Do not place them on the host root filesystem. Use Hugging Face only for model downloads.

## Required behavior

- Make the Qwen4Exp target verifier and MTP proposal flow compatible with the QSA portions of the hybrid `int8_block32` KV cache.
- Preserve GDN/Mamba recurrent state, compressed QSA state, fixed circular QSA ring semantics, PLE behavior, and model-runner committed-token semantics.
- Preserve batch-final scale selection, historical page requantization when a scale grows, page reset and reuse, arbitrary block tables, nonzero storage offsets, prefix-cache reuse, Mamba alignment, and CUDA Graph replay.
- Keep malformed metadata and unsupported shapes on a safe exact fallback or fail-fast contract. Do not silently reinterpret INT8 cache bytes as FP16 or FP8.
- Keep FP16, BF16, FP8, DFlash, Eagle, and non-Qwen behavior unchanged.
- Keep INT8 KV and MTP explicit opt-ins. Do not enable either globally by default.

## Acceptance

- Reproduce the current FP16-KV native-MTP route and the current INT8-KV plus MTP failure before implementation.
- Load the fixed model on TP4 V100 with `kv_cache_dtype=int8_block32` and native MTP4 enabled.
- Complete deterministic eager and CUDA-graph generations with nonzero MTP draft and acceptance counters.
- Prove that runtime route evidence binds Qwen4Exp MTP, `FLASH_ATTN_V100`, and `int8_block32` for the same requests.
- Validate batch one and a heterogeneous transition batch, prefix-cache hits, graph replay, cache scale growth, cache reset and page reuse, relevant QSA cache groups, and exact fallback boundaries.
- Compare matched INT8-KV MTP and FP16-KV MTP runs with identical model revision, prompt and output lengths, TP topology, graph mode, sampling, speculative configuration, and throughput definition.
- Keep matched INT8 verifier round latency no worse than 1.25 times the FP16-KV MTP control, unless a prediction-bearing probe proves that this oracle is structurally invalid and renewed user approval changes it.
- Keep formal mean acceptance length at least 95 percent of the matched FP16-KV MTP control on the bounded acceptance workload.
- Preserve bounded semantic quality with zero invalid outputs and no score regression against the matched FP16-KV MTP control.
- Do not reduce measured cache-token capacity. Record the actual hybrid-cache memory and capacity effect.
- Produce reproducible artifacts that bind source, model revision, weight format, target and draft cache dtypes, attention backend, graph policy, TP topology, lengths, sampling, MTP counters, environment, module provenance, and hashes.
- Pass affected unit, CUDA, integration, formatting, static, and package checks. Pass an independent red team and a distinct adequacy assessment on the frozen candidate.

## Initial discrimination

First establish the earliest incompatible boundary. Distinguish cache-spec planning, cache allocation/layout, KV writing, MTP draft attention, target verification, graph metadata, prefix reconciliation, and model-specific QSA behavior. Run the cheapest vertical TP4 smoke after each material boundary repair. Do not optimize an unverified path.

## Exclusions

- TP8 claims.
- SGLang integration.
- Vision-tower execution.
- Global default enablement of INT8 KV or MTP.
- Image, wheel, or deployment promotion beyond merge-ready validation artifacts.
- Checkpoint mutation or conversion.
- New prerelease dependencies.

## Authority and adaptation

The controller owns diagnosis, internal architecture, kernel and Python organization, tests, benchmark tooling, validation order, and removal of superseded experiments within this outcome. The controller can revise internal tactics after evidence, but it cannot weaken acceptance or expand exclusions. Renew user approval before changing the model or revision, checkpoint bytes, representation semantics, topology, dependencies, defaults, performance or quality gates, paid resources, or deployment effects.


<!-- iron-controller-brief:v2:end -->

## Amendments and decisions

<!-- iron-amendment:v2:A001:start -->
### Model-weight correction: official block-scaled FP8

The user replaced the NVFP4 checkpoint target before controller dispatch.

- Supersede `RadixArk/Qwen3.8-Flash-Next-NVFP4` with the official Hugging Face checkpoint `Qwen/Qwen3.8-Flash-Next-FP8`.
- Pin the current inspected Hugging Face revision `236dfdf285828023ca3bcd3f37366c58a3469b13` unless Hugging Face proves that this immutable revision is unavailable.
- Preserve the official checkpoint bytes. Do not convert the model to INT8, NVFP4, AWQ, or another weight format.
- The inspected official configuration declares FP8 quantization, dynamic activations, and `weight_block_size=[128,128]`.
- Exercise the exact SM70 TurboMind block-FP8 W8A16 path. On V100, the accepted claim is FP8 weight storage and scale handling with FP16 HMMA Tensor Core computation. Do not claim native FP8 or INT8 Tensor Core execution.
- Verify that eligible dense and MoE weights retain the intended block-scaled FP8 route. Identify every full-precision exclusion from the official checkpoint configuration.
- Keep the confirmed signed `int8_block32` KV cache and native Qwen4Exp MTP4 requirements unchanged.
- Start with matched performance characterization of official FP8 weights. Compare against the safest same-checkpoint control that changes only the SM70 weight backend when memory permits.
- Record weight-route logs, prepared weight/scaling metadata, GPU memory, model-load time, MTP acceptance, verifier round latency, pure decode throughput, whole-request throughput, and prefill separately.
- Use Hugging Face only. Place the model, Hugging Face cache, source workspace, build cache, and results on localpool.

The controller retains authority to tune the existing TurboMind FP8 block-scaled route within the confirmed acceptance contract. Renew user approval before adding a new INT8 model-weight implementation or changing the official checkpoint bytes.
<!-- iron-amendment:v2:A001:end -->

## Outcome

## Continuity
