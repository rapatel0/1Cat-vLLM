# SM70 MTP FP8 scale forensics

Date: 2026-04-09

Official FP8 checkpoint: `Qwen/Qwen3.8-Flash-Next-FP8` snapshot `236dfdf285828023ca3bcd3f37366c58a3469b13`

BF16 MTP source: `dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` fused tensors `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`

RadixArk NVFP4 MTP fused tensors are byte-identical to the abliterated MTP fused tensors.

## Official checkpoint config

- `quant_method`: `fp8`
- `activation_scheme`: `dynamic`
- `weight_per_tensor`: false
- `act_per_tensor`: false
- `weight_block_size`: `[128, 128]`
- `modules_to_not_convert`: 943 entries. MTP attn, shared expert, HC, router, embeddings stay BF16.
- `modules_to_convert`: 1 entry (expert matmuls, including MTP routed experts).

## Official MTP expert layout (expert 0)

| Tensor | dtype | shape | logical |
| --- | --- | --- | --- |
| `gate_proj.weight` | `float8_e4m3fn` | `(640, 2560)` | `[N, K]` |
| `gate_proj.weight_scale_inv` | `bfloat16` | `(5, 20)` | `[ceil(N/128), ceil(K/128)]` |
| `up_proj.weight` | `float8_e4m3fn` | `(640, 2560)` | `[N, K]` |
| `up_proj.weight_scale_inv` | `bfloat16` | `(5, 20)` | same |
| `down_proj.weight` | `float8_e4m3fn` | `(2560, 640)` | `[N, K]` |
| `down_proj.weight_scale_inv` | `bfloat16` | `(20, 5)` | same |

NVFP4 fused BF16: `gate_up_proj` `(512, 1280, 2560)` = concat(`gate`, `up`) on dim 1. `down_proj` `(512, 2560, 640)`.

Block order is row-major over `[N, K]`. No transpose.

## Scale convention

Dequant is multiply:

```
dq = fp8.float() * weight_scale_inv.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
```

`weight_scale_inv` matches `amax / 448`, not `448 / amax`.

Sample of 6 experts x 3 projections (18 tensors):

- `official_scale / (amax / 448)` median = 1.0
- mean about 1.0, std about 0.0015 (bfloat16 scale rounding)
- reconstruct vs BF16: rel L2 about 0.0266, cosine about 0.9996, sat 0%
- runtime `amax / 448` reconstructs the same BF16 to the same rel L2 within 0.00005

Raw BF16 weights are not equal to FP8 values. Reconstruction error is FP8 rounding, not a different source tensor.

## Names

Call the load-time path **runtime-amax FP8 MTP**. Do not call it ModelOpt MSE parity. Official stored scales match runtime-amax on these tensors.

## Next

Reuse official per-expert FP8 tensors when identity holds, or serialize fused runtime-amax FP8. Then add metadata dispatch and the three-way bench.
