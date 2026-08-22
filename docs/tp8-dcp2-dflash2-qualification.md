# TP8+DCP2 DFlash2 qualification

## Decision

Do not retain DFlash2. Do not create a DFlash2 candidate commit.

The isolated upstream compatibility port passed model and correctness gates. It failed
acceptance, capacity, c1 throughput, and c32 safety gates. No sequence-count result is
valid because the c32 run stopped during warmup.

## Scope

| Item | Value |
| --- | --- |
| Retained source | `3bc07baedc9066d079237cc34b74bcdeecbc380b` |
| Draft source | `5fa53b487980f4bdf2a81a61d10aa6356ac65177` |
| Upstream basis | vLLM PR `#52816` |
| Candidate projection | `de88395908a95da53698165053de478655dcb23e` |
| Topology | TP8 with DCP2 A2A |
| Draft policy | DFlash q8 with greedy selection |
| Candidate commit | none |

The port added DFlash2 model routing, V1 proposer integration, model sharing, and
selector support. It supports greedy selection only. Sampling and DDTree reject work
fail closed. They are not qualified.

## Checkpoint

| Item | Value |
| --- | --- |
| Repository | `incoai/Qwen3.8-27B-DFlash2` |
| Revision | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` |
| Staged path | `/srv/models/Qwen3.8-27B-DFlash2` |
| `model.safetensors` SHA256 | `67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c` |
| Projection manifest SHA256 | `bafeb1833c2e6b956cbb2500fe1c0f6f9fae235b43764b7883c9db0111b8049d` |

The public checkpoint required no Hugging Face credential. This report contains no
credential material.

## Passed gates

- The runtime resolved `DFlash2DraftModel`.
- All eight ranks shared target embeddings and the target LM head.
- All eight ranks hit the dynamic grouped-convolution and selector top-k-16 route.
- The q8 verifier CUDA graph captured for `8x1..4`.
- DFlash2 graph memory was 0.81 GiB.
- Five changed-input greedy responses exactly matched native MTP3.
- The matched response JSONL SHA256 was
  `657f7c0ce0fd5be3afc2989e0dd4c697695326d7afd5c39d0764c5f7d4c034a5`.
- Exact 8K, 32K, 128K, and repeated-32K retrieval checks passed.

## Failed gates

| Gate | Result |
| --- | --- |
| DFlash KV capacity | 1,437,416 tokens, below the 1,800,000-token floor |
| C1 throughput | 17.6333 tok/s over 120 seconds |
| Retained MTP3 C1 anchor | about 54.052 tok/s |
| Draft acceptance | 0 accepted of 14,798 drafts across 2,114 verifier steps |
| Mean acceptance length | 1.0 |
| C32 execution | stopped during warmup |

The c32 stale-draft guard stopped execution with this error:

```text
Speculative decode scheduled draft input slots, but the worker has no draft token tensor to scatter.
```

The guard prevented stale GPU `input_ids` reuse. It proves that c32 output could become
invalid. Therefore no c32 throughput, utilization, power, admission, latency, or
sequence-count result is valid.

The c1 window also did not saturate the hardware. It had one active request, mean SM
use of 64.85%, and mean power of 92.80 W. These readings do not support sequence tuning.

## Restoration

The service was restored without a DFlash runtime setting.

| Item | Result |
| --- | --- |
| Deployment | one desired, ready, and available replica |
| Source | `3bc07baedc9066d079237cc34b74bcdeecbc380b` |
| Speculator | native MTP3 |
| Candidate environment variables | none |
| Health endpoint | HTTP 200 |
| Final exact retrieval | 8K pass at 8,178 prompt tokens |
| Final retrieval SHA256 | `89dbd5412fd307b14a64db9f6cafde20cc58ea79dd774b7686c2af726de7d1c4` |
| Restored KV capacity | 2,103,266 tokens |
| Restored graph memory | 2.11 GiB |

Historical annotation labels remain on the deployment. They do not enable DFlash2.
The live command uses native MTP3. No environment variable contains `DFLASH` or
`CANDIDATE`.

## Blocking issue and next gate

The blocker is the DFlash multi-request draft-input contract. The scheduler assigns
draft slots, but the worker receives no draft-token tensor. The next engineering gate
must fix this contract before a new service benchmark.

The gate must prove all of these results:

1. All eight ranks receive and scatter fresh DFlash draft tokens for c32 changed inputs.
2. The stale-draft guard remains active and does not trigger.
3. A fixed-corpus c32 warmup reaches 32 active requests with zero waiting requests.
4. Greedy acceptance becomes nonzero and output parity remains exact.
5. Capacity remains at or above the 1.8-million-token floor.

Only after that gate passes can a sequence-count sweep compare c32, c48, or c64 with
matched MTP3 windows.

## Artifacts

```text
/srv/dev/dcp2-direct-lse-profile-53893bfb47/dflash2-qualification-de88395908
```

Artifact manifest SHA256:

```text
1df4f586e407e0e9012f437254e8aa63e7ae90b7807b16f9a49e790536e1c742
```

See `docs/tp8-dcp2-dflash2-qualification.json` for the compact evidence index.

## Review findings

- **Blocker:** The c32 DFlash worker lacks the draft-token tensor for scheduled slots.
- **High:** DFlash accepted no draft tokens in the c1 window.
- **High:** DFlash KV capacity missed its required floor by 362,584 tokens.
- **Correct:** The retained native-MTP3 service was restored and passed its final 8K gate.

## Residual risks

- The zero-acceptance result can indicate a selector or draft-token semantic mismatch.
- The candidate only qualifies greedy behavior. Sampling and DDTree remain unsupported.
- The c1 utilization result does not characterize a saturated DFlash service.
