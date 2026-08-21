# TP8+DCP2 native MTP depth sweep

## Verdict

MTP3 remains the active winner under the required retention rule.

MTP2 produced the highest raw c32 median at 638.2849 tok/s. MTP4 was effectively tied at 638.1720 tok/s.

Both depths failed the c1 gate. MTP2 regressed c1 by 6.83%, and MTP4 regressed c1 by 4.50%.

MTP1 regressed c1 by 18.82%. Its c32 median gain was 1.54%, but its pooled c32 gain was only 0.06%.

MTP3 produced the highest c1 median at 60.7642 tok/s. It remains the fastest eligible balanced configuration.

## Preflight

The checkpoint contains `text_config.mtp_num_hidden_layers=1`. Explicit depths one through four are structurally valid.

The sweep changed only `num_speculative_tokens`. All other service policy remained fixed.

| Policy | Fixed value |
|---|---|
| Runtime source | `d299e4acbc`, byte-equivalent to `59a5fa11f0` |
| TP / DCP | TP8 / DCP2 |
| DCP route | packed A2A with direct query gather |
| Attention backend | `FLASH_ATTN_V100` |
| Graph mode | `FULL_AND_PIECEWISE` |
| Sequence parallelism | `enable_sp=False` |
| Draft sampling | probabilistic |
| Draft reduction | local argmax |
| GDN full-forward | automatic SM70 route |
| Maximum sequences | 32 |
| Maximum batch tokens | 16,384 |
| GPU memory utilization | 0.78 |
| Image digest | `sha256:253e98bfd4a3f9e89187321b37dae01dd27642b3dc11546be881ce188df96c72` |

The source enables fused GDN ZBA extraction only for MTP3 and MTP4. MTP1 and MTP2 used the unfused default route.

The source seam is `vllm/model_executor/models/qwen3_5.py:123`.

## Protocol

The measured sequence was:

```text
MTP3, MTP1, MTP4, MTP2,
MTP2, MTP4, MTP1, MTP3,
MTP4, MTP3, MTP2, MTP1
```

Each depth received three clean starts. Each start used two fixed c32 warmups.

Each clean start measured:

- three fixed c1 cohorts
- three fixed c32 cohorts
- 256 completion tokens per request
- temperature 0.0
- top-p 1.0
- `ignore_eos=True`
- one exact 8K retrieval.

The first start for each depth also measured one 32K retrieval and one 2,048-token quality output.

All prompt text was source-neutral. Prompt text contained no depth name or source label.

All c1 and c32 corpus hashes matched across every depth. The normalized deployment fingerprint also matched across all starts.

No extra starts were necessary. C32 variance did not affect the decision because every non-MTP3 depth failed the independent c1 gate.

## Performance

Values below are medians across nine matched cohorts per depth.

| Depth | c1 tok/s | c1 vs MTP3 | c32 tok/s | c32 vs MTP3 | Pooled c32 vs MTP3 | Verifier step | Completion/step | Accepted/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MTP1 | 49.3309 | -18.82% | 612.5845 | +1.54% | +0.06% | 93.721 ms | 1.7953 | 0.7913 |
| MTP2 | 56.6139 | -6.83% | **638.2849** | +5.80% | +6.09% | 114.187 ms | 2.2642 | 1.2622 |
| MTP3 | **60.7642** | control | 603.2672 | control | control | 136.777 ms | 2.5785 | 1.5804 |
| MTP4 | 58.0315 | -4.50% | 638.1720 | +5.79% | +2.99% | 139.010 ms | 2.7508 | 1.7555 |

The three c32 start medians were:

| Depth | Start medians, tok/s |
|---|---|
| MTP1 | 610.1281, 612.5845, 629.5890 |
| MTP2 | 638.2849, 655.7465, 632.7821 |
| MTP3 | 603.2672, 632.3116, 602.8543 |
| MTP4 | 616.1527, 638.1720, 653.1659 |

C32 cohort ranges remained wide. The pooled results and corpus-specific medians supported the same c1 disqualifications.

## Acceptance

Pooled c32 acceptance stayed monotonic at every depth.

| Depth | Mean acceptance length | Per-position acceptance |
|---|---:|---|
| MTP1 | 1.7911 | 0.7911 |
| MTP2 | 2.2638 | 0.7570, 0.5068 |
| MTP3 | 2.5817 | 0.7601, 0.5010, 0.3206 |
| MTP4 | 2.7556 | 0.7546, 0.4946, 0.3146, 0.1919 |

No measured cohort produced all-position acceptance saturation.

## Correctness and quality

All 12 measured 8K retrievals passed. The final clean MTP3 restart also passed the same 8K retrieval.

The model removed underscores from each 8K secret. The assertion removed punctuation and whitespace only.

It preserved all alphanumeric characters, case, and order. Raw and normalized outputs remain in every benchmark record.

All four 32K retrievals passed. Each depth returned the complete exact secret.

All four 2,048-token quality outputs passed:

- all 20-token n-gram unique ratios were 1.0
- maximum identical line run was one
- maximum digit run was four
- acceptance stayed monotonic
- no output showed all-position saturation

The live requests exercised graph replay and the depth-specific recurrent-state contract. No source change entered the test.

## Graph and capacity

Every clean start completed CUDA graph capture. No runtime log reported a graph miss or eager fallback.

| Depth | Verifier graph shape | CUDA graph memory | KV capacity range | Fused ZBA |
|---|---|---:|---:|---|
| MTP1 | q=2, `2x1..32` | 1.94 GiB | 2,161,715 | no |
| MTP2 | q=3, `3x1..32` | 1.97 GiB | 2,134,931-2,142,641 | no |
| MTP3 | q=4, `4x1..32` | 1.99 GiB | 2,123,901 | yes |
| MTP4 | q=5, `5x1..32` | 2.03 GiB | 2,090,088-2,097,152 | yes |

All depths passed startup, graph, and capacity gates.

## Power and thermal sanity

Mean power increased from 110.1 W at MTP1 to 126.5 W at MTP4. Mean GPU use stayed between 68.2% and 70.7%.

The maximum measured temperature was 55 C. Thermal limits did not affect the result.

Some telemetry samples exceeded the nominal 200 W software limit. Mean power remained far below that limit.

## Decision

A non-MTP3 depth required all of these results:

- at least 1.5% higher fresh c32 median
- no c1 regression larger than 2%
- no correctness, quality, graph, or capacity regression
- stable matched evidence

MTP1, MTP2, and MTP4 failed the c1 requirement. MTP3 therefore remains active.

## Harness correction

The first attempted sequence contained an assertion defect. It treated underscore removal in a correct secret as a retrieval failure.

That attempt stopped before all c1 and c32 cohorts. Its data did not enter any performance result.

The complete 12-start sequence ran again after the assertion correction. The invalid artifacts remain separate for audit.

## Historical comparison

Prior MTP3/MTP4 and common-GDN results are not direct controls for this sweep.

Those campaigns used different corpora, prompt lengths, warmups, or source-labelled prompts. This report uses only its fresh fixed-corpus results.

## Final state

- Old TP8: `0/0`.
- TP8+DCP2: `1/1`, healthy.
- Active depth: MTP3.
- Runtime source: byte-equivalent to `59a5fa11f0`.
- KV capacity: 2,123,901 tokens.
- CUDA graph memory: 1.99 GiB.
- Final exact 8K retrieval: pass.
- Runtime source changes: none.

## Artifacts

Persistent root:

```text
/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp-depth-sweep
```

Manifest SHA256:

```text
8de6904def574787ffbe2d1d6803287012f3d03c2513dd95e78e31b574db7337
```

Archive SHA256:

```text
ce9fe5fdbf7eb1210458b7be8f191a7daba550de80cc2cff87e8678d795a28ae
```

## Review findings

- **High:** MTP1 failed the c1 gate by 18.82%.
- **High:** MTP2 failed the c1 gate by 6.83%.
- **High:** MTP4 failed the c1 gate by 4.50%.
- **Correct:** All retrieval, quality, graph, and acceptance gates passed.
- **Correct:** MTP3 is the only depth that satisfies the complete retention rule.

## Residual risks

- C32 cohort ranges remained wide.
- The quality matrix used one 2,048-token output per depth.
- The retrieval matrix used 32K, not 128K, as the extended gate.
- Power telemetry included transient readings above the nominal limit.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Recorded source paths, severity-ranked findings, 12 measured clean starts, 36 c1 cohorts, 36 c32 cohorts, correctness gates, graph evidence, capacity, and the final active MTP3 state."
    }
  ],
  "changedFiles": [
    "docs/tp8-dcp2-mtp-depth-sweep.md",
    "docs/tp8-dcp2-mtp-depth-sweep.json"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "checkpoint and live configuration preflight",
      "result": "passed",
      "summary": "Confirmed mtp_num_hidden_layers=1 and valid explicit depths one through four."
    },
    {
      "command": "12-start Latin-square config-only sweep",
      "result": "passed",
      "summary": "Completed three clean starts, nine c1 cohorts, and nine c32 cohorts per depth."
    },
    {
      "command": "8K, 32K, and 2,048-token quality gates",
      "result": "passed",
      "summary": "All measured retrieval and quality gates passed at every depth."
    },
    {
      "command": "graph, capacity, acceptance, power, and thermal checks",
      "result": "passed",
      "summary": "All graphs captured, acceptance stayed monotonic, capacity passed, and temperature stayed at or below 55 C."
    },
    {
      "command": "final clean MTP3 restart and exact 8K retrieval",
      "result": "passed",
      "summary": "MTP3 returned to healthy 1/1 with 2,123,901 KV tokens and 1.99 GiB graphs."
    },
    {
      "command": "artifact manifest verification",
      "result": "passed",
      "summary": "The persistent archive and every extracted artifact matched the recorded SHA256 manifest."
    }
  ],
  "validationOutput": [
    "MTP1 c1/c32 medians: 49.3309 / 612.5845 tok/s",
    "MTP2 c1/c32 medians: 56.6139 / 638.2849 tok/s",
    "MTP3 c1/c32 medians: 60.7642 / 603.2672 tok/s",
    "MTP4 c1/c32 medians: 58.0315 / 638.1720 tok/s",
    "All 12 measured 8K needles and all four 32K needles passed",
    "Final active depth: MTP3, healthy 1/1"
  ],
  "residualRisks": [
    "C32 cohort ranges remained wide, but every alternative failed the independent c1 gate.",
    "The extended correctness gate used 32K rather than 128K.",
    "The quality gate used one 2,048-token output per depth."
  ],
  "noStagedFiles": true,
  "diffSummary": "Added durable Markdown and JSON results only. Runtime source and serving policy did not change.",
  "reviewFindings": [
    "high: MTP1 c1 median regressed 18.82 percent.",
    "high: MTP2 c1 median regressed 6.83 percent.",
    "high: MTP4 c1 median regressed 4.50 percent.",
    "no correctness or capacity blocker. MTP3 remains the only eligible winner."
  ],
  "manualNotes": "The first attempted sequence used a defective punctuation-sensitive needle assertion. The complete measured sequence ran again after correction, and invalid artifacts remain separate."
}
```
