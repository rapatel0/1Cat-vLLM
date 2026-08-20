# TP8+DCP2 MTP4 TP collective campaign

## Verdict

No TP collective candidate was retained.

The qualified automatic PyNCCL policy remains fastest and active on `gpu-01`.
The active projection stays at source commit `8b760b5f65`.

The current qualified result remains 54.2509 tok/s c1 and 734.9341 tok/s c32.
The active service retains the 2.097M-token MTP4 KV result.

The operator reference remains about 120 tok/s c1 and 1,700 tok/s c32 for TP8
without MTP. The qualified MTP4 result reaches 45.2% and 43.2% of those values.

## Active TP route

The TP group contains global ranks 0 through 7 on physical CUDA devices 0
through 7. Each process uses the same local-rank and device index.

The live route log reports only `PYNCCL` for `tp:0`. It reports the same route
for `dcp:0`.

The q=5 verifier all-reduce payload is:

- shape: `[160, 5120]`;
- dtype: FP16;
- size: 1,638,400 bytes;
- byte alignment: a multiple of 16;
- layout: contiguous;
- TP world size: eight.

The payload passes the size, alignment, and layout gates for custom all-reduce.
The physical topology fails its all-to-all NVLink gate.

`CustomAllreduce` rejects a group larger than two when the group is not fully
connected. All eight workers report this rejection. The controlled custom-on
and custom-off tests therefore both used PyNCCL.

The PyNCCL call is graph-safe. The CUDA graph stores stable input and output
addresses for each reduction. The hot q=5 graph has 127 all-reduces per target
step.

## Physical topology and NCCL rings

`nvidia-smi topo -m` reports a DGX-1-style hybrid cube mesh. Several GPU pairs
use `SYS`, so the eight-device group lacks direct all-to-all NVLink.

The NCCL topology dump maps ranks to PCI devices as follows:

| rank | PCI bus | GPU UUID suffix |
|---:|---|---|
| 0 | `1A:00.0` | `0300bd506c22` |
| 1 | `1B:00.0` | `3a8b5d0c6278` |
| 2 | `3D:00.0` | `f6645f3c33eb` |
| 3 | `3E:00.0` | `f65d0c4b23e0` |
| 4 | `88:00.0` | `79d7c38ce61e` |
| 5 | `89:00.0` | `a3fd15a0ec03` |
| 6 | `B2:00.0` | `b8f421bb100b` |
| 7 | `B3:00.0` | `1eb597391f4b` |

Automatic NCCL 2.27.5 selects twelve ring channels. It uses these four ring
orders:

- `0 3 2 1 5 6 7 4`, four channels;
- `0 4 7 6 5 1 2 3`, four channels;
- `0 1 3 7 5 4 6 2`, two channels;
- `0 2 6 4 5 7 3 1`, two channels.

The selected transport is `P2P/CUMEM`. NCCL reports `PXN 0` and `GDR 1`.
The two four-channel rings use the strongest direct NVLink cycle.

A dumped topology file reproduced the same twelve rings after reload. It did
not provide a stable latency reduction. No topology file was retained.

A rank-to-device reorder was not tested live. Such a reorder changes the
physical DCP pair map and violates this campaign contract.

## All-rank q=5 trace

The prior all-rank Torch trace contains six steady q=5 target steps. It shows:

- 762 matched q=5 all-reduce calls on each rank;
- 127 calls per target step;
- twelve NCCL channels for every matched call;
- zero overlap with another CUDA kernel or device copy;
- 30.285 microseconds median rank-arrival skew;
- 58.611 microseconds p95 arrival skew;
- 76.993 microseconds maximum arrival skew;
- 6.864 microseconds median completion skew.

Rank 6 arrived last for 243 of 762 calls. No other rank arrived last more than
100 times. The collective remains a hard serial boundary after each row-parallel
projection.

Per-rank median NCCL kernel time ranged from 95.232 to 101.985 microseconds.
The trace measured about 13.1 ms of serialized TP all-reduce time per q=5 target
step on most ranks.

## Exact route sweep

A CUDA-graph microbenchmark used the exact `[160, 5120]` FP16 payload. Each
captured graph contained 127 consecutive PyNCCL all-reduces.

Every reported route produced the exact rank-marker sum on all eight GPUs.

| route | channels | median microseconds per call | result |
|---|---:|---:|---|
| automatic Ring/LL128 | 12 | 78.599 | control |
| Ring/LL | 12 | 67.907 | 13.6% faster in isolation |
| Ring/LL | 16 | 66.354 | 15.6% faster in isolation |
| Ring/LL | 24 | **54.364** | 30.8% faster in isolation |
| Ring/LL | 32 | 55.937 | slower than 24 channels |
| Ring/LL128 | 32 | 76.146 | 3.1% faster in isolation |
| Ring/Simple | 12 | 105.024 | reject |
| Tree/LL128 | 4 | 129.025 | reject |
| P2P disabled | automatic | 626.019 | reject |

The 24-channel debug run confirmed the requested route. NCCL selected
`Algo RING proto LL channel{Lo..Hi}={0..23}` and repeated the known ring paths.

`NCCL_P2P_LEVEL` values `NVL`, `PIX`, `PXB`, `PHB`, and `SYS` did not produce a
stable gain. Direct-P2P disable and shared-memory disable also produced no gain.
Full P2P disable was 7.96 times slower than the stable automatic control.

The matched custom-attempt results were 78.561 and 78.599 microseconds per call
for custom-on and custom-off. Custom-on reported `custom_disabled=true`.

## Live candidate results

The environment knobs apply to every NCCL communicator in each worker. The
current PyNCCL wrapper uses `ncclCommInitRank` and exposes no per-communicator
protocol selector.

### Force LL protocol

`NCCL_PROTO=LL` passed the 8K exact needle gate and MTP4 graphs.

| metric | result | change from qualified baseline |
|---|---:|---:|
| c1, three-run median | 55.7576 tok/s | +2.78% |
| c32, three-run median | 619.6260 tok/s | -15.69% |
| c32, seven-run median | 697.3356 tok/s | **-5.12%** |
| c32 batch-step latency | 116.275 ms | +6.58% |
| accepted drafts per request step | 1.5102 | no material change |
| completion tokens per request step | 2.5096 | no material change |
| KV tokens | 2,090,873 | above floor |
| graph memory | 2.50 GiB | above baseline |

This candidate failed the live throughput gate and was removed.

### Force Ring/LL with 24 channels

`NCCL_ALGO=Ring`, `NCCL_PROTO=LL`, and 24 fixed channels passed the 8K exact
needle gate and MTP4 graphs.

| metric | result | change from qualified baseline |
|---|---:|---:|
| c1, three-run median | 55.3317 tok/s | +1.99% |
| c32, three-run median | 614.6222 tok/s | **-16.37%** |
| c32 batch-step latency | 130.373 ms | +19.50% |
| accepted drafts per request step | 1.5023 | no material change |
| completion tokens per request step | 2.5006 | no material change |
| KV tokens | 2,069,681 | above floor |
| graph memory | 2.21 GiB | above baseline |

The candidate reduced isolated TP latency but failed the live throughput gate.
It was removed before the long correctness matrix.

The live results prove that serialized microbenchmark sums do not predict this
system. A global NCCL override also changes DCP collectives and graph resources.
The current wrapper cannot scope the protocol to `tp:0`.

## Restored qualified state

The automatic NCCL policy is active with no protocol, algorithm, channel, P2P,
or topology override.

The final state passed an exact 8K needle check. A post-restore seven-run c32
check measured 727.9066 tok/s. This is 0.96% below the qualified 734.9341 tok/s
median and within the prior run variance.

The final post-restore c32 metrics were:

- 1.4945 accepted drafts per request step;
- 2.4954 completion tokens per request step;
- 110.655 ms per c32 batch step.

The old TP8 deployment remains at zero replicas. TP8+DCP2 remains healthy at
one replica. The gateway and shared virtual environment did not change.

## Stop decision and next bottleneck

No supported existing TP route beats automatic PyNCCL in the full MTP4 service.
The custom route cannot execute on this physical topology.

A per-communicator LL selector does not exist in the current wrapper. Adding a
new NCCL tuner plugin or a second loader namespace exceeds this narrow task.

The next exact target is the work before each collective. Rank 6 is the most
frequent late arrival, and linear GEMMs remain 48.48% of serialized CUDA time.
A future run must attribute that rank skew to specific preceding projections.

## Artifacts

Persistent directory:

`/srv/dev/dcp2-direct-lse-profile-53893bfb47/mtp-tp-collective`

Key records:

- `tp-ar-trace-analysis.json`, SHA256 `5cfd29886e5f25f171ec4e285f43eb23497e4895bedd062656a9c69f4fabc691`
- `nccl-topo.xml`, SHA256 `c88c8eb537dc44032f693433f4cfccfe424e7582520ac353b75ee19c70da6f3d`
- `tp-ar-topology-dump.log`, SHA256 `2211a33be195e251149f0879961baa9465ea22711c72648324e8313035c2bb22`
- `tp-ar-ring-ll-ch24-debug.log`, SHA256 `4494272034a184db27617c9bd8699663fb9b3a72e74f74e30890200bea68a382`
- `tp-ar-microbench-results.json`, SHA256 `96344e3e9e8e1e2ae6755941ac3f600a4854933f2ab065bb5020c6ab6991989d`
