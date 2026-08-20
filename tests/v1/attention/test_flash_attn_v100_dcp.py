# SPDX-License-Identifier: Apache-2.0
"""CPU numerical checks for the Flash-V100 DCP attention integration."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import torch

ROOT = Path(__file__).parents[3]
BACKEND = ROOT / "vllm/v1/attention/backends/flash_attn_v100.py"
FLASH_INTERFACE = ROOT / "flash-attention-v100/flash_attn_v100/flash_attn_interface.py"
ATTENTION_UTILS = ROOT / "vllm/v1/attention/backends/utils.py"
DCP_QUERY_GATHER = ROOT / "vllm/v1/attention/ops/dcp_query_gather.py"
FUSED_MHA_FORWARD = ROOT / "flash-attention-v100/kernel/fused_mha_forward.cu"
FLASH_DECODE = ROOT / "flash-attention-v100/kernel/flash_decode_paged.cu"


def _dense_attention(q, k, v, *, causal, softmax_scale, **_kwargs):
    """CPU reference with FlashAttention BMHD output and BHM LSE layouts."""
    repeat = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(repeat, dim=2)
    v = v.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bmhd,bnhd->bhmn", q.float(), k.float()) * softmax_scale
    if causal:
        m, n = q.shape[1], k.shape[1]
        q_pos = torch.arange(m).view(m, 1) + (n - m)
        k_pos = torch.arange(n).view(1, n)
        scores = scores.masked_fill((k_pos > q_pos).view(1, 1, m, n), -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.einsum("bhmn,bnhd->bmhd", probs, v)
    return out.to(q.dtype), lse, None


def _merge_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse):
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    prefix_weight = torch.exp(prefix_lse - max_lse).transpose(0, 1).unsqueeze(-1)
    suffix_weight = torch.exp(suffix_lse - max_lse).transpose(0, 1).unsqueeze(-1)
    output.copy_(
        (prefix_output * prefix_weight + suffix_output * suffix_weight)
        / (prefix_weight + suffix_weight)
    )


_CURRENT_GROUP = None
_RECORDED_ROUTES: list[str] = []


class _FakeGroup:
    world_size = 2

    def __init__(self, rank, q_all, prefix_k, prefix_v, local_k, local_v, scale):
        self.rank_in_group = rank
        self.q_all = q_all
        self.prefix_k = prefix_k
        self.prefix_v = prefix_v
        self.local_k = local_k
        self.local_v = local_v
        self.scale = scale
        self.combine_calls = 0

    def all_gather(self, query, dim):
        assert dim == 1
        heads = query.shape[1]
        torch.testing.assert_close(
            query,
            self.q_all[
                :, self.rank_in_group * heads : (self.rank_in_group + 1) * heads
            ],
        )
        return self.q_all

    def combine(self, local_out, local_lse, return_lse):
        assert return_lse
        self.combine_calls += 1
        q_bmhd = self.q_all.unsqueeze(0)
        if self.local_k.shape[0] == 0:
            assert torch.count_nonzero(local_out) == 0
            assert torch.isneginf(local_lse).all()
        else:
            expected_out, expected_lse, _ = _dense_attention(
                q_bmhd,
                self.local_k.unsqueeze(0),
                self.local_v.unsqueeze(0),
                causal=False,
                softmax_scale=self.scale,
            )
            torch.testing.assert_close(local_out, expected_out.squeeze(0))
            torch.testing.assert_close(
                local_lse,
                expected_lse.squeeze(0).transpose(0, 1),
            )

        full_out, full_lse, _ = _dense_attention(
            q_bmhd,
            self.prefix_k.unsqueeze(0),
            self.prefix_v.unsqueeze(0),
            causal=False,
            softmax_scale=self.scale,
        )
        heads_per_rank = self.q_all.shape[1] // self.world_size
        head_slice = slice(
            self.rank_in_group * heads_per_rank,
            (self.rank_in_group + 1) * heads_per_rank,
        )
        return (
            full_out.squeeze(0)[:, head_slice],
            full_lse.squeeze(0)[head_slice].transpose(0, 1).contiguous(),
        )


class _FakeBatchGroup:
    world_size = 2

    def __init__(
        self,
        rank,
        q_all,
        prefix_ks,
        prefix_vs,
        local_ks,
        local_vs,
        scale,
    ):
        self.rank_in_group = rank
        self.q_all = list(q_all)
        self.prefix_ks = prefix_ks
        self.prefix_vs = prefix_vs
        self.local_ks = local_ks
        self.local_vs = local_vs
        self.scale = scale
        self.combine_calls = 0
        self.next_fallback_seq = 0

    def all_gather(self, query, dim):
        assert dim == 1
        q_flat = torch.cat(self.q_all)
        heads = query.shape[1]
        torch.testing.assert_close(
            query,
            q_flat[
                :,
                self.rank_in_group * heads : (self.rank_in_group + 1) * heads,
            ],
        )
        return q_flat

    @staticmethod
    def _attention(q, k, v, scale):
        if k.shape[0] == 0:
            return (
                torch.zeros_like(q),
                torch.full(q.shape[:-1], -torch.inf),
            )
        out, lse, _ = _dense_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            causal=False,
            softmax_scale=scale,
        )
        return out.squeeze(0), lse.squeeze(0).transpose(0, 1).contiguous()

    def combine(self, local_out, local_lse, return_lse):
        assert return_lse
        self.combine_calls += 1
        total_tokens = sum(q.shape[0] for q in self.q_all)
        if local_out.shape[0] == total_tokens:
            seq_indices = range(len(self.q_all))
        else:
            seq_indices = [self.next_fallback_seq]
            self.next_fallback_seq += 1

        expected_local_out = []
        expected_local_lse = []
        full_out = []
        full_lse = []
        for seq_idx in seq_indices:
            q_seq = self.q_all[seq_idx]
            out, lse = self._attention(
                q_seq,
                self.local_ks[seq_idx],
                self.local_vs[seq_idx],
                self.scale,
            )
            expected_local_out.append(out)
            expected_local_lse.append(lse)
            out, lse = self._attention(
                q_seq,
                self.prefix_ks[seq_idx],
                self.prefix_vs[seq_idx],
                self.scale,
            )
            full_out.append(out)
            full_lse.append(lse)

        torch.testing.assert_close(local_out, torch.cat(expected_local_out))
        torch.testing.assert_close(local_lse, torch.cat(expected_local_lse))
        heads_per_rank = self.q_all[0].shape[1] // self.world_size
        head_slice = slice(
            self.rank_in_group * heads_per_rank,
            (self.rank_in_group + 1) * heads_per_rank,
        )
        return (
            torch.cat(full_out)[:, head_slice],
            torch.cat(full_lse)[:, head_slice].contiguous(),
        )


def _dcp_local_lens(seq_lens, dcp_size, dcp_rank, interleave):
    base = seq_lens // interleave // dcp_size * interleave
    remainder = seq_lens - base * dcp_size
    remainder = torch.clamp(remainder - dcp_rank * interleave, 0, interleave)
    return base + remainder


def _extract_kv(*, kv_cache, seq_lens, total_tokens, **_kwargs):
    assert int(seq_lens[0]) == total_tokens
    return (
        kv_cache[0, 0, :total_tokens].clone(),
        kv_cache[0, 1, :total_tokens].clone(),
    )


def _cp_lse_combine(
    out,
    lse,
    group,
    return_lse=False,
    trace_range=None,
):
    if trace_range is None:
        result = group.combine(out, lse, True)
    else:
        with trace_range("lse_prepare"):
            pass
        with trace_range(f"lse_all_gather bytes={lse.numel() * lse.element_size()}"):
            pass
        with trace_range("output_correction"):
            result = group.combine(out, lse, True)
        with trace_range(
            f"output_reduce_scatter bytes={out.numel() * out.element_size()}"
        ):
            pass
    return result if return_lse else result[0]


def _query_all_gather(query, group, trace_range=None):
    query_bytes = query.numel() * query.element_size()
    if trace_range is None:
        return group.all_gather(query.contiguous(), dim=1), False
    with trace_range(f"query_gather_prepare_fallback bytes={query_bytes}"):
        prepared = query.contiguous()
    with trace_range(f"query_all_gather_fallback bytes={query_bytes}"):
        gathered = group.all_gather(prepared, dim=1)
    return gathered, False


def _a2a_lse_combine(
    out,
    lse,
    group,
    return_lse=False,
    trace_range=None,
    allow_unmanaged_buffers=False,
    use_persistent_buffers=False,
):
    del allow_unmanaged_buffers, use_persistent_buffers
    payload_bytes = (out.numel() + 2 * lse.numel()) * out.element_size()
    if trace_range is None:
        result = group.combine(out, lse, True)
    else:
        with trace_range(f"a2a_pack bytes={payload_bytes}"):
            pass
        with trace_range(f"a2a_all_to_all bytes={payload_bytes}"):
            result = group.combine(out, lse, True)
        with trace_range(f"a2a_unpack bytes={payload_bytes}"):
            pass
    return result if return_lse else result[0]


def _namespace():
    return {
        "torch": torch,
        "get_dcp_group": lambda: _CURRENT_GROUP,
        "get_dcp_local_seq_lens": _dcp_local_lens,
        "_split_paged_kv_cache": lambda cache: (cache[:, 0], cache[:, 1]),
        "_extract_contiguous_kv_from_paged_cache": _extract_kv,
        "_dequantize_fp8_contiguous_kv": lambda k, v, *_args: (k, v),
        "_normalize_query_start_loc_for_available_tokens": lambda qsl, _n: qsl,
        "cp_lse_ag_out_rs": _cp_lse_combine,
        "dcp_a2a_lse_reduce": _a2a_lse_combine,
        "dcp_query_all_gather": _query_all_gather,
        "_decode_xqa_allowed_for_q_per_kv": lambda q_per_kv, _metadata: (
            q_per_kv in (6, 8)
        ),
        "merge_attn_states": _merge_states,
        "_record_route": _RECORDED_ROUTES.append,
        "_is_cuda_graph_capturing": lambda _query: False,
    }


def _load_method(name: str):
    tree = ast.parse(BACKEND.read_text())
    impl = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlashAttnV100Impl"
    )
    method = next(
        node
        for node in impl.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    method.decorator_list = []
    method.returns = None
    for arg in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = _namespace()
    exec(compile(module, str(BACKEND), "exec"), namespace)
    return namespace[name]


def _load_top_level_function(path: Path, name: str):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    function.returns = None
    for arg in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ):
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {
        "os": os,
        "torch": torch,
        "_dcp_query_gather_buffer_cache": {},
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class _StubImpl:
    dcp_world_size = 2
    dcp_comm_backend = "ag_rs"
    _dcp_decode_profile_enabled = False
    smallq_decode_max_query_len = 16
    cp_kv_cache_interleave_size = 1
    kv_cache_dtype = "auto"
    scale = 0.5
    logits_soft_cap = 0.0
    alibi_slopes = None
    flash_attn_func = staticmethod(_dense_attention)

    @staticmethod
    def flash_attn_prefill_paged(q, *_args, **kwargs):
        if isinstance(_CURRENT_GROUP, _FakeBatchGroup):
            outputs = []
            lses = []
            if q.shape[0] == 1:
                seq_idx = _CURRENT_GROUP.next_fallback_seq
                batch_rows = [(q[0], seq_idx)]
            else:
                batch_rows = [(q_seq, i) for i, q_seq in enumerate(q)]
            for q_seq, seq_idx in batch_rows:
                local_k = _CURRENT_GROUP.local_ks[seq_idx]
                local_v = _CURRENT_GROUP.local_vs[seq_idx]
                if local_k.shape[0] == 0:
                    outputs.append(torch.zeros_like(q_seq))
                    lses.append(
                        torch.full(
                            (q_seq.shape[1], q_seq.shape[0]),
                            -torch.inf,
                            dtype=torch.float32,
                        )
                    )
                else:
                    out, lse, _ = _dense_attention(
                        q_seq.unsqueeze(0),
                        local_k.unsqueeze(0),
                        local_v.unsqueeze(0),
                        causal=kwargs["causal"],
                        softmax_scale=kwargs["softmax_scale"],
                    )
                    outputs.append(out.squeeze(0))
                    lses.append(lse.squeeze(0))
            assert kwargs["return_lse"]
            return torch.stack(outputs), torch.stack(lses)

        out, lse, _ = _dense_attention(
            q,
            _CURRENT_GROUP.local_k.unsqueeze(0),
            _CURRENT_GROUP.local_v.unsqueeze(0),
            causal=kwargs["causal"],
            softmax_scale=kwargs["softmax_scale"],
        )
        assert kwargs["return_lse"]
        return out, lse

    def _dcp_decode_profile_range(self, _stage):
        raise AssertionError("profile ranges are disabled in CPU prefill tests")

    @staticmethod
    def _flash_v100_window_size(_causal):
        return (-1, -1)


class _RecordingRange:
    def __init__(self, stages, stage):
        self.stages = stages
        self.stage = stage

    def __enter__(self):
        self.stages.append(self.stage)

    def __exit__(self, *_args):
        return False


class _StubDecodeImpl:
    dcp_world_size = 2
    cp_kv_cache_interleave_size = 1
    kv_cache_dtype = "auto"
    scale = 0.5
    use_decode_xqa = False
    flash_attn_decode_paged_xqa = None
    _flash_decode_paged_xqa_kwargs = {"return_lse", "dcp_trace_range"}
    _get_dcp_decode_output_workspace = _load_method("_get_dcp_decode_output_workspace")
    _dcp_decode_xqa_eligible = _load_method("_dcp_decode_xqa_eligible")

    def __init__(self, rank, profile=False, backend="ag_rs", use_xqa=False):
        self.dcp_rank = rank
        self.dcp_comm_backend = backend
        self._dcp_decode_profile_enabled = profile
        self._dcp_decode_output_workspaces = {}
        self.profile_stages = []
        self.scalar_calls = 0
        self.xqa_calls = 0
        self.use_decode_xqa = use_xqa
        if use_xqa:
            self.flash_attn_decode_paged_xqa = self._flash_attn_decode_paged_xqa

    def _dcp_decode_profile_range(self, stage):
        return _RecordingRange(self.profile_stages, stage)

    @staticmethod
    def _flash_v100_window_size(*, causal):
        assert causal
        return (-1, -1)

    def _decode_reference(self, query, seq_lens, kwargs, stage):
        assert kwargs["return_lse"]
        assert int(seq_lens[0]) == self.expected_local_len
        trace_range = kwargs.get("dcp_trace_range")
        if trace_range is None:
            out, lse, _ = _dense_attention(
                query.unsqueeze(0),
                _CURRENT_GROUP.local_k.unsqueeze(0),
                _CURRENT_GROUP.local_v.unsqueeze(0),
                causal=False,
                softmax_scale=self.scale,
            )
        else:
            with trace_range(stage):
                out, lse, _ = _dense_attention(
                    query.unsqueeze(0),
                    _CURRENT_GROUP.local_k.unsqueeze(0),
                    _CURRENT_GROUP.local_v.unsqueeze(0),
                    causal=False,
                    softmax_scale=self.scale,
                )
        kwargs["out"].copy_(out.squeeze(0))
        return kwargs["out"], lse.squeeze(0).transpose(0, 1).contiguous()

    def _call_flash_attn_decode_paged(
        self,
        query,
        _key_cache,
        _value_cache,
        _block_table,
        seq_lens,
        **kwargs,
    ):
        self.scalar_calls += 1
        return self._decode_reference(query, seq_lens, kwargs, "local_attention")

    def _flash_attn_decode_paged_xqa(
        self,
        query,
        _key_cache,
        _value_cache,
        _block_table,
        seq_lens,
        **kwargs,
    ):
        self.xqa_calls += 1
        return self._decode_reference(
            query,
            seq_lens,
            kwargs,
            "local_attention_xqa",
        )


class _Metadata:
    causal = True


class _Layer:
    _k_scale_float = 1.0
    _v_scale_float = 1.0


def _run_prefill_case(method, prefix_len, backend="ag_rs"):
    global _CURRENT_GROUP
    torch.manual_seed(7 + prefix_len)
    query_len, total_heads, local_heads, dim = 3, 4, 2, 4
    q_all = torch.randn(query_len, total_heads, dim)
    prefix_k = torch.randn(prefix_len, 1, dim)
    prefix_v = torch.randn(prefix_len, 1, dim)
    suffix_k = torch.randn(query_len, 1, dim)
    suffix_v = torch.randn(query_len, 1, dim)
    rank_outputs = []

    for rank in range(2):
        owned = torch.arange(prefix_len) % 2 == rank
        local_k, local_v = prefix_k[owned], prefix_v[owned]
        cache = torch.zeros(1, 2, max(prefix_len, 1), 1, dim)
        cache[0, 0, : local_k.shape[0]] = local_k
        cache[0, 1, : local_v.shape[0]] = local_v
        metadata = _Metadata()
        metadata.num_actual_tokens = query_len
        metadata.query_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
        metadata.query_start_loc_cpu = metadata.query_start_loc
        metadata.seq_lens = torch.tensor([prefix_len + query_len], dtype=torch.int32)
        metadata.block_table = torch.zeros(1, 1, dtype=torch.int32)
        _CURRENT_GROUP = _FakeGroup(
            rank, q_all, prefix_k, prefix_v, local_k, local_v, _StubImpl.scale
        )
        impl = _StubImpl()
        impl.dcp_rank = rank
        impl.dcp_comm_backend = backend
        output = torch.empty(query_len, local_heads, dim)
        method(
            impl,
            _Layer(),
            q_all[:, rank * local_heads : (rank + 1) * local_heads],
            suffix_k,
            suffix_v,
            cache,
            metadata,
            output,
        )
        assert _CURRENT_GROUP.combine_calls == 1
        rank_outputs.append(output)

    actual = torch.cat(rank_outputs, dim=1)
    full_k = torch.cat((prefix_k, suffix_k), dim=0)
    full_v = torch.cat((prefix_v, suffix_v), dim=0)
    expected, _, _ = _dense_attention(
        q_all.unsqueeze(0),
        full_k.unsqueeze(0),
        full_v.unsqueeze(0),
        causal=True,
        softmax_scale=_StubImpl.scale,
    )
    torch.testing.assert_close(actual, expected.squeeze(0), rtol=1e-5, atol=1e-5)


def _run_multi_request_prefill_case(
    method,
    query_lens,
    prefix_lens,
    backend="a2a",
):
    global _CURRENT_GROUP
    torch.manual_seed(97 + sum(query_lens) + sum(prefix_lens))
    total_heads, local_heads, dim = 4, 2, 4
    q_all = [torch.randn(q_len, total_heads, dim) for q_len in query_lens]
    prefix_ks = [torch.randn(length, 1, dim) for length in prefix_lens]
    prefix_vs = [torch.randn(length, 1, dim) for length in prefix_lens]
    suffix_ks = [torch.randn(q_len, 1, dim) for q_len in query_lens]
    suffix_vs = [torch.randn(q_len, 1, dim) for q_len in query_lens]
    query_flat = torch.cat(q_all)
    key_flat = torch.cat(suffix_ks)
    value_flat = torch.cat(suffix_vs)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    rank_outputs = []
    rank_combine_calls = []
    _RECORDED_ROUTES.clear()

    for rank in range(2):
        local_ks = []
        local_vs = []
        for prefix_k, prefix_v in zip(prefix_ks, prefix_vs):
            owned = torch.arange(prefix_k.shape[0]) % 2 == rank
            local_ks.append(prefix_k[owned])
            local_vs.append(prefix_v[owned])
        max_local_len = max(1, *(local_k.shape[0] for local_k in local_ks))
        cache = torch.zeros(1, 2, max_local_len, 1, dim)
        metadata = _Metadata()
        metadata.num_actual_tokens = query_flat.shape[0]
        metadata.query_start_loc = query_start_loc
        metadata.query_start_loc_cpu = query_start_loc
        metadata.seq_lens = torch.tensor(
            [prefix_len + q_len for prefix_len, q_len in zip(prefix_lens, query_lens)],
            dtype=torch.int32,
        )
        metadata.block_table = torch.zeros(len(query_lens), 1, dtype=torch.int32)
        _CURRENT_GROUP = _FakeBatchGroup(
            rank,
            q_all,
            prefix_ks,
            prefix_vs,
            local_ks,
            local_vs,
            _StubImpl.scale,
        )
        impl = _StubImpl()
        impl.dcp_rank = rank
        impl.dcp_comm_backend = backend
        output = torch.empty(query_flat.shape[0], local_heads, dim)
        method(
            impl,
            _Layer(),
            query_flat[:, rank * local_heads : (rank + 1) * local_heads],
            key_flat,
            value_flat,
            cache,
            metadata,
            output,
        )
        rank_combine_calls.append(_CURRENT_GROUP.combine_calls)
        rank_outputs.append(output)

    actual = torch.cat(rank_outputs, dim=1)
    expected = []
    for q_seq, prefix_k, prefix_v, suffix_k, suffix_v in zip(
        q_all,
        prefix_ks,
        prefix_vs,
        suffix_ks,
        suffix_vs,
    ):
        out, _, _ = _dense_attention(
            q_seq.unsqueeze(0),
            torch.cat((prefix_k, suffix_k)).unsqueeze(0),
            torch.cat((prefix_v, suffix_v)).unsqueeze(0),
            causal=True,
            softmax_scale=_StubImpl.scale,
        )
        expected.append(out.squeeze(0))
    torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-5)
    return rank_combine_calls, list(_RECORDED_ROUTES)


def _run_decode_case(
    method,
    prefix_len,
    profile=False,
    backend="ag_rs",
    use_xqa=False,
):
    global _CURRENT_GROUP
    torch.manual_seed(31 + prefix_len)
    total_heads, local_heads, dim = (6, 3, 256) if use_xqa else (4, 2, 4)
    dtype = torch.float16 if use_xqa else torch.float32
    q_all = torch.randn(1, total_heads, dim, dtype=dtype)
    prefix_k = torch.randn(prefix_len, 1, dim, dtype=dtype)
    prefix_v = torch.randn(prefix_len, 1, dim, dtype=dtype)
    rank_outputs = []
    rank_profile_stages = []
    _RECORDED_ROUTES.clear()

    for rank in range(2):
        owned = torch.arange(prefix_len) % 2 == rank
        local_k, local_v = prefix_k[owned], prefix_v[owned]
        cache = torch.zeros(
            1,
            2,
            max(prefix_len, 1),
            1,
            dim,
            dtype=dtype,
        )
        _CURRENT_GROUP = _FakeGroup(
            rank, q_all, prefix_k, prefix_v, local_k, local_v, _StubImpl.scale
        )
        metadata = _Metadata()
        metadata.num_actual_tokens = 1
        metadata.max_query_len = 1
        metadata.seq_lens = torch.tensor([prefix_len], dtype=torch.int32)
        metadata.block_table = torch.zeros(1, 1, dtype=torch.int32)
        impl = _StubDecodeImpl(
            rank,
            profile=profile,
            backend=backend,
            use_xqa=use_xqa,
        )
        impl.expected_local_len = local_k.shape[0]
        output = torch.empty(1, local_heads, dim, dtype=dtype)
        method(
            impl,
            _Layer(),
            q_all[:, rank * local_heads : (rank + 1) * local_heads],
            torch.empty(0),
            torch.empty(0),
            cache,
            metadata,
            output,
        )
        assert _CURRENT_GROUP.combine_calls == 1
        if use_xqa:
            assert impl.xqa_calls == 1
            assert impl.scalar_calls == 0
        else:
            assert impl.xqa_calls == 0
            assert impl.scalar_calls == 1
        rank_outputs.append(output)
        rank_profile_stages.append(impl.profile_stages)

    actual = torch.cat(rank_outputs, dim=1)
    expected, _, _ = _dense_attention(
        q_all.unsqueeze(0),
        prefix_k.unsqueeze(0),
        prefix_v.unsqueeze(0),
        causal=False,
        softmax_scale=_StubImpl.scale,
    )
    torch.testing.assert_close(actual, expected.squeeze(0), rtol=1e-5, atol=1e-5)
    return rank_profile_stages


def test_rank_specific_dcp_local_seq_lens():
    get_local_lens = _load_top_level_function(ATTENTION_UTILS, "get_dcp_local_seq_lens")
    seq_lens = torch.tensor([0, 1, 2, 3, 7, 8, 9], dtype=torch.int64)
    rank0 = get_local_lens(seq_lens, 2, 0, 2)
    rank1 = get_local_lens(seq_lens, 2, 1, 2)
    expected_rank0 = torch.tensor([0, 1, 2, 2, 4, 4, 5], dtype=torch.int32)
    expected_rank1 = torch.tensor([0, 0, 0, 1, 3, 4, 4], dtype=torch.int32)
    torch.testing.assert_close(rank0, expected_rank0)
    torch.testing.assert_close(rank1, expected_rank1)
    torch.testing.assert_close(rank0 + rank1, seq_lens.to(torch.int32))


def test_dcp_prefill_matches_dense_attention():
    source = BACKEND.read_text()
    assert "from vllm.distributed.parallel_state import get_dcp_group" in source
    assert "from vllm.v1.attention.ops.common import cp_lse_ag_out_rs" in source
    assert "return_attn_probs=True" in source
    method = _load_method("_forward_with_dcp")
    _run_prefill_case(method, prefix_len=0)
    _run_prefill_case(method, prefix_len=5)
    _run_prefill_case(method, prefix_len=1)


def test_dcp_prefix_prefill_a2a_matches_dense_attention():
    method = _load_method("_forward_with_dcp")
    _run_prefill_case(method, prefix_len=5, backend="a2a")
    _run_prefill_case(method, prefix_len=1, backend="a2a")


def test_dcp_uniform_mtp4_batch_combines_once_with_uneven_empty_contexts():
    method = _load_method("_forward_with_dcp")
    combine_calls, routes = _run_multi_request_prefill_case(
        method,
        query_lens=[5, 5, 5, 5],
        prefix_lens=[0, 1, 6, 9],
        backend="a2a",
    )
    assert combine_calls == [1, 1]
    assert routes.count("prefill_prefix_dcp_uniform_batch_a2a") == 2
    assert routes.count("prefill_prefix_dcp_uniform_batch") == 2
    assert "prefill_prefix_dcp_per_request_fallback" not in routes


def test_dcp_uniform_batch_ag_rs_matches_dense_attention():
    method = _load_method("_forward_with_dcp")
    combine_calls, routes = _run_multi_request_prefill_case(
        method,
        query_lens=[4, 4, 4],
        prefix_lens=[2, 7, 0],
        backend="ag_rs",
    )
    assert combine_calls == [1, 1]
    assert routes.count("prefill_prefix_dcp_uniform_batch_ag_rs") == 2


def test_dcp_irregular_query_lengths_use_per_request_fallback():
    method = _load_method("_forward_with_dcp")
    combine_calls, routes = _run_multi_request_prefill_case(
        method,
        query_lens=[5, 3, 4],
        prefix_lens=[0, 7, 2],
        backend="a2a",
    )
    assert combine_calls == [3, 3]
    assert routes.count("prefill_prefix_dcp_per_request_fallback") == 2
    assert routes.count("prefill_prefix_dcp_a2a") == 6
    assert "prefill_prefix_dcp_uniform_batch" not in routes


def test_dense_flash_exposes_existing_lse():
    source = FUSED_MHA_FORWARD.read_text()
    assert 'TORCH_CHECK(!return_softmax, "return_softmax not supported")' not in source
    assert "return {out_fp16, softmax_lse, p, rng_state};" in source


def test_paged_prefill_exposes_existing_lse():
    source = (
        ROOT / "flash-attention-v100/kernel/fused_mha_forward_paged.cu"
    ).read_text()
    assert "std::vector<at::Tensor> flash_attention_prefill_paged(" in source
    assert "return {out_fp16, softmax_lse};" in source


def test_decode_interface_uses_direct_reduction_lse():
    interface_source = FLASH_INTERFACE.read_text()
    kernel_source = FLASH_DECODE.read_text()

    assert "dcp_trace_range" in interface_source
    assert 'dcp_trace_range("local_attention")' in interface_source
    assert 'dcp_trace_range("local_attention_xqa")' in interface_source
    assert 'dcp_trace_range("lse_reconstruction")' not in interface_source
    assert "_decode_lse_from_workspace" not in interface_source
    assert "final_lse if return_lse else None" in interface_source
    assert "return attn_out, final_lse" in interface_source

    assert "float* __restrict__ final_lse" in kernel_source
    assert "global_max + logf(global_sum)" in kernel_source
    assert "global_sum > 0.f" in kernel_source
    assert "kNegativeInfinity" in kernel_source
    assert kernel_source.count("final_lse->stride(0)") == 2
    assert kernel_source.count("final_lse->stride(1)") == 2


def test_dcp_decode_matches_dense_attention():
    method = _load_method("_flash_v100_decode")
    _run_decode_case(method, prefix_len=5)
    _run_decode_case(method, prefix_len=1)


def test_dcp_decode_a2a_matches_dense_attention():
    method = _load_method("_flash_v100_decode")
    _run_decode_case(method, prefix_len=5, backend="a2a")
    _run_decode_case(method, prefix_len=1, backend="a2a")
    assert _RECORDED_ROUTES.count("decode_scalar_paged_dcp_fallback") == 2
    assert _RECORDED_ROUTES.count("decode_scalar_paged_dcp_a2a") == 2
    assert "decode_xqa_paged_dcp" not in _RECORDED_ROUTES


def test_dcp_q1_xqa_q_per_kv6_matches_dense_attention_and_uses_a2a():
    method = _load_method("_flash_v100_decode")
    _run_decode_case(method, prefix_len=6, backend="a2a", use_xqa=True)
    _run_decode_case(method, prefix_len=1, backend="a2a", use_xqa=True)
    assert _RECORDED_ROUTES.count("decode_xqa_paged_dcp") == 2
    assert _RECORDED_ROUTES.count("decode_xqa_paged_dcp_a2a") == 2
    assert _RECORDED_ROUTES.count("decode_xqa_paged_dcp_complete") == 2
    assert "decode_scalar_paged_dcp_fallback" not in _RECORDED_ROUTES


def test_dcp_xqa_selection_excludes_q_greater_than_one_and_bad_shapes():
    eligible = _load_method("_dcp_decode_xqa_eligible")
    impl = _StubDecodeImpl(rank=0, use_xqa=True)
    key_cache = torch.empty(1, 4, 1, 256, dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)
    metadata = _Metadata()
    metadata.max_query_len = 1
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)

    assert eligible(
        impl,
        torch.empty(2, 6, 256, dtype=torch.float16),
        key_cache,
        value_cache,
        metadata,
        (-1, -1),
    )

    metadata.max_query_len = 5
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    assert not eligible(
        impl,
        torch.empty(10, 6, 256, dtype=torch.float16),
        key_cache,
        value_cache,
        metadata,
        (-1, -1),
    )

    metadata.max_query_len = 1
    assert not eligible(
        impl,
        torch.empty(2, 5, 256, dtype=torch.float16),
        key_cache,
        value_cache,
        metadata,
        (-1, -1),
    )
    assert not eligible(
        impl,
        torch.empty(2, 6, 256, dtype=torch.float16),
        key_cache,
        value_cache,
        metadata,
        (128, -1),
    )


def test_dcp_decode_rejects_unknown_communication_backend():
    method = _load_method("_flash_v100_decode")
    try:
        _run_decode_case(method, prefix_len=5, backend="invalid")
    except RuntimeError as error:
        assert "unsupported communication backend" in str(error)
    else:
        raise AssertionError("unknown DCP communication backend was accepted")


def test_dcp_decode_nvtx_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_FLASH_V100_DCP_DECODE_NVTX", raising=False)
    enabled = _load_top_level_function(BACKEND, "_dcp_decode_nvtx_enabled")
    assert not enabled()
    monkeypatch.setenv("VLLM_FLASH_V100_DCP_DECODE_NVTX", "1")
    assert enabled()


def test_dcp_decode_output_workspace_reuses_stable_storage():
    impl = _StubDecodeImpl(rank=0)
    reference = torch.empty(3, 4, 5)
    first = impl._get_dcp_decode_output_workspace(reference)
    second = impl._get_dcp_decode_output_workspace(reference)
    assert first is second
    assert first.data_ptr() == second.data_ptr()

    different_shape = impl._get_dcp_decode_output_workspace(torch.empty(2, 4, 5))
    assert different_shape is not first
    assert len(impl._dcp_decode_output_workspaces) == 2


def test_dcp_query_gather_reformat_preserves_rank_head_order():
    reformat = _load_top_level_function(DCP_QUERY_GATHER, "_reformat_rank_major_query")
    world_size, tokens, local_heads, head_dim = 2, 3, 2, 4
    rank_major = torch.empty(world_size, tokens, local_heads, head_dim)
    for rank in range(world_size):
        for token in range(tokens):
            for head in range(local_heads):
                rank_major[rank, token, head].fill_(100 * rank + 10 * token + head)
    head_major = torch.empty(tokens, world_size * local_heads, head_dim)
    actual = reformat(rank_major, head_major)
    expected = torch.cat([rank_major[rank] for rank in range(world_size)], dim=1)
    assert actual is head_major
    torch.testing.assert_close(actual, expected)


def test_dcp_query_gather_workspace_is_stable_separate_and_non_aliasing():
    get_buffers = _load_top_level_function(
        DCP_QUERY_GATHER, "_dcp_query_gather_buffers"
    )
    query = torch.empty(5, 3, 8)
    first = get_buffers(query, 2, "dcp:test")
    second = get_buffers(query, 2, "dcp:test")
    assert all(left is right for left, right in zip(first, second))

    spans = sorted(
        (
            tensor.data_ptr(),
            tensor.data_ptr() + tensor.numel() * tensor.element_size(),
        )
        for tensor in first
    )
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))

    different_shape = get_buffers(torch.empty(4, 3, 8), 2, "dcp:test")
    strided = torch.empty(5, 3, 16)[..., ::2]
    different_stride = get_buffers(strided, 2, "dcp:test")
    different_group = get_buffers(query, 2, "dcp:other")
    assert different_shape[0] is not first[0]
    assert different_stride[0] is not first[0]
    assert different_group[0] is not first[0]


def test_dcp_query_gather_source_has_direct_and_fail_closed_fallback():
    source = DCP_QUERY_GATHER.read_text()
    assert "pynccl.all_gather(rank_major, prepared)" in source
    assert "query_all_gather_direct bytes=" in source
    assert "query_gather_reformat bytes=" in source
    assert "query_gather_cache_acquire" in source
    assert "query_all_gather_fallback bytes=" in source
    assert "cp_group.all_gather(prepared, dim=1)" in source
    assert "torch.compiler.is_compiling()" in source
    assert "_is_fake_tensor(query)" in source


def test_dcp_decode_profile_covers_hot_path_without_changing_output():
    method = _load_method("_flash_v100_decode")
    rank_stages = _run_decode_case(method, prefix_len=5, profile=True)
    expected_stages = {
        "local_attention",
        "lse_prepare",
        "output_correction",
        "output_workspace_acquire",
    }
    for stages in rank_stages:
        assert expected_stages.issubset(stages)
        assert any(
            stage.startswith("query_gather_prepare_fallback bytes=") for stage in stages
        )
        assert any(
            stage.startswith("query_all_gather_fallback bytes=") for stage in stages
        )
        assert any(stage.startswith("lse_all_gather bytes=") for stage in stages)
        assert any(stage.startswith("output_reduce_scatter bytes=") for stage in stages)
        assert any(stage.startswith("output_copy bytes=") for stage in stages)
        assert not any(stage.startswith("a2a_") for stage in stages)


def test_dcp_decode_a2a_profile_has_one_packed_collective():
    method = _load_method("_flash_v100_decode")
    rank_stages = _run_decode_case(method, prefix_len=5, profile=True, backend="a2a")
    for stages in rank_stages:
        assert any(
            stage.startswith("query_all_gather_fallback bytes=") for stage in stages
        )
        assert any(stage.startswith("a2a_pack bytes=") for stage in stages)
        assert any(stage.startswith("a2a_all_to_all bytes=") for stage in stages)
        assert any(stage.startswith("a2a_unpack bytes=") for stage in stages)
        assert any(stage.startswith("output_copy bytes=") for stage in stages)
        assert not any(stage.startswith("lse_all_gather") for stage in stages)
        assert not any(stage.startswith("output_reduce_scatter") for stage in stages)


def test_dcp_q1_xqa_profile_proves_xqa_and_packed_a2a_routes():
    method = _load_method("_flash_v100_decode")
    rank_stages = _run_decode_case(
        method,
        prefix_len=7,
        profile=True,
        backend="a2a",
        use_xqa=True,
    )
    for stages in rank_stages:
        assert "local_attention_xqa" in stages
        assert "local_attention" not in stages
        assert any(stage.startswith("a2a_all_to_all bytes=") for stage in stages)
    assert _RECORDED_ROUTES.count("decode_xqa_paged_dcp") == 2
    assert _RECORDED_ROUTES.count("decode_xqa_paged_dcp_a2a") == 2
