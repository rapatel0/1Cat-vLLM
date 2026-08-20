# SPDX-License-Identifier: Apache-2.0
"""CPU numerical checks for the Flash-V100 DCP attention integration."""

from __future__ import annotations

import ast
from pathlib import Path

import torch

ROOT = Path(__file__).parents[3]
BACKEND = ROOT / "vllm/v1/attention/backends/flash_attn_v100.py"
FLASH_INTERFACE = ROOT / "flash-attention-v100/flash_attn_v100/flash_attn_interface.py"
ATTENTION_UTILS = ROOT / "vllm/v1/attention/backends/utils.py"
FUSED_MHA_FORWARD = ROOT / "flash-attention-v100/kernel/fused_mha_forward.cu"


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


def _cp_lse_combine(out, lse, group, return_lse=False):
    result = group.combine(out, lse, True)
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
        "merge_attn_states": _merge_states,
        "_record_route": lambda _route: None,
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
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class _StubImpl:
    dcp_world_size = 2
    cp_kv_cache_interleave_size = 1
    kv_cache_dtype = "auto"
    scale = 0.5
    logits_soft_cap = 0.0
    alibi_slopes = None
    flash_attn_func = staticmethod(_dense_attention)

    @staticmethod
    def flash_attn_prefill_paged(q, *_args, **kwargs):
        out, lse, _ = _dense_attention(
            q,
            _CURRENT_GROUP.local_k.unsqueeze(0),
            _CURRENT_GROUP.local_v.unsqueeze(0),
            causal=kwargs["causal"],
            softmax_scale=kwargs["softmax_scale"],
        )
        assert kwargs["return_lse"]
        return out, lse

    @staticmethod
    def _flash_v100_window_size(_causal):
        return (-1, -1)


class _StubDecodeImpl:
    dcp_world_size = 2
    cp_kv_cache_interleave_size = 1
    kv_cache_dtype = "auto"
    scale = 0.5
    use_decode_xqa = False

    def __init__(self, rank):
        self.dcp_rank = rank

    @staticmethod
    def _flash_v100_window_size(*, causal):
        assert causal
        return (-1, -1)

    def _call_flash_attn_decode_paged(
        self,
        query,
        _key_cache,
        _value_cache,
        _block_table,
        seq_lens,
        **kwargs,
    ):
        assert kwargs["return_lse"]
        assert int(seq_lens[0]) == self.expected_local_len
        out, lse, _ = _dense_attention(
            query.unsqueeze(0),
            _CURRENT_GROUP.local_k.unsqueeze(0),
            _CURRENT_GROUP.local_v.unsqueeze(0),
            causal=False,
            softmax_scale=self.scale,
        )
        kwargs["out"].copy_(out.squeeze(0))
        return kwargs["out"], lse.squeeze(0).transpose(0, 1).contiguous()


class _Metadata:
    causal = True


class _Layer:
    _k_scale_float = 1.0
    _v_scale_float = 1.0


def _run_prefill_case(method, prefix_len):
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
        assert _CURRENT_GROUP.combine_calls == (1 if prefix_len else 0)
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


def _run_decode_case(method, prefix_len):
    global _CURRENT_GROUP
    torch.manual_seed(31 + prefix_len)
    total_heads, local_heads, dim = 4, 2, 4
    q_all = torch.randn(1, total_heads, dim)
    prefix_k = torch.randn(prefix_len, 1, dim)
    prefix_v = torch.randn(prefix_len, 1, dim)
    rank_outputs = []

    for rank in range(2):
        owned = torch.arange(prefix_len) % 2 == rank
        local_k, local_v = prefix_k[owned], prefix_v[owned]
        cache = torch.zeros(1, 2, max(prefix_len, 1), 1, dim)
        _CURRENT_GROUP = _FakeGroup(
            rank, q_all, prefix_k, prefix_v, local_k, local_v, _StubImpl.scale
        )
        metadata = _Metadata()
        metadata.num_actual_tokens = 1
        metadata.seq_lens = torch.tensor([prefix_len], dtype=torch.int32)
        metadata.block_table = torch.zeros(1, 1, dtype=torch.int32)
        impl = _StubDecodeImpl(rank)
        impl.expected_local_len = local_k.shape[0]
        output = torch.empty(1, local_heads, dim)
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
        rank_outputs.append(output)

    actual = torch.cat(rank_outputs, dim=1)
    expected, _, _ = _dense_attention(
        q_all.unsqueeze(0),
        prefix_k.unsqueeze(0),
        prefix_v.unsqueeze(0),
        causal=False,
        softmax_scale=_StubImpl.scale,
    )
    torch.testing.assert_close(actual, expected.squeeze(0), rtol=1e-5, atol=1e-5)


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


def test_decode_workspace_lse_reconstruction():
    reconstruct_lse = _load_top_level_function(
        FLASH_INTERFACE, "_decode_lse_from_workspace"
    )
    max_logits = torch.tensor(
        [
            [[2.0, -1.0, 99.0], [0.5, 3.0, -88.0]],
            [[-2.0, 1.0, 5.0], [4.0, -7.0, 12.0]],
        ]
    )
    exp_sums = torch.tensor(
        [
            [[3.0, 4.0, 1000.0], [2.0, 5.0, 0.0]],
            [[1.5, 2.5, 3.5], [4.0, 0.0, 9.0]],
        ]
    )
    seq_lens = torch.tensor([5, 2], dtype=torch.int32)
    actual = reconstruct_lse(max_logits, exp_sums, seq_lens, partition_size=2)

    expected = torch.empty(2, 2)
    expected[0] = torch.logsumexp(
        max_logits[0, :, :3] + torch.log(exp_sums[0, :, :3]), dim=-1
    )
    expected[1] = max_logits[1, :, 0] + torch.log(exp_sums[1, :, 0])
    torch.testing.assert_close(actual, expected)


def test_dcp_decode_matches_dense_attention():
    method = _load_method("_flash_v100_decode")
    _run_decode_case(method, prefix_len=5)
    _run_decode_case(method, prefix_len=1)
