# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from benchmarks import benchmark_hy3_max_context_service as benchmark


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _prompt(path: Path, *, source: str, domain: str, offset: int) -> Path:
    token_ids = [
        (index * 17 + offset) % 131_071 for index in range(benchmark.PROMPT_TOKEN_COUNT)
    ]
    return _write_json(
        path,
        {
            "schema_version": 1,
            "source_id": source,
            "semantic_domain": domain,
            "source_reference": f"https://example.test/{source}",
            "source_sha256": f"{offset:064x}",
            "prompt_token_ids": token_ids,
        },
    )


def _provenance_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    server = _write_json(
        tmp_path / "server.json",
        {
            "deployment_uid": "deployment-uid",
            "pod_uid": "pod-uid",
            "image": "localhost:32000/hy3:test",
            "image_id": "docker-pullable://hy3@sha256:" + "1" * 64,
            "served_model": "cyankiwi/Hy3-AWQ-INT4",
            "server_started_at": "2026-07-18T00:00:00Z",
        },
    )
    source = _write_json(
        tmp_path / "source.json",
        {
            "git_commit": "2" * 40,
            "git_tree": "3" * 40,
            "git_dirty": False,
            "source_snapshot_sha256": "4" * 64,
        },
    )
    config = _write_json(
        tmp_path / "config.json",
        {
            "config_id": "hy3-test",
            "deployment_spec_sha256": "5" * 64,
            "runtime_args_sha256": "6" * 64,
            "runtime_env_sha256": "7" * 64,
            "max_model_len": 65_536,
            "max_num_seqs": 2,
            "tensor_parallel_size": 8,
            "attention_backend": "TRITON_ATTN",
        },
    )
    return server, source, config


def test_prometheus_parser_sums_labeled_workers() -> None:
    parsed = benchmark._parse_prometheus_metrics(
        """
# HELP vllm:num_requests_running Number running.
vllm:num_requests_running{model_name="hy3",worker="0"} 1
vllm:num_requests_running{model_name="hy3",worker="1"} 1
vllm:num_requests_waiting{model_name="hy3"} 0
other_metric 99
"""
    )

    assert parsed == {
        benchmark.METRIC_RUNNING: 2.0,
        benchmark.METRIC_WAITING: 0.0,
    }


def test_batch2_metrics_proof_rejects_any_queue() -> None:
    clean_samples = [
        {"error": None, "running": 0.0, "waiting": 0.0},
        {"error": None, "running": 2.0, "waiting": 0.0},
        {"error": None, "running": 2.0, "waiting": 0.0},
    ]
    proof = benchmark._metrics_proof(clean_samples, expected_batch=2)
    assert proof["passed"] is True
    assert proof["simultaneously_running_not_waiting_samples"] == 2

    queued_samples = [
        *clean_samples,
        {"error": None, "running": 1.0, "waiting": 1.0},
    ]
    with pytest.raises(benchmark.ContractError, match="queued requests"):
        benchmark._metrics_proof(queued_samples, expected_batch=2)


def test_paired_statistics_cover_five_complete_repeats() -> None:
    groups = []
    for repeat in range(1, benchmark.PAIRED_REPEATS + 1):
        for slot in ("a", "b"):
            groups.append(
                {
                    "phase": "b1",
                    "repeat": repeat,
                    "records": [
                        {
                            "slot": slot,
                            "steady_decode_tokens_per_second": 10.0,
                        }
                    ],
                }
            )
        groups.append(
            {
                "phase": "b2",
                "repeat": repeat,
                "aggregate_steady_decode_tokens_per_second": 18.0,
                "records": [
                    {
                        "slot": slot,
                        "steady_decode_tokens_per_second": 9.0,
                    }
                    for slot in ("a", "b")
                ],
            }
        )

    result = benchmark._paired_statistics(groups)
    assert result["paired_b2_over_b1_per_slot"]["median"] == 0.9
    assert result["paired_b2_over_b1_per_slot"]["mad"] == 0.0
    assert result["paired_b2_aggregate_over_b1_reference"]["median"] == 1.8
    assert len(result["paired_repeats"]) == 5


def test_sse_decoder_handles_fragmented_crlf() -> None:
    decoder = benchmark.SSEDecoder()

    assert decoder.feed(b": ping\r\n\r\n") == []
    assert decoder.feed(b'data: {"choices":[{"index":0,') == []
    events = decoder.feed(b'"token_ids":[9]}]}\r\n\r\n')
    assert events == [{"choices": [{"index": 0, "token_ids": [9]}]}]
    assert decoder.feed(b"data: [DONE]\n\n") == []
    assert decoder.finish() == []


def test_sse_decoder_handles_split_utf8_code_point() -> None:
    decoder = benchmark.SSEDecoder()
    encoded = 'data: {"text":"r\u00e9sum\u00e9"}\n\n'.encode()
    split = encoded.index(b"\xc3") + 1

    assert decoder.feed(encoded[:split]) == []
    assert decoder.feed(encoded[split:]) == [{"text": "r\u00e9sum\u00e9"}]
    assert decoder.feed(b"data: [DONE]\n\n") == []
    assert decoder.finish() == []


def test_dry_run_validates_exact_contract_without_server(
    tmp_path: Path,
    capsys,
) -> None:
    prompt_a = _prompt(
        tmp_path / "prompt-a.json",
        source="kernel-manual",
        domain="systems",
        offset=1,
    )
    prompt_b = _prompt(
        tmp_path / "prompt-b.json",
        source="case-law",
        domain="legal",
        offset=2,
    )
    server, source, config = _provenance_files(tmp_path)

    result = benchmark.main(
        [
            "--base-url",
            "http://hy3.test",
            "--model",
            "cyankiwi/Hy3-AWQ-INT4",
            "--prompt-token-json",
            str(prompt_a),
            "--prompt-token-json",
            str(prompt_b),
            "--server-provenance-json",
            str(server),
            "--source-provenance-json",
            str(source),
            "--config-provenance-json",
            str(config),
            "--pvc-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "unused"),
            "--run-id",
            "dry-run",
            "--dry-run",
        ]
    )

    plan = json.loads(capsys.readouterr().out)
    assert result == 0
    assert plan["dry_run"] is True
    assert plan["contract"]["prompt_token_count_per_slot"] == 64_001
    assert plan["contract"]["output_token_count_per_slot"] == 256
    assert plan["contract"]["paired_repeats"] == 5
    assert plan["scored_request_count"] == 20
    assert [prompt["semantic_domain"] for prompt in plan["prompts"]] == [
        "systems",
        "legal",
    ]
