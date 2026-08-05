# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import benchmark_hy3_max_context_admission as benchmark


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _argv_sha256(argv: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(argv, separators=(",", ":")).encode("ascii")
    ).hexdigest()


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


def _provenance_files(tmp_path: Path) -> dict[str, Path]:
    return {
        "server": _write_json(
            tmp_path / "server.json",
            {
                "deployment_uid": "deployment-uid",
                "pod_uid": "pod-uid",
                "image": "localhost:32000/hy3:test",
                "image_id": "docker-pullable://hy3@sha256:" + "1" * 64,
                "served_model": "cyankiwi/Hy3-AWQ-INT4",
                "server_started_at": "2026-07-20T00:00:00Z",
                "ready": True,
                "restart_count": 0,
            },
        ),
        "source": _write_json(
            tmp_path / "source.json",
            {
                "git_commit": "2" * 40,
                "git_tree": "3" * 40,
                "git_dirty": False,
                "source_snapshot_sha256": "4" * 64,
            },
        ),
        "config": _write_json(
            tmp_path / "config.json",
            {
                "runtime_config_id": "hy3-admission-a1",
                "max_model_len": 65_536,
                "max_num_seqs": 2,
                "tensor_parallel_size": 8,
                "pipeline_parallel_size": 1,
                "kv_cache_dtype": "fp16",
                "physical_kv_cache_tokens": 128_514,
                "max_num_batched_tokens": 4096,
                "max_num_partial_prefills": 2,
                "max_long_partial_prefills": 2,
                "long_prefill_token_threshold": 2048,
                "compilation_config_mode": 3,
            },
        ),
        "argv": _write_json(
            tmp_path / "argv.json",
            {
                "argv": ["vllm", "serve", "cyankiwi/Hy3-AWQ-INT4"],
                "argv_sha256": _argv_sha256(["vllm", "serve", "cyankiwi/Hy3-AWQ-INT4"]),
            },
        ),
        "gpu": _write_json(
            tmp_path / "gpu.json",
            {"gpu_uuids": [f"GPU-{index:032x}" for index in range(8)]},
        ),
        "jit": _write_json(
            tmp_path / "jit.json",
            {
                "checked_at": "2026-07-20T00:00:00Z",
                "log_sha256": "6" * 64,
                "unexpected_jit": False,
            },
        ),
    }


def _dry_run_argv(tmp_path: Path) -> list[str]:
    prompt_a = _prompt(
        tmp_path / "prompt-a.json",
        source="manual",
        domain="systems",
        offset=1,
    )
    prompt_b = _prompt(
        tmp_path / "prompt-b.json",
        source="case-law",
        domain="legal",
        offset=2,
    )
    provenance = _provenance_files(tmp_path)
    return [
        "--base-url",
        "http://hy3.test",
        "--model",
        "cyankiwi/Hy3-AWQ-INT4",
        "--prompt-token-json",
        str(prompt_a),
        "--prompt-token-json",
        str(prompt_b),
        "--server-provenance-json",
        str(provenance["server"]),
        "--source-provenance-json",
        str(provenance["source"]),
        "--config-provenance-json",
        str(provenance["config"]),
        "--argv-provenance-json",
        str(provenance["argv"]),
        "--gpu-provenance-json",
        str(provenance["gpu"]),
        "--jit-provenance-json",
        str(provenance["jit"]),
        "--pvc-root",
        str(tmp_path),
        "--output-root",
        str(tmp_path),
        "--run-id",
        "admission-a1",
        "--dry-run",
    ]


def test_dry_run_records_fixed_two_slot_contract(tmp_path: Path, capsys) -> None:
    assert benchmark.main(_dry_run_argv(tmp_path)) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["contract"]["prompt_token_count_per_slot"] == 64_001
    assert plan["contract"]["output_token_count_per_slot"] == 1
    assert plan["contract"]["metrics_interval_seconds"] == 0.05
    assert plan["contract"]["required_running"] == 2
    assert len(plan["provenance"]["gpu"]["value"]["gpu_uuids"]) == 8


def test_admission_proof_rejects_queue_or_single_running_sample() -> None:
    records = [
        {
            "request_start_monotonic_ns": 100,
            "request_end_monotonic_ns": 300,
            "cache_salt_sha256": "a",
        },
        {
            "request_start_monotonic_ns": 110,
            "request_end_monotonic_ns": 290,
            "cache_salt_sha256": "b",
        },
    ]
    clean_samples = [
        {"sampled_at_monotonic_ns": 120, "running": 2.0, "waiting": 0.0, "error": None},
        {"sampled_at_monotonic_ns": 200, "running": 2.0, "waiting": 0.0, "error": None},
    ]
    proof = benchmark._admission_proof(clean_samples, records)
    assert proof["passed"] is True
    assert proof["active_interval_sample_count"] == 2

    queued = [
        *clean_samples,
        {
            "sampled_at_monotonic_ns": 150,
            "running": 1.0,
            "waiting": 1.0,
            "error": None,
        },
    ]
    with pytest.raises(benchmark.ContractError, match="continuously admitted"):
        benchmark._admission_proof(queued, records)


def test_config_provenance_rejects_insufficient_kv(tmp_path: Path) -> None:
    paths = _provenance_files(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["physical_kv_cache_tokens"] = benchmark.MIN_PHYSICAL_KV_TOKENS - 1
    _write_json(paths["config"], config)
    provenance = {
        kind: benchmark._load_provenance(path, kind) for kind, path in paths.items()
    }

    with pytest.raises(benchmark.ContractError, match="two-slot minimum"):
        benchmark._validate_admission_provenance(provenance)


def test_exclusive_writer_refuses_an_artifact_collision(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    benchmark._write_json_exclusive(path, {"first": True})

    with pytest.raises(FileExistsError):
        benchmark._write_json_exclusive(path, {"second": True})
