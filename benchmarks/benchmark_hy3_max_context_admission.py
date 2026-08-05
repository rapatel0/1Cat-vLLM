# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed two-slot admission probe for the Hy3 64K service campaign.

The probe is intentionally not a throughput benchmark. It proves that two
independent, realistic 64,001-token requests are concurrently admitted before
the five-repeat B1/B2 service harness is allowed to score a scheduler variant.
All inputs and outputs must reside below one mounted PVC; writes are exclusive
and are bound by a SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import contextlib
import hashlib
import json
import math
import os
import secrets
import shutil
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

PROMPT_TOKEN_COUNT = 64_001
OUTPUT_TOKEN_COUNT = 1
MAX_MODEL_LEN = 65_536
TENSOR_PARALLEL_SIZE = 8
PIPELINE_PARALLEL_SIZE = 1
MIN_PHYSICAL_KV_TOKENS = 128_514
METRICS_INTERVAL_SECONDS = 0.05
METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"


class ContractError(RuntimeError):
    """Raised when the admission contract cannot be proved."""


@dataclass(frozen=True)
class PromptSpec:
    slot: str
    source_id: str
    semantic_domain: str
    source_reference: str
    source_sha256: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    input_path: Path
    input_file_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "source_id": self.source_id,
            "semantic_domain": self.semantic_domain,
            "source_reference": self.source_reference,
            "source_sha256": self.source_sha256,
            "prompt_token_count": len(self.token_ids),
            "prompt_token_ids_sha256": self.token_ids_sha256,
            "input_path": str(self.input_path),
            "input_file_sha256": self.input_file_sha256,
        }


@dataclass(frozen=True)
class ProvenanceInput:
    kind: str
    path: Path
    file_sha256: str
    value: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "value": self.value,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_token_ids(token_ids: Sequence[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return _sha256_bytes(encoded)


def _is_lower_hex(value: Any, lengths: set[int]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_run_id(value: str) -> bool:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    return (
        1 <= len(value) <= 128
        and value[0].isascii()
        and value[0].isalnum()
        and all(character in allowed for character in value)
    )


def _require_fields(value: dict[str, Any], required: set[str], *, label: str) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise ContractError(f"{label} is missing required fields: {missing}")
    empty = sorted(key for key in required if value[key] is None or value[key] == "")
    if empty:
        raise ContractError(f"{label} has empty required fields: {empty}")


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ContractError(f"{label} does not exist or is not a file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain one JSON object: {path}")
    return value, _sha256_bytes(raw)


def _load_prompt(path: Path, slot: str) -> PromptSpec:
    value, file_sha256 = _load_json_object(path, label=f"prompt {slot}")
    required = {
        "schema_version",
        "source_id",
        "semantic_domain",
        "source_reference",
        "source_sha256",
        "prompt_token_ids",
    }
    _require_fields(value, required, label=f"prompt {slot}")
    if value["schema_version"] != 1:
        raise ContractError(f"prompt {slot} schema_version must be 1")
    if not _is_lower_hex(value["source_sha256"], {64}):
        raise ContractError(f"prompt {slot} source_sha256 must be lowercase SHA-256")
    token_ids = value["prompt_token_ids"]
    if not isinstance(token_ids, list):
        raise ContractError(f"prompt {slot} prompt_token_ids must be a JSON array")
    if len(token_ids) != PROMPT_TOKEN_COUNT:
        raise ContractError(
            f"prompt {slot} must contain exactly {PROMPT_TOKEN_COUNT:,} token IDs, "
            f"got {len(token_ids):,}"
        )
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id > 2**31 - 1
        for token_id in token_ids
    ):
        raise ContractError(
            f"prompt {slot} token IDs must be non-negative signed 32-bit integers"
        )
    for key in ("source_id", "semantic_domain", "source_reference"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ContractError(f"prompt {slot} {key} must be a non-empty string")
    immutable_ids = tuple(token_ids)
    return PromptSpec(
        slot=slot,
        source_id=value["source_id"].strip(),
        semantic_domain=value["semantic_domain"].strip(),
        source_reference=value["source_reference"].strip(),
        source_sha256=value["source_sha256"],
        token_ids=immutable_ids,
        token_ids_sha256=_sha256_token_ids(immutable_ids),
        input_path=path.resolve(),
        input_file_sha256=file_sha256,
    )


def _validate_prompt_independence(prompts: Sequence[PromptSpec]) -> None:
    if len(prompts) != 2:
        raise ContractError("exactly two prompt-token JSON inputs are required")
    checks = {
        "source_id": {prompt.source_id.casefold() for prompt in prompts},
        "semantic_domain": {prompt.semantic_domain.casefold() for prompt in prompts},
        "source_reference": {prompt.source_reference.casefold() for prompt in prompts},
        "source_sha256": {prompt.source_sha256 for prompt in prompts},
        "prompt_token_ids_sha256": {prompt.token_ids_sha256 for prompt in prompts},
    }
    duplicated = sorted(name for name, values in checks.items() if len(values) != 2)
    if duplicated:
        raise ContractError(
            "the two prompts are not independently identified; "
            f"duplicate fields: {duplicated}"
        )


def _load_provenance(path: Path, kind: str) -> ProvenanceInput:
    value, file_sha256 = _load_json_object(path, label=f"{kind} provenance")
    required_by_kind = {
        "server": {
            "deployment_uid",
            "pod_uid",
            "image",
            "image_id",
            "served_model",
            "server_started_at",
            "ready",
            "restart_count",
        },
        "source": {
            "git_commit",
            "git_tree",
            "git_dirty",
            "source_snapshot_sha256",
        },
        "config": {
            "runtime_config_id",
            "max_model_len",
            "max_num_seqs",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "kv_cache_dtype",
            "physical_kv_cache_tokens",
            "max_num_batched_tokens",
            "max_num_partial_prefills",
            "max_long_partial_prefills",
            "long_prefill_token_threshold",
            "compilation_config_mode",
        },
        "argv": {"argv", "argv_sha256"},
        "gpu": {"gpu_uuids"},
        "jit": {"checked_at", "log_sha256", "unexpected_jit"},
    }
    _require_fields(value, required_by_kind[kind], label=f"{kind} provenance")
    if kind == "source":
        if value["git_dirty"] is not False:
            raise ContractError("source provenance must identify a clean tree")
        for key in ("git_commit", "git_tree"):
            if not _is_lower_hex(value[key], {40, 64}):
                raise ContractError(f"source provenance {key} is not a git object")
        if not _is_lower_hex(value["source_snapshot_sha256"], {64}):
            raise ContractError("source provenance snapshot is not SHA-256")
    elif kind == "server":
        if not isinstance(value["ready"], bool) or not isinstance(
            value["restart_count"], int
        ):
            raise ContractError(
                "server provenance ready and restart_count have invalid types"
            )
    elif kind == "config":
        integer_fields = {
            "max_model_len",
            "max_num_seqs",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "physical_kv_cache_tokens",
            "max_num_batched_tokens",
            "max_num_partial_prefills",
            "max_long_partial_prefills",
            "long_prefill_token_threshold",
            "compilation_config_mode",
        }
        if any(
            isinstance(value[key], bool) or not isinstance(value[key], int)
            for key in integer_fields
        ):
            raise ContractError("config provenance has non-integer numeric fields")
        if not isinstance(value["kv_cache_dtype"], str):
            raise ContractError("config provenance kv_cache_dtype must be a string")
    elif kind == "argv":
        if not isinstance(value["argv"], list) or not all(
            isinstance(argument, str) and argument for argument in value["argv"]
        ):
            raise ContractError("argv provenance argv must be a non-empty string list")
        if not _is_lower_hex(value["argv_sha256"], {64}):
            raise ContractError("argv provenance argv_sha256 must be SHA-256")
        canonical_argv = json.dumps(
            value["argv"], separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        if _sha256_bytes(canonical_argv) != value["argv_sha256"]:
            raise ContractError("argv provenance argv_sha256 does not match argv")
    elif kind == "gpu":
        uuids = value["gpu_uuids"]
        if not isinstance(uuids, list) or not all(
            isinstance(uuid, str) for uuid in uuids
        ):
            raise ContractError("GPU provenance gpu_uuids must be a string list")
    elif kind == "jit":
        if not isinstance(value["unexpected_jit"], bool):
            raise ContractError("JIT provenance unexpected_jit must be a boolean")
        if not _is_lower_hex(value["log_sha256"], {64}):
            raise ContractError("JIT provenance log_sha256 must be SHA-256")
    return ProvenanceInput(
        kind=kind,
        path=path.resolve(),
        file_sha256=file_sha256,
        value=value,
    )


def _validate_admission_provenance(
    provenance: dict[str, ProvenanceInput],
) -> None:
    server = provenance["server"].value
    if server["ready"] is not True or server["restart_count"] != 0:
        raise ContractError(
            "server provenance must identify a Ready server with zero restarts"
        )
    config = provenance["config"].value
    expected = {
        "max_model_len": MAX_MODEL_LEN,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
    }
    for key, expected_value in expected.items():
        if config[key] != expected_value:
            raise ContractError(
                f"config provenance {key} must be {expected_value!r}, "
                f"got {config[key]!r}"
            )
    if config["max_num_seqs"] < 2:
        raise ContractError("config provenance max_num_seqs must be an integer >= 2")
    if config["kv_cache_dtype"].casefold() not in {"fp16", "float16"}:
        raise ContractError("config provenance must identify FP16 KV")
    if config["physical_kv_cache_tokens"] < MIN_PHYSICAL_KV_TOKENS:
        raise ContractError(
            "config provenance physical_kv_cache_tokens is below the "
            f"two-slot minimum of {MIN_PHYSICAL_KV_TOKENS:,}"
        )
    uuids = provenance["gpu"].value["gpu_uuids"]
    if (
        len(uuids) != TENSOR_PARALLEL_SIZE
        or len(set(uuids)) != TENSOR_PARALLEL_SIZE
        or not all(uuid.startswith("GPU-") for uuid in uuids)
    ):
        raise ContractError("GPU provenance must identify eight distinct GPU UUIDs")
    if provenance["jit"].value["unexpected_jit"] is not False:
        raise ContractError("JIT provenance reports unexpected JIT")


def _validate_url(value: str, *, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ContractError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ContractError(f"{label} must not contain a query or fragment")
    return value.rstrip("/")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_pvc_paths(
    *, pvc_root: Path, output_root: Path, inputs: Iterable[Path]
) -> tuple[Path, Path]:
    root = pvc_root.resolve(strict=True)
    if root in {Path("/"), Path("/tmp"), Path("/var/tmp")}:
        raise ContractError(f"refusing unsafe PVC root: {root}")
    if not root.is_dir() or not os.path.ismount(root):
        raise ContractError(f"--pvc-root is not a mounted directory: {root}")
    output = output_root.resolve(strict=True)
    if not output.is_dir() or not _is_relative_to(output, root):
        raise ContractError(
            f"--output-root must be an existing PVC directory: {output}"
        )
    for input_path in inputs:
        resolved = input_path.resolve(strict=True)
        if not _is_relative_to(resolved, root):
            raise ContractError(f"input is outside PVC root {root}: {resolved}")
        if resolved.stat().st_dev != root.stat().st_dev:
            raise ContractError(f"input is on a different filesystem: {resolved}")
    if shutil.disk_usage(root).free < 100 * 1024 * 1024:
        raise ContractError(f"PVC has less than 100 MiB free: {root}")
    return root, output


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) < 2 or fields[0].startswith("#"):
            continue
        name = fields[0].split("{", 1)[0]
        if name not in {METRIC_RUNNING, METRIC_WAITING}:
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if not math.isfinite(value):
            raise ContractError(f"non-finite Prometheus value for {name}")
        totals[name] = totals.get(name, 0.0) + value
    return totals


class SSEDecoder:
    """Incremental UTF-8 SSE decoder that requires a terminating DONE event."""

    def __init__(self) -> None:
        self._buffer = ""
        self._data_lines: list[str] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self.done = False

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        if self.done and chunk.strip():
            raise ContractError("received bytes after SSE [DONE]")
        try:
            self._buffer += self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise ContractError("SSE stream was not valid UTF-8") from exc
        events: list[dict[str, Any]] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                events.extend(self._dispatch())
            elif line.startswith(":"):
                continue
            elif line.startswith("data:"):
                self._data_lines.append(line[5:].lstrip(" "))
        return events

    def finish(self) -> list[dict[str, Any]]:
        try:
            self._buffer += self._decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ContractError("SSE stream ended within a UTF-8 code point") from exc
        events: list[dict[str, Any]] = []
        if self._buffer:
            line = self._buffer.removesuffix("\r")
            self._buffer = ""
            if line.startswith("data:"):
                self._data_lines.append(line[5:].lstrip(" "))
            elif line and not line.startswith(":"):
                raise ContractError(f"unsupported trailing SSE field: {line!r}")
        if self._data_lines:
            events.extend(self._dispatch())
        if not self.done:
            raise ContractError("SSE stream ended without data: [DONE]")
        return events

    def _dispatch(self) -> list[dict[str, Any]]:
        if not self._data_lines:
            return []
        payload = "\n".join(self._data_lines)
        self._data_lines.clear()
        if payload == "[DONE]":
            self.done = True
            return []
        if self.done:
            raise ContractError("received SSE event after [DONE]")
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid SSE JSON: {payload[:200]!r}") from exc
        if not isinstance(event, dict):
            raise ContractError("SSE data event must be a JSON object")
        return [event]


async def _fetch_text(
    session: aiohttp.ClientSession, url: str, *, timeout_s: float
) -> tuple[int, str]:
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as response:
            return response.status, await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise ContractError(f"GET {url} failed: {exc}") from exc


async def _preflight(
    session: aiohttp.ClientSession, *, base_url: str, metrics_url: str, model: str
) -> dict[str, Any]:
    health_status, health_body = await _fetch_text(
        session, base_url + "/health", timeout_s=10.0
    )
    if health_status != 200:
        raise ContractError(f"health endpoint returned HTTP {health_status}")
    models_status, models_body = await _fetch_text(
        session, base_url + "/v1/models", timeout_s=30.0
    )
    if models_status != 200:
        raise ContractError(f"models endpoint returned HTTP {models_status}")
    try:
        model_ids = [
            item["id"]
            for item in json.loads(models_body)["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ContractError("models endpoint returned an unexpected schema") from exc
    if model not in model_ids:
        raise ContractError(f"model {model!r} is not served: {model_ids}")
    metrics_status, metrics_body = await _fetch_text(
        session, metrics_url, timeout_s=10.0
    )
    metrics = _parse_prometheus_metrics(metrics_body) if metrics_status == 200 else {}
    if metrics_status != 200 or set(metrics) != {METRIC_RUNNING, METRIC_WAITING}:
        raise ContractError("metrics preflight is missing required scheduler gauges")
    if metrics[METRIC_RUNNING] != 0 or metrics[METRIC_WAITING] != 0:
        raise ContractError(f"server is not idle: {metrics}")
    return {
        "health_http_status": health_status,
        "health_body_sha256": _sha256_bytes(health_body.encode()),
        "models_http_status": models_status,
        "models_body_sha256": _sha256_bytes(models_body.encode()),
        "model_ids": model_ids,
        "metrics_http_status": metrics_status,
        "metrics_body_sha256": _sha256_bytes(metrics_body.encode()),
        "idle_metrics": metrics,
    }


async def _sample_metrics(
    session: aiohttp.ClientSession,
    *,
    metrics_url: str,
    stop: asyncio.Event,
    ready: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    while True:
        sampled_at_ns = time.monotonic_ns()
        try:
            status, raw_text = await _fetch_text(session, metrics_url, timeout_s=5.0)
            metrics = _parse_prometheus_metrics(raw_text) if status == 200 else {}
            samples.append(
                {
                    "sampled_at_monotonic_ns": sampled_at_ns,
                    "http_status": status,
                    "running": metrics.get(METRIC_RUNNING),
                    "waiting": metrics.get(METRIC_WAITING),
                    "raw_metrics": raw_text,
                    "raw_metrics_sha256": _sha256_bytes(raw_text.encode()),
                    "error": None,
                }
            )
        except Exception as exc:
            samples.append(
                {
                    "sampled_at_monotonic_ns": sampled_at_ns,
                    "http_status": None,
                    "running": None,
                    "waiting": None,
                    "raw_metrics": None,
                    "raw_metrics_sha256": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        ready.set()
        if stop.is_set():
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=METRICS_INTERVAL_SECONDS)


def _choice(event: dict[str, Any]) -> dict[str, Any] | None:
    choices = event.get("choices")
    if choices == [] and event.get("usage") is not None:
        return None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ContractError("each completion SSE event must contain one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("index") != 0:
        raise ContractError("completion SSE choice must have index zero")
    return choice


async def _stream_completion(
    session: aiohttp.ClientSession,
    *,
    endpoint: str,
    model: str,
    prompt: PromptSpec,
    run_id: str,
    start_gate: asyncio.Event,
    timeout_s: float,
) -> dict[str, Any]:
    salt = secrets.token_hex(32)
    request_id = f"{run_id}-admission-{prompt.slot}"
    payload = {
        "model": model,
        "prompt": list(prompt.token_ids),
        "max_tokens": OUTPUT_TOKEN_COUNT,
        "min_tokens": OUTPUT_TOKEN_COUNT,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "ignore_eos": True,
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "n": 1,
        "echo": False,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_salt": salt,
        "request_id": request_id,
    }
    await start_gate.wait()
    request_start_ns = time.monotonic_ns()
    decoder = SSEDecoder()
    response_id: str | None = None
    response_model: str | None = None
    echoed_prompt_hash: str | None = None
    output_token_ids: list[int] = []
    output_text: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    try:
        async with session.post(
            endpoint,
            json=payload,
            headers={"X-Request-ID": request_id},
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as response:
            if response.status != 200:
                detail = (await response.text())[:4_000]
                raise ContractError(
                    f"POST {endpoint} returned HTTP {response.status}: {detail}"
                )
            async for chunk in response.content.iter_any():
                for event in decoder.feed(chunk):
                    response_id = response_id or event.get("id")
                    response_model = response_model or event.get("model")
                    if event.get("usage") is not None:
                        if not isinstance(event["usage"], dict):
                            raise ContractError("completion usage must be an object")
                        usage = event["usage"]
                    choice = _choice(event)
                    if choice is None:
                        continue
                    prompt_ids = choice.get("prompt_token_ids")
                    if prompt_ids is not None:
                        if not isinstance(prompt_ids, list):
                            raise ContractError("echoed prompt_token_ids is not a list")
                        echoed_prompt_hash = _sha256_token_ids(prompt_ids)
                        if (
                            len(prompt_ids) != PROMPT_TOKEN_COUNT
                            or echoed_prompt_hash != prompt.token_ids_sha256
                        ):
                            raise ContractError("server did not echo the exact prompt")
                    token_ids = choice.get("token_ids")
                    if token_ids is not None:
                        if not isinstance(token_ids, list) or any(
                            isinstance(token_id, bool)
                            or not isinstance(token_id, int)
                            or token_id < 0
                            or token_id > 2**31 - 1
                            for token_id in token_ids
                        ):
                            raise ContractError("stream token_ids are invalid")
                        output_token_ids.extend(token_ids)
                    text = choice.get("text")
                    if text is not None:
                        if not isinstance(text, str):
                            raise ContractError("stream text delta must be a string")
                        output_text.append(text)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
            decoder.finish()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise ContractError(f"completion request failed: {exc}") from exc
    request_end_ns = time.monotonic_ns()
    if not isinstance(response_id, str) or not response_id:
        raise ContractError("completion response is missing an ID")
    if response_model != model:
        raise ContractError(f"response model mismatch: {response_model!r}")
    if echoed_prompt_hash != prompt.token_ids_sha256:
        raise ContractError("stream never echoed the exact prompt_token_ids")
    if len(output_token_ids) != OUTPUT_TOKEN_COUNT:
        raise ContractError(
            f"expected {OUTPUT_TOKEN_COUNT} output token, got {len(output_token_ids)}"
        )
    if finish_reason != "length":
        raise ContractError(f"expected finish_reason='length', got {finish_reason!r}")
    if usage is None or usage.get("prompt_tokens") != PROMPT_TOKEN_COUNT:
        raise ContractError(f"unexpected completion usage: {usage}")
    if usage.get("completion_tokens") != OUTPUT_TOKEN_COUNT:
        raise ContractError(f"unexpected completion usage: {usage}")
    return {
        "slot": prompt.slot,
        "request_id": request_id,
        "response_id": response_id,
        "cache_salt_sha256": _sha256_bytes(salt.encode()),
        "cache_salt_length": len(salt),
        "prompt_token_count": PROMPT_TOKEN_COUNT,
        "prompt_token_ids_sha256": prompt.token_ids_sha256,
        "echoed_prompt_token_ids_sha256": echoed_prompt_hash,
        "output_token_count": len(output_token_ids),
        "output_token_ids_sha256": _sha256_token_ids(output_token_ids),
        "output_text_sha256": _sha256_bytes("".join(output_text).encode()),
        "finish_reason": finish_reason,
        "usage": usage,
        "request_start_monotonic_ns": request_start_ns,
        "request_end_monotonic_ns": request_end_ns,
    }


def _admission_proof(
    samples: Sequence[dict[str, Any]], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(records) != 2:
        raise ContractError("admission requires exactly two completed requests")
    active_start = max(record["request_start_monotonic_ns"] for record in records)
    active_end = min(record["request_end_monotonic_ns"] for record in records)
    if active_end <= active_start:
        raise ContractError("two client request intervals did not overlap")
    interval_samples = [
        sample
        for sample in samples
        if active_start <= sample["sampled_at_monotonic_ns"] <= active_end
    ]
    if not interval_samples:
        raise ContractError("no 50 ms metrics samples covered the active interval")
    bad_active = [
        sample
        for sample in interval_samples
        if sample["error"] is not None
        or sample["running"] != 2.0
        or sample["waiting"] != 0.0
    ]
    if bad_active:
        raise ContractError(
            "active interval was not continuously admitted as running=2, waiting=0"
        )
    request_window_start = min(
        record["request_start_monotonic_ns"] for record in records
    )
    request_window_end = max(record["request_end_monotonic_ns"] for record in records)
    request_samples = [
        sample
        for sample in samples
        if request_window_start
        <= sample["sampled_at_monotonic_ns"]
        <= request_window_end
    ]
    if any(
        sample["error"] is not None or sample["waiting"] != 0.0
        for sample in request_samples
    ):
        raise ContractError("a metrics sample observed a queue or scrape failure")
    if any(
        sample["running"] is not None and sample["running"] > 2.0 for sample in samples
    ):
        raise ContractError("metrics observed more than two running requests")
    salts = [record["cache_salt_sha256"] for record in records]
    if len(set(salts)) != 2:
        raise ContractError("the two admission requests did not use unique cache salts")
    return {
        "passed": True,
        "active_interval_start_monotonic_ns": active_start,
        "active_interval_end_monotonic_ns": active_end,
        "active_interval_sample_count": len(interval_samples),
        "request_window_sample_count": len(request_samples),
        "required_running": 2,
        "required_waiting": 0,
        "unique_cache_salt_hash_count": len(set(salts)),
    }


async def _run_probe(
    *,
    args: argparse.Namespace,
    prompts: Sequence[PromptSpec],
    records: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), trust_env=False
    ) as session:
        preflight = await _preflight(
            session,
            base_url=args.base_url,
            metrics_url=args.metrics_url,
            model=args.model,
        )
        stop = asyncio.Event()
        ready = asyncio.Event()
        start_gate = asyncio.Event()
        sampler = asyncio.create_task(
            _sample_metrics(
                session,
                metrics_url=args.metrics_url,
                stop=stop,
                ready=ready,
                samples=samples,
            )
        )
        await asyncio.wait_for(ready.wait(), timeout=10.0)
        initial = samples[-1]
        if (
            initial["error"] is not None
            or initial["running"] != 0.0
            or initial["waiting"] != 0.0
        ):
            stop.set()
            await sampler
            raise ContractError(f"server was not idle before admission: {initial}")
        tasks = [
            asyncio.create_task(
                _stream_completion(
                    session,
                    endpoint=args.base_url + "/v1/completions",
                    model=args.model,
                    prompt=prompt,
                    run_id=args.run_id,
                    start_gate=start_gate,
                    timeout_s=args.request_timeout_s,
                )
            )
            for prompt in prompts
        ]
        start_gate.set()
        try:
            records.extend(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            stop.set()
            await sampler
    proof = _admission_proof(samples, records)
    return preflight, records, samples, proof


def _write_json_exclusive(path: Path, value: Any) -> dict[str, Any]:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return {"sha256": _sha256_bytes(encoded), "size_bytes": len(encoded)}


def _write_manifest(
    output_dir: Path,
    *,
    run_id: str,
    input_paths: Sequence[Path],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "benchmark": "forge001_hy3_two_slot_admission",
        "run_id": run_id,
        "created_at": _utc_now(),
        "artifacts": artifacts,
        "inputs": {
            str(path): {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in input_paths
        },
    }
    return _write_json_exclusive(output_dir / "manifest.json", manifest)


def _build_plan(
    *,
    args: argparse.Namespace,
    prompts: Sequence[PromptSpec],
    provenance: dict[str, ProvenanceInput],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "forge001_hy3_two_slot_admission",
        "run_id": args.run_id,
        "base_url": args.base_url,
        "metrics_url": args.metrics_url,
        "model": args.model,
        "contract": {
            "prompt_token_count_per_slot": PROMPT_TOKEN_COUNT,
            "output_token_count_per_slot": OUTPUT_TOKEN_COUNT,
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
            "minimum_physical_kv_tokens": MIN_PHYSICAL_KV_TOKENS,
            "metrics_interval_seconds": METRICS_INTERVAL_SECONDS,
            "required_running": 2,
            "required_waiting": 0,
            "unique_cache_salts": True,
            "raw_metrics_retained": True,
        },
        "prompts": [prompt.record() for prompt in prompts],
        "provenance": {kind: item.record() for kind, item in provenance.items()},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metrics-url")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-token-json", type=Path, action="append", required=True
    )
    parser.add_argument("--server-provenance-json", type=Path, required=True)
    parser.add_argument("--source-provenance-json", type=Path, required=True)
    parser.add_argument("--config-provenance-json", type=Path, required=True)
    parser.add_argument("--argv-provenance-json", type=Path, required=True)
    parser.add_argument("--gpu-provenance-json", type=Path, required=True)
    parser.add_argument("--jit-provenance-json", type=Path, required=True)
    parser.add_argument("--pvc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-timeout-s", type=float, default=7_200.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not _valid_run_id(args.run_id):
        raise ContractError("--run-id contains unsafe characters")
    args.base_url = _validate_url(args.base_url, label="--base-url")
    args.metrics_url = _validate_url(
        args.metrics_url or args.base_url + "/metrics", label="--metrics-url"
    )
    if len(args.prompt_token_json) != 2:
        raise ContractError("--prompt-token-json must be provided exactly twice")
    if args.request_timeout_s <= 0:
        raise ContractError("--request-timeout-s must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    prompts = [
        _load_prompt(path, slot)
        for path, slot in zip(args.prompt_token_json, ("a", "b"), strict=True)
    ]
    _validate_prompt_independence(prompts)
    provenance_paths = {
        "server": args.server_provenance_json,
        "source": args.source_provenance_json,
        "config": args.config_provenance_json,
        "argv": args.argv_provenance_json,
        "gpu": args.gpu_provenance_json,
        "jit": args.jit_provenance_json,
    }
    provenance = {
        kind: _load_provenance(path, kind) for kind, path in provenance_paths.items()
    }
    if provenance["server"].value["served_model"] != args.model:
        raise ContractError("server provenance served_model does not match --model")
    plan = _build_plan(args=args, prompts=prompts, provenance=provenance)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    input_paths = [*args.prompt_token_json, *provenance_paths.values()]
    pvc_root, output_root = _validate_pvc_paths(
        pvc_root=args.pvc_root, output_root=args.output_root, inputs=input_paths
    )
    output_dir = output_root / args.run_id
    try:
        output_dir.mkdir(mode=0o750)
    except FileExistsError as exc:
        raise ContractError(
            f"refusing to reuse output directory: {output_dir}"
        ) from exc
    artifacts: dict[str, dict[str, Any]] = {}
    artifacts["plan.json"] = _write_json_exclusive(output_dir / "plan.json", plan)
    records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    try:
        _validate_admission_provenance(provenance)
        preflight, records, samples, proof = asyncio.run(
            _run_probe(
                args=args,
                prompts=prompts,
                records=records,
                samples=samples,
            )
        )
        artifacts["requests.json"] = _write_json_exclusive(
            output_dir / "requests.json", records
        )
        artifacts["metrics.json"] = _write_json_exclusive(
            output_dir / "metrics.json", samples
        )
        summary = {
            "schema_version": 1,
            "benchmark": "forge001_hy3_two_slot_admission",
            "status": "passed",
            "finished_at": _utc_now(),
            "run": {
                "run_id": args.run_id,
                "pvc_root": str(pvc_root),
                "output_dir": str(output_dir),
            },
            "plan": plan,
            "preflight": preflight,
            "requests": records,
            "admission_proof": proof,
            "gates": {
                "two_request_overlap": True,
                "running_two_waiting_zero_throughout_active_interval": True,
                "zero_queued_samples": True,
                "exact_prompt_lengths": True,
                "one_token_outputs": True,
                "unique_cache_salts": True,
                "no_unexpected_jit": True,
                "pvc_only_exclusive_manifest": True,
            },
        }
        artifacts["summary.json"] = _write_json_exclusive(
            output_dir / "summary.json", summary
        )
    except BaseException as exc:
        artifacts["requests.json"] = _write_json_exclusive(
            output_dir / "requests.json", records
        )
        artifacts["metrics.json"] = _write_json_exclusive(
            output_dir / "metrics.json", samples
        )
        failure = {
            "schema_version": 1,
            "benchmark": "forge001_hy3_two_slot_admission",
            "status": "failed",
            "failed_at": _utc_now(),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "plan": plan,
        }
        artifacts["failure.json"] = _write_json_exclusive(
            output_dir / "failure.json", failure
        )
        _write_manifest(
            output_dir,
            run_id=args.run_id,
            input_paths=input_paths,
            artifacts=artifacts,
        )
        raise
    manifest = _write_manifest(
        output_dir,
        run_id=args.run_id,
        input_paths=input_paths,
        artifacts=artifacts,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest_sha256": manifest["sha256"],
                "manifest_size_bytes": manifest["size_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
