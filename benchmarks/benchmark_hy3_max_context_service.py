# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed Hy3 maximum-context HTTP service benchmark.

The scored contract is deliberately fixed: two independent 64,001-token
prompts, 256 generated tokens, batch sizes one and two, and five paired
repeats. Results are written only below a caller-identified mounted PVC.

Each ``--prompt-token-json`` is a schema-version-1 object containing
``source_id``, ``semantic_domain``, ``source_reference``, ``source_sha256``,
and ``prompt_token_ids``. The server, source, and config provenance files are
also required inputs so a result cannot silently outlive the image or source
tree it measured. ``--dry-run`` validates all inputs without HTTP or writes.
"""

import argparse
import asyncio
import codecs
import contextlib
import hashlib
import json
import math
import os
import shutil
import statistics
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
OUTPUT_TOKEN_COUNT = 256
MAX_MODEL_LEN = 65_536
PAIRED_REPEATS = 5
METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"

SERVER_PROVENANCE_FIELDS = {
    "deployment_uid",
    "pod_uid",
    "image",
    "image_id",
    "served_model",
    "server_started_at",
}
SOURCE_PROVENANCE_FIELDS = {
    "git_commit",
    "git_tree",
    "git_dirty",
    "source_snapshot_sha256",
}
CONFIG_PROVENANCE_FIELDS = {
    "config_id",
    "deployment_spec_sha256",
    "runtime_args_sha256",
    "runtime_env_sha256",
    "max_model_len",
    "max_num_seqs",
    "tensor_parallel_size",
    "attention_backend",
}


class ContractError(RuntimeError):
    """Raised when a run cannot prove the fixed benchmark contract."""


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

    def metadata(self) -> dict[str, Any]:
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
    encoded = json.dumps(
        token_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
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
        and value[0].isalnum()
        and value[0].isascii()
        and all(character in allowed for character in value)
    )


def _require_fields(
    value: dict[str, Any],
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        raise ContractError(f"{label} is missing required fields: {missing}")
    empty = sorted(
        key
        for key in required
        if value[key] is None or value[key] == "" or value[key] == []
    )
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
    source_sha256 = value["source_sha256"]
    if not _is_lower_hex(source_sha256, {64}):
        raise ContractError(f"prompt {slot} source_sha256 must be lowercase SHA-256")
    token_ids = value["prompt_token_ids"]
    if not isinstance(token_ids, list):
        raise ContractError(f"prompt {slot} prompt_token_ids must be a JSON array")
    if len(token_ids) != PROMPT_TOKEN_COUNT:
        raise ContractError(
            f"prompt {slot} must contain exactly {PROMPT_TOKEN_COUNT:,} token "
            f"IDs, got {len(token_ids):,}"
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
        source_sha256=source_sha256,
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
            "the two prompt sources are not independently identified; "
            f"duplicate fields: {duplicated}"
        )


def _load_provenance(path: Path, kind: str) -> ProvenanceInput:
    value, file_sha256 = _load_json_object(path, label=f"{kind} provenance")
    required_by_kind = {
        "server": SERVER_PROVENANCE_FIELDS,
        "source": SOURCE_PROVENANCE_FIELDS,
        "config": CONFIG_PROVENANCE_FIELDS,
    }
    required = required_by_kind[kind]
    _require_fields(value, required, label=f"{kind} provenance")
    string_fields_by_kind = {
        "server": SERVER_PROVENANCE_FIELDS,
        "source": {
            "git_commit",
            "git_tree",
            "source_snapshot_sha256",
        },
        "config": {
            "config_id",
            "deployment_spec_sha256",
            "runtime_args_sha256",
            "runtime_env_sha256",
            "attention_backend",
        },
    }
    for key in string_fields_by_kind[kind]:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ContractError(f"{kind} provenance {key} must be a non-empty string")
    if kind == "source":
        if not isinstance(value["git_dirty"], bool):
            raise ContractError("source provenance git_dirty must be a boolean")
        if value["git_dirty"]:
            raise ContractError("source provenance must identify a clean tree")
        for key in ("git_commit", "git_tree"):
            if not _is_lower_hex(value[key], {40, 64}):
                raise ContractError(f"source provenance {key} is not a git object")
        if not _is_lower_hex(value["source_snapshot_sha256"], {64}):
            raise ContractError(
                "source provenance source_snapshot_sha256 is not SHA-256"
            )
    if kind == "config":
        for key in (
            "deployment_spec_sha256",
            "runtime_args_sha256",
            "runtime_env_sha256",
        ):
            if not _is_lower_hex(value[key], {64}):
                raise ContractError(f"config provenance {key} is not SHA-256")
        if value["max_model_len"] != MAX_MODEL_LEN:
            raise ContractError(
                f"config max_model_len must be exactly {MAX_MODEL_LEN:,}"
            )
        if (
            isinstance(value["max_num_seqs"], bool)
            or not isinstance(value["max_num_seqs"], int)
            or value["max_num_seqs"] < 2
        ):
            raise ContractError("config max_num_seqs must be an integer >= 2")
        if value["tensor_parallel_size"] != 8:
            raise ContractError("config tensor_parallel_size must be exactly 8")
    return ProvenanceInput(
        kind=kind,
        path=path.resolve(),
        file_sha256=file_sha256,
        value=value,
    )


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
    *,
    pvc_root: Path,
    output_root: Path,
    inputs: Iterable[Path],
) -> tuple[Path, Path]:
    root = pvc_root.resolve(strict=True)
    if root in {Path("/"), Path("/tmp"), Path("/var/tmp")}:
        raise ContractError(f"refusing unsafe PVC root: {root}")
    if not root.is_dir() or not os.path.ismount(root):
        raise ContractError(f"--pvc-root is not a mounted directory: {root}")

    output_parent = output_root.parent.resolve(strict=True)
    output = output_parent / output_root.name
    if not _is_relative_to(output, root):
        raise ContractError(f"output root escapes PVC mount {root}: {output}")
    if output.exists():
        raise ContractError(f"output root already exists: {output}")

    for input_path in inputs:
        resolved = input_path.resolve(strict=True)
        if not _is_relative_to(resolved, root):
            raise ContractError(f"input is outside PVC mount {root}: {resolved}")
        if resolved.stat().st_dev != root.stat().st_dev:
            raise ContractError(f"input is on a different filesystem: {resolved}")

    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 100 * 1024 * 1024:
        raise ContractError(
            f"PVC has less than 100 MiB free: {free_bytes} bytes at {root}"
        )
    return root, output


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
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
    """Incremental UTF-8 SSE decoder with strict JSON data events."""

    def __init__(self) -> None:
        self._buffer = ""
        self._data_lines: list[str] = []
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self.done = False

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        if self.done and chunk.strip():
            raise ContractError("received bytes after SSE [DONE]")
        try:
            self._buffer += self._utf8_decoder.decode(chunk, final=False)
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
        events: list[dict[str, Any]] = []
        try:
            self._buffer += self._utf8_decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ContractError("SSE stream ended within a UTF-8 code point") from exc
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
            raise ContractError(f"invalid JSON SSE event: {payload[:200]!r}") from exc
        if not isinstance(event, dict):
            raise ContractError("SSE data event must be a JSON object")
        return [event]


def _cache_salt(run_id: str, phase: str, repeat: int, slot: str) -> str:
    identity = f"{run_id}\0{phase}\0{repeat}\0{slot}".encode()
    return hashlib.sha256(identity).hexdigest()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ContractError("cannot divide by a non-positive benchmark duration")
    return numerator / denominator


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ContractError("summary statistics require finite values")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    minimum = min(values)
    maximum = max(values)
    return {
        "count": len(values),
        "median": median,
        "mad": statistics.median(deviations),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "values": list(values),
    }


async def _fetch_text(
    session: aiohttp.ClientSession,
    url: str,
    *,
    timeout_s: float,
) -> tuple[int, str]:
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as response:
            return response.status, await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise ContractError(f"GET {url} failed: {exc}") from exc


async def _preflight(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    metrics_url: str,
    model: str,
) -> dict[str, Any]:
    health_status, health_body = await _fetch_text(
        session,
        base_url + "/health",
        timeout_s=10.0,
    )
    if health_status != 200:
        raise ContractError(f"health endpoint returned HTTP {health_status}")
    models_status, models_body = await _fetch_text(
        session,
        base_url + "/v1/models",
        timeout_s=30.0,
    )
    if models_status != 200:
        raise ContractError(f"models endpoint returned HTTP {models_status}")
    try:
        models_response = json.loads(models_body)
        model_ids = [
            entry["id"]
            for entry in models_response["data"]
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        ]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ContractError(
            "models endpoint did not return the expected schema"
        ) from exc
    if model not in model_ids:
        raise ContractError(f"model {model!r} is not served; available: {model_ids}")

    metrics_status, metrics_body = await _fetch_text(
        session,
        metrics_url,
        timeout_s=10.0,
    )
    if metrics_status != 200:
        raise ContractError(f"metrics endpoint returned HTTP {metrics_status}")
    metrics = _parse_prometheus_metrics(metrics_body)
    missing = sorted({METRIC_RUNNING, METRIC_WAITING} - metrics.keys())
    if missing:
        raise ContractError(f"metrics endpoint is missing required gauges: {missing}")
    if metrics[METRIC_RUNNING] != 0 or metrics[METRIC_WAITING] != 0:
        raise ContractError(
            "server is not idle at preflight: "
            f"running={metrics[METRIC_RUNNING]}, "
            f"waiting={metrics[METRIC_WAITING]}"
        )
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
    interval_s: float,
    stop: asyncio.Event,
    ready: asyncio.Event,
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    while True:
        timestamp_ns = time.monotonic_ns()
        try:
            status, body = await _fetch_text(
                session,
                metrics_url,
                timeout_s=max(2.0, interval_s * 5),
            )
            parsed = _parse_prometheus_metrics(body) if status == 200 else {}
            samples.append(
                {
                    "monotonic_ns": timestamp_ns,
                    "http_status": status,
                    "running": parsed.get(METRIC_RUNNING),
                    "waiting": parsed.get(METRIC_WAITING),
                    "body_sha256": _sha256_bytes(body.encode()),
                    "error": None,
                }
            )
        except Exception as exc:
            samples.append(
                {
                    "monotonic_ns": timestamp_ns,
                    "http_status": None,
                    "running": None,
                    "waiting": None,
                    "body_sha256": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        ready.set()
        if stop.is_set():
            return samples
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)


def _stream_event_choice(event: dict[str, Any]) -> dict[str, Any] | None:
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
    phase: str,
    repeat: int,
    run_id: str,
    timeout_s: float,
    start_gate: asyncio.Event,
) -> dict[str, Any]:
    salt = _cache_salt(run_id, phase, repeat, prompt.slot)
    request_id = f"{run_id}-{phase}-r{repeat}-{prompt.slot}"
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
    output_token_ids: list[int] = []
    token_arrival_ns: list[int] = []
    text_parts: list[str] = []
    echoed_prompt_hash: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    headers = {"X-Request-ID": request_id}
    try:
        async with session.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as response:
            if response.status != 200:
                detail = (await response.text())[:4_000]
                raise ContractError(
                    f"POST {endpoint} returned HTTP {response.status}: {detail}"
                )
            async for chunk in response.content.iter_any():
                arrival_ns = time.monotonic_ns()
                for event in decoder.feed(chunk):
                    if response_id is None:
                        response_id = event.get("id")
                        response_model = event.get("model")
                    elif event.get("id") not in {None, response_id}:
                        raise ContractError("completion response ID changed mid-stream")
                    elif event.get("model") not in {None, response_model}:
                        raise ContractError(
                            "completion response model changed mid-stream"
                        )
                    if event.get("usage") is not None:
                        if not isinstance(event["usage"], dict):
                            raise ContractError("completion usage must be an object")
                        usage = event["usage"]
                    choice = _stream_event_choice(event)
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
                            raise ContractError(
                                "server did not echo the exact submitted prompt"
                            )
                    delta_ids = choice.get("token_ids")
                    if delta_ids is not None:
                        if not isinstance(delta_ids, list) or any(
                            isinstance(token_id, bool)
                            or not isinstance(token_id, int)
                            or token_id < 0
                            or token_id > 2**31 - 1
                            for token_id in delta_ids
                        ):
                            raise ContractError(
                                "stream token_ids must be non-negative "
                                "signed 32-bit integers"
                            )
                        output_token_ids.extend(delta_ids)
                        token_arrival_ns.extend([arrival_ns] * len(delta_ids))
                        if len(output_token_ids) > OUTPUT_TOKEN_COUNT:
                            raise ContractError(
                                "server returned too many output tokens"
                            )
                    text = choice.get("text")
                    if text is not None:
                        if not isinstance(text, str):
                            raise ContractError("stream text delta must be a string")
                        text_parts.append(text)
                    if choice.get("finish_reason") is not None:
                        next_finish_reason = choice["finish_reason"]
                        if (
                            finish_reason is not None
                            and next_finish_reason != finish_reason
                        ):
                            raise ContractError(
                                "finish_reason changed within the stream"
                            )
                        finish_reason = next_finish_reason
            for event in decoder.finish():
                choice = _stream_event_choice(event)
                if choice is not None:
                    raise ContractError("trailing SSE event was not consumed")
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise ContractError(f"completion request failed: {exc}") from exc
    request_end_ns = time.monotonic_ns()

    if response_id != f"cmpl-{request_id}":
        raise ContractError(
            f"response ID mismatch: expected 'cmpl-{request_id}', got {response_id!r}"
        )
    if response_model != model:
        raise ContractError(
            f"response model mismatch: expected {model!r}, got {response_model!r}"
        )
    if echoed_prompt_hash != prompt.token_ids_sha256:
        raise ContractError("stream never echoed exact prompt_token_ids")
    if len(output_token_ids) != OUTPUT_TOKEN_COUNT:
        raise ContractError(
            f"expected {OUTPUT_TOKEN_COUNT} output tokens, got {len(output_token_ids)}"
        )
    if len(token_arrival_ns) != OUTPUT_TOKEN_COUNT:
        raise ContractError("missing monotonic timestamp for an output token")
    if any(right < left for left, right in zip(token_arrival_ns, token_arrival_ns[1:])):
        raise ContractError("token arrival timestamps are not monotonic")
    if token_arrival_ns[-1] <= token_arrival_ns[0]:
        raise ContractError("stream did not expose a measurable decode interval")
    if finish_reason != "length":
        raise ContractError(f"expected finish_reason='length', got {finish_reason!r}")
    if usage is None:
        raise ContractError("stream did not include final usage")
    if (
        usage.get("prompt_tokens") != PROMPT_TOKEN_COUNT
        or usage.get("completion_tokens") != OUTPUT_TOKEN_COUNT
    ):
        raise ContractError(f"unexpected completion usage: {usage}")

    first_token_ns = token_arrival_ns[0]
    last_token_ns = token_arrival_ns[-1]
    decode_seconds = (last_token_ns - first_token_ns) / 1e9
    return {
        "phase": phase,
        "repeat": repeat,
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
        "output_text_sha256": _sha256_bytes("".join(text_parts).encode()),
        "finish_reason": finish_reason,
        "usage": usage,
        "request_start_monotonic_ns": request_start_ns,
        "first_token_monotonic_ns": first_token_ns,
        "last_token_monotonic_ns": last_token_ns,
        "request_end_monotonic_ns": request_end_ns,
        "token_arrival_monotonic_ns": token_arrival_ns,
        "ttft_seconds": (first_token_ns - request_start_ns) / 1e9,
        "decode_seconds": decode_seconds,
        "steady_decode_tokens": OUTPUT_TOKEN_COUNT - 1,
        "steady_decode_tokens_per_second": _safe_ratio(
            OUTPUT_TOKEN_COUNT - 1,
            decode_seconds,
        ),
    }


def _metrics_proof(
    samples: Sequence[dict[str, Any]],
    *,
    expected_batch: int,
) -> dict[str, Any]:
    good = [
        sample
        for sample in samples
        if sample["error"] is None
        and sample["running"] is not None
        and sample["waiting"] is not None
    ]
    if len(good) < 3:
        raise ContractError(
            f"only {len(good)} valid metrics samples; at least 3 are required"
        )
    running_values = [float(sample["running"]) for sample in good]
    waiting_values = [float(sample["waiting"]) for sample in good]
    max_running = max(running_values)
    max_waiting = max(waiting_values)
    simultaneous_samples = sum(
        running >= expected_batch and waiting == 0
        for running, waiting in zip(running_values, waiting_values)
    )
    if max_running != expected_batch:
        raise ContractError(
            f"batch {expected_batch} observed max running={max_running}, "
            f"expected exactly {expected_batch}"
        )
    if max_waiting != 0:
        raise ContractError(
            f"batch {expected_batch} observed queued requests: max waiting="
            f"{max_waiting}"
        )
    if simultaneous_samples < 1:
        raise ContractError(
            f"batch {expected_batch} never had all requests running with no queue"
        )
    return {
        "sample_count": len(samples),
        "valid_sample_count": len(good),
        "max_running": max_running,
        "max_waiting": max_waiting,
        "simultaneously_running_not_waiting_samples": simultaneous_samples,
        "passed": True,
    }


async def _run_group(
    session: aiohttp.ClientSession,
    *,
    prompts: Sequence[PromptSpec],
    phase: str,
    repeat: int,
    run_id: str,
    endpoint: str,
    metrics_url: str,
    model: str,
    timeout_s: float,
    metrics_interval_s: float,
) -> dict[str, Any]:
    expected_batch = len(prompts)
    if expected_batch not in {1, 2}:
        raise ContractError(f"unsupported scored batch size: {expected_batch}")
    stop = asyncio.Event()
    ready = asyncio.Event()
    start_gate = asyncio.Event()
    samples: list[dict[str, Any]] = []
    sampler = asyncio.create_task(
        _sample_metrics(
            session,
            metrics_url=metrics_url,
            interval_s=metrics_interval_s,
            stop=stop,
            ready=ready,
            samples=samples,
        )
    )
    await asyncio.wait_for(ready.wait(), timeout=10.0)
    baseline = samples[-1]
    if (
        baseline["error"] is not None
        or baseline["running"] != 0
        or baseline["waiting"] != 0
    ):
        stop.set()
        await sampler
        raise ContractError(
            "server was not idle before scored group: "
            f"running={baseline['running']}, waiting={baseline['waiting']}, "
            f"error={baseline['error']}"
        )
    requests = [
        asyncio.create_task(
            _stream_completion(
                session,
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                phase=phase,
                repeat=repeat,
                run_id=run_id,
                timeout_s=timeout_s,
                start_gate=start_gate,
            )
        )
        for prompt in prompts
    ]
    start_gate.set()
    request_error: BaseException | None = None
    try:
        records = await asyncio.gather(*requests)
    except BaseException as exc:
        for task in requests:
            task.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        request_error = exc
        records = []
    finally:
        stop.set()
    await sampler
    if request_error is not None:
        raise request_error.with_traceback(request_error.__traceback__)
    proof = _metrics_proof(samples, expected_batch=expected_batch)
    overlap_seconds: float | None = None
    decode_overlap_seconds: float | None = None
    aggregate_tps: float | None = None
    if expected_batch == 2:
        overlap_ns = min(record["last_token_monotonic_ns"] for record in records) - max(
            record["request_start_monotonic_ns"] for record in records
        )
        if overlap_ns <= 0:
            raise ContractError("batch-2 client request intervals did not overlap")
        overlap_seconds = overlap_ns / 1e9
        decode_overlap_ns = min(
            record["last_token_monotonic_ns"] for record in records
        ) - max(record["first_token_monotonic_ns"] for record in records)
        if decode_overlap_ns <= 0:
            raise ContractError("batch-2 decode intervals did not overlap")
        decode_overlap_seconds = decode_overlap_ns / 1e9
        aggregate_decode_ns = max(
            record["last_token_monotonic_ns"] for record in records
        ) - min(record["first_token_monotonic_ns"] for record in records)
        aggregate_tps = _safe_ratio(
            expected_batch * (OUTPUT_TOKEN_COUNT - 1),
            aggregate_decode_ns / 1e9,
        )
    return {
        "phase": phase,
        "repeat": repeat,
        "expected_batch": expected_batch,
        "records": records,
        "metrics_samples": samples,
        "metrics_proof": proof,
        "client_interval_overlap_seconds": overlap_seconds,
        "client_decode_interval_overlap_seconds": decode_overlap_seconds,
        "aggregate_steady_decode_tokens_per_second": aggregate_tps,
    }


def _paired_statistics(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paired_records = []
    b1_values: list[float] = []
    b2_values: list[float] = []
    b2_aggregate_values: list[float] = []
    paired_ratios: list[float] = []
    paired_deltas: list[float] = []
    scaling_values: list[float] = []
    for repeat in range(1, PAIRED_REPEATS + 1):
        by_phase_slot: dict[tuple[str, str], dict[str, Any]] = {}
        b2_group: dict[str, Any] | None = None
        for group in groups:
            if group["repeat"] != repeat:
                continue
            if group["phase"] == "b2":
                b2_group = group
            for record in group["records"]:
                by_phase_slot[(group["phase"], record["slot"])] = record
        expected_keys = {
            ("b1", "a"),
            ("b1", "b"),
            ("b2", "a"),
            ("b2", "b"),
        }
        if by_phase_slot.keys() != expected_keys or b2_group is None:
            raise ContractError(f"repeat {repeat} does not have a complete pair")
        repeat_ratios = []
        repeat_deltas = []
        repeat_b1 = []
        repeat_b2 = []
        for slot in ("a", "b"):
            b1 = by_phase_slot[("b1", slot)]["steady_decode_tokens_per_second"]
            b2 = by_phase_slot[("b2", slot)]["steady_decode_tokens_per_second"]
            ratio = _safe_ratio(b2, b1)
            delta = b2 - b1
            b1_values.append(b1)
            b2_values.append(b2)
            paired_ratios.append(ratio)
            paired_deltas.append(delta)
            repeat_ratios.append(ratio)
            repeat_deltas.append(delta)
            repeat_b1.append(b1)
            repeat_b2.append(b2)
        b1_reference = statistics.median(repeat_b1)
        b2_aggregate = b2_group["aggregate_steady_decode_tokens_per_second"]
        if b2_aggregate is None:
            raise ContractError(f"repeat {repeat} is missing batch-2 throughput")
        scaling = _safe_ratio(b2_aggregate, b1_reference)
        b2_aggregate_values.append(b2_aggregate)
        scaling_values.append(scaling)
        paired_records.append(
            {
                "repeat": repeat,
                "b1_per_slot_tps": dict(zip(("a", "b"), repeat_b1)),
                "b2_per_slot_tps": dict(zip(("a", "b"), repeat_b2)),
                "b2_over_b1_per_slot": dict(zip(("a", "b"), repeat_ratios)),
                "b2_minus_b1_per_slot_tps": dict(zip(("a", "b"), repeat_deltas)),
                "b1_reference_per_slot_tps": b1_reference,
                "b2_aggregate_tps": b2_aggregate,
                "b2_aggregate_over_b1_reference": scaling,
            }
        )
    return {
        "paired_repeats": paired_records,
        "b1_per_slot_tps": _summary_stats(b1_values),
        "b2_per_slot_tps": _summary_stats(b2_values),
        "b2_aggregate_tps": _summary_stats(b2_aggregate_values),
        "paired_b2_over_b1_per_slot": _summary_stats(paired_ratios),
        "paired_b2_minus_b1_per_slot_tps": _summary_stats(paired_deltas),
        "paired_b2_aggregate_over_b1_reference": _summary_stats(scaling_values),
    }


async def _run_benchmark(
    *,
    args: argparse.Namespace,
    prompts: Sequence[PromptSpec],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=False,
    ) as session:
        preflight = await _preflight(
            session,
            base_url=args.base_url,
            metrics_url=args.metrics_url,
            model=args.model,
        )
        endpoint = args.base_url + "/v1/completions"
        groups: list[dict[str, Any]] = []
        for repeat in range(1, PAIRED_REPEATS + 1):
            phase_order = ("b1", "b2") if repeat % 2 else ("b2", "b1")
            for phase in phase_order:
                if phase == "b2":
                    groups.append(
                        await _run_group(
                            session,
                            prompts=prompts,
                            phase=phase,
                            repeat=repeat,
                            run_id=args.run_id,
                            endpoint=endpoint,
                            metrics_url=args.metrics_url,
                            model=args.model,
                            timeout_s=args.request_timeout_s,
                            metrics_interval_s=args.metrics_interval_s,
                        )
                    )
                else:
                    prompt_order = prompts if repeat % 2 else tuple(reversed(prompts))
                    for prompt in prompt_order:
                        groups.append(
                            await _run_group(
                                session,
                                prompts=(prompt,),
                                phase=phase,
                                repeat=repeat,
                                run_id=args.run_id,
                                endpoint=endpoint,
                                metrics_url=args.metrics_url,
                                model=args.model,
                                timeout_s=args.request_timeout_s,
                                metrics_interval_s=args.metrics_interval_s,
                            )
                        )
    return preflight, groups


def _write_json_exclusive(path: Path, value: Any) -> tuple[str, int]:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha256_bytes(encoded), len(encoded)


def _write_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    groups: Sequence[dict[str, Any]],
    input_paths: Sequence[Path],
    plan_sha256: str,
    plan_size_bytes: int,
) -> dict[str, Any]:
    requests = [record for group in groups for record in group["records"]]
    metrics_groups = [
        {
            "phase": group["phase"],
            "repeat": group["repeat"],
            "expected_batch": group["expected_batch"],
            "proof": group["metrics_proof"],
            "client_interval_overlap_seconds": group["client_interval_overlap_seconds"],
            "client_decode_interval_overlap_seconds": group[
                "client_decode_interval_overlap_seconds"
            ],
            "samples": group["metrics_samples"],
        }
        for group in groups
    ]
    request_path = output_dir / "requests.json"
    metrics_path = output_dir / "metrics.json"
    request_sha, request_size = _write_json_exclusive(request_path, requests)
    metrics_sha, metrics_size = _write_json_exclusive(metrics_path, metrics_groups)
    summary["artifacts"] = {
        "requests.json": {
            "sha256": request_sha,
            "size_bytes": request_size,
        },
        "metrics.json": {
            "sha256": metrics_sha,
            "size_bytes": metrics_size,
        },
        "plan.json": {
            "sha256": plan_sha256,
            "size_bytes": plan_size_bytes,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_sha, summary_size = _write_json_exclusive(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "run_id": summary["run"]["run_id"],
        "created_at": _utc_now(),
        "artifacts": {
            "requests.json": {
                "sha256": request_sha,
                "size_bytes": request_size,
            },
            "metrics.json": {
                "sha256": metrics_sha,
                "size_bytes": metrics_size,
            },
            "summary.json": {
                "sha256": summary_sha,
                "size_bytes": summary_size,
            },
            "plan.json": {
                "sha256": plan_sha256,
                "size_bytes": plan_size_bytes,
            },
        },
        "inputs": {
            str(path): {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_paths
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_sha, manifest_size = _write_json_exclusive(manifest_path, manifest)
    return {
        "output_dir": str(output_dir),
        "manifest_sha256": manifest_sha,
        "manifest_size_bytes": manifest_size,
        "summary_sha256": summary_sha,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--metrics-url",
        help="Prometheus endpoint; defaults to BASE_URL/metrics",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-token-json",
        action="append",
        type=Path,
        required=True,
        help="Repeat exactly twice for independent prompt-token JSON inputs",
    )
    parser.add_argument("--server-provenance-json", type=Path, required=True)
    parser.add_argument("--source-provenance-json", type=Path, required=True)
    parser.add_argument("--config-provenance-json", type=Path, required=True)
    parser.add_argument("--pvc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-timeout-s", type=float, default=7_200.0)
    parser.add_argument("--metrics-interval-s", type=float, default=0.2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the exact plan without HTTP or writes",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not _valid_run_id(args.run_id):
        raise ContractError("--run-id contains unsafe characters")
    args.base_url = _validate_url(args.base_url, label="--base-url")
    args.metrics_url = _validate_url(
        args.metrics_url or args.base_url + "/metrics",
        label="--metrics-url",
    )
    if args.request_timeout_s <= 0:
        raise ContractError("--request-timeout-s must be positive")
    if not 0.05 <= args.metrics_interval_s <= 5.0:
        raise ContractError("--metrics-interval-s must be between 0.05 and 5")


def _contract_record() -> dict[str, Any]:
    return {
        "prompt_token_count_per_slot": PROMPT_TOKEN_COUNT,
        "output_token_count_per_slot": OUTPUT_TOKEN_COUNT,
        "max_model_len": MAX_MODEL_LEN,
        "batch_sizes": [1, 2],
        "paired_repeats": PAIRED_REPEATS,
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
        "return_token_ids": True,
        "unique_cache_salt_per_slot_repeat_phase": True,
        "token_clock": "time.monotonic_ns",
        "prompt_hash": "sha256-canonical-json-int-array-v1",
    }


def _failure_artifact(output_dir: Path, exc: BaseException) -> None:
    if not output_dir.is_dir():
        return
    path = output_dir / "failure.json"
    if path.exists():
        return
    _write_json_exclusive(
        path,
        {
            "schema_version": 1,
            "status": "failed",
            "failed_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    if len(args.prompt_token_json) != 2:
        raise ContractError("--prompt-token-json must be provided exactly twice")
    prompts = [
        _load_prompt(path, slot)
        for path, slot in zip(args.prompt_token_json, ("a", "b"))
    ]
    _validate_prompt_independence(prompts)
    provenance = {
        "server": _load_provenance(args.server_provenance_json, "server"),
        "source": _load_provenance(args.source_provenance_json, "source"),
        "config": _load_provenance(args.config_provenance_json, "config"),
    }
    if provenance["server"].value["served_model"] != args.model:
        raise ContractError("server provenance served_model does not match --model")
    if "hy3" not in args.model.casefold():
        raise ContractError("--model must identify the Hy3 deployment")

    input_paths = [
        *args.prompt_token_json,
        args.server_provenance_json,
        args.source_provenance_json,
        args.config_provenance_json,
    ]
    plan = {
        "schema_version": 1,
        "benchmark": "forge001_hy3_exact_max_context_service",
        "dry_run": args.dry_run,
        "run_id": args.run_id,
        "base_url": args.base_url,
        "metrics_url": args.metrics_url,
        "model": args.model,
        "contract": _contract_record(),
        "prompts": [prompt.metadata() for prompt in prompts],
        "provenance": {kind: item.record() for kind, item in provenance.items()},
        "phase_order": [
            {
                "repeat": repeat,
                "phases": ["b1", "b2"] if repeat % 2 else ["b2", "b1"],
                "b1_slot_order": ["a", "b"] if repeat % 2 else ["b", "a"],
                "b2_slots": ["a", "b"],
            }
            for repeat in range(1, PAIRED_REPEATS + 1)
        ],
        "scored_request_count": PAIRED_REPEATS * 4,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    _, output_dir = _validate_pvc_paths(
        pvc_root=args.pvc_root,
        output_root=args.output_root / args.run_id,
        inputs=input_paths,
    )
    started_at = _utc_now()
    output_dir.mkdir(mode=0o750)
    plan_sha256, plan_size_bytes = _write_json_exclusive(
        output_dir / "plan.json",
        plan,
    )
    try:
        preflight, groups = asyncio.run(_run_benchmark(args=args, prompts=prompts))
        salts = [
            record["cache_salt_sha256"]
            for group in groups
            for record in group["records"]
        ]
        if len(salts) != PAIRED_REPEATS * 4 or len(set(salts)) != len(salts):
            raise ContractError("cache salts were not unique across scored requests")
        statistics_record = _paired_statistics(groups)
        summary = {
            "schema_version": 1,
            "benchmark": "forge001_hy3_exact_max_context_service",
            "status": "passed",
            "contract": _contract_record(),
            "run": {
                "run_id": args.run_id,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "base_url": args.base_url,
                "metrics_url": args.metrics_url,
                "model": args.model,
                "pvc_root": str(args.pvc_root.resolve()),
                "output_dir": str(output_dir),
            },
            "prompts": [prompt.metadata() for prompt in prompts],
            "provenance": {kind: item.record() for kind, item in provenance.items()},
            "preflight": preflight,
            "phase_order": plan["phase_order"],
            "scored_request_count": len(salts),
            "unique_cache_salt_hash_count": len(set(salts)),
            "statistics": statistics_record,
            "gates": {
                "exact_prompt_lengths": True,
                "independent_prompt_sources": True,
                "exact_output_lengths": True,
                "monotonic_token_timestamps": True,
                "five_paired_repeats": True,
                "unique_cache_salts": True,
                "batch2_simultaneously_running": True,
                "batch2_never_queued": True,
                "exact_provenance": True,
                "pvc_output": True,
            },
        }
        artifact = _write_artifacts(
            output_dir=output_dir,
            summary=summary,
            groups=groups,
            input_paths=input_paths,
            plan_sha256=plan_sha256,
            plan_size_bytes=plan_size_bytes,
        )
    except BaseException as exc:
        _failure_artifact(output_dir, exc)
        raise
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
