"""Fail-closed transport utilities for the DCU preflight worker.

This module deliberately contains only the project-owned boundary between the
CPU orchestration process and a future DCU model process.  It does not import
LeRobot, construct a model, or select an accelerator.  Tensor payloads are
persisted as ``safetensors`` files and control messages are one JSON object per
line.  The real model/CLI implementation is kept out of this contract layer so
that the boundary can be tested without a DCU or a simulator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Protocol, TypeAlias

import torch


class DCUPreflightError(RuntimeError):
    """Base error for a violated preflight contract."""


# A ``-1`` dimension is an explicitly variable sequence length.  It is used
# only for language tensors in REQUEST_SCHEMA; all model/action dimensions
# remain exact.
TensorSpec: TypeAlias = tuple[tuple[int, ...], tuple[torch.dtype, ...]]
Schema: TypeAlias = Mapping[str, Any]


# The request contains the exact tensors needed by the frozen SmolVLA action
# API.  Dtypes include the CPU reference and the DCU model's BF16 execution
# path; shape is intentionally fixed for this pilot contract.
REQUEST_SCHEMA: dict[str, TensorSpec] = {
    "observation.state": ((1, 8), (torch.float32, torch.bfloat16)),
    "observation.images.image": ((1, 3, 360, 360), (torch.float32, torch.bfloat16)),
    "observation.images.image2": ((1, 3, 360, 360), (torch.float32, torch.bfloat16)),
    "observation.language.tokens": ((1, -1), (torch.int64,)),
    "observation.language.attention_mask": ((1, -1), (torch.bool,)),
    "noise": ((1, 50, 32), (torch.float32, torch.bfloat16)),
}

# A response is either a complete normalized action chunk or one action
# selected from the worker's queue.  The explicit one-of handling is useful:
# accepting both keys would hide a malformed response.
RESPONSE_SCHEMA: dict[str, Any] = {
    "action_chunk": ((1, 50, 7), (torch.float32, torch.bfloat16)),
    "action": ((1, 7), (torch.float32, torch.bfloat16)),
}

_RESPONSE_KEYS = frozenset(RESPONSE_SCHEMA)
_COMMANDS = frozenset({"reset", "predict_action_chunk", "select_action", "ping", "shutdown", "close"})


def _load_safetensors_module() -> Any:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
    except Exception as exc:  # pragma: no cover - depends on runtime image
        raise DCUPreflightError("safetensors is required for the DCU tensor boundary") from exc
    return safe_open, load_file, save_file


def _normalise_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.name != candidate.name.strip() or not candidate.name:
        raise DCUPreflightError("tensor bundle path must name a regular .safetensors file")
    if candidate.suffix != ".safetensors":
        raise DCUPreflightError(f"tensor bundle must use .safetensors: {candidate}")
    return candidate


def _normalise_spec(name: str, raw: Any) -> TensorSpec:
    """Validate and normalize the compact ``(shape, dtypes)`` schema form."""

    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise DCUPreflightError(f"invalid schema for tensor {name!r}")
    raw_shape, raw_dtypes = raw
    try:
        shape = tuple(int(item) for item in raw_shape)
    except (TypeError, ValueError) as exc:
        raise DCUPreflightError(f"invalid shape schema for tensor {name!r}") from exc
    if any(item < -1 for item in shape):
        raise DCUPreflightError(f"negative shape schema for tensor {name!r}")
    if isinstance(raw_dtypes, torch.dtype):
        dtypes = (raw_dtypes,)
    else:
        try:
            dtypes = tuple(raw_dtypes)
        except TypeError as exc:
            raise DCUPreflightError(f"invalid dtype schema for tensor {name!r}") from exc
    if not dtypes or not all(isinstance(item, torch.dtype) for item in dtypes):
        raise DCUPreflightError(f"invalid dtype schema for tensor {name!r}")
    return shape, dtypes


def _schema_alternatives(schema: Schema) -> tuple[dict[str, TensorSpec], ...]:
    """Return exact allowed key sets for a schema.

    ``REQUEST_SCHEMA`` is one exact mapping.  ``RESPONSE_SCHEMA`` is the
    intentionally small one-of mapping described above.  Custom callers may
    also pass ``{"__alternatives__": [mapping, ...]}`` without changing the
    public API.
    """

    if not isinstance(schema, Mapping):
        raise DCUPreflightError("tensor schema must be a mapping")
    explicit = schema.get("__alternatives__")
    if explicit is not None:
        if not isinstance(explicit, Iterable) or isinstance(explicit, (str, bytes)):
            raise DCUPreflightError("schema alternatives must be an iterable of mappings")
        alternatives: list[dict[str, TensorSpec]] = []
        for index, item in enumerate(explicit):
            if not isinstance(item, Mapping) or not item:
                raise DCUPreflightError(f"invalid schema alternative {index}")
            alternatives.append({str(key): _normalise_spec(str(key), value) for key, value in item.items()})
        if not alternatives:
            raise DCUPreflightError("schema alternatives cannot be empty")
        return tuple(alternatives)

    if schema.keys() == _RESPONSE_KEYS:
        # Keep RESPONSE_SCHEMA ergonomic for callers: one of the two response
        # keys is required, never both and never neither.
        return (
            {"action_chunk": _normalise_spec("action_chunk", schema["action_chunk"])},
            {"action": _normalise_spec("action", schema["action"])},
        )
    return ({str(key): _normalise_spec(str(key), value) for key, value in schema.items()},)


def _validate_tensor_value(name: str, tensor: Any, spec: TensorSpec) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        raise DCUPreflightError(f"tensor bundle value {name!r} is not a torch.Tensor")
    shape, dtypes = spec
    actual_shape = tuple(int(item) for item in tensor.shape)
    if len(actual_shape) != len(shape) or any(
        expected != -1 and actual != expected
        for actual, expected in zip(actual_shape, shape)
    ):
        raise DCUPreflightError(f"tensor {name!r} shape {actual_shape} does not match {shape}")
    if tensor.dtype not in dtypes:
        allowed = ", ".join(str(dtype) for dtype in dtypes)
        raise DCUPreflightError(f"tensor {name!r} dtype {tensor.dtype} is not one of {allowed}")
    if tensor.is_floating_point() or tensor.is_complex():
        finite = bool(torch.isfinite(tensor.detach()).all().item())
        if not finite:
            raise DCUPreflightError(f"tensor {name!r} contains non-finite values")
    else:
        finite = True
    return {
        "shape": list(actual_shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": finite,
        "numel": int(tensor.numel()),
    }


def _select_schema(bundle: Mapping[str, Any], schema: Schema) -> dict[str, TensorSpec]:
    alternatives = _schema_alternatives(schema)
    keys = set(bundle)
    for alternative in alternatives:
        if keys == set(alternative):
            return alternative
    expected = " or ".join(sorted(str(sorted(item)) for item in alternatives))
    raise DCUPreflightError(f"tensor bundle keys {sorted(keys)} do not match schema {expected}")


def validate_tensor_bundle(bundle: Mapping[str, Any], *, schema: Schema) -> dict[str, Any]:
    """Validate exact keys, shapes, dtypes, devices and finite values.

    The returned metadata is JSON-safe and intentionally does not expose the
    tensor payload.  Input tensors are never moved or mutated.
    """

    if not isinstance(bundle, Mapping):
        raise DCUPreflightError("tensor bundle must be a mapping")
    selected = _select_schema(bundle, schema)
    tensors: dict[str, Any] = {}
    finite = True
    for name, spec in selected.items():
        description = _validate_tensor_value(name, bundle[name], spec)
        tensors[name] = description
        finite = finite and bool(description["finite"])
    return {
        "keys": sorted(tensors),
        "tensors": tensors,
        "finite": finite,
    }


def save_tensor_bundle(path: str | Path, bundle: Mapping[str, torch.Tensor], *, schema: Schema) -> Path:
    """Validate and write a tensor bundle as a new safetensors file."""

    target = _normalise_path(path)
    validate_tensor_bundle(bundle, schema=schema)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _, _, save_file = _load_safetensors_module()
    # ``save_file`` requires CPU-compatible dense tensors.  Clone into a
    # contiguous detached view; the caller's tensors remain untouched.
    payload = {
        str(name): tensor.detach().contiguous().cpu()
        for name, tensor in bundle.items()
    }
    try:
        save_file(payload, str(target), metadata={"shiftvla_schema": "tensor-bundle-v1"})
    except Exception:
        # Do not leave a misleading partial artifact behind.  The path is
        # explicit and newly allocated, so unlinking it is recoverable.
        if target.exists():
            target.unlink()
        raise
    return target


def load_tensor_bundle(path: str | Path, *, schema: Schema) -> dict[str, torch.Tensor]:
    """Load, then validate, a local safetensors tensor bundle."""

    target = _normalise_path(path)
    if not target.is_file():
        raise DCUPreflightError(f"tensor bundle does not exist as a regular file: {target}")
    _, load_file, _ = _load_safetensors_module()
    try:
        bundle = load_file(str(target), device="cpu")
    except Exception as exc:
        raise DCUPreflightError(f"could not load safetensors bundle {target}: {exc}") from exc
    validate_tensor_bundle(bundle, schema=schema)
    return dict(bundle)


def encode_json_line(message: Mapping[str, Any]) -> str:
    """Encode one strict JSON control message, including its newline."""

    if not isinstance(message, Mapping):
        raise DCUPreflightError("worker message must be a JSON object")
    payload = dict(message)
    identifier = payload.get("id")
    command = payload.get("command")
    if not isinstance(identifier, str) or not identifier:
        raise DCUPreflightError("worker message requires a non-empty string id")
    if not isinstance(command, str) or command not in _COMMANDS:
        allowed = ", ".join(sorted(_COMMANDS))
        raise DCUPreflightError(f"worker message requires a supported command ({allowed})")
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DCUPreflightError(f"worker message is not strict JSON: {exc}") from exc
    if "\n" in encoded or "\r" in encoded:
        raise DCUPreflightError("worker message may not contain raw line breaks")
    return encoded + "\n"


def decode_json_line(line: str | bytes) -> dict[str, Any]:
    """Decode exactly one newline-terminated JSON object."""

    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DCUPreflightError("worker message is not UTF-8") from exc
    if not isinstance(line, str) or not line.endswith("\n"):
        raise DCUPreflightError("worker message must be exactly one newline-terminated JSON line")
    if line.count("\n") != 1 or "\r" in line:
        raise DCUPreflightError("worker message must contain exactly one JSON line")
    raw = line[:-1]
    if not raw.strip():
        raise DCUPreflightError("worker message cannot be empty")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DCUPreflightError(f"worker message is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise DCUPreflightError("worker message must decode to a JSON object")
    # Re-run the same structural validation as the encoder.  Decode is not a
    # bypass around the protocol contract.
    encoded = encode_json_line(value)
    if encoded[:-1] != raw:
        # This rejects duplicate keys and non-canonical spellings that would
        # be silently normalized by json.loads/dumps.
        raise DCUPreflightError("worker message is not canonical strict JSON")
    return value


def _decode_response_line(line: str | bytes) -> dict[str, Any]:
    """Decode one canonical response line without imposing client semantics."""

    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DCUPreflightError("worker response is not UTF-8") from exc
    if not isinstance(line, str) or not line.endswith("\n"):
        raise DCUPreflightError("worker response must be exactly one newline-terminated JSON line")
    if line.count("\n") != 1 or "\r" in line:
        raise DCUPreflightError("worker response must contain exactly one JSON line")
    raw = line[:-1]
    if not raw.strip():
        raise DCUPreflightError("worker response cannot be empty")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DCUPreflightError(f"worker response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise DCUPreflightError("worker response must be a JSON object")
    try:
        canonical = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DCUPreflightError(f"worker response is not strict JSON: {exc}") from exc
    if canonical != raw:
        raise DCUPreflightError("worker response is not canonical strict JSON")
    identifier = value.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise DCUPreflightError("worker response requires a non-empty string id")
    return value


class SubprocessWorkerTransport:
    """Fail-closed canonical JSONL transport for a dedicated worker process."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        shutdown_timeout: float = 5.0,
    ) -> None:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
            raise DCUPreflightError("worker argv must be a non-empty sequence of strings")
        command = tuple(argv)
        if not all(isinstance(item, str) and item and "\x00" not in item for item in command):
            raise DCUPreflightError("worker argv must contain non-empty strings without NUL")
        if not isinstance(command[0], str) or not command[0].strip():
            raise DCUPreflightError("worker argv[0] must be a non-empty executable")
        if isinstance(shutdown_timeout, bool):
            raise DCUPreflightError("worker shutdown_timeout must be a finite non-negative number")
        try:
            timeout = float(shutdown_timeout)
        except (TypeError, ValueError) as exc:
            raise DCUPreflightError("worker shutdown_timeout must be a finite non-negative number") from exc
        if timeout < 0 or timeout != timeout or timeout == float("inf"):
            raise DCUPreflightError("worker shutdown_timeout must be a finite non-negative number")
        if env is None:
            process_env = os.environ.copy()
        else:
            if not isinstance(env, Mapping):
                raise DCUPreflightError("worker env must be a mapping of strings")
            process_env = {}
            for key, value in env.items():
                if (
                    not isinstance(key, str)
                    or not key
                    or "\x00" in key
                    or not isinstance(value, str)
                    or "\x00" in value
                ):
                    raise DCUPreflightError("worker env keys and values must be strings without NUL")
                process_env[key] = value

        self.argv = command
        self.env = dict(process_env)
        self.process: subprocess.Popen[str]
        self._shutdown_timeout = timeout
        self._lock = threading.Lock()
        self._request_number = 0
        self._closed = False
        self._failed = False
        self._stderr_chunks: list[str] = []
        self._stderr_lock = threading.Lock()
        try:
            self.process = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd) if cwd is not None else None,
                env=self.env,
                shell=False,
                text=True,
                encoding="utf-8",
                errors="strict",
                bufsize=1,
            )
        except (OSError, ValueError) as exc:
            raise DCUPreflightError(f"could not start worker process: {exc}") from exc
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="shiftvla-worker-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        try:
            for chunk in stream:
                with self._stderr_lock:
                    self._stderr_chunks.append(chunk)
        except (OSError, UnicodeError, ValueError):
            return

    @property
    def stderr_text(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_chunks)

    def _ensure_open(self) -> None:
        if self._closed:
            raise DCUPreflightError("worker transport is closed")
        if self._failed:
            raise DCUPreflightError("worker transport is unusable after a protocol failure")
        return_code = self.process.poll()
        if return_code is not None:
            self._failed = True
            raise DCUPreflightError(f"worker exited before request with return code {return_code}")
        if self.process.stdin is None or self.process.stdout is None:
            self._failed = True
            raise DCUPreflightError("worker transport pipes are unavailable")

    def _request_locked(self, command: str, payload: Mapping[str, object]) -> Mapping[str, Any]:
        self._ensure_open()
        if "id" in payload or "command" in payload:
            raise DCUPreflightError("worker request payload may not override id or command")
        self._request_number += 1
        identifier = f"request-{self._request_number}"
        message = {"id": identifier, "command": command, **dict(payload)}
        line = encode_json_line(message)
        try:
            self.process.stdin.write(line)
            self.process.stdin.flush()
            response_line = self.process.stdout.readline()
        except (BrokenPipeError, OSError, UnicodeError, ValueError) as exc:
            self._failed = True
            raise DCUPreflightError(f"worker request failed: {exc}") from exc
        if response_line == "":
            self._failed = True
            return_code = self.process.poll()
            raise DCUPreflightError(
                "worker exited before response"
                if return_code is None
                else f"worker exited before response with return code {return_code}"
            )
        try:
            response = _decode_response_line(response_line)
        except DCUPreflightError:
            self._failed = True
            raise
        if response["id"] != identifier:
            self._failed = True
            raise DCUPreflightError(
                f"worker response id {response['id']!r} does not match request id {identifier!r}"
            )
        return response

    def request(self, command: str, **payload: object) -> Mapping[str, Any]:
        """Send one command and return its matching canonical response."""

        with self._lock:
            return self._request_locked(command, payload)

    def _finish_locked(self, *, force: bool) -> None:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except (OSError, ValueError):
            pass
        if self.process.poll() is None and force:
            self.process.terminate()
        try:
            self.process.wait(timeout=self._shutdown_timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._stderr_thread.join(timeout=self._shutdown_timeout)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        self._closed = True

    def shutdown(self) -> Mapping[str, Any]:
        """Request worker shutdown and wait for a clean process exit."""

        with self._lock:
            if self._closed:
                raise DCUPreflightError("worker transport is closed")
            try:
                response = self._request_locked("shutdown", {})
                self._finish_locked(force=False)
            except Exception:
                self._failed = True
                self._finish_locked(force=True)
                raise
            return response

    def close(self) -> None:
        """Shut down the worker, forcefully cleaning up after protocol failure."""

        with self._lock:
            if self._closed:
                return
            if not self._failed and self.process.poll() is None:
                try:
                    self._request_locked("shutdown", {})
                except Exception:
                    self._failed = True
            self._finish_locked(force=self._failed)


class WorkerTransport(Protocol):
    """Minimal transport contract consumed by :class:`DCUWorkerClient`."""

    def request(self, command: str, **payload: object) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _require_response(response: Mapping[str, Any], command: str) -> Mapping[str, Any]:
    if not isinstance(response, Mapping):
        raise DCUPreflightError(f"worker {command} response must be a JSON object")
    if response.get("ok") is not True:
        detail = response.get("error", "unknown worker error")
        raise DCUPreflightError(f"worker {command} failed: {detail}")
    return response


def _queue_value(response: Mapping[str, Any], key: str) -> int | None:
    value = response.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DCUPreflightError(f"worker response {key} must be a non-negative integer")
    return value


class DCUWorkerClient:
    """Client-side queue/shape contract for a dedicated DCU worker.

    The client does not run model code.  A transport object owns the process
    and IPC details; this class validates reset ordering, persisted response
    tensors, action dimensions and the queue evidence reported by the worker.
    """

    def __init__(
        self,
        *,
        transport: WorkerTransport,
        action_dim: int = 7,
        chunk_size: int = 50,
        n_action_steps: int = 1,
    ) -> None:
        if action_dim <= 0 or chunk_size <= 0 or n_action_steps <= 0:
            raise DCUPreflightError("worker client dimensions must be positive")
        if n_action_steps > chunk_size:
            raise DCUPreflightError("n_action_steps cannot exceed chunk_size")
        if not hasattr(transport, "request"):
            raise DCUPreflightError("worker transport must provide request()")
        self.transport = transport
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.n_action_steps = int(n_action_steps)
        self._reset_done = False
        self.last_queue_evidence: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise DCUPreflightError("worker reset seed must be a non-negative integer")
        payload = {} if seed is None else {"seed": seed}
        response = _require_response(self.transport.request("reset", **payload), "reset")
        before = _queue_value(response, "queue_length_before")
        after = _queue_value(response, "queue_length_after")
        reported = _queue_value(response, "queue_length")
        if reported is not None and reported != 0:
            raise DCUPreflightError(f"worker reset left a non-empty action queue: {reported}")
        if before is not None and before != 0:
            raise DCUPreflightError(f"worker reset queue_length_before must be 0, got {before}")
        if after is not None and after != 0:
            raise DCUPreflightError(f"worker reset queue_length_after must be 0, got {after}")
        self._reset_done = True
        self.last_queue_evidence = {
            "queue_length_before": before,
            "queue_length_after": after,
            "queue_length": reported,
        }
        return response

    def _require_reset(self) -> None:
        if not self._reset_done:
            raise DCUPreflightError("worker client must be reset before requesting an action")

    @staticmethod
    def _response_path(response: Mapping[str, Any], command: str) -> Path:
        path = response.get("response_path")
        if not isinstance(path, str) or not path:
            raise DCUPreflightError(f"worker {command} response requires response_path")
        return Path(path)

    def predict_action_chunk(self, request_path: str | Path) -> torch.Tensor:
        self._require_reset()
        response = _require_response(
            self.transport.request("predict_action_chunk", request_path=str(request_path)),
            "predict_action_chunk",
        )
        bundle = load_tensor_bundle(self._response_path(response, "predict_action_chunk"), schema=RESPONSE_SCHEMA)
        if set(bundle) != {"action_chunk"}:
            raise DCUPreflightError("predict_action_chunk response must contain action_chunk only")
        chunk = bundle["action_chunk"]
        if tuple(chunk.shape) != (1, self.chunk_size, self.action_dim):
            raise DCUPreflightError(
                f"action chunk shape {tuple(chunk.shape)} does not match "
                f"(1, {self.chunk_size}, {self.action_dim})"
            )
        self.last_queue_evidence = {
            "queue_length_before": _queue_value(response, "queue_length_before"),
            "queue_length_after": _queue_value(response, "queue_length_after"),
            "new_chunk_generated": True,
        }
        return chunk

    def select_action(self, request_path: str | Path) -> torch.Tensor:
        self._require_reset()
        response = _require_response(
            self.transport.request("select_action", request_path=str(request_path)),
            "select_action",
        )
        bundle = load_tensor_bundle(self._response_path(response, "select_action"), schema=RESPONSE_SCHEMA)
        if set(bundle) != {"action"}:
            raise DCUPreflightError("select_action response must contain action only")
        action = bundle["action"]
        if tuple(action.shape) != (1, self.action_dim):
            raise DCUPreflightError(f"action shape {tuple(action.shape)} does not match (1, {self.action_dim})")
        before = _queue_value(response, "queue_length_before")
        after = _queue_value(response, "queue_length_after")
        generated = response.get("new_chunk_generated")
        if not isinstance(generated, bool):
            raise DCUPreflightError("worker select_action response requires boolean new_chunk_generated")
        if before is not None and after is not None:
            if generated and before != 0:
                raise DCUPreflightError("new_chunk_generated must start with an empty queue")
            if after >= before and not generated:
                raise DCUPreflightError("select_action must consume a queued action")
        self.last_queue_evidence = {
            "queue_length_before": before,
            "queue_length_after": after,
            "new_chunk_generated": generated,
        }
        return action

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close is not None:
            close()
