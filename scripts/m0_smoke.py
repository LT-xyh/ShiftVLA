#!/usr/bin/env python3
"""One-episode, fail-closed M0 SmolVLA smoke harness.

The official LeRobot factories and rollout own model preprocessing, action
queueing, environment stepping, and termination.  This module only wraps those
callables to record contracts and latency without changing their behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
RENDERER_ENV = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
}
ACTION_KEY = "action"
EXPECTED_PROJECT_SHA = "5f412cadc2e591e6af9231d973b38b266c34f948"


def _finite(value: Any) -> bool | None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value).all().item())
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return bool(np.isfinite(value).all())
        return True
    if isinstance(value, (float, int, np.floating, np.integer)):
        return bool(math.isfinite(float(value)))
    return None


def describe(value: Any) -> Any:
    """Return JSON-safe shape/dtype/finite metadata without exposing tensors."""
    if isinstance(value, torch.Tensor):
        result: dict[str, Any] = {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": _finite(value),
        }
        if value.numel() and (value.is_floating_point() or value.is_complex()):
            result["min"] = float(value.detach().amin().item())
            result["max"] = float(value.detach().amax().item())
        return result
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": _finite(value),
        }
    if isinstance(value, Mapping):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [describe(item) for item in value],
        }
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _copy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    return value


def _as_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _action_from_transition(value: Any) -> Any:
    if isinstance(value, Mapping) and ACTION_KEY in value:
        return value[ACTION_KEY]
    return value


class TraceStore:
    """Mutable per-rollout evidence store used by the transparent proxies."""

    def __init__(self, chunk_size: int, n_action_steps: int, action_dim: int = 7) -> None:
        self.chunk_size = int(chunk_size)
        self.n_action_steps = int(n_action_steps)
        self.action_dim = int(action_dim)
        self.decisions: list[dict[str, Any]] = []
        self.inference_latencies: list[float] = []
        self.env_step_latencies: list[float] = []
        self.render_latencies: list[float] = []
        self.render_count = 0
        self.gl: dict[str, Any] = {}
        self._current: dict[str, Any] | None = None
        self._raw: dict[str, Any] = {}

    @property
    def current(self) -> dict[str, Any] | None:
        return self._current

    def begin_decision(self) -> dict[str, Any]:
        decision = {
            "decision_index": len(self.decisions),
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
        }
        self.decisions.append(decision)
        self._current = decision
        self._raw = {}
        return decision

    def _require_current(self) -> dict[str, Any]:
        if self._current is None:
            return self.begin_decision()
        return self._current

    def record(self, key: str, value: Any) -> None:
        decision = self._require_current()
        decision[key] = describe(value)
        self._raw[key] = _copy_value(value)

    def update(self, **values: Any) -> None:
        self._require_current().update(values)

    def raw(self, key: str) -> Any:
        return self._raw.get(key)


def _queue_length(policy: Any) -> int | None:
    queues = getattr(policy, "_queues", None)
    if not isinstance(queues, Mapping):
        return None
    queue = queues.get(ACTION_KEY)
    try:
        return len(queue) if queue is not None else None
    except TypeError:
        return None


class ProcessorProxy:
    """Delegate a processor pipeline and record only input/output metadata."""

    def __init__(self, delegate: Any, name: str, trace: TraceStore) -> None:
        self.delegate = delegate
        self.name = name
        self.trace = trace

    def __call__(self, value: Any) -> Any:
        if self.name == "env_processor":
            self.trace.begin_decision()
            self.trace.record("observation_env_processor_input", value)
        elif self.name == "policy_processor":
            self.trace.record("observation_policy_processor_input", value)
        elif self.name == "postprocessor":
            self.trace.record("postprocessor_input", value)
        elif self.name == "env_postprocessor":
            self.trace.record("env_postprocessor_input", value)

        result = self.delegate(value)

        if self.name == "env_processor":
            self.trace.record("observation_env_processor", result)
        elif self.name == "policy_processor":
            self.trace.record("observation_policy_processor", result)
        elif self.name == "postprocessor":
            self.trace.record("postprocessed_action", result)
        elif self.name == "env_postprocessor":
            self.trace.record("env_postprocessed_action", _action_from_transition(result))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class PolicyProxy(nn.Module):
    """An nn.Module wrapper that leaves official policy calls untouched."""

    def __init__(self, delegate: nn.Module, trace: TraceStore) -> None:
        super().__init__()
        self._delegate = delegate
        self.trace = trace
        self.call_count = 0

    def reset(self) -> None:
        self._delegate.reset()

    def select_action(self, batch: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        if self.trace.current is None:
            self.trace.begin_decision()
        before = _queue_length(self._delegate)
        started = time.perf_counter()
        result = self._delegate.select_action(batch, *args, **kwargs)
        elapsed = time.perf_counter() - started
        after = _queue_length(self._delegate)
        self.call_count += 1
        self.trace.record("normalized_action", result)
        self.trace.inference_latencies.append(elapsed)
        self.trace.update(
            queue_length_before=before,
            queue_length_after=after,
            new_chunk_generated=(before == 0),
            inference_latency_seconds=elapsed,
        )
        return result

    def eval(self) -> PolicyProxy:
        self._delegate.eval()
        super().eval()
        return self

    def train(self, mode: bool = True) -> PolicyProxy:
        self._delegate.train(mode)
        super().train(mode)
        return self

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, "_modules")
            delegate = modules.get("_delegate")
            if delegate is None:
                raise
            return getattr(delegate, name)


class VectorEnvProxy:
    """Delegate a vector environment while validating the 1x7 action seam."""

    def __init__(self, delegate: Any, trace: TraceStore) -> None:
        self.delegate = delegate
        self.trace = trace

    def step(self, action: Any) -> Any:
        if not isinstance(action, np.ndarray):
            raise TypeError(f"official rollout must pass a NumPy action, got {type(action).__name__}")
        expected_shape = (int(self.delegate.num_envs), self.trace.action_dim)
        if action.shape != expected_shape:
            raise ValueError(f"vector env action shape mismatch: expected {expected_shape}, got {action.shape}")
        if action.dtype != np.dtype(np.float32):
            raise TypeError(f"vector env action dtype must be float32, got {action.dtype}")
        if not np.isfinite(action).all():
            raise ValueError("vector env action is non-finite")

        self.trace.record("env_step_action", action)
        postprocessed = _as_numpy(_action_from_transition(self.trace.raw("env_postprocessed_action")))
        if postprocessed is None:
            postprocessed = _as_numpy(self.trace.raw("postprocessed_action"))
        if postprocessed is None:
            raise RuntimeError("missing postprocessed action before env.step")
        if postprocessed.shape != action.shape or not np.array_equal(postprocessed, action):
            self.trace.update(env_step_equals_postprocessed=False)
            raise RuntimeError("env.step action differs from postprocessed action")
        self.trace.update(env_step_equals_postprocessed=True)

        started = time.perf_counter()
        result = self.delegate.step(action)
        elapsed = time.perf_counter() - started
        self.trace.env_step_latencies.append(elapsed)
        observation, reward, terminated, truncated, info = result
        env_done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        self.trace.update(
            env_step_latency_seconds=elapsed,
            reward=np.asarray(reward).tolist(),
            terminated=np.asarray(terminated, dtype=bool).tolist(),
            truncated=np.asarray(truncated, dtype=bool).tolist(),
            env_done=env_done.tolist(),
            done=env_done.tolist(),
            success=_success_values(info, int(self.delegate.num_envs)),
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _success_values(info: Any, n_envs: int) -> list[bool]:
    if not isinstance(info, Mapping):
        return [False] * n_envs
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, Mapping):
            value = final_info.get("is_success", [False] * n_envs)
            values = np.asarray(value).reshape(-1).tolist()
            return [bool(item) for item in (values or [False] * n_envs)][:n_envs]
        values: list[bool] = []
        for item in final_info:
            values.append(bool(item.get("is_success", False)) if isinstance(item, Mapping) else False)
        return (values + [False] * n_envs)[:n_envs]
    if "is_success" in info:
        values = np.asarray(info["is_success"]).reshape(-1).tolist()
        return [bool(item) for item in (values or [False] * n_envs)][:n_envs]
    return [False] * n_envs


def _gl_evidence() -> dict[str, Any]:
    try:
        from OpenGL import GL

        def text(name: int) -> str | None:
            value = GL.glGetString(name)
            return value.decode() if value else None

        return {
            "vendor": text(GL.GL_VENDOR),
            "renderer": text(GL.GL_RENDERER),
            "version": text(GL.GL_VERSION),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _assert_renderer_environment() -> dict[str, Any]:
    mismatches = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in RENDERER_ENV.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"renderer environment must be exact EGL settings: {mismatches}")
    return {"variables": dict(RENDERER_ENV), "exact": True}


def _assert_gl_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"GL identity unavailable: {identity!r}")
    if "error" in identity:
        raise RuntimeError(f"GL identity probe failed: {identity['error']}")
    missing = [
        key
        for key in ("vendor", "renderer", "version")
        if not isinstance(identity.get(key), str) or not identity[key].strip()
    ]
    if missing:
        raise RuntimeError(f"GL identity is incomplete; missing {missing}")
    return dict(identity)


def make_render_callback(trace: TraceStore):
    def render_callback(env: Any) -> None:
        started = time.perf_counter()
        frames = env.call("render")
        elapsed = time.perf_counter() - started
        if not trace.gl:
            trace.gl = _gl_evidence()
        _assert_gl_identity(trace.gl)
        trace.render_latencies.append(elapsed)
        trace.render_count += 1
        if trace.current is not None:
            trace.current["render_latency_seconds"] = elapsed
        frame = frames[0] if isinstance(frames, (list, tuple)) else frames
        metadata = describe(frame)
        if metadata.get("finite") is False:
            raise ValueError("render callback received a non-finite frame")

    return render_callback


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p95": float(np.percentile(np.asarray(values), 95)),
        "max": max(values),
        "total": sum(values),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_runtime_lock(path: Path) -> dict[str, Any]:
    """Parse and self-validate the immutable runtime lock payload."""
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    marker = b"[pip_freeze_all]\n"
    if raw.count(marker) != 1:
        raise RuntimeError(f"runtime lock must contain exactly one final {marker!r} section")
    prefix, freeze_bytes = raw.split(marker, 1)
    if not freeze_bytes or not freeze_bytes.endswith(b"\n"):
        raise RuntimeError("runtime lock pip freeze payload must be non-empty and newline-terminated")
    try:
        prefix_text = prefix.decode("utf-8")
        freeze_text = freeze_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"runtime lock is not valid UTF-8: {path}") from exc
    if any(line.startswith("[") and line.endswith("]") for line in freeze_text.splitlines()):
        raise RuntimeError("runtime lock pip freeze payload contains an unexpected section")

    metadata: dict[str, str] = {}
    section: str | None = None
    pip_check_lines: list[str] | None = None
    reading_pip_check = False
    for line_number, line in enumerate(prefix_text.splitlines(), start=1):
        if line.startswith("#"):
            continue
        if not line:
            if section == "packages" and reading_pip_check:
                reading_pip_check = False
            continue
        if line.startswith("["):
            if not line.endswith("]"):
                raise RuntimeError(f"malformed runtime lock section at line {line_number}")
            section = line[1:-1]
            reading_pip_check = False
            continue
        if section is None:
            if ":" not in line:
                raise RuntimeError(f"malformed runtime lock metadata at line {line_number}")
            key, value = line.split(":", 1)
            key = key.strip()
            if not key or key in metadata:
                raise RuntimeError(f"duplicate or empty runtime lock metadata key at line {line_number}")
            metadata[key] = value.strip()
        elif section == "packages" and line.startswith("pip_check:"):
            if pip_check_lines is not None:
                raise RuntimeError("runtime lock contains duplicate pip_check entries")
            inline = line.split(":", 1)[1].strip()
            pip_check_lines = [inline] if inline else []
            reading_pip_check = True
        elif section == "packages" and reading_pip_check:
            pip_check_lines.append(line)

    required = ("python_executable", "pip_freeze_all_lines", "pip_freeze_all_sha256")
    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise RuntimeError(f"runtime lock missing required metadata: {missing}")
    executable = Path(metadata["python_executable"])
    if not executable.is_absolute():
        raise RuntimeError("runtime lock python_executable must be absolute")
    try:
        expected_lines = int(metadata["pip_freeze_all_lines"])
    except ValueError as exc:
        raise RuntimeError("runtime lock pip_freeze_all_lines must be an integer") from exc
    expected_sha256 = metadata["pip_freeze_all_sha256"]
    if expected_lines <= 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("runtime lock freeze line count/hash metadata is invalid")
    actual_lines = len(freeze_bytes.splitlines())
    actual_sha256 = _sha256_bytes(freeze_bytes)
    if actual_lines != expected_lines or actual_sha256 != expected_sha256:
        raise RuntimeError(
            "runtime lock freeze payload does not match its declared line count/hash: "
            f"lines={actual_lines}/{expected_lines}, sha256={actual_sha256}/{expected_sha256}"
        )
    if pip_check_lines is None or not pip_check_lines:
        raise RuntimeError("runtime lock must contain a non-empty [packages] pip_check record")
    return {
        "path": str(path),
        "lock_sha256": _sha256_bytes(raw),
        "python_executable": str(executable),
        "pip_freeze_all_lines": expected_lines,
        "pip_freeze_all_sha256": expected_sha256,
        "pip_freeze_all_bytes": freeze_bytes,
        "pip_check_lines": pip_check_lines,
    }


def _validate_runtime_freeze(
    output: bytes,
    expected_lines: int,
    expected_sha256: str,
    expected_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate raw ``pip freeze --all`` bytes without normalizing them."""
    actual_lines = len(output.splitlines())
    actual_sha256 = _sha256_bytes(output)
    if actual_lines != int(expected_lines) or actual_sha256 != expected_sha256:
        raise RuntimeError(
            "pip freeze --all differs from runtime lock: "
            f"lines={actual_lines}/{expected_lines}, sha256={actual_sha256}/{expected_sha256}"
        )
    if expected_bytes is not None and output != expected_bytes:
        raise RuntimeError("pip freeze --all bytes differ from the locked payload")
    return {
        "expected_lines": int(expected_lines),
        "actual_lines": actual_lines,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "exact_bytes_match": expected_bytes is None or output == expected_bytes,
    }


def _verify_runtime_lock(path: Path) -> dict[str, Any]:
    """Fail closed unless the running interpreter and package set are exact."""
    parsed = _parse_runtime_lock(path)
    expected_executable = Path(parsed["python_executable"]).resolve()
    actual_executable = Path(sys.executable).resolve()
    if actual_executable != expected_executable:
        raise RuntimeError(
            "runtime python executable differs from lock: "
            f"actual={actual_executable}, expected={expected_executable}"
        )

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=False,
        check=False,
    )
    if freeze.returncode != 0:
        raise RuntimeError(f"pip freeze --all failed with return code {freeze.returncode}")
    try:
        freeze_stderr = freeze.stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("pip freeze --all stderr is not valid UTF-8") from exc
    freeze_evidence = _validate_runtime_freeze(
        freeze.stdout,
        parsed["pip_freeze_all_lines"],
        parsed["pip_freeze_all_sha256"],
        parsed["pip_freeze_all_bytes"],
    )

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=False,
        check=False,
    )
    try:
        pip_check_stdout = pip_check.stdout.decode("utf-8")
        pip_check_stderr = pip_check.stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("pip check output is not valid UTF-8") from exc
    actual_check_lines = pip_check_stdout.splitlines()
    if pip_check.returncode != 0 or actual_check_lines != parsed["pip_check_lines"]:
        raise RuntimeError(
            "pip check differs from the runtime lock: "
            f"returncode={pip_check.returncode}, output={actual_check_lines!r}, "
            f"expected={parsed['pip_check_lines']!r}"
        )
    return {
        "path": parsed["path"],
        "lock_sha256": parsed["lock_sha256"],
        "python_executable": {
            "expected": str(expected_executable),
            "actual": str(actual_executable),
        },
        "pip_freeze_all": freeze_evidence,
        "pip_check": {
            "args": [sys.executable, "-m", "pip", "check"],
            "returncode": pip_check.returncode,
            "stdout": pip_check_stdout,
            "stderr": pip_check_stderr,
            "expected_stdout_lines": parsed["pip_check_lines"],
        },
    }


def _run_command(args: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
        return {
            "args": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"args": args, "error": f"{type(exc).__name__}: {exc}"}


def _git_evidence(root: Path, run_dir: Path | None = None) -> dict[str, Any]:
    status = _run_command(["git", "status", "--short", "--untracked-files=all"], root)
    dirty: list[dict[str, Any]] = []
    run_relative = run_dir.relative_to(root) if run_dir is not None else None
    for line in str(status.get("stdout", "")).splitlines():
        relative = line[3:] if len(line) >= 4 else ""
        relative_path = Path(relative)
        if not relative or (
            run_relative is not None
            and (relative_path == run_relative or run_relative in relative_path.parents)
        ):
            continue
        path = root / relative
        item: dict[str, Any] = {"path": relative, "status": line[:2]}
        if path.is_file():
            item["sha256"] = _sha256(path)
            item["bytes"] = path.stat().st_size
        dirty.append(item)

    dependencies = {}
    for name in ("lerobot", "hf-libero", "robosuite", "mujoco"):
        checkout = root / "external" / name
        dependencies[name] = {
            "head": _run_command(["git", "rev-parse", "HEAD"], checkout).get("stdout"),
            "status": _run_command(["git", "status", "--short"], checkout),
        }
    return {
        "head": _run_command(["git", "rev-parse", "HEAD"], root).get("stdout"),
        "status": status,
        "dirty_files": dirty,
        "dependencies": dependencies,
    }


def _verify_source_checkouts(root: Path, approved_manifest: Path) -> dict[str, Any]:
    with approved_manifest.open() as handle:
        approved = json.load(handle)
    expected = dict(approved.get("project", {}).get("external_source_shas", {}))
    required = {
        "lerobot": "lerobot",
        "hf_libero": "hf-libero",
        "robosuite": "robosuite",
        "mujoco": "mujoco",
    }
    if set(expected) != set(required):
        raise RuntimeError(f"approved source SHA inventory changed: {sorted(expected)}")
    pins = _load_yaml(root / "external" / "pins.yaml")
    pin_entries = {
        str(item["id"]).replace("-", "_"): str(item["revision"])
        for item in pins.get("dependencies", [])
        if isinstance(item, Mapping) and "id" in item and "revision" in item
    }
    evidence: dict[str, Any] = {}
    for key, directory_name in required.items():
        checkout = root / "external" / directory_name
        expected_sha = str(expected[key])
        pin_sha = pin_entries.get(key)
        if pin_sha != expected_sha:
            raise RuntimeError(f"pins.yaml source SHA mismatch for {key}: {pin_sha} != {expected_sha}")
        head = _run_command(["git", "rev-parse", "HEAD"], checkout)
        branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], checkout)
        status = _run_command(["git", "status", "--short", "--untracked-files=all"], checkout)
        actual_sha = head.get("stdout")
        clean = status.get("returncode") == 0 and not str(status.get("stdout", "")).strip()
        if actual_sha != expected_sha or branch.get("stdout") != "HEAD" or not clean:
            raise RuntimeError(
                f"source checkout gate failed for {key}: "
                f"head={actual_sha}, expected={expected_sha}, branch={branch.get('stdout')}, clean={clean}"
            )
        evidence[key] = {
            "path": str(checkout.resolve()),
            "expected_sha": expected_sha,
            "actual_sha": actual_sha,
            "detached": branch.get("stdout") == "HEAD",
            "clean": clean,
            "status": status,
        }
    return evidence


def _verify_runtime_module_paths(
    root: Path, modules: Mapping[str, Any], purelib: Path | None = None
) -> dict[str, Any]:
    purelib = (purelib or Path(sysconfig.get_paths()["purelib"])).resolve()
    forbidden_roots = [
        (root / "external" / name).resolve()
        for name in ("lerobot", "hf-libero", "libero", "libero-plus", "robosuite", "mujoco")
    ]
    evidence: dict[str, Any] = {"purelib": str(purelib), "modules": {}}
    for name in ("lerobot", "libero", "robosuite", "mujoco"):
        module = modules.get(name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise RuntimeError(f"runtime module has no __file__: {name}")
        path = Path(module_file).resolve()
        if not path.is_relative_to(purelib):
            raise RuntimeError(f"runtime module escaped isolated site-packages: {name} -> {path}")
        if any(path.is_relative_to(forbidden) for forbidden in forbidden_roots):
            raise RuntimeError(f"runtime module resolved to an external source tree: {name} -> {path}")
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        evidence["modules"][name] = {
            "__file__": str(path),
            "__spec__.origin": str(Path(origin).resolve()) if origin and origin != "built-in" else origin,
        }
    return evidence


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return value


def _validate_config(config: dict[str, Any], root: Path) -> None:
    required = {
        "seed",
        "suite",
        "task_id",
        "init_state_id",
        "n_envs",
        "use_async_envs",
        "obs_type",
        "observation_height",
        "observation_width",
        "action_dim",
        "chunk_size",
        "n_action_steps",
        "checkpoint",
        "base_model",
        "assets",
        "preflight_manifest",
        "runtime_lock",
        "libero_config_path",
        "output_root",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    if config["n_envs"] != 1 or config["use_async_envs"] is not False:
        raise ValueError("M0 smoke requires one synchronous vector environment")
    if config["obs_type"] != "pixels_agent_pos":
        raise ValueError("M0 smoke requires official pixels_agent_pos observations")
    if tuple((config["observation_height"], config["observation_width"])) != (360, 360):
        raise ValueError("M0 smoke requires the official 360x360 LIBERO resolution")
    if config["action_dim"] != 7 or config["chunk_size"] != 50 or config["n_action_steps"] != 1:
        raise ValueError("M0 smoke action/chunk contract is 7D, chunk_size=50, n_action_steps=1")
    if config["init_state_id"] != 0:
        raise ValueError("M0 smoke requires init-state ID 0")
    if not isinstance(config["checkpoint"], Mapping) or not isinstance(config["base_model"], Mapping):
        raise ValueError("checkpoint and base_model must be mappings")
    for key in ("path", "revision", "repo_id"):
        if not config["checkpoint"].get(key) or not config["base_model"].get(key):
            raise ValueError(f"artifact entries require {key}")
    for key in ("path", "revision", "repo_id"):
        if not config["assets"].get(key):
            raise ValueError(f"assets entry requires {key}")
    for field in (
        "checkpoint",
        "base_model",
        "assets",
        "preflight_manifest",
        "runtime_lock",
        "libero_config_path",
    ):
        value = Path(config[field]["path"] if field in ("checkpoint", "base_model", "assets") else config[field])
        if not value.is_absolute():
            raise ValueError(f"{field} must be an absolute path")
    if Path(config["libero_config_path"]).joinpath("config.yaml").is_file() is False:
        raise FileNotFoundError(f"LIBERO config missing: {config['libero_config_path']}")
    if Path(config["preflight_manifest"]).is_file() is False:
        raise FileNotFoundError(f"preflight artifact manifest missing: {config['preflight_manifest']}")
    if Path(config["runtime_lock"]).is_file() is False:
        raise FileNotFoundError(f"runtime lock missing: {config['runtime_lock']}")
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = root / output_root
    config["_output_root"] = str(output_root.resolve())


def _assert_offline_and_hf_home() -> dict[str, Any]:
    for key, expected in OFFLINE_ENV.items():
        if os.environ.get(key) != expected:
            raise RuntimeError(f"{key} must be set to {expected} before runtime imports")
    hf_home_text = os.environ.get("HF_HOME")
    if not hf_home_text:
        raise RuntimeError("HF_HOME must point to a fresh empty directory")
    hf_home = Path(hf_home_text).resolve()
    if hf_home.exists() and any(hf_home.iterdir()):
        raise RuntimeError(f"HF_HOME must be fresh and empty: {hf_home}")
    hf_home.mkdir(parents=True, exist_ok=True)
    return {"variables": {key: os.environ.get(key) for key in (*OFFLINE_ENV, "HF_HOME")}, "fresh_empty": True}


def _ensure_asset_symlink(target: Path, link: Path) -> dict[str, Any]:
    target = target.resolve(strict=True)
    if not target.is_dir():
        raise RuntimeError(f"asset target is not a directory: {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        resolved = link.resolve(strict=False)
        if resolved != target:
            raise RuntimeError(f"asset symlink points outside the exact tree: {link} -> {resolved}")
        link.unlink()
        link.symlink_to(target, target_is_directory=True)
    elif link.exists():
        raise RuntimeError(f"asset binding is a non-symlink and will not be removed: {link}")
    else:
        link.symlink_to(target, target_is_directory=True)
    resolved = link.resolve(strict=True)
    if not link.is_symlink() or resolved != target:
        raise RuntimeError(f"asset symlink verification failed: {link} -> {resolved}")
    return {"link": str(link), "target": str(target), "resolved": str(resolved), "is_symlink": True}


def _safe_asset_symlink(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    del root
    configured = Path(config["assets"]["path"]).resolve(strict=True)
    purelib = Path(sysconfig.get_paths()["purelib"])
    link = purelib / "libero" / "libero" / "assets"
    return _ensure_asset_symlink(configured, link)


def _verify_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    with Path(config["preflight_manifest"]).open() as handle:
        approved = json.load(handle)
    by_revision = {str(item["revision"]): item for item in approved.get("artifacts", [])}
    specs = [config["assets"], config["checkpoint"], config["base_model"]]
    summary: list[dict[str, Any]] = []
    for spec in specs:
        revision = str(spec["revision"])
        item = by_revision.get(revision)
        if item is None:
            raise RuntimeError(f"artifact revision not in approved Preflight-B manifest: {revision}")
        root = Path(spec["path"]).resolve(strict=True)
        if root != Path(item["local_path"]).resolve():
            raise RuntimeError(f"artifact path mismatch for revision {revision}")
        if item.get("verification", {}).get("status") != "PASS":
            raise RuntimeError(f"approved artifact manifest is not PASS for {revision}")
        file_hashes = []
        expected_paths = {record["path"] for record in item["files"]}
        actual_paths = {
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and ".cache" not in path.relative_to(root).parts
        }
        if actual_paths != expected_paths:
            unknown = sorted(actual_paths - expected_paths)
            missing = sorted(expected_paths - actual_paths)
            raise RuntimeError(
                f"artifact payload inventory mismatch for {revision}: "
                f"unknown={unknown[:5]}, missing={missing[:5]}"
            )
        for record in item["files"]:
            path = root / record["path"]
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"artifact file is missing or symlinked: {path}")
            size = path.stat().st_size
            digest = _sha256(path)
            if size != int(record["bytes"]) or digest != record["sha256"]:
                raise RuntimeError(f"artifact hash/size mismatch: {path}")
            with path.open("rb") as handle:
                prefix = handle.read(64)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(f"Git-LFS pointer remained materialized: {path}")
            file_hashes.append({"path": record["path"], "bytes": size, "sha256": digest})
        summary.append(
            {
                "repo_id": item["repo_id"],
                "repo_type": item["repo_type"],
                "revision": revision,
                "path": str(root),
                "file_count": len(file_hashes),
                "total_bytes": sum(entry["bytes"] for entry in file_hashes),
                "files": file_hashes,
            }
        )
    return {"approved_manifest": str(Path(config["preflight_manifest"]).resolve()), "artifacts": summary}


def _runtime_evidence(chosen_device: str) -> dict[str, Any]:
    import gymnasium
    import mujoco
    import transformers

    cuda_available = bool(torch.cuda.is_available())
    device_name = None
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
    hipcc = shutil.which("hipcc")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": torch.version.hip,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "device_name": device_name,
        "chosen_device": chosen_device,
        "hipcc": _run_command([hipcc, "--version"]) if hipcc else None,
    }


def _assert_project_head(head: str | None) -> None:
    if head != EXPECTED_PROJECT_SHA:
        raise RuntimeError(f"project HEAD must be {EXPECTED_PROJECT_SHA}, got {head}")


def _assert_action_space(env: Any, action_dim: int) -> dict[str, Any]:
    action_space = getattr(env, "action_space", None)
    single_action_space = getattr(env, "single_action_space", None)
    if action_space is None or single_action_space is None:
        raise RuntimeError("vector environment must expose action_space and single_action_space")
    expected_batch_shape = (int(env.num_envs), action_dim)
    if tuple(action_space.shape) != expected_batch_shape or action_space.dtype != np.dtype(np.float32):
        raise RuntimeError(
            f"vector action_space must be {expected_batch_shape} float32, "
            f"got shape={action_space.shape}, dtype={action_space.dtype}"
        )
    if tuple(single_action_space.shape) != (action_dim,) or single_action_space.dtype != np.dtype(np.float32):
        raise RuntimeError(
            f"single_action_space must be ({action_dim},) float32, "
            f"got shape={single_action_space.shape}, dtype={single_action_space.dtype}"
        )
    return {
        "batch_action_shape": list(expected_batch_shape),
        "batch_action_dtype": "float32",
        "single_action_shape": list(single_action_space.shape),
        "single_action_dtype": str(single_action_space.dtype),
    }


def _assert_rollout_acceptance(
    trace: TraceStore,
    rollout_data: Mapping[str, Any],
    policy: PolicyProxy,
    render_count: int,
) -> None:
    rollout_length = int(rollout_data[ACTION_KEY].shape[1])
    if rollout_length < 1:
        raise RuntimeError("official rollout produced no steps")
    if policy.call_count != len(trace.env_step_latencies) or policy.call_count != len(trace.decisions):
        raise RuntimeError(
            "policy/env/decision call counts differ: "
            f"{policy.call_count}/{len(trace.env_step_latencies)}/{len(trace.decisions)}"
        )
    if policy.call_count != rollout_length:
        raise RuntimeError(f"rollout length differs from policy calls: {rollout_length} vs {policy.call_count}")
    for decision in trace.decisions:
        if decision.get("env_step_action", {}).get("shape") != [1, trace.action_dim]:
            raise RuntimeError(f"invalid action shape in decision {decision['decision_index']}")
        if decision.get("env_step_action", {}).get("finite") is not True:
            raise RuntimeError(f"non-finite action in decision {decision['decision_index']}")
        if decision.get("env_step_equals_postprocessed") is not True:
            raise RuntimeError(f"action equality failed in decision {decision['decision_index']}")
        if (
            decision.get("queue_length_before") != 0
            or decision.get("queue_length_after") != 0
            or decision.get("new_chunk_generated") is not True
        ):
            raise RuntimeError(f"queue semantics failed in decision {decision['decision_index']}")
    final_done = rollout_data["done"].detach().cpu().numpy()
    if not bool(final_done[0, -1]):
        raise RuntimeError("official rollout did not report final done")
    if render_count != rollout_length + 1:
        raise RuntimeError(f"render count must equal steps+1: {render_count} vs {rollout_length + 1}")


def _action_summary(trace: TraceStore) -> dict[str, Any]:
    actions = [row.get("env_step_action", {}) for row in trace.decisions]
    finite = [item.get("finite") is True for item in actions]
    shapes = [tuple(item.get("shape", [])) for item in actions]
    dtypes = [item.get("dtype") for item in actions]
    minima = [item["min"] for item in actions if "min" in item]
    maxima = [item["max"] for item in actions if "max" in item]
    return {
        "count": len(actions),
        "shape_counts": {str(shape): shapes.count(shape) for shape in sorted(set(shapes))},
        "dtype_counts": {str(dtype): dtypes.count(dtype) for dtype in sorted(set(dtypes))},
        "all_shape_1x7": all(shape == (1, trace.action_dim) for shape in shapes),
        "all_finite": all(finite),
        "all_equal_postprocessed": all(
            row.get("env_step_equals_postprocessed") is True for row in trace.decisions
        ),
        "min": min(minima) if minima else None,
        "max": max(maxima) if maxima else None,
    }


def _choose_device(requested: str | None) -> str:
    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested device unavailable: {requested}")
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _command_text() -> str:
    env_keys = (
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "LIBERO_CONFIG_PATH",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "PYTHONDONTWRITEBYTECODE",
        "MPLCONFIGDIR",
        "XDG_CACHE_HOME",
        "PYTHONNOUSERSITE",
        "PIP_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
    )
    prefix = " ".join(f"{key}={shlex.quote(os.environ[key])}" for key in env_keys if key in os.environ)
    command = shlex.join([sys.executable, *sys.argv])
    return f"{prefix} {command}".strip()


class _Tee:
    def __init__(self, stream: Any, path: Path) -> None:
        self.stream = stream
        self.file = path.open("a", buffering=1)

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


def _new_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}_{os.urandom(4).hex()}"
        candidate = output_root / run_id
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate unique run directory below {output_root}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _set_official_done(trace: TraceStore, rollout_data: Mapping[str, Any]) -> None:
    done = rollout_data["done"].detach().cpu().numpy()
    success = rollout_data["success"].detach().cpu().numpy()
    rewards = rollout_data["reward"].detach().cpu().numpy()
    for index, decision in enumerate(trace.decisions):
        decision["done"] = [bool(done[0, index])]
        decision["success"] = [bool(success[0, index])]
        decision["rollout_reward"] = [float(rewards[0, index])]
        if decision.get("env_done") == [True]:
            if decision.get("terminated") == [True]:
                decision["termination_reason"] = "environment_termination"
            elif decision.get("truncated") == [True]:
                decision["termination_reason"] = "environment_truncation"
        elif decision["done"] == [True]:
            decision["termination_reason"] = "official_horizon"
        else:
            decision["termination_reason"] = "not_done"


def _episode_result(trace: TraceStore, rollout_data: Mapping[str, Any], policy: PolicyProxy) -> dict[str, Any]:
    rewards = rollout_data["reward"].detach().cpu().numpy()
    success = rollout_data["success"].detach().cpu().numpy()
    done = rollout_data["done"].detach().cpu().numpy()
    return {
        "policy_call_count": policy.call_count,
        "env_step_call_count": len(trace.env_step_latencies),
        "success": bool(success.any()),
        "reward_sum": float(rewards.sum()),
        "rollout_length": int(rollout_data[ACTION_KEY].shape[1]),
        "termination_reason": trace.decisions[-1].get("termination_reason") if trace.decisions else "no_decision",
        "action_summary": _action_summary(trace),
        "official_done_final": bool(done[0, -1]) if done.size else False,
        "latency_seconds": {
            "inference": _latency_summary(trace.inference_latencies),
            "env_step": _latency_summary(trace.env_step_latencies),
            "render": _latency_summary(trace.render_latencies),
        },
        "render_call_count": trace.render_count,
        "queue_semantics": {
            "configured_chunk_size": trace.chunk_size,
            "configured_n_action_steps": trace.n_action_steps,
            "all_decisions_start_empty": all(
                row.get("queue_length_before") == 0 for row in trace.decisions
            ),
            "all_decisions_end_empty": all(
                row.get("queue_length_after") == 0 for row in trace.decisions
            ),
            "all_decisions_generated_chunk": all(
                row.get("new_chunk_generated") is True for row in trace.decisions
            ),
        },
    }


def _manifest_base(
    root: Path, config_path: Path, run_dir: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "RUNNING",
        "project": {
            "root": str(root),
            "expected_sha": EXPECTED_PROJECT_SHA,
            "git": _git_evidence(root, run_dir),
        },
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "run_dir": str(run_dir.resolve()),
        "runtime_lock": {
            "path": str(Path(config["runtime_lock"]).resolve()),
            "lock_sha256": _sha256(Path(config["runtime_lock"]).resolve()),
        },
        "command": _command_text(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "HF_HUB_DISABLE_TELEMETRY",
                "LIBERO_CONFIG_PATH",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "MUJOCO_EGL_DEVICE_ID",
                "PYTHONDONTWRITEBYTECODE",
                "MPLCONFIGDIR",
                "XDG_CACHE_HOME",
                "PYTHONNOUSERSITE",
                "PIP_CACHE_DIR",
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
            )
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "no_network": True,
        "frozen_policy": True,
        "training": False,
        "optimizer_created": False,
        "backward_called": False,
        "lora": False,
        "world_model": False,
        "sae": False,
        "perturbation": False,
        "matched_sampling": False,
        "activation_patching": False,
        "flow_noise": {
            "source": "ordinary global PyTorch RNG",
            "paired": False,
        },
    }


def run(config_path: Path) -> int:
    root = Path(__file__).resolve().parents[1]
    config = _load_yaml(config_path)
    _validate_config(config, root)
    run_dir = _new_run_dir(Path(config["_output_root"]))
    sys.stdout = _Tee(sys.stdout, run_dir / "stdout.log")
    sys.stderr = _Tee(sys.stderr, run_dir / "stderr.log")
    manifest = _manifest_base(root, config_path, run_dir, config)
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "config_resolved.json", config)
    (run_dir / "command.txt").write_text(manifest["command"] + "\n")
    trace = TraceStore(config["chunk_size"], config["n_action_steps"], config["action_dim"])
    envs = None
    started = time.perf_counter()

    try:
        _assert_project_head(manifest["project"]["git"]["head"])
        manifest["offline"] = _assert_offline_and_hf_home()
        manifest["runtime_lock"] = _verify_runtime_lock(Path(config["runtime_lock"]).resolve())
        manifest["renderer_environment"] = _assert_renderer_environment()
        libero_config_dir = Path(config["libero_config_path"]).resolve()
        libero_config_file = libero_config_dir / "config.yaml"
        if os.environ.get("LIBERO_CONFIG_PATH") != str(libero_config_dir):
            raise RuntimeError("LIBERO_CONFIG_PATH must equal the configured pinned LIBERO config")
        manifest["libero_config"] = {
            "directory": str(libero_config_dir),
            "config_file": str(libero_config_file),
            "config_file_sha256": _sha256(libero_config_file),
            "config_file_bytes": libero_config_file.stat().st_size,
        }
        asset_link = _safe_asset_symlink(root, config)
        manifest["asset_symlink"] = asset_link
        manifest["artifacts"] = _verify_artifacts(config)
        manifest["source_checkouts"] = _verify_source_checkouts(
            root, Path(config["preflight_manifest"]).resolve()
        )

        chosen_device = _choose_device(config.get("device"))
        manifest["runtime"] = _runtime_evidence(chosen_device)

        import lerobot
        import libero
        import mujoco
        import robosuite
        from libero.libero import benchmark, get_assets_path, get_libero_path
        from lerobot.envs import (
            close_envs,
            make_env,
            make_env_config,
            make_env_pre_post_processors,
        )
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.scripts.lerobot_eval import rollout

        manifest["runtime_imports"] = _verify_runtime_module_paths(
            root,
            {
                "lerobot": lerobot,
                "libero": libero,
                "robosuite": robosuite,
                "mujoco": mujoco,
            },
        )

        asset_path = Path(get_assets_path()).resolve(strict=True)
        if asset_path != Path(config["assets"]["path"]).resolve():
            raise RuntimeError(f"get_assets_path() escaped the exact artifact tree: {asset_path}")
        suite = benchmark.get_benchmark_dict()[config["suite"]]()
        task = suite.get_task(int(config["task_id"]))
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_state = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        expected_bddl_root = root / "external" / "hf-libero" / "libero" / "libero" / "bddl_files"
        expected_init_root = root / "external" / "hf-libero" / "libero" / "libero" / "init_files"
        if not bddl.is_file() or not init_state.is_file():
            raise RuntimeError("pinned LIBERO BDDL/init state is missing")
        if not bddl.resolve().is_relative_to(expected_bddl_root.resolve()):
            raise RuntimeError(f"BDDL path escaped pinned hf-libero tree: {bddl}")
        if not init_state.resolve().is_relative_to(expected_init_root.resolve()):
            raise RuntimeError(f"init-state path escaped pinned hf-libero tree: {init_state}")

        env_cfg = make_env_config(
            "libero",
            task=config["suite"],
            task_ids=[int(config["task_id"])],
            obs_type=config["obs_type"],
            render_mode="rgb_array",
            init_states=True,
            hard_reset=True,
            observation_height=int(config["observation_height"]),
            observation_width=int(config["observation_width"]),
        )
        envs = make_env(
            env_cfg,
            n_envs=int(config["n_envs"]),
            use_async_envs=False,
            trust_remote_code=False,
        )
        env = envs[config["suite"]][int(config["task_id"])]
        if int(env.num_envs) != 1:
            raise RuntimeError(f"unexpected env batch size: {env.num_envs}")
        manifest["action_space"] = _assert_action_space(env, int(config["action_dim"]))
        max_steps = int(env.call("_max_episode_steps")[0])
        if max_steps != 280:
            raise RuntimeError(f"official libero_spatial horizon changed: {max_steps}")
        init_ids = env.get_attr("init_state_id")
        if int(np.asarray(init_ids).reshape(-1)[0]) != int(config["init_state_id"]):
            raise RuntimeError(f"unexpected initial state ID: {init_ids}")

        policy_config = SmolVLAConfig.from_pretrained(
            str(Path(config["checkpoint"]["path"]).resolve()), local_files_only=True
        )
        policy_config.device = chosen_device
        policy_config.vlm_model_name = str(Path(config["base_model"]["path"]).resolve())
        policy_config.pretrained_path = Path(config["checkpoint"]["path"]).resolve()
        policy_config.pretrained_revision = str(config["checkpoint"]["revision"])
        policy_config.use_peft = False
        if policy_config.chunk_size != config["chunk_size"] or policy_config.n_action_steps != config["n_action_steps"]:
            raise RuntimeError("checkpoint queue configuration does not match frozen smoke config")
        policy = make_policy(cfg=policy_config, env_cfg=env_cfg, rename_map=None)
        # make_policy delegates to from_pretrained with strict=False in v0.6.1.
        # Re-load the same exact local tensor file through the official class gate.
        SmolVLAPolicy._load_as_safetensor(
            policy,
            str(Path(config["checkpoint"]["path"]) / "model.safetensors"),
            chosen_device,
            strict=True,
        )
        policy.eval()
        model_device = str(next(policy.parameters()).device)
        manifest["model"] = {
            "policy_type": type(policy).__name__,
            "training": bool(policy.training),
            "chosen_device": chosen_device,
            "model_device": model_device,
            "strict_checkpoint_load": True,
            "checkpoint_path": str(Path(config["checkpoint"]["path"]).resolve()),
            "base_model_path": str(Path(config["base_model"]["path"]).resolve()),
            "chunk_size": int(policy_config.chunk_size),
            "n_action_steps": int(policy_config.n_action_steps),
            "use_peft": bool(policy_config.use_peft),
            "parameters": sum(parameter.numel() for parameter in policy.parameters()),
        }

        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=policy_config
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(Path(config["checkpoint"]["path"]).resolve()),
            pretrained_revision=str(config["checkpoint"]["revision"]),
            preprocessor_overrides={
                "tokenizer_processor": {
                    "tokenizer_name": str(Path(config["base_model"]["path"]).resolve())
                },
                "device_processor": {"device": chosen_device},
            },
            postprocessor_overrides={"device_processor": {"device": chosen_device}},
        )
        manifest["processors"] = {
            "env_preprocessor_steps": [type(step).__name__ for step in env_preprocessor.steps],
            "env_postprocessor_steps": [type(step).__name__ for step in env_postprocessor.steps],
            "policy_preprocessor_steps": [type(step).__name__ for step in preprocessor.steps],
            "policy_postprocessor_steps": [type(step).__name__ for step in postprocessor.steps],
            "tokenizer_override": str(Path(config["base_model"]["path"]).resolve()),
        }

        policy_proxy = PolicyProxy(policy, trace).eval()
        env_proxy = VectorEnvProxy(env, trace)
        render_callback = make_render_callback(trace)
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        random.seed(int(config["seed"]))
        manifest["task"] = {
            "suite": config["suite"],
            "task_id": int(config["task_id"]),
            "init_state_id": int(config["init_state_id"]),
            "seed": int(config["seed"]),
            "task_name": task.name,
            "task_description": task.language,
            "bddl": str(bddl.resolve()),
            "init_state": str(init_state.resolve()),
            "horizon": max_steps,
            "n_envs": int(config["n_envs"]),
            "sync_vector_env": True,
        }
        manifest["flow_noise_seed"] = int(config["seed"])
        manifest["episode_attempt_started_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["episode_attempt_count"] = 1
        rollout_data = rollout(
            env=env_proxy,
            policy=policy_proxy,
            env_preprocessor=ProcessorProxy(env_preprocessor, "env_processor", trace),
            env_postprocessor=ProcessorProxy(env_postprocessor, "env_postprocessor", trace),
            preprocessor=ProcessorProxy(preprocessor, "policy_processor", trace),
            postprocessor=ProcessorProxy(postprocessor, "postprocessor", trace),
            seeds=[int(config["seed"])],
            return_observations=False,
            render_callback=render_callback,
        )
        _set_official_done(trace, rollout_data)
        _assert_rollout_acceptance(trace, rollout_data, policy_proxy, trace.render_count)
        manifest["egl"] = trace.gl
        manifest["episode_result"] = _episode_result(trace, rollout_data, policy_proxy)
        manifest["status"] = "PASS"
        manifest["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["duration_seconds"] = time.perf_counter() - started
        _write_json(run_dir / "episode_result.json", manifest["episode_result"])
        _write_jsonl(run_dir / "decisions.jsonl", trace.decisions)
        return 0
    except Exception as exc:
        manifest["status"] = "FAIL"
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        manifest["completed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["duration_seconds"] = time.perf_counter() - started
        if trace.gl:
            manifest["egl"] = trace.gl
        _write_json(run_dir / "episode_result.json", {"status": "FAIL", "failure_reason": manifest["failure_reason"]})
        _write_jsonl(run_dir / "decisions.jsonl", trace.decisions)
        print(manifest["traceback"], file=sys.stderr)
        return 1
    finally:
        if envs is not None:
            try:
                from lerobot.envs import close_envs

                close_envs(envs)
            except Exception as exc:
                manifest.setdefault("cleanup_errors", []).append(f"{type(exc).__name__}: {exc}")
        _write_json(run_dir / "run_manifest.json", manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
