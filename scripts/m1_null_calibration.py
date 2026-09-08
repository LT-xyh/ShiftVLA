#!/usr/bin/env python3
"""Fresh-process, policy-free M1-N0 null calibration.

The module owns only the duplicate-control protocol.  Source registry and
envelope primitives are loaded lazily from the existing hard-gate modules so
importing this file cannot load LIBERO, MuJoCo, LeRobot, or a policy stack.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np


if __package__ in {None, ""}:
    # A direct ``python scripts/m1_null_calibration.py`` invocation has only
    # ``scripts/`` on sys.path.  Keep the repository package importable while
    # retaining lazy environment imports below.
    _ROOT_FOR_IMPORT = str(Path(__file__).resolve().parents[1])
    if _ROOT_FOR_IMPORT not in sys.path:
        sys.path.insert(0, _ROOT_FOR_IMPORT)


SCHEMA_VERSION = 1
PAIR_COUNT = 20
PAIR_PREFIX = "m1n0-pair-"
ACTION_SHAPE = (82, 7)
ACTION_DTYPE = np.dtype("float32")
TASK = {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027}
TARGET_OBJECT_NAME = "akita_black_bowl_1"
REGIME_NAMES: tuple[str, ...] = (
    "free_motion",
    "pre_contact",
    "contact",
    "grasp",
    "carried",
    "release",
    "predicate_transition",
)
DEFAULT_QUANTITY_ROOTS = ("qpos", "qvel", "objects", "gripper_physical")
DEFAULT_CAMERA = "agentview"
DEFAULT_RENDER_KEY = "render_rgb"
DEFAULT_WORKER_TIMEOUT_SECONDS = 600.0
# These are the names emitted by the pinned LeRobot LIBERO adapter after its
# official camera-name mapping.  The path is retained in evidence, while the
# camera group is resolved from the leaf rather than guessed from a requested
# output label.
OFFICIAL_RGB_LEAF_CAMERAS = {
    "image": "agentview",
    "image2": "robot0_eye_in_hand",
    "agentview_image": "agentview",
    "robot0_eye_in_hand_image": "robot0_eye_in_hand",
    "agentview_rgb": "agentview",
    "robot0_eye_in_hand_rgb": "robot0_eye_in_hand",
}
EXPECTED_PAIR_IDS = tuple(f"{PAIR_PREFIX}{index:03d}" for index in range(PAIR_COUNT))
STRICT_QUANTITY_ROOTS = (
    "integration",
    "qpos",
    "qvel",
    "controller",
    "eef_pose",
    "gripper",
    "gripper_physical",
    "objects",
    "contacts",
)
STRICT_INVARIANT_ROOTS = (
    "integration",
    "qpos",
    "qvel",
    "controller",
    "eef_pose",
    "gripper",
    "gripper_physical",
    "objects",
    "body_xpos",
    "body_xquat",
    "contacts",
    "predicates",
    "success",
    "counters",
    "observables",
    "observable_cache",
)
TERMINAL_FIELDS = ("step", "termination_reason", "success", "terminated", "truncated")
DEFAULT_TERMINAL_CONTRACT = {
    "step": ACTION_SHAPE[0],
    "termination_reason": "predicate_transition",
    "success": True,
    "terminated": True,
    "truncated": False,
}
FORBIDDEN_PROTOCOL_FIELDS = (
    "reset_count",
    "restore_count",
    "capture_count",
    "policy_calls",
    "processor_calls",
    "processors_calls",
    "retry_count",
    "post_terminal_steps",
    "set_init_state_count",
    "settle_count",
    "dummy_action_count",
    "autoreset_count",
)
POST_CONSTRUCTION_FORBIDDEN_OPERATIONS = (
    "reset",
    "set_init_state",
    "settle",
    "restore",
    "capture",
    "policy_calls",
    "processor_calls",
    "retry",
    "post_terminal_step",
    "dummy_action",
    "autoreset",
)


class CalibrationError(RuntimeError):
    """Base class for fail-closed calibration errors."""


class ProvenanceError(CalibrationError):
    """Frozen config, registry, or tape provenance is invalid."""


class ProtocolError(CalibrationError):
    """An attempt or pair violated the frozen null protocol."""


class PublicationError(CalibrationError):
    """An immutable artifact would be overwritten or is unverifiable."""


class NullCalibrationError(CalibrationError):
    """A complete null envelope cannot be derived."""


def _json_safe(value: Any) -> Any:
    """Encode finite values without pickle or inline array payloads.

    Arrays are represented by their immutable content digest for canonical
    payload hashing.  Published JSON documents call
    :func:`_materialize_array_artifacts` first, so the on-disk representation
    is an explicit content-addressed sidecar reference rather than a large
    nested JSON list.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("object arrays are not publication values")
        array = np.ascontiguousarray(value)
        if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
            raise ValueError("publication arrays must be finite")
        return {
            "__ndarray_digest__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": sha256_bytes(array.tobytes(order="C")),
        }
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("publication mapping keys must be strings")
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("publication values must be finite")
        return value
    raise TypeError(f"unsupported publication value: {type(value).__name__}")


def _json_restore(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        if value.get("__ndarray_ref__") is True:
            path_value = value.get("path")
            expected_sha = value.get("sha256")
            if not isinstance(path_value, (str, Path)) or not isinstance(expected_sha, str):
                raise ProvenanceError("invalid NumPy sidecar reference")
            target = Path(path_value)
            try:
                verify_artifact_hash(target, expected_sha)
                array = np.load(target, allow_pickle=False)
            except Exception as exc:
                raise ProvenanceError(f"could not load NumPy sidecar {target}: {exc}") from exc
            array = np.ascontiguousarray(np.asarray(array)).copy()
            try:
                expected_dtype = np.dtype(str(value["dtype"]))
                expected_shape = tuple(int(item) for item in value["shape"])
            except Exception as exc:
                raise ProvenanceError(f"invalid NumPy sidecar metadata: {target}") from exc
            if array.dtype != expected_dtype or array.shape != expected_shape:
                raise ProvenanceError(f"NumPy sidecar metadata mismatch: {target}")
            if array.dtype.kind == "O":
                raise ProvenanceError(f"NumPy sidecar contains an object array: {target}")
            array.setflags(write=False)
            return array
        if value.get("__ndarray__") is True:
            try:
                array = np.asarray(value["data"], dtype=np.dtype(str(value["dtype"])))
                if list(array.shape) != list(value["shape"]):
                    array = array.reshape(tuple(int(item) for item in value["shape"]))
                array.setflags(write=False)
                return array
            except Exception as exc:
                raise ProvenanceError(f"invalid serialized array: {exc}") from exc
        return {str(key): _json_restore(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_restore(child) for child in value]
    return value


def canonical_json(value: Any) -> str:
    """Canonical finite JSON used by every self-hash and artifact."""

    try:
        return json.dumps(
            _json_safe(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ProvenanceError(f"expected a regular file for SHA-256: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _without_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    return result


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = _without_hash(value, "output_sha256")
    # Parent-side transport metadata is added after a worker publishes its
    # result file.  It describes framing/logs, not the attempt payload.
    payload.pop("worker_transport", None)
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def config_contract_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(_without_hash(value, "config_sha256")).encode("utf-8"))


def verify_self_hash(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        return False
    try:
        return declared.lower() == sha256_bytes(
            canonical_json(_without_hash(value, field)).encode("utf-8")
        ).lower()
    except CalibrationError:
        return False


def verify_artifact_hash(path: str | Path, expected_sha256: str) -> bool:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise PublicationError("expected artifact SHA-256 is invalid")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise PublicationError(f"artifact hash mismatch for {path}: {actual} != {expected_sha256}")
    return True


def _exclusive_bytes(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    if target.is_symlink() or target.exists():
        raise PublicationError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PublicationError(f"refusing to overwrite existing artifact: {target}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _write_array_sidecar(array: np.ndarray, artifact_root: Path) -> dict[str, Any]:
    """Write one deterministic, no-pickle NumPy sidecar exactly once."""

    value = np.ascontiguousarray(array)
    if value.dtype.kind == "O":
        raise PublicationError("object arrays cannot be published")
    if value.dtype.kind in {"f", "c"} and not np.all(np.isfinite(value)):
        raise PublicationError("non-finite arrays cannot be published")
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    payload = stream.getvalue()
    digest = sha256_bytes(payload)
    artifact_root = artifact_root.resolve()
    target = artifact_root / f"{digest}.npy"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise PublicationError(f"NumPy sidecar path is not a regular file: {target}")
        verify_artifact_hash(target, digest)
    else:
        try:
            _exclusive_bytes(target, payload)
        except PublicationError:
            # Two independent publishers may race on the same content address.
            # Reuse only if the winner published the exact bytes; never
            # overwrite or accept a colliding/tampered sidecar.
            if target.is_symlink() or not target.is_file():
                raise
        verify_artifact_hash(target, digest)
    return {
        "path": str(target),
        "sha256": digest,
        "size": len(payload),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "format": "npy",
        "allow_pickle": False,
        "verified": True,
    }


def _materialize_array_artifacts(
    value: Any, artifact_root: str | Path
) -> tuple[Any, list[dict[str, Any]]]:
    """Replace every ndarray with a reconstructible content-addressed ref.

    The returned object contains only JSON-safe values and sidecar references;
    the input object is never mutated.  Equal arrays in one publication share
    one sidecar record.
    """

    root = Path(artifact_root)
    records: dict[str, dict[str, Any]] = {}

    def visit(node: Any) -> Any:
        if isinstance(node, np.ndarray):
            record = _write_array_sidecar(node, root)
            records.setdefault(record["sha256"], record)
            return {
                "__ndarray_ref__": True,
                "path": record["path"],
                "sha256": record["sha256"],
                "dtype": record["dtype"],
                "shape": record["shape"],
                "format": record["format"],
                "allow_pickle": False,
            }
        if isinstance(node, Mapping):
            if any(type(key) is not str for key in node):
                raise PublicationError("publication mapping keys must be strings")
            return {key: visit(child) for key, child in node.items()}
        if isinstance(node, tuple):
            return [visit(child) for child in node]
        if isinstance(node, list):
            return [visit(child) for child in node]
        return node

    return visit(value), list(records.values())


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Write canonical JSON exactly once and return its content hash."""

    materialized, array_artifacts = _materialize_array_artifacts(
        value, Path(path).parent / "arrays"
    )
    payload = (canonical_json(materialized) + "\n").encode("utf-8")
    target = _exclusive_bytes(path, payload)
    digest = sha256_bytes(payload)
    if sha256_file(target) != digest:
        raise PublicationError(f"published artifact hash could not be verified: {target}")
    return {
        "path": str(target),
        "sha256": digest,
        "size": len(payload),
        "array_artifacts": array_artifacts,
    }


def _load_document(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ProvenanceError(f"document is not a regular file: {target}")
    text = target.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]

            parsed = yaml.safe_load(text)
        except ModuleNotFoundError as exc:
            raise ProvenanceError("PyYAML is required for YAML config files") from exc
        except Exception as exc:
            raise ProvenanceError(f"could not parse YAML document {target}: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ProvenanceError(f"document must contain a mapping: {target}")
    restored = _json_restore(parsed)
    if not isinstance(restored, Mapping):
        raise ProvenanceError(f"document did not restore to a mapping: {target}")
    return dict(restored)


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_document(path)


_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, *, base: Path = _ROOT) -> Path:
    target = Path(value)
    return target if target.is_absolute() else base / target


def _official_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the complete frozen state-replay construction configuration."""

    if not bool(config.get("strict_runtime_contract")):
        return dict(config)
    obs_type = config.get("obs_type")
    if not isinstance(obs_type, str) or not obs_type.strip():
        raise ProvenanceError("strict null config requires obs_type")
    reference = config.get("state_replay_config")
    if not isinstance(reference, Mapping):
        raise ProvenanceError("strict null config requires state_replay_config path and SHA")
    path_value, expected_sha = reference.get("path"), reference.get("sha256")
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        raise ProvenanceError("state_replay_config.path is required")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ProvenanceError("state_replay_config.sha256 is required")
    state_path = _resolve_path(path_value)
    state_config = _load_document(state_path)
    try:
        from scripts import m1_state_replay

        # The state-replay contract hashes the canonical parsed config with
        # its self-reference removed from ``paths``; its validator is the
        # authority for the complete pinned runtime construction fields.
        paths = state_config.get("paths")
        actual_sha = paths.get("config_sha256") if isinstance(paths, Mapping) else None
        if str(actual_sha).lower() != expected_sha.lower():
            raise ProvenanceError("state_replay_config SHA drift")
        m1_state_replay.validate_config(state_config)
    except ProvenanceError:
        raise
    except Exception as exc:
        raise ProvenanceError(f"state replay runtime config is not valid: {exc}") from exc
    merged = copy.deepcopy(state_config)
    # Null calibration owns the same task identity but never the replay tape
    # or policy settings.  Preserve the full validated runtime sections.
    merged["task"] = copy.deepcopy(config.get("task", state_config.get("task", TASK)))
    merged["obs_type"] = obs_type
    merged["strict_provenance"] = True
    merged["authoritative"] = True
    return merged


def _strict_task(value: Any, *, field_name: str = "task") -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != TASK:
        raise ProvenanceError(f"{field_name} must equal the frozen M1 task identity {TASK!r}")
    return dict(TASK)


def _terminal_contract_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("terminal_contract", DEFAULT_TERMINAL_CONTRACT)
    if not isinstance(raw, Mapping):
        raise ProvenanceError("terminal_contract must be a mapping")
    contract: dict[str, Any] = {}
    for field_name in TERMINAL_FIELDS:
        if field_name not in raw:
            raise ProvenanceError(f"terminal_contract.{field_name} is required")
        contract[field_name] = copy.deepcopy(raw[field_name])
    if isinstance(contract["step"], bool) or int(contract["step"]) != ACTION_SHAPE[0]:
        raise ProvenanceError("terminal_contract.step must be exactly 82")
    if not isinstance(contract["termination_reason"], str) or not contract["termination_reason"].strip():
        raise ProvenanceError("terminal_contract.termination_reason is required")
    for field_name in ("success", "terminated", "truncated"):
        if not isinstance(contract[field_name], bool):
            raise ProvenanceError(f"terminal_contract.{field_name} must be boolean")
    return contract


def _registry_terminal_contract(
    registry: Mapping[str, Any], trace: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve terminal semantics from source coverage and the frozen config.

    The hard-gate registry records legal step/reason at the release and
    predicate-transition windows.  The null config freezes the complete
    returned-terminal tuple, including the three boolean fields that older
    registries did not repeat in their source-coverage rows.
    """

    contract = _terminal_contract_from_config(config)
    observed: list[Mapping[str, Any]] = []
    for owner in (trace, registry):
        terminal = owner.get("terminal") if isinstance(owner, Mapping) else None
        if isinstance(terminal, Mapping):
            observed.append(terminal)
    windows = trace.get("windows") if isinstance(trace, Mapping) else None
    if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
        for window in windows:
            if not isinstance(window, Mapping):
                continue
            coverage = window.get("source_coverage")
            if isinstance(coverage, Mapping):
                observed.append(coverage)
    for item in observed:
        if "legal_terminal_step" in item:
            try:
                legal_step = int(item["legal_terminal_step"])
            except (TypeError, ValueError) as exc:
                raise ProvenanceError("registry legal terminal step is not an integer") from exc
            if legal_step != contract["step"]:
                raise ProvenanceError("registry legal terminal step differs from terminal_contract")
        reason = item.get("terminal_reason", item.get("termination_reason"))
        if reason is not None and str(reason) != contract["termination_reason"]:
            raise ProvenanceError("registry terminal reason differs from terminal_contract")
        for field_name in ("success", "terminated", "truncated"):
            if field_name in item and item[field_name] != contract[field_name]:
                raise ProvenanceError(f"registry terminal {field_name} differs from terminal_contract")
    if not any(
        "legal_terminal_step" in item or "terminal_reason" in item or "termination_reason" in item
        for item in observed
    ):
        raise ProvenanceError("registry has no frozen legal terminal step/reason")
    if not any(
        isinstance(item.get("terminal_evidence", item.get("evidence")), Mapping)
        and bool(item.get("terminal_evidence", item.get("evidence")))
        for item in observed
    ):
        raise ProvenanceError("registry has no frozen official terminal evidence")
    return contract


def _validate_strict_null_config(config: Mapping[str, Any], runtime_config: Mapping[str, Any]) -> None:
    """Validate the null-specific fields before any official construction."""

    if not bool(config.get("strict_runtime_contract")):
        return
    _strict_task(config.get("task"), field_name="task")
    if config.get("obs_type") != runtime_config.get("obs_type"):
        raise ProvenanceError("null config obs_type differs from state-replay construction config")
    paths = runtime_config.get("paths")
    if not isinstance(paths, Mapping):
        raise ProvenanceError("state-replay construction config has no paths")
    configured_python = config.get("python")
    pinned_python = paths.get("cpu_python")
    if not isinstance(configured_python, str) or not configured_python.strip():
        raise ProvenanceError("strict null config requires pinned python")
    if str(Path(configured_python).resolve()) != str(Path(str(pinned_python)).resolve()):
        raise ProvenanceError("null config python differs from frozen state-replay python")
    renderer = config.get("renderer")
    if not isinstance(renderer, Mapping):
        raise ProvenanceError("strict null config requires renderer camera and observation_key")
    if not isinstance(renderer.get("camera"), str) or not renderer["camera"].strip():
        raise ProvenanceError("strict null config renderer.camera is required")
    if renderer["camera"] != DEFAULT_CAMERA:
        raise ProvenanceError(
            f"strict null config renderer.camera is frozen to {DEFAULT_CAMERA!r}"
        )
    if not isinstance(renderer.get("observation_key"), str) or not renderer["observation_key"].strip():
        raise ProvenanceError("strict null config renderer.observation_key is required")
    timeout = config.get("worker_timeout_seconds")
    if isinstance(timeout, bool):
        raise ProvenanceError("strict null config worker_timeout_seconds must be positive")
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProvenanceError("strict null config worker_timeout_seconds must be positive") from exc
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ProvenanceError("strict null config worker_timeout_seconds must be positive")
    configured_runtime = config.get("runtime")
    frozen_runtime = runtime_config.get("runtime")
    if not isinstance(configured_runtime, Mapping) or not isinstance(frozen_runtime, Mapping):
        raise ProvenanceError("strict null config requires the complete runtime contract")
    for field_name, expected in (
        ("offline", True),
        ("local_only", True),
        ("include_policy", False),
        ("call_policy", False),
        ("call_processors", False),
        ("fresh_processes", True),
        ("retry_count", 0),
    ):
        if configured_runtime.get(field_name) != expected:
            raise ProvenanceError(f"runtime.{field_name} must equal frozen value {expected!r}")
    frozen_renderer = frozen_runtime.get("renderer")
    configured_renderer = configured_runtime.get("renderer")
    if not isinstance(frozen_renderer, Mapping) or not isinstance(configured_renderer, Mapping):
        raise ProvenanceError("strict runtime renderer contract is missing")
    for field_name in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "expected_gl"):
        if field_name not in configured_renderer:
            raise ProvenanceError(f"runtime.renderer.{field_name} is required")
        if not _exact_equal(configured_renderer[field_name], frozen_renderer.get(field_name)):
            raise ProvenanceError(f"runtime.renderer.{field_name} differs from frozen runtime")
    quantity = config.get("quantity_selection")
    if not isinstance(quantity, Mapping):
        raise ProvenanceError("strict null config requires quantity_selection")
    roots = quantity.get("root_patterns")
    if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)):
        raise ProvenanceError("strict null quantity root_patterns are required")
    normalized_roots = tuple(str(root) for root in roots)
    if normalized_roots != STRICT_QUANTITY_ROOTS:
        raise ProvenanceError(
            "strict null quantity roots must be the complete frozen set "
            f"{list(STRICT_QUANTITY_ROOTS)!r}"
        )
    invariants = config.get("invariants")
    required_roots = invariants.get("required_roots") if isinstance(invariants, Mapping) else None
    if tuple(str(root) for root in required_roots or ()) != STRICT_INVARIANT_ROOTS:
        raise ProvenanceError("strict null invariant required_roots are incomplete or reordered")
    _terminal_contract_from_config(config)


def _load_verified_registry(path: str | Path) -> dict[str, Any]:
    """Load the existing strict registry only when preparation/run is asked."""

    from scripts import m1_hard_gate

    return dict(m1_hard_gate.load_frozen_registry(path))


def _verify_registry_payload(payload: Mapping[str, Any], path: Path, expected: str | None = None) -> None:
    if not verify_self_hash(payload, "registry_sha256"):
        raise ProvenanceError(f"registry self-hash is invalid: {path}")
    actual = str(payload["registry_sha256"]).lower()
    if expected is not None and actual != str(expected).lower():
        raise ProvenanceError(f"registry hash drift: {actual} != {expected}")
    if tuple(payload.get("regimes", ())) != REGIME_NAMES:
        raise ProvenanceError("registry regimes do not match the frozen seven-regime contract")
    traces = payload.get("traces")
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        raise ProvenanceError("registry has no frozen traces")


def _validate_trace_frozen_inputs(trace: Mapping[str, Any], tape_sha: str) -> None:
    expected = {
        "suite": TASK["suite"],
        "task_id": TASK["task_id"],
        "init_state_id": TASK["init_state_id"],
        "seed": TASK["seed"],
        "action_shape": list(ACTION_SHAPE),
        "action_dtype": "float32",
        "action_sha256": tape_sha,
    }
    for key, expected_value in expected.items():
        actual = trace.get(key)
        if key == "action_sha256" and str(actual).lower() == str(expected_value).lower():
            continue
        if not _exact_equal(actual, expected_value):
            raise ProvenanceError(f"registry trace {key} differs from frozen task/action input")


def _validate_attempt_schedule_record(
    record: Mapping[str, Any], *, pair_id: str, side: str, ordinal: int
) -> None:
    """Validate one immutable A/B schedule record, including its exact ID."""

    expected = {
        "attempt_id": f"{pair_id}-{side}",
        "pair_id": pair_id,
        "side": side,
        "ordinal": ordinal,
        "status": "scheduled",
    }
    for name, value in expected.items():
        if name not in record or not _exact_equal(record[name], value):
            raise ProvenanceError(
                f"frozen schedule record for {pair_id}/{side} has invalid {name}: "
                f"{record.get(name)!r} != {value!r}"
            )


def _validate_frozen_schedule(
    run_spec: Mapping[str, Any],
    pair_registry: Mapping[str, Any],
    *,
    source_registry: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed unless the persisted run and pair registries are exact."""

    if not verify_self_hash(run_spec, "run_spec_sha256"):
        raise ProvenanceError("run specification self-hash is invalid")
    if not verify_self_hash(pair_registry, "pair_registry_sha256"):
        raise ProvenanceError("pair registry self-hash is invalid")
    if run_spec.get("schema_version") != SCHEMA_VERSION or run_spec.get("run_type") != "m1_null_calibration":
        raise ProvenanceError("run specification schema/type is not frozen")
    if pair_registry.get("schema_version") != SCHEMA_VERSION or pair_registry.get("registry_type") != "m1_null_calibration_pair_registry":
        raise ProvenanceError("pair registry schema/type is not frozen")
    if run_spec.get("pair_count") != PAIR_COUNT or run_spec.get("pair_prefix") != PAIR_PREFIX:
        raise ProvenanceError("run specification pair count/prefix is not frozen")
    if pair_registry.get("pair_count") != PAIR_COUNT:
        raise ProvenanceError("pair registry pair count is not exactly 20")
    if pair_registry.get("run_spec_sha256") != run_spec.get("run_spec_sha256"):
        raise ProvenanceError("pair registry does not belong to this run specification")
    if pair_registry.get("registry_sha256") != run_spec.get("registry_sha256"):
        raise ProvenanceError("pair registry source registry hash differs from run specification")
    run_pairs = run_spec.get("pairs")
    pairs = pair_registry.get("pairs")
    # The pair registry is the authoritative schedule.  The run specification
    # also repeats it so a detached pair registry cannot be substituted.
    if not isinstance(run_pairs, Sequence) or isinstance(run_pairs, (str, bytes)):
        raise ProvenanceError("run specification has no frozen pair schedule")
    if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes)):
        raise ProvenanceError("pair registry has no frozen pair schedule")
    if len(run_pairs) != PAIR_COUNT or len(pairs) != PAIR_COUNT:
        raise ProvenanceError("frozen schedule must contain exactly 20 pairs")
    expected_trace_id = run_spec.get("trace_id")
    seen_attempt_ids: set[str] = set()
    for index, expected_pair_id in enumerate(EXPECTED_PAIR_IDS):
        run_pair, pair = run_pairs[index], pairs[index]
        if not isinstance(run_pair, Mapping) or not isinstance(pair, Mapping):
            raise ProvenanceError(f"frozen pair {expected_pair_id} is malformed")
        if run_pair.get("pair_id") != expected_pair_id or pair.get("pair_id") != expected_pair_id:
            raise ProvenanceError(f"frozen pair ID is not exactly {expected_pair_id}")
        if run_pair.get("trace_id") != expected_trace_id or pair.get("trace_id") != expected_trace_id:
            raise ProvenanceError(f"frozen pair {expected_pair_id} trace ID differs from run specification")
        run_attempts, pair_attempts = run_pair.get("attempts"), pair.get("attempts")
        if not isinstance(run_attempts, Sequence) or isinstance(run_attempts, (str, bytes)) or len(run_attempts) != 2:
            raise ProvenanceError(f"frozen run-spec pair {expected_pair_id} lacks exactly A and B")
        if not isinstance(pair_attempts, Sequence) or isinstance(pair_attempts, (str, bytes)) or len(pair_attempts) != 2:
            raise ProvenanceError(f"frozen pair {expected_pair_id} lacks exactly A and B")
        for side_index, side in enumerate(("A", "B")):
            run_attempt, pair_attempt = run_attempts[side_index], pair_attempts[side_index]
            if not isinstance(run_attempt, Mapping) or not isinstance(pair_attempt, Mapping):
                raise ProvenanceError(f"frozen attempt {expected_pair_id}-{side} is malformed")
            _validate_attempt_schedule_record(
                run_attempt,
                pair_id=expected_pair_id,
                side=side,
                ordinal=index * 2 + side_index,
            )
            _validate_attempt_schedule_record(
                pair_attempt,
                pair_id=expected_pair_id,
                side=side,
                ordinal=index * 2 + side_index,
            )
            if run_attempt.get("trace_id") != expected_trace_id or pair_attempt.get("trace_id") != expected_trace_id:
                raise ProvenanceError(f"frozen attempt {expected_pair_id}-{side} trace ID differs")
            attempt_id = str(pair_attempt["attempt_id"])
            if attempt_id in seen_attempt_ids:
                raise ProvenanceError(f"duplicate frozen attempt ID {attempt_id}")
            seen_attempt_ids.add(attempt_id)
    if seen_attempt_ids != {f"{pair_id}-{side}" for pair_id in EXPECTED_PAIR_IDS for side in ("A", "B")}:
        raise ProvenanceError("frozen attempt ID set is incomplete")
    if source_registry is not None:
        trace = _trace_record(source_registry)
        if str(trace.get("trace_id")) != str(expected_trace_id):
            raise ProvenanceError("run specification trace ID is not the source registry trace")
    if config is not None:
        task = config.get("task")
        if bool(config.get("strict_runtime_contract")) and task != TASK:
            raise ProvenanceError("run specification task identity is not frozen")


def _trace_record(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    traces = registry.get("traces")
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        raise ProvenanceError("registry has no trace record")
    record = traces[0]
    if not isinstance(record, Mapping):
        raise ProvenanceError("registry trace record is malformed")
    return record


def _tape_provenance(registry: Mapping[str, Any], trace_id: str) -> Mapping[str, Any]:
    raw = registry.get("tape_provenance")
    if not isinstance(raw, Mapping):
        raise ProvenanceError("registry tape_provenance is missing")
    per_trace = raw.get("per_trace")
    if isinstance(per_trace, Mapping) and isinstance(per_trace.get(trace_id), Mapping):
        return per_trace[trace_id]
    return raw


def _load_tape(registry: Mapping[str, Any], *, config: Mapping[str, Any], trace_id: str) -> tuple[np.ndarray, str]:
    provenance = _tape_provenance(registry, trace_id)
    path_value = provenance.get("path", provenance.get("tape_path", provenance.get("action_tape_path")))
    if path_value is None:
        configured = config.get("action_tape")
        if isinstance(configured, Mapping):
            path_value = configured.get("path")
    if path_value is None:
        raise ProvenanceError("registry tape provenance has no persisted path")
    tape_path = _resolve_path(path_value)
    if tape_path.suffix.lower() == ".npy":
        try:
            tape = np.load(tape_path, allow_pickle=False)
        except Exception as exc:
            raise ProvenanceError(f"could not load persisted action tape: {tape_path}") from exc
    else:
        try:
            tape = np.fromfile(tape_path, dtype=ACTION_DTYPE)
        except Exception as exc:
            raise ProvenanceError(f"could not load persisted raw action tape: {tape_path}") from exc
    tape = np.asarray(tape)
    declared_shape = provenance.get("shape", list(ACTION_SHAPE))
    declared_dtype = str(provenance.get("dtype", "float32"))
    if declared_dtype not in {"float32", "<f4", "|f4"}:
        raise ProvenanceError("frozen action tape dtype is not float32")
    if tuple(int(item) for item in declared_shape) != ACTION_SHAPE:
        raise ProvenanceError("frozen action tape shape is not (82, 7)")
    if tape.dtype != ACTION_DTYPE or tuple(tape.shape) != ACTION_SHAPE:
        raise ProvenanceError("persisted action tape has the wrong dtype or shape")
    tape = np.ascontiguousarray(tape)
    if not np.all(np.isfinite(tape)):
        raise ProvenanceError("persisted action tape is not finite")
    actual = sha256_bytes(tape.tobytes(order="C"))
    declared = str(provenance.get("sha256", "")).lower()
    if declared != actual:
        raise ProvenanceError(f"action tape hash drift: {actual} != {declared}")
    config_tape = config.get("action_tape")
    if isinstance(config_tape, Mapping):
        if str(config_tape.get("sha256", actual)).lower() != actual:
            raise ProvenanceError("config action tape SHA-256 differs from frozen registry")
        if list(config_tape.get("shape", list(ACTION_SHAPE))) != list(ACTION_SHAPE):
            raise ProvenanceError("config action tape shape differs from frozen contract")
        if str(config_tape.get("dtype", "float32")) not in {"float32", "<f4", "|f4"}:
            raise ProvenanceError("config action tape dtype differs from frozen contract")
    return tape, actual


def _windows(
    registry: Mapping[str, Any], trace: Mapping[str, Any], *, require_all: bool = True
) -> list[dict[str, Any]]:
    raw = trace.get("windows")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raw = [
            {
                "window_id": regime,
                "regimes": [regime],
                "capture_offset": trace.get("capture_offset", 1),
                "continuation_horizon": trace.get("continuation_horizon", 10),
                "source_coverage": {"regimes": [regime]},
            }
            for regime in trace.get("regimes", REGIME_NAMES)
        ]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ProvenanceError(f"registry window {index} is malformed")
        regimes = item.get("regimes", item.get("regime"))
        if isinstance(regimes, str):
            regimes = [regimes]
        if not isinstance(regimes, Sequence) or isinstance(regimes, (str, bytes)) or not regimes:
            raise ProvenanceError(f"registry window {index} has no regimes")
        normalized = [str(regime) for regime in regimes]
        if any(regime not in REGIME_NAMES for regime in normalized):
            raise ProvenanceError(f"registry window {index} has an unknown regime")
        offset = int(item.get("capture_offset"))
        horizon = int(item.get("continuation_horizon"))
        if offset <= 0 or horizon <= 0:
            raise ProvenanceError(f"registry window {index} has an invalid offset/horizon")
        key = str(item.get("window_id", normalized[0]))
        if key in seen:
            raise ProvenanceError(f"duplicate registry window {key}")
        seen.add(key)
        result.append(
            {
                **dict(item),
                "window_id": key,
                "regimes": normalized,
                "capture_offset": offset,
                "continuation_horizon": horizon,
                "source_coverage": dict(item.get("source_coverage", {"regimes": normalized})),
            }
        )
    covered = {regime for item in result for regime in item["regimes"]}
    if require_all and covered != set(REGIME_NAMES):
        raise ProvenanceError("frozen registry windows do not cover exactly all seven regimes")
    return result


@dataclass
class PreparedRun:
    config_path: Path
    config: dict[str, Any]
    source_registry_path: Path
    source_registry: dict[str, Any]
    run_spec_path: Path
    run_spec: dict[str, Any]
    pair_registry_path: Path
    pair_registry: dict[str, Any]
    tape: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class PairValidation:
    pair_id: str
    trace_id: str | None
    valid: bool
    reasons: tuple[str, ...] = ()
    attempts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, repr=False)
    discrete: Mapping[str, Any] = field(default_factory=dict)
    # Populated by the parent only after A/B validation.  It contains scalar
    # physics deltas and renderer disagreement metrics, never trajectory
    # snapshots or RGB arrays.
    measurements: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "trace_id": self.trace_id,
            "valid": self.valid,
            "reasons": list(self.reasons),
            "discrete": dict(self.discrete),
            "measurement_summary": {
                "physics_samples": len(self.measurements.get("physics_samples", ()))
                if isinstance(self.measurements, Mapping)
                else 0,
                "renderer_samples": len(self.measurements.get("renderer_samples", ()))
                if isinstance(self.measurements, Mapping)
                else 0,
            },
        }


@dataclass(frozen=True)
class EnvelopeBundle:
    physics: dict[str, Any]
    renderer: dict[str, Any]
    discrete: dict[str, Any]


def _attempt_protocol(attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    value = attempt.get("protocol")
    return value if isinstance(value, Mapping) else attempt


def _pid(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _proc_start_identity(pid: int) -> str:
    """Return a Linux process identity tied to ``/proc`` starttime and boot."""

    stat_path = Path(f"/proc/{int(pid)}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
        closing = raw.rfind(")")
        if closing < 0:
            raise ValueError("malformed /proc stat comm field")
        fields = raw[closing + 1 :].split()
        # The remainder starts at stat field 3 (state), so field 22
        # (starttime) is offset 19 here.
        starttime_ticks = fields[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except (OSError, IndexError, ValueError) as exc:
        raise ProtocolError(f"could not capture /proc identity for PID {pid}: {exc}") from exc
    if not boot_id:
        raise ProtocolError(f"/proc boot identity is empty for PID {pid}")
    return f"pid={int(pid)};starttime_ticks={starttime_ticks};boot_id={boot_id}"


_PROC_IDENTITY_RE = re.compile(
    r"^pid=(?P<pid>[1-9][0-9]*);starttime_ticks=(?P<start>[1-9][0-9]*);boot_id=(?P<boot>[0-9a-fA-F-]{8,})$"
)


def _parse_proc_start_identity(value: Any) -> tuple[int, str, str] | None:
    if not isinstance(value, str):
        return None
    match = _PROC_IDENTITY_RE.fullmatch(value)
    if match is None:
        return None
    return int(match.group("pid")), match.group("start"), match.group("boot").lower()


def _frozen_inputs(attempt: Mapping[str, Any]) -> Mapping[str, Any]:
    value = attempt.get("frozen_inputs")
    return value if isinstance(value, Mapping) else attempt


def _exact_equal(left: Any, right: Any) -> bool:
    try:
        from scripts import m1_hard_gate

        return bool(m1_hard_gate.exact_discrete_equal(left, right))
    except Exception:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
        return type(left) is type(right) and left == right


def _invariants(snapshot: Any) -> Mapping[str, Any]:
    if isinstance(snapshot, Mapping) and isinstance(snapshot.get("invariants"), Mapping):
        return snapshot["invariants"]
    if isinstance(snapshot, Mapping):
        return snapshot
    return {}


def _contact_entry(value: Any) -> tuple[tuple[str, str], Any] | None:
    """Extract one contact identity and its optional numeric distance."""

    if isinstance(value, Mapping):
        left = value.get("geom1", value.get("first", value.get("a")))
        right = value.get("geom2", value.get("second", value.get("b")))
        distance = value.get("distance", value.get("dist"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        left, right = value[0], value[1]
        distance = value[2] if len(value) >= 3 else None
    else:
        return None
    if left is None or right is None:
        return None
    pair = tuple(sorted((str(left), str(right))))
    return pair, distance


def _canonical_contact_entries(values: Any) -> list[tuple[tuple[str, str], Any]]:
    """Canonicalize contacts while retaining multiplicity.

    MuJoCo does not promise that equal contact rows are emitted in a stable
    order.  Identity is therefore sorted by the unordered geometry pair and
    duplicate rows are retained.  For duplicate identities, a finite distance
    is only a deterministic tie-breaker; it is not part of discrete contact
    identity.
    """

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    entries = [entry for value in values if (entry := _contact_entry(value)) is not None]

    def sort_key(entry: tuple[tuple[str, str], Any]) -> tuple[Any, ...]:
        pair, distance = entry
        try:
            numeric = float(distance)
            if math.isfinite(numeric):
                distance_key: tuple[int, float, str] = (0, numeric, "")
            else:
                distance_key = (1, 0.0, repr(distance))
        except (TypeError, ValueError, OverflowError):
            distance_key = (1, 0.0, repr(distance))
        return (*pair, *distance_key)

    return sorted(entries, key=sort_key)


def _snapshot_contacts(snapshot: Any) -> tuple[tuple[str, str], ...]:
    values = _invariants(snapshot).get("contacts", ())
    return tuple(pair for pair, _distance in _canonical_contact_entries(values))


def _contact_records(snapshot: Any) -> tuple[tuple[str, str], ...]:
    """Return actual named contact topology from the invariant snapshot."""

    values = _invariants(snapshot).get("contacts", ())
    return tuple(pair for pair, _distance in _canonical_contact_entries(values))


def _name_contains_token(value: Any, token: str) -> bool:
    text = str(value).lower()
    target = str(token).lower()
    if text == target:
        return True
    return re.search(rf"(?<![a-z0-9]){re.escape(target)}(?![a-z0-9])", text) is not None


def _target_object_tokens(snapshot: Any, *, strict: bool = False) -> set[str]:
    """Return names belonging only to the frozen target object."""

    objects = _invariants(snapshot).get("objects", {})
    if not isinstance(objects, Mapping):
        return set()
    tokens: set[str] = set()
    for name, value in objects.items():
        object_tokens = {str(name).lower()}
        if isinstance(value, Mapping):
            for field_name in ("name", "root_body", "body_name", "geom_name", "geom_names", "geoms"):
                candidate = value.get(field_name)
                if isinstance(candidate, str):
                    object_tokens.add(candidate.lower())
                elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                    object_tokens.update(str(item).lower() for item in candidate)
        is_target = any(_name_contains_token(candidate, TARGET_OBJECT_NAME) for candidate in object_tokens)
        # Non-strict historical test fixtures used ``target`` as a shorthand;
        # strict workers must expose the canonical frozen object identity.
        if not is_target and not strict and str(name).lower() == "target":
            is_target = True
            object_tokens.update({"target_geom"})
        if is_target:
            tokens.update(object_tokens)
    return {token for token in tokens if token}


def _terminal_semantics(attempt: Mapping[str, Any]) -> dict[str, Any]:
    terminal = attempt.get("terminal")
    return dict(terminal) if isinstance(terminal, Mapping) else {}


def _terminal_raw_evidence(terminal: Mapping[str, Any]) -> Any:
    """Return the authoritative raw-reason evidence, preserving ``None``."""

    if "raw_termination_evidence" in terminal:
        return terminal.get("raw_termination_evidence")
    if "official_evidence" in terminal:
        return terminal.get("official_evidence")
    return None


def _reset_provenance_errors(value: Any, *, strict: bool) -> list[str]:
    """Validate construction-vs-post-construction provenance evidence."""

    if value is None:
        return ["reset provenance is missing"] if strict else []
    if not isinstance(value, Mapping):
        return ["reset provenance is not a mapping"]
    construction = value.get("construction")
    if not isinstance(construction, Mapping):
        return ["reset provenance construction section is missing"]
    operations = construction.get("operations")
    forbidden = construction.get("post_construction_forbidden")
    errors: list[str] = []
    if not isinstance(operations, Mapping):
        errors.append("reset provenance construction operations are missing")
    else:
        reset = operations.get("reset")
        if (
            not isinstance(reset, Mapping)
            or reset.get("allowed") is not True
            or reset.get("observed") is not True
            or reset.get("count") != 1
        ):
            errors.append("reset provenance construction reset is not exactly one allowed operation")
        for name in ("set_init_state", "settle"):
            operation = operations.get(name)
            count = operation.get("count") if isinstance(operation, Mapping) else None
            valid_count = (
                not isinstance(count, bool)
                and isinstance(count, (int, np.integer))
                and int(count) >= 0
            )
            if (
                not isinstance(operation, Mapping)
                or operation.get("allowed") is not True
                or operation.get("observed") is not True
                or not valid_count
            ):
                errors.append(f"reset provenance construction {name} evidence is missing")
    if not isinstance(forbidden, Mapping):
        errors.append("reset provenance post-construction forbidden section is missing")
    else:
        for name in POST_CONSTRUCTION_FORBIDDEN_OPERATIONS:
            operation = forbidden.get(name)
            if not isinstance(operation, Mapping):
                errors.append(f"reset provenance forbidden operation is missing: {name}")
                continue
            if operation.get("allowed") is not False or operation.get("observed") is not True:
                errors.append(f"reset provenance marks forbidden operation as allowed: {name}")
            count = operation.get("count")
            valid_count = (
                not isinstance(count, bool)
                and isinstance(count, (int, np.integer))
                and int(count) >= 0
            )
            if not valid_count:
                errors.append(f"reset provenance forbidden count is invalid: {name}")
            elif int(count) != 0:
                errors.append(f"post-construction forbidden operation observed: {name}={count}")
    if strict:
        # A lazy LeRobot child is not observable by a probe installed before
        # LiberoEnv.reset().  The authoritative worker must therefore publish
        # source-bound records plus the actual post-reset state, and a
        # complete concrete-inner probe.  A nonnegative zero is insufficient
        # evidence for any of these fields.
        records = construction.get("authoritative_records") if isinstance(construction, Mapping) else None
        if not isinstance(records, Mapping):
            errors.append("reset provenance authoritative construction records are missing")
        else:
            expected_counts = {
                "outer_reset": 1,
                "inner_reset": 2,
                "set_init_state": 1,
                "settle": 10,
                "dummy_action": 10,
            }
            for name, expected in expected_counts.items():
                record = records.get(name)
                if not isinstance(record, Mapping):
                    errors.append(f"authoritative construction record is missing: {name}")
                    continue
                if record.get("observed") is not True or record.get("count") != expected:
                    errors.append(
                        f"authoritative construction record is not exact: {name}"
                    )
                if name != "outer_reset" and not isinstance(record.get("source"), Mapping):
                    errors.append(f"authoritative construction record source is missing: {name}")
            source_evidence = records.get("source_evidence")
            expected_source = {
                "path": "external/lerobot/src/lerobot/envs/libero.py",
                "sha256": "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71",
            }
            lerobot_source = source_evidence.get("lerobot_libero") if isinstance(source_evidence, Mapping) else None
            if not isinstance(lerobot_source, Mapping):
                errors.append("authoritative construction source evidence is missing: lerobot_libero")
            else:
                for field_name, expected in expected_source.items():
                    if lerobot_source.get(field_name) != expected:
                        errors.append(
                            f"authoritative construction source evidence differs: lerobot_libero.{field_name}"
                        )
            post_state = records.get("post_reset_state")
            if not isinstance(post_state, Mapping):
                errors.append("authoritative construction post-reset state is missing")
            else:
                expected_post_state = {
                    "inner_env_exists": True,
                    "num_steps_wait": 10,
                    "post_reset_timestep": 10,
                    "matches_num_steps_wait": True,
                }
                for field_name, expected in expected_post_state.items():
                    if post_state.get(field_name) != expected:
                        errors.append(
                            f"authoritative construction post-reset state differs: {field_name}"
                        )
        monitoring = construction.get("post_construction_monitoring") if isinstance(construction, Mapping) else None
        if not isinstance(monitoring, Mapping):
            errors.append("post-construction monitoring evidence is missing")
        else:
            if monitoring.get("complete") is not True or monitoring.get("observed") is not True:
                errors.append("post-construction monitoring evidence is incomplete")
            required_operations = monitoring.get("required_operations")
            if sorted(str(item) for item in (required_operations or ())) != ["reset", "set_init_state", "step"]:
                errors.append("post-construction monitoring required operations are incomplete")
            targets = monitoring.get("targets")
            installed = monitoring.get("installed")
            installed_operations = {
                str(item.get("operation"))
                for item in (installed or ())
                if isinstance(item, Mapping)
            }
            if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)) or not targets:
                errors.append("post-construction monitoring targets are missing")
            if not {"reset", "set_init_state", "step"} <= installed_operations:
                errors.append("post-construction monitoring hooks are incomplete")
    return errors


def _predicate_goal_values(snapshot: Any) -> tuple[bool, ...] | None:
    """Return the exact goal-predicate values exposed by one snapshot."""

    predicates = _invariants(snapshot).get("predicates")
    if not isinstance(predicates, Mapping) or predicates.get("available") is not True:
        return None
    goals = predicates.get("goals")
    if not isinstance(goals, Sequence) or isinstance(goals, (str, bytes)) or not goals:
        return None
    values: list[bool] = []
    for goal in goals:
        if not isinstance(goal, Mapping) or not isinstance(goal.get("value"), (bool, np.bool_)):
            return None
        values.append(bool(goal["value"]))
    return tuple(values)


def _predicate_transition_evidence(previous: Any, current: Any) -> dict[str, Any]:
    """Prove a predicate transition from adjacent official invariant frames."""

    before = _predicate_goal_values(previous)
    after = _predicate_goal_values(current)
    transitioned = bool(
        before is not None
        and after is not None
        and len(before) == len(after)
        and any(not old and new for old, new in zip(before, after))
    )
    return {
        "available": before is not None and after is not None,
        "previous": list(before) if before is not None else None,
        "current": list(after) if after is not None else None,
        "predicate_transition": transitioned,
    }


def _window_map(attempt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = attempt.get("windows")
    if isinstance(raw, Mapping):
        result = {str(key): value for key, value in raw.items() if isinstance(value, Mapping)}
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        result = {
            str(item.get("window_id", item.get("regime", index))): item
            for index, item in enumerate(raw)
            if isinstance(item, Mapping)
        }
    else:
        result = {}
    # The persisted worker form may key a single-regime window by its regime.
    for value in list(result.values()):
        regimes = value.get("regimes", value.get("regime"))
        if isinstance(regimes, str):
            regimes = [regimes]
        if isinstance(regimes, Sequence) and not isinstance(regimes, (str, bytes)):
            for regime in regimes:
                result.setdefault(str(regime), value)
    return result


def _window_snapshots(window: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw = window.get("snapshots", window.get("states"))
    if not isinstance(raw, Mapping):
        return {}
    result: dict[int, Mapping[str, Any]] = {}
    for key, value in raw.items():
        try:
            step = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            result[step] = value
    return dict(sorted(result.items()))


def _window_contract_map(value: Any) -> dict[str, Mapping[str, Any]]:
    """Return only persisted window IDs, without regime aliases."""

    raw = value.get("windows") if isinstance(value, Mapping) else value
    if isinstance(raw, Mapping):
        items = list(raw.values())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = list(raw)
    else:
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        window_id = str(item.get("window_id", item.get("regime", index)))
        if window_id in result:
            raise ProtocolError(f"duplicate persisted window ID: {window_id}")
        result[window_id] = item
    return result


def _validate_invariant_snapshot(
    snapshot: Mapping[str, Any], *, required_roots: Sequence[str] = STRICT_INVARIANT_ROOTS
) -> None:
    invariants = snapshot.get("invariants") if isinstance(snapshot.get("invariants"), Mapping) else snapshot
    if not isinstance(invariants, Mapping):
        raise ProtocolError("invariant snapshot is not a mapping")
    missing = [str(root) for root in required_roots if str(root) not in invariants]
    if missing:
        raise ProtocolError("invariant snapshot is missing required roots: " + ", ".join(missing))
    predicates = invariants.get("predicates")
    if not isinstance(predicates, Mapping) or not isinstance(predicates.get("available"), bool) or "goals" not in predicates:
        raise ProtocolError("predicate invariant schema/labels are missing")
    if not isinstance(invariants.get("success"), (bool, np.bool_)):
        raise ProtocolError("success invariant must be boolean")
    counters = invariants.get("counters")
    if not isinstance(counters, Mapping):
        raise ProtocolError("counter invariant schema is missing")
    for field_name in ("done", "terminated", "truncated"):
        if not isinstance(counters.get(field_name), (bool, np.bool_)):
            raise ProtocolError(f"counter invariant {field_name} must be boolean")
    contacts = invariants.get("contacts")
    if not isinstance(contacts, Sequence) or isinstance(contacts, (str, bytes)):
        raise ProtocolError("contact invariant schema is missing")
    gripper = invariants.get("gripper")
    if not isinstance(gripper, Mapping) or "current_action" not in gripper:
        raise ProtocolError("discrete gripper current_action invariant is missing")
    try:
        action = np.asarray(gripper["current_action"])
    except Exception as exc:
        raise ProtocolError("discrete gripper current_action is not numeric") from exc
    if action.dtype.kind not in {"b", "i", "u", "f"} or action.size == 0:
        raise ProtocolError("discrete gripper current_action is not a non-empty numeric array")


def _configured_invariant_roots(config: Mapping[str, Any]) -> tuple[str, ...]:
    invariants = config.get("invariants")
    if isinstance(invariants, Mapping) and isinstance(invariants.get("required_roots"), Sequence):
        roots = tuple(str(root) for root in invariants["required_roots"])
        if roots:
            return roots
    return STRICT_INVARIANT_ROOTS


def _validate_terminal(
    terminal: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
    *,
    require_official_evidence: bool = False,
) -> None:
    missing = [field_name for field_name in TERMINAL_FIELDS if field_name not in terminal]
    if missing:
        raise ProtocolError("terminal is missing required fields: " + ", ".join(missing))
    if isinstance(terminal["step"], bool) or int(terminal["step"]) <= 0:
        raise ProtocolError("terminal.step must be a positive integer")
    if not isinstance(terminal["termination_reason"], str) or not terminal["termination_reason"].strip():
        raise ProtocolError("terminal termination_reason is required")
    for field_name in ("success", "terminated", "truncated"):
        if not isinstance(terminal[field_name], (bool, np.bool_)):
            raise ProtocolError(f"terminal {field_name} must be boolean")
    if require_official_evidence:
        evidence = terminal.get("official_evidence")
        if not isinstance(evidence, Mapping):
            raise ProtocolError("terminal official evidence is missing")
        source = evidence.get("source")
        if source not in {"step_result", "step_info", "adapter", "missing"}:
            raise ProtocolError("terminal official evidence source is invalid")
        if evidence.get("returned_step") is not True:
            raise ProtocolError("terminal official evidence is not tied to returned step")
        for field_name in ("terminated", "truncated"):
            if not _exact_equal(evidence.get(field_name), terminal[field_name]):
                raise ProtocolError(f"terminal official evidence {field_name} differs")
        semantic = evidence.get("semantic_derivation")
        if terminal["termination_reason"] == "predicate_transition":
            # LIBERO/LeRobot may report only a generic wrapper termination or
            # no raw reason at all.  The semantic claim is valid only when the
            # adjacent official predicate frames independently prove the
            # false-to-true transition at the frozen successful terminal step.
            if not isinstance(semantic, Mapping):
                raise ProtocolError(
                    "predicate_transition terminal lacks independent semantic proof"
                )
            predicate_evidence = semantic.get("predicate_evidence")
            if not (
                semantic.get("derived") is True
                and semantic.get("frozen_terminal_step") is True
                and semantic.get("success") is True
                and semantic.get("predicate_transition") is True
                and isinstance(predicate_evidence, Mapping)
                and predicate_evidence.get("available") is True
                and predicate_evidence.get("predicate_transition") is True
            ):
                raise ProtocolError(
                    "predicate_transition terminal lacks independent predicate proof"
                )
            semantic_reason = semantic.get("semantic_reason")
            if semantic_reason is not None and semantic_reason != "predicate_transition":
                raise ProtocolError("terminal semantic proof reason differs")
    if expected is not None:
        for field_name in TERMINAL_FIELDS:
            if not _exact_equal(terminal[field_name], expected[field_name]):
                raise ProtocolError(
                    f"terminal {field_name} differs from frozen contract: "
                    f"{terminal[field_name]!r} != {expected[field_name]!r}"
                )


def _numeric_leaves(value: Any, path: str = "") -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.update(_numeric_leaves(child, child_path))
        return result
    if isinstance(value, (bool, np.bool_)) or value is None or isinstance(value, (str, bytes)):
        return result
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        array = np.asarray(value)
    elif isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except Exception:
            return result
    else:
        return result
    if array.dtype.kind not in {"b", "i", "u", "f", "c"} or array.size == 0:
        return result
    if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
        return result
    result[path] = np.ascontiguousarray(array).copy()
    return result


def _contact_distance_leaves(value: Any, path: str = "contacts") -> dict[str, np.ndarray]:
    """Extract contact distances without treating geometry labels as numbers.

    Contact topology is a discrete exact field, while MuJoCo's signed contact
    distance is a floating physical quantity.  The canonical invariant uses
    ``(geom1, geom2, distance)`` tuples, but accepting named mappings here
    keeps the quantity collector aligned with the official observation tree.
    """

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    result: dict[str, np.ndarray] = {}
    occurrence_by_identity: dict[tuple[str, str], int] = {}
    for identity, distance in _canonical_contact_entries(value):
        if distance is None or isinstance(distance, (bool, np.bool_)):
            continue
        try:
            numeric = np.asarray(distance, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if numeric.shape != () or not np.all(np.isfinite(numeric)):
            continue
        occurrence = occurrence_by_identity.get(identity, 0)
        occurrence_by_identity[identity] = occurrence + 1
        # JSON encoding makes the pair boundary unambiguous even when a
        # geometry name contains punctuation used by this diagnostic path.
        identity_token = json.dumps(list(identity), ensure_ascii=True, separators=(",", ":"))
        result[f"{path}[{identity_token}][{occurrence}].distance"] = np.asarray(
            [float(numeric)], dtype=np.float64
        )
    return result


def _selected_numeric_leaves(snapshot: Any, roots: Sequence[str]) -> dict[str, np.ndarray]:
    invariants = _invariants(snapshot)
    result: dict[str, np.ndarray] = {}
    for root in roots:
        if root not in invariants:
            continue
        if root == "contacts":
            result.update(_contact_distance_leaves(invariants[root], root))
            continue
        for path, value in _numeric_leaves(invariants[root], root).items():
            result[path] = value
    # Official observation-tree numeric leaves are physics quantities too,
    # except RGB images, which belong exclusively to the renderer envelope.
    # Keep their namespace distinct from invariant roots so the envelope can
    # be independently reconstructed by regime and continuation horizon.
    raw_observation = snapshot.get("raw_observation") if isinstance(snapshot, Mapping) else None

    def visit_observation(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit_observation(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit_observation(child, f"{path}[{index}]")
            return
        try:
            array = np.asarray(value)
        except Exception:
            return
        if array.dtype.kind not in {"b", "i", "u", "f", "c"} or array.size == 0:
            return
        if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
            return
        # Official camera leaves are excluded by semantic key even if a
        # backend exposes them in a non-uint8 dtype.  Unknown uint8 image-like
        # leaves are also renderer-only; strict RGB key validation occurs in
        # _renderer_images.
        if _renderer_camera_for_key(path) is not None or (
            array.dtype == np.dtype("uint8") and array.ndim >= 2
        ):
            return
        result[f"observation.{path}"] = np.ascontiguousarray(array).copy()

    if raw_observation is not None:
        visit_observation(raw_observation, "")
    return result


def _actual_robot_object_contact(snapshot: Any, *, strict: bool = False) -> bool:
    contacts = _contact_records(snapshot)
    object_tokens = _target_object_tokens(snapshot, strict=strict)
    if not object_tokens:
        return False
    for left, right in contacts:
        left_text, right_text = left.lower(), right.lower()
        object_side = next(
            (side for side in (left_text, right_text) if any(token in side for token in object_tokens)),
            None,
        )
        robot_side = right_text if object_side == left_text else left_text if object_side == right_text else None
        if object_side is not None and robot_side is not None and any(
            token in robot_side for token in ("robot", "finger", "gripper", "hand")
        ):
            return True
    return False


def _close_command(snapshot: Any, action: Any = None) -> bool:
    if action is not None:
        try:
            values = np.asarray(action, dtype=np.float32).reshape(-1)
            if values.size >= 7:
                # The pinned robosuite PandaGripper contract is -1=open,
                # +1=closed.  Keep the final command component as the
                # semantic scalar even when a persisted gripper state has two
                # opposing finger channels.
                return bool(values[6] > 0.0)
        except Exception:
            pass
    current = _invariants(snapshot).get("gripper", {})
    if isinstance(current, Mapping):
        value = current.get("current_action")
        if value is not None:
            try:
                values = np.asarray(value, dtype=np.float64).reshape(-1)
                if values.size:
                    # PandaGripper.current_action is [left, right] and its
                    # right channel carries the same open/closed sign as the
                    # one-dimensional command.  Do not infer closure from a
                    # minimum over the opposing channels.
                    return bool(values[-1] > 0.0)
            except Exception:
                pass
    return False


def _object_positions(snapshot: Any) -> list[np.ndarray]:
    objects = _invariants(snapshot).get("objects", {})
    result: list[np.ndarray] = []
    if isinstance(objects, Mapping):
        for value in objects.values():
            if isinstance(value, Mapping):
                position = value.get("body_pos", value.get("position"))
                if position is not None:
                    try:
                        array = np.asarray(position, dtype=np.float64)
                        if array.size:
                            result.append(array.reshape(-1))
                    except Exception:
                        pass
    return result


def _object_position_map(snapshot: Any, *, target_only: bool = False, strict: bool = False) -> dict[str, np.ndarray]:
    objects = _invariants(snapshot).get("objects", {})
    if not isinstance(objects, Mapping):
        return {}
    target_tokens = _target_object_tokens(snapshot, strict=strict) if target_only else set()
    result: dict[str, np.ndarray] = {}
    for name, value in objects.items():
        if not isinstance(value, Mapping):
            continue
        if target_only:
            object_tokens = {str(name).lower()}
            for field_name in ("name", "root_body", "body_name", "geom_name", "geom_names", "geoms"):
                candidate = value.get(field_name)
                if isinstance(candidate, str):
                    object_tokens.add(candidate.lower())
                elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                    object_tokens.update(str(item).lower() for item in candidate)
            if not target_tokens or not any(token in target_tokens for token in object_tokens):
                continue
        position = value.get("body_pos", value.get("position"))
        if position is None:
            continue
        try:
            array = np.asarray(position, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if array.size and np.all(np.isfinite(array)):
            result[str(name)] = array
    return result


def _object_motion(first: Any, current: Any, *, strict: bool = False) -> bool:
    left = _object_position_map(first, target_only=True, strict=strict)
    right = _object_position_map(current, target_only=True, strict=strict)
    if set(left) != set(right):
        return False
    return any(a.shape == right[name].shape and not np.array_equal(a, right[name]) for name, a in left.items())


def _independent_evidence(
    regime: str,
    first: Any,
    current: Any,
    *,
    action: Any = None,
    history: Sequence[Mapping[str, Any]] | None = None,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    frames = list(history or (first, current))
    contacts = [_actual_robot_object_contact(frame, strict=strict) for frame in frames]
    closes = [_close_command(frame, frame.get("action") if isinstance(frame, Mapping) else action) for frame in frames]
    contact = bool(_actual_robot_object_contact(current, strict=strict))
    close = _close_command(current, action)
    sustained = any(all(contacts[index : index + 2]) and all(closes[index : index + 2]) for index in range(max(0, len(frames) - 1)))
    if len(frames) < 2:
        sustained = False
    carried = any(_object_motion(first, frame, strict=strict) for frame in frames[1:])
    contact_observed = any(contacts)
    source = "attempt_snapshots_and_frozen_actions"
    return {
        "contact": {
            "present": contact,
            "observed": contact_observed,
            "source": source,
            "target_object": TARGET_OBJECT_NAME,
            "topology": list(_contact_records(current)),
        },
        "grasp": {
            "present": bool(contact and close and sustained),
            "observed": bool(sustained),
            "source": source,
            "target_object": TARGET_OBJECT_NAME,
            "sustained": sustained,
        },
        "carried": {
            "present": carried,
            "observed": carried,
            "source": source,
            "target_object": TARGET_OBJECT_NAME,
        },
    }


def _window_regime_requirements(
    regime: str, history: Sequence[Mapping[str, Any]], *, strict: bool = False
) -> dict[str, bool]:
    """Check regime evidence across its full designated continuation window."""

    frames = list(history)
    if not frames:
        return {"contact": False, "grasp": False, "carried": False}
    derived = _independent_evidence(
        str(regime),
        frames[0],
        frames[-1],
        action=frames[-1].get("action") if isinstance(frames[-1], Mapping) else None,
        history=frames,
        strict=strict,
    )
    return {
        "contact": bool(derived.get("contact", {}).get("observed")),
        "grasp": bool(derived.get("grasp", {}).get("observed")),
        "carried": bool(derived.get("carried", {}).get("observed")),
    }


def validate_pair(
    pair: Mapping[str, Any],
    attempts: Mapping[str, Mapping[str, Any]],
    *,
    parent_pid: int | None = None,
    expected_frozen_inputs: Mapping[str, Any] | None = None,
    expected_terminal: Mapping[str, Any] | None = None,
    expected_runtime_identity: Mapping[str, Any] | None = None,
    expected_windows: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    expected_action_tape: np.ndarray | None = None,
    required_invariant_roots: Sequence[str] = STRICT_INVARIANT_ROOTS,
    strict: bool = False,
) -> PairValidation:
    """Validate one complete A/B pair without applying numeric null limits."""

    pair_id = str(pair.get("pair_id", ""))
    reasons: list[str] = []
    if not pair_id:
        reasons.append("pair_id is missing")
    elif pair_id not in EXPECTED_PAIR_IDS:
        reasons.append(f"pair_id is outside the frozen schedule: {pair_id}")
    if not isinstance(attempts, Mapping) or not isinstance(attempts.get("A"), Mapping) or not isinstance(attempts.get("B"), Mapping):
        reasons.append("pair does not have exactly one A and one B attempt")
        return PairValidation(pair_id, None, False, tuple(reasons), {}, {"categorical_gate": False})
    left = attempts["A"]
    right = attempts["B"]
    if set(str(key) for key in attempts) != {"A", "B"}:
        reasons.append("pair has unexpected attempt sides")
    for side, attempt in (("A", left), ("B", right)):
        expected_attempt_id = f"{pair_id}-{side}"
        if attempt.get("attempt_id") != expected_attempt_id:
            reasons.append(
                f"{side} attempt_id differs from frozen registry: "
                f"{attempt.get('attempt_id')!r} != {expected_attempt_id!r}"
            )
        if attempt.get("pair_id") != pair_id:
            reasons.append(f"{side} attempt pair_id differs from frozen pair: {attempt.get('pair_id')!r}")
        if attempt.get("side") != side:
            reasons.append(f"{side} attempt side field differs from frozen side")
        if attempt.get("status") != "completed":
            reasons.append(f"{side} attempt is not completed")
        pid = _pid(attempt.get("pid"))
        if pid is None:
            reasons.append(f"{side} attempt has no positive PID")
        elif parent_pid is not None and pid == int(parent_pid):
            reasons.append(f"{side} attempt PID is the parent PID")
        if _pid(attempt.get("ppid")) is None:
            reasons.append(f"{side} attempt has no positive PPID")
        elif strict and parent_pid is not None and _pid(attempt.get("ppid")) != int(parent_pid):
            reasons.append(f"{side} attempt PPID is not the launching parent PID")
        process_identity = attempt.get("process_start_identity")
        if not isinstance(process_identity, str) or not process_identity:
            reasons.append(f"{side} attempt lacks process-start identity")
        elif strict:
            if attempt.get("process_start_identity_source") != "/proc/<pid>/stat:starttime_ticks+boot_id":
                reasons.append(f"{side} process-start identity source is not /proc")
            parsed_identity = _parse_proc_start_identity(process_identity)
            if parsed_identity is None:
                reasons.append(f"{side} attempt lacks a real /proc process-start identity")
            elif pid is None or parsed_identity[0] != pid:
                reasons.append(f"{side} /proc identity PID does not match attempt PID")
            elif _pid(attempt.get("ppid")) == pid:
                reasons.append(f"{side} attempt PPID equals its own PID")
        protocol = _attempt_protocol(attempt)
        terminal = _terminal_semantics(attempt)
        provenance = attempt.get("reset_provenance")
        provenance_errors = _reset_provenance_errors(provenance, strict=strict)
        for error in provenance_errors:
            reasons.append(f"{side} {error}")
        if strict and "post_construction_forbidden" not in protocol:
            reasons.append(f"{side} protocol post-construction provenance is missing")
        if isinstance(provenance, Mapping) and isinstance(protocol.get("post_construction_forbidden"), Mapping):
            construction = provenance.get("construction")
            expected_forbidden = construction.get("post_construction_forbidden") if isinstance(construction, Mapping) else None
            if isinstance(expected_forbidden, Mapping) and not _exact_equal(
                protocol.get("post_construction_forbidden"), expected_forbidden
            ):
                reasons.append(f"{side} protocol post-construction provenance disagrees with reset provenance")
        if strict and not isinstance(terminal, Mapping):
            reasons.append(f"{side} terminal evidence is missing")
        if strict and isinstance(terminal, Mapping):
            for field_name in TERMINAL_FIELDS + ("raw_termination_reason", "raw_termination_evidence", "official_evidence"):
                if field_name not in terminal:
                    reasons.append(f"{side} terminal field is missing: {field_name}")
        if strict:
            required_protocol = (
                "construction_reset_count",
                *FORBIDDEN_PROTOCOL_FIELDS,
                "actions_executed",
                "step_calls",
                "render_calls",
                "invariant_collections",
                "reset_provenance",
                "post_construction_forbidden",
            )
            for field_name in required_protocol:
                if field_name not in protocol:
                    reasons.append(f"{side} observed protocol counter is missing: {field_name}")
            for field_name in ("step_calls", "invariant_collections"):
                if field_name in protocol:
                    try:
                        valid_count = (
                            not isinstance(protocol[field_name], bool)
                            and int(protocol[field_name]) == protocol[field_name]
                        )
                    except (TypeError, ValueError, OverflowError):
                        valid_count = False
                    if not valid_count or int(protocol[field_name]) != ACTION_SHAPE[0]:
                        reasons.append(f"{side} did not observe exactly 82 {field_name}")
            if "render_calls" in protocol:
                try:
                    valid_count = (
                        not isinstance(protocol["render_calls"], bool)
                        and int(protocol["render_calls"]) == protocol["render_calls"]
                    )
                except (TypeError, ValueError, OverflowError):
                    valid_count = False
                if not valid_count or int(protocol["render_calls"]) != ACTION_SHAPE[0]:
                    reasons.append(f"{side} did not observe exactly 82 render_calls")
        for field_name in FORBIDDEN_PROTOCOL_FIELDS:
            value = protocol.get(field_name, 0)
            if strict and field_name in protocol:
                try:
                    valid_zero = (
                        not isinstance(value, bool)
                        and int(value) == value
                        and int(value) == 0
                    )
                except (TypeError, ValueError, OverflowError):
                    valid_zero = value is False or value is None
                if not valid_zero:
                    reasons.append(f"{side} forbidden protocol activity: {field_name}={value}")
            elif value not in (0, False, None):
                reasons.append(f"{side} forbidden protocol activity: {field_name}={value}")
        if "construction_reset_count" in protocol and protocol.get("construction_reset_count") != 1:
            reasons.append(f"{side} construction reset count is not exactly one")
        if "actions_executed" in protocol:
            try:
                valid_count = (
                    not isinstance(protocol["actions_executed"], bool)
                    and int(protocol["actions_executed"]) == protocol["actions_executed"]
                )
            except (TypeError, ValueError, OverflowError):
                valid_count = False
            if not valid_count or int(protocol["actions_executed"]) != ACTION_SHAPE[0]:
                reasons.append(f"{side} did not execute exactly 82 actions")
        output_sha = attempt.get("output_sha256")
        artifact_sha = attempt.get("artifact_sha256")
        if strict and not isinstance(output_sha, str):
            reasons.append(f"{side} output payload SHA is missing")
        if strict and not isinstance(artifact_sha, str):
            reasons.append(f"{side} actual artifact SHA is missing")
        # The strict worker contract binds both payload and sidecar hashes.
        # Lightweight non-strict fixtures may be edited in-memory to exercise
        # regime diagnostics and do not carry an immutable artifact record.
        if output_sha is not None and (strict or artifact_sha is not None):
            if artifact_sha is not None:
                artifact_path = attempt.get("artifact_path")
                if artifact_path is None:
                    reasons.append(f"{side} artifact path is missing for output SHA")
                else:
                    try:
                        verify_artifact_hash(artifact_path, str(artifact_sha))
                    except CalibrationError as exc:
                        reasons.append(f"{side} output artifact hash is invalid: {exc}")
                payload = dict(attempt)
                for metadata_name in (
                    "artifact_path",
                    "artifact_sha256",
                    "artifact_size",
                    "artifact_integrity",
                ):
                    payload.pop(metadata_name, None)
                try:
                    if output_sha != payload_sha256(payload):
                        reasons.append(f"{side} output payload hash is invalid")
                except CalibrationError as exc:
                    reasons.append(f"{side} output payload hash is invalid: {exc}")
            elif output_sha != payload_sha256(attempt):
                reasons.append(f"{side} output artifact hash is invalid")
    left_pid, right_pid = _pid(left.get("pid")), _pid(right.get("pid"))
    if left_pid is not None and right_pid is not None and left_pid == right_pid:
        reasons.append("A and B reuse the same process PID")
    left_start, right_start = left.get("process_start_identity"), right.get("process_start_identity")
    if left_start == right_start:
        reasons.append("A and B reuse the same process-start identity")

    runtime_identities: dict[str, Mapping[str, Any]] = {}
    if strict:
        for side, attempt in (("A", left), ("B", right)):
            runtime = attempt.get("runtime")
            identity = runtime.get("identity") if isinstance(runtime, Mapping) else None
            if not isinstance(identity, Mapping) or not identity:
                reasons.append(f"{side} runtime identity facts are missing")
                continue
            runtime_identities[side] = identity
            for name in ("python_executable", "runtime_lock", "facts", "expected_gl", "gl_identity"):
                if name not in identity:
                    reasons.append(f"{side} runtime identity field is missing: {name}")
            expected_gl = identity.get("expected_gl")
            live_gl = identity.get("gl_identity")
            if not isinstance(expected_gl, Mapping) or not isinstance(live_gl, Mapping):
                reasons.append(f"{side} live/frozen GL identity is missing")
            elif not _exact_equal(expected_gl, live_gl):
                reasons.append(f"{side} live GL identity differs from frozen GL identity")
            runtime_python = runtime.get("python_executable") if isinstance(runtime, Mapping) else None
            if runtime_python is not None and runtime_python != identity.get("python_executable"):
                reasons.append(f"{side} runtime executable differs from its identity record")
        if "A" in runtime_identities and "B" in runtime_identities:
            if not _exact_equal(runtime_identities["A"], runtime_identities["B"]):
                reasons.append("A and B runtime facts/GL identities differ")
        if expected_runtime_identity is not None:
            for side, identity in runtime_identities.items():
                for name, expected in expected_runtime_identity.items():
                    if name == "gl_identity":
                        continue
                    if name not in identity or not _exact_equal(identity[name], expected):
                        reasons.append(f"{side} runtime identity {name} differs from the run contract")

    left_inputs, right_inputs = _frozen_inputs(left), _frozen_inputs(right)
    required_inputs = (
        "trace_id",
        "task",
        "action_tape_sha256",
        "action_shape",
        "action_dtype",
        "physics_model_fingerprint",
        "observation_model_fingerprint",
    )
    trace_id: str | None = None
    for name in required_inputs:
        if name not in left_inputs or name not in right_inputs:
            reasons.append(f"frozen input {name} is missing")
            continue
        if not _exact_equal(left_inputs[name], right_inputs[name]):
            reasons.append(f"frozen input {name} differs between A and B")
        if name == "trace_id":
            trace_id = str(left_inputs[name])
        if strict and name in {"physics_model_fingerprint", "observation_model_fingerprint"}:
            if not isinstance(left_inputs[name], Mapping) or not left_inputs[name]:
                reasons.append(f"frozen input {name} is empty")
            if not isinstance(right_inputs[name], Mapping) or not right_inputs[name]:
                reasons.append(f"frozen input {name} is empty on B")
    if expected_frozen_inputs is not None:
        for name, expected in expected_frozen_inputs.items():
            if name not in left_inputs or name not in right_inputs:
                reasons.append(f"frozen input {name} is missing from A or B")
                continue
            if not _exact_equal(left_inputs[name], expected):
                reasons.append(f"A frozen input {name} differs from the run specification")
            if not _exact_equal(right_inputs[name], expected):
                reasons.append(f"B frozen input {name} differs from the run specification")
    # Some callers retain a human/source alias on the pair record while the
    # authoritative attempt carries the concrete episode ID.  Enforce the
    # pair-level identity when both attempts explicitly persist it, otherwise
    # the frozen-input equality check above is the source of truth.
    if (
        pair.get("trace_id") is not None
        and trace_id is not None
        and ("trace_id" in left or "trace_id" in right)
        and str(pair["trace_id"]) != trace_id
    ):
        reasons.append("pair trace_id differs from attempt frozen input")

    left_terminal, right_terminal = _terminal_semantics(left), _terminal_semantics(right)
    if left_terminal or right_terminal:
        for field_name in TERMINAL_FIELDS:
            if not _exact_equal(left_terminal.get(field_name), right_terminal.get(field_name)):
                reasons.append(f"terminal {field_name} differs between A and B")
        for field_name in ("raw_termination_reason", "raw_termination_evidence", "official_evidence"):
            if field_name not in left_terminal and field_name not in right_terminal:
                continue
            if not _exact_equal(left_terminal.get(field_name), right_terminal.get(field_name)):
                reasons.append(f"raw termination authoritative {field_name} differs between A and B")
        if not _exact_equal(
            _terminal_raw_evidence(left_terminal), _terminal_raw_evidence(right_terminal)
        ):
            reasons.append("raw termination authoritative evidence differs between A and B")

    left_provenance, right_provenance = left.get("reset_provenance"), right.get("reset_provenance")
    if left_provenance is not None or right_provenance is not None:
        if not _exact_equal(left_provenance, right_provenance):
            reasons.append("reset provenance differs between A and B")
    left_protocol_provenance = _attempt_protocol(left).get("post_construction_forbidden")
    right_protocol_provenance = _attempt_protocol(right).get("post_construction_forbidden")
    if left_protocol_provenance is not None or right_protocol_provenance is not None:
        if not _exact_equal(left_protocol_provenance, right_protocol_provenance):
            reasons.append("post-construction protocol provenance differs between A and B")

    left_windows, right_windows = _window_map(left), _window_map(right)
    if strict:
        expected_window_map = _window_contract_map(expected_windows) if expected_windows is not None else {}
        left_contract_windows = _window_contract_map(left)
        right_contract_windows = _window_contract_map(right)
        if expected_windows is not None:
            if set(left_contract_windows) != set(expected_window_map):
                reasons.append("A persisted window IDs differ from frozen schedule")
            if set(right_contract_windows) != set(expected_window_map):
                reasons.append("B persisted window IDs differ from frozen schedule")
            for window_id, frozen in expected_window_map.items():
                for side, actual in (
                    ("A", left_contract_windows.get(window_id)),
                    ("B", right_contract_windows.get(window_id)),
                ):
                    if actual is None:
                        continue
                    for field_name in ("window_id", "regimes", "capture_offset", "continuation_horizon"):
                        expected_value = frozen.get(field_name)
                        actual_value = actual.get(field_name)
                        if field_name == "regimes":
                            expected_value = [str(item) for item in (expected_value or ())]
                            actual_value = [str(item) for item in (actual_value or ())]
                        if not _exact_equal(actual_value, expected_value):
                            reasons.append(
                                f"{side} window {window_id} {field_name} differs from frozen schedule"
                            )
                    actual_snapshots = _window_snapshots(actual)
                    try:
                        frozen_horizon = int(frozen["continuation_horizon"])
                    except (KeyError, TypeError, ValueError):
                        frozen_horizon = -1
                    if set(actual_snapshots) != set(range(frozen_horizon + 1)):
                        reasons.append(f"{side} window {window_id} snapshot coordinates are not frozen")
        elif set(left_contract_windows) != set(right_contract_windows):
            reasons.append("persisted window IDs differ between A and B")
    regime_keys = sorted(set(left_windows) | set(right_windows))
    if not regime_keys:
        reasons.append("pair has no frozen regime windows")
    discrete = {
        "categorical_gate": True,
        "contact_identity_exact": True,
        "predicate_exact": True,
        "success_exact": True,
        "done_termination_exact": True,
        "terminal_timing_exact": True,
        "gripper_exact": True,
        "action_exact": True,
        "observation_exact": True,
        "regime_exact": set(left_windows) == set(right_windows),
        "divergences": [],
    }
    if not discrete["regime_exact"]:
        reasons.append("regime window coordinate sets differ")
        discrete["categorical_gate"] = False
    for regime in regime_keys:
        lwindow, rwindow = left_windows.get(regime), right_windows.get(regime)
        if lwindow is None or rwindow is None:
            continue
        lsnap, rsnap = _window_snapshots(lwindow), _window_snapshots(rwindow)
        if set(lsnap) != set(rsnap) or not lsnap:
            reasons.append(f"window {regime} snapshot coordinate sets differ or are empty")
            discrete["categorical_gate"] = False
            continue
        # A frozen capture offset denotes the beginning of a continuation
        # window, not a requirement that the semantic event already hold at
        # horizon zero.  In particular, the contact window is captured at
        # step 43 and first contact is observed at step 44 (horizon one).
        # Derive the requirement once over the complete window while retaining
        # the per-horizon evidence below for diagnostics and envelope grouping.
        left_history = [lsnap[step] for step in sorted(lsnap)]
        right_history = [rsnap[step] for step in sorted(rsnap)]
        left_window_requirements = _window_regime_requirements(
            regime, left_history, strict=strict
        )
        right_window_requirements = _window_regime_requirements(
            regime, right_history, strict=strict
        )
        if left_window_requirements != right_window_requirements:
            discrete["regime_exact"] = False
            discrete["categorical_gate"] = False
            message = f"{regime} continuation evidence differs between A and B"
            discrete["divergences"].append(message)
            reasons.append(message)
        for horizon in sorted(set(lsnap) & set(rsnap)):
            if strict:
                for side, snapshot in (("A", lsnap[horizon]), ("B", rsnap[horizon])):
                    try:
                        _validate_invariant_snapshot(
                            snapshot,
                            required_roots=required_invariant_roots,
                        )
                    except ProtocolError as exc:
                        discrete["categorical_gate"] = False
                        reasons.append(f"{side} {regime}@{horizon} invariant evidence is invalid: {exc}")
            li, ri = _invariants(lsnap[horizon]), _invariants(rsnap[horizon])
            exact_fields = (
                ("predicates", "predicate_exact"),
                ("success", "success_exact"),
                ("counters.done", "done_termination_exact"),
                ("counters.terminated", "done_termination_exact"),
                ("counters.truncated", "done_termination_exact"),
            )
            for field_name, category in exact_fields:
                def nested(value: Mapping[str, Any], path: str) -> Any:
                    current: Any = value
                    for token in path.split("."):
                        current = current.get(token) if isinstance(current, Mapping) else None
                    return current
                lv, rv = nested(li, field_name), nested(ri, field_name)
                if not _exact_equal(lv, rv):
                    discrete[category] = False
                    discrete["categorical_gate"] = False
                    message = f"{regime}@{horizon} {field_name} differs"
                    discrete["divergences"].append(message)
                    reasons.append(message)
            if _snapshot_contacts(lsnap[horizon]) != _snapshot_contacts(rsnap[horizon]):
                discrete["contact_identity_exact"] = False
                discrete["categorical_gate"] = False
                message = f"{regime}@{horizon} contact identity differs"
                discrete["divergences"].append(message)
                reasons.append(message)
            if strict:
                try:
                    left_action = _snapshot_action(lsnap[horizon])
                    right_action = _snapshot_action(rsnap[horizon])
                except ProtocolError as exc:
                    left_action = right_action = None
                    discrete["action_exact"] = False
                    discrete["categorical_gate"] = False
                    reasons.append(f"{regime}@{horizon} action evidence is invalid: {exc}")
                if (
                    left_action is None
                    or right_action is None
                    or left_action.tobytes(order="C") != right_action.tobytes(order="C")
                ):
                    discrete["action_exact"] = False
                    discrete["categorical_gate"] = False
                    message = f"{regime}@{horizon} frozen action differs or is missing"
                    discrete["divergences"].append(message)
                    reasons.append(message)
                expected_tape_sha = left_inputs.get("action_tape_sha256")
                for side, window in (("A", lwindow), ("B", rwindow)):
                    window_sha = window.get("action_sha256")
                    if expected_tape_sha is None or str(window_sha).lower() != str(expected_tape_sha).lower():
                        discrete["action_exact"] = False
                        discrete["categorical_gate"] = False
                        reasons.append(f"{side} {regime}@{horizon} action tape hash differs from frozen input")
                if expected_action_tape is not None:
                    try:
                        expected_action = np.asarray(expected_action_tape)[
                            int(lwindow["capture_offset"]) + int(horizon) - 1
                        ]
                        if (
                            left_action is None
                            or right_action is None
                            or left_action.tobytes(order="C") != expected_action.tobytes(order="C")
                            or right_action.tobytes(order="C") != expected_action.tobytes(order="C")
                        ):
                            discrete["action_exact"] = False
                            discrete["categorical_gate"] = False
                            message = f"{regime}@{horizon} action differs from frozen tape"
                            discrete["divergences"].append(message)
                            reasons.append(message)
                    except (KeyError, IndexError, TypeError, ValueError):
                        discrete["action_exact"] = False
                        discrete["categorical_gate"] = False
                        reasons.append(f"{regime}@{horizon} frozen tape coordinate is invalid")
                for side, snapshot in (("A", lsnap[horizon]), ("B", rsnap[horizon])):
                    metadata = snapshot.get("observation_metadata")
                    raw_observation = snapshot.get("raw_observation")
                    if not isinstance(metadata, Mapping):
                        discrete["observation_exact"] = False
                        discrete["categorical_gate"] = False
                        reasons.append(f"{side} {regime}@{horizon} observation metadata is missing")
                    else:
                        try:
                            observed_metadata = _observation_metadata(raw_observation)
                        except ProtocolError as exc:
                            discrete["observation_exact"] = False
                            discrete["categorical_gate"] = False
                            reasons.append(f"{side} {regime}@{horizon} observation evidence is invalid: {exc}")
                        else:
                            if not _exact_equal(dict(metadata), observed_metadata):
                                discrete["observation_exact"] = False
                                discrete["categorical_gate"] = False
                                reasons.append(
                                    f"{side} {regime}@{horizon} observation metadata is inconsistent"
                                )
                left_metadata = lsnap[horizon].get("observation_metadata")
                right_metadata = rsnap[horizon].get("observation_metadata")
                if not _exact_equal(
                    _observation_structure(left_metadata),
                    _observation_structure(right_metadata),
                ):
                    discrete["observation_exact"] = False
                    discrete["categorical_gate"] = False
                    message = f"{regime}@{horizon} observation structure differs"
                    discrete["divergences"].append(message)
                    reasons.append(message)
                for side, window, snapshots, action in (
                    ("A", lwindow, lsnap, left_action),
                    ("B", rwindow, rsnap, right_action),
                ):
                    history = [snapshots[step] for step in sorted(snapshots)]
                    derived = _independent_evidence(
                        regime,
                        history[0],
                        snapshots[horizon],
                        action=action,
                        history=history,
                        strict=strict,
                    )
                    persisted = window.get("evidence")
                    if not isinstance(persisted, Mapping):
                        discrete["categorical_gate"] = False
                        reasons.append(f"{side} {regime}@{horizon} bilateral evidence is missing")
                    elif horizon == max(snapshots) and not _exact_equal(persisted, derived):
                        discrete["categorical_gate"] = False
                        reasons.append(f"{side} {regime}@{horizon} bilateral evidence is inconsistent")
                    # Use the full continuation-window proof for semantic
                    # regime gates.  ``derived`` remains horizon-local and is
                    # retained in the persisted evidence comparison above.
                    requirements = {
                        "contact": bool(left_window_requirements["contact"])
                        if side == "A"
                        else bool(right_window_requirements["contact"]),
                        "grasp": bool(left_window_requirements["grasp"])
                        if side == "A"
                        else bool(right_window_requirements["grasp"]),
                        "carried": bool(left_window_requirements["carried"])
                        if side == "A"
                        else bool(right_window_requirements["carried"]),
                    }
                    required = {
                        "contact": regime in {"contact", "grasp"},
                        "grasp": regime == "grasp",
                        "carried": regime == "carried",
                    }
                    for evidence_name, needed in required.items():
                        if needed and not requirements[evidence_name]:
                            discrete["categorical_gate"] = False
                            reasons.append(
                                f"{side} {regime}@{horizon} lacks required {evidence_name} evidence"
                            )
            lgrip = li.get("gripper", {}) if isinstance(li.get("gripper"), Mapping) else {}
            rgrip = ri.get("gripper", {}) if isinstance(ri.get("gripper"), Mapping) else {}
            if not _exact_equal(lgrip.get("current_action"), rgrip.get("current_action")):
                discrete["gripper_exact"] = False
                discrete["categorical_gate"] = False
                message = f"{regime}@{horizon} discrete gripper action differs"
                discrete["divergences"].append(message)
                reasons.append(message)
    left_terminal, right_terminal = _terminal_semantics(left), _terminal_semantics(right)
    if left.get("status") == "completed":
        try:
            _validate_terminal(left_terminal, expected_terminal, require_official_evidence=strict)
        except ProtocolError as exc:
            reasons.append(f"A terminal contract is invalid: {exc}")
    if right.get("status") == "completed":
        try:
            _validate_terminal(right_terminal, expected_terminal, require_official_evidence=strict)
        except ProtocolError as exc:
            reasons.append(f"B terminal contract is invalid: {exc}")
    for name in TERMINAL_FIELDS:
        if left.get("status") == "completed" and name not in left_terminal:
            reasons.append(f"A terminal {name} is missing")
        if right.get("status") == "completed" and name not in right_terminal:
            reasons.append(f"B terminal {name} is missing")
        if not _exact_equal(left_terminal.get(name), right_terminal.get(name)):
            discrete["terminal_timing_exact"] = False
            discrete["categorical_gate"] = False
            message = f"terminal {name} differs"
            discrete["divergences"].append(message)
            reasons.append(message)
        if expected_terminal is not None:
            if name not in left_terminal or not _exact_equal(left_terminal.get(name), expected_terminal.get(name)):
                reasons.append(f"A terminal {name} differs from frozen registry contract")
            if name not in right_terminal or not _exact_equal(right_terminal.get(name), expected_terminal.get(name)):
                reasons.append(f"B terminal {name} differs from frozen registry contract")
    return PairValidation(
        pair_id,
        trace_id,
        not reasons and bool(discrete["categorical_gate"]),
        tuple(dict.fromkeys(reasons)),
        {"A": left, "B": right},
        discrete,
    )


def _runtime_fingerprint(
    adapter: Any, kind: str, *, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    strict = bool(config and config.get("strict_runtime_contract"))
    # Strict calibration must fingerprint the concrete compiled model and the
    # configured renderer through the canonical state-replay implementation.
    # Adapter-provided convenience attributes are not an authoritative source
    # because they may be synthetic or stale.
    model = getattr(adapter, "model", None)
    if strict:
        if model is None:
            raise ProtocolError(f"strict {kind} model fingerprint requires the compiled model")
        try:
            from scripts import m1_state_replay

            if kind == "physics":
                value = m1_state_replay.physics_model_fingerprint(
                    model, getattr(adapter, "mujoco", None)
                )
            else:
                runtime = config.get("runtime") if isinstance(config, Mapping) else None
                renderer = runtime.get("renderer", {}) if isinstance(runtime, Mapping) else {}
                value = m1_state_replay.observation_model_fingerprint(
                    model,
                    renderer if isinstance(renderer, Mapping) else {},
                    getattr(adapter, "mujoco", None),
                )
        except Exception as exc:
            raise ProtocolError(f"could not capture strict {kind} model fingerprint: {exc}") from exc
        if not isinstance(value, Mapping) or not value:
            raise ProtocolError(f"strict {kind} model fingerprint is empty")
        required = ("kind", "hash", "fields")
        if any(name not in value for name in required) or not value.get("fields"):
            raise ProtocolError(f"strict {kind} model fingerprint lacks field evidence")
        if str(value.get("kind")) != kind:
            raise ProtocolError(f"strict {kind} model fingerprint kind is invalid")
        return copy.deepcopy(dict(value))

    names = (f"{kind}_model_fingerprint", f"{kind}_fingerprint")
    for owner in (adapter, getattr(adapter, "inner", None), getattr(adapter, "model", None)):
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            try:
                value = value() if callable(value) else value
            except Exception:
                value = None
            if isinstance(value, Mapping) and value:
                return copy.deepcopy(dict(value))
    # This compatibility value is used only by the small non-strict test
    # seam.  A strict run always takes the concrete model-fingerprint branch.
    return {"type": type(model).__name__ if model is not None else type(adapter).__name__}


_RUNTIME_PROBE = r"""
import importlib.metadata as metadata
import json
import platform
import sys

def version(*names):
    for name in names:
        try:
            return metadata.version(name)
        except Exception:
            pass
    return None

try:
    import mujoco
    mujoco_version = getattr(mujoco, "__version__", None) or version("mujoco")
except Exception:
    mujoco_version = version("mujoco")
try:
    import torch
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
except Exception:
    cuda_version = None
    cuda_available = False
print(json.dumps({
    "python": platform.python_version(),
    "numpy": version("numpy"),
    "pytorch": version("torch"),
    "transformers": version("transformers"),
    "gymnasium": version("gymnasium"),
    "lerobot": version("lerobot"),
    "robosuite": version("robosuite"),
    "hf_libero": version("hf-libero", "hf_libero"),
    "mujoco": mujoco_version,
    "cuda": str(cuda_version) if cuda_version else "none",
    "cuda_available": cuda_available,
    "gpu": "CPU" if not cuda_available and not cuda_version else "accelerator",
}, sort_keys=True))
"""


def _adapter_gl_identity(adapter: Any) -> dict[str, str] | None:
    for owner in (adapter, getattr(adapter, "inner", None), getattr(adapter, "env", None)):
        value = getattr(owner, "gl_identity", None) if owner is not None else None
        if isinstance(value, Mapping):
            result = {str(key): str(item) for key, item in value.items()}
            if all(result.get(key, "").strip() for key in ("vendor", "renderer", "version")):
                return result
    return None


def _runtime_identity_audit(
    config: Mapping[str, Any],
    *,
    adapter: Any | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the pinned interpreter/lock and capture live runtime facts."""

    if not bool(config.get("strict_runtime_contract")):
        return dict(previous or {})
    runtime_config = _official_runtime_config(config)
    paths = runtime_config.get("paths")
    if not isinstance(paths, Mapping):
        raise ProvenanceError("strict runtime has no pinned paths")
    python_value = config.get("python", paths.get("cpu_python"))
    if not isinstance(python_value, (str, Path)) or not str(python_value).strip():
        raise ProvenanceError("strict runtime has no configured pinned python")
    python_path = Path(str(python_value)).resolve()
    locked_python = Path(str(paths.get("cpu_python", ""))).resolve()
    if python_path != locked_python or not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ProvenanceError(
            f"configured python is not the frozen executable: {python_path} != {locked_python}"
        )
    lock_value = paths.get("runtime_lock")
    lock_sha = paths.get("runtime_lock_sha256")
    if not isinstance(lock_value, (str, Path)) or not isinstance(lock_sha, str):
        raise ProvenanceError("strict runtime lock path/SHA is required")
    lock_path = _resolve_path(lock_value)
    actual_lock_sha = sha256_file(lock_path)
    if actual_lock_sha.lower() != lock_sha.lower():
        raise ProvenanceError("runtime lock SHA drift")
    environment = runtime_config.get("runtime", {}).get("environment", {})
    child_env = os.environ.copy()
    if isinstance(environment, Mapping):
        child_env.update(
            {str(key): str(value) for key, value in environment.items() if str(key) != "empty_hf_cache"}
        )
    probe = subprocess.run(
        [str(python_path), "-c", _RUNTIME_PROBE],
        cwd=str(_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise ProvenanceError(
            f"configured python runtime probe failed ({probe.returncode}): {probe.stderr[-1000:]}"
        )
    try:
        facts = json.loads(str(probe.stdout).strip())
    except json.JSONDecodeError as exc:
        raise ProvenanceError("configured python runtime probe was not JSON") from exc
    if not isinstance(facts, Mapping):
        raise ProvenanceError("configured python runtime probe did not return a mapping")
    pins = runtime_config.get("pins")
    if not isinstance(pins, Mapping):
        raise ProvenanceError("strict runtime pins are required")
    mismatches = []
    for key in ("python", "numpy", "pytorch", "transformers", "gymnasium", "lerobot", "robosuite", "hf_libero", "mujoco", "cuda", "gpu"):
        if key not in pins:
            mismatches.append(f"missing pin {key}")
        elif str(facts.get(key)) != str(pins[key]):
            mismatches.append(f"{key}: expected {pins[key]!r}, got {facts.get(key)!r}")
    if mismatches:
        raise ProvenanceError("configured runtime facts differ from frozen pins: " + "; ".join(mismatches))
    identity: dict[str, Any] = {
        "python_executable": str(python_path),
        "runtime_lock": {"path": str(lock_path), "sha256": actual_lock_sha},
        "facts": dict(facts),
        "environment": {str(key): str(value) for key, value in environment.items()},
        "expected_gl": copy.deepcopy(
            runtime_config.get("runtime", {}).get("renderer", {}).get("expected_gl", {})
            if isinstance(runtime_config.get("runtime"), Mapping)
            and isinstance(runtime_config.get("runtime", {}).get("renderer"), Mapping)
            else {}
        ),
    }
    if adapter is not None:
        gl = _adapter_gl_identity(adapter)
        expected_gl = identity["expected_gl"]
        if not isinstance(expected_gl, Mapping) or not expected_gl:
            raise ProvenanceError("strict runtime expected GL identity is missing")
        if gl is None:
            raise ProvenanceError("strict runtime did not capture live GL identity")
        expected_gl = {str(key): str(value) for key, value in expected_gl.items()}
        if gl != expected_gl:
            raise ProvenanceError(f"live GL identity differs from frozen EGL identity: {gl!r} != {expected_gl!r}")
        identity["gl_identity"] = gl
    return identity


def _step_parts(result: Any) -> tuple[Any, bool, bool, Mapping[str, Any]]:
    observation: Any = None
    terminated = truncated = False
    info: Mapping[str, Any] = {}
    if isinstance(result, Mapping):
        observation = result.get("observation", result.get("obs", result.get("terminal_observation")))
        terminated = bool(result.get("terminated", result.get("done", False)))
        truncated = bool(result.get("truncated", False))
        if isinstance(result.get("info"), Mapping):
            info = result["info"]
        if "termination_reason" in result and "termination_reason" not in info:
            info = {**dict(info), "termination_reason": result["termination_reason"]}
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) >= 1:
            observation = result[0]
        if len(result) >= 5:
            terminated, truncated = bool(result[2]), bool(result[3])
            info = result[4] if isinstance(result[4], Mapping) else {}
        elif len(result) >= 4:
            terminated = bool(result[2])
            info = result[3] if isinstance(result[3], Mapping) else {}
    return observation, terminated, truncated, info


def _terminal_reason_evidence(
    adapter: Any,
    result: Any,
    info: Mapping[str, Any],
    *,
    terminated: bool,
    truncated: bool,
) -> tuple[Any, dict[str, Any]]:
    """Return the returned-step reason and the official source for it.

    A frozen terminal contract is not evidence that the environment actually
    returned that reason.  Keep the source explicit so strict validation can
    reject a worker that merely copied the configured contract.
    """

    candidates: list[tuple[str, str, Any]] = []
    if isinstance(result, Mapping):
        for field_name in ("termination_reason", "terminal_reason"):
            if result.get(field_name) is not None:
                candidates.append(("step_result", field_name, result[field_name]))
    for field_name in ("termination_reason", "terminal_reason"):
        if info.get(field_name) is not None:
            candidates.append(("step_info", field_name, info[field_name]))
    for owner in (adapter, getattr(adapter, "inner", None)):
        if owner is None:
            continue
        for field_name in (
            "last_termination_reason",
            "termination_reason",
            "last_terminal_reason",
            "terminal_reason",
        ):
            value = getattr(owner, field_name, None)
            if value is not None and not callable(value):
                candidates.append(("adapter", field_name, value))
    if not candidates:
        return None, {
            "source": "missing",
            "field": None,
            "raw_reason": None,
            "returned_step": True,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    source, field_name, reason = candidates[0]
    return reason, {
        "source": source,
        "field": field_name,
        "raw_reason": reason,
        "returned_step": True,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _derive_terminal_reason(
    raw_reason: Any,
    *,
    step: int,
    success: bool,
    previous_snapshot: Any,
    terminal_snapshot: Any,
    expected_terminal: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Separate raw wrapper termination from the frozen semantic terminal.

    The LeRobot/LIBERO step API does not expose a semantic reason.  In that
    case the only permitted ``predicate_transition`` claim is the conjunction
    of the frozen terminal step, successful completion, and an observed
    false-to-true goal-predicate transition in adjacent invariant frames.
    """

    expected_step = expected_terminal.get("step")
    expected_reason = expected_terminal.get("termination_reason")
    predicate_evidence = _predicate_transition_evidence(previous_snapshot, terminal_snapshot)
    frozen_step = (
        not isinstance(expected_step, bool)
        and isinstance(expected_step, (int, np.integer))
        and int(step) == int(expected_step)
    )
    success_observed = success is True
    derived = bool(
        expected_reason == "predicate_transition"
        and frozen_step
        and success_observed
        and predicate_evidence["predicate_transition"]
    )
    semantic_reason: str | None
    if derived:
        semantic_reason = "predicate_transition"
    elif expected_reason == "predicate_transition" and str(raw_reason) == "predicate_transition":
        # Do not accept a copied/configured semantic reason without its
        # independent terminal evidence.  ``None`` makes the normal terminal
        # validator fail closed and leaves the raw reason intact below.
        semantic_reason = None
    elif raw_reason is None:
        semantic_reason = None
    else:
        semantic_reason = str(raw_reason)
    return semantic_reason, {
        "source": "frozen_terminal_step_success_predicate_transition",
        "frozen_terminal_step": frozen_step,
        "success": success_observed,
        "predicate_transition": bool(predicate_evidence["predicate_transition"]),
        "predicate_evidence": predicate_evidence,
        "derived": derived,
        "raw_reason": raw_reason,
        "semantic_reason": semantic_reason,
    }


def _is_terminal(result: Any) -> bool:
    _, terminated, truncated, _ = _step_parts(result)
    return terminated or truncated


def _adapter_snapshot(adapter: Any) -> Mapping[str, Any]:
    collector = getattr(adapter, "collect_invariants", None)
    if not callable(collector):
        return {}
    value = collector()
    if not isinstance(value, Mapping):
        raise ProtocolError("adapter invariant collector did not return a mapping")
    return copy.deepcopy(dict(value))


def _observation_metadata(value: Any) -> dict[str, Any]:
    """Describe the complete observation tree recursively without pickle."""

    arrays: dict[str, dict[str, Any]] = {}

    def visit(node: Any, path: str) -> dict[str, Any]:
        if isinstance(node, Mapping):
            children: dict[str, Any] = {}
            for key, child in node.items():
                if type(key) is not str:
                    raise ProtocolError("official observation mapping keys must be strings")
                child_path = f"{path}.{key}" if path else key
                children[key] = visit(child, child_path)
            return {"kind": "mapping", "keys": list(children), "children": children}
        if isinstance(node, np.ndarray):
            if node.dtype.kind == "O":
                raise ProtocolError("official observation tree contains an object array")
            array = np.ascontiguousarray(node)
            if array.dtype.kind in {"f", "c"} and not np.all(np.isfinite(array)):
                raise ProtocolError("official observation tree contains non-finite values")
            metadata = {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": sha256_bytes(array.tobytes(order="C")),
            }
            arrays[path] = metadata
            return {"kind": "array", **metadata}
        if isinstance(node, (list, tuple)):
            children = [visit(child, f"{path}[{index}]") for index, child in enumerate(node)]
            return {
                "kind": "tuple" if isinstance(node, tuple) else "list",
                "length": len(children),
                "children": children,
            }
        if isinstance(node, (bool, np.bool_)):
            return {"kind": "scalar", "dtype": "bool"}
        if isinstance(node, (int, np.integer)):
            return {"kind": "scalar", "dtype": "int"}
        if isinstance(node, (float, np.floating)):
            if not math.isfinite(float(node)):
                raise ProtocolError("official observation scalar is non-finite")
            return {"kind": "scalar", "dtype": "float"}
        if node is None or isinstance(node, str):
            return {"kind": "scalar", "dtype": "none" if node is None else "str"}
        raise ProtocolError(f"unsupported official observation node: {type(node).__name__}")

    if value is None:
        raise ProtocolError("official observation is missing")
    return {"schema_version": 1, "tree": visit(value, ""), "arrays": arrays}


def _observation_structure(value: Any) -> Any:
    """Drop numeric content hashes while retaining official tree structure."""

    if isinstance(value, Mapping):
        return {
            str(key): _observation_structure(child)
            for key, child in value.items()
            if key != "sha256"
        }
    if isinstance(value, list):
        return [_observation_structure(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_observation_structure(child) for child in value)
    return value


def _adapter_rgb(adapter: Any) -> np.ndarray | None:
    renderer = getattr(adapter, "render_rgb", None)
    if not callable(renderer):
        return None
    value = renderer()
    if value is None:
        return None
    image = np.asarray(value)
    if image.dtype != np.dtype("uint8") or image.ndim < 2 or image.size == 0:
        raise ProtocolError("direct renderer output must be a non-empty uint8 array")
    return np.ascontiguousarray(image).copy()


def _terminal_success(adapter: Any, info: Mapping[str, Any]) -> bool:
    for owner in (adapter, getattr(adapter, "inner", None)):
        checker = getattr(owner, "_check_success", None) if owner is not None else None
        if callable(checker):
            try:
                value = checker()
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
            except Exception:
                pass
    value = info.get("is_success", info.get("success", False))
    return bool(value) if isinstance(value, (bool, np.bool_)) else False


def _adapter_counter(adapter: Any, field_name: str, *aliases: str) -> tuple[Any, str]:
    for owner in (adapter, getattr(adapter, "inner", None), getattr(adapter, "env", None)):
        if owner is None:
            continue
        for name in (field_name, *aliases):
            value = getattr(owner, name, None)
            if value is not None and not callable(value):
                return value, f"{type(owner).__name__}.{name}"
    counters = getattr(adapter, "protocol_counters", None)
    if isinstance(counters, Mapping):
        for name in (field_name, *aliases):
            if name in counters:
                return counters[name], f"protocol_counters.{name}"
    return None, "unavailable"


def _protocol_counters(adapter: Any, observed: Mapping[str, int], terminal_step: int) -> dict[str, Any]:
    """Publish counters from the adapter or directly measured protocol calls."""

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    provenance_getter = getattr(adapter, "reset_provenance_snapshot", None)
    if callable(provenance_getter):
        reset_provenance = provenance_getter()
    else:
        reset_provenance = getattr(adapter, "reset_provenance", None)
    if not isinstance(reset_provenance, Mapping):
        reset_provenance = None
    construction = reset_provenance.get("construction", {}) if reset_provenance else {}
    construction_operations = construction.get("operations", {}) if isinstance(construction, Mapping) else {}
    post_forbidden = (
        construction.get("post_construction_forbidden", {})
        if isinstance(construction, Mapping)
        else {}
    )
    if not isinstance(construction_operations, Mapping):
        construction_operations = {}
    if not isinstance(post_forbidden, Mapping):
        post_forbidden = {}

    def provenance_count(operation: str, *, construction_phase: bool = False) -> tuple[Any, str] | None:
        section = construction_operations if construction_phase else post_forbidden
        entry = section.get(operation) if isinstance(section, Mapping) else None
        if not isinstance(entry, Mapping) or "count" not in entry:
            return None
        count = entry.get("count")
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 0:
            raise ProtocolError(f"reset provenance count is invalid: {operation}")
        phase = "construction" if construction_phase else "post_construction_forbidden"
        return int(count), f"RuntimeAdapter.reset_provenance.{phase}.{operation}.count"

    fields = {
        "construction_reset_count": ("construction_reset_count",),
        "reset_count": ("reset_count",),
        "restore_count": ("restore_count",),
        "capture_count": ("capture_count",),
        "policy_calls": ("policy_calls",),
        "processor_calls": ("processor_calls", "processors_calls"),
        "processors_calls": ("processors_calls", "processor_calls"),
        "retry_count": ("retry_count",),
        "post_terminal_steps": ("post_terminal_steps",),
        "set_init_state_count": ("set_init_state_count",),
        "settle_count": ("settle_count",),
        "dummy_action_count": ("dummy_action_count",),
        "autoreset_count": ("autoreset_count",),
    }
    for field_name, aliases in fields.items():
        provenance = None
        if field_name == "construction_reset_count":
            provenance = provenance_count("reset", construction_phase=True)
        else:
            operation_name = {
                "reset_count": "reset",
                "set_init_state_count": "set_init_state",
                "settle_count": "settle",
                "restore_count": "restore",
                "capture_count": "capture",
                "policy_calls": "policy_calls",
                "processor_calls": "processor_calls",
                "processors_calls": "processor_calls",
                "retry_count": "retry",
                "dummy_action_count": "dummy_action",
                "autoreset_count": "autoreset",
                "post_terminal_steps": "post_terminal_step",
            }.get(field_name)
            if operation_name is not None:
                provenance = provenance_count(operation_name)
        if provenance is not None:
            value, source = provenance
        else:
            value, source = _adapter_counter(adapter, field_name, *aliases)
        if value is None:
            if field_name == "construction_reset_count":
                value = int(bool(getattr(adapter, "_construction_reset_done", False)))
                source = "RuntimeAdapter._construction_reset_done"
            elif field_name == "post_terminal_steps":
                value = int(observed.get(field_name, 0))
                source = "measured_protocol_calls"
            else:
                # These operations have no call site in this protocol.  Keep
                # the measured zero explicit and identify its source rather
                # than fabricating a runtime counter value.
                value = 0
                source = "null_protocol_call_site_audit"
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"protocol counter {field_name} is not an integer") from exc
        if value < 0:
            raise ProtocolError(f"protocol counter {field_name} is negative")
        values[field_name] = value
        sources[field_name] = source
    values["reset_provenance"] = copy.deepcopy(dict(reset_provenance)) if reset_provenance else None
    values["post_construction_forbidden"] = copy.deepcopy(dict(post_forbidden)) if post_forbidden else {
        name: {"allowed": False, "observed": True, "count": 0, "source": "unavailable"}
        for name in POST_CONSTRUCTION_FORBIDDEN_OPERATIONS
    }
    values["actions_executed"] = int(observed.get("actions_executed", terminal_step))
    values["step_calls"] = int(observed.get("step_calls", values["actions_executed"]))
    values["render_calls"] = int(observed.get("render_calls", 0))
    values["invariant_collections"] = int(observed.get("invariant_collections", 0))
    values["counter_sources"] = sources
    return values


def _attempt_failure(attempt: Mapping[str, Any], error: Exception | str) -> dict[str, Any]:
    try:
        identity = _proc_start_identity(os.getpid())
    except Exception:
        identity = None
    protocol = {"construction_reset_count": 0}
    protocol.update({field_name: 0 for field_name in FORBIDDEN_PROTOCOL_FIELDS})
    protocol.update(
        {
            "actions_executed": 0,
            "step_calls": 0,
            "render_calls": 0,
            "invariant_collections": 0,
            "counter_sources": {
                field_name: "failed_before_protocol_observation"
                for field_name in FORBIDDEN_PROTOCOL_FIELDS
            },
        }
    )
    return {
        "attempt_id": attempt.get("attempt_id"),
        "pair_id": attempt.get("pair_id"),
        "side": attempt.get("side"),
        "status": "failed",
        "error": f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "process_start_identity": identity,
        "protocol": protocol,
    }


def execute_attempt(
    attempt: Mapping[str, Any],
    *,
    adapter_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Execute one tape exactly once from one freshly constructed adapter."""

    adapter: Any = None
    result: dict[str, Any] | None = None
    actions: np.ndarray
    try:
        config = attempt.get("config") if isinstance(attempt.get("config"), Mapping) else {}
        strict = bool(config.get("strict_runtime_contract"))
        expected_attempt_id = f"{attempt.get('pair_id')}-{attempt.get('side')}"
        if attempt.get("attempt_id") != expected_attempt_id:
            raise ProtocolError("worker attempt ID does not match frozen pair/side")
        tape_value = attempt.get("tape")
        if tape_value is None:
            tape_path = attempt.get("tape_path")
            if tape_path is None:
                raise ProtocolError("worker attempt has no persisted tape")
            tape_value = np.load(_resolve_path(tape_path), allow_pickle=False)
        actions = np.asarray(tape_value)
        if actions.dtype != ACTION_DTYPE or tuple(actions.shape) != ACTION_SHAPE or not np.all(np.isfinite(actions)):
            raise ProtocolError("worker tape is not exact finite float32 (82, 7)")
        actions = np.ascontiguousarray(actions)
        actual_tape_sha = sha256_bytes(actions.tobytes(order="C"))
        expected_tape_sha = attempt.get("action_tape_sha256")
        if expected_tape_sha is None and isinstance(config.get("action_tape"), Mapping):
            expected_tape_sha = config["action_tape"].get("sha256")
        if expected_tape_sha is not None and str(expected_tape_sha).lower() != actual_tape_sha.lower():
            raise ProtocolError("worker action tape hash differs from frozen registry")
        if strict:
            _strict_task(config.get("task"), field_name="task")
            _runtime_identity_audit(config)
        adapter = adapter_factory(config) if adapter_factory is not None else _construct_adapter(config)
        windows = _window_map_from_registry(attempt)
        expected_terminal = attempt.get("terminal_contract")
        if expected_terminal is None and isinstance(config.get("terminal_contract"), Mapping):
            expected_terminal = config["terminal_contract"]
        if not isinstance(expected_terminal, Mapping):
            expected_terminal = DEFAULT_TERMINAL_CONTRACT
        timeline: dict[int, dict[str, Any]] = {}
        terminal: dict[str, Any] | None = None
        observed: dict[str, int] = {
            "actions_executed": 0,
            "step_calls": 0,
            "render_calls": 0,
            "invariant_collections": 0,
            "post_terminal_steps": 0,
        }
        for index, action in enumerate(actions):
            if terminal is not None:
                raise ProtocolError("post-terminal action was submitted")
            result = getattr(adapter, "step", None)
            if not callable(result):
                raise ProtocolError("fresh adapter does not expose step()")
            step_result = result(np.asarray(action, dtype=ACTION_DTYPE).copy())
            observed["step_calls"] += 1
            observed["actions_executed"] += 1
            step = index + 1
            observation, terminated, truncated, info = _step_parts(step_result)
            invariant = _adapter_snapshot(adapter)
            observed["invariant_collections"] += 1
            if strict:
                _validate_invariant_snapshot(invariant, required_roots=_configured_invariant_roots(config))
            observation_metadata = _observation_metadata(observation)
            if strict and not observation_metadata["arrays"]:
                raise ProtocolError("strict official observation tree has no array leaves")
            renderer = getattr(adapter, "render_rgb", None)
            if callable(renderer):
                observed["render_calls"] += 1
            rgb = _adapter_rgb(adapter)
            timeline[step] = {
                "invariants": invariant,
                "raw_observation": observation,
                "observation_metadata": observation_metadata,
                "renderer": rgb,
                "action": np.asarray(action, dtype=ACTION_DTYPE).copy(),
            }
            if terminated or truncated:
                reason, official_evidence = _terminal_reason_evidence(
                    adapter,
                    step_result,
                    info,
                    terminated=terminated,
                    truncated=truncated,
                )
                success = _terminal_success(adapter, info)
                previous_snapshot = timeline.get(step - 1)
                semantic_reason, semantic_evidence = _derive_terminal_reason(
                    reason,
                    step=step,
                    success=success,
                    previous_snapshot=previous_snapshot,
                    terminal_snapshot=timeline[step],
                    expected_terminal=expected_terminal,
                )
                official_evidence = {
                    **official_evidence,
                    "raw_reason": reason,
                    "semantic_reason": semantic_reason,
                    "semantic_derivation": semantic_evidence,
                }
                terminal = {
                    "step": step,
                    "termination_reason": semantic_reason,
                    "raw_termination_reason": reason,
                    "raw_termination_evidence": copy.deepcopy(official_evidence),
                    "success": success,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "official_evidence": official_evidence,
                    "terminal_observation": observation,
                }
                if str(reason) == "environment_termination":
                    # Preserve the M0 wrapper-level evidence separately from
                    # the semantic predicate-transition classification.
                    terminal["environment_termination_evidence"] = {
                        "reason": "environment_termination",
                        "source": official_evidence.get("source"),
                        "field": official_evidence.get("field"),
                        "returned_step": official_evidence.get("returned_step") is True,
                    }
                break
        if terminal is None:
            raise ProtocolError("frozen trace did not terminate legally at step 82")
        if int(terminal["step"]) != ACTION_SHAPE[0]:
            raise ProtocolError(f"frozen trace terminated at unexpected step {terminal['step']}")
        _validate_terminal(
            terminal,
            expected_terminal if isinstance(expected_terminal, Mapping) else None,
            require_official_evidence=strict,
        )
        output_windows: dict[str, Any] = {}
        for window in windows:
            offset = int(window["capture_offset"])
            horizon = int(window["continuation_horizon"])
            snapshots: dict[str, Any] = {}
            for relative in range(horizon + 1):
                absolute = offset + relative
                if absolute not in timeline:
                    raise ProtocolError(f"missing frozen window coordinate {window['window_id']}@{relative}")
                snapshots[str(relative)] = timeline[absolute]
            snapshot_history = [snapshots[str(relative)] for relative in range(horizon + 1)]
            window_evidence = _independent_evidence(
                str(window["window_id"]),
                snapshot_history[0],
                snapshot_history[-1],
                action=_snapshot_action(snapshot_history[-1]),
                history=snapshot_history,
                strict=strict,
            )
            output_windows[str(window["window_id"])] = {
                "window_id": str(window["window_id"]),
                "regimes": list(window["regimes"]),
                "capture_offset": offset,
                "continuation_horizon": horizon,
                "snapshots": snapshots,
                "action_sha256": actual_tape_sha,
                "evidence": window_evidence,
            }
        runtime_identity = _runtime_identity_audit(config, adapter=adapter) if strict else {}
        process_identity = _proc_start_identity(os.getpid())
        protocol = _protocol_counters(adapter, observed, int(terminal["step"]))
        provenance_getter = getattr(adapter, "reset_provenance_snapshot", None)
        reset_provenance = (
            provenance_getter()
            if callable(provenance_getter)
            else copy.deepcopy(getattr(adapter, "reset_provenance", None))
        )
        result = {
            "attempt_id": attempt.get("attempt_id"),
            "pair_id": attempt.get("pair_id"),
            "side": attempt.get("side"),
            "status": "completed",
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "process_start_identity": process_identity,
            "process_start_identity_source": "/proc/<pid>/stat:starttime_ticks+boot_id",
            "frozen_inputs": {
                "trace_id": attempt.get("trace_id"),
                "task": dict(config.get("task", TASK)) if isinstance(config.get("task", TASK), Mapping) else dict(TASK),
                "action_tape_sha256": actual_tape_sha,
                "action_shape": list(ACTION_SHAPE),
                "action_dtype": "float32",
                "physics_model_fingerprint": _runtime_fingerprint(adapter, "physics", config=config),
                "observation_model_fingerprint": _runtime_fingerprint(adapter, "observation", config=config),
            },
            "windows": output_windows,
            "terminal": terminal,
            "protocol": protocol,
            "reset_provenance": reset_provenance,
            "runtime": {
                "python": sys.version.split()[0],
                "python_executable": sys.executable,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "process_start_identity": process_identity,
                "identity_source": "/proc/<pid>/stat:starttime_ticks+boot_id",
                "identity": runtime_identity,
            },
        }
        if strict:
            expected_runtime = attempt.get("runtime_identity_contract")
            if isinstance(expected_runtime, Mapping):
                expected_facts = expected_runtime.get("facts")
                actual_facts = runtime_identity.get("facts")
                if not _exact_equal(expected_facts, actual_facts):
                    raise ProtocolError("worker runtime facts differ from frozen runtime contract")
        result["output_sha256"] = payload_sha256(result)
    except Exception as exc:
        result = _attempt_failure(attempt, exc)
    finally:
        close_evidence: dict[str, Any]
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    close_evidence = {
                        "attempted": True,
                        "success": False,
                        "source": "adapter.close",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                else:
                    close_evidence = {
                        "attempted": True,
                        "success": True,
                        "source": "adapter.close",
                    }
            else:
                close_evidence = {
                    "attempted": False,
                    "success": False,
                    "source": "adapter.close",
                    "error": "fresh adapter does not expose close()",
                }
        else:
            close_evidence = {
                "attempted": False,
                "success": False,
                "source": "adapter.close",
                "error": "adapter was not constructed",
            }
        if result is None:
            result = _attempt_failure(attempt, "worker produced no attempt result")
        result["close_evidence"] = close_evidence
        if not close_evidence["success"] and result.get("status") == "completed":
            result["status"] = "failed"
            result["error"] = f"close cleanup failed: {close_evidence.get('error', 'unknown error')}"
        elif not close_evidence["success"] and close_evidence.get("error"):
            result["error"] = (
                f"{result.get('error', 'attempt failed')}; "
                f"close cleanup failed: {close_evidence['error']}"
            )
        # Cleanup evidence is part of the immutable worker payload.  Recompute
        # the payload hash after the close outcome is known; a close exception
        # therefore cannot leave a falsely completed/hash-valid attempt.
        result.pop("output_sha256", None)
        try:
            result["output_sha256"] = payload_sha256(result)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"attempt payload hashing failed: {type(exc).__name__}: {exc}"
            result.pop("output_sha256", None)
    return result


def _window_map_from_registry(attempt: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = attempt.get("registry")
    if not isinstance(registry, Mapping):
        raise ProvenanceError("worker attempt has no frozen registry")
    trace_id = str(attempt.get("trace_id", ""))
    traces = registry.get("traces")
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
        raise ProvenanceError("worker registry has no traces")
    trace = next((item for item in traces if isinstance(item, Mapping) and str(item.get("trace_id")) == trace_id), None)
    if trace is None:
        trace = traces[0] if traces and isinstance(traces[0], Mapping) else None
    if not isinstance(trace, Mapping):
        raise ProvenanceError("worker trace is missing")
    return _windows(registry, trace, require_all=False)


def _construct_adapter(config: Mapping[str, Any]) -> Any:
    """Construct one official policy-free adapter at the worker boundary."""

    from scripts import m1_state_replay

    construction_config = _official_runtime_config(config)
    runtime = m1_state_replay.build_official_runtime(construction_config)
    if callable(getattr(runtime, "step", None)):
        return runtime
    return m1_state_replay.RuntimeAdapter.construct_fresh(
        construction_config, runtime_builder=lambda _config: runtime
    )


def _prepare_attempt(prepared: PreparedRun, pair: Mapping[str, Any], side: str) -> dict[str, Any]:
    record = next(item for item in pair["attempts"] if item["side"] == side)
    return {
        **dict(record),
        "config": copy.deepcopy(prepared.config),
        "config_path": str(prepared.config_path),
        "registry": copy.deepcopy(prepared.source_registry),
        "registry_path": str(prepared.source_registry_path),
        "trace_id": pair["trace_id"],
        "tape_path": str(_tape_path(prepared.source_registry, pair["trace_id"])),
        "action_tape_sha256": prepared.run_spec["action_tape"]["sha256"],
        "terminal_contract": copy.deepcopy(prepared.run_spec.get("terminal_contract", {})),
        "runtime_identity_contract": copy.deepcopy(prepared.run_spec.get("runtime_identity_contract", {})),
        "expected_registry_sha256": prepared.run_spec["registry_sha256"],
        "expected_run_spec_sha256": prepared.run_spec["run_spec_sha256"],
        "expected_config_sha256": prepared.run_spec["config_sha256"],
        "run_spec_path": str(prepared.run_spec_path),
        "pair_registry_path": str(prepared.pair_registry_path),
        "pair_registry_sha256": prepared.pair_registry["pair_registry_sha256"],
        "python": prepared.run_spec["python"],
        "worker_timeout_seconds": prepared.run_spec["worker_timeout_seconds"],
        "output_root": prepared.run_spec["output_root"],
        "parent_pid": os.getpid(),
    }


def _tape_path(registry: Mapping[str, Any], trace_id: str) -> Path:
    provenance = _tape_provenance(registry, trace_id)
    value = provenance.get("path", provenance.get("tape_path"))
    if value is None:
        raise ProvenanceError("frozen tape path is missing")
    return _resolve_path(value)


def _same_resolved_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (TypeError, ValueError, OSError):
        return False


def _verify_worker_binding(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any]]:
    """Verify every persisted worker contract before environment construction."""

    config_path_value = job.get("config_path")
    if not isinstance(config_path_value, (str, Path)) or not str(config_path_value).strip():
        raise ProvenanceError("worker job has no config path")
    config_path = _resolve_path(config_path_value).resolve()
    config = load_config(config_path)
    expected_config_sha = job.get("expected_config_sha256", job.get("config_sha256"))
    if not isinstance(expected_config_sha, str) or len(expected_config_sha) != 64:
        raise ProvenanceError("worker job has no expected config contract hash")
    actual_config_sha = config_contract_sha256(config)
    if actual_config_sha.lower() != expected_config_sha.lower():
        raise ProvenanceError(
            f"worker config contract hash drift: {actual_config_sha} != {expected_config_sha}"
        )
    declared_config_sha = config.get("config_sha256")
    if declared_config_sha is not None and str(declared_config_sha).lower() != actual_config_sha.lower():
        raise ProvenanceError("worker config self-hash is invalid")

    run_spec_path_value = job.get("run_spec_path")
    if not isinstance(run_spec_path_value, (str, Path)) or not str(run_spec_path_value).strip():
        raise ProvenanceError("worker job has no run specification path")
    run_spec_path = _resolve_path(run_spec_path_value).resolve()
    run_spec = _load_document(run_spec_path)
    if not verify_self_hash(run_spec, "run_spec_sha256"):
        raise ProvenanceError("worker run specification self-hash is invalid")
    expected_run_spec_sha = job.get("expected_run_spec_sha256")
    if not isinstance(expected_run_spec_sha, str) or run_spec["run_spec_sha256"].lower() != expected_run_spec_sha.lower():
        raise ProvenanceError("worker run specification hash differs from the scheduled contract")
    if not _same_resolved_path(run_spec.get("config_path"), config_path):
        raise ProvenanceError("worker config path is not bound to the run specification")
    if run_spec.get("config_sha256", "").lower() != actual_config_sha.lower():
        raise ProvenanceError("worker run specification config hash differs from the loaded config")

    registry_path_value = job.get("registry_path")
    if not isinstance(registry_path_value, (str, Path)) or not str(registry_path_value).strip():
        raise ProvenanceError("worker job has no source registry path")
    registry_path = _resolve_path(registry_path_value).resolve()
    expected_registry_sha = job.get("expected_registry_sha256")
    if not isinstance(expected_registry_sha, str) or len(expected_registry_sha) != 64:
        raise ProvenanceError("worker job has no expected source registry hash")
    registry = _load_verified_registry(registry_path)
    _verify_registry_payload(registry, registry_path, expected_registry_sha)
    if str(run_spec.get("registry_sha256", "")).lower() != expected_registry_sha.lower():
        raise ProvenanceError("worker run specification source registry hash differs")
    if not _same_resolved_path(run_spec.get("source_registry_path"), registry_path):
        raise ProvenanceError("worker source registry path is not bound to the run specification")

    pair_registry_path_value = job.get("pair_registry_path")
    if not isinstance(pair_registry_path_value, (str, Path)) or not str(pair_registry_path_value).strip():
        raise ProvenanceError("worker job has no pair registry path")
    pair_registry_path = _resolve_path(pair_registry_path_value).resolve()
    pair_registry = _load_document(pair_registry_path)
    if not verify_self_hash(pair_registry, "pair_registry_sha256"):
        raise ProvenanceError("worker pair registry self-hash is invalid")
    expected_pair_registry_sha = job.get("pair_registry_sha256")
    if not isinstance(expected_pair_registry_sha, str) or pair_registry["pair_registry_sha256"].lower() != expected_pair_registry_sha.lower():
        raise ProvenanceError("worker pair registry hash differs from the scheduled contract")
    if not _same_resolved_path(run_spec.get("pair_registry_path"), pair_registry_path):
        raise ProvenanceError("worker pair registry path is not bound to the run specification")
    if not _same_resolved_path(pair_registry.get("run_spec_path"), run_spec_path):
        raise ProvenanceError("worker pair registry run-spec path is not bound")
    if not _same_resolved_path(pair_registry.get("source_registry_path"), registry_path):
        raise ProvenanceError("worker pair registry source path is not bound")
    _validate_frozen_schedule(
        run_spec,
        pair_registry,
        source_registry=registry,
        config=config,
    )

    pair_id = str(job.get("pair_id", ""))
    side = str(job.get("side", ""))
    if pair_id not in EXPECTED_PAIR_IDS or side not in {"A", "B"}:
        raise ProvenanceError("worker job pair/side is outside the frozen schedule")
    pairs = pair_registry.get("pairs")
    scheduled_pair = next(
        (item for item in pairs if isinstance(item, Mapping) and item.get("pair_id") == pair_id),
        None,
    ) if isinstance(pairs, Sequence) else None
    if not isinstance(scheduled_pair, Mapping):
        raise ProvenanceError("worker pair is not in the frozen pair registry")
    scheduled_attempt = next(
        (
            item
            for item in scheduled_pair.get("attempts", ())
            if isinstance(item, Mapping) and item.get("side") == side
        ),
        None,
    )
    if not isinstance(scheduled_attempt, Mapping):
        raise ProvenanceError("worker attempt side is not in the frozen pair registry")
    for field_name in ("attempt_id", "pair_id", "side", "ordinal", "status", "trace_id"):
        if field_name in scheduled_attempt and not _exact_equal(
            job.get(field_name), scheduled_attempt.get(field_name)
        ):
            raise ProvenanceError(f"worker schedule binding differs for {field_name}")
    if str(job.get("trace_id")) != str(run_spec.get("trace_id")):
        raise ProvenanceError("worker trace ID differs from the run specification")
    if str(job.get("action_tape_sha256", "")).lower() != str(run_spec["action_tape"]["sha256"]).lower():
        raise ProvenanceError("worker action tape hash differs from the run specification")
    try:
        job_timeout = float(job.get("worker_timeout_seconds"))
        frozen_timeout = float(run_spec.get("worker_timeout_seconds"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProvenanceError("worker timeout is missing or invalid") from exc
    if (
        isinstance(job.get("worker_timeout_seconds"), bool)
        or not math.isfinite(job_timeout)
        or job_timeout <= 0
        or not math.isfinite(frozen_timeout)
        or frozen_timeout <= 0
        or job_timeout != frozen_timeout
    ):
        raise ProvenanceError("worker timeout differs from the run specification")

    trace_id = str(run_spec["trace_id"])
    tape, tape_sha = _load_tape(registry, config=config, trace_id=trace_id)
    action_contract = run_spec.get("action_tape")
    if not isinstance(action_contract, Mapping) or str(action_contract.get("sha256", "")).lower() != tape_sha.lower():
        raise ProvenanceError("worker persisted tape differs from the run specification")
    if list(action_contract.get("shape", ())) != list(ACTION_SHAPE) or str(
        action_contract.get("dtype", "")
    ) not in {"float32", "<f4", "|f4"}:
        raise ProvenanceError("worker run-spec tape shape/dtype is not frozen")
    expected_tape_path = _tape_path(registry, trace_id)
    if job.get("tape_path") is not None and not _same_resolved_path(job.get("tape_path"), expected_tape_path):
        raise ProvenanceError("worker tape path is not bound to the source registry")
    return config, registry, tape, run_spec, pair_registry


def _verify_prepared(prepared: PreparedRun) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    current = load_config(prepared.config_path)
    if not verify_self_hash(prepared.run_spec, "run_spec_sha256"):
        raise ProvenanceError("run specification self-hash is invalid")
    computed = config_contract_sha256(current)
    if computed != str(prepared.run_spec.get("config_sha256", "")):
        raise ProvenanceError("config hash drift detected before worker launch")
    declared = current.get("config_sha256")
    if declared is not None and str(declared).lower() != computed.lower():
        raise ProvenanceError("config self-hash is invalid")
    runtime_config = _official_runtime_config(current)
    _validate_strict_null_config(current, runtime_config)
    if bool(current.get("strict_runtime_contract")):
        _strict_task(current.get("task"), field_name="task")
    registry_path = Path(prepared.run_spec["source_registry_path"])
    registry = _load_verified_registry(registry_path)
    _verify_registry_payload(registry, registry_path, str(prepared.run_spec["registry_sha256"]))
    pair_registry = _load_document(prepared.pair_registry_path)
    _validate_frozen_schedule(
        prepared.run_spec,
        pair_registry,
        source_registry=registry,
        config=current,
    )
    trace = _trace_record(registry)
    trace_id = str(trace.get("trace_id"))
    tape, _ = _load_tape(registry, config=current, trace_id=trace_id)
    if bool(current.get("strict_runtime_contract")):
        _validate_trace_frozen_inputs(trace, sha256_bytes(tape.tobytes(order="C")))
        expected_terminal = prepared.run_spec.get("terminal_contract")
        if not isinstance(expected_terminal, Mapping):
            raise ProvenanceError("run specification has no terminal contract")
        _registry_terminal_contract(registry, trace, current)
        if dict(expected_terminal) != _terminal_contract_from_config(current):
            raise ProvenanceError("run specification terminal contract drifted")
    return current, pair_registry, tape


def prepare_run(*, config_path: str | Path) -> PreparedRun:
    """Freeze source provenance and the complete 20-pair schedule only."""

    config_target = Path(config_path).resolve()
    config = load_config(config_target)
    computed_config_sha = config_contract_sha256(config)
    declared_config_sha = config.get("config_sha256")
    if declared_config_sha is not None and str(declared_config_sha).lower() != computed_config_sha.lower():
        raise ProvenanceError("config self-hash is invalid")
    runtime_config = _official_runtime_config(config)
    if int(config.get("pair_count", PAIR_COUNT)) != PAIR_COUNT:
        raise ProvenanceError("pair_count is frozen at exactly 20")
    prefix = str(config.get("pair_prefix", PAIR_PREFIX))
    if prefix != PAIR_PREFIX:
        raise ProvenanceError("pair_prefix is not frozen")
    registry_cfg = config.get("registry")
    if isinstance(registry_cfg, Mapping):
        registry_value = registry_cfg.get("path")
        expected_registry_sha = registry_cfg.get("sha256")
    else:
        registry_value = registry_cfg
        expected_registry_sha = config.get("registry_sha256")
    if registry_value is None:
        raise ProvenanceError("config registry path is missing")
    source_registry_path = _resolve_path(registry_value)
    source_registry = _load_verified_registry(source_registry_path)
    _verify_registry_payload(source_registry, source_registry_path, str(expected_registry_sha) if expected_registry_sha else None)
    trace = _trace_record(source_registry)
    trace_id = str(trace.get("trace_id"))
    tape, tape_sha = _load_tape(source_registry, config=config, trace_id=trace_id)
    windows = _windows(source_registry, trace)
    _validate_strict_null_config(config, runtime_config)
    if bool(config.get("strict_runtime_contract")):
        _validate_trace_frozen_inputs(trace, tape_sha)
        terminal_contract = _registry_terminal_contract(source_registry, trace, config)
    else:
        terminal_contract = _terminal_contract_from_config(config)
    output_value = config.get("output_root", "runs/m1_null_calibration/20260901_task000_init000_null20")
    output_root = _resolve_path(output_value)
    run_spec_path = output_root / "run_spec.json"
    pair_registry_path = output_root / "pair_registry.json"
    attempts = [
        {
            "attempt_id": f"{PAIR_PREFIX}{index:03d}-{side}",
            "pair_id": f"{PAIR_PREFIX}{index:03d}",
            "side": side,
            "ordinal": index * 2 + (0 if side == "A" else 1),
            "status": "scheduled",
            "trace_id": trace_id,
        }
        for index in range(PAIR_COUNT)
        for side in ("A", "B")
    ]
    pairs = [
        {
            "pair_id": f"{PAIR_PREFIX}{index:03d}",
            "trace_id": trace_id,
            "attempts": [attempt for attempt in attempts if attempt["pair_id"] == f"{PAIR_PREFIX}{index:03d}"],
        }
        for index in range(PAIR_COUNT)
    ]
    run_spec_body = {
        "schema_version": SCHEMA_VERSION,
        "run_type": "m1_null_calibration",
        "config_path": str(config_target),
        "config_sha256": computed_config_sha,
        "source_registry_path": str(source_registry_path),
        "registry_sha256": str(source_registry["registry_sha256"]),
        "output_root": str(output_root),
        "pair_count": PAIR_COUNT,
        "pair_prefix": PAIR_PREFIX,
        "trace_id": trace_id,
        "task": dict(config.get("task", TASK)) if isinstance(config.get("task", TASK), Mapping) else dict(TASK),
        "action_tape": {"sha256": tape_sha, "shape": list(ACTION_SHAPE), "dtype": "float32"},
        "windows": windows,
        "pairs": copy.deepcopy(pairs),
        "terminal_contract": terminal_contract,
        "python": str(config.get("python", sys.executable)),
        "worker_timeout_seconds": float(
            config.get("worker_timeout_seconds", DEFAULT_WORKER_TIMEOUT_SECONDS)
        ),
        "runtime": {
            "include_policy": False,
            "call_policy": False,
            "call_processors": False,
            "fresh_processes": True,
            "retry_count": 0,
        },
        "obs_type": config.get("obs_type"),
        "runtime_config": copy.deepcopy(runtime_config.get("runtime", {})),
        "state_replay_config": copy.deepcopy(config.get("state_replay_config")),
        "pair_registry_path": str(pair_registry_path),
    }
    run_spec = {**run_spec_body, "run_spec_sha256": sha256_bytes(canonical_json(run_spec_body).encode("utf-8"))}
    write_json_atomic(run_spec_path, run_spec)
    pair_registry_body = {
        "schema_version": SCHEMA_VERSION,
        "registry_type": "m1_null_calibration_pair_registry",
        "run_spec_path": str(run_spec_path),
        "run_spec_sha256": run_spec["run_spec_sha256"],
        "source_registry_path": str(source_registry_path),
        "registry_sha256": str(source_registry["registry_sha256"]),
        "action_tape": run_spec["action_tape"],
        "terminal_contract": terminal_contract,
        "pair_count": PAIR_COUNT,
        "pairs": pairs,
    }
    pair_registry = {
        **pair_registry_body,
        "pair_registry_sha256": sha256_bytes(canonical_json(pair_registry_body).encode("utf-8")),
    }
    _validate_frozen_schedule(run_spec, pair_registry, source_registry=source_registry, config=config)
    write_json_atomic(pair_registry_path, pair_registry)
    return PreparedRun(
        config_target,
        config,
        source_registry_path,
        source_registry,
        run_spec_path,
        run_spec,
        pair_registry_path,
        pair_registry,
        tape,
    )


def _normalise_process_result(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_restore(dict(value))
    if isinstance(value, subprocess.CompletedProcess):
        stdout = value.stdout.decode("utf-8", errors="replace") if isinstance(value.stdout, bytes) else value.stdout
        if isinstance(stdout, str) and stdout.strip():
            try:
                parsed = json.loads(stdout)
                if isinstance(parsed, Mapping):
                    return _json_restore(dict(parsed))
            except json.JSONDecodeError:
                pass
        return {"status": "failed", "error": "worker returned no valid JSON", "returncode": value.returncode}
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return {"status": "failed", "error": f"worker output is not JSON: {exc}"}
        return _json_restore(dict(parsed)) if isinstance(parsed, Mapping) else {"status": "failed", "error": "worker output is not a mapping"}
    return {"status": "failed", "error": f"unsupported process result: {type(value).__name__}"}


def _finalize_process_result(
    result: Mapping[str, Any], request: Mapping[str, Any], *, strict: bool
) -> dict[str, Any]:
    """Apply the parent-side status/hash contract without hiding omissions."""

    normalized = _json_restore(dict(result))
    if normalized.get("status") is None:
        normalized["status"] = "completed" if normalized.get("terminal") else "failed"
    if normalized.get("status") == "completed" and "output_sha256" not in normalized:
        if strict:
            failure = _attempt_failure(
                request,
                ProtocolError("completed worker result is missing output_sha256"),
            )
            # Preserve the independently written result/log references.  Do
            # not retain the full trajectory in the parent failure record.
            if isinstance(normalized.get("worker_transport"), Mapping):
                failure["worker_transport"] = copy.deepcopy(normalized["worker_transport"])
            failure["worker_result_status"] = "completed"
            return failure
        normalized["output_sha256"] = payload_sha256(normalized)
    return normalized


def _bind_worker_result(result: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """Reject worker output that cannot be bound to its frozen attempt slot."""

    expected = {
        "attempt_id": request.get("attempt_id"),
        "pair_id": request.get("pair_id"),
        "side": request.get("side"),
    }
    mismatches = [
        f"{name}={result.get(name)!r} expected {value!r}"
        for name, value in expected.items()
        if result.get(name) is not None and result.get(name) != value
    ]
    if mismatches:
        failure = _attempt_failure(request, ProtocolError("worker identity mismatch: " + "; ".join(mismatches)))
        failure["worker_result"] = copy.deepcopy(dict(result))
        return failure
    bound = copy.deepcopy(dict(result))
    for name, value in expected.items():
        bound[name] = value
    return bound


def _stream_artifact(path: Path, value: str | bytes | None) -> dict[str, Any]:
    raw = value.encode("utf-8") if isinstance(value, str) else bytes(value or b"")
    record = _exclusive_bytes(path, raw)
    return {"path": str(record), "sha256": sha256_bytes(raw), "size": len(raw)}


def _default_process_runner(attempt: Mapping[str, Any]) -> dict[str, Any]:
    job_dir = _resolve_path(attempt.get("output_root", "runs/m1_null_calibration")) / "jobs"
    job_path = job_dir / f"{attempt['attempt_id']}.json"
    result_path = job_dir / f"{attempt['attempt_id']}.result.json"
    stdout_path = job_dir / f"{attempt['attempt_id']}.stdout.log"
    stderr_path = job_dir / f"{attempt['attempt_id']}.stderr.log"
    write_json_atomic(job_path, dict(attempt))
    configured_python = attempt.get("python")
    if not isinstance(configured_python, str) or not configured_python.strip():
        configured_python = sys.executable
    command = [
        configured_python,
        str(Path(__file__).resolve()),
        "--worker",
        "--job",
        str(job_path),
        "--result",
        str(result_path),
    ]
    raw_timeout = attempt.get("worker_timeout_seconds", DEFAULT_WORKER_TIMEOUT_SECONDS)
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("worker_timeout_seconds must be finite and positive") from exc
    if isinstance(raw_timeout, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ProtocolError("worker_timeout_seconds must be finite and positive")
    try:
        completed = subprocess.run(
            command,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = getattr(exc, "stdout", None)
        if partial_stdout is None:
            partial_stdout = getattr(exc, "output", None)
        partial_stderr = getattr(exc, "stderr", None)
        stdout_record = _stream_artifact(stdout_path, partial_stdout)
        stderr_record = _stream_artifact(stderr_path, partial_stderr)
        transport = {
            "protocol": "dedicated_result_file_v1",
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "result": {
                "path": str(result_path),
                "exists": result_path.is_file() and not result_path.is_symlink(),
            },
            "stdout": stdout_record,
            "stderr": stderr_record,
        }
        failure = _attempt_failure(
            attempt,
            ProtocolError(f"worker timed out after {timeout_seconds:g} seconds"),
        )
        failure["worker_transport"] = transport
        return failure
    transport: dict[str, Any] = {
        "protocol": "dedicated_result_file_v1",
        "returncode": int(completed.returncode),
        "timed_out": False,
        "timeout_seconds": timeout_seconds,
        "result": {
            "path": str(result_path),
            "exists": result_path.is_file() and not result_path.is_symlink(),
        },
        "stdout": _stream_artifact(stdout_path, completed.stdout),
        "stderr": _stream_artifact(stderr_path, completed.stderr),
    }
    if result_path.is_file() and not result_path.is_symlink():
        try:
            result = _load_document(result_path)
            transport["result"].update(
                {
                    "sha256": sha256_file(result_path),
                    "size": result_path.stat().st_size,
                }
            )
            result = _json_restore(result)
        except Exception as exc:
            result = {
                "status": "failed",
                "error": f"worker result file is invalid: {type(exc).__name__}: {exc}",
            }
    else:
        result = {
            "status": "failed",
            "error": "worker did not publish its dedicated result file",
        }
    result["worker_transport"] = transport
    return result


def _attempt_artifact_path(prepared: PreparedRun, attempt_id: str) -> Path:
    return _resolve_path(prepared.run_spec["output_root"]) / "attempts" / f"{attempt_id}.json"


def _verified_artifact_record(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    target = Path(path)
    digest = sha256_file(target)
    if expected_sha256 is not None and digest.lower() != str(expected_sha256).lower():
        raise PublicationError(f"artifact hash mismatch for {target}: {digest} != {expected_sha256}")
    return {
        "path": str(target),
        "sha256": digest,
        "size": target.stat().st_size,
        "verified": verify_artifact_hash(target, digest),
    }


def run_calibration(
    prepared: PreparedRun,
    *,
    process_runner: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run all 40 frozen attempts once, retaining every failure."""

    config, pair_registry, _ = _verify_prepared(prepared)
    pairs = pair_registry.get("pairs")
    if not isinstance(pairs, Sequence) or len(pairs) != PAIR_COUNT:
        raise ProvenanceError("pair registry does not contain exactly 20 pairs")
    runner = process_runner or _default_process_runner
    output_root = _resolve_path(prepared.run_spec["output_root"])
    strict = bool(config.get("strict_runtime_contract"))
    runtime_identity_contract = _runtime_identity_audit(config) if strict else {}
    expected_frozen_inputs = {
        "trace_id": prepared.run_spec["trace_id"],
        "task": copy.deepcopy(prepared.run_spec["task"]),
        "action_tape_sha256": prepared.run_spec["action_tape"]["sha256"],
        "action_shape": list(ACTION_SHAPE),
        "action_dtype": "float32",
    }
    # Only compact attempt summaries survive the pair loop.  Full A/B
    # trajectories exist at most for the currently validating pair.
    attempts: list[dict[str, Any]] = []
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    integrity_errors: list[str] = []
    attempt_artifacts: list[dict[str, Any]] = []
    validations: list[PairValidation] = []
    measurement_records: list[dict[str, Any]] = []
    parent_pid = os.getpid()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ProvenanceError("pair registry contains a malformed pair")
        pair_id = str(pair["pair_id"])
        pair_attempts: dict[str, Mapping[str, Any]] = {}
        for side in ("A", "B"):
            request = _prepare_attempt(prepared, pair, side)
            if strict:
                request["runtime_identity_contract"] = copy.deepcopy(runtime_identity_contract)
            try:
                result = _bind_worker_result(_normalise_process_result(runner(request)), request)
                result = _finalize_process_result(result, request, strict=strict)
            except Exception as exc:
                result = _attempt_failure(request, exc)
                result.update({"attempt_id": request["attempt_id"], "pair_id": pair_id, "side": side})
            artifact_path = _attempt_artifact_path(prepared, str(request["attempt_id"]))
            artifact_record = write_json_atomic(artifact_path, result)
            artifact_verified = False
            try:
                verified_record = _verified_artifact_record(artifact_path, artifact_record["sha256"])
                array_artifacts = []
                for sidecar in artifact_record.get("array_artifacts", []):
                    if not isinstance(sidecar, Mapping):
                        raise PublicationError("attempt sidecar record is malformed")
                    sidecar_path = sidecar.get("path")
                    sidecar_sha = sidecar.get("sha256")
                    if not isinstance(sidecar_path, (str, Path)) or not isinstance(sidecar_sha, str):
                        raise PublicationError("attempt sidecar record lacks path/SHA")
                    verify_artifact_hash(sidecar_path, sidecar_sha)
                    array_artifacts.append(
                        {
                            "path": str(sidecar_path),
                            "sha256": sidecar_sha,
                            "size": int(sidecar.get("size", Path(sidecar_path).stat().st_size)),
                            "format": "npy",
                            "allow_pickle": False,
                            "verified": True,
                        }
                    )
                attempt_artifacts.append(
                    {
                        "attempt_id": str(request["attempt_id"]),
                        **verified_record,
                        "array_artifacts": array_artifacts,
                    }
                )
                artifact_verified = bool(verified_record["verified"])
            except Exception as exc:
                integrity_errors.append(
                    f"{request['attempt_id']}: attempt artifact integrity failed: {type(exc).__name__}: {exc}"
                )
            result = {
                **result,
                "artifact_path": str(artifact_path),
                "artifact_sha256": artifact_record["sha256"],
                "artifact_size": artifact_record["size"],
                "artifact_integrity": {
                    "verified": artifact_verified,
                    "hash_domain": "canonical_json_bytes_with_trailing_newline",
                },
            }
            summary = _compact_attempt_summary(result)
            attempts.append(summary)
            by_pair.setdefault(pair_id, {})[side] = summary
            pair_attempts[side] = result
        full_validation = validate_pair(
            pair,
            pair_attempts,
            parent_pid=parent_pid,
            expected_frozen_inputs=expected_frozen_inputs,
            expected_terminal=prepared.run_spec.get("terminal_contract"),
            expected_runtime_identity=runtime_identity_contract if strict else None,
            expected_windows=prepared.run_spec.get("windows") if strict else None,
            expected_action_tape=prepared.tape if strict else None,
            required_invariant_roots=_configured_invariant_roots(config),
            strict=strict,
        )
        compact_measurements = _compact_pair_measurements(full_validation, config=config)
        compact_validation = PairValidation(
            full_validation.pair_id,
            full_validation.trace_id,
            full_validation.valid,
            full_validation.reasons,
            {},
            full_validation.discrete,
            compact_measurements,
        )
        validations.append(compact_validation)
        measurement_records.append(compact_measurements)
        # Explicitly drop the trajectory-bearing objects before launching the
        # next pair; immutable attempt artifacts remain independently readable.
        pair_attempts.clear()
        full_validation = None
    complete_validations = [item for item in validations if item.valid]
    candidate_valid_count = len(complete_validations)
    valid_count = candidate_valid_count if candidate_valid_count == PAIR_COUNT else 0
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED",
        "attempts": attempts,
        "pair_results": [item.to_dict() for item in validations],
        "counts": {
            "scheduled_attempts": PAIR_COUNT * 2,
            "completed_attempts": sum(item.get("status") == "completed" for item in attempts),
            "failed_attempts": sum(item.get("status") != "completed" for item in attempts),
            "candidate_valid_pairs": candidate_valid_count,
            "valid_pairs": valid_count,
        },
        "gates": {
            "provenance": True,
            "pair_count": valid_count == PAIR_COUNT,
            "process_independence": all(item.valid for item in validations),
            "regime_coverage": False,
            "physics": False,
            "discrete": all(item.discrete.get("categorical_gate", False) for item in validations),
            "renderer": False,
            "integrity": not integrity_errors,
        },
        "errors": [f"{item.pair_id}: {reason}" for item in validations for reason in item.reasons]
        + integrity_errors,
    }
    envelope_paths: dict[str, str] = {}
    envelope_publication_records: dict[str, dict[str, Any]] = {}
    if valid_count == PAIR_COUNT:
        try:
            envelope = build_envelopes(validations, required_pair_ids=[str(pair["pair_id"]) for pair in pairs], config=config)
            result["envelopes"] = {
                "physics": envelope.physics,
                "renderer": envelope.renderer,
                "discrete": envelope.discrete,
            }
            physics_path = output_root / "physics_envelope.json"
            renderer_path = output_root / "renderer_envelope.json"
            physics_publication = write_json_atomic(physics_path, envelope.physics)
            renderer_publication = write_json_atomic(renderer_path, envelope.renderer)
            envelope_paths = {
                "physics": str(physics_path),
                "renderer": str(renderer_path),
            }
            envelope_publication_records = {
                "physics": physics_publication,
                "renderer": renderer_publication,
            }
            result["envelope_paths"] = envelope_paths
            result["gates"].update({"regime_coverage": True, "physics": True, "renderer": True})
            result["status"] = "PASS" if all(result["gates"].values()) else "BLOCKED"
        except Exception as exc:
            result["errors"].append(f"envelope construction failed: {type(exc).__name__}: {exc}")
    final_pair_registry_path = output_root / "final_pair_registry.json"
    attempt_artifacts_by_id = {
        str(item["attempt_id"]): item for item in attempt_artifacts if isinstance(item, Mapping)
    }
    final_pairs = []
    for pair, validation in zip(pairs, validations):
        pair_id = str(pair["pair_id"])
        final_pairs.append(
            {
                "pair_id": pair_id,
                "trace_id": pair.get("trace_id"),
                # Attempt payloads (including full invariant/observation
                # arrays) live exactly once in their immutable artifacts.
                # The final registry is a reconstructible index, not a second
                # copy of the trajectory evidence.
                "attempts": [
                    {
                        "attempt_id": f"{pair_id}-{side}",
                        "side": side,
                        "status": by_pair.get(pair_id, {}).get(side, {}).get("status", "missing"),
                        "artifact": copy.deepcopy(
                            attempt_artifacts_by_id.get(
                                f"{pair_id}-{side}",
                                {"verified": False, "missing": True},
                            )
                        ),
                    }
                    for side in ("A", "B")
                ],
                "validation": validation.to_dict(),
            }
        )
    final_registry_body = {
        "schema_version": SCHEMA_VERSION,
        "registry_type": "m1_null_calibration_final_pair_registry",
        "run_spec_path": str(prepared.run_spec_path),
        "run_spec_sha256": prepared.run_spec["run_spec_sha256"],
        "pair_registry_path": str(prepared.pair_registry_path),
        "pair_registry_sha256": pair_registry["pair_registry_sha256"],
        "pair_count": PAIR_COUNT,
        "attempt_count": PAIR_COUNT * 2,
        "pairs": final_pairs,
    }
    final_pair_registry = {
        **final_registry_body,
        "final_pair_registry_sha256": sha256_bytes(canonical_json(final_registry_body).encode("utf-8")),
    }
    final_registry_record = write_json_atomic(final_pair_registry_path, final_pair_registry)
    try:
        _verified_artifact_record(final_pair_registry_path, final_registry_record["sha256"])
        if not verify_self_hash(final_pair_registry, "final_pair_registry_sha256"):
            raise PublicationError("final pair registry self-hash is invalid")
    except Exception as exc:
        integrity_errors.append(
            f"final pair registry integrity failed: {type(exc).__name__}: {exc}"
        )
    pair_measurement_path = output_root / "pair_measurements.json"
    pair_measurement_record = write_json_atomic(
        pair_measurement_path,
        {
            "schema_version": SCHEMA_VERSION,
            "final_pair_registry_sha256": final_pair_registry["final_pair_registry_sha256"],
            # Compact scalar measurements are reduced and retained by the
            # parent immediately after each pair is validated.  Keep this
            # explicit record separate from the validation/index summaries;
            # the full trajectories remain only in the immutable attempt
            # artifacts referenced below.
            "measurement_records": measurement_records,
            "pairs": result["pair_results"],
            # Validation summaries and immutable artifact references are
            # sufficient to reconstruct every pair independently; embedding
            # the raw arrays again would create an unverifiable duplicate.
            "attempt_artifacts": attempt_artifacts,
        },
    )
    try:
        _verified_artifact_record(pair_measurement_path, pair_measurement_record["sha256"])
    except Exception as exc:
        integrity_errors.append(
            f"pair measurements integrity failed: {type(exc).__name__}: {exc}"
        )
    result["pair_measurements_path"] = str(pair_measurement_path)
    result["gates"]["integrity"] = not integrity_errors
    if integrity_errors:
        result["errors"].extend(item for item in integrity_errors if item not in result["errors"])
        result["status"] = "BLOCKED"
    artifact_records = {
        "attempts": attempt_artifacts,
        "run_spec": _verified_artifact_record(
            prepared.run_spec_path,
            sha256_file(prepared.run_spec_path),
        ),
        "pair_registry": _verified_artifact_record(
            prepared.pair_registry_path,
            sha256_file(prepared.pair_registry_path),
        ),
        "final_pair_registry": _verified_artifact_record(
            final_pair_registry_path,
            final_registry_record["sha256"],
        ),
        "pair_measurements": _verified_artifact_record(
            pair_measurement_path,
            pair_measurement_record["sha256"],
        ),
    }
    terminal_body = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "m1_n0_terminal",
        "status": result["status"],
        "run_spec_path": str(prepared.run_spec_path),
        "run_spec_sha256": prepared.run_spec["run_spec_sha256"],
        "pair_registry_path": str(prepared.pair_registry_path),
        "pair_registry_sha256": pair_registry["pair_registry_sha256"],
        "final_pair_registry_path": str(final_pair_registry_path),
        "final_pair_registry_sha256": final_pair_registry["final_pair_registry_sha256"],
        "counts": result["counts"],
        "gates": result["gates"],
        "errors": result["errors"],
        "artifacts": {
            **artifact_records,
            **{
                name: {
                    **_verified_artifact_record(path, sha256_file(path)),
                    "array_artifacts": list(
                        envelope_publication_records.get(name, {}).get("array_artifacts", [])
                    ),
                }
                for name, path in envelope_paths.items()
            },
        },
        "integrity": {
            "verified": not integrity_errors,
            "errors": list(integrity_errors),
        },
    }
    terminal = {**terminal_body, "terminal_manifest_sha256": sha256_bytes(canonical_json(terminal_body).encode("utf-8"))}
    terminal_path = output_root / "terminal_manifest.json"
    terminal_record = write_json_atomic(terminal_path, terminal)
    try:
        _verified_artifact_record(terminal_path, terminal_record["sha256"])
    except Exception as exc:
        # The terminal manifest is immutable; retain a failed integrity gate
        # in the returned result even if a filesystem readback is unavailable.
        result["gates"]["integrity"] = False
        result["status"] = "BLOCKED"
        result["errors"].append(
            f"terminal manifest integrity failed: {type(exc).__name__}: {exc}"
        )
    result["terminal_manifest_path"] = str(terminal_path)
    return result


def _attempt_control_pair(
    left: Mapping[str, Any], right: Mapping[str, Any], *, left_action: Any = None, right_action: Any = None, strict: bool = False
) -> list[Any]:
    if left_action is not None and right_action is not None:
        controls = [np.asarray(left_action), np.asarray(right_action)]
        for control in controls:
            if control.dtype != ACTION_DTYPE or control.shape != (7,) or not np.all(np.isfinite(control)):
                raise NullCalibrationError("renderer controls must be exact finite float32 actions")
        return controls
    for key in ("duplicate_controls", "controls", "control_tape"):
        lvalue, rvalue = left.get(key), right.get(key)
        if isinstance(lvalue, Mapping) and isinstance(rvalue, Mapping):
            if "A" in lvalue and "B" in rvalue:
                return [lvalue["A"], rvalue["B"]]
        if lvalue is not None and rvalue is not None:
            return [lvalue, rvalue]
    if strict:
        raise NullCalibrationError("renderer sample lacks actual frozen controls")
    return [np.asarray([0], dtype=np.float32), np.asarray([0], dtype=np.float32)]


def _snapshot_action(snapshot: Mapping[str, Any]) -> np.ndarray | None:
    value = snapshot.get("action")
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype != ACTION_DTYPE or array.shape != (7,):
        raise ProtocolError("window action is not exact float32 shape (7,)")
    return np.ascontiguousarray(array).copy()


def _renderer_camera_for_key(key: str) -> str | None:
    """Resolve an official RGB observation key to its actual camera/view."""

    normalized = str(key)
    if normalized == DEFAULT_RENDER_KEY:
        return DEFAULT_CAMERA
    leaf = normalized.rsplit(".", 1)[-1]
    if "[" in leaf:
        leaf = leaf.split("[", 1)[0]
    return OFFICIAL_RGB_LEAF_CAMERAS.get(leaf)


def _renderer_images(
    snapshot: Mapping[str, Any],
    *,
    camera: str = DEFAULT_CAMERA,
    configured_key: str = DEFAULT_RENDER_KEY,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Collect direct and official-tree RGB images with fail-closed view IDs."""

    result: dict[str, np.ndarray] = {}
    configured_camera = str(camera)
    if strict and configured_camera != DEFAULT_CAMERA:
        # ``render_rgb`` is an unlabelled direct render.  It is only permitted
        # as the official agentview diagnostic and may not be relabelled as a
        # side/wrist view by configuration.
        raise NullCalibrationError(
            f"direct renderer is frozen to {DEFAULT_CAMERA}, got {configured_camera!r}"
        )
    direct = snapshot.get("renderer")
    if direct is not None:
        value = np.asarray(direct)
        if value.dtype != np.dtype("uint8") or value.ndim < 2 or value.size == 0:
            raise NullCalibrationError("direct renderer output must be a non-empty uint8 image")
        result[str(configured_key)] = np.ascontiguousarray(value)
    raw = snapshot.get("raw_observation")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        try:
            array = np.asarray(value)
        except Exception:
            return
        if array.dtype == np.dtype("uint8") and array.ndim >= 2 and array.size:
            resolved_camera = _renderer_camera_for_key(path)
            if resolved_camera is None:
                if strict:
                    raise NullCalibrationError(
                        f"unknown official RGB observation key/camera mapping: {path}"
                    )
                # Legacy synthetic fixtures used a single generic ``pixels``
                # leaf.  Keep that non-authoritative compatibility path while
                # strict workers require a named official camera key.
                resolved_camera = DEFAULT_CAMERA if path == "pixels" else None
            if resolved_camera is None:
                return
            result[path] = np.ascontiguousarray(array)
    visit(raw, "")
    return result


def _compact_pair_measurements(validation: PairValidation, *, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Reduce one validated pair to aggregate-ready scalar evidence.

    The full A/B attempt remains available in its immutable artifact on disk
    while this parent-side record retains only numeric deltas, RGB metrics,
    controls, and provenance coordinates needed for envelope aggregation.
    """

    if not validation.valid:
        return {
            "schema_version": 1,
            "pair_id": validation.pair_id,
            "trace_id": validation.trace_id,
            "physics_samples": [],
            "renderer_samples": [],
        }
    attempts = validation.attempts
    left, right = attempts.get("A"), attempts.get("B")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise NullCalibrationError("validated pair has no A/B evidence for compaction")
    strict_contract = bool(config and config.get("strict_runtime_contract"))
    roots = DEFAULT_QUANTITY_ROOTS
    if isinstance(config, Mapping):
        selection = config.get("quantity_selection")
        if isinstance(selection, Mapping) and isinstance(selection.get("root_patterns"), Sequence):
            roots = tuple(str(item) for item in selection["root_patterns"])
    renderer_config = config.get("renderer", {}) if isinstance(config, Mapping) else {}
    configured_camera = (
        str(renderer_config.get("camera", DEFAULT_CAMERA))
        if isinstance(renderer_config, Mapping)
        else DEFAULT_CAMERA
    )
    configured_key = (
        str(renderer_config.get("observation_key", DEFAULT_RENDER_KEY))
        if isinstance(renderer_config, Mapping)
        else DEFAULT_RENDER_KEY
    )
    left_windows, right_windows = _window_map(left), _window_map(right)
    physics_samples: list[dict[str, Any]] = []
    renderer_samples: list[dict[str, Any]] = []
    hard_gate = _load_hard_gate_for_envelope()
    for window_id in sorted(left_windows):
        lwindow, rwindow = left_windows.get(window_id), right_windows.get(window_id)
        if lwindow is None or rwindow is None:
            raise NullCalibrationError(f"pair {validation.pair_id} has an incomplete window set")
        lsnap, rsnap = _window_snapshots(lwindow), _window_snapshots(rwindow)
        for horizon in sorted(lsnap):
            left_values = _selected_numeric_leaves(lsnap[horizon], roots)
            right_values = _selected_numeric_leaves(rsnap[horizon], roots)
            if set(left_values) != set(right_values):
                raise NullCalibrationError(
                    f"quantity path set differs at {validation.pair_id}/{window_id}/{horizon}"
                )
            regimes = lwindow.get("regimes", lwindow.get("regime", [window_id]))
            if isinstance(regimes, str):
                regimes = [regimes]
            left_action = _snapshot_action(lsnap[horizon])
            right_action = _snapshot_action(rsnap[horizon])
            if strict_contract and (
                left_action is None
                or right_action is None
                or left_action.tobytes(order="C") != right_action.tobytes(order="C")
            ):
                raise NullCalibrationError(
                    f"{validation.pair_id}/{window_id}/{horizon} frozen action evidence differs"
                )
            history_left = [lsnap[step] for step in sorted(lsnap)]
            history_right = [rsnap[step] for step in sorted(rsnap)]
            for quantity in sorted(left_values):
                left_value, right_value = left_values[quantity], right_values[quantity]
                if left_value.shape != right_value.shape:
                    raise NullCalibrationError(
                        f"quantity shape differs at {validation.pair_id}/{window_id}/{horizon}"
                    )
                difference = np.asarray(left_value, dtype=np.float64) - np.asarray(
                    right_value, dtype=np.float64
                )
                if difference.size == 0 or not np.all(np.isfinite(difference)):
                    raise NullCalibrationError("compact physics difference is empty or non-finite")
                max_abs = float(np.max(np.abs(difference)))
                for regime in (str(item) for item in regimes):
                    physics_samples.append(
                        {
                            "pair_id": validation.pair_id,
                            "trace_id": validation.trace_id,
                            "regime": regime,
                            "quantity": quantity,
                            "horizon": int(horizon),
                            "difference": max_abs,
                            "independent_evidence": _independent_evidence(
                                regime,
                                history_left[0],
                                lsnap[horizon],
                                action=left_action,
                                history=history_left,
                                strict=strict_contract,
                            ),
                        }
                    )
            left_images = _renderer_images(
                lsnap[horizon],
                camera=configured_camera,
                configured_key=configured_key,
                strict=strict_contract,
            )
            right_images = _renderer_images(
                rsnap[horizon],
                camera=configured_camera,
                configured_key=configured_key,
                strict=strict_contract,
            )
            if set(left_images) != set(right_images):
                raise NullCalibrationError(
                    f"renderer observation key set differs at {validation.pair_id}/{window_id}/{horizon}"
                )
            controls = _attempt_control_pair(
                left,
                right,
                left_action=left_action,
                right_action=right_action,
                strict=strict_contract,
            )
            for key in sorted(left_images):
                metrics = hard_gate.rgb_disagreement_metrics(left_images[key], right_images[key])
                image_camera = _renderer_camera_for_key(key) or configured_camera
                for regime in (str(item) for item in regimes):
                    renderer_samples.append(
                        {
                            "pair_id": validation.pair_id,
                            "trace_id": validation.trace_id,
                            "camera": image_camera,
                            "key": key,
                            "regime": regime,
                            "horizon": int(horizon),
                            "duplicate_controls": [
                                np.ascontiguousarray(np.asarray(item)).copy() for item in controls
                            ],
                            "metrics": {
                                "differing_pixel_count": int(metrics["differing_pixel_count"]),
                                "max_abs": float(metrics["max_abs"]),
                                "mean_abs": float(metrics["mean_abs"]),
                            },
                        }
                    )
    return {
        "schema_version": 1,
        "pair_id": validation.pair_id,
        "trace_id": validation.trace_id,
        "physics_samples": physics_samples,
        "renderer_samples": renderer_samples,
    }


def _compact_attempt_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only non-trajectory attempt metadata in parent memory/results."""

    fields = (
        "attempt_id",
        "pair_id",
        "side",
        "status",
        "error",
        "pid",
        "ppid",
        "process_start_identity",
        "process_start_identity_source",
        "output_sha256",
        "artifact_path",
        "artifact_sha256",
        "artifact_size",
        "artifact_integrity",
        "worker_transport",
        "worker_result_status",
        "frozen_inputs",
        "terminal",
        "protocol",
        "reset_provenance",
        "runtime",
        "close_evidence",
    )
    summary = {name: copy.deepcopy(result[name]) for name in fields if name in result}
    terminal = summary.get("terminal")
    if isinstance(terminal, Mapping):
        terminal = dict(terminal)
        terminal.pop("terminal_observation", None)
        summary["terminal"] = terminal
    return summary


def build_envelopes(
    pair_results: Iterable[PairValidation],
    *,
    required_pair_ids: Sequence[str],
    config: Mapping[str, Any] | None = None,
) -> EnvelopeBundle:
    """Derive strict regime×quantity×horizon physics and RGB envelopes."""

    required = [str(item) for item in required_pair_ids]
    if len(required) != PAIR_COUNT or len(set(required)) != PAIR_COUNT:
        raise NullCalibrationError("exactly 20 distinct required pair IDs are required")
    values = list(pair_results)
    by_id = {item.pair_id: item for item in values if isinstance(item, PairValidation)}
    if set(by_id) != set(required):
        raise NullCalibrationError("pair envelope input IDs do not match the frozen 20-pair schedule")
    if any(not by_id[pair_id].valid for pair_id in required):
        raise NullCalibrationError("invalid or semantically divergent pair cannot enter envelope aggregation")
    # The authoritative run drops full A/B trajectories immediately after
    # validating each pair.  Aggregate its compact scalar records directly;
    # legacy fixture callers with full attempts continue through the detailed
    # structural-validation path below.
    compact_flags = [
        isinstance(by_id[pair_id].measurements, Mapping)
        and "physics_samples" in by_id[pair_id].measurements
        and "renderer_samples" in by_id[pair_id].measurements
        for pair_id in required
    ]
    if any(compact_flags):
        if not all(compact_flags):
            raise NullCalibrationError("pair envelope inputs mix compact and full trajectory records")
        return _build_envelopes_from_compact(
            values,
            required_pair_ids=required,
            config=config,
        )
    strict_contract = bool(config and config.get("strict_runtime_contract"))
    required_invariant_roots = _configured_invariant_roots(config or {}) if strict_contract else ()
    roots = DEFAULT_QUANTITY_ROOTS
    if isinstance(config, Mapping):
        selection = config.get("quantity_selection")
        if isinstance(selection, Mapping) and isinstance(selection.get("root_patterns"), Sequence):
            roots = tuple(str(item) for item in selection["root_patterns"])
    if not roots or any(not item or item.startswith("/") for item in roots):
        raise NullCalibrationError("quantity-selection root patterns must be explicit relative paths")

    hard_gate = _load_hard_gate_for_envelope()
    physics_samples: list[dict[str, Any]] = []
    renderer_samples: list[dict[str, Any]] = []
    # Quantity paths are frozen per semantic regime and continuation horizon.
    # Contact lists can legitimately change cardinality between regimes, so a
    # single global path set would incorrectly reject a valid exact schedule.
    path_sets: dict[tuple[str, int], set[str]] = {}
    contact_identity_exact = True
    categorical = True
    for pair_id in required:
        pair = by_id[pair_id]
        left, right = pair.attempts["A"], pair.attempts["B"]
        left_windows, right_windows = _window_map(left), _window_map(right)
        for window_id in sorted(set(left_windows) | set(right_windows)):
            lwindow, rwindow = left_windows.get(window_id), right_windows.get(window_id)
            if lwindow is None or rwindow is None:
                raise NullCalibrationError(f"pair {pair_id} has an incomplete window set")
            lsnap, rsnap = _window_snapshots(lwindow), _window_snapshots(rwindow)
            if set(lsnap) != set(rsnap) or not lsnap:
                raise NullCalibrationError(f"pair {pair_id} window {window_id} has incomplete coordinates")
            regimes = lwindow.get("regimes", lwindow.get("regime", [window_id]))
            if isinstance(regimes, str):
                regimes = [regimes]
            for horizon in sorted(lsnap):
                if strict_contract:
                    for side, snapshot in (("A", lsnap[horizon]), ("B", rsnap[horizon])):
                        try:
                            _validate_invariant_snapshot(
                                snapshot,
                                required_roots=required_invariant_roots,
                            )
                        except ProtocolError as exc:
                            raise NullCalibrationError(
                                f"{pair_id}/{window_id}/{horizon} {side} invariant roots are invalid: {exc}"
                            ) from exc
                if _snapshot_contacts(lsnap[horizon]) != _snapshot_contacts(rsnap[horizon]):
                    contact_identity_exact = False
                    categorical = False
                    raise NullCalibrationError(
                        f"contact identity differs before numeric distance comparison at "
                        f"{pair_id}/{window_id}/{horizon}"
                    )
                left_values = _selected_numeric_leaves(lsnap[horizon], roots)
                right_values = _selected_numeric_leaves(rsnap[horizon], roots)
                if set(left_values) != set(right_values):
                    raise NullCalibrationError(f"quantity path set differs at {pair_id}/{window_id}/{horizon}")
                normalized_regimes = [str(regime) for regime in regimes]
                for regime in normalized_regimes:
                    group_key = (regime, int(horizon))
                    current_paths = set(left_values)
                    previous_paths = path_sets.get(group_key)
                    if previous_paths is None:
                        path_sets[group_key] = current_paths
                    elif current_paths != previous_paths:
                        raise NullCalibrationError(
                            "quantity-selection path set is inconsistent at "
                            f"{regime}@{horizon}"
                        )
                history_left = [lsnap[step] for step in sorted(lsnap)]
                history_right = [rsnap[step] for step in sorted(rsnap)]
                left_action = _snapshot_action(lsnap[horizon])
                right_action = _snapshot_action(rsnap[horizon])
                if strict_contract and (
                    left_action is None
                    or right_action is None
                    or left_action.tobytes(order="C") != right_action.tobytes(order="C")
                ):
                    raise NullCalibrationError(
                        f"{pair_id}/{window_id}/{horizon} frozen action evidence differs or is missing"
                    )
                for quantity in sorted(left_values):
                    if left_values[quantity].shape != right_values[quantity].shape:
                        raise NullCalibrationError(f"quantity shape differs at {pair_id}/{quantity}")
                    for regime in normalized_regimes:
                        evidence = _independent_evidence(
                            str(regime),
                            history_left[0],
                            lsnap[horizon],
                            action=left_action,
                            history=history_left,
                            strict=strict_contract,
                        )
                        physics_samples.append(
                            {
                                "pair_id": pair_id,
                                "trace_id": pair.trace_id,
                                "regime": str(regime),
                                "quantity": quantity,
                                "horizon": horizon,
                                "value_a": left_values[quantity],
                                "value_b": right_values[quantity],
                                "independent_evidence": evidence,
                            }
                        )
                if _snapshot_contacts(lsnap[horizon]) != _snapshot_contacts(rsnap[horizon]):
                    contact_identity_exact = False
                    categorical = False
                renderer_config = config.get("renderer", {}) if isinstance(config, Mapping) else {}
                configured_camera = (
                    str(renderer_config.get("camera", DEFAULT_CAMERA))
                    if isinstance(renderer_config, Mapping)
                    else DEFAULT_CAMERA
                )
                configured_key = (
                    str(renderer_config.get("observation_key", DEFAULT_RENDER_KEY))
                    if isinstance(renderer_config, Mapping)
                    else DEFAULT_RENDER_KEY
                )
                if not configured_camera.strip() or not configured_key.strip():
                    raise NullCalibrationError("renderer camera and observation_key must be non-empty")
                left_images = _renderer_images(
                    lsnap[horizon],
                    camera=configured_camera,
                    configured_key=configured_key,
                    strict=strict_contract,
                )
                right_images = _renderer_images(
                    rsnap[horizon],
                    camera=configured_camera,
                    configured_key=configured_key,
                    strict=strict_contract,
                )
                if set(left_images) != set(right_images):
                    raise NullCalibrationError(f"renderer observation key set differs at {pair_id}/{window_id}/{horizon}")
                controls = _attempt_control_pair(
                    left,
                    right,
                    left_action=left_action,
                    right_action=right_action,
                    strict=strict_contract,
                )
                for key in sorted(left_images):
                    image_camera = _renderer_camera_for_key(key) or configured_camera
                    for regime in normalized_regimes:
                        renderer_samples.append(
                            {
                                "pair_id": pair_id,
                                "trace_id": pair.trace_id,
                                "camera": image_camera,
                                "key": key,
                                "regime": str(regime),
                                "horizon": horizon,
                                "duplicate_controls": controls,
                                "rgb_a": left_images[key],
                                "rgb_b": right_images[key],
                            }
                        )
    if not path_sets or not physics_samples:
        raise NullCalibrationError("no explicitly selected physics quantities were observed")
    required_groups = sorted(
        {(str(item["regime"]), str(item["quantity"]), int(item["horizon"])) for item in physics_samples}
    )
    physics = hard_gate.build_grouped_null_envelope(
        physics_samples,
        selected_trace_ids=sorted({str(item.trace_id) for item in by_id.values() if item.trace_id is not None}),
        required_groups=required_groups,
        min_pairs=PAIR_COUNT,
        min_pairs_per_trace=5,
        min_samples_per_regime=5,
        global_tolerance=None,
    )
    if not renderer_samples:
        raise NullCalibrationError("no direct or official-tree RGB observations were observed")
    renderer = hard_gate.build_renderer_envelopes(
        renderer_samples,
        exact_only=None,
        min_samples_per_group=PAIR_COUNT,
    )
    renderer.setdefault("coverage", {}).update(
        {
            "n_pairs": PAIR_COUNT,
            "pair_ids": sorted(required),
            "configured_camera": configured_camera,
            "configured_observation_key": configured_key,
        }
    )
    discrete = {
        "categorical_gate": categorical and all(item.discrete.get("categorical_gate", False) for item in by_id.values()),
        "contact_identity_exact": contact_identity_exact and all(item.discrete.get("contact_identity_exact", False) for item in by_id.values()),
        "predicate_exact": all(item.discrete.get("predicate_exact", False) for item in by_id.values()),
        "success_exact": all(item.discrete.get("success_exact", False) for item in by_id.values()),
        "done_termination_exact": all(item.discrete.get("done_termination_exact", False) for item in by_id.values()),
        "terminal_timing_exact": all(item.discrete.get("terminal_timing_exact", False) for item in by_id.values()),
        "gripper_exact": all(item.discrete.get("gripper_exact", False) for item in by_id.values()),
        "action_exact": all(item.discrete.get("action_exact", False) for item in by_id.values()),
        "observation_exact": all(item.discrete.get("observation_exact", False) for item in by_id.values()),
        "pair_count": PAIR_COUNT,
        "divergences": [
            {"pair_id": item.pair_id, "reasons": list(item.reasons)}
            for item in by_id.values()
            if item.reasons
        ],
    }
    return EnvelopeBundle(physics=physics, renderer=renderer, discrete=discrete)


def _load_hard_gate_for_envelope() -> Any:
    from scripts import m1_hard_gate

    return m1_hard_gate


def _build_envelopes_from_compact(
    pair_results: Sequence[PairValidation],
    *,
    required_pair_ids: Sequence[str],
    config: Mapping[str, Any] | None = None,
) -> EnvelopeBundle:
    """Aggregate pair-reduced measurements without retaining raw trajectories.

    ``run_calibration`` validates and compacts one pair before launching the
    next pair.  The compact records contain scalar physics deltas, renderer
    metrics, frozen controls, and independent-evidence summaries; all raw
    snapshots remain in their immutable attempt artifacts.  Keep the legacy
    full-attempt path in :func:`build_envelopes` for small fixture callers,
    while making the authoritative schedule use this bounded-memory path.
    """

    required = [str(item) for item in required_pair_ids]
    if len(required) != PAIR_COUNT or len(set(required)) != PAIR_COUNT:
        raise NullCalibrationError("exactly 20 distinct required pair IDs are required")
    by_id = {item.pair_id: item for item in pair_results if isinstance(item, PairValidation)}
    if set(by_id) != set(required):
        raise NullCalibrationError("pair envelope input IDs do not match the frozen 20-pair schedule")
    if any(not by_id[pair_id].valid for pair_id in required):
        raise NullCalibrationError("invalid or semantically divergent pair cannot enter envelope aggregation")

    physics_samples: list[dict[str, Any]] = []
    renderer_samples: list[dict[str, Any]] = []
    path_sets: dict[tuple[str, int], set[str]] = {}
    per_pair_paths: dict[tuple[str, str, int], set[str]] = {}
    traces: set[str] = set()
    hard_gate = _load_hard_gate_for_envelope()
    for pair_id in required:
        pair = by_id[pair_id]
        measurements = pair.measurements
        if not isinstance(measurements, Mapping):
            raise NullCalibrationError(f"pair {pair_id} compact measurements are missing")
        raw_physics = measurements.get("physics_samples")
        raw_renderer = measurements.get("renderer_samples")
        if not isinstance(raw_physics, Sequence) or isinstance(raw_physics, (str, bytes)):
            raise NullCalibrationError(f"pair {pair_id} compact physics measurements are missing")
        if not isinstance(raw_renderer, Sequence) or isinstance(raw_renderer, (str, bytes)):
            raise NullCalibrationError(f"pair {pair_id} compact renderer measurements are missing")
        if pair.trace_id is None:
            raise NullCalibrationError(f"pair {pair_id} compact trace identity is missing")
        traces.add(str(pair.trace_id))
        for index, raw_sample in enumerate(raw_physics):
            if not isinstance(raw_sample, Mapping):
                raise NullCalibrationError(f"pair {pair_id} compact physics sample {index} is malformed")
            sample = copy.deepcopy(dict(raw_sample))
            if str(sample.get("pair_id")) != pair_id or str(sample.get("trace_id")) != str(pair.trace_id):
                raise NullCalibrationError(f"pair {pair_id} compact physics provenance is inconsistent")
            regime = hard_gate.normalize_regime(sample.get("regime"))
            quantity = str(sample.get("quantity", ""))
            raw_horizon = sample.get("horizon")
            if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, (int, np.integer)) or int(raw_horizon) < 0:
                raise NullCalibrationError(f"pair {pair_id} compact physics horizon is invalid")
            horizon = int(raw_horizon)
            if not quantity:
                raise NullCalibrationError(f"pair {pair_id} compact physics quantity is missing")
            path_sets.setdefault((regime, horizon), set()).add(quantity)
            per_pair_paths.setdefault((pair_id, regime, horizon), set()).add(quantity)
            physics_samples.append(sample)
        for index, raw_sample in enumerate(raw_renderer):
            if not isinstance(raw_sample, Mapping):
                raise NullCalibrationError(f"pair {pair_id} compact renderer sample {index} is malformed")
            sample = copy.deepcopy(dict(raw_sample))
            if str(sample.get("pair_id")) != pair_id or str(sample.get("trace_id")) != str(pair.trace_id):
                raise NullCalibrationError(f"pair {pair_id} compact renderer provenance is inconsistent")
            renderer_samples.append(sample)

    for (pair_id, regime, horizon), quantities in sorted(per_pair_paths.items()):
        expected = path_sets[(regime, horizon)]
        if quantities != expected:
            raise NullCalibrationError(
                f"compact quantity-selection path set is inconsistent at {pair_id}/{regime}/{horizon}"
            )
    if not physics_samples:
        raise NullCalibrationError("no compact physics quantities were observed")
    required_groups = sorted(
        {
            (hard_gate.normalize_regime(item["regime"]), str(item["quantity"]), int(item["horizon"]))
            for item in physics_samples
        }
    )
    physics = hard_gate.build_grouped_null_envelope(
        physics_samples,
        selected_trace_ids=sorted(traces),
        required_groups=required_groups,
        min_pairs=PAIR_COUNT,
        min_pairs_per_trace=5,
        min_samples_per_regime=5,
        global_tolerance=None,
    )
    if not renderer_samples:
        raise NullCalibrationError("no compact RGB observations were observed")
    renderer = hard_gate.build_renderer_envelopes(
        renderer_samples,
        exact_only=None,
        min_samples_per_group=PAIR_COUNT,
    )
    renderer_config = config.get("renderer", {}) if isinstance(config, Mapping) else {}
    configured_camera = (
        str(renderer_config.get("camera", DEFAULT_CAMERA))
        if isinstance(renderer_config, Mapping)
        else DEFAULT_CAMERA
    )
    configured_key = (
        str(renderer_config.get("observation_key", DEFAULT_RENDER_KEY))
        if isinstance(renderer_config, Mapping)
        else DEFAULT_RENDER_KEY
    )
    renderer.setdefault("coverage", {}).update(
        {
            "n_pairs": PAIR_COUNT,
            "pair_ids": sorted(required),
            "configured_camera": configured_camera,
            "configured_observation_key": configured_key,
        }
    )
    discrete = {
        "categorical_gate": all(item.discrete.get("categorical_gate", False) for item in by_id.values()),
        "contact_identity_exact": all(item.discrete.get("contact_identity_exact", False) for item in by_id.values()),
        "predicate_exact": all(item.discrete.get("predicate_exact", False) for item in by_id.values()),
        "success_exact": all(item.discrete.get("success_exact", False) for item in by_id.values()),
        "done_termination_exact": all(item.discrete.get("done_termination_exact", False) for item in by_id.values()),
        "terminal_timing_exact": all(item.discrete.get("terminal_timing_exact", False) for item in by_id.values()),
        "gripper_exact": all(item.discrete.get("gripper_exact", False) for item in by_id.values()),
        "action_exact": all(item.discrete.get("action_exact", False) for item in by_id.values()),
        "observation_exact": all(item.discrete.get("observation_exact", False) for item in by_id.values()),
        "pair_count": PAIR_COUNT,
        "divergences": [
            {"pair_id": item.pair_id, "reasons": list(item.reasons)}
            for item in by_id.values()
            if item.reasons
        ],
    }
    return EnvelopeBundle(physics=physics, renderer=renderer, discrete=discrete)


def _load_prepared_from_run_spec(run_spec_path: str | Path) -> PreparedRun:
    run_spec_target = Path(run_spec_path).resolve()
    run_spec = _load_document(run_spec_target)
    if not verify_self_hash(run_spec, "run_spec_sha256"):
        raise ProvenanceError("run specification self-hash is invalid")
    config_path = Path(run_spec["config_path"])
    config = load_config(config_path)
    source_registry_path = Path(run_spec["source_registry_path"])
    source_registry = _load_verified_registry(source_registry_path)
    pair_registry_path = Path(run_spec["pair_registry_path"])
    pair_registry = _load_document(pair_registry_path)
    trace_id = str(run_spec["trace_id"])
    tape, _ = _load_tape(source_registry, config=config, trace_id=trace_id)
    return PreparedRun(config_path, config, source_registry_path, source_registry, run_spec_target, run_spec, pair_registry_path, pair_registry, tape)


def _worker_main(job_path: str | Path, result_path: str | Path | None = None) -> int:
    job = _load_document(job_path)
    target = Path(result_path) if result_path is not None else Path(job_path).with_suffix(".result.json")
    try:
        # All hashes, paths, schedule records, and tape bindings are checked
        # before the first environment constructor can run.  The embedded job
        # config/registry are transport hints only; canonical files remain
        # authoritative.
        config, registry, tape, run_spec, _pair_registry = _verify_worker_binding(job)
        trace_id = str(run_spec["trace_id"])
        attempt = {
            **job,
            "config": config,
            "registry": registry,
            "tape": tape,
            "trace_id": trace_id,
            "action_tape_sha256": run_spec["action_tape"]["sha256"],
            "terminal_contract": copy.deepcopy(run_spec.get("terminal_contract", {})),
        }
        result = execute_attempt(attempt)
    except Exception as exc:
        result = _attempt_failure(job, exc)
    # The result file is the only machine-readable worker channel.  stdout is
    # intentionally a one-line frame so environment diagnostics cannot corrupt
    # the parent-side payload parser.
    write_json_atomic(target, result)
    sys.stdout.write(f"M1N0_RESULT {target}\n")
    return 0 if result.get("status") == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("prepare", "run", "worker"))
    parser.add_argument("--schema-version", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-spec", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    if args.schema_version:
        print(SCHEMA_VERSION)
        return 0
    mode = args.mode
    if args.prepare:
        mode = "prepare"
    if args.run:
        mode = "run"
    if args.worker:
        mode = "worker"
    if mode == "prepare":
        if args.config is None:
            parser.error("prepare requires --config")
        prepared = prepare_run(config_path=args.config)
        print(prepared.run_spec_path)
        return 0
    if mode == "run":
        if args.run_spec is not None:
            prepared = _load_prepared_from_run_spec(args.run_spec)
        elif args.config is not None:
            config = load_config(args.config)
            output_root = _resolve_path(config.get("output_root", "runs/m1_null_calibration"))
            prepared = _load_prepared_from_run_spec(output_root / "run_spec.json")
        else:
            parser.error("run requires --run-spec or --config")
        result = run_calibration(prepared)
        print(canonical_json({"status": result["status"], "terminal_manifest_path": result.get("terminal_manifest_path")}))
        return 0 if result["status"] == "PASS" else 1
    if mode == "worker":
        if args.job is None:
            parser.error("worker requires --job")
        return _worker_main(args.job, args.result)
    parser.error("select prepare, run, or worker mode")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
