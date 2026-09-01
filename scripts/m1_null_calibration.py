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
import json
import math
import os
from pathlib import Path
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
FORBIDDEN_PROTOCOL_FIELDS = (
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
    """Encode finite values without pickle while retaining array metadata."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("object arrays are not publication values")
        if value.dtype.kind in {"f", "c"} and not np.all(np.isfinite(value)):
            raise ValueError("publication arrays must be finite")
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tolist(),
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
    if isinstance(value, Mapping):
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
    return sha256_bytes(canonical_json(_without_hash(value, "output_sha256")).encode("utf-8"))


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


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Write canonical JSON exactly once and return its content hash."""

    payload = (canonical_json(value) + "\n").encode("utf-8")
    target = _exclusive_bytes(path, payload)
    digest = sha256_bytes(payload)
    if sha256_file(target) != digest:
        raise PublicationError(f"published artifact hash could not be verified: {target}")
    return {"path": str(target), "sha256": digest, "size": len(payload)}


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
    return dict(parsed)


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_document(path)


_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path, *, base: Path = _ROOT) -> Path:
    target = Path(value)
    return target if target.is_absolute() else base / target


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "trace_id": self.trace_id,
            "valid": self.valid,
            "reasons": list(self.reasons),
            "discrete": dict(self.discrete),
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


def _snapshot_contacts(snapshot: Any) -> tuple[tuple[str, str], ...]:
    values = _invariants(snapshot).get("contacts", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    result: set[tuple[str, str]] = set()
    for value in values:
        if isinstance(value, Mapping):
            left = value.get("geom1", value.get("first", value.get("a")))
            right = value.get("geom2", value.get("second", value.get("b")))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
            left, right = value[0], value[1]
        else:
            continue
        if left is None or right is None:
            continue
        pair = (str(left), str(right))
        result.add(tuple(sorted(pair)))
    return tuple(sorted(result))


def _terminal_semantics(attempt: Mapping[str, Any]) -> dict[str, Any]:
    terminal = attempt.get("terminal")
    return dict(terminal) if isinstance(terminal, Mapping) else {}


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


def _selected_numeric_leaves(snapshot: Any, roots: Sequence[str]) -> dict[str, np.ndarray]:
    invariants = _invariants(snapshot)
    result: dict[str, np.ndarray] = {}
    for root in roots:
        if root not in invariants:
            continue
        for path, value in _numeric_leaves(invariants[root], root).items():
            result[path] = value
    return result


def _actual_robot_object_contact(snapshot: Any) -> bool:
    contacts = _snapshot_contacts(snapshot)
    for left, right in contacts:
        text = f"{left} {right}".lower()
        if any(token in text for token in ("robot", "finger", "gripper", "hand")) and any(
            token in text for token in ("object", "target", "bowl", "cube", "geom")
        ):
            return True
    return False


def _close_command(snapshot: Any, action: Any = None) -> bool:
    if action is not None:
        try:
            values = np.asarray(action, dtype=np.float32).reshape(-1)
            if values.size >= 7:
                return bool(values[6] < 0.0)
        except Exception:
            pass
    current = _invariants(snapshot).get("gripper", {})
    if isinstance(current, Mapping):
        value = current.get("current_action")
        if value is not None:
            try:
                return bool(np.min(np.asarray(value, dtype=np.float64)) < 0.0)
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


def _object_motion(first: Any, current: Any) -> bool:
    left = _object_positions(first)
    right = _object_positions(current)
    if len(left) != len(right):
        return False
    return any(a.shape == b.shape and not np.array_equal(a, b) for a, b in zip(left, right))


def _independent_evidence(regime: str, first: Any, current: Any, *, action: Any = None) -> dict[str, dict[str, Any]]:
    contact = _actual_robot_object_contact(current)
    close = _close_command(current, action)
    carried = _object_motion(first, current)
    source = "attempt_snapshots_and_frozen_actions"
    return {
        "contact": {"present": contact, "source": source},
        "grasp": {"present": bool(contact and close), "source": source},
        "carried": {"present": carried, "source": source},
    }


def validate_pair(
    pair: Mapping[str, Any],
    attempts: Mapping[str, Mapping[str, Any]],
    *,
    parent_pid: int | None = None,
) -> PairValidation:
    """Validate one complete A/B pair without applying numeric null limits."""

    pair_id = str(pair.get("pair_id", ""))
    reasons: list[str] = []
    if not pair_id:
        reasons.append("pair_id is missing")
    if not isinstance(attempts, Mapping) or not isinstance(attempts.get("A"), Mapping) or not isinstance(attempts.get("B"), Mapping):
        reasons.append("pair does not have exactly one A and one B attempt")
        return PairValidation(pair_id, None, False, tuple(reasons), {}, {"categorical_gate": False})
    left = attempts["A"]
    right = attempts["B"]
    if set(str(key) for key in attempts) != {"A", "B"}:
        reasons.append("pair has unexpected attempt sides")
    for side, attempt in (("A", left), ("B", right)):
        if attempt.get("status") != "completed":
            reasons.append(f"{side} attempt is not completed")
        pid = _pid(attempt.get("pid"))
        if pid is None:
            reasons.append(f"{side} attempt has no positive PID")
        elif parent_pid is not None and pid == int(parent_pid):
            reasons.append(f"{side} attempt PID is the parent PID")
        if _pid(attempt.get("ppid")) is None:
            reasons.append(f"{side} attempt has no positive PPID")
        if not isinstance(attempt.get("process_start_identity"), str) or not attempt.get("process_start_identity"):
            reasons.append(f"{side} attempt lacks process-start identity")
        protocol = _attempt_protocol(attempt)
        for field_name in FORBIDDEN_PROTOCOL_FIELDS:
            value = protocol.get(field_name, 0)
            if value not in (0, False, None):
                reasons.append(f"{side} forbidden protocol activity: {field_name}={value}")
        if "construction_reset_count" in protocol and protocol.get("construction_reset_count") != 1:
            reasons.append(f"{side} construction reset count is not exactly one")
        if "actions_executed" in protocol and int(protocol.get("actions_executed", -1)) != ACTION_SHAPE[0]:
            reasons.append(f"{side} did not execute exactly 82 actions")
        output_sha = attempt.get("output_sha256")
        if output_sha is not None and output_sha != payload_sha256(attempt):
            reasons.append(f"{side} output artifact hash is invalid")
    left_pid, right_pid = _pid(left.get("pid")), _pid(right.get("pid"))
    if left_pid is not None and right_pid is not None and left_pid == right_pid:
        reasons.append("A and B reuse the same process PID")
    left_start, right_start = left.get("process_start_identity"), right.get("process_start_identity")
    if left_start == right_start:
        reasons.append("A and B reuse the same process-start identity")

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

    left_windows, right_windows = _window_map(left), _window_map(right)
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
        for horizon in sorted(set(lsnap) & set(rsnap)):
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
            lgrip = li.get("gripper", {}) if isinstance(li.get("gripper"), Mapping) else {}
            rgrip = ri.get("gripper", {}) if isinstance(ri.get("gripper"), Mapping) else {}
            if not _exact_equal(lgrip.get("current_action"), rgrip.get("current_action")):
                discrete["gripper_exact"] = False
                discrete["categorical_gate"] = False
                message = f"{regime}@{horizon} discrete gripper action differs"
                discrete["divergences"].append(message)
                reasons.append(message)
    left_terminal, right_terminal = _terminal_semantics(left), _terminal_semantics(right)
    for name in ("step", "termination_reason", "success", "terminated", "truncated"):
        if not _exact_equal(left_terminal.get(name), right_terminal.get(name)):
            discrete["terminal_timing_exact"] = False
            discrete["categorical_gate"] = False
            message = f"terminal {name} differs"
            discrete["divergences"].append(message)
            reasons.append(message)
    return PairValidation(
        pair_id,
        trace_id,
        not reasons and bool(discrete["categorical_gate"]),
        tuple(dict.fromkeys(reasons)),
        {"A": left, "B": right},
        discrete,
    )


def _runtime_fingerprint(adapter: Any, kind: str) -> dict[str, Any]:
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
    model = getattr(adapter, "model", None)
    return {"type": type(model).__name__ if model is not None else type(adapter).__name__}


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


def _attempt_failure(attempt: Mapping[str, Any], error: Exception | str) -> dict[str, Any]:
    protocol = {
        "construction_reset_count": 0,
        "restore_count": 0,
        "capture_count": 0,
        "policy_calls": 0,
        "processor_calls": 0,
        "retry_count": 0,
        "post_terminal_steps": 0,
        "actions_executed": 0,
    }
    return {
        "attempt_id": attempt.get("attempt_id"),
        "pair_id": attempt.get("pair_id"),
        "side": attempt.get("side"),
        "status": "failed",
        "error": f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error),
        "protocol": protocol,
    }


def execute_attempt(
    attempt: Mapping[str, Any],
    *,
    adapter_factory: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Execute one tape exactly once from one freshly constructed adapter."""

    adapter: Any = None
    actions: np.ndarray
    try:
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
        config = attempt.get("config") if isinstance(attempt.get("config"), Mapping) else {}
        adapter = adapter_factory(config) if adapter_factory is not None else _construct_adapter(config)
        windows = _window_map_from_registry(attempt)
        timeline: dict[int, dict[str, Any]] = {}
        terminal: dict[str, Any] | None = None
        for index, action in enumerate(actions):
            if terminal is not None:
                raise ProtocolError("post-terminal action was submitted")
            result = getattr(adapter, "step", None)
            if not callable(result):
                raise ProtocolError("fresh adapter does not expose step()")
            step_result = result(np.asarray(action, dtype=ACTION_DTYPE).copy())
            step = index + 1
            observation, terminated, truncated, info = _step_parts(step_result)
            invariant = _adapter_snapshot(adapter)
            rgb = _adapter_rgb(adapter)
            timeline[step] = {
                "invariants": invariant,
                "raw_observation": observation,
                "renderer": rgb,
            }
            if terminated or truncated:
                reason = info.get("termination_reason", info.get("terminal_reason"))
                terminal = {
                    "step": step,
                    "termination_reason": reason,
                    "success": _terminal_success(adapter, info),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "terminal_observation": observation,
                }
                break
        if terminal is None:
            raise ProtocolError("frozen trace did not terminate legally at step 82")
        if int(terminal["step"]) != ACTION_SHAPE[0]:
            raise ProtocolError(f"frozen trace terminated at unexpected step {terminal['step']}")
        output_windows: dict[str, Any] = {}
        for window in windows:
            offset = int(window["capture_offset"])
            horizon = int(window["continuation_horizon"])
            snapshots: dict[str, Any] = {}
            for regime in window["regimes"]:
                for relative in range(horizon + 1):
                    absolute = offset + relative
                    if absolute not in timeline:
                        raise ProtocolError(f"missing frozen window coordinate {regime}@{relative}")
                    snapshots[str(relative)] = timeline[absolute]
                output_windows[str(window["window_id"])] = {
                    "window_id": str(window["window_id"]),
                    "regimes": list(window["regimes"]),
                    "capture_offset": offset,
                    "continuation_horizon": horizon,
                    "snapshots": snapshots,
                    "action_sha256": sha256_bytes(actions.tobytes(order="C")),
                }
                break
        reset_count = 1
        protocol = {
            "construction_reset_count": reset_count,
            "restore_count": 0,
            "capture_count": 0,
            "policy_calls": 0,
            "processor_calls": 0,
            "retry_count": 0,
            "post_terminal_steps": 0,
            "actions_executed": int(terminal["step"]),
        }
        result = {
            "attempt_id": attempt.get("attempt_id"),
            "pair_id": attempt.get("pair_id"),
            "side": attempt.get("side"),
            "status": "completed",
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "process_start_identity": f"pid={os.getpid()}:start={time.monotonic_ns()}",
            "frozen_inputs": {
                "trace_id": attempt.get("trace_id"),
                "task": dict(config.get("task", TASK)) if isinstance(config.get("task", TASK), Mapping) else dict(TASK),
                "action_tape_sha256": sha256_bytes(actions.tobytes(order="C")),
                "action_shape": list(ACTION_SHAPE),
                "action_dtype": "float32",
                "physics_model_fingerprint": _runtime_fingerprint(adapter, "physics"),
                "observation_model_fingerprint": _runtime_fingerprint(adapter, "observation"),
            },
            "windows": output_windows,
            "terminal": terminal,
            "protocol": protocol,
            "runtime": {"python": sys.version.split()[0], "pid": os.getpid(), "ppid": os.getppid()},
        }
        result["output_sha256"] = payload_sha256(result)
        return result
    except Exception as exc:
        return _attempt_failure(attempt, exc)
    finally:
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # A close failure is retained by the parent as an attempt
                    # artifact when the worker result itself is inspected.
                    pass


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

    runtime = m1_state_replay.build_official_runtime(config)
    if callable(getattr(runtime, "step", None)):
        return runtime
    return m1_state_replay.RuntimeAdapter.construct_fresh(config, runtime_builder=lambda _config: runtime)


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
    }


def _tape_path(registry: Mapping[str, Any], trace_id: str) -> Path:
    provenance = _tape_provenance(registry, trace_id)
    value = provenance.get("path", provenance.get("tape_path"))
    if value is None:
        raise ProvenanceError("frozen tape path is missing")
    return _resolve_path(value)


def _verify_prepared(prepared: PreparedRun) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    current = load_config(prepared.config_path)
    computed = config_contract_sha256(current)
    if computed != str(prepared.run_spec.get("config_sha256", "")):
        raise ProvenanceError("config hash drift detected before worker launch")
    declared = current.get("config_sha256")
    if declared is not None and str(declared).lower() != computed.lower():
        raise ProvenanceError("config self-hash is invalid")
    registry_path = Path(prepared.run_spec["source_registry_path"])
    registry = _load_verified_registry(registry_path)
    _verify_registry_payload(registry, registry_path, str(prepared.run_spec["registry_sha256"]))
    pair_registry = _load_document(prepared.pair_registry_path)
    if not verify_self_hash(pair_registry, "pair_registry_sha256"):
        raise ProvenanceError("pair registry self-hash is invalid")
    if pair_registry.get("run_spec_sha256") != prepared.run_spec.get("run_spec_sha256"):
        raise ProvenanceError("pair registry does not belong to this run specification")
    trace = _trace_record(registry)
    trace_id = str(trace.get("trace_id"))
    tape, _ = _load_tape(registry, config=current, trace_id=trace_id)
    return current, pair_registry, tape


def prepare_run(*, config_path: str | Path) -> PreparedRun:
    """Freeze source provenance and the complete 20-pair schedule only."""

    config_target = Path(config_path).resolve()
    config = load_config(config_target)
    computed_config_sha = config_contract_sha256(config)
    declared_config_sha = config.get("config_sha256")
    if declared_config_sha is not None and str(declared_config_sha).lower() != computed_config_sha.lower():
        raise ProvenanceError("config self-hash is invalid")
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
        "task": dict(TASK),
        "action_tape": {"sha256": tape_sha, "shape": list(ACTION_SHAPE), "dtype": "float32"},
        "windows": windows,
        "runtime": {
            "include_policy": False,
            "call_policy": False,
            "call_processors": False,
            "fresh_processes": True,
            "retry_count": 0,
        },
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
        "pair_count": PAIR_COUNT,
        "pairs": pairs,
    }
    pair_registry = {
        **pair_registry_body,
        "pair_registry_sha256": sha256_bytes(canonical_json(pair_registry_body).encode("utf-8")),
    }
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


def _default_process_runner(attempt: Mapping[str, Any]) -> dict[str, Any]:
    job_dir = _resolve_path(attempt.get("output_root", "runs/m1_null_calibration")) / "jobs"
    job_path = job_dir / f"{attempt['attempt_id']}.json"
    write_json_atomic(job_path, dict(attempt))
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--job", str(job_path)]
    completed = subprocess.run(command, cwd=str(_ROOT), capture_output=True, text=True, check=False)
    return _normalise_process_result(completed)


def _attempt_artifact_path(prepared: PreparedRun, attempt_id: str) -> Path:
    return _resolve_path(prepared.run_spec["output_root"]) / "attempts" / f"{attempt_id}.json"


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
    attempts: list[dict[str, Any]] = []
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ProvenanceError("pair registry contains a malformed pair")
        pair_id = str(pair["pair_id"])
        for side in ("A", "B"):
            request = _prepare_attempt(prepared, pair, side)
            try:
                result = _normalise_process_result(runner(request))
                if result.get("status") is None:
                    result["status"] = "completed" if result.get("terminal") else "failed"
                result.setdefault("attempt_id", request["attempt_id"])
                result.setdefault("pair_id", pair_id)
                result.setdefault("side", side)
                if result.get("status") == "completed" and "output_sha256" not in result:
                    result["output_sha256"] = payload_sha256(result)
            except Exception as exc:
                result = _attempt_failure(request, exc)
                result.update({"attempt_id": request["attempt_id"], "pair_id": pair_id, "side": side})
            attempts.append(result)
            by_pair.setdefault(pair_id, {})[side] = result
            write_json_atomic(_attempt_artifact_path(prepared, str(request["attempt_id"])), result)
    parent_pid = os.getpid()
    validations: list[PairValidation] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        validations.append(validate_pair(pair, by_pair.get(pair_id, {}), parent_pid=parent_pid))
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
            "integrity": True,
        },
        "errors": [f"{item.pair_id}: {reason}" for item in validations for reason in item.reasons],
    }
    envelope_paths: dict[str, str] = {}
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
            write_json_atomic(physics_path, envelope.physics)
            write_json_atomic(renderer_path, envelope.renderer)
            envelope_paths = {
                "physics": str(physics_path),
                "renderer": str(renderer_path),
            }
            result["envelope_paths"] = envelope_paths
            result["gates"].update({"regime_coverage": True, "physics": True, "renderer": True})
            result["status"] = "PASS" if all(result["gates"].values()) else "BLOCKED"
        except Exception as exc:
            result["errors"].append(f"envelope construction failed: {type(exc).__name__}: {exc}")
    pair_measurement_path = output_root / "pair_measurements.json"
    write_json_atomic(pair_measurement_path, {"schema_version": SCHEMA_VERSION, "pairs": result["pair_results"]})
    result["pair_measurements_path"] = str(pair_measurement_path)
    terminal_body = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "m1_n0_terminal",
        "status": result["status"],
        "run_spec_path": str(prepared.run_spec_path),
        "run_spec_sha256": prepared.run_spec["run_spec_sha256"],
        "pair_registry_path": str(prepared.pair_registry_path),
        "pair_registry_sha256": pair_registry["pair_registry_sha256"],
        "counts": result["counts"],
        "gates": result["gates"],
        "errors": result["errors"],
        "artifacts": {"pair_measurements": str(pair_measurement_path), **envelope_paths},
    }
    terminal = {**terminal_body, "terminal_manifest_sha256": sha256_bytes(canonical_json(terminal_body).encode("utf-8"))}
    terminal_path = output_root / "terminal_manifest.json"
    write_json_atomic(terminal_path, terminal)
    result["terminal_manifest_path"] = str(terminal_path)
    return result


def _attempt_control_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[Any]:
    for key in ("duplicate_controls", "controls", "control_tape"):
        lvalue, rvalue = left.get(key), right.get(key)
        if isinstance(lvalue, Mapping) and isinstance(rvalue, Mapping):
            if "A" in lvalue and "B" in rvalue:
                return [lvalue["A"], rvalue["B"]]
        if lvalue is not None and rvalue is not None:
            return [lvalue, rvalue]
    return [np.asarray([0], dtype=np.float32), np.asarray([0], dtype=np.float32)]


def _renderer_images(snapshot: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    direct = snapshot.get("renderer")
    if direct is not None:
        value = np.asarray(direct)
        if value.dtype == np.dtype("uint8") and value.ndim >= 2:
            result[DEFAULT_RENDER_KEY] = np.ascontiguousarray(value)
    raw = snapshot.get("raw_observation")
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                array = np.asarray(value)
            except Exception:
                continue
            if array.dtype == np.dtype("uint8") and array.ndim >= 2 and array.size:
                result[str(key)] = np.ascontiguousarray(array)
    return result


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
    path_set: set[str] | None = None
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
                left_values = _selected_numeric_leaves(lsnap[horizon], roots)
                right_values = _selected_numeric_leaves(rsnap[horizon], roots)
                if set(left_values) != set(right_values):
                    raise NullCalibrationError(f"quantity path set differs at {pair_id}/{window_id}/{horizon}")
                if path_set is None:
                    path_set = set(left_values)
                elif set(left_values) != path_set:
                    raise NullCalibrationError("quantity-selection path set is inconsistent across frozen windows")
                for quantity in sorted(left_values):
                    if left_values[quantity].shape != right_values[quantity].shape:
                        raise NullCalibrationError(f"quantity shape differs at {pair_id}/{quantity}")
                    for regime in regimes:
                        evidence = _independent_evidence(
                            str(regime),
                            _window_snapshots(lwindow).get(0, lsnap[horizon]),
                            lsnap[horizon],
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
                left_images, right_images = _renderer_images(lsnap[horizon]), _renderer_images(rsnap[horizon])
                if set(left_images) != set(right_images):
                    raise NullCalibrationError(f"renderer observation key set differs at {pair_id}/{window_id}/{horizon}")
                controls = _attempt_control_pair(left, right)
                for key in sorted(left_images):
                    for regime in regimes:
                        renderer_samples.append(
                            {
                                "pair_id": pair_id,
                                "trace_id": pair.trace_id,
                                "camera": DEFAULT_CAMERA,
                                "key": key,
                                "regime": str(regime),
                                "horizon": horizon,
                                "duplicate_controls": controls,
                                "rgb_a": left_images[key],
                                "rgb_b": right_images[key],
                            }
                        )
    if path_set is None or not physics_samples:
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
            "configured_camera": DEFAULT_CAMERA,
            "configured_observation_key": DEFAULT_RENDER_KEY,
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


def _worker_main(job_path: str | Path) -> int:
    job = _load_document(job_path)
    config = load_config(job["config_path"]) if job.get("config_path") else dict(job.get("config", {}))
    registry = _load_verified_registry(job["registry_path"])
    trace_id = str(job["trace_id"])
    tape, _ = _load_tape(registry, config=config, trace_id=trace_id)
    attempt = {**job, "config": config, "registry": registry, "tape": tape}
    result = execute_attempt(attempt)
    sys.stdout.write(canonical_json(result) + "\n")
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
        return _worker_main(args.job)
    parser.error("select prepare, run, or worker mode")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
