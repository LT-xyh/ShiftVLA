#!/usr/bin/env python3
"""Standalone, policy-independent ReplayVLA M1 corpus hard gates.

This module consumes persisted M0 episode artifacts and produces only
provenance/corpus/calibration data.  It intentionally has no import path to
LeRobot, LIBERO, MuJoCo, a replay runner, a policy, or SmolVLA.  The public
functions are small seams that a later M1 runner may call after its own
runtime has produced immutable artifacts.

The implementation is fail-closed: malformed provenance, ambiguous actions,
unknown regimes, insufficient calibration coverage, and attempted registry
overwrites raise a typed error rather than silently dropping evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
ACTION_DIM = 7
MIN_NULL_PAIRS = 20
MIN_PAIRS_PER_TRACE = 5
MIN_SAMPLES_PER_REGIME = 5
MIN_PRIMARY_TRACES_PER_REGIME = 1

# These are the seven canonical persisted identifiers.  Human-readable labels
# are deliberately not accepted as alternate persisted names.
REGIME_NAMES = (
    "free_motion",
    "pre_contact",
    "contact",
    "grasp",
    "carried",
    "release",
    "predicate_transition",
)
REGIME_NAME_ALIASES = frozenset()
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_OUTCOME_KEYS = frozenset(
    {
        "success",
        "reward",
        "rewards",
        "return",
        "returns",
        "terminated",
        "truncated",
        "done",
        "outcome",
        "replay",
        "replay_outcome",
        "policy",
        "action",
        "actions",
        "action_evidence",
        "crashed",
        "status",
        "steps",
        "rate",
        "result",
        "pass",
        "fail",
        "metric",
    }
)


class M1HardGateError(RuntimeError):
    """Base error for a failed-closed hard gate."""


class ProvenanceError(M1HardGateError):
    """M0 source, identity, action, or parent-manifest evidence is invalid."""


class CorpusSelectionError(M1HardGateError):
    """A deterministic source-coverage corpus cannot be selected."""


class RegistryError(M1HardGateError):
    """A frozen registry is malformed or would be overwritten."""


class RegimeError(M1HardGateError):
    """A regime name or metadata definition is not one of the seven names."""


class NullCalibrationError(M1HardGateError):
    """Grouped null calibration has incomplete or unsafe coverage."""


class RendererCalibrationError(NullCalibrationError):
    """Renderer duplicate controls or RGB envelope evidence is invalid."""


class DiscreteGateError(M1HardGateError):
    """An exact/discrete field comparison failed."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise M1HardGateError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise M1HardGateError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise M1HardGateError(f"{name} must be finite")
    return result


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Return SHA-256 over the supplied bytes without conversions."""

    return _sha256(bytes(value))


def sha256_file(path: str | Path) -> str:
    """Hash one regular, non-symlink file exactly as persisted."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ProvenanceError(f"expected a regular file for SHA-256: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert finite NumPy values and paths to canonical JSON values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("object arrays are not canonical JSON values")
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError("canonical JSON values must be finite")
        return value.tolist()
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("canonical JSON values must be finite")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        raise TypeError(f"unsupported NumPy scalar: {type(value).__name__}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON values must be finite")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("canonical JSON mapping keys must be strings")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize finite values with stable key ordering and no whitespace."""

    try:
        return json.dumps(
            _json_safe(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise M1HardGateError(f"value is not canonical JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def _load_json(path: str | Path, *, name: str) -> Any:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ProvenanceError(f"{name} is not a regular file: {target}")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProvenanceError(f"{name} is not valid JSON: {target}: {exc}") from exc


def _mapping(value: Any, name: str, error: type[M1HardGateError] = ProvenanceError) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str, error: type[M1HardGateError] = ProvenanceError) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str, error: type[M1HardGateError] = ProvenanceError) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise error(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise error(f"{name} must be non-negative")
    return result


def _positive_int(value: Any, name: str, error: type[M1HardGateError] = ProvenanceError) -> int:
    result = _nonnegative_int(value, name, error)
    if result <= 0:
        raise error(f"{name} must be positive")
    return result


def _validate_sha(value: Any, name: str, error: type[M1HardGateError] = ProvenanceError) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise error(f"{name} must be a 64-character SHA-256 digest")
    return value.lower()


def normalize_regime(value: Any) -> str:
    """Return the one canonical identifier for a seven-regime label."""

    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise RegimeError("regime name must be a string")
    name = value
    if name not in REGIME_NAMES:
        raise RegimeError(f"unknown M1 regime: {value!r}")
    return name


class Regime(str, Enum):
    FREE_MOTION = "free_motion"
    PRE_CONTACT = "pre_contact"
    CONTACT = "contact"
    GRASP = "grasp"
    CARRIED = "carried"
    RELEASE = "release"
    PREDICATE_TRANSITION = "predicate_transition"


@dataclass(frozen=True)
class RegimeMetadata:
    """Frozen metadata describing one of the seven semantic regimes."""

    name: str
    event: str
    evidence_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        canonical = normalize_regime(self.name)
        if canonical != self.name:
            raise RegimeError(f"persisted regime must use canonical name: {self.name!r}")
        if not isinstance(self.event, str) or not self.event:
            raise RegimeError("regime event must be a non-empty string")
        fields = tuple(self.evidence_fields)
        if not fields or any(not isinstance(field, str) or not field for field in fields):
            raise RegimeError("regime evidence_fields must be non-empty strings")
        object.__setattr__(self, "evidence_fields", fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "event": self.event,
            "evidence_fields": list(self.evidence_fields),
        }


REGIME_METADATA = {
    name: RegimeMetadata(
        name=name,
        event={
            "free_motion": "free_motion_event",
            "pre_contact": "pre_contact_event",
            "contact": "contact_onset_or_sustained_event",
            "grasp": "grasp_acquisition_or_maintenance_event",
            "carried": "carried_object_motion_event",
            "release": "release_or_separation_event",
            "predicate_transition": "predicate_transition_event",
        }[name],
        evidence_fields=("state", "contact_pairs", "gripper", "object_motion"),
    )
    for name in REGIME_NAMES
}


def _has_only_float_scalars(value: Any) -> bool:
    """Check source values before NumPy can erase their scalar types."""

    if isinstance(value, np.ndarray):
        return value.dtype.kind == "f"
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_has_only_float_scalars(item) for item in value)
    return isinstance(value, (float, np.floating)) and not isinstance(value, (bool, np.bool_))


def _normalise_action(value: Any, name: str) -> np.ndarray:
    """Normalize one M0 action from either ``[7]`` or ``[[7]]`` form."""

    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ProvenanceError(f"{name} is not an array-like action") from exc
    # JSON has no dtype marker.  A cast from bool/int to float32 would erase a
    # material part of the persisted evidence, so only floating-point source
    # values are accepted here.  Check leaves as well as the inferred NumPy
    # dtype because a mixed ``[1, 0.5]`` list becomes float64 on conversion.
    if raw.dtype.kind != "f" or not _has_only_float_scalars(value):
        raise ProvenanceError(f"{name} must contain finite exact float32 evidence")
    if raw.shape == (1, ACTION_DIM):
        raw = raw[0]
    elif raw.shape != (ACTION_DIM,):
        raise ProvenanceError(f"{name} must have shape [7] or [1, 7], got {list(raw.shape)}")
    try:
        result = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"{name} must contain float32-compatible numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ProvenanceError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _actions_from_row(row: Mapping[str, Any]) -> np.ndarray:
    if "actions" not in row or "action_evidence" not in row:
        raise ProvenanceError("M0 row must persist both actions and action_evidence")
    if row.get("actions_dtype") != "float32":
        raise ProvenanceError("actions dtype must be float32")
    if row.get("actions_finite") is not True:
        raise ProvenanceError("M0 action finite evidence is false")
    raw_actions = row["actions"]
    evidence = row["action_evidence"]
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise ProvenanceError("actions must be a sequence")
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise ProvenanceError("action_evidence must be a sequence")
    if len(raw_actions) != len(evidence):
        raise ProvenanceError("action_evidence length/shape does not match actions")
    declared_shapes = row.get("actions_shape")
    if not isinstance(declared_shapes, Sequence) or isinstance(declared_shapes, (str, bytes)):
        raise ProvenanceError("actions_shape must be a sequence")
    if len(declared_shapes) != len(raw_actions):
        raise ProvenanceError("actions_shape length does not match actions")
    if "steps" not in row:
        raise ProvenanceError("M0 row must persist integer steps")
    _nonnegative_int(row.get("steps"), "steps")
    if row.get("steps") != len(raw_actions):
        raise ProvenanceError("steps does not match persisted actions")
    values: list[np.ndarray] = []
    for index, (raw, item) in enumerate(zip(raw_actions, evidence)):
        if declared_shapes is not None:
            try:
                raw_shape = list(np.asarray(raw).shape)
            except Exception as exc:
                raise ProvenanceError(f"actions[{index}] shape evidence is unreadable") from exc
            if declared_shapes[index] != raw_shape:
                raise ProvenanceError(f"actions_shape[{index}] does not match actions[{index}]")
        action = _normalise_action(raw, f"actions[{index}]")
        if not isinstance(item, Mapping):
            raise ProvenanceError(f"action_evidence[{index}] must be structured float32 evidence")
        if isinstance(item, Mapping):
            if item.get("dtype") != "float32":
                raise ProvenanceError(f"action_evidence[{index}] dtype is not float32")
            if item.get("finite") is not True:
                raise ProvenanceError(f"action_evidence[{index}] finite evidence is false")
            if "values" not in item:
                raise ProvenanceError(f"action_evidence[{index}] lacks values")
            evidence_value = item["values"]
            shape = item.get("shape")
            if shape != [1, ACTION_DIM]:
                raise ProvenanceError(f"action_evidence[{index}] shape is not [1, 7]")
            try:
                evidence_shape = list(np.asarray(item["values"]).shape)
            except Exception as exc:
                raise ProvenanceError(f"action_evidence[{index}] shape evidence is unreadable") from exc
            if shape != evidence_shape:
                raise ProvenanceError(f"action_evidence[{index}] shape does not match values")
        evidence_action = _normalise_action(evidence_value, f"action_evidence[{index}]")
        if not np.array_equal(action, evidence_action, equal_nan=False):
            raise ProvenanceError(f"action_evidence[{index}] does not exactly equal actions[{index}]")
        values.append(action)
    if not values:
        raise ProvenanceError("M0 trace has no persisted actions")
    result = np.ascontiguousarray(np.stack(values, axis=0), dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != ACTION_DIM or not result.flags.c_contiguous:
        raise ProvenanceError("actions must be C-contiguous float32[:, 7]")
    if not np.all(np.isfinite(result)):
        raise ProvenanceError("actions must contain only finite values")
    return result


def _extract_expected_action_hash(row: Mapping[str, Any]) -> str | None:
    found: list[tuple[str, str]] = []
    for key in (
        "action_sha256",
        "actions_sha256",
        "actions_hash",
        "action_hash",
        "action_tape_sha256",
    ):
        if key in row and row[key] is not None:
            found.append((key, _validate_sha(row[key], key)))
    if not found:
        return None
    if len({value for _, value in found}) != 1:
        raise ProvenanceError("persisted action hash fields disagree")
    return found[0][1]


def _find_manifest_paths(
    *,
    parent_run_dir: str | Path | None,
    parent_manifest_path: str | Path | None,
    terminal_manifest_path: str | Path | None,
) -> tuple[Path, Path, Path]:
    if parent_run_dir is None and parent_manifest_path is None:
        raise ProvenanceError(
            "parent_run_dir or parent_manifest_path is required; manifest discovery is not implicit"
        )
    if parent_manifest_path is not None:
        run_manifest = Path(parent_manifest_path)
        run_dir = run_manifest.parent
        if parent_run_dir is not None and Path(parent_run_dir).resolve() != run_dir.resolve():
            raise ProvenanceError("parent_run_dir does not match parent_manifest_path")
    elif parent_run_dir is not None:
        run_dir = Path(parent_run_dir)
        run_manifest = run_dir / "run_manifest.json"
    terminal = Path(terminal_manifest_path) if terminal_manifest_path is not None else run_dir / "terminal_manifest.json"
    if terminal.resolve().parent != run_dir.resolve():
        raise ProvenanceError("terminal_manifest_path must be inside the parent run directory")
    if run_manifest.is_symlink() or terminal.is_symlink() or not run_manifest.is_file() or not terminal.is_file():
        raise ProvenanceError("parent run and terminal manifests must be regular files")
    return run_dir.resolve(), run_manifest.resolve(), terminal.resolve()


def _manifest_run_directory(value: Any, base: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve()
    except OSError:
        return path


def _validate_parent_manifest(
    row: Mapping[str, Any],
    *,
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    terminal_manifest: Mapping[str, Any],
    allow_relocated_run_directory: bool = False,
    declared_original_run_directory: str | Path | None = None,
) -> None:
    expected_run = run_dir.resolve()
    declared_original = None
    if declared_original_run_directory is not None:
        declared_original = Path(declared_original_run_directory).resolve()
    relocation_seen = False
    for name, manifest in (("run", run_manifest), ("terminal", terminal_manifest)):
        declared = _manifest_run_directory(manifest.get("run_directory"), run_dir)
        if declared is not None and declared != expected_run:
            if not allow_relocated_run_directory or declared_original is None or declared != declared_original:
                raise ProvenanceError(f"parent {name} manifest run_directory does not match source run")
            relocation_seen = True
    if allow_relocated_run_directory and not relocation_seen:
        raise ProvenanceError("relocation mode requires manifests to carry the declared original run directory")
    if terminal_manifest.get("terminal") is not True:
        raise ProvenanceError("parent terminal manifest is not terminal")
    run_status = run_manifest.get("status")
    terminal_status = terminal_manifest.get("status")
    allowed_statuses = {"pass", "completed", "provenance_closed", "closed", "success"}
    if not isinstance(run_status, str) or run_status.lower() not in allowed_statuses:
        raise ProvenanceError(f"parent run manifest status is not complete: {run_status}")
    if not isinstance(terminal_status, str) or terminal_status.lower() not in allowed_statuses:
        raise ProvenanceError(f"parent terminal manifest status is not complete: {terminal_status}")
    episode_id = _string(row.get("episode_id"), "episode_id")
    scope = run_manifest.get("scope")
    if not isinstance(scope, Mapping):
        raise ProvenanceError("parent run manifest must persist an episode scope")
    for key in ("episode_ids", "trajectory_ids"):
        values = scope.get(key)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(not isinstance(item, str) for item in values)
            or episode_id not in values
        ):
            raise ProvenanceError(f"parent scope does not contain {episode_id} in {key}")
    provenance = run_manifest.get("provenance")
    if not isinstance(provenance, Mapping) or "seed" not in provenance:
        raise ProvenanceError("parent run manifest must persist provenance seed")
    if provenance["seed"] != row.get("seed"):
        raise ProvenanceError("M0 row seed does not match parent provenance seed")
    # Some closure writers carry a digest in the opposite manifest.  When it
    # exists, validate it, but do not invent one for older M0 manifests.
    for manifest, aliases, name in (
        (run_manifest, ("terminal_manifest_sha256", "terminal_sha256"), "terminal manifest"),
        (terminal_manifest, ("run_manifest_sha256", "parent_run_manifest_sha256", "run_sha256"), "run manifest"),
    ):
        for key in aliases:
            if key in manifest and manifest[key] is not None:
                _validate_sha(manifest[key], f"{name} field {key}")


def _validate_matrix_identity(row: Mapping[str, Any], run_dir: Path) -> None:
    matrix_path = run_dir / "episode_matrix.json"
    if not matrix_path.is_file():
        raise ProvenanceError("episode matrix is required for persisted M0 provenance")
    matrix = _load_json(matrix_path, name="episode matrix")
    if isinstance(matrix, Mapping):
        matrix = matrix.get("rows")
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise ProvenanceError("episode matrix must be a sequence")
    episode_id = row.get("episode_id")
    matches = [item for item in matrix if isinstance(item, Mapping) and item.get("episode_id") == episode_id]
    if len(matches) != 1:
        raise ProvenanceError("episode matrix must contain exactly one matching episode")
    matrix_row = matches[0]
    for key in ("trajectory_id", "suite", "task_id", "task_name", "init_state_id", "seed"):
        if key in matrix_row and matrix_row.get(key) != row.get(key):
            raise ProvenanceError(f"episode matrix {key} does not match M0 row")


def _validate_source_layout(source_path: Path, run_dir: Path) -> Path:
    """Require the persisted M0 row to live in the canonical child layout."""

    try:
        resolved_source = source_path.resolve()
        resolved_run = run_dir.resolve()
        relative = resolved_source.relative_to(resolved_run)
    except (OSError, ValueError) as exc:
        raise ProvenanceError("M0 row must be inside the parent run directory") from exc
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != "children"
        or not parts[1]
        or parts[2] != "episode_rows"
        or not parts[3]
        or Path(parts[3]).suffix != ".json"
    ):
        raise ProvenanceError(
            "M0 row must use parent_run_dir/children/<worker>/episode_rows/<row>.json layout"
        )
    return resolved_source


def _project_root_for_run(run_dir: Path) -> Path | None:
    """Infer the project root used by relative closure inventory paths."""

    resolved = run_dir.resolve()
    parts = resolved.parts
    try:
        index = parts.index("runs")
    except ValueError:
        return None
    if index == 0:
        return Path(resolved.anchor or os.sep)
    return Path(*parts[:index])


def _closure_path_matches(value: Any, source_path: Path, run_dir: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    declared = Path(value)
    candidates = [declared]
    if not declared.is_absolute():
        candidates.extend((run_dir / declared, run_dir.parent / declared))
        project_root = _project_root_for_run(run_dir)
        if project_root is not None:
            candidates.append(project_root / declared)
    try:
        expected = source_path.resolve()
    except OSError:
        expected = source_path
    for candidate in candidates:
        try:
            if candidate.resolve() == expected:
                return True
        except OSError:
            continue
    return False


def _validate_closure_inventory(
    *,
    terminal_manifest: Mapping[str, Any],
    source_path: Path,
    source_sha256: str,
    run_dir: Path,
) -> None:
    closure = terminal_manifest.get("post_run_provenance_closure")
    if not isinstance(closure, Mapping):
        raise ProvenanceError("parent terminal manifest lacks post-run provenance closure")
    artifacts = closure.get("source_artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ProvenanceError("post-run provenance closure source_artifacts must be a sequence")
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        if not _closure_path_matches(item.get("path"), source_path, run_dir):
            continue
        declared_sha = _validate_sha(item.get("sha256"), "closure source artifact SHA-256")
        if declared_sha != source_sha256:
            raise ProvenanceError("closure source artifact SHA-256 does not match persisted row")
        if "size_bytes" in item:
            size = item["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) != source_path.stat().st_size:
                raise ProvenanceError("closure source artifact size does not match persisted row")
        return
    raise ProvenanceError("M0 row is absent from the post-run provenance closure inventory")


def _resolve_pinned_hash(
    name: str,
    *values: Any,
    error: type[M1HardGateError] = ProvenanceError,
) -> str:
    present = [value for value in values if value is not None]
    if not present:
        raise error(f"caller must provide pinned {name}")
    normalized = {_validate_sha(value, f"pinned {name}", error) for value in present}
    if len(normalized) != 1:
        raise error(f"caller supplied conflicting pinned {name} values")
    return normalized.pop()


@dataclass(frozen=True)
class M0Trace:
    """Validated immutable M0 row plus complete source/tape provenance."""

    episode_id: str
    suite: str
    task_id: int
    init_state_id: int
    seed: int
    task_name: str
    actions: np.ndarray
    source_file: str
    source_file_sha256: str
    parent_run_manifest: str
    parent_run_manifest_sha256: str
    parent_terminal_manifest: str
    parent_terminal_manifest_sha256: str
    action_sha256: str
    success: bool = True
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _string(self.episode_id, "trace episode_id")
        _string(self.suite, "trace suite")
        _nonnegative_int(self.task_id, "trace task_id")
        _nonnegative_int(self.init_state_id, "trace init_state_id")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ProvenanceError("trace seed must be an integer")
        _string(self.task_name, "trace task_name")
        value = np.asarray(self.actions)
        if value.dtype != np.dtype("float32") or value.ndim != 2 or value.shape[1] != ACTION_DIM:
            raise ProvenanceError("trace actions must be float32[:, 7]")
        if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
            raise ProvenanceError("trace actions must be finite and C-contiguous")
        value = np.ascontiguousarray(value).copy()
        value.setflags(write=False)
        object.__setattr__(self, "actions", value)
        for key in (
            "source_file_sha256",
            "parent_run_manifest_sha256",
            "parent_terminal_manifest_sha256",
            "action_sha256",
        ):
            _validate_sha(getattr(self, key), f"trace {key}")
        if _sha256(value.tobytes(order="C")) != self.action_sha256.lower():
            raise ProvenanceError("trace action_sha256 does not match exact float32 action bytes")
        if self.success is not True:
            raise ProvenanceError("M0 trace must be successful")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def trajectory_id(self) -> str:
        return self.episode_id

    def provenance(self) -> dict[str, Any]:
        result = {
            "trace_id": self.episode_id,
            "episode_id": self.episode_id,
            "trajectory_id": self.trajectory_id,
            "suite": self.suite,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "init_state_id": self.init_state_id,
            "seed": int(self.seed),
            "source_file": self.source_file,
            "source_file_sha256": self.source_file_sha256,
            "parent_run_manifest": self.parent_run_manifest,
            "parent_run_manifest_sha256": self.parent_run_manifest_sha256,
            "parent_terminal_manifest": self.parent_terminal_manifest,
            "parent_terminal_manifest_sha256": self.parent_terminal_manifest_sha256,
            "action_sha256": self.action_sha256,
            "action_shape": list(self.actions.shape),
            "action_dtype": "float32",
        }
        relocation = self.metadata.get("_m1_relocation") if isinstance(self.metadata, Mapping) else None
        if isinstance(relocation, Mapping):
            result["relocation"] = dict(relocation)
        return result

    def __getitem__(self, key: str) -> Any:
        return self.provenance()[key]

    def to_dict(self) -> dict[str, Any]:
        return self.provenance()


def extract_m0_trace(
    source: str | Path | Mapping[str, Any],
    *,
    parent_run_dir: str | Path | None = None,
    parent_manifest_path: str | Path | None = None,
    terminal_manifest_path: str | Path | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    expected_source_sha256: str | None = None,
    expected_source_file_sha256: str | None = None,
    source_sha256: str | None = None,
    source_file_sha256: str | None = None,
    expected_parent_run_manifest_sha256: str | None = None,
    expected_parent_manifest_sha256: str | None = None,
    parent_run_sha256: str | None = None,
    parent_manifest_sha256: str | None = None,
    expected_terminal_manifest_sha256: str | None = None,
    expected_parent_terminal_manifest_sha256: str | None = None,
    terminal_sha256: str | None = None,
    parent_terminal_sha256: str | None = None,
    expected_action_sha256: str | None = None,
    pinned_hashes: Mapping[str, Any] | None = None,
    expected_hashes: Mapping[str, Any] | None = None,
    allow_relocated_run_directory: bool = False,
    declared_original_run_directory: str | Path | None = None,
) -> M0Trace:
    """Load and validate one persisted M0 episode row.

    ``source`` is normally ``children/*/episode_rows/*.json``.  Callers must
    supply the persisted parent run directory or explicit manifest paths;
    manifest discovery from unrelated ancestors is intentionally forbidden.
    The parent run and terminal SHA-256 values always refer to the exact file
    bytes read from disk.
    """

    if isinstance(source, Mapping):
        raise ProvenanceError("M0 source must be a persisted episode-row path, not a mapping")
    if not isinstance(source, (str, Path)):
        raise ProvenanceError("M0 source must be a persisted episode-row path")

    run_dir, run_manifest_path, terminal_path = _find_manifest_paths(
        parent_run_dir=parent_run_dir,
        parent_manifest_path=parent_manifest_path,
        terminal_manifest_path=terminal_manifest_path,
    )
    source_path = Path(source)
    # A relative source is interpreted relative to the caller's current
    # project, as ordinary filesystem APIs do.  If it is not found there,
    # accepting the explicit parent-run-relative form is useful while keeping
    # the subsequent containment/layout check strict.
    if not source_path.is_absolute() and not source_path.exists():
        source_path = run_dir / source_path
    source_path = _validate_source_layout(source_path, run_dir)
    row = _load_json(source_path, name="M0 episode row")
    source_sha = sha256_file(source_path)
    pinned_values: list[Mapping[str, Any]] = []
    if pinned_hashes is not None:
        pinned_values.append(_mapping(pinned_hashes, "pinned_hashes", ProvenanceError))
    if expected_hashes is not None:
        pinned_values.append(_mapping(expected_hashes, "expected_hashes", ProvenanceError))

    def pinned_for(*keys: str) -> list[Any]:
        return [mapping[key] for mapping in pinned_values for key in keys if key in mapping]

    expected_source = _resolve_pinned_hash(
        "source file SHA-256",
        expected_source_sha256,
        expected_source_file_sha256,
        source_sha256,
        source_file_sha256,
        *pinned_for("source_sha256", "source_sha", "source_file_sha256", "source_file"),
    )
    expected_parent = _resolve_pinned_hash(
        "parent run manifest SHA-256",
        expected_parent_run_manifest_sha256,
        expected_parent_manifest_sha256,
        parent_run_sha256,
        parent_manifest_sha256,
        *pinned_for(
            "parent_run_manifest_sha256",
            "parent_manifest_sha256",
            "run_manifest_sha256",
            "parent_run_manifest",
        ),
    )
    expected_terminal = _resolve_pinned_hash(
        "parent terminal manifest SHA-256",
        expected_terminal_manifest_sha256,
        expected_parent_terminal_manifest_sha256,
        terminal_sha256,
        parent_terminal_sha256,
        *pinned_for(
            "parent_terminal_manifest_sha256",
            "terminal_manifest_sha256",
            "terminal_sha256",
            "terminal_manifest",
            "parent_terminal_manifest",
        ),
    )
    if expected_source != source_sha:
        raise ProvenanceError("pinned source file SHA-256 does not match persisted bytes")
    run_manifest_sha = sha256_file(run_manifest_path)
    terminal_manifest_sha = sha256_file(terminal_path)
    if expected_parent != run_manifest_sha:
        raise ProvenanceError("pinned parent run manifest SHA-256 does not match persisted bytes")
    if expected_terminal != terminal_manifest_sha:
        raise ProvenanceError("pinned parent terminal manifest SHA-256 does not match persisted bytes")
    row = _mapping(row, "M0 episode row")
    source_sha = _validate_sha(source_sha, "source file SHA-256")
    expected_source_sha = row.get("source_file_sha256", row.get("source_file_sha"))
    if expected_source_sha is not None and _validate_sha(expected_source_sha, "row source_file_sha256") != source_sha:
        raise ProvenanceError("source file SHA-256 does not match persisted row")

    episode_id = _string(row.get("episode_id"), "episode_id")
    trajectory_id = row.get("trajectory_id", episode_id)
    if trajectory_id != episode_id:
        raise ProvenanceError("trajectory_id does not match episode_id")
    suite = _string(row.get("suite"), "suite")
    task_id = _nonnegative_int(row.get("task_id"), "task_id")
    init_state_id = _nonnegative_int(row.get("init_state_id"), "init_state_id")
    if isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), (int, np.integer)):
        raise ProvenanceError("seed must be an integer")
    seed = int(row["seed"])
    task_name = _string(row.get("task_name", ""), "task_name")
    if row.get("completed") is not None and row.get("completed") is not True:
        raise ProvenanceError("M0 row is not completed")
    if row.get("status") != "completed":
        raise ProvenanceError("M0 row status must be completed")
    if row.get("success") is not True:
        raise ProvenanceError("M0 row success must be true")
    if row.get("crashed") is not None and row.get("crashed") is not False:
        raise ProvenanceError("M0 row crashed flag must be false")
    if row.get("attempt_count") != 1:
        raise ProvenanceError("M0 row attempt_count must be exactly 1")
    if row.get("no_retry") is not True:
        raise ProvenanceError("M0 row no_retry must be true")
    source_evidence = row.get("task_source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise ProvenanceError("M0 row must persist structured task_source_evidence")
    for key, expected in (("suite", suite), ("task_id", task_id), ("task_name", task_name)):
        if key not in source_evidence or source_evidence[key] != expected:
            raise ProvenanceError(f"task_source_evidence.{key} does not match row")
    if expected_identity:
        identity = dict(expected_identity)
        nested_task = identity.get("task")
        if isinstance(nested_task, Mapping):
            identity = {**identity, **nested_task}
        actual = {
            "episode_id": episode_id,
            "trajectory_id": episode_id,
            "suite": suite,
            "task_id": task_id,
            "task_name": task_name,
            "init_state_id": init_state_id,
            "seed": seed,
        }
        for key, expected in identity.items():
            if key in actual and actual[key] != expected:
                raise ProvenanceError(f"trace identity {key} does not match expected identity")

    actions = _actions_from_row(row)
    action_sha = _sha256(actions.tobytes(order="C"))
    expected_action_sha = _extract_expected_action_hash(row)
    pinned_action = [value for value in (expected_action_sha256, *pinned_for("action_sha256", "action_file_sha256")) if value is not None]
    if expected_action_sha is None and not pinned_action:
        raise ProvenanceError("M0 row must persist an action SHA-256 or caller must provide a pinned action SHA-256")
    if expected_action_sha is not None and expected_action_sha != action_sha:
        raise ProvenanceError("action hash does not match exact float32 action bytes")
    if pinned_action:
        expected_action_pin = _resolve_pinned_hash("action SHA-256", *pinned_action)
        if expected_action_pin != action_sha:
            raise ProvenanceError("pinned action SHA-256 does not match exact float32 action bytes")
    run_manifest = _load_json(run_manifest_path, name="parent run manifest")
    terminal_manifest = _load_json(terminal_path, name="parent terminal manifest")
    run_manifest = _mapping(run_manifest, "parent run manifest")
    terminal_manifest = _mapping(terminal_manifest, "parent terminal manifest")
    _validate_parent_manifest(
        row,
        run_dir=run_dir,
        run_manifest=run_manifest,
        terminal_manifest=terminal_manifest,
        allow_relocated_run_directory=allow_relocated_run_directory,
        declared_original_run_directory=declared_original_run_directory,
    )
    _validate_matrix_identity(row, run_dir)
    if not re.fullmatch(r"\d+-" + re.escape(episode_id) + r"\.json", source_path.name):
        raise ProvenanceError("M0 row filename does not match episode_id")
    _validate_closure_inventory(
        terminal_manifest=terminal_manifest,
        source_path=source_path,
        source_sha256=source_sha,
        run_dir=run_dir,
    )
    for manifest, keys, actual, label in (
        (
            run_manifest,
            ("run_manifest_sha256", "manifest_sha256", "parent_run_manifest_sha256"),
            run_manifest_sha,
            "parent run manifest",
        ),
        (
            terminal_manifest,
            ("terminal_manifest_sha256", "manifest_sha256", "parent_terminal_manifest_sha256"),
            terminal_manifest_sha,
            "parent terminal manifest",
        ),
    ):
        candidates = [manifest]
        nested = manifest.get("provenance")
        if isinstance(nested, Mapping):
            candidates.append(nested)
        for candidate in candidates:
            for key in keys:
                if key in candidate and candidate[key] is not None:
                    if _validate_sha(candidate[key], f"{label} {key}") != actual:
                        raise ProvenanceError(f"{label} hash field {key} does not match persisted bytes")
    metadata = dict(row)
    if allow_relocated_run_directory:
        metadata["_m1_relocation"] = {
            "allowed": True,
            "declared_original_run_directory": str(Path(declared_original_run_directory).resolve())
            if declared_original_run_directory is not None
            else None,
            "current_run_directory": str(run_dir.resolve()),
            "reason": "content-addressed M0 bundle relocation",
        }
    return M0Trace(
        episode_id=episode_id,
        suite=suite,
        task_id=task_id,
        init_state_id=init_state_id,
        seed=seed,
        task_name=task_name,
        actions=actions,
        source_file=str(source_path),
        source_file_sha256=source_sha,
        parent_run_manifest=str(run_manifest_path),
        parent_run_manifest_sha256=run_manifest_sha,
        parent_terminal_manifest=str(terminal_path),
        parent_terminal_manifest_sha256=terminal_manifest_sha,
        action_sha256=action_sha,
        success=True,
        metadata=metadata,
    )

def _reject_outcomes(value: Any, path: str = "selection") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in _OUTCOME_KEYS or any(token in key_text for token in ("outcome", "reward", "replay")):
                raise CorpusSelectionError(f"selection input contains outcome field {path}.{key}")
            _reject_outcomes(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_outcomes(item, f"{path}[{index}]")


def _coverage_regimes(
    coverage: Mapping[str, Any],
    fallback: str | None = None,
    *,
    error: type[M1HardGateError] = CorpusSelectionError,
) -> tuple[str, ...]:
    values: Any = coverage.get("regimes", coverage.get("covered_regimes"))
    if values is None and "regime" in coverage:
        values = [coverage["regime"]]
    if values is None and fallback is not None:
        values = [fallback]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise CorpusSelectionError("source_coverage must name one or more regimes")
    try:
        names = tuple(dict.fromkeys(normalize_regime(item) for item in values))
    except RegimeError as exc:
        raise error(f"source_coverage contains an unknown regime: {exc}") from exc
    if not names:
        raise CorpusSelectionError("source_coverage has no regimes")
    return names


@dataclass(frozen=True)
class _Candidate:
    trace_id: str
    regimes: tuple[str, ...]
    role: str
    source_coverage: Mapping[str, Any]


def _candidate(
    value: Any,
    *,
    fallback_regime: str | None,
    role: str,
    coverage_override: Mapping[str, Any] | None = None,
) -> _Candidate:
    if isinstance(value, M0Trace):
        coverage = dict(coverage_override or ({"regime": fallback_regime} if fallback_regime else {}))
        if not coverage:
            raise CorpusSelectionError(
                "M0Trace selection requires an explicit source_coverage candidate; "
                "outcomes are not selection inputs"
            )
        return _Candidate(value.episode_id, _coverage_regimes(coverage, fallback_regime), role, coverage)
    # A mapping is only a transport wrapper around an already validated trace.
    # Episode IDs or provenance-looking dictionaries are not evidence: callers
    # must pass the M0Trace returned by extract_m0_trace.
    if isinstance(value, Mapping):
        validated = value.get("m0_trace", value.get("trace"))
        if not isinstance(validated, M0Trace):
            raise CorpusSelectionError("corpus candidates must be validated M0Trace objects")
        _reject_outcomes({key: item for key, item in value.items() if key not in {"m0_trace", "trace"}})
        item = value
        value = validated
    else:
        raise CorpusSelectionError("corpus candidates must be validated M0Trace objects")
    if isinstance(value, M0Trace):
        trace_id = value.episode_id
        coverage = coverage_override or (item.get("source_coverage") if isinstance(item, Mapping) else None)
        if coverage is None:
            coverage = {"regime": fallback_regime} if fallback_regime else {}
        coverage = _mapping(coverage, "candidate source_coverage", CorpusSelectionError)
        regimes = _coverage_regimes(coverage, fallback_regime)
        declared_role = item.get("role", role) if isinstance(item, Mapping) else role
        if not isinstance(declared_role, str):
            raise CorpusSelectionError("candidate role must be a string")
        normalized_role = declared_role.lower()
        if role == "primary" and normalized_role not in {"primary", "required"}:
            raise CorpusSelectionError(f"primary candidate {trace_id} has non-primary role {declared_role}")
        return _Candidate(trace_id, regimes, declared_role, dict(coverage))

    # The branch is intentionally unreachable; retaining a single return path
    # makes the type guard above explicit to static checkers.
    raise CorpusSelectionError("corpus candidates must be validated M0Trace objects")


@dataclass(frozen=True)
class CorpusSelection:
    """Deterministic source-only minimum corpus selection."""

    trace_ids: tuple[str, ...]
    regimes: tuple[str, ...]
    source_coverage: Mapping[str, tuple[str, ...]]
    primary_trace_ids: tuple[str, ...]
    supplement_trace_ids: tuple[str, ...]
    source_coverage_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_ids": list(self.trace_ids),
            "regimes": list(self.regimes),
            "source_coverage": {key: list(value) for key, value in self.source_coverage.items()},
            "primary_trace_ids": list(self.primary_trace_ids),
            "supplement_trace_ids": list(self.supplement_trace_ids),
            "source_coverage_only": self.source_coverage_only,
        }


def _iter_candidates(value: Any, *, default_role: str) -> list[tuple[str | None, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        result: list[tuple[str | None, Any]] = []
        for key, items in value.items():
            if key in REGIME_NAMES or key in REGIME_NAME_ALIASES:
                regime = normalize_regime(key)
                if isinstance(items, (str, Mapping, M0Trace)):
                    items = [items]
                if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                    raise CorpusSelectionError(f"candidate list for {regime} must be a sequence")
                result.extend((regime, item) for item in items)
            else:
                # A mapping keyed by trace ID is accepted only when each value
                # itself carries source coverage; this keeps IDs deterministic
                # without allowing outcome fields.
                if isinstance(items, M0Trace):
                    result.append((None, items))
                elif isinstance(items, Mapping):
                    item = dict(items)
                    item.setdefault("episode_id", str(key))
                    result.append((None, item))
                else:
                    raise CorpusSelectionError(f"unknown candidate mapping key: {key}")
        return result
    if isinstance(value, (str, Mapping, M0Trace)):
        return [(None, value)]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise CorpusSelectionError("candidate collection must be a sequence or regime mapping")
    return [(None, item) for item in value]


def _minimum_set_cover(
    candidates: Sequence[_Candidate],
    *,
    minimum_per_regime: int | Sequence[int],
    supplement_ids: set[str],
) -> tuple[_Candidate, ...] | None:
    """Return the exact minimum-cardinality quota cover.

    The candidate count for the frozen pilot is small, so exhaustive
    branch-and-bound is preferable to a greedy cover: a multi-regime trace can
    make a greedy prefix strictly larger than the true minimum.  IDs provide
    the complete deterministic tie-break after cardinality and supplement
    count.
    """

    ordered = tuple(sorted(candidates, key=lambda item: item.trace_id))
    if not ordered:
        return None
    regime_index = {name: index for index, name in enumerate(REGIME_NAMES)}
    if isinstance(minimum_per_regime, int):
        target = tuple(minimum_per_regime for _ in REGIME_NAMES)
    else:
        target = tuple(minimum_per_regime)
        if len(target) != len(REGIME_NAMES) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in target
        ):
            raise CorpusSelectionError("set-cover quotas must be seven non-negative integers")
    if all(item == 0 for item in target):
        return ()
    coverage_masks: list[tuple[int, ...]] = []
    for candidate in ordered:
        coverage_masks.append(tuple(regime_index[name] for name in candidate.regimes))
    suffix_possible = [set() for _ in range(len(ordered) + 1)]
    for index in range(len(ordered) - 1, -1, -1):
        suffix_possible[index] = suffix_possible[index + 1] | {
            REGIME_NAMES[regime_index_value] for regime_index_value in coverage_masks[index]
        }

    best: tuple[tuple[int, int, tuple[str, ...]], tuple[_Candidate, ...]] | None = None
    counts = [0] * len(REGIME_NAMES)

    def visit(index: int, chosen: list[_Candidate]) -> None:
        nonlocal best
        if all(count >= need for count, need in zip(counts, target)):
            ids = tuple(sorted(item.trace_id for item in chosen))
            supplement_count = sum(item in supplement_ids for item in ids)
            key = (len(chosen), supplement_count, ids)
            if best is None or key < best[0]:
                best = (key, tuple(chosen))
            return
        if index >= len(ordered):
            return
        if any(
            count < need and regime not in suffix_possible[index]
            for regime, count, need in zip(REGIME_NAMES, counts, target)
        ):
            return
        if best is not None and len(chosen) > best[0][0]:
            return

        candidate = ordered[index]
        for regime_index_value in coverage_masks[index]:
            counts[regime_index_value] += 1
        chosen.append(candidate)
        visit(index + 1, chosen)
        chosen.pop()
        for regime_index_value in coverage_masks[index]:
            counts[regime_index_value] -= 1
        visit(index + 1, chosen)

    visit(0, [])
    return None if best is None else best[1]


def select_minimum_corpus(
    primary_candidates: Any = None,
    preregistered_supplements: Any = None,
    *,
    minimum_per_regime: int = MIN_PRIMARY_TRACES_PER_REGIME,
    source_coverage: Mapping[str, Any] | None = None,
) -> CorpusSelection:
    """Select a minimum source-coverage corpus with primary precedence.

    For every regime, lexicographically smallest primary episode IDs fill the
    quota first.  Only an explicitly preregistered supplement collection may
    fill a primary shortfall.  The selector never reads an episode outcome,
    action, reward, replay, or status field; passing one is rejected.
    """

    if source_coverage is not None:
        source_coverage = _mapping(source_coverage, "source_coverage", CorpusSelectionError)
        _reject_outcomes(source_coverage, path="source_coverage")
    if isinstance(minimum_per_regime, bool) or not isinstance(minimum_per_regime, int) or minimum_per_regime <= 0:
        raise CorpusSelectionError("minimum_per_regime must be a positive integer")
    primary_items = _iter_candidates(primary_candidates, default_role="primary")
    supplement_items = _iter_candidates(preregistered_supplements, default_role="supplement")
    primary: list[_Candidate] = []
    supplements: list[_Candidate] = []
    for fallback, item in primary_items:
        if isinstance(item, str) and source_coverage is not None:
            item = {"episode_id": item, "source_coverage": source_coverage.get(item, {"regime": fallback})}
        override = None
        if isinstance(item, M0Trace) and source_coverage is not None:
            candidate_coverage = source_coverage.get(item.episode_id)
            if isinstance(candidate_coverage, Mapping):
                override = candidate_coverage
        primary.append(_candidate(item, fallback_regime=fallback, role="primary", coverage_override=override))
    for fallback, item in supplement_items:
        if isinstance(item, str):
            item = {
                "episode_id": item,
                "source_coverage": source_coverage.get(item, {"regime": fallback})
                if source_coverage is not None
                else {"regime": fallback},
                "preregistered": True,
            }
        override = None
        if isinstance(item, M0Trace) and source_coverage is not None:
            candidate_coverage = source_coverage.get(item.episode_id)
            if isinstance(candidate_coverage, Mapping):
                override = candidate_coverage
        parsed = _candidate(item, fallback_regime=fallback, role="supplement", coverage_override=override)
        explicit_preregistered = isinstance(item, Mapping) and (
            item.get("preregistered") is True or str(item.get("role", "")).lower() in {"supplement", "preregistered"}
        )
        # The named ``preregistered_supplements`` argument is itself a
        # preregistration boundary, but a conflicting explicit role is not.
        if isinstance(item, Mapping) and not explicit_preregistered:
            raise CorpusSelectionError(f"supplement {parsed.trace_id} is not preregistered")
        if isinstance(item, str):
            raise CorpusSelectionError(f"supplement {parsed.trace_id} is not preregistered")
        supplements.append(parsed)
    all_candidates = primary + supplements
    by_id: dict[str, _Candidate] = {}
    for item in all_candidates:
        previous = by_id.get(item.trace_id)
        if previous is not None and (previous.regimes != item.regimes or previous.role != item.role):
            raise CorpusSelectionError(f"candidate {item.trace_id} has conflicting source coverage/role")
        by_id[item.trace_id] = item
    primary_by_id = {item.trace_id: item for item in primary}
    supplement_by_id = {item.trace_id: item for item in supplements}
    primary_unique = tuple(primary_by_id.values())
    primary_available = {
        regime: sum(regime in item.regimes for item in primary_unique) for regime in REGIME_NAMES
    }
    primary_quota = tuple(min(minimum_per_regime, primary_available[regime]) for regime in REGIME_NAMES)
    selected_primary = _minimum_set_cover(
        primary_unique,
        minimum_per_regime=primary_quota,
        supplement_ids=set(),
    )
    if selected_primary is None:
        raise CorpusSelectionError("primary source-coverage quotas cannot be selected")
    primary_counts = {regime: 0 for regime in REGIME_NAMES}
    for item in selected_primary:
        for regime in item.regimes:
            primary_counts[regime] += 1
    supplement_quota = tuple(
        max(0, minimum_per_regime - primary_counts[regime]) for regime in REGIME_NAMES
    )
    selected_supplements = _minimum_set_cover(
        tuple(supplement_by_id.values()),
        minimum_per_regime=supplement_quota,
        supplement_ids=set(supplement_by_id),
    )
    if selected_supplements is None:
        missing = []
        all_covered = {
            regime
            for item in (*selected_primary, *supplement_by_id.values())
            for regime in item.regimes
        }
        missing.extend(
            regime
            for regime in REGIME_NAMES
            if regime not in all_covered or primary_counts[regime] < minimum_per_regime
        )
        detail = f"; missing regimes={missing}" if missing else ""
        raise CorpusSelectionError(
            f"regime quotas lack {minimum_per_regime} validated source-coverage traces{detail}; "
            "non-preregistered supplements cannot fill the gap"
        )
    selected_items = (*selected_primary, *selected_supplements)
    selected = {item.trace_id: item for item in selected_items}
    trace_ids = tuple(sorted(selected))
    primary_ids = tuple(sorted(set(selected) & set(primary_by_id)))
    supplement_ids = tuple(sorted(set(selected) & set(supplement_by_id)))
    coverage = {trace_id: selected[trace_id].regimes for trace_id in trace_ids}
    return CorpusSelection(
        trace_ids=trace_ids,
        regimes=REGIME_NAMES,
        source_coverage=coverage,
        primary_trace_ids=primary_ids,
        supplement_trace_ids=supplement_ids,
        source_coverage_only=True,
    )


def _trace_provenance(
    value: Any,
    *,
    allow_mapping: bool = False,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    if isinstance(value, M0Trace):
        result = value.provenance()
        for path_key, sha_key in (
            ("source_file", "source_file_sha256"),
            ("parent_run_manifest", "parent_run_manifest_sha256"),
            ("parent_terminal_manifest", "parent_terminal_manifest_sha256"),
        ):
            path = Path(result[path_key])
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
            if path.is_symlink() or not path.is_file():
                raise RegistryError(f"registry {path_key} must be an actual regular file")
            if sha256_file(path) != result[sha_key]:
                raise RegistryError(f"registry {path_key} SHA-256 does not match actual path")
        return result
    if not allow_mapping:
        raise RegistryError("registry traces must be validated M0Trace objects")
    item = _mapping(value, "registry trace", RegistryError)
    # Loader-only path for already frozen JSON records.  New registries must be
    # built from M0Trace objects so a mapping cannot masquerade as validation.
    trace_id = item.get("trace_id", item.get("episode_id"))
    result = {
        "trace_id": _string(trace_id, "registry trace_id", RegistryError),
        "episode_id": _string(item.get("episode_id", trace_id), "registry episode_id", RegistryError),
        "trajectory_id": _string(item.get("trajectory_id", trace_id), "registry trajectory_id", RegistryError),
        "suite": _string(item.get("suite"), "registry suite", RegistryError),
        "task_id": _nonnegative_int(item.get("task_id"), "registry task_id", RegistryError),
        "task_name": _string(item.get("task_name"), "registry task_name", RegistryError),
        "init_state_id": _nonnegative_int(item.get("init_state_id"), "registry init_state_id", RegistryError),
        "seed": _nonnegative_int(item.get("seed"), "registry seed", RegistryError),
        "source_file": _string(item.get("source_file"), "registry source_file", RegistryError),
        "source_file_sha256": _validate_sha(item.get("source_file_sha256"), "registry source_file_sha256", RegistryError),
        "parent_run_manifest": _string(item.get("parent_run_manifest"), "registry parent_run_manifest", RegistryError),
        "parent_run_manifest_sha256": _validate_sha(
            item.get("parent_run_manifest_sha256"), "registry parent_run_manifest_sha256", RegistryError
        ),
        "parent_terminal_manifest": _string(
            item.get("parent_terminal_manifest"), "registry parent_terminal_manifest", RegistryError
        ),
        "parent_terminal_manifest_sha256": _validate_sha(
            item.get("parent_terminal_manifest_sha256"),
            "registry parent_terminal_manifest_sha256",
            RegistryError,
        ),
        "action_sha256": _validate_sha(item.get("action_sha256", item.get("actions_sha256")), "registry action_sha256", RegistryError),
        "action_shape": list(item.get("action_shape", [])),
        "action_dtype": item.get("action_dtype", "float32"),
    }
    if isinstance(item.get("relocation"), Mapping):
        result["relocation"] = dict(item["relocation"])
    if result["trajectory_id"] != result["episode_id"]:
        raise RegistryError("registry trace episode/trajectory identity mismatch")
    if (
        result["action_dtype"] != "float32"
        or len(result["action_shape"]) != 2
        or result["action_shape"][-1] != ACTION_DIM
    ):
        raise RegistryError("registry trace action provenance is not float32[:, 7]")
    for path_key, sha_key in (
        ("source_file", "source_file_sha256"),
        ("parent_run_manifest", "parent_run_manifest_sha256"),
        ("parent_terminal_manifest", "parent_terminal_manifest_sha256"),
    ):
        path = Path(result[path_key])
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if path.is_symlink() or not path.is_file():
            raise RegistryError(f"registry {path_key} must be an actual regular file")
        if sha256_file(path) != result[sha_key]:
            raise RegistryError(f"registry {path_key} SHA-256 does not match actual path")
    return result


def _validate_tape_provenance(value: Mapping[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    result = dict(value)
    per_trace = result.get("per_trace")
    if per_trace is not None:
        if not isinstance(per_trace, Mapping) or not per_trace:
            raise RegistryError("tape provenance per_trace must be a non-empty mapping")
        checked: dict[str, Any] = {}
        for trace_id, raw in sorted(per_trace.items(), key=lambda item: str(item[0])):
            if not isinstance(trace_id, str) or not trace_id:
                raise RegistryError("tape provenance per_trace keys must be non-empty strings")
            checked[trace_id] = _validate_tape_provenance(_mapping(raw, f"tape provenance per_trace[{trace_id}]", RegistryError), base_dir=base_dir)
        result["per_trace"] = checked
        result.setdefault("algorithm", "validated_m0_action_tape_per_trace")
        result.setdefault("dtype", "float32")
        result.setdefault("registry", True)
        return result
    if "path" not in result:
        for alias in ("tape_path", "action_tape_path", "source_file"):
            if alias in result:
                result["path"] = result[alias]
                break
    for key in ("algorithm", "dtype", "shape", "sha256", "path"):
        if key not in result:
            raise RegistryError(f"tape provenance lacks {key}")
    _string(result["algorithm"], "tape algorithm", RegistryError)
    if result["dtype"] not in ("float32", "<f4", "|f4"):
        raise RegistryError("tape provenance dtype must be float32")
    shape = result["shape"]
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 2
        or list(shape)[-1:] != [ACTION_DIM]
        or isinstance(shape[0], bool)
        or not isinstance(shape[0], (int, np.integer))
        or int(shape[0]) <= 0
    ):
        raise RegistryError("tape provenance shape must be a non-empty [N, 7]")
    tape_sha = _validate_sha(result["sha256"], "tape provenance sha256", RegistryError)
    tape_path = result["path"]
    if not isinstance(tape_path, str) or not tape_path:
        raise RegistryError("tape provenance path must be a non-empty string")
    target = Path(tape_path)
    if not target.is_absolute() and base_dir is not None:
        target = base_dir / target
    if target.is_symlink() or not target.is_file():
        raise RegistryError("tape provenance path must be an actual regular file")
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise RegistryError("tape provenance path is unreadable") from exc
    expected_bytes = int(shape[0]) * ACTION_DIM * np.dtype("float32").itemsize
    actual_raw_sha = sha256_bytes(raw)
    actual_tape_sha = actual_raw_sha
    # Support both a raw float32 tape and a NumPy .npy persistence format, but
    # always verify the declared shape/dtype and exact contiguous action bytes.
    if actual_raw_sha == tape_sha and target.suffix != ".npy":
        if len(raw) != expected_bytes:
            raise RegistryError("persisted raw tape byte length does not match tape provenance shape")
        raw_array = np.frombuffer(raw, dtype=np.dtype("float32"))
        if not np.all(np.isfinite(raw_array)):
            raise RegistryError("persisted action tape must contain only finite float32 values")
    else:
        try:
            loaded = np.load(target, allow_pickle=False)
            loaded = np.asarray(loaded)
            if loaded.dtype != np.dtype("float32") or tuple(loaded.shape) != tuple(shape):
                raise ValueError
            loaded = np.ascontiguousarray(loaded)
            if not np.all(np.isfinite(loaded)):
                raise ValueError
            actual_tape_sha = sha256_bytes(loaded.tobytes(order="C"))
        except Exception as exc:
            # A raw float32 tape whose bytes do not hash to the declaration is
            # never salvageable by interpreting it as .npy.
            if actual_raw_sha != tape_sha:
                raise RegistryError("tape provenance sha256 does not match persisted tape bytes") from exc
        if actual_tape_sha != tape_sha and actual_raw_sha != tape_sha:
            raise RegistryError("tape provenance sha256 does not match persisted tape bytes")
    result["path"] = str(target if target.is_absolute() else target)
    if "seed" in result:
        _nonnegative_int(result["seed"], "tape provenance seed", RegistryError)
    for key in ("capture_offsets", "offsets"):
        if key in result:
            offsets = result[key]
            if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
                raise RegistryError(f"tape provenance {key} must be a sequence")
            for offset in offsets:
                _nonnegative_int(offset, f"tape provenance {key} offset", RegistryError)
                if int(offset) >= int(shape[0]):
                    raise RegistryError(f"tape provenance {key} offset is outside persisted tape bounds")
    return result


def _registry_payload_without_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("registry_sha256", None)
    return result


def _write_exclusive(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        raise RegistryError(f"refusing to overwrite existing registry: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RegistryError(f"refusing to overwrite existing registry: {path}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_frozen_registry(
    path: str | Path,
    traces: Sequence[Any] | None = None,
    tape_provenance: Mapping[str, Any] | None = None,
    per_trace: Mapping[str, Mapping[str, Any]] | None = None,
    selection: Mapping[str, Any] | CorpusSelection | None = None,
) -> Path:
    """Write one complete immutable registry, refusing every overwrite."""

    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        raise RegistryError("registry traces must be a non-empty sequence")
    if tape_provenance is None:
        raise RegistryError("registry requires complete tape provenance")
    if per_trace is None:
        raise RegistryError("registry requires per-trace regimes, capture offsets, and horizons")
    per_trace = _mapping(per_trace, "per_trace", RegistryError)
    trace_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in traces:
        provenance = _trace_provenance(trace)
        trace_id = provenance["trace_id"]
        if trace_id in seen:
            raise RegistryError(f"duplicate registry trace {trace_id}")
        seen.add(trace_id)
        metadata = per_trace.get(trace_id)
        if metadata is None:
            raise RegistryError(f"missing per-trace registry metadata for {trace_id}")
        metadata = _mapping(metadata, f"per_trace[{trace_id}]", RegistryError)
        raw_windows = metadata.get("windows")
        windows: list[dict[str, Any]] = []
        if raw_windows is not None:
            if not isinstance(raw_windows, Sequence) or isinstance(raw_windows, (str, bytes)) or not raw_windows:
                raise RegistryError(f"per_trace[{trace_id}] windows must be a non-empty sequence")
            for wi, raw_window in enumerate(raw_windows):
                window = _mapping(raw_window, f"per_trace[{trace_id}].windows[{wi}]", RegistryError)
                wr = window.get("regimes", window.get("regime"))
                if isinstance(wr, str):
                    wr = [wr]
                if not isinstance(wr, Sequence) or isinstance(wr, (str, bytes)) or not wr:
                    raise RegistryError(f"per_trace[{trace_id}].windows[{wi}] regimes must be a sequence")
                wregimes = tuple(dict.fromkeys(normalize_regime(item) for item in wr))
                woffset = _nonnegative_int(window.get("capture_offset"), f"per_trace[{trace_id}].windows[{wi}] capture_offset", RegistryError)
                whorizon = _positive_int(window.get("continuation_horizon"), f"per_trace[{trace_id}].windows[{wi}] continuation_horizon", RegistryError)
                wcoverage = _mapping(window.get("source_coverage", {"regimes": list(wregimes)}), f"per_trace[{trace_id}].windows[{wi}] source_coverage", RegistryError)
                if not set(wregimes).issubset(set(_coverage_regimes(wcoverage, error=RegistryError))):
                    raise RegistryError(f"per_trace[{trace_id}].windows[{wi}] source_coverage does not cover all regimes")
                windows.append({"regimes": list(wregimes), "capture_offset": woffset, "continuation_horizon": whorizon, "source_coverage": dict(wcoverage)})
        raw_regimes = metadata.get("regimes", metadata.get("regime"))
        if isinstance(raw_regimes, str):
            raw_regimes = [raw_regimes]
        if not isinstance(raw_regimes, Sequence) or isinstance(raw_regimes, (str, bytes)):
            raise RegistryError(f"per_trace[{trace_id}] regimes must be a sequence")
        regimes = tuple(dict.fromkeys(normalize_regime(item) for item in raw_regimes))
        if not regimes:
            raise RegistryError(f"per_trace[{trace_id}] has no regimes")
        capture_offset = _nonnegative_int(metadata.get("capture_offset"), f"per_trace[{trace_id}] capture_offset", RegistryError)
        continuation_horizon = _positive_int(
            metadata.get("continuation_horizon"),
            f"per_trace[{trace_id}] continuation_horizon",
            RegistryError,
        )
        coverage = _mapping(metadata.get("source_coverage", {}), f"per_trace[{trace_id}] source_coverage", RegistryError)
        # Coverage may have richer detector details, but it must still state
        # exactly the regimes claimed by the frozen registry.
        coverage_regimes = _coverage_regimes(coverage, error=RegistryError)
        if not set(regimes).issubset(set(coverage_regimes)):
            raise RegistryError(f"per_trace[{trace_id}] source_coverage does not cover all regimes")
        record = {
            **provenance,
            "regimes": list(regimes),
            "capture_offset": capture_offset,
            "continuation_horizon": continuation_horizon,
            "source_coverage": dict(coverage),
        }
        if windows:
            record["windows"] = windows
        trace_records.append(record)
    trace_records.sort(key=lambda item: item["trace_id"])
    all_regimes = {
        regime
        for item in trace_records
        for regime in (
            [regime for window in item.get("windows", ()) for regime in window.get("regimes", ())]
            if item.get("windows")
            else item["regimes"]
        )
    }
    if all_regimes != set(REGIME_NAMES):
        raise RegistryError("frozen registry must cover exactly all seven M1 regimes")
    tape = _validate_tape_provenance(_mapping(tape_provenance, "tape_provenance", RegistryError))
    for record in trace_records:
        windows_to_check = record.get("windows") or [record]
        trace_tape = tape.get("per_trace", {}).get(record["trace_id"], tape) if isinstance(tape.get("per_trace"), Mapping) else tape
        if isinstance(tape.get("per_trace"), Mapping) and record["trace_id"] not in tape["per_trace"]:
            raise RegistryError(f"tape provenance is missing trace {record['trace_id']}")
        if trace_tape.get("dtype") != record["action_dtype"]:
            raise RegistryError(f"per_trace[{record['trace_id']}] tape dtype differs from M0 action provenance")
        if list(trace_tape.get("shape", ())) != list(record["action_shape"]):
            raise RegistryError(f"per_trace[{record['trace_id']}] tape shape differs from M0 action provenance")
        if trace_tape.get("sha256") != record["action_sha256"]:
            raise RegistryError(f"per_trace[{record['trace_id']}] tape SHA differs from M0 action provenance")
        tape_length = int(trace_tape["shape"][0])
        for window in windows_to_check:
            start = int(window["capture_offset"])
            horizon = int(window["continuation_horizon"])
            if start >= tape_length or start + horizon > tape_length:
                raise RegistryError(
                    f"per_trace[{record['trace_id']}] capture window is outside persisted tape bounds"
                )
    if selection is None:
        selection_payload: Mapping[str, Any] = {"rule": "primary_always_minimum_source_coverage"}
    elif isinstance(selection, CorpusSelection):
        selection_payload = selection.to_dict()
    else:
        selection_payload = _mapping(selection, "selection", RegistryError)
    _reject_outcomes(selection_payload, path="selection")
    if selection is not None:
        selected_ids = selection_payload.get("trace_ids")
        if not isinstance(selected_ids, Sequence) or isinstance(selected_ids, (str, bytes)):
            raise RegistryError("selection must persist trace_ids for registry reconciliation")
        normalized_selected = tuple(sorted(_string(item, "selection trace_id", RegistryError) for item in selected_ids))
        if normalized_selected != tuple(sorted(item["trace_id"] for item in trace_records)):
            raise RegistryError("selection trace_ids do not reconcile with registry traces")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "registry_type": "replayvla_m1_frozen_registry",
        "regimes": list(REGIME_NAMES),
        "traces": trace_records,
        "tape_provenance": tape,
        "selection": dict(selection_payload),
    }
    try:
        canonical_payload = canonical_json_bytes(payload)
    except M1HardGateError as exc:
        raise RegistryError(f"registry contains non-canonical JSON values: {exc}") from exc
    payload["registry_sha256"] = _sha256(canonical_payload)
    try:
        output = canonical_json_bytes(payload) + b"\n"
    except M1HardGateError as exc:
        raise RegistryError(f"registry contains non-canonical JSON values: {exc}") from exc
    return _write_exclusive(Path(path), output)


def load_frozen_registry(path: str | Path) -> dict[str, Any]:
    """Load and verify a frozen registry's own SHA and structural coverage."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RegistryError(f"registry is not a regular file: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RegistryError(f"registry is not valid JSON: {exc}") from exc
    payload = _mapping(payload, "registry", RegistryError)
    expected = _validate_sha(payload.get("registry_sha256"), "registry_sha256", RegistryError)
    try:
        actual = _sha256(canonical_json_bytes(_registry_payload_without_hash(payload)))
    except M1HardGateError as exc:
        raise RegistryError(f"registry contains non-canonical JSON values: {exc}") from exc
    if expected != actual:
        raise RegistryError("registry hash does not match canonical registry contents")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError("unsupported registry schema version")
    if tuple(payload.get("regimes", ())) != REGIME_NAMES:
        raise RegistryError("registry regime names are not the exact seven M1 names")
    traces = payload.get("traces")
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        raise RegistryError("registry traces are missing")
    seen: set[str] = set()
    all_regimes: set[str] = set()
    for index, item in enumerate(traces):
        item = _mapping(item, f"registry traces[{index}]", RegistryError)
        trace_id = _string(item.get("trace_id"), f"registry traces[{index}].trace_id", RegistryError)
        if trace_id in seen:
            raise RegistryError(f"duplicate registry trace {trace_id}")
        seen.add(trace_id)
        raw_regimes = item.get("regimes")
        if not isinstance(raw_regimes, Sequence) or isinstance(raw_regimes, (str, bytes)):
            raise RegistryError(f"registry traces[{index}] regimes are missing")
        regimes = tuple(dict.fromkeys(normalize_regime(item) for item in raw_regimes))
        windows = item.get("windows")
        if windows is not None:
            if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
                raise RegistryError(f"registry traces[{index}] windows are malformed")
            for wi, window in enumerate(windows):
                window = _mapping(window, f"registry traces[{index}].windows[{wi}]", RegistryError)
                wregimes = window.get("regimes")
                if not isinstance(wregimes, Sequence) or isinstance(wregimes, (str, bytes)) or not wregimes:
                    raise RegistryError(f"registry traces[{index}].windows[{wi}] regimes are missing")
                all_regimes.update(normalize_regime(value) for value in wregimes)
                _nonnegative_int(window.get("capture_offset"), f"registry traces[{index}].windows[{wi}] capture_offset", RegistryError)
                _positive_int(window.get("continuation_horizon"), f"registry traces[{index}].windows[{wi}] continuation_horizon", RegistryError)
        else:
            all_regimes.update(regimes)
        _nonnegative_int(item.get("capture_offset"), f"registry traces[{index}] capture_offset", RegistryError)
        _positive_int(item.get("continuation_horizon"), f"registry traces[{index}] continuation_horizon", RegistryError)
        _trace_provenance(item, allow_mapping=True, base_dir=target.parent)
        _coverage_regimes(
            _mapping(item.get("source_coverage"), "registry source_coverage", RegistryError),
            error=RegistryError,
        )
    if all_regimes != set(REGIME_NAMES):
        raise RegistryError("registry does not cover all seven regimes")
    tape = _validate_tape_provenance(
        _mapping(payload.get("tape_provenance"), "registry tape_provenance", RegistryError),
        base_dir=target.parent,
    )
    for index, item in enumerate(traces):
        windows_to_check = item.get("windows") or [item]
        trace_tape = tape.get("per_trace", {}).get(item["trace_id"], tape) if isinstance(tape.get("per_trace"), Mapping) else tape
        if isinstance(tape.get("per_trace"), Mapping) and item["trace_id"] not in tape["per_trace"]:
            raise RegistryError(f"registry tape provenance is missing trace {item['trace_id']}")
        if trace_tape.get("dtype") != item.get("action_dtype"):
            raise RegistryError(f"registry trace {item['trace_id']} tape dtype differs from action provenance")
        if list(trace_tape.get("shape", ())) != list(item.get("action_shape", ())):
            raise RegistryError(f"registry trace {item['trace_id']} tape shape differs from action provenance")
        if trace_tape.get("sha256") != item.get("action_sha256"):
            raise RegistryError(f"registry trace {item['trace_id']} tape SHA differs from action provenance")
        tape_length = int(trace_tape["shape"][0])
        for wi, window in enumerate(windows_to_check):
            start = _nonnegative_int(window.get("capture_offset"), f"registry traces[{index}] window[{wi}] capture_offset", RegistryError)
            horizon = _positive_int(window.get("continuation_horizon"), f"registry traces[{index}] window[{wi}] continuation_horizon", RegistryError)
            if start >= tape_length or start + horizon > tape_length:
                raise RegistryError(f"registry traces[{index}] capture window is outside persisted tape bounds")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise RegistryError("registry selection is missing")
    _reject_outcomes(selection, path="selection")
    selected_ids = selection.get("trace_ids")
    if selected_ids is not None:
        if not isinstance(selected_ids, Sequence) or isinstance(selected_ids, (str, bytes)):
            raise RegistryError("registry selection trace_ids must be a sequence")
        normalized_selected = tuple(sorted(_string(item, "selection trace_id", RegistryError) for item in selected_ids))
        if normalized_selected != tuple(sorted(seen)):
            raise RegistryError("registry selection trace_ids do not reconcile with registry traces")
    return json.loads(json.dumps(payload))


def _sample_value_abs(sample: Mapping[str, Any]) -> float:
    if "value_a" in sample and "value_b" in sample:
        try:
            first = np.asarray(sample["value_a"])
            second = np.asarray(sample["value_b"])
            if first.dtype.kind == "O" or second.dtype.kind == "O" or first.shape != second.shape:
                raise ValueError
            delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise NullCalibrationError("null pair values must be finite numeric arrays of equal shape") from exc
    elif "difference" in sample:
        delta = np.asarray(sample["difference"])
    elif "delta" in sample:
        delta = np.asarray(sample["delta"])
    elif "value" in sample:
        delta = np.asarray(sample["value"])
    else:
        raise NullCalibrationError("null sample needs value_a/value_b, difference, delta, or value")
    if delta.dtype.kind in {"O", "U", "S"}:
        raise NullCalibrationError("null sample values must be numeric")
    try:
        delta = np.asarray(delta, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NullCalibrationError("null sample values must be numeric") from exc
    if delta.size == 0 or not np.all(np.isfinite(delta)):
        raise NullCalibrationError("null sample values must be finite and non-empty")
    return float(np.max(np.abs(delta)))


def _independent_evidence(sample: Mapping[str, Any]) -> dict[str, Any]:
    nested = sample.get("independent_evidence")
    if nested is not None:
        nested = _mapping(nested, "independent_evidence", NullCalibrationError)
    evidence: dict[str, Any] = {}
    for name in ("contact", "grasp", "carried"):
        key = f"{name}_evidence"
        value = sample.get(key, nested.get(name) if isinstance(nested, Mapping) else None)
        if value is None and name in sample:
            value = sample[name]
        if value is None:
            raise NullCalibrationError(f"null sample lacks independent {name} evidence")
        if isinstance(value, Mapping):
            if not value:
                raise NullCalibrationError(f"independent {name} evidence is empty")
            if "source" not in value or not isinstance(value["source"], str) or not value["source"]:
                raise NullCalibrationError(f"independent {name} evidence needs a source")
            if "regime" in value:
                raise NullCalibrationError(f"{name} evidence may not be derived from regime")
            if "present" not in value and "detected" not in value:
                raise NullCalibrationError(f"independent {name} evidence needs an explicit present flag")
            present = value.get("present", value.get("detected"))
            if not isinstance(present, bool):
                raise NullCalibrationError(f"independent {name} evidence present must be bool")
            if "present" in value and "detected" in value and value["present"] != value["detected"]:
                raise NullCalibrationError(f"independent {name} evidence present/detected fields disagree")
        else:
            raise NullCalibrationError(f"independent {name} evidence must be a structured mapping")
        evidence[name] = {"present": present}
    return evidence


def build_grouped_null_envelope(
    samples: Iterable[Mapping[str, Any]],
    *,
    selected_trace_ids: Sequence[str] | None = None,
    required_groups: Iterable[Sequence[Any]] | None = None,
    min_pairs: int = MIN_NULL_PAIRS,
    min_pairs_per_trace: int = MIN_PAIRS_PER_TRACE,
    min_samples_per_regime: int = MIN_SAMPLES_PER_REGIME,
    global_tolerance: Any = None,
) -> dict[str, Any]:
    """Build strict regime×quantity×horizon null envelopes.

    ``min_pairs`` applies to every grouped key, preserving the calibration
    semantics when quantities or horizons differ.  There is intentionally no
    global fallback threshold; callers must use the matching grouped key.
    """

    if global_tolerance is not None:
        raise NullCalibrationError("global tolerance fallback is forbidden; use grouped envelopes")
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < MIN_NULL_PAIRS:
        raise NullCalibrationError(f"min_pairs must be at least {MIN_NULL_PAIRS}")
    if isinstance(min_pairs_per_trace, bool) or not isinstance(min_pairs_per_trace, int) or min_pairs_per_trace < MIN_PAIRS_PER_TRACE:
        raise NullCalibrationError(f"min_pairs_per_trace must be at least {MIN_PAIRS_PER_TRACE}")
    if isinstance(min_samples_per_regime, bool) or not isinstance(min_samples_per_regime, int) or min_samples_per_regime < MIN_SAMPLES_PER_REGIME:
        raise NullCalibrationError(f"min_samples_per_regime must be at least {MIN_SAMPLES_PER_REGIME}")
    if selected_trace_ids is None:
        raise NullCalibrationError("selected_trace_ids is required for null calibration")
    if hasattr(selected_trace_ids, "trace_ids"):
        selected_trace_ids = getattr(selected_trace_ids, "trace_ids")
    if isinstance(selected_trace_ids, (str, bytes)) or not isinstance(selected_trace_ids, Iterable):
        raise NullCalibrationError("selected_trace_ids must be an iterable of trace IDs")
    selected = {_string(item, "selected trace ID", NullCalibrationError) for item in selected_trace_ids}
    if not selected:
        raise NullCalibrationError("selected_trace_ids cannot be empty")
    if required_groups is None:
        raise NullCalibrationError("required_groups is required for explicit null coverage")
    required: set[tuple[str, str, int]] = set()
    for index, raw_group in enumerate(required_groups):
        if not isinstance(raw_group, Sequence) or isinstance(raw_group, (str, bytes)) or len(raw_group) != 3:
            raise NullCalibrationError(f"required_groups[{index}] must be (regime, quantity, horizon)")
        required.add(
            (
                normalize_regime(raw_group[0]),
                _string(raw_group[1], "required null quantity", NullCalibrationError),
                _nonnegative_int(raw_group[2], "required null horizon", NullCalibrationError),
            )
        )
    if not required:
        raise NullCalibrationError("required_groups cannot be empty")
    grouped: dict[tuple[str, str, int], list[tuple[str, str, float, dict[str, Any]]]] = {}
    group_pair_ids: dict[tuple[str, str, int], set[str]] = {}
    trace_pair_ids: dict[str, set[str]] = {}
    regime_pair_ids = {regime: set() for regime in REGIME_NAMES}
    independent_regime_presence = {name: 0 for name in ("contact", "grasp", "carried")}
    pair_traces: dict[str, str] = {}
    sample_count = 0
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"null sample {index}", NullCalibrationError)
        pair_id = _string(sample.get("pair_id"), "null pair_id", NullCalibrationError)
        trace_id = _string(sample.get("trace_id", sample.get("episode_id")), "null trace_id", NullCalibrationError)
        prior_trace = pair_traces.setdefault(pair_id, trace_id)
        if prior_trace != trace_id:
            raise NullCalibrationError(f"null pair_id {pair_id} spans multiple traces")
        if selected is not None and trace_id not in selected:
            raise NullCalibrationError(f"null sample trace {trace_id} is not in selected corpus")
        regime = normalize_regime(sample.get("regime"))
        quantity = _string(sample.get("quantity"), "null quantity", NullCalibrationError)
        horizon = _nonnegative_int(sample.get("horizon"), "null horizon", NullCalibrationError)
        value_abs = _sample_value_abs(sample)
        independent = _independent_evidence(sample)
        group_key = (regime, quantity, horizon)
        seen_in_group = group_pair_ids.setdefault(group_key, set())
        if pair_id in seen_in_group:
            raise NullCalibrationError(
                f"duplicate null pair_id {pair_id} within regime={regime} quantity={quantity} horizon={horizon}"
            )
        seen_in_group.add(pair_id)
        grouped.setdefault(group_key, []).append((pair_id, trace_id, value_abs, independent))
        for evidence_name in independent_regime_presence:
            if regime == evidence_name and independent[evidence_name]["present"]:
                independent_regime_presence[evidence_name] += 1
        trace_pair_ids.setdefault(trace_id, set()).add(pair_id)
        regime_pair_ids[regime].add(pair_id)
        sample_count += 1
    if not grouped:
        raise NullCalibrationError("null calibration has no samples")
    if selected is not None:
        missing = selected.difference(trace_pair_ids)
        if missing:
            raise NullCalibrationError(f"selected traces have no null pairs: {sorted(missing)}")
    for trace_id, pair_values in trace_pair_ids.items():
        count = len(pair_values)
        if count < min_pairs_per_trace:
            raise NullCalibrationError(f"trace {trace_id} has {count} null pairs; minimum is {min_pairs_per_trace}")
    regime_counts = {regime: len(values) for regime, values in regime_pair_ids.items()}
    missing_regimes = [regime for regime, count in regime_counts.items() if count < min_samples_per_regime]
    if missing_regimes:
        raise NullCalibrationError(f"regimes lack minimum samples: {missing_regimes}")
    for key, values in grouped.items():
        if len(values) < min_pairs:
            regime, quantity, horizon = key
            raise NullCalibrationError(
                f"group regime={regime} quantity={quantity} horizon={horizon} has "
                f"{len(values)} pairs; minimum is {min_pairs}"
            )
    missing_required = sorted(required.difference(grouped))
    if missing_required:
        raise NullCalibrationError(f"required null groups are missing: {missing_required}")
    independent_coverage: dict[str, dict[str, int]] = {}
    for evidence_name in ("contact", "grasp", "carried"):
        present = 0
        total_evidence = 0
        for values in grouped.values():
            for _, _, _, evidence in values:
                total_evidence += 1
                if evidence[evidence_name]["present"]:
                    present += 1
        if not present or not independent_regime_presence[evidence_name]:
            raise NullCalibrationError(
                f"independent {evidence_name} evidence has no positive samples in its {evidence_name} regime"
            )
        independent_coverage[evidence_name] = {
            "present_samples": present,
            "present_in_regime": independent_regime_presence[evidence_name],
            "samples": total_evidence,
        }
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for (regime, quantity, horizon), values in sorted(grouped.items()):
        magnitudes = np.asarray([item[2] for item in values], dtype=np.float64)
        group = {
            "regime": regime,
            "quantity": quantity,
            "horizon": horizon,
            "n_pairs": len(values),
            "n_traces": len({item[1] for item in values}),
            "trace_ids": sorted({item[1] for item in values}),
            "pair_ids": sorted(item[0] for item in values),
            "max_abs": float(np.max(magnitudes)),
            "mean_abs": float(np.mean(magnitudes)),
            "q50_abs": float(np.quantile(magnitudes, 0.50)),
            "q95_abs": float(np.quantile(magnitudes, 0.95)),
            "q99_abs": float(np.quantile(magnitudes, 0.99)),
        }
        groups.setdefault(regime, {}).setdefault(quantity, {})[str(horizon)] = group
    return {
        "schema_version": SCHEMA_VERSION,
        "envelope_type": "replayvla_m1_grouped_null",
        "groups": groups,
        "coverage": {
            "n_pairs": len(pair_traces),
            "n_samples": sample_count,
            "n_traces": len(trace_pair_ids),
            "minimum_pairs_per_group": min_pairs,
            "minimum_pairs_per_trace": min_pairs_per_trace,
            "minimum_samples_per_regime": min_samples_per_regime,
            "regime_sample_counts": regime_counts,
            "independent_evidence": independent_coverage,
            "global_tolerance": None,
            "required_groups": [list(item) for item in sorted(required)],
        },
    }


def check_null_value(
    envelope: Mapping[str, Any],
    *,
    value: Any,
    regime: str,
    quantity: str,
    horizon: int,
) -> bool:
    """Check one value against its explicit group; missing groups fail closed."""

    regime_name = normalize_regime(regime)
    if not isinstance(quantity, str) or not quantity:
        raise NullCalibrationError("quantity must be a non-empty string")
    horizon_value = _nonnegative_int(horizon, "horizon", NullCalibrationError)
    groups = _mapping(envelope.get("groups"), "null envelope groups", NullCalibrationError)
    regime_group = groups.get(regime_name)
    if not isinstance(regime_group, Mapping) or quantity not in regime_group:
        raise NullCalibrationError(f"no explicit null group for regime={regime_name} quantity={quantity}")
    quantity_group = _mapping(regime_group[quantity], "null quantity group", NullCalibrationError)
    item = quantity_group.get(str(horizon_value))
    if not isinstance(item, Mapping):
        raise NullCalibrationError(f"no explicit null group for regime={regime_name} horizon={horizon_value}")
    observed = _sample_value_abs({"value": value})
    threshold = _finite_number(item.get("max_abs"), "null group max_abs")
    return observed <= threshold


def rgb_disagreement_metrics(expected: Any, actual: Any) -> dict[str, int | float]:
    """Return exact pixel disagreement count plus max/mean absolute error."""

    left = np.asarray(expected)
    right = np.asarray(actual)
    if left.dtype.kind == "O" or right.dtype.kind == "O" or left.shape != right.shape:
        raise RendererCalibrationError("RGB controls must have equal non-object shapes")
    if left.size == 0:
        raise RendererCalibrationError("RGB controls cannot be empty")
    try:
        difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RendererCalibrationError("RGB controls must be numeric") from exc
    if not np.all(np.isfinite(difference)):
        raise RendererCalibrationError("RGB disagreement is not finite")
    absolute = np.abs(difference)
    differing = absolute != 0.0
    if absolute.ndim >= 3:
        differing_pixels = np.any(differing, axis=-1)
    else:
        differing_pixels = differing
    return {
        "differing_pixel_count": int(np.count_nonzero(differing_pixels)),
        "max_abs": float(np.max(absolute)),
        "mean_abs": float(np.mean(absolute)),
    }


def _controls_bitwise_identical(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise RendererCalibrationError("duplicate_controls must contain at least two arrays")
    arrays = [np.asarray(item) for item in value]
    first = arrays[0]
    if first.dtype.kind == "O" or any(item.dtype.kind == "O" for item in arrays):
        raise RendererCalibrationError("duplicate controls may not be object arrays")
    return all(item.dtype == first.dtype and item.shape == first.shape and item.tobytes(order="C") == first.tobytes(order="C") for item in arrays[1:])


def build_renderer_envelopes(
    samples: Iterable[Mapping[str, Any]],
    *,
    exact_only: bool | None = None,
    min_samples_per_group: int = 1,
) -> dict[str, Any]:
    """Build renderer envelopes keyed camera×observation-key×regime×horizon."""

    if isinstance(min_samples_per_group, bool) or not isinstance(min_samples_per_group, int) or min_samples_per_group <= 0:
        raise RendererCalibrationError("min_samples_per_group must be positive")
    grouped: dict[tuple[str, str, str, int], list[tuple[dict[str, int | float], bool]]] = {}
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"renderer sample {index}", RendererCalibrationError)
        camera = _string(sample.get("camera"), "renderer camera", RendererCalibrationError)
        key = _string(sample.get("key", sample.get("observation_key")), "renderer key", RendererCalibrationError)
        regime = normalize_regime(sample.get("regime"))
        horizon = _nonnegative_int(sample.get("horizon"), "renderer horizon", RendererCalibrationError)
        controls = sample.get("duplicate_controls")
        if controls is None and "duplicate_control_a" in sample and "duplicate_control_b" in sample:
            controls = [sample["duplicate_control_a"], sample["duplicate_control_b"]]
        if controls is None:
            raise RendererCalibrationError("renderer sample requires duplicate_controls")
        compact_metrics = sample.get("metrics")
        if compact_metrics is not None:
            metrics_mapping = _mapping(compact_metrics, "renderer metrics", RendererCalibrationError)
            try:
                differing_pixel_count = int(metrics_mapping["differing_pixel_count"])
                max_abs = float(metrics_mapping["max_abs"])
                mean_abs = float(metrics_mapping["mean_abs"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RendererCalibrationError("renderer metrics are incomplete") from exc
            if (
                differing_pixel_count < 0
                or not math.isfinite(max_abs)
                or max_abs < 0
                or not math.isfinite(mean_abs)
                or mean_abs < 0
            ):
                raise RendererCalibrationError("renderer metrics must be finite and nonnegative")
            metrics = {
                "differing_pixel_count": differing_pixel_count,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
            }
        else:
            left = sample.get("rgb_a", sample.get("expected"))
            right = sample.get("rgb_b", sample.get("actual"))
            if left is None or right is None:
                left, right = controls[0], controls[1]
            metrics = rgb_disagreement_metrics(left, right)
        controls_identical = _controls_bitwise_identical(controls)
        grouped.setdefault((camera, key, regime, horizon), []).append((metrics, controls_identical))
    if not grouped:
        raise RendererCalibrationError("renderer envelope has no samples")
    output: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for (camera, key, regime, horizon), values in sorted(grouped.items()):
        if len(values) < min_samples_per_group:
            raise RendererCalibrationError(f"renderer group {camera}/{key}/{regime}/{horizon} lacks samples")
        controls_all_identical = all(item[1] for item in values)
        if not controls_all_identical:
            raise RendererCalibrationError(
                "renderer null calibration requires bitwise-identical duplicate controls"
            )
        metrics = [item[0] for item in values]
        rgb_all_exact = all(
            item["differing_pixel_count"] == 0 and item["max_abs"] == 0.0 and item["mean_abs"] == 0.0
            for item in metrics
        )
        if exact_only is True and not rgb_all_exact:
            raise RendererCalibrationError(
                "exact-only renderer envelope requires bitwise-identical duplicate RGB outputs"
            )
        group = {
            "camera": camera,
            "key": key,
            "regime": regime,
            "horizon": horizon,
            "n_samples": len(values),
            "duplicate_controls_bitwise_identical": controls_all_identical,
            "exact_only": rgb_all_exact if exact_only is None else bool(exact_only),
            "differing_pixel_count_max": max(int(item["differing_pixel_count"]) for item in metrics),
            "max_abs": max(float(item["max_abs"]) for item in metrics),
            "mean_abs": max(float(item["mean_abs"]) for item in metrics),
        }
        output.setdefault(camera, {}).setdefault(key, {}).setdefault(regime, {})[str(horizon)] = group
    return {
        "schema_version": SCHEMA_VERSION,
        "envelope_type": "replayvla_m1_renderer",
        "groups": output,
        "coverage": {"global_tolerance": None, "group_count": len(grouped)},
    }


def check_renderer_value(
    envelope: Mapping[str, Any],
    *,
    camera: str,
    key: str,
    regime: str,
    horizon: int,
    expected: Any,
    actual: Any,
) -> bool:
    """Check RGB metrics against the matching renderer group only."""

    camera = _string(camera, "renderer camera", RendererCalibrationError)
    key = _string(key, "renderer key", RendererCalibrationError)
    regime = normalize_regime(regime)
    horizon = _nonnegative_int(horizon, "renderer horizon", RendererCalibrationError)
    groups = _mapping(envelope.get("groups"), "renderer envelope groups", RendererCalibrationError)
    camera_group = groups.get(camera)
    if not isinstance(camera_group, Mapping) or key not in camera_group:
        raise RendererCalibrationError(f"no explicit renderer group for camera={camera} key={key}")
    key_group = _mapping(camera_group[key], "renderer key group", RendererCalibrationError)
    regime_group = key_group.get(regime)
    if not isinstance(regime_group, Mapping) or str(horizon) not in regime_group:
        raise RendererCalibrationError(
            f"no explicit renderer group for camera={camera} key={key} regime={regime} horizon={horizon}"
        )
    group = _mapping(regime_group[str(horizon)], "renderer group", RendererCalibrationError)
    metrics = rgb_disagreement_metrics(expected, actual)
    if group.get("exact_only") is True:
        return metrics["differing_pixel_count"] == 0 and metrics["max_abs"] == 0.0 and metrics["mean_abs"] == 0.0
    return (
        metrics["differing_pixel_count"] <= int(group.get("differing_pixel_count_max"))
        and float(metrics["max_abs"]) <= _finite_number(group.get("max_abs"), "renderer max_abs")
        and float(metrics["mean_abs"]) <= _finite_number(group.get("mean_abs"), "renderer mean_abs")
    )


def exact_discrete_equal(expected: Any, actual: Any) -> bool:
    """Compare discrete values exactly, including scalar type and array dtype."""

    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        if not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray):
            return False
        if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.dtype.kind == "O":
            return False
        if expected.dtype.kind in {"f", "c"} and (
            not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual))
        ):
            return False
        # ``array_equal`` treats +0.0 and -0.0 as equal.  Discrete evidence is
        # persisted bytes, so compare the exact representation instead.
        return expected.tobytes(order="C") == actual.tobytes(order="C")
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping) or len(expected) != len(actual):
            return False
        unmatched = list(actual.items())
        for expected_key, expected_value in expected.items():
            match_index = next(
                (
                    index
                    for index, (actual_key, _) in enumerate(unmatched)
                    if type(expected_key) is type(actual_key)
                    and exact_discrete_equal(expected_key, actual_key)
                ),
                None,
            )
            if match_index is None:
                return False
            _, actual_value = unmatched.pop(match_index)
            if not exact_discrete_equal(expected_value, actual_value):
                return False
        return not unmatched
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            return False
        return all(exact_discrete_equal(left, right) for left, right in zip(expected, actual))
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, np.generic) or isinstance(actual, np.generic):
        if type(expected) is not type(actual):
            return False
        if expected.dtype.kind in {"f", "c"}:
            if not np.isfinite(expected) or not np.isfinite(actual):
                return False
            return expected.tobytes() == actual.tobytes()
        return bool(expected == actual)
    if isinstance(expected, (int, float, str, type(None))) or isinstance(actual, (int, float, str, type(None))):
        if type(expected) is not type(actual):
            return False
        if isinstance(expected, float) and (not math.isfinite(expected) or not math.isfinite(actual)):
            return False
        if isinstance(expected, float) and expected == 0.0 and math.copysign(1.0, expected) != math.copysign(1.0, actual):
            return False
        return expected == actual
    return type(expected) is type(actual) and expected == actual


def require_exact_discrete(expected: Any, actual: Any, *, field: str = "discrete field") -> None:
    if not exact_discrete_equal(expected, actual):
        raise DiscreteGateError(f"{field} failed exact discrete equality")


def validate_legitimate_short_terminal(
    record: Mapping[str, Any],
    *,
    nominal_horizon: int | None = None,
    horizon: int | None = None,
) -> bool:
    """Validate a synchronous successful terminal before the nominal horizon.

    A legitimate short terminal must be returned by the same step that marks
    termination, include a terminal observation, and show no autoreset,
    retry, truncation, or second attempt.  Missing evidence is invalid.
    """

    if not isinstance(record, Mapping):
        return False
    try:
        steps_value = record.get("steps")
        if isinstance(steps_value, bool) or not isinstance(steps_value, (int, np.integer)):
            return False
        steps = int(steps_value)
        horizon_value = nominal_horizon if nominal_horizon is not None else horizon
        if horizon_value is None:
            horizon_value = record.get("nominal_horizon", record.get("horizon"))
        if isinstance(horizon_value, bool) or not isinstance(horizon_value, (int, np.integer)):
            return False
        horizon = int(horizon_value)
        if steps <= 0 or horizon <= 0 or steps >= horizon:
            return False
        if record.get("status") != "completed" or record.get("completed") is not True:
            return False
        if record.get("success") is not True:
            return False
        if record.get("terminated") is not True or record.get("truncated") is not False:
            return False
        if record.get("step_returned_terminal") is not True:
            return False
        if "synchronous" in record and record.get("synchronous") is not True:
            return False
        if "terminal_step" in record:
            terminal_step = record.get("terminal_step")
            if isinstance(terminal_step, bool) or not isinstance(terminal_step, (int, np.integer)):
                return False
            if int(terminal_step) != steps - 1:
                return False
        if "terminal_event" in record and record.get("terminal_event") not in {
            "step_return",
            "terminal_step",
            "terminated",
            "success",
        }:
            return False
        if "termination_source" in record and record.get("termination_source") not in {
            "step_return",
            "synchronous_step",
        }:
            return False
        if "termination_reason" in record and record.get("termination_reason") not in {
            "success",
            "task_success",
            "predicate",
            "predicate_success",
            "predicate_transition",
            "environment_termination",
            "terminated",
            "termination",
            "done",
        }:
            return False
        terminal_observation = record.get(
            "terminal_observation",
            record.get("final_observation", record.get("observation")),
        )
        if terminal_observation is None and record.get("terminal_observation_present") is True:
            terminal_observation = True
        if terminal_observation is None or terminal_observation is False:
            return False
        if record.get("autoreset") is not False:
            return False
        if record.get("attempt_count") != 1 or record.get("no_retry") is not True:
            return False
        if record.get("retry_count", 0) != 0:
            return False
        if record.get("reset_count", 0) != 0:
            return False
        return True
    except (TypeError, ValueError):
        return False


def assert_legitimate_short_terminal(
    record: Mapping[str, Any], *, nominal_horizon: int | None = None, horizon: int | None = None
) -> None:
    if not validate_legitimate_short_terminal(record, nominal_horizon=nominal_horizon, horizon=horizon):
        raise ProvenanceError("terminal record is not a synchronous legitimate short terminal")


def main(argv: Sequence[str] | None = None) -> int:
    """Provide a dependency-free smokeable CLI surface for this primitive module."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-version", action="store_true", help="print the hard-gate schema version")
    args = parser.parse_args(argv)
    if args.schema_version:
        print(SCHEMA_VERSION)
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line convenience
    raise SystemExit(main())
