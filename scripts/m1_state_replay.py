#!/usr/bin/env python3
"""Fail-closed exact-state replay harness for the frozen ShiftVLA M1 pilot.

The module is deliberately import-light.  NumPy is the only runtime package
imported at module import time; the official LeRobot/MuJoCo environment is
loaded lazily by :func:`build_official_runtime` and only when an explicit real
run is requested.  Unit tests use the public pure helpers and the injectable
``RuntimeAdapter`` seam with fake objects.

M1 is an exact replay diagnostic, not a policy or repair implementation.  The
single source trajectory owns one persisted action tape and each capture is
restored into three independently constructed official environments.  Unknown
mutable state and any failed provenance/publication check fail closed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import argparse
import copy
from functools import wraps
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Callable, Iterator
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "m1" / "state_replay.yaml"
SCHEMA_VERSION = 1
# The surrounding M1 config/manifests remain schema v1.  ReplayState is a
# separately versioned persisted payload so v2 can add exact Python/RNG,
# model-identity, observation, task, and tape provenance without changing
# the older run-manifest contract.
REPLAY_STATE_SCHEMA_VERSION = 2
M0_CLOSURE_SHA256 = "1daf4b2d69b482c5b3110db283cb79a0fc74424a"
ACTION_TAPE_SEED = 12027
ACTION_TAPE_STEPS = 110
ACTION_DIM = 7
ARM_ACTION_DIM = 6
GRIPPER_BLOCK_STEPS = 10
CAPTURE_STEPS = (10, 50, 100)
FUTURE_STEPS = 10
RESTORES_PER_CAPTURE = 3
FLOATING_RTOL = 0.0
FLOATING_ATOL = 1.0e-12
INTEGRATION_SPEC_NAME = "mjSTATE_INTEGRATION"
POST_PROCESS_ALLOWLIST = frozenset(("vis_site_names", "model.site_rgba"))
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

# These values are duplicated in the checked-in YAML on purpose.  Validation
# compares both copies so an edited config cannot promote a different source,
# artifact, or runtime into the authoritative path.
FROZEN_OBS_TYPE = "pixels_agent_pos"
FROZEN_TRAJECTORY_ID = "libero_spatial-task000-init000"
FROZEN_SOURCE_CHECKOUTS = {
    "libero": {
        "path": "external/hf-libero",
        "git_sha": "8561c60eea2fb93096146f240194649df73d8b1e",
        "clean": True,
        "detached": True,
        "url": "https://github.com/huggingface/LIBERO.git",
    },
    "lerobot": {
        "path": "external/lerobot",
        "git_sha": "7e241bd630a3719a56157a497ce5d08f244784f1",
        "clean": True,
        "detached": True,
        "url": "https://github.com/huggingface/lerobot.git",
    },
    "robosuite": {
        "path": "external/robosuite",
        "git_sha": "fbee5844ff5632f5b5698e204ec5357ca50be0df",
        "clean": True,
        "detached": True,
        "url": "https://github.com/ARISE-Initiative/robosuite.git",
    },
    "mujoco": {
        "path": "external/mujoco",
        "git_sha": "72cb2b210da666617924de709406d6aadbe60c71",
        "clean": True,
        "detached": True,
        "url": "https://github.com/google-deepmind/mujoco.git",
    },
}
FROZEN_SOURCE_EVIDENCE_SHA256 = {
    "libero_env_wrapper": "a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0",
    "lerobot_libero": "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71",
    "robosuite_base": "16a044bf28cd39623c286d72f8be2f3789d53e215d77bef3b88ca33511c87687",
    "controller_base": "6dc278b614e78d8c3e7656b430c59cc600c42fcd1db0f4293bea1bf4925541dd",
    "osc": "a5887a950f53c16c67874c1e4f2ea6919dc0d0e5d1b618c15c699bd84e269477",
    "panda_gripper": "1c395d5ab96d097aa95e4542981901a6d54609d27fb72217788a7e63167052f5",
    "bddl_base_domain": "4f4da47dd241ac6590d66c7559d76e44b68669924207f53143ca3a4962921f24",
}
FROZEN_BDDL_SHA256 = "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b"
FROZEN_INIT_STATE_SHA256 = "cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a"
FROZEN_LIBERO_CONFIG_SHA256 = "3794964a45a33c0d545118d3c44064bae497931eadd14cb26974008cdd3b55af"
FROZEN_RUNTIME_LOCK_SHA256 = "921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc"
FROZEN_ARTIFACT_MANIFEST_SHA256 = "1a94ebb8cc42614744d8dc9ebdad16f5461006f8806cc7d75869118013198a85"
FROZEN_CHECKPOINT_SHA256 = "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"
FROZEN_BASE_MODEL_SHA256 = "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3"
FROZEN_ASSET_MANIFEST_SHA256 = FROZEN_ARTIFACT_MANIFEST_SHA256
FROZEN_CONFIG_CONTRACT_SHA256 = "730aff4a41fd91fb837102ca5f363a4a142bf7010140f5176940700f1d1fd5f0"
FROZEN_WORKTREE_AUDIT = {
    "timing": "pre_output",
    "allowed_output_root": "runs/m1_state_replay",
    "implementation_paths": (
        "scripts/m1_state_replay.py",
        "configs/m1/state_replay.yaml",
        "tests/test_m1_state_replay.py",
        "docs/m1_state_replay.md",
        "docs/superpowers/plans/2026-08-30-m1-exact-state-replay.md",
    ),
}
FROZEN_RUNTIME_ENVIRONMENT = {
    "LIBERO_CONFIG_PATH": "/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/m1/libero_config",
    "HF_HOME": "/public/home/xuyinghao/tmp/shiftvla-empty-hf-home-g",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MUJOCO_EGL_DEVICE_ID": "8",
}


class M1Error(RuntimeError):
    """Base class for a failed-closed M1 run."""


class M1ConfigError(M1Error):
    """Configuration or immutable identity is outside the frozen contract."""


class M1SchemaError(M1Error):
    """A persisted record, comparison, or state schema is malformed."""


class M1RuntimeError(M1Error):
    """The selected runtime did not satisfy the replay protocol."""


class UnknownMutableStateError(M1RuntimeError):
    """A stateful field reachable from the selected step path is unclassified."""


class ForbiddenRestoreError(M1RuntimeError):
    """A restore attempted a reset, public init-state, or settle operation."""


class PublicationError(M1Error):
    """An immutable artifact could not be published without overwrite."""


class EarlyTerminationError(M1RuntimeError):
    """The source trajectory terminated before the required horizon."""


def _json_default(value: Any) -> Any:
    """Convert only unambiguous NumPy scalars/arrays for canonical JSON."""

    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError("object arrays are not valid canonical JSON values")
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError("canonical JSON values must be finite")
        return value.tolist()
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, float) and not math.isfinite(scalar):
            raise ValueError("canonical JSON values must be finite")
        return scalar
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    # Comparison records are deliberately dataclasses rather than mutable
    # dictionaries at the comparison seam.  Convert them recursively at the
    # serialization boundary so persisted manifests never contain Python
    # objects or enum instances.
    if isinstance(value, ComparisonResult):
        return comparison_result_to_dict(value)
    raise TypeError(f"value of type {type(value).__name__} is not canonical JSON")


def canonical_json(value: Any) -> str:
    """Return deterministic, finite JSON with no insignificant whitespace."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise M1SchemaError(f"value is not canonical JSON: {exc}") from exc


def _config_contract_sha256(root: Mapping[str, Any]) -> str:
    """Hash the parsed frozen config while omitting its self-hash field."""

    value = copy.deepcopy(dict(root))
    paths = value.get("paths")
    if isinstance(paths, Mapping):
        paths = dict(paths)
        paths.pop("config_sha256", None)
        value["paths"] = paths
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _config_contract_sha256_file(path: str | Path) -> str:
    target = _resolve_path(path)
    if target.is_symlink() or not target.is_file():
        raise M1SchemaError(f"config contract path is not a regular file: {target}")
    try:
        import yaml  # type: ignore[import-not-found]

        parsed = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise M1SchemaError(f"could not parse config contract: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise M1SchemaError("config contract must contain a mapping")
    return _config_contract_sha256(parsed)


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    """Hash raw bytes without implicit text or array conversions."""

    return hashlib.sha256(bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise M1SchemaError(f"expected regular file for SHA-256: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_path(path: Path, *, allow_missing: bool = False) -> None:
    if path.is_symlink():
        raise PublicationError(f"symlinks are not allowed in M1 artifacts: {path}")
    if path.exists() and not path.is_file():
        raise PublicationError(f"artifact path is not a regular file: {path}")
    if not allow_missing and not path.exists():
        raise M1SchemaError(f"artifact does not exist: {path}")


def _atomic_create_bytes(path: str | Path, payload: bytes) -> Path:
    """Create one file atomically, refusing an existing target."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_path(target, allow_missing=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PublicationError(f"refusing to overwrite existing artifact: {target}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_create_text(path: str | Path, text: str) -> Path:
    return _atomic_create_bytes(path, text.encode("utf-8"))


def safe_save_npy(path: str | Path, array: np.ndarray) -> Path:
    """Persist one non-object float64 array without pickle or overwrite."""

    target = Path(path)
    value = np.asarray(array)
    if value.dtype.kind == "O":
        raise M1SchemaError("object/pickle arrays are not allowed")
    if value.dtype != np.dtype(np.float64):
        raise M1SchemaError(f"payload dtype must be float64, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise M1SchemaError("payload must contain only finite values")
    value = np.ascontiguousarray(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_path(target, allow_missing=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npy", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PublicationError(f"refusing to overwrite existing artifact: {target}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def safe_load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy payload with pickle explicitly disabled and exact dtype."""

    target = Path(path)
    _require_regular_path(target)
    try:
        value = np.load(target, allow_pickle=False)
    except Exception as exc:
        raise M1SchemaError(f"could not load safe NumPy payload: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise M1SchemaError("NumPy payload must be an ndarray")
    if value.dtype.kind == "O":
        raise M1SchemaError("object/pickle arrays are not allowed")
    if value.dtype != np.dtype(np.float64):
        raise M1SchemaError(f"payload dtype must be float64, got {value.dtype}")
    if not np.all(np.isfinite(value)):
        raise M1SchemaError("payload must contain only finite values")
    value = np.ascontiguousarray(value).copy()
    value.setflags(write=False)
    return value


def safe_save_uint8_npy(path: str | Path, array: np.ndarray) -> Path:
    """Persist one renderer image as immutable uint8 NumPy bytes."""

    target = Path(path)
    value = np.asarray(array)
    if value.dtype != np.dtype(np.uint8):
        raise M1SchemaError(f"RGB payload dtype must be uint8, got {value.dtype}")
    if value.ndim < 2 or value.size == 0:
        raise M1SchemaError("RGB payload must be a non-empty image")
    value = np.ascontiguousarray(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_path(target, allow_missing=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npy", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PublicationError(f"refusing to overwrite existing artifact: {target}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def safe_load_uint8_npy(path: str | Path) -> np.ndarray:
    """Load an immutable renderer image without pickle."""

    target = Path(path)
    _require_regular_path(target)
    try:
        value = np.load(target, allow_pickle=False)
    except Exception as exc:
        raise M1SchemaError(f"could not load safe RGB payload: {exc}") from exc
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.uint8):
        raise M1SchemaError("RGB payload must be a uint8 ndarray")
    if value.ndim < 2 or value.size == 0:
        raise M1SchemaError("RGB payload must be a non-empty image")
    value = np.ascontiguousarray(value).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class MatchedStateRecord:
    """Deterministic metadata plus a safe float64 payload."""

    metadata: Mapping[str, Any]
    payload: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.payload)
        if value.dtype != np.dtype(np.float64):
            raise M1SchemaError(f"record payload must be float64, got {value.dtype}")
        if value.dtype.kind == "O" or not np.all(np.isfinite(value)):
            raise M1SchemaError("record payload must be finite and non-object")
        if not isinstance(self.metadata, Mapping):
            raise M1SchemaError("record metadata must be a mapping")
        # Validate now so a record cannot carry a value that cannot be
        # reproduced by its canonical JSON envelope.
        canonical_json(dict(self.metadata))
        value = np.ascontiguousarray(value)
        value.setflags(write=False)
        object.__setattr__(self, "payload", value)
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    @property
    def payload_sha256(self) -> str:
        return sha256_bytes(self.payload.tobytes(order="C"))

    def envelope(self) -> dict[str, Any]:
        # Generic records remain v1 unless their metadata explicitly carries
        # the ReplayState v2 marker.  This preserves old captures while making
        # a v2 state bundle self-describing at both metadata and envelope
        # boundaries.
        record_schema = self.metadata.get("schema_version", 1)
        if record_schema not in (1, REPLAY_STATE_SCHEMA_VERSION):
            raise M1SchemaError(f"unsupported record schema version: {record_schema!r}")
        return {
            "schema_version": int(record_schema),
            "metadata": copy.deepcopy(dict(self.metadata)),
            "payload_dtype": "float64",
            "payload_shape": list(self.payload.shape),
            "payload_sha256": self.payload_sha256,
        }


def save_record_bundle(record: MatchedStateRecord, directory: str | Path, stem: str) -> tuple[Path, Path]:
    """Publish ``stem.json`` and ``stem.npy`` as one no-overwrite bundle."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{stem}.json"
    npy_path = target_dir / f"{stem}.npy"
    # Write the payload first; if metadata publication fails, the caller sees
    # the publication failure and never receives a falsely complete record.
    safe_save_npy(npy_path, record.payload)
    try:
        _atomic_create_text(json_path, canonical_json(record.envelope()) + "\n")
    except Exception:
        # The payload is retained as an immutable forensic artifact.  Nothing
        # is overwritten or silently removed after a publication error.
        raise
    return json_path, npy_path


def load_record_bundle(directory: str | Path, stem: str) -> MatchedStateRecord:
    target_dir = Path(directory)
    json_path = target_dir / f"{stem}.json"
    npy_path = target_dir / f"{stem}.npy"
    _require_regular_path(json_path)
    try:
        envelope = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise M1SchemaError(f"record metadata is not valid JSON: {exc}") from exc
    if not isinstance(envelope, Mapping):
        raise M1SchemaError("record metadata envelope must be a mapping")
    if envelope.get("schema_version") not in (1, REPLAY_STATE_SCHEMA_VERSION):
        raise M1SchemaError("unsupported record schema version")
    value = safe_load_npy(npy_path)
    expected_shape = tuple(envelope.get("payload_shape", ()))
    if tuple(value.shape) != expected_shape:
        raise M1SchemaError("payload shape does not match record metadata")
    if envelope.get("payload_dtype") != "float64":
        raise M1SchemaError("record payload dtype metadata must be float64")
    expected_hash = envelope.get("payload_sha256")
    if not isinstance(expected_hash, str) or expected_hash != sha256_bytes(value.tobytes(order="C")):
        raise M1SchemaError("record payload hash mismatch")
    metadata = envelope.get("metadata")
    if not isinstance(metadata, Mapping):
        raise M1SchemaError("record metadata must be a mapping")
    metadata_schema = metadata.get("schema_version")
    if metadata_schema is not None and metadata_schema != envelope.get("schema_version"):
        raise M1SchemaError("record envelope and metadata schema versions differ")
    return MatchedStateRecord(metadata=dict(metadata), payload=value)


def action_tape_sha256(tape: np.ndarray) -> str:
    value = np.asarray(tape)
    return sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def generate_action_tape(
    *,
    seed: int = ACTION_TAPE_SEED,
    n_steps: int = ACTION_TAPE_STEPS,
    action_dim: int = ACTION_DIM,
    arm_low: float = -0.2,
    arm_high: float = 0.2,
    gripper_block_steps: int = GRIPPER_BLOCK_STEPS,
    zero_orientation_steps: Sequence[int] = (11, 51, 101),
) -> np.ndarray:
    """Generate the one immutable PCG64 action tape from the frozen contract."""

    if action_dim != 7 or n_steps != 110 or seed != ACTION_TAPE_SEED:
        raise M1ConfigError("M1 action tape dimensions and seed are immutable")
    if gripper_block_steps != 10:
        raise M1ConfigError("M1 gripper block length is immutable")
    if not math.isfinite(arm_low) or not math.isfinite(arm_high) or arm_low >= arm_high:
        raise M1ConfigError("arm action bounds are invalid")
    rng = np.random.Generator(np.random.PCG64(seed))
    tape = np.empty((n_steps, action_dim), dtype=np.float32)
    tape[:, :ARM_ACTION_DIM] = rng.uniform(
        arm_low, arm_high, size=(n_steps, ARM_ACTION_DIM)
    ).astype(np.float32)
    blocks = np.asarray((-1.0, 1.0), dtype=np.float32)
    tape[:, 6] = np.resize(np.repeat(blocks, gripper_block_steps), n_steps)
    for one_based_step in zero_orientation_steps:
        if one_based_step < 1 or one_based_step > n_steps:
            raise M1ConfigError(f"orientation-zero action step is out of range: {one_based_step}")
        tape[one_based_step - 1, 3:6] = np.float32(0.0)
    if not np.all(np.isfinite(tape)):
        raise M1SchemaError("generated action tape is non-finite")
    tape.setflags(write=False)
    return tape


def validate_action_tape(
    tape: np.ndarray,
    *,
    expected_sha256: str | None = None,
) -> np.ndarray:
    value = np.asarray(tape)
    if value.shape != (ACTION_TAPE_STEPS, ACTION_DIM) or value.dtype != np.dtype(np.float32):
        raise M1SchemaError("action tape must have exact float32 shape (110, 7)")
    if not np.all(np.isfinite(value)):
        raise M1SchemaError("action tape must be finite")
    if np.any(value[:, :6] < np.float32(-0.2)) or np.any(value[:, :6] > np.float32(0.2)):
        raise M1SchemaError("arm actions fall outside the frozen uniform bounds")
    expected_gripper = np.resize(np.repeat(np.asarray((-1.0, 1.0), dtype=np.float32), 10), 110)
    if not np.array_equal(value[:, 6], expected_gripper):
        raise M1SchemaError("gripper action blocks do not match the frozen tape")
    for step in (11, 51, 101):
        if not np.array_equal(value[step - 1, 3:6], np.zeros(3, dtype=np.float32)):
            raise M1SchemaError(f"orientation dimensions of action {step} are not zero")
    if expected_sha256 is not None and action_tape_sha256(value) != str(expected_sha256).lower():
        raise M1SchemaError("action tape hash mismatch")
    value = np.ascontiguousarray(value).copy()
    value.setflags(write=False)
    return value


def persist_action_tape(
    path: str | Path,
    tape: np.ndarray,
    *,
    expected_sha256: str | None = None,
) -> str:
    value = validate_action_tape(tape, expected_sha256=expected_sha256)
    _safe_save_action_tape(path, value)
    return action_tape_sha256(value)


def _safe_save_action_tape(path: str | Path, tape: np.ndarray) -> Path:
    value = validate_action_tape(tape)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_path(target, allow_missing=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npy", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
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


def persist_action_tape_exact(
    path: str | Path,
    tape: np.ndarray,
    *,
    expected_sha256: str | None = None,
) -> str:
    value = validate_action_tape(tape, expected_sha256=expected_sha256)
    _safe_save_action_tape(path, value)
    return action_tape_sha256(value)


def load_action_tape(path: str | Path, *, expected_sha256: str | None = None) -> np.ndarray:
    target = Path(path)
    _require_regular_path(target)
    try:
        value = np.load(target, allow_pickle=False)
    except Exception as exc:
        raise M1SchemaError(f"could not load action tape: {exc}") from exc
    if not isinstance(value, np.ndarray):
        raise M1SchemaError("action tape must be an ndarray")
    return validate_action_tape(value, expected_sha256=expected_sha256)


@dataclass(frozen=True)
class CaptureWindow:
    """One post-step capture and its inclusive reference-state window."""

    capture_step: int
    state_steps: tuple[int, ...]
    action_indices: tuple[int, ...]
    phase: str = "after_env_step_before_next_action"


def capture_phase(step: int, future_steps: int) -> str:
    if step <= 0 or future_steps <= 0:
        raise M1SchemaError("capture step and future length must be positive")
    return "after_env_step_before_next_action"


def build_capture_windows(
    capture_steps: Sequence[int] = CAPTURE_STEPS,
    *,
    future_steps: int = FUTURE_STEPS,
    total_steps: int = ACTION_TAPE_STEPS,
) -> tuple[CaptureWindow, ...]:
    values = tuple(int(step) for step in capture_steps)
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise M1SchemaError("capture steps must be sorted and unique")
    if not values or any(step <= 0 or step >= total_steps for step in values):
        raise M1SchemaError("capture step must be after a real step and before the tape end")
    if future_steps <= 0 or any(step + future_steps > total_steps for step in values):
        raise M1SchemaError("capture step plus future window exceeds the source tape")
    return tuple(
        CaptureWindow(
            capture_step=step,
            state_steps=tuple(range(step, step + future_steps + 1)),
            action_indices=tuple(range(step, step + future_steps)),
        )
        for step in values
    )


class ComparisonClass(str, Enum):
    EXACT = "exact"
    FLOATING_PHYSICAL = "floating_physical"
    DERIVED_DIAGNOSTIC = "derived_diagnostic"


@dataclass(frozen=True)
class ComparisonResult:
    comparison_class: ComparisonClass
    passed: bool
    path: str = ""
    detail: str = ""
    gates_pass: bool = True
    max_abs: float | None = None
    mean_abs: float | None = None
    tolerance: float | None = None
    different_count: int | None = None
    subresults: Mapping[str, "ComparisonResult"] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return self.passed == other
        if not isinstance(other, ComparisonResult):
            return NotImplemented
        return (
            self.comparison_class == other.comparison_class
            and self.passed == other.passed
            and self.path == other.path
            and self.detail == other.detail
            and self.gates_pass == other.gates_pass
            and self.max_abs == other.max_abs
            and self.mean_abs == other.mean_abs
            and self.tolerance == other.tolerance
            and self.different_count == other.different_count
            and dict(self.subresults) == dict(other.subresults)
        )


class _CountsDict(dict[str, int]):
    """Manifest count mapping with compatibility for the original 3-field API."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping) and "static_comparisons" not in other:
            return {key: value for key, value in self.items() if key != "static_comparisons"} == dict(other)
        return dict.__eq__(self, other)


def comparison_result_to_dict(result: ComparisonResult) -> dict[str, Any]:
    """Return a fully JSON-safe comparison result, including child rows."""

    return {
        "comparison_class": result.comparison_class.value,
        "passed": bool(result.passed),
        "path": result.path,
        "detail": result.detail,
        "gates_pass": bool(result.gates_pass),
        "max_abs": result.max_abs,
        "mean_abs": result.mean_abs,
        "tolerance": result.tolerance,
        "different_count": result.different_count,
        "subresults": {
            str(key): comparison_result_to_dict(child)
            for key, child in sorted(result.subresults.items(), key=lambda item: str(item[0]))
        },
    }


def _exact_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        if not isinstance(expected, np.ndarray) or not isinstance(actual, np.ndarray):
            return False
        return expected.dtype == actual.dtype and expected.shape == actual.shape and np.array_equal(
            expected, actual, equal_nan=False
        )
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return False
        return set(expected) == set(actual) and all(_exact_equal(expected[k], actual[k]) for k in expected)
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not isinstance(expected, (list, tuple)) or not isinstance(actual, (list, tuple)):
            return False
        return len(expected) == len(actual) and all(_exact_equal(a, b) for a, b in zip(expected, actual))
    if isinstance(expected, (float, np.floating)) and isinstance(actual, (float, np.floating)):
        return bool(math.isfinite(float(expected)) and math.isfinite(float(actual)) and float(expected) == float(actual))
    return type(expected) is type(actual) and expected == actual


def compare_exact(expected: Any, actual: Any, *, path: str = "") -> ComparisonResult:
    passed = _exact_equal(expected, actual)
    return ComparisonResult(
        ComparisonClass.EXACT,
        passed,
        path=path,
        detail="" if passed else "exact values differ",
    )


def compare_floating(
    expected: Any,
    actual: Any,
    *,
    path: str = "",
    rtol: float = FLOATING_RTOL,
    atol: float = FLOATING_ATOL,
    quaternion_sign_invariant: bool = False,
) -> ComparisonResult:
    if rtol != FLOATING_RTOL or atol != FLOATING_ATOL:
        raise M1ConfigError("floating comparison tolerances are frozen at rtol=0 and atol=1e-12")

    def floating_equal(left_value: Any, right_value: Any) -> tuple[bool, float | None, float | None]:
        if left_value is None or right_value is None:
            passed = left_value is None and right_value is None
            return passed, 0.0 if passed else None, 0.0 if passed else None
        if isinstance(left_value, Mapping) or isinstance(right_value, Mapping):
            if not isinstance(left_value, Mapping) or not isinstance(right_value, Mapping):
                return False, None, None
            if set(left_value) != set(right_value):
                return False, None, None
            metrics = [floating_equal(left_value[key], right_value[key]) for key in left_value]
            finite_metrics = [(maximum, mean) for passed, maximum, mean in metrics if maximum is not None and mean is not None]
            return (
                all(passed for passed, _maximum, _mean in metrics),
                max((maximum for maximum, _mean in finite_metrics), default=None),
                float(np.mean([mean for _maximum, mean in finite_metrics])) if finite_metrics else None,
            )
        left_array = np.asarray(left_value)
        right_array = np.asarray(right_value)
        if left_array.shape != right_array.shape:
            return False, None, None
        try:
            if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
                return False, None, None
            left_float = np.asarray(left_array, dtype=np.float64)
            right_float = np.asarray(right_array, dtype=np.float64)
            if quaternion_sign_invariant and left_float.ndim >= 1 and left_float.shape[-1] == 4:
                left_rows = left_float.reshape((-1, 4))
                right_rows = right_float.reshape((-1, 4))
                differences = []
                for left_row, right_row in zip(left_rows, right_rows):
                    direct = np.abs(left_row - right_row)
                    flipped = np.abs(left_row + right_row)
                    differences.append(flipped if float(np.max(flipped)) < float(np.max(direct)) else direct)
                diff = np.concatenate(differences) if differences else np.asarray([], dtype=np.float64)
            else:
                diff = np.abs(left_float - right_float).reshape(-1)
            maximum = float(np.max(diff)) if diff.size else 0.0
            mean = float(np.mean(diff)) if diff.size else 0.0
            passed_here = bool(np.all(diff <= np.float64(atol)))
            return passed_here, maximum, mean
        except (TypeError, ValueError):
            return False, None, None

    passed, maximum, mean = floating_equal(expected, actual)
    return ComparisonResult(
        ComparisonClass.FLOATING_PHYSICAL,
        passed,
        path=path,
        detail="" if passed else "floating values differ beyond the exact M1 gate",
        max_abs=maximum,
        mean_abs=mean,
        tolerance=float(atol),
    )


def compare_diagnostic(expected: Any, actual: Any, *, path: str = "") -> ComparisonResult:
    passed = _exact_equal(expected, actual)
    # Diagnostic mismatches are recorded but intentionally do not gate M1.
    return ComparisonResult(
        ComparisonClass.DERIVED_DIAGNOSTIC,
        passed,
        path=path,
        detail="" if passed else "derived diagnostic differs",
        gates_pass=True,
    )


def compare_rgb(expected: Any, actual: Any, *, path: str = "rgb") -> ComparisonResult:
    """Compare persisted uint8 RGB captures as non-gating diagnostics."""

    try:
        left = np.asarray(expected)
        right = np.asarray(actual)
        if left.dtype != np.dtype(np.uint8) or right.dtype != np.dtype(np.uint8):
            raise ValueError("RGB captures must be uint8")
        if left.shape != right.shape or left.ndim < 2:
            return ComparisonResult(
                ComparisonClass.DERIVED_DIAGNOSTIC,
                False,
                path=path,
                detail="RGB shape or dimensionality differs",
                gates_pass=True,
            )
        difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
        if difference.ndim >= 3:
            different_count = int(np.count_nonzero(np.any(difference != 0, axis=-1)))
        else:
            different_count = int(np.count_nonzero(difference))
        flat = difference.reshape(-1)
        return ComparisonResult(
            ComparisonClass.DERIVED_DIAGNOSTIC,
            bool(different_count == 0),
            path=path,
            detail="" if different_count == 0 else "RGB pixels differ",
            gates_pass=True,
            max_abs=float(np.max(flat)) if flat.size else 0.0,
            mean_abs=float(np.mean(flat)) if flat.size else 0.0,
            tolerance=0.0,
            different_count=different_count,
        )
    except (TypeError, ValueError):
        return ComparisonResult(
            ComparisonClass.DERIVED_DIAGNOSTIC,
            False,
            path=path,
            detail="RGB capture is not a valid uint8 image",
            gates_pass=True,
        )


def _contact_item(value: Any) -> tuple[str, str, float]:
    if isinstance(value, Mapping):
        first = value.get("geom1", value.get("geom_a", value.get("first")))
        second = value.get("geom2", value.get("geom_b", value.get("second")))
        distance = value.get("distance", value.get("dist"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 3:
            raise M1SchemaError("contact pair sequence must have two geoms and distance")
        first, second, distance = value[0], value[1], value[2]
    else:
        first = getattr(value, "geom1", getattr(value, "geom_a", None))
        second = getattr(value, "geom2", getattr(value, "geom_b", None))
        distance = getattr(value, "distance", getattr(value, "dist", None))
    if first is None or second is None or distance is None:
        raise M1SchemaError(f"contact pair is missing geometry or distance: {value!r}")
    distance_float = float(distance)
    if not math.isfinite(distance_float):
        raise M1SchemaError("contact distance must be finite")
    left, right = str(first), str(second)
    if right < left:
        left, right = right, left
    return left, right, distance_float


def canonicalize_contact_pairs(pairs: Iterable[Any]) -> tuple[tuple[str, str, float], ...]:
    return tuple(sorted((_contact_item(pair) for pair in pairs), key=lambda item: (item[0], item[1], item[2])))


def _runtime_contact_pairs(data: Any, model: Any, mujoco_module: Any | None) -> tuple[tuple[str, str, float], ...]:
    """Extract MuJoCo contacts as named canonical pairs without mutating data."""

    raw_pairs = getattr(data, "contact_pairs", None)
    if raw_pairs is not None:
        return canonicalize_contact_pairs(raw_pairs)
    try:
        count = int(getattr(data, "ncon", 0))
    except (TypeError, ValueError):
        count = 0
    contacts = getattr(data, "contact", None)
    if count <= 0 or contacts is None:
        return ()
    raw_model = _raw_mujoco_handle(model, "_model")
    geom_name = getattr(raw_model, "geom_id2name", None)
    if not callable(geom_name) and mujoco_module is not None:
        obj_name = getattr(mujoco_module, "mjOBJ_GEOM", None)
        object_enum = getattr(mujoco_module, "mjtObj", None)
        if obj_name is None and object_enum is not None:
            obj_name = getattr(object_enum, "mjOBJ_GEOM", None)
        id2name = getattr(mujoco_module, "mj_id2name", None)
        if obj_name is not None and callable(id2name):
            geom_name = lambda index, _obj=obj_name, _fn=id2name: _fn(raw_model, _obj, index)
    pairs: list[tuple[Any, Any, float]] = []
    for index in range(count):
        try:
            contact = contacts[index]
            geom1 = int(getattr(contact, "geom1"))
            geom2 = int(getattr(contact, "geom2"))
            distance = float(getattr(contact, "dist"))
            left = geom_name(geom1) if callable(geom_name) else geom1
            right = geom_name(geom2) if callable(geom_name) else geom2
            pairs.append((left, right, distance))
        except Exception as exc:
            raise M1RuntimeError(f"could not collect MuJoCo contact {index}: {exc}") from exc
    return canonicalize_contact_pairs(pairs)


def compare_contact_pairs(expected: Iterable[Any], actual: Iterable[Any], *, path: str = "contacts") -> ComparisonResult:
    left = canonicalize_contact_pairs(expected)
    right = canonicalize_contact_pairs(actual)
    if len(left) != len(right):
        pair_result = compare_exact([item[:2] for item in left], [item[:2] for item in right], path=f"{path}.geom_pairs")
        distance_result = compare_floating([], [], path=f"{path}.distances")
        return ComparisonResult(
            ComparisonClass.EXACT,
            False,
            path=path,
            detail="contact count differs",
            gates_pass=False,
            subresults={"geom_pairs": pair_result, "distances": distance_result},
        )
    pair_result = compare_exact([item[:2] for item in left], [item[:2] for item in right], path=f"{path}.geom_pairs")
    distance_result = compare_floating(
        [item[2] for item in left], [item[2] for item in right], path=f"{path}.distances"
    )
    passed = pair_result.passed and distance_result.passed
    return ComparisonResult(
        ComparisonClass.EXACT,
        passed,
        path=path,
        detail="" if passed else "canonical contact geometry or distance differs",
        gates_pass=passed,
        subresults={"geom_pairs": pair_result, "distances": distance_result},
    )


SERIALIZED = "SERIALIZED"
DERIVED_RECONSTRUCTED = "DERIVED/RECONSTRUCTED"
IMMUTABLE = "IMMUTABLE"
PROVEN_UNUSED = "PROVEN UNUSED"
STATE_CLASSES = frozenset((SERIALIZED, DERIVED_RECONSTRUCTED, IMMUTABLE, PROVEN_UNUSED))

DEFAULT_STATE_CLASSIFICATION: dict[str, tuple[str, ...]] = {
    SERIALIZED: (
        "mjdata.integration",
        "fixture.model.body_pos",
        "fixture.model.body_quat",
        "env.timestep",
        "env.cur_time",
        "env.done",
        "controller.initial_joint",
        "controller.goal_pos",
        "controller.goal_ori",
        "controller.new_update",
        "gripper.current_action",
        "identity.task",
        "identity.instruction",
        "identity.source_step",
        "identity.action_tape_sha256",
        "observable.timers_cache",
    ),
    DERIVED_RECONSTRUCTED: (
        "mjdata.kinematics",
        "mjdata.contacts",
        "eef.pose",
        "objects.pose",
        "objects.joints",
        "controller.kinematic_caches",
        "observables.values",
        "task0.visual_state",
    ),
    IMMUTABLE: (
        "task.instruction",
        "bddl.xml.assets.pins",
        "model.layout",
        "model.ordered_names_dims",
        "controller.configuration_gains",
        "renderer.configuration",
        "action_tape.algorithm_config_hash",
        "comparison.schema_tolerances",
    ),
    PROVEN_UNUSED: (
        "policy.processors.noise",
        "robot.recent_buffers",
        "viewer.internals",
        "post_reset.rng",
    ),
}


def _flatten_classification(classification: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(classification, Mapping):
        raise M1SchemaError("state classification must be a mapping")
    flattened: dict[str, str] = {}
    keys = set(str(key) for key in classification)
    if keys <= STATE_CLASSES:
        for state_class, fields in classification.items():
            if not isinstance(fields, Iterable) or isinstance(fields, (str, bytes)):
                raise M1SchemaError(f"classification {state_class} must contain field names")
            for field_name in fields:
                field = str(field_name)
                if field in flattened:
                    raise M1SchemaError(f"field appears in multiple state classes: {field}")
                flattened[field] = str(state_class)
        return flattened
    # Also accept the convenient field -> class form for callers loading a
    # machine-generated audit report.
    for field_name, state_class in classification.items():
        if str(state_class) not in STATE_CLASSES:
            raise M1SchemaError(f"unknown state class for {field_name}: {state_class}")
        if str(field_name) in flattened:
            raise M1SchemaError(f"duplicate state field: {field_name}")
        flattened[str(field_name)] = str(state_class)
    return flattened


def validate_state_classification(
    classification: Mapping[str, Any],
    *,
    reachable_mutable_fields: Iterable[str] | None = None,
) -> dict[str, str]:
    flattened = _flatten_classification(classification)
    if set(flattened.values()) != STATE_CLASSES:
        missing = sorted(STATE_CLASSES - set(flattened.values()))
        raise UnknownMutableStateError(f"state classification must include exactly four classes; missing {missing}")
    if reachable_mutable_fields is not None:
        reachable = {str(field) for field in reachable_mutable_fields}
        unknown = sorted(reachable - set(flattened))
        if unknown:
            raise UnknownMutableStateError(f"unknown/unclassified mutable state: {unknown}")
    return flattened


def audit_mutable_state(
    reachable_mutable_fields: Iterable[str],
    classification: Mapping[str, Any] = DEFAULT_STATE_CLASSIFICATION,
) -> dict[str, str]:
    return validate_state_classification(
        classification, reachable_mutable_fields=reachable_mutable_fields
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M1ConfigError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise M1ConfigError(f"config must be a regular file: {target}")
    try:
        import yaml  # type: ignore[import-not-found]

        value = yaml.safe_load(target.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        raise M1ConfigError("PyYAML is required to load the M1 config")
    except Exception as exc:
        raise M1ConfigError(f"could not parse M1 YAML config: {exc}") from exc
    return dict(_mapping(value, "config"))


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(config, "config")
    if root.get("schema_version") != 1 or root.get("name") != "state_replay":
        raise M1ConfigError("M1 config schema/name is not frozen")
    task = _mapping(root.get("task"), "task")
    expected_task = {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027}
    if dict(task) != expected_task:
        raise M1ConfigError(f"task must equal the frozen M1 task identity, got {task!r}")
    tape = _mapping(root.get("action_tape"), "action_tape")
    if tape.get("seed") != ACTION_TAPE_SEED or tape.get("shape") != [110, 7]:
        raise M1ConfigError("action tape seed/shape do not match the frozen contract")
    if tape.get("dtype") != "float32" or tape.get("arm_low") != -0.2 or tape.get("arm_high") != 0.2:
        raise M1ConfigError("action tape dtype/bounds do not match the frozen contract")
    if tape.get("gripper_block_steps") != 10 or tape.get("zero_orientation_steps") != [11, 51, 101]:
        raise M1ConfigError("action tape gripper/orientation contract does not match")
    if "algorithm" in tape and tape.get("algorithm") != "numpy.Generator":
        raise M1ConfigError("action tape algorithm must be the frozen numpy.Generator")
    if "bit_generator" in tape and tape.get("bit_generator") != "PCG64":
        raise M1ConfigError("action tape bit_generator must be PCG64")
    if "gripper_values" in tape and tape.get("gripper_values") != [-1.0, 1.0]:
        raise M1ConfigError("action tape gripper_values are frozen at [-1.0, 1.0]")
    captures = root.get("capture_steps")
    if captures != [10, 50, 100] or root.get("future_steps") != 10 or root.get("restores_per_capture") != 3:
        raise M1ConfigError("capture/restore counts do not match the frozen contract")
    tolerances = _mapping(root.get("tolerances"), "tolerances")
    if tolerances.get("rtol") != 0.0 or tolerances.get("atol") != 1.0e-12:
        raise M1ConfigError("tolerances must be exactly rtol=0 and atol=1e-12")
    paths = _mapping(root.get("paths"), "paths")
    for key in ("action_tape", "output_root", "cpu_python", "config"):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise M1ConfigError(f"paths.{key} must be a non-empty string")
    runtime = _mapping(root.get("runtime"), "runtime")
    if runtime.get("include_policy") is not False or runtime.get("offline") is not True:
        raise M1ConfigError("M1 runtime must be offline and include_policy=false")
    if "strict_provenance" in root and root.get("strict_provenance") is not True:
        raise M1ConfigError("strict_provenance must remain true for the frozen M1 configuration")
    # The checked-in experiment config opts into the complete frozen schema.
    # Small injected configs used by pure adapter tests intentionally omit
    # source/artifact pins, but once a state audit is supplied we must reject
    # silently substituted defaults and validate every frozen field exactly.
    if "state_audit" in root or bool(root.get("strict_provenance", False)):
        _validate_strict_config(root)
    return dict(root)


def _validate_strict_config(root: Mapping[str, Any]) -> None:
    if root.get("strict_provenance") is not True:
        raise M1ConfigError("strict_provenance must be exactly true for the frozen authoritative M1 config")
    if root.get("obs_type") != FROZEN_OBS_TYPE:
        raise M1ConfigError(f"obs_type must equal the frozen value {FROZEN_OBS_TYPE!r}")
    if root.get("trajectory_id") != FROZEN_TRAJECTORY_ID:
        raise M1ConfigError("trajectory_id is not the frozen M1 trajectory")
    tape = _mapping(root.get("action_tape"), "action_tape")
    expected_tape = {
        "algorithm": "numpy.Generator",
        "bit_generator": "PCG64",
        "seed": ACTION_TAPE_SEED,
        "shape": [ACTION_TAPE_STEPS, ACTION_DIM],
        "dtype": "float32",
        "arm_low": -0.2,
        "arm_high": 0.2,
        "gripper_block_steps": 10,
        "gripper_values": [-1.0, 1.0],
        "zero_orientation_steps": [11, 51, 101],
    }
    for key, expected in expected_tape.items():
        if tape.get(key) != expected:
            raise M1ConfigError(f"frozen action_tape.{key} must equal {expected!r}")
    tape_hash = tape.get("sha256")
    if not isinstance(tape_hash, str) or len(tape_hash) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in tape_hash):
        raise M1ConfigError("frozen action_tape.sha256 is required")
    if root.get("source_steps") != ACTION_TAPE_STEPS:
        raise M1ConfigError("source_steps must be exactly 110")

    runtime = _mapping(root.get("runtime"), "runtime")
    expected_runtime = {
        "offline": True,
        "local_only": True,
        "include_policy": False,
        "call_policy": False,
        "call_processors": False,
        "use_async_envs": False,
        "n_envs": 1,
        "controller_class": "OperationalSpaceController",
        "gripper_class": "PandaGripper",
        "sleeping": False,
        "environment": {**FROZEN_RUNTIME_ENVIRONMENT, "empty_hf_cache": True},
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise M1ConfigError(f"runtime.{key} must equal frozen value {expected!r}")
    renderer = _mapping(runtime.get("renderer"), "runtime.renderer")
    expected_renderer = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "8",
        "expected_gl": {
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
            "version": "3.1 Mesa 21.1.5",
        },
    }
    if dict(renderer) != expected_renderer:
        raise M1ConfigError("runtime.renderer does not equal the frozen EGL contract")

    paths = _mapping(root.get("paths"), "paths")
    expected_paths = {
        "config": "configs/m1/state_replay.yaml",
        "action_tape": "action_tape.npy",
        "output_root": "runs/m1_state_replay",
        "cpu_python": "/public/home/xuyinghao/tmp/shiftvla-libero/bin/python",
        "official_factory": "scripts.dcu_preflight.build_cpu_runtime",
        "runtime_lock": "runtime/locks/shiftvla-libero-runtime.txt",
        "runtime_lock_sha256": FROZEN_RUNTIME_LOCK_SHA256,
        "artifact_manifest": "runtime/manifests/m0_preflight_b_artifacts.json",
        "artifact_manifest_sha256": FROZEN_ARTIFACT_MANIFEST_SHA256,
        "bddl": "external/hf-libero/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
        "init_state": "external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init",
    }
    for key, expected in expected_paths.items():
        if paths.get(key) != expected:
            raise M1ConfigError(f"paths.{key} must equal the frozen value {expected!r}")
    for key in ("official_factory", "runtime_lock", "artifact_manifest"):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise M1ConfigError(f"paths.{key} is required by the frozen provenance contract")
    source_evidence = _mapping(paths.get("source_evidence"), "paths.source_evidence")
    if dict(source_evidence) != {
        "libero_env_wrapper": "external/hf-libero/libero/libero/envs/env_wrapper.py",
        "lerobot_libero": "external/lerobot/src/lerobot/envs/libero.py",
        "robosuite_base": "external/robosuite/robosuite/environments/base.py",
        "controller_base": "external/robosuite/robosuite/controllers/base_controller.py",
        "osc": "external/robosuite/robosuite/controllers/osc.py",
        "panda_gripper": "external/robosuite/robosuite/models/grippers/panda_gripper.py",
        "bddl_base_domain": "external/hf-libero/libero/libero/envs/bddl_base_domain.py",
    }:
        raise M1ConfigError("paths.source_evidence is not the exact frozen source list")
    source_evidence_hashes = _mapping(
        paths.get("source_evidence_sha256"), "paths.source_evidence_sha256"
    )
    if dict(source_evidence_hashes) != FROZEN_SOURCE_EVIDENCE_SHA256:
        raise M1ConfigError("paths.source_evidence_sha256 is not the exact frozen digest map")

    source_checkouts = _mapping(root.get("source_checkouts"), "source_checkouts")
    if dict(source_checkouts) != FROZEN_SOURCE_CHECKOUTS:
        raise M1ConfigError("source_checkouts are not the exact frozen checkout audit")
    worktree_audit = _mapping(root.get("worktree_audit"), "worktree_audit")
    expected_worktree_audit = {
        "timing": FROZEN_WORKTREE_AUDIT["timing"],
        "allowed_output_root": FROZEN_WORKTREE_AUDIT["allowed_output_root"],
        "implementation_paths": list(FROZEN_WORKTREE_AUDIT["implementation_paths"]),
    }
    if dict(worktree_audit) != expected_worktree_audit:
        raise M1ConfigError("worktree_audit is not the exact pre-output frozen audit")
    if paths.get("config_sha256") != FROZEN_CONFIG_CONTRACT_SHA256:
        raise M1ConfigError("paths.config_sha256 is not the frozen config-contract digest")
    if _config_contract_sha256(root) != FROZEN_CONFIG_CONTRACT_SHA256:
        raise M1ConfigError("config contents do not match the frozen config-contract digest")

    expected_sections = {
        "checkpoint": {
            "repo_id": "HuggingFaceVLA/smolvla_libero",
            "revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
            "path": "/public/home/xuyinghao/workspace/vla/ShiftVLA/external/artifacts/smolvla_libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
            "sha256": FROZEN_CHECKPOINT_SHA256,
        },
        "base_model": {
            "repo_id": "HuggingFaceTB/SmolVLM2-500M-Instruct",
            "revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            "path": "/public/home/xuyinghao/workspace/vla/ShiftVLA/external/artifacts/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            "sha256": FROZEN_BASE_MODEL_SHA256,
        },
        "assets": {
            "repo_id": "lerobot/libero-assets",
            "revision": "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
            "path": "/public/home/xuyinghao/workspace/vla/ShiftVLA/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
            "manifest_path": "/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/manifests/m0_preflight_b_artifacts.json",
            "manifest_sha256": FROZEN_ASSET_MANIFEST_SHA256,
        },
        "libero_config": {
            "path": "/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/m1/libero_config/config.yaml",
            "sha256": FROZEN_LIBERO_CONFIG_SHA256,
        },
    }
    for section_name in ("checkpoint", "base_model", "assets", "libero_config"):
        section = _mapping(root.get(section_name), section_name)
        if dict(section) != expected_sections[section_name]:
            raise M1ConfigError(f"{section_name} is not the exact frozen identity/digest record")
        for key in ("repo_id", "revision", "path") if section_name != "libero_config" else ("path", "sha256"):
            if not isinstance(section.get(key), str) or not section[key].strip():
                raise M1ConfigError(f"{section_name}.{key} is required")
        digest_key = "manifest_sha256" if section_name == "assets" else "sha256"
        digest = section.get(digest_key)
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise M1ConfigError(f"{section_name}.{digest_key} must be a 64-character SHA-256")

    pins = _mapping(root.get("pins"), "pins")
    expected_pins = {
        "lerobot_git_sha": FROZEN_SOURCE_CHECKOUTS["lerobot"]["git_sha"],
        "libero_git_sha": FROZEN_SOURCE_CHECKOUTS["libero"]["git_sha"],
        "robosuite_git_sha": FROZEN_SOURCE_CHECKOUTS["robosuite"]["git_sha"],
        "mujoco_git_sha": FROZEN_SOURCE_CHECKOUTS["mujoco"]["git_sha"],
        "pytorch": "2.11.0+cpu",
        "transformers": "5.5.4",
        "python": "3.12.6",
        "numpy": "2.2.6",
        "gymnasium": "1.3.0",
        "mujoco": "3.7.0",
        "lerobot": "0.6.1",
        "robosuite": "1.4.0",
        "hf_libero": "0.1.4",
        "cuda": "none",
        "gpu": "CPU",
    }
    if dict(pins) != expected_pins:
        raise M1ConfigError("pins are not the exact frozen runtime pins")
    required_pins = (
        "lerobot_git_sha",
        "libero_git_sha",
        "robosuite_git_sha",
        "mujoco_git_sha",
        "pytorch",
        "transformers",
        "python",
        "numpy",
        "gymnasium",
        "mujoco",
        "robosuite",
        "hf_libero",
        "cuda",
        "gpu",
    )
    for key in required_pins:
        value = pins.get(key)
        if not isinstance(value, str) or not value.strip() or "pending" in value.lower():
            raise M1ConfigError(f"pins.{key} must be a resolved non-pending value")
    for key in required_pins[:4]:
        if len(str(pins[key])) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in str(pins[key])):
            raise M1ConfigError(f"pins.{key} must be a 40-character Git SHA")

    init_evidence = root.get("init_state_id_evidence")
    if init_evidence != 0:
        raise M1ConfigError("init_state_id_evidence must preserve the requested pre-reset selection 0")

    closure_sha = root.get("m0_closure_sha256")
    if closure_sha != M0_CLOSURE_SHA256:
        raise M1ConfigError("m0_closure_sha256 does not match the closed M0 evidence")
    closure = _mapping(root.get("m0_closure"), "m0_closure")
    if closure.get("sha256") != M0_CLOSURE_SHA256 or not isinstance(closure.get("path"), str):
        raise M1ConfigError("m0_closure path/hash is required")

    expected_hashes = {
        "bddl": {
            "path": expected_paths["bddl"],
            "sha256": FROZEN_BDDL_SHA256,
        },
        "init_state": {
            "path": expected_paths["init_state"],
            "sha256": FROZEN_INIT_STATE_SHA256,
        },
    }
    if dict(_mapping(root.get("hashes"), "hashes")) != expected_hashes:
        raise M1ConfigError("hashes are not the exact frozen BDDL/init-state digest record")

    state_audit = _mapping(root.get("state_audit"), "state_audit")
    classification = state_audit.get("classification")
    reachable = state_audit.get("reachable_fields")
    if not isinstance(classification, Mapping) or not isinstance(reachable, Iterable) or isinstance(reachable, (str, bytes)):
        raise M1ConfigError("state_audit classification and reachable_fields are required")
    expected_reachable = sorted(field for fields in DEFAULT_STATE_CLASSIFICATION.values() for field in fields)
    if sorted(str(field) for field in reachable) != expected_reachable:
        raise M1ConfigError("state_audit.reachable_fields is not the explicit frozen list")
    flattened = _flatten_classification(classification)
    expected_flattened = _flatten_classification(DEFAULT_STATE_CLASSIFICATION)
    if flattened != expected_flattened:
        raise M1ConfigError("state_audit.classification is not the exact four-class frozen map")
    post_process = _mapping(root.get("post_process_guard"), "post_process_guard")
    if post_process.get("enabled") is not True or tuple(post_process.get("allowlist", ())) != tuple(sorted(POST_PROCESS_ALLOWLIST)):
        raise M1ConfigError("post_process_guard flags/allowlist are not frozen")
    publication = _mapping(root.get("publication"), "publication")
    if dict(publication) != {
        "no_overwrite": True,
        "atomic": True,
        "terminal_manifest_on_exit": True,
    }:
        raise M1ConfigError("publication flags are not the frozen no-overwrite contract")
    allowed_dirty = _mapping(root.get("allowed_dirty_evidence"), "allowed_dirty_evidence")
    if dict(allowed_dirty) != {
        "agents_guidance": {
            "path": "AGENTS.md",
            "sha256": "956a88bf24253c7120be85ec5b446771e60ff9cac95b7932e7a10d5e3c5fc5d8",
            "reason": "pre-existing user-owned workspace guidance",
        },
        "m0_raw_evidence": {
            "path": "runs/m0_baseline_a",
            "sha256": "91501f4a017a0d5a633f8ccd0ac2878b8d33209d809f0e1cdecf19d439135384",
            "reason": "pre-existing raw M0 retention evidence",
        },
    }:
        raise M1ConfigError("allowed_dirty_evidence is not the exact pre-existing evidence record")

    def reject_pending(value: Any, path: str = "config") -> None:
        if isinstance(value, str) and "pending" in value.lower():
            raise M1ConfigError(f"{path} contains an unresolved pending placeholder")
        if isinstance(value, Mapping):
            for key, child in value.items():
                reject_pending(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                reject_pending(child, f"{path}[{index}]")

    reject_pending(root)


def _raw_mujoco_handle(value: Any, wrapper_attr: str) -> Any:
    """Return the raw MuJoCo model/data behind a robosuite wrapper."""

    raw = getattr(value, wrapper_attr, None)
    return value if raw is None else raw


def _mujoco_integration_spec(mujoco_module: Any) -> Any | None:
    """Resolve ``mjSTATE_INTEGRATION`` for MuJoCo 3.7 enum-style bindings."""

    spec = getattr(mujoco_module, INTEGRATION_SPEC_NAME, None)
    if spec is not None:
        return spec
    state_enum = getattr(mujoco_module, "mjtState", None)
    return getattr(state_enum, INTEGRATION_SPEC_NAME, None)


def _mujoco_state_evidence(mujoco_module: Any, model: Any, data: Any) -> dict[str, Any]:
    """Describe the exact MuJoCo integration mask and component layout.

    The evidence is collected from the raw MuJoCo handles.  Composite enum
    values are intentionally not used as component rows: each base component
    is sized independently so offsets and the total selected state size are
    auditable in the terminal manifest.
    """

    if mujoco_module is None:
        raise M1RuntimeError("MuJoCo module is required for state evidence")
    raw_model = _raw_mujoco_handle(model, "_model")
    raw_data = _raw_mujoco_handle(data, "_data")
    spec = _mujoco_integration_spec(mujoco_module)
    if spec is None:
        raise M1RuntimeError(f"MuJoCo module does not expose {INTEGRATION_SPEC_NAME}")
    try:
        mask = int(spec)
    except (TypeError, ValueError) as exc:
        raise M1RuntimeError("MuJoCo integration mask is not an integer enum") from exc
    size_fn = getattr(mujoco_module, "mj_stateSize", None)
    if not callable(size_fn):
        raise M1RuntimeError("MuJoCo module does not expose mj_stateSize")
    try:
        total_size = int(size_fn(raw_model, spec))
    except Exception as exc:
        raise M1RuntimeError(f"could not determine MuJoCo integration-state size: {exc}") from exc
    state_enum = getattr(mujoco_module, "mjtState", None)
    component_names = (
        "mjSTATE_TIME",
        "mjSTATE_QPOS",
        "mjSTATE_QVEL",
        "mjSTATE_ACT",
        "mjSTATE_HISTORY",
        "mjSTATE_WARMSTART",
        "mjSTATE_CTRL",
        "mjSTATE_QFRC_APPLIED",
        "mjSTATE_XFRC_APPLIED",
        "mjSTATE_EQ_ACTIVE",
        "mjSTATE_MOCAP_POS",
        "mjSTATE_MOCAP_QUAT",
        "mjSTATE_USERDATA",
        "mjSTATE_PLUGIN",
    )
    entries: list[dict[str, Any]] = []
    offset = 0
    for name in component_names:
        component = getattr(mujoco_module, name, None)
        if component is None and state_enum is not None:
            component = getattr(state_enum, name, None)
        if component is None:
            continue
        try:
            component_mask = int(component)
        except (TypeError, ValueError):
            raise M1RuntimeError(f"MuJoCo state component {name} is not an integer enum")
        if not mask & component_mask:
            continue
        try:
            component_size = int(size_fn(raw_model, component))
        except Exception as exc:
            raise M1RuntimeError(f"could not size MuJoCo component {name}: {exc}") from exc
        entries.append(
            {
                "name": name,
                "mask": component_mask,
                "size": component_size,
                "offset": offset,
            }
        )
        offset += component_size
    # A small injected module may expose only the composite enum.  It is
    # still useful for fake-only tests, while a real MuJoCo 3.7 module always
    # contributes the complete base-component table above.
    if not entries:
        entries.append({"name": INTEGRATION_SPEC_NAME, "mask": mask, "size": total_size, "offset": 0})
        offset = total_size
    if offset != total_size:
        raise M1RuntimeError(
            "MuJoCo integration component layout does not sum to the selected state size: "
            f"{offset} != {total_size}"
        )
    return {
        "version": getattr(mujoco_module, "__version__", None) or _package_version("mujoco") or "unknown",
        "mask_name": INTEGRATION_SPEC_NAME,
        "mask": mask,
        "size": total_size,
        "component_layout": {"components": entries, "sum_component_sizes": offset},
        "model_layout": model_layout_signature(raw_model, mujoco_module),
        "raw_handles": {"model": type(raw_model).__name__, "data": type(raw_data).__name__},
    }


def set_integration_state_exact(
    mujoco_module: Any,
    model: Any,
    data: Any,
    payload: np.ndarray,
    *,
    require_readback: bool = True,
) -> np.ndarray:
    """Set and immediately exact-readback integration state, before forward."""

    expected = np.asarray(payload, dtype=np.float64)
    if expected.ndim != 1 or not np.all(np.isfinite(expected)):
        raise M1SchemaError("integration state must be a finite one-dimensional float64 vector")
    raw_model = _raw_mujoco_handle(model, "_model")
    raw_data = _raw_mujoco_handle(data, "_data")
    spec = _mujoco_integration_spec(mujoco_module)
    if spec is None:
        raise M1RuntimeError(f"MuJoCo module does not expose {INTEGRATION_SPEC_NAME}")
    try:
        mujoco_module.mj_setState(raw_model, raw_data, expected, spec)
    except Exception as exc:
        raise M1RuntimeError(f"mj_setState({INTEGRATION_SPEC_NAME}) failed: {exc}") from exc
    if not require_readback:
        return expected.copy()
    readback = np.empty_like(expected)
    try:
        mujoco_module.mj_getState(raw_model, raw_data, readback, spec)
    except Exception as exc:
        raise M1RuntimeError(f"mj_getState({INTEGRATION_SPEC_NAME}) failed: {exc}") from exc
    if readback.dtype != expected.dtype or readback.shape != expected.shape or not np.array_equal(
        readback, expected, equal_nan=False
    ):
        raise M1RuntimeError("integration state exact readback mismatch before mj_forward")
    return readback


def _copy_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _maybe_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value).copy()
    except Exception:
        return None


def _name_list(model: Any, attr: str) -> list[str] | None:
    value = getattr(model, attr, None)
    if value is None:
        return None
    try:
        result: list[str] = []
        for item in value:
            if isinstance(item, bytes):
                result.append(item.decode("utf-8", errors="surrogateescape"))
            else:
                result.append(str(item))
        return result
    except Exception:
        return None


# MuJoCo's ``MjModel`` is a compiled object rather than a dataclass.  The
# public Python binding exposes the model identity as dimensions, name tables,
# options, and a large family of typed arrays.  Keep the classification here,
# next to the existing layout signature, so an array added to a fake/model
# proxy cannot silently become an un-gated part of replay identity.
MODEL_FINGERPRINT_SCHEMA_VERSION = 1
_MODEL_DIMENSION_FIELDS = (
    "nq", "nv", "na", "nu", "nbody", "nbvh", "njnt", "ngeom", "nsite", "ncam",
    "nlight", "nmesh", "nmeshvert", "nmeshtexvert", "nmeshface", "nmeshgraph",
    "nhfield", "nhfielddata", "ntex", "ntexdata", "nmat", "npair", "nexclude",
    "neq", "ntendon", "nwrap", "nsensor", "nnumeric", "nnumericdata", "ntuple",
    "ntupledata", "nkey", "nmocap", "nuser_body", "nuser_jnt", "nuser_geom",
    "nuser_site", "nuser_cam", "nuser_tendon", "nuser_actuator", "nuser_sensor",
    "nuserdata", "nplugin", "npluginattr", "npluginstate", "nhistory", "nM", "nD",
    "nactuator",
)
_MODEL_NAME_KINDS = (
    "body", "joint", "geom", "site", "camera", "cam", "light", "actuator", "sensor",
    "tendon", "wrap", "mesh", "hfield", "texture", "material", "pair", "exclude",
    "equality", "plugin", "numeric", "tuple", "key",
)
_MODEL_PHYSICS_NAME_KINDS = frozenset(
    {"body", "joint", "geom", "actuator", "sensor", "tendon", "wrap", "equality", "plugin", "pair", "exclude", "numeric", "tuple", "key", "hfield"}
)
_MODEL_OBSERVATION_NAME_KINDS = frozenset(
    {"site", "camera", "cam", "light", "mesh", "texture", "material"}
)
_MODEL_PHYSICS_EXACT_FIELDS = frozenset(
    {
        # Execution/initialization arrays and compiled name storage.
        "qpos0", "qpos_spring", "qvel0", "actuator_length0", "actuator_acc0",
        "body_invweight0", "body_invweight1", "dof_M0", "dof_invweight0", "dof_invweight1",
        "names", "name_bodyadr", "name_jntadr", "name_geomadr", "name_siteadr", "name_camadr",
        "name_lightadr", "name_meshadr", "name_hfieldadr", "name_matadr", "name_pairadr",
        "name_excludeadr", "name_eqadr", "name_tendonadr", "name_actuatoradr", "name_sensoradr",
        "name_numericadr", "name_tupleadr", "name_keyadr", "name_pluginadr",
        # Arrays whose values are used by the execution path but do not carry
        # one of the object prefixes below.
        "actuator_ctrlrange", "actuator_forcerange", "actuator_actrange", "actuator_gear",
        "actuator_cranklength", "actuator_acc0", "actuator_length0", "actuator_lengthrange",
        "actuator_dynprm", "actuator_gainprm", "actuator_biasprm", "actuator_user",
        "actuator_group", "actuator_plugin", "actuator_actlimited", "actuator_ctrllimited",
        "actuator_forcelimited", "actuator_trntype", "actuator_dyntype", "actuator_gaintype",
        "actuator_biastype", "actuator_trnid", "actuator_scale",
    }
)
_MODEL_OBSERVATION_EXACT_FIELDS = frozenset(
    {
        "site_rgba", "geom_rgba", "mat_rgba", "mat_texid", "mat_texuniform", "mat_emission",
        "mat_specular", "mat_shininess", "mat_reflectance", "mat_metallic", "mat_roughness",
        "mat_texrepeat", "mat_texcoord", "mat_texrepeat",
    }
)
_MODEL_OBSERVATION_PREFIXES = (
    "cam_", "camera_", "light_", "mat_", "material_", "tex_", "texture_",
    "visual_", "render_", "resource_", "skin_",
)
_MODEL_PHYSICS_PREFIXES = (
    "body_", "geom_", "jnt_", "joint_", "dof_", "actuator_", "eq_", "equality_",
    "tendon_", "wrap_", "plugin_", "sensor_", "pair_", "exclude_", "numeric_", "tuple_",
    "key_",
)
_MODEL_PHYSICS_EXECUTION_FIELDS = frozenset(
    {
        "qpos0", "qpos_spring", "qvel0", "body_invweight0", "body_invweight1", "dof_M0",
        "dof_invweight0", "dof_invweight1", "qpos", "qvel", "act", "ctrl", "qfrc_applied",
        "xfrc_applied", "mocap_pos", "mocap_quat", "userdata", "plugin_state",
    }
)
_MODEL_PHYSICS_DIMENSIONS = frozenset(
    {
        "nq", "nv", "na", "nu", "nbody", "njnt", "ngeom", "nactuator", "ntendon", "nwrap",
        "neq", "npair", "nexclude", "nsensor", "nnumeric", "nnumericdata", "ntuple", "ntupledata",
        "nkey", "nplugin", "npluginattr", "npluginstate", "nuserdata", "nmocap", "nsite",
        "nuser_body", "nuser_jnt",
        "nuser_geom", "nuser_site", "nuser_tendon", "nuser_actuator", "nuser_sensor", "nhistory", "nM", "nD",
    }
)
_MODEL_OBSERVATION_DIMENSIONS = frozenset(
    {"ncam", "nlight", "nmesh", "nmeshvert", "nmeshtexvert", "nmeshface", "nmeshgraph", "nhfield", "nhfielddata", "ntex", "ntexdata", "nmat", "nuser_cam"}
)


def _fingerprint_json_value(value: Any) -> Any:
    """Convert a metadata value without exposing object addresses."""

    if isinstance(value, Enum):
        return _fingerprint_json_value(value.value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return [_fingerprint_json_value(item) for item in value.tolist()]
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _fingerprint_json_value(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_json_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"encoding": "bytes", "data": value.decode("utf-8", errors="surrogateescape")}
    raise M1SchemaError(f"model fingerprint value is not deterministic: {type(value).__name__}")


def _fingerprint_field(value: Any) -> dict[str, Any]:
    """Return stable dtype/shape/SHA-256 evidence for one compiled field."""

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise M1SchemaError(f"could not inspect model fingerprint field: {exc}") from exc
    if array.dtype.kind == "O":
        payload = canonical_json(_fingerprint_json_value(value)).encode("utf-8")
        dtype = "json"
        shape = list(array.shape)
    else:
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.tobytes(order="C")
        dtype = contiguous.dtype.str
        shape = list(contiguous.shape)
    return {"dtype": dtype, "shape": shape, "sha256": sha256_bytes(payload)}


def _model_attr_names(model: Any) -> tuple[str, ...]:
    names: set[str] = set()
    try:
        names.update(str(name) for name in dir(model))
    except Exception:
        pass
    try:
        names.update(str(name) for name in vars(model))
    except (TypeError, AttributeError):
        pass
    return tuple(sorted(names))


def _model_array_category(name: str) -> str | None:
    lower = str(name).lower()
    # Site transforms/attachments and mesh/height-field geometry participate
    # in forward dynamics, contacts, sensors, or predicates.  Their visual
    # appearance/resource arrays remain observation-only.
    if lower.startswith((
        "site_pos", "site_quat", "site_type", "site_bodyid", "site_group", "site_user",
        "site_sameframe", "site_size",
        "hfield_", "mesh_vert", "mesh_face", "mesh_graph", "mesh_scale",
        "mesh_bvh", "mesh_oct", "mesh_poly", "mesh_pos", "mesh_quat",
        "bvh_", "oct_", "flexedge_", "flexvert_", "ten_", "tree_",
        "b_colind", "b_rowadr", "b_rownnz", "d_colind", "d_diag", "d_rowadr",
        "d_rownnz", "m_colind", "m_rowadr", "m_rownnz", "mapd", "mapm",
    )):
        return "physics"
    if lower.startswith("flex_"):
        if lower in {"flex_rgba", "flex_matid", "flex_group", "flex_texcoord", "flex_elemtexcoord"}:
            return "observation"
        return "physics"
    if lower in {"site_rgba", "site_material", "site_matid"} or lower.startswith((
        "geom_rgba", "geom_material", "geom_matid", "mat_", "material_", "tex_", "texture_",
        "visual_", "render_", "resource_", "skin_", "mesh_normal", "mesh_texcoord",
        "mesh_path",
    )):
        return "observation"
    if lower in _MODEL_OBSERVATION_EXACT_FIELDS or any(lower.startswith(prefix) for prefix in _MODEL_OBSERVATION_PREFIXES):
        # ``geom_rgba`` is explicitly visual even though the broader geom
        # family is physics; same for material/texture resource tables.
        return "observation"
    if lower in _MODEL_PHYSICS_EXACT_FIELDS or lower in _MODEL_PHYSICS_EXECUTION_FIELDS:
        return "physics"
    if any(lower.startswith(prefix) for prefix in _MODEL_PHYSICS_PREFIXES):
        if (
            lower.endswith("_rgba")
            or lower.endswith("_material")
            or lower.endswith("_matid")
            or lower.endswith("_group")
        ):
            return "observation"
        return "physics"
    # Known raw name tables are metadata, not an opaque unknown array.  They
    # are included in both?  Include them in physics because changing names
    # changes object lookup and therefore execution identity.
    if lower == "names" or lower.startswith("name_"):
        return "physics"
    if lower in {"names_map", "plugin"} or lower.startswith("text_"):
        return "physics"
    return None


def _collect_model_fingerprint(
    model: Any,
    *,
    kind: str,
    renderer_config: Mapping[str, Any] | None = None,
    mujoco_module: Any | None = None,
) -> dict[str, Any]:
    raw_model = _raw_mujoco_handle(model, "_model")
    dimensions: dict[str, int] = {}
    fields: dict[str, dict[str, Any]] = {}
    for name in _MODEL_DIMENSION_FIELDS:
        if not hasattr(raw_model, name):
            continue
        if kind == "physics" and name not in _MODEL_PHYSICS_DIMENSIONS:
            continue
        if kind == "observation" and name not in _MODEL_OBSERVATION_DIMENSIONS:
            continue
        try:
            value = int(getattr(raw_model, name))
        except (TypeError, ValueError):
            raise M1SchemaError(f"compiled model dimension {name} is not an integer")
        dimensions[name] = value
        fields[f"dimensions.{name}"] = _fingerprint_field(value)

    names: dict[str, list[str]] = {}
    for object_kind in _MODEL_NAME_KINDS:
        count_name = {
            "joint": "njnt", "body": "nbody", "geom": "ngeom", "site": "nsite",
            "camera": "ncam", "cam": "ncam", "light": "nlight", "actuator": "nactuator",
            "sensor": "nsensor", "tendon": "ntendon", "wrap": "nwrap", "mesh": "nmesh",
            "hfield": "nhfield", "texture": "ntex", "material": "nmat", "pair": "npair",
            "exclude": "nexclude", "equality": "neq", "plugin": "nplugin", "numeric": "nnumeric",
            "tuple": "ntuple", "key": "nkey",
        }.get(object_kind)
        candidates = (f"{object_kind}_names", f"{object_kind}_name_list", f"{object_kind}s")
        value: list[str] | None = None
        for attr in candidates:
            value = _name_list(raw_model, attr)
            if value is not None:
                break
        if value is None and count_name is not None and count_name in dimensions:
            id2name = getattr(raw_model, f"{object_kind}_id2name", None)
            if not callable(id2name) and mujoco_module is not None:
                object_enum = getattr(mujoco_module, f"mjOBJ_{object_kind.upper()}", None)
                object_types = getattr(mujoco_module, "mjtObj", None)
                if object_enum is None and object_types is not None:
                    object_enum = getattr(object_types, f"mjOBJ_{object_kind.upper()}", None)
                id2name_fn = getattr(mujoco_module, "mj_id2name", None)
                if object_enum is not None and callable(id2name_fn):
                    id2name = lambda index, _obj=object_enum, _fn=id2name_fn: _fn(raw_model, _obj, index)
            if callable(id2name):
                value = []
                for index in range(dimensions[count_name]):
                    named = id2name(index)
                    value.append("" if named is None else str(named))
        name_kind_allowed = (
            object_kind in _MODEL_PHYSICS_NAME_KINDS
            if kind == "physics"
            else object_kind in _MODEL_OBSERVATION_NAME_KINDS
        )
        if value is not None and name_kind_allowed:
            names[object_kind] = value
            fields[f"names.{object_kind}"] = _fingerprint_field(value)

    options: dict[str, Any] = {}
    option_owner = getattr(raw_model, "opt", None)
    if kind == "physics" and option_owner is not None:
        for attr in sorted(str(name) for name in dir(option_owner) if not str(name).startswith("_")):
            try:
                value = getattr(option_owner, attr)
            except Exception:
                continue
            if callable(value):
                continue
            try:
                encoded = _fingerprint_json_value(value)
                _fingerprint_field(value)
            except M1SchemaError:
                continue
            options[attr] = encoded
            fields[f"options.{attr}"] = _fingerprint_field(value)

    unknown_arrays: list[str] = []
    array_values: dict[str, Any] = {}
    for name in _model_attr_names(raw_model):
        if name.startswith("_") or name in {"opt"} or name.endswith("_names") or name.endswith("_name_list"):
            continue
        try:
            value = getattr(raw_model, name)
        except Exception:
            continue
        # Only ndarray-like fields are considered compiled arrays.  Lists
        # under a known prefix are included too, which keeps fake compiled
        # models useful without making arbitrary Python bookkeeping fields
        # part of the identity.
        category = _model_array_category(name)
        is_array = isinstance(value, np.ndarray)
        if not is_array and category is not None and isinstance(value, (list, tuple)):
            try:
                is_array = np.asarray(value).dtype.kind != "O"
            except Exception:
                is_array = False
        if not is_array:
            continue
        if category is None:
            # Fail closed for both Python proxies and MuJoCo's C-extension
            # properties.  A newly exposed compiled array must be assigned to
            # physics or observation identity before an authoritative run.
            unknown_arrays.append(name)
            continue
        array_values[name] = value

    if unknown_arrays:
        raise M1SchemaError(
            "unclassified compiled model array fields: " + ", ".join(sorted(unknown_arrays))
        )
    for name, value in sorted(array_values.items()):
        category = _model_array_category(name)
        if (kind == "physics" and category != "physics") or (kind == "observation" and category != "observation"):
            continue
        fields[name] = _fingerprint_field(value)

    if kind == "observation" and renderer_config is not None:
        def add_renderer(path: str, value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                    add_renderer(f"{path}.{key}" if path else str(key), child)
                return
            fields[f"renderer.{path}"] = _fingerprint_field(value)
        add_renderer("", renderer_config)

    # The aggregate hash is over the field evidence itself.  This makes a
    # digest collision or a future metadata omission visible to the exact
    # equality gate, rather than relying on one opaque concatenated blob.
    ordered_fields = {key: fields[key] for key in sorted(fields)}
    aggregate = sha256_bytes(canonical_json(ordered_fields).encode("utf-8"))
    return {
        "schema_version": MODEL_FINGERPRINT_SCHEMA_VERSION,
        "kind": kind,
        "hash": aggregate,
        "sha256": aggregate,
        "model_hash": aggregate,
        "fields": ordered_fields,
        "field_metadata": ordered_fields,
        "field_hashes": {key: value["sha256"] for key, value in ordered_fields.items()},
        "dimensions": dimensions,
        "names": names,
        "options": options,
        "unknown_fields": [],
    }


def physics_model_fingerprint(model: Any, mujoco_module: Any | None = None) -> dict[str, Any]:
    """Hash dynamics-critical compiled ``mjModel`` identity exactly."""

    return _collect_model_fingerprint(model, kind="physics", mujoco_module=mujoco_module)


def observation_model_fingerprint(
    model: Any,
    renderer_config: Mapping[str, Any] | None = None,
    mujoco_module: Any | None = None,
) -> dict[str, Any]:
    """Hash camera/rendering/material/resource identity independently."""

    return _collect_model_fingerprint(
        model,
        kind="observation",
        renderer_config=renderer_config,
        mujoco_module=mujoco_module,
    )


def exact_model_fingerprint_equal(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Return true only when kind, field evidence, and aggregate hash match."""

    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return False
    return (
        expected.get("schema_version") == actual.get("schema_version")
        and expected.get("kind") == actual.get("kind")
        and expected.get("hash") == actual.get("hash")
        and _exact_equal(expected.get("fields"), actual.get("fields"))
    )


# Short alias for callers that use the gate as a predicate.
model_fingerprint_equal = exact_model_fingerprint_equal


def model_layout_signature(model: Any, mujoco_module: Any | None = None) -> dict[str, Any]:
    """Collect dimensions and ordered names used as exact compatibility gates."""

    model = _raw_mujoco_handle(model, "_model")
    dimensions: dict[str, int] = {}
    for name in (
        "nq",
        "nv",
        "na",
        "nu",
        "nbody",
        "ngeom",
        "nsite",
        "njnt",
        "ncam",
        "nactuator",
        "neq",
        "nmocap",
        "nuserdata",
        "npluginstate",
        "nplugin",
        "nhistory",
    ):
        if hasattr(model, name):
            try:
                dimensions[name] = int(getattr(model, name))
            except (TypeError, ValueError):
                pass
    names: dict[str, list[str]] = {}
    for kind in ("joint", "body", "geom", "site", "camera", "actuator", "plugin"):
        for attr in (f"{kind}_names", f"{kind}_name_list", f"{kind}s"):
            values = _name_list(model, attr)
            if values is not None:
                names[kind] = values
                break
        if kind in names:
            continue
        count = dimensions.get({
            "joint": "njnt",
            "body": "nbody",
            "geom": "ngeom",
            "site": "nsite",
            "camera": "ncam",
            "actuator": "nactuator",
            "plugin": "nplugin",
        }[kind])
        if count is None:
            continue
        id2name = getattr(model, f"{kind}_id2name", None)
        if not callable(id2name) and mujoco_module is not None:
            obj_name = getattr(mujoco_module, f"mjOBJ_{kind.upper()}", None)
            if obj_name is None:
                object_enum = getattr(mujoco_module, "mjtObj", None)
                obj_name = getattr(object_enum, f"mjOBJ_{kind.upper()}", None) if object_enum else None
            mj_id2name = getattr(mujoco_module, "mj_id2name", None)
            if obj_name is not None and callable(mj_id2name):
                id2name = lambda index, _obj=obj_name, _fn=mj_id2name: _fn(model, _obj, index)
        if callable(id2name):
            values: list[str] = []
            for index in range(count):
                name = id2name(index)
                values.append("" if name is None else str(name))
            names[kind] = values
    options: dict[str, Any] = {}
    model_options = getattr(model, "opt", None)
    if model_options is not None:
        for name in (
            "timestep",
            "integrator",
            "cone",
            "jacobian",
            "solver",
            "iterations",
            "ls_iterations",
            "noslip_iterations",
            "sdf_iterations",
            "sdf_initpoints",
            "tolerance",
            "ls_tolerance",
            "noslip_tolerance",
            "gravity",
            "wind",
            "magnetic",
            "density",
            "viscosity",
            "impratio",
            "enableflags",
            "disableflags",
        ):
            if hasattr(model_options, name):
                value = getattr(model_options, name)
                options[name] = np.asarray(value).tolist() if isinstance(value, (np.ndarray, list, tuple)) else value
    return {"dimensions": dimensions, "ordered_names": names, "options": options}


def _get_controller(env: Any) -> Any | None:
    robots = getattr(env, "robots", None)
    if robots is None:
        return None
    try:
        robot = list(robots)[0]
    except (TypeError, IndexError):
        return None
    return getattr(robot, "controller", None)


def _get_gripper(env: Any) -> Any | None:
    robots = getattr(env, "robots", None)
    if robots is None:
        return None
    try:
        robot = list(robots)[0]
    except (TypeError, IndexError):
        return None
    for attr in ("gripper", "gripper_model"):
        value = getattr(robot, attr, None)
        if value is not None:
            return value
    return None


def _gripper_kinematics(env: Any) -> dict[str, Any]:
    """Read only physical gripper views used by the invariant collector."""

    robots = getattr(env, "robots", None)
    try:
        robot = list(robots)[0] if robots is not None else None
    except (TypeError, IndexError):
        robot = None
    result: dict[str, Any] = {}
    for name in ("gripper_qpos", "gripper_qvel"):
        if robot is not None and hasattr(robot, name):
            result[name] = _maybe_array(getattr(robot, name))
    gripper = _get_gripper(env)
    if gripper is not None:
        for name in ("qpos", "qvel"):
            if name not in result and hasattr(gripper, name):
                result[f"gripper_{name}"] = _maybe_array(getattr(gripper, name))
    if robot is not None:
        sim = getattr(robot, "sim", None)
        data = getattr(sim, "data", None)
        if data is not None:
            for result_name, index_attr, data_attr in (
                ("gripper_qpos", "_ref_gripper_joint_pos_indexes", "qpos"),
                ("gripper_qvel", "_ref_gripper_joint_vel_indexes", "qvel"),
            ):
                if result_name in result:
                    continue
                indexes = getattr(robot, index_attr, None)
                values = getattr(data, data_attr, None)
                if indexes is not None and values is not None:
                    result[result_name] = np.asarray(values[list(indexes)]).copy()
    return result


def _visual_state_snapshot(inner: Any) -> dict[str, Any]:
    """Snapshot exactly the task-0 visual fields allowed by post-processing."""

    nested: Any = None
    object_properties = getattr(inner, "object_properties", None)
    if isinstance(object_properties, Mapping):
        nested = object_properties.get("vis_site_names")
        if nested is None:
            # Some task wrappers keep the table/fixture entry one level down.
            for value in object_properties.values():
                if isinstance(value, Mapping) and "vis_site_names" in value:
                    nested = value["vis_site_names"]
                    break
    top_level = getattr(inner, "vis_site_names", None)

    def without_vis_site_names(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: without_vis_site_names(child)
                for key, child in value.items()
                if str(key) != "vis_site_names"
            }
        if isinstance(value, list):
            return [without_vis_site_names(child) for child in value]
        if isinstance(value, tuple):
            return tuple(without_vis_site_names(child) for child in value)
        return copy.deepcopy(value)

    fixture_visual: dict[str, Any] = {}
    fixture_other: dict[str, Any] = {}
    fixture_dict = getattr(inner, "fixtures_dict", None)
    if fixture_dict is None:
        fixture_dict = getattr(inner, "_fixtures_dict", None)
    if isinstance(fixture_dict, Mapping):
        for fixture_name, fixture in fixture_dict.items():
            properties = getattr(fixture, "object_properties", None)
            if isinstance(properties, Mapping):
                if "vis_site_names" in properties:
                    fixture_visual[str(fixture_name)] = copy.deepcopy(properties["vis_site_names"])
                fixture_other[str(fixture_name)] = without_vis_site_names(properties)
            elif properties is not None and hasattr(properties, "__dict__"):
                properties_dict = vars(properties)
                if "vis_site_names" in properties_dict:
                    fixture_visual[str(fixture_name)] = copy.deepcopy(properties_dict["vis_site_names"])
                fixture_other[str(fixture_name)] = without_vis_site_names(properties_dict)

    return {
        "vis_site_names": {
            "top_level": copy.deepcopy(top_level),
            "object_properties": copy.deepcopy(nested),
            "fixtures": fixture_visual,
        },
        # Every other object_properties field is physical/semantic state for
        # this guard, even when it is nested below a task-specific object.
        "object_properties_other": without_vis_site_names(object_properties),
        "fixture_object_properties_other": fixture_other,
    }


def _unwrap_official_env(value: Any) -> Any:
    """Select exactly one synchronous child without touching lazy ``sim``.

    LeRobot's vector wrapper may defer simulator construction until the
    official child has been reset.  In particular, probing ``hasattr(child,
    "sim")`` here can execute a property and bind the simulator too early.
    Only the explicit one-child vector boundary is traversed; the selected
    child remains the official ``LiberoEnv`` whose ``step`` performs the
    normal success/observation path.
    """

    current = value
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        envs = getattr(current, "envs", None)
        if envs is None:
            break
        try:
            if len(envs) != 1:
                raise M1RuntimeError("M1 requires exactly one synchronous environment child")
            current = envs[0]
        except TypeError as exc:
            raise M1RuntimeError("M1 synchronous environment children are not inspectable") from exc
    return current


def _find_inner_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    first_sim_env: Any | None = None
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "robots") and hasattr(current, "sim"):
            if first_sim_env is None:
                first_sim_env = current
            # The underlying robosuite task is the owner of fixture model
            # mutations and timestep/cur_time/done.  LiberoEnv also exposes
            # robots/sim as delegating properties, so prefer the deepest env
            # that owns those mutable fields when traversing its _env chain.
            if hasattr(current, "timestep") or hasattr(current, "fixtures_dict"):
                return current
        next_value = getattr(current, "_env", None)
        if next_value is None:
            next_value = getattr(current, "env", None)
        if next_value is None:
            break
        current = next_value
    return first_sim_env if first_sim_env is not None else env


def _integration_state(mujoco_module: Any, model: Any, data: Any) -> np.ndarray:
    raw_model = _raw_mujoco_handle(model, "_model")
    raw_data = _raw_mujoco_handle(data, "_data")
    spec = _mujoco_integration_spec(mujoco_module)
    if spec is None:
        value = getattr(raw_data, "integration_state", getattr(raw_data, "state", None))
        if value is None:
            raise M1RuntimeError("runtime cannot expose integration state")
        return np.asarray(value, dtype=np.float64).copy()
    size_fn = getattr(mujoco_module, "mj_stateSize", None)
    if callable(size_fn):
        try:
            size = int(size_fn(raw_model, spec))
        except Exception as exc:
            raise M1RuntimeError(f"could not determine integration-state size: {exc}") from exc
    else:
        value = getattr(raw_data, "integration_state", getattr(raw_data, "state", None))
        if value is None:
            raise M1RuntimeError("runtime cannot determine integration-state size")
        size = int(np.asarray(value).size)
    out = np.empty(size, dtype=np.float64)
    try:
        mujoco_module.mj_getState(raw_model, raw_data, out, spec)
    except Exception as exc:
        raise M1RuntimeError(f"could not capture integration state: {exc}") from exc
    if not np.all(np.isfinite(out)):
        raise M1RuntimeError("captured integration state is non-finite")
    return out


def _fixture_snapshot(inner: Any) -> dict[str, Any]:
    sim = getattr(inner, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        return {"body_pos": {}, "body_quat": {}}
    fixture_dict = getattr(inner, "fixtures_dict", None)
    if fixture_dict is None:
        fixture_dict = getattr(inner, "_fixtures_dict", None)
    names: list[str] = []
    if isinstance(fixture_dict, Mapping):
        for value in fixture_dict.values():
            root_body = getattr(value, "root_body", value if isinstance(value, str) else None)
            if root_body is not None:
                names.append(str(root_body))
    explicit = getattr(inner, "fixture_body_names", None)
    if explicit is not None:
        names.extend(str(name) for name in explicit)
    names = sorted(set(names))
    body_pos = getattr(model, "body_pos", None)
    body_quat = getattr(model, "body_quat", None)
    result_pos: dict[str, list[float]] = {}
    result_quat: dict[str, list[float]] = {}
    for name in names:
        body_id_fn = getattr(model, "body_name2id", None)
        if not callable(body_id_fn) or body_pos is None or body_quat is None:
            continue
        try:
            body_id = int(body_id_fn(name))
            result_pos[name] = np.asarray(body_pos[body_id], dtype=np.float64).tolist()
            result_quat[name] = np.asarray(body_quat[body_id], dtype=np.float64).tolist()
        except Exception as exc:
            raise M1RuntimeError(f"could not capture fixture body {name}: {exc}") from exc
    return {"body_pos": result_pos, "body_quat": result_quat}


def _fixture_root_names(inner: Any) -> tuple[str, ...]:
    """Return the non-empty set of fixture roots exposed by the task env."""

    fixture_dict = getattr(inner, "fixtures_dict", None)
    if fixture_dict is None:
        fixture_dict = getattr(inner, "_fixtures_dict", None)
    names: list[str] = []
    if isinstance(fixture_dict, Mapping):
        for value in fixture_dict.values():
            root_body = getattr(value, "root_body", value if isinstance(value, str) else None)
            if root_body is not None and str(root_body).strip():
                names.append(str(root_body))
    explicit = getattr(inner, "fixture_body_names", None)
    if explicit is not None:
        names.extend(str(name) for name in explicit if str(name).strip())
    return tuple(sorted(set(names)))


def fixture_physics_fingerprint(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Return an exact digest for the fixture pose subset of compiled state."""

    if not isinstance(fixture, Mapping):
        raise M1SchemaError("fixture state must be a mapping")
    fields: dict[str, dict[str, Any]] = {}
    for name in ("body_pos", "body_quat"):
        value = fixture.get(name, {})
        fields[name] = _fingerprint_field(value)
    digest = sha256_bytes(canonical_json(fields).encode("utf-8"))
    return {
        "schema_version": 1,
        "kind": "fixture_physics",
        "hash": digest,
        "sha256": digest,
        "fields": fields,
    }


def _restore_fixture(inner: Any, fixture: Mapping[str, Any]) -> None:
    sim = getattr(inner, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        if fixture.get("body_pos") or fixture.get("body_quat"):
            raise M1RuntimeError("fixture state exists but runtime has no model")
        return
    body_pos = getattr(model, "body_pos", None)
    body_quat = getattr(model, "body_quat", None)
    body_id_fn = getattr(model, "body_name2id", None)
    for name, value in dict(fixture.get("body_pos", {})).items():
        if not callable(body_id_fn) or body_pos is None:
            raise M1RuntimeError("runtime cannot restore fixture body positions")
        body_pos[int(body_id_fn(name))] = np.asarray(value, dtype=np.float64)
    for name, value in dict(fixture.get("body_quat", {})).items():
        if not callable(body_id_fn) or body_quat is None:
            raise M1RuntimeError("runtime cannot restore fixture body orientations")
        body_quat[int(body_id_fn(name))] = np.asarray(value, dtype=np.float64)


_MUJOCO_JOINT_WIDTHS: dict[int, tuple[int, int]] = {
    # mjJNT_FREE, mjJNT_BALL, mjJNT_SLIDE, mjJNT_HINGE
    0: (7, 6),
    1: (4, 3),
    2: (1, 1),
    3: (1, 1),
}


def _joint_state_slices(
    model: Any,
    joint_id: int,
    qpos: Any,
    qvel: Any,
) -> tuple[slice, slice, tuple[int, int]]:
    """Return type-width-checked qpos/qvel slices for one MuJoCo joint."""

    joint_types = getattr(model, "jnt_type", None)
    qpos_addresses = getattr(model, "jnt_qposadr", None)
    qvel_addresses = getattr(model, "jnt_dofadr", None)
    if joint_types is None or qpos_addresses is None or qvel_addresses is None:
        raise M1RuntimeError(
            "MuJoCo named-joint state requires jnt_type, jnt_qposadr, and jnt_dofadr"
        )
    try:
        type_code = int(np.asarray(joint_types).reshape(-1)[joint_id])
        qpos_start = int(np.asarray(qpos_addresses).reshape(-1)[joint_id])
        qvel_start = int(np.asarray(qvel_addresses).reshape(-1)[joint_id])
    except (IndexError, TypeError, ValueError) as exc:
        raise M1RuntimeError(f"invalid MuJoCo joint address/type for joint {joint_id}") from exc
    widths = _MUJOCO_JOINT_WIDTHS.get(type_code)
    if widths is None:
        raise M1RuntimeError(f"unsupported MuJoCo joint type {type_code} for joint {joint_id}")
    qpos_width, qvel_width = widths

    def checked_slice(
        start: int,
        width: int,
        addresses: Any,
        vector: Any,
        label: str,
    ) -> slice:
        values = np.asarray(vector).reshape(-1)
        if start < 0 or start + width > values.size:
            raise M1RuntimeError(
                f"MuJoCo {label} address slice is outside runtime vector: "
                f"start={start}, width={width}, size={values.size}"
            )
        # Address arrays are authoritative for the next joint boundary.  A
        # malformed/non-monotonic address table cannot be allowed to make a
        # free/ball slice bleed into the next joint.
        try:
            all_addresses = [int(item) for item in np.asarray(addresses).reshape(-1)]
        except (TypeError, ValueError) as exc:
            raise M1RuntimeError(f"MuJoCo {label} address table is not numeric") from exc
        later = [item for item in all_addresses if item > start]
        if later and start + width > min(later):
            raise M1RuntimeError(
                f"MuJoCo {label} type-width slice crosses next joint address: "
                f"start={start}, width={width}, next={min(later)}"
            )
        return slice(start, start + width)

    return (
        checked_slice(qpos_start, qpos_width, qpos_addresses, qpos, "qpos"),
        checked_slice(qvel_start, qvel_width, qvel_addresses, qvel, "qvel"),
        widths,
    )


def _object_snapshot(inner: Any) -> dict[str, Any]:
    """Collect named movable-object body poses and joint qpos/qvel views."""

    objects = getattr(inner, "objects_dict", None)
    if not isinstance(objects, Mapping):
        return {}
    sim = getattr(inner, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return {}
    body_xpos = getattr(data, "body_xpos", getattr(data, "xpos", None))
    body_xquat = getattr(data, "body_xquat", getattr(data, "xquat", None))
    body_name2id = getattr(model, "body_name2id", None)
    joint_name2id = getattr(model, "joint_name2id", None)
    qpos = getattr(data, "qpos", None)
    qvel = getattr(data, "qvel", None)
    qpos_addr = getattr(model, "jnt_qposadr", None)
    qvel_addr = getattr(model, "jnt_dofadr", None)
    result: dict[str, Any] = {}
    for object_name, obj in objects.items():
        name = str(object_name)
        entry: dict[str, Any] = {}
        root_body = getattr(obj, "root_body", None)
        if callable(body_name2id) and root_body is not None and body_xpos is not None and body_xquat is not None:
            try:
                body_id = int(body_name2id(root_body))
                entry["body_pos"] = np.asarray(body_xpos[body_id], dtype=np.float64).tolist()
                entry["body_quat"] = np.asarray(body_xquat[body_id], dtype=np.float64).tolist()
            except Exception as exc:
                raise M1RuntimeError(f"could not capture object body {name}: {exc}") from exc
        joints = getattr(obj, "joints", None)
        if callable(joint_name2id) and joints and qpos is not None and qvel is not None:
            joint_values: dict[str, Any] = {}
            for joint_name in joints:
                try:
                    joint_id = int(joint_name2id(joint_name))
                    qpos_slice, qvel_slice, (qpos_width, qvel_width) = _joint_state_slices(
                        model, joint_id, qpos, qvel
                    )
                    joint_values[str(joint_name)] = {
                        "qpos": np.asarray(qpos).reshape(-1)[qpos_slice].astype(np.float64).tolist(),
                        "qvel": np.asarray(qvel).reshape(-1)[qvel_slice].astype(np.float64).tolist(),
                        "qpos_width": qpos_width,
                        "qvel_width": qvel_width,
                    }
                except Exception as exc:
                    raise M1RuntimeError(f"could not capture object joint {joint_name}: {exc}") from exc
            if joint_values:
                entry["joints"] = joint_values
        if entry:
            result[name] = entry
    return result


def _controller_state(controller: Any) -> dict[str, Any]:
    if controller is None:
        return {}
    result: dict[str, Any] = {}
    for field_name in (
        "initial_joint",
        "goal_pos",
        "goal_ori",
        "new_update",
        "relative_ori",
        "ori_ref",
    ):
        if not hasattr(controller, field_name):
            continue
        value = getattr(controller, field_name)
        if field_name == "new_update":
            result[field_name] = bool(value)
        elif value is None:
            result[field_name] = None
        else:
            result[field_name] = np.asarray(value).tolist()
    return result


def _restore_controller(controller: Any, state: Mapping[str, Any]) -> None:
    _validate_controller_state_payload(state)
    if controller is None:
        if state:
            raise M1RuntimeError("controller state exists but fresh runtime has no controller")
        return
    for field_name in (
        "initial_joint",
        "goal_pos",
        "goal_ori",
        "relative_ori",
        "ori_ref",
    ):
        if field_name in state:
            value = state[field_name]
            setattr(
                controller,
                field_name,
                None if value is None else np.asarray(value, dtype=np.float64).copy(),
            )
    if "new_update" in state:
        setattr(controller, "new_update", bool(state["new_update"]))


def _gripper_state(gripper: Any) -> dict[str, Any]:
    if gripper is None or not hasattr(gripper, "current_action"):
        return {}
    return {"current_action": np.asarray(getattr(gripper, "current_action")).tolist()}


def _restore_gripper(gripper: Any, state: Mapping[str, Any]) -> None:
    if not state:
        return
    if gripper is None or not hasattr(gripper, "current_action"):
        raise M1RuntimeError("gripper state exists but fresh runtime has no current_action")
    gripper.current_action = np.asarray(state["current_action"], dtype=np.float64).copy()


def _controller_configuration(controller: Any) -> dict[str, Any]:
    """Capture immutable OSC configuration/gains for fresh-runtime gating."""

    if controller is None:
        return {}
    result: dict[str, Any] = {}
    for field_name in (
        "impedance_mode",
        "use_delta",
        "use_ori",
        "eef_name",
        "joint_dim",
        "control_freq",
        "kp",
        "kd",
        "input_min",
        "input_max",
        "output_min",
        "output_max",
    ):
        if hasattr(controller, field_name):
            value = getattr(controller, field_name)
            result[field_name] = np.asarray(value).tolist() if isinstance(value, (np.ndarray, list, tuple)) else value
    return result


def _counter_state(inner: Any, outer: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("timestep", "cur_time", "done", "terminated", "truncated"):
        for owner in (inner, outer):
            if hasattr(owner, name):
                value = getattr(owner, name)
                result[name] = bool(value) if isinstance(value, (bool, np.bool_)) else value
                break
    return result


def _restore_counters(inner: Any, outer: Any, state: Mapping[str, Any]) -> None:
    for name, value in state.items():
        owner = inner if hasattr(inner, name) else outer if hasattr(outer, name) else None
        if owner is not None:
            setattr(owner, name, value)


def _renderer_config_for_runtime(env: Any, inner: Any, config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Collect renderer identity without importing a renderer or policy."""

    if isinstance(config, Mapping):
        runtime = config.get("runtime")
        if isinstance(runtime, Mapping) and isinstance(runtime.get("renderer"), Mapping):
            return dict(runtime["renderer"])
    for owner in (env, inner):
        value = getattr(owner, "renderer_config", None)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _task_state_snapshot(outer: Any, inner: Any, source_step: int) -> dict[str, Any]:
    """Capture task/discrete state as an explicit v2 replay seam."""

    result: dict[str, Any] = {"source_step": int(source_step)}
    for name in ("task_suite_name", "suite", "task_id", "task", "task_description", "instruction", "init_state_id"):
        for owner in (outer, inner):
            if hasattr(owner, name):
                result[name] = copy.deepcopy(getattr(owner, name))
                break
    task = getattr(inner, "task", None)
    if isinstance(task, Mapping):
        result["task_mapping"] = copy.deepcopy(dict(task))
    for name in ("object_states_dict", "tracking_object_states_change"):
        for owner in (inner, outer):
            if hasattr(owner, name):
                result[name] = _snapshot_serialized_value(getattr(owner, name))
                break
    return result


def _identity_state(
    outer: Any,
    inner: Any,
    tape_hash: str | None,
    source_step: int,
    *,
    mujoco_module: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = getattr(outer, "task", getattr(inner, "task", None))
    instruction = getattr(outer, "task_description", getattr(inner, "task_description", None))
    init_state = getattr(outer, "init_state_id", getattr(inner, "init_state_id", None))
    sim = getattr(inner, "sim", None)
    model = getattr(sim, "model", None)
    configured_task = config.get("task", {}) if isinstance(config, Mapping) else {}
    configured_hashes = config.get("hashes", {}) if isinstance(config, Mapping) else {}
    bddl_hash = configured_hashes.get("bddl", {}) if isinstance(configured_hashes, Mapping) else {}
    init_hash = configured_hashes.get("init_state", {}) if isinstance(configured_hashes, Mapping) else {}
    physics_fingerprint = physics_model_fingerprint(model, mujoco_module) if model is not None else {}
    observation_fingerprint = (
        observation_model_fingerprint(
            model,
            _renderer_config_for_runtime(outer, inner, config),
            mujoco_module,
        )
        if model is not None
        else {}
    )
    fixture_fingerprint = fixture_physics_fingerprint(_fixture_snapshot(inner)) if model is not None else {}
    return {
        "suite": getattr(outer, "task_suite_name", getattr(outer, "suite", "libero_spatial")),
        "task": task,
        "instruction": instruction,
        "init_state_id": int(init_state) if init_state is not None else None,
        "source_step": int(source_step),
        "action_tape_sha256": tape_hash,
        "config_identity": {
            "suite": configured_task.get("suite") if isinstance(configured_task, Mapping) else None,
            "task_id": configured_task.get("task_id") if isinstance(configured_task, Mapping) else None,
            "task_name": configured_task.get("name") if isinstance(configured_task, Mapping) else None,
            "instruction": configured_task.get("instruction") if isinstance(configured_task, Mapping) else None,
            "init_state_id": configured_task.get("init_state_id") if isinstance(configured_task, Mapping) else None,
            "seed": configured_task.get("seed") if isinstance(configured_task, Mapping) else None,
            "trajectory_id": config.get("trajectory_id") if isinstance(config, Mapping) else None,
            "bddl_sha256": bddl_hash.get("sha256") if isinstance(bddl_hash, Mapping) else bddl_hash,
            "init_state_sha256": init_hash.get("sha256") if isinstance(init_hash, Mapping) else init_hash,
        },
        "model_layout": model_layout_signature(model, mujoco_module) if model is not None else {},
        "physics_model_fingerprint": physics_fingerprint,
        "observation_model_fingerprint": observation_fingerprint,
        "fixture_physics_fingerprint": fixture_fingerprint,
        "controller_configuration": _controller_configuration(_get_controller(inner)),
    }


@dataclass(frozen=True)
class ReplayState:
    integration_state: np.ndarray
    fixture: Mapping[str, Any]
    counters: Mapping[str, Any]
    controller: Mapping[str, Any]
    gripper: Mapping[str, Any]
    identity: Mapping[str, Any]
    reward: Any = None
    success: Any = None
    # Concrete SERIALIZED Python runtime values (currently Observable
    # timers/delays/sampled/current/cache fields).  Keeping this separate from
    # derived observable evidence makes the capture/persist/restore/readback
    # set equality auditable.
    serialized_runtime: Mapping[str, Any] = field(default_factory=dict)
    # ReplayState v2 makes the state boundary explicit.  The aggregate
    # ``serialized_runtime`` field remains for v1 compatibility while these
    # fields expose the Python/RNG, model, observation, task, and tape seams.
    python_state: Mapping[str, Any] = field(default_factory=dict)
    rng: Mapping[str, Any] = field(default_factory=dict)
    physics_model_fingerprint: Mapping[str, Any] = field(default_factory=dict)
    observation_model_fingerprint: Mapping[str, Any] = field(default_factory=dict)
    fixture_physics_fingerprint: Mapping[str, Any] = field(default_factory=dict)
    raw_observation: Any = None
    task_state: Mapping[str, Any] = field(default_factory=dict)
    tape_provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = REPLAY_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or int(self.schema_version) not in (1, REPLAY_STATE_SCHEMA_VERSION):
            raise M1SchemaError(
                f"replay state schema_version must be 1 or {REPLAY_STATE_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        value = np.asarray(self.integration_state, dtype=np.float64)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise M1SchemaError("replay integration state must be finite and one-dimensional")
        value = np.ascontiguousarray(value)
        value.setflags(write=False)
        object.__setattr__(self, "integration_state", value)
        for name in (
            "fixture", "counters", "controller", "gripper", "identity", "serialized_runtime",
            "python_state", "rng", "physics_model_fingerprint", "observation_model_fingerprint",
            "fixture_physics_fingerprint",
            "task_state", "tape_provenance",
        ):
            mapping = dict(getattr(self, name))
            if name == "controller":
                _validate_controller_state_payload(mapping)
            if name in {"serialized_runtime", "python_state", "rng", "task_state", "tape_provenance"}:
                # Normalize direct ReplayState construction as well as live
                # captures, so tuple/set/ndarray values are persistable even
                # when a caller did not go through ``capture`` first.
                mapping = {
                    str(path): _snapshot_serialized_value(item)
                    for path, item in mapping.items()
                }
            object.__setattr__(self, name, copy.deepcopy(mapping))
        if self.raw_observation is None:
            object.__setattr__(self, "raw_observation", None)
        else:
            object.__setattr__(
                self,
                "raw_observation",
                # Keep the in-memory ReplayState semantically raw.  The
                # lossless tagged encoding is applied at the record boundary,
                # so callers can inspect ndarray dtype/shape directly.
                copy.deepcopy(self.raw_observation),
            )
        object.__setattr__(self, "reward", copy.deepcopy(self.reward))
        object.__setattr__(self, "success", copy.deepcopy(self.success))

    @property
    def python_persistent_state(self) -> Mapping[str, Any]:
        return self.python_state

    @property
    def rng_state(self) -> Mapping[str, Any]:
        return self.rng

    @property
    def physics_fingerprint(self) -> Mapping[str, Any]:
        return self.physics_model_fingerprint

    @property
    def observation_fingerprint(self) -> Mapping[str, Any]:
        return self.observation_model_fingerprint

    @property
    def task_discrete_state(self) -> Mapping[str, Any]:
        return self.task_state

    @property
    def state_schema_version(self) -> int:
        return self.schema_version


_OBSERVABLE_FIELD_CLASSES: dict[str, str] = {
    "name": IMMUTABLE,
    "sensor": IMMUTABLE,
    "corrupter": IMMUTABLE,
    "filter": IMMUTABLE,
    "delayer": IMMUTABLE,
    "sampling_timestep": IMMUTABLE,
    "enabled": IMMUTABLE,
    "active": IMMUTABLE,
    "is_number": IMMUTABLE,
    "data_shape": IMMUTABLE,
    "time_since_last_sample": SERIALIZED,
    "timer": SERIALIZED,
    "current_delay": SERIALIZED,
    "delay": SERIALIZED,
    "sampled": SERIALIZED,
    # The values are diagnostic after refresh, but the Observable's own
    # mutable cursor/cache is part of the state used by the next update.  It
    # is therefore captured and restored exactly.  Treating these as merely
    # derived would let a timer/cache difference hide behind a forced refresh.
    "current_observed_value": SERIALIZED,
    "cache": SERIALIZED,
    "obs_cache": SERIALIZED,
}


# The selected step path is intentionally small and explicit.  These names
# are the registry of instance fields that may occur on the corresponding
# owner roots.  Source inspection is evidence attached to each registered
# field; it never expands the registry.  Any other instance field is an
# unknown mutable-state failure, including a harmless-looking scalar or a
# source-defined mutable container.
_INNER_AUDITED_FIELDS = frozenset(
    {
        # simulator / ownership handles
        "sim", "model", "robots", "_robots", "fixtures_dict", "_fixtures_dict",
        "objects_dict", "_objects_dict", "object_sites_dict", "_object_sites_dict",
        "object_states_dict", "_object_states_dict", "objects", "fixtures",
        "placement_initializer", "conditional_placement_initializer",
        "conditional_placement_on_objects_initializer", "object_property_initializers",
        "tracking_object_states_change", "obj_body_id", "observables", "_observables",
        "_obs_cache", "obs_cache", "object_properties", "task", "task_description",
        "task_suite_name", "suite", "task_id", "init_states", "_init_states",
        "_task_bddl_file", "bddl_file_name", "parsed_problem", "obj_of_interest",
        "_arena_type", "_arena_xml", "_arena_properties", "custom_asset_dir",
        # Explicit base-environment/task fields from the pinned robosuite and
        # LIBERO source.  These are audited even when a particular task does
        # not use them; no source-name heuristic adds fields at runtime.
        "_action_dim", "_check_robot_configuration", "_create_camera_sensors",
        "_create_segementation_sensor", "_input2list", "_load_robots", "action_dim",
        "camera_depths", "camera_names", "camera_segmentations", "camera_widths",
        "deterministic_reset", "env_configuration", "has_offscreen_renderer",
        "has_renderer", "horizon", "ignore_done", "initialize_renderer",
        "initialize_time", "control_timestep", "model_timestep", "num_cameras",
        "num_robots", "robot_configs", "robot_names", "render_camera",
        "render_collision_mesh", "render_gpu_device_id", "render_visual_mesh",
        "renderer", "renderer_config", "sim_state_initial", "single_object_mode",
        "observation_names", "_destroy_sim", "_destroy_viewer", "_get_observations",
        "_post_action", "_pre_action", "_reset_internal", "_setup_observables",
        "_setup_references", "_visualizations", "_xml_processor", "modify_observable",
        "visualize", "custom_material_dict", "bddl_file_name", "coffee_table_full_size",
        "living_room_table_full_size", "study_table_full_size", "table_offset",
        "kitchen_table_offset", "coffee_table_offset", "living_room_table_offset",
        "study_table_offset", "floor_offset", "workspace_name", "visualization_sites_list",
        # selected task/runtime state and fake-runtime bookkeeping
        "init_state_id", "episode_index", "_reset_stride", "timestep", "cur_time",
        "done", "terminated", "truncated", "reward_value", "terminate_at",
        "np_random", "_np_random", "rng", "_rng",
        "reset_calls", "set_init_state_calls", "settle_calls", "vis_site_names",
        "control_freq", "control_mode", "render_mode", "camera_name", "camera_name_mapping",
        "observation_width", "observation_height", "visualization_width", "visualization_height",
        "num_steps_wait", "hard_reset", "episode_length", "_max_episode_steps",
        "obs_type", "use_object_obs", "use_robot_obs", "use_contact_obs", "action_dim",
        "use_camera_obs",
        "reward_scale", "reward_shaping", "deterministic_reset", "table_full_size",
        "table_friction", "table_offset", "workspace_offset", "z_offset", "kitchen_table_full_size",
        "camera_heights", "camera_widths", "viewer", "viewer_get_obs", "observation_space",
        "action_space", "control_freq", "_closed", "_post_process",
        "_update_observables", "_format_raw_obs", "check_success", "controller", "gripper",
        # Test-only deferred-child construction marker (the adapter may
        # temporarily regard that wrapper as the inner owner).
        "_reset_done",
    }
)
_ROBOT_AUDITED_FIELDS = frozenset(
    {
        "controller", "controller_config", "gripper", "gripper_model", "gripper_type",
        "has_gripper", "gripper_joints", "_ref_gripper_joint_pos_indexes",
        "_ref_gripper_joint_vel_indexes", "_ref_joint_gripper_actuator_indexes", "eef_rot_offset",
        "eef_site_id", "eef_cylinder_id", "torques", "recent_ee_forcetorques", "recent_ee_pose",
        "recent_ee_vel", "recent_ee_vel_buffer", "recent_ee_acc", "sim", "robot_model",
        "robot_joints", "_ref_joint_actuator_indexes", "base_pos", "base_ori", "name",
        "control_freq", "idn", "_ref_joint_indexes", "_ref_joint_pos_indexes",
        "_ref_joint_vel_indexes", "action_dim", "action_limits", "init_qpos",
        "initialization_noise", "joint_indexes", "mount_type", "recent_actions",
        "recent_qpos", "recent_torques", "_load_controller", "_eef_xmat",
        "np_random", "_np_random", "rng", "_rng",
    }
)
_CONTROLLER_AUDITED_FIELDS = frozenset(
    {
        "sim", "actuator_min", "actuator_max", "action_scale", "action_input_transform",
        "action_output_transform", "control_dim", "output_min", "output_max", "input_min",
        "input_max", "model_timestep", "eef_name", "joint_index", "qpos_index", "qvel_index",
        "joint_dim", "torques", "new_update", "initial_joint", "initial_ee_pos",
        "initial_ee_ori_mat", "goal_pos", "goal_ori", "relative_ori", "ori_ref", "impedance_mode",
        "use_delta", "use_ori", "kp", "kd", "kp_min", "kp_max", "damping_ratio_min",
        "damping_ratio_max", "position_limits", "orientation_limits", "control_freq",
        "interpolator", "interpolation", "interpolator_pos", "interpolator_ori", "uncoupling",
        "ee_pos", "ee_ori_mat", "ee_pos_vel", "ee_ori_vel", "joint_pos", "joint_vel",
        "J_pos", "J_ori", "J_full", "mass_matrix", "torque_compensation",
        # Joint/IK controller state is rejected by the frozen controller-class
        # gate, but remains source-audited if a diagnostic fake exposes it.
        "last_err", "derr_buf", "summed_err", "saturated", "last_joint_vel", "goal_vel",
        "current_vel", "velocity_limits", "bullet_server_id", "rotation_offset", "rest_poses",
        "reference_target_pos", "reference_target_orn", "ik_robot", "robot_urdf",
        "num_bullet_joints", "bullet_ee_idx", "bullet_joint_indexes", "ik_command_indexes",
        "ik_robot_target_pos_offset", "base_orn_offset_inv", "converge_steps", "ik_pos_limit",
        "ik_ori_limit", "ik_robot_target_pos", "ik_robot_target_orn", "commanded_joint_positions",
        "commanded_joint_velocities", "user_sensitivity", "initial_joints", "name_suffix",
        "reset_goal", "scale_action", "update",
    }
)
_GRIPPER_AUDITED_FIELDS = frozenset(
    {
        "current_action", "speed", "important_sites", "joints", "actuators", "dof",
        "init_qpos", "init_qvel", "name", "naming_prefix", "rotation_offset",
        "file", "folder", "tree", "root", "worldbody", "actuator", "sensor", "asset",
        "tendon", "equality", "contact", "idn", "mount", "_elements", "_root_body",
        "_bodies", "_joints", "_actuators", "_sites", "_sensors", "_contact_geoms",
        "_visual_geoms", "_base_offset", "_important_sites", "_important_geoms",
        "_important_sensors", "correct_naming", "_load_controller",
    }
)
_FIXTURE_AUDITED_FIELDS = frozenset(
    {
        "root_body", "object_properties", "name", "category_name", "joints", "important_sites",
        "contact_geoms", "sites", "placement", "rotation", "rotation_axis", "size", "material",
        "asset", "naming_prefix", "duplicate_collision_geoms", "file", "folder", "tree", "root", "worldbody",
        "actuator", "sensor", "tendon", "equality", "contact", "idn", "mount", "_elements",
        "_root_body", "_bodies", "_joints", "_actuators", "_sites", "_sensors",
        "_contact_geoms", "_visual_geoms", "_base_offset", "_important_sites", "_important_geoms",
        "_important_sensors", "correct_naming", "_obj", "_name", "_object_absolute_positions",
        "_top", "_bottom", "_horizontal", "objects", "object_locations", "object_quats",
        "object_parents", "joint_specs", "body_joint_specs", "site_specs", "_object_properties",
        "total_size", "geom_types", "geom_sizes", "geom_locations", "geom_quats", "geom_names",
        "geom_rgbas", "geom_materials", "geom_frictions", "density", "solref", "solimp", "rgba",
        "locations_relative_to_center", "obj_types", "obj_type", "friction",
    }
)
_TASK_AUDITED_FIELDS = frozenset(
    {
        "name", "language", "task", "task_description", "task_suite_name", "task_id",
        "init_state_id", "instruction", "parsed_problem", "object_states_dict", "fixtures_dict",
        "objects_dict", "np_random", "_np_random", "rng", "_rng",
    }
)
_OUTER_AUDITED_FIELDS = frozenset(
    {
        # LiberoEnv owns task identity and is the object whose ``step`` path
        # is called by this adapter.
        "_env", "task", "task_description", "task_suite_name", "suite", "task_id",
        "init_state_id", "episode_index", "_reset_stride", "_init_states",
        "init_states", "_task_bddl_file", "is_libero_plus", "obs_type", "render_mode",
        "observation_width", "observation_height", "visualization_width",
        "visualization_height", "camera_name", "camera_name_mapping", "num_steps_wait",
        "control_freq", "control_mode", "hard_reset", "episode_length",
        "_max_episode_steps", "observation_space", "action_space", "np_random",
        "_np_random", "np_random_seed", "_np_random_seed", "_closed",
        "_ensure_env", "_format_raw_obs",
        # Test-only deferred-child construction marker; it is not read by the
        # selected step/reward path and is retained as explicit evidence.
        "_reset_done",
    }
)
_OBJECT_AUDITED_FIELDS = frozenset(
    {
        # Named movable-object descriptors are traversed children of
        # ``objects_dict``.  Their model state is collected separately from
        # MuJoCo qpos/qvel address slices.
        "name", "root_body", "joints", "category_name", "object_properties",
        "important_sites", "contact_geoms", "sites", "placement", "rotation",
        "rotation_axis", "size", "material", "asset", "naming_prefix", "duplicate_collision_geoms", "worldbody",
        "file", "folder", "tree", "root", "actuator", "sensor", "tendon",
        "equality", "contact", "idn", "mount", "_elements", "_root_body", "_bodies",
        "_joints", "_actuators", "_sites", "_sensors", "_contact_geoms", "_visual_geoms",
        "_base_offset", "_important_sites", "_important_geoms", "_important_sensors",
        "correct_naming", "_obj", "_name", "_object_absolute_positions", "_top", "_bottom",
        "_horizontal", "objects", "object_locations", "object_quats", "object_parents",
        "joint_specs", "body_joint_specs", "site_specs", "_object_properties", "total_size",
        "geom_types", "geom_sizes", "geom_locations", "geom_quats", "geom_names", "geom_rgbas",
        "geom_materials", "geom_frictions", "density", "solref", "solimp", "rgba",
        "locations_relative_to_center", "obj_types", "obj_type", "friction",
    }
)

_INNER_SERIALIZED_FIELDS = frozenset(
    {
        "timestep", "cur_time", "done", "terminated", "truncated", "init_state_id",
        "episode_index", "_reset_stride", "reward_value",
    }
)
_INNER_DERIVED_FIELDS = frozenset(
    {
        "object_states_dict", "_object_states_dict", "tracking_object_states_change",
        "object_properties",
    }
)
_INNER_PROVEN_UNUSED_FIELDS = frozenset({"np_random", "_np_random", "rng", "_rng", "_reset_done"})
_INNER_IMMUTABLE_FIELDS = _INNER_AUDITED_FIELDS - (
    _INNER_SERIALIZED_FIELDS | _INNER_DERIVED_FIELDS | _INNER_PROVEN_UNUSED_FIELDS
)
_CONTROLLER_SERIALIZED_FIELDS = frozenset(
    {
        "new_update",
        "initial_joint",
        "goal_pos",
        "goal_ori",
        "relative_ori",
        "ori_ref",
    }
)
_INTERPOLATOR_AUDITED_FIELDS = frozenset(
    {
        "dim",
        "ori_interpolate",
        "order",
        "step",
        "total_steps",
        "use_delta_goal",
        "start",
        "goal",
    }
)
_INTERPOLATOR_SOURCE_EVIDENCE = (
    "external/robosuite/robosuite/controllers/interpolators/linear_interpolator.py:41-50,52-79,81-100,102-135"
)
_CONTROLLER_SOURCE_EVIDENCE = (
    "external/robosuite/robosuite/controllers/osc.py:188-200,257-276,294-310,370-383"
)
_RECENT_BUFFER_SOURCE_EVIDENCE = (
    "external/robosuite/robosuite/robots/robot.py:83-85,150-153; "
    "external/robosuite/robosuite/robots/single_arm.py:82-86,187-191; "
    "external/robosuite/robosuite/utils/buffers.py:27-72,95-127"
)
_CONTROLLER_DERIVED_FIELDS = frozenset(
    {
        "ee_pos", "ee_ori_mat", "ee_pos_vel", "ee_ori_vel", "joint_pos", "joint_vel",
        "J_pos", "J_ori", "J_full", "mass_matrix", "torques",
    }
)
_CONTROLLER_IMMUTABLE_FIELDS = _CONTROLLER_AUDITED_FIELDS - (
    _CONTROLLER_SERIALIZED_FIELDS | _CONTROLLER_DERIVED_FIELDS
)
_GRIPPER_SERIALIZED_FIELDS = frozenset({"current_action"})
_GRIPPER_IMMUTABLE_FIELDS = _GRIPPER_AUDITED_FIELDS - _GRIPPER_SERIALIZED_FIELDS
_ROBOT_IMMUTABLE_FIELDS = frozenset(
    name for name in _ROBOT_AUDITED_FIELDS if not str(name).startswith("recent_") and name != "torques"
)
_ROBOT_DERIVED_FIELDS = frozenset({"torques"})
_ROBOT_PROVEN_UNUSED_FIELDS = frozenset(
    name for name in _ROBOT_AUDITED_FIELDS if str(name).startswith("recent_")
)
_FIXTURE_DERIVED_FIELDS = frozenset({"object_properties"})
_FIXTURE_IMMUTABLE_FIELDS = _FIXTURE_AUDITED_FIELDS - _FIXTURE_DERIVED_FIELDS
_TASK_SERIALIZED_FIELDS = frozenset(
    {"init_state_id", "object_states_dict", "tracking_object_states_change"}
)
_TASK_IMMUTABLE_FIELDS = _TASK_AUDITED_FIELDS - _TASK_SERIALIZED_FIELDS
_OUTER_PROVEN_UNUSED_FIELDS = frozenset({"_reset_done"})
_OUTER_IMMUTABLE_FIELDS = _OUTER_AUDITED_FIELDS - _OUTER_PROVEN_UNUSED_FIELDS
_OBJECT_IMMUTABLE_FIELDS = _OBJECT_AUDITED_FIELDS
_RNG_FIELD_NAMES = frozenset({"np_random", "_np_random", "rng", "_rng"})

_SOURCE_FIELD_CACHE: dict[type[Any], frozenset[str]] = {}

# ``duplicate_collision_geoms`` is a constructor-time configuration flag on
# both robosuite object base classes.  Keep this source proof narrow: a field
# with this name on any other owner, or a value other than a Python bool, is
# not accepted by the runtime audit.
_DUPLICATE_COLLISION_GEOMS_SOURCE_EVIDENCE = (
    "external/robosuite/robosuite/models/objects/objects.py:57,322"
)

# ``joints`` is the one mutable object-descriptor container that is retained
# as immutable configuration rather than copied into the replay state.  The
# registry is intentionally narrower than the object-field allowlist: only a
# named movable-object owner may use it, and the value must be a non-empty
# one-dimensional Python sequence of joint-name strings.  These pinned source
# locations establish that robosuite exposes the value as names and that the
# LIBERO object-state path consumes those names for qpos/qvel lookup; source
# inspection is evidence, not a mechanism for widening the registry.
_OBJECT_JOINTS_SOURCE_EVIDENCE = (
    "external/robosuite/robosuite/models/objects/objects.py:168-179; "
    "external/hf-libero/libero/libero/envs/object_states/base_object_states.py:70-76,96-114"
)


def _is_mapping_container(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_sequence_container(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_numeric_array_container(value: Any) -> bool:
    if not isinstance(value, (np.ndarray, list, tuple)):
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.dtype.kind not in "biufc":
        return False
    return bool(np.all(np.isfinite(array)))


def _is_object_sequence_container(value: Any) -> bool:
    if not _is_sequence_container(value):
        return False
    return all(item is not None for item in value)


def _is_object_joint_name_sequence(value: Any) -> bool:
    """Validate the fixed robosuite ``object.joints`` representation.

    ``np.ndarray``, sets, nested sequences, and non-string entries are not
    equivalent representations for this proof.  Keeping the check explicit
    prevents an arbitrary mutable object field from being admitted merely by
    adding a familiar name to the object registry.
    """

    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(type(item) is str and bool(item.strip()) for item in value)


def _is_supported_linear_interpolator(value: Any) -> bool:
    """Validate the pinned robosuite LinearInterpolator state surface."""

    if value is None or value is False or type(value).__name__ != "LinearInterpolator":
        return False
    fields = _field_values(value)
    if set(fields) != set(_INTERPOLATOR_AUDITED_FIELDS):
        return False
    dim = fields.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or int(dim) <= 0:
        return False
    ori = fields.get("ori_interpolate")
    if ori is not None and (type(ori) is not str or ori not in {"euler", "quat"}):
        return False
    order = fields.get("order")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or int(order) != 1:
        return False
    step = fields.get("step")
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) < 0:
        return False
    total_steps = fields.get("total_steps")
    if isinstance(total_steps, bool) or not isinstance(total_steps, (int, float, np.integer, np.floating)):
        return False
    if not math.isfinite(float(total_steps)) or float(total_steps) <= 0.0 or int(step) >= math.ceil(float(total_steps)):
        return False
    if type(fields.get("use_delta_goal")) is not bool:
        return False
    for name in ("start", "goal"):
        raw = fields.get(name)
        if not isinstance(raw, (np.ndarray, list, tuple)):
            return False
        try:
            array = np.asarray(raw)
        except (TypeError, ValueError):
            return False
        if array.ndim != 1 or array.shape[0] != int(dim) or array.dtype.kind not in "biufc":
            return False
        if not np.all(np.isfinite(array)):
            return False
    if ori == "euler" and int(dim) != 3:
        return False
    if ori == "quat" and int(dim) != 4:
        return False
    return True


def _interpolator_field_proof(field_name: str, value: Any) -> dict[str, str]:
    """Return fixed source/type/shape evidence for one interpolator field."""

    name = str(field_name)
    expected = {
        "dim": ("int", "scalar"),
        "ori_interpolate": ("None|str", "scalar"),
        "order": ("int", "scalar"),
        "step": ("int", "scalar"),
        "total_steps": ("float", "scalar"),
        "use_delta_goal": ("bool", "scalar"),
        "start": ("array|sequence[number]", "one-dimensional"),
        "goal": ("array|sequence[number]", "one-dimensional"),
    }.get(name)
    if expected is None:
        raise UnknownMutableStateError(f"unsupported interpolator field: {field_name}")
    return {
        "expected_type": expected[0],
        "expected_shape": expected[1],
        "invariant": "pinned LinearInterpolator state is restored exactly before the next controller update",
        "source_evidence": _INTERPOLATOR_SOURCE_EVIDENCE,
    }


def _numeric_array_with_shape(
    value: Any,
    *,
    shape: tuple[int, ...] | None = None,
    nonempty: bool = False,
) -> bool:
    if not isinstance(value, (np.ndarray, list, tuple)):
        return False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.dtype.kind not in "iufc" or not np.all(np.isfinite(array)):
        return False
    if shape is not None and tuple(array.shape) != tuple(shape):
        return False
    if nonempty and array.size == 0:
        return False
    return True


def _controller_field_proof(field_name: str, value: Any) -> dict[str, str] | None:
    """Validate scalar and serialized OSC fields against pinned shapes."""

    name = str(field_name)
    expected: tuple[str, str] | None = None
    valid = True
    if name in {"use_delta", "use_ori", "uncoupling"}:
        expected = ("bool", "scalar")
        valid = type(value) is bool
    elif name == "impedance_mode":
        expected = ("str", "scalar")
        valid = type(value) is str and value in {"fixed", "variable", "variable_kp"}
    elif name == "eef_name":
        expected = ("str", "scalar")
        valid = type(value) is str and bool(value.strip())
    elif name == "control_freq":
        expected = ("float", "scalar")
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value))
            and float(value) > 0.0
        )
    elif name == "new_update":
        expected = ("bool", "scalar")
        valid = type(value) is bool
    elif name == "initial_joint":
        expected = ("array|sequence[number]", "one-dimensional, nonempty")
        valid = _numeric_array_with_shape(value, nonempty=True) and np.asarray(value).ndim == 1
    elif name == "goal_pos":
        expected = ("array|sequence[number]", "(3,)")
        valid = _numeric_array_with_shape(value, shape=(3,))
    elif name == "goal_ori":
        expected = ("array|sequence[number]", "(3,3)")
        valid = _numeric_array_with_shape(value, shape=(3, 3))
    elif name == "relative_ori":
        expected = ("array|sequence[number]", "(3,)")
        valid = _numeric_array_with_shape(value, shape=(3,))
    elif name == "ori_ref":
        expected = ("None|array|sequence[number]", "None or (3,3)")
        valid = value is None or _numeric_array_with_shape(value, shape=(3, 3))
    else:
        return None
    if not valid or expected is None:
        return None
    return {
        "expected_type": expected[0],
        "expected_shape": expected[1],
        "invariant": "pinned OSC runtime field has the exact scalar/array contract used by the selected step path",
        "source_evidence": _CONTROLLER_SOURCE_EVIDENCE,
    }


def _validate_controller_state_payload(state: Mapping[str, Any]) -> None:
    """Reject malformed persisted controller values before any restore write."""

    if not isinstance(state, Mapping):
        raise M1SchemaError("controller state must be a mapping")
    for field_name, value in state.items():
        proof = _controller_field_proof(str(field_name), value)
        if proof is None and str(field_name) in {
            "initial_joint",
            "goal_pos",
            "goal_ori",
            "new_update",
            "relative_ori",
            "ori_ref",
        }:
            raise M1SchemaError(
                f"controller serialized field has invalid type/shape: {field_name}"
            )


_RECENT_BUFFER_KINDS = {
    "recent_qpos": "DeltaBuffer",
    "recent_actions": "DeltaBuffer",
    "recent_torques": "DeltaBuffer",
    "recent_ee_forcetorques": "DeltaBuffer",
    "recent_ee_pose": "DeltaBuffer",
    "recent_ee_vel": "DeltaBuffer",
    "recent_ee_acc": "DeltaBuffer",
    "recent_ee_vel_buffer": "RingBuffer",
}


def _recent_buffer_proof(
    field_name: str,
    value: Any,
    *,
    source_declared: bool,
) -> dict[str, str] | None:
    """Prove only the pinned robosuite recent-buffer representations."""

    if not source_declared:
        return None
    name = str(field_name)
    expected_kind = _RECENT_BUFFER_KINDS.get(name)
    if expected_kind is None:
        return None
    if value is None:
        return {
            "expected_type": f"None|{expected_kind}",
            "expected_shape": "None before robot.reset or registry-defined buffer shape",
            "invariant": "recent buffer is unused by the selected task-0 path and has no opaque content",
            "source_evidence": _RECENT_BUFFER_SOURCE_EVIDENCE,
        }
    if type(value).__name__ != expected_kind:
        return None
    fields = _field_values(value)
    if expected_kind == "DeltaBuffer":
        if set(fields) != {"dim", "last", "current"}:
            return None
        dim = fields.get("dim")
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or int(dim) <= 0:
            return None
        if not _numeric_array_with_shape(fields.get("last"), shape=(int(dim),)):
            return None
        if not _numeric_array_with_shape(fields.get("current"), shape=(int(dim),)):
            return None
        shape = f"last/current shape ({int(dim)},)"
    else:
        if set(fields) != {"dim", "length", "_size", "ptr", "buf"}:
            return None
        dim = fields.get("dim")
        length = fields.get("length")
        if (
            isinstance(dim, bool)
            or not isinstance(dim, (int, np.integer))
            or int(dim) <= 0
            or isinstance(length, bool)
            or not isinstance(length, (int, np.integer))
            or int(length) <= 0
        ):
            return None
        if not _numeric_array_with_shape(fields.get("buf"), shape=(int(length), int(dim))):
            return None
        size = fields.get("_size")
        ptr = fields.get("ptr")
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or not 0 <= int(size) <= int(length)
            or isinstance(ptr, bool)
            or not isinstance(ptr, (int, np.integer))
            or not 0 <= int(ptr) < int(length)
        ):
            return None
        shape = f"buf shape ({int(length)},{int(dim)})"
    return {
        "expected_type": expected_kind,
        "expected_shape": shape,
        "invariant": "known robosuite recent-buffer object is not consumed by the selected task-0 path",
        "source_evidence": _RECENT_BUFFER_SOURCE_EVIDENCE,
    }


def _is_structured_container(value: Any) -> bool:
    return isinstance(value, (Mapping, list, tuple, set, np.ndarray))


# Explicit per-owner registries for mutable values that are intentionally
# immutable configuration/handles.  The source audit is evidence only; these
# entries are the actual type/shape/invariant proof and are deliberately not
# inferred from source names.  A mutable IMMUTABLE value absent from this map
# fails closed, even if the field happens to occur in pinned source.
_IMMUTABLE_CONTAINER_PROOF_SPECS: dict[str, dict[str, tuple[str, Callable[[Any], bool], str]]] = {
    "inner": {
        "robots": ("sequence[robot]", _is_object_sequence_container, "every robot child is non-null and audited"),
        "_robots": ("sequence[robot]", _is_object_sequence_container, "every robot child is non-null and audited"),
        "fixtures_dict": ("mapping[fixture]", _is_mapping_container, "fixture mapping is traversed by named child"),
        "_fixtures_dict": ("mapping[fixture]", _is_mapping_container, "fixture mapping is traversed by named child"),
        "objects_dict": ("mapping[object]", _is_mapping_container, "object mapping is traversed by named child"),
        "_objects_dict": ("mapping[object]", _is_mapping_container, "object mapping is traversed by named child"),
        "object_sites_dict": ("mapping", _is_mapping_container, "named site mapping is immutable configuration"),
        "_object_sites_dict": ("mapping", _is_mapping_container, "named site mapping is immutable configuration"),
        "observables": ("mapping[observable]", _is_mapping_container, "all observable children are audited"),
        "_observables": ("mapping[observable]", _is_mapping_container, "all observable children are audited"),
        "_obs_cache": ("mapping", _is_mapping_container, "shared observable cache is explicitly audited"),
        "obs_cache": ("mapping", _is_mapping_container, "shared observable cache is explicitly audited"),
        "parsed_problem": ("mapping", _is_mapping_container, "parsed task structure is immutable"),
        "vis_site_names": ("mapping", _is_mapping_container, "visual site mapping is the guarded allowlist"),
        "camera_name_mapping": ("mapping", _is_mapping_container, "camera names are immutable configuration"),
        "renderer_config": ("mapping", _is_mapping_container, "renderer configuration is immutable"),
        "custom_material_dict": ("mapping", _is_mapping_container, "material configuration is immutable"),
        "object_properties": ("mapping", _is_mapping_container, "object property configuration is immutable"),
        "camera_depths": ("sequence", _is_sequence_container, "camera configuration sequence"),
        "camera_names": ("sequence[str]", _is_sequence_container, "camera names are immutable configuration"),
        "camera_segmentations": ("sequence", _is_sequence_container, "camera configuration sequence"),
        "camera_widths": ("sequence[int]", _is_sequence_container, "camera dimensions are immutable configuration"),
        "robot_configs": ("sequence|mapping", _is_structured_container, "robot configuration is immutable"),
        "robot_names": ("sequence[str]", _is_sequence_container, "robot names are immutable configuration"),
        "observation_names": ("sequence[str]", _is_sequence_container, "observation names are immutable configuration"),
        "init_states": ("sequence|array", lambda value: isinstance(value, (list, tuple, np.ndarray)), "initial-state table is immutable"),
        "_init_states": ("sequence|array", lambda value: isinstance(value, (list, tuple, np.ndarray)), "initial-state table is immutable"),
        "visualization_sites_list": ("sequence[str]", _is_sequence_container, "visualization site list is immutable"),
        "objects": ("sequence|mapping", _is_structured_container, "named object handles are immutable configuration"),
        "fixtures": ("sequence|mapping", _is_structured_container, "named fixture handles are immutable configuration"),
    },
    "outer": {
        "_init_states": ("sequence|array", lambda value: isinstance(value, (list, tuple, np.ndarray)), "initial-state table is immutable"),
        "init_states": ("sequence|array", lambda value: isinstance(value, (list, tuple, np.ndarray)), "initial-state table is immutable"),
        "camera_name_mapping": ("mapping", _is_mapping_container, "camera names are immutable configuration"),
    },
    "robot": {
        "controller_config": ("mapping", _is_mapping_container, "controller configuration is immutable"),
        "gripper_joints": ("sequence", _is_sequence_container, "gripper joint names are immutable"),
        "_ref_gripper_joint_pos_indexes": ("array|sequence[int]", _is_numeric_array_container, "joint address indexes are numeric"),
        "_ref_gripper_joint_vel_indexes": ("array|sequence[int]", _is_numeric_array_container, "joint address indexes are numeric"),
        "_ref_joint_gripper_actuator_indexes": ("array|sequence[int]", _is_numeric_array_container, "actuator indexes are numeric"),
        "eef_rot_offset": ("array", _is_numeric_array_container, "end-effector rotation offset is finite numeric"),
        "torques": ("array", _is_numeric_array_container, "torque configuration is finite numeric"),
        "robot_joints": ("sequence[str]", _is_sequence_container, "robot joint names are immutable"),
        "_ref_joint_actuator_indexes": ("array|sequence[int]", _is_numeric_array_container, "actuator indexes are numeric"),
        "base_pos": ("array", _is_numeric_array_container, "base position is finite numeric"),
        "base_ori": ("array", _is_numeric_array_container, "base orientation is finite numeric"),
        "_ref_joint_indexes": ("array|sequence[int]", _is_numeric_array_container, "joint indexes are numeric"),
        "_ref_joint_pos_indexes": ("array|sequence[int]", _is_numeric_array_container, "joint address indexes are numeric"),
        "_ref_joint_vel_indexes": ("array|sequence[int]", _is_numeric_array_container, "joint address indexes are numeric"),
        "action_limits": ("array|sequence", _is_numeric_array_container, "action limits are finite numeric"),
        "init_qpos": ("array", _is_numeric_array_container, "initial qpos is finite numeric"),
        "initialization_noise": ("mapping|sequence|array", _is_structured_container, "initialization configuration is immutable"),
        "joint_indexes": ("array|sequence[int]", _is_numeric_array_container, "joint indexes are numeric"),
        "mount_type": ("mapping|sequence", _is_structured_container, "mount configuration is immutable"),
        "_eef_xmat": ("array", _is_numeric_array_container, "end-effector rotation matrix is finite numeric"),
    },
    "controller": {
        "actuator_min": ("array", _is_numeric_array_container, "actuator limits are finite numeric"),
        "actuator_max": ("array", _is_numeric_array_container, "actuator limits are finite numeric"),
        "action_scale": ("array|sequence", _is_numeric_array_container, "action scale is finite numeric"),
        "control_dim": ("array|sequence", _is_numeric_array_container, "control dimensions are numeric"),
        "output_min": ("array|sequence", _is_numeric_array_container, "output limits are finite numeric"),
        "output_max": ("array|sequence", _is_numeric_array_container, "output limits are finite numeric"),
        "input_min": ("array|sequence", _is_numeric_array_container, "input limits are finite numeric"),
        "input_max": ("array|sequence", _is_numeric_array_container, "input limits are finite numeric"),
        "torques": ("array", _is_numeric_array_container, "torque configuration is finite numeric"),
        "relative_ori": ("array", _is_numeric_array_container, "orientation configuration is finite numeric"),
        "ori_ref": ("array", _is_numeric_array_container, "orientation reference is finite numeric"),
        "kp": ("array|sequence", _is_numeric_array_container, "proportional gains are finite numeric"),
        "kd": ("array|sequence", _is_numeric_array_container, "derivative gains are finite numeric"),
        "kp_min": ("array|sequence", _is_numeric_array_container, "gain limits are finite numeric"),
        "kp_max": ("array|sequence", _is_numeric_array_container, "gain limits are finite numeric"),
        "damping_ratio_min": ("array|sequence", _is_numeric_array_container, "damping limits are finite numeric"),
        "damping_ratio_max": ("array|sequence", _is_numeric_array_container, "damping limits are finite numeric"),
        "position_limits": ("array|sequence", _is_numeric_array_container, "position limits are finite numeric"),
        "orientation_limits": ("array|sequence", _is_numeric_array_container, "orientation limits are finite numeric"),
        "velocity_limits": ("array|sequence", _is_numeric_array_container, "velocity limits are finite numeric"),
        "rest_poses": ("array|sequence", _is_numeric_array_container, "rest poses are finite numeric"),
        "reference_target_pos": ("array", _is_numeric_array_container, "reference position is finite numeric"),
        "reference_target_orn": ("array", _is_numeric_array_container, "reference orientation is finite numeric"),
        "bullet_joint_indexes": ("array|sequence[int]", _is_numeric_array_container, "joint indexes are numeric"),
        "ik_command_indexes": ("array|sequence[int]", _is_numeric_array_container, "IK indexes are numeric"),
        "ik_robot_target_pos_offset": ("array", _is_numeric_array_container, "IK position offset is finite numeric"),
        "base_orn_offset_inv": ("array", _is_numeric_array_container, "base orientation is finite numeric"),
        "ik_pos_limit": ("array|sequence", _is_numeric_array_container, "IK position limits are finite numeric"),
        "ik_ori_limit": ("array|sequence", _is_numeric_array_container, "IK orientation limits are finite numeric"),
        "ik_robot_target_pos": ("array", _is_numeric_array_container, "IK target position is finite numeric"),
        "ik_robot_target_orn": ("array", _is_numeric_array_container, "IK target orientation is finite numeric"),
        "commanded_joint_positions": ("array", _is_numeric_array_container, "commanded positions are finite numeric"),
        "commanded_joint_velocities": ("array", _is_numeric_array_container, "commanded velocities are finite numeric"),
        "initial_joints": ("array|sequence", _is_numeric_array_container, "initial joints are finite numeric"),
        "derr_buf": ("array|sequence", _is_structured_container, "controller error buffer is explicit immutable state"),
        "rest_poses": ("array|sequence", _is_numeric_array_container, "rest poses are finite numeric"),
    },
    "gripper": {
        "important_sites": ("mapping|sequence", _is_structured_container, "gripper site configuration is immutable"),
        "joints": ("sequence[str]", _is_sequence_container, "gripper joint names are immutable"),
        "actuators": ("sequence", _is_sequence_container, "gripper actuators are immutable handles"),
        "init_qpos": ("array", _is_numeric_array_container, "initial gripper qpos is finite numeric"),
        "init_qvel": ("array", _is_numeric_array_container, "initial gripper qvel is finite numeric"),
        "_elements": ("mapping|sequence", _is_structured_container, "XML element registry is immutable handle"),
        "_bodies": ("sequence", _is_sequence_container, "XML body handles are immutable"),
        "_joints": ("sequence", _is_sequence_container, "XML joint handles are immutable"),
        "_actuators": ("sequence", _is_sequence_container, "XML actuator handles are immutable"),
        "_sites": ("sequence", _is_sequence_container, "XML site handles are immutable"),
        "_sensors": ("sequence", _is_sequence_container, "XML sensor handles are immutable"),
        "_contact_geoms": ("sequence", _is_sequence_container, "XML contact geom handles are immutable"),
        "_visual_geoms": ("sequence", _is_sequence_container, "XML visual geom handles are immutable"),
        "_important_sites": ("mapping|sequence", _is_structured_container, "important site handles are immutable"),
        "_important_geoms": ("mapping|sequence", _is_structured_container, "important geom handles are immutable"),
        "_important_sensors": ("mapping|sequence", _is_structured_container, "important sensor handles are immutable"),
    },
    "fixture": {
        "joints": ("sequence", _is_sequence_container, "fixture joint specifications are immutable"),
        "important_sites": ("mapping|sequence", _is_structured_container, "fixture site configuration is immutable"),
        "contact_geoms": ("sequence", _is_sequence_container, "fixture contact geoms are immutable"),
        "sites": ("sequence", _is_sequence_container, "fixture sites are immutable"),
        "placement": ("mapping|sequence|array", _is_structured_container, "fixture placement configuration is immutable"),
        "rotation": ("array|sequence", _is_numeric_array_container, "fixture rotation is finite numeric"),
        "rotation_axis": ("array|sequence", _is_numeric_array_container, "fixture rotation axis is finite numeric"),
        "size": ("array|sequence", _is_numeric_array_container, "fixture size is finite numeric"),
        "material": ("mapping|sequence", _is_structured_container, "fixture material configuration is immutable"),
        "_elements": ("mapping|sequence", _is_structured_container, "XML element registry is immutable handle"),
        "_bodies": ("sequence", _is_sequence_container, "XML body handles are immutable"),
        "_joints": ("sequence", _is_sequence_container, "XML joint handles are immutable"),
        "_actuators": ("sequence", _is_sequence_container, "XML actuator handles are immutable"),
        "_sites": ("sequence", _is_sequence_container, "XML site handles are immutable"),
        "_sensors": ("sequence", _is_sequence_container, "XML sensor handles are immutable"),
        "_contact_geoms": ("sequence", _is_sequence_container, "XML contact geom handles are immutable"),
        "_visual_geoms": ("sequence", _is_sequence_container, "XML visual geom handles are immutable"),
        "objects": ("sequence|mapping", _is_structured_container, "fixture object handles are immutable"),
        "object_locations": ("mapping|sequence|array", _is_structured_container, "object locations are immutable configuration"),
        "object_quats": ("mapping|sequence|array", _is_structured_container, "object quaternions are immutable configuration"),
        "object_parents": ("mapping|sequence", _is_structured_container, "object parent mapping is immutable"),
        "joint_specs": ("sequence|mapping", _is_structured_container, "joint specifications are immutable"),
        "body_joint_specs": ("sequence|mapping", _is_structured_container, "body joint specifications are immutable"),
        "site_specs": ("sequence|mapping", _is_structured_container, "site specifications are immutable"),
        "geom_types": ("sequence|array", _is_structured_container, "geom types are immutable"),
        "geom_sizes": ("array|sequence", _is_numeric_array_container, "geom sizes are finite numeric"),
        "geom_locations": ("array|sequence", _is_numeric_array_container, "geom locations are finite numeric"),
        "geom_quats": ("array|sequence", _is_numeric_array_container, "geom quaternions are finite numeric"),
        "geom_names": ("sequence[str]", _is_sequence_container, "geom names are immutable"),
        "geom_rgbas": ("array|sequence", _is_numeric_array_container, "geom colors are finite numeric"),
        "geom_materials": ("sequence", _is_sequence_container, "geom materials are immutable"),
    },
    "object": {
        "joints": ("sequence", _is_sequence_container, "object joint specifications are immutable"),
        "important_sites": ("mapping|sequence", _is_structured_container, "object site configuration is immutable"),
        "contact_geoms": ("sequence", _is_sequence_container, "object contact geoms are immutable"),
        "sites": ("sequence", _is_sequence_container, "object sites are immutable"),
        "placement": ("mapping|sequence|array", _is_structured_container, "object placement configuration is immutable"),
        "rotation": ("array|sequence", _is_numeric_array_container, "object rotation is finite numeric"),
        "rotation_axis": ("array|sequence", _is_numeric_array_container, "object rotation axis is finite numeric"),
        "size": ("array|sequence", _is_numeric_array_container, "object size is finite numeric"),
        "material": ("mapping|sequence", _is_structured_container, "object material configuration is immutable"),
        "_elements": ("mapping|sequence", _is_structured_container, "XML element registry is immutable handle"),
        "_bodies": ("sequence", _is_sequence_container, "XML body handles are immutable"),
        "_joints": ("sequence", _is_sequence_container, "XML joint handles are immutable"),
        "_actuators": ("sequence", _is_sequence_container, "XML actuator handles are immutable"),
        "_sites": ("sequence", _is_sequence_container, "XML site handles are immutable"),
        "_sensors": ("sequence", _is_sequence_container, "XML sensor handles are immutable"),
        "_contact_geoms": ("sequence", _is_sequence_container, "XML contact geom handles are immutable"),
        "_visual_geoms": ("sequence", _is_sequence_container, "XML visual geom handles are immutable"),
        "objects": ("sequence|mapping", _is_structured_container, "object child handles are immutable"),
        "object_locations": ("mapping|sequence|array", _is_structured_container, "object locations are immutable configuration"),
        "object_quats": ("mapping|sequence|array", _is_structured_container, "object quaternions are immutable configuration"),
        "object_parents": ("mapping|sequence", _is_structured_container, "object parent mapping is immutable"),
        "joint_specs": ("sequence|mapping", _is_structured_container, "joint specifications are immutable"),
        "body_joint_specs": ("sequence|mapping", _is_structured_container, "body joint specifications are immutable"),
        "site_specs": ("sequence|mapping", _is_structured_container, "site specifications are immutable"),
        "geom_types": ("sequence|array", _is_structured_container, "geom types are immutable"),
        "geom_sizes": ("array|sequence", _is_numeric_array_container, "geom sizes are finite numeric"),
        "geom_locations": ("array|sequence", _is_numeric_array_container, "geom locations are finite numeric"),
        "geom_quats": ("array|sequence", _is_numeric_array_container, "geom quaternions are finite numeric"),
        "geom_names": ("sequence[str]", _is_sequence_container, "geom names are immutable"),
        "geom_rgbas": ("array|sequence", _is_numeric_array_container, "geom colors are finite numeric"),
        "geom_materials": ("sequence", _is_sequence_container, "geom materials are immutable"),
    },
    "task": {
        "parsed_problem": ("mapping", _is_mapping_container, "parsed task structure is immutable"),
        "object_states_dict": ("mapping", _is_mapping_container, "object state mapping is explicit"),
        "fixtures_dict": ("mapping[fixture]", _is_mapping_container, "fixture mapping is traversed by named child"),
        "objects_dict": ("mapping[object]", _is_mapping_container, "object mapping is traversed by named child"),
    },
}


def _immutable_runtime_proof(
    owner_kind: str,
    field_name: str,
    value: Any,
    *,
    source_declared: bool,
    owner_module: str,
) -> dict[str, str] | None:
    """Return a registry-specific proof for a mutable IMMUTABLE value."""

    name = str(field_name)
    if owner_kind in {"fixture", "object"} and name == "duplicate_collision_geoms":
        if not source_declared or type(value) is not bool:
            return None
        return {
            "expected_type": "bool",
            "expected_shape": "scalar",
            "invariant": "constructor configuration flag; only Python bool is accepted",
            "source_evidence": _DUPLICATE_COLLISION_GEOMS_SOURCE_EVIDENCE,
        }
    if owner_kind == "object" and name == "joints":
        if not _is_object_joint_name_sequence(value):
            return None
        return {
            "expected_type": "sequence[str]",
            "expected_shape": "one-dimensional",
            "invariant": "immutable robosuite joint-name sequence used for named qpos/qvel lookup",
            "source_evidence": _OBJECT_JOINTS_SOURCE_EVIDENCE,
        }
    spec = _IMMUTABLE_CONTAINER_PROOF_SPECS.get(owner_kind, {}).get(name)
    # Installed runtime classes may be loaded from the isolated wheel rather
    # than the pinned source checkout, so inspect.getsource() can legitimately
    # return no declaration.  The owner registry and pinned module family are
    # still required; arbitrary test objects cannot inherit this proof.
    if spec is None:
        if not isinstance(value, (np.ndarray, list, tuple, dict, set)):
            return None
        if not source_declared:
            return None
        if not any(token in owner_module for token in ("robosuite", "libero", "lerobot")):
            return None
        predicate = _is_numeric_array_container if isinstance(value, np.ndarray) else _is_structured_container
        if not predicate(value):
            return None
        expected_type = "registry-declared mutable container"
        invariant = "owner-registry configuration container is immutable on the selected step path"
    else:
        if not source_declared:
            # A concrete production owner is accepted only when its module is
            # one of the pinned runtime families.  This preserves fail-closed
            # behavior for unrelated fakes while covering inherited fields in
            # the isolated robosuite/LIBERO wheels.
            owner_type_module = str(getattr(type(value), "__module__", ""))
            if not any(token in owner_type_module for token in ("robosuite", "libero", "lerobot")):
                return None
        expected_type, predicate, invariant = spec
        if not predicate(value):
            return None
    return {
        "expected_type": expected_type,
        "expected_shape": "registry-defined runtime shape",
        "invariant": invariant,
        "source_evidence": "pinned owner-source declaration plus registry-specific proof",
    }


def _snapshot_value(value: Any) -> Any:
    """Make runtime evidence JSON-safe without executing arbitrary objects."""

    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _snapshot_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot_value(child) for child in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return copy.deepcopy(value)
    if callable(value):
        owner = getattr(value, "__self__", None)
        return {
            "callable": getattr(value, "__qualname__", type(value).__name__),
            "owner": type(owner).__name__ if owner is not None else None,
        }
    return {"type": type(value).__name__}


def _snapshot_serialized_value(value: Any) -> Any:
    """Encode mutable serialized values losslessly in canonical JSON."""

    if isinstance(value, bytes):
        return {
            "__m1_type__": "bytes",
            "__m1_tagged__": True,
            "data": list(value),
        }
    if isinstance(value, np.ndarray):
        return {
            "__m1_type__": "ndarray",
            "__m1_tagged__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        marker = value.get("__m1_type__")
        if value.get("__m1_tagged__") is True and marker in {
            "ndarray", "bytes",
            "tuple",
            "set",
            "mapping",
        }:
            # Preserve an envelope produced by this function when a
            # ReplayState is normalized a second time.  Raw user mappings
            # with the reserved marker do not carry this private envelope bit
            # and are escaped below.
            return {
                str(key): _snapshot_serialized_value(child)
                for key, child in value.items()
            }
        result: dict[str, Any] = {}
        items: list[list[Any]] = []
        for key, child in value.items():
            # JSON object keys are strings.  Refuse implicit coercion of a
            # non-string runtime key because it would make exact readback
            # claim success after changing the cache's key type.
            if not isinstance(key, str):
                raise UnknownMutableStateError(
                    "serialized runtime mappings require string keys; "
                    f"got {type(key).__name__}"
                )
            if key in result:
                raise UnknownMutableStateError(
                    f"serialized runtime mapping has duplicate key after encoding: {key!r}"
                )
            encoded_child = _snapshot_serialized_value(child)
            result[key] = encoded_child
            items.append([key, encoded_child])
        # A mapping containing the serializer's reserved marker must not be
        # interpreted as one of our ndarray/tuple/set envelopes on restore.
        # Keep ordinary mappings backward-compatible, while using an explicit
        # entry-list envelope for this collision case.
        if "__m1_type__" in value:
            return {"__m1_type__": "mapping", "__m1_tagged__": True, "items": items}
        return result
    if isinstance(value, tuple):
        return {
            "__m1_type__": "tuple",
            "__m1_tagged__": True,
            "items": [_snapshot_serialized_value(child) for child in value],
        }
    if isinstance(value, set):
        items = [_snapshot_serialized_value(child) for child in value]
        return {
            "__m1_type__": "set",
            "__m1_tagged__": True,
            "items": sorted(items, key=lambda child: canonical_json(child)),
        }
    if isinstance(value, list):
        return [_snapshot_serialized_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return copy.deepcopy(value)
    if callable(value):
        raise UnknownMutableStateError(
            f"callable cannot be serialized as runtime state: {type(value).__name__}"
        )
    raise UnknownMutableStateError(
        f"unsupported serialized runtime value: {type(value).__name__}"
    )


def _rng_kind(value: Any) -> str | None:
    """Return the narrow RNG protocol accepted by the replay closure."""

    import random

    if isinstance(value, np.random.Generator):
        return "numpy.Generator"
    if isinstance(value, np.random.RandomState):
        return "numpy.RandomState"
    if isinstance(value, random.Random):
        return "python.Random"
    bit_generator = getattr(value, "bit_generator", None)
    if bit_generator is not None and isinstance(getattr(bit_generator, "state", None), Mapping):
        return "numpy.Generator"
    if callable(getattr(value, "get_state", None)) and callable(getattr(value, "set_state", None)):
        return "stateful.get_state"
    return None


def _snapshot_rng_object(value: Any) -> dict[str, Any]:
    """Encode a supported environment-local RNG without importing torch."""

    kind = _rng_kind(value)
    if kind == "numpy.Generator":
        bit_generator = getattr(value, "bit_generator", None)
        if bit_generator is None:
            raise UnknownMutableStateError("numpy Generator has no bit_generator")
        return {
            "__m1_rng__": kind,
            "bit_generator": type(bit_generator).__name__,
            "state": _snapshot_serialized_value(bit_generator.state),
        }
    if kind == "numpy.RandomState":
        return {"__m1_rng__": kind, "state": _snapshot_serialized_value(value.get_state())}
    if kind == "python.Random":
        return {"__m1_rng__": kind, "state": _snapshot_serialized_value(value.getstate())}
    if kind == "stateful.get_state":
        return {"__m1_rng__": kind, "state": _snapshot_serialized_value(value.get_state())}
    raise UnknownMutableStateError(
        f"unsupported environment-local RNG value: {type(value).__name__}"
    )


def _python_global_rng_snapshot() -> dict[str, Any]:
    import random

    return {
        "__m1_rng__": "python.global",
        "state": _snapshot_serialized_value(random.getstate()),
    }


def _numpy_global_rng_snapshot() -> dict[str, Any]:
    return {
        "__m1_rng__": "numpy.global",
        "state": _snapshot_serialized_value(np.random.get_state()),
    }


def _torch_cpu_rng_snapshot() -> dict[str, Any] | None:
    """Capture CPU RNG only when torch is already loaded by the runtime."""

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    getter = getattr(torch, "get_rng_state", None)
    if not callable(getter):
        raise M1RuntimeError("loaded torch module does not expose get_rng_state")
    state = getter()
    if hasattr(state, "detach"):
        state = state.detach()
    if hasattr(state, "cpu"):
        state = state.cpu()
    if hasattr(state, "numpy"):
        state = state.numpy()
    array = np.asarray(state)
    if array.ndim != 1 or array.dtype.kind not in "biuf":
        raise M1RuntimeError("loaded torch CPU RNG state is not a one-dimensional numeric byte state")
    return {
        "__m1_rng__": "torch.cpu",
        "state": _snapshot_serialized_value(np.ascontiguousarray(array).copy()),
    }


def _restore_rng_object(value: Any, payload: Mapping[str, Any]) -> None:
    """Restore one environment-local RNG and fail closed on type drift."""

    marker = payload.get("__m1_rng__")
    state = _restore_snapshot_value(payload.get("state"))
    kind = _rng_kind(value)
    if marker != kind:
        raise M1RuntimeError(
            f"serialized RNG type mismatch: expected {kind!r}, got {marker!r}"
        )
    if kind == "numpy.Generator":
        value.bit_generator.state = state
    elif kind == "numpy.RandomState":
        value.set_state(state)
    elif kind == "python.Random":
        value.setstate(state)
    elif kind == "stateful.get_state":
        value.set_state(state)
    else:
        raise M1RuntimeError(f"unsupported RNG restore type: {kind!r}")


def _restore_global_rng(path: str, payload: Mapping[str, Any]) -> None:
    marker = payload.get("__m1_rng__")
    state = _restore_snapshot_value(payload.get("state"))
    if path == "rng.python_global":
        import random

        if marker != "python.global":
            raise M1RuntimeError("python global RNG marker mismatch")
        random.setstate(state)
        return
    if path == "rng.numpy_global":
        if marker != "numpy.global":
            raise M1RuntimeError("numpy global RNG marker mismatch")
        np.random.set_state(state)
        return
    if path == "rng.torch_cpu":
        torch = sys.modules.get("torch")
        if torch is None or marker != "torch.cpu":
            raise M1RuntimeError("torch CPU RNG is not available for restore")
        setter = getattr(torch, "set_rng_state", None)
        if not callable(setter):
            raise M1RuntimeError("loaded torch module does not expose set_rng_state")
        state_array = np.asarray(state, dtype=np.uint8).copy()
        tensor_factory = getattr(torch, "as_tensor", None)
        if callable(tensor_factory):
            setter(tensor_factory(state_array, dtype=getattr(torch, "uint8", None)))
        else:
            setter(state_array)
        return
    raise M1RuntimeError(f"unsupported global RNG path: {path}")


def _global_rng_audit() -> tuple[dict[str, Any], list[str]]:
    """Audit process-level RNGs without loading optional accelerator modules."""

    audit: dict[str, Any] = {
        "python_global": {
            "classification": SERIALIZED,
            "status": "loaded",
            "source": "python random module global state",
        },
        "numpy_global": {
            "classification": SERIALIZED,
            "status": "loaded",
            "source": "numpy.random legacy global RandomState",
        },
        "accelerators": {"cuda": "N/A", "mps": "N/A"},
    }
    errors: list[str] = []
    try:
        torch_state = _torch_cpu_rng_snapshot()
    except M1Error as exc:
        audit["torch_cpu"] = {
            "classification": "BLOCK",
            "status": "loaded_but_uninspectable",
            "error": str(exc),
        }
        errors.append(str(exc))
    else:
        if torch_state is None:
            audit["torch_cpu"] = {
                "classification": PROVEN_UNUSED,
                "status": "not loaded",
                "source": "torch is absent from sys.modules; no optional import performed",
            }
        else:
            audit["torch_cpu"] = {
                "classification": SERIALIZED,
                "status": "loaded",
                "source": "loaded torch CPU generator state",
            }
    return audit, errors


_SLEEPING_ISLAND_ARRAY_FIELDS = (
    "tree_asleep",
    "tree_awake",
    "body_awake",
    "tree_island",
    "dof_island",
    "island_ntree",
    "island_nv",
    "island_ne",
    "island_nf",
    "island_nefc",
)


def _sleeping_island_audit(data: Any) -> tuple[dict[str, Any], list[str]]:
    """Inspect MuJoCo sleep/island counters and reject unsupported state."""

    evidence: dict[str, Any] = {}
    blockers: list[str] = []

    def summarize(value: Any) -> tuple[dict[str, Any], np.ndarray | None]:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return {"type": type(value).__name__, "inspectable": False}, None
        if array.ndim == 0:
            try:
                scalar = array.item()
            except ValueError:
                scalar = None
            return {
                "type": type(value).__name__,
                "shape": [],
                "value": scalar,
                "inspectable": True,
            }, array
        numeric = array.dtype.kind in "biufc"
        summary: dict[str, Any] = {
            "type": type(value).__name__,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "inspectable": numeric,
        }
        if numeric and array.size:
            summary["min"] = float(np.min(array))
            summary["max"] = float(np.max(array))
            summary["nonzero_count"] = int(np.count_nonzero(array))
        return summary, array if numeric else None

    for name in ("ntree_awake", "nbody_awake", "nisland", "nidof"):
        if not hasattr(data, name):
            continue
        summary, array = summarize(getattr(data, name))
        evidence[name] = summary
        if array is None or not summary.get("inspectable"):
            blockers.append(f"MuJoCo sleeping/island field {name} is not inspectable")
            continue
        value = float(array.reshape(-1)[0]) if array.size else 0.0
        # ``nisland`` and ``nidof`` are ordinary topology counts, not sleep
        # indicators.  Likewise ntree/nbody awake counts may legitimately be
        # below the total when MuJoCo uses sentinel rows.  Record them as
        # evidence and gate only explicit awake/asleep markers below.
        if name == "ntree_awake" and hasattr(data, "ntree") and value < 0:
            blockers.append(f"MuJoCo sleeping state {name} is below ntree")
        elif name == "nbody_awake" and value < 0:
            blockers.append(f"MuJoCo sleeping state {name} is below nbody")

    for name in _SLEEPING_ISLAND_ARRAY_FIELDS:
        if not hasattr(data, name):
            continue
        summary, array = summarize(getattr(data, name))
        evidence[name] = summary
        if array is None or not summary.get("inspectable"):
            blockers.append(f"MuJoCo sleeping/island field {name} is not inspectable")
            continue
        values = array.reshape(-1)
        if name == "tree_asleep":
            # MuJoCo uses negative sentinels for trees that are not asleep;
            # non-negative entries identify an actually sleeping tree.
            active = bool(np.any(values >= 0))
        elif name == "tree_awake":
            active = bool(np.any(values == 0))
        elif name == "body_awake":
            # Body arrays may contain -1 for bodies outside a dynamic tree;
            # zero is the explicit asleep marker.
            active = bool(np.any(values == 0))
        else:
            # island IDs and per-island counts are topology/constraint
            # diagnostics, not sleeping state.  Never reject non-zero IDs.
            active = False
        if active:
            blockers.append(f"MuJoCo sleeping/island state {name} is nonzero or active")
    return evidence, blockers


def _restore_snapshot_value(value: Any) -> Any:
    """Decode tagged values emitted by :func:`_snapshot_value`."""

    if isinstance(value, Mapping):
        marker = value.get("__m1_type__")
        if marker == "mapping":
            items = value.get("items")
            if not isinstance(items, list):
                raise M1SchemaError("serialized mapping items must be a list")
            result: dict[str, Any] = {}
            for item in items:
                if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                    raise M1SchemaError("serialized mapping entries must be [string_key, value] pairs")
                if item[0] in result:
                    raise M1SchemaError(f"serialized mapping has duplicate key: {item[0]!r}")
                result[item[0]] = _restore_snapshot_value(item[1])
            return result
        if marker == "ndarray":
            try:
                array = np.asarray(value.get("data"), dtype=np.dtype(str(value.get("dtype"))))
                shape = tuple(int(item) for item in value.get("shape", ()))
                if tuple(array.shape) != shape:
                    raise ValueError(f"shape metadata {shape!r} does not match data {array.shape!r}")
                return np.ascontiguousarray(array).copy()
            except (TypeError, ValueError) as exc:
                raise M1SchemaError(f"invalid serialized ndarray metadata: {exc}") from exc
        if marker == "bytes":
            data = value.get("data")
            if not isinstance(data, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
                for item in data
            ):
                raise M1SchemaError("serialized bytes data must be a list of uint8 integers")
            return bytes(data)
        if marker == "tuple":
            items = value.get("items")
            if not isinstance(items, list):
                raise M1SchemaError("serialized tuple items must be a list")
            return tuple(_restore_snapshot_value(child) for child in items)
        if marker == "set":
            items = value.get("items")
            if not isinstance(items, list):
                raise M1SchemaError("serialized set items must be a list")
            return set(_restore_snapshot_value(child) for child in items)
        return {str(key): _restore_snapshot_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_restore_snapshot_value(child) for child in value]
    return copy.deepcopy(value)


def _observable_objects(inner: Any) -> Mapping[str, Any] | None:
    for name in ("observables", "_observables"):
        value = getattr(inner, name, None)
        if isinstance(value, Mapping):
            return value
    return None


def _discover_observable_state(inner: Any) -> tuple[dict[str, str], list[str], dict[str, Any]]:
    """Discover every Observable instance field and classify it fail-closed."""

    observables = _observable_objects(inner)
    if observables is None:
        return {}, [], {}
    classification: dict[str, str] = {}
    unknown: list[str] = []
    evidence: dict[str, Any] = {}
    for observable_name, observable in observables.items():
        prefix = f"observables.{observable_name}"
        fields = _field_values(observable)
        if not fields and not hasattr(observable, "__dict__") and not getattr(type(observable), "__slots__", ()):
            # A few lightweight diagnostic fakes expose an already-materialized
            # value instead of an Observable instance.  There are no mutable
            # timer/cache fields to audit in that representation; retain the
            # value as derived evidence while reserving unknown failures for
            # actual owner objects with unclassified instance fields.
            classification[prefix] = DERIVED_RECONSTRUCTED
            evidence[str(observable_name)] = {"value": _snapshot_value(observable)}
            continue
        evidence[str(observable_name)] = {
            str(field): _snapshot_value(value) for field, value in fields.items()
        }
        for field in fields:
            short_name = str(field).lstrip("_").lower()
            state_class = _OBSERVABLE_FIELD_CLASSES.get(short_name)
            if state_class is None:
                unknown.append(f"{prefix}.{field}")
            else:
                classification[f"{prefix}.{field}"] = state_class
    shared_cache = getattr(inner, "_obs_cache", getattr(inner, "obs_cache", None))
    if isinstance(shared_cache, Mapping):
        classification["observable_cache.__shared__"] = SERIALIZED
        evidence["__shared__"] = _snapshot_value(shared_cache)
    return classification, unknown, evidence


def _source_declared_instance_fields(value: Any) -> set[str]:
    """Return ``self.<field>`` names visible in the pinned owner source.

    This is deliberately a read-only evidence audit.  It avoids depending on
    private third-party class APIs and records source occurrences for review,
    but it never expands the explicit owner registry: a source-defined field
    remains unknown until a concrete classification entry is added.
    """

    value_type = type(value)
    cached = _SOURCE_FIELD_CACHE.get(value_type)
    if cached is not None:
        return set(cached)
    fields: set[str] = set()
    for cls in getattr(value_type, "__mro__", (value_type,)):
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        # A lightweight token scan is safer than executing or importing the
        # source.  It also catches assignments in reset/step methods, where
        # the selected path can mutate state after construction.
        import re

        fields.update(re.findall(r"\bself\.([A-Za-z_]\w*)", source))
    _SOURCE_FIELD_CACHE[value_type] = frozenset(fields)
    return fields


def _owner_field_classification(
    owner_kind: str,
    field_name: str,
    value: Any,
    *,
    source_declared: bool = False,
) -> str | None:
    """Classify one source-audited owner field.

    The exact logical four-class state map remains the public M1 contract;
    this owner-level map supplies a closed-loop audit over concrete fields.
    Handles are reported separately by ``_audit_runtime_owners``.
    """

    name = str(field_name)
    lower = name.lower().lstrip("_")
    if name in _RNG_FIELD_NAMES and owner_kind in {"inner", "outer", "robot", "task"}:
        if value is None:
            return PROVEN_UNUSED
        if _rng_kind(value) is not None:
            return SERIALIZED
        return None
    if owner_kind == "observable":
        return _OBSERVABLE_FIELD_CLASSES.get(lower)
    if owner_kind == "outer":
        if name in _OUTER_PROVEN_UNUSED_FIELDS:
            return PROVEN_UNUSED
        return IMMUTABLE if name in _OUTER_IMMUTABLE_FIELDS else None
    if owner_kind == "object":
        return IMMUTABLE if name in _OBJECT_IMMUTABLE_FIELDS else None
    if owner_kind == "controller":
        if name in _CONTROLLER_SERIALIZED_FIELDS:
            return SERIALIZED
        if name in _CONTROLLER_DERIVED_FIELDS:
            return DERIVED_RECONSTRUCTED
        if name in _CONTROLLER_IMMUTABLE_FIELDS:
            return IMMUTABLE
        return None
    if owner_kind == "interpolator":
        return SERIALIZED if name in _INTERPOLATOR_AUDITED_FIELDS else None
    if owner_kind == "gripper":
        if name in _GRIPPER_SERIALIZED_FIELDS:
            return SERIALIZED
        if name in _GRIPPER_IMMUTABLE_FIELDS:
            return IMMUTABLE
        return None
    if owner_kind == "robot":
        if name in _ROBOT_PROVEN_UNUSED_FIELDS:
            return PROVEN_UNUSED if _recent_buffer_proof(
                name,
                value,
                source_declared=source_declared,
            ) is not None else None
        if name in _ROBOT_DERIVED_FIELDS:
            return DERIVED_RECONSTRUCTED
        if name in _ROBOT_IMMUTABLE_FIELDS:
            return IMMUTABLE
        return None
    if owner_kind == "fixture":
        if name in _FIXTURE_DERIVED_FIELDS:
            return DERIVED_RECONSTRUCTED
        if name in _FIXTURE_IMMUTABLE_FIELDS:
            return IMMUTABLE
        return None
    if owner_kind == "task":
        if name in _TASK_SERIALIZED_FIELDS:
            return SERIALIZED
        if name in _TASK_IMMUTABLE_FIELDS:
            return IMMUTABLE
        return None
    # Environment counters and task state are the only mutable Python state
    # the selected Libero/robosuite path carries outside owners above.
    if owner_kind == "inner":
        if name in _INNER_DERIVED_FIELDS:
            # Object-state wrappers are reconstructed from the named MuJoCo
            # object/fixture poses and the guarded task-0 post-process.  Their
            # Python containers are traversed handles, not opaque serialized
            # values.
            return DERIVED_RECONSTRUCTED
        if name in _INNER_SERIALIZED_FIELDS:
            return SERIALIZED
        if name in _INNER_PROVEN_UNUSED_FIELDS:
            return PROVEN_UNUSED
        if name.startswith("recent_"):
            return PROVEN_UNUSED
        if name in _INNER_IMMUTABLE_FIELDS:
            return IMMUTABLE
        return None
    return None


def _state_behavior_evidence(state_class: str, owner_kind: str, field_name: str) -> dict[str, str]:
    """Attach an explicit operation/proof to every concrete state class."""

    if state_class == SERIALIZED:
        return {
            "operation": "generic serialized_runtime capture/restore with exact readback",
            "comparison": "EXACT",
        }
    if state_class == DERIVED_RECONSTRUCTED:
        if owner_kind == "controller":
            operation = "controller.update(force=True)"
        elif owner_kind == "observable":
            operation = "forced observable refresh (_update_observables(force=True))"
        elif owner_kind == "fixture":
            operation = "guarded task-0 fixture visual reconstruction (_post_process)"
        elif field_name in {"object_states_dict", "_object_states_dict", "tracking_object_states_change"}:
            operation = "named object/fixture pose reconstruction plus guarded _post_process"
        else:
            operation = "mj_forward plus invariant collector"
        return {"reconstruction_op": operation, "comparison": "FLOATING_PHYSICAL or DERIVED_DIAGNOSTIC"}
    if state_class == IMMUTABLE:
        return {
            "proof": "pinned source/config/layout audit; value is not written on the selected step path",
            "comparison": "EXACT",
        }
    if state_class == PROVEN_UNUSED:
        return {
            "source_evidence": (
                "pinned selected-step source audit found no read by physics, reward, "
                "controller, gripper, or invariant collector"
            ),
            "comparison": "not applicable; retained as audit evidence",
        }
    raise M1SchemaError(f"unknown state class for behavior evidence: {state_class}")


def _field_values(value: Any) -> Mapping[str, Any]:
    """Reflect instance fields without invoking arbitrary properties."""

    if isinstance(value, Mapping):
        return value
    try:
        fields = dict(vars(value))
    except (TypeError, AttributeError):
        fields = {}
    if not isinstance(fields, Mapping):
        fields = {}
    # A small wrapper may use ``__slots__`` instead of ``__dict__``.  Slots
    # are explicit instance fields and therefore part of the same closed-loop
    # audit; reading a slot does not invoke an arbitrary property descriptor.
    for cls in getattr(type(value), "__mro__", (type(value),)):
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if not isinstance(slots, Iterable):
            continue
        for field_name in slots:
            name = str(field_name)
            if name in {"__dict__", "__weakref__"} or name in fields:
                continue
            try:
                fields[name] = getattr(value, name)
            except (AttributeError, TypeError):
                continue
    return fields


def _object_joints_property_descriptor(value: Any) -> property | None:
    """Return only the fixed ``joints`` property descriptor, if present.

    The owner audit intentionally does not walk arbitrary descriptors.  The
    robosuite object registry is the sole property-backed exception because
    ``MujocoObject.joints`` exposes the immutable named-joint list used by the
    LIBERO object-state path.
    """

    for cls in getattr(type(value), "__mro__", (type(value),)):
        descriptor = getattr(cls, "__dict__", {}).get("joints")
        if descriptor is None:
            continue
        return descriptor if isinstance(descriptor, property) else None
    return None


def _audit_runtime_owners(inner: Any, env: Any) -> dict[str, Any]:
    """Reflect all selected owner roots and reject unknown mutable fields.

    ``owner_roots`` and ``handle_classification`` are intentionally included
    in the returned evidence.  This makes the reset-only audit inspectable and
    prevents a future implementation from accidentally narrowing the audit to
    the fields currently used by capture/restore.
    """

    roots: list[tuple[str, Any, str]] = [("inner", inner, "inner")]
    unsupported_interpolators: list[str] = []
    # The selected outer LiberoEnv owns the public ``step`` path and task
    # identity.  Audit it as a first-class owner even when the wrapped
    # robosuite task object has no ``task`` field of its own.
    if env is not inner:
        roots.append(("outer", env, "outer"))
    robots = getattr(inner, "robots", None)
    try:
        robot_values = list(robots) if robots is not None else []
    except Exception as exc:
        raise UnknownMutableStateError(f"robots handle is not traversable: {exc}") from exc
    for index, robot in enumerate(robot_values):
        robot_path = f"robot[{index}]"
        roots.append((robot_path, robot, "robot"))
        controller = getattr(robot, "controller", None)
        if controller is not None:
            controller_path = f"{robot_path}.controller"
            roots.append((controller_path, controller, "controller"))
            for interpolator_name in (
                "interpolator",
                "interpolation",
                "interpolator_pos",
                "interpolator_ori",
            ):
                if not hasattr(controller, interpolator_name):
                    continue
                interpolator = getattr(controller, interpolator_name)
                if interpolator is None or interpolator is False:
                    continue
                interpolator_path = f"{controller_path}.{interpolator_name}"
                if not _is_supported_linear_interpolator(interpolator):
                    unsupported_interpolators.append(
                        f"{interpolator_path} (unsupported or malformed nonempty interpolator)"
                    )
                else:
                    roots.append((interpolator_path, interpolator, "interpolator"))
        gripper = getattr(robot, "gripper", getattr(robot, "gripper_model", None))
        if gripper is not None:
            roots.append((f"{robot_path}.gripper", gripper, "gripper"))

    observables = _observable_objects(inner)
    roots.append(("observables", observables if observables is not None else {}, "handle"))
    for name, observable in (observables or {}).items():
        roots.append((f"observables.{name}", observable, "observable"))

    fixtures = getattr(inner, "fixtures_dict", getattr(inner, "_fixtures_dict", None))
    roots.append(("fixtures", fixtures if fixtures is not None else {}, "handle"))
    if isinstance(fixtures, Mapping):
        for name, fixture in fixtures.items():
            roots.append((f"fixtures.{name}", fixture, "fixture"))

    task = getattr(inner, "task", None)
    if task is not None:
        roots.append(("task", task, "task"))

    # ``objects_dict`` is an owner graph, not an opaque label container.  The
    # mapping itself is recorded as a traversed handle and every named
    # movable-object descriptor is audited as a child root.  This catches a
    # mutable field added to one object after construction.
    objects = getattr(inner, "objects_dict", getattr(inner, "_objects_dict", None))
    roots.append(("objects", objects if objects is not None else {}, "handle"))
    if isinstance(objects, Mapping):
        for name, obj in objects.items():
            roots.append((f"objects.{name}", obj, "object"))

    sim = getattr(inner, "sim", None)
    roots.append(("inner.sim", sim, "handle"))
    if sim is not None:
        roots.append(("inner.sim.model", getattr(sim, "model", None), "handle"))
        roots.append(("inner.sim.data", getattr(sim, "data", None), "handle"))

    unknown: list[str] = list(unsupported_interpolators)
    classification: dict[str, str] = {}
    handles: dict[str, str] = {}
    field_evidence: dict[str, Any] = {}
    for path, owner, owner_kind in roots:
        if owner is None:
            if owner_kind != "handle":
                unknown.append(path)
            continue
        # Mapping/list roots are traversed explicitly through their child
        # owner roots above.  Their containers are still classified as
        # traversed handles rather than silently ignored.
        if owner_kind == "handle":
            if path == "objects":
                if not isinstance(owner, Mapping):
                    unknown.append(f"{path} (named object mapping is not traversable)")
                else:
                    handles[path] = "traversed"
            else:
                handles[path] = "traversed" if path in {"observables", "fixtures", "inner.sim"} else "immutable_handle"
            continue
        fields = dict(_field_values(owner))
        property_registry_fields: set[str] = set()
        if owner_kind == "object" and "joints" not in fields:
            joints_descriptor = _object_joints_property_descriptor(owner)
            if joints_descriptor is not None:
                try:
                    fields["joints"] = joints_descriptor.__get__(owner, type(owner))
                    property_registry_fields.add("joints")
                except Exception as exc:
                    unknown.append(
                        f"{path}.joints (registered property could not be read: {exc})"
                    )
        declared = _source_declared_instance_fields(owner)
        if owner_kind == "inner":
            allowed_names = _INNER_AUDITED_FIELDS
        elif owner_kind == "outer":
            allowed_names = _OUTER_AUDITED_FIELDS
        elif owner_kind == "object":
            allowed_names = _OBJECT_AUDITED_FIELDS
        elif owner_kind == "robot":
            allowed_names = _ROBOT_AUDITED_FIELDS
        elif owner_kind == "controller":
            allowed_names = _CONTROLLER_AUDITED_FIELDS
        elif owner_kind == "interpolator":
            allowed_names = _INTERPOLATOR_AUDITED_FIELDS
        elif owner_kind == "gripper":
            allowed_names = _GRIPPER_AUDITED_FIELDS
        elif owner_kind == "fixture":
            allowed_names = _FIXTURE_AUDITED_FIELDS
        elif owner_kind == "task":
            allowed_names = _TASK_AUDITED_FIELDS
        elif owner_kind == "observable":
            allowed_names = {
                str(field_name)
                for field_name in fields
                if _OBSERVABLE_FIELD_CLASSES.get(str(field_name).lstrip("_").lower()) is not None
            }
        else:
            allowed_names = frozenset()
        for field_name, field_value in fields.items():
            field_path = f"{path}.{field_name}"
            if callable(field_value) and getattr(field_value, "_m1_operation_probe", False):
                classification[field_path] = IMMUTABLE
                field_evidence[field_path] = {
                    "class": IMMUTABLE,
                    "type": type(field_value).__name__,
                    "mutable_value": False,
                    "operation_probe": True,
                    "source_evidence": "RuntimeAdapter construction/post-construction operation probe",
                    **_state_behavior_evidence(IMMUTABLE, owner_kind, str(field_name)),
                }
                handles[field_path] = "immutable_handle"
                continue
            # Bound methods/callable hooks are behavior handles, not mutable
            # runtime state.  They still appear in ``vars`` when a test or
            # wrapper monkeypatches a hook (for example ``_post_process``),
            # so classify the handle explicitly before applying the mutable
            # field allowlist.
            if str(field_name) not in allowed_names:
                unknown.append(field_path)
                continue
            if callable(field_value):
                classification[field_path] = IMMUTABLE
                field_evidence[field_path] = {
                    "class": IMMUTABLE,
                    "type": type(field_value).__name__,
                    "mutable_value": False,
                    "source_declared": str(field_name) in declared
                    or str(field_name) in property_registry_fields,
                    **_state_behavior_evidence(IMMUTABLE, owner_kind, str(field_name)),
                }
                handles[field_path] = "immutable_handle"
                continue
            # A concrete source declaration is authoritative, but a field
            # with an object/container value still needs an explicit handle
            # classification or traversal.  The selected owner children are
            # traversed above; all other opaque handles fail closed.
            state_class = _owner_field_classification(
                owner_kind,
                str(field_name),
                field_value,
                source_declared=str(field_name) in declared,
            )
            if state_class is None:
                unknown.append(field_path)
                continue
            controller_proof: dict[str, str] | None = None
            if owner_kind == "controller":
                controller_proof = _controller_field_proof(
                    str(field_name),
                    field_value,
                )
                if str(field_name) in {
                    "use_delta",
                    "use_ori",
                    "uncoupling",
                    "impedance_mode",
                    "eef_name",
                    "control_freq",
                    "new_update",
                    "initial_joint",
                    "goal_pos",
                    "goal_ori",
                    "relative_ori",
                    "ori_ref",
                } and controller_proof is None:
                    unknown.append(
                        f"{field_path} (controller field has invalid type/shape)"
                    )
                    continue
            mutable_container = isinstance(field_value, (np.ndarray, list, dict, set, tuple))
            needs_immutable_proof = (
                state_class == IMMUTABLE
                and (
                    mutable_container
                    or (
                        owner_kind in {"fixture", "object"}
                        and str(field_name) == "duplicate_collision_geoms"
                    )
                )
            )
            immutable_proof: dict[str, str] | None = None
            if needs_immutable_proof:
                immutable_proof = _immutable_runtime_proof(
                    owner_kind,
                    str(field_name),
                    field_value,
                    source_declared=str(field_name) in declared,
                    owner_module=str(getattr(type(owner), "__module__", "")),
                )
                if immutable_proof is None:
                    unknown.append(
                        f"{field_path} (immutable mutable value lacks source/type/shape proof)"
                    )
                    continue
            classification[field_path] = state_class
            field_evidence[field_path] = {
                "class": state_class,
                "type": type(field_value).__name__,
                "source_declared": str(field_name) in declared
                or str(field_name) in property_registry_fields,
                "mutable_value": isinstance(field_value, (np.ndarray, list, dict, set, tuple))
                or isinstance(field_value, (bool, int, float, complex, np.generic)),
                **_state_behavior_evidence(state_class, owner_kind, str(field_name)),
            }
            if immutable_proof is not None:
                field_evidence[field_path].update(immutable_proof)
            if controller_proof is not None:
                field_evidence[field_path].update(controller_proof)
            if owner_kind == "object" and str(field_name) in property_registry_fields:
                field_evidence[field_path]["property_registry"] = "robosuite.object.joints"
            if owner_kind == "interpolator":
                field_evidence[field_path].update(
                    _interpolator_field_proof(str(field_name), field_value)
                )
            if owner_kind == "robot" and str(field_name).startswith("recent_"):
                recent_proof = _recent_buffer_proof(
                    str(field_name),
                    field_value,
                    source_declared=str(field_name) in declared,
                )
                if recent_proof is None:
                    unknown.append(
                        f"{field_path} (recent buffer lacks pinned type/shape/source proof)"
                    )
                    continue
                field_evidence[field_path].update(recent_proof)
            if str(field_name) in _RNG_FIELD_NAMES and owner_kind in {
                "inner",
                "outer",
                "robot",
                "task",
            }:
                field_evidence[field_path].update(
                    {
                        "expected_type": _rng_kind(field_value) or "None",
                        "expected_shape": "opaque RNG protocol state",
                        "invariant": "the selected-step RNG state is captured and restored exactly",
                        "source_evidence": "runtime-provided environment-local RNG object; no RNG type-name expansion",
                    }
                )
            if isinstance(field_value, (np.ndarray, list, dict, set, tuple)) or (
                field_value is not None
                and not isinstance(field_value, (str, bytes, bool, int, float, complex, np.generic))
                and not callable(field_value)
            ):
                if str(field_name) in {"sim", "robots", "_robots", "fixtures_dict", "_fixtures_dict", "objects_dict", "_objects_dict", "observables", "_observables", "_obs_cache", "obs_cache", "controller", "gripper", "gripper_model", "robot_model", "object_properties"}:
                    handles[field_path] = "traversed" if str(field_name) in {"sim", "robots", "_robots", "fixtures_dict", "_fixtures_dict", "objects_dict", "_objects_dict", "observables", "_observables"} else "immutable_handle"
                elif owner_kind == "controller" and str(field_name) in {"interpolator", "interpolator_pos", "interpolator_ori", "derr_buf"}:
                    handles[field_path] = "immutable_handle"
                elif str(field_name).startswith("recent_"):
                    handles[field_path] = "immutable_handle"
                elif isinstance(field_value, (list, dict, set, tuple, np.ndarray)):
                    # A mutable container that is not one of the explicitly
                    # traversed roots still needs an explicit handle record.
                    # SERIALIZED containers are captured by the generic path
                    # map; all source-audited immutable/derived containers are
                    # retained as explicit immutable handles or reconstruction
                    # inputs rather than being silently ignored.
                    handles[field_path] = (
                        "traversed"
                        if state_class == DERIVED_RECONSTRUCTED
                        else "immutable_handle"
                    )
                elif not callable(field_value):
                    # Non-container source-audited objects are immutable
                    # handles unless the owner classifier explicitly marks
                    # them serialized (which would require a value encoder).
                    if state_class == SERIALIZED and str(field_name) in _RNG_FIELD_NAMES:
                        handles[field_path] = "serialized_rng"
                    elif state_class in {IMMUTABLE, PROVEN_UNUSED, DERIVED_RECONSTRUCTED}:
                        handles[field_path] = (
                            "traversed"
                            if state_class == DERIVED_RECONSTRUCTED
                            else "immutable_handle"
                        )
                    else:
                        unknown.append(field_path)

        # Inspect the nested fixture ``object_properties`` object/mapping,
        # because _post_process is allowed to touch only its visual field.
        if owner_kind == "fixture" and hasattr(owner, "object_properties"):
            properties = getattr(owner, "object_properties")
            if properties is not None and not isinstance(properties, Mapping) and not hasattr(properties, "__dict__"):
                unknown.append(f"{path}.object_properties")

    # A standalone observable classifier also catches unknown fields in
    # objects that do not expose ``__dict__`` and records the canonical paths.
    _observable_classification, unknown_observables, _observable_evidence = _discover_observable_state(inner)
    unknown.extend(unknown_observables)
    classification.update(_observable_classification)
    for path, state_class in _observable_classification.items():
        field_evidence.setdefault(
            str(path),
            {
                "class": state_class,
                "type": "observable_field",
                "mutable_value": True,
                **_state_behavior_evidence(
                    state_class,
                    "observable",
                    str(path).rsplit(".", 1)[-1],
                ),
            },
        )
    global_rng_audit, global_rng_errors = _global_rng_audit()
    unknown.extend(global_rng_errors)
    global_rng_classification = {
        "rng.python_global": SERIALIZED,
        "rng.numpy_global": SERIALIZED,
    }
    if global_rng_audit.get("torch_cpu", {}).get("classification") == SERIALIZED:
        global_rng_classification["rng.torch_cpu"] = SERIALIZED
    elif global_rng_audit.get("torch_cpu", {}).get("classification") == PROVEN_UNUSED:
        global_rng_classification["rng.torch_cpu"] = PROVEN_UNUSED
    classification.update(global_rng_classification)
    field_evidence.update(
        {
            "rng.python_global": {
                "class": SERIALIZED,
                "type": "python.random global",
                "mutable_value": True,
                **_state_behavior_evidence(SERIALIZED, "inner", "rng.python_global"),
                "source_evidence": "python random.getstate()/setstate process-global protocol",
            },
            "rng.numpy_global": {
                "class": SERIALIZED,
                "type": "numpy.random global",
                "mutable_value": True,
                **_state_behavior_evidence(SERIALIZED, "inner", "rng.numpy_global"),
                "source_evidence": "numpy.random.get_state()/set_state process-global protocol",
            },
        }
    )
    torch_info = global_rng_audit.get("torch_cpu", {})
    field_evidence["rng.torch_cpu"] = {
        "class": torch_info.get("classification"),
        "type": "torch CPU RNG",
        "mutable_value": torch_info.get("classification") == SERIALIZED,
        "status": torch_info.get("status"),
        "source_evidence": torch_info.get("source", torch_info.get("error")),
    }
    env_local_rng = {
        str(path): {
            "classification": state_class,
            "type": field_evidence.get(str(path), {}).get("expected_type"),
        }
        for path, state_class in classification.items()
        if str(path).split(".")[-1] in _RNG_FIELD_NAMES
        and str(path).split(".")[0] in {"inner", "outer", "robot", "task"}
    }
    global_rng_audit["env_local"] = env_local_rng
    unknown = sorted(set(unknown))
    outer_identity = {
        str(path).split(".", 1)[1]: state_class
        for path, state_class in classification.items()
        if str(path).startswith("outer.")
    }
    return {
        "owner_roots": [path for path, _owner, _kind in roots],
        "handle_classification": handles,
        "field_classification": classification,
        "field_evidence": field_evidence,
        "outer_identity": outer_identity,
        "unknown_paths": unknown,
        "observable_count": len(observables) if observables is not None else 0,
        "rng": global_rng_audit,
    }


_MISSING_RUNTIME_PATH = object()


def _runtime_path_get(inner: Any, path: str, *, outer: Any | None = None) -> Any:
    """Resolve one audited owner path without evaluating arbitrary code."""

    import re

    if str(path) == "observable_cache.__shared__":
        return getattr(inner, "_obs_cache", getattr(inner, "obs_cache", _MISSING_RUNTIME_PATH))
    value: Any = inner
    outer = inner if outer is None else outer
    components = str(path).split(".")
    if not components:
        return _MISSING_RUNTIME_PATH
    first = components.pop(0)
    if first == "inner":
        value = inner
    elif first == "outer":
        value = outer
    else:
        match = re.fullmatch(r"robot\[(\d+)\]", first)
        if match:
            try:
                robots = list(getattr(inner, "robots"))
                value = robots[int(match.group(1))]
            except (AttributeError, IndexError, TypeError, ValueError):
                return _MISSING_RUNTIME_PATH
        elif first == "observables":
            value = _observable_objects(inner)
        elif first == "fixtures":
            value = getattr(inner, "fixtures_dict", getattr(inner, "_fixtures_dict", None))
        elif first == "task":
            value = getattr(inner, "task", None)
        elif first == "observable_cache":
            value = getattr(inner, "_obs_cache", getattr(inner, "obs_cache", None))
        else:
            return _MISSING_RUNTIME_PATH
    for component in components:
        if isinstance(value, Mapping):
            if component in value:
                value = value[component]
                continue
            # Owner names are serialized as strings in the path, while a
            # lightweight runtime may retain an equivalent non-string key.
            found = False
            for key, child in value.items():
                if str(key) == component:
                    value = child
                    found = True
                    break
            if not found:
                return _MISSING_RUNTIME_PATH
        else:
            try:
                value = getattr(value, component)
            except (AttributeError, TypeError):
                return _MISSING_RUNTIME_PATH
    return value


def _runtime_path_parent(
    inner: Any,
    path: str,
    *,
    outer: Any | None = None,
) -> tuple[Any, str] | None:
    components = str(path).split(".")
    if len(components) < 2:
        return None
    parent_path = ".".join(components[:-1])
    parent = _runtime_path_get(inner, parent_path, outer=outer)
    if parent is _MISSING_RUNTIME_PATH:
        return None
    return parent, components[-1]


def _runtime_path_set(
    inner: Any,
    path: str,
    value: Any,
    *,
    outer: Any | None = None,
) -> None:
    """Set one already-classified serialized owner field."""

    if str(path) in {"rng.python_global", "rng.numpy_global", "rng.torch_cpu"}:
        _restore_global_rng(str(path), value)
        return
    current_rng = _runtime_path_get(inner, path, outer=outer)
    if current_rng is not _MISSING_RUNTIME_PATH and _rng_kind(current_rng) is not None:
        if not isinstance(value, Mapping):
            raise M1SchemaError(f"serialized RNG path is not a mapping: {path}")
        _restore_rng_object(current_rng, value)
        return

    if str(path) == "observable_cache.__shared__":
        decoded = _restore_snapshot_value(value)
        current = getattr(inner, "_obs_cache", _MISSING_RUNTIME_PATH)
        field_name = "_obs_cache"
        if current is _MISSING_RUNTIME_PATH:
            current = getattr(inner, "obs_cache", _MISSING_RUNTIME_PATH)
            field_name = "obs_cache"
        if current is _MISSING_RUNTIME_PATH:
            raise UnknownMutableStateError(f"serialized runtime path is not reachable: {path}")
        try:
            setattr(inner, field_name, decoded)
        except Exception as exc:
            raise M1RuntimeError(f"serialized runtime field is not writable: {path}") from exc
        return
    parent_info = _runtime_path_parent(inner, path, outer=outer)
    if parent_info is None:
        raise UnknownMutableStateError(f"serialized runtime path is not reachable: {path}")
    parent, field_name = parent_info
    decoded = _restore_snapshot_value(value)
    if isinstance(parent, Mapping):
        # Mapping roots are traversed only when the concrete mapping is
        # mutable.  Refuse immutable mappings rather than silently changing a
        # detached copy and claiming successful restoration.
        try:
            parent[field_name] = decoded  # type: ignore[index]
        except Exception as exc:
            raise M1RuntimeError(f"serialized runtime mapping is not writable: {path}") from exc
        return
    try:
        current = getattr(parent, field_name)
    except (AttributeError, TypeError) as exc:
        raise UnknownMutableStateError(f"serialized runtime field is not readable: {path}") from exc
    # Backward-compatible envelopes may contain an untagged JSON list for an
    # ndarray.  Preserve the fresh runtime's dtype in that case.
    if isinstance(current, np.ndarray) and not isinstance(decoded, np.ndarray):
        decoded = np.asarray(decoded, dtype=current.dtype).copy()
    elif isinstance(current, tuple) and isinstance(decoded, list):
        decoded = tuple(decoded)
    elif isinstance(current, set) and isinstance(decoded, (list, tuple)):
        decoded = set(decoded)
    try:
        if isinstance(current, np.ndarray) and isinstance(decoded, np.ndarray):
            if current.shape != decoded.shape or current.dtype != decoded.dtype:
                raise M1RuntimeError(f"serialized runtime ndarray shape/dtype mismatch: {path}")
            current[...] = decoded
            return
        if isinstance(current, dict) and isinstance(decoded, Mapping):
            current.clear()
            current.update(decoded)
            return
        if isinstance(current, list) and isinstance(decoded, list):
            current[:] = decoded
            return
        if isinstance(current, set) and isinstance(decoded, set):
            current.clear()
            current.update(decoded)
            return
        setattr(parent, field_name, decoded)
    except Exception as exc:
        raise M1RuntimeError(f"serialized runtime field is not writable: {path}") from exc


def _serialized_runtime_snapshot(
    inner: Any,
    *,
    owner_audit: Mapping[str, Any] | None = None,
    outer: Any | None = None,
) -> dict[str, Any]:
    """Return every concrete owner field classified ``SERIALIZED``."""

    owner_audit = owner_audit or _audit_runtime_owners(inner, inner)
    unknown = owner_audit.get("unknown_paths", ())
    if unknown:
        raise UnknownMutableStateError(
            "unknown/unclassified mutable state: " + ", ".join(sorted(str(path) for path in unknown))
        )
    classification = owner_audit.get("field_classification", {})
    if not isinstance(classification, Mapping):
        raise M1SchemaError("runtime owner audit did not return field classification")
    values: dict[str, Any] = {}
    for path, state_class in sorted(classification.items(), key=lambda item: str(item[0])):
        if state_class != SERIALIZED:
            continue
        path_text = str(path)
        if path_text == "rng.python_global":
            values[path_text] = _python_global_rng_snapshot()
            continue
        if path_text == "rng.numpy_global":
            values[path_text] = _numpy_global_rng_snapshot()
            continue
        if path_text == "rng.torch_cpu":
            torch_state = _torch_cpu_rng_snapshot()
            if torch_state is None:
                raise UnknownMutableStateError("serialized torch CPU RNG path is not reachable")
            values[path_text] = torch_state
            continue
        current = _runtime_path_get(inner, path_text, outer=outer)
        if current is _MISSING_RUNTIME_PATH:
            raise UnknownMutableStateError(f"serialized runtime path is not reachable: {path}")
        if _rng_kind(current) is not None:
            values[path_text] = _snapshot_rng_object(current)
        else:
            values[path_text] = _snapshot_serialized_value(current)
    return values


def _is_observable_serialized_path(path: str) -> bool:
    return str(path).startswith("observables.") or str(path).startswith("observable_cache.")


def _serialized_classified_paths(runtime_audit: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the concrete SERIALIZED paths from one runtime audit."""

    classification = runtime_audit.get("field_classification", {})
    if not isinstance(classification, Mapping):
        raise M1SchemaError("runtime audit is missing concrete field classification")
    return tuple(
        sorted(
            str(path)
            for path, state_class in classification.items()
            if state_class == SERIALIZED
        )
    )


_REQUIRED_SERIALIZATION_TEMPLATE_CLASSES: dict[str, str] = {
    "mjdata.integration": SERIALIZED,
    "fixture.model.body_pos": SERIALIZED,
    "fixture.model.body_quat": SERIALIZED,
    "env.timestep": SERIALIZED,
    "env.cur_time": SERIALIZED,
    "env.done": SERIALIZED,
    "controller": SERIALIZED,
    "gripper.current_action": SERIALIZED,
    "identity": SERIALIZED,
    "observables": SERIALIZED,
    "objects.joints": DERIVED_RECONSTRUCTED,
}
_REQUIRED_SERIALIZATION_TEMPLATES = frozenset(_REQUIRED_SERIALIZATION_TEMPLATE_CLASSES)
_REQUIRED_INTEGRATION_TEMPLATE_SEAMS = {
    "capture_artifact": "ReplayState.integration_state",
    "persistence_artifact": "capture_<10|50|100>.npy + capture metadata",
    "restore_operation": "mj_setState -> exact mj_getState readback",
    "readback_comparison": "exact integration-state array comparison before mj_forward",
}


def _template_expansions(concrete: Mapping[str, Any]) -> dict[str, list[str]]:
    """Compute the exact concrete paths owned by each source design template."""

    concrete_paths = {str(path) for path in concrete}
    def owner_paths(owner: str, suffix: str | None = None) -> list[str]:
        result = []
        for path in concrete_paths:
            parts = path.split(".")
            if owner not in parts:
                continue
            if suffix is not None and not path.endswith(f".{suffix}"):
                continue
            result.append(path)
        return sorted(result)

    return {
        "mjdata.integration": [],
        "fixture.model.body_pos": [],
        "fixture.model.body_quat": [],
        "env.timestep": sorted(
            path for path in concrete_paths if path.rsplit(".", 1)[-1] == "timestep"
        ),
        "env.cur_time": sorted(
            path for path in concrete_paths if path.rsplit(".", 1)[-1] == "cur_time"
        ),
        "env.done": sorted(
            path for path in concrete_paths if path.rsplit(".", 1)[-1] == "done"
        ),
        "controller": owner_paths("controller"),
        "gripper.current_action": owner_paths("gripper", "current_action"),
        "identity": [],
        "observables": sorted(
            path
            for path in concrete_paths
            if _is_observable_serialized_path(path)
        ),
        "objects.joints": [],
    }


def _serialization_coverage(
    classification: Mapping[str, Any], *, observable_count: int | None = None
) -> dict[str, Any]:
    """Expand design categories into one concrete serialization contract.

    The frozen YAML categories are useful source-design templates, but they
    are not runtime paths.  Gate D therefore consumes only ``concrete`` rows,
    each of which names the capture, persistence, restore, and readback seam
    for one path emitted by ``serialized_runtime``.
    """

    concrete: dict[str, dict[str, str]] = {}
    for path, state_class in sorted(classification.items(), key=lambda item: str(item[0])):
        if state_class != SERIALIZED:
            continue
        path_text = str(path)
        concrete[path_text] = {
            "class": SERIALIZED,
            "capture_artifact": f"ReplayState.serialized_runtime[{path_text!r}]",
            "persistence_artifact": (
                f"capture_<10|50|100>.json.metadata.serialized_runtime[{path_text!r}]"
            ),
            "restore_operation": f"_runtime_path_set({path_text!r}) before refresh",
            "readback_comparison": (
                f"_serialized_runtime_snapshot exact readback[{path_text!r}]"
            ),
        }

    def template(
        state_class: str,
        expanded: Iterable[str],
        capture: str,
        persistence: str,
        restore: str,
        readback: str,
    ) -> dict[str, Any]:
        return {
            "class": state_class,
            "runtime_classified": False,
            "expanded_concrete_paths": sorted(str(path) for path in expanded),
            "capture_artifact": capture,
            "persistence_artifact": persistence,
            "restore_operation": restore,
            "readback_comparison": readback,
        }

    expansions = _template_expansions(concrete)
    controller_paths = expansions["controller"]
    gripper_paths = expansions["gripper.current_action"]
    observable_paths = expansions["observables"]
    templates = {
        "mjdata.integration": template(
            SERIALIZED,
            (),
            "ReplayState.integration_state",
            "capture_<10|50|100>.npy + capture metadata",
            "mj_setState -> exact mj_getState readback",
            "exact integration-state array comparison before mj_forward",
        ),
        "fixture.model.body_pos": template(
            SERIALIZED,
            (),
            "ReplayState.fixture.body_pos",
            "capture_<10|50|100>.json.metadata.fixture.body_pos",
            "_restore_fixture model.body_pos",
            "floating physical fixture body-pos comparison",
        ),
        "fixture.model.body_quat": template(
            SERIALIZED,
            (),
            "ReplayState.fixture.body_quat",
            "capture_<10|50|100>.json.metadata.fixture.body_quat",
            "_restore_fixture model.body_quat",
            "floating physical fixture body-quat comparison",
        ),
        "env.timestep": template(
            SERIALIZED,
            expansions["env.timestep"],
            "ReplayState.counters.timestep",
            "capture_<10|50|100>.json.metadata.counters.timestep",
            "_restore_counters",
            "exact counter comparison",
        ),
        "env.cur_time": template(
            SERIALIZED,
            expansions["env.cur_time"],
            "ReplayState.counters.cur_time",
            "capture_<10|50|100>.json.metadata.counters.cur_time",
            "_restore_counters",
            "exact counter comparison",
        ),
        "env.done": template(
            SERIALIZED,
            expansions["env.done"],
            "ReplayState.counters.done",
            "capture_<10|50|100>.json.metadata.counters.done",
            "_restore_counters",
            "exact counter comparison",
        ),
        "controller": template(
            SERIALIZED,
            controller_paths,
            "ReplayState.controller + serialized_runtime controller paths",
            "capture_<10|50|100>.json.metadata.controller and serialized_runtime",
            "controller.update(force=True) then _restore_controller",
            "exact serialized readback + physical controller cache comparison",
        ),
        "gripper.current_action": template(
            SERIALIZED,
            gripper_paths,
            "ReplayState.gripper.current_action + serialized_runtime",
            "capture_<10|50|100>.json.metadata.gripper and serialized_runtime",
            "_restore_gripper",
            "exact gripper action and physical-state comparison",
        ),
        "identity": template(
            SERIALIZED,
            (),
            "ReplayState.identity",
            "capture_<10|50|100>.json.metadata.identity",
            "_validate_restore_identity before mutable writes",
            "exact task/instruction/layout/tape identity comparison",
        ),
        "observables": template(
            SERIALIZED,
            observable_paths,
            "ReplayState.serialized_runtime observable paths",
            "capture_<10|50|100>.json.metadata.serialized_runtime",
            "_restore_serialized_runtime + forced observable refresh",
            "exact observable timer/cache readback",
        ),
        "objects.joints": template(
            DERIVED_RECONSTRUCTED,
            (),
            "ReplayState-derived invariant objects.<name>.joints",
            "reference_<10|50|100>.json invariants.objects.<name>.joints",
            "mj_forward object qpos/qvel address slices",
            "floating physical qpos/qvel arrays with width schema",
        ),
    }
    return {
        "source_design_templates": templates,
        "concrete": concrete,
        "observable_count": (
            int(observable_count)
            if observable_count is not None
            else len({
                path.split(".", 2)[1]
                for path in observable_paths
                if path.startswith("observables.") and len(path.split(".", 2)) >= 2
            })
        ),
    }


def _serialization_coverage_contract_pass(
    coverage: Any,
    classified: Iterable[str],
    *,
    require_official_observables: bool = False,
) -> bool:
    """Check exact concrete/expanded serialization coverage for Gate D."""

    if not isinstance(coverage, Mapping):
        return False
    concrete = coverage.get("concrete")
    templates = coverage.get("source_design_templates")
    if not isinstance(concrete, Mapping) or not isinstance(templates, Mapping):
        return False
    if {str(name) for name in templates} != _REQUIRED_SERIALIZATION_TEMPLATES:
        return False
    classified_paths = {str(path) for path in classified}
    concrete_paths = {str(path) for path in concrete}
    if concrete_paths != classified_paths:
        return False
    required_fields = (
        "capture_artifact",
        "persistence_artifact",
        "restore_operation",
        "readback_comparison",
    )
    if any(
        not isinstance(row, Mapping)
        or row.get("class") != SERIALIZED
        or any(not row.get(field_name) for field_name in required_fields)
        for row in concrete.values()
    ):
        return False
    expected_expansions = _template_expansions(concrete)
    for template_name, expected_class in _REQUIRED_SERIALIZATION_TEMPLATE_CLASSES.items():
        row = templates.get(template_name)
        if not isinstance(row, Mapping):
            return False
        if row.get("class") != expected_class or row.get("runtime_classified") is not False:
            return False
        if template_name == "mjdata.integration":
            for field_name, expected in _REQUIRED_INTEGRATION_TEMPLATE_SEAMS.items():
                if row.get(field_name) != expected:
                    return False
        expanded = row.get("expanded_concrete_paths", ())
        if isinstance(expanded, (str, bytes)):
            return False
        try:
            if sorted(str(path) for path in expanded) != expected_expansions[template_name]:
                return False
        except TypeError:
            return False
        if any(not row.get(field_name) for field_name in required_fields):
            return False

    observable_paths = set(expected_expansions["observables"])
    observable_names = {
        path.split(".", 2)[1]
        for path in observable_paths
        if path.startswith("observables.") and len(path.split(".", 2)) >= 2
    }
    try:
        observed_count = int(coverage.get("observable_count"))
    except (TypeError, ValueError):
        return False
    if observable_paths:
        if not observable_names or len(observable_names) != 31:
            return False
        if observed_count != len(observable_names):
            return False
    elif observed_count != 0:
        return False
    if require_official_observables and (
        observed_count != 31 or len(observable_names) != 31 or not observable_paths
    ):
        return False
    return True


def _restore_serialized_runtime(
    inner: Any,
    serialized_runtime: Mapping[str, Any],
    *,
    only_observables: bool = False,
    owner_audit: Mapping[str, Any] | None = None,
    outer: Any | None = None,
) -> dict[str, Any]:
    """Restore serialized owner fields and require exact concrete readback."""

    if not isinstance(serialized_runtime, Mapping):
        raise M1SchemaError("serialized_runtime must be a mapping")
    expected_paths = {
        str(path): value
        for path, value in serialized_runtime.items()
        if not only_observables or _is_observable_serialized_path(str(path))
    }
    owner_audit = owner_audit or _audit_runtime_owners(inner, inner)
    full_classification = owner_audit.get("field_classification", {})
    if not isinstance(full_classification, Mapping):
        raise M1SchemaError("runtime owner audit did not return field classification")
    serialized_paths = {
        str(path)
        for path, state_class in full_classification.items()
        if state_class == SERIALIZED
    }
    if only_observables:
        serialized_paths = {path for path in serialized_paths if _is_observable_serialized_path(path)}
    if set(expected_paths) != serialized_paths:
        missing = sorted(serialized_paths - set(expected_paths))
        extra = sorted(set(expected_paths) - serialized_paths)
        raise M1RuntimeError(
            "serialized runtime path set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    for path, value in expected_paths.items():
        _runtime_path_set(inner, path, value, outer=outer)
    readback = _serialized_runtime_snapshot(inner, owner_audit=owner_audit, outer=outer)
    readback_selected = {
        path: value
        for path, value in readback.items()
        if not only_observables or _is_observable_serialized_path(path)
    }
    if set(readback_selected) != set(expected_paths) or not _exact_equal(readback_selected, expected_paths):
        raise M1RuntimeError("serialized runtime exact readback mismatch")
    return copy.deepcopy(readback_selected)


def _observable_snapshot(inner: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    observables = _observable_objects(inner)
    if observables is None:
        return {}, {}
    values: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    for name, observable in observables.items():
        if hasattr(observable, "__dict__"):
            try:
                value = getattr(observable, "obs")
            except Exception:
                value = getattr(observable, "_current_observed_value", None)
            fields = vars(observable)
        else:
            value = observable
            fields = {}
        values[str(name)] = _snapshot_value(value)
        cache[str(name)] = {
            str(field): _snapshot_value(value)
            for field, value in fields.items()
            if str(field).lstrip("_").lower()
            in {"time_since_last_sample", "timer", "current_delay", "delay", "sampled", "cache", "obs_cache"}
        }
    shared_cache = getattr(inner, "_obs_cache", getattr(inner, "obs_cache", None))
    if isinstance(shared_cache, Mapping):
        cache["__shared__"] = _snapshot_value(shared_cache)
    return values, cache


REQUIRED_INVARIANT_PATHS = (
    "integration",
    "fixture.body_pos",
    "fixture.body_quat",
    "objects",
    "controller",
    "eef_pose.pos",
    "eef_pose.orientation",
    "eef_pose.linear_velocity",
    "eef_pose.angular_velocity",
    "controller_kinematic_caches.ee_pos",
    "controller_kinematic_caches.ee_ori_mat",
    "controller_kinematic_caches.ee_pos_vel",
    "controller_kinematic_caches.ee_ori_vel",
    "controller_kinematic_caches.J_full",
    "controller_kinematic_caches.J_pos",
    "controller_kinematic_caches.J_ori",
    "controller_kinematic_caches.mass_matrix",
    "gripper.current_action",
    "gripper_physical",
    "contacts",
    "constraints.efc_force",
    "constraints.efc_type",
    "constraints.efc_id",
    "constraints.efc_state",
    "observables",
    "observable_cache",
    "serialized_runtime",
    "counters.timestep",
    "counters.cur_time",
    "counters.done",
)


def validate_invariant_snapshot(
    snapshot: Mapping[str, Any],
    *,
    required_paths: Iterable[str] = REQUIRED_INVARIANT_PATHS,
) -> dict[str, Any]:
    """Require every primary physical/diagnostic invariant before comparison."""

    if not isinstance(snapshot, Mapping):
        raise M1SchemaError("invariant snapshot must be a mapping")
    missing: list[str] = []
    for path in required_paths:
        current: Any = snapshot
        present = True
        for component in str(path).split("."):
            if not isinstance(current, Mapping) or component not in current:
                present = False
                break
            current = current[component]
        if not present or current is None:
            missing.append(str(path))
    # ``objects`` is a required named-state seam, not just a presence bit.
    # Require each movable object's body pose and every named joint's qpos and
    # qvel so a collector cannot publish an object entry that is structurally
    # present while silently omitting the state needed for replay validation.
    objects = snapshot.get("objects")
    if isinstance(objects, Mapping):
        if not objects:
            missing.append("objects")
        for object_name, object_state in objects.items():
            object_path = f"objects.{object_name}"
            if not isinstance(object_state, Mapping):
                missing.append(object_path)
                continue
            for field_name in ("body_pos", "body_quat", "joints"):
                if field_name not in object_state or object_state[field_name] is None:
                    missing.append(f"{object_path}.{field_name}")
            joints = object_state.get("joints")
            if isinstance(joints, Mapping):
                for joint_name, joint_state in joints.items():
                    joint_path = f"{object_path}.joints.{joint_name}"
                    if not isinstance(joint_state, Mapping):
                        missing.append(joint_path)
                        continue
                    for field_name in ("qpos", "qvel"):
                        if field_name not in joint_state or joint_state[field_name] is None:
                            missing.append(f"{joint_path}.{field_name}")
                    if "qpos_width" not in joint_state:
                        missing.append(f"{joint_path}.qpos_width")
                    if "qvel_width" not in joint_state:
                        missing.append(f"{joint_path}.qvel_width")
                    if "qpos" in joint_state and "qpos_width" in joint_state:
                        try:
                            if len(np.asarray(joint_state["qpos"]).reshape(-1)) != int(
                                joint_state["qpos_width"]
                            ):
                                missing.append(f"{joint_path}.qpos_width")
                        except (TypeError, ValueError):
                            missing.append(f"{joint_path}.qpos_width")
                    if "qvel" in joint_state and "qvel_width" in joint_state:
                        try:
                            if len(np.asarray(joint_state["qvel"]).reshape(-1)) != int(
                                joint_state["qvel_width"]
                            ):
                                missing.append(f"{joint_path}.qvel_width")
                        except (TypeError, ValueError):
                            missing.append(f"{joint_path}.qvel_width")
                    try:
                        width_pair = (int(joint_state["qpos_width"]), int(joint_state["qvel_width"]))
                        if width_pair not in set(_MUJOCO_JOINT_WIDTHS.values()):
                            missing.append(f"{joint_path}.joint_type_width")
                    except (KeyError, TypeError, ValueError):
                        missing.append(f"{joint_path}.joint_type_width")
    if missing:
        raise M1SchemaError("required invariant fields are missing: " + ", ".join(sorted(missing)))
    return dict(snapshot)


def replay_state_record(state: ReplayState) -> MatchedStateRecord:
    metadata = {
        "schema_version": int(state.schema_version),
        "fixture": state.fixture,
        "counters": state.counters,
        "controller": state.controller,
        "gripper": state.gripper,
        "identity": state.identity,
        "reward": state.reward,
        "success": state.success,
        "serialized_runtime": state.serialized_runtime,
    }
    if state.schema_version >= REPLAY_STATE_SCHEMA_VERSION:
        metadata.update(
            {
                "python_state": state.python_state,
                "rng": state.rng,
                "physics_model_fingerprint": state.physics_model_fingerprint,
                "observation_model_fingerprint": state.observation_model_fingerprint,
                "fixture_physics_fingerprint": state.fixture_physics_fingerprint,
                "raw_observation": _snapshot_serialized_value(state.raw_observation)
                if state.raw_observation is not None
                else None,
                "task_state": state.task_state,
                "tape_provenance": state.tape_provenance,
            }
        )
    return MatchedStateRecord(metadata=metadata, payload=state.integration_state)


def replay_state_from_record(record: MatchedStateRecord) -> ReplayState:
    metadata = dict(record.metadata)
    schema_version = metadata.get("schema_version", 1)
    if schema_version not in (1, REPLAY_STATE_SCHEMA_VERSION):
        raise M1SchemaError(f"unsupported ReplayState schema version: {schema_version!r}")
    # v1 records predate the split Python/RNG/model/observation/task/tape
    # fields.  Keep their explicit schema marker and leave the new fields
    # empty rather than manufacturing provenance that was never captured.
    return ReplayState(
        integration_state=record.payload,
        fixture=metadata.get("fixture", {}),
        counters=metadata.get("counters", {}),
        controller=metadata.get("controller", {}),
        gripper=metadata.get("gripper", {}),
        identity=metadata.get("identity", {}),
        reward=metadata.get("reward"),
        success=metadata.get("success"),
        serialized_runtime=metadata.get("serialized_runtime", {}),
        python_state=metadata.get("python_state", {}),
        rng=metadata.get("rng", {}),
        physics_model_fingerprint=metadata.get("physics_model_fingerprint", {}),
        observation_model_fingerprint=metadata.get("observation_model_fingerprint", {}),
        fixture_physics_fingerprint=metadata.get("fixture_physics_fingerprint", {}),
        raw_observation=(
            _restore_snapshot_value(metadata.get("raw_observation"))
            if metadata.get("raw_observation") is not None
            else None
        ),
        task_state=metadata.get("task_state", {}),
        tape_provenance=metadata.get("tape_provenance", {}),
        schema_version=int(schema_version),
    )


def _effective_serialized_runtime(state: ReplayState) -> dict[str, Any]:
    """Use v2 split Python/RNG fields when an older caller omitted the aggregate."""

    result = dict(state.serialized_runtime)
    if state.schema_version >= REPLAY_STATE_SCHEMA_VERSION:
        for source in (state.python_state, state.rng):
            for path, value in source.items():
                result.setdefault(str(path), copy.deepcopy(value))
    return result


def _predicate_snapshot(inner: Any) -> dict[str, Any]:
    """Evaluate every concrete LIBERO goal predicate as exact discrete state."""

    parsed = getattr(inner, "parsed_problem", None)
    if not isinstance(parsed, Mapping):
        return {"available": False, "goals": []}
    goals = parsed.get("goal_state")
    if not isinstance(goals, Sequence) or isinstance(goals, (str, bytes)):
        raise M1RuntimeError("parsed LIBERO task has no inspectable goal_state")
    evaluator = getattr(inner, "_eval_predicate", None)
    if goals and not callable(evaluator):
        raise M1RuntimeError("parsed LIBERO task does not expose _eval_predicate")
    rows: list[dict[str, Any]] = []
    for index, goal in enumerate(goals):
        try:
            value = evaluator(goal) if callable(evaluator) else False
        except Exception as exc:
            raise M1RuntimeError(f"goal predicate {index} could not be evaluated: {exc}") from exc
        if not isinstance(value, (bool, np.bool_)):
            raise M1RuntimeError(f"goal predicate {index} did not return bool")
        rows.append(
            {
                "index": index,
                # Persist predicate expressions in the canonical JSON-like
                # sequence form used by the exact-state contract.  LIBERO
                # commonly exposes goal tuples, while the artifact schema
                # intentionally represents them as lists.
                "expression": list(goal) if isinstance(goal, tuple) else _snapshot_value(goal),
                "value": bool(value),
            }
        )
    return {"available": True, "goals": rows}


class RuntimeAdapter:
    """Project-owned adapter around one official synchronous LeRobot env.

    The adapter never calls ``reset``, ``set_init_state``, or dummy settling
    during :meth:`restore`.  A fresh official reset is construction-only and is
    performed by :meth:`construct_fresh` before any restore attempt.
    """

    def __init__(
        self,
        runtime: Mapping[str, Any] | Any,
        *,
        config: Mapping[str, Any] | None = None,
        tape_hash: str | None = None,
        mujoco_module: Any | None = None,
        selected_env: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = dict(config or {})
        self.tape_hash = tape_hash
        if isinstance(runtime, Mapping):
            self.vector_env = runtime.get("env")
            self.mujoco = mujoco_module or runtime.get("mujoco")
        else:
            self.vector_env = runtime
            self.mujoco = mujoco_module
        # ``selected_env`` is supplied by ``construct_fresh`` only after the
        # official reset.  Direct construction remains useful for callers
        # that already own an initialized environment.
        self.env = selected_env if selected_env is not None else _unwrap_official_env(self.vector_env)
        self.inner = _find_inner_env(self.env)
        self.sim = getattr(self.env, "sim", getattr(self.inner, "sim", None))
        if self.sim is None:
            raise M1RuntimeError("official runtime adapter could not locate the simulator")
        self.model = getattr(self.sim, "model", None)
        self.data = getattr(self.sim, "data", None)
        if self.model is None or self.data is None:
            raise M1RuntimeError("official runtime adapter requires simulator model and data")
        # LIBERO reset performs ten official settle steps.  Keep that
        # underlying robosuite cursor separate from the experiment cursor:
        # M1 boundaries count only actions issued after reset.
        self.reset_underlying_timestep = int(getattr(self.inner, "timestep", 0))
        self.reset_underlying_cur_time = float(getattr(self.inner, "cur_time", 0.0))
        self.experiment_step_count = 0
        self.step_count = 0
        self.reset_provenance: dict[str, Any] = {}
        self.last_reward: float | None = None
        self.last_info: Any = None
        self.last_raw_observation: Any = None
        self.last_terminated = False
        self.last_truncated = False
        self._construction_reset_done = False
        self._closed = False
        self._restore_in_progress = False
        self.rgb_records: dict[str, dict[str, Any]] = {}
        self.gl_identity: dict[str, str] | None = None
        self.post_process_proof: dict[str, Any] = {}
        # These counters are owned by the adapter, so a zero is an observed
        # call-site fact rather than a synthetic assertion made by a caller.
        # Construction provenance snapshots them before any restore/capture
        # operation can occur.
        self.post_construction_operation_counts: dict[str, int] = {
            name: 0 for name in POST_CONSTRUCTION_FORBIDDEN_OPERATIONS
        }
        self._post_construction_counter_baseline: dict[str, int | None] = {}
        self._operation_probe: dict[str, Any] | None = None
        self.state_classification = validate_state_classification(DEFAULT_STATE_CLASSIFICATION)
        self.runtime_audit: dict[str, Any] = {}
        configured_allowlist = self.config.get("post_process_guard", {}).get("allowlist") if isinstance(
            self.config.get("post_process_guard"), Mapping
        ) else None
        self.post_process_allowlist = frozenset(
            str(item) for item in (configured_allowlist or POST_PROCESS_ALLOWLIST)
        )
        if not self.post_process_allowlist <= POST_PROCESS_ALLOWLIST:
            raise M1ConfigError(
                f"post-process allowlist contains unsupported fields: "
                f"{sorted(self.post_process_allowlist - POST_PROCESS_ALLOWLIST)}"
            )
        configured_reachable = self.config.get("reachable_mutable_fields")
        if configured_reachable is not None:
            self.audit_state_classification(configured_reachable)

    def audit_state_classification(
        self,
        reachable_mutable_fields: Iterable[str],
        classification: Mapping[str, Any] = DEFAULT_STATE_CLASSIFICATION,
    ) -> dict[str, str]:
        """Require every runtime-audited mutable field to be classified."""

        self.state_classification = audit_mutable_state(
            reachable_mutable_fields, classification=classification
        )
        return dict(self.state_classification)

    # Alias kept explicit for callers that name this operation a runtime audit.
    def audit_runtime_state(
        self,
        reachable_mutable_fields: Iterable[str] | None = None,
        classification: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | dict[str, str]:
        if reachable_mutable_fields is None:
            return self._runtime_audit(classification=classification)
        return self.audit_state_classification(
            reachable_mutable_fields,
            classification=classification or DEFAULT_STATE_CLASSIFICATION,
        )

    def _owner_audit_for_serialized(self, *, refresh: bool = False) -> Mapping[str, Any]:
        """Return the closed-loop owner audit used by snapshot seams.

        Construction records the audit for the per-step hot path.  Capture
        boundaries refresh it so a field added during a source step cannot be
        silently omitted from the next persisted state record.
        """

        if not refresh and self.runtime_audit.get("field_classification"):
            return self.runtime_audit
        audit = _audit_runtime_owners(self.inner, self.env)
        if audit.get("unknown_paths"):
            raise UnknownMutableStateError(
                "unknown/unclassified mutable state: "
                + ", ".join(sorted(str(path) for path in audit["unknown_paths"]))
            )
        if refresh and self.runtime_audit:
            # Preserve the broader runtime-audit evidence while refreshing
            # only the concrete owner classification used by the capture.
            concrete = dict(audit.get("field_classification", {}))
            self.runtime_audit["field_classification"] = copy.deepcopy(concrete)
            self.runtime_audit["classification"] = copy.deepcopy(concrete)
            self.runtime_audit["classification_evidence"] = copy.deepcopy(
                audit.get("field_evidence", {})
            )
            self.runtime_audit["field_evidence"] = copy.deepcopy(
                audit.get("field_evidence", {})
            )
            self.runtime_audit["serialized_paths"] = sorted(
                str(path)
                for path, state_class in concrete.items()
                if state_class == SERIALIZED
            )
            self.runtime_audit["serialization_coverage"] = _serialization_coverage(
                concrete,
                observable_count=int(audit.get("observable_count", 0)),
            )
            self.runtime_audit["outer_identity"] = copy.deepcopy(
                audit.get("outer_identity", {})
            )
            self.runtime_audit["rng"] = copy.deepcopy(audit.get("rng", {}))
            self.runtime_audit["unknown_paths"] = []
        return audit

    def _runtime_audit(
        self,
        *,
        classification: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Audit the required runtime seam on every freshly built instance.

        A structurally empty fake is intentionally rejected just like an
        incomplete official runtime.  The audit is conservative: missing
        required fields, enabled plugins/sleeping, unsupported controller
        modes, and unclassified reachable state all block construction.
        """

        errors: list[str] = []
        owner_audit = _audit_runtime_owners(self.inner, self.env)
        if owner_audit.get("unknown_paths"):
            raise UnknownMutableStateError(
                "unknown/unclassified mutable state: "
                + ", ".join(str(path) for path in owner_audit["unknown_paths"])
            )
        observable_classification, unknown_observable_fields, observable_evidence = _discover_observable_state(
            self.inner
        )
        observables = _observable_objects(self.inner)
        if self.config.get("strict_provenance") and (
            observables is None or len(observables) != 31
        ):
            errors.append(
                "strict M1 runtime must expose exactly 31 observables, "
                f"got {0 if observables is None else len(observables)}"
            )
        configured_runtime = self.config.get("runtime")
        if self.config.get("strict_provenance") and isinstance(configured_runtime, Mapping):
            if configured_runtime.get("controller_class") != "OperationalSpaceController":
                errors.append("runtime controller_class is not OperationalSpaceController")
            if configured_runtime.get("gripper_class") != "PandaGripper":
                errors.append("runtime gripper_class is not PandaGripper")
            if configured_runtime.get("sleeping") is not False:
                errors.append("runtime sleeping must be disabled")
        controller = _get_controller(self.inner)
        if controller is None:
            errors.append("OperationalSpaceController is missing")
        else:
            if self.config.get("strict_provenance") and type(controller).__name__ != "OperationalSpaceController":
                errors.append(
                    "controller class must be OperationalSpaceController, "
                    f"got {type(controller).__name__}"
                )
            required_controller = (
                "initial_joint",
                "goal_pos",
                "goal_ori",
                "new_update",
                "impedance_mode",
                "use_delta",
                "update",
            )
            missing = [name for name in required_controller if not hasattr(controller, name)]
            if missing:
                errors.append(f"controller fields missing: {missing}")
            if callable(getattr(controller, "update", None)) is False:
                errors.append("controller.update is not callable")
            if hasattr(controller, "impedance_mode") and getattr(controller, "impedance_mode") != "fixed":
                errors.append("controller impedance_mode must be fixed")
            if hasattr(controller, "use_delta") and getattr(controller, "use_delta") is not True:
                errors.append("controller use_delta must be true")

        gripper = _get_gripper(self.inner)
        if gripper is None:
            errors.append("PandaGripper is missing")
        elif not hasattr(gripper, "current_action"):
            errors.append("gripper current_action is missing")
        elif self.config.get("strict_provenance") and getattr(gripper, "current_action", None) is None:
            errors.append("gripper current_action is not initialized")
        if self.config.get("strict_provenance") and gripper is not None and type(gripper).__name__ != "PandaGripper":
            errors.append(f"gripper class must be PandaGripper, got {type(gripper).__name__}")

        fixture_roots = _fixture_root_names(self.inner)
        if not fixture_roots:
            errors.append("fixture roots are empty")
        if not hasattr(self.inner, "timestep") or not hasattr(self.inner, "cur_time"):
            errors.append("environment counters timestep/cur_time are missing")
        if not hasattr(self.inner, "done") and not hasattr(self.env, "done"):
            errors.append("environment done counter is missing")
        if self.config.get("strict_provenance") or self.config.get("authoritative") is True:
            objects = getattr(self.inner, "objects_dict", None)
            if not isinstance(objects, Mapping) or not objects:
                errors.append("named movable objects are missing")
            else:
                for object_name, obj in objects.items():
                    if not getattr(obj, "root_body", None):
                        errors.append(f"object {object_name} has no named root body")
                    joints = getattr(obj, "joints", None)
                    if not joints:
                        errors.append(f"object {object_name} has no named joints")

        model = _raw_mujoco_handle(self.model, "_model")
        plugin_counts: dict[str, int | str] = {}
        for plugin_field in ("nplugin", "npluginstate"):
            raw_plugin_count = getattr(model, plugin_field, None)
            if raw_plugin_count is None:
                plugin_counts[plugin_field] = "MISSING"
                if self.config.get("strict_provenance"):
                    errors.append(f"MuJoCo {plugin_field} is required and missing")
                continue
            if isinstance(raw_plugin_count, bool):
                plugin_counts[plugin_field] = "UNINSPECTABLE"
                errors.append(f"MuJoCo {plugin_field} must be an integer, not bool")
                continue
            try:
                plugin_value = int(raw_plugin_count)
            except (TypeError, ValueError):
                plugin_counts[plugin_field] = "UNINSPECTABLE"
                errors.append(f"MuJoCo {plugin_field} is not inspectable")
                continue
            plugin_counts[plugin_field] = plugin_value
            if plugin_value != 0:
                errors.append(f"MuJoCo {plugin_field} is nonzero: {plugin_value}")
        inspected_plugin_counts = [
            value for value in plugin_counts.values() if isinstance(value, int)
        ]
        if self.config.get("strict_provenance") and set(plugin_counts) != {"nplugin", "npluginstate"}:
            errors.append("strict MuJoCo plugin audit requires nplugin and npluginstate")
        if not inspected_plugin_counts:
            errors.append("MuJoCo plugin count is not inspectable")
            plugin_count = 0
        else:
            plugin_count = inspected_plugin_counts[0]
        sleeping_values: list[tuple[str, Any]] = []
        for owner in (self.env, self.inner, model):
            for name in ("sleeping", "enable_sleeping", "mj_enable_sleep"):
                if hasattr(owner, name):
                    sleeping_values.append((name, getattr(owner, name)))
        # MuJoCo 3.7 exposes sleeping through the model option bit rather
        # than an environment ``sleeping`` attribute.  Read that raw option
        # so strict construction proves the actual simulator setting.
        model_options = getattr(model, "opt", None)
        enable_flags = getattr(model_options, "enableflags", None)
        sleep_enum = getattr(getattr(self.mujoco, "mjtEnableBit", None), "mjENBL_SLEEP", None)
        if enable_flags is not None and sleep_enum is not None:
            try:
                sleeping_values.append(
                    ("model.opt.enableflags.mjENBL_SLEEP", bool(int(enable_flags) & int(sleep_enum)))
                )
            except (TypeError, ValueError):
                errors.append("MuJoCo sleeping enable flag is not inspectable")
        elif self.config.get("strict_provenance"):
            errors.append("MuJoCo sleeping enable flag is not inspectable")
        if any(bool(value) for _name, value in sleeping_values):
            errors.append("MuJoCo sleeping must be disabled")
        sleeping_island_evidence, sleeping_island_errors = _sleeping_island_audit(self.data)
        errors.extend(sleeping_island_errors)

        # Observable modifiers are allowed only when deterministic.  The
        # official stack normally exposes none; if a test/runtime exposes a
        # mapping, reject obvious random/noise modifiers and non-callable
        # placeholders rather than silently treating them as harmless.
        modifiers_observed = bool(observables)
        modifier_evidence: dict[str, Any] = {}
        modifiers_deterministic = True if observables else None
        for observable_name, observable in (observables or {}).items():
            fields: dict[str, Any] = {}
            for field_name in ("_sensor", "_corrupter", "_filter", "_delayer"):
                modifier = getattr(observable, field_name, None)
                if not callable(modifier):
                    errors.append(f"observable {observable_name} {field_name} is not callable")
                    modifiers_deterministic = False
                    fields[field_name] = {"callable": False}
                    continue
                code = getattr(modifier, "__code__", None)
                names = tuple(str(name).lower() for name in getattr(code, "co_names", ()))
                module_name = str(getattr(modifier, "__module__", "")).lower()
                nondeterministic = any(
                    token in module_name or token in names
                    for token in ("random", "normal", "uniform", "choice", "rand")
                )
                fields[field_name] = {
                    "callable": True,
                    "qualname": getattr(modifier, "__qualname__", type(modifier).__name__),
                    "module": getattr(modifier, "__module__", None),
                    "nondeterministic_symbol": nondeterministic,
                }
                if nondeterministic:
                    modifiers_deterministic = False
                    errors.append(
                        f"observable {observable_name} {field_name} is not deterministic"
                    )
            modifier_evidence[str(observable_name)] = fields
        for owner in (self.env, self.inner):
            modifiers = getattr(owner, "observable_modifiers", None)
            if modifiers is None:
                continue
            modifiers_observed = True
            if not isinstance(modifiers, Mapping):
                errors.append("observable modifiers are not inspectable")
                continue
            nondeterministic = [
                str(name) for name, value in modifiers.items()
                if any(token in str(name).lower() for token in ("random", "noise", "stochastic"))
                or bool(getattr(value, "random", False))
            ]
            if nondeterministic:
                errors.append(f"observable modifiers are not deterministic: {nondeterministic}")
                modifiers_deterministic = False

        configured_audit = self.config.get("state_audit")
        reachable: tuple[str, ...]
        if isinstance(configured_audit, Mapping) and "reachable_fields" in configured_audit:
            reachable = tuple(str(field) for field in configured_audit["reachable_fields"])
            audit_classification = configured_audit.get("classification", classification or DEFAULT_STATE_CLASSIFICATION)
        else:
            configured_reachable = self.config.get("reachable_mutable_fields")
            reachable = tuple(str(field) for field in configured_reachable) if configured_reachable is not None else tuple(
                sorted({field for fields in DEFAULT_STATE_CLASSIFICATION.values() for field in fields})
            )
            audit_classification = classification or DEFAULT_STATE_CLASSIFICATION
        try:
            flattened = validate_state_classification(
                audit_classification,
                reachable_mutable_fields=reachable,
            )
        except M1Error as exc:
            errors.append(str(exc))
            flattened = {}
        concrete_owner_classification = owner_audit.get("field_classification", {})
        if not isinstance(concrete_owner_classification, Mapping):
            concrete_owner_classification = {}
        # The config-level map is a source-design template only.  Runtime
        # classification is the concrete owner map produced by the closed
        # loop audit; abstract entries must not masquerade as captured paths.
        concrete_classification = dict(concrete_owner_classification)
        concrete_classification.update(observable_classification)
        classification_templates = dict(flattened)
        classification_template_evidence: dict[str, Any] = {}
        for path, state_class in classification_templates.items():
            path_text = str(path)
            components = path_text.split(".")
            owner_kind = {
                "controller": "controller",
                "observables": "observable",
                "observable_cache": "observable",
                "gripper": "gripper",
                "fixture": "fixture",
                "task": "task",
                "env": "inner",
                "inner": "inner",
                "robot": "robot",
            }.get(components[0], "inner")
            classification_template_evidence[path_text] = {
                "class": state_class,
                **_state_behavior_evidence(
                    state_class,
                    owner_kind,
                    components[-1],
                ),
            }
        serialization_coverage = _serialization_coverage(
            concrete_classification,
            observable_count=len(observables) if observables is not None else 0,
        )
        audited_reachable = tuple(
            sorted(set(reachable) | set(concrete_classification))
        )
        if self.config.get("strict_provenance") and not errors:
            try:
                required_snapshot = self.collect_invariants()
            except Exception as exc:
                errors.append(f"required invariant collection failed: {exc}")
            else:
                try:
                    validate_invariant_snapshot(required_snapshot)
                except M1SchemaError as exc:
                    errors.append(str(exc))
                required_roots = (
                    "integration",
                    "fixture",
                    "objects",
                    "eef_pose",
                    "controller_kinematic_caches",
                    "gripper",
                    "gripper_physical",
                    "contacts",
                )
                missing_roots = [name for name in required_roots if name not in required_snapshot]
                if missing_roots:
                    errors.append(f"required invariants are missing: {missing_roots}")
                fixture = required_snapshot.get("fixture")
                if not isinstance(fixture, Mapping) or not fixture.get("body_pos") or not fixture.get("body_quat"):
                    errors.append("named fixture model poses are missing")
                objects_snapshot = required_snapshot.get("objects")
                if not isinstance(objects_snapshot, Mapping) or not objects_snapshot:
                    errors.append("named object poses/joints are missing")
                elif any(
                    not isinstance(value, Mapping)
                    or "body_pos" not in value
                    or "body_quat" not in value
                    or not value.get("joints")
                    for value in objects_snapshot.values()
                ):
                    errors.append("named object pose/quaternion or joint qpos/qvel is missing")
                if not isinstance(required_snapshot.get("gripper_physical"), Mapping) or not required_snapshot["gripper_physical"]:
                    errors.append("physical gripper state is missing")
        if not self.post_process_proof:
            if self.config.get("strict_provenance"):
                try:
                    self._prove_post_process()
                except Exception as exc:
                    self.post_process_proof = {"observed": True, "passed": False, "error": str(exc)}
            else:
                # Test-only injected runtimes must not be reported as having
                # an authoritative reset proof.  Their normal guard still
                # runs on restore, but construction does not perturb event
                # order with an extra post-process call.
                self.post_process_proof = {
                    "observed": False,
                    "passed": True,
                    "authoritative": False,
                    "reason": "test-only runtime",
                }
        if self.config.get("strict_provenance") and not self.post_process_proof.get("passed", False):
            errors.append("guarded _post_process proof did not pass")
        if unknown_observable_fields:
            raise UnknownMutableStateError(
                "unknown/unclassified observable mutable state: "
                + ", ".join(sorted(unknown_observable_fields))
            )
        if errors:
            raise M1RuntimeError("required runtime audit failed: " + "; ".join(errors))
        self.state_classification = dict(concrete_classification)
        self.runtime_audit = {
            "owner_roots": list(owner_audit.get("owner_roots", ())),
            "handle_classification": copy.deepcopy(owner_audit.get("handle_classification", {})),
            "field_classification": copy.deepcopy(owner_audit.get("field_classification", {})),
            "field_evidence": copy.deepcopy(owner_audit.get("field_evidence", {})),
            "outer_identity": copy.deepcopy(owner_audit.get("outer_identity", {})),
            # Concrete evidence is kept separate from source-design template
            # evidence so an abstract SERIALIZED category cannot be mistaken
            # for a runtime-classified path.
            "classification_evidence": copy.deepcopy(owner_audit.get("field_evidence", {})),
            "classification_template_evidence": classification_template_evidence,
            "classification_templates": classification_templates,
            "unknown_paths": list(owner_audit.get("unknown_paths", ())),
            "controller_class": type(controller).__name__ if controller is not None else None,
            "controller_fields": sorted(
                name for name in ("initial_joint", "goal_pos", "goal_ori", "new_update", "impedance_mode", "use_delta")
                if controller is not None and hasattr(controller, name)
            ),
            "gripper_class": type(gripper).__name__ if gripper is not None else None,
            "fixture_roots": list(fixture_roots),
            "plugin_count": int(plugin_count),
            "plugin_counts": copy.deepcopy(plugin_counts),
            "sleeping": {
                "observed": bool(sleeping_values),
                "enabled": any(bool(value) for _name, value in sleeping_values)
                if sleeping_values
                else None,
                "island_state": copy.deepcopy(sleeping_island_evidence),
            },
            "rng": copy.deepcopy(owner_audit.get("rng", {})),
            "observable_modifiers": {
                "observed": modifiers_observed,
                "deterministic": modifiers_deterministic,
                "fields": modifier_evidence,
            },
            "observables": {
                "count": len(observables) if observables is not None else 0,
                "fields": observable_evidence,
                "classification": dict(observable_classification),
            },
            "reachable_fields": list(audited_reachable),
            "classification": dict(concrete_classification),
            "serialized_paths": list(
                sorted(
                    path for path, state_class in concrete_classification.items()
                    if state_class == SERIALIZED
                )
            ),
            "serialization_coverage": serialization_coverage,
        }
        return copy.deepcopy(self.runtime_audit)

    @classmethod
    def construct_fresh(
        cls,
        config: Mapping[str, Any],
        *,
        runtime_builder: Callable[[Mapping[str, Any]], Mapping[str, Any] | Any] | None = None,
        tape_hash: str | None = None,
        mujoco_module: Any | None = None,
    ) -> "RuntimeAdapter":
        builder = runtime_builder or build_official_runtime
        runtime = builder(config)
        vector_env = runtime.get("env") if isinstance(runtime, Mapping) else runtime
        selected_env: Any | None = None
        adapter: RuntimeAdapter | None = None
        try:
            # Keep child selection inside the cleanup guard: a malformed
            # vector (for example, more than one child) is still an owned
            # runtime that must be closed before construction fails.
            selected_env = _unwrap_official_env(vector_env)
            # Bind no lazy simulator/model/data attributes until this one
            # official reset has completed.  ``RuntimeAdapter.__init__`` does
            # not reset and therefore cannot accidentally perform a second
            # construction reset.
            task = _mapping(config.get("task", {}), "task")
            requested_init_state = int(task.get("init_state_id", 0))
            # LeRobot selects the current cursor before reset and advances it
            # by one after the selected initial state is applied.  Capture the
            # pre-reset selection before invoking the one seeded reset.
            selected_pre_reset = int(getattr(selected_env, "init_state_id", requested_init_state))
            construction_counter_before = {
                name: cls._read_construction_counter(selected_env, name)
                for name in ("reset_calls", "set_init_state_calls", "settle_calls")
            }
            operation_probe = cls._install_operation_probes(selected_env)
            reset_result = cls._construction_reset_target(selected_env, config)
            adapter = cls(
                runtime,
                config=config,
                tape_hash=tape_hash,
                mujoco_module=mujoco_module,
                selected_env=selected_env,
            )
            adapter._construction_reset_done = True
            operation_probe["adapter"] = adapter
            operation_probe["phase"] = "post_construction"
            adapter._operation_probe = operation_probe
            observed_post_cursor = getattr(selected_env, "init_state_id", None)
            expected_post_cursor = selected_pre_reset + 1
            if observed_post_cursor is not None:
                try:
                    observed_post_cursor = int(observed_post_cursor)
                except (TypeError, ValueError) as exc:
                    raise M1RuntimeError("post-reset init-state cursor is not an integer") from exc
            # Minimal test-only doubles do not implement LeRobot's cursor
            # increment.  Their evidence still records the official expected
            # cursor; the strict official path must observe the increment.
            if config.get("strict_provenance") and observed_post_cursor != expected_post_cursor:
                raise M1RuntimeError(
                    "post-reset init-state cursor mismatch: "
                    f"expected {expected_post_cursor}, got {observed_post_cursor}"
                )
            adapter.reset_provenance = {
                "requested_init_state_id": requested_init_state,
                "requested_seed": task.get("seed"),
                "post_reset_init_state_id": getattr(
                    selected_env,
                    "init_state_id",
                    getattr(adapter.inner, "init_state_id", None),
                ),
                "post_reset_timestep": int(getattr(adapter.inner, "timestep", 0)),
                "post_reset_cur_time": float(getattr(adapter.inner, "cur_time", 0.0)),
                "init_state_id_evidence": {
                    "requested": requested_init_state,
                    "selected_pre_reset": selected_pre_reset,
                    "post_reset_cursor": expected_post_cursor
                    if observed_post_cursor is None or observed_post_cursor == selected_pre_reset
                    else observed_post_cursor,
                },
                "construction": cls._construction_provenance(
                    selected_env,
                    reset_result=reset_result,
                    counter_before=construction_counter_before,
                    post_construction_counts=adapter.post_construction_operation_counts,
                    construction_operation_counts=operation_probe["construction_counts"],
                    operation_probe=operation_probe,
                ),
            }
            adapter._post_construction_counter_baseline = {
                name: cls._read_construction_counter(selected_env, name)
                for name in ("reset_calls", "set_init_state_calls", "settle_calls")
            }
            adapter._runtime_audit()
            return adapter
        except Exception as primary:
            # A reset, simulator binding, or audit failure must not leak the
            # freshly built official child.  Once the adapter exists it owns
            # the close policy; before that, close the vector owner when it
            # exposes close and otherwise close the selected child.
            try:
                if adapter is not None:
                    adapter.close()
                else:
                    owner = vector_env if callable(getattr(vector_env, "close", None)) else selected_env
                    close = getattr(owner, "close", None)
                    if callable(close):
                        close()
            except Exception as cleanup:
                primary.add_note(
                    f"fresh environment cleanup failed: {type(cleanup).__name__}: {cleanup}"
                )
            raise

    @staticmethod
    def _read_construction_counter(target: Any, name: str) -> int | None:
        value = getattr(target, name, None)
        if isinstance(value, (bool, np.bool_)):
            return None
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return integer if integer >= 0 else None

    @staticmethod
    def _operation_probe_targets(target: Any) -> tuple[Any, ...]:
        """Return the selected wrapper and its official nested env owners."""

        result: list[Any] = []
        pending = [target]
        seen: set[int] = set()
        while pending and len(result) < 8:
            owner = pending.pop(0)
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            result.append(owner)
            for name in ("_env", "env", "unwrapped", "inner"):
                child = getattr(owner, name, None)
                if child is not None and not callable(child):
                    pending.append(child)
        return tuple(result)

    @classmethod
    def _install_operation_probes(cls, target: Any) -> dict[str, Any]:
        """Trace reset/init/settle calls across the official wrapper chain."""

        holder: dict[str, Any] = {
            "phase": "construction",
            "construction_counts": {name: 0 for name in ("reset", "set_init_state", "settle")},
            "installed": [],
        }

        def observe(name: str) -> None:
            if holder["phase"] == "construction":
                holder["construction_counts"][name] += 1
                return
            adapter = holder.get("adapter")
            if adapter is not None:
                adapter._record_post_construction_operation(name)

        for owner in cls._operation_probe_targets(target):
            for name in ("reset", "set_init_state", "settle"):
                original = getattr(owner, name, None)
                if not callable(original):
                    continue

                @wraps(original)
                def wrapped(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any) -> Any:
                    observe(_name)
                    return _original(*args, **kwargs)

                setattr(wrapped, "_m1_operation_probe", True)
                try:
                    setattr(owner, name, wrapped)
                except (AttributeError, TypeError):
                    continue
                holder["installed"].append(
                    {"owner": type(owner).__name__, "operation": name}
                )
        return holder

    @classmethod
    def _construction_operation_evidence(
        cls,
        target: Any,
        name: str,
        *,
        before: int | None,
        allowed: bool,
    ) -> dict[str, Any]:
        after = cls._read_construction_counter(target, name)
        if before is not None and after is not None and after >= before:
            return {
                "allowed": allowed,
                "observed": True,
                "count": after - before,
                "source": f"{type(target).__name__}.{name}",
            }
        return {
            "allowed": allowed,
            "observed": False,
            "count": None,
            "source": f"{type(target).__name__}.{name}:counter_unavailable",
        }

    @classmethod
    def _construction_provenance(
        cls,
        target: Any,
        *,
        reset_result: Any,
        counter_before: Mapping[str, int | None],
        post_construction_counts: Mapping[str, Any],
        construction_operation_counts: Mapping[str, Any] | None = None,
        operation_probe: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        forbidden: dict[str, dict[str, Any]] = {}
        for name in POST_CONSTRUCTION_FORBIDDEN_OPERATIONS:
            value = post_construction_counts.get(name)
            observed = (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, np.integer))
                and int(value) >= 0
            )
            forbidden[name] = {
                "allowed": False,
                "observed": observed,
                "count": int(value) if observed else None,
                "source": "RuntimeAdapter.post_construction_operation_counts",
            }
        probe_counts = construction_operation_counts or {}

        def construction_evidence(name: str, counter_name: str) -> dict[str, Any]:
            probe_value = probe_counts.get(name)
            if isinstance(probe_value, (int, np.integer)) and not isinstance(probe_value, (bool, np.bool_)):
                return {
                    "allowed": True,
                    "observed": True,
                    "count": int(probe_value),
                    "source": f"RuntimeAdapter.construct_fresh.operation_probe.{name}",
                }
            return cls._construction_operation_evidence(
                target,
                counter_name,
                before=counter_before.get(counter_name),
                allowed=True,
            )

        reset_evidence = construction_evidence("reset", "reset_calls")
        # The adapter invoked exactly one seeded reset even when a minimal
        # official wrapper does not expose a public reset counter.  The call
        # site itself is authoritative in that case; later post-construction
        # calls are still fail-closed unless a counter or adapter hook records
        # them.
        if reset_evidence.get("count") is None:
            reset_evidence = {
                "allowed": True,
                "observed": True,
                "count": 1,
                "source": "RuntimeAdapter.construct_fresh._construction_reset_target",
            }
        return {
            "phase": "construction",
            "reset_result_type": type(reset_result).__name__,
            "operation_probe": {
                "source": "RuntimeAdapter.construct_fresh.operation_probe",
                "installed": sorted(
                    f"{item.get('owner')}:{item.get('operation')}"
                    for item in (operation_probe or {}).get("installed", ())
                    if isinstance(item, Mapping)
                ),
                "construction_counts": {
                    str(name): int(value)
                    for name, value in probe_counts.items()
                    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                },
            },
            "operations": {
                "reset": reset_evidence,
                "set_init_state": construction_evidence("set_init_state", "set_init_state_calls"),
                "settle": construction_evidence("settle", "settle_calls"),
            },
            "post_construction_forbidden": forbidden,
        }

    def _record_post_construction_operation(self, name: str) -> None:
        if name not in self.post_construction_operation_counts:
            raise M1RuntimeError(f"unknown post-construction operation: {name}")
        self.post_construction_operation_counts[name] += 1

    def reset_provenance_snapshot(self) -> dict[str, Any]:
        """Return construction evidence plus live post-construction deltas.

        ``reset``/``set_init_state``/``settle`` are legitimate only during
        official construction.  Their underlying runtime counters therefore
        need a post-reset baseline; publishing the raw totals would falsely
        classify construction work as forbidden activity.
        """

        provenance = copy.deepcopy(self.reset_provenance)
        construction = provenance.get("construction") if isinstance(provenance, Mapping) else None
        if not isinstance(construction, dict):
            return provenance if isinstance(provenance, dict) else {}
        forbidden = construction.get("post_construction_forbidden")
        if not isinstance(forbidden, dict):
            forbidden = {}
            construction["post_construction_forbidden"] = forbidden
        counter_names = {
            "reset": "reset_calls",
            "set_init_state": "set_init_state_calls",
            "settle": "settle_calls",
        }
        for name in POST_CONSTRUCTION_FORBIDDEN_OPERATIONS:
            tracked = self.post_construction_operation_counts.get(name, 0)
            count: int | None = int(tracked) if isinstance(tracked, (int, np.integer)) else None
            source = "RuntimeAdapter.post_construction_operation_counts"
            counter_name = counter_names.get(name)
            baseline = self._post_construction_counter_baseline.get(counter_name) if counter_name else None
            if counter_name is not None:
                current = self._read_construction_counter(self.env, counter_name)
                if baseline is not None and current is not None and current >= baseline:
                    delta = int(current - baseline)
                    count = max(int(count or 0), delta)
                    source = f"{type(self.env).__name__}.{counter_name}-post_construction_baseline"
                elif count is None:
                    count = 0
            if count is None or count < 0:
                count = 0
            forbidden[name] = {
                "allowed": False,
                "observed": True,
                "count": int(count),
                "source": source,
            }
        construction["phase"] = "construction"
        return provenance

    @staticmethod
    def _construction_reset_target(target: Any, config: Mapping[str, Any]) -> Any:
        """Reset one selected official child exactly once."""

        reset = getattr(target, "reset", None)
        if not callable(reset):
            raise M1RuntimeError("fresh official environment does not expose reset")
        seed = _mapping(config.get("task", {}), "task").get("seed", 2027)
        try:
            signature = inspect.signature(reset)
        except (TypeError, ValueError) as exc:
            raise M1RuntimeError("cannot validate official reset signature before reset") from exc
        try:
            signature.bind(seed=int(seed))
        except TypeError:
            try:
                signature.bind(int(seed))
            except TypeError as exc:
                raise M1RuntimeError(
                    "official reset must accept the configured seed; refusing unseeded retry"
                ) from exc
            result = reset(int(seed))
        else:
            result = reset(seed=int(seed))
        return result

    def _construction_reset(self) -> Any:
        """Compatibility helper that refuses a second construction reset."""

        if getattr(self, "_construction_reset_done", False):
            raise M1RuntimeError("construction reset was already performed")
        result = self._construction_reset_target(self.env, self.config)
        self.step_count = 0
        self.experiment_step_count = 0
        self.last_reward = None
        self.last_info = None
        self.last_raw_observation = None
        self.last_terminated = False
        self.last_truncated = False
        self._construction_reset_done = True
        return result

    def _identity(self, source_step: int) -> dict[str, Any]:
        return _identity_state(
            self.env,
            self.inner,
            self.tape_hash,
            source_step,
            mujoco_module=self.mujoco,
            config=self.config,
        )

    def _read_raw_observation_no_update(self) -> Any:
        """Read h=0 observation state without advancing an Observable."""

        getter = getattr(self.inner, "_get_observations", None)
        if not callable(getter):
            getter = getattr(self.env, "_get_observations", None)
        if not callable(getter):
            raise M1RuntimeError("runtime does not expose inner._get_observations(force_update=False)")
        try:
            raw = getter(force_update=False)
        except TypeError as exc:
            raise M1RuntimeError(
                "inner._get_observations must support force_update=False for ReplayState v2"
            ) from exc
        formatter = getattr(self.env, "_format_raw_obs", None)
        if not callable(formatter):
            formatter = getattr(self.inner, "_format_raw_obs", None)
        if callable(formatter):
            try:
                raw = formatter(raw)
            except Exception as exc:
                raise M1RuntimeError(f"raw observation formatting failed: {exc}") from exc
        return raw

    def capture(self, *, source_step: int | None = None) -> ReplayState:
        self._record_post_construction_operation("capture")
        if source_step is None:
            source_step = self.step_count
        owner_audit = self._owner_audit_for_serialized(refresh=True)
        serialized_runtime = _serialized_runtime_snapshot(
            self.inner,
            owner_audit=owner_audit,
            outer=self.env,
        )
        python_state = {
            str(path): value
            for path, value in serialized_runtime.items()
            if not str(path).startswith("rng.")
        }
        rng_state = {
            str(path): value
            for path, value in serialized_runtime.items()
            if str(path).startswith("rng.")
        }
        raw_observation = None
        # Official Libero's step return is authoritative when available.  A
        # direct capture on an initialized v2 runtime may have no return yet;
        # use the no-update path in that case without forcing observable
        # refresh.  Tiny v1 fakes have neither seam and retain their legacy
        # restore behavior.
        if callable(getattr(self.inner, "_get_observations", None)) or callable(
            getattr(self.env, "_get_observations", None)
        ):
            raw_observation = self._read_raw_observation_no_update()
        fixture = _fixture_snapshot(self.inner)
        return ReplayState(
            integration_state=_integration_state(self.mujoco, self.model, self.data),
            fixture=fixture,
            counters={
                **_counter_state(self.inner, self.env),
                "reset_underlying_timestep": int(self.reset_underlying_timestep),
                "reset_underlying_cur_time": float(self.reset_underlying_cur_time),
                "experiment_step_count": int(self.experiment_step_count),
            },
            controller=_controller_state(_get_controller(self.inner)),
            gripper=_gripper_state(_get_gripper(self.inner)),
            identity=self._identity(source_step),
            reward=self.last_reward,
            success=self._check_success(),
            serialized_runtime=serialized_runtime,
            python_state=python_state,
            rng=rng_state,
            physics_model_fingerprint=physics_model_fingerprint(self.model, self.mujoco),
            observation_model_fingerprint=observation_model_fingerprint(
                self.model,
                _renderer_config_for_runtime(self.env, self.inner, self.config),
                self.mujoco,
            ),
            fixture_physics_fingerprint=fixture_physics_fingerprint(fixture),
            raw_observation=raw_observation,
            task_state=_task_state_snapshot(self.env, self.inner, int(source_step)),
            tape_provenance={
                "sha256": self.tape_hash,
                "source_step": int(source_step),
                "action_dim": ACTION_DIM,
            },
            schema_version=REPLAY_STATE_SCHEMA_VERSION,
        )

    def render_rgb(self) -> np.ndarray | None:
        """Render the selected official child as a uint8 RGB diagnostic."""

        render = getattr(self.env, "render", None)
        if not callable(render):
            render = getattr(self.inner, "render", None)
        if not callable(render):
            return None
        try:
            image = render()
        except TypeError:
            try:
                image = render(mode="rgb_array")
            except Exception as exc:
                raise M1RuntimeError(f"RGB render failed: {exc}") from exc
        except Exception as exc:
            raise M1RuntimeError(f"RGB render failed: {exc}") from exc
        if image is None:
            return None
        value = np.asarray(image)
        if value.dtype != np.dtype(np.uint8) or value.ndim < 2 or value.size == 0:
            raise M1RuntimeError("RGB render must return a non-empty uint8 image")
        self.gl_identity = _live_gl_identity()
        expected = {}
        runtime = self.config.get("runtime")
        if isinstance(runtime, Mapping) and isinstance(runtime.get("renderer"), Mapping):
            configured = runtime["renderer"].get("expected_gl")
            if isinstance(configured, Mapping):
                expected = {str(key): str(item) for key, item in configured.items()}
        if self.config.get("strict_provenance") and expected and self.gl_identity != expected:
            raise M1RuntimeError(
                f"active GL identity differs from frozen renderer: expected {expected!r}, got {self.gl_identity!r}"
            )
        return np.ascontiguousarray(value).copy()

    def capture_rgb(self, label: str, path: str | Path) -> dict[str, Any] | None:
        image = self.render_rgb()
        if image is None:
            if self.config.get("strict_provenance"):
                raise M1RuntimeError("strict M1 runtime does not expose RGB rendering")
            return None
        target = safe_save_uint8_npy(path, image)
        record = {
            "label": str(label),
            "path": str(target),
            "sha256": sha256_file(target),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "gl": copy.deepcopy(self.gl_identity),
        }
        self.rgb_records[str(label)] = record
        return copy.deepcopy(record)

    def _validate_restore_identity(self, state: ReplayState) -> None:
        expected = dict(state.identity)
        actual = self._identity(int(expected.get("source_step", self.step_count)))
        # Source-step and tape hash are provenance values, not fresh-runtime
        # values, so compare only runtime identity/layout fields here.
        for key in (
            "suite",
            "task",
            "instruction",
            "init_state_id",
            "config_identity",
            "model_layout",
            "controller_configuration",
        ):
            if expected.get(key) != actual.get(key):
                raise M1RuntimeError(f"restore identity/layout mismatch at {key}")
        if self.config.get("strict_provenance") or self.config.get("authoritative") is True:
            for key in (
                "physics_model_fingerprint",
                "observation_model_fingerprint",
                "fixture_physics_fingerprint",
            ):
                expected_fingerprint = state.__dict__.get(key) or expected.get(key)
                actual_fingerprint = actual.get(key)
                if not isinstance(expected_fingerprint, Mapping) or not expected_fingerprint:
                    raise M1RuntimeError(f"restore identity/layout mismatch: missing {key}")
                if not isinstance(actual_fingerprint, Mapping) or not exact_model_fingerprint_equal(
                    expected_fingerprint, actual_fingerprint
                ):
                    raise M1RuntimeError(f"restore identity/layout mismatch at {key}")
        if self.tape_hash is not None and expected.get("action_tape_sha256") != self.tape_hash:
            raise M1RuntimeError("restore action-tape provenance mismatch")

    def _post_process_snapshot(self) -> dict[str, Any]:
        controller = _get_controller(self.inner)
        gripper = _get_gripper(self.inner)
        reward = self.last_reward
        reward_fn = getattr(self.env, "reward", None)
        if not callable(reward_fn):
            reward_fn = getattr(self.inner, "reward", None)
        if callable(reward_fn):
            try:
                reward = reward_fn()
            except TypeError:
                reward = reward_fn(None)
        if reward is None:
            reward = getattr(self.inner, "reward_value", getattr(self.env, "reward_value", None))
        snapshot: dict[str, Any] = {
            "integration": _integration_state(self.mujoco, self.model, self.data),
            "qpos": _maybe_array(getattr(self.data, "qpos", None)),
            "qvel": _maybe_array(getattr(self.data, "qvel", None)),
            "time": getattr(self.data, "time", None),
            "fixture": _fixture_snapshot(self.inner),
            "controller": _controller_state(controller),
            "gripper": _gripper_state(gripper),
            "gripper_physical": _gripper_kinematics(self.inner),
            "counters": _counter_state(self.inner, self.env),
            "reward": reward,
            "success": self._check_success(),
            "predicates": _predicate_snapshot(self.inner),
            "serialized_runtime": _serialized_runtime_snapshot(
                self.inner,
                owner_audit=self._owner_audit_for_serialized(),
                outer=self.env,
            ),
            "visual": _visual_state_snapshot(self.inner),
            "site_rgba": _maybe_array(getattr(self.model, "site_rgba", None)),
        }
        snapshot["controller_kinematic_caches"] = {
            name: _maybe_array(getattr(controller, name))
            for name in (
                "ee_pos",
                "ee_ori_mat",
                "ee_pos_vel",
                "ee_ori_vel",
                "joint_pos",
                "joint_vel",
                "J_full",
                "J_pos",
                "J_ori",
                "mass_matrix",
            )
            if controller is not None and hasattr(controller, name)
        }
        return snapshot

    def _check_success(self) -> Any:
        checker = getattr(self.env, "check_success", None)
        if not callable(checker):
            checker = getattr(self.inner, "_check_success", None)
        if not callable(checker):
            return None
        try:
            return checker()
        except Exception as exc:
            raise M1RuntimeError(f"success check failed during restore guard: {exc}") from exc

    def _prove_post_process(self) -> dict[str, Any]:
        """Run the task-0 hook once after reset and prove it is visual-only."""

        post_process = getattr(self.env, "_post_process", None)
        if not callable(post_process):
            post_process = getattr(self.inner, "_post_process", None)
        if not callable(post_process):
            self.post_process_proof = {
                "observed": False,
                "passed": False,
                "reason": "runtime does not expose _post_process",
            }
            return copy.deepcopy(self.post_process_proof)
        before = self._post_process_snapshot()
        visual_guard = before.get("visual", {}).get("vis_site_names", {})
        fixture_visual = visual_guard.get("fixtures") if isinstance(visual_guard, Mapping) else None
        if self.config.get("strict_provenance") and (
            not isinstance(fixture_visual, Mapping)
            or not fixture_visual
            or not any(value not in ({}, None) for value in fixture_visual.values())
        ):
            raise M1RuntimeError(
                "strict task-0 reset-only proof requires a nonempty fixtures_dict object_properties.vis_site_names guard"
            )
        post_process()
        after = self._post_process_snapshot()
        physical_keys = (
            "integration",
            "qpos",
            "qvel",
            "time",
            "fixture",
            "controller",
            "controller_kinematic_caches",
            "gripper",
            "gripper_physical",
            "counters",
            "reward",
            "success",
            "serialized_runtime",
        )
        changed = [key for key in physical_keys if not _exact_equal(before[key], after[key])]
        if not _exact_equal(
            before["visual"].get("object_properties_other"),
            after["visual"].get("object_properties_other"),
        ):
            changed.append("object_properties_other")
        if not _exact_equal(
            before["visual"].get("fixture_object_properties_other"),
            after["visual"].get("fixture_object_properties_other"),
        ):
            changed.append("fixture_object_properties_other")
        if "vis_site_names" not in self.post_process_allowlist and not _exact_equal(
            before["visual"].get("vis_site_names"), after["visual"].get("vis_site_names")
        ):
            changed.append("vis_site_names")
        if "model.site_rgba" not in self.post_process_allowlist and not _exact_equal(
            before["site_rgba"], after["site_rgba"]
        ):
            changed.append("site_rgba")
        if changed:
            raise M1RuntimeError("_post_process proof found non-visual mutation: " + ", ".join(changed))
        self.post_process_proof = {
            "observed": True,
            "passed": True,
            "allowlist": sorted(self.post_process_allowlist),
        }
        return copy.deepcopy(self.post_process_proof)

    def _guarded_post_process(self) -> None:
        if not self.post_process_proof.get("passed", False):
            raise M1RuntimeError("_post_process cannot run without a passing reset-only proof")
        post_process = getattr(self.env, "_post_process", None)
        if not callable(post_process):
            post_process = getattr(self.inner, "_post_process", None)
        if not callable(post_process):
            return
        before = self._post_process_snapshot()
        post_process()
        after = self._post_process_snapshot()
        for key in (
            "integration",
            "qpos",
            "qvel",
            "time",
            "fixture",
            "controller",
            "controller_kinematic_caches",
            "gripper",
            "gripper_physical",
            "counters",
            "reward",
            "success",
            "serialized_runtime",
        ):
            if not _exact_equal(before[key], after[key]):
                raise M1RuntimeError(f"_post_process changed non-visual state: {key}")
        before_visual = before["visual"]
        after_visual = after["visual"]
        if not _exact_equal(
            before_visual.get("object_properties_other"),
            after_visual.get("object_properties_other"),
        ):
            raise M1RuntimeError("_post_process changed non-allowlisted object_properties state")
        if not _exact_equal(
            before_visual.get("fixture_object_properties_other"),
            after_visual.get("fixture_object_properties_other"),
        ):
            raise M1RuntimeError("_post_process changed non-allowlisted fixture object state")
        if "vis_site_names" not in self.post_process_allowlist and not _exact_equal(
            before_visual.get("vis_site_names"), after_visual.get("vis_site_names")
        ):
            raise M1RuntimeError("_post_process changed non-allowlisted visual state: vis_site_names")
        if before["site_rgba"] is not None or after["site_rgba"] is not None:
            if before["site_rgba"] is None or after["site_rgba"] is None:
                raise M1RuntimeError("_post_process changed the site_rgba representation")
            if before["site_rgba"].shape != after["site_rgba"].shape:
                raise M1RuntimeError("_post_process changed site_rgba shape")
            if (
                "model.site_rgba" not in self.post_process_allowlist
                and not _exact_equal(before["site_rgba"], after["site_rgba"])
            ):
                raise M1RuntimeError("_post_process changed non-allowlisted site_rgba")
        # The only allowed mutable differences were explicitly snapshotted
        # above.  Keeping this check centralized makes future fields fail
        # closed instead of being silently treated as visual.

    def _refresh_observables(self) -> Any:
        refresh = getattr(self.env, "_update_observables", None)
        if not callable(refresh):
            refresh = getattr(self.inner, "_update_observables", None)
        if callable(refresh):
            try:
                return refresh(force=True)
            except TypeError as exc:
                raise M1RuntimeError(
                    "observable refresh must support the required force=True call"
                ) from exc
        raise M1RuntimeError("runtime does not expose a forced observable refresh")

    def restore(
        self,
        state: ReplayState,
        *,
        static_reference: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._restore_in_progress:
            raise M1RuntimeError("nested restore is not allowed")
        self._record_post_construction_operation("restore")
        self._restore_in_progress = True
        try:
            strict_restore = bool(
                self.config.get("strict_provenance")
                or self.config.get("authoritative") is True
            )
            serialized_runtime = _effective_serialized_runtime(state)
            # Re-audit the complete owner graph at the restore boundary so a
            # mutable field injected after construction cannot be hidden by a
            # stale cached classification.
            owner_audit = self._owner_audit_for_serialized(refresh=True)
            # Identity/layout is validated before any mutable write.
            self._validate_restore_identity(state)
            if strict_restore:
                # Fixtures are compiled ``mjModel`` body arrays.  An
                # authoritative restore must never patch them; it compares
                # the fresh reset pose/physics identity and blocks if the
                # persisted source differs.
                fresh_fixture = _fixture_snapshot(self.inner)
                expected_fixture_fp = state.fixture_physics_fingerprint or fixture_physics_fingerprint(state.fixture)
                actual_fixture_fp = fixture_physics_fingerprint(fresh_fixture)
                if not _exact_equal(state.fixture, fresh_fixture) or not exact_model_fingerprint_equal(
                    expected_fixture_fp, actual_fixture_fp
                ):
                    raise M1RuntimeError(
                        "strict restore fixture/physics fingerprint mismatch; "
                        "compiled mjModel fixture arrays are not writable"
                    )
            else:
                # Legacy injected tests used fixture patching before the v2
                # authoritative contract existed.  Keep this non-authoritative
                # compatibility path isolated from strict restore.
                _restore_fixture(self.inner, state.fixture)
            readback = set_integration_state_exact(
                self.mujoco, self.model, self.data, state.integration_state, require_readback=True
            )
            forward = getattr(self.mujoco, "mj_forward", None)
            if callable(forward):
                forward(
                    _raw_mujoco_handle(self.model, "_model"),
                    _raw_mujoco_handle(self.data, "_data"),
                )
            else:
                sim_forward = getattr(self.sim, "forward", None)
                if not callable(sim_forward):
                    raise M1RuntimeError("runtime cannot perform mj_forward")
                sim_forward()
            controller = _get_controller(self.inner)
            if controller is not None:
                update = getattr(controller, "update", None)
                if not callable(update):
                    raise M1RuntimeError("controller does not expose update(force=True)")
                update(force=True)
            _restore_controller(controller, state.controller)
            _restore_gripper(_get_gripper(self.inner), state.gripper)
            _restore_counters(self.inner, self.env, state.counters)
            self.experiment_step_count = int(
                state.counters.get(
                    "experiment_step_count",
                    state.identity.get("source_step", self.experiment_step_count),
                )
            )
            self.step_count = self.experiment_step_count
            if "reset_underlying_timestep" in state.counters:
                self.reset_underlying_timestep = int(state.counters["reset_underlying_timestep"])
            if "reset_underlying_cur_time" in state.counters:
                self.reset_underlying_cur_time = float(state.counters["reset_underlying_cur_time"])
            # Official MuJoCo integration state carries simulation time.  A
            # lightweight injected runtime may expose ``time`` separately and
            # not synchronize it from its fake state vector, so mirror the
            # serialized counter when that explicit field exists.
            if "cur_time" in state.counters and hasattr(self.data, "time"):
                setattr(self.data, "time", float(state.counters["cur_time"]))
            self.last_reward = state.reward
            self.last_raw_observation = copy.deepcopy(state.raw_observation)
            self.last_terminated = bool(
                state.counters.get("terminated", state.counters.get("done", False))
            )
            self.last_truncated = bool(state.counters.get("truncated", False))
            # Restore all concrete SERIALIZED owner fields before the guarded
            # task hook.  The hook and the forced observable refresh must see
            # the same counters/task/fixture/controller state as the source.
            # Refresh may legitimately rebuild derived observable values, so
            # restore the observable subset once more immediately afterwards.
            _restore_serialized_runtime(
                self.inner,
                serialized_runtime,
                owner_audit=owner_audit,
                outer=self.env,
            )
            self._guarded_post_process()
            # ReplayState v2/strict restores read h=0 from the restored
            # observable cursor without a dummy action or forced refresh.
            # Legacy non-authoritative fakes retain the original forced
            # refresh path when they expose no no-update observation seam.
            observation_reader_available = bool(
                callable(getattr(self.inner, "_get_observations", None))
                or callable(getattr(self.env, "_get_observations", None))
            )
            v2_observation_requested = bool(
                state.schema_version >= REPLAY_STATE_SCHEMA_VERSION
                or strict_restore
            )
            if v2_observation_requested and not observation_reader_available and (
                state.raw_observation is not None or strict_restore
            ):
                raise M1RuntimeError(
                    "ReplayState v2/strict restore requires inner._get_observations(force_update=False)"
                )
            use_no_update_observation = bool(v2_observation_requested and observation_reader_available)
            raw_observation_comparison: dict[str, Any] | None = None
            observed_raw_observation: Any = None
            if use_no_update_observation:
                # The no-update read must observe the restored serialized
                # cursor/cache, not a value reconstructed by a forced refresh.
                _restore_serialized_runtime(
                    self.inner,
                    serialized_runtime,
                    only_observables=True,
                    owner_audit=owner_audit,
                    outer=self.env,
                )
                observed_raw_observation = self._read_raw_observation_no_update()
                observed_encoded = _snapshot_serialized_value(observed_raw_observation)
                if state.raw_observation is None:
                    if strict_restore:
                        raise M1RuntimeError("strict ReplayState v2 is missing raw_observation")
                    raw_observation_comparison = comparison_result_to_dict(
                        compare_exact(None, observed_encoded, path="raw_observation")
                    )
                else:
                    raw_observation_comparison = comparison_result_to_dict(
                        compare_exact(
                            _snapshot_serialized_value(state.raw_observation),
                            observed_encoded,
                            path="raw_observation",
                        )
                    )
            else:
                self._refresh_observables()
                _restore_serialized_runtime(
                    self.inner,
                    serialized_runtime,
                    only_observables=True,
                    owner_audit=owner_audit,
                    outer=self.env,
                )
            serialized_readback = _serialized_runtime_snapshot(
                self.inner,
                owner_audit=owner_audit,
                outer=self.env,
            )
            if set(serialized_readback) != set(serialized_runtime) or not _exact_equal(
                serialized_readback,
                serialized_runtime,
            ):
                raise M1RuntimeError("serialized runtime full readback mismatch after refresh")
            actual = self.collect_invariants()
            result: dict[str, Any] = {
                "invariants": actual,
                "serialized_runtime_readback": serialized_readback,
                "immediate_readback": {
                    "exact": bool(np.array_equal(readback, state.integration_state, equal_nan=False)),
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "shape": list(readback.shape),
                    "dtype": str(readback.dtype),
                    "expected": state.integration_state.copy(),
                    "readback": readback.copy(),
                    "expected_sha256": sha256_bytes(state.integration_state.tobytes(order="C")),
                    "readback_sha256": sha256_bytes(readback.tobytes(order="C")),
                },
            }
            if strict_restore:
                current_physics = physics_model_fingerprint(self.model, self.mujoco)
                current_observation = observation_model_fingerprint(
                    self.model,
                    _renderer_config_for_runtime(self.env, self.inner, self.config),
                    self.mujoco,
                )
                result["physics_static_identity"] = exact_model_fingerprint_equal(
                    state.physics_model_fingerprint,
                    current_physics,
                )
                result["observation_static_identity"] = exact_model_fingerprint_equal(
                    state.observation_model_fingerprint,
                    current_observation,
                )
                result["static_fingerprints"] = {
                    "physics": current_physics,
                    "observation": current_observation,
                }
            if raw_observation_comparison is not None:
                result["raw_observation"] = observed_raw_observation
                result["raw_observation_comparison"] = raw_observation_comparison
            if static_reference is not None:
                static_comparisons = compare_invariants(static_reference, actual)
                result["static"] = {
                    "gate": aggregate_gate(static_comparisons),
                    "comparisons": [comparison_result_to_dict(item) for item in static_comparisons],
                }
            return result
        except (ForbiddenRestoreError, UnknownMutableStateError):
            raise
        except Exception as exc:
            if isinstance(exc, M1Error):
                raise
            raise M1RuntimeError(f"state restore failed: {exc}") from exc
        finally:
            self._restore_in_progress = False

    def close(self) -> None:
        """Close the vector/official environment exactly once."""

        if self._closed:
            return
        self._closed = True
        # The official vector wrapper owns its selected child.  Calling both
        # close methods can double-close MuJoCo resources, so prefer the
        # owner when it exposes close and fall back to the child only for
        # minimal injected runtimes.
        vector_close = getattr(self.vector_env, "close", None)
        if callable(vector_close):
            targets = [self.vector_env]
        else:
            targets = [self.env] if self.env is not None else []
        errors: list[str] = []
        for target in targets:
            close = getattr(target, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            raise M1RuntimeError("environment close failed: " + "; ".join(errors))

    def step(self, action: np.ndarray) -> Any:
        if self._restore_in_progress:
            raise M1RuntimeError("cannot step while restoring")
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (ACTION_DIM,):
            raise M1SchemaError("M1 actions must have shape (7,)")
        # This intentionally calls the official LiberoEnv child rather than
        # the vector wrapper, avoiding vector autoreset mutable state.
        step = getattr(self.env, "step", None)
        if not callable(step):
            raise M1RuntimeError("official LiberoEnv child does not expose step")
        result = step(value)
        self.experiment_step_count += 1
        self.step_count = self.experiment_step_count
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
            if len(result) >= 1:
                # Preserve the exact source observation before any policy
                # processor or later refresh can transform its Observable
                # cache.  ReplayState v2 serializes this value losslessly.
                self.last_raw_observation = copy.deepcopy(result[0])
            if len(result) >= 2:
                try:
                    self.last_reward = float(result[1])
                except (TypeError, ValueError):
                    self.last_reward = None
            if len(result) >= 5:
                self.last_terminated = bool(result[2])
                self.last_truncated = bool(result[3])
                self.last_info = result[4]
            elif len(result) >= 4:
                self.last_terminated = bool(result[2])
                self.last_truncated = False
                self.last_info = result[3]
        return result

    def collect_invariants(self) -> dict[str, Any]:
        controller = _get_controller(self.inner)
        gripper = _get_gripper(self.inner)
        counters = _counter_state(self.inner, self.env)
        counters["reset_underlying_timestep"] = int(self.reset_underlying_timestep)
        counters["reset_underlying_cur_time"] = float(self.reset_underlying_cur_time)
        counters["experiment_step_count"] = int(self.experiment_step_count)
        counters.setdefault("terminated", bool(self.last_terminated))
        counters.setdefault("truncated", bool(self.last_truncated))
        reward = self.last_reward
        if reward is None:
            for owner in (self.env, self.inner):
                reward_fn = getattr(owner, "reward", None)
                if callable(reward_fn):
                    try:
                        reward = reward_fn()
                    except TypeError:
                        try:
                            reward = reward_fn(None)
                        except Exception:
                            reward = None
                    except Exception:
                        reward = None
                    if reward is not None:
                        break
        observable_values, observable_cache = _observable_snapshot(self.inner)
        constraints: dict[str, Any] = {}
        for name in (
            "efc_force",
            "efc_type",
            "efc_state",
            "efc_id",
            "efc_J",
            "efc_pos",
            "efc_margin",
            "efc_solref",
            "efc_solimp",
            "efc_friction",
        ):
            if hasattr(self.data, name):
                constraints[name] = _maybe_array(getattr(self.data, name))
        result: dict[str, Any] = {
            "integration": _integration_state(self.mujoco, self.model, self.data),
            "time": float(getattr(self.data, "time", np.nan)) if hasattr(self.data, "time") else None,
            "qpos": _maybe_array(getattr(self.data, "qpos", None)),
            "qvel": _maybe_array(getattr(self.data, "qvel", None)),
            "controller": _controller_state(controller),
            "gripper": _gripper_state(gripper),
            "gripper_physical": _gripper_kinematics(self.inner),
            "fixture": _fixture_snapshot(self.inner),
            "body_xpos": _maybe_array(getattr(self.data, "xpos", getattr(self.data, "body_xpos", None))),
            "body_xquat": _maybe_array(getattr(self.data, "xquat", getattr(self.data, "body_xquat", None))),
            "counters": counters,
            "reward": reward,
            "success": self._check_success(),
            # Goal predicates are discrete task state.  Persist every
            # concrete predicate so paired comparisons cannot hide a goal
            # transition behind the numeric envelope.
            "predicates": _predicate_snapshot(self.inner),
            "observables": observable_values,
            "observable_cache": observable_cache,
            "serialized_runtime": _serialized_runtime_snapshot(
                self.inner,
                owner_audit=self._owner_audit_for_serialized(),
                outer=self.env,
            ),
            "constraints": constraints,
            "diagnostic": {
                **_visual_state_snapshot(self.inner),
                "site_rgba": _maybe_array(getattr(self.model, "site_rgba", None)),
            },
        }
        if controller is not None:
            for name in (
                "ee_pos",
                "ee_ori_mat",
                "ee_pos_vel",
                "ee_ori_vel",
                "joint_pos",
                "joint_vel",
                "J_full",
                "J_pos",
                "J_ori",
                "mass_matrix",
            ):
                if hasattr(controller, name):
                    result[name] = _maybe_array(getattr(controller, name))
            if hasattr(controller, "ee_pos") or hasattr(controller, "ee_ori_mat"):
                result["eef_pose"] = {
                    "pos": _maybe_array(getattr(controller, "ee_pos", None)),
                    "orientation": _maybe_array(getattr(controller, "ee_ori_mat", None)),
                    "linear_velocity": _maybe_array(getattr(controller, "ee_pos_vel", None)),
                    "angular_velocity": _maybe_array(getattr(controller, "ee_ori_vel", None)),
                }
            result["controller_kinematic_caches"] = {
                name: _maybe_array(getattr(controller, name))
                for name in (
                    "ee_pos",
                    "ee_ori_mat",
                    "ee_pos_vel",
                    "ee_ori_vel",
                    "joint_pos",
                    "joint_vel",
                    "J_full",
                    "J_pos",
                    "J_ori",
                    "mass_matrix",
                )
                if hasattr(controller, name)
            }
        result["objects"] = _object_snapshot(self.inner)
        result["contacts"] = _runtime_contact_pairs(self.data, self.model, self.mujoco)
        # Construction audits the official instance once, but mutable fields
        # must remain present for every source/reference and restore sample.
        # Otherwise a field absent from both sides could evade the comparison
        # union and become a false pass.
        if self.config.get("strict_provenance"):
            validate_invariant_snapshot(result)
        return result


def compare_invariants(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> tuple[ComparisonResult, ...]:
    """Compare each invariant leaf with its contract-specific comparison class."""

    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise M1SchemaError("invariant snapshots must be mappings")

    floating_roots = {
        "integration",
        "time",
        "qpos",
        "qvel",
        "fixture",
        "body_xpos",
        "body_xquat",
        "ee_pos",
        "ee_ori_mat",
        "joint_pos",
        "joint_vel",
        "gripper",
        "gripper_physical",
        "eef_pose",
        "objects",
        "controller_kinematic_caches",
        "constraints",
        "serialized_runtime",
    }
    exact_roots = {
        "reward",
        "success",
        "predicates",
        "identity",
        "controller_configuration",
        "counters.timestep",
        "counters.done",
        "counters.terminated",
        "counters.truncated",
        "controller.new_update",
        "controller.impedance_mode",
        "controller.use_delta",
        "controller.use_ori",
        "controller.eef_name",
        "controller.joint_dim",
        "controller.control_freq",
        "constraints.efc_type",
        "constraints.efc_id",
        "constraints.efc_state",
    }
    # Observable values remain derived diagnostic evidence, while the
    # concrete timer/cache cursor fields under serialized_runtime are exact
    # serialized state and therefore gate the replay.
    diagnostic_roots = {"diagnostic", "rgb", "observables", "observable_cache"}

    def comparison_for(path: str, left: Any, right: Any) -> ComparisonResult:
        root = path.split(".", 1)[0]
        if root in diagnostic_roots:
            return compare_diagnostic(left, right, path=path)
        if path in exact_roots or root in {"identity", "controller_configuration", "serialized_runtime"}:
            return compare_exact(left, right, path=path)
        if path == "contacts":
            return compare_contact_pairs(left or (), right or (), path=path)
        if root in floating_roots or root == "controller" or path == "counters.cur_time":
            sign_invariant = any(token in path.lower() for token in ("quat", "orientation"))
            return compare_floating(left, right, path=path, quaternion_sign_invariant=sign_invariant)
        return compare_exact(left, right, path=path)

    results: list[ComparisonResult] = []

    def visit(path: str, left_present: bool, right_present: bool, left: Any, right: Any) -> None:
        if not left_present or not right_present:
            root = path.split(".", 1)[0]
            if root in diagnostic_roots:
                result = ComparisonResult(
                    ComparisonClass.DERIVED_DIAGNOSTIC,
                    False,
                    path=path,
                    detail="invariant is missing from one side",
                    gates_pass=True,
                )
            elif path in exact_roots or root in {"identity", "controller_configuration", "serialized_runtime"}:
                result = ComparisonResult(
                    ComparisonClass.EXACT,
                    False,
                    path=path,
                    detail="invariant is missing from one side",
                    gates_pass=False,
                )
            elif root in floating_roots or root == "controller" or path == "counters.cur_time":
                result = ComparisonResult(
                    ComparisonClass.FLOATING_PHYSICAL,
                    False,
                    path=path,
                    detail="invariant is missing from one side",
                    gates_pass=True,
                )
            else:
                result = ComparisonResult(
                    ComparisonClass.EXACT,
                    False,
                    path=path,
                    detail="invariant is missing from one side",
                    gates_pass=False,
                )
            results.append(result)
            return
        if path == "contacts":
            results.append(comparison_for(path, left, right))
            return
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                results.append(comparison_for(path, left, right))
                return
            keys = sorted(set(left) | set(right), key=str)
            if not keys:
                results.append(comparison_for(path, left, right))
                return
            for key in keys:
                child_path = f"{path}.{key}" if path else str(key)
                visit(child_path, key in left, key in right, left.get(key), right.get(key))
            return
        results.append(comparison_for(path, left, right))

    keys = sorted(set(expected) | set(actual), key=str)
    for key in keys:
        visit(str(key), key in expected, key in actual, expected.get(key), actual.get(key))
    return tuple(results)


def aggregate_gate(results: Iterable[ComparisonResult]) -> dict[str, Any]:
    values = tuple(results)
    exact = [result for result in values if result.comparison_class is ComparisonClass.EXACT]
    floating = [result for result in values if result.comparison_class is ComparisonClass.FLOATING_PHYSICAL]
    diagnostic = [result for result in values if result.comparison_class is ComparisonClass.DERIVED_DIAGNOSTIC]
    return {
        "pass": all(result.passed for result in (*exact, *floating)),
        "exact": {
            "pass": all(result.passed for result in exact),
            "results": [comparison_result_to_dict(result) for result in exact],
        },
        "floating_physical": {
            "pass": all(result.passed for result in floating),
            "results": [comparison_result_to_dict(result) for result in floating],
        },
        "derived_diagnostic": {
            "pass": all(result.passed for result in diagnostic),
            "gates_pass": True,
            "results": [comparison_result_to_dict(result) for result in diagnostic],
        },
    }


def build_official_runtime(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Call the project-owned policy-free official CPU environment factory."""

    _apply_runtime_environment(config)
    try:
        from scripts.dcu_preflight import build_cpu_environment_runtime

        return build_cpu_environment_runtime(config, phase="compare")
    except M1Error:
        raise
    except Exception as exc:
        raise M1RuntimeError(f"official CPU runtime construction failed: {exc}") from exc


def _git_sha() -> str | None:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.rstrip("\r\n")
    except Exception:
        return None
    return value


def _resolve_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else ROOT / target


def _runtime_environment_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return and validate the exact offline/EGL environment contract."""

    runtime = _mapping(config.get("runtime", {}), "runtime")
    environment = _mapping(runtime.get("environment"), "runtime.environment")
    expected = {**FROZEN_RUNTIME_ENVIRONMENT, "empty_hf_cache": True}
    if dict(environment) != expected:
        raise M1ConfigError("runtime.environment is not the exact frozen offline/EGL contract")
    libero_config_path = Path(str(environment["LIBERO_CONFIG_PATH"]))
    hf_home = Path(str(environment["HF_HOME"]))
    if not libero_config_path.is_dir() or not (libero_config_path / "config.yaml").is_file():
        raise M1ConfigError(f"frozen LIBERO_CONFIG_PATH is unavailable: {libero_config_path}")
    if not hf_home.is_dir():
        raise M1ConfigError(f"frozen HF_HOME is unavailable: {hf_home}")
    if any(hf_home.iterdir()):
        raise M1ConfigError(f"frozen HF_HOME must be empty/fresh: {hf_home}")
    return dict(environment)


def _apply_runtime_environment(config: Mapping[str, Any]) -> dict[str, str]:
    """Set the pinned process environment before importing official runtime code."""

    if config.get("strict_provenance") is not True:
        return {}
    environment = _runtime_environment_contract(config)
    values = {
        str(key): str(value)
        for key, value in environment.items()
        if str(key) != "empty_hf_cache"
    }
    os.environ.update(values)
    return values


def _git_command(path: Path, *args: str) -> str | None:
    try:
        value = subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
            # Preserve the two-column porcelain status prefix.  ``strip()`` would
            # drop its leading space for a worktree-modified file and shift the
            # parsed path by one character (for example ``AGENTS.md`` ->
            # ``GENTS.md``), defeating the allowed-dirty audit.
        ).stdout.rstrip("\r\n")
    except Exception:
        return None
    return value


def _source_checkout_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit every pinned external checkout without modifying it."""

    expected_root = config.get("source_checkouts")
    if not isinstance(expected_root, Mapping):
        return {"required": False, "pass": False, "entries": [], "errors": ["source_checkouts missing"]}
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in sorted(expected_root, key=str):
        expected = expected_root[name]
        if not isinstance(expected, Mapping):
            errors.append(f"source_checkouts.{name} is not a mapping")
            continue
        target = _resolve_path(str(expected.get("path", "")))
        head = _git_command(target, "rev-parse", "HEAD")
        status = _git_command(target, "status", "--porcelain", "--untracked-files=all")
        branch = _git_command(target, "symbolic-ref", "-q", "--short", "HEAD")
        url = _git_command(target, "remote", "get-url", "origin")
        actual = {
            "path": str(target),
            "git_sha": head,
            "clean": status == "" if status is not None else False,
            "detached": branch is None,
            "url": url,
        }
        entries.append({"name": str(name), "expected": dict(expected), "actual": actual})
        for key in ("git_sha", "clean", "detached", "url"):
            if actual.get(key) != expected.get(key):
                errors.append(
                    f"source checkout {name} {key} mismatch: expected {expected.get(key)!r}, got {actual.get(key)!r}"
                )
    return {
        "required": True,
        "pass": not errors and len(entries) == len(expected_root),
        "entries": entries,
        "errors": errors,
    }


def _worktree_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit the root before M1 output exists, with no expected HEAD value.

    The authoritative implementation identity is the actual current HEAD.
    Script/config bytes must match their HEAD blobs exactly, while the other
    implementation files must also be clean relative to HEAD.  Only the
    explicitly hashed AGENTS guidance and raw M0 retention subtree may remain
    dirty; all other status entries are retained as blocking evidence.
    """

    expected = config.get("worktree_audit")
    if not isinstance(expected, Mapping):
        return {"required": False, "pass": False, "errors": ["worktree_audit missing"]}
    head = _git_command(ROOT, "rev-parse", "HEAD")
    status = _git_command(ROOT, "status", "--porcelain", "--untracked-files=all")
    branch = _git_command(ROOT, "symbolic-ref", "-q", "--short", "HEAD")
    status_text = status or ""

    def status_entries(value: str) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for line in value.splitlines():
            if not line:
                continue
            code = line[:2] if len(line) >= 2 else line
            relative = line[3:] if len(line) >= 4 else ""
            # Rename entries are represented by the destination path for
            # allowlist checks; retain the raw line for forensic evidence.
            if " -> " in relative:
                relative = relative.split(" -> ", 1)[1]
            entries.append({"code": code, "path": relative, "raw": line})
        return entries

    entries = status_entries(status_text)
    implementation_paths_value = expected.get("implementation_paths", FROZEN_WORKTREE_AUDIT["implementation_paths"])
    implementation_paths = tuple(
        str(path).replace(os.sep, "/").lstrip("./")
        for path in implementation_paths_value
        if str(path).strip()
    ) if isinstance(implementation_paths_value, Iterable) and not isinstance(implementation_paths_value, (str, bytes)) else ()

    def relative_path(path: str | Path) -> str:
        target = Path(path)
        if target.is_absolute():
            try:
                target = target.relative_to(ROOT)
            except ValueError:
                return str(target).replace(os.sep, "/")
        return str(target).replace(os.sep, "/").lstrip("./")

    def under(relative: str, prefix: str) -> bool:
        return relative == prefix or relative.startswith(prefix.rstrip("/") + "/")

    allowed_items = config.get("allowed_dirty_evidence", {})
    if isinstance(allowed_items, Mapping):
        iterable: Iterable[tuple[Any, Any]] = allowed_items.items()
    elif isinstance(allowed_items, (list, tuple)):
        iterable = ((f"entry_{index}", value) for index, value in enumerate(allowed_items))
    else:
        iterable = ()
    allowed_records: list[dict[str, Any]] = []
    allowed_prefixes: list[str] = []
    allowed_output_root_value = expected.get("allowed_output_root")
    allowed_output_prefix = ""
    if isinstance(allowed_output_root_value, (str, Path)) and str(allowed_output_root_value).strip():
        allowed_output_prefix = relative_path(_resolve_path(allowed_output_root_value))
        allowed_prefixes.append(allowed_output_prefix)
    errors: list[str] = []
    if head is None:
        errors.append("root HEAD is unavailable")
    if status is None:
        errors.append("root worktree status is unavailable")
    for label, value in iterable:
        record = {"label": str(label)}
        if isinstance(value, Mapping):
            path = value.get("path")
            expected_hash = value.get("sha256")
            record["reason"] = value.get("reason")
        else:
            path = label
            expected_hash = value
        if not isinstance(path, (str, Path)) or not str(path).strip():
            errors.append(f"allowed dirty evidence {label} has no path")
            allowed_records.append(record)
            continue
        target = _resolve_path(path)
        relative = relative_path(target)
        allowed_prefixes.append(relative)
        record["path"] = str(target)
        record["expected"] = str(expected_hash).lower() if expected_hash is not None else None
        try:
            actual_hash = _hash_path(target)
        except Exception as exc:
            errors.append(f"allowed dirty evidence {label} hash failed: {exc}")
            actual_hash = None
        record["sha256"] = actual_hash
        if record["expected"] is None or actual_hash != record["expected"]:
            errors.append(
                f"allowed dirty evidence {label} hash mismatch: expected {record['expected']!r}, got {actual_hash!r}"
            )
        allowed_records.append(record)

    def head_blob(relative: str) -> bytes | None:
        if head is None or not relative or relative.startswith("/") or ".." in Path(relative).parts:
            return None
        try:
            result = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
        except Exception:
            return None
        return bytes(result.stdout)

    implementation_unexpected: list[str] = []
    implementation_missing: list[str] = []
    implementation_rows: list[dict[str, Any]] = []
    for relative in implementation_paths:
        target = ROOT / relative
        working_hash = sha256_file(target) if target.is_file() and not target.is_symlink() else None
        blob = head_blob(relative)
        head_hash = sha256_bytes(blob) if blob is not None else None
        exact = bool(blob is not None and target.is_file() and not target.is_symlink() and target.read_bytes() == blob)
        row = {
            "path": relative,
            "working_sha256": working_hash,
            "head_sha256": head_hash,
            "exact_head_blob": exact,
        }
        implementation_rows.append(row)
        if blob is None or not target.is_file() or target.is_symlink() or not exact:
            implementation_unexpected.append(relative)
            if blob is None or not target.is_file():
                implementation_missing.append(relative)

    allowed_status: list[dict[str, str]] = []
    allowed_output_status: list[dict[str, str]] = []
    implementation_status: list[dict[str, str]] = []
    other_status: list[dict[str, str]] = []
    for entry in entries:
        path = entry["path"]
        if allowed_output_prefix and under(path, allowed_output_prefix):
            allowed_output_status.append(entry)
            allowed_status.append(entry)
        elif any(under(path, prefix) for prefix in allowed_prefixes):
            allowed_status.append(entry)
        elif any(path == implementation or under(path, implementation) for implementation in implementation_paths):
            implementation_status.append(entry)
            if path not in implementation_unexpected:
                implementation_unexpected.append(path)
        else:
            other_status.append(entry)

    if expected.get("timing") != "pre_output":
        errors.append("worktree audit timing must be pre_output")
    if not implementation_paths:
        errors.append("worktree audit implementation_paths are empty")
    if implementation_unexpected:
        errors.append("implementation files are not exact clean HEAD blobs")
    if other_status:
        errors.append("unexpected pre-existing worktree dirt is present")
    actual = {
        "head": head,
        "branch": branch,
        "detached": branch is None,
        "clean": status_text == "" if status is not None else False,
        "status": status_text,
        "status_entries": entries,
        "timing": "pre_output",
        "allowed_output_root": str(expected.get("allowed_output_root", "")),
    }
    implementation = {
        "paths": list(implementation_paths),
        "clean": not implementation_unexpected,
        "head_contains_exact": all(row["exact_head_blob"] for row in implementation_rows),
        "rows": implementation_rows,
        "unexpected": sorted(set(implementation_unexpected)),
        "missing": sorted(set(implementation_missing)),
        "status": implementation_status,
    }
    return {
        "required": True,
        "pass": not errors and bool(head) and implementation["clean"] and implementation["head_contains_exact"] and not other_status,
        "expected": {
            "timing": expected.get("timing"),
            "allowed_output_root": expected.get("allowed_output_root"),
            "implementation_paths": list(implementation_paths),
        },
        "actual": actual,
        "implementation": implementation,
        "allowed_dirty": {
            "records": allowed_records,
            "status": allowed_status,
        },
        "allowed_output": {
            "root": allowed_output_prefix,
            "status": allowed_output_status,
        },
        "other_dirt": other_status,
        "errors": errors,
    }


def _tree_sha256(path: str | Path) -> str:
    """Hash a directory deterministically by relative names and file bytes."""

    root = _resolve_path(path)
    if not root.is_dir() or root.is_symlink():
        raise M1SchemaError(f"expected regular directory for tree SHA-256: {root}")
    digest = hashlib.sha256()
    for item in sorted((item for item in root.rglob("*") if item.is_file() and not item.is_symlink()), key=lambda item: str(item.relative_to(root))):
        relative = str(item.relative_to(root)).replace(os.sep, "/").encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hash_path(path: str | Path) -> str:
    target = _resolve_path(path)
    if target.is_dir():
        return _tree_sha256(target)
    return sha256_file(target)


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata as importlib_metadata

        return importlib_metadata.version(name)
    except Exception:
        return None


def _live_gl_identity() -> dict[str, str] | None:
    """Read the active GL vendor/renderer/version after a render call."""

    try:
        from OpenGL import GL  # type: ignore[import-not-found]

        def read(name: int) -> str | None:
            value = GL.glGetString(name)
            if value is None:
                return None
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        values = {
            "vendor": read(GL.GL_VENDOR),
            "renderer": read(GL.GL_RENDERER),
            "version": read(GL.GL_VERSION),
        }
    except Exception:
        return None
    if any(value is None or not value.strip() for value in values.values()):
        return None
    return {key: str(value) for key, value in values.items()}


def _hash_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect configured input hashes and fail closed only for strict config."""

    entries: list[dict[str, Any]] = []
    allowed_dirty_entries: list[dict[str, Any]] = []
    errors: list[str] = []

    def add(label: str, path: Any, expected: Any | None = None) -> None:
        if not isinstance(path, (str, Path)) or not str(path).strip():
            return
        target = _resolve_path(path)
        if not target.exists() or target.is_symlink():
            if expected is not None:
                errors.append(f"{label} path is missing: {target}")
            return
        try:
            # The checked-in config carries a self-referential digest.  Its
            # expected value therefore names the canonical contract with the
            # digest field removed, while every other input uses raw bytes.
            actual = (
                _config_contract_sha256_file(target)
                if label == "config" and expected is not None
                else _hash_path(target)
            )
        except Exception as exc:
            errors.append(f"{label} hash failed: {exc}")
            return
        expected_text = str(expected).lower() if expected is not None else None
        if expected_text is not None and actual != expected_text:
            # Checkpoint/base model SHA values name model.safetensors while the
            # configured path names its immutable directory.
            model_file = target / "model.safetensors" if target.is_dir() else None
            if model_file is not None and model_file.is_file() and sha256_file(model_file) == expected_text:
                actual = expected_text
                target = model_file
            else:
                errors.append(f"{label} hash mismatch: expected {expected_text}, got {actual}")
        entries.append({"label": label, "path": str(target), "sha256": actual, "expected": expected_text})

    for label, name in (("checkpoint", "checkpoint"), ("base_model", "base_model"), ("assets", "assets")):
        value = config.get(name)
        if isinstance(value, Mapping):
            expected = value.get("manifest_sha256", value.get("sha256"))
            manifest_path = value.get("manifest_path") if label == "assets" else None
            add(label, manifest_path or value.get("path"), expected)
    libero_config = config.get("libero_config")
    if isinstance(libero_config, Mapping):
        add("libero_config", libero_config.get("path"), libero_config.get("sha256"))
    paths = config.get("paths")
    if isinstance(paths, Mapping):
        add("config", paths.get("config"), paths.get("config_sha256"))
        source_evidence = paths.get("source_evidence")
        source_evidence_hashes = paths.get("source_evidence_sha256")
        if isinstance(source_evidence, Mapping):
            for label, path in source_evidence.items():
                expected = (
                    source_evidence_hashes.get(label)
                    if isinstance(source_evidence_hashes, Mapping)
                    else None
                )
                add(f"source.{label}", path, expected)
        add("runtime_lock", paths.get("runtime_lock"), paths.get("runtime_lock_sha256"))
        add("artifact_manifest", paths.get("artifact_manifest"), paths.get("artifact_manifest_sha256"))
    closure = config.get("m0_closure")
    if isinstance(closure, Mapping):
        closure_path = closure.get("path")
        if bool(config.get("strict_provenance", False)) and (
            not isinstance(closure_path, (str, Path)) or not _resolve_path(closure_path).is_file()
        ):
            errors.append("m0_closure path is missing")
        elif isinstance(closure_path, (str, Path)) and str(closure_path).strip():
            # The M0 closure identifier is intentionally distinct from the
            # file digest; retain both pieces of evidence without comparing
            # unrelated hashes.
            add("m0_closure_file", closure_path)
    # The M0 closure identifier is a provenance/commit identity rather than a
    # digest of its manifest file; it is recorded in provenance but is not
    # conflated with the artifact hash audit here.
    configured_hashes = config.get("hashes")
    if isinstance(configured_hashes, Mapping):
        for label, value in configured_hashes.items():
            if isinstance(value, Mapping):
                add(str(label), value.get("path"), value.get("sha256"))
            else:
                add(str(label), label, value)
    dirty = config.get("allowed_dirty_evidence")
    if isinstance(dirty, Mapping):
        dirty_items: Iterable[tuple[Any, Any]] = dirty.items()
    elif isinstance(dirty, (list, tuple)):
        dirty_items = ((f"entry_{index}", value) for index, value in enumerate(dirty))
    else:
        dirty_items = ()
    for label, value in dirty_items:
        if isinstance(value, Mapping):
            path = value.get("path")
            expected = value.get("sha256")
        else:
            path = label
            expected = value
        if not isinstance(path, (str, Path)) or not str(path).strip():
            errors.append(f"allowed_dirty_evidence.{label} path is required")
            continue
        target = _resolve_path(path)
        if target.is_symlink() or not target.exists() or not (target.is_file() or target.is_dir()):
            errors.append(f"allowed_dirty_evidence.{label} path is missing: {target}")
            continue
        actual = _hash_path(target)
        expected_text = str(expected).lower() if expected is not None else None
        if expected_text is None or len(expected_text) != 64 or actual != expected_text:
            errors.append(
                f"allowed_dirty_evidence.{label} hash mismatch: expected {expected_text}, got {actual}"
            )
        allowed_dirty_entries.append(
            {
                "label": str(label),
                "path": str(target),
                "sha256": actual,
                "expected": expected_text,
            }
        )
    strict = bool(config.get("strict_provenance", False))
    if strict and errors:
        raise M1ConfigError("hash audit failed: " + "; ".join(errors))
    return {
        "pass": not errors,
        "entries": entries,
        "allowed_dirty_evidence": allowed_dirty_entries,
        "errors": errors,
    }


def _runtime_lock_facts(config: Mapping[str, Any]) -> dict[str, str]:
    """Read the approved package facts from the immutable runtime lock."""

    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        return {}
    raw_path = paths.get("runtime_lock")
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        return {}
    target = _resolve_path(raw_path)
    if not target.is_file():
        return {}
    text = target.read_text(encoding="utf-8", errors="strict")
    facts: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key == "python_version":
            facts["python"] = value.split()[0]
        elif key in {"torch_version", "transformers_version", "numpy_version", "mujoco_version"}:
            fact_key = {
                "torch_version": "pytorch",
                "transformers_version": "transformers",
                "numpy_version": "numpy",
                "mujoco_version": "mujoco",
            }[key]
            facts[fact_key] = value
        elif key in {"gymnasium_version", "robosuite_version", "hf_libero_version", "lerobot_version"}:
            facts[key.removesuffix("_version")] = value
        elif key == "torch_cuda_version":
            facts["cuda"] = "none" if value.lower() in {"none", "null", "n/a"} else value
        elif key == "backend":
            facts["gpu"] = "CPU" if value.lower() == "cpu" else value
    # The lock's pip-freeze section is authoritative for packages whose
    # accelerator summary does not repeat a version.
    for line in text.splitlines():
        stripped = line.strip()
        for package, key in (
            ("gymnasium", "gymnasium"),
            ("robosuite", "robosuite"),
            ("hf_libero", "hf_libero"),
            ("lerobot", "lerobot"),
        ):
            prefix = package + "=="
            if stripped.startswith(prefix):
                facts[key] = stripped[len(prefix) :].split()[0]
    return facts


def _runtime_facts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect installed runtime facts without falling back to config pins."""

    try:
        import mujoco  # type: ignore[import-not-found]

        mujoco_version = getattr(mujoco, "__version__", None)
    except Exception:
        mujoco_version = None
    package_names = {
        "pytorch": "torch",
        "transformers": "transformers",
        "numpy": "numpy",
        "gymnasium": "gymnasium",
        "lerobot": "lerobot",
        "robosuite": "robosuite",
    }
    package_values = {key: _package_version(name) for key, name in package_names.items()}
    hf_libero = _package_version("hf-libero") or _package_version("hf_libero")
    cuda = "none"
    try:
        import torch  # type: ignore[import-not-found]

        cuda_value = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_value:
            cuda = str(cuda_value)
    except Exception:
        pass
    facts: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": package_values["numpy"] or np.__version__,
        "pytorch": package_values["pytorch"],
        "transformers": package_values["transformers"],
        "gymnasium": package_values["gymnasium"],
        "lerobot": package_values["lerobot"],
        "robosuite": package_values["robosuite"],
        "hf_libero": hf_libero,
        "mujoco": mujoco_version or _package_version("mujoco"),
        "cuda": cuda,
        "gpu": "CPU" if cuda == "none" else (platform.processor() or "unknown"),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("MUJOCO_EGL_DEVICE_ID", "cpu")),
    }
    return facts


_RUNTIME_FACT_KEYS: tuple[str, ...] = (
    "python",
    "numpy",
    "pytorch",
    "transformers",
    "gymnasium",
    "lerobot",
    "robosuite",
    "hf_libero",
    "mujoco",
    "cuda",
    "gpu",
)


def _runtime_fact_expectations(config: Mapping[str, Any]) -> dict[str, str | None]:
    """Resolve the exact approved expectation for every gated runtime fact."""

    pins = _mapping(config.get("pins"), "pins")
    lock = _runtime_lock_facts(config)
    return {
        key: str(pins[key]) if key in pins else str(lock[key]) if key in lock else None
        for key in _RUNTIME_FACT_KEYS
    }


def _runtime_fact_audit(
    config: Mapping[str, Any], facts: Mapping[str, Any]
) -> dict[str, Any]:
    """Return publishable runtime mismatch evidence without raising."""

    actual = {str(key): value for key, value in dict(facts).items()}
    expected = _runtime_fact_expectations(config)
    lock = _runtime_lock_facts(config)
    mismatches: list[dict[str, Any]] = []
    errors: list[str] = []
    for key in _RUNTIME_FACT_KEYS:
        expected_value = expected.get(key)
        actual_value = actual.get(key)
        if expected_value is None:
            message = f"runtime expectation is missing for {key}"
            errors.append(message)
            mismatches.append(
                {"key": key, "expected": None, "actual": actual_value, "reason": message}
            )
            continue
        if actual_value is None:
            message = f"runtime fact {key} is unavailable; expected {expected_value!r}"
            errors.append(message)
            mismatches.append(
                {"key": key, "expected": expected_value, "actual": None, "reason": message}
            )
        elif str(actual_value) != str(expected_value):
            message = (
                f"runtime fact {key} mismatch: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )
            errors.append(message)
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "actual": actual_value,
                    "reason": message,
                }
            )
        if key in lock and str(expected_value) != str(lock[key]):
            message = (
                f"runtime config/lock mismatch for {key}: "
                f"{expected_value!r} != {lock[key]!r}"
            )
            errors.append(message)
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "lock": lock[key],
                    "actual": actual_value,
                    "reason": message,
                }
            )
    return {
        "pass": not errors,
        "expected": expected,
        "actual": actual,
        "errors": errors,
        "mismatches": mismatches,
    }


def _validate_runtime_facts(config: Mapping[str, Any], facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Require installed facts to match both config pins and the lock file."""

    if config.get("strict_provenance") is not True:
        return dict(facts if facts is not None else _runtime_facts(config))
    actual = dict(facts if facts is not None else _runtime_facts(config))
    audit = _runtime_fact_audit(config, actual)
    if not audit["pass"]:
        raise M1ConfigError("runtime version/fact audit failed: " + "; ".join(audit["errors"]))
    return actual


def _provenance(
    config: Mapping[str, Any],
    *,
    tape_hash: str,
    runtime_audit: Mapping[str, Any] | None = None,
    mujoco_state: Mapping[str, Any] | None = None,
    command: Sequence[str] | None = None,
    worktree_audit: Mapping[str, Any] | None = None,
    runtime_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _mapping(config.get("runtime", {}), "runtime")
    pins = dict(_mapping(config.get("pins", {}), "pins"))
    task = _mapping(config.get("task", {}), "task")
    renderer = runtime.get("renderer", {}) if isinstance(runtime.get("renderer"), Mapping) else {}
    strict = config.get("strict_provenance") is True
    # Validate runtime facts before worktree/hash checks so a wrong installed
    # package can never be hidden by an unrelated dirty-worktree failure.
    observed_runtime_facts = _validate_runtime_facts(config, facts=runtime_facts)
    runtime_facts_gate = (
        _runtime_fact_audit(config, observed_runtime_facts)
        if strict
        else {
            "pass": True,
            "expected": {},
            "actual": observed_runtime_facts,
            "errors": [],
            "mismatches": [],
            "authoritative": False,
        }
    )
    init_evidence = config.get("init_state_id_evidence")
    if strict and init_evidence != 0:
        raise M1ConfigError(
            "init_state_id_evidence must preserve the requested pre-reset selection 0"
        )
    environment_contract = (
        _runtime_environment_contract(config)
        if strict
        else dict(runtime.get("environment", {})) if isinstance(runtime.get("environment"), Mapping) else {}
    )
    source_checkout_audit = _source_checkout_audit(config)
    worktree_evidence = dict(worktree_audit) if worktree_audit is not None else _worktree_audit(config)
    if strict and not source_checkout_audit.get("pass"):
        raise M1ConfigError(
            "source checkout audit failed: "
            + "; ".join(str(item) for item in source_checkout_audit.get("errors", ()))
        )
    if strict and not worktree_evidence.get("pass"):
        raise M1ConfigError(
            "root worktree audit failed: "
            + "; ".join(str(item) for item in worktree_evidence.get("errors", ()))
        )
    hash_audit = _hash_audit(config)
    command_value = list(command) if command is not None else [sys.executable, *sys.argv]
    state_evidence = copy.deepcopy(dict(mujoco_state or {}))
    state_evidence.setdefault("version", observed_runtime_facts["mujoco"])
    state_evidence.setdefault("mask_name", INTEGRATION_SPEC_NAME)
    return {
        "m0_closure_sha256": str(config.get("m0_closure_sha256", M0_CLOSURE_SHA256)),
        "m1_git_sha": _git_sha(),
        "git_sha": _git_sha(),
        "trajectory_id": config.get("trajectory_id", "libero_spatial-task000-init000"),
        "suite": task.get("suite"),
        "task_id": task.get("task_id"),
        "seed": task.get("seed"),
        "configured_init_state": task.get("init_state_id"),
        "source_steps": config.get("source_steps", ACTION_TAPE_STEPS),
        "configured_step": config.get("source_steps", ACTION_TAPE_STEPS),
        "action_generation": {
            "algorithm": config.get("action_tape", {}).get("algorithm"),
            "bit_generator": config.get("action_tape", {}).get("bit_generator"),
            "seed": config.get("action_tape", {}).get("seed"),
            "configured_init_state": task.get("init_state_id"),
        },
        "action_tape_sha256": tape_hash,
        "pins": pins,
        "checkpoint": copy.deepcopy(config.get("checkpoint")),
        "base_model": copy.deepcopy(config.get("base_model")),
        "assets": copy.deepcopy(config.get("assets")),
        "runtime_facts": observed_runtime_facts,
        "runtime_facts_gate": runtime_facts_gate,
        "mujoco_state": state_evidence,
        "runtime_audit": copy.deepcopy(dict(runtime_audit or {})),
        "source_checkout_audit": source_checkout_audit,
        "worktree_audit": worktree_evidence,
        "renderer": copy.deepcopy(dict(renderer)),
        "environment": {
            "offline": runtime.get("offline"),
            "local_only": runtime.get("local_only"),
            "include_policy": runtime.get("include_policy"),
            "call_policy": runtime.get("call_policy"),
            "call_processors": runtime.get("call_processors"),
            "use_async_envs": runtime.get("use_async_envs"),
            "n_envs": runtime.get("n_envs"),
            "command": command_value,
            "env": {
                key: os.environ.get(key, environment_contract.get(key))
                for key in sorted(environment_contract)
                if key != "empty_hf_cache"
            },
            "configured": copy.deepcopy(environment_contract),
        },
        "input_hash_audit": hash_audit,
        "config": copy.deepcopy(config.get("provenance", {})),
        "allowed_dirty_evidence": copy.deepcopy(config.get("allowed_dirty_evidence", {})),
        "init_state_id_evidence": {
            "requested": int(init_evidence) if init_evidence is not None else None,
            "selected_pre_reset": int(init_evidence) if init_evidence is not None else None,
            "post_reset_cursor": int(init_evidence) + 1 if init_evidence is not None else None,
        },
    }


def create_run_directory(root: str | Path, run_id: str | None = None) -> Path:
    target_root = Path(root)
    target_root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    target = target_root / run_id
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise PublicationError(f"run directory already exists: {target}") from exc
    return target


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_create_text(path, canonical_json(value) + "\n")


def _load_json_mapping(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    _require_regular_path(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise M1SchemaError(f"could not load JSON artifact {target}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise M1SchemaError(f"JSON artifact must be a mapping: {target}")
    return dict(value)


def _journal(run_directory: Path, event: Mapping[str, Any]) -> None:
    journal_dir = run_directory / "journal"
    journal_dir.mkdir(exist_ok=True)
    event_id = f"{len(tuple(journal_dir.glob('*.json'))):06d}_{uuid.uuid4().hex[:8]}"
    _write_json_new(journal_dir / f"{event_id}.json", dict(event))


def _publish_artifact_manifest(run_directory: Path, config: Mapping[str, Any]) -> Path:
    """Publish immutable hashes for every completed run artifact."""

    files: list[dict[str, Any]] = []
    for path in sorted(run_directory.rglob("*"), key=lambda item: str(item.relative_to(run_directory))):
        if not path.is_file() or path.name in {"artifact_manifest.json", "terminal_manifest.json"}:
            continue
        relative = str(path.relative_to(run_directory)).replace(os.sep, "/")
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    script_path = ROOT / "scripts" / "m1_state_replay.py"
    config_path = _resolve_path(_mapping(config.get("paths", {}), "paths").get("config", DEFAULT_CONFIG_PATH))
    payload = {
        "schema_version": 1,
        "files": files,
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "config": {
            "path": str(config_path),
            "sha256": _config_contract_sha256_file(config_path) if config_path.is_file() else None,
            "raw_sha256": sha256_file(config_path) if config_path.is_file() else None,
        },
        "m0_closure_sha256": str(config.get("m0_closure_sha256", M0_CLOSURE_SHA256)),
        "input_hash_audit": _hash_audit(config),
    }
    target = run_directory / "artifact_manifest.json"
    _write_json_new(target, payload)
    return target


def _contains_pending(value: Any) -> bool:
    if isinstance(value, str):
        return "pending" in value.lower()
    if isinstance(value, Mapping):
        return any(_contains_pending(key) or _contains_pending(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_pending(child) for child in value)
    return False


def _gate_summary(
    manifest: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    *,
    persisted_serialization: bool,
    authoritative: bool,
) -> dict[str, Any]:
    static_rows = [row for attempt in attempts for row in attempt.get("static", ())]
    dynamic_rows = [row for attempt in attempts for row in attempt.get("rows", ())]
    counts = manifest.get("counts", {})
    provenance = manifest.get("provenance", {})
    input_hash_audit = provenance.get("input_hash_audit", {}) if isinstance(provenance, Mapping) else {}
    static_pass = len(static_rows) == 9 and all(
        bool(row.get("gate", {}).get("pass", False))
        and bool(row.get("immediate_readback", {}).get("exact", False))
        for row in static_rows
    )
    dynamic_pass = len(dynamic_rows) == 90 and all(
        bool(row.get("gate", {}).get("pass", False)) for row in dynamic_rows
    )
    serialized_contract = (
        provenance.get("serialized_runtime_contract", {})
        if isinstance(provenance, Mapping)
        else {}
    )
    serialization_pass = bool(
        persisted_serialization
        and isinstance(serialized_contract, Mapping)
        and serialized_contract.get("pass") is True
        and _serialization_coverage_contract_pass(
            serialized_contract.get("coverage"),
            serialized_contract.get("classified", ()),
            require_official_observables=bool(manifest.get("authoritative")),
        )
    )
    classification = bool(
        isinstance(provenance, Mapping)
        and isinstance(provenance.get("runtime_audit"), Mapping)
        and provenance.get("runtime_audit", {}).get("classification")
        and provenance.get("runtime_audit", {}).get("field_classification")
        and not provenance.get("runtime_audit", {}).get("unknown_paths")
    )
    def reset_evidence_pass(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        expected = value.get("requested")
        return (
            expected == 0
            and value.get("selected_pre_reset") == expected
            and value.get("post_reset_cursor") == expected + 1
        )

    source_reset = provenance.get("source_reset_provenance", {}) if isinstance(provenance, Mapping) else {}
    reset_pass = reset_evidence_pass(
        source_reset.get("init_state_id_evidence") if isinstance(source_reset, Mapping) else None
    ) and bool(attempts) and all(
        reset_evidence_pass(
            attempt.get("reset_provenance", {}).get("init_state_id_evidence")
            if isinstance(attempt.get("reset_provenance"), Mapping)
            else None
        )
        for attempt in attempts
    )
    runtime_facts_pass = bool(
        isinstance(provenance, Mapping)
        and isinstance(provenance.get("runtime_facts_gate"), Mapping)
        and provenance.get("runtime_facts_gate", {}).get("pass") is True
    )
    provenance_pass = bool(
        isinstance(input_hash_audit, Mapping)
        and input_hash_audit.get("pass") is True
        and not _contains_pending(provenance)
        and runtime_facts_pass
        and reset_pass
        and isinstance(provenance.get("mujoco_state"), Mapping)
        and all(
            provenance["mujoco_state"].get(key) is not None
            for key in ("mask", "size", "component_layout")
        )
    )
    orchestration = bool(
        counts.get("source_trajectories") == 1
        and counts.get("restore_attempts") == 9
        and counts.get("static_comparisons") == 9
        and counts.get("dynamic_comparisons") == 90
        and len({attempt.get("attempt_id") for attempt in attempts}) == 9
    )
    hash_pass = bool(manifest.get("artifact_manifest_sha256"))
    authoritative_pass = bool(
        authoritative
        and manifest.get("config", {}).get("strict_provenance") is True
        and classification
        and provenance_pass
        and isinstance(provenance.get("source_checkout_audit"), Mapping)
        and provenance.get("source_checkout_audit", {}).get("pass") is True
        and isinstance(provenance.get("worktree_audit"), Mapping)
        and provenance.get("worktree_audit", {}).get("pass") is True
        and isinstance(provenance.get("post_process_proof"), Mapping)
        and provenance.get("post_process_proof", {}).get("passed") is True
        and bool(manifest.get("rgb_source"))
        and isinstance(provenance.get("renderer_live"), Mapping)
    )
    gates = {
        "A_static": {"pass": static_pass, "count": len(static_rows)},
        "B_dynamic": {"pass": dynamic_pass, "count": len(dynamic_rows)},
        "C_classification": {"pass": classification},
        "D_serialization": {"pass": serialization_pass},
        "E_provenance": {"pass": provenance_pass},
        "F_orchestration": {"pass": orchestration and reset_pass},
        "G_reset_provenance": {"pass": reset_pass},
        "hash_audit": {"pass": hash_pass},
        "authoritative": {"pass": authoritative_pass},
        # Stable short aliases make the terminal schema easy to consume while
        # retaining the explicit A-F names used in the protocol.
        "static": static_pass,
        "dynamic": dynamic_pass,
        "classification": classification,
        "serialization": serialization_pass,
        "provenance": provenance_pass,
        "orchestration": orchestration,
    }
    return gates


def _is_terminated(result: Any) -> bool:
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) >= 5:
            return bool(result[2]) or bool(result[3])
        if len(result) >= 4:
            return bool(result[2])
    return False


def run_source_trajectory(
    adapter: RuntimeAdapter,
    tape: np.ndarray,
    windows: Sequence[CaptureWindow],
    *,
    rgb_directory: str | Path | None = None,
) -> tuple[dict[int, ReplayState], dict[int, dict[str, Any]]]:
    """Run one source exactly once and retain post-step windows."""

    captures: dict[int, ReplayState] = {}
    references: dict[int, dict[str, Any]] = {}
    capture_steps = {window.capture_step for window in windows}
    for index, action in enumerate(validate_action_tape(tape)):
        result = adapter.step(action)
        step = index + 1
        # The capture occurs only after the official step has returned.
        if step in capture_steps:
            captures[step] = adapter.capture(source_step=step)
            if rgb_directory is not None:
                adapter.capture_rgb(
                    f"source_{step}",
                    Path(rgb_directory) / f"source_{step}.npy",
                )
        if any(step in window.state_steps for window in windows):
            references[step] = adapter.collect_invariants()
        if _is_terminated(result) and step < ACTION_TAPE_STEPS:
            raise EarlyTerminationError(f"source terminated before required step {ACTION_TAPE_STEPS}: {step}")
    if adapter.step_count != ACTION_TAPE_STEPS:
        raise EarlyTerminationError(
            f"source ended at step {adapter.step_count}, required {ACTION_TAPE_STEPS}"
        )
    if set(captures) != capture_steps or any(step not in references for window in windows for step in window.state_steps):
        raise EarlyTerminationError("source did not produce every required capture/reference state")
    return captures, references


def compare_restore_window(
    adapter: RuntimeAdapter,
    tape: np.ndarray,
    capture: ReplayState,
    reference_states: Mapping[int, Mapping[str, Any]],
    window: CaptureWindow,
    *,
    rgb_directory: str | Path | None = None,
    source_rgb: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    restored = adapter.restore(
        capture,
        static_reference=reference_states.get(window.capture_step),
    )
    static_gate = restored.get("static", {}).get(
        "gate",
        {"pass": False, "error": "static reference was not supplied"},
    )
    rgb_result: dict[str, Any] | None = None
    if rgb_directory is not None:
        label = f"restored_{window.capture_step}"
        restored_rgb = adapter.capture_rgb(
            label,
            Path(rgb_directory) / f"{label}.npy",
        )
        source_record = (source_rgb or {}).get(f"source_{window.capture_step}")
        if restored_rgb is not None and isinstance(source_record, Mapping):
            source_path = source_record.get("path")
            if isinstance(source_path, str):
                source_image = safe_load_uint8_npy(source_path)
                restored_image = safe_load_uint8_npy(restored_rgb["path"])
                rgb_comparison = compare_rgb(
                    source_image,
                    restored_image,
                    path=f"rgb.capture_{window.capture_step}",
                )
                rgb_result = {
                    "source": copy.deepcopy(dict(source_record)),
                    "restored": copy.deepcopy(dict(restored_rgb)),
                    "comparison": comparison_result_to_dict(rgb_comparison),
                }
        elif restored_rgb is not None or source_record is not None:
            rgb_result = {
                "source": copy.deepcopy(dict(source_record)) if isinstance(source_record, Mapping) else None,
                "restored": copy.deepcopy(dict(restored_rgb)) if restored_rgb is not None else None,
                "comparison": comparison_result_to_dict(
                    compare_diagnostic(
                        source_record,
                        restored_rgb,
                        path=f"rgb.capture_{window.capture_step}",
                    )
                ),
            }
    static_row = {
        "step": window.capture_step,
        "action_index": None,
        "terminated": bool(adapter.last_terminated),
        "truncated": bool(adapter.last_truncated),
        "done": bool(_counter_state(adapter.inner, adapter.env).get("done", False)),
        "reward": adapter.last_reward,
        "success": adapter._check_success(),
        "comparisons": restored.get("static", {}).get("comparisons", []),
        "gate": static_gate,
        "immediate_readback": restored["immediate_readback"],
        "serialized_runtime_readback": restored.get("serialized_runtime_readback", {}),
    }
    if rgb_result is not None:
        static_row["rgb"] = rgb_result
    rows: list[dict[str, Any]] = []
    for action_index, state_step in zip(window.action_indices, window.state_steps[1:]):
        result = adapter.step(tape[action_index])
        actual = adapter.collect_invariants()
        comparisons = compare_invariants(reference_states[state_step], actual)
        counters = actual.get("counters", {})
        rows.append(
            {
                "step": state_step,
                "action_index": action_index,
                "terminated": _is_terminated(result),
                "truncated": bool(adapter.last_truncated),
                "done": bool(counters.get("done", False)),
                "reward": actual.get("reward"),
                "success": actual.get("success"),
                "comparisons": [
                    comparison_result_to_dict(item)
                    for item in comparisons
                ],
                "gate": aggregate_gate(comparisons),
            }
        )
    return {
        "capture_step": window.capture_step,
        "invariants": restored.get("invariants", {}),
        "static": [static_row],
        "rows": rows,
        "serialized_runtime_readback": restored.get("serialized_runtime_readback", {}),
        "pass": bool(static_gate.get("pass", False)) and all(row["gate"]["pass"] for row in rows),
    }


def run_experiment(
    config: Mapping[str, Any],
    *,
    runtime_builder: Callable[[Mapping[str, Any]], Mapping[str, Any] | Any] | None = None,
    run_directory: str | Path | None = None,
    tape: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute the exact one-source/3x3x10 replay when explicitly requested."""

    normalized = validate_config(config)
    paths = _mapping(normalized["paths"], "paths")
    # A caller-supplied directory is an immutable publication target.  Reject
    # it before tape generation, factory invocation, or any artifact write.
    if run_directory is not None:
        run_dir = Path(run_directory)
        if run_dir.exists():
            raise PublicationError(f"caller-supplied run directory already exists: {run_dir}")
    # The empty run directory is not a worktree mutation.  Capture this audit
    # before the first manifest/journal file so generated M1 output cannot
    # make its own dirty-worktree evidence self-defeating.
    pre_output_worktree_audit = (
        _worktree_audit(normalized) if normalized.get("strict_provenance") is True else None
    )
    if run_directory is not None:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_dir.mkdir()
        except FileExistsError as exc:
            raise PublicationError(f"run directory already exists: {run_dir}") from exc
    else:
        run_dir = create_run_directory(paths["output_root"])
    terminal_path = run_dir / "terminal_manifest.json"
    command = [sys.executable, *sys.argv]
    authority_requested = runtime_builder is None and normalized.get("strict_provenance") is True
    # Authority is granted only after every preflight/provenance gate passes.
    # A strict invocation with a wrong installed runtime must publish a
    # terminal BLOCKED manifest with ``authoritative=false``.
    authoritative = False
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "config": normalized,
        "authoritative": authoritative,
        "provenance": {},
        "counts": _CountsDict({
            "source_trajectories": 0,
            "restore_attempts": 0,
            "static_comparisons": 0,
            "dynamic_comparisons": 0,
        }),
    }
    tape_hash: str | None = None
    expected_hash = normalized.get("action_tape", {}).get("sha256")
    source: RuntimeAdapter | None = None
    attempts: list[dict[str, Any]] = []
    persisted_serialization = False
    status = "BLOCKED"
    error: str | None = None
    primary_exception: BaseException | None = None
    cleanup_exception: BaseException | None = None
    cleanup_failures: list[str] = []
    artifact_manifest_path: Path | None = None
    try:
        # Publish the attempt journal immediately after the protected output
        # directory is created.  Keeping this inside the guarded block means
        # even an early write failure gets a terminal publication attempt.
        _write_json_new(run_dir / "run_manifest.json", manifest)
        _journal(run_dir, {"event": "run_started", "run_directory": str(run_dir), "source_trajectories": 0})
        _atomic_create_text(run_dir / "command.txt", canonical_json(command) + "\n")
        _apply_runtime_environment(normalized)
        # Retain the pre-output audit before provenance validation can fail.
        # In particular, a blocked authoritative invocation must still publish
        # the complete status list and hashes for unrelated dirty files; an
        # exception from ``_provenance`` must not erase that forensic record.
        if pre_output_worktree_audit is not None:
            manifest["provenance"]["worktree_audit"] = copy.deepcopy(
                pre_output_worktree_audit
            )
        observed_runtime_facts: Mapping[str, Any] | None = None
        if normalized.get("strict_provenance") is True:
            try:
                observed_runtime_facts = _runtime_facts(normalized)
                runtime_facts_gate = _runtime_fact_audit(
                    normalized, observed_runtime_facts
                )
            except Exception as exc:
                observed_runtime_facts = {}
                runtime_facts_gate = {
                    "pass": False,
                    "expected": _runtime_fact_expectations(normalized),
                    "actual": {},
                    "errors": [
                        f"runtime fact collection failed: {type(exc).__name__}: {exc}"
                    ],
                    "mismatches": [],
                }
            manifest["provenance"]["runtime_facts"] = copy.deepcopy(
                dict(observed_runtime_facts)
            )
            manifest["provenance"]["runtime_facts_gate"] = copy.deepcopy(
                runtime_facts_gate
            )
            if not runtime_facts_gate.get("pass"):
                raise M1ConfigError(
                    "runtime version/fact audit failed: "
                    + "; ".join(str(item) for item in runtime_facts_gate.get("errors", ()))
                )
        # Generate exactly once, persist before constructing any environment,
        # and then reload the immutable bytes exactly once for all consumers.
        if tape is None:
            tape = generate_action_tape()
        else:
            tape = validate_action_tape(tape)
        tape_hash = action_tape_sha256(tape)
        if expected_hash and tape_hash != str(expected_hash).lower():
            raise M1ConfigError("configured action tape hash does not match generated tape")
        manifest["provenance"] = _provenance(
            normalized,
            tape_hash=tape_hash,
            command=command,
            worktree_audit=pre_output_worktree_audit,
            runtime_facts=observed_runtime_facts,
        )
        authoritative = authority_requested
        manifest["authoritative"] = authoritative
        tape_path = run_dir / "action_tape.npy"
        persist_action_tape_exact(tape_path, tape, expected_sha256=expected_hash)
        _atomic_create_text(run_dir / "action_tape.sha256", tape_hash + "\n")
        tape = load_action_tape(tape_path, expected_sha256=tape_hash)
        windows = build_capture_windows(
            normalized["capture_steps"],
            future_steps=int(normalized["future_steps"]),
            total_steps=int(normalized.get("source_steps", ACTION_TAPE_STEPS)),
        )
        source = RuntimeAdapter.construct_fresh(
            normalized, runtime_builder=runtime_builder, tape_hash=tape_hash
        )
        manifest["provenance"]["runtime_audit"] = copy.deepcopy(source.runtime_audit)
        classified_serialized_paths = _serialized_classified_paths(source.runtime_audit)
        serialization_coverage = source.runtime_audit.get("serialization_coverage", {})
        if not isinstance(serialization_coverage, Mapping):
            raise M1SchemaError("runtime audit is missing serialization_coverage mapping")
        concrete_coverage = serialization_coverage.get("concrete", {})
        if not isinstance(concrete_coverage, Mapping):
            raise M1SchemaError("runtime serialization coverage concrete map is invalid")
        concrete_coverage_paths = tuple(sorted(str(path) for path in concrete_coverage))
        if concrete_coverage_paths != classified_serialized_paths:
            raise M1SchemaError(
                "runtime serialization coverage does not equal classified SERIALIZED paths: "
                f"classified={classified_serialized_paths!r}, coverage={concrete_coverage_paths!r}"
            )
        serialized_contract: dict[str, Any] = {
            "classified": list(classified_serialized_paths),
            "coverage": copy.deepcopy(dict(serialization_coverage)),
            "captures": {},
            "persisted_metadata": {},
            "persisted": {},
            "restore_readbacks": {},
            "restore_invariants": {},
            "pass": False,
        }
        manifest["provenance"]["serialized_runtime_contract"] = serialized_contract
        manifest["provenance"]["post_process_proof"] = copy.deepcopy(source.post_process_proof)
        manifest["provenance"]["source_reset_provenance"] = copy.deepcopy(source.reset_provenance)
        manifest["provenance"]["mujoco_state"] = _mujoco_state_evidence(
            source.mujoco, source.model, source.data
        )
        captures, references = run_source_trajectory(
            source,
            tape,
            windows,
            rgb_directory=run_dir / "rgb" / "source",
        )
        manifest["rgb_source"] = copy.deepcopy(source.rgb_records)
        manifest["provenance"]["renderer_live"] = copy.deepcopy(source.gl_identity)
        if normalized.get("strict_provenance") and not source.rgb_records:
            raise M1RuntimeError("strict M1 runtime produced no persisted source RGB captures")
        manifest["counts"]["source_trajectories"] = 1

        # Persist every capture/reference first, then reload those artifacts;
        # no restore is permitted to consume only an in-memory source object.
        persisted_captures: dict[int, ReplayState] = {}
        persisted_references: dict[int, dict[str, Any]] = {}
        for window in windows:
            capture_step = window.capture_step
            capture_state = captures[capture_step]
            capture_paths = tuple(sorted(str(path) for path in capture_state.serialized_runtime))
            serialized_contract["captures"][str(capture_step)] = list(capture_paths)
            if capture_paths != classified_serialized_paths:
                raise M1SchemaError(
                    "captured SERIALIZED path set differs from classified set at "
                    f"step {capture_step}: expected {classified_serialized_paths!r}, got {capture_paths!r}"
                )
            capture_record = replay_state_record(capture_state)
            save_record_bundle(
                capture_record,
                run_dir,
                f"capture_{capture_step}",
            )
            loaded_capture_record = load_record_bundle(run_dir, f"capture_{capture_step}")
            metadata_runtime = loaded_capture_record.metadata.get("serialized_runtime", {})
            if not isinstance(metadata_runtime, Mapping):
                raise M1SchemaError(
                    f"persisted capture metadata has no serialized_runtime mapping at step {capture_step}"
                )
            metadata_paths = tuple(sorted(str(path) for path in metadata_runtime))
            serialized_contract["persisted_metadata"][str(capture_step)] = list(metadata_paths)
            if metadata_paths != classified_serialized_paths:
                raise M1SchemaError(
                    "persisted capture metadata SERIALIZED path set differs from classified set at "
                    f"step {capture_step}: expected {classified_serialized_paths!r}, got {metadata_paths!r}"
                )
            persisted_captures[capture_step] = replay_state_from_record(loaded_capture_record)
            persisted_paths = tuple(sorted(str(path) for path in persisted_captures[capture_step].serialized_runtime))
            serialized_contract["persisted"][str(capture_step)] = list(persisted_paths)
            if persisted_paths != classified_serialized_paths or not _exact_equal(
                capture_state.serialized_runtime,
                persisted_captures[capture_step].serialized_runtime,
            ):
                raise M1SchemaError(
                    "persisted capture SERIALIZED state differs from source capture "
                    f"at step {capture_step}"
                )
            reference_payload = {
                str(step): references[step] for step in window.state_steps
            }
            reference_path = run_dir / f"reference_{capture_step}.json"
            _write_json_new(reference_path, reference_payload)
            loaded_reference = _load_json_mapping(reference_path)
            for step in window.state_steps:
                key = str(step)
                if key not in loaded_reference or not isinstance(loaded_reference[key], Mapping):
                    raise M1SchemaError(f"persisted reference is missing state step {step}")
                persisted_references[step] = dict(loaded_reference[key])
        persisted_serialization = (
            len(persisted_captures) == len(windows)
            and all(step in persisted_references for window in windows for step in window.state_steps)
            and all(
                serialized_contract["captures"].get(str(window.capture_step))
                == list(classified_serialized_paths)
                and serialized_contract["persisted_metadata"].get(str(window.capture_step))
                == list(classified_serialized_paths)
                and serialized_contract["persisted"].get(str(window.capture_step))
                == list(classified_serialized_paths)
                for window in windows
            )
        )
        if not persisted_serialization:
            raise M1SchemaError("persisted capture/reference serialization is incomplete")
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                cleanup_failures.append(f"source: {type(exc).__name__}: {exc}")
                raise
            else:
                source = None

        for window in windows:
            for ordinal in range(1, RESTORES_PER_CAPTURE + 1):
                attempt_id = f"capture_{window.capture_step}_restore_{ordinal}"
                adapter: RuntimeAdapter | None = None
                attempt_exception: BaseException | None = None
                try:
                    adapter = RuntimeAdapter.construct_fresh(
                        normalized, runtime_builder=runtime_builder, tape_hash=tape_hash
                    )
                    result = compare_restore_window(
                        adapter,
                        tape,
                        persisted_captures[window.capture_step],
                        persisted_references,
                        window,
                        rgb_directory=run_dir / "rgb" / attempt_id,
                        source_rgb=manifest.get("rgb_source", {}),
                    )
                    restored_readback = result.get("serialized_runtime_readback")
                    if not isinstance(restored_readback, Mapping):
                        raise M1SchemaError(
                            f"restore {attempt_id} did not publish serialized runtime readback"
                        )
                    restored_paths = tuple(sorted(str(path) for path in restored_readback))
                    invariant_runtime = result.get("invariants", {}).get("serialized_runtime")
                    if not isinstance(invariant_runtime, Mapping):
                        raise M1SchemaError(
                            f"restore {attempt_id} did not publish serialized runtime invariants"
                        )
                    invariant_paths = tuple(sorted(str(path) for path in invariant_runtime))
                    serialized_contract["restore_readbacks"][attempt_id] = list(restored_paths)
                    serialized_contract["restore_invariants"][attempt_id] = list(invariant_paths)
                    if restored_paths != classified_serialized_paths or invariant_paths != classified_serialized_paths:
                        raise M1SchemaError(
                            f"restore {attempt_id} SERIALIZED path set differs from classified set"
                        )
                    result["reset_provenance"] = copy.deepcopy(adapter.reset_provenance)
                except Exception as exc:
                    attempt_exception = exc
                    raise
                finally:
                    if adapter is not None:
                        try:
                            adapter.close()
                        except Exception as exc:
                            cleanup_failures.append(
                                f"{attempt_id}: {type(exc).__name__}: {exc}"
                            )
                            if attempt_exception is not None:
                                attempt_exception.add_note(
                                    f"fresh environment cleanup failed: {type(exc).__name__}: {exc}"
                                )
                            else:
                                raise
                result["attempt_id"] = attempt_id
                attempts.append(result)
                manifest["counts"]["restore_attempts"] += 1
                manifest["counts"]["static_comparisons"] += len(result.get("static", ()))
                manifest["counts"]["dynamic_comparisons"] += len(result.get("rows", ()))
                _journal(
                    run_dir,
                    {"event": "restore_complete", "attempt_id": attempt_id, "pass": result["pass"]},
                )
        if len(attempts) != len(windows) * RESTORES_PER_CAPTURE:
            raise M1RuntimeError("restore attempt count is not exactly nine")
        if manifest["counts"]["static_comparisons"] != 9 or manifest["counts"]["dynamic_comparisons"] != 90:
            raise M1RuntimeError("M1 comparison counts are not exactly static=9 and dynamic=90")
        serialized_contract["pass"] = bool(
            persisted_serialization
            and set(serialized_contract["classified"])
            and _serialization_coverage_contract_pass(
                serialized_contract.get("coverage"),
                serialized_contract["classified"],
                require_official_observables=bool(manifest.get("authoritative")),
            )
            and all(
                serialized_contract["restore_readbacks"].get(attempt["attempt_id"])
                == serialized_contract["classified"]
                and serialized_contract["restore_invariants"].get(attempt["attempt_id"])
                == serialized_contract["classified"]
                for attempt in attempts
            )
        )
        persisted_serialization = bool(persisted_serialization and serialized_contract["pass"])
        if not persisted_serialization:
            raise M1SchemaError(
                "classified SERIALIZED paths are not equal across capture, persistence, "
                "restore readback, and invariant comparison"
            )
        _write_json_new(run_dir / "restore_results.json", {"attempts": attempts})
        artifact_manifest_path = _publish_artifact_manifest(run_dir, normalized)
        manifest["artifact_manifest_sha256"] = sha256_file(artifact_manifest_path)
        manifest["artifact_manifest"] = _load_json_mapping(artifact_manifest_path)
        manifest["provenance"]["artifact_manifest_sha256"] = manifest["artifact_manifest_sha256"]
        manifest["provenance"]["script_sha256"] = sha256_file(ROOT / "scripts" / "m1_state_replay.py")
        config_path = _resolve_path(paths.get("config", DEFAULT_CONFIG_PATH))
        if config_path.is_file():
            manifest["provenance"]["config_sha256"] = _config_contract_sha256_file(config_path)
            manifest["provenance"]["config_raw_sha256"] = sha256_file(config_path)
        manifest["provenance"]["action_tape_sha256"] = tape_hash
        manifest["attempt_ids"] = [attempt["attempt_id"] for attempt in attempts]
        manifest["gates"] = _gate_summary(
            manifest,
            attempts,
            persisted_serialization=persisted_serialization,
            authoritative=authoritative,
        )
        status = "PASS" if all(
            isinstance(value, Mapping) and value.get("pass") is True
            for key, value in manifest["gates"].items()
            if key.startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_")) or key == "hash_audit"
        ) and manifest["gates"]["authoritative"]["pass"] else (
            "TEST_ONLY" if not authoritative and all(
                isinstance(value, Mapping) and value.get("pass") is True
                for key, value in manifest["gates"].items()
                if key.startswith(("A_", "B_", "C_", "D_", "E_", "F_", "G_")) or key == "hash_audit"
            ) else "BLOCKED"
        )
        manifest["status"] = status
    except Exception as exc:
        primary_exception = exc
        error = f"{type(exc).__name__}: {exc}; run_directory={run_dir}"
        manifest["error"] = error
        status = "BLOCKED"
        # Build the gates before re-raising so an early preflight mismatch is
        # still a publishable, machine-readable terminal result.
        try:
            manifest["gates"] = _gate_summary(
                manifest,
                attempts,
                persisted_serialization=persisted_serialization,
                authoritative=False,
            )
        except Exception as gate_error:
            manifest["gates"] = {
                "E_provenance": {
                    "pass": False,
                    "error": f"gate construction failed: {type(gate_error).__name__}: {gate_error}",
                },
                "authoritative": {"pass": False},
            }
        raise
    finally:
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                cleanup_exception = exc
                cleanup_failures.append(f"source: {type(exc).__name__}: {exc}")
                if primary_exception is None:
                    status = "BLOCKED"
                    error = f"{type(exc).__name__}: {exc}; run_directory={run_dir}"
                    manifest["error"] = error
        manifest["cleanup_errors"] = "; ".join(cleanup_failures)
        manifest["status"] = status
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        if error is not None:
            manifest["error"] = error
        # Publication errors intentionally escape this finally block; a
        # terminal manifest that cannot be atomically created is itself a
        # failed publication and must be visible to the caller.
        try:
            _write_json_new(terminal_path, manifest)
        except Exception as exc:
            publication_error = PublicationError(
                f"terminal manifest publication failed for run_directory={run_dir}: {exc}"
            )
            # Do not mask the first runtime failure with cleanup/publication
            # failures.  Attach the latter to the original exception so the
            # caller and traceback still receive both pieces of evidence.
            if primary_exception is not None:
                primary_exception.add_note(str(publication_error))
            else:
                if cleanup_exception is not None:
                    publication_error.add_note(
                        f"cleanup also failed: {type(cleanup_exception).__name__}: {cleanup_exception}"
                    )
                raise publication_error from exc
        if primary_exception is None and cleanup_exception is not None:
            raise cleanup_exception
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--validate-only", action="store_true", help="validate config and tape without constructing LIBERO")
    parser.add_argument("--run", action="store_true", help="opt in to the real one-source/9-restore experiment")
    args = parser.parse_args(argv)
    try:
        config = validate_config(load_config(args.config))
        tape = generate_action_tape()
        expected_hash = config.get("action_tape", {}).get("sha256")
        actual_hash = action_tape_sha256(tape)
        if expected_hash and str(expected_hash).lower() != actual_hash:
            raise M1ConfigError("configured tape hash does not match deterministic tape")
        if args.validate_only or not args.run:
            print(canonical_json({"status": "VALIDATED", "action_tape_sha256": actual_hash}))
            return 0
        manifest = run_experiment(config, tape=tape)
        print(canonical_json({"status": manifest["status"], "run": manifest.get("finished_at")}))
        return 0 if manifest["status"] == "PASS" else 2
    except Exception as exc:
        print(f"M1 BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI exercised separately
    raise SystemExit(main())


__all__ = [
    "ACTION_DIM",
    "ACTION_TAPE_SEED",
    "CaptureWindow",
    "ComparisonClass",
    "ComparisonResult",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_STATE_CLASSIFICATION",
    "DERIVED_RECONSTRUCTED",
    "EarlyTerminationError",
    "FLOATING_ATOL",
    "FLOATING_RTOL",
    "MODEL_FINGERPRINT_SCHEMA_VERSION",
    "REPLAY_STATE_SCHEMA_VERSION",
    "ForbiddenRestoreError",
    "IMMUTABLE",
    "M1ConfigError",
    "M1Error",
    "M1RuntimeError",
    "M1SchemaError",
    "MatchedStateRecord",
    "PROVEN_UNUSED",
    "POST_PROCESS_ALLOWLIST",
    "PublicationError",
    "REQUIRED_INVARIANT_PATHS",
    "ReplayState",
    "RESTORES_PER_CAPTURE",
    "RuntimeAdapter",
    "SERIALIZED",
    "UnknownMutableStateError",
    "action_tape_sha256",
    "aggregate_gate",
    "audit_mutable_state",
    "build_capture_windows",
    "build_official_runtime",
    "canonical_json",
    "canonicalize_contact_pairs",
    "capture_phase",
    "compare_contact_pairs",
    "compare_diagnostic",
    "compare_exact",
    "compare_floating",
    "compare_rgb",
    "compare_invariants",
    "create_run_directory",
    "generate_action_tape",
    "physics_model_fingerprint",
    "observation_model_fingerprint",
    "exact_model_fingerprint_equal",
    "model_fingerprint_equal",
    "fixture_physics_fingerprint",
    "load_action_tape",
    "load_config",
    "load_record_bundle",
    "main",
    "model_layout_signature",
    "persist_action_tape",
    "persist_action_tape_exact",
    "replay_state_from_record",
    "replay_state_record",
    "run_experiment",
    "run_source_trajectory",
    "safe_load_npy",
    "safe_load_uint8_npy",
    "safe_save_npy",
    "safe_save_uint8_npy",
    "set_integration_state_exact",
    "sha256_bytes",
    "sha256_file",
    "validate_action_tape",
    "validate_config",
    "validate_invariant_snapshot",
    "validate_state_classification",
]
