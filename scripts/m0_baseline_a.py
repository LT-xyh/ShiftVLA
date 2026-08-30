#!/usr/bin/env python3
"""Frozen-policy M0 Baseline-A planning and rollout harness.

The public helpers in this module are intentionally runtime-light.  They
validate the immutable 15-episode matrix, record the official postprocessed
action contract, aggregate fail-closed evidence, and persist artifacts without
overwriting an earlier result.  Imports of LeRobot, LIBERO, MuJoCo, and the
SmolVLA policy are lazy and occur only in the real child execution path.

The normal CLI supports ``--validate-only`` for a hardware-free pre-run audit.
Real execution is guarded by the same config/manifest contracts and launches
two CPU children before waiting for either child; each child owns one
persistent nested DCU model worker.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

import numpy as np

# ``python scripts/m0_baseline_a.py`` places ``scripts/`` (rather than the
# repository root) on ``sys.path``.  Keep direct CLI execution equivalent to
# module execution without importing any runtime package here.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "m0" / "baseline_a.yaml"
SCHEMA_VERSION = 1
EXPECTED_NAME = "baseline_a"
EXPECTED_SEED = 2027
EXPECTED_SUITE = "libero_spatial"
EXPECTED_TASK_IDS = (0, 4, 9)
EXPECTED_INIT_STATE_IDS = (0, 1, 2, 3, 4)
EXPECTED_TASK_NAMES = {
    "0": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "4": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "9": "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}
EXPECTED_ACTION_DIM = 7
EXPECTED_CHUNK_SIZE = 50
EXPECTED_N_ACTION_STEPS = 1
EXPECTED_HORIZON = 280
EXPECTED_PHYSICAL_DEVICES = (0, 1)
EXPECTED_LOGICAL_DEVICE = "cuda:0"
EXPECTED_EGL_DEVICE_ID = "8"
EXPECTED_PREFLIGHT_CONFIG_SHA256 = "3ac615f60a254fe10968f7cf39d51f0d501bbc3b02992ac49fbdcbe2912ba058"
EXPECTED_PARENT_COLLECTION_TIMEOUT_SECONDS = 86400.0
EXPECTED_RENDERER = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MUJOCO_EGL_DEVICE_ID": EXPECTED_EGL_DEVICE_ID,
}
EXPECTED_OFFLINE = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}
EXPECTED_GL_IDENTITY = {
    "vendor": "Mesa/X.org",
    "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
    "version": "3.1 Mesa 21.1.5",
}
EXPECTED_PROVENANCE = {
    "lerobot_git_sha": "7e241bd630a3719a56157a497ce5d08f244784f1",
    "libero_git_sha": "8561c60eea2fb93096146f240194649df73d8b1e",
    "python": "3.11.16",
    "pytorch": "2.7.1+das.opt1.dtk25042",
    "transformers": "5.5.4",
    "cuda": "none-under-hip",
    "hip": "6.x-host-locked",
    "gpu": "K100_AI",
}
# Journal events are deliberately independent immutable files.  A directory
# gives every attempt-start/terminal decision its own atomic no-overwrite
# boundary; a single append-only file could be torn at an arbitrary byte.
EPISODE_JOURNAL_NAME = "episode_journal"
EPISODE_ROWS_NAME = "episode_rows"
WORKERS = ("A", "B")
TERMINAL_STATUSES = frozenset({"PASS", "FAIL", "TERMINATED", "CRASHED"})


class BaselineError(RuntimeError):
    """Base error for a failed-closed Baseline-A contract."""


class BaselineConfigError(BaselineError):
    """The supplied config is not the frozen Baseline-A config."""


class BaselineSchemaError(BaselineError):
    """An evidence record or artifact violates the Baseline-A schema."""


class BaselineRuntimeError(BaselineError):
    """A real runtime child failed and must not be retried."""


def _mapping(value: Any, name: str, error: type[BaselineError] = BaselineConfigError) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, error: type[BaselineError] = BaselineConfigError) -> Any:
    if key not in mapping:
        raise error(f"missing required config field {key}")
    return mapping[key]


def _string(value: Any, name: str, error: type[BaselineError] = BaselineConfigError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{name} must be a non-empty string")
    return value


def _int(value: Any, name: str, *, nonnegative: bool = False, error: type[BaselineError] = BaselineConfigError) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise error(f"{name} must be non-negative")
    return int(value)


def _sha(value: Any, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise BaselineConfigError(f"{name} must be a SHA-256 hex digest")
    return text.lower()


def _finite_nonnegative(value: Any, name: str, *, allow_none: bool = False, error: type[BaselineError] = BaselineSchemaError) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise error(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise error(f"{name} must be a finite non-negative number")
    return result


def _finite_number(value: Any, name: str, *, error: type[BaselineError] = BaselineSchemaError) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise error(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{name} must be a finite number")
    return result


def _strip_yaml_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in "'\"":
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _minimal_yaml_scalar(value: str) -> Any:
    value = _strip_yaml_comment(value.strip())
    if not value:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    try:
        return int(value, 10)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _minimal_yaml_load(text: str) -> Any:
    """Parse the small YAML subset used by project-owned config files.

    PyYAML remains preferred when available.  This fallback avoids adding a
    dependency to the frozen runtime and supports mappings, scalar lists, and
    inline JSON/Python-style lists/maps.
    """

    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if "\t" in raw[:indent]:
            raise BaselineConfigError("config indentation must use spaces")
        lines.append((indent, _strip_yaml_comment(raw[indent:])))
    if not lines:
        return {}
    root: Any = {} if not lines[0][1].startswith("- ") else []
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, (indent, content) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise BaselineConfigError(f"invalid indentation near line {index + 1}")
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise BaselineConfigError(f"list item has no list parent near line {index + 1}")
            parent.append(_minimal_yaml_scalar(content[2:]))
            continue
        if ":" not in content:
            raise BaselineConfigError(f"mapping entry is missing ':' near line {index + 1}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not key:
            raise BaselineConfigError(f"mapping key is empty near line {index + 1}")
        raw_value = raw_value.strip()
        if raw_value:
            if not isinstance(parent, dict):
                raise BaselineConfigError(f"mapping entry has no mapping parent near line {index + 1}")
            parent[key] = _minimal_yaml_scalar(raw_value)
            continue
        # A blank value introduces a nested mapping unless the next line is a
        # list item, in which case it introduces a list.
        next_content = lines[index + 1][1] if index + 1 < len(lines) else ""
        child: Any = [] if next_content.startswith("- ") else {}
        if not isinstance(parent, dict):
            raise BaselineConfigError(f"nested mapping has no mapping parent near line {index + 1}")
        parent[key] = child
        stack.append((indent, child))
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one local config without touching model/simulator runtimes."""

    input_path = Path(path)
    if input_path.is_symlink() or not input_path.is_file():
        raise BaselineConfigError(f"config must be a regular file: {input_path}")
    target = input_path.resolve(strict=True)
    text = target.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        value = _minimal_yaml_load(text)
    else:
        try:
            value = yaml.safe_load(text)
        except Exception as exc:  # pragma: no cover - PyYAML-specific parser errors
            raise BaselineConfigError(f"could not parse config YAML: {exc}") from exc
    return dict(_mapping(value, "config"))


def _validate_sequence(value: Any, name: str, *, expected: Sequence[int]) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BaselineConfigError(f"{name} must be a sequence")
    result = [_int(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if result != list(expected):
        raise BaselineConfigError(f"{name} must equal {list(expected)}, got {result}")
    if len(set(result)) != len(result):
        raise BaselineConfigError(f"{name} must not contain duplicates")
    return result


def _validate_env_mapping(value: Any, name: str, expected: Mapping[str, str]) -> dict[str, str]:
    section = dict(_mapping(value, name))
    for key, expected_value in expected.items():
        actual = _string(section.get(key), f"{name}.{key}")
        if actual != expected_value:
            raise BaselineConfigError(f"{name}.{key} must equal {expected_value!r}, got {actual!r}")
    if set(section) != set(expected):
        raise BaselineConfigError(f"{name} keys must equal {sorted(expected)}, got {sorted(section)}")
    return section


def validate_config(
    config: Mapping[str, Any],
    *,
    expected_project_sha: str | None = None,
    actual_project_sha: str | None = None,
    require_pinned_provenance: bool = False,
) -> dict[str, Any]:
    """Validate the exact frozen Baseline-A identity and return normalized data."""

    root = _mapping(config, "config")
    if _required(root, "schema_version") != SCHEMA_VERSION:
        raise BaselineConfigError("schema_version must be 1")
    if _required(root, "name") != EXPECTED_NAME:
        raise BaselineConfigError("name must be baseline_a")
    if expected_project_sha is not None or actual_project_sha is not None:
        expected = _string(expected_project_sha, "expected project SHA")
        actual = _string(actual_project_sha, "actual project SHA")
        if expected != actual:
            raise BaselineConfigError(f"project SHA mismatch: expected {expected}, got {actual}")

    seed = _int(_required(root, "seed"), "seed", nonnegative=True)
    if seed != EXPECTED_SEED:
        raise BaselineConfigError(f"seed must equal {EXPECTED_SEED}")
    suite = _string(_required(root, "suite"), "suite")
    if suite != EXPECTED_SUITE:
        raise BaselineConfigError(f"suite must equal {EXPECTED_SUITE}")
    task_ids = _validate_sequence(_required(root, "task_ids"), "task_ids", expected=EXPECTED_TASK_IDS)
    init_state_ids = _validate_sequence(
        _required(root, "init_state_ids"), "init_state_ids", expected=EXPECTED_INIT_STATE_IDS
    )
    task_names_raw = _mapping(_required(root, "task_names"), "task_names")
    task_names = {str(key): _string(value, f"task_names.{key}") for key, value in task_names_raw.items()}
    if set(task_names) != {str(item) for item in task_ids}:
        raise BaselineConfigError("task_names keys must cover exactly task_ids")
    if task_names != {key: EXPECTED_TASK_NAMES[key] for key in (str(item) for item in task_ids)}:
        raise BaselineConfigError("task_names must equal the pinned LIBERO task names")

    observation = _mapping(_required(root, "observation"), "observation")
    if _string(_required(observation, "type"), "observation.type") != "pixels_agent_pos":
        raise BaselineConfigError("observation.type must be pixels_agent_pos")
    height = _int(_required(observation, "height"), "observation.height")
    width = _int(_required(observation, "width"), "observation.width")
    if (height, width) != (360, 360):
        raise BaselineConfigError("observation dimensions must be 360x360")
    action = _mapping(_required(root, "action"), "action")
    action_dim = _int(_required(action, "dim"), "action.dim")
    chunk_size = _int(_required(action, "chunk_size"), "action.chunk_size")
    n_action_steps = _int(_required(action, "n_action_steps"), "action.n_action_steps")
    if (action_dim, chunk_size, n_action_steps) != (
        EXPECTED_ACTION_DIM,
        EXPECTED_CHUNK_SIZE,
        EXPECTED_N_ACTION_STEPS,
    ):
        raise BaselineConfigError("action contract must be dim=7, chunk_size=50, n_action_steps=1")
    horizon = _int(_required(root, "horizon"), "horizon")
    if horizon != EXPECTED_HORIZON:
        raise BaselineConfigError(f"horizon must equal {EXPECTED_HORIZON}")

    for artifact_name in ("checkpoint", "base_model", "assets"):
        artifact = _mapping(_required(root, artifact_name), artifact_name)
        for field in ("repo_id", "revision", "path"):
            _string(_required(artifact, field), f"{artifact_name}.{field}")
        digest_field = "manifest_sha256" if artifact_name == "assets" else "sha256"
        _sha(_required(artifact, digest_field), f"{artifact_name}.{digest_field}")

    runtime = _mapping(_required(root, "runtime"), "runtime")
    for field in ("cpu_python", "dcu_python", "cpu_runtime_lock", "dcu_runtime_lock"):
        _string(_required(runtime, field), f"runtime.{field}")
    physical = _validate_sequence(
        _required(runtime, "physical_devices"),
        "runtime.physical_devices",
        expected=EXPECTED_PHYSICAL_DEVICES,
    )
    logical = _string(_required(runtime, "logical_device"), "runtime.logical_device")
    if logical != EXPECTED_LOGICAL_DEVICE:
        raise BaselineConfigError("runtime.logical_device must be cuda:0")
    renderer = _validate_env_mapping(_required(runtime, "renderer"), "runtime.renderer", EXPECTED_RENDERER)
    offline = _validate_env_mapping(_required(runtime, "offline"), "runtime.offline", EXPECTED_OFFLINE)
    workers = _int(runtime.get("workers", 2), "runtime.workers")
    if workers != 2:
        raise BaselineConfigError("runtime.workers must equal 2")
    for field, expected in (
        ("worker_startup_timeout_seconds", 300),
        ("worker_forward_timeout_seconds", 300),
        ("worker_shutdown_timeout_seconds", 30),
    ):
        timeout = _int(_required(runtime, field), f"runtime.{field}", nonnegative=True)
        if timeout != expected:
            raise BaselineConfigError(f"runtime.{field} must equal the dcu_preflight timeout {expected}")
    parent_timeout = runtime.get("parent_collection_timeout_seconds", EXPECTED_PARENT_COLLECTION_TIMEOUT_SECONDS)
    if isinstance(parent_timeout, bool) or not isinstance(parent_timeout, (int, float)):
        raise BaselineConfigError("runtime.parent_collection_timeout_seconds must be numeric")
    parent_timeout = float(parent_timeout)
    if not math.isfinite(parent_timeout) or parent_timeout <= 0:
        raise BaselineConfigError("runtime.parent_collection_timeout_seconds must be finite and positive")
    for lock_field in ("cpu_runtime_lock_sha256", "dcu_runtime_lock_sha256"):
        if runtime.get(lock_field) is not None:
            _sha(runtime[lock_field], f"runtime.{lock_field}")
    if root.get("renderer") is not None:
        if dict(_mapping(root["renderer"], "renderer")) != renderer:
            raise BaselineConfigError("top-level renderer alias must equal runtime.renderer")
    if root.get("offline") is not None:
        if dict(_mapping(root["offline"], "offline")) != offline:
            raise BaselineConfigError("top-level offline alias must equal runtime.offline")
    if root.get("libero_config") is not None:
        libero_config = _mapping(root["libero_config"], "libero_config")
        _string(_required(libero_config, "path"), "libero_config.path")
        _sha(_required(libero_config, "sha256"), "libero_config.sha256")

    output = _mapping(_required(root, "output"), "output")
    output_root = _string(_required(output, "root"), "output.root")
    preflight_config = _string(_required(root, "preflight_config"), "preflight_config")
    preflight_config_sha256 = _sha(
        _required(root, "preflight_config_sha256"), "preflight_config_sha256"
    )
    if preflight_config_sha256 != EXPECTED_PREFLIGHT_CONFIG_SHA256:
        raise BaselineConfigError("preflight_config_sha256 does not match the pinned dcu_preflight config")
    if _required(root, "perturbation") is not False:
        raise BaselineConfigError("perturbation must be false for Baseline-A")
    action_noise = _mapping(_required(root, "action_noise"), "action_noise")
    noise_source = _string(_required(action_noise, "source"), "action_noise.source")
    if noise_source != "native_policy_rng":
        raise BaselineConfigError("action_noise.source must be native_policy_rng")
    if _required(action_noise, "explicit") is not False:
        raise BaselineConfigError("action_noise.explicit must be false")

    provenance = dict(root.get("provenance", {}))
    if not isinstance(provenance, dict):
        raise BaselineConfigError("provenance must be a mapping")
    for field in (
        "lerobot_git_sha",
        "libero_git_sha",
        "python",
        "pytorch",
        "transformers",
        "cuda",
        "hip",
        "gpu",
    ):
        _string(provenance.get(field), f"provenance.{field}")
    if require_pinned_provenance:
        provenance = _validate_pinned_provenance(provenance)

    return {
        "schema_version": SCHEMA_VERSION,
        "name": EXPECTED_NAME,
        "seed": seed,
        "suite": suite,
        "task_ids": task_ids,
        "init_state_ids": init_state_ids,
        "task_names": task_names,
        "observation": {"type": "pixels_agent_pos", "height": height, "width": width},
        "action": {"dim": action_dim, "chunk_size": chunk_size, "n_action_steps": n_action_steps},
        "horizon": horizon,
        "checkpoint": dict(_mapping(root["checkpoint"], "checkpoint")),
        "base_model": dict(_mapping(root["base_model"], "base_model")),
        "assets": dict(_mapping(root["assets"], "assets")),
        "libero_config": (
            dict(_mapping(root["libero_config"], "libero_config"))
            if root.get("libero_config") is not None
            else None
        ),
        "runtime": {
            **dict(runtime),
            "physical_devices": physical,
            "logical_device": logical,
            "renderer": renderer,
            "offline": offline,
            "workers": workers,
            "parent_collection_timeout_seconds": parent_timeout,
        },
        "output": {"root": output_root},
        "preflight_config": preflight_config,
        "preflight_config_sha256": preflight_config_sha256,
        "perturbation": False,
        "action_noise": {"source": noise_source, "explicit": False},
        "provenance": provenance,
        "planned_episodes": len(task_ids) * len(init_state_ids),
    }


def _validate_pinned_provenance(provenance: Mapping[str, Any]) -> dict[str, str]:
    """Reject validate-only evidence whose recorded runtime identity drifted."""

    section = dict(_mapping(provenance, "provenance", BaselineConfigError))
    for field, expected in EXPECTED_PROVENANCE.items():
        actual = _string(section.get(field), f"provenance.{field}")
        if actual != expected:
            raise BaselineConfigError(
                f"provenance.{field} must equal the pinned value {expected!r}, got {actual!r}"
            )
    return section


def assignment_for_index(index: int, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the deterministic zero-based task-major worker assignment."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise BaselineSchemaError("episode matrix index must be a non-negative integer")
    devices = list(EXPECTED_PHYSICAL_DEVICES)
    if config is not None:
        runtime = _mapping(config.get("runtime"), "runtime", BaselineSchemaError)
        raw_devices = runtime.get("physical_devices")
        if isinstance(raw_devices, Sequence) and not isinstance(raw_devices, (str, bytes)):
            devices = [_int(item, "runtime.physical_devices", error=BaselineSchemaError) for item in raw_devices]
    if devices != list(EXPECTED_PHYSICAL_DEVICES):
        raise BaselineSchemaError("worker physical devices must be [0, 1]")
    worker_index = index % 2
    return {
        "worker": WORKERS[worker_index],
        "physical_device": devices[worker_index],
        "logical_device": EXPECTED_LOGICAL_DEVICE,
    }


def build_episode_matrix(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build and validate the exact task-major 3x5 episode matrix."""

    normalized = validate_config(config)
    rows: list[dict[str, Any]] = []
    order = 0
    for task_id in normalized["task_ids"]:
        for init_state_id in normalized["init_state_ids"]:
            assignment = assignment_for_index(order, normalized)
            episode_id = f"{normalized['suite']}-task{task_id:03d}-init{init_state_id:03d}"
            rows.append(
                {
                    "episode_id": episode_id,
                    "trajectory_id": episode_id,
                    "order": order,
                    "suite": normalized["suite"],
                    "task_id": task_id,
                    "task_name": normalized["task_names"][str(task_id)],
                    "init_state_id": init_state_id,
                    "seed": normalized["seed"],
                    "worker": assignment["worker"],
                    "physical_device": assignment["physical_device"],
                    "logical_device": assignment["logical_device"],
                    "horizon": normalized["horizon"],
                    "status": "planned",
                }
            )
            order += 1
    validate_episode_matrix(rows, normalized)
    return rows


def validate_episode_matrix(matrix: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    """Reject missing, duplicate, reordered, or cross-worker matrix rows."""

    normalized = validate_config(config)
    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise BaselineSchemaError("episode matrix must be a sequence")
    expected_pairs = [
        (task_id, init_state_id)
        for task_id in normalized["task_ids"]
        for init_state_id in normalized["init_state_ids"]
    ]
    if len(matrix) != len(expected_pairs):
        if len({str(row.get("episode_id")) for row in matrix if isinstance(row, Mapping)}) != len(matrix):
            raise BaselineSchemaError("episode matrix contains duplicate episode IDs")
        raise BaselineSchemaError(f"episode matrix must contain {len(expected_pairs)} rows")
    seen: set[str] = set()
    for order, (row, pair) in enumerate(zip(matrix, expected_pairs, strict=True)):
        item = _mapping(row, f"episode_matrix[{order}]", BaselineSchemaError)
        episode_id = _string(item.get("episode_id"), f"episode_matrix[{order}].episode_id", BaselineSchemaError)
        if episode_id in seen:
            raise BaselineSchemaError(f"duplicate episode ID {episode_id}")
        seen.add(episode_id)
        if item.get("order") != order:
            raise BaselineSchemaError(f"episode matrix order must be zero-based at row {order}")
        if (item.get("task_id"), item.get("init_state_id")) != pair:
            raise BaselineSchemaError("episode matrix must use task-major order")
        expected_assignment = assignment_for_index(order, normalized)
        for field, value in expected_assignment.items():
            if item.get(field) != value:
                raise BaselineSchemaError(f"episode_matrix[{order}].{field} does not match assignment")
        if item.get("suite") != EXPECTED_SUITE or item.get("seed") != EXPECTED_SEED:
            raise BaselineSchemaError("episode matrix row identity does not match frozen config")
    return True


def _expected_scoped_matrix(
    matrix: Sequence[Mapping[str, Any]], config: Mapping[str, Any], scope: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate a parent matrix or one exact worker projection of it."""

    normalized = validate_config(config)
    full = build_episode_matrix(normalized)
    scope_type = scope.get("type", "parent")
    if scope_type == "parent":
        expected = full
    elif scope_type == "worker":
        worker = scope.get("worker")
        physical_device = scope.get("physical_device")
        if worker not in WORKERS or physical_device != EXPECTED_PHYSICAL_DEVICES[WORKERS.index(worker)]:
            raise BaselineSchemaError("worker scope identity is invalid")
        expected = [row for row in full if row["worker"] == worker]
    else:
        raise BaselineSchemaError("manifest scope type is invalid")
    actual_ids = [str(row.get("episode_id")) for row in matrix]
    expected_ids = [row["episode_id"] for row in expected]
    if actual_ids != expected_ids:
        raise BaselineSchemaError("manifest matrix rows do not match its declared scope")
    if scope.get("planned_count", len(matrix)) != len(expected):
        raise BaselineSchemaError("manifest scope planned_count does not match matrix")
    if scope.get("episode_ids", expected_ids) != expected_ids:
        raise BaselineSchemaError("manifest scope episode_ids do not match matrix")
    if scope.get("trajectory_ids", [row["trajectory_id"] for row in expected]) != [
        row["trajectory_id"] for row in expected
    ]:
        raise BaselineSchemaError("manifest scope trajectory_ids do not match matrix")
    return expected


def build_cpu_child_environment(
    config: Mapping[str, Any],
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Set offline/EGL variables and remove compute visibility from CPU child."""

    root = _mapping(config, "config", BaselineSchemaError)
    runtime = _mapping(root.get("runtime"), "runtime", BaselineSchemaError)
    environment = dict(os.environ if base_environment is None else base_environment)
    offline = _mapping(runtime.get("offline"), "runtime.offline", BaselineSchemaError)
    renderer = _mapping(runtime.get("renderer"), "runtime.renderer", BaselineSchemaError)
    environment.update({str(key): str(value) for key, value in offline.items()})
    environment.update({str(key): str(value) for key, value in renderer.items()})
    environment.pop("HIP_VISIBLE_DEVICES", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONUNBUFFERED"] = "1"
    libero_config = root.get("libero_config")
    if isinstance(libero_config, Mapping) and isinstance(libero_config.get("path"), str):
        environment["LIBERO_CONFIG_PATH"] = str(Path(libero_config["path"]).resolve().parent)
    validate_renderer_environment(environment)
    return environment


def build_nested_worker_environment(
    config: Mapping[str, Any], physical_device: int, base_environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Set one physical K100 visible as logical ``cuda:0`` in model worker."""

    if physical_device not in EXPECTED_PHYSICAL_DEVICES:
        raise BaselineSchemaError("nested worker physical device must be 0 or 1")
    environment = build_cpu_child_environment(config, base_environment)
    environment["HIP_VISIBLE_DEVICES"] = str(physical_device)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    return environment


def build_worker_environments(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    return {
        worker: build_nested_worker_environment(config, index)
        for worker, index in zip(WORKERS, EXPECTED_PHYSICAL_DEVICES, strict=True)
    }


def _cross_check_preflight_identity(
    config: Mapping[str, Any], preflight_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Cross-check only duplicated identity fields against dcu_preflight.

    The underlying dcu_preflight module remains authoritative for its complete
    runner validation.  This narrow boundary prevents Baseline-A from
    accidentally selecting a different artifact, lock, or runtime mapping.
    """

    baseline = validate_config(config)
    dcu = _mapping(preflight_config, "preflight config", BaselineConfigError)

    def equal(actual: Any, expected: Any, name: str) -> None:
        if actual != expected:
            raise BaselineConfigError(
                f"preflight cross-identity mismatch for {name}: expected {expected!r}, got {actual!r}"
            )

    equal(dcu.get("schema_version"), 1, "schema_version")
    equal(dcu.get("name"), "dcu_preflight", "name")
    equal(dcu.get("mode"), "explicit_phase", "mode")
    for field in ("seed", "suite", "obs_type", "observation_height", "observation_width", "action_dim", "chunk_size", "n_action_steps", "horizon"):
        baseline_value = {
            "seed": baseline["seed"],
            "suite": baseline["suite"],
            "obs_type": baseline["observation"]["type"],
            "observation_height": baseline["observation"]["height"],
            "observation_width": baseline["observation"]["width"],
            "action_dim": baseline["action"]["dim"],
            "chunk_size": baseline["action"]["chunk_size"],
            "n_action_steps": baseline["action"]["n_action_steps"],
            "horizon": baseline["horizon"],
        }[field]
        equal(dcu.get(field), baseline_value, field)
    task = _mapping(dcu.get("task"), "preflight.task", BaselineConfigError)
    equal(task.get("suite"), baseline["suite"], "task.suite")
    equal(task.get("seed"), baseline["seed"], "task.seed")
    equal(task.get("task_id"), EXPECTED_TASK_IDS[0], "task.task_id")
    equal(task.get("init_state_id"), EXPECTED_INIT_STATE_IDS[0], "task.init_state_id")

    for name, digest_name in (("checkpoint", "model_sha256"), ("base_model", "model_sha256"), ("assets", "manifest_sha256")):
        baseline_artifact = _mapping(baseline[name], name, BaselineConfigError)
        dcu_artifact = _mapping(dcu.get(name), f"preflight.{name}", BaselineConfigError)
        for field in ("repo_id", "revision", "path"):
            equal(dcu_artifact.get(field), baseline_artifact.get(field), f"{name}.{field}")
        equal(dcu_artifact.get(digest_name), baseline_artifact.get("sha256" if name != "assets" else "manifest_sha256"), f"{name}.{digest_name}")

    dcu_runtime = _mapping(dcu.get("runtime"), "preflight.runtime", BaselineConfigError)
    baseline_runtime = baseline["runtime"]
    for field in (
        "cpu_python",
        "dcu_python",
        "cpu_runtime_lock",
        "dcu_runtime_lock",
        "logical_device",
        "physical_devices",
        "worker_startup_timeout_seconds",
        "worker_forward_timeout_seconds",
        "worker_shutdown_timeout_seconds",
    ):
        equal(dcu_runtime.get(field), baseline_runtime.get(field), f"runtime.{field}")
    for baseline_field, preflight_field in (
        ("cpu_runtime_lock_sha256", "runtime_lock_sha256"),
        ("dcu_runtime_lock_sha256", "dcu_runtime_lock_sha256"),
    ):
        expected_digest = baseline_runtime.get(baseline_field)
        if expected_digest is not None:
            equal(dcu.get(preflight_field), expected_digest, preflight_field)
    equal(
        dcu.get("runtime_lock", dcu_runtime.get("runtime_lock")),
        baseline_runtime["cpu_runtime_lock"],
        "runtime_lock",
    )
    equal(dcu.get("offline"), baseline_runtime["offline"], "offline")
    equal(dcu.get("renderer"), baseline_runtime["renderer"], "renderer")
    if baseline.get("libero_config") is not None:
        dcu_libero = _mapping(dcu.get("libero_config"), "preflight.libero_config", BaselineConfigError)
        for field in ("path", "sha256"):
            equal(dcu_libero.get(field), baseline["libero_config"].get(field), f"libero_config.{field}")
    return {
        "checked": [
            "task_contract",
            "checkpoint",
            "base_model",
            "assets",
            "runtime_locks",
            "offline",
            "renderer",
            "action_contract",
        ],
        "exact": True,
    }


def _run_preflight_gates_bounded(
    gate_runner: Any,
    *,
    timeout_seconds: float,
    **kwargs: Any,
) -> Any:
    """Run the synchronous preflight gate runner under a finite POSIX deadline."""

    if not callable(gate_runner):
        raise BaselineRuntimeError("preflight gate runner is not callable")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise BaselineRuntimeError("preflight gate timeout must be a finite positive number")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise BaselineRuntimeError("preflight gate timeout must be a finite positive number")
    if threading.current_thread() is not threading.main_thread():
        raise BaselineRuntimeError("preflight gate deadline requires the main thread")
    if not all(
        hasattr(signal, name) and callable(getattr(signal, name, None))
        for name in ("signal", "getsignal", "setitimer", "getitimer")
    ) or not all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL")):
        raise BaselineRuntimeError("preflight gate deadline requires POSIX SIGALRM/setitimer")

    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
    except BaseException as exc:
        raise BaselineRuntimeError(f"could not inspect the preflight signal deadline: {exc}") from exc

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise BaselineRuntimeError(f"preflight gate runner timed out after {timeout:.3f}s")

    handler_installed = False
    timer_installed = False
    restoration_errors: list[str] = []
    try:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            handler_installed = True
        except BaseException as exc:
            raise BaselineRuntimeError(f"could not install the preflight signal deadline: {exc}") from exc
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout, 0.0)
            timer_installed = True
        except BaseException as exc:
            raise BaselineRuntimeError(f"could not arm the preflight signal deadline: {exc}") from exc
        return gate_runner(**kwargs)
    finally:
        if timer_installed:
            try:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            except BaseException as exc:
                restoration_errors.append(f"timer restoration failed: {type(exc).__name__}: {exc}")
        if handler_installed:
            try:
                signal.signal(signal.SIGALRM, previous_handler)
            except BaseException as exc:
                restoration_errors.append(f"handler restoration failed: {type(exc).__name__}: {exc}")
        if restoration_errors:
            raise BaselineRuntimeError("; ".join(restoration_errors))


def validate_preflight_contract(
    config: Mapping[str, Any],
    *,
    expected_project_sha: str,
    actual_project_sha: str,
    run_directory: str | Path,
    preflight_module: Any | None = None,
    run_gates: bool = True,
) -> dict[str, Any]:
    """Load, hash, cross-check, and optionally gate the frozen preflight config."""

    normalized = validate_config(config)
    path = Path(normalized["preflight_config"])
    if path.is_symlink() or not path.is_file():
        raise BaselineConfigError(f"preflight config must be a regular file: {path}")
    path = path.resolve(strict=True)
    actual_hash = _sha256_file(path)
    expected_hash = normalized["preflight_config_sha256"]
    if actual_hash != expected_hash:
        raise BaselineConfigError(
            f"preflight config SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    if preflight_module is None:
        from scripts import dcu_preflight as preflight_module
    loader = getattr(preflight_module, "_load_yaml_config", None)
    if not callable(loader):
        raise BaselineRuntimeError("dcu_preflight does not expose its config loader")
    preflight_config = dict(_mapping(loader(path), "preflight config", BaselineConfigError))
    cross_identity = _cross_check_preflight_identity(normalized, preflight_config)
    gates: Mapping[str, Any]
    if run_gates:
        run_path = Path(run_directory).resolve()
        run_path.mkdir(parents=True, exist_ok=True)
        gate_runner = getattr(preflight_module, "run_preflight_gates", None)
        if not callable(gate_runner):
            raise BaselineRuntimeError("dcu_preflight does not expose run_preflight_gates")
        gates = _run_preflight_gates_bounded(
            gate_runner,
            timeout_seconds=float(normalized["runtime"]["worker_startup_timeout_seconds"]),
            config=preflight_config,
            phase="concurrency",
            expected_project_sha=str(expected_project_sha),
            actual_project_sha=str(actual_project_sha),
            run_directory=run_path,
            prepare_assets=False,
        )
        if not isinstance(gates, Mapping):
            raise BaselineRuntimeError("dcu_preflight gates must return a mapping")
    else:
        gates = {"not_run": True}
    return {
        "config_path": str(path),
        "config_sha256": actual_hash,
        "cross_identity": cross_identity,
        "gates": dict(gates),
        "phase": "concurrency",
        "prepare_assets": False,
    }


def validate_renderer_environment(
    environment: Mapping[str, str], gl_identity: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the independent EGL ordinal and, when supplied, llvmpipe identity."""

    mismatches = {
        key: {"expected": expected, "actual": environment.get(key)}
        for key, expected in EXPECTED_RENDERER.items()
        if environment.get(key) != expected
    }
    if mismatches:
        raise BaselineSchemaError(f"renderer environment is not the frozen EGL mapping: {mismatches}")
    evidence: dict[str, Any] = {"variables": dict(EXPECTED_RENDERER), "exact": True}
    if gl_identity is not None:
        if dict(gl_identity) != EXPECTED_GL_IDENTITY:
            raise BaselineSchemaError(
                f"GL identity does not match the validated llvmpipe renderer: {gl_identity!r}"
            )
        evidence["gl_identity"] = dict(EXPECTED_GL_IDENTITY)
    return evidence


def select_init_state_before_reset(env: Any, init_state_id: int) -> dict[str, Any]:
    """Set and verify an official vector-env init state before ``reset``.

    LIBERO increments its ``init_state_id`` during reset.  Therefore the
    selected value and the post-reset value are deliberately separate pieces
    of evidence; callers must invoke the official reset between them.
    """

    init_state_id = _int(init_state_id, "init_state_id", nonnegative=True, error=BaselineSchemaError)
    setter = getattr(env, "set_attr", None)
    getter = getattr(env, "get_attr", None)
    if not callable(setter) or not callable(getter):
        raise BaselineSchemaError("environment must expose official set_attr/get_attr")
    setter("init_state_id", init_state_id)
    values = np.asarray(getter("init_state_id")).reshape(-1)
    if values.size != 1 or int(values[0]) != init_state_id:
        raise BaselineSchemaError(
            f"selected init_state_id was not verified before reset: expected {init_state_id}, got {values.tolist()}"
        )
    return {"selected_init_state_id": init_state_id, "verified_before_reset": True}


def validate_runtime_task_identity(planned: Mapping[str, Any], runtime_task: Mapping[str, Any]) -> bool:
    """Require the official runtime task source to equal the matrix row."""

    planned_row = _mapping(planned, "planned episode", BaselineRuntimeError)
    task = _mapping(runtime_task, "runtime task", BaselineRuntimeError)
    for field in ("suite", "task_id", "task_name"):
        if task.get(field) != planned_row.get(field):
            raise BaselineRuntimeError(
                f"runtime {field} does not match planned episode: "
                f"expected {planned_row.get(field)!r}, got {task.get(field)!r}"
            )
    return True


def episode_ipc_directory(child_directory: str | Path, planned: Mapping[str, Any]) -> Path:
    """Return one collision-free IPC directory for exactly one episode."""

    row = _mapping(planned, "planned episode", BaselineSchemaError)
    order = _int(row.get("order"), "planned order", nonnegative=True, error=BaselineSchemaError)
    episode_id = _string(row.get("episode_id"), "planned episode_id", BaselineSchemaError)
    safe_id = "".join(char if char.isalnum() or char in "-_." else "_" for char in episode_id)
    return Path(child_directory).resolve() / "ipc" / f"{order:04d}-{safe_id}"


def _strict_child_path(child_directory: str | Path, candidate: str | Path, name: str) -> Path:
    """Resolve a cleanup target only after checking its lexical path."""

    # ``Path.resolve`` follows symlinks.  Check the lexical child-relative
    # path first so an alias such as ``child/ipc/alias -> child/episode_rows``
    # cannot become the trusted target before the symlink guard runs.
    child_raw = Path(child_directory)
    candidate_raw = Path(candidate)
    if not child_raw.is_absolute():
        child_raw = Path.cwd() / child_raw
    if not candidate_raw.is_absolute():
        candidate_raw = Path.cwd() / candidate_raw

    def first_symlink(path: Path) -> Path | None:
        current = Path(path.anchor or os.sep)
        for component in path.parts:
            if component in {"", path.anchor, "."}:
                continue
            if component == "..":
                current = current.parent
                continue
            current /= component
            if current.is_symlink():
                return current
        return None

    lexical_symlink = first_symlink(candidate_raw) or first_symlink(child_raw)
    if lexical_symlink is not None:
        raise BaselineSchemaError(f"{name} must not traverse a symlink: {lexical_symlink}")
    child_lexical = Path(os.path.abspath(os.fspath(child_raw)))
    candidate_lexical = Path(os.path.abspath(os.fspath(candidate_raw)))
    try:
        relative = candidate_lexical.relative_to(child_lexical)
    except ValueError as exc:
        raise BaselineSchemaError(f"{name} escapes child directory: {candidate_lexical}") from exc
    current = child_lexical
    if current.is_symlink():
        raise BaselineSchemaError(f"{name} has a symlinked child directory: {current}")
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise BaselineSchemaError(f"{name} must not traverse a symlink: {current}")
    child = child_lexical.resolve(strict=False)
    target = candidate_lexical.resolve(strict=False)
    try:
        target.relative_to(child)
    except ValueError as exc:
        raise BaselineSchemaError(f"{name} escapes child directory: {target}") from exc
    return target


def _remove_ipc_target(child_directory: str | Path, target: Path, removed: list[str]) -> None:
    checked = _strict_child_path(child_directory, target, "IPC cleanup target")
    if checked.is_symlink():
        raise BaselineSchemaError(f"IPC cleanup target must not be a symlink: {checked}")
    if checked.is_dir():
        for nested in checked.rglob("*"):
            _strict_child_path(child_directory, nested, "IPC cleanup entry")
            if nested.is_symlink():
                raise BaselineSchemaError(f"IPC cleanup entry must not be a symlink: {nested}")
        shutil.rmtree(checked)
    elif checked.exists():
        checked.unlink()
    removed.append(str(checked))


def cleanup_episode_ipc(child_directory: str | Path, planned: Mapping[str, Any]) -> dict[str, Any]:
    """Delete exactly one completed episode's transient request artifacts."""

    child = Path(child_directory).resolve()
    target = episode_ipc_directory(child, planned)
    removed: list[str] = []
    if target.exists() or target.is_symlink():
        _remove_ipc_target(child, target, removed)
    return {"strict_under_child": True, "removed_paths": removed, "removed_count": len(removed)}


def cleanup_child_ipc(child_directory: str | Path) -> dict[str, Any]:
    """Remove orphan worker/episode IPC entries after an outer child exits."""

    child = Path(child_directory).resolve()
    ipc = _strict_child_path(child, child / "ipc", "IPC root")
    removed: list[str] = []
    if ipc.exists() or ipc.is_symlink():
        if ipc.is_symlink() or not ipc.is_dir():
            raise BaselineSchemaError(f"IPC root must be a regular directory: {ipc}")
        for entry in sorted(ipc.iterdir(), key=lambda item: item.name):
            _remove_ipc_target(child, entry, removed)
    return {"strict_under_child": True, "removed_paths": removed, "removed_count": len(removed)}


def configure_remote_timeouts(remote: Any, runtime: Mapping[str, Any], preflight_module: Any | None = None) -> Any:
    """Apply the existing dcu_preflight startup/forward timeout contract."""

    if preflight_module is None:
        from scripts import dcu_preflight as preflight_module
    timeout = getattr(preflight_module, "_timeout_seconds", None)
    if callable(timeout):
        remote.reset_timeout_seconds = float(timeout(runtime, "startup"))
        remote.forward_timeout_seconds = float(timeout(runtime, "forward"))
    else:
        remote.reset_timeout_seconds = float(runtime["worker_startup_timeout_seconds"])
        remote.forward_timeout_seconds = float(runtime["worker_forward_timeout_seconds"])
    return remote


def parent_collection_timeout_seconds(config: Mapping[str, Any]) -> float:
    normalized = validate_config(config)
    value = normalized["runtime"].get(
        "parent_collection_timeout_seconds", EXPECTED_PARENT_COLLECTION_TIMEOUT_SECONDS
    )
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise BaselineConfigError("parent collection timeout must be finite and positive")
    return timeout


def close_runtime_environment(runtime: Mapping[str, Any] | None) -> None:
    """Close all official environments through the runtime-owned seam."""

    if not runtime:
        return
    close_envs = runtime.get("close_envs")
    envs = runtime.get("envs")
    if callable(close_envs) and envs is not None:
        close_envs(envs)


def worker_peak_memory_bytes(
    worker_client: Any, *, start_index: int = 0, responses: Sequence[Any] | None = None
) -> int:
    """Return peak-memory evidence for one episode's worker responses only."""

    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise BaselineSchemaError("worker response start_index must be a non-negative integer")
    if responses is None:
        transport = getattr(worker_client, "transport", None)
        all_responses = getattr(transport, "responses", [])
        if not isinstance(all_responses, Sequence) or isinstance(all_responses, (str, bytes)):
            raise BaselineRuntimeError("worker transport responses must be a sequence")
        responses = all_responses[start_index:]
    peaks: list[int] = []
    for index, response in enumerate(responses):
        if not isinstance(response, Mapping) or "peak_memory_bytes" not in response:
            continue
        value = response["peak_memory_bytes"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BaselineRuntimeError(f"worker response {index} peak memory is invalid")
        peaks.append(int(value))
    return max(peaks, default=0)


def validate_worker_evidence(
    evidence: Mapping[str, Any], *, worker: str, physical_device: int
) -> bool:
    """Validate identity fields from dcu_preflight's live worker summary."""

    item = _mapping(evidence, "worker evidence", BaselineRuntimeError)
    if item.get("physical_device") != physical_device:
        raise BaselineRuntimeError("worker ping physical device does not match assignment")
    if item.get("logical_device") != EXPECTED_LOGICAL_DEVICE:
        raise BaselineRuntimeError("worker ping logical device is not cuda:0")
    if item.get("torch_logical_device") != EXPECTED_LOGICAL_DEVICE:
        raise BaselineRuntimeError("worker ping torch device is not cuda:0")
    compute = _mapping(item.get("compute_environment"), "worker.compute_environment", BaselineRuntimeError)
    if compute.get("physical_k100") != physical_device:
        raise BaselineRuntimeError("worker compute physical device does not match assignment")
    if compute.get("HIP_VISIBLE_DEVICES") != str(physical_device):
        raise BaselineRuntimeError("worker HIP_VISIBLE_DEVICES does not match assignment")
    if compute.get("CUDA_VISIBLE_DEVICES") != "0":
        raise BaselineRuntimeError("worker CUDA_VISIBLE_DEVICES must expose logical cuda:0")
    if item.get("model_load_success") is not True:
        raise BaselineRuntimeError(f"worker {worker} model load evidence is not successful")
    return True


def validate_action(action: Any) -> dict[str, Any]:
    """Validate one exact postprocessed environment action."""

    if not isinstance(action, np.ndarray):
        raise BaselineSchemaError(f"environment action must be a NumPy array, got {type(action).__name__}")
    if action.shape != (1, EXPECTED_ACTION_DIM):
        raise BaselineSchemaError(f"environment action shape must be (1, 7), got {action.shape}")
    if action.dtype != np.dtype(np.float32):
        raise BaselineSchemaError(f"environment action dtype must be float32, got {action.dtype}")
    finite = bool(np.isfinite(action).all())
    if not finite:
        raise BaselineSchemaError("environment action contains non-finite values")
    return {"shape": list(action.shape), "dtype": str(action.dtype), "finite": finite, "values": action.tolist()}


def termination_reason(
    *, terminated: bool, truncated: bool, success: bool, steps: int, horizon: int = EXPECTED_HORIZON
) -> str:
    """Classify official environment termination, including the horizon."""

    steps = _int(steps, "steps", nonnegative=True, error=BaselineSchemaError)
    horizon = _int(horizon, "horizon", nonnegative=False, error=BaselineSchemaError)
    if steps < 1 or steps > horizon:
        raise BaselineSchemaError(f"steps must be in [1, {horizon}]")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool) or not isinstance(success, bool):
        raise BaselineSchemaError("terminated, truncated, and success must be booleans")
    if terminated and truncated:
        raise BaselineSchemaError("terminated and truncated cannot both be true")
    if terminated:
        return "environment_termination"
    if truncated and steps == horizon:
        return "official_horizon"
    if truncated:
        return "environment_truncation"
    if steps == horizon:
        return "official_horizon"
    if success:
        raise BaselineSchemaError("successful episode must report environment termination")
    raise BaselineSchemaError("episode must terminate, truncate, or reach the official horizon")


def _latency_summary(values: Sequence[Any], name: str) -> dict[str, Any]:
    numbers = [_finite_nonnegative(item, f"{name}[{index}]") for index, item in enumerate(values)]
    if not numbers:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "total": 0.0}
    array = np.asarray(numbers, dtype=np.float64)
    return {
        "count": len(numbers),
        "mean": float(statistics.mean(numbers)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "total": float(sum(numbers)),
    }


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BaselineSchemaError(f"{name} must be a sequence")
    return list(value)


def make_episode_record(
    planned: Mapping[str, Any],
    *,
    success: bool,
    rewards: Sequence[Any],
    terminated: bool,
    truncated: bool,
    steps: int,
    policy_calls: int,
    inference_latencies: Sequence[Any],
    env_step_latencies: Sequence[Any],
    actions: Sequence[np.ndarray],
    model_peak_memory_bytes: int,
    wall_time_seconds: float,
    offline: bool = True,
    hub_fallback: bool = False,
    attempt_count: int = 1,
    no_retry: bool = True,
    termination: str | None = None,
    init_state_evidence: Mapping[str, Any] | None = None,
    task_source_evidence: Mapping[str, Any] | None = None,
    worker_evidence: Mapping[str, Any] | None = None,
    gl_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one completed evidence row; exactly one attempt is permitted."""

    planned_row = _mapping(planned, "planned episode", BaselineSchemaError)
    episode_id = _string(planned_row.get("episode_id"), "planned episode_id", BaselineSchemaError)
    if attempt_count != 1 or no_retry is not True:
        raise BaselineSchemaError("attempt_count must be exactly 1 and no_retry must be true")
    if not isinstance(success, bool):
        raise BaselineSchemaError("success must be a boolean")
    steps = _int(steps, "steps", nonnegative=False, error=BaselineSchemaError)
    horizon = _int(planned_row.get("horizon", EXPECTED_HORIZON), "horizon", error=BaselineSchemaError)
    if steps < 1 or steps > horizon:
        raise BaselineSchemaError(f"steps must be in [1, {horizon}]")
    rewards_list = _sequence(rewards, "rewards")
    inference_list = _sequence(inference_latencies, "inference_latencies")
    env_steps_list = _sequence(env_step_latencies, "env_step_latencies")
    action_list = _sequence(actions, "actions")
    if any(len(values) != steps for values in (rewards_list, inference_list, env_steps_list, action_list)):
        raise BaselineSchemaError("rewards, latencies, and actions must each have exactly steps entries")
    policy_calls = _int(policy_calls, "policy_calls", nonnegative=True, error=BaselineSchemaError)
    if policy_calls != steps:
        raise BaselineSchemaError("native n_action_steps=1 requires one policy call per environment step")
    reward_values = [_finite_number(item, f"rewards[{index}]") for index, item in enumerate(rewards_list)]
    action_evidence = [validate_action(item) for item in action_list]
    peak_memory = _int(model_peak_memory_bytes, "model_peak_memory_bytes", nonnegative=True, error=BaselineSchemaError)
    wall = float(_finite_nonnegative(wall_time_seconds, "wall_time_seconds"))
    if not isinstance(offline, bool) or offline is not True:
        raise BaselineSchemaError("offline must be true")
    if not isinstance(hub_fallback, bool) or hub_fallback is not False:
        raise BaselineSchemaError("hub_fallback must be false")
    reason = termination_reason(
        terminated=terminated,
        truncated=truncated,
        success=success,
        steps=steps,
        horizon=horizon,
    )
    if termination is not None and termination != reason:
        raise BaselineSchemaError("provided termination reason disagrees with official flags")
    init_evidence = dict(_mapping(init_state_evidence, "init_state_evidence", BaselineSchemaError))
    if init_evidence.get("selected_init_state_id") != planned_row.get("init_state_id"):
        raise BaselineSchemaError("selected_init_state_id does not match planned episode")
    if init_evidence.get("verified_before_reset") is not True:
        raise BaselineSchemaError("init state was not verified before reset")
    if init_evidence.get("post_reset_init_state_id") != int(planned_row.get("init_state_id")) + 1:
        raise BaselineSchemaError("post_reset_init_state_id must equal planned init_state_id + 1")
    task_evidence = dict(_mapping(task_source_evidence, "task_source_evidence", BaselineSchemaError))
    for field in ("suite", "task_id", "task_name"):
        if task_evidence.get(field) != planned_row.get(field):
            raise BaselineSchemaError(f"task_source_evidence.{field} does not match planned episode")
    worker_evidence_map = dict(_mapping(worker_evidence, "worker_evidence", BaselineSchemaError))
    try:
        validate_worker_evidence(
            worker_evidence_map,
            worker=str(planned_row.get("worker")),
            physical_device=int(planned_row.get("physical_device")),
        )
    except BaselineError as exc:
        raise BaselineSchemaError(f"worker evidence is invalid: {exc}") from exc
    gl_evidence_map = dict(_mapping(gl_evidence, "gl_evidence", BaselineSchemaError))
    try:
        validate_renderer_environment(
            _mapping(gl_evidence_map.get("variables"), "gl_evidence.variables", BaselineSchemaError),
            gl_identity=_mapping(gl_evidence_map.get("gl_identity"), "gl_evidence.gl_identity", BaselineSchemaError),
        )
    except BaselineError as exc:
        raise BaselineSchemaError(f"GL evidence is invalid: {exc}") from exc
    inference_summary = _latency_summary(inference_list, "inference_latencies")
    env_summary = _latency_summary(env_steps_list, "env_step_latencies")
    result: dict[str, Any] = {
        **dict(planned_row),
        "status": "completed",
        "success": success,
        "reward_sum": float(sum(reward_values)),
        "rewards": reward_values,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": reason,
        "steps": steps,
        "policy_calls": policy_calls,
        "inference_latencies": [float(item) for item in inference_list],
        "inference_mean_seconds": inference_summary["mean"],
        "inference_p50_seconds": inference_summary["p50"],
        "inference_p95_seconds": inference_summary["p95"],
        "env_step_latencies": [float(item) for item in env_steps_list],
        "env_step_mean_seconds": env_summary["mean"],
        "env_step_p50_seconds": env_summary["p50"],
        "env_step_p95_seconds": env_summary["p95"],
        "model_peak_memory_bytes": peak_memory,
        "model_peak_memory": peak_memory,
        "actions": [item["values"] for item in action_evidence],
        "action_evidence": action_evidence,
        "actions_finite": all(item["finite"] for item in action_evidence),
        "actions_dtype": "float32",
        "actions_shape": [item["shape"] for item in action_evidence],
        "offline": offline,
        "hub_fallback": hub_fallback,
        "wall_time_seconds": wall,
        "attempt_count": 1,
        "no_retry": True,
        "crashed": False,
        **init_evidence,
        "task_source_evidence": task_evidence,
        "worker_evidence": worker_evidence_map,
        "gl_evidence": gl_evidence_map,
    }
    return result


def make_episode_failure_record(
    planned: Mapping[str, Any], *, reason: str, kind: str = "runtime", wall_time_seconds: float = 0.0
) -> dict[str, Any]:
    """Represent a terminal runtime/schema failure without a retry."""

    if kind not in {"runtime", "schema", "numerical", "crashed"}:
        raise BaselineSchemaError("failure kind is not recognized")
    planned_row = dict(_mapping(planned, "planned episode", BaselineSchemaError))
    _string(planned_row.get("episode_id"), "planned episode_id", BaselineSchemaError)
    _string(reason, "failure reason", BaselineSchemaError)
    wall = float(_finite_nonnegative(wall_time_seconds, "wall_time_seconds"))
    return {
        **planned_row,
        "status": "crashed",
        "failure_kind": kind,
        "failure_reason": reason,
        "success": False,
        "terminated": False,
        "truncated": False,
        "reward_sum": None,
        "rewards": [],
        "steps": 0,
        "policy_calls": 0,
        "inference_mean_seconds": None,
        "inference_p50_seconds": None,
        "inference_p95_seconds": None,
        "env_step_mean_seconds": None,
        "env_step_p50_seconds": None,
        "env_step_p95_seconds": None,
        "model_peak_memory_bytes": None,
        "model_peak_memory": None,
        "actions": [],
        "action_evidence": [],
        "actions_finite": None,
        "actions_dtype": None,
        "actions_shape": [],
        "offline": True,
        "hub_fallback": False,
        "attempt_count": 1,
        "no_retry": True,
        "wall_time_seconds": wall,
        "crashed": True,
    }


def _planned_rows(planned: int | Sequence[Mapping[str, Any]]) -> tuple[int, list[Mapping[str, Any]] | None]:
    if isinstance(planned, bool):
        raise BaselineSchemaError("planned must be an integer or matrix")
    if isinstance(planned, int):
        if planned < 0:
            raise BaselineSchemaError("planned must be non-negative")
        return planned, None
    if not isinstance(planned, Sequence) or isinstance(planned, (str, bytes)):
        raise BaselineSchemaError("planned must be an integer or matrix")
    rows = list(planned)
    return len(rows), rows


def _validate_episode_row(row: Mapping[str, Any], planned_by_id: Mapping[str, Mapping[str, Any]]) -> None:
    episode_id = _string(row.get("episode_id"), "episode_id", BaselineSchemaError)
    if episode_id not in planned_by_id:
        raise BaselineSchemaError(f"episode {episode_id} is not in the planned matrix")
    if row.get("attempt_count") != 1 or row.get("no_retry") is not True:
        raise BaselineSchemaError(f"episode {episode_id} violates one-attempt/no-retry contract")
    planned = planned_by_id[episode_id]
    for field in (
        "order",
        "trajectory_id",
        "suite",
        "task_id",
        "task_name",
        "init_state_id",
        "seed",
        "worker",
        "physical_device",
        "logical_device",
        "horizon",
    ):
        if row.get(field) != planned.get(field):
            raise BaselineSchemaError(f"episode {episode_id} {field} does not match planned matrix")
    status = row.get("status")
    if status == "completed":
        if not isinstance(row.get("success"), bool):
            raise BaselineSchemaError(f"episode {episode_id} success must be boolean")
        if not isinstance(row.get("terminated"), bool) or not isinstance(row.get("truncated"), bool):
            raise BaselineSchemaError(f"episode {episode_id} termination flags must be boolean")
        if row["terminated"] and row["truncated"]:
            raise BaselineSchemaError(f"episode {episode_id} termination flags conflict")
        steps = _int(row.get("steps"), f"episode {episode_id}.steps", error=BaselineSchemaError)
        horizon = _int(planned.get("horizon"), f"episode {episode_id}.horizon", error=BaselineSchemaError)
        reason = row.get("termination_reason")
        expected_reason = termination_reason(
            terminated=row["terminated"],
            truncated=row["truncated"],
            success=row["success"],
            steps=steps,
            horizon=horizon,
        )
        if reason != expected_reason:
            raise BaselineSchemaError(f"episode {episode_id} termination reason is inconsistent")
        if row.get("policy_calls") != steps:
            raise BaselineSchemaError(f"episode {episode_id} policy_calls must equal steps")
        rewards = row.get("rewards")
        if not isinstance(rewards, Sequence) or isinstance(rewards, (str, bytes)) or len(rewards) != steps:
            raise BaselineSchemaError(f"episode {episode_id} rewards length is invalid")
        reward_values = [
            _finite_number(value, f"episode {episode_id}.rewards[{index}]")
            for index, value in enumerate(rewards)
        ]
        reward_sum = _finite_number(row.get("reward_sum"), f"episode {episode_id}.reward_sum")
        if not math.isclose(reward_sum, float(sum(reward_values)), rel_tol=1e-9, abs_tol=1e-9):
            raise BaselineSchemaError(f"episode {episode_id} reward_sum does not match rewards")
        for field, raw_name in (
            ("inference_latencies", "inference_latencies"),
            ("env_step_latencies", "env_step_latencies"),
        ):
            values = row.get(field)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != steps:
                raise BaselineSchemaError(f"episode {episode_id} {field} length is invalid")
            summary = _latency_summary(values, f"episode {episode_id}.{raw_name}")
            prefix = "inference" if field.startswith("inference") else "env_step"
            for statistic in ("mean", "p50", "p95"):
                actual = _finite_nonnegative(
                    row.get(f"{prefix}_{statistic}_seconds"),
                    f"episode {episode_id}.{prefix}_{statistic}_seconds",
                )
                expected = summary[statistic]
                if expected is None or not math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-9):
                    raise BaselineSchemaError(
                        f"episode {episode_id} {prefix}_{statistic}_seconds does not match latencies"
                    )
        _finite_nonnegative(row.get("wall_time_seconds"), f"episode {episode_id}.wall_time_seconds")
        _int(
            row.get("model_peak_memory_bytes"),
            f"episode {episode_id}.model_peak_memory_bytes",
            nonnegative=True,
            error=BaselineSchemaError,
        )
        if row.get("selected_init_state_id") != planned.get("init_state_id"):
            raise BaselineSchemaError(f"episode {episode_id} selected_init_state_id does not match planned")
        if row.get("verified_before_reset") is not True:
            raise BaselineSchemaError(f"episode {episode_id} init state was not verified before reset")
        if row.get("post_reset_init_state_id") != int(planned.get("init_state_id")) + 1:
            raise BaselineSchemaError(f"episode {episode_id} post_reset_init_state_id is invalid")
        task_source = _mapping(
            row.get("task_source_evidence"), f"episode {episode_id}.task_source_evidence", BaselineSchemaError
        )
        for field in ("suite", "task_id", "task_name"):
            if task_source.get(field) != planned.get(field):
                raise BaselineSchemaError(f"episode {episode_id} task_source_evidence.{field} is invalid")
        try:
            validate_worker_evidence(
                _mapping(row.get("worker_evidence"), f"episode {episode_id}.worker_evidence", BaselineSchemaError),
                worker=str(planned.get("worker")),
                physical_device=int(planned.get("physical_device")),
            )
        except BaselineError as exc:
            raise BaselineSchemaError(f"episode {episode_id} worker evidence is invalid: {exc}") from exc
        gl = _mapping(row.get("gl_evidence"), f"episode {episode_id}.gl_evidence", BaselineSchemaError)
        try:
            validate_renderer_environment(
                _mapping(gl.get("variables"), f"episode {episode_id}.gl_evidence.variables", BaselineSchemaError),
                gl_identity=_mapping(gl.get("gl_identity"), f"episode {episode_id}.gl_evidence.gl_identity", BaselineSchemaError),
            )
        except BaselineError as exc:
            raise BaselineSchemaError(f"episode {episode_id} gl evidence is invalid: {exc}") from exc
        if row.get("offline") is not True or row.get("hub_fallback") is not False:
            raise BaselineSchemaError(f"episode {episode_id} offline/no-Hub evidence is invalid")
        if row.get("actions_dtype") != "float32" or row.get("actions_finite") is not True:
            raise BaselineSchemaError(f"episode {episode_id} action dtype/finite evidence is invalid")
        actions = row.get("actions")
        evidence = row.get("action_evidence")
        shapes = row.get("actions_shape")
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise BaselineSchemaError(f"episode {episode_id} actions must be a sequence")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)) or len(evidence) != steps:
            raise BaselineSchemaError(f"episode {episode_id} action evidence length is invalid")
        if not isinstance(shapes, Sequence) or isinstance(shapes, (str, bytes)) or len(shapes) != steps:
            raise BaselineSchemaError(f"episode {episode_id} action shapes are invalid")
        if len(actions) != steps:
            raise BaselineSchemaError(f"episode {episode_id} actions length is invalid")
        for index, (action, item, shape) in enumerate(zip(actions, evidence, shapes, strict=True)):
            if not isinstance(item, Mapping) or item.get("dtype") != "float32" or item.get("finite") is not True:
                raise BaselineSchemaError(f"episode {episode_id} action evidence {index} is invalid")
            if shape != [1, EXPECTED_ACTION_DIM] or item.get("shape") != [1, EXPECTED_ACTION_DIM]:
                raise BaselineSchemaError(f"episode {episode_id} action shape {index} is invalid")
            try:
                checked = np.asarray(action, dtype=np.float32)
                checked_evidence = np.asarray(item.get("values"), dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise BaselineSchemaError(f"episode {episode_id} action {index} is not numeric") from exc
            validate_action(checked)
            validate_action(checked_evidence)
            if checked.tolist() != checked_evidence.tolist():
                raise BaselineSchemaError(f"episode {episode_id} action {index} evidence does not match action")
        if row.get("crashed") is not False:
            raise BaselineSchemaError(f"episode {episode_id} completed row cannot be crashed")
    elif status == "crashed":
        if row.get("success") is not False or row.get("crashed") is not True:
            raise BaselineSchemaError(f"episode {episode_id} crashed row flags are invalid")
        if not isinstance(row.get("failure_kind"), str) or not row.get("failure_kind"):
            raise BaselineSchemaError(f"episode {episode_id} failure kind is missing")
        _string(row.get("failure_reason"), f"episode {episode_id}.failure_reason", BaselineSchemaError)
        if row.get("offline") is not True or row.get("hub_fallback") is not False:
            raise BaselineSchemaError(f"episode {episode_id} crashed offline/no-Hub evidence is invalid")
        if row.get("steps") != 0 or row.get("policy_calls") != 0:
            raise BaselineSchemaError(f"episode {episode_id} crashed row must have zero steps/calls")
        if row.get("actions") != [] or row.get("action_evidence") != [] or row.get("actions_shape") != []:
            raise BaselineSchemaError(f"episode {episode_id} crashed row must have empty action evidence")
        if row.get("actions_finite") is not None or row.get("actions_dtype") is not None:
            raise BaselineSchemaError(f"episode {episode_id} crashed action evidence must be null")
    else:
        raise BaselineSchemaError(f"episode {episode_id} has unknown status")


def _worker_metrics(
    records: Sequence[Mapping[str, Any]],
    worker: str,
    *,
    worker_metrics: Mapping[str, Any] | None,
    total_wall_time_seconds: float,
) -> dict[str, Any]:
    worker_records = [row for row in records if row.get("worker") == worker]
    provided = worker_metrics.get(worker, {}) if isinstance(worker_metrics, Mapping) else {}
    provided_map = _mapping(provided, f"worker_metrics.{worker}", BaselineSchemaError)
    busy_default = sum(float(row.get("wall_time_seconds", 0.0)) for row in worker_records)
    # A child span includes startup, environment setup, and rollout.  Falling
    # back to the parent span keeps an omitted metric conservative and avoids
    # manufacturing 100% utilization from episode busy time alone.
    wall_default = total_wall_time_seconds
    busy = float(provided_map.get("busy_time_seconds", busy_default))
    wall = float(provided_map.get("wall_time_seconds", wall_default or total_wall_time_seconds))
    _finite_nonnegative(busy, f"worker_metrics.{worker}.busy_time_seconds")
    _finite_nonnegative(wall, f"worker_metrics.{worker}.wall_time_seconds")
    if busy > wall:
        raise BaselineSchemaError(f"worker_metrics.{worker}.busy_time_seconds exceeds wall_time_seconds")
    completed = sum(row.get("status") == "completed" for row in worker_records)
    successes = sum(row.get("status") == "completed" and row.get("success") is True for row in worker_records)
    failed = sum(row.get("status") == "completed" and row.get("success") is False for row in worker_records)
    crashed = sum(row.get("status") == "crashed" for row in worker_records)
    throughput = completed / wall if wall else 0.0
    utilization = busy / wall if wall else 0.0
    return {
        "planned": 0,
        "attempted": len(worker_records),
        "completed": completed,
        "successes": successes,
        "failed": failed,
        "crashed": crashed,
        "busy_time_seconds": busy,
        "wall_time_seconds": wall,
        "throughput_episodes_per_second": throughput,
        # This is process busy-time / observed process wall-time, not hardware utilization.
        "utilization": utilization,
        "utilization_definition": "recorded serial worker busy_time_seconds / observed worker wall_time_seconds; not hardware utilization",
    }


def _worker_metrics_contract(
    metrics: Mapping[str, Any] | None,
    *,
    require_explicit: bool,
    required_workers: Sequence[str] | None = None,
) -> bool:
    """Check genuine child timing metrics and their derived quantities."""

    if not isinstance(metrics, Mapping):
        return False
    if require_explicit:
        if required_workers is None:
            workers = list(WORKERS)
        else:
            workers = list(required_workers)
            if not workers or any(worker not in WORKERS for worker in workers):
                return False
        if any(worker not in metrics for worker in workers):
            return False
    else:
        workers = list(WORKERS)
    for worker in workers:
        item = metrics.get(worker)
        if not isinstance(item, Mapping):
            return False
        if require_explicit and not {"wall_time_seconds", "busy_time_seconds"}.issubset(item):
            return False
        try:
            wall = _finite_nonnegative(item.get("wall_time_seconds"), f"worker_metrics.{worker}.wall_time_seconds")
            busy = _finite_nonnegative(item.get("busy_time_seconds"), f"worker_metrics.{worker}.busy_time_seconds")
            if wall is None or wall <= 0 or busy is None or busy > wall:
                return False
            throughput = _finite_nonnegative(
                item.get("throughput_episodes_per_second"),
                f"worker_metrics.{worker}.throughput_episodes_per_second",
            )
            utilization = _finite_nonnegative(
                item.get("utilization"), f"worker_metrics.{worker}.utilization"
            )
            completed = _int(item.get("completed"), f"worker_metrics.{worker}.completed", nonnegative=True)
            if throughput is None or utilization is None:
                return False
            if not math.isclose(throughput, completed / wall, rel_tol=1e-9, abs_tol=1e-12):
                return False
            if not math.isclose(utilization, busy / wall, rel_tol=1e-9, abs_tol=1e-12):
                return False
        except (BaselineError, TypeError, ValueError):
            return False
    return True


def _gate_map(
    *,
    matrix_ok: bool,
    records: Sequence[Mapping[str, Any]],
    planned_count: int,
    runtime_failures: Sequence[Mapping[str, Any]],
    numerical_anomalies: Sequence[Mapping[str, Any]],
    worker_metrics_ok: bool,
) -> dict[str, bool]:
    attempts_ok = all(row.get("attempt_count") == 1 and row.get("no_retry") is True for row in records)
    actions_ok = all(
        row.get("status") != "completed"
        or (
            row.get("actions_finite") is True
            and row.get("actions_dtype") == "float32"
            and isinstance(row.get("actions"), Sequence)
            and all(
                isinstance(action, Sequence)
                and tuple(np.asarray(action, dtype=np.float32).shape) == (1, EXPECTED_ACTION_DIM)
                and bool(np.isfinite(np.asarray(action, dtype=np.float32)).all())
                for action in row.get("actions", [])
            )
        )
        for row in records
    )
    offline_ok = all(
        row.get("offline") is True
        and row.get("hub_fallback") is False
        for row in records
    )
    no_runtime_failure = not runtime_failures
    no_numerical_anomaly = not numerical_anomalies
    return {
        "A": bool(matrix_ok),
        "B": attempts_ok,
        "C": actions_ok,
        "D": offline_ok,
        "E": no_runtime_failure,
        "F": no_numerical_anomaly,
        # A planned episode is accepted only after one terminal record.  This
        # keeps a partial run from being presented as a clean zero-SR result.
        "G": bool(planned_count == len(records)),
        # Child wall/busy spans are required for an accepted positive result;
        # this is linked into the manifest so a parent cannot silently invent
        # throughput or utilization from episode rows.
        "H": bool(worker_metrics_ok),
    }


def aggregate_results(
    episodes: Sequence[Mapping[str, Any]],
    *,
    planned: int | Sequence[Mapping[str, Any]],
    runtime_failures: Sequence[Mapping[str, Any]] | None = None,
    numerical_anomalies: Sequence[Mapping[str, Any]] | None = None,
    total_wall_time_seconds: float | None = None,
    worker_metrics: Mapping[str, Any] | None = None,
    require_worker_metrics: bool = False,
    worker_scope: str | None = None,
) -> dict[str, Any]:
    """Aggregate episode evidence and assign the conservative final verdict."""

    if worker_scope is not None:
        if worker_scope not in WORKERS:
            raise BaselineSchemaError("worker_scope must be A or B")
        if not require_worker_metrics:
            raise BaselineSchemaError("worker_scope requires explicit worker metrics")

    planned_count, planned_rows = _planned_rows(planned)
    planned_by_id: dict[str, Mapping[str, Any]] = {}
    matrix_ok = True
    if planned_rows is not None:
        for index, row in enumerate(planned_rows):
            item = _mapping(row, f"planned[{index}]", BaselineSchemaError)
            episode_id = _string(item.get("episode_id"), f"planned[{index}].episode_id", BaselineSchemaError)
            if episode_id in planned_by_id:
                raise BaselineSchemaError(f"duplicate planned episode ID {episode_id}")
            planned_by_id[episode_id] = item
        matrix_ok = len(planned_by_id) == planned_count
    records = list(episodes)
    seen: set[str] = set()
    for index, row in enumerate(records):
        item = _mapping(row, f"episodes[{index}]", BaselineSchemaError)
        episode_id = _string(item.get("episode_id"), f"episodes[{index}].episode_id", BaselineSchemaError)
        if episode_id in seen:
            raise BaselineSchemaError(f"duplicate episode ID {episode_id}")
        seen.add(episode_id)
        if planned_rows is not None:
            _validate_episode_row(item, planned_by_id)
        elif item.get("attempt_count") != 1 or item.get("no_retry") is not True:
            raise BaselineSchemaError(f"episode {episode_id} violates one-attempt/no-retry contract")
    runtime_failures_list = [dict(_mapping(row, "runtime failure", BaselineSchemaError)) for row in (runtime_failures or [])]
    anomalies_list = [dict(_mapping(row, "numerical anomaly", BaselineSchemaError)) for row in (numerical_anomalies or [])]
    successful = [row for row in records if row.get("status") == "completed" and row.get("success") is True]
    completed = [row for row in records if row.get("status") == "completed"]
    failed = [row for row in completed if row.get("success") is False]
    crashed_records = [row for row in records if row.get("status") == "crashed"]
    # Episode-linked runtime failures are represented by their terminal crashed
    # row.  Count worker-level failures without an episode row separately, so
    # the same failed attempt cannot be counted twice.
    crashed_record_ids = {
        str(row.get("episode_id")) for row in crashed_records if row.get("episode_id") is not None
    }
    unmatched_runtime_failures = [
        failure
        for failure in runtime_failures_list
        if str(failure.get("episode_id")) not in crashed_record_ids
    ]
    crashed = len(crashed_records) + len(unmatched_runtime_failures)
    total_wall = float(total_wall_time_seconds) if total_wall_time_seconds is not None else sum(float(row.get("wall_time_seconds", 0.0)) for row in records)
    _finite_nonnegative(total_wall, "total_wall_time_seconds")
    overall_sr = len(successful) / len(completed) if completed else 0.0
    per_task: dict[str, dict[str, Any]] = {}
    task_ids = sorted({str(row.get("task_id")) for row in (planned_rows or records) if row.get("task_id") is not None})
    for task_id in task_ids:
        task_completed = [row for row in completed if str(row.get("task_id")) == task_id]
        task_successes = [row for row in task_completed if row.get("success") is True]
        per_task[task_id] = {
            "successes": len(task_successes),
            "completed": len(task_completed),
            "sr": len(task_successes) / len(task_completed) if task_completed else 0.0,
        }
    mean_steps = statistics.mean(int(row["steps"]) for row in successful) if successful else None
    unattempted = [
        str(row.get("episode_id"))
        for row in (planned_rows or [])
        if str(row.get("episode_id")) not in seen
    ]
    metrics: dict[str, dict[str, Any]] = {}
    planned_by_worker = {worker: 0 for worker in WORKERS}
    if planned_rows is not None:
        for row in planned_rows:
            if row.get("worker") in planned_by_worker:
                planned_by_worker[str(row["worker"])] += 1
    for worker in WORKERS:
        metrics[worker] = _worker_metrics(
            records,
            worker,
            worker_metrics=worker_metrics,
            total_wall_time_seconds=total_wall,
        )
        metrics[worker]["planned"] = planned_by_worker[worker]
    metrics_ok = _worker_metrics_contract(
        worker_metrics if (require_worker_metrics or successful) else metrics,
        require_explicit=require_worker_metrics or bool(successful),
        required_workers=(worker_scope,) if worker_scope is not None else None,
    )
    gates = _gate_map(
        matrix_ok=matrix_ok,
        records=records,
        planned_count=planned_count,
        runtime_failures=runtime_failures_list,
        numerical_anomalies=anomalies_list,
        worker_metrics_ok=metrics_ok,
    )
    non_metric_gates_ok = all(value for key, value in gates.items() if key != "H")
    if runtime_failures_list or anomalies_list or crashed or not non_metric_gates_ok or not metrics_ok:
        verdict = "BLOCKED"
    elif overall_sr == 0.0:
        verdict = "NEEDS_INVESTIGATION"
    else:
        verdict = "PASS"
    return {
        "schema_version": SCHEMA_VERSION,
        "total": planned_count,
        "planned": planned_count,
        "attempted": len(records),
        "completed": len(completed),
        "failed": len(failed),
        "crashed": crashed,
        "successes": len(successful),
        "overall_sr": overall_sr,
        "overall_success_rate": overall_sr,
        "per_task_sr": {task_id: data["sr"] for task_id, data in per_task.items()},
        "per_task": per_task,
        "mean_steps_to_success": float(mean_steps) if mean_steps is not None else None,
        "total_wall_time_seconds": total_wall,
        "per_worker": metrics,
        "runtime_failures": runtime_failures_list,
        "numerical_anomalies": anomalies_list,
        "unattempted_episode_ids": unattempted,
        "gates": gates,
        "final_verdict": verdict,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False, separators=(",", ":")))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return (json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BaselineSchemaError(f"value is not strict JSON: {exc}") from exc


def _canonical_json_line(value: Any) -> bytes:
    try:
        return (
            json.dumps(_json_safe(value), sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BaselineSchemaError(f"value is not strict JSON: {exc}") from exc


class JournalIntegrityError(BaselineSchemaError):
    """Malformed journal with the valid prefix retained for recovery."""

    def __init__(self, message: str, *, partial_events: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.partial_events = [dict(item) for item in partial_events]


def _safe_episode_id(value: Any, name: str = "episode_id") -> str:
    episode_id = _string(value, name, BaselineSchemaError)
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in episode_id)


def _fsync_directory(path: str | Path) -> None:
    """Durably publish a just-created immutable entry when the platform allows it."""

    target = Path(path).resolve()
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(target, flags)
    except OSError:
        # Some test filesystems do not expose directory fsync.  The file
        # itself was still fsynced before publication; do not hide the useful
        # evidence behind a portability-only failure.
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _journal_event_path(path: str | Path, planned: Mapping[str, Any], event: str) -> Path:
    row = _mapping(planned, "planned episode", BaselineSchemaError)
    order = _int(row.get("order"), "planned order", nonnegative=True, error=BaselineSchemaError)
    episode_id = _safe_episode_id(row.get("episode_id"))
    if event == "attempt_start":
        suffix = "attempt-start"
    elif event == "terminal":
        suffix = "terminal"
    else:
        raise BaselineSchemaError("episode journal event must be attempt_start or terminal")
    return Path(path).resolve() / f"{order:04d}-{episode_id}.{suffix}.json"


def _journal_error(message: str, events: Sequence[Mapping[str, Any]]) -> JournalIntegrityError:
    return JournalIntegrityError(message, partial_events=events)


def read_episode_journal(path: str | Path) -> list[dict[str, Any]]:
    """Read independent immutable journal entries in deterministic order.

    A malformed entry fails closed, but ``JournalIntegrityError.partial_events``
    retains the valid prefix so parent recovery can keep earlier attempts.
    """

    target = Path(path).resolve()
    if not target.exists():
        return []
    if target.is_symlink() or not target.is_dir():
        raise _journal_error(f"episode journal must be a regular directory: {target}", [])
    events: list[dict[str, Any]] = []
    state: dict[str, str] = {}
    try:
        entries = sorted(target.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise _journal_error(f"episode journal directory is unreadable: {target}: {exc}", events) from exc
    for entry in entries:
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
            raise _journal_error(f"episode journal entry is not an immutable JSON file: {entry}", events)
        try:
            raw = entry.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            event = _mapping(parsed, f"episode journal entry {entry.name}", BaselineSchemaError)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, BaselineSchemaError) as exc:
            raise _journal_error(f"episode journal entry is malformed: {entry}", events) from exc
        event_map = dict(event)
        try:
            episode_id = _string(
                event_map.get("episode_id"),
                f"episode journal entry {entry.name}.episode_id",
                BaselineSchemaError,
            )
            kind = event_map.get("event")
            if kind not in {"attempt_start", "terminal"}:
                raise BaselineSchemaError(f"episode journal entry event is invalid: {entry.name}")
            prior = state.get(episode_id)
            if kind == "attempt_start":
                if prior is not None:
                    raise BaselineSchemaError(f"episode journal has duplicate attempt for {episode_id}")
                state[episode_id] = "started"
            else:
                if prior != "started":
                    raise BaselineSchemaError(f"episode journal terminal has no unique attempt start for {episode_id}")
                if event_map.get("status") not in {"completed", "crashed"}:
                    raise BaselineSchemaError(f"episode journal terminal status is invalid for {episode_id}")
                state[episode_id] = "terminal"
        except BaselineSchemaError as exc:
            raise _journal_error(str(exc), events) from exc
        event_map["_path"] = str(entry)
        events.append(event_map)
    return events


def append_episode_journal(
    path: str | Path,
    planned: Mapping[str, Any],
    *,
    event: str,
    status: str | None = None,
) -> Path:
    """Record exactly one attempt-start and one terminal immutable event."""

    planned_row = _mapping(planned, "planned episode", BaselineSchemaError)
    episode_id = _string(planned_row.get("episode_id"), "planned episode_id", BaselineSchemaError)
    if event not in {"attempt_start", "terminal"}:
        raise BaselineSchemaError("episode journal event must be attempt_start or terminal")
    if event == "terminal" and status not in {"completed", "crashed"}:
        raise BaselineSchemaError("episode journal terminal status must be completed or crashed")
    journal = Path(path).resolve()
    if journal.exists() and (journal.is_symlink() or not journal.is_dir()):
        raise BaselineSchemaError(f"episode journal must be a regular directory: {journal}")
    journal.mkdir(parents=True, exist_ok=True)
    existing = read_episode_journal(journal)
    state = {item["episode_id"]: item["event"] for item in existing}
    if event == "attempt_start" and episode_id in state:
        raise BaselineSchemaError(f"episode {episode_id} already has a journal attempt")
    if event == "terminal" and state.get(episode_id) != "attempt_start":
        raise BaselineSchemaError(f"episode {episode_id} terminal journal requires one attempt start")
    target = _journal_event_path(journal, planned_row, event)
    payload = {
        "event": event,
        "episode_id": episode_id,
        "trajectory_id": planned_row.get("trajectory_id"),
        "order": planned_row.get("order"),
        "worker": planned_row.get("worker"),
        "physical_device": planned_row.get("physical_device"),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if status is not None:
        payload["status"] = status
    written = write_json_no_overwrite(target, payload)
    _fsync_directory(journal)
    return written


def _journal_row_bijection(
    events: Sequence[Mapping[str, Any]],
    *,
    planned: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the row/event bijection and return recoverable crash rows."""

    planned_rows = list(planned)
    planned_by_id = {
        _string(row.get("episode_id"), "planned episode_id", BaselineSchemaError): row for row in planned_rows
    }
    existing_rows = list(existing)
    existing_by_id: dict[str, Mapping[str, Any]] = {}
    for row in existing_rows:
        episode_id = _string(row.get("episode_id"), "existing episode_id", BaselineSchemaError)
        if episode_id in existing_by_id:
            raise BaselineSchemaError(f"duplicate existing episode {episode_id}")
        if episode_id not in planned_by_id:
            raise BaselineSchemaError(f"existing episode {episode_id} is not planned")
        existing_by_id[episode_id] = row
    starts = {str(item.get("episode_id")): item for item in events if item.get("event") == "attempt_start"}
    terminals = {str(item.get("episode_id")): item for item in events if item.get("event") == "terminal"}
    problems: list[str] = []
    for episode_id in set(starts) | set(terminals) | set(existing_by_id):
        if episode_id not in planned_by_id:
            problems.append(f"journal/row episode {episode_id} is not planned")
    recovered: list[dict[str, Any]] = []
    for row in sorted(planned_rows, key=lambda item: int(item.get("order", 0))):
        episode_id = str(row.get("episode_id"))
        start = starts.get(episode_id)
        terminal = terminals.get(episode_id)
        existing_row = existing_by_id.get(episode_id)
        if start is None and (terminal is not None or existing_row is not None):
            problems.append(f"episode {episode_id} has a row/terminal without attempt_start")
        if terminal is None:
            if existing_row is not None:
                problems.append(f"episode {episode_id} row has no terminal journal event")
            elif start is not None:
                recovered.append(
                    make_episode_failure_record(
                        row,
                        reason="child interrupted after attempt start without a terminal record",
                        kind="crashed",
                    )
                )
        elif existing_row is None:
            problems.append(f"episode {episode_id} terminal journal event has no row")
        elif terminal.get("status") != existing_row.get("status"):
            problems.append(f"episode {episode_id} terminal status does not match row status")
        if start is not None:
            for field in ("order", "worker", "physical_device", "trajectory_id"):
                if start.get(field) != row.get(field):
                    problems.append(f"episode {episode_id} attempt_start {field} does not match planned row")
        if terminal is not None:
            for field in ("order", "worker", "physical_device", "trajectory_id"):
                if terminal.get(field) != row.get(field):
                    problems.append(f"episode {episode_id} terminal {field} does not match planned row")
    return recovered, problems


def recover_journal_episodes(
    path: str | Path,
    *,
    planned: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Synthesize one crashed row for each started-but-unfinished episode."""

    events = read_episode_journal(path)
    recovered, problems = _journal_row_bijection(events, planned=planned, existing=existing)
    if problems:
        raise BaselineSchemaError("; ".join(problems))
    return recovered


def write_json_no_overwrite(path: str | Path, value: Any) -> Path:
    """Atomically create one JSON artifact and reject every later write."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json(value)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(target.parent)
    return target


def write_text_no_overwrite(path: str | Path, value: str) -> Path:
    if not isinstance(value, str):
        raise BaselineSchemaError("text artifact must be a string")
    return _write_bytes_no_overwrite(path, value.encode("utf-8"))


def _write_bytes_no_overwrite(path: str | Path, data: bytes) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(target.parent)
    return target


def _write_bytes_if_absent_or_identical(path: str | Path, data: bytes) -> Path:
    """Keep an incrementally journaled artifact immutable once created."""

    target = Path(path).resolve()
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError(f"artifact already exists with different content: {target}")
        return target
    return _write_bytes_no_overwrite(target, data)


def _episode_row_path(child_directory: str | Path, row: Mapping[str, Any]) -> Path:
    item = _mapping(row, "episode row", BaselineSchemaError)
    order = _int(item.get("order"), "episode row order", nonnegative=True, error=BaselineSchemaError)
    episode_id = _safe_episode_id(item.get("episode_id"))
    return Path(child_directory).resolve() / EPISODE_ROWS_NAME / f"{order:04d}-{episode_id}.json"


def persist_episode_row(child_directory: str | Path, row: Mapping[str, Any]) -> Path:
    """Persist one terminal row as an immutable per-episode JSON artifact."""

    target = _episode_row_path(child_directory, row)
    written = write_json_no_overwrite(target, row)
    _fsync_directory(target.parent)
    return written


def terminalize_child(
    path: str | Path,
    *,
    worker: str,
    physical_device: int,
    planned: Sequence[Mapping[str, Any]],
    attempted: Sequence[Mapping[str, Any]],
    reason: str,
    status: str = "TERMINATED",
    returncode: int | None = None,
    worker_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one terminal child manifest with explicit unattempted IDs."""

    if worker not in WORKERS:
        raise BaselineSchemaError("child worker must be A or B")
    if physical_device != EXPECTED_PHYSICAL_DEVICES[WORKERS.index(worker)]:
        raise BaselineSchemaError("child physical device does not match worker")
    if status not in {"TERMINATED", "CRASHED", "FAIL"}:
        raise BaselineSchemaError("child terminal status is not recognized")
    _string(reason, "child terminal reason", BaselineSchemaError)
    planned_rows = list(planned)
    attempted_rows = list(attempted)
    planned_ids = [_string(row.get("episode_id"), "planned episode_id", BaselineSchemaError) for row in planned_rows]
    attempted_ids = [_string(row.get("episode_id"), "attempted episode_id", BaselineSchemaError) for row in attempted_rows]
    if len(set(planned_ids)) != len(planned_ids) or len(set(attempted_ids)) != len(attempted_ids):
        raise BaselineSchemaError("child planned/attempted episode IDs must be unique")
    if not set(attempted_ids).issubset(planned_ids):
        raise BaselineSchemaError("child attempted episode is not planned")
    if returncode is not None:
        _int(returncode, "returncode", error=BaselineSchemaError)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": EXPECTED_NAME,
        "status": status,
        "worker": worker,
        "physical_device": physical_device,
        "logical_device": EXPECTED_LOGICAL_DEVICE,
        "scope": {
            "type": "worker",
            "worker": worker,
            "physical_device": physical_device,
            "planned_count": len(planned_rows),
            "episode_ids": planned_ids,
            "trajectory_ids": [row.get("trajectory_id") for row in planned_rows],
        },
        "matrix_count": len(planned_rows),
        "trajectory_ids": [row.get("trajectory_id") for row in planned_rows],
        "reason": reason,
        "returncode": returncode,
        "planned_episode_ids": planned_ids,
        "attempted_episode_ids": attempted_ids,
        "unattempted_episode_ids": [episode_id for episode_id in planned_ids if episode_id not in attempted_ids],
        "attempt_count": len(attempted_ids),
        "no_retry": True,
        "worker_metrics": dict(worker_metrics or {}),
        "terminalized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    write_json_no_overwrite(path, manifest)
    return manifest


def terminalize_parent(
    path: str | Path,
    *,
    config: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    reason: str,
    runtime_failures: Sequence[Mapping[str, Any]] | None = None,
    numerical_anomalies: Sequence[Mapping[str, Any]] | None = None,
    preflight_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a terminal parent manifest while retaining all partial evidence."""

    _string(reason, "parent terminal reason", BaselineSchemaError)
    aggregate = aggregate_results(
        episodes,
        planned=matrix,
        runtime_failures=runtime_failures or [{"reason": reason}],
        numerical_anomalies=numerical_anomalies,
    )
    normalized = validate_config(config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": EXPECTED_NAME,
        "status": "BLOCKED",
        "reason": reason,
        "matrix_count": len(matrix),
        "aggregate": aggregate,
        "gate_verdict": "BLOCKED",
        "unattempted_episode_ids": aggregate["unattempted_episode_ids"],
        "runtime": normalized["runtime"],
        "preflight": dict(preflight_evidence or {}),
        "provenance": {
            **normalized["provenance"],
            "seed": normalized["seed"],
            "action_noise_source": normalized["action_noise"]["source"],
            "perturbation": False,
            "offline": True,
            "hub_fallback": False,
        },
    }
    write_json_no_overwrite(path, manifest)
    return manifest


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(root: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, check=False
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise BaselineRuntimeError(f"could not resolve project HEAD: {completed.stderr.strip()}")
    return completed.stdout.strip()


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    run_directory: str | Path,
    project_sha: str,
    matrix: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    provenance: Mapping[str, Any],
    status: str = "VALIDATED",
    preflight_evidence: Mapping[str, Any] | None = None,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the parent manifest with complete pinned provenance."""

    normalized = validate_config(config)
    matrix_rows = [dict(row) for row in matrix]
    if scope is None:
        scope_map = {
            "type": "parent",
            "planned_count": len(matrix_rows),
            "episode_ids": [row.get("episode_id") for row in matrix_rows],
            "trajectory_ids": [row.get("trajectory_id") for row in matrix_rows],
        }
    else:
        scope_map = dict(_mapping(scope, "scope", BaselineSchemaError))
        scope_map.setdefault("type", "parent")
        scope_map.setdefault("planned_count", len(matrix_rows))
        scope_map.setdefault("episode_ids", [row.get("episode_id") for row in matrix_rows])
        scope_map.setdefault("trajectory_ids", [row.get("trajectory_id") for row in matrix_rows])
    _expected_scoped_matrix(matrix_rows, normalized, scope_map)
    aggregate_map = dict(_mapping(aggregate, "aggregate", BaselineSchemaError))
    provenance_map = dict(_mapping(provenance, "provenance", BaselineSchemaError))
    for field in (
        "git_sha",
        "lerobot_git_sha",
        "libero_git_sha",
        "checkpoint_revision",
        "python",
        "pytorch",
        "transformers",
        "hip",
        "gpu",
        "trajectory_ids",
    ):
        if field not in provenance_map:
            raise BaselineSchemaError(f"provenance.{field} is required")
    config_file = Path(config_path).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": EXPECTED_NAME,
        "status": status,
        "run_directory": str(Path(run_directory).resolve()),
        "project": {"git_sha": str(project_sha), "expected_project_sha": str(project_sha)},
        "config": {"path": str(config_file), "sha256": _sha256_file(config_file) if config_file.is_file() else None},
        "runtime": normalized["runtime"],
        "preflight": dict(preflight_evidence or {}),
        "checkpoint": normalized["checkpoint"],
        "base_model": normalized["base_model"],
        "assets": normalized["assets"],
        "episode_matrix": {
            "order": "task_major",
            "count": len(matrix_rows),
            "task_ids": normalized["task_ids"],
            "init_state_ids": normalized["init_state_ids"],
            "rows": matrix_rows,
        },
        "scope": scope_map,
        "aggregate": aggregate_map,
        "gates": aggregate_map.get("gates", {}),
        "gate_verdict": aggregate_map.get("final_verdict"),
        "no_retry": True,
        "provenance": {
            **provenance_map,
            "seed": normalized["seed"],
            "task_ids": normalized["task_ids"],
            "episode_ids": [row["episode_id"] for row in matrix_rows],
            "trajectory_ids": [row["trajectory_id"] for row in matrix_rows],
            "action_noise_source": normalized["action_noise"]["source"],
            "action_noise_seed": normalized["seed"],
            "perturbation": False,
            "offline": True,
            "hub_fallback": False,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    validate_manifest(manifest)
    return manifest


def _assert_json_finite(value: Any, path: str = "manifest") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BaselineSchemaError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")


def validate_manifest(manifest: Mapping[str, Any]) -> bool:
    """Validate parent/worker scope, counts, provenance, and verdict linkage."""

    root = _mapping(manifest, "manifest", BaselineSchemaError)
    if root.get("schema_version") != SCHEMA_VERSION or root.get("name") != EXPECTED_NAME:
        raise BaselineSchemaError("manifest schema identity is invalid")
    status = _string(root.get("status"), "manifest.status", BaselineSchemaError)
    if status not in {"VALIDATED", "PASS", "NEEDS_INVESTIGATION", "BLOCKED", "RUNNING"}:
        raise BaselineSchemaError("manifest status is invalid")
    project = _mapping(root.get("project"), "manifest.project", BaselineSchemaError)
    _string(project.get("git_sha"), "manifest.project.git_sha", BaselineSchemaError)
    config = _mapping(root.get("config"), "manifest.config", BaselineSchemaError)
    _string(config.get("path"), "manifest.config.path", BaselineSchemaError)
    matrix = _mapping(root.get("episode_matrix"), "manifest.episode_matrix", BaselineSchemaError)
    rows = matrix.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise BaselineSchemaError("manifest episode_matrix.rows must be a sequence")
    matrix_count = matrix.get("count")
    if matrix_count != len(rows):
        raise BaselineSchemaError("manifest episode_matrix.count does not match rows")
    row_ids = [_string(row.get("episode_id"), "manifest episode row episode_id", BaselineSchemaError) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise BaselineSchemaError("manifest episode_matrix rows contain duplicate episode IDs")
    scope = _mapping(root.get("scope"), "manifest.scope", BaselineSchemaError)
    scope_type = scope.get("type")
    if scope_type not in {"parent", "worker"}:
        raise BaselineSchemaError("manifest scope type is invalid")
    if scope.get("planned_count") != len(rows):
        raise BaselineSchemaError("manifest scope planned_count does not match matrix count")
    if scope.get("episode_ids") != row_ids:
        raise BaselineSchemaError("manifest scope episode_ids do not match matrix rows")
    trajectory_ids = [
        _string(row.get("trajectory_id"), "manifest episode row trajectory_id", BaselineSchemaError)
        for row in rows
    ]
    if scope.get("trajectory_ids") != trajectory_ids:
        raise BaselineSchemaError("manifest scope trajectory_ids do not match matrix rows")
    if scope_type == "worker":
        worker = scope.get("worker")
        physical_device = scope.get("physical_device")
        if worker not in WORKERS or physical_device != EXPECTED_PHYSICAL_DEVICES[WORKERS.index(worker)]:
            raise BaselineSchemaError("manifest worker scope identity is invalid: physical_device/worker mismatch")
        if any(row.get("worker") != worker or row.get("physical_device") != physical_device for row in rows):
            raise BaselineSchemaError("manifest worker scope does not match matrix rows")
    elif "worker" in scope or "physical_device" in scope:
        raise BaselineSchemaError("parent manifest scope cannot declare a worker")
    aggregate = _mapping(root.get("aggregate"), "manifest.aggregate", BaselineSchemaError)
    if aggregate.get("planned") != len(rows) or aggregate.get("total") != len(rows):
        raise BaselineSchemaError("manifest aggregate planned/total counts do not match matrix count")
    verdict = aggregate.get("final_verdict")
    if verdict not in {"PASS", "NEEDS_INVESTIGATION", "BLOCKED"}:
        raise BaselineSchemaError("manifest aggregate final_verdict is invalid")
    if root.get("gate_verdict") != verdict:
        raise BaselineSchemaError("manifest gate_verdict must match aggregate final_verdict")
    if root.get("gates", {}) != aggregate.get("gates", {}):
        raise BaselineSchemaError("manifest gates do not match aggregate gates")
    if root.get("no_retry") is not True:
        raise BaselineSchemaError("manifest no_retry must be true")
    if status not in {"VALIDATED", "RUNNING"} and status != verdict:
        raise BaselineSchemaError("manifest status does not match aggregate final_verdict")
    provenance = _mapping(root.get("provenance"), "manifest.provenance", BaselineSchemaError)
    for field in (
        "git_sha",
        "lerobot_git_sha",
        "libero_git_sha",
        "checkpoint_revision",
        "python",
        "pytorch",
        "transformers",
        "cuda",
        "hip",
        "gpu",
        "trajectory_ids",
        "episode_ids",
        "action_noise_source",
    ):
        if field not in provenance:
            raise BaselineSchemaError(f"manifest.provenance.{field} is required")
    if provenance.get("perturbation") is not False:
        raise BaselineSchemaError("manifest provenance perturbation must be false")
    if provenance.get("offline") is not True or provenance.get("hub_fallback") is not False:
        raise BaselineSchemaError("manifest provenance must prove offline/no-Hub-fallback execution")
    if provenance.get("trajectory_ids") != trajectory_ids:
        raise BaselineSchemaError("manifest provenance trajectory_ids do not match matrix rows")
    if provenance.get("episode_ids") != row_ids:
        raise BaselineSchemaError("manifest provenance episode_ids do not match matrix rows")
    runtime = _mapping(root.get("runtime"), "manifest.runtime", BaselineSchemaError)
    if runtime.get("logical_device") != EXPECTED_LOGICAL_DEVICE:
        raise BaselineSchemaError("manifest runtime logical device must be cuda:0")
    _assert_json_finite(root)
    return True


def build_child_command(
    *,
    config_path: str | Path,
    expected_project_sha: str,
    worker: str,
    physical_device: int,
    child_directory: str | Path,
    project_python: str | Path | None = None,
) -> list[str]:
    if worker not in WORKERS or physical_device != EXPECTED_PHYSICAL_DEVICES[WORKERS.index(worker)]:
        raise BaselineSchemaError("worker/device assignment is invalid")
    return [
        str(project_python or sys.executable),
        str(Path(__file__).resolve()),
        "--child",
        "--config",
        str(Path(config_path).resolve()),
        "--expected-project-sha",
        str(expected_project_sha),
        "--worker",
        worker,
        "--physical-device",
        str(physical_device),
        "--child-directory",
        str(Path(child_directory).resolve()),
    ]


def _open_exclusive_child_log(path: str | Path) -> Any:
    """Open one parent-owned binary log without replacing an earlier file."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    return os.fdopen(fd, "wb", buffering=0)


def _close_child_log_handles(metadata: dict[str, Any], *, timeout: float = 1.0) -> list[str]:
    """Flush/fsync/close parent-owned child logs under one finite deadline."""

    if metadata.get("log_handles_closed") is True:
        return list(metadata.get("log_cleanup_errors", []))
    handles = [metadata.get("stdout_handle"), metadata.get("stderr_handle")]
    handles = [handle for handle in handles if handle is not None]
    errors: list[str] = []
    deadline = time.monotonic() + float(timeout)
    for handle in handles:
        result: dict[str, Any] = {}

        def close_one(current: Any = handle) -> None:
            try:
                flush = getattr(current, "flush", None)
                if callable(flush):
                    flush()
                fileno = getattr(current, "fileno", None)
                if callable(fileno):
                    os.fsync(fileno())
                close = getattr(current, "close", None)
                if callable(close):
                    close()
            except BaseException as exc:  # pragma: no cover - filesystem failure
                result["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=close_one, name="baseline-a-log-close", daemon=True)
        thread.start()
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            errors.append("child log flush/fsync/close timed out")
        elif result.get("error"):
            errors.append(str(result["error"]))
    metadata["log_handles_closed"] = True
    metadata["log_cleanup_errors"] = errors
    metadata["log_cleanup"] = {
        "paths": [str(metadata[key]) for key in ("stdout_log", "stderr_log") if metadata.get(key)],
        "closed": not errors,
        "errors": list(errors),
    }
    return errors


def launch_children(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    expected_project_sha: str,
    run_directory: str | Path,
    popen: Any = subprocess.Popen,
    project_python: str | Path | None = None,
    wait: bool = True,
    collection_timeout_seconds: float | None = None,
    children_out: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Launch both assigned CPU children before waiting on either one.

    ``popen`` is an explicit seam for fake-only tests.  Production callers
    leave it at ``subprocess.Popen``; no child receives compute visibility.
    """

    normalized = validate_config(config)
    _string(expected_project_sha, "expected project SHA", BaselineSchemaError)
    matrix = build_episode_matrix(normalized)
    root = Path(run_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    children_root = root / "children"
    children_root.mkdir(exist_ok=False)
    python_executable = project_python or normalized["runtime"].get("cpu_python") or sys.executable
    children: list[dict[str, Any]] = children_out if children_out is not None else []
    children.clear()
    try:
        for worker, physical_device in zip(WORKERS, EXPECTED_PHYSICAL_DEVICES, strict=True):
            child_directory = children_root / f"worker-{worker}"
            command = build_child_command(
                config_path=config_path,
                expected_project_sha=expected_project_sha,
                worker=worker,
                physical_device=physical_device,
                child_directory=child_directory,
                project_python=python_executable,
            )
            environment = build_cpu_child_environment(normalized)
            config_path_resolved = Path(config_path).resolve()
            config_sha256 = _sha256_file(config_path_resolved)
            child_directory.mkdir(parents=True, exist_ok=False)
            stdout_log = child_directory / "process_stdout.log"
            stderr_log = child_directory / "process_stderr.log"
            metadata = {
                "worker": worker,
                "physical_device": physical_device,
                "logical_device": EXPECTED_LOGICAL_DEVICE,
                "planned": [row for row in matrix if row["worker"] == worker],
                "command": command,
                "environment": {
                    "HIP_VISIBLE_DEVICES": environment.get("HIP_VISIBLE_DEVICES"),
                    "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
                    "MUJOCO_EGL_DEVICE_ID": environment.get("MUJOCO_EGL_DEVICE_ID"),
                },
                "directory": child_directory,
                # Register the child as soon as its directory exists.  This
                # keeps a Popen/log-opening failure in the same cleanup and
                # terminal-evidence boundary as a successfully started child.
                "process": None,
                "pgid": None,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "stdout_handle": None,
                "stderr_handle": None,
                "expected_project_sha": str(expected_project_sha),
                "config_path": config_path_resolved,
                "config_sha256": config_sha256,
            }
            children.append(metadata)
            metadata["stdout_handle"] = _open_exclusive_child_log(stdout_log)
            metadata["stderr_handle"] = _open_exclusive_child_log(stderr_log)
            # This timestamp belongs to the parent and must be captured before
            # Popen so child-reported startup/rollout timing cannot replace it.
            metadata["parent_observed_start_monotonic"] = time.monotonic()
            process = popen(
                command,
                cwd=str(ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=metadata["stdout_handle"],
                stderr=metadata["stderr_handle"],
                shell=False,
                start_new_session=True,
            )
            # Store the process before inspecting its PID so an unusual Popen
            # object whose pid property raises is still cleanup-accounted.
            metadata["process"] = process
            process_pid = getattr(process, "pid", None)
            metadata["pgid"] = process_pid if isinstance(process_pid, int) and process_pid > 0 else None
    except BaseException as exc:
        for child in children:
            _terminate_child_process(child, f"launch failure: {type(exc).__name__}")
            _close_child_log_handles(child, timeout=1.0)
            try:
                child["ipc_cleanup"] = cleanup_child_ipc(Path(str(child["directory"])))
            except BaseException:
                pass
            child["returncode"] = getattr(child["process"], "returncode", None)
            try:
                terminalize_child(
                    Path(child["directory"]) / "terminal_manifest.json",
                    worker=child["worker"],
                    physical_device=child["physical_device"],
                    planned=child["planned"],
                    attempted=[],
                    reason=f"launch failure: {type(exc).__name__}: {exc}",
                    status="CRASHED",
                    returncode=child["returncode"],
                )
            except Exception:
                pass
        raise BaselineRuntimeError(f"child launch failed: {type(exc).__name__}: {exc}") from exc
    # This loop deliberately starts only after the launch loop above has
    # returned, proving that both children are live before the first wait.
    if wait:
        collect_children_fail_closed(
            children,
            timeout_seconds=(
                parent_collection_timeout_seconds(normalized)
                if collection_timeout_seconds is None
                else float(collection_timeout_seconds)
            ),
        )
    return children


def _process_group_descendants(pgid: int, *, exclude_pid: int | None = None) -> list[int]:
    """Return live PIDs in an owned process group other than its outer PID."""

    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 0:
        return []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    descendants: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if exclude_pid is not None and pid == exclude_pid:
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            close_paren = stat.rfind(")")
            fields = stat[close_paren + 2 :].split()
            # fields[0] is state (field 3); fields[2] is pgrp (field 5).
            if len(fields) >= 3 and fields[2] == str(pgid) and fields[0] != "Z":
                descendants.append(pid)
        except (OSError, UnicodeDecodeError, ValueError):
            continue
    return sorted(descendants)


def _process_is_alive(process: Any) -> bool:
    poll = getattr(process, "poll", None)
    if callable(poll):
        try:
            return poll() is None
        except BaseException:
            return True
    return getattr(process, "returncode", None) is None


def _terminate_process_bounded(process: Any) -> list[str]:
    """Escalate one outer process and verify that it is no longer alive."""

    if not _process_is_alive(process):
        return []
    errors: list[str] = []
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        try:
            terminate()
        except BaseException as exc:
            errors.append(f"outer process TERM failed: {type(exc).__name__}: {exc}")
    wait = getattr(process, "wait", None)
    if callable(wait):
        try:
            wait(timeout=1.0)
        except BaseException:
            pass
    if _process_is_alive(process):
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except BaseException as exc:
                errors.append(f"outer process KILL failed: {type(exc).__name__}: {exc}")
        if callable(wait):
            try:
                wait(timeout=1.0)
            except BaseException:
                pass
    if _process_is_alive(process):
        errors.append("outer process remained alive after bounded termination")
    return errors


def _cleanup_owned_process_group(child: Mapping[str, Any], reason: str) -> list[str]:
    """Boundedly kill and verify only a child-owned process group."""

    if not isinstance(child, dict):
        return []
    process = child.get("process")
    pgid = child.get("pgid") or getattr(process, "pid", None)
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 0:
        if _process_is_alive(process):
            error = "outer process remains alive without a valid process group"
            child["group_cleanup_error"] = error
            child.pop("group_cleanup_done", None)
            return [error]
        child["group_cleanup_done"] = True
        return []
    outer_pid = getattr(process, "pid", None)
    try:
        remaining = _process_group_descendants(pgid, exclude_pid=outer_pid)
    except BaseException as exc:  # pragma: no cover - defensive proc reader
        error = f"process-group verification failed: {type(exc).__name__}: {exc}"
        child["group_cleanup_error"] = error
        return [error]
    if remaining:
        try:
            from scripts import dcu_preflight

            killer = getattr(dcu_preflight, "_kill_concurrency_process", None)
            if not callable(killer):
                raise BaselineRuntimeError("dcu_preflight._kill_concurrency_process is unavailable")
            killer(child, reason=reason)
        except BaseException as exc:
            error = f"process-group cleanup failed: {type(exc).__name__}: {exc}"
            child["group_cleanup_error"] = error
            return [error]
        try:
            remaining = _process_group_descendants(pgid, exclude_pid=outer_pid)
        except BaseException as exc:  # pragma: no cover - defensive proc reader
            remaining = [pgid]
            child["group_cleanup_error"] = f"post-cleanup verification failed: {exc}"
        if remaining:
            error = f"owned process group still has descendants: {remaining}"
            child["group_cleanup_error"] = error
            child.pop("group_cleanup_done", None)
            return [error]
    if _process_is_alive(process):
        error = f"owned outer process remains alive: {outer_pid}"
        child["group_cleanup_error"] = error
        child.pop("group_cleanup_done", None)
        return [error]
    child["group_cleanup_done"] = True
    child["group_cleanup"] = {"pgid": pgid, "descendants_after": []}
    return []


def _terminate_child_process(child: Mapping[str, Any], reason: str) -> None:
    process = child.get("process")
    if process is None:
        return
    child_metadata = child if isinstance(child, dict) else None
    if child_metadata is not None and child_metadata.get("group_cleanup_done"):
        return
    if child_metadata is not None:
        child_metadata["termination_reason"] = reason
        child_metadata["forced_termination"] = True
    pid = child.get("pgid") or getattr(process, "pid", None)
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=1.0)
            except Exception:
                pass
        cleanup_errors = _terminate_process_bounded(process)
        cleanup_errors.extend(_cleanup_owned_process_group(child, reason))
        if cleanup_errors and child_metadata is not None:
            child_metadata["group_cleanup_error"] = "; ".join(cleanup_errors)
    else:
        cleanup_errors = _terminate_process_bounded(process)
        if cleanup_errors and child_metadata is not None:
            child_metadata["group_cleanup_error"] = "; ".join(cleanup_errors)
        elif child_metadata is not None:
            child_metadata["group_cleanup_done"] = True


def collect_children_fail_closed(
    children: Sequence[dict[str, Any]],
    *,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Collect both children with a bound and terminate siblings on failure."""

    timeout = float(_finite_nonnegative(timeout_seconds, "timeout_seconds"))
    if timeout <= 0:
        raise BaselineSchemaError("timeout_seconds must be positive")
    pending = list(children)
    failures: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while pending:
        progressed = False
        for child in list(pending):
            process = child.get("process")
            poll = getattr(process, "poll", None)
            returncode = poll() if callable(poll) else None
            if returncode is None:
                continue
            progressed = True
            child["returncode"] = getattr(process, "returncode", returncode)
            child["parent_observed_finish_monotonic"] = time.monotonic()
            _close_child_log_handles(child, timeout=max(0.0, deadline - time.monotonic()))
            if child.get("log_cleanup_errors"):
                failures.append(
                    {
                        "worker": child.get("worker"),
                        "reason": "child log cleanup failed: " + "; ".join(child["log_cleanup_errors"]),
                    }
                )
            cleanup_errors = _cleanup_owned_process_group(child, "child exited")
            if cleanup_errors:
                failures.extend({"worker": child.get("worker"), "reason": error} for error in cleanup_errors)
            try:
                child["ipc_cleanup"] = cleanup_child_ipc(Path(str(child["directory"])))
            except BaseException as exc:
                failures.append({"worker": child.get("worker"), "reason": f"orphan IPC cleanup failed: {exc}"})
            pending.remove(child)
            if child["returncode"] != 0:
                failures.append(
                    {
                        "worker": child.get("worker"),
                        "reason": f"child returned {child['returncode']}",
                    }
                )
                for sibling in pending:
                    _terminate_child_process(sibling, f"sibling {child.get('worker')} failed")
        if pending and time.monotonic() >= deadline:
            for child in pending:
                _terminate_child_process(child, "child collection timeout")
                child["parent_observed_finish_monotonic"] = time.monotonic()
                _close_child_log_handles(child, timeout=1.0)
                child["returncode"] = getattr(child.get("process"), "returncode", None)
                failures.append({"worker": child.get("worker"), "reason": "child collection timeout"})
                failures.extend(
                    {"worker": child.get("worker"), "reason": error}
                    for error in child.get("log_cleanup_errors", [])
                )
                failures.extend(
                    {"worker": child.get("worker"), "reason": error}
                    for error in child.get("group_cleanup_error", "").split("; ")
                    if error
                )
                try:
                    child["ipc_cleanup"] = cleanup_child_ipc(Path(str(child["directory"])))
                except BaseException as exc:
                    failures.append({"worker": child.get("worker"), "reason": f"orphan IPC cleanup failed: {exc}"})
            pending.clear()
            break
        if not progressed:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return failures


def _derive_parent_worker_metrics(
    children: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Derive parent worker timing from parent observations and assigned rows.

    Child manifests may retain their own internal timing for diagnostics, but
    parent throughput/utilization is accepted only when the parent recorded a
    monotonic start immediately before ``Popen`` and a finish while collecting
    that child.  Episode busy time is independently summed from the rows
    assigned to each worker.
    """

    if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
        raise BaselineRuntimeError("parent observed worker timing children must be a sequence")
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)):
        raise BaselineRuntimeError("parent observed worker timing episodes must be a sequence")

    by_worker: dict[str, Mapping[str, Any]] = {}
    for index, child in enumerate(children):
        item = _mapping(child, f"children[{index}]", BaselineRuntimeError)
        worker = item.get("worker")
        if worker not in WORKERS:
            raise BaselineRuntimeError(f"parent observed worker timing has invalid worker: {worker!r}")
        if worker in by_worker:
            raise BaselineRuntimeError(f"parent observed worker timing has duplicate worker: {worker}")
        by_worker[worker] = item
    if set(by_worker) != set(WORKERS):
        missing = sorted(set(WORKERS) - set(by_worker))
        raise BaselineRuntimeError(f"parent observed worker timing is missing workers: {missing}")

    episode_by_id: dict[str, Mapping[str, Any]] = {}
    for index, episode in enumerate(episodes):
        item = _mapping(episode, f"episodes[{index}]", BaselineRuntimeError)
        episode_id = _string(item.get("episode_id"), f"episodes[{index}].episode_id", BaselineRuntimeError)
        if episode_id in episode_by_id:
            raise BaselineRuntimeError(f"parent observed worker timing has duplicate episode: {episode_id}")
        episode_by_id[episode_id] = item

    metrics: dict[str, dict[str, Any]] = {}
    for worker in WORKERS:
        child = by_worker[worker]
        start = _finite_nonnegative(
            child.get("parent_observed_start_monotonic"),
            f"parent_observed_start_monotonic.{worker}",
            error=BaselineRuntimeError,
        )
        finish = _finite_nonnegative(
            child.get("parent_observed_finish_monotonic"),
            f"parent_observed_finish_monotonic.{worker}",
            error=BaselineRuntimeError,
        )
        if start is None or finish is None or finish <= start:
            raise BaselineRuntimeError(
                f"parent observed worker timing for {worker} must have finish after start"
            )
        parent_wall = float(finish - start)

        planned = child.get("planned")
        if not isinstance(planned, Sequence) or isinstance(planned, (str, bytes)):
            raise BaselineRuntimeError(f"parent observed worker timing planned rows for {worker} are invalid")
        assigned_ids: list[str] = []
        for index, planned_row in enumerate(planned):
            item = _mapping(planned_row, f"children[{worker}].planned[{index}]", BaselineRuntimeError)
            episode_id = _string(
                item.get("episode_id"),
                f"children[{worker}].planned[{index}].episode_id",
                BaselineRuntimeError,
            )
            if episode_id in assigned_ids:
                raise BaselineRuntimeError(f"parent observed worker timing has duplicate assignment: {episode_id}")
            assigned_ids.append(episode_id)

        assigned_rows = [episode_by_id[episode_id] for episode_id in assigned_ids if episode_id in episode_by_id]
        busy = 0.0
        for index, row in enumerate(assigned_rows):
            row_wall = _finite_nonnegative(
                row.get("wall_time_seconds"),
                f"parent assigned row {worker}[{index}].wall_time_seconds",
                error=BaselineRuntimeError,
            )
            if row_wall is not None:
                busy += row_wall
        if busy > parent_wall:
            raise BaselineRuntimeError(
                f"parent observed worker wall for {worker} is less than assigned row busy time"
            )
        attempted = len(assigned_rows)
        completed = sum(row.get("status") == "completed" for row in assigned_rows)
        successes = sum(row.get("status") == "completed" and row.get("success") is True for row in assigned_rows)
        failed = sum(row.get("status") == "completed" and row.get("success") is False for row in assigned_rows)
        crashed = sum(row.get("status") == "crashed" for row in assigned_rows)
        metrics[worker] = {
            "planned": len(assigned_ids),
            "attempted": attempted,
            "completed": completed,
            "successes": successes,
            "failed": failed,
            "crashed": crashed,
            "busy_time_seconds": busy,
            "wall_time_seconds": parent_wall,
            "throughput_episodes_per_second": completed / parent_wall,
            "utilization": busy / parent_wall,
            "utilization_definition": "assigned row busy_time_seconds / parent-observed worker wall_time_seconds; not hardware utilization",
            "timing_source": "parent_monotonic_observation",
        }
    return metrics


def _new_run_directory(root: str | Path) -> Path:
    output_root = Path(root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(32):
        candidate = output_root / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}_{os.urandom(4).hex()}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise BaselineRuntimeError("could not allocate a unique run directory")


def _write_run_artifacts(
    run_directory: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    project_sha: str,
    matrix: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    provenance: Mapping[str, Any],
    status: str,
    command: Sequence[str],
    preflight_evidence: Mapping[str, Any] | None = None,
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_run_manifest(
        config=config,
        config_path=config_path,
        run_directory=run_directory,
        project_sha=project_sha,
        matrix=matrix,
        aggregate=aggregate,
        provenance=provenance,
        status=status,
        preflight_evidence=preflight_evidence,
        scope=scope,
    )
    write_json_no_overwrite(run_directory / "config_resolved.json", config)
    write_json_no_overwrite(run_directory / "episode_matrix.json", list(matrix))
    _write_bytes_if_absent_or_identical(
        run_directory / "episodes.jsonl",
        b"".join(_canonical_json_line(row) for row in episodes),
    )
    write_json_no_overwrite(run_directory / "aggregate.json", aggregate)
    _write_bytes_no_overwrite(run_directory / "command.txt", (shlex.join([str(item) for item in command]) + "\n").encode())
    _write_bytes_no_overwrite(run_directory / "stdout.log", b"")
    _write_bytes_no_overwrite(run_directory / "stderr.log", b"")
    # Publish the manifest last so a normal PASS cannot become visible before
    # every other required artifact has been durably written.
    write_json_no_overwrite(run_directory / "run_manifest.json", manifest)
    return manifest


def validate_only(
    config_path: str | Path,
    *,
    expected_project_sha: str,
    output_root: str | Path | None = None,
    actual_project_sha: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate and persist a complete planned-run artifact set without execution."""

    input_path = Path(config_path).resolve(strict=True)
    raw_config = load_config(input_path)
    actual = actual_project_sha or _git_sha(ROOT)
    config = validate_config(
        raw_config,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual,
        require_pinned_provenance=True,
    )
    matrix = build_episode_matrix(config)
    aggregate = aggregate_results([], planned=matrix)
    root = output_root if output_root is not None else config["output"]["root"]
    _validate_pinned_provenance(config["provenance"])
    preflight_evidence = validate_preflight_contract(
        config,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual,
        # ``run_gates=False`` is deliberately local/read-only, so the output
        # root is sufficient here and no run directory is left on failure.
        run_directory=root,
        run_gates=False,
    )
    run_directory = _new_run_directory(root)
    provenance = {
        **config["provenance"],
        "git_sha": actual,
        "checkpoint_revision": config["checkpoint"]["revision"],
        "trajectory_ids": [row["trajectory_id"] for row in matrix],
    }
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(input_path),
        "--expected-project-sha",
        expected_project_sha,
        "--seed",
        str(config["seed"]),
        "--validate-only",
    ]
    manifest = _write_run_artifacts(
        run_directory,
        config=config,
        config_path=input_path,
        project_sha=actual,
        matrix=matrix,
        episodes=[],
        aggregate=aggregate,
        provenance=provenance,
        status="VALIDATED",
        command=command,
        preflight_evidence=preflight_evidence,
    )
    return run_directory, manifest


def _run_episode_official_impl(
    config: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    worker_client: Any,
    child_directory: Path,
    worker_evidence: Mapping[str, Any],
    runtime_holder: dict[str, Any],
) -> dict[str, Any]:
    """Execute one episode through the existing official runtime seams.

    This function is intentionally unreachable from validation-only tests.  It
    reuses ``dcu_preflight`` and ``m0_smoke`` wrappers so processor ordering,
    queue semantics, and environment action conversion remain owned by the
    pinned LeRobot code.
    """

    from scripts import dcu_preflight, m0_smoke
    from scripts.dcu_model_worker import FEATURE_SCHEMA
    from scripts.dcu_worker import save_tensor_bundle

    episode_config = deepcopy(dict(config))
    task = dict(episode_config["task"]) if isinstance(episode_config.get("task"), Mapping) else {}
    task.update(
        {
            "suite": planned["suite"],
            "task_id": int(planned["task_id"]),
            "init_state_id": int(planned["init_state_id"]),
            "seed": int(planned["seed"]),
            "horizon": int(planned.get("horizon", EXPECTED_HORIZON)),
        }
    )
    episode_config["task"] = task
    # dcu_preflight expects its historical top-level aliases as well.
    episode_config.update(
        {
            "suite": task["suite"],
            "task_id": task["task_id"],
            "init_state_id": task["init_state_id"],
            "seed": task["seed"],
            "obs_type": config["observation"]["type"],
            "observation_height": config["observation"]["height"],
            "observation_width": config["observation"]["width"],
            "action_dim": config["action"]["dim"],
            "chunk_size": config["action"]["chunk_size"],
            "n_action_steps": config["action"]["n_action_steps"],
            "horizon": task["horizon"],
            "libero_config": config.get("libero_config", {}),
            "offline": config["runtime"]["offline"],
            "renderer": config["runtime"]["renderer"],
        }
    )
    runtime = dcu_preflight.build_cpu_runtime(episode_config, include_policy=False, phase="concurrency")
    runtime_holder["runtime"] = runtime
    env = runtime["env"]
    validate_runtime_task_identity(planned, runtime.get("task", {}))
    # LeRobot's SyncVectorEnv officially supports set_attr/get_attr.  Verify
    # the selected initialization state immediately before reset; the value
    # returned after reset is retained as evidence of the runtime increment.
    getter = getattr(env, "get_attr", None)
    if not callable(getter):
        raise BaselineRuntimeError("official environment lacks set_attr/get_attr for init_state_id")
    init_state_evidence = select_init_state_before_reset(env, int(planned["init_state_id"]))
    # The child runtime owns the IPC directory and shares the existing worker
    # client boundary; only feature requests are persisted.
    ipc = episode_ipc_directory(child_directory, planned)
    ipc.mkdir(parents=True, exist_ok=False)
    counter = 0

    def request_writer(bundle: Mapping[str, Any]) -> Path:
        nonlocal counter
        path = ipc / f"features-{counter:04d}.safetensors"
        counter += 1
        save_tensor_bundle(path, bundle, schema=FEATURE_SCHEMA)
        return path

    trace = m0_smoke.TraceStore(EXPECTED_CHUNK_SIZE, EXPECTED_N_ACTION_STEPS, EXPECTED_ACTION_DIM)
    remote = dcu_preflight.FeatureOnlyRemotePolicy(worker_client, request_writer=request_writer, trace=trace)
    configure_remote_timeouts(remote, config["runtime"], dcu_preflight)
    policy = m0_smoke.PolicyProxy(remote, trace).eval()
    env_preprocessor = m0_smoke.ProcessorProxy(runtime["env_preprocessor"], "env_processor", trace)
    env_postprocessor = m0_smoke.ProcessorProxy(runtime["env_postprocessor"], "env_postprocessor", trace)
    preprocessor = m0_smoke.ProcessorProxy(runtime["preprocessor"], "policy_processor", trace)
    postprocessor = m0_smoke.ProcessorProxy(runtime["postprocessor"], "postprocessor", trace)
    wrapped_env = m0_smoke.VectorEnvProxy(env, trace)
    action_capture: list[np.ndarray] = []

    class _ActionCapturingEnv:
        """Capture exact arrays at the final official env.step boundary."""

        def step(self, action: Any) -> Any:
            action_capture.append(np.array(action, copy=True))
            return wrapped_env.step(action)

        def __getattr__(self, name: str) -> Any:
            return getattr(wrapped_env, name)

    capturing_env = _ActionCapturingEnv()
    from lerobot.scripts.lerobot_eval import rollout

    transport = getattr(worker_client, "transport", None)
    response_start_index = len(getattr(transport, "responses", [])) if transport is not None else 0
    started = time.perf_counter()
    rollout_data = rollout(
        capturing_env,
        policy,
        env_preprocessor,
        env_postprocessor,
        preprocessor,
        postprocessor,
        seeds=[int(planned["seed"])],
        return_observations=False,
        render_callback=m0_smoke.make_render_callback(trace),
    )
    steps = int(rollout_data["action"].shape[1])
    rewards = np.asarray(rollout_data["reward"].detach().cpu().numpy()).reshape(-1)
    success = bool(np.asarray(rollout_data["success"].detach().cpu().numpy()).any())
    done = bool(np.asarray(rollout_data["done"].detach().cpu().numpy()).reshape(-1)[-1])
    if not done:
        raise BaselineRuntimeError("official rollout did not terminate at success or horizon")
    m0_smoke._set_official_done(trace, rollout_data)
    m0_smoke._assert_rollout_acceptance(trace, rollout_data, policy, trace.render_count)
    gl_evidence = validate_renderer_environment(config["runtime"]["renderer"], gl_identity=trace.gl)
    if not callable(getter):
        raise BaselineRuntimeError("official environment lost get_attr after reset")
    after_reset = int(np.asarray(getter("init_state_id")).reshape(-1)[0])
    if after_reset != int(planned["init_state_id"]) + 1:
        raise BaselineRuntimeError(
            "official reset did not advance init_state_id from the selected state: "
            f"selected={planned['init_state_id']}, after_reset={after_reset}"
        )
    terminated = trace.decisions[-1].get("terminated") == [True] if trace.decisions else False
    truncated = trace.decisions[-1].get("truncated") == [True] if trace.decisions else False
    if not terminated and not truncated and steps != EXPECTED_HORIZON:
        raise BaselineRuntimeError("episode ended without official termination or horizon")
    if not terminated and not truncated:
        truncated = steps == EXPECTED_HORIZON
    peak_memory = worker_peak_memory_bytes(worker_client, start_index=response_start_index)
    if len(action_capture) != steps:
        raise BaselineRuntimeError("trace did not retain every exact environment action")
    result = make_episode_record(
        planned,
        success=success,
        rewards=rewards.tolist(),
        terminated=terminated,
        truncated=truncated,
        steps=steps,
        policy_calls=policy.call_count,
        inference_latencies=trace.inference_latencies,
        env_step_latencies=trace.env_step_latencies,
        actions=action_capture,
        model_peak_memory_bytes=peak_memory,
        wall_time_seconds=time.perf_counter() - started,
        init_state_evidence={
            **init_state_evidence,
            "post_reset_init_state_id": after_reset,
        },
        task_source_evidence=runtime["task"],
        worker_evidence=worker_evidence,
        gl_evidence=gl_evidence,
    )
    return result


def _run_episode_official(
    config: Mapping[str, Any],
    planned: Mapping[str, Any],
    *,
    worker_client: Any,
    child_directory: Path,
    worker_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one episode, close official resources, and remove transient IPC."""

    holder: dict[str, Any] = {}
    result: dict[str, Any] | None = None
    try:
        result = _run_episode_official_impl(
            config,
            planned,
            worker_client=worker_client,
            child_directory=child_directory,
            worker_evidence=worker_evidence or {},
            runtime_holder=holder,
        )
    finally:
        close_runtime_environment(holder.get("runtime"))
    cleanup = cleanup_episode_ipc(child_directory, planned)
    if result is None:  # pragma: no cover - defensive control-flow guard
        raise BaselineRuntimeError("episode returned no terminal row")
    result["ipc_cleanup"] = cleanup
    return result


def _append_episode_record(path: str | Path, row: Mapping[str, Any]) -> Path:
    """Compatibility seam for an immutable one-line snapshot."""

    return _write_bytes_if_absent_or_identical(path, _canonical_json_line(row))


def _worker_process(worker_state: Mapping[str, Any] | None) -> Any:
    if not worker_state:
        return None
    transport = worker_state.get("transport")
    base = getattr(transport, "transport", transport)
    return getattr(base, "process", None)


def _close_worker_bounded(worker_state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Use the existing bounded worker shutdown seam and verify termination."""

    if not worker_state:
        return {"attempted": False, "closed": True}
    from scripts import dcu_preflight

    closer = getattr(dcu_preflight, "_close_worker", None)
    if not callable(closer):
        raise BaselineRuntimeError("dcu_preflight._close_worker is unavailable")
    closer(worker_state)
    process = _worker_process(worker_state)
    poll = getattr(process, "poll", None)
    if callable(poll) and poll() is None:
        raise BaselineRuntimeError("nested worker remained alive after bounded cleanup")
    return {"attempted": True, "closed": True, "process_returncode": getattr(process, "returncode", None)}


def _run_child(
    config_path: Path,
    *,
    expected_project_sha: str,
    worker: str,
    physical_device: int,
    child_directory: Path,
) -> int:
    """Run one assigned list serially with one attempt per episode."""

    raw = load_config(config_path)
    actual = _git_sha(ROOT)
    config = validate_config(
        raw,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual,
        require_pinned_provenance=True,
    )
    matrix = build_episode_matrix(config)
    assigned = [row for row in matrix if row["worker"] == worker]
    expected_device = EXPECTED_PHYSICAL_DEVICES[WORKERS.index(worker)]
    if physical_device != expected_device:
        raise BaselineRuntimeError("child physical device does not match assigned worker")
    child_directory.mkdir(parents=True, exist_ok=True)
    if child_directory.is_symlink() or not child_directory.is_dir():
        raise BaselineRuntimeError(f"child directory is not a regular directory: {child_directory}")
    journal_path = child_directory / EPISODE_JOURNAL_NAME
    episodes: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    worker_state: dict[str, Any] | None = None
    preflight_evidence: dict[str, Any] | None = None
    worker_evidence: dict[str, Any] = {}
    try:
        # The existing dcu_preflight worker seam launches exactly one
        # persistent nested model process for this child.
        from scripts import dcu_preflight

        # Re-read and cross-check the exact preflight file in every child
        # immediately before starting its persistent model worker.  The parent
        # already ran the expensive live gates; this child check guards against
        # a changed file or cross-identity drift between the two boundaries.
        preflight_evidence = validate_preflight_contract(
            config,
            expected_project_sha=expected_project_sha,
            actual_project_sha=actual,
            run_directory=child_directory,
            run_gates=False,
        )
        worker_config_path = config["preflight_config"]
        worker_state = dcu_preflight._start_worker(
            {
                **dict(config),
                "runtime": {
                    **dict(config["runtime"]),
                    "dcu_python": config["runtime"]["dcu_python"],
                    "worker_startup_timeout_seconds": 300,
                    "worker_forward_timeout_seconds": 300,
                    "worker_shutdown_timeout_seconds": 30,
                },
                "offline": config["runtime"]["offline"],
                "renderer": config["runtime"]["renderer"],
            },
            run_directory=child_directory,
            config_path=worker_config_path,
            physical_device=physical_device,
        )
        worker_client = worker_state["client"]
        worker_summary = getattr(dcu_preflight, "_worker_summary", None)
        if not callable(worker_summary):
            raise BaselineRuntimeError("dcu_preflight worker summary seam is unavailable")
        worker_evidence = dict(worker_summary(worker_state))
        validate_worker_evidence(worker_evidence, worker=worker, physical_device=physical_device)
        for planned in assigned:
            attempt_started = time.perf_counter()
            append_episode_journal(journal_path, planned, event="attempt_start")
            try:
                episode = _run_episode_official(
                    config,
                    planned,
                    worker_client=worker_client,
                    child_directory=child_directory,
                    worker_evidence=worker_evidence,
                )
            except Exception as exc:
                # One terminal decision; no retry and all later episodes are
                # explicitly represented as unattempted by the child manifest.
                reason = f"{type(exc).__name__}: {exc}"
                episode = make_episode_failure_record(
                    planned,
                    reason=reason,
                    kind="runtime" if not isinstance(exc, BaselineSchemaError) else "schema",
                    wall_time_seconds=time.perf_counter() - attempt_started,
                )
                persist_episode_row(child_directory, episode)
                append_episode_journal(journal_path, planned, event="terminal", status="crashed")
                episodes.append(episode)
                runtime_failures.append({
                    "episode_id": planned["episode_id"],
                    "worker": worker,
                    "reason": reason,
                    "wall_time_seconds": time.perf_counter() - attempt_started,
                })
                break
            persist_episode_row(child_directory, episode)
            append_episode_journal(journal_path, planned, event="terminal", status="completed")
            episodes.append(episode)
        recovered, journal_problems = _journal_row_bijection(
            read_episode_journal(journal_path), planned=assigned, existing=episodes
        )
        if recovered:
            raise BaselineRuntimeError("child journal ended with an unterminalized attempt")
        if journal_problems:
            raise BaselineRuntimeError("child journal/row bijection failed: " + "; ".join(journal_problems))
        cleanup_evidence = _close_worker_bounded(worker_state)
        worker_state = None
        child_wall_time = time.perf_counter() - started
        busy_time = sum(float(row.get("wall_time_seconds", 0.0)) for row in episodes)
        completed = sum(row.get("status") == "completed" for row in episodes)
        worker_metrics = {
            worker: {
                "wall_time_seconds": child_wall_time,
                "busy_time_seconds": busy_time,
                "completed": completed,
                "throughput_episodes_per_second": completed / child_wall_time if child_wall_time else 0.0,
                "utilization": busy_time / child_wall_time if child_wall_time else 0.0,
            }
        }
        aggregate = aggregate_results(
            episodes,
            planned=assigned,
            runtime_failures=runtime_failures,
            total_wall_time_seconds=child_wall_time,
            worker_metrics=worker_metrics,
            require_worker_metrics=True,
            worker_scope=worker,
        )
        worker_metrics = {worker: dict(aggregate["per_worker"][worker])}
        provenance = {
            **config["provenance"],
            "git_sha": actual,
            "checkpoint_revision": config["checkpoint"]["revision"],
            "trajectory_ids": [row["trajectory_id"] for row in assigned],
            "worker_evidence": worker_evidence,
            "worker_metrics": worker_metrics,
            "worker_cleanup": cleanup_evidence,
        }
        if aggregate["final_verdict"] in {"PASS", "NEEDS_INVESTIGATION"}:
            _write_run_artifacts(
                child_directory,
                config=config,
                config_path=config_path,
                project_sha=actual,
                matrix=assigned,
                episodes=episodes,
                aggregate=aggregate,
                provenance=provenance,
                status=aggregate["final_verdict"],
                command=sys.argv,
                preflight_evidence=preflight_evidence,
                scope={
                    "type": "worker",
                    "worker": worker,
                    "physical_device": physical_device,
                    "planned_count": len(assigned),
                    "episode_ids": [row["episode_id"] for row in assigned],
                    "trajectory_ids": [row["trajectory_id"] for row in assigned],
                },
            )
        else:
            # A blocked child is represented only by terminal_manifest.json;
            # never publish a normal BLOCKED worker manifest.
            write_json_no_overwrite(child_directory / "config_resolved.json", config)
            write_json_no_overwrite(child_directory / "episode_matrix.json", list(assigned))
            _write_bytes_if_absent_or_identical(
                child_directory / "episodes.jsonl",
                b"".join(_canonical_json_line(row) for row in episodes),
            )
            write_json_no_overwrite(child_directory / "aggregate.json", aggregate)
            _write_bytes_no_overwrite(
                child_directory / "command.txt", (shlex.join([str(item) for item in sys.argv]) + "\n").encode()
            )
            _write_bytes_no_overwrite(child_directory / "stdout.log", b"")
            _write_bytes_no_overwrite(child_directory / "stderr.log", b"")
        if runtime_failures or aggregate["final_verdict"] == "BLOCKED":
            terminalize_child(
                child_directory / "terminal_manifest.json",
                worker=worker,
                physical_device=physical_device,
                planned=assigned,
                attempted=episodes,
                reason=str(runtime_failures[0]["reason"] if runtime_failures else "child aggregate gates blocked"),
                status="CRASHED",
                worker_metrics=worker_metrics,
            )
        return 0 if not runtime_failures else 1
    except BaseException as exc:
        try:
            terminalize_child(
                child_directory / "terminal_manifest.json",
                worker=worker,
                physical_device=physical_device,
                planned=assigned,
                attempted=episodes,
                reason=f"{type(exc).__name__}: {exc}",
                status="CRASHED",
                worker_metrics={
                    worker: {
                        "wall_time_seconds": time.perf_counter() - started,
                        "busy_time_seconds": sum(float(row.get("wall_time_seconds", 0.0)) for row in episodes),
                    }
                },
            )
        except Exception:
            pass
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    finally:
        if worker_state is not None:
            try:
                _close_worker_bounded(worker_state)
            except BaseException as cleanup_exc:
                runtime_failures.append({"worker": worker, "reason": f"worker cleanup failed: {cleanup_exc}"})
                try:
                    terminalize_child(
                        child_directory / "terminal_manifest.json",
                        worker=worker,
                        physical_device=physical_device,
                        planned=assigned,
                        attempted=episodes,
                        reason=f"worker cleanup failed: {cleanup_exc}",
                        status="CRASHED",
                    )
                except FileExistsError:
                    pass


def _validate_child_terminal_manifest(
    manifest: Mapping[str, Any], child: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the compact terminal evidence emitted after child failure."""

    item = _mapping(manifest, "child terminal manifest", BaselineSchemaError)
    if item.get("schema_version") != SCHEMA_VERSION or item.get("name") != EXPECTED_NAME:
        raise BaselineSchemaError("child terminal manifest schema identity is invalid")
    worker = child.get("worker")
    physical_device = child.get("physical_device")
    if item.get("worker") != worker or item.get("physical_device") != physical_device:
        raise BaselineSchemaError("child terminal manifest worker identity is invalid")
    scope = _mapping(item.get("scope"), "child terminal manifest.scope", BaselineSchemaError)
    planned = list(child.get("planned", []))
    planned_ids = [row.get("episode_id") for row in planned]
    if scope.get("type") != "worker" or scope.get("worker") != worker:
        raise BaselineSchemaError("child terminal manifest scope is invalid")
    if scope.get("planned_count") != len(planned) or scope.get("episode_ids") != planned_ids:
        raise BaselineSchemaError("child terminal manifest scope does not match assignment")
    if item.get("planned_episode_ids") != planned_ids:
        raise BaselineSchemaError("child terminal manifest planned IDs do not match assignment")
    attempted = item.get("attempted_episode_ids")
    if not isinstance(attempted, Sequence) or isinstance(attempted, (str, bytes)):
        raise BaselineSchemaError("child terminal manifest attempted IDs are invalid")
    if len(set(attempted)) != len(attempted) or not set(attempted).issubset(set(planned_ids)):
        raise BaselineSchemaError("child terminal manifest attempted IDs are invalid")
    return dict(item)


def _load_child_manifest(child: Mapping[str, Any]) -> dict[str, Any] | None:
    directory = Path(str(child["directory"]))
    run_path = directory / "run_manifest.json"
    terminal_path = directory / "terminal_manifest.json"
    episodes_path = directory / "episodes.jsonl"
    journal_path = directory / EPISODE_JOURNAL_NAME
    # Never follow a child-controlled manifest alias.  Check both names before
    # selecting the normal-vs-terminal branch so a symlink cannot be hidden by
    # the other manifest's presence.
    if run_path.is_symlink():
        raise BaselineSchemaError(f"child run manifest must not be a symlink: {run_path}")
    if terminal_path.is_symlink():
        raise BaselineSchemaError(f"child terminal manifest must not be a symlink: {terminal_path}")
    if run_path.is_file():
        try:
            manifest = dict(_mapping(json.loads(run_path.read_text(encoding="utf-8")), "child manifest", BaselineSchemaError))
        except json.JSONDecodeError as exc:
            raise BaselineSchemaError(f"child manifest JSON is malformed: {run_path}") from exc
        validate_manifest(manifest)
        scope = _mapping(manifest.get("scope"), "child manifest.scope", BaselineSchemaError)
        if scope.get("type") != "worker" or scope.get("worker") != child.get("worker"):
            raise BaselineSchemaError("child manifest scope worker does not match parent assignment")
        if scope.get("physical_device") != child.get("physical_device"):
            raise BaselineSchemaError("child manifest scope physical_device does not match parent assignment")
        planned_ids = [row.get("episode_id") for row in child.get("planned", [])]
        if scope.get("episode_ids") != planned_ids:
            raise BaselineSchemaError("child manifest scope episode_ids do not match parent assignment")
        matrix = _mapping(manifest.get("episode_matrix"), "child manifest.episode_matrix", BaselineSchemaError)
        manifest_rows_raw = matrix.get("rows")
        if not isinstance(manifest_rows_raw, Sequence) or isinstance(manifest_rows_raw, (str, bytes)):
            raise BaselineSchemaError("child manifest episode_matrix rows are invalid")
        manifest_rows = [
            dict(_mapping(row, f"child manifest episode_matrix.rows[{index}]", BaselineSchemaError))
            for index, row in enumerate(manifest_rows_raw)
        ]
        assigned_rows = [
            dict(_mapping(row, f"child assignment[{index}]", BaselineSchemaError))
            for index, row in enumerate(child.get("planned", []))
        ]
        if manifest_rows != assigned_rows:
            raise BaselineSchemaError("child manifest episode_matrix rows do not exactly match parent assignment")
        return manifest
    if terminal_path.is_file():
        try:
            manifest = json.loads(terminal_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BaselineSchemaError(f"child terminal manifest JSON is malformed: {terminal_path}") from exc
        return _validate_child_terminal_manifest(manifest, child)
    return None


def _single_worker_metrics_contract(metrics: Mapping[str, Any] | None, worker: str) -> bool:
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get(worker), Mapping):
        return False
    item = metrics[worker]
    try:
        wall = _finite_nonnegative(item.get("wall_time_seconds"), f"worker_metrics.{worker}.wall_time_seconds")
        busy = _finite_nonnegative(item.get("busy_time_seconds"), f"worker_metrics.{worker}.busy_time_seconds")
        completed = _int(item.get("completed"), f"worker_metrics.{worker}.completed", nonnegative=True)
        throughput = _finite_nonnegative(item.get("throughput_episodes_per_second"), f"worker_metrics.{worker}.throughput")
        utilization = _finite_nonnegative(item.get("utilization"), f"worker_metrics.{worker}.utilization")
    except (BaselineError, TypeError, ValueError):
        return False
    return bool(
        wall is not None
        and wall > 0
        and busy is not None
        and busy <= wall
        and throughput is not None
        and utilization is not None
        and math.isclose(throughput, completed / wall, rel_tol=1e-9, abs_tol=1e-12)
        and math.isclose(utilization, busy / wall, rel_tol=1e-9, abs_tol=1e-12)
    )


def _child_manifest_trust_failures(
    manifest: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None,
    expected_project_sha: str | None,
    config_path: str | Path | None,
    rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any] | None,
) -> list[str]:
    """Check every parent-visible identity field of a normal worker manifest."""

    failures: list[str] = []
    status = manifest.get("status")
    if status not in {"PASS", "NEEDS_INVESTIGATION"}:
        failures.append(f"normal worker manifest status is not accepted: {status!r}")
    if manifest.get("no_retry") is not True:
        failures.append("normal worker manifest no_retry is not true")
    project = manifest.get("project")
    if expected_project_sha is not None:
        if not isinstance(project, Mapping) or project.get("git_sha") != expected_project_sha:
            failures.append("child manifest project SHA does not match parent")
        elif project.get("expected_project_sha") != expected_project_sha:
            failures.append("child manifest expected project SHA does not match parent")
    manifest_config = manifest.get("config")
    if config_path is not None:
        expected_path = str(Path(config_path).resolve())
        expected_hash = _sha256_file(config_path)
        if not isinstance(manifest_config, Mapping) or manifest_config.get("path") != expected_path:
            failures.append("child manifest config path does not match parent")
        if not isinstance(manifest_config, Mapping) or manifest_config.get("sha256") != expected_hash:
            failures.append("child manifest config hash does not match parent")
    normalized = validate_config(config) if config is not None else None
    for name in ("checkpoint", "base_model", "assets"):
        if normalized is not None and manifest.get(name) != normalized.get(name):
            failures.append(f"child manifest {name} identity does not match parent")
    if normalized is not None and manifest.get("runtime") != normalized.get("runtime"):
        failures.append("child manifest runtime identity does not match parent")
    scope = manifest.get("scope")
    if not isinstance(scope, Mapping) or scope.get("type") != "worker":
        failures.append("child normal manifest scope is not worker")
    else:
        if scope.get("worker") != child.get("worker") or scope.get("physical_device") != child.get("physical_device"):
            failures.append("child normal manifest worker scope does not match parent")
        if normalized is not None:
            try:
                _expected_scoped_matrix(list(manifest.get("episode_matrix", {}).get("rows", [])), normalized, scope)
            except BaselineError as exc:
                failures.append(f"child normal manifest assigned rows are invalid: {exc}")
    assigned = list(child.get("planned", []))
    assigned_ids = [row.get("episode_id") for row in assigned]
    matrix = manifest.get("episode_matrix")
    manifest_rows = matrix.get("rows", []) if isinstance(matrix, Mapping) else []
    manifest_ids = [row.get("episode_id") for row in manifest_rows] if isinstance(manifest_rows, Sequence) else []
    if manifest_ids != assigned_ids:
        failures.append("child normal manifest rows do not match parent assignment")
    try:
        assigned_dicts = [
            dict(_mapping(row, f"child assignment[{index}]", BaselineSchemaError))
            for index, row in enumerate(assigned)
        ]
        manifest_dicts = [
            dict(_mapping(row, f"child normal manifest episode_matrix.rows[{index}]", BaselineSchemaError))
            for index, row in enumerate(manifest_rows)
        ]
    except BaselineError as exc:
        failures.append(f"child normal manifest assigned rows are malformed: {exc}")
    else:
        if manifest_dicts != assigned_dicts:
            failures.append("child normal manifest rows do not exactly match parent assignment")
    row_ids = [row.get("episode_id") for row in rows]
    if row_ids != assigned_ids:
        failures.append("child normal manifest does not account for every assigned row")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        failures.append("child normal manifest provenance is missing")
    else:
        expected_ids = assigned_ids
        expected_trajectories = [row.get("trajectory_id") for row in assigned]
        for field, expected in (
            ("git_sha", expected_project_sha),
            ("checkpoint_revision", normalized.get("checkpoint", {}).get("revision") if normalized else None),
            ("trajectory_ids", expected_trajectories),
            ("episode_ids", expected_ids),
        ):
            if expected is not None and provenance.get(field) != expected:
                failures.append(f"child normal manifest provenance.{field} does not match parent")
        if provenance.get("offline") is not True or provenance.get("hub_fallback") is not False:
            failures.append("child normal manifest provenance is not offline/no-Hub")
        if provenance.get("perturbation") is not False:
            failures.append("child normal manifest provenance perturbation is not false")
        child_provenance_metrics = provenance.get("worker_metrics")
        if metrics is not None and child_provenance_metrics != metrics:
            failures.append("child worker metrics do not match manifest provenance")
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, Mapping):
        failures.append("child normal manifest aggregate is missing")
    else:
        row_counts = {
            "planned": len(assigned),
            "attempted": len(rows),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "successes": sum(row.get("status") == "completed" and row.get("success") is True for row in rows),
            "failed": sum(row.get("status") == "completed" and row.get("success") is False for row in rows),
            "crashed": sum(row.get("status") == "crashed" for row in rows),
        }
        aggregate_counts = {
            "planned": row_counts["planned"],
            "total": row_counts["planned"],
            "attempted": row_counts["attempted"],
            "completed": row_counts["completed"],
            "successes": row_counts["successes"],
            "failed": row_counts["failed"],
            "crashed": row_counts["crashed"],
        }
        for field, expected in aggregate_counts.items():
            if aggregate.get(field) != expected:
                failures.append(
                    f"child normal manifest aggregate {field} count does not match assigned rows: "
                    f"expected {expected}, got {aggregate.get(field)!r}"
                )
        expected_unattempted = [row["episode_id"] for row in assigned if row["episode_id"] not in set(row_ids)]
        if aggregate.get("unattempted_episode_ids") != expected_unattempted:
            failures.append("child normal manifest aggregate unattempted IDs do not match rows")
        if aggregate.get("final_verdict") != status:
            failures.append("child normal manifest status does not match aggregate verdict")
        worker = str(child.get("worker"))
        manifest_metrics = aggregate.get("per_worker")
        if metrics is not None and (not isinstance(manifest_metrics, Mapping) or manifest_metrics.get(worker) != metrics.get(worker)):
            failures.append("child worker metrics do not match aggregate manifest metrics")
        worker_metric = metrics.get(worker) if isinstance(metrics, Mapping) else None
        if isinstance(worker_metric, Mapping):
            for field, expected in row_counts.items():
                if worker_metric.get(field) != expected:
                    failures.append(
                        f"child worker metrics {field} count does not match assigned rows: "
                        f"expected {expected}, got {worker_metric.get(field)!r}"
                    )
            busy_time = 0.0
            for index, row in enumerate(rows):
                try:
                    wall_time = _finite_nonnegative(
                        row.get("wall_time_seconds"),
                        f"child row {index}.wall_time_seconds",
                    )
                except BaselineError as exc:
                    failures.append(f"child worker busy time cannot be derived from rows: {exc}")
                    continue
                if wall_time is not None:
                    busy_time += wall_time
            reported_busy = worker_metric.get("busy_time_seconds")
            if isinstance(reported_busy, (int, float, np.integer, np.floating)) and not isinstance(reported_busy, bool):
                if not math.isclose(float(reported_busy), busy_time, rel_tol=1e-9, abs_tol=1e-12):
                    failures.append(
                        "child worker busy_time_seconds does not equal the sum of assigned row wall_time_seconds"
                    )
            reported_wall = worker_metric.get("wall_time_seconds")
            if (
                isinstance(reported_wall, (int, float, np.integer, np.floating))
                and not isinstance(reported_wall, bool)
                and isinstance(reported_busy, (int, float, np.integer, np.floating))
                and not isinstance(reported_busy, bool)
                and float(reported_busy) > float(reported_wall)
            ):
                failures.append("child worker wall_time_seconds is less than busy_time_seconds")
    return failures


def _read_child_rows(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read immutable per-episode rows, preserving rows before a torn entry."""

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    rows_dir = directory / EPISODE_ROWS_NAME
    if rows_dir.exists():
        if rows_dir.is_symlink() or not rows_dir.is_dir():
            return rows, [f"episode row directory is not a regular directory: {rows_dir}"]
        for entry in sorted(rows_dir.iterdir(), key=lambda item: item.name):
            if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
                failures.append(f"episode row artifact is malformed: {entry}")
                continue
            try:
                rows.append(dict(_mapping(json.loads(entry.read_text(encoding="utf-8")), "child episode", BaselineSchemaError)))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, BaselineSchemaError) as exc:
                failures.append(f"child episode artifact is malformed: {entry}: {exc}")
    snapshot = directory / "episodes.jsonl"
    if snapshot.is_file():
        try:
            for line_number, line in enumerate(snapshot.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    item = dict(_mapping(json.loads(line), "child episode", BaselineSchemaError))
                except (json.JSONDecodeError, BaselineSchemaError) as exc:
                    failures.append(f"child episode JSON is malformed at line {line_number}: {exc}")
                    continue
                existing = next((current for current in rows if current.get("episode_id") == item.get("episode_id")), None)
                if existing is None:
                    rows.append(item)
                elif existing != item:
                    failures.append(f"child episode snapshot disagrees with immutable row {item.get('episode_id')}")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"child episode snapshot is unreadable: {exc}")
    return rows, failures


def _read_child_artifacts(
    children: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
    expected_project_sha: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Read child rows, genuine metrics, and explicit trust/runtime failures."""

    episodes: list[dict[str, Any]] = []
    worker_metrics: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for child in children:
        directory = Path(str(child["directory"])).resolve()
        try:
            manifest = _load_child_manifest(child)
        except BaselineError as exc:
            manifest = None
            failures.append({"worker": child.get("worker"), "reason": f"child manifest unreadable: {exc}"})
        child_rows, row_failures = _read_child_rows(directory)
        failures.extend({"worker": child.get("worker"), "reason": reason} for reason in row_failures)
        journal_path = directory / EPISODE_JOURNAL_NAME
        events: list[dict[str, Any]] = []
        journal_error: str | None = None
        if journal_path.exists():
            try:
                events = read_episode_journal(journal_path)
            except JournalIntegrityError as exc:
                events = list(exc.partial_events)
                journal_error = str(exc)
            except BaselineError as exc:
                journal_error = str(exc)
        elif child_rows:
            journal_error = "child episode rows exist but journal directory is missing"
        if journal_error:
            failures.append({"worker": child.get("worker"), "reason": f"episode journal invalid: {journal_error}"})
        if journal_path.exists() or child_rows:
            try:
                recovered, problems = _journal_row_bijection(
                    events,
                    planned=list(child.get("planned", [])),
                    existing=child_rows,
                )
            except BaselineError as exc:
                recovered, problems = [], [str(exc)]
            failures.extend(
                {"worker": child.get("worker"), "reason": f"journal/row bijection: {reason}"}
                for reason in problems
            )
            for recovered_row in recovered:
                try:
                    persist_episode_row(directory, recovered_row)
                    append_episode_journal(journal_path, recovered_row, event="terminal", status="crashed")
                    child_rows.append(recovered_row)
                except BaseException as exc:
                    failures.append({"worker": child.get("worker"), "reason": f"journal recovery persistence failed: {exc}"})
        if manifest is None:
            failures.append({"worker": child.get("worker"), "reason": "normal worker run manifest is missing"})
        else:
            status = manifest.get("status")
            if status not in {"PASS", "NEEDS_INVESTIGATION"}:
                failures.append({"worker": child.get("worker"), "reason": f"child terminal/non-normal manifest status: {status!r}"})
            candidate: dict[str, Any] | None = None
            aggregate = manifest.get("aggregate")
            if isinstance(aggregate, Mapping) and isinstance(aggregate.get("per_worker"), Mapping):
                value = aggregate["per_worker"].get(str(child.get("worker")))
                if isinstance(value, Mapping):
                    candidate = dict(value)
            worker = str(child.get("worker"))
            candidate_contract_ok = False
            if candidate is None:
                failures.append({"worker": worker, "reason": "child worker metrics are missing from normal manifest"})
            else:
                candidate_contract_ok = _single_worker_metrics_contract({worker: candidate}, worker)
                if not candidate_contract_ok:
                    failures.append({"worker": worker, "reason": "child worker metrics are missing or not genuinely derived"})
            candidate_failures = _child_manifest_trust_failures(
                manifest,
                child,
                config=config,
                expected_project_sha=expected_project_sha,
                config_path=config_path,
                rows=child_rows,
                metrics=({worker: candidate} if candidate is not None else None),
            )
            if candidate is not None and candidate_contract_ok and not candidate_failures:
                worker_metrics[worker] = candidate
            failures.extend(
                {"worker": child.get("worker"), "reason": reason}
                for reason in candidate_failures
            )
        for row in child_rows:
            try:
                episode_id = _string(row.get("episode_id"), "child episode_id", BaselineSchemaError)
            except BaselineError as exc:
                failures.append({"worker": child.get("worker"), "reason": str(exc)})
                continue
            if episode_id in seen:
                failures.append({"worker": child.get("worker"), "reason": f"duplicate child episode {episode_id}"})
                continue
            seen.add(episode_id)
            episodes.append(row)
    return episodes, worker_metrics, failures


def _read_child_episodes(children: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows, _metrics, failures = _read_child_artifacts(children)
    if failures:
        raise BaselineSchemaError("; ".join(str(item.get("reason")) for item in failures))
    return rows


def _persist_child_process_outputs(children: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Close/fsync direct-to-file child logs; never read them into RAM."""

    failures: list[dict[str, Any]] = []
    for child in children:
        errors = _close_child_log_handles(child if isinstance(child, dict) else dict(child), timeout=1.0)
        failures.extend(
            {"worker": child.get("worker"), "reason": f"child log cleanup failed: {error}"}
            for error in errors
        )
    return failures


def _cleanup_child_orphan_ipc(children: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for child in children:
        try:
            evidence = cleanup_child_ipc(Path(str(child["directory"])))
            if isinstance(child, dict):
                child["ipc_cleanup"] = evidence
        except BaseException as exc:
            failures.append({"worker": child.get("worker"), "reason": f"orphan IPC cleanup failed: {exc}"})
    return failures


def _terminalize_children_for_failure(
    children: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    reason_by_worker: Mapping[str, str],
) -> None:
    for child in children:
        worker = str(child["worker"])
        returncode = child.get("returncode")
        reason = reason_by_worker.get(worker)
        if reason is None and child.get("forced_termination"):
            reason = str(child.get("termination_reason", "child forcibly terminated"))
        if reason is None and returncode in (None, 0):
            continue
        reason = reason or f"child returned {returncode}"
        attempted = [row for row in episodes if row.get("worker") == worker]
        try:
            terminalize_child(
                Path(str(child["directory"])) / "terminal_manifest.json",
                worker=worker,
                physical_device=int(child["physical_device"]),
                planned=child["planned"],
                attempted=attempted,
                reason=reason,
                status="CRASHED",
                returncode=int(returncode) if isinstance(returncode, int) else None,
            )
        except FileExistsError:
            pass


def _parent_provenance(
    config: Mapping[str, Any], actual_project_sha: str, matrix: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        **config["provenance"],
        "git_sha": actual_project_sha,
        "checkpoint_revision": config["checkpoint"]["revision"],
        "trajectory_ids": [row["trajectory_id"] for row in matrix],
    }


def _fallback_blocked_aggregate(
    matrix: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Build minimal finite failure evidence without invoking validation code."""

    planned_ids = [str(row.get("episode_id")) for row in matrix]
    attempted_ids = [str(row.get("episode_id")) for row in episodes if isinstance(row, Mapping)]
    attempted_ids = list(dict.fromkeys(item for item in attempted_ids if item in planned_ids))
    completed = sum(isinstance(row, Mapping) and row.get("status") == "completed" for row in episodes)
    crashed = sum(isinstance(row, Mapping) and row.get("status") == "crashed" for row in episodes)
    return {
        "schema_version": SCHEMA_VERSION,
        "total": len(matrix),
        "planned": len(matrix),
        "attempted": len(attempted_ids),
        "completed": completed,
        "failed": 0,
        "crashed": crashed + 1,
        "successes": 0,
        "overall_sr": 0.0,
        "overall_success_rate": 0.0,
        "per_task_sr": {},
        "per_task": {},
        "mean_steps_to_success": None,
        "total_wall_time_seconds": 0.0,
        "per_worker": {},
        "runtime_failures": [{"reason": str(reason)}],
        "numerical_anomalies": [],
        "unattempted_episode_ids": [item for item in planned_ids if item not in attempted_ids],
        "gates": {key: False for key in "ABCDEFGH"},
        "final_verdict": "BLOCKED",
    }


def _write_terminal_parent_fallback(
    *,
    run_directory: Path,
    config: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    reason: str,
    runtime_failures: Sequence[Mapping[str, Any]],
    preflight_evidence: Mapping[str, Any] | None,
) -> None:
    """Write terminal parent evidence even when normal finalization failed."""

    aggregate = _fallback_blocked_aggregate(matrix, episodes, reason)
    attempted = [
        str(row.get("episode_id")) for row in episodes if isinstance(row, Mapping) and row.get("episode_id") is not None
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": EXPECTED_NAME,
        "status": "BLOCKED",
        "reason": str(reason),
        "matrix_count": len(matrix),
        "planned_episode_ids": [str(row.get("episode_id")) for row in matrix],
        "attempted_episode_ids": list(dict.fromkeys(attempted)),
        "unattempted_episode_ids": aggregate["unattempted_episode_ids"],
        "aggregate": aggregate,
        "gate_verdict": "BLOCKED",
        "runtime_failures": [dict(item) for item in runtime_failures if isinstance(item, Mapping)],
        "runtime": dict(config.get("runtime", {})) if isinstance(config, Mapping) else {},
        "preflight": dict(preflight_evidence or {}),
        "provenance": dict(config.get("provenance", {})) if isinstance(config, Mapping) else {},
        "terminalized_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        write_json_no_overwrite(run_directory / "terminal_manifest.json", manifest)
    except FileExistsError:
        pass
    except BaseException as exc:
        raise BaselineRuntimeError(f"terminal provenance publication failed: {exc}") from exc


def _write_parent_failure_evidence(
    *,
    run_directory: Path,
    config: Mapping[str, Any],
    config_path: Path,
    project_sha: str,
    matrix: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    runtime_failures: Sequence[Mapping[str, Any]],
    preflight_evidence: Mapping[str, Any] | None,
    reason: str,
) -> None:
    try:
        aggregate = aggregate_results(
            episodes,
            planned=matrix,
            runtime_failures=runtime_failures,
            total_wall_time_seconds=0.0,
        )
        provenance = _parent_provenance(config, project_sha, matrix)
        _write_run_artifacts(
            run_directory,
            config=config,
            config_path=config_path,
            project_sha=project_sha,
            matrix=matrix,
            episodes=episodes,
            aggregate=aggregate,
            provenance=provenance,
            status="BLOCKED",
            command=sys.argv,
            preflight_evidence=preflight_evidence,
        )
    except BaseException:
        # The separate terminal manifest remains the authoritative failure
        # evidence even if a malformed partial child artifact prevents the
        # normal aggregate bundle from being finalized.
        pass
    _write_terminal_parent_fallback(
        run_directory=run_directory,
        config=config,
        matrix=matrix,
        episodes=episodes,
        reason=reason,
        runtime_failures=runtime_failures,
        preflight_evidence=preflight_evidence,
    )


def run(
    config_path: str | Path,
    *,
    expected_project_sha: str,
    seed: int | None = None,
    validate_only_mode: bool = False,
    output_root: str | Path | None = None,
) -> int:
    if seed is not None and seed != EXPECTED_SEED:
        raise BaselineConfigError(f"explicit seed must equal frozen config seed {EXPECTED_SEED}")
    if validate_only_mode:
        validate_only(config_path, expected_project_sha=expected_project_sha, output_root=output_root)
        return 0
    input_path = Path(config_path).resolve(strict=True)
    raw = load_config(input_path)
    actual = _git_sha(ROOT)
    config = validate_config(
        raw,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual,
        require_pinned_provenance=True,
    )
    matrix = build_episode_matrix(config)
    run_directory = _new_run_directory(output_root or config["output"]["root"])
    children: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    preflight_evidence: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        preflight_evidence = validate_preflight_contract(
            config,
            expected_project_sha=expected_project_sha,
            actual_project_sha=actual,
            run_directory=run_directory,
        )
        launch_result = launch_children(
            config,
            config_path=input_path,
            expected_project_sha=expected_project_sha,
            run_directory=run_directory,
            project_python=config["runtime"].get("cpu_python"),
            wait=False,
            children_out=children,
        )
        children = launch_result
        runtime_failures.extend(
            collect_children_fail_closed(
                children,
                timeout_seconds=parent_collection_timeout_seconds(config),
            )
        )
    except BaseException as exc:
        reason = f"parent execution failure: {type(exc).__name__}: {exc}"
        runtime_failures.append({"reason": reason})
        for child in children:
            try:
                _terminate_child_process(child, reason)
            except BaseException as cleanup_exc:
                runtime_failures.append({"worker": child.get("worker"), "reason": f"cleanup failed: {cleanup_exc}"})
        try:
            runtime_failures.extend(_persist_child_process_outputs(children))
        except BaseException as persist_exc:
            runtime_failures.append({"reason": f"post-collection persistence failed: {persist_exc}"})
        try:
            runtime_failures.extend(_cleanup_child_orphan_ipc(children))
        except BaseException as cleanup_exc:
            runtime_failures.append({"reason": f"orphan IPC cleanup failed: {cleanup_exc}"})
        episodes: list[dict[str, Any]] = []
        try:
            episodes, _worker_metrics, child_failures = _read_child_artifacts(
                children,
                config=config,
                expected_project_sha=actual,
                config_path=input_path,
            )
            runtime_failures.extend(child_failures)
        except BaseException as parse_exc:
            runtime_failures.append({"reason": f"post-collection child parsing failed: {parse_exc}"})
        try:
            _terminalize_children_for_failure(
                children,
                episodes,
                {str(item.get("worker")): reason for item in runtime_failures if item.get("worker")},
            )
        except BaseException as terminal_exc:
            runtime_failures.append({"reason": f"child terminalization failed: {terminal_exc}"})
        try:
            _write_parent_failure_evidence(
                run_directory=run_directory,
                config=config,
                config_path=input_path,
                project_sha=actual,
                matrix=matrix,
                episodes=episodes,
                runtime_failures=runtime_failures,
                preflight_evidence=preflight_evidence,
                reason=reason,
            )
        except BaseException as evidence_exc:
            _write_terminal_parent_fallback(
                run_directory=run_directory,
                config=config,
                matrix=matrix,
                episodes=episodes,
                reason=f"{reason}; terminal evidence write failed: {evidence_exc}",
                runtime_failures=runtime_failures,
                preflight_evidence=preflight_evidence,
            )
        return 1

    # Everything after collection is one fail-closed boundary.  Parsing a
    # malformed child row, aggregating inconsistent evidence, or writing a
    # final artifact must all leave terminal parent evidence behind.
    episodes: list[dict[str, Any]] = []
    try:
        runtime_failures.extend(_persist_child_process_outputs(children))
        episodes, _child_worker_metrics, child_failures = _read_child_artifacts(
            children,
            config=config,
            expected_project_sha=actual,
            config_path=input_path,
        )
        runtime_failures.extend(child_failures)
        try:
            # Parent acceptance never consumes child-reported wall/busy spans.
            # They remain diagnostic evidence in the child manifest only.
            worker_metrics = _derive_parent_worker_metrics(children, episodes)
        except BaselineError as exc:
            worker_metrics = {}
            runtime_failures.append({"reason": f"parent observed worker timing invalid: {exc}"})
        reason_by_worker: dict[str, str] = {}
        for failure in runtime_failures:
            worker = failure.get("worker")
            if worker is not None:
                reason_by_worker[str(worker)] = str(failure.get("reason", "child failure"))
        for child in children:
            returncode = child.get("returncode")
            if returncode not in (None, 0):
                worker = str(child["worker"])
                reason_by_worker.setdefault(worker, f"child returned {returncode}")
                if not any(str(item.get("worker")) == worker for item in runtime_failures):
                    runtime_failures.append({"worker": worker, "reason": reason_by_worker[worker]})
        _terminalize_children_for_failure(children, episodes, reason_by_worker)
        aggregate = aggregate_results(
            episodes,
            planned=matrix,
            runtime_failures=runtime_failures,
            total_wall_time_seconds=time.perf_counter() - started,
            worker_metrics=worker_metrics,
            require_worker_metrics=True,
        )
        provenance = _parent_provenance(config, actual, matrix)
        _write_run_artifacts(
            run_directory,
            config=config,
            config_path=input_path,
            project_sha=actual,
            matrix=matrix,
            episodes=episodes,
            aggregate=aggregate,
            provenance=provenance,
            status=aggregate["final_verdict"],
            command=sys.argv,
            preflight_evidence=preflight_evidence,
        )
        if aggregate["final_verdict"] == "BLOCKED":
            reason = next(
                (str(item.get("reason")) for item in runtime_failures if item.get("reason")),
                "aggregate gates incomplete",
            )
            _write_parent_failure_evidence(
                run_directory=run_directory,
                config=config,
                config_path=input_path,
                project_sha=actual,
                matrix=matrix,
                episodes=episodes,
                runtime_failures=runtime_failures or [{"reason": reason}],
                preflight_evidence=preflight_evidence,
                reason=reason,
            )
        return 0 if aggregate["final_verdict"] == "PASS" else 1
    except BaseException as exc:
        reason = f"parent post-collection failure: {type(exc).__name__}: {exc}"
        runtime_failures.append({"reason": reason})
        for child in children:
            try:
                _terminate_child_process(child, reason)
            except BaseException:
                pass
        try:
            _terminalize_children_for_failure(children, episodes, {str(item.get("worker")): reason for item in runtime_failures if item.get("worker")})
        except BaseException:
            pass
        try:
            _write_parent_failure_evidence(
                run_directory=run_directory,
                config=config,
                config_path=input_path,
                project_sha=actual,
                matrix=matrix,
                episodes=episodes,
                runtime_failures=runtime_failures,
                preflight_evidence=preflight_evidence,
                reason=reason,
            )
        except BaseException as evidence_exc:
            _write_terminal_parent_fallback(
                run_directory=run_directory,
                config=config,
                matrix=matrix,
                episodes=episodes,
                reason=f"{reason}; terminal evidence write failed: {evidence_exc}",
                runtime_failures=runtime_failures,
                preflight_evidence=preflight_evidence,
            )
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-project-sha", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=WORKERS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--physical-device", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-directory", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.child:
            if args.worker is None or args.physical_device is None or args.child_directory is None:
                raise BaselineConfigError("child mode requires worker, physical device, and child directory")
            return _run_child(
                args.config.resolve(),
                expected_project_sha=args.expected_project_sha,
                worker=args.worker,
                physical_device=args.physical_device,
                child_directory=args.child_directory.resolve(),
            )
        return run(
            args.config.resolve(),
            expected_project_sha=args.expected_project_sha,
            seed=args.seed,
            validate_only_mode=args.validate_only,
            output_root=args.output_root,
        )
    except BaseException as exc:
        print(f"baseline-a failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


# Names used by downstream code/reviewers that mirror the existing preflight
# vocabulary.  Keep one implementation for the contract.
assign_worker = assignment_for_index
worker_assignment = assignment_for_index
validate_episode_record = _validate_episode_row
aggregate_episodes = aggregate_results
terminalize_child_manifest = terminalize_child
_load_yaml = load_config
_validate_config = validate_config
build_episode_assignment = build_episode_matrix
validate_run_manifest = validate_manifest


if __name__ == "__main__":  # pragma: no cover - exercised by explicit CLI
    raise SystemExit(main())
