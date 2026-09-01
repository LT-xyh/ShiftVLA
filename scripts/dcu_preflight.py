"""Fail-closed DCU preflight contracts and explicit phase runner.

Importing this module is fake-only: benchmark/model imports, accelerator
selection, and worker subprocesses are lazy and occur only after the immutable
gates pass.  The CLI executes exactly one requested phase (``compare``,
``closed-loop``, or ``concurrency``); ``one-step-child`` is reachable only by
the private token emitted by the concurrency parent.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import random
import shlex
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import traceback
from typing import Any

import numpy as np
import torch

# ``python scripts/dcu_preflight.py`` places ``scripts/`` (rather than the
# repository root) on ``sys.path``.  Keep direct CLI execution equivalent to
# module execution without importing any runtime package here.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Re-export the worker's one project error type.  In particular, do not define
# a second exception here: the transport and preflight layers share this exact
# identity at their public boundary.
from scripts.dcu_worker import DCUPreflightError


_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PATCH_SHA256 = "3ebaba1e8d305c93d0c758c9c9d3a575215ebe3513e486ae5503da444f8d573d"
EXPECTED_OVERLAY_WHEEL_SHA256 = "d03459fe530b556398bab41ad7de60b47c3354ffa7a5ce44db6249571fcdd2ff"
PATCH_PATH = _ROOT / "runtime" / "patches" / "lerobot-v0.6.1-python311.patch"
FAILURE_MANIFEST_NAME = "failure_manifest.json"

# The runner is intentionally stricter than the transport module.  These
# values are the immutable M0 identity; a YAML edit cannot silently select a
# different checkpoint, benchmark, or action contract.
EXPECTED_SEED = 2027
EXPECTED_SUITE = "libero_spatial"
EXPECTED_TASK_ID = 0
EXPECTED_INIT_STATE_ID = 0
EXPECTED_OBS_TYPE = "pixels_agent_pos"
EXPECTED_N_ENVS = 1
EXPECTED_OBSERVATION_SHAPE = (360, 360)
EXPECTED_ACTION_DIM = 7
EXPECTED_CHUNK_SIZE = 50
EXPECTED_N_ACTION_STEPS = 1
EXPECTED_HORIZON = 280
EXPECTED_PHYSICAL_DEVICES = (0, 1)
EXPECTED_CONCURRENCY_STEPS_PER_WORKER = 2
EXPECTED_COMPARE_PHYSICAL_DEVICE = 1
EXPECTED_CLOSED_LOOP_PHYSICAL_DEVICE = 1
EXPECTED_LOGICAL_DEVICE = "cuda:0"

EXPECTED_EGL_DEVICE_ID = "8"
EXPECTED_GL_IDENTITY = {
    "vendor": "Mesa/X.org",
    "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
    "version": "3.1 Mesa 21.1.5",
}

EXPECTED_CHECKPOINT_REPO = "HuggingFaceVLA/smolvla_libero"
EXPECTED_CHECKPOINT_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
EXPECTED_CHECKPOINT_RELATIVE_PATH = Path(
    "external/artifacts/smolvla_libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
)
EXPECTED_CHECKPOINT_MODEL_SHA256 = "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"
EXPECTED_BASE_MODEL_REPO = "HuggingFaceTB/SmolVLM2-500M-Instruct"
EXPECTED_BASE_MODEL_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
EXPECTED_BASE_MODEL_RELATIVE_PATH = Path(
    "external/artifacts/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
EXPECTED_BASE_MODEL_SHA256 = "b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3"
EXPECTED_ASSETS_REPO = "lerobot/libero-assets"
EXPECTED_ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
EXPECTED_ASSETS_RELATIVE_PATH = Path(
    "external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
)
EXPECTED_PREFLIGHT_MANIFEST_SHA256 = "1a94ebb8cc42614744d8dc9ebdad16f5461006f8806cc7d75869118013198a85"
EXPECTED_CPU_LOCK_SHA256 = "921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc"
EXPECTED_DCU_LOCK_SHA256 = "cc507d48c64e72a9d2f6552bc5fa638217f9e6a2addef0067ded3d0adce30ed2"
EXPECTED_CPU_LOCK_PATH = _ROOT / "runtime" / "locks" / "shiftvla-libero-runtime.txt"
EXPECTED_DCU_LOCK_PATH = _ROOT / "runtime" / "locks" / "shiftvla-libero-dcu-runtime.txt"
EXPECTED_WORKER_STARTUP_TIMEOUT_SECONDS = 300
EXPECTED_WORKER_FORWARD_TIMEOUT_SECONDS = 300
EXPECTED_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 30
EXPECTED_LIBERO_CONFIG_SHA256 = "98d57e0d3d7b70bab5c66a110ecc333c737895e9dcd457c06ec94b4150e0ecc8"
EXPECTED_OVERLAY_WHEEL_PATH = Path(
    "/public/home/xuyinghao/tmp/shiftvla-lerobot-py311/wheels/lerobot-0.6.1-py3-none-any.whl"
)
EXPECTED_REFERENCE_RUN_MANIFEST_SHA256 = "bf5c510f1e2c9fbdb0c27dca55f8682bcaaaad6e9bc8c3385aa788501eb91506"
EXPECTED_REFERENCE_DECISIONS_SHA256 = "6d4c65d0fb8e3829971fb8b1023641172683c7fcf019f831823719ccd2299267"
EXPECTED_REFERENCE_RUN_RELATIVE_PATH = Path(
    "runs/m0_smoke/20260825T075758Z_2_f664c2fe/run_manifest.json"
)
EXPECTED_REFERENCE_DECISIONS_RELATIVE_PATH = Path(
    "runs/m0_smoke/20260825T075758Z_2_f664c2fe/decisions.jsonl"
)
EXPECTED_ALLOWED_DIRTY_PATHS = ("AGENTS.md",)
EXPECTED_AGENTS_SHA256 = "956a88bf24253c7120be85ec5b446771e60ff9cac95b7932e7a10d5e3c5fc5d8"
EXPECTED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}
EXPECTED_RENDERER_ENV = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    # Host-specific EGL ordinal preserving the exact frozen M0 renderer identity.
    "MUJOCO_EGL_DEVICE_ID": "8",
}

PHASES = ("compare", "closed-loop", "concurrency")
_ONE_STEP_CHILD_PHASE = "one-step-child"
# This token is deliberately not accepted from configuration.  The parent
# concurrency phase supplies it in its explicitly generated child command.
ONE_STEP_CHILD_TOKEN = "shiftvla-internal-one-step-v1"

EXPECTED_PATCH_FILES = frozenset(
    {
        "src/lerobot/datasets/aggregate.py",
        "src/lerobot/datasets/streaming_dataset.py",
        "src/lerobot/motors/motors_bus.py",
        "src/lerobot/processor/pipeline.py",
        "src/lerobot/utils/io_utils.py",
    }
)
EXPECTED_PATCH_INSERTIONS = 17
EXPECTED_PATCH_DELETIONS = 13
EXPECTED_FLOW_NOISE_SHAPE = (1, 50, 32)

# Backends that would make a later model process use an unapproved accelerated
# path.  Matching is by the top-level import name, so nested modules such as
# ``flash_attn.flash_attn_interface`` are caught as well.
FORBIDDEN_BACKENDS = frozenset(
    {
        "apex",
        "bitsandbytes",
        "deepspeed",
        "flash_attn",
        "mamba",
        "mamba_ssm",
        "triton",
        "xformers",
    }
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DCUPreflightError(f"{name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise DCUPreflightError(f"missing required config field {path}")
        current = current[component]
    return current


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DCUPreflightError(f"{path} must be a non-empty string")
    return value


def _require_positive_int(value: Any, path: str) -> int:
    if not _is_int(value) or value <= 0:
        raise DCUPreflightError(f"{path} must be a positive integer")
    return int(value)


def _require_nonnegative_int(value: Any, path: str) -> int:
    if not _is_int(value) or value < 0:
        raise DCUPreflightError(f"{path} must be a non-negative integer")
    return int(value)


def _require_sha256(value: Any, path: str) -> str:
    digest = _require_string(value, path)
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise DCUPreflightError(f"{path} must be a SHA-256 hex digest")
    return digest.lower()


def _expected_project_path(relative: Path) -> Path:
    """Resolve an immutable project-owned path without consulting config."""

    return (_ROOT / relative).resolve()


def _require_exact(value: Any, expected: Any, path: str) -> Any:
    if value != expected:
        raise DCUPreflightError(f"{path} must equal the frozen value {expected!r}, got {value!r}")
    return value


def validate_config_identity(
    config: Mapping[str, Any],
    *,
    expected_project_sha: str,
    actual_project_sha: str,
) -> dict[str, Any]:
    """Validate the frozen M0 config's identity and internal duplication.

    The project revision is supplied by the caller because this utility must
    not invoke Git.  The rest of the checks are structural and cross-check the
    duplicated task fields, dimensions, immutable artifact revisions and
    runtime isolation settings in the YAML mapping.
    """

    root = _require_mapping(config, "config")
    expected_sha = _require_string(expected_project_sha, "expected project SHA")
    actual_sha = _require_string(actual_project_sha, "actual project SHA")
    if expected_sha != actual_sha:
        raise DCUPreflightError(
            f"project SHA mismatch: expected {expected_sha}, got {actual_sha}"
        )

    schema_version = _required(root, "schema_version")
    if schema_version != 1:
        raise DCUPreflightError(f"schema_version must be 1, got {schema_version!r}")
    if _required(root, "name") != "dcu_preflight":
        raise DCUPreflightError("config name must be dcu_preflight")
    mode = _require_string(_required(root, "mode"), "mode")
    if mode != "explicit_phase":
        raise DCUPreflightError("mode must be the phase-neutral value 'explicit_phase'")

    # These scalar fields are intentionally duplicated in the config for
    # shell-level inspection.  The nested task mapping is authoritative and
    # must agree exactly with them.
    duplicated_fields = (
        "suite",
        "task_id",
        "init_state_id",
        "seed",
        "observation_height",
        "observation_width",
        "horizon",
    )
    task = _require_mapping(_required(root, "task"), "task")
    for field in duplicated_fields:
        top_path = field
        task_path = f"task.{field}"
        top_value = _required(root, top_path)
        task_value = _required(task, field)
        if top_value != task_value:
            raise DCUPreflightError(
                f"config identity mismatch: {top_path} != {task_path}"
            )
    _require_string(task["suite"], "task.suite")
    for field in ("task_id", "init_state_id", "seed", "horizon"):
        _require_nonnegative_int(task[field], f"task.{field}")
    for field in ("observation_height", "observation_width"):
        _require_positive_int(task[field], f"task.{field}")

    _require_exact(task["suite"], EXPECTED_SUITE, "task.suite")
    _require_exact(task["task_id"], EXPECTED_TASK_ID, "task.task_id")
    _require_exact(task["init_state_id"], EXPECTED_INIT_STATE_ID, "task.init_state_id")
    _require_exact(task["seed"], EXPECTED_SEED, "task.seed")
    _require_exact(
        (task["observation_height"], task["observation_width"]),
        EXPECTED_OBSERVATION_SHAPE,
        "task observation shape",
    )
    _require_exact(task["horizon"], EXPECTED_HORIZON, "task.horizon")

    for field in ("task_id", "init_state_id", "seed"):
        _require_nonnegative_int(root[field], field)
    for field in (
        "observation_height",
        "observation_width",
        "action_dim",
        "chunk_size",
        "horizon",
    ):
        _require_positive_int(root[field], field)
    n_action_steps = _require_positive_int(root["n_action_steps"], "n_action_steps")
    if n_action_steps > int(root["chunk_size"]):
        raise DCUPreflightError("n_action_steps cannot exceed chunk_size")

    _require_exact(_required(root, "seed"), EXPECTED_SEED, "seed")
    _require_exact(_required(root, "suite"), EXPECTED_SUITE, "suite")
    _require_exact(_required(root, "task_id"), EXPECTED_TASK_ID, "task_id")
    _require_exact(_required(root, "init_state_id"), EXPECTED_INIT_STATE_ID, "init_state_id")
    _require_exact(_required(root, "obs_type"), EXPECTED_OBS_TYPE, "obs_type")
    _require_exact(_required(root, "n_envs"), EXPECTED_N_ENVS, "n_envs")
    _require_exact(_required(root, "use_async_envs"), False, "use_async_envs")
    _require_exact(
        (_required(root, "observation_height"), _required(root, "observation_width")),
        EXPECTED_OBSERVATION_SHAPE,
        "observation shape",
    )
    _require_exact(_required(root, "action_dim"), EXPECTED_ACTION_DIM, "action_dim")
    _require_exact(_required(root, "chunk_size"), EXPECTED_CHUNK_SIZE, "chunk_size")
    _require_exact(_required(root, "n_action_steps"), EXPECTED_N_ACTION_STEPS, "n_action_steps")
    _require_exact(_required(root, "horizon"), EXPECTED_HORIZON, "horizon")

    checkpoint = _require_mapping(_required(root, "checkpoint"), "checkpoint")
    for field in ("repo_id", "revision", "path"):
        _require_string(_required(root, f"checkpoint.{field}"), f"checkpoint.{field}")
    _require_exact(checkpoint["repo_id"], EXPECTED_CHECKPOINT_REPO, "checkpoint.repo_id")
    _require_exact(checkpoint["revision"], EXPECTED_CHECKPOINT_REVISION, "checkpoint.revision")
    _require_exact(
        Path(str(checkpoint["path"])).resolve(),
        _expected_project_path(EXPECTED_CHECKPOINT_RELATIVE_PATH),
        "checkpoint.path",
    )
    _require_exact(
        _require_sha256(_required(root, "checkpoint.model_sha256"), "checkpoint.model_sha256"),
        EXPECTED_CHECKPOINT_MODEL_SHA256,
        "checkpoint.model_sha256",
    )

    base_model = _require_mapping(_required(root, "base_model"), "base_model")
    for field in ("repo_id", "revision", "path"):
        _require_string(_required(root, f"base_model.{field}"), f"base_model.{field}")
    _require_exact(base_model["repo_id"], EXPECTED_BASE_MODEL_REPO, "base_model.repo_id")
    _require_exact(base_model["revision"], EXPECTED_BASE_MODEL_REVISION, "base_model.revision")
    _require_exact(
        Path(str(base_model["path"])).resolve(),
        _expected_project_path(EXPECTED_BASE_MODEL_RELATIVE_PATH),
        "base_model.path",
    )
    _require_exact(
        _require_sha256(_required(root, "base_model.model_sha256"), "base_model.model_sha256"),
        EXPECTED_BASE_MODEL_SHA256,
        "base_model.model_sha256",
    )

    assets = _require_mapping(_required(root, "assets"), "assets")
    for field in ("repo_id", "revision", "path"):
        _require_string(_required(root, f"assets.{field}"), f"assets.{field}")
    _require_exact(assets["repo_id"], EXPECTED_ASSETS_REPO, "assets.repo_id")
    _require_exact(assets["revision"], EXPECTED_ASSETS_REVISION, "assets.revision")
    _require_exact(
        Path(str(assets["path"])).resolve(),
        _expected_project_path(EXPECTED_ASSETS_RELATIVE_PATH),
        "assets.path",
    )
    assets_hash = assets.get("manifest_sha256", assets.get("tree_sha256"))
    _require_exact(
        _require_sha256(assets_hash, "assets.manifest_sha256"),
        EXPECTED_PREFLIGHT_MANIFEST_SHA256,
        "assets.manifest_sha256",
    )

    runtime = _require_mapping(_required(root, "runtime"), "runtime")
    for field in (
        "cpu_python",
        "dcu_python",
        "cpu_runtime_lock",
        "dcu_runtime_lock",
        "card0_reason",
        "parent_device",
        "logical_device",
        "renderer_device",
    ):
        _require_string(_required(root, f"runtime.{field}"), f"runtime.{field}")
    # Startup and forward calls have a generous but finite bound: a busy
    # accelerator must not turn a valid, merely unoptimised inference into a
    # protocol failure.  Shutdown remains short and is handled separately.
    for timeout_field, expected_timeout in (
        ("worker_startup_timeout_seconds", EXPECTED_WORKER_STARTUP_TIMEOUT_SECONDS),
        ("worker_forward_timeout_seconds", EXPECTED_WORKER_FORWARD_TIMEOUT_SECONDS),
        ("worker_shutdown_timeout_seconds", EXPECTED_WORKER_SHUTDOWN_TIMEOUT_SECONDS),
    ):
        timeout = _require_positive_int(
            _required(root, f"runtime.{timeout_field}"), f"runtime.{timeout_field}"
        )
        _require_exact(timeout, expected_timeout, f"runtime.{timeout_field}")
    if "worker_timeout_seconds" in runtime:
        _require_positive_int(_required(root, "runtime.worker_timeout_seconds"), "runtime.worker_timeout_seconds")
    physical_devices = _required(root, "runtime.physical_devices")
    if not isinstance(physical_devices, Sequence) or isinstance(physical_devices, (str, bytes)):
        raise DCUPreflightError("runtime.physical_devices must be a sequence")
    if not physical_devices:
        raise DCUPreflightError("runtime.physical_devices cannot be empty")
    normalized_devices: list[int] = []
    for index, device in enumerate(physical_devices):
        normalized_devices.append(_require_nonnegative_int(device, f"runtime.physical_devices[{index}]"))
    if len(set(normalized_devices)) != len(normalized_devices):
        raise DCUPreflightError("runtime.physical_devices must be unique")
    if tuple(normalized_devices) != EXPECTED_PHYSICAL_DEVICES:
        raise DCUPreflightError(
            f"runtime.physical_devices must equal {list(EXPECTED_PHYSICAL_DEVICES)}, got {normalized_devices}"
        )
    for field in ("compare_physical_device", "closed_loop_physical_device"):
        device = _require_nonnegative_int(_required(root, f"runtime.{field}"), f"runtime.{field}")
        if device not in normalized_devices:
            raise DCUPreflightError(f"runtime.{field} is not in runtime.physical_devices")
    _require_exact(
        _required(root, "runtime.compare_physical_device"),
        EXPECTED_COMPARE_PHYSICAL_DEVICE,
        "runtime.compare_physical_device",
    )
    _require_exact(
        _required(root, "runtime.closed_loop_physical_device"),
        EXPECTED_CLOSED_LOOP_PHYSICAL_DEVICE,
        "runtime.closed_loop_physical_device",
    )
    _require_exact(_required(root, "runtime.logical_device"), EXPECTED_LOGICAL_DEVICE, "runtime.logical_device")
    _require_exact(_required(root, "runtime.parent_device"), "cpu", "runtime.parent_device")
    if Path(str(runtime["cpu_runtime_lock"])).resolve() != EXPECTED_CPU_LOCK_PATH.resolve():
        raise DCUPreflightError("runtime.cpu_runtime_lock does not identify the pinned CPU lock")
    if Path(str(runtime["dcu_runtime_lock"])).resolve() != EXPECTED_DCU_LOCK_PATH.resolve():
        raise DCUPreflightError("runtime.dcu_runtime_lock does not identify the pinned DCU lock")
    _require_exact(_required(root, "runtime.renderer_device"), "llvmpipe", "runtime.renderer_device")

    patch = _require_mapping(_required(root, "patch"), "patch")
    patch_path = _require_string(_required(root, "patch.path"), "patch.path")
    if Path(patch_path).resolve() != PATCH_PATH.resolve():
        raise DCUPreflightError("patch.path does not identify the pinned patch")
    patch_sha = _require_sha256(_required(root, "patch.sha256"), "patch.sha256")
    if patch_sha != EXPECTED_PATCH_SHA256:
        raise DCUPreflightError("patch.sha256 does not match the pinned patch")
    overlay_sha = _require_sha256(
        _required(root, "patch.overlay_wheel_sha256"), "patch.overlay_wheel_sha256"
    )
    if overlay_sha != EXPECTED_OVERLAY_WHEEL_SHA256:
        raise DCUPreflightError("patch.overlay_wheel_sha256 does not match the pinned wheel")
    if _required(root, "patch.applies_only_to") != "explicit_dcu_runtime":
        raise DCUPreflightError("patch.applies_only_to must be explicit_dcu_runtime")
    overlay_path = _require_string(
        _required(root, "patch.overlay_wheel_path"), "patch.overlay_wheel_path"
    )
    if Path(overlay_path).resolve() != EXPECTED_OVERLAY_WHEEL_PATH.resolve():
        raise DCUPreflightError("patch.overlay_wheel_path does not identify the pinned wheel")

    for section_name in ("offline", "renderer"):
        section = _require_mapping(_required(root, section_name), section_name)
        for key, value in section.items():
            _require_string(value, f"{section_name}.{key}")
    if dict(_required(root, "offline")) != EXPECTED_OFFLINE_ENV:
        raise DCUPreflightError("offline environment must equal the frozen fail-closed mapping")
    if dict(_required(root, "renderer")) != EXPECTED_RENDERER_ENV:
        raise DCUPreflightError("renderer environment must equal the frozen EGL mapping")

    libero_config = _required(root, "libero_config")
    if isinstance(libero_config, Mapping):
        libero_path = _require_string(_required(libero_config, "path"), "libero_config.path")
        libero_hash = _require_sha256(
            _required(libero_config, "sha256"), "libero_config.sha256"
        )
    else:
        libero_path = _require_string(libero_config, "libero_config")
        libero_hash = _require_sha256(
            _required(root, "libero_config_sha256"), "libero_config_sha256"
        )
    config_path_alias = _require_string(
        _required(root, "libero_config_path"), "libero_config_path"
    )
    # The environment variable/API takes the LIBERO config *directory*, while
    # the archived hash is for its concrete config.yaml file.
    if Path(libero_path).resolve().parent != Path(config_path_alias).resolve():
        raise DCUPreflightError("libero_config_path must be the directory containing libero_config.path")
    if not Path(libero_path).is_absolute() or Path(libero_path).name != "config.yaml":
        raise DCUPreflightError("libero_config.path must be an absolute config.yaml path")
    if libero_hash != EXPECTED_LIBERO_CONFIG_SHA256:
        raise DCUPreflightError("libero_config.sha256 does not match the pinned config")

    manifest_path = _require_string(_required(root, "preflight_manifest"), "preflight_manifest")
    manifest_hash = _require_sha256(
        _required(root, "preflight_manifest_sha256"), "preflight_manifest_sha256"
    )
    if Path(manifest_path).resolve() != _ROOT / "runtime/manifests/m0_preflight_b_artifacts.json":
        raise DCUPreflightError("preflight_manifest does not identify the pinned artifact manifest")
    if manifest_hash != EXPECTED_PREFLIGHT_MANIFEST_SHA256:
        raise DCUPreflightError("preflight_manifest_sha256 does not match the pinned manifest")

    reference = _require_mapping(_required(root, "m0_reference"), "m0_reference")
    for key in ("run_manifest_path", "decisions_path"):
        _require_string(_required(reference, key), f"m0_reference.{key}")
    if Path(reference["run_manifest_path"]).resolve() != _expected_project_path(
        EXPECTED_REFERENCE_RUN_RELATIVE_PATH
    ):
        raise DCUPreflightError("m0_reference.run_manifest_path is not the archived M0 run")
    if Path(reference["decisions_path"]).resolve() != _expected_project_path(
        EXPECTED_REFERENCE_DECISIONS_RELATIVE_PATH
    ):
        raise DCUPreflightError("m0_reference.decisions_path is not the archived M0 decisions")
    if _require_sha256(reference.get("run_manifest_sha256"), "m0_reference.run_manifest_sha256") != EXPECTED_REFERENCE_RUN_MANIFEST_SHA256:
        raise DCUPreflightError("M0 reference run manifest hash does not match")
    if _require_sha256(reference.get("decisions_sha256"), "m0_reference.decisions_sha256") != EXPECTED_REFERENCE_DECISIONS_SHA256:
        raise DCUPreflightError("M0 reference decisions hash does not match")

    runtime_lock = _require_string(_required(root, "runtime_lock"), "runtime_lock")
    if Path(runtime_lock).resolve() != Path(_required(runtime, "cpu_runtime_lock")).resolve():
        raise DCUPreflightError("runtime_lock must alias runtime.cpu_runtime_lock")
    if _require_sha256(_required(root, "runtime_lock_sha256"), "runtime_lock_sha256") != EXPECTED_CPU_LOCK_SHA256:
        raise DCUPreflightError("runtime_lock_sha256 does not match the pinned CPU lock")
    if _require_sha256(_required(root, "dcu_runtime_lock_sha256"), "dcu_runtime_lock_sha256") != EXPECTED_DCU_LOCK_SHA256:
        raise DCUPreflightError("dcu_runtime_lock_sha256 does not match the pinned DCU lock")

    output = _require_mapping(_required(root, "output"), "output")
    _require_string(_required(output, "root"), "output.root")
    project = _require_mapping(_required(root, "project"), "project")
    allowed_dirty_paths = _required(project, "allowed_dirty_paths")
    if not isinstance(allowed_dirty_paths, Sequence) or isinstance(allowed_dirty_paths, (str, bytes)):
        raise DCUPreflightError("project.allowed_dirty_paths must be a sequence")
    for index, path in enumerate(allowed_dirty_paths):
        _require_string(path, f"project.allowed_dirty_paths[{index}]")
    dirty_hashes = _required(project, "allowed_dirty_sha256")
    dirty_hashes = _require_mapping(dirty_hashes, "project.allowed_dirty_sha256")
    for path, digest in dirty_hashes.items():
        _require_string(path, "project.allowed_dirty_sha256 key")
        _require_sha256(digest, f"project.allowed_dirty_sha256.{path}")
    if tuple(allowed_dirty_paths) != EXPECTED_ALLOWED_DIRTY_PATHS:
        raise DCUPreflightError("only AGENTS.md may be dirty during a real preflight")
    if dict(dirty_hashes) != {"AGENTS.md": EXPECTED_AGENTS_SHA256}:
        raise DCUPreflightError("allowed dirty AGENTS.md hash is not the approved one")

    return {
        "project_sha": actual_sha,
        "schema_version": int(schema_version),
        "mode": mode,
        "seed": int(root["seed"]),
        "task": {
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "init_state_id": int(task["init_state_id"]),
        },
        "checkpoint_revision": str(checkpoint["revision"]),
        "physical_devices": normalized_devices,
        "patch_sha256": patch_sha,
    }


def _normalise_noise_shape(shape: Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, (str, bytes)):
        raise DCUPreflightError(f"noise shape must be {EXPECTED_FLOW_NOISE_SHAPE}")
    try:
        normalized = tuple(shape)
    except TypeError as exc:
        raise DCUPreflightError(f"noise shape must be {EXPECTED_FLOW_NOISE_SHAPE}") from exc
    if not all(_is_int(item) for item in normalized):
        raise DCUPreflightError(f"noise shape must be {EXPECTED_FLOW_NOISE_SHAPE}")
    return tuple(int(item) for item in normalized)


def generate_flow_noise(*, seed: int, shape: Sequence[int]) -> tuple[torch.Tensor, str]:
    """Generate explicit CPU float32 flow noise and its content digest."""

    if not _is_int(seed) or seed < 0:
        raise DCUPreflightError("noise seed must be a non-negative integer")
    normalized_shape = _normalise_noise_shape(shape)
    if normalized_shape != EXPECTED_FLOW_NOISE_SHAPE:
        raise DCUPreflightError(
            f"noise shape {normalized_shape} does not match {EXPECTED_FLOW_NOISE_SHAPE}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(
        normalized_shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    if not bool(torch.isfinite(noise).all().item()):  # pragma: no cover - torch invariant
        raise DCUPreflightError("generated flow noise is not finite")
    digest = hashlib.sha256(noise.detach().contiguous().numpy().tobytes()).hexdigest()
    return noise, digest


def _numeric_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        raise DCUPreflightError(f"{name} must be a torch.Tensor or numpy.ndarray")
    if tensor.dtype == torch.bool or not (tensor.is_floating_point() or tensor.is_complex()):
        # Integer action tensors are still numeric, but converting here makes
        # the finite and difference checks unambiguous.
        if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex or tensor.dtype in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }):
            raise DCUPreflightError(f"{name} must contain numeric values")
    if tensor.is_complex():
        raise DCUPreflightError(f"{name} must be real-valued")
    if not bool(torch.isfinite(tensor).all().item()):
        raise DCUPreflightError(f"{name} contains non-finite values")
    return tensor


def _finite_scalar(value: Any, name: str, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DCUPreflightError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise DCUPreflightError(f"{name} must be a finite non-negative number")
    return result


def compare_outputs(
    cpu_chunk: torch.Tensor | np.ndarray,
    dcu_chunk: torch.Tensor | np.ndarray,
    cpu_action: torch.Tensor | np.ndarray,
    dcu_action: torch.Tensor | np.ndarray,
    *,
    cpu_latency: float | None = None,
    dcu_latency: float | None = None,
    dcu_peak_memory: int | None = None,
) -> dict[str, Any]:
    """Compare finite, shape-matched normalized chunks and final actions."""

    cpu_chunk_tensor = _numeric_tensor(cpu_chunk, "cpu_chunk")
    dcu_chunk_tensor = _numeric_tensor(dcu_chunk, "dcu_chunk")
    cpu_action_tensor = _numeric_tensor(cpu_action, "cpu_action")
    dcu_action_tensor = _numeric_tensor(dcu_action, "dcu_action")
    if tuple(cpu_chunk_tensor.shape) != tuple(dcu_chunk_tensor.shape):
        raise DCUPreflightError(
            f"chunk shape mismatch: {tuple(cpu_chunk_tensor.shape)} != {tuple(dcu_chunk_tensor.shape)}"
        )
    if tuple(cpu_action_tensor.shape) != tuple(dcu_action_tensor.shape):
        raise DCUPreflightError(
            f"action shape mismatch: {tuple(cpu_action_tensor.shape)} != {tuple(dcu_action_tensor.shape)}"
        )
    if cpu_chunk_tensor.numel() == 0 or cpu_action_tensor.numel() == 0:
        raise DCUPreflightError("comparison tensors cannot be empty")

    chunk_diff = (
        cpu_chunk_tensor.to(device="cpu", dtype=torch.float64)
        - dcu_chunk_tensor.to(device="cpu", dtype=torch.float64)
    ).abs()
    action_diff = (
        cpu_action_tensor.to(device="cpu", dtype=torch.float64)
        - dcu_action_tensor.to(device="cpu", dtype=torch.float64)
    ).abs()
    cpu_latency_value = _finite_scalar(cpu_latency, "cpu_latency")
    dcu_latency_value = _finite_scalar(dcu_latency, "dcu_latency")
    if dcu_peak_memory is not None:
        if isinstance(dcu_peak_memory, bool) or not isinstance(dcu_peak_memory, (int, np.integer)):
            raise DCUPreflightError("dcu_peak_memory must be a non-negative integer")
        if int(dcu_peak_memory) < 0:
            raise DCUPreflightError("dcu_peak_memory must be a non-negative integer")
        dcu_peak_memory_value: int | None = int(dcu_peak_memory)
    else:
        dcu_peak_memory_value = None
    return {
        "finite": True,
        "chunk_shape": [int(item) for item in cpu_chunk_tensor.shape],
        "action_shape": [int(item) for item in cpu_action_tensor.shape],
        "normalized_max_abs_diff": float(chunk_diff.max().item()),
        "normalized_mean_abs_diff": float(chunk_diff.mean().item()),
        "final_action_max_abs_diff": float(action_diff.max().item()),
        "final_action_mean_abs_diff": float(action_diff.mean().item()),
        "cpu_latency_seconds": cpu_latency_value,
        "dcu_latency_seconds": dcu_latency_value,
        "dcu_peak_memory_bytes": dcu_peak_memory_value,
    }


def map_physical_to_logical(physical_index: int) -> dict[str, Any]:
    """Describe one isolated physical accelerator as logical ``cuda:0``."""

    if not _is_int(physical_index) or physical_index < 0:
        raise DCUPreflightError("physical device index must be a non-negative integer")
    physical = int(physical_index)
    return {
        "physical_index": physical,
        "hip_visible_devices": str(physical),
        "logical_device": "cuda:0",
    }


def build_worker_environment(
    physical_index: int,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy an environment and pin one physical device to logical ``cuda:0``."""

    mapping = map_physical_to_logical(physical_index)
    result = _copy_environment(base_environment)
    expected_hip = mapping["hip_visible_devices"]
    for key, expected in (
        ("HIP_VISIBLE_DEVICES", expected_hip),
        ("CUDA_VISIBLE_DEVICES", "0"),
    ):
        if key in result and result[key] != expected:
            raise DCUPreflightError(f"worker environment {key} conflicts with physical device mapping")
        result[key] = expected
    return result


def _copy_environment(base_environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy a subprocess environment after applying the same strict checks.

    Environment construction is deliberately kept separate from device
    visibility.  A CPU LIBERO child needs the validated renderer/offline
    variables, while the nested model worker receives the compute mapping
    from :func:`build_worker_environment` at its own process boundary.
    """

    if base_environment is None:
        base_environment = {}
    if not isinstance(base_environment, Mapping):
        raise DCUPreflightError("worker environment must be a mapping")
    result: dict[str, str] = {}
    for key, value in base_environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise DCUPreflightError("worker environment keys and values must be strings")
        if "\x00" in key or "\x00" in value:
            raise DCUPreflightError("worker environment contains an invalid NUL/newline")
        if key.startswith("BASH_FUNC_") and key.endswith("%%"):
            continue
        if "\n" in key or "\n" in value:
            raise DCUPreflightError("worker environment contains an invalid NUL/newline")
        result[key] = value
    return result


def _module_names(imported_modules: Mapping[str, Any] | Iterable[Any] | None) -> set[str]:
    if imported_modules is None:
        return {str(name) for name in sys.modules if isinstance(name, str)}
    if isinstance(imported_modules, Mapping):
        return {str(name) for name in imported_modules if isinstance(name, str)}
    try:
        return {str(name) for name in imported_modules}
    except TypeError as exc:
        raise DCUPreflightError("imported_modules must be a mapping or iterable") from exc


def validate_forbidden_backends(
    *,
    imported_modules: Mapping[str, Any] | Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Fail closed when an unapproved optional backend is already imported."""

    names = _module_names(imported_modules)
    forbidden = sorted(
        {
            name
            for name in names
            if name == name.strip()
            and name.split(".", 1)[0].lower() in FORBIDDEN_BACKENDS
        }
    )
    if forbidden:
        raise DCUPreflightError(f"forbidden backend imported: {', '.join(forbidden)}")
    return {"forbidden": forbidden}


def _validate_seed(seed: int) -> int:
    if not _is_int(seed) or seed < 0:
        raise DCUPreflightError("seed must be a non-negative integer")
    return int(seed)


def enforce_one_step_boundary(env: Any, client: Any, *, seed: int) -> dict[str, Any]:
    """Execute exactly one fake reset, one decision and one environment step."""

    normalized_seed = _validate_seed(seed)
    if not callable(getattr(env, "reset", None)) or not callable(getattr(env, "step", None)):
        raise DCUPreflightError("environment must provide reset() and step()")
    if not callable(getattr(client, "reset", None)) or not callable(getattr(client, "select_action", None)):
        raise DCUPreflightError("client must provide reset() and select_action()")

    reset_result = env.reset(seed=normalized_seed)
    client.reset()
    decision_observation = reset_result
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        decision_observation = reset_result[0]
    action = client.select_action(decision_observation)
    if isinstance(action, torch.Tensor):
        if action.is_complex() or not bool(torch.isfinite(action).all().item()):
            raise DCUPreflightError("selected action contains non-finite values")
        action_for_env: Any = action.detach().cpu().numpy()
    elif isinstance(action, np.ndarray):
        if not np.isfinite(action).all():
            raise DCUPreflightError("selected action contains non-finite values")
        action_for_env = action
    else:
        action_for_env = action

    step_result = env.step(action_for_env)
    if not isinstance(step_result, tuple) or len(step_result) != 5:
        raise DCUPreflightError("environment step must return (observation, reward, terminated, truncated, info)")
    _, reward, terminated, truncated, info = step_result
    if isinstance(terminated, np.bool_):
        terminated = bool(terminated)
    if isinstance(truncated, np.bool_):
        truncated = bool(truncated)
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise DCUPreflightError("environment terminated/truncated flags must be boolean")
    reward_value = _finite_scalar(reward, "reward", allow_none=False)
    if not isinstance(info, Mapping):
        raise DCUPreflightError("environment info must be a mapping")
    return {
        "seed": normalized_seed,
        "decision_count": 1,
        "env_step_count": 1,
        "terminated": terminated,
        "truncated": truncated,
        "done": terminated or truncated,
        "reward": reward_value,
        "info": dict(info),
    }


def create_run_directory(root: str | Path) -> Path:
    """Create a fresh UTC-stamped run directory without overwriting one."""

    root_path = Path(root)
    if root_path.exists() and not root_path.is_dir():
        raise NotADirectoryError(root_path)
    root_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for suffix in range(10000):
        name = stamp if suffix == 0 else f"{stamp}_{suffix}"
        candidate = root_path / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not allocate a unique run directory below {root_path}")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DCUPreflightError("failure manifest contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    raise DCUPreflightError(f"failure manifest value is not JSON serializable: {type(value).__name__}")


def write_failure_manifest(
    run_directory: str | Path,
    error: BaseException,
    *,
    context: Mapping[str, Any],
) -> Path:
    """Write a single failure manifest atomically with exclusive creation."""

    directory = Path(run_directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if not isinstance(context, Mapping):
        raise DCUPreflightError("failure manifest context must be a mapping")
    payload = {
        "status": "FAIL",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "context": _json_safe(context),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = directory / FAILURE_MANIFEST_NAME
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return target


class _PatchHunk:
    __slots__ = ("old_start", "old_count", "new_start", "new_count", "lines")

    def __init__(
        self,
        old_start: int,
        old_count: int,
        new_start: int,
        new_count: int,
        lines: list[str],
    ) -> None:
        self.old_start = old_start
        self.old_count = old_count
        self.new_start = new_start
        self.new_count = new_count
        self.lines = lines


class _PatchSection:
    __slots__ = ("path", "old_path", "new_path", "hunks")

    def __init__(self, path: str, old_path: str, new_path: str, hunks: list[_PatchHunk]) -> None:
        self.path = path
        self.old_path = old_path
        self.new_path = new_path
        self.hunks = hunks


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+?)(?:\n)?$")
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*?)(?:\n)?$"
)


def _strip_line_ending(line: str) -> str:
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def _parse_patch(text: str) -> list[_PatchSection]:
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts:
        raise DCUPreflightError("patch contains no unified diff sections")
    sections: list[_PatchSection] = []
    for section_index, start in enumerate(starts):
        stop = starts[section_index + 1] if section_index + 1 < len(starts) else len(lines)
        header = _strip_line_ending(lines[start])
        match = _DIFF_HEADER_RE.fullmatch(header)
        if match is None or match.group(1) != match.group(2):
            raise DCUPreflightError("patch must contain same-path a/ and b/ entries")
        path = match.group(1)
        old_path: str | None = None
        new_path: str | None = None
        hunks: list[_PatchHunk] = []
        cursor = start + 1
        while cursor < stop:
            line = lines[cursor]
            stripped = _strip_line_ending(line)
            if stripped.startswith("--- "):
                old_path = stripped[4:]
                if old_path.startswith("a/"):
                    old_path = old_path[2:]
                cursor += 1
                continue
            if stripped.startswith("+++ "):
                new_path = stripped[4:]
                if new_path.startswith("b/"):
                    new_path = new_path[2:]
                cursor += 1
                continue
            hunk_match = _HUNK_HEADER_RE.fullmatch(stripped)
            if hunk_match is None:
                cursor += 1
                continue
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2) or "1")
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4) or "1")
            body: list[str] = []
            cursor += 1
            while cursor < stop:
                body_line = lines[cursor]
                body_stripped = _strip_line_ending(body_line)
                if _HUNK_HEADER_RE.fullmatch(body_stripped) is not None or body_line.startswith("diff --git "):
                    break
                if body_line.startswith((" ", "+", "-")):
                    body.append(body_line)
                elif body_stripped == "\\ No newline at end of file":
                    # This marker describes the preceding payload and is not
                    # itself part of either side of the hunk.
                    pass
                else:
                    raise DCUPreflightError(f"invalid unified diff hunk line in {path}")
                cursor += 1
            hunks.append(_PatchHunk(old_start, old_count, new_start, new_count, body))
        if old_path != path or new_path != path:
            raise DCUPreflightError(f"patch path metadata does not match {path}")
        if not hunks:
            raise DCUPreflightError(f"patch section has no hunks: {path}")
        sections.append(_PatchSection(path, old_path, new_path, hunks))
    return sections


def _apply_hunks_in_memory(source: list[str], hunks: Sequence[_PatchHunk], path: str) -> tuple[list[str], int, int]:
    output: list[str] = []
    cursor = 0
    insertions = 0
    deletions = 0
    for hunk in hunks:
        if hunk.old_start == 0:
            start = 0
        elif hunk.old_count == 0:
            start = hunk.old_start
        else:
            start = hunk.old_start - 1
        if start < cursor or start > len(source):
            raise DCUPreflightError(f"patch hunk position is invalid for {path}")
        output.extend(source[cursor:start])
        cursor = start
        old_seen = 0
        new_seen = 0
        for line in hunk.lines:
            prefix = line[:1]
            payload = line[1:]
            if prefix == " ":
                if cursor >= len(source) or source[cursor] != payload:
                    raise DCUPreflightError(f"patch context does not apply to pinned source: {path}")
                output.append(source[cursor])
                cursor += 1
                old_seen += 1
                new_seen += 1
            elif prefix == "-":
                if cursor >= len(source) or source[cursor] != payload:
                    raise DCUPreflightError(f"patch deletion does not apply to pinned source: {path}")
                cursor += 1
                old_seen += 1
                deletions += 1
            elif prefix == "+":
                output.append(payload)
                new_seen += 1
                insertions += 1
            else:  # pragma: no cover - parser rejects before this point
                raise DCUPreflightError(f"invalid patch operation for {path}")
        if old_seen != hunk.old_count or new_seen != hunk.new_count:
            raise DCUPreflightError(f"patch hunk line counts are invalid for {path}")
    output.extend(source[cursor:])
    return output, insertions, deletions


def validate_patch(
    patch_path: str | Path,
    *,
    source_root: str | Path,
) -> dict[str, Any]:
    """Validate the pinned patch hash, allowlist, line counts and application.

    Applying is performed entirely in memory against the source files.  This
avoids mutating the external checkout and deliberately avoids an external command
    or a Git CLI dependency in the fake-only layer.
    """

    patch = Path(patch_path)
    if not patch.is_file():
        raise FileNotFoundError(patch)
    raw = patch.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_PATCH_SHA256:
        raise DCUPreflightError(
            f"patch sha256 mismatch: expected {EXPECTED_PATCH_SHA256}, got {digest}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DCUPreflightError("patch must be UTF-8 text") from exc
    sections = _parse_patch(text)
    files = {section.path for section in sections}
    if files != set(EXPECTED_PATCH_FILES):
        raise DCUPreflightError(
            f"patch file allowlist mismatch: expected {sorted(EXPECTED_PATCH_FILES)}, got {sorted(files)}"
        )

    root = Path(source_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    resolved_root = root.resolve()
    total_insertions = 0
    total_deletions = 0
    for section in sections:
        relative = Path(section.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DCUPreflightError(f"patch path escapes source root: {section.path}")
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise DCUPreflightError(f"patch path escapes source root: {section.path}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        source_text = candidate.read_text(encoding="utf-8")
        _, insertions, deletions = _apply_hunks_in_memory(
            source_text.splitlines(keepends=True), section.hunks, section.path
        )
        total_insertions += insertions
        total_deletions += deletions
    if total_insertions != EXPECTED_PATCH_INSERTIONS or total_deletions != EXPECTED_PATCH_DELETIONS:
        raise DCUPreflightError(
            "patch insertion/deletion counts do not match the pinned contract: "
            f"got {total_insertions}/{total_deletions}"
        )
    return {
        "sha256": digest,
        "files": files,
        "insertions": total_insertions,
        "deletions": total_deletions,
        "applies": True,
        "source_root": str(resolved_root),
    }


# ---------------------------------------------------------------------------
# Explicit phase runner
# ---------------------------------------------------------------------------


def _normalise_phase(phase: str) -> str:
    if not isinstance(phase, str):
        raise DCUPreflightError("phase must be a string")
    normalized = phase.strip().lower().replace("_", "-")
    if normalized not in PHASES and normalized != _ONE_STEP_CHILD_PHASE:
        raise DCUPreflightError(f"unsupported phase: {phase}")
    return normalized


def _sha256_file(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise DCUPreflightError(f"expected a regular file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file() or target.is_symlink():
        raise DCUPreflightError(f"config must be a regular file: {target}")
    try:
        import yaml

        value = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DCUPreflightError(f"could not load config {target}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise DCUPreflightError("config must decode to a mapping")
    return dict(value)


def _json_dump(path: str | Path, value: Any, *, exclusive: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    encoded = json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
    return target


def _atomic_json_dump(path: str | Path, value: Any) -> Path:
    """Replace one JSON artifact atomically, including stale child manifests."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def _jsonl_dump(path: str | Path, rows: Sequence[Mapping[str, Any]], *, exclusive: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with target.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    return target


class _Tee:
    """Write diagnostics to both the caller's stream and a phase log."""

    def __init__(self, stream: Any, path: str | Path) -> None:
        self.stream = stream
        self.file = Path(path).open("a", encoding="utf-8", buffering=1)

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.file.write(value)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class _RemoteQueue:
    def __init__(self) -> None:
        self.length = 0

    def __len__(self) -> int:
        return int(self.length)


class FeatureOnlyRemotePolicy(torch.nn.Module):
    """An ``nn.Module`` seam that delegates only queue-native action selection.

    The CPU process owns the simulator and all official preprocessing.  This
    wrapper filters the processor output down to the exact feature schema,
    writes it to the phase-local IPC directory, and asks the DCU client for
    one queued action.  It intentionally has no model ``forward`` path and
    rejects explicit noise for closed-loop calls.
    """

    def __init__(
        self,
        client: Any,
        *,
        request_writer: Callable[[Mapping[str, torch.Tensor]], str | Path],
        trace: Any | None = None,
    ) -> None:
        super().__init__()
        if not callable(getattr(client, "select_action", None)):
            raise DCUPreflightError("remote client must provide select_action()")
        if not callable(request_writer):
            raise DCUPreflightError("request_writer must be callable")
        self.client = client
        self.request_writer = request_writer
        self.trace = trace
        self.last_queue_evidence: dict[str, Any] = {}
        self._remote_queue = _RemoteQueue()
        self._queues = {"action": self._remote_queue}
        self._seed = EXPECTED_SEED
        self.reset_timeout_seconds: float | None = None
        self.forward_timeout_seconds: float | None = None

    @staticmethod
    def _feature_bundle(features: Mapping[str, Any]) -> Mapping[str, torch.Tensor] | Mapping[str, Any]:
        if not isinstance(features, Mapping):
            raise DCUPreflightError("remote policy features must be a mapping")
        if "noise" in features:
            raise DCUPreflightError("feature-only select_action request may not contain noise")
        # The tiny path-only mapping is useful for fake transport tests; the
        # actual runner always supplies the complete official feature mapping.
        if set(features) == {"path"}:
            return features
        try:
            from scripts.dcu_model_worker import FEATURE_SCHEMA

            expected = set(FEATURE_SCHEMA)
        except Exception as exc:  # pragma: no cover - project dependency invariant
            raise DCUPreflightError(f"could not load feature schema: {exc}") from exc
        missing = sorted(expected - set(features))
        if missing:
            raise DCUPreflightError(f"feature-only request is missing keys: {missing}")
        bundle = {key: features[key] for key in expected}
        # Validation is repeated by save_tensor_bundle, but doing it here
        # provides a clear policy-boundary error before any IPC is attempted.
        from scripts.dcu_worker import validate_tensor_bundle

        validate_tensor_bundle(bundle, schema=FEATURE_SCHEMA)
        return bundle

    def reset(self, *, seed: int = EXPECTED_SEED) -> Mapping[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise DCUPreflightError("remote policy reset seed must be a non-negative integer")
        reset_call = lambda: self.client.reset(seed=int(seed))
        if self.reset_timeout_seconds is not None:
            response = _bounded_call(
                reset_call,
                timeout=self.reset_timeout_seconds,
                name="remote-reset",
            )
        else:
            response = reset_call()
        if not isinstance(response, Mapping):
            raise DCUPreflightError("remote reset response must be a mapping")
        self._seed = int(seed)
        self.last_queue_evidence = dict(response.get("queue_evidence", response))
        after = response.get("queue_length_after", response.get("queue_length", 0))
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise DCUPreflightError("remote reset queue length is invalid")
        self._remote_queue.length = int(after)
        return response

    def select_action(self, features: Mapping[str, Any], *args: Any, **kwargs: Any) -> torch.Tensor:
        del args
        if kwargs:
            raise DCUPreflightError("feature-only select_action does not accept extra arguments")
        bundle = self._feature_bundle(features)
        request_path = Path(self.request_writer(bundle))
        if request_path.suffix != ".safetensors":
            raise DCUPreflightError("feature-only request must be a .safetensors path")
        started = time.perf_counter()
        select_call = lambda: self.client.select_action(request_path)
        if self.forward_timeout_seconds is not None:
            result = _bounded_call(
                select_call,
                timeout=self.forward_timeout_seconds,
                name="remote-select-action",
            )
        else:
            result = select_call()
        wall_latency = time.perf_counter() - started
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, EXPECTED_ACTION_DIM):
            raise DCUPreflightError(
                f"remote select_action output shape {getattr(result, 'shape', None)} does not match (1, 7)"
            )
        if result.dtype not in (torch.float32, torch.bfloat16) or not bool(torch.isfinite(result).all().item()):
            raise DCUPreflightError("remote select_action output must be finite float32 or bfloat16")
        evidence = getattr(self.client, "last_queue_evidence", {})
        if evidence is None:
            evidence = {}
        if not isinstance(evidence, Mapping):
            raise DCUPreflightError("remote client queue evidence must be a mapping")
        self.last_queue_evidence = dict(evidence)
        before = self.last_queue_evidence.get("queue_length_before")
        after = self.last_queue_evidence.get("queue_length_after")
        generated = self.last_queue_evidence.get("new_chunk_generated")
        for name, value in (("queue_length_before", before), ("queue_length_after", after)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise DCUPreflightError(f"remote {name} is invalid")
        if generated is not None and not isinstance(generated, bool):
            raise DCUPreflightError("remote new_chunk_generated must be boolean")
        if generated is True and before not in (None, 0):
            raise DCUPreflightError("remote new_chunk_generated requires an empty queue")
        if generated is False and before is not None and after is not None and after >= before:
            raise DCUPreflightError("remote select_action did not consume a queued action")
        after = self.last_queue_evidence.get("queue_length_after")
        if after is not None:
            if isinstance(after, bool) or not isinstance(after, int) or after < 0:
                raise DCUPreflightError("remote queue_length_after is invalid")
            self._remote_queue.length = int(after)
        if self.trace is not None:
            record = getattr(self.trace, "record", None)
            if callable(record):
                record("remote_action", result)
            update = getattr(self.trace, "update", None)
            if callable(update):
                update(
                    worker_wall_latency_seconds=wall_latency,
                    worker_queue_evidence=dict(self.last_queue_evidence),
                )
                response = getattr(self.client, "last_response", None)
                if response is None:
                    response = getattr(getattr(self.client, "transport", None), "last_response", None)
                if isinstance(response, Mapping):
                    update(worker_response=dict(response))
        return result

    def predict_action_chunk(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise DCUPreflightError("remote closed-loop policy exposes select_action only")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise DCUPreflightError("remote closed-loop policy does not expose forward()")


def validate_runner_config(
    config: Mapping[str, Any],
    *,
    phase: str,
    expected_project_sha: str,
    actual_project_sha: str,
) -> dict[str, Any]:
    """Validate the exact frozen config plus phase-specific restrictions."""

    normalized_phase = _normalise_phase(phase)
    identity = validate_config_identity(
        config,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual_project_sha,
    )
    if normalized_phase == _ONE_STEP_CHILD_PHASE:
        raise DCUPreflightError("one-step-child is internal and requires its token gate")
    concurrency = _require_mapping(config.get("concurrency"), "concurrency")
    workers = _require_exact(concurrency.get("workers"), 2, "concurrency.workers")
    if list(concurrency.get("physical_devices", [])) != list(EXPECTED_PHYSICAL_DEVICES):
        raise DCUPreflightError("concurrency.physical_devices must be exactly [0, 1]")
    if concurrency.get("logical_device") != EXPECTED_LOGICAL_DEVICE:
        raise DCUPreflightError("concurrency.logical_device must be cuda:0 per child")
    _require_exact(
        concurrency.get("steps_per_worker"),
        EXPECTED_CONCURRENCY_STEPS_PER_WORKER,
        "concurrency.steps_per_worker",
    )
    if config.get("mode") != "explicit_phase":
        raise DCUPreflightError("config mode must be the phase-neutral value 'explicit_phase'")
    identity.update(
        {
            "phase": normalized_phase,
            "n_envs": EXPECTED_N_ENVS,
            "obs_type": EXPECTED_OBS_TYPE,
            "workers": int(workers),
            "compare_physical_device": EXPECTED_COMPARE_PHYSICAL_DEVICE,
            "closed_loop_physical_device": EXPECTED_CLOSED_LOOP_PHYSICAL_DEVICE,
        }
    )
    return identity


def _run_command(args: Sequence[str], *, cwd: Path | None = None) -> dict[str, Any]:
    command = [str(item) for item in args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"args": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "args": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.rstrip("\r\n"),
        "stderr": completed.stderr.rstrip("\r\n"),
    }


def _project_head(root: Path) -> str:
    evidence = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    if evidence.get("returncode") != 0 or not evidence.get("stdout"):
        raise DCUPreflightError(f"could not resolve project HEAD: {evidence}")
    return str(evidence["stdout"])


def _dirty_evidence(root: Path, run_directory: Path | None = None) -> dict[str, Any]:
    evidence = _run_command(["git", "status", "--short", "--untracked-files=all"], cwd=root)
    if evidence.get("returncode") != 0:
        raise DCUPreflightError(f"could not inspect project Git status: {evidence}")
    dirty: list[dict[str, Any]] = []
    run_relative: Path | None = None
    if run_directory is not None:
        try:
            run_relative = run_directory.resolve().relative_to(root.resolve())
        except ValueError:
            run_relative = None
    for line in str(evidence.get("stdout", "")).splitlines():
        relative_text = line[3:] if len(line) >= 4 else ""
        relative = Path(relative_text)
        if run_relative is not None and (relative == run_relative or run_relative in relative.parents):
            continue
        item: dict[str, Any] = {"path": relative_text, "status": line[:2]}
        path = root / relative
        if path.is_file() and not path.is_symlink():
            item["bytes"] = path.stat().st_size
            item["sha256"] = _sha256_file(path)
        dirty.append(item)
    unexpected = [
        item
        for item in dirty
        if item.get("path") != "AGENTS.md"
        or item.get("sha256") != EXPECTED_AGENTS_SHA256
    ]
    if unexpected:
        raise DCUPreflightError(f"project worktree has unapproved dirty files: {unexpected}")
    return {
        "head": _project_head(root),
        "dirty_files": dirty,
        "status": evidence,
        "allowed_dirty_paths": list(EXPECTED_ALLOWED_DIRTY_PATHS),
    }


def _verify_exact_file_hash(path: str | Path, expected: str, name: str) -> dict[str, Any]:
    target = Path(path).resolve()
    digest = _sha256_file(target)
    if digest != expected:
        raise DCUPreflightError(f"{name} sha256 mismatch: expected {expected}, got {digest}")
    return {"path": str(target), "sha256": digest, "bytes": target.stat().st_size}


def _verify_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    reference = _require_mapping(config["m0_reference"], "m0_reference")
    run_evidence = _verify_exact_file_hash(
        reference["run_manifest_path"], EXPECTED_REFERENCE_RUN_MANIFEST_SHA256, "M0 reference run manifest"
    )
    decisions_evidence = _verify_exact_file_hash(
        reference["decisions_path"], EXPECTED_REFERENCE_DECISIONS_SHA256, "M0 reference decisions"
    )
    with Path(reference["decisions_path"]).open(encoding="utf-8") as handle:
        first_line = handle.readline()
    if not first_line:
        raise DCUPreflightError("M0 reference decisions are empty")
    try:
        first_decision = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise DCUPreflightError("M0 reference first decision is not JSON") from exc
    if not isinstance(first_decision, Mapping):
        raise DCUPreflightError("M0 reference first decision must be a mapping")
    return {
        "run_manifest": run_evidence,
        "decisions": decisions_evidence,
        "first_decision": dict(first_decision),
    }


def _decode_command_bytes(value: Any, name: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DCUPreflightError(f"{name} output is not valid UTF-8") from exc
    if isinstance(value, str):
        return value
    raise DCUPreflightError(f"{name} output must be bytes or text")


def _verify_live_runtime_lock(lock_path: str | Path, python_path: str | Path) -> dict[str, Any]:
    """Match one configured interpreter's raw freeze/check output to its lock.

    The CPU helper in ``m0_smoke`` performs this check for the parent runtime.
    The DCU interpreter is a separate process, so the same byte-preserving
    contract must be evaluated explicitly with the configured DCU executable.
    """

    from scripts import m0_smoke

    target = Path(lock_path).resolve(strict=True)
    configured_python = Path(python_path).resolve()
    try:
        parsed = m0_smoke._parse_runtime_lock(target)
    except Exception as exc:
        raise DCUPreflightError(f"DCU runtime lock could not be parsed: {exc}") from exc
    locked_python = Path(str(parsed["python_executable"])).resolve()
    if locked_python != configured_python:
        raise DCUPreflightError(
            "DCU runtime lock python executable differs from config: "
            f"locked={locked_python}, configured={configured_python}"
        )

    freeze = subprocess.run(
        [str(configured_python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=False,
        check=False,
    )
    if int(getattr(freeze, "returncode", -1)) != 0:
        raise DCUPreflightError(
            f"DCU pip freeze --all failed with return code {getattr(freeze, 'returncode', None)}"
        )
    freeze_stdout = getattr(freeze, "stdout", b"")
    try:
        freeze_evidence = m0_smoke._validate_runtime_freeze(
            freeze_stdout,
            parsed["pip_freeze_all_lines"],
            parsed["pip_freeze_all_sha256"],
            parsed["pip_freeze_all_bytes"],
        )
    except Exception as exc:
        raise DCUPreflightError(f"DCU pip freeze --all differs from runtime lock: {exc}") from exc

    pip_check = subprocess.run(
        [str(configured_python), "-m", "pip", "check"],
        capture_output=True,
        text=False,
        check=False,
    )
    check_stdout = _decode_command_bytes(getattr(pip_check, "stdout", b""), "DCU pip check stdout")
    check_stderr = _decode_command_bytes(getattr(pip_check, "stderr", b""), "DCU pip check stderr")
    check_lines = check_stdout.splitlines()
    expected_check_lines = list(parsed["pip_check_lines"])
    if int(getattr(pip_check, "returncode", -1)) != 0 or check_lines != expected_check_lines:
        raise DCUPreflightError(
            "DCU pip check is not the exact locked PASS result: "
            f"returncode={getattr(pip_check, 'returncode', None)}, "
            f"output={check_lines!r}, expected={expected_check_lines!r}"
        )

    python_version = subprocess.run(
        [str(configured_python), "--version"],
        capture_output=True,
        text=False,
        check=False,
    )
    if int(getattr(python_version, "returncode", -1)) != 0:
        raise DCUPreflightError(
            "DCU Python version probe failed with return code "
            f"{getattr(python_version, 'returncode', None)}"
        )
    python_stdout = _decode_command_bytes(getattr(python_version, "stdout", b""), "DCU Python version stdout")
    python_stderr = _decode_command_bytes(getattr(python_version, "stderr", b""), "DCU Python version stderr")
    version_text = (python_stdout.strip() or python_stderr.strip())
    if not version_text:
        raise DCUPreflightError("DCU Python version probe returned no version")
    return {
        "path": str(target),
        "lock_sha256": _sha256_file(target),
        "python_executable": {
            "configured": str(configured_python),
            "locked": str(locked_python),
            "version": version_text,
        },
        "pip_freeze_all": freeze_evidence,
        "pip_check": {
            "args": [str(configured_python), "-m", "pip", "check"],
            "returncode": int(pip_check.returncode),
            "stdout": check_stdout,
            "stderr": check_stderr,
            "expected_stdout_lines": expected_check_lines,
            "exact_pass": True,
        },
        "torch_runtime": {
            "probe": "deferred_to_dcu_worker_ping",
            "hip_version": None,
            "cuda_version": None,
        },
    }


def _probe_hardware_toolchain() -> dict[str, Any]:
    """Capture successful read-only host tool probes in the preflight gate."""

    hy_smi = _run_command(["hy-smi"])
    hipcc = _run_command(["hipcc", "--version"])
    for name, evidence in (("hy-smi", hy_smi), ("hipcc", hipcc)):
        if evidence.get("returncode") != 0:
            raise DCUPreflightError(f"{name} probe failed: {evidence}")
    return {"hy_smi": hy_smi, "hipcc": hipcc}


def _verify_overlay_and_patch(config: Mapping[str, Any]) -> dict[str, Any]:
    patch_evidence = validate_patch(PATCH_PATH, source_root=_ROOT / "external" / "lerobot")
    overlay = _require_mapping(config["patch"], "patch")
    overlay_evidence = _verify_exact_file_hash(
        overlay["overlay_wheel_path"], EXPECTED_OVERLAY_WHEEL_SHA256, "overlay wheel"
    )
    return {"patch": patch_evidence, "overlay_wheel": overlay_evidence}


def _verify_artifacts_and_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(config["preflight_manifest"])).resolve()
    manifest_evidence = _verify_exact_file_hash(
        manifest_path, EXPECTED_PREFLIGHT_MANIFEST_SHA256, "preflight manifest"
    )
    checkpoint = _require_mapping(config["checkpoint"], "checkpoint")
    base_model = _require_mapping(config["base_model"], "base_model")
    checkpoint_model = _verify_exact_file_hash(
        Path(str(checkpoint["path"])) / "model.safetensors",
        EXPECTED_CHECKPOINT_MODEL_SHA256,
        "checkpoint model",
    )
    base_model_file = _verify_exact_file_hash(
        Path(str(base_model["path"])) / "model.safetensors",
        EXPECTED_BASE_MODEL_SHA256,
        "base model",
    )
    # The M0 helper performs the complete inventory, size, pointer rejection,
    # and per-file digest check against the archived materialization manifest.
    from scripts import m0_smoke

    inventory = m0_smoke._verify_artifacts(dict(config))
    return {
        "manifest": manifest_evidence,
        "checkpoint_model": checkpoint_model,
        "base_model": base_model_file,
        "inventory": inventory,
    }


def _verify_runtime_and_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _require_mapping(config["runtime"], "runtime")
    cpu_python = Path(str(runtime["cpu_python"])).resolve()
    if Path(sys.executable).resolve() != cpu_python:
        raise DCUPreflightError(
            f"CPU parent interpreter differs from pinned runtime: {sys.executable} != {cpu_python}"
        )
    from scripts import m0_smoke

    lock_evidence = m0_smoke._verify_runtime_lock(Path(str(config["runtime_lock"])).resolve())
    if lock_evidence.get("lock_sha256") != EXPECTED_CPU_LOCK_SHA256:
        raise DCUPreflightError(
            "CPU runtime lock sha256 mismatch: "
            f"expected {EXPECTED_CPU_LOCK_SHA256}, got {lock_evidence.get('lock_sha256')}"
        )
    dcu_lock_evidence = _verify_exact_file_hash(
        runtime["dcu_runtime_lock"], EXPECTED_DCU_LOCK_SHA256, "DCU runtime lock"
    )
    live_dcu_runtime = _verify_live_runtime_lock(
        runtime["dcu_runtime_lock"], runtime["dcu_python"]
    )
    dcu_lock_evidence.update(live_dcu_runtime)
    libero_config = _require_mapping(config["libero_config"], "libero_config")
    libero_config_evidence = _verify_exact_file_hash(
        libero_config["path"],
        EXPECTED_LIBERO_CONFIG_SHA256,
        "LIBERO config.yaml",
    )
    if os.environ.get("LIBERO_CONFIG_PATH") != str(Path(str(config["libero_config"]["path"])).resolve().parent):
        raise DCUPreflightError("LIBERO_CONFIG_PATH must identify the pinned LIBERO config directory")
    offline = dict(config["offline"])
    offline_values: dict[str, str] = {}
    for key, expected in offline.items():
        if os.environ.get(key) != expected:
            raise DCUPreflightError(f"{key} must equal {expected!r} for offline execution")
        offline_values[key] = str(expected)
    renderer = dict(config["renderer"])
    renderer_values: dict[str, str] = {}
    for key, expected in renderer.items():
        if os.environ.get(key) != expected:
            raise DCUPreflightError(f"{key} must equal {expected!r} for EGL execution")
        renderer_values[key] = str(expected)
    from scripts import m0_smoke as smoke

    offline_evidence = smoke._assert_offline_and_hf_home()
    toolchain = _probe_hardware_toolchain()
    return {
        "cpu_runtime_lock": lock_evidence,
        "dcu_runtime_lock": dcu_lock_evidence,
        "toolchain": toolchain,
        "libero_config": libero_config_evidence,
        "offline": {"variables": offline_values, **offline_evidence},
        "renderer": {"variables": renderer_values, "exact": True},
    }


def _verify_assets_binding(config: Mapping[str, Any], *, create: bool) -> dict[str, Any]:
    from scripts import m0_smoke

    target = Path(str(config["assets"]["path"])).resolve(strict=True)
    if not target.is_dir():
        raise DCUPreflightError(f"assets path is not a directory: {target}")
    if create:
        return m0_smoke._safe_asset_symlink(_ROOT, dict(config))
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    link = purelib / "libero" / "libero" / "assets"
    if not link.is_symlink() or link.resolve(strict=True) != target:
        raise DCUPreflightError(f"asset symlink does not point to the pinned tree: {link}")
    return {"link": str(link), "target": str(target), "resolved": str(link.resolve()), "is_symlink": True}


def run_preflight_gates(
    *,
    config: Mapping[str, Any],
    phase: str,
    expected_project_sha: str,
    actual_project_sha: str,
    run_directory: str | Path,
    prepare_assets: bool = False,
) -> dict[str, Any]:
    """Run all read-only/fail-closed gates before importing real runtimes."""

    normalized_phase = _normalise_phase(phase)
    identity = validate_runner_config(
        config,
        phase=normalized_phase,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual_project_sha,
    )
    run_dir = Path(run_directory).resolve()
    root_git = _dirty_evidence(_ROOT, run_dir)
    from scripts import m0_smoke

    source_checkouts = m0_smoke._verify_source_checkouts(
        _ROOT, Path(str(config["preflight_manifest"])).resolve()
    )
    environment = _verify_runtime_and_environment(config)
    artifacts = _verify_artifacts_and_manifest(config)
    reference = _verify_reference(config)
    patch = _verify_overlay_and_patch(config)
    assets = _verify_assets_binding(config, create=prepare_assets)
    return {
        "identity": identity,
        "project": root_git,
        "source_checkouts": source_checkouts,
        "environment": environment,
        "artifacts": artifacts,
        "reference": {
            "run_manifest": reference["run_manifest"],
            "decisions": reference["decisions"],
            "first_decision": reference["first_decision"],
        },
        "patch": patch,
        "assets": assets,
        "network": {"offline": True, "fallback": False},
        "scientific_scope": {
            "training": False,
            "backward": False,
            "lora": False,
            "hooks": False,
            "replay": False,
            "perturbation": False,
            "plus": False,
            "matched_sampling": False,
            "amp": False,
            "precision_changes": False,
            "fused_backend": False,
        },
    }


def _feature_schema() -> Mapping[str, Any]:
    from scripts.dcu_model_worker import FEATURE_SCHEMA

    return FEATURE_SCHEMA


def _feature_keys() -> tuple[str, ...]:
    return tuple(sorted(_feature_schema()))


def _describe_runtime_value(value: Any) -> Any:
    try:
        from scripts.m0_smoke import describe

        return describe(value)
    except Exception:
        if isinstance(value, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True,
            }
        return {"type": type(value).__name__}


def _feature_bundle(processed: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(processed, Mapping):
        raise DCUPreflightError("policy preprocessor output must be a mapping")
    schema = _feature_schema()
    missing = sorted(set(schema) - set(processed))
    if missing:
        raise DCUPreflightError(f"policy preprocessor output is missing feature keys: {missing}")
    bundle = {key: processed[key] for key in schema}
    from scripts.dcu_worker import validate_tensor_bundle

    validate_tensor_bundle(bundle, schema=schema)
    return bundle


def _feature_metadata(bundle: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    from scripts.dcu_worker import validate_tensor_bundle

    validated = validate_tensor_bundle(bundle, schema=_feature_schema())
    tensors = {
        key: dict(value) for key, value in validated["tensors"].items()
    }
    for key, tensor in bundle.items():
        item = tensors[key]
        item["type"] = "torch.Tensor"
        if tensor.numel() and (tensor.is_floating_point() or tensor.is_complex()):
            item["min"] = float(tensor.detach().amin().item())
            item["max"] = float(tensor.detach().amax().item())
    return {
        "keys": validated["keys"],
        "tensors": tensors,
        "finite": validated["finite"],
    }


def _reference_feature_metadata(reference: Mapping[str, Any]) -> dict[str, Any]:
    first = _require_mapping(reference.get("first_decision"), "reference.first_decision")
    metadata = _require_mapping(first.get("observation_policy_processor"), "reference.observation_policy_processor")
    expected: dict[str, Any] = {}
    for key in _feature_keys():
        item = _require_mapping(metadata.get(key), f"reference.observation_policy_processor.{key}")
        expected[key] = {
            "shape": list(item.get("shape", [])),
            "dtype": item.get("dtype"),
            "type": item.get("type"),
        }
        for field in ("min", "max"):
            value = item.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                expected[key][field] = float(value)
    return expected


def _assert_reference_feature_metadata(
    bundle: Mapping[str, torch.Tensor], reference: Mapping[str, Any]
) -> dict[str, Any]:
    actual = _feature_metadata(bundle)
    expected = _reference_feature_metadata(reference)
    mismatches: list[dict[str, Any]] = []
    for key in _feature_keys():
        observed = actual["tensors"][key]
        want = expected[key]
        for field in ("shape", "dtype"):
            if observed[field] != want[field]:
                mismatches.append({"key": key, "field": field, "expected": want[field], "actual": observed[field]})
        for field in ("min", "max"):
            if field not in want:
                continue
            actual_value = observed.get(field)
            if not isinstance(actual_value, (int, float)) or not math.isclose(
                float(actual_value), float(want[field]), rel_tol=1e-7, abs_tol=1e-7
            ):
                mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "expected": want[field],
                        "actual": actual_value,
                        "tolerance": {"rel": 1e-7, "abs": 1e-7},
                    }
                )
        if want.get("type") and observed.get("type", "torch.Tensor") != want["type"]:
            # validate_tensor_bundle intentionally returns compact metadata;
            # the type is fixed by the wire contract and checked explicitly.
            if want["type"] != "torch.Tensor":
                mismatches.append({"key": key, "field": "type", "expected": want["type"], "actual": "torch.Tensor"})
    if mismatches:
        raise DCUPreflightError(f"feature metadata differs from archived M0 first decision: {mismatches}")
    return {
        "actual": actual,
        "reference": expected,
        "matched": True,
        "same_initial_observation": "reconstructed from task/seed/init_state; not a historic tensor bitwise hash",
        "floating_min_max_tolerance": {"relative": 1e-7, "absolute": 1e-7},
    }


def _add_task_to_observation(observation: Any, env: Any) -> Any:
    if not isinstance(observation, Mapping):
        raise DCUPreflightError("environment reset observation must be a mapping")
    result = dict(observation)
    task: Any = None
    call = getattr(env, "call", None)
    if callable(call):
        for name in ("task_description", "task"):
            try:
                values = call(name)
                task = list(values)
                break
            except (AttributeError, NotImplementedError, KeyError):
                continue
    if task is None:
        task = [""]
    result["task"] = task
    return result


def _reset_vector_env(env: Any, seed: int) -> tuple[Any, Any]:
    options = {"is_new_rollout": True}
    try:
        from lerobot.scripts.lerobot_eval import NEW_ROLLOUT_OPTION

        options = {NEW_ROLLOUT_OPTION: True}
    except Exception:
        pass
    try:
        result = env.reset(seed=[seed], options=options)
    except TypeError:
        result = env.reset(seed=[seed])
    if not isinstance(result, tuple) or len(result) != 2:
        raise DCUPreflightError("vector environment reset must return (observation, info)")
    return result


def _render_once(env: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    call = getattr(env, "call", None)
    if not callable(call):
        raise DCUPreflightError("vector environment must provide call('render')")
    frames = call("render")
    elapsed = time.perf_counter() - started
    frame = frames[0] if isinstance(frames, (list, tuple)) else frames
    if isinstance(frame, torch.Tensor):
        finite = bool(torch.isfinite(frame).all().item()) if frame.is_floating_point() else True
        shape = tuple(frame.shape)
        dtype = str(frame.dtype)
    else:
        frame_array = np.asarray(frame)
        finite = bool(np.isfinite(frame_array).all()) if np.issubdtype(frame_array.dtype, np.number) else True
        shape = tuple(frame_array.shape)
        dtype = str(frame_array.dtype)
    if not finite or shape != EXPECTED_OBSERVATION_SHAPE + (3,):
        raise DCUPreflightError(f"render output must be finite 360x360x3, got shape={shape}, dtype={dtype}")
    return frame, elapsed


def _gl_identity_after_render() -> dict[str, Any]:
    """Use the M0 GL probe and identity gate after a real render call."""

    from scripts import m0_smoke

    identity = m0_smoke._gl_evidence()
    return m0_smoke._assert_gl_identity(identity)


def _assert_exact_llvmpipe(identity: Any) -> dict[str, Any]:
    """Require the renderer identity already validated for the frozen M0 run."""

    if not isinstance(identity, Mapping):
        raise DCUPreflightError(f"GL identity unavailable: {identity!r}")
    mismatches = {
        key: {"expected": expected, "actual": identity.get(key)}
        for key, expected in EXPECTED_GL_IDENTITY.items()
        if identity.get(key) != expected
    }
    if mismatches:
        raise DCUPreflightError(f"GL identity is not the validated exact llvmpipe renderer: {mismatches}")
    return {**dict(identity), "exact_validated": True}


def _egl_probe_evidence(*, selected_device: str = EXPECTED_EGL_DEVICE_ID) -> dict[str, Any]:
    """Enumerate EGL devices independently of CUDA/HIP visibility.

    The package's C++ probes are intentionally used instead of a CUDA-derived
    ordinal.  ``query_devices`` emits one strict integer count and
    ``test_device`` validates the selected ordinal in that same EGL namespace.
    """

    selected = _require_string(selected_device, "MUJOCO_EGL_DEVICE_ID")
    if not selected.isdigit():
        raise DCUPreflightError("MUJOCO_EGL_DEVICE_ID must be a decimal EGL ordinal")
    selected_index = int(selected)
    try:
        import egl_probe

        package_file = Path(str(egl_probe.__file__)).resolve()
    except Exception as exc:
        raise DCUPreflightError(f"could not import pinned egl_probe: {exc}") from exc
    build = package_file.parent / "build"
    query = build / "query_devices"
    test = build / "test_device"
    for binary, name in ((query, "eglQueryDevicesEXT query_devices"), (test, "egl test_device")):
        if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
            raise DCUPreflightError(f"{name} binary is unavailable: {binary}")
    queried = subprocess.run(
        [str(query)],
        capture_output=True,
        text=False,
        check=False,
    )
    query_stdout = _decode_command_bytes(getattr(queried, "stdout", b""), "egl query stdout")
    query_stderr = _decode_command_bytes(getattr(queried, "stderr", b""), "egl query stderr")
    if int(getattr(queried, "returncode", -1)) != 0:
        raise DCUPreflightError(
            f"eglQueryDevicesEXT query_devices failed: returncode={getattr(queried, 'returncode', None)}, "
            f"stdout={query_stdout!r}, stderr={query_stderr!r}"
        )
    count_match = re.fullmatch(r"\s*(\d+)\s*", query_stdout)
    if count_match is None:
        raise DCUPreflightError(f"eglQueryDevicesEXT count is not a strict integer: {query_stdout!r}")
    count = int(count_match.group(1))
    if selected_index >= count:
        raise DCUPreflightError(
            f"selected EGL device {selected_index} is outside eglQueryDevicesEXT count {count}"
        )
    tested = subprocess.run(
        [str(test), selected],
        capture_output=True,
        text=False,
        check=False,
    )
    test_stdout = _decode_command_bytes(getattr(tested, "stdout", b""), "egl test stdout")
    test_stderr = _decode_command_bytes(getattr(tested, "stderr", b""), "egl test stderr")
    if int(getattr(tested, "returncode", -1)) != 0:
        raise DCUPreflightError(
            f"EGL device {selected_index} validation failed: returncode={getattr(tested, 'returncode', None)}, "
            f"stdout={test_stdout!r}, stderr={test_stderr!r}"
        )
    return {
        "probe_package": str(package_file),
        "query_devices": {
            "path": str(query),
            "returncode": int(queried.returncode),
            "stdout": query_stdout,
            "stderr": query_stderr,
        },
        "eglQueryDevicesEXT_device_count": count,
        "egl_device_count": count,
        "selected_MUJOCO_EGL_DEVICE_ID": selected,
        "selected_egl_device_id": selected,
        "test_device": {
            "path": str(test),
            "ordinal": selected_index,
            "returncode": int(tested.returncode),
            "stdout": test_stdout,
            "stderr": test_stderr,
        },
        "namespace": "EGL independent of CUDA_VISIBLE_DEVICES",
    }


def _task_source_evidence(libero: Any, task: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Resolve and validate the same pinned task/BDDL/init seams as M0."""

    try:
        from libero.libero import benchmark, get_libero_path

        suite = benchmark.get_benchmark_dict()[str(task["suite"])]()
        task_spec = suite.get_task(int(task["task_id"]))
        bddl = Path(get_libero_path("bddl_files")) / task_spec.problem_folder / task_spec.bddl_file
        init_state = Path(get_libero_path("init_states")) / task_spec.problem_folder / task_spec.init_states_file
    except Exception as exc:
        raise DCUPreflightError(f"could not resolve pinned LIBERO task sources: {exc}") from exc
    expected_bddl_root = root / "external" / "hf-libero" / "libero" / "libero" / "bddl_files"
    expected_init_root = root / "external" / "hf-libero" / "libero" / "libero" / "init_files"
    if not bddl.is_file() or not init_state.is_file():
        raise DCUPreflightError("pinned LIBERO BDDL/init state is missing")
    try:
        bddl_resolved = bddl.resolve(strict=True)
        init_resolved = init_state.resolve(strict=True)
        bddl_resolved.relative_to(expected_bddl_root.resolve())
        init_resolved.relative_to(expected_init_root.resolve())
    except ValueError as exc:
        raise DCUPreflightError("LIBERO BDDL/init state escaped pinned hf-libero tree") from exc
    except FileNotFoundError as exc:
        raise DCUPreflightError("LIBERO BDDL/init state is not a regular file") from exc
    description = getattr(task_spec, "language", getattr(task_spec, "description", None))
    if not isinstance(description, str) or not description.strip():
        raise DCUPreflightError("LIBERO task description is missing")
    task_name = getattr(task_spec, "name", None)
    if not isinstance(task_name, str) or not task_name.strip():
        raise DCUPreflightError("LIBERO task name is missing")
    del libero
    return {
        "suite": str(task["suite"]),
        "task_id": int(task["task_id"]),
        "init_state_id": int(task["init_state_id"]),
        "task_name": task_name,
        "task_description": description,
        "bddl_path": str(bddl_resolved),
        "bddl_file": bddl_resolved.name,
        "init_state_path": str(init_resolved),
        "init_state_file": init_resolved.name,
        "horizon": EXPECTED_HORIZON,
        "source_check": "pinned_hf_libero_paths",
    }


def _init_state_evidence(env: Any) -> dict[str, Any]:
    get_attr = getattr(env, "get_attr", None)
    if not callable(get_attr):
        raise DCUPreflightError("CPU environment must expose get_attr('init_state_id')")
    try:
        values = get_attr("init_state_id")
    except Exception as exc:
        raise DCUPreflightError(f"could not read environment init_state_id: {exc}") from exc
    try:
        array = np.asarray(values).reshape(-1)
        actual_value = int(array[0]) if array.size else None
    except (TypeError, ValueError, IndexError) as exc:
        raise DCUPreflightError(f"environment init_state_id is invalid: {values!r}") from exc
    if array.size != EXPECTED_N_ENVS or actual_value != EXPECTED_INIT_STATE_ID:
        raise DCUPreflightError(
            f"environment init_state_id must be exactly {EXPECTED_INIT_STATE_ID}, got {values!r}"
        )
    return {
        "attribute": "init_state_id",
        "value": actual_value,
        "expected": EXPECTED_INIT_STATE_ID,
        "n_envs": int(array.size),
    }


def _assert_environment_action_space(env: Any, action_dim: int) -> dict[str, Any]:
    """Validate the synchronous environment action boundary without M0 imports."""

    action_space = getattr(env, "action_space", None)
    single_action_space = getattr(env, "single_action_space", None)
    if action_space is None or single_action_space is None:
        raise DCUPreflightError("vector environment must expose action_space and single_action_space")
    expected_batch_shape = (int(getattr(env, "num_envs", -1)), int(action_dim))
    if tuple(action_space.shape) != expected_batch_shape or action_space.dtype != np.dtype(np.float32):
        raise DCUPreflightError(
            f"vector action_space must be {expected_batch_shape} float32, "
            f"got shape={action_space.shape}, dtype={action_space.dtype}"
        )
    if tuple(single_action_space.shape) != (int(action_dim),) or single_action_space.dtype != np.dtype(np.float32):
        raise DCUPreflightError(
            f"single_action_space must be ({int(action_dim)},) float32, "
            f"got shape={single_action_space.shape}, dtype={single_action_space.dtype}"
        )
    return {
        "batch_action_shape": list(expected_batch_shape),
        "batch_action_dtype": "float32",
        "single_action_shape": list(single_action_space.shape),
        "single_action_dtype": str(single_action_space.dtype),
    }


def _loaded_policy_module_names() -> tuple[str, ...]:
    """Return policy-bearing modules already visible in this process."""

    names = []
    for name in sys.modules:
        lowered = str(name).lower()
        if "smolvla" in lowered or lowered == "lerobot.policies" or lowered.startswith("lerobot.policies."):
            names.append(str(name))
    return tuple(sorted(names))


def build_cpu_environment_runtime(
    config: Mapping[str, Any],
    *,
    phase: str = "compare",
) -> dict[str, Any]:
    """Construct only the pinned CPU LIBERO environment.

    This is the policy-free M1 boundary.  It deliberately does not inspect
    checkpoint/base-model config and does not import policy classes or
    LeRobot processor factories.  Policy-bearing callers use
    :func:`build_cpu_runtime` below, while exact-state replay uses this seam.
    """

    normalized_phase = _normalise_phase(phase)
    if normalized_phase == _ONE_STEP_CHILD_PHASE:
        normalized_phase = "concurrency"
    root = _ROOT
    task = _require_mapping(config["task"], "task")
    policy_modules_before = set(_loaded_policy_module_names())
    if policy_modules_before:
        raise DCUPreflightError(
            "environment-only M1 process already contains policy-bearing modules: "
            + ", ".join(sorted(policy_modules_before))
        )
    try:
        import lerobot
        import libero
        import mujoco
        import robosuite
        from lerobot.envs import (
            close_envs,
            make_env,
            make_env_config,
        )
    except Exception as exc:
        raise DCUPreflightError(f"pinned CPU runtime imports failed: {exc}") from exc
    policy_modules_after = set(_loaded_policy_module_names())
    newly_imported_policy_modules = sorted(policy_modules_after - policy_modules_before)
    if newly_imported_policy_modules:
        raise DCUPreflightError(
            "environment-only runtime imported policy-bearing modules: "
            + ", ".join(newly_imported_policy_modules)
        )
    # Keep source/module path validation local to the environment-only
    # boundary.  Importing the M0 smoke module here would pull a policy
    # bearing module into the M1 process even though no policy is needed.
    imported: dict[str, Any] = {"modules": {}}
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    forbidden_roots = [
        (root / name).resolve()
        for name in ("external/lerobot", "external/hf-libero", "external/libero", "external/robosuite", "external/mujoco")
    ]
    for module_name, module in (("lerobot", lerobot), ("libero", libero), ("robosuite", robosuite), ("mujoco", mujoco)):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise DCUPreflightError(f"runtime module has no __file__: {module_name}")
        module_path = Path(module_file).resolve()
        if not module_path.is_relative_to(purelib) or any(module_path.is_relative_to(forbidden) for forbidden in forbidden_roots):
            raise DCUPreflightError(f"runtime module escaped isolated site-packages: {module_name} -> {module_path}")
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        imported["modules"][module_name] = {
            "__file__": str(module_path),
            "__spec__.origin": str(Path(origin).resolve()) if origin and origin != "built-in" else origin,
        }
    imported["purelib"] = str(purelib)
    imported["policy_import_audit"] = {
        "before": sorted(policy_modules_before),
        "after": sorted(policy_modules_after),
        "new": newly_imported_policy_modules,
    }
    task_source = _task_source_evidence(libero, task, root)
    env_cfg = make_env_config(
        "libero",
        task=str(task["suite"]),
        task_ids=[int(task["task_id"])],
        obs_type=str(config["obs_type"]),
        render_mode="rgb_array",
        init_states=True,
        hard_reset=True,
        observation_height=EXPECTED_OBSERVATION_SHAPE[0],
        observation_width=EXPECTED_OBSERVATION_SHAPE[1],
    )
    envs = make_env(
        env_cfg,
        n_envs=EXPECTED_N_ENVS,
        use_async_envs=False,
        trust_remote_code=False,
    )
    env = envs[str(task["suite"])][int(task["task_id"])]
    if int(getattr(env, "num_envs", -1)) != EXPECTED_N_ENVS:
        raise DCUPreflightError("CPU runtime must expose exactly one synchronous environment")
    action_space = _assert_environment_action_space(env, EXPECTED_ACTION_DIM)
    max_steps = int(env.call("_max_episode_steps")[0])
    if max_steps != EXPECTED_HORIZON:
        raise DCUPreflightError(f"LIBERO horizon must be {EXPECTED_HORIZON}, got {max_steps}")
    init_state = _init_state_evidence(env)
    task_evidence = {
        **task_source,
        "horizon": max_steps,
        "init_state_id_evidence": init_state,
    }

    return {
        "lerobot": lerobot,
        "libero": libero,
        "mujoco": mujoco,
        "robosuite": robosuite,
        "env_cfg": env_cfg,
        "envs": envs,
        "env": env,
        "close_envs": close_envs,
        "action_space": action_space,
        "max_steps": max_steps,
        "task": task_evidence,
        "init_state_id_evidence": init_state,
        "runtime_imports": imported,
        "phase": normalized_phase,
    }


def build_cpu_runtime(
    config: Mapping[str, Any],
    *,
    include_policy: bool = True,
    phase: str = "compare",
) -> dict[str, Any]:
    """Construct the pinned vanilla LIBERO env and optional policy stack."""

    normalized_phase = _normalise_phase(phase)
    if normalized_phase == _ONE_STEP_CHILD_PHASE:
        normalized_phase = "concurrency"
    checkpoint = _require_mapping(config["checkpoint"], "checkpoint")
    base_model = _require_mapping(config["base_model"], "base_model")
    environment_runtime = build_cpu_environment_runtime(config, phase=normalized_phase)
    env_cfg = environment_runtime["env_cfg"]
    lerobot = environment_runtime["lerobot"]
    env = environment_runtime["env"]
    try:
        from lerobot.envs import make_env_pre_post_processors, preprocess_observation
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except Exception as exc:
        raise DCUPreflightError(f"pinned CPU policy imports failed: {exc}") from exc

    policy_cfg = SmolVLAConfig.from_pretrained(
        str(Path(str(checkpoint["path"])).resolve()),
        revision=str(checkpoint["revision"]),
        local_files_only=True,
    )
    policy_cfg.device = "cpu"
    policy_cfg.vlm_model_name = str(Path(str(base_model["path"])).resolve())
    policy_cfg.pretrained_path = Path(str(checkpoint["path"])).resolve()
    policy_cfg.pretrained_revision = str(checkpoint["revision"])
    policy_cfg.base_model_revision = str(base_model["revision"])
    policy_cfg.use_peft = False
    policy_cfg.use_amp = False
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False
    if int(getattr(policy_cfg, "chunk_size", -1)) != EXPECTED_CHUNK_SIZE:
        raise DCUPreflightError("checkpoint chunk_size differs from the frozen contract")
    if int(getattr(policy_cfg, "n_action_steps", -1)) != EXPECTED_N_ACTION_STEPS:
        raise DCUPreflightError("checkpoint n_action_steps differs from the frozen contract")
    policy = None
    model_evidence: dict[str, Any] | None = None
    if include_policy:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=None)
        SmolVLAPolicy._load_as_safetensor(
            policy,
            str(Path(str(checkpoint["path"])) / "model.safetensors"),
            "cpu",
            strict=True,
        )
        policy.eval()
        model_evidence = {
            "policy_type": type(policy).__name__,
            "training": bool(getattr(policy, "training", False)),
            "device": str(next(policy.parameters()).device),
            "checkpoint_path": str(Path(str(checkpoint["path"])).resolve()),
            "base_model_path": str(Path(str(base_model["path"])).resolve()),
            "strict_checkpoint_load": True,
            "use_peft": bool(getattr(policy_cfg, "use_peft", False)),
            "use_amp": bool(getattr(policy_cfg, "use_amp", False)),
            "compile_model": bool(getattr(policy_cfg, "compile_model", False)),
            "chunk_size": EXPECTED_CHUNK_SIZE,
            "n_action_steps": EXPECTED_N_ACTION_STEPS,
        }
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(Path(str(checkpoint["path"])).resolve()),
        pretrained_revision=str(checkpoint["revision"]),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(Path(str(base_model["path"])).resolve())},
            "device_processor": {"device": "cpu"},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return {
        **environment_runtime,
        "policy_cfg": policy_cfg,
        "policy": policy,
        "env_preprocessor": env_preprocessor,
        "env_postprocessor": env_postprocessor,
        "preprocessor": preprocessor,
        "postprocessor": postprocessor,
        "preprocess_observation": preprocess_observation,
        "model": model_evidence,
        "phase": normalized_phase,
    }


# ---------------------------------------------------------------------------
# Worker/phase execution seams
# ---------------------------------------------------------------------------


class _RecordingTransport:
    """Record JSON responses while retaining the worker transport contract."""

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.responses: list[dict[str, Any]] = []
        self.last_response: dict[str, Any] | None = None

    def request(self, command: str, **payload: object) -> Mapping[str, Any]:
        response = self.transport.request(command, **payload)
        if not isinstance(response, Mapping):
            raise DCUPreflightError("worker response must be a mapping")
        normalized = {str(key): value for key, value in response.items()}
        self.responses.append(normalized)
        self.last_response = normalized
        return normalized

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()

    @property
    def stderr_text(self) -> str:
        value = getattr(self.transport, "stderr_text", "")
        return str(value)


def _timeout_seconds(runtime: Mapping[str, Any], kind: str) -> float:
    """Return one finite timeout from the explicit worker timeout contract."""

    names = {
        "startup": ("worker_startup_timeout_seconds", "worker_timeout_seconds"),
        "forward": ("worker_forward_timeout_seconds", "worker_timeout_seconds"),
        "shutdown": ("worker_shutdown_timeout_seconds", "worker_timeout_seconds"),
    }
    if kind not in names:
        raise DCUPreflightError(f"unknown worker timeout kind: {kind}")
    value: Any = None
    for name in names[kind]:
        if name in runtime:
            value = runtime[name]
            break
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DCUPreflightError(f"runtime {kind} timeout is not numeric")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise DCUPreflightError(f"runtime {kind} timeout must be finite and positive")
    return timeout


def _bounded_call(call: Callable[[], Any], *, timeout: float, name: str) -> Any:
    """Bound a blocking worker request without changing the worker protocol."""

    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise DCUPreflightError(f"{name} timeout must be finite and positive")
    finished = threading.Event()
    result: list[Any] = []
    error: list[BaseException] = []

    def invoke() -> None:
        try:
            result.append(call())
        except BaseException as exc:  # propagate the original protocol error
            error.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=invoke, name=f"shiftvla-{name}", daemon=True)
    thread.start()
    if not finished.wait(float(timeout)):
        raise DCUPreflightError(f"worker {name} timed out after {float(timeout):.3f}s")
    if error:
        raise error[0]
    if not result:  # pragma: no cover - defensive against a broken callable
        raise DCUPreflightError(f"worker {name} returned no result")
    return result[0]


def _config_file_path(config: Mapping[str, Any], config_path: str | Path | None) -> Path:
    if config_path is not None:
        target = Path(config_path).resolve()
    else:
        target = (_ROOT / "configs" / "m0" / "dcu_preflight.yaml").resolve()
    if not target.is_file() or target.is_symlink():
        raise DCUPreflightError(f"worker config must be a regular file: {target}")
    return target


def _worker_environment(config: Mapping[str, Any], physical_device: int) -> dict[str, str]:
    runtime = _require_mapping(config["runtime"], "runtime")
    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in _require_mapping(config["offline"], "offline").items()})
    environment.update({str(k): str(v) for k, v in _require_mapping(config["renderer"], "renderer").items()})
    libero_config = _require_mapping(config["libero_config"], "libero_config")
    environment["LIBERO_CONFIG_PATH"] = str(Path(str(libero_config["path"])).resolve().parent)
    environment["PYTHONUNBUFFERED"] = "1"
    # Ignore any parent visibility: this boundary is the source of truth for
    # the nested worker's physical-to-logical mapping.
    environment.pop("HIP_VISIBLE_DEVICES", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    del runtime
    return build_worker_environment(physical_device, environment)


def build_cpu_child_environment(
    config: Mapping[str, Any],
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the CPU LIBERO child's environment without compute visibility.

    CUDA/HIP visibility belongs to the nested model-worker process, not to the
    CPU process which imports robosuite and creates the EGL context.  In
    particular, robosuite's EGL assertion treats a non-empty
    ``CUDA_VISIBLE_DEVICES`` as an EGL ordinal list; keeping that variable out
    of this boundary preserves the validated independent ordinal 8 renderer.
    """

    _require_mapping(config, "config")
    environment = _copy_environment(os.environ.copy() if base_environment is None else base_environment)
    offline = _require_mapping(config["offline"], "offline")
    renderer = _require_mapping(config["renderer"], "renderer")
    libero_config = _require_mapping(config["libero_config"], "libero_config")
    environment.update({str(k): str(v) for k, v in offline.items()})
    environment.update({str(k): str(v) for k, v in renderer.items()})
    environment["LIBERO_CONFIG_PATH"] = str(Path(str(libero_config["path"])).resolve().parent)
    environment["PYTHONUNBUFFERED"] = "1"
    # These are intentionally absent, rather than set to an arbitrary
    # logical/physical index.  The nested worker gets its own mapping.
    environment.pop("HIP_VISIBLE_DEVICES", None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if environment.get("MUJOCO_EGL_DEVICE_ID") != EXPECTED_EGL_DEVICE_ID:
        raise DCUPreflightError(
            "CPU child renderer must select the validated independent EGL ordinal "
            f"{EXPECTED_EGL_DEVICE_ID!r}"
        )
    return environment


def _start_worker(
    config: Mapping[str, Any],
    *,
    run_directory: str | Path,
    config_path: str | Path | None,
    physical_device: int,
) -> dict[str, Any]:
    """Start and ping one isolated DCU worker with a bounded startup wait."""

    normalized_device = _require_nonnegative_int(physical_device, "physical_device")
    if normalized_device not in EXPECTED_PHYSICAL_DEVICES:
        raise DCUPreflightError("physical_device must be one of the pinned [0, 1] devices")
    runtime = _require_mapping(config["runtime"], "runtime")
    dcu_python = _require_string(runtime["dcu_python"], "runtime.dcu_python")
    config_file = _config_file_path(config, config_path)
    ipc = Path(run_directory).resolve() / "ipc"
    ipc.mkdir(parents=True, exist_ok=True)
    argv = [
        dcu_python,
        str((_ROOT / "scripts" / "dcu_model_worker.py").resolve()),
        "--config",
        str(config_file),
        "--seed",
        str(EXPECTED_SEED),
        "--response-dir",
        str(ipc),
    ]
    from scripts.dcu_worker import DCUWorkerClient, SubprocessWorkerTransport

    worker_environment = _worker_environment(config, normalized_device)
    transport = SubprocessWorkerTransport(
        argv,
        env=worker_environment,
        cwd=_ROOT,
        shutdown_timeout=_timeout_seconds(runtime, "shutdown"),
    )
    recorder = _RecordingTransport(transport)
    client = DCUWorkerClient(
        transport=recorder,
        action_dim=EXPECTED_ACTION_DIM,
        chunk_size=EXPECTED_CHUNK_SIZE,
        n_action_steps=EXPECTED_N_ACTION_STEPS,
    )
    try:
        ping = _bounded_call(
            lambda: recorder.request("ping"),
            timeout=_timeout_seconds(runtime, "startup"),
            name="startup",
        )
        if not isinstance(ping, Mapping) or ping.get("ok") is not True:
            raise DCUPreflightError(f"worker startup ping failed: {ping!r}")
    except BaseException:
        _force_terminate_transport(recorder)
        raise
    return {
        "client": client,
        "transport": recorder,
        "ping": dict(ping),
        "argv": argv,
        "physical_device": normalized_device,
        "logical_device": EXPECTED_LOGICAL_DEVICE,
        "compute_environment": {
            "physical_k100": normalized_device,
            "HIP_VISIBLE_DEVICES": worker_environment.get("HIP_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": worker_environment.get("CUDA_VISIBLE_DEVICES"),
            "torch_logical_device": EXPECTED_LOGICAL_DEVICE,
        },
        "ipc": ipc,
        "runtime": runtime,
    }


def _close_worker(worker: Mapping[str, Any] | None) -> None:
    if not worker:
        return
    client = worker.get("client")
    close = getattr(client, "close", None)
    if callable(close):
        try:
            runtime = _require_mapping(worker.get("runtime", {}), "worker.runtime")
            _bounded_call(
                close,
                timeout=_timeout_seconds(runtime, "shutdown"),
                name="shutdown",
            )
        except Exception:
            # The original phase exception is more useful than a best-effort
            # cleanup failure.  The transport itself force-cleans on errors.
            _force_terminate_transport(worker.get("transport"))


def _force_terminate_transport(transport: Any) -> None:
    """Terminate a transport without taking its request lock after a timeout."""

    base = getattr(transport, "transport", transport)
    process = getattr(base, "process", None)
    if process is not None:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=1.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except Exception:
                pass
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None) if process is not None else None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


def _worker_request(
    worker: Mapping[str, Any],
    method_name: str,
    *args: Any,
    timeout_kind: str = "forward",
    **kwargs: Any,
) -> Any:
    client = worker.get("client")
    method = getattr(client, method_name, None)
    if not callable(method):
        raise DCUPreflightError(f"worker client does not provide {method_name}()")
    runtime = _require_mapping(worker["runtime"], "worker.runtime")
    return _bounded_call(
        lambda: method(*args, **kwargs),
        timeout=_timeout_seconds(runtime, timeout_kind),
        name=method_name,
    )


def _worker_summary(worker: Mapping[str, Any] | None) -> dict[str, Any]:
    if not worker:
        return {}
    transport = worker.get("transport")
    responses = getattr(transport, "responses", [])
    ping = worker.get("ping", {})
    ping_mapping = ping if isinstance(ping, Mapping) else {}
    physical = worker.get("physical_device")
    compute_environment = worker.get("compute_environment")
    if not isinstance(compute_environment, Mapping):
        compute_environment = {
            "physical_k100": physical,
            "HIP_VISIBLE_DEVICES": str(physical) if _is_int(physical) else None,
            "CUDA_VISIBLE_DEVICES": "0",
            "torch_logical_device": worker.get("logical_device", EXPECTED_LOGICAL_DEVICE),
        }
    torch_device = ping_mapping.get("device", worker.get("logical_device"))
    torch_device_name = ping_mapping.get("device_name")
    model = ping_mapping.get("model")
    return {
        "argv": list(worker.get("argv", [])),
        "physical_device": physical,
        "logical_device": worker.get("logical_device"),
        "compute_environment": _summary_value(compute_environment),
        "torch_logical_device": torch_device,
        "torch_device_name": torch_device_name,
        "model_load_success": bool(ping_mapping.get("ok") is True and isinstance(model, Mapping)),
        "model": _summary_value(model),
        "startup_ping": _summary_value(ping_mapping),
        "torch_runtime": {
            "hip_version": ping_mapping.get("torch_hip_version"),
            "cuda_version": ping_mapping.get("torch_cuda_version"),
            "cuda_available": ping_mapping.get("torch_cuda_available"),
            "device": torch_device,
            "device_name": torch_device_name,
            "source": "dcu_worker_ping",
        },
        "responses": _summary_value(list(responses)),
        "stderr": str(getattr(transport, "stderr_text", "")),
    }


def _response_for(worker: Mapping[str, Any], command: str) -> dict[str, Any]:
    transport = worker.get("transport")
    responses = getattr(transport, "responses", [])
    for item in reversed(responses):
        if isinstance(item, Mapping) and item.get("command") == command:
            return dict(item)
    return {}


def _path_inside(path: str | Path, root: str | Path, name: str) -> Path:
    candidate = Path(path).resolve()
    parent = Path(root).resolve()
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise DCUPreflightError(f"{name} escaped the phase IPC directory: {candidate}") from exc
    return candidate


def _save_bundle(path: str | Path, bundle: Mapping[str, torch.Tensor], schema: Mapping[str, Any]) -> dict[str, Any]:
    from scripts.dcu_worker import save_tensor_bundle, validate_tensor_bundle

    target = save_tensor_bundle(path, bundle, schema=schema)
    return {
        "path": str(target.resolve()),
        "sha256": _sha256_file(target),
        "bytes": target.stat().st_size,
        "metadata": validate_tensor_bundle(bundle, schema=schema),
    }


def _valid_action_chunk(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DCUPreflightError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != (1, EXPECTED_CHUNK_SIZE, EXPECTED_ACTION_DIM):
        raise DCUPreflightError(f"{name} must have shape (1, 50, 7), got {tuple(value.shape)}")
    if value.dtype not in (torch.float32, torch.bfloat16):
        raise DCUPreflightError(f"{name} must be float32 or bfloat16, got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        raise DCUPreflightError(f"{name} contains non-finite values")
    return value


def _valid_action(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise DCUPreflightError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != (1, EXPECTED_ACTION_DIM):
        raise DCUPreflightError(f"{name} must have shape (1, 7), got {tuple(value.shape)}")
    if value.dtype not in (torch.float32, torch.bfloat16):
        raise DCUPreflightError(f"{name} must be float32 or bfloat16, got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        raise DCUPreflightError(f"{name} contains non-finite values")
    return value


def _prepare_features(runtime: Mapping[str, Any], observation: Any, env: Any) -> dict[str, torch.Tensor]:
    preprocess_observation = runtime.get("preprocess_observation")
    if not callable(preprocess_observation):
        raise DCUPreflightError("CPU runtime did not expose official preprocess_observation")
    processed = preprocess_observation(observation)
    processed = _add_task_to_observation(processed, env)
    env_processed = runtime["env_preprocessor"](processed)
    policy_processed = runtime["preprocessor"](env_processed)
    return _feature_bundle(policy_processed)


def _postprocess_first_action(
    postprocessor: Callable[[Any], Any],
    env_postprocessor: Callable[[Any], Any],
    chunk: torch.Tensor,
) -> torch.Tensor:
    first = _valid_action(chunk[:, 0, :], "first action")
    postprocessed = postprocessor(first)
    if isinstance(postprocessed, Mapping):
        postprocessed = postprocessed.get("action", postprocessed)
    if not isinstance(postprocessed, torch.Tensor):
        raise DCUPreflightError("official postprocessor must return an action tensor")
    transition = env_postprocessor({"action": postprocessed})
    if not isinstance(transition, Mapping) or "action" not in transition:
        raise DCUPreflightError("official env postprocessor must return an action transition")
    return _valid_action(transition["action"], "env-postprocessed first action")


def _seed_cpu(seed: int) -> None:
    normalized = _validate_seed(seed)
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)


def _phase_physical_device(phase: str, requested: int | None) -> int:
    normalized_phase = _normalise_phase(phase)
    if normalized_phase == "compare":
        expected = EXPECTED_COMPARE_PHYSICAL_DEVICE
    elif normalized_phase == "closed-loop":
        expected = EXPECTED_CLOSED_LOOP_PHYSICAL_DEVICE
    else:
        raise DCUPreflightError(f"single-card physical-device selection is invalid for {normalized_phase}")
    selected = expected if requested is None else _require_nonnegative_int(requested, "physical_device")
    if selected != expected:
        raise DCUPreflightError(
            f"{normalized_phase} is pinned to physical device {expected}; got {selected}"
        )
    return selected


def _run_compare_impl(
    *,
    config: Mapping[str, Any],
    phase: str,
    run_directory: str | Path,
    preflight: Mapping[str, Any],
    config_path: str | Path | None = None,
    physical_device: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Run reset/render/feature/explicit-noise comparison without env.step."""

    from scripts import m0_smoke
    from scripts.dcu_worker import REQUEST_SCHEMA, RESPONSE_SCHEMA

    runtime = build_cpu_runtime(config, include_policy=True, phase=phase)
    env = runtime["env"]
    worker: dict[str, Any] | None = None
    ipc = Path(run_directory).resolve() / "ipc"
    ipc.mkdir(parents=True, exist_ok=True)
    try:
        observation, reset_info = _reset_vector_env(env, EXPECTED_SEED)
        _, render_latency = _render_once(env)
        gl = _gl_identity_after_render()
        features = _prepare_features(runtime, observation, env)
        reference = _require_mapping(preflight.get("reference"), "preflight.reference")
        feature_match = _assert_reference_feature_metadata(features, reference)
        noise, noise_hash = generate_flow_noise(seed=EXPECTED_SEED, shape=EXPECTED_FLOW_NOISE_SHAPE)
        request_artifact = _save_bundle(
            ipc / "compare-request.safetensors",
            {**features, "noise": noise},
            REQUEST_SCHEMA,
        )

        policy = runtime.get("policy")
        if policy is None or not callable(getattr(policy, "predict_action_chunk", None)):
            raise DCUPreflightError("CPU runtime did not build the frozen policy")
        policy.reset()
        _seed_cpu(EXPECTED_SEED)
        cpu_started = time.perf_counter()
        with torch.inference_mode():
            cpu_chunk = policy.predict_action_chunk(features, noise=noise)
        cpu_wall = time.perf_counter() - cpu_started
        cpu_chunk = _valid_action_chunk(cpu_chunk, "CPU action chunk")
        cpu_artifact = _save_bundle(
            ipc / "cpu-action-chunk.safetensors", {"action_chunk": cpu_chunk}, RESPONSE_SCHEMA
        )

        selected_physical = _phase_physical_device("compare", physical_device)
        worker = _start_worker(
            config,
            run_directory=run_directory,
            config_path=config_path,
            physical_device=selected_physical,
        )
        _worker_request(worker, "reset", seed=EXPECTED_SEED, timeout_kind="startup")
        dcu_started = time.perf_counter()
        dcu_chunk = _worker_request(
            worker, "predict_action_chunk", request_artifact["path"], timeout_kind="forward"
        )
        dcu_wall = time.perf_counter() - dcu_started
        dcu_chunk = _valid_action_chunk(dcu_chunk, "DCU action chunk")
        dcu_response = _response_for(worker, "predict_action_chunk")
        response_path = dcu_response.get("response_path")
        if not isinstance(response_path, str):
            raise DCUPreflightError("DCU predict response did not identify a safetensors artifact")
        dcu_response_path = _path_inside(response_path, ipc, "DCU response")
        dcu_artifact = {
            "path": str(dcu_response_path),
            "sha256": _sha256_file(dcu_response_path),
            "bytes": dcu_response_path.stat().st_size,
        }

        cpu_action = _postprocess_first_action(
            runtime["postprocessor"], runtime["env_postprocessor"], cpu_chunk
        )
        dcu_action = _postprocess_first_action(
            runtime["postprocessor"], runtime["env_postprocessor"], dcu_chunk
        )
        comparison = compare_outputs(
            cpu_chunk,
            dcu_chunk,
            cpu_action,
            dcu_action,
            cpu_latency=cpu_wall,
            dcu_latency=dcu_wall,
            dcu_peak_memory=(
                int(dcu_response["peak_memory_bytes"])
                if isinstance(dcu_response.get("peak_memory_bytes"), int)
                else None
            ),
        )
        comparison.update(
            {
                "status": "PASS",
                "noise_seed": EXPECTED_SEED,
                "noise_sha256": noise_hash,
                "cpu": {
                    "wall_latency_seconds": cpu_wall,
                    "inference_latency_seconds": cpu_wall,
                    "model": runtime.get("model"),
                },
                "dcu": {
                    "wall_latency_seconds": dcu_wall,
                    "inference_latency_seconds": dcu_response.get("latency_seconds"),
                    "model": worker["ping"].get("model"),
                    "response": dcu_response,
                },
                "device": {
                    "physical_device": selected_physical,
                    "logical_device": EXPECTED_LOGICAL_DEVICE,
                    "card0_reason": config["runtime"]["card0_reason"],
                },
                "render_latency_seconds": render_latency,
                "gl": gl,
                "env_step_count": 0,
                "reset_info": _summary_value(reset_info),
                "task": runtime.get("task"),
                "feature_metadata": feature_match,
                "artifacts": {
                    "request": request_artifact,
                    "cpu_chunk": cpu_artifact,
                    "dcu_chunk": dcu_artifact,
                },
            }
        )
        decision = {
            "decision_index": 0,
            "feature_metadata": feature_match["actual"],
            "cpu_chunk": _describe_runtime_value(cpu_chunk),
            "dcu_chunk": _describe_runtime_value(dcu_chunk),
            "cpu_postprocessed_action": _describe_runtime_value(cpu_action),
            "dcu_postprocessed_action": _describe_runtime_value(dcu_action),
            "gl": gl,
            "env_step_count": 0,
            "finite": True,
        }
        return {
            "status": "PASS",
            "comparison": comparison,
            "decisions": [decision],
            "worker": _worker_summary(worker),
        }
    finally:
        _close_worker(worker)
        try:
            runtime["close_envs"](runtime["envs"])
        except Exception:
            pass


def _run_closed_loop_impl(
    *,
    config: Mapping[str, Any],
    phase: str,
    run_directory: str | Path,
    config_path: str | Path | None = None,
    physical_device: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Run one bounded official episode with remote queue-native selection only."""

    from scripts import m0_smoke
    from scripts.dcu_model_worker import FEATURE_SCHEMA
    from scripts.dcu_worker import save_tensor_bundle

    runtime = build_cpu_runtime(config, include_policy=False, phase=phase)
    env = runtime["env"]
    trace = m0_smoke.TraceStore(
        chunk_size=EXPECTED_CHUNK_SIZE,
        n_action_steps=EXPECTED_N_ACTION_STEPS,
        action_dim=EXPECTED_ACTION_DIM,
    )
    worker: dict[str, Any] | None = None
    ipc = Path(run_directory).resolve() / "ipc"
    ipc.mkdir(parents=True, exist_ok=True)
    request_number = 0

    def request_writer(bundle: Mapping[str, torch.Tensor]) -> Path:
        nonlocal request_number
        target = ipc / f"closed-loop-features-{request_number:04d}.safetensors"
        request_number += 1
        save_tensor_bundle(target, bundle, schema=FEATURE_SCHEMA)
        return target

    try:
        selected_physical = _phase_physical_device("closed-loop", physical_device)
        worker = _start_worker(
            config,
            run_directory=run_directory,
            config_path=config_path,
            physical_device=selected_physical,
        )
        remote = FeatureOnlyRemotePolicy(
            worker["client"], request_writer=request_writer, trace=trace
        )
        remote.reset_timeout_seconds = _timeout_seconds(worker["runtime"], "startup")
        remote.forward_timeout_seconds = _timeout_seconds(worker["runtime"], "forward")
        policy = m0_smoke.PolicyProxy(remote, trace)
        env_preprocessor = m0_smoke.ProcessorProxy(runtime["env_preprocessor"], "env_processor", trace)
        preprocessor = m0_smoke.ProcessorProxy(runtime["preprocessor"], "policy_processor", trace)
        postprocessor = m0_smoke.ProcessorProxy(runtime["postprocessor"], "postprocessor", trace)
        env_postprocessor = m0_smoke.ProcessorProxy(
            runtime["env_postprocessor"], "env_postprocessor", trace
        )
        wrapped_env = m0_smoke.VectorEnvProxy(env, trace)
        policy.eval()
        from lerobot.scripts.lerobot_eval import rollout

        rollout_data = rollout(
            wrapped_env,
            policy,
            env_preprocessor,
            env_postprocessor,
            preprocessor,
            postprocessor,
            seeds=[EXPECTED_SEED],
            return_observations=False,
            render_callback=m0_smoke.make_render_callback(trace),
        )
        m0_smoke._set_official_done(trace, rollout_data)
        m0_smoke._assert_rollout_acceptance(
            trace, rollout_data, policy, render_count=trace.render_count
        )
        actions = rollout_data["action"]
        rewards = rollout_data["reward"]
        successes = rollout_data["success"]
        dones = rollout_data["done"]
        steps = int(actions.shape[1])
        if not 1 <= steps <= EXPECTED_HORIZON:
            raise DCUPreflightError(
                f"closed-loop rollout must run between 1 and {EXPECTED_HORIZON} steps, got {steps}"
            )
        if not bool(dones.detach().cpu().bool()[0, -1].item()):
            raise DCUPreflightError("closed-loop rollout final official done must be true")
        gl = m0_smoke._assert_gl_identity(trace.gl)
        episode_result = {
            "status": "PASS",
            "episodes": 1,
            "steps": steps,
            "horizon": EXPECTED_HORIZON,
            "completed_by": "done_or_horizon",
            "termination_reason": trace.decisions[-1].get("termination_reason") if trace.decisions else "no_decision",
            "success_observed": bool(successes.detach().cpu().bool().any().item()),
            "success_threshold_enforced": False,
            "reward": rewards.detach().cpu().tolist(),
            "done": dones.detach().cpu().tolist(),
            "render_count": trace.render_count,
            "render_latency": _summary_value(trace.render_latencies),
            "env_step_latency": _summary_value(trace.env_step_latencies),
            "inference_latency": _summary_value(trace.inference_latencies),
            "gl": gl,
            "task": runtime.get("task"),
            "device": {
                "physical_device": selected_physical,
                "logical_device": EXPECTED_LOGICAL_DEVICE,
                "card0_reason": config["runtime"]["card0_reason"],
            },
        }
        return {
            "status": "PASS",
            "episode_result": episode_result,
            "decisions": trace.decisions,
            "worker": _worker_summary(worker),
        }
    finally:
        _close_worker(worker)
        try:
            runtime["close_envs"](runtime["envs"])
        except Exception:
            pass


def _run_one_step_child_impl(
    *,
    config: Mapping[str, Any],
    phase: str,
    run_directory: str | Path,
    config_path: str | Path | None = None,
    physical_device: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Execute the fixed two-decision/two-step concurrency child boundary."""

    from scripts import m0_smoke
    from scripts.dcu_model_worker import FEATURE_SCHEMA
    from scripts.dcu_worker import save_tensor_bundle

    concurrency = _require_mapping(config["concurrency"], "concurrency")
    steps = _require_exact(
        concurrency.get("steps_per_worker"),
        EXPECTED_CONCURRENCY_STEPS_PER_WORKER,
        "concurrency.steps_per_worker",
    )
    if int(steps) != EXPECTED_CONCURRENCY_STEPS_PER_WORKER:
        raise DCUPreflightError("concurrency child step boundary is not the frozen two-step contract")
    # Validate the actual process boundary as well as the environment builder.
    # A direct/private child invocation with compute visibility is rejected so
    # robosuite can never interpret a CUDA ordinal as an EGL ordinal list.
    cpu_compute_visibility = {
        key: os.environ.get(key)
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    }
    if any(value is not None for value in cpu_compute_visibility.values()):
        raise DCUPreflightError(
            "CPU concurrency child must not receive HIP/CUDA compute visibility: "
            f"{cpu_compute_visibility}"
        )
    cpu_environment = build_cpu_child_environment(config, base_environment=os.environ.copy())
    runtime = build_cpu_runtime(config, include_policy=False, phase="concurrency")
    env = runtime["env"]
    trace = m0_smoke.TraceStore(
        chunk_size=EXPECTED_CHUNK_SIZE,
        n_action_steps=EXPECTED_N_ACTION_STEPS,
        action_dim=EXPECTED_ACTION_DIM,
    )
    worker: dict[str, Any] | None = None
    run_dir = Path(run_directory).resolve()
    ipc = run_dir / "ipc"
    ipc.mkdir(parents=True, exist_ok=True)
    selected_physical = (
        _require_nonnegative_int(physical_device, "physical_device")
        if physical_device is not None
        else EXPECTED_PHYSICAL_DEVICES[0]
    )
    if selected_physical not in EXPECTED_PHYSICAL_DEVICES:
        raise DCUPreflightError("one-step child physical_device must be 0 or 1")
    request_counter = 0

    def request_writer(bundle: Mapping[str, torch.Tensor]) -> Path:
        nonlocal request_counter
        target = ipc / f"one-step-features-{request_counter:04d}.safetensors"
        request_counter += 1
        save_tensor_bundle(target, bundle, schema=FEATURE_SCHEMA)
        return target

    try:
        worker = _start_worker(
            config,
            run_directory=run_dir,
            config_path=config_path,
            physical_device=selected_physical,
        )
        worker_evidence = _worker_summary(worker)
        compute_environment = _require_mapping(
            worker_evidence.get("compute_environment"), "worker.compute_environment"
        )
        if compute_environment.get("physical_k100") != selected_physical:
            raise DCUPreflightError("nested worker physical K100 evidence does not match child assignment")
        if compute_environment.get("HIP_VISIBLE_DEVICES") != str(selected_physical):
            raise DCUPreflightError("nested worker HIP_VISIBLE_DEVICES does not match physical K100")
        if compute_environment.get("CUDA_VISIBLE_DEVICES") != "0":
            raise DCUPreflightError("nested worker CUDA_VISIBLE_DEVICES must expose logical cuda:0")
        if worker_evidence.get("torch_logical_device") != EXPECTED_LOGICAL_DEVICE:
            raise DCUPreflightError("nested worker torch logical device is not cuda:0")
        if worker_evidence.get("model_load_success") is not True:
            raise DCUPreflightError("nested SmolVLA model load evidence is not successful")
        remote = FeatureOnlyRemotePolicy(worker["client"], request_writer=request_writer, trace=trace)
        remote.reset_timeout_seconds = _timeout_seconds(worker["runtime"], "startup")
        remote.forward_timeout_seconds = _timeout_seconds(worker["runtime"], "forward")
        policy = m0_smoke.PolicyProxy(remote, trace)
        env_preprocessor = m0_smoke.ProcessorProxy(runtime["env_preprocessor"], "env_processor", trace)
        preprocessor = m0_smoke.ProcessorProxy(runtime["preprocessor"], "policy_processor", trace)
        postprocessor = m0_smoke.ProcessorProxy(runtime["postprocessor"], "postprocessor", trace)
        env_postprocessor = m0_smoke.ProcessorProxy(
            runtime["env_postprocessor"], "env_postprocessor", trace
        )
        wrapped_env = m0_smoke.VectorEnvProxy(env, trace)
        policy.reset()
        observation, reset_info = _reset_vector_env(env, EXPECTED_SEED)
        m0_smoke.make_render_callback(trace)(wrapped_env)
        gl = _assert_exact_llvmpipe(m0_smoke._assert_gl_identity(trace.gl))
        egl = _egl_probe_evidence(
            selected_device=str(cpu_environment["MUJOCO_EGL_DEVICE_ID"])
        )
        action_evidence: list[dict[str, Any]] = []
        for step_index in range(int(steps)):
            # Keep the official processor order for every independent policy
            # decision: observation -> env processor -> policy processor ->
            # frozen remote policy -> postprocessors -> environment step.
            processed = runtime["preprocess_observation"](observation)
            processed = _add_task_to_observation(processed, env)
            processed = env_preprocessor(processed)
            features = preprocessor(processed)
            action = policy.select_action(_feature_bundle(features))
            action = postprocessor(action)
            transition = env_postprocessor({"action": action})
            if not isinstance(transition, Mapping) or "action" not in transition:
                raise DCUPreflightError("official env postprocessor returned no action")
            action_tensor = _valid_action(transition["action"], f"two-step action {step_index}")
            action_numpy = action_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            if action_numpy.shape != (1, EXPECTED_ACTION_DIM) or not np.isfinite(action_numpy).all():
                raise DCUPreflightError("concurrency action must be finite float32 with shape (1, 7)")
            step_result = wrapped_env.step(action_numpy)
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise DCUPreflightError("vector environment step must return five values")
            terminated = np.asarray(step_result[2], dtype=bool).reshape(-1)
            truncated = np.asarray(step_result[3], dtype=bool).reshape(-1)
            if terminated.size != EXPECTED_N_ENVS or truncated.size != EXPECTED_N_ENVS:
                raise DCUPreflightError("concurrency child step done flags must have one environment")
            done = bool(np.logical_or(terminated, truncated).any())
            if done and step_index < int(steps) - 1:
                raise DCUPreflightError(
                    "concurrency child episode terminated before completing the fixed two steps"
                )
            if not trace.decisions:
                raise DCUPreflightError("policy decision was not recorded")
            trace.decisions[-1]["gl"] = gl
            trace.decisions[-1]["step_index"] = step_index
            trace.decisions[-1]["action_contract"] = {
                "shape": list(action_numpy.shape),
                "dtype": str(action_numpy.dtype),
                "finite": bool(np.isfinite(action_numpy).all()),
            }
            action_evidence.append(trace.decisions[-1]["action_contract"] | {"decision_index": step_index})
            observation = step_result[0]
        if len(trace.decisions) != int(steps) or len(trace.env_step_latencies) != int(steps):
            raise DCUPreflightError(
                "concurrency child did not record exactly two decisions and two env.step calls"
            )
        if trace.render_count < 1:
            raise DCUPreflightError("concurrency child did not render after reset")
        worker_response = _response_for(worker, "select_action")
        peak_memory = worker_response.get("peak_memory_bytes")
        if peak_memory is not None and (
            isinstance(peak_memory, bool) or not isinstance(peak_memory, int) or peak_memory < 0
        ):
            raise DCUPreflightError("worker peak_memory_bytes must be a non-negative integer")
        latency = {
            "inference_seconds": _summary_value(trace.inference_latencies),
            "env_step_seconds": _summary_value(trace.env_step_latencies),
            "render_seconds": _summary_value(trace.render_latencies),
            "worker_wall_seconds": _summary_value(
                [item.get("worker_wall_latency_seconds") for item in trace.decisions]
            ),
        }
        memory = {
            "dcu_peak_memory_bytes": peak_memory,
            "worker_response": _summary_value(worker_response),
        }
        cpu_environment_evidence = {
            "physical_k100": selected_physical,
            "HIP_VISIBLE_DEVICES": cpu_compute_visibility["HIP_VISIBLE_DEVICES"],
            "CUDA_VISIBLE_DEVICES": cpu_compute_visibility["CUDA_VISIBLE_DEVICES"],
            "MUJOCO_EGL_DEVICE_ID": cpu_environment["MUJOCO_EGL_DEVICE_ID"],
            "MUJOCO_GL": cpu_environment.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": cpu_environment.get("PYOPENGL_PLATFORM"),
            "offline": {
                key: cpu_environment.get(key) for key in EXPECTED_OFFLINE_ENV
            },
        }
        device_evidence = {
            "physical_k100": selected_physical,
            "cpu_child_cuda_visible_devices": cpu_compute_visibility["CUDA_VISIBLE_DEVICES"],
            "nested_worker_hip_visible_devices": compute_environment["HIP_VISIBLE_DEVICES"],
            "nested_worker_cuda_visible_devices": compute_environment["CUDA_VISIBLE_DEVICES"],
            "torch_logical_device": worker_evidence["torch_logical_device"],
            "torch_device_name": worker_evidence.get("torch_device_name"),
        }
        return {
            "status": "PASS",
            "child_result": {
                "status": "PASS",
                "physical_device": selected_physical,
                "logical_device": EXPECTED_LOGICAL_DEVICE,
                "seed": EXPECTED_SEED,
                "reset_count": 1,
                "decision_count": int(steps),
                "env_step_count": int(steps),
                "reset_info": _summary_value(reset_info),
                "gl": gl,
                "egl": egl,
                "environment": {"cpu_child": cpu_environment_evidence},
                "device": device_evidence,
                "model_load_success": True,
                "model": worker_evidence.get("model"),
                "action_evidence": action_evidence,
                "latency": latency,
                "memory": memory,
                "network": {"offline": True, "hub_fallback": False},
                "render_count": trace.render_count,
                "task": runtime.get("task"),
                "decisions": trace.decisions,
                "worker": worker_evidence,
            },
            "decisions": trace.decisions,
        }
    finally:
        _close_worker(worker)
        try:
            runtime["close_envs"](runtime["envs"])
        except Exception:
            pass


def build_concurrency_child_command(
    *,
    config_path: str | Path,
    expected_project_sha: str,
    physical_device: int,
    run_directory: str | Path,
    token: str = ONE_STEP_CHILD_TOKEN,
) -> list[str]:
    """Build the only command through which a one-step child is reachable."""

    if not isinstance(token, str) or token != ONE_STEP_CHILD_TOKEN:
        raise DCUPreflightError("invalid one-step-child internal token")
    device = _require_nonnegative_int(physical_device, "physical_device")
    if device not in EXPECTED_PHYSICAL_DEVICES:
        raise DCUPreflightError("concurrency child physical device must be 0 or 1")
    expected = _require_string(expected_project_sha, "expected project SHA")
    config_file = Path(config_path).resolve()
    run_dir = Path(run_directory).resolve()
    return [
        sys.executable,
        str((_ROOT / "scripts" / "dcu_preflight.py").resolve()),
        _ONE_STEP_CHILD_PHASE,
        "--config",
        str(config_file),
        "--expected-project-sha",
        expected,
        "--physical-device",
        str(device),
        "--run-directory",
        str(run_dir),
        "--internal-token",
        ONE_STEP_CHILD_TOKEN,
    ]


def _terminalize_concurrency_manifest(
    child_directory: str | Path,
    *,
    status: str,
    returncode: int | None,
    reason: str,
) -> Path:
    """Make a child run manifest terminal even when its process was killed."""

    if status not in {"PASS", "FAIL", "TERMINATED"}:
        raise DCUPreflightError(f"invalid concurrency child terminal status: {status}")
    directory = Path(child_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "run_manifest.json"
    current: dict[str, Any] = {}
    if target.is_file():
        try:
            parsed = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                current = dict(parsed)
        except (OSError, json.JSONDecodeError):
            # A partial/invalid RUNNING file is replaced with an explicit
            # terminal record; the parent owns this recovery boundary.
            current = {}
    current.update(
        {
            "status": status,
            "terminal": True,
            "returncode": returncode,
            "termination_reason": str(reason),
            "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
    )
    return _atomic_json_dump(target, current)


def _close_concurrency_process_pipes(process: Any) -> None:
    """Close pipes after bounded collection can no longer reap their streams."""

    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _decode_process_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _cleanup_started_concurrency_process(process: Any) -> None:
    """Reap a just-started child whose PID metadata was invalid."""

    if getattr(process, "poll", lambda: None)() is not None:
        return
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        try:
            terminate()
        except Exception:
            pass
    wait = getattr(process, "wait", None)
    if not callable(wait):
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass
        _close_concurrency_process_pipes(process)
        return
    try:
        wait(timeout=1.0)
    except Exception:
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass
        try:
            wait(timeout=1.0)
        except Exception:
            pass
    _close_concurrency_process_pipes(process)


def _kill_concurrency_process(metadata: dict[str, Any], *, reason: str) -> None:
    """Terminate only the process group created for one concurrency child."""

    process = metadata["process"]
    pgid = metadata.get("pgid")
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 0:
        raise DCUPreflightError("concurrency child process group id is missing or invalid")
    # The group may still contain a nested worker after the outer child exits;
    # signal the owned group even when ``poll()`` is no longer ``None``.
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    wait = getattr(process, "wait", None)
    if process.poll() is None and callable(wait):
        try:
            wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
    # Always make a best-effort SIGKILL pass over this process's own group.
    # This is what closes the descendant-leak window after a fast outer exit.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None and callable(wait):
        try:
            wait(timeout=1.0)
        except Exception:
            metadata["wait_timeout"] = True
    metadata["group_cleanup_done"] = True
    metadata["forced_termination"] = True
    metadata["termination_reason"] = str(reason)
    metadata["returncode"] = getattr(process, "returncode", None)


def _collect_concurrency_process(
    metadata: dict[str, Any],
    *,
    timeout: float,
) -> tuple[str, str]:
    """Collect one child without allowing a killed sibling to linger."""

    process = metadata["process"]
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as first_timeout:
        metadata["timeout_error"] = str(first_timeout)
        if not metadata.get("forced_termination"):
            _kill_concurrency_process(metadata, reason=f"timeout after {timeout:.3f}s")
        try:
            stdout, stderr = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired as second_timeout:
            metadata["collection_timeout"] = True
            metadata["terminal_collection_error"] = str(second_timeout)
            stdout = getattr(second_timeout, "output", None) or getattr(first_timeout, "output", None) or ""
            stderr = getattr(second_timeout, "stderr", None) or getattr(first_timeout, "stderr", None) or ""
            _close_concurrency_process_pipes(process)
        except Exception as exc:
            metadata["collection_error"] = str(exc)
            stdout = getattr(first_timeout, "output", None) or ""
            stderr = getattr(first_timeout, "stderr", None) or ""
            _close_concurrency_process_pipes(process)
    metadata["stdout"] = _decode_process_output(stdout)
    metadata["stderr"] = _decode_process_output(stderr)
    process_returncode = getattr(process, "returncode", None)
    if metadata.get("returncode") is None or metadata.get("forced_termination") is not True:
        metadata["returncode"] = process_returncode
    return metadata["stdout"], metadata["stderr"]


def _validate_concurrency_child_result(
    child_result: Mapping[str, Any],
    *,
    expected_physical_device: int,
    expected_steps: int,
) -> dict[str, Any]:
    """Validate the complete evidence contract emitted by one child."""

    def require_field(mapping: Mapping[str, Any], field: str, path: str) -> Any:
        if field not in mapping:
            raise DCUPreflightError(f"child evidence missing {path}")
        return mapping[field]

    if require_field(child_result, "status", "status") != "PASS":
        raise DCUPreflightError("child evidence status must be PASS")
    for field, expected in (
        ("physical_device", expected_physical_device),
        ("logical_device", EXPECTED_LOGICAL_DEVICE),
        ("reset_count", 1),
        ("decision_count", expected_steps),
        ("env_step_count", expected_steps),
    ):
        if require_field(child_result, field, field) != expected:
            raise DCUPreflightError(f"child evidence {field} does not match {expected!r}")
    if require_field(child_result, "model_load_success", "model_load_success") is not True:
        raise DCUPreflightError("child evidence model_load_success must be true")

    egl = require_field(child_result, "egl", "egl")
    if not isinstance(egl, Mapping):
        raise DCUPreflightError("child evidence egl must be a mapping")
    count = require_field(egl, "eglQueryDevicesEXT_device_count", "egl.count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= int(EXPECTED_EGL_DEVICE_ID):
        raise DCUPreflightError("child evidence EGL count must be greater than selected ordinal 8")
    if require_field(egl, "selected_MUJOCO_EGL_DEVICE_ID", "egl.selected") != EXPECTED_EGL_DEVICE_ID:
        raise DCUPreflightError("child evidence selected EGL ordinal must be 8")
    test_device = require_field(egl, "test_device", "egl.test_device")
    if not isinstance(test_device, Mapping) or require_field(test_device, "returncode", "egl.test_device.returncode") != 0:
        raise DCUPreflightError("child evidence EGL test_device must return 0")

    gl = require_field(child_result, "gl", "gl")
    if not isinstance(gl, Mapping):
        raise DCUPreflightError("child evidence gl must be a mapping")
    try:
        _assert_exact_llvmpipe(gl)
    except DCUPreflightError as exc:
        raise DCUPreflightError(f"child evidence exact GL identity failed: {exc}") from exc

    action_evidence = require_field(child_result, "action_evidence", "action_evidence")
    if isinstance(action_evidence, (str, bytes)) or not isinstance(action_evidence, Sequence):
        raise DCUPreflightError("child evidence action_evidence must be a sequence")
    if len(action_evidence) != expected_steps:
        raise DCUPreflightError("child evidence action_evidence count does not equal two")
    for index, action in enumerate(action_evidence):
        if not isinstance(action, Mapping):
            raise DCUPreflightError(f"child evidence action_evidence[{index}] must be a mapping")
        if require_field(action, "decision_index", f"action_evidence[{index}].decision_index") != index:
            raise DCUPreflightError("child evidence action decision indices are not contiguous")
        if require_field(action, "shape", f"action_evidence[{index}].shape") != [1, EXPECTED_ACTION_DIM]:
            raise DCUPreflightError("child evidence action shape must be [1, 7]")
        if require_field(action, "dtype", f"action_evidence[{index}].dtype") != "float32":
            raise DCUPreflightError("child evidence action dtype must be float32")
        if require_field(action, "finite", f"action_evidence[{index}].finite") is not True:
            raise DCUPreflightError("child evidence action must be finite")

    latency = require_field(child_result, "latency", "latency")
    if not isinstance(latency, Mapping):
        raise DCUPreflightError("child evidence latency must be a mapping")

    def finite_nonnegative_sequence(field: str, minimum_length: int) -> None:
        values = require_field(latency, field, f"latency.{field}")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise DCUPreflightError(f"child evidence latency.{field} must be a sequence")
        if len(values) < minimum_length:
            raise DCUPreflightError(
                f"child evidence latency.{field} must contain at least {minimum_length} values"
            )
        for index, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise DCUPreflightError(f"child evidence latency.{field}[{index}] must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise DCUPreflightError(
                    f"child evidence latency.{field}[{index}] must be finite and non-negative"
                )

    for latency_field in ("inference_seconds", "env_step_seconds", "worker_wall_seconds"):
        finite_nonnegative_sequence(latency_field, expected_steps)
        if len(latency[latency_field]) != expected_steps:
            raise DCUPreflightError(
                f"child evidence latency.{latency_field} must contain exactly {expected_steps} values"
            )
    finite_nonnegative_sequence("render_seconds", 1)

    memory = require_field(child_result, "memory", "memory")
    if not isinstance(memory, Mapping):
        raise DCUPreflightError("child evidence memory must be a mapping")
    peak_memory = require_field(memory, "dcu_peak_memory_bytes", "memory.dcu_peak_memory_bytes")
    if isinstance(peak_memory, bool) or not isinstance(peak_memory, (int, np.integer)) or peak_memory < 0:
        raise DCUPreflightError("child evidence memory peak must be a non-negative integer")
    if not isinstance(require_field(memory, "worker_response", "memory.worker_response"), Mapping):
        raise DCUPreflightError("child evidence memory.worker_response must be a mapping")

    network = require_field(child_result, "network", "network")
    if not isinstance(network, Mapping):
        raise DCUPreflightError("child evidence network must be a mapping")
    if network.get("offline") is not True or network.get("hub_fallback") is not False:
        raise DCUPreflightError("child evidence must prove offline execution without Hub fallback")

    environment = require_field(child_result, "environment", "environment")
    if not isinstance(environment, Mapping):
        raise DCUPreflightError("child evidence environment must be a mapping")
    cpu_child = require_field(environment, "cpu_child", "environment.cpu_child")
    if not isinstance(cpu_child, Mapping):
        raise DCUPreflightError("child evidence CPU environment must be a mapping")
    for field in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if require_field(cpu_child, field, f"environment.cpu_child.{field}") is not None:
            raise DCUPreflightError(f"child evidence CPU environment {field} must be unset")
    if require_field(cpu_child, "MUJOCO_EGL_DEVICE_ID", "environment.cpu_child.MUJOCO_EGL_DEVICE_ID") != EXPECTED_EGL_DEVICE_ID:
        raise DCUPreflightError("child evidence CPU environment EGL ordinal must be 8")

    device = require_field(child_result, "device", "device")
    if not isinstance(device, Mapping):
        raise DCUPreflightError("child evidence device must be a mapping")
    for field, expected in (
        ("physical_k100", expected_physical_device),
        ("nested_worker_hip_visible_devices", str(expected_physical_device)),
        ("nested_worker_cuda_visible_devices", "0"),
        ("torch_logical_device", EXPECTED_LOGICAL_DEVICE),
    ):
        if require_field(device, field, f"device.{field}") != expected:
            raise DCUPreflightError(f"child evidence device.{field} does not match {expected!r}")

    worker = require_field(child_result, "worker", "worker")
    if not isinstance(worker, Mapping):
        raise DCUPreflightError("child evidence worker must be a mapping")
    if require_field(worker, "physical_device", "worker.physical_device") != expected_physical_device:
        raise DCUPreflightError("child evidence worker physical device does not match assignment")
    if require_field(worker, "torch_logical_device", "worker.torch_logical_device") != EXPECTED_LOGICAL_DEVICE:
        raise DCUPreflightError("child evidence worker logical device must be cuda:0")
    torch_device_name = require_field(worker, "torch_device_name", "worker.torch_device_name")
    if not isinstance(torch_device_name, str) or not torch_device_name.strip():
        raise DCUPreflightError("child evidence worker torch device name is missing")
    if require_field(worker, "model_load_success", "worker.model_load_success") is not True:
        raise DCUPreflightError("child evidence worker model load must succeed")
    nested = require_field(worker, "compute_environment", "worker.compute_environment")
    if not isinstance(nested, Mapping):
        raise DCUPreflightError("child evidence nested worker environment must be a mapping")
    for field, expected in (
        ("physical_k100", expected_physical_device),
        ("HIP_VISIBLE_DEVICES", str(expected_physical_device)),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("torch_logical_device", EXPECTED_LOGICAL_DEVICE),
    ):
        if require_field(nested, field, f"worker.compute_environment.{field}") != expected:
            raise DCUPreflightError(
                f"child evidence worker.compute_environment.{field} does not match {expected!r}"
            )
    return dict(child_result)


def _run_concurrency_impl(
    *,
    config: Mapping[str, Any],
    phase: str,
    run_directory: str | Path,
    expected_project_sha: str,
    actual_project_sha: str,
    config_path: str | Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Popen two independent fixed-boundary children before waiting on either."""

    del phase, actual_project_sha
    run_dir = Path(run_directory).resolve()
    runtime = _require_mapping(config["runtime"], "runtime")
    concurrency = _require_mapping(config["concurrency"], "concurrency")
    steps = _require_exact(
        concurrency.get("steps_per_worker"),
        EXPECTED_CONCURRENCY_STEPS_PER_WORKER,
        "concurrency.steps_per_worker",
    )
    config_file = _config_file_path(config, config_path)
    children_root = run_dir / "children"
    if children_root.exists():
        raise FileExistsError(children_root)
    children_root.mkdir(parents=True, exist_ok=False)
    devices = list(EXPECTED_PHYSICAL_DEVICES)
    commands = [
        build_concurrency_child_command(
            config_path=config_file,
            expected_project_sha=expected_project_sha,
            physical_device=device,
            run_directory=children_root / f"child{index}",
        )
        for index, device in enumerate(devices)
    ]
    children: list[dict[str, Any]] = []
    processes: list[dict[str, Any]] = []
    first_error: DCUPreflightError | None = None
    started = time.perf_counter()
    try:
        # Deliberately complete this loop before calling communicate(): both
        # CPU children are live concurrently, each with its own worker IPC.
        for index, (device, command) in enumerate(zip(devices, commands, strict=True)):
            child_dir = children_root / f"child{index}"
            if child_dir.exists():
                raise FileExistsError(child_dir)
            # Only the nested model worker receives compute visibility.  The
            # CPU child keeps the independent validated EGL ordinal.
            environment = build_cpu_child_environment(config)
            if environment.get("HIP_VISIBLE_DEVICES") is not None or environment.get("CUDA_VISIBLE_DEVICES") is not None:
                raise DCUPreflightError("CPU child environment must not contain HIP/CUDA visibility")
            if environment.get("MUJOCO_EGL_DEVICE_ID") != EXPECTED_EGL_DEVICE_ID:
                raise DCUPreflightError("CPU child environment must retain EGL ordinal 8")
            process = subprocess.Popen(
                command,
                cwd=str(_ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="strict",
                shell=False,
                start_new_session=True,
            )
            process_pid = getattr(process, "pid", None)
            if isinstance(process_pid, bool) or not isinstance(process_pid, int) or process_pid <= 0:
                _cleanup_started_concurrency_process(process)
                raise DCUPreflightError("concurrency child process did not expose a valid pid")
            processes.append(
                {
                    "index": index,
                    "physical_device": device,
                    "logical_device": EXPECTED_LOGICAL_DEVICE,
                    "command": command,
                    "directory": child_dir,
                    "process": process,
                    # start_new_session makes the child PID the process-group
                    # ID; only this group is ever signalled during cleanup.
                    "pgid": process_pid,
                    "cpu_environment": {
                        "HIP_VISIBLE_DEVICES": environment.get("HIP_VISIBLE_DEVICES"),
                        "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
                        "MUJOCO_EGL_DEVICE_ID": environment.get("MUJOCO_EGL_DEVICE_ID"),
                        "MUJOCO_GL": environment.get("MUJOCO_GL"),
                        "PYOPENGL_PLATFORM": environment.get("PYOPENGL_PLATFORM"),
                    },
                }
            )
    except BaseException:
        for metadata in processes:
            _kill_concurrency_process(metadata, reason="parent launch failure")
            try:
                _collect_concurrency_process(metadata, timeout=1.0)
            except Exception:
                pass
            _terminalize_concurrency_manifest(
                metadata["directory"],
                status="TERMINATED",
                returncode=metadata.get("returncode"),
                reason=str(metadata.get("termination_reason", "parent launch failure")),
            )
        raise

    timeout = _timeout_seconds(runtime, "startup") + _timeout_seconds(runtime, "forward") + _timeout_seconds(runtime, "shutdown")
    try:
        # Waiting in launch order retains deterministic manifests.  Any first
        # failure immediately terminates every still-live sibling before it can
        # leak a RUNNING manifest or process.
        for metadata in processes:
            process = metadata["process"]
            if first_error is not None and process.poll() is None:
                _kill_concurrency_process(
                    metadata, reason=f"terminated after child {first_error.args[0]}"
                )
            try:
                stdout, stderr = _collect_concurrency_process(metadata, timeout=timeout)
            except BaseException as exc:
                if first_error is None:
                    first_error = DCUPreflightError(
                        f"concurrency child {metadata['index']} collection failed: {exc}"
                    )
                _kill_concurrency_process(metadata, reason=f"collection failure: {exc}")
                try:
                    stdout, stderr = _collect_concurrency_process(metadata, timeout=1.0)
                except Exception:
                    stdout, stderr = "", ""
            child_dir = metadata["directory"]
            child_dir.mkdir(parents=True, exist_ok=True)
            process_stdout = child_dir / "process_stdout.log"
            process_stderr = child_dir / "process_stderr.log"
            if not process_stdout.exists():
                _write_text_exclusive(process_stdout, str(stdout or ""))
            if not process_stderr.exists():
                _write_text_exclusive(process_stderr, str(stderr or ""))
            returncode = metadata.get("returncode", getattr(process, "returncode", None))
            metadata["returncode"] = returncode
            manifest_path = child_dir / "run_manifest.json"
            result_path = child_dir / "child_result.json"
            if metadata.get("forced_termination"):
                reason = str(metadata.get("termination_reason", "terminated by concurrency parent"))
                _terminalize_concurrency_manifest(
                    child_dir,
                    status="TERMINATED",
                    returncode=returncode,
                    reason=reason,
                )
                if first_error is None:
                    first_error = DCUPreflightError(
                        f"concurrency child {metadata['index']} terminated ({reason})"
                    )
                continue
            if returncode != 0:
                reason = f"concurrency child {metadata['index']} returned {returncode}"
                _terminalize_concurrency_manifest(
                    child_dir, status="FAIL", returncode=returncode, reason=reason
                )
                if first_error is None:
                    first_error = DCUPreflightError(reason)
                continue
            if not manifest_path.is_file() or not result_path.is_file():
                reason = f"concurrency child {metadata['index']} did not produce its result manifest"
                _terminalize_concurrency_manifest(
                    child_dir, status="FAIL", returncode=returncode, reason=reason
                )
                if first_error is None:
                    first_error = DCUPreflightError(reason)
                continue
            try:
                child_result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                reason = f"concurrency child {metadata['index']} result is unreadable: {exc}"
                _terminalize_concurrency_manifest(
                    child_dir, status="FAIL", returncode=returncode, reason=reason
                )
                if first_error is None:
                    first_error = DCUPreflightError(reason)
                continue
            if not isinstance(child_result, Mapping):
                reason = f"concurrency child {metadata['index']} result is not a mapping"
                _terminalize_concurrency_manifest(
                    child_dir, status="FAIL", returncode=returncode, reason=reason
                )
                if first_error is None:
                    first_error = DCUPreflightError(reason)
                continue
            try:
                validated_result = _validate_concurrency_child_result(
                    child_result,
                    expected_physical_device=int(metadata["physical_device"]),
                    expected_steps=int(steps),
                )
            except DCUPreflightError as exc:
                reason = f"concurrency child {metadata['index']} evidence invalid: {exc}"
                _terminalize_concurrency_manifest(
                    child_dir, status="FAIL", returncode=returncode, reason=reason
                )
                if first_error is None:
                    first_error = DCUPreflightError(reason)
                continue
            _terminalize_concurrency_manifest(
                child_dir, status="PASS", returncode=returncode, reason="completed"
            )
            metadata["manifest"] = str(manifest_path)
            metadata["result"] = validated_result
            metadata.pop("process", None)
            children.append(metadata)
    finally:
        # Every child still carrying a process handle is incomplete.  Its
        # outer process may already have exited while a nested worker keeps
        # the owned process group alive, so cleanup is intentionally not
        # guarded by poll().
        for metadata in processes:
            process = metadata.get("process")
            if process is None:
                continue
            manifest_path = metadata["directory"] / "run_manifest.json"
            existing_status = None
            existing_reason = "concurrency parent cleanup"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest, Mapping):
                    existing_status = manifest.get("status")
                    if manifest.get("termination_reason"):
                        existing_reason = str(manifest["termination_reason"])
            except (OSError, json.JSONDecodeError):
                pass
            if metadata.get("group_cleanup_done") is not True:
                _kill_concurrency_process(metadata, reason="concurrency parent cleanup")
            try:
                _collect_concurrency_process(metadata, timeout=1.0)
            except Exception:
                pass
            status = existing_status if existing_status in {"PASS", "FAIL", "TERMINATED"} else "TERMINATED"
            _terminalize_concurrency_manifest(
                metadata["directory"],
                status=status,
                returncode=metadata.get("returncode"),
                reason=existing_reason,
            )
    elapsed = time.perf_counter() - started
    if first_error is not None:
        raise first_error
    if len(children) != 2:
        raise DCUPreflightError(f"concurrency requires two completed children, got {len(children)}")
    total_env_steps = sum(int(child["result"]["env_step_count"]) for child in children)
    total_policy_decisions = sum(int(child["result"]["decision_count"]) for child in children)
    env_steps_per_second = total_env_steps / elapsed if elapsed else None
    policy_decisions_per_second = total_policy_decisions / elapsed if elapsed else None
    throughput = {
        "workers": 2,
        "wall_seconds": elapsed,
        "workers_per_second": 2.0 / elapsed if elapsed else None,
        "total_env_steps": total_env_steps,
        "total_policy_decisions": total_policy_decisions,
        "env_steps_per_second": env_steps_per_second,
        "policy_decisions_per_second": policy_decisions_per_second,
    }
    return {
        "status": "PASS",
        "workers": 2,
        "physical_devices": devices,
        "logical_device_per_child": EXPECTED_LOGICAL_DEVICE,
        "card0_reason": config["runtime"]["card0_reason"],
        "wall_throughput": throughput,
        "concurrent_throughput": dict(throughput),
        "children": children,
    }


def _summary_value(value: Any) -> Any:
    """JSON-safe evidence conversion that never serializes live runtime objects."""

    if isinstance(value, torch.Tensor):
        return _describe_runtime_value(value)
    if isinstance(value, np.ndarray):
        return _describe_runtime_value(value)
    if isinstance(value, Mapping):
        return {str(key): _summary_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _summary_value(value.item())
    return {"type": type(value).__name__}


def _touch_exclusive(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n"):
        pass
    return target


def _write_text_exclusive(path: str | Path, value: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(str(value))
    return target


def _phase_command(
    phase: str,
    *,
    config_path: str | Path | None,
    expected_project_sha: str,
    physical_device: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str((_ROOT / "scripts" / "dcu_preflight.py").resolve()),
        phase,
    ]
    if config_path is not None:
        command.extend(["--config", str(Path(config_path).resolve())])
    command.extend(["--expected-project-sha", str(expected_project_sha)])
    if physical_device is not None:
        command.extend(["--physical-device", str(physical_device)])
    return command


def _persist_phase_artifacts(run_directory: Path, result: Mapping[str, Any], phase: str) -> None:
    if "comparison" in result:
        _json_dump(run_directory / "comparison.json", result["comparison"])
    if "episode_result" in result:
        _json_dump(run_directory / "episode_result.json", result["episode_result"])
    if "concurrency" in result:
        _json_dump(run_directory / "concurrency.json", result["concurrency"])
    if "children" in result and phase == "concurrency":
        _json_dump(run_directory / "concurrency.json", result)
    if "child_result" in result:
        _json_dump(run_directory / "child_result.json", result["child_result"])
    if "decisions" in result:
        _jsonl_dump(run_directory / "decisions.jsonl", result["decisions"])
    if not any(key in result for key in ("comparison", "episode_result", "children", "child_result", "decisions")):
        _json_dump(run_directory / "phase_result.json", _summary_value(result))


def _artifact_manifest_identity(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    spec = _require_mapping(config[name], name)
    hash_key = "manifest_sha256" if name == "assets" else "model_sha256"
    return {
        "repo_id": str(spec["repo_id"]),
        "revision": str(spec["revision"]),
        "path": str(Path(str(spec["path"])).resolve()),
        "sha256": str(spec[hash_key]),
    }


def _phase_noise_manifest(phase: str, result: Mapping[str, Any]) -> dict[str, Any]:
    if phase == "compare":
        comparison = result.get("comparison")
        comparison = comparison if isinstance(comparison, Mapping) else {}
        return {
            "mode": "explicit_same_noise",
            "seed": EXPECTED_SEED,
            "sha256": comparison.get("noise_sha256"),
            "native_worker_rng": False,
        }
    if phase in {"closed-loop", "one-step-child", "concurrency"}:
        children: list[dict[str, Any]] = []
        if phase == "concurrency":
            raw_children = result.get("children")
            if isinstance(raw_children, Sequence) and not isinstance(raw_children, (str, bytes)):
                for child in raw_children:
                    if isinstance(child, Mapping):
                        children.append(
                            {
                                "physical_device": child.get("physical_device"),
                                "seed": child.get("result", {}).get("seed")
                                if isinstance(child.get("result"), Mapping)
                                else EXPECTED_SEED,
                            }
                        )
        elif phase == "one-step-child":
            child = result.get("child_result")
            children.append(
                {
                    "physical_device": child.get("physical_device")
                    if isinstance(child, Mapping)
                    else None,
                    "seed": child.get("seed") if isinstance(child, Mapping) else EXPECTED_SEED,
                }
            )
        return {
            "mode": "native_worker_torch_rng",
            "seed": EXPECTED_SEED,
            "explicit": False,
            "children": children,
        }
    raise DCUPreflightError(f"unsupported phase for noise manifest: {phase}")


def _phase_manifest_provenance(
    *,
    config: Mapping[str, Any],
    phase: str,
    expected_project_sha: str,
    actual_project_sha: str,
    input_config_path: Path,
    resolved_config_path: Path,
    preflight: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build final provenance without turning configured values into claims."""

    runtime = _require_mapping(config["runtime"], "runtime")
    preflight_mapping = preflight if isinstance(preflight, Mapping) else {}
    result_mapping = result if isinstance(result, Mapping) else {}
    phase_result_name = {
        "compare": "comparison.json",
        "closed-loop": "episode_result.json",
        "concurrency": "concurrency.json",
        "one-step-child": "child_result.json",
    }[phase]
    environment_evidence = preflight_mapping.get("environment")
    worker_evidence = result_mapping.get("worker")
    return {
        "input_config": {
            "path": str(input_config_path.resolve()),
            "sha256": _sha256_file(input_config_path),
        },
        "resolved_config_artifact": {
            "path": str(resolved_config_path.resolve()),
            "sha256": _sha256_file(resolved_config_path),
        },
        "project": {
            "expected_sha": str(expected_project_sha),
            "actual_sha": str(actual_project_sha),
            "code_sha": str(actual_project_sha),
        },
        "runtime_locks": {
            "cpu": {
                "path": str(Path(str(runtime["cpu_runtime_lock"])).resolve()),
                "sha256": str(config["runtime_lock_sha256"]),
            },
            "dcu": {
                "path": str(Path(str(runtime["dcu_runtime_lock"])).resolve()),
                "sha256": str(config["dcu_runtime_lock_sha256"]),
            },
        },
        "artifacts": {
            "checkpoint": _artifact_manifest_identity(config, "checkpoint"),
            "base_model": _artifact_manifest_identity(config, "base_model"),
            "assets": _artifact_manifest_identity(config, "assets"),
        },
        "seed": EXPECTED_SEED,
        "action_noise": _phase_noise_manifest(phase, result_mapping),
        "runtime_hardware_evidence_index": {
            "preflight": "preflight.json",
            "preflight_environment": "preflight.environment",
            "python": "preflight.environment.dcu_runtime_lock.python_executable",
            "pip_freeze": "preflight.environment.dcu_runtime_lock.pip_freeze_all",
            "pip_check": "preflight.environment.dcu_runtime_lock.pip_check",
            "toolchain": "preflight.environment.toolchain",
            "phase_result": phase_result_name,
            "worker": "result.worker" if worker_evidence is not None else None,
            "gl": {
                "compare": "result.comparison.gl",
                "closed-loop": "result.episode_result.gl",
                "one-step-child": "result.child_result.gl",
                "concurrency": "result.children[*].result.gl",
            }[phase],
            "environment_evidence_present": isinstance(environment_evidence, Mapping),
            "worker_evidence_present": worker_evidence is not None,
            "toolchain_evidence": _summary_value(
                environment_evidence.get("toolchain")
                if isinstance(environment_evidence, Mapping)
                else None
            ),
            "worker_torch_runtime": _summary_value(
                worker_evidence.get("torch_runtime")
                if isinstance(worker_evidence, Mapping)
                else None
            ),
        },
    }


def run_phase(
    phase: str,
    *,
    config: Mapping[str, Any],
    expected_project_sha: str,
    actual_project_sha: str,
    output_root: str | Path | None = None,
    config_path: str | Path | None = None,
    run_directory: str | Path | None = None,
    physical_device: int | None = None,
    preflight_gate: Callable[..., Mapping[str, Any]] | None = None,
    phase_runner: Callable[..., Any] | None = None,
    internal_token: str | None = None,
    command: Sequence[str] | None = None,
) -> int:
    """Execute exactly one explicit phase and persist a no-overwrite result."""

    normalized_phase = _normalise_phase(phase)
    if normalized_phase == _ONE_STEP_CHILD_PHASE:
        if internal_token != ONE_STEP_CHILD_TOKEN:
            raise DCUPreflightError("one-step-child requires the private internal token")
        validation_phase = "concurrency"
    else:
        if internal_token is not None:
            raise DCUPreflightError("internal token is only valid for one-step-child")
        validation_phase = normalized_phase
    identity = validate_runner_config(
        config,
        phase=validation_phase,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual_project_sha,
    )
    identity["phase"] = normalized_phase
    if physical_device is not None:
        device = _require_nonnegative_int(physical_device, "physical_device")
        if device not in EXPECTED_PHYSICAL_DEVICES:
            raise DCUPreflightError("physical_device must be one of the pinned [0, 1] devices")
        if normalized_phase in {"compare", "closed-loop"}:
            _phase_physical_device(normalized_phase, device)
    if run_directory is None:
        if output_root is None:
            output_mapping = _require_mapping(config["output"], "output")
            output_root = output_mapping["root"]
        run_dir = create_run_directory(output_root)
    else:
        run_dir = Path(run_directory).resolve()
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=False)
    input_config_path = _config_file_path(config, config_path)
    resolved_config_path = run_dir / "config_resolved.json"
    _json_dump(resolved_config_path, config)
    command_list = list(command or _phase_command(
        normalized_phase,
        config_path=config_path,
        expected_project_sha=expected_project_sha,
        physical_device=physical_device,
    ))
    _touch_exclusive(run_dir / "stdout.log")
    _touch_exclusive(run_dir / "stderr.log")
    with (run_dir / "command.txt").open("x", encoding="utf-8", newline="") as command_handle:
        command_handle.write(shlex.join([str(item) for item in command_list]) + "\n")
    started = time.perf_counter()
    provenance = _phase_manifest_provenance(
        config=config,
        phase=normalized_phase,
        expected_project_sha=expected_project_sha,
        actual_project_sha=actual_project_sha,
        input_config_path=input_config_path,
        resolved_config_path=resolved_config_path,
        preflight=None,
        result=None,
    )
    _json_dump(
        run_dir / "run_manifest.json",
        {
            "status": "RUNNING",
            "phase": normalized_phase,
            "identity": identity,
            "command": [str(item) for item in command_list],
            **provenance,
            "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    )
    preflight: Mapping[str, Any] = {}
    try:
        gate = preflight_gate or run_preflight_gates
        gate_out = _Tee(sys.stdout, run_dir / "stdout.log")
        gate_err = _Tee(sys.stderr, run_dir / "stderr.log")
        try:
            with redirect_stdout(gate_out), redirect_stderr(gate_err):
                preflight = gate(
                    config=config,
                    phase=validation_phase,
                    expected_project_sha=expected_project_sha,
                    actual_project_sha=actual_project_sha,
                    run_directory=run_dir,
                    prepare_assets=normalized_phase != _ONE_STEP_CHILD_PHASE,
                )
        finally:
            gate_out.close()
            gate_err.close()
        if not isinstance(preflight, Mapping):
            raise DCUPreflightError("preflight gate must return a mapping")
        if phase_runner is None:
            runner = {
                "compare": _run_compare_impl,
                "closed-loop": _run_closed_loop_impl,
                "concurrency": _run_concurrency_impl,
                _ONE_STEP_CHILD_PHASE: _run_one_step_child_impl,
            }[normalized_phase]
        else:
            runner = phase_runner
        tee_out = _Tee(sys.stdout, run_dir / "stdout.log")
        tee_err = _Tee(sys.stderr, run_dir / "stderr.log")
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                raw_result = runner(
                    config=config,
                    phase=normalized_phase,
                    run_directory=run_dir,
                    preflight=preflight,
                    config_path=config_path,
                    physical_device=physical_device,
                    expected_project_sha=expected_project_sha,
                    actual_project_sha=actual_project_sha,
                )
        finally:
            tee_out.close()
            tee_err.close()
        if isinstance(raw_result, bool):
            if not raw_result:
                raise DCUPreflightError("phase runner returned false")
            result: dict[str, Any] = {"status": "PASS"}
        elif isinstance(raw_result, int):
            if raw_result != 0:
                raise DCUPreflightError(f"phase runner returned {raw_result}")
            result = {"status": "PASS"}
        elif isinstance(raw_result, Mapping):
            result = dict(raw_result)
            if result.get("status", "PASS") != "PASS":
                raise DCUPreflightError(f"phase runner did not pass: {result.get('status')!r}")
            result["status"] = "PASS"
        else:
            raise DCUPreflightError("phase runner must return a mapping or zero")
        _persist_phase_artifacts(run_dir, result, normalized_phase)
        _json_dump(run_dir / "preflight.json", _summary_value(preflight))
        provenance = _phase_manifest_provenance(
            config=config,
            phase=normalized_phase,
            expected_project_sha=expected_project_sha,
            actual_project_sha=actual_project_sha,
            input_config_path=input_config_path,
            resolved_config_path=resolved_config_path,
            preflight=preflight,
            result=result,
        )
        manifest = {
            "status": "PASS",
            "phase": normalized_phase,
            "identity": identity,
            "command": [str(item) for item in command_list],
            **provenance,
            "preflight": _summary_value(preflight),
            "result": _summary_value(result),
            "elapsed_seconds": time.perf_counter() - started,
            "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        _json_dump(run_dir / "run_manifest.json", manifest)
        return 0
    except BaseException as exc:
        provenance = _phase_manifest_provenance(
            config=config,
            phase=normalized_phase,
            expected_project_sha=expected_project_sha,
            actual_project_sha=actual_project_sha,
            input_config_path=input_config_path,
            resolved_config_path=resolved_config_path,
            preflight=preflight,
            result=None,
        )
        failure_context = {
            "phase": normalized_phase,
            "identity": identity,
            "command": [str(item) for item in command_list],
            "preflight": _summary_value(preflight),
            "elapsed_seconds": time.perf_counter() - started,
            "traceback": traceback.format_exc(),
        }
        try:
            write_failure_manifest(run_dir, exc, context=failure_context)
        except FileExistsError:
            raise
        _json_dump(
            run_dir / "run_manifest.json",
            {
                "status": "FAIL",
                "phase": normalized_phase,
                "identity": identity,
                "command": [str(item) for item in command_list],
                **provenance,
                "failure_manifest": FAILURE_MANIFEST_NAME,
                "elapsed_seconds": time.perf_counter() - started,
                "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        return 1


def run_compare(**kwargs: Any) -> int:
    return run_phase("compare", phase_runner=kwargs.pop("phase_runner", None), **kwargs)


def run_closed_loop(**kwargs: Any) -> int:
    return run_phase("closed-loop", phase_runner=kwargs.pop("phase_runner", None), **kwargs)


def run_concurrency(**kwargs: Any) -> int:
    return run_phase("concurrency", phase_runner=kwargs.pop("phase_runner", None), **kwargs)


def run_one_step_child(**kwargs: Any) -> int:
    token = kwargs.pop("internal_token", None)
    return run_phase("one-step-child", internal_token=token, **kwargs)


def dispatch_phase(phase: str, **kwargs: Any) -> int:
    """Dispatch one requested phase; never invokes a following phase."""

    normalized_phase = _normalise_phase(phase)
    if normalized_phase == "compare":
        return run_compare(**kwargs)
    if normalized_phase == "closed-loop":
        return run_closed_loop(**kwargs)
    if normalized_phase == "concurrency":
        return run_concurrency(**kwargs)
    raise DCUPreflightError("one-step-child is private and cannot be dispatched directly")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=[*PHASES, _ONE_STEP_CHILD_PHASE])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-project-sha", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-directory", type=Path, default=None)
    parser.add_argument("--physical-device", type=int, default=None)
    parser.add_argument("--internal-token", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = _load_yaml_config(args.config)
        actual = _project_head(_ROOT)
        if args.phase == _ONE_STEP_CHILD_PHASE:
            if args.internal_token != ONE_STEP_CHILD_TOKEN:
                raise DCUPreflightError("one-step-child requires the private internal token")
            return run_one_step_child(
                config=config,
                expected_project_sha=args.expected_project_sha,
                actual_project_sha=actual,
                output_root=args.output_root,
                config_path=args.config,
                run_directory=args.run_directory,
                physical_device=args.physical_device,
                internal_token=args.internal_token,
                command=list(argv) if argv is not None else sys.argv[1:],
            )
        if args.internal_token is not None:
            raise DCUPreflightError("internal token is only valid for one-step-child")
        return dispatch_phase(
            args.phase,
            config=config,
            expected_project_sha=args.expected_project_sha,
            actual_project_sha=actual,
            output_root=args.output_root,
            config_path=args.config,
            run_directory=args.run_directory,
            physical_device=args.physical_device,
            command=list(argv) if argv is not None else sys.argv[1:],
        )
    except BaseException as exc:
        print(f"preflight failed before manifest allocation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by explicit phase CLI
    raise SystemExit(main())


__all__ = [
    "DCUPreflightError",
    "FeatureOnlyRemotePolicy",
    "EXPECTED_FLOW_NOISE_SHAPE",
    "EXPECTED_CPU_LOCK_PATH",
    "EXPECTED_CPU_LOCK_SHA256",
    "EXPECTED_DCU_LOCK_PATH",
    "EXPECTED_DCU_LOCK_SHA256",
    "EXPECTED_LIBERO_CONFIG_SHA256",
    "ONE_STEP_CHILD_TOKEN",
    "PHASES",
    "EXPECTED_OVERLAY_WHEEL_SHA256",
    "EXPECTED_PATCH_DELETIONS",
    "EXPECTED_PATCH_FILES",
    "EXPECTED_PATCH_INSERTIONS",
    "EXPECTED_PATCH_SHA256",
    "EXPECTED_WORKER_FORWARD_TIMEOUT_SECONDS",
    "EXPECTED_WORKER_SHUTDOWN_TIMEOUT_SECONDS",
    "EXPECTED_WORKER_STARTUP_TIMEOUT_SECONDS",
    "FAILURE_MANIFEST_NAME",
    "FORBIDDEN_BACKENDS",
    "PATCH_PATH",
    "build_worker_environment",
    "build_cpu_child_environment",
    "build_cpu_environment_runtime",
    "build_concurrency_child_command",
    "build_cpu_runtime",
    "compare_outputs",
    "create_run_directory",
    "enforce_one_step_boundary",
    "generate_flow_noise",
    "map_physical_to_logical",
    "dispatch_phase",
    "main",
    "run_closed_loop",
    "run_compare",
    "run_concurrency",
    "run_one_step_child",
    "run_phase",
    "run_preflight_gates",
    "validate_runner_config",
    "validate_config_identity",
    "validate_forbidden_backends",
    "validate_patch",
    "write_failure_manifest",
]
