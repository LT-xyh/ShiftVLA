"""Isolated, fail-closed SmolVLA model worker for the DCU process boundary.

The process consumes one canonical JSON object per line on stdin and emits one
canonical JSON object per line on stdout.  Tensor payloads stay in local
``safetensors`` files; only their paths and JSON-safe execution evidence cross
the control boundary.  No simulator or observation preprocessing is owned by
this module: callers must provide the feature tensors expected by the pinned
SmolVLA policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, TextIO

# ``python scripts/dcu_model_worker.py`` places ``scripts/`` (rather than the
# repository root) on ``sys.path``.  Add the root only for this direct CLI
# form so the project-owned transport module remains importable without an
# ambient PYTHONPATH.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.dcu_worker import (
    DCUPreflightError,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    decode_json_line,
    encode_json_line,
    load_tensor_bundle,
    save_tensor_bundle,
)


# Keep the wire schemas separate even though their tensor specifications are
# shared.  In particular, a queue-native select_action request must not carry
# a caller-supplied flow noise tensor.
FEATURE_SCHEMA: dict[str, Any] = {
    key: value for key, value in REQUEST_SCHEMA.items() if key != "noise"
}
PREDICT_REQUEST_SCHEMA: dict[str, Any] = dict(REQUEST_SCHEMA)
SELECT_REQUEST_SCHEMA: dict[str, Any] = dict(FEATURE_SCHEMA)
NOISE_SCHEMA: dict[str, Any] = {"noise": REQUEST_SCHEMA["noise"]}

EXPECTED_DEVICE = "cuda:0"
EXPECTED_ACTION_SHAPE = (1, 7)
EXPECTED_CHUNK_SHAPE = (1, 50, 7)
_FORBIDDEN_BACKENDS = frozenset(
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
_COMMANDS = frozenset(
    {"reset", "predict_action_chunk", "select_action", "ping", "shutdown", "close"}
)


def _error(message: str) -> DCUPreflightError:
    return DCUPreflightError(message)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise _error(f"missing required config field {path}")
        current = current[component]
    return current


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(f"{path} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(f"{path} must be a non-negative integer")
    return int(value)


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{path} must be a non-empty string")
    return value


def _local_path(
    value: Any,
    path: str,
    *,
    require_directory: bool = True,
    check_exists: bool = True,
) -> Path:
    raw = _nonempty_string(value, path)
    if "://" in raw:
        raise _error(f"{path} must be an exact local path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise _error(f"{path} must be an absolute local path")
    if check_exists and require_directory and not candidate.is_dir():
        raise _error(f"{path} is not a local directory: {candidate}")
    if check_exists and not require_directory and not candidate.is_file():
        raise _error(f"{path} is not a local file: {candidate}")
    return candidate.resolve()


def load_worker_config(path: str | Path) -> dict[str, Any]:
    """Load one local YAML mapping without contacting the Hub."""

    config_path = Path(path)
    if not config_path.is_file():
        raise _error(f"worker config does not exist as a regular file: {config_path}")
    try:
        import yaml

        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _error(f"could not load worker config {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise _error("worker config must decode to a mapping")
    return dict(value)


def validate_worker_config(config: Mapping[str, Any], *, check_paths: bool = True) -> dict[str, Any]:
    """Validate the immutable local-path and shape contract used by the CLI."""

    root = _require_mapping(config, "config")
    if root.get("schema_version") != 1:
        raise _error(f"schema_version must be 1, got {root.get('schema_version')!r}")
    if root.get("name") != "dcu_preflight":
        raise _error("config name must be dcu_preflight")

    seed = _nonnegative_int(_required(root, "seed"), "seed")
    action_dim = _positive_int(_required(root, "action_dim"), "action_dim")
    chunk_size = _positive_int(_required(root, "chunk_size"), "chunk_size")
    n_action_steps = _positive_int(_required(root, "n_action_steps"), "n_action_steps")
    if (action_dim, chunk_size) != (7, 50):
        raise _error("worker action contract must be action_dim=7 and chunk_size=50")
    if n_action_steps > chunk_size:
        raise _error("n_action_steps cannot exceed chunk_size")

    task = _require_mapping(_required(root, "task"), "task")
    suite = _nonempty_string(_required(task, "suite"), "task.suite")
    task_id = _nonnegative_int(_required(task, "task_id"), "task.task_id")
    observation_height = _positive_int(
        _required(task, "observation_height"), "task.observation_height"
    )
    observation_width = _positive_int(
        _required(task, "observation_width"), "task.observation_width"
    )
    for field, nested_value in (
        ("suite", suite),
        ("task_id", task_id),
        ("seed", seed),
        ("observation_height", observation_height),
        ("observation_width", observation_width),
    ):
        if field in root and root[field] != nested_value:
            raise _error(f"config identity mismatch: {field} != task.{field}")

    checkpoint = _require_mapping(_required(root, "checkpoint"), "checkpoint")
    checkpoint_path = _local_path(
        _required(checkpoint, "path"),
        "checkpoint.path",
        require_directory=True,
        check_exists=check_paths,
    )
    checkpoint_revision = _nonempty_string(
        _required(checkpoint, "revision"), "checkpoint.revision"
    )
    _nonempty_string(_required(checkpoint, "repo_id"), "checkpoint.repo_id")
    model_file = checkpoint_path / "model.safetensors"
    if check_paths and not model_file.is_file():
        raise _error(f"checkpoint.model.safetensors is missing: {model_file}")

    base_model = _require_mapping(_required(root, "base_model"), "base_model")
    base_model_path = _local_path(
        _required(base_model, "path"),
        "base_model.path",
        require_directory=True,
        check_exists=check_paths,
    )
    base_model_revision = _nonempty_string(
        _required(base_model, "revision"), "base_model.revision"
    )
    _nonempty_string(_required(base_model, "repo_id"), "base_model.repo_id")

    runtime = _require_mapping(_required(root, "runtime"), "runtime")
    logical_device = _nonempty_string(
        _required(runtime, "logical_device"), "runtime.logical_device"
    )
    if logical_device != EXPECTED_DEVICE:
        raise _error(f"runtime.logical_device must be {EXPECTED_DEVICE}")
    offline = _require_mapping(_required(root, "offline"), "offline")
    offline_values: dict[str, str] = {}
    for key, value in offline.items():
        if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
            raise _error("offline environment keys and values must be non-empty strings")
        offline_values[key] = value
    if not offline_values:
        raise _error("offline environment cannot be empty")

    return {
        "seed": seed,
        "suite": suite,
        "task_id": task_id,
        "observation_height": observation_height,
        "observation_width": observation_width,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "checkpoint_path": checkpoint_path,
        "checkpoint_revision": checkpoint_revision,
        "base_model_path": base_model_path,
        "base_model_revision": base_model_revision,
        "offline": offline_values,
        "logical_device": logical_device,
    }


def validate_offline_environment(
    config_or_environment: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Require every configured offline flag to be present with exact value."""

    mapping = _require_mapping(config_or_environment, "offline environment")
    expected_raw: Any = mapping.get("offline", mapping)
    expected = _require_mapping(expected_raw, "offline")
    actual = os.environ if environment is None else environment
    if not isinstance(actual, Mapping):
        raise _error("environment must be a mapping")
    result: dict[str, str] = {}
    for key, value in expected.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise _error("offline environment keys and values must be strings")
        observed = actual.get(key)
        if observed != value:
            raise _error(f"offline environment {key} must equal {value!r}, got {observed!r}")
        result[key] = observed
    return result


class TorchDeviceAdapter:
    """Small injectable adapter around the selected logical accelerator."""

    def is_available(self, device: str) -> bool:
        return bool(device == EXPECTED_DEVICE and torch.cuda.is_available())

    def device_count(self) -> int:
        return int(torch.cuda.device_count())

    def get_device_name(self, index: int) -> str:
        return str(torch.cuda.get_device_name(index))

    def torch_hip_version(self) -> str | None:
        return torch.version.hip

    def torch_cuda_version(self) -> str | None:
        return torch.version.cuda

    def to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        return tensor.to(device)

    def synchronize(self) -> None:
        torch.cuda.synchronize(0)

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats(0)

    def max_memory_allocated(self) -> int:
        return int(torch.cuda.max_memory_allocated(0))

    def manual_seed_all(self, seed: int) -> None:
        torch.cuda.manual_seed_all(seed)

    def basic_tensor_ops(self) -> dict[str, dict[str, Any]]:
        """Run small eager FP32 and BF16 matmul-plus-add probes on cuda:0."""

        fp32_left = torch.ones((2, 2), dtype=torch.float32, device=EXPECTED_DEVICE)
        fp32_right = torch.eye(2, dtype=torch.float32, device=EXPECTED_DEVICE)
        fp32_result = torch.matmul(fp32_left, fp32_right) + fp32_left

        bf16_left = torch.ones((2, 2), dtype=torch.bfloat16, device=EXPECTED_DEVICE)
        bf16_right = torch.eye(2, dtype=torch.bfloat16, device=EXPECTED_DEVICE)
        bf16_result = torch.matmul(bf16_left, bf16_right) + bf16_left

        self.synchronize()
        return {
            "fp32": {
                "shape": list(fp32_result.shape),
                "dtype": str(fp32_result.dtype),
                "finite": bool(torch.isfinite(fp32_result).all().item()),
                "device": str(fp32_result.device),
            },
            "bf16": {
                "shape": list(bf16_result.shape),
                "dtype": str(bf16_result.dtype),
                "finite": bool(torch.isfinite(bf16_result).all().item()),
                "device": str(bf16_result.device),
            },
        }


def validate_device(
    device_adapter: Any,
    device: str = EXPECTED_DEVICE,
) -> dict[str, Any]:
    """Require one visible K100 accelerator as logical ``cuda:0``."""

    if device != EXPECTED_DEVICE:
        raise _error(f"worker device must be exactly {EXPECTED_DEVICE}")
    try:
        available = bool(device_adapter.is_available(device))
        count = int(device_adapter.device_count())
        name = str(device_adapter.get_device_name(0))
    except Exception as exc:
        raise _error(f"could not inspect {EXPECTED_DEVICE}: {exc}") from exc
    if not available or count < 1:
        raise _error(f"{EXPECTED_DEVICE} is not available")
    if "k100" not in name.lower():
        raise _error(f"visible device is not a K100: {name}")
    return {
        "device": EXPECTED_DEVICE,
        "device_index": 0,
        "device_count": count,
        "device_name": name,
        "available": available,
    }


def _seed_torch(seed: int, device_adapter: Any | None = None) -> None:
    normalized = _nonnegative_int(seed, "seed")
    torch.manual_seed(normalized)
    manual_seed_all = getattr(device_adapter, "manual_seed_all", None)
    if callable(manual_seed_all):
        try:
            manual_seed_all(normalized)
        except Exception as exc:
            raise _error(f"could not seed accelerator RNG: {exc}") from exc
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(normalized)


def _forbidden_modules() -> list[str]:
    names = {
        str(name).split(".", 1)[0].lower()
        for name in sys.modules
        if isinstance(name, str)
    }
    return sorted(name for name in names if name in _FORBIDDEN_BACKENDS)


def _model_parameter_evidence(policy: Any) -> dict[str, Any]:
    """Summarize parameters and fail closed on any non-cuda:0 parameter."""

    parameters_method = getattr(policy, "parameters", None)
    if not callable(parameters_method):
        raise _error("loaded policy must provide parameters()")
    try:
        parameters = tuple(parameters_method())
    except Exception as exc:
        raise _error(f"could not inspect loaded policy parameters: {exc}") from exc

    parameter_count = 0
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    for parameter in parameters:
        try:
            count = int(parameter.numel())
            dtype = str(parameter.dtype)
            device = str(parameter.device)
        except Exception as exc:
            raise _error(f"loaded policy parameter metadata is invalid: {exc}") from exc
        if device != EXPECTED_DEVICE:
            raise _error(
                f"loaded policy parameter is on {device}; all parameters must be on {EXPECTED_DEVICE}"
            )
        parameter_count += count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        device_counts[device] = device_counts.get(device, 0) + 1
    return {
        "parameter_count": parameter_count,
        "parameter_dtype_counts": dtype_counts,
        "parameter_device_counts": device_counts,
        # Keep short aliases in the evidence for consumers that use the
        # generic dtype_counts/device_counts terminology.
        "dtype_counts": dict(dtype_counts),
        "device_counts": dict(device_counts),
    }


def _callable_or_import(
    value: Any,
    module_name: str,
    attr_name: str,
) -> Any:
    if value is not None:
        return value
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def build_official_policy(
    config: Mapping[str, Any],
    *,
    device_adapter: Any | None = None,
    environment: Mapping[str, str] | None = None,
    make_env_config_fn: Callable[..., Any] | None = None,
    make_policy_fn: Callable[..., Any] | None = None,
    smolvla_config_cls: Any | None = None,
    smolvla_policy_cls: Any | None = None,
    stderr: TextIO | None = None,
) -> Any:
    """Construct the pinned policy through LeRobot's official factories.

    The optional callables are dependency-injection seams for fake-only tests;
    the default path imports only the pinned LeRobot modules.  No benchmark
    environment is constructed; callers supply feature tensors directly.
    """

    log = sys.stderr if stderr is None else stderr
    validated = validate_worker_config(config, check_paths=True)
    validate_offline_environment(config, environment=environment)
    adapter = TorchDeviceAdapter() if device_adapter is None else device_adapter
    device_evidence = validate_device(adapter, EXPECTED_DEVICE)
    forbidden = _forbidden_modules()
    if forbidden:
        raise _error(f"forbidden backend imported: {', '.join(forbidden)}")
    _seed_torch(validated["seed"], adapter)
    synchronize = getattr(adapter, "synchronize", None)
    reset_peak = getattr(adapter, "reset_peak_memory_stats", None)
    max_memory = getattr(adapter, "max_memory_allocated", None)

    with redirect_stdout(log):
        if callable(synchronize):
            synchronize()
        if callable(reset_peak):
            reset_peak()
        load_started = time.perf_counter()
        if make_env_config_fn is None:
            make_env_config_fn = _callable_or_import(None, "lerobot.envs", "make_env_config")
        if make_policy_fn is None:
            make_policy_fn = _callable_or_import(None, "lerobot.policies.factory", "make_policy")
        if smolvla_config_cls is None:
            module = __import__(
                "lerobot.policies.smolvla.configuration_smolvla",
                fromlist=["SmolVLAConfig"],
            )
            smolvla_config_cls = getattr(module, "SmolVLAConfig")
        if smolvla_policy_cls is None:
            module = __import__(
                "lerobot.policies.smolvla.modeling_smolvla",
                fromlist=["SmolVLAPolicy"],
            )
            smolvla_policy_cls = getattr(module, "SmolVLAPolicy")

        env_cfg = make_env_config_fn(
            "libero",
            task=validated["suite"],
            task_ids=[validated["task_id"]],
            obs_type=str(config.get("obs_type", "pixels_agent_pos")),
            render_mode="rgb_array",
            init_states=True,
            hard_reset=True,
            observation_height=validated["observation_height"],
            observation_width=validated["observation_width"],
        )
        policy_cfg = smolvla_config_cls.from_pretrained(
            str(validated["checkpoint_path"]),
            revision=validated["checkpoint_revision"],
            local_files_only=True,
        )
        # The base VLM is addressed by an exact local path.  The revision is
        # retained as explicit evidence because v0.6.1 does not forward it to
        # the nested Transformers loader.
        policy_cfg.device = EXPECTED_DEVICE
        policy_cfg.vlm_model_name = str(validated["base_model_path"])
        policy_cfg.pretrained_path = validated["checkpoint_path"]
        policy_cfg.pretrained_revision = validated["checkpoint_revision"]
        policy_cfg.base_model_revision = validated["base_model_revision"]
        policy_cfg.use_peft = False
        policy_cfg.use_amp = False
        if hasattr(policy_cfg, "compile_model"):
            policy_cfg.compile_model = False
        if int(getattr(policy_cfg, "chunk_size", validated["chunk_size"])) != validated["chunk_size"]:
            raise _error("SmolVLA config chunk_size does not match worker config")
        if int(getattr(policy_cfg, "n_action_steps", validated["n_action_steps"])) != validated["n_action_steps"]:
            raise _error("SmolVLA config n_action_steps does not match worker config")
        policy = make_policy_fn(cfg=policy_cfg, env_cfg=env_cfg, rename_map=None)
        reload_file = validated["checkpoint_path"] / "model.safetensors"
        smolvla_policy_cls._load_as_safetensor(
            policy,
            str(reload_file),
            EXPECTED_DEVICE,
            strict=True,
        )
        eval_method = getattr(policy, "eval", None)
        if callable(eval_method):
            eval_method()
        if callable(synchronize):
            synchronize()
        load_latency = time.perf_counter() - load_started
        peak_memory = int(max_memory()) if callable(max_memory) else 0

    forbidden_after = _forbidden_modules()
    if forbidden_after:
        raise _error(f"forbidden backend imported: {', '.join(forbidden_after)}")
    if bool(getattr(policy, "training", False)):
        raise _error("loaded policy must be in eval mode")
    parameter_evidence = _model_parameter_evidence(policy)
    adapter_hip = getattr(adapter, "torch_hip_version", None)
    adapter_cuda = getattr(adapter, "torch_cuda_version", None)
    hip_version = adapter_hip() if callable(adapter_hip) else torch.version.hip
    cuda_version = adapter_cuda() if callable(adapter_cuda) else torch.version.cuda
    policy_evidence = {
        "policy_type": type(policy).__name__,
        "checkpoint_path": str(validated["checkpoint_path"]),
        "checkpoint_revision": validated["checkpoint_revision"],
        "base_model_path": str(validated["base_model_path"]),
        "base_model_revision": validated["base_model_revision"],
        "device": EXPECTED_DEVICE,
        "strict_checkpoint_load": True,
        "use_peft": bool(getattr(policy_cfg, "use_peft", False)),
        "use_amp": bool(getattr(policy_cfg, "use_amp", False)),
        "compile_model": bool(getattr(policy_cfg, "compile_model", False)),
        "training": bool(getattr(policy, "training", False)),
        "chunk_size": int(getattr(policy_cfg, "chunk_size", 50)),
        "n_action_steps": int(getattr(policy_cfg, "n_action_steps", 1)),
        "device_evidence": device_evidence,
        "load_latency_seconds": float(load_latency),
        "peak_memory_bytes": peak_memory,
        "torch_version": str(torch.__version__),
        "torch_hip_version": hip_version,
        "torch_cuda_version": cuda_version,
        **parameter_evidence,
    }
    setattr(policy, "_dcu_model_evidence", policy_evidence)
    return policy


def _tensor_dtypes(bundle: Mapping[str, torch.Tensor]) -> dict[str, str]:
    return {str(key): str(value.dtype) for key, value in bundle.items()}


def _run_basic_tensor_ops(device_adapter: Any) -> dict[str, dict[str, Any]]:
    """Run and validate the mandatory eager runtime probe through the adapter."""

    basic_tensor_ops = getattr(device_adapter, "basic_tensor_ops", None)
    if not callable(basic_tensor_ops):
        raise _error("device adapter must provide basic_tensor_ops()")
    synchronize = getattr(device_adapter, "synchronize", None)
    if not callable(synchronize):
        raise _error("device adapter must provide synchronize() for basic_tensor_ops")
    try:
        synchronize()
        raw_evidence = basic_tensor_ops()
        synchronize()
    except DCUPreflightError:
        raise
    except Exception as exc:
        raise _error(f"basic_tensor_ops failed: {exc}") from exc
    if not isinstance(raw_evidence, Mapping):
        raise _error("basic_tensor_ops must return a mapping")

    expected_dtypes = {"fp32": "torch.float32", "bf16": "torch.bfloat16"}
    evidence: dict[str, dict[str, Any]] = {}
    for name, expected_dtype in expected_dtypes.items():
        entry = raw_evidence.get(name)
        if not isinstance(entry, Mapping):
            raise _error(f"basic_tensor_ops evidence missing {name} mapping")
        raw_shape = entry.get("shape")
        if not isinstance(raw_shape, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw_shape
        ):
            raise _error(f"basic_tensor_ops {name} shape must be an integer sequence")
        shape = [int(item) for item in raw_shape]
        if shape != [2, 2]:
            raise _error(f"basic_tensor_ops {name} shape must be [2, 2], got {shape}")
        dtype = str(entry.get("dtype"))
        if dtype != expected_dtype:
            raise _error(
                f"basic_tensor_ops {name} dtype must be {expected_dtype}, got {dtype}"
            )
        if entry.get("finite") is not True:
            raise _error(f"basic_tensor_ops {name} result must be finite")
        device = str(entry.get("device", EXPECTED_DEVICE))
        if device != EXPECTED_DEVICE:
            raise _error(
                f"basic_tensor_ops {name} ran on {device}; expected {EXPECTED_DEVICE}"
            )
        evidence[name] = {
            "shape": shape,
            "dtype": dtype,
            "finite": True,
            "device": device,
        }
    return evidence


def _queue_length(policy: Any) -> int | None:
    queues = getattr(policy, "_queues", None)
    if not isinstance(queues, Mapping):
        return None
    try:
        action_queue = queues.get("action")
        return None if action_queue is None else int(len(action_queue))
    except (TypeError, ValueError):
        return None


def _validate_seed_message(value: Any, default: int) -> int:
    if value is None:
        return default
    return _nonnegative_int(value, "reset.seed")


@dataclass
class DCUModelWorker:
    """Model-side command handler with explicit reset and queue semantics."""

    policy: Any
    device_adapter: Any | None = None
    seed: int = 2027
    config: Mapping[str, Any] | None = None
    stderr: TextIO | None = None
    environment: Mapping[str, str] | None = None
    device: str = EXPECTED_DEVICE

    def __post_init__(self) -> None:
        self.device_adapter = TorchDeviceAdapter() if self.device_adapter is None else self.device_adapter
        self.stderr = sys.stderr if self.stderr is None else self.stderr
        self.seed = _nonnegative_int(self.seed, "seed")
        self.closed = False
        self._reset_done = False
        if self.config is not None:
            validate_worker_config(self.config, check_paths=True)
            validate_offline_environment(self.config, environment=self.environment)
        with redirect_stdout(self.stderr):
            validate_device(self.device_adapter, self.device)
            forbidden = _forbidden_modules()
            if forbidden:
                raise _error(f"forbidden backend imported: {', '.join(forbidden)}")
            eval_method = getattr(self.policy, "eval", None)
            if callable(eval_method):
                eval_method()
            _seed_torch(self.seed, self.device_adapter)
            self._startup_basic_tensor_ops = _run_basic_tensor_ops(self.device_adapter)

    def _require_open(self) -> None:
        if self.closed:
            raise _error("worker is shut down")

    def _require_reset(self) -> None:
        self._require_open()
        if not self._reset_done:
            raise _error("worker must be reset before requesting an action")

    def _move(self, bundle: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        moved: dict[str, torch.Tensor] = {}
        for key, tensor in bundle.items():
            moved[str(key)] = self.device_adapter.to_device(tensor, EXPECTED_DEVICE)
        return moved

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        self._require_open()
        normalized_seed = _validate_seed_message(seed, self.seed)
        before = _queue_length(self.policy)
        _seed_torch(normalized_seed, self.device_adapter)
        reset_method = getattr(self.policy, "reset", None)
        if not callable(reset_method):
            raise _error("policy must provide reset()")
        with redirect_stdout(self.stderr):
            reset_method()
        # Reseed after reset as well: the official reset only clears queues,
        # but this keeps the request boundary deterministic for fake policies
        # and future compatible implementations.
        _seed_torch(normalized_seed, self.device_adapter)
        after = _queue_length(self.policy)
        if after is not None and after != 0:
            raise _error(f"policy reset left a non-empty action queue: {after}")
        self.seed = normalized_seed
        self._reset_done = True
        return {
            "ok": True,
            "seed": normalized_seed,
            "queue_length_before": before,
            "queue_length_after": after,
            "queue_length": after,
        }

    def _request_bundle(
        self,
        message: Mapping[str, Any],
        *,
        command: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        request_path = message.get("request_path")
        features_path = message.get("features_path", message.get("feature_path"))
        noise_path = message.get("noise_path")
        if command == "select_action" and noise_path is not None:
            raise _error("select_action requires a feature-only request")
        if request_path is not None and not isinstance(request_path, str):
            raise _error("request_path must be a string")
        if features_path is not None and not isinstance(features_path, str):
            raise _error("features_path must be a string")
        if noise_path is not None and not isinstance(noise_path, str):
            raise _error("noise_path must be a string")

        if request_path is not None and features_path is not None:
            raise _error("request_path and features_path are mutually exclusive")

        if request_path is not None:
            if command == "select_action":
                # Accept either shape long enough to give a precise contract
                # error when a caller accidentally supplies the predict bundle.
                features = load_tensor_bundle(
                    request_path,
                    schema={"__alternatives__": [SELECT_REQUEST_SCHEMA, PREDICT_REQUEST_SCHEMA]},
                )
                if "noise" in features:
                    raise _error("select_action requires a feature-only request")
            elif noise_path is None:
                features = load_tensor_bundle(request_path, schema=PREDICT_REQUEST_SCHEMA)
            else:
                features = load_tensor_bundle(request_path, schema=SELECT_REQUEST_SCHEMA)
                noise = load_tensor_bundle(noise_path, schema=NOISE_SCHEMA)
                return features, noise
            if command == "predict_action_chunk":
                return ({key: value for key, value in features.items() if key != "noise"}, {"noise": features["noise"]})
            return features, {}
        if features_path is None:
            raise _error(f"{command} requires request_path or features_path")
        features = load_tensor_bundle(features_path, schema=SELECT_REQUEST_SCHEMA)
        if command == "select_action":
            return features, {}
        if noise_path is None:
            raise _error("predict_action_chunk requires explicit noise")
        noise = load_tensor_bundle(noise_path, schema=NOISE_SCHEMA)
        return features, noise

    def predict_action_chunk(self, message: Mapping[str, Any], *, response_path: Path) -> dict[str, Any]:
        self._require_reset()
        features, noise_bundle = self._request_bundle(message, command="predict_action_chunk")
        batch = self._move(features)
        noise = self._move(noise_bundle)["noise"]
        input_bundle = {**batch, "noise": noise}
        queue_before = _queue_length(self.policy)
        synchronize = getattr(self.device_adapter, "synchronize", None)
        reset_peak = getattr(self.device_adapter, "reset_peak_memory_stats", None)
        max_memory = getattr(self.device_adapter, "max_memory_allocated", None)
        with redirect_stdout(self.stderr):
            if callable(synchronize):
                synchronize()
            if callable(reset_peak):
                reset_peak()
            started = time.perf_counter()
            with torch.inference_mode():
                # Deliberately pass the same explicit tensor to the official API.
                result = self.policy.predict_action_chunk(batch, noise=noise)
            if callable(synchronize):
                synchronize()
            latency = time.perf_counter() - started
            peak_memory = int(max_memory()) if callable(max_memory) else 0
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != EXPECTED_CHUNK_SHAPE:
            shape = getattr(result, "shape", None)
            raise _error(f"predict_action_chunk output shape {shape} does not match {EXPECTED_CHUNK_SHAPE}")
        if result.dtype not in (torch.float32, torch.bfloat16) or not bool(torch.isfinite(result).all().item()):
            raise _error("predict_action_chunk output must be finite float32 or bfloat16")
        save_tensor_bundle(response_path, {"action_chunk": result}, schema=RESPONSE_SCHEMA)
        queue_after = _queue_length(self.policy)
        return {
            "ok": True,
            "response_path": str(response_path.resolve()),
            "action_chunk_shape": list(EXPECTED_CHUNK_SHAPE),
            "output_dtype": str(result.dtype),
            "input_dtypes": _tensor_dtypes(input_bundle),
            "dtypes": {"inputs": _tensor_dtypes(input_bundle), "output": str(result.dtype)},
            "latency_seconds": float(latency),
            "peak_memory_bytes": peak_memory,
            "queue_length_before": queue_before,
            "queue_length_after": queue_after,
            "new_chunk_generated": True,
        }

    def select_action(self, message: Mapping[str, Any], *, response_path: Path) -> dict[str, Any]:
        self._require_reset()
        features, noise_bundle = self._request_bundle(message, command="select_action")
        if noise_bundle:
            raise _error("select_action requires a feature-only request")
        batch = self._move(features)
        queue_before = _queue_length(self.policy)
        synchronize = getattr(self.device_adapter, "synchronize", None)
        reset_peak = getattr(self.device_adapter, "reset_peak_memory_stats", None)
        max_memory = getattr(self.device_adapter, "max_memory_allocated", None)
        with redirect_stdout(self.stderr):
            if callable(synchronize):
                synchronize()
            if callable(reset_peak):
                reset_peak()
            started = time.perf_counter()
            with torch.inference_mode():
                # Do not pass a noise keyword: this preserves the official queue
                # and native RNG behavior for closed-loop action selection.
                result = self.policy.select_action(batch)
            if callable(synchronize):
                synchronize()
            latency = time.perf_counter() - started
            peak_memory = int(max_memory()) if callable(max_memory) else 0
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != EXPECTED_ACTION_SHAPE:
            shape = getattr(result, "shape", None)
            raise _error(f"select_action output shape {shape} does not match {EXPECTED_ACTION_SHAPE}")
        if result.dtype not in (torch.float32, torch.bfloat16) or not bool(torch.isfinite(result).all().item()):
            raise _error("select_action output must be finite float32 or bfloat16")
        save_tensor_bundle(response_path, {"action": result}, schema=RESPONSE_SCHEMA)
        queue_after = _queue_length(self.policy)
        generated = queue_before == 0 if queue_before is not None else None
        return {
            "ok": True,
            "response_path": str(response_path.resolve()),
            "action_shape": list(EXPECTED_ACTION_SHAPE),
            "output_dtype": str(result.dtype),
            "input_dtypes": _tensor_dtypes(batch),
            "dtypes": {"inputs": _tensor_dtypes(batch), "output": str(result.dtype)},
            "latency_seconds": float(latency),
            "peak_memory_bytes": peak_memory,
            "queue_length_before": queue_before,
            "queue_length_after": queue_after,
            "new_chunk_generated": generated if generated is not None else False,
        }

    def ping(self) -> dict[str, Any]:
        self._require_open()
        with redirect_stdout(self.stderr):
            device_evidence = validate_device(self.device_adapter, EXPECTED_DEVICE)
            basic_tensor_ops = _run_basic_tensor_ops(self.device_adapter)
            adapter_hip = getattr(self.device_adapter, "torch_hip_version", None)
            adapter_cuda = getattr(self.device_adapter, "torch_cuda_version", None)
            hip_version = adapter_hip() if callable(adapter_hip) else torch.version.hip
            cuda_version = adapter_cuda() if callable(adapter_cuda) else torch.version.cuda
        evidence = {
            **device_evidence,
            "torch_hip_version": hip_version,
            "torch_cuda_version": cuda_version,
            "torch_cuda_available": bool(device_evidence["available"]),
            "torch_device_count": int(device_evidence["device_count"]),
            "queue_length": _queue_length(self.policy),
            "policy_type": type(self.policy).__name__,
            "policy_training": bool(getattr(self.policy, "training", False)),
            "basic_tensor_ops": basic_tensor_ops,
            "startup_basic_tensor_ops": dict(self._startup_basic_tensor_ops),
        }
        model_evidence = getattr(self.policy, "_dcu_model_evidence", None)
        if isinstance(model_evidence, Mapping):
            evidence["model"] = dict(model_evidence)
        else:
            evidence["model"] = {
                "policy_type": type(self.policy).__name__,
                "training": bool(getattr(self.policy, "training", False)),
            }
        return {
            "ok": True,
            "evidence": evidence,
            "torch_hip_version": hip_version,
            "torch_cuda_version": cuda_version,
            "device": EXPECTED_DEVICE,
            "device_name": device_evidence["device_name"],
            "model": evidence["model"],
            "basic_tensor_ops": basic_tensor_ops,
            "startup_basic_tensor_ops": dict(self._startup_basic_tensor_ops),
            "seed": self.seed,
        }

    def shutdown(self) -> dict[str, Any]:
        if not self.closed:
            synchronize = getattr(self.device_adapter, "synchronize", None)
            with redirect_stdout(self.stderr):
                if callable(synchronize):
                    synchronize()
            self.closed = True
        return {"ok": True, "shutdown": True}

    close = shutdown

    def handle(self, message: Mapping[str, Any], *, response_path: Path | None = None) -> dict[str, Any]:
        self._require_open()
        if not isinstance(message, Mapping):
            raise _error("worker message must be a JSON object")
        identifier = message.get("id")
        command = message.get("command")
        if not isinstance(identifier, str) or not identifier:
            raise _error("worker message requires a non-empty string id")
        if not isinstance(command, str) or command not in _COMMANDS:
            raise _error("worker message requires a supported command")
        if command == "reset":
            payload = self.reset(seed=message.get("seed"))
        elif command == "ping":
            payload = self.ping()
        elif command in {"shutdown", "close"}:
            payload = self.shutdown()
        else:
            if response_path is None:
                raise _error(f"{command} requires a response path")
            if response_path.exists():
                raise FileExistsError(response_path)
            if command == "predict_action_chunk":
                payload = self.predict_action_chunk(message, response_path=response_path)
            else:
                payload = self.select_action(message, response_path=response_path)
        return {"id": identifier, "command": command, **payload}


def _safe_identifier(identifier: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("._")
    return value or "request"


class JSONLWorkerServer:
    """stdio adapter that keeps protocol output isolated from diagnostics."""

    def __init__(
        self,
        worker: DCUModelWorker,
        *,
        response_dir: str | Path,
    ) -> None:
        self.worker = worker
        response_root = Path(response_dir)
        response_root.mkdir(parents=True, exist_ok=True)
        self.response_dir = response_root.resolve()
        self._response_counter = 0

    def _allocate_response_path(self, command: str, identifier: str) -> Path:
        stem = f"{command}-{_safe_identifier(identifier)}"
        while True:
            suffix = self._response_counter
            self._response_counter += 1
            candidate = self.response_dir / f"{stem}-{suffix}.safetensors"
            if not candidate.exists():
                break
        if candidate.exists():
            raise FileExistsError(candidate)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def _contain_ipc_paths(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve every tensor input under the worker-owned IPC directory."""

        normalized = dict(message)
        for field in ("request_path", "features_path", "feature_path", "noise_path"):
            if field not in normalized:
                continue
            raw = normalized[field]
            if not isinstance(raw, str) or not raw:
                raise _error(f"{field} must be a string path")
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.response_dir / candidate
            resolved = candidate.resolve()
            try:
                resolved.relative_to(self.response_dir)
            except ValueError as exc:
                raise _error(
                    f"{field} must resolve under configured response_dir {self.response_dir}"
                ) from exc
            normalized[field] = str(resolved)
        return normalized

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(message, Mapping):
            return self.worker.handle(message)
        if "response_path" in message:
            raise _error("response_path is worker-owned")
        normalized = self._contain_ipc_paths(message)
        command = normalized.get("command")
        identifier = normalized.get("id")
        if command in {"predict_action_chunk", "select_action"}:
            response_path = self._allocate_response_path(str(command), str(identifier))
            return self.worker.handle(normalized, response_path=response_path)
        return self.worker.handle(normalized)

    def run(
        self,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
    ) -> int:
        self.worker.stderr = stderr
        for raw_line in stdin:
            try:
                message = decode_json_line(raw_line)
                response = self.handle(message)
                stdout.write(encode_json_line(response))
                stdout.flush()
            except Exception as exc:
                identifier = None
                parsed: Any = None
                try:
                    if isinstance(raw_line, str):
                        parsed = json.loads(raw_line)
                        if isinstance(parsed, Mapping) and isinstance(parsed.get("id"), str):
                            identifier = parsed["id"]
                except Exception:
                    pass
                print(f"worker request failed: {type(exc).__name__}: {exc}", file=stderr)
                if identifier is not None:
                    response_command = (
                        parsed.get("command")
                        if isinstance(parsed, Mapping) and parsed.get("command") in _COMMANDS
                        else "ping"
                    )
                    error_response = {
                        "id": identifier,
                        "command": response_command,
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                    stdout.write(encode_json_line(error_response))
                    stdout.flush()
                    continue
                return 1
            if self.worker.closed:
                break
        return 0


def run_server(
    worker: DCUModelWorker,
    *,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    response_dir: str | Path,
) -> int:
    return JSONLWorkerServer(worker, response_dir=response_dir).run(stdin, stdout, stderr)


def _build_cli_worker(config: Mapping[str, Any], stderr: TextIO) -> DCUModelWorker:
    validated = validate_worker_config(config, check_paths=True)
    validate_offline_environment(config)
    policy = build_official_policy(config, stderr=stderr)
    return DCUModelWorker(
        policy=policy,
        device_adapter=TorchDeviceAdapter(),
        seed=validated["seed"],
        config=config,
        stderr=stderr,
    )


class _StderrArgumentParser(argparse.ArgumentParser):
    def print_help(self, file: TextIO | None = None) -> None:
        super().print_help(sys.stderr if file is None else file)


def main(argv: list[str] | None = None) -> int:
    parser = _StderrArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="explicit Torch seed; when supplied it must match config.seed",
    )
    parser.add_argument("--response-dir", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        config = load_worker_config(args.config)
        if args.seed is not None:
            requested_seed = _nonnegative_int(args.seed, "--seed")
            configured_seed = _nonnegative_int(config.get("seed"), "seed")
            if requested_seed != configured_seed:
                raise _error(
                    f"--seed must match config.seed ({configured_seed}), got {requested_seed}"
                )
        worker = _build_cli_worker(config, sys.stderr)
        return run_server(worker, response_dir=args.response_dir)
    except Exception as exc:
        print(f"worker startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised by the runtime shell
    raise SystemExit(main())


__all__ = [
    "DCUModelWorker",
    "EXPECTED_DEVICE",
    "FEATURE_SCHEMA",
    "JSONLWorkerServer",
    "NOISE_SCHEMA",
    "PREDICT_REQUEST_SCHEMA",
    "SELECT_REQUEST_SCHEMA",
    "TorchDeviceAdapter",
    "build_official_policy",
    "load_worker_config",
    "main",
    "run_server",
    "validate_device",
    "validate_offline_environment",
    "validate_worker_config",
]
