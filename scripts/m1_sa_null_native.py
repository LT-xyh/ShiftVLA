#!/usr/bin/env python3
"""M1-SA-native qualification/orchestration for ReplayVLA-P1 F3N.

This module deliberately avoids the legacy state_replay.yaml provenance
contract. Importing it is static-only: environment/runtime packages are loaded
only inside explicit qualification or construction functions.
"""

from __future__ import annotations

from collections.abc import Mapping
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
NATIVE_CONTRACT = "m1_sa_native_v1"
AUTHORITY_COMMIT = "3902dea1879a4bdf3fbb1f76ad60aa46295e8ed8"
LEGACY_LIBERO_CONFIG = "/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/m1/libero_config"
EXPECTED_TASK = {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027}
EXPECTED_RUNTIME_LOCK_SHA256 = "921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc"
EXPECTED_BDDL_SHA256 = "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b"
EXPECTED_INIT_SHA256 = "cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a"
EXPECTED_ACTION_SHA256 = "c17bc44ad8195fecb42a80b3b272828761a9df6d88dd2bafe45db01a6cb04bbf"
EXPECTED_REGISTRY_SHA256 = "2be994dc8915b53b9fb94b5a55fa5d95e13979ea21d6a5aba6112a1611f66608"
EXPECTED_LIBERO_CONFIG_SHA256 = "743d98647daf57c0cd4e3f54db3ca11f8c93ec326ff1627b606fc9960fab02ba"
EXPECTED_LIBERO_SHA = "8561c60eea2fb93096146f240194649df73d8b1e"
EXPECTED_GL = {
    "vendor": "Mesa/X.org",
    "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
    "version": "3.1 Mesa 21.1.5",
}


class NativeQualificationError(RuntimeError):
    pass


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeQualificationError(f"{name} must be a mapping")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: str | Path) -> str:
    target = _resolve(path)
    if target.is_symlink() or not target.is_file():
        raise NativeQualificationError(f"expected regular file: {target}")
    h = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_head(path: str | Path) -> str | None:
    target = _resolve(path)
    try:
        return subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _repo_head() -> str | None:
    return _git_head(ROOT)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    target = _resolve(path)
    if target.is_symlink() or not target.is_file():
        raise NativeQualificationError(f"config is not a regular file: {target}")
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise NativeQualificationError("config must be a mapping")
    return dict(value)


def _require_sha(path: str | Path, expected: str, role: str) -> dict[str, Any]:
    target = _resolve(path)
    actual = _sha256_file(target)
    if actual.lower() != expected.lower():
        raise NativeQualificationError(f"{role} SHA drift: {actual} != {expected}")
    return {"path": str(target), "sha256": actual, "size": target.stat().st_size}


def _config_without_dynamic_requirements(config: Mapping[str, Any]) -> None:
    if config.get("runtime_contract") != NATIVE_CONTRACT:
        raise NativeQualificationError("runtime_contract is not m1_sa_native_v1")
    if config.get("authority_commit") != AUTHORITY_COMMIT:
        raise NativeQualificationError("F3N authority commit drift")
    if dict(_mapping(config.get("task"), "task")) != EXPECTED_TASK:
        raise NativeQualificationError("task identity drift")
    if config.get("obs_type") != "pixels_agent_pos":
        raise NativeQualificationError("obs_type drift")
    if int(config.get("pair_count", -1)) != 20 or config.get("pair_prefix") != "m1n0-pair-":
        raise NativeQualificationError("F3N schedule must be exactly 20 pairs")
    if config.get("state_replay_config") is not None:
        raise NativeQualificationError("F3N must not reference legacy state_replay_config")
    serialized = json.dumps(config, sort_keys=True)
    if LEGACY_LIBERO_CONFIG in serialized:
        raise NativeQualificationError("legacy LIBERO_CONFIG_PATH leaked into F3N config")
    runtime = _mapping(config.get("runtime"), "runtime")
    required_runtime = {
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
        "retry_count": 0,
        "fresh_processes": True,
    }
    for key, expected in required_runtime.items():
        if runtime.get(key) != expected:
            raise NativeQualificationError(f"runtime.{key} drift: {runtime.get(key)!r} != {expected!r}")
    env = _mapping(runtime.get("environment"), "runtime.environment")
    if env.get("MUJOCO_GL") != "egl" or env.get("PYOPENGL_PLATFORM") != "egl":
        raise NativeQualificationError("F3N renderer backend must be EGL")
    if str(env.get("MUJOCO_EGL_DEVICE_ID")) != "8":
        raise NativeQualificationError("F3N EGL ordinal must remain 8")
    libero_cfg = _mapping(config.get("libero_config"), "libero_config")
    expected_cfg_dir = str(_resolve(libero_cfg["path"]).parent)
    if str(Path(str(env.get("LIBERO_CONFIG_PATH", ""))).resolve()) != str(Path(expected_cfg_dir).resolve()):
        raise NativeQualificationError("runtime LIBERO_CONFIG_PATH does not bind the F3N LIBERO config")
    if config.get("python") != "/public/home/xuyinghao/tmp/shiftvla-libero/bin/python":
        raise NativeQualificationError("CPU interpreter drift")
    if "source_checkouts" in config:
        raise NativeQualificationError("F3N must not reintroduce legacy third-party source checkout gates")


def validate_native_config(
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any] | None = None,
    *,
    check_files: bool = True,
) -> dict[str, Any]:
    """Validate only the M1-SA-native null contract.

    Missing external/lerobot, external/robosuite, and external/mujoco source
    checkouts are intentionally not gates.
    """

    _config_without_dynamic_requirements(config)
    if runtime_config is not None and dict(runtime_config) != dict(config):
        raise NativeQualificationError("native runtime config must be the direct F3N config")
    if not check_files:
        return {"status": "PASS", "contract": NATIVE_CONTRACT, "file_checks": "SKIPPED"}

    python_path = Path(str(config["python"]))
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise NativeQualificationError(f"CPU interpreter unavailable: {python_path}")

    lock = _mapping(config.get("runtime_lock"), "runtime_lock")
    if str(lock.get("sha256")).lower() != EXPECTED_RUNTIME_LOCK_SHA256:
        raise NativeQualificationError("runtime lock declaration drift")
    lock_evidence = _require_sha(lock["path"], EXPECTED_RUNTIME_LOCK_SHA256, "runtime lock")

    libero_cfg = _mapping(config.get("libero_config"), "libero_config")
    if str(libero_cfg.get("sha256")).lower() != EXPECTED_LIBERO_CONFIG_SHA256:
        raise NativeQualificationError("LIBERO config declaration drift")
    libero_cfg_evidence = _require_sha(
        libero_cfg["path"], EXPECTED_LIBERO_CONFIG_SHA256, "ReplayVLA-P1 LIBERO config"
    )

    input_bindings = _mapping(config.get("input_bindings"), "input_bindings")
    bddl = _mapping(input_bindings.get("bddl"), "input_bindings.bddl")
    init_state = _mapping(input_bindings.get("init_state"), "input_bindings.init_state")
    if str(bddl.get("sha256")).lower() != EXPECTED_BDDL_SHA256:
        raise NativeQualificationError("BDDL declaration drift")
    if str(init_state.get("sha256")).lower() != EXPECTED_INIT_SHA256:
        raise NativeQualificationError("init-state declaration drift")
    bddl_evidence = _require_sha(bddl["path"], EXPECTED_BDDL_SHA256, "BDDL")
    init_evidence = _require_sha(init_state["path"], EXPECTED_INIT_SHA256, "init-state")

    recovery = _mapping(input_bindings.get("recovery_manifest"), "input_bindings.recovery_manifest")
    recovery_path = _resolve(recovery["path"])
    if recovery_path.is_symlink() or not recovery_path.is_file():
        raise NativeQualificationError("recovery manifest is unavailable")

    checkout = _mapping(config.get("libero_checkout"), "libero_checkout")
    actual_libero_head = _git_head(checkout["path"])
    if checkout.get("git_sha") != EXPECTED_LIBERO_SHA or actual_libero_head != EXPECTED_LIBERO_SHA:
        raise NativeQualificationError(
            f"pinned LIBERO checkout drift: {actual_libero_head!r} != {EXPECTED_LIBERO_SHA}"
        )

    assets = _mapping(config.get("assets"), "assets")
    asset_path = Path(str(assets["path"]))
    if assets.get("revision") != "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6":
        raise NativeQualificationError("asset revision drift")
    if not asset_path.is_dir():
        raise NativeQualificationError(f"LIBERO assets unavailable: {asset_path}")

    registry = _mapping(config.get("registry"), "registry")
    if str(registry.get("sha256")).lower() != EXPECTED_REGISTRY_SHA256:
        raise NativeQualificationError("derived registry declaration drift")
    # The strict loader verifies the registry self-hash, rebound source bytes,
    # and tape provenance without importing any environment runtime.
    from scripts.m1_hard_gate import load_frozen_registry

    loaded_registry = load_frozen_registry(_resolve(registry["path"]))
    if str(loaded_registry.get("registry_sha256")).lower() != EXPECTED_REGISTRY_SHA256:
        raise NativeQualificationError("derived registry strict-loader identity drift")

    # Action semantic bytes are checked using no-pickle NumPy loading.
    action = _mapping(config.get("action_tape"), "action_tape")
    import numpy as np

    tape_path = _resolve(action["path"])
    tape = np.load(tape_path, allow_pickle=False)
    tape = np.ascontiguousarray(np.asarray(tape))
    if tape.dtype != np.dtype("float32") or tuple(tape.shape) != (82, 7) or not np.isfinite(tape).all():
        raise NativeQualificationError("action tape shape/dtype/finite contract drift")
    semantic_sha = hashlib.sha256(tape.tobytes(order="C")).hexdigest()
    if semantic_sha != EXPECTED_ACTION_SHA256 or str(action.get("sha256")).lower() != EXPECTED_ACTION_SHA256:
        raise NativeQualificationError("action semantic bytes drift")

    # Validate the actual F2-proven LIBERO config targets. The datasets field is
    # retained as configuration evidence but is not a policy-free construction
    # gate because build_cpu_environment_runtime does not consume datasets.
    parsed_libero_cfg = _load_yaml(libero_cfg["path"])
    for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
        value = parsed_libero_cfg.get(key)
        if not isinstance(value, str) or not Path(value).exists():
            raise NativeQualificationError(f"LIBERO config target unavailable: {key}={value!r}")
    if Path(parsed_libero_cfg["bddl_files"]).resolve() != _resolve(bddl["path"]).parent.parent.resolve():
        # bddl_files is the suite root; compare against the checked-out root explicitly below.
        expected = (ROOT / "external/hf-libero/libero/libero/bddl_files").resolve()
        if Path(parsed_libero_cfg["bddl_files"]).resolve() != expected:
            raise NativeQualificationError("LIBERO config BDDL root drift")
    expected_init_root = (ROOT / "external/hf-libero/libero/libero/init_files").resolve()
    if Path(parsed_libero_cfg["init_states"]).resolve() != expected_init_root:
        raise NativeQualificationError("LIBERO config init-state root drift")
    if Path(parsed_libero_cfg["assets"]).resolve() != asset_path.resolve():
        raise NativeQualificationError("LIBERO config assets root drift")

    return {
        "status": "PASS",
        "contract": NATIVE_CONTRACT,
        "python": str(python_path.resolve()),
        "runtime_lock": lock_evidence,
        "libero_config": libero_cfg_evidence,
        "libero_checkout": {"path": str(_resolve(checkout["path"])), "git_sha": actual_libero_head},
        "bddl": bddl_evidence,
        "init_state": init_evidence,
        "registry_sha256": EXPECTED_REGISTRY_SHA256,
        "action_semantic_sha256": semantic_sha,
        "assets": {"path": str(asset_path.resolve()), "revision": assets["revision"]},
        "recovery_manifest": {
            "path": str(recovery_path.resolve()),
            "sha256": _sha256_file(recovery_path),
            "size": recovery_path.stat().st_size,
        },
        "datasets": {
            "path": parsed_libero_cfg.get("datasets"),
            "classification": "NOT_USED_ON_TRACED_F3N_POLICY_FREE_PATH",
        },
        "missing_source_checkouts_are_not_gates": ["external/lerobot", "external/robosuite", "external/mujoco"],
    }


def native_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_native_config(config, check_files=True)
    return copy.deepcopy(dict(config))


_RUNTIME_PROBE = r"""
import hashlib, importlib, importlib.metadata as md, json, pathlib, platform, sys, sysconfig
specs = json.loads(sys.argv[1])
rows = {}
for key, spec in specs.items():
    module = importlib.import_module(spec["module"])
    path = pathlib.Path(module.__file__).resolve()
    dist = md.distribution(spec["distribution"])
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    rows[key] = {
        "module": spec["module"],
        "distribution": spec["distribution"],
        "version": dist.version,
        "module_file": str(path),
        "module_file_sha256": h,
        "distribution_location": str(pathlib.Path(dist.locate_file("")).resolve()),
        "evidence_class": "TRUSTED_DEPENDENCY",
        "source_to_binary": "NOT_INDEPENDENTLY_ATTESTED",
    }
print(json.dumps({
    "python": platform.python_version(),
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "purelib": str(pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()),
    "dependencies": rows,
}, sort_keys=True))
"""


def runtime_identity_audit(
    config: Mapping[str, Any],
    *,
    adapter: Any | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    static = validate_native_config(config, check_files=True)
    python_path = str(Path(str(config["python"])).resolve())
    deps = _mapping(config.get("trusted_dependencies"), "trusted_dependencies")
    env = os.environ.copy()
    runtime_env = _mapping(_mapping(config["runtime"], "runtime").get("environment"), "runtime.environment")
    env.update({str(k): str(v) for k, v in runtime_env.items()})
    probe = subprocess.run(
        [python_path, "-c", _RUNTIME_PROBE, json.dumps(deps, sort_keys=True)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise NativeQualificationError(
            f"trusted-dependency probe failed ({probe.returncode}): {probe.stderr[-1200:]}"
        )
    try:
        observed = json.loads(probe.stdout.strip())
    except json.JSONDecodeError as exc:
        raise NativeQualificationError("trusted-dependency probe output is not JSON") from exc
    if str(Path(observed.get("executable", "")).resolve()) != python_path:
        raise NativeQualificationError("runtime probe used a different interpreter")
    expected_versions = {str(k): str(v["version"]) for k, v in deps.items()}
    actual_rows = _mapping(observed.get("dependencies"), "probe.dependencies")
    errors = []
    purelib = Path(str(observed["purelib"])).resolve()
    for key, version in expected_versions.items():
        row = _mapping(actual_rows.get(key), f"probe.dependencies.{key}")
        if str(row.get("version")) != version:
            errors.append(f"{key} version {row.get('version')!r} != {version!r}")
        module_file = Path(str(row.get("module_file", ""))).resolve()
        try:
            module_file.relative_to(purelib)
        except ValueError:
            errors.append(f"{key} module escaped purelib: {module_file}")
    if errors:
        raise NativeQualificationError("trusted dependency identity drift: " + "; ".join(errors))

    facts = {
        "python": observed.get("python"),
        **{str(key): str(row.get("version")) for key, row in actual_rows.items()},
    }
    identity = {
        "contract": NATIVE_CONTRACT,
        "python_executable": python_path,
        "python": {"executable": python_path, "version": observed.get("python")},
        "facts": facts,
        "runtime_lock": static["runtime_lock"],
        "dependencies": dict(actual_rows),
        "environment": {str(k): str(v) for k, v in runtime_env.items()},
        "expected_gl": copy.deepcopy(EXPECTED_GL),
        "evidence_class": "OBSERVED_RUNTIME_IDENTITY",
    }
    if adapter is not None:
        gl = None
        for owner in (adapter, getattr(adapter, "inner", None), getattr(adapter, "env", None)):
            value = getattr(owner, "gl_identity", None) if owner is not None else None
            if isinstance(value, Mapping):
                candidate = {str(k): str(v) for k, v in value.items()}
                if all(candidate.get(k) for k in ("vendor", "renderer", "version")):
                    gl = candidate
                    break
        if gl is None:
            raise NativeQualificationError("live GL identity was not captured")
        if gl != EXPECTED_GL:
            raise NativeQualificationError(f"live GL identity drift: {gl!r} != {EXPECTED_GL!r}")
        identity["gl_identity"] = gl
    return identity


def construct_native_adapter(config: Mapping[str, Any]) -> Any:
    """Construct one fresh official env through the F2-proven policy-free seam."""

    validate_native_config(config, check_files=True)
    runtime = _mapping(config["runtime"], "runtime")
    env = _mapping(runtime["environment"], "runtime.environment")
    os.environ.update({str(k): str(v) for k, v in env.items()})

    # LIBERO resolves assets through LIBERO_CONFIG_PATH/config.yaml. Verify
    # that the imported package resolves the exact frozen asset root instead
    # of mutating an undocumented private cache attribute.
    import libero.libero as libero_package

    asset_path = Path(str(_mapping(config["assets"], "assets")["path"])).resolve()
    get_libero_path = getattr(libero_package, "get_libero_path", None)
    if not callable(get_libero_path):
        raise NativeQualificationError("installed LIBERO does not expose get_libero_path")
    resolved_asset_path = Path(str(get_libero_path("assets"))).resolve()
    if resolved_asset_path != asset_path:
        raise NativeQualificationError(
            f"LIBERO asset binding drift: {resolved_asset_path} != {asset_path}"
        )

    from scripts.dcu_preflight import build_cpu_environment_runtime
    from scripts.m1_state_replay import RuntimeAdapter

    runtime_bundle = build_cpu_environment_runtime(config, phase="compare")
    tape_hash = str(_mapping(config["action_tape"], "action_tape")["sha256"])
    return RuntimeAdapter.construct_fresh(
        config,
        runtime_builder=lambda _cfg: runtime_bundle,
        tape_hash=tape_hash,
    )


def _write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NativeQualificationError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def _copy_bytes_no_overwrite(source: Path, target: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file():
        raise NativeQualificationError(f"source evidence is not a regular file: {source}")
    if target.exists() or target.is_symlink():
        raise NativeQualificationError(f"refusing to overwrite compact evidence: {target}")
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return {"path": str(target), "sha256": digest, "size": len(data)}


def static_qualify(config_path: str | Path) -> dict[str, Any]:
    config_target = _resolve(config_path).resolve()
    config = _load_yaml(config_target)
    static = validate_native_config(config, check_files=True)
    runtime_identity = runtime_identity_audit(config)
    current_head = _repo_head()
    if current_head is None:
        raise NativeQualificationError("repository HEAD unavailable")
    return {
        "phase": "F3N-static",
        "status": "PASS",
        "authority_commit": AUTHORITY_COMMIT,
        "execution_commit": current_head,
        "config_path": str(config_target),
        "config_file_sha256": _sha256_file(config_target),
        "static_inputs": static,
        "runtime_identity": runtime_identity,
        "legacy_state_replay_consulted": False,
        "policy_or_processor_import_authorized": False,
    }


class FailFastProcessRunner:
    """Launch no further workers after one pre-construction setup failure."""

    def __init__(self, delegate: Callable[[Mapping[str, Any]], Any]):
        self.delegate = delegate
        self.blocked = False
        self.block_reason: str | None = None
        self.dynamic_launches = 0

    def __call__(self, request: Mapping[str, Any]) -> Any:
        from scripts import m1_null_calibration as nullcal

        if self.blocked:
            value = nullcal._attempt_failure(
                request,
                "NOT_RUN_DUE_TO_COHORT_BLOCK: " + str(self.block_reason),
            )
            value["not_run_due_to_cohort_block"] = True
            return value
        self.dynamic_launches += 1
        value = self.delegate(request)
        if isinstance(value, Mapping) and value.get("status") != "completed":
            protocol = value.get("protocol")
            construction_count = (
                protocol.get("construction_reset_count")
                if isinstance(protocol, Mapping)
                else None
            )
            if construction_count in (0, None):
                self.blocked = True
                self.block_reason = str(value.get("error", "pre-construction worker failure"))
        return value


def renderer_entry_check(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(config["runtime"], "runtime")
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in _mapping(runtime["environment"], "runtime.environment").items()})
    cmd = [str(Path(str(config["python"])).resolve()), "-B", str(ROOT / "scripts/m1_sa_renderer.py"), "discover"]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    parsed = None
    for line in reversed(result.stdout.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if result.returncode != 0 or not isinstance(parsed, Mapping):
        raise NativeQualificationError(
            f"renderer entry check failed: returncode={result.returncode}, stderr={result.stderr[-1000:]}"
        )
    if parsed.get("status") != "PASS" or int(parsed.get("selected_ordinal", -1)) != 8:
        raise NativeQualificationError(f"renderer entry check did not uniquely select ordinal 8: {parsed!r}")
    return {
        "status": "PASS",
        "selected_ordinal": 8,
        "selection_rule": parsed.get("selection_rule"),
        "devices": parsed.get("devices"),
    }


def execute_f3n(
    config_path: str | Path,
    qualification_path: str | Path,
) -> dict[str, Any]:
    from scripts import m1_null_calibration as nullcal

    config_target = _resolve(config_path).resolve()
    config = _load_yaml(config_target)
    qualification_target = _resolve(qualification_path).resolve()
    qualification = json.loads(qualification_target.read_text(encoding="utf-8"))
    if qualification.get("status") != "PASS":
        raise NativeQualificationError("F3N static qualification is not PASS")
    if qualification.get("config_file_sha256") != _sha256_file(config_target):
        raise NativeQualificationError("F3N config bytes changed after qualification")
    if qualification.get("execution_commit") != _repo_head():
        raise NativeQualificationError("repository HEAD changed after F3N qualification")

    compact_root = ROOT / "runtime/replayvla-p1/f3n"
    if (compact_root / "f3n_summary.json").exists():
        raise NativeQualificationError("F3N compact summary already exists; no rerun is authorized")

    try:
        prepared = nullcal.prepare_run(config_path=config_target)
        schedule_record = _copy_bytes_no_overwrite(
            Path(prepared.run_spec_path), compact_root / "null_schedule.json"
        )
        pair_registry_record = _copy_bytes_no_overwrite(
            Path(prepared.pair_registry_path), compact_root / "pair_registry.json"
        )
    except Exception as exc:
        summary = {
            "phase": "F3N",
            "status": "BLOCKED",
            "stage": "prepare_run",
            "reason": f"{type(exc).__name__}: {exc}",
            "completed_pairs": 0,
            "completed_attempts": 0,
            "technical_failures": 1,
            "f3n": {
                "authority_commit": AUTHORITY_COMMIT,
                "execution_commit": _repo_head(),
                "static_qualification_path": str(qualification_target),
                "static_qualification_sha256": _sha256_file(qualification_target),
                "dynamic_worker_launches": 0,
                "cohort_fail_fast_triggered": False,
            },
        }
        _write_json_no_overwrite(compact_root / "f3n_summary.json", summary)
        return summary

    base_meta = {
        "authority_commit": AUTHORITY_COMMIT,
        "execution_commit": _repo_head(),
        "static_qualification_path": str(qualification_target),
        "static_qualification_sha256": _sha256_file(qualification_target),
        "schedule": schedule_record,
        "pair_registry": pair_registry_record,
    }

    try:
        entry = renderer_entry_check(config)
    except Exception as exc:
        summary = {
            "phase": "F3N",
            "status": "BLOCKED",
            "stage": "renderer_entry_check",
            "reason": f"{type(exc).__name__}: {exc}",
            "completed_pairs": 0,
            "completed_attempts": 0,
            "technical_failures": 0,
            "f3n": {**base_meta, "renderer_entry_check": "BLOCKED"},
        }
        _write_json_no_overwrite(compact_root / "f3n_summary.json", summary)
        return summary

    runner = FailFastProcessRunner(nullcal._default_process_runner)
    try:
        result = nullcal.run_calibration(prepared, process_runner=runner)
    except Exception as exc:
        summary = {
            "phase": "F3N",
            "status": "BLOCKED",
            "stage": "null_runner",
            "reason": f"{type(exc).__name__}: {exc}",
            "completed_pairs": 0,
            "completed_attempts": 0,
            "technical_failures": 1,
            "f3n": {
                **base_meta,
                "renderer_entry_check": entry,
                "dynamic_worker_launches": runner.dynamic_launches,
                "cohort_fail_fast_triggered": runner.blocked,
                "cohort_block_reason": runner.block_reason,
            },
        }
        _write_json_no_overwrite(compact_root / "f3n_summary.json", summary)
        return summary

    result["f3n"] = {
        **base_meta,
        "renderer_entry_check": entry,
        "dynamic_worker_launches": runner.dynamic_launches,
        "cohort_fail_fast_triggered": runner.blocked,
        "cohort_block_reason": runner.block_reason,
    }
    terminal_path = result.get("terminal_manifest_path")
    raw_manifest = {
        "output_root": str(prepared.run_spec["output_root"]),
        "terminal_manifest": (
            {
                "path": str(terminal_path),
                "sha256": _sha256_file(terminal_path),
                "size": Path(str(terminal_path)).stat().st_size,
            }
            if isinstance(terminal_path, str) and Path(terminal_path).is_file()
            else None
        ),
        "schedule": schedule_record,
        "pair_registry": pair_registry_record,
        "raw_artifacts_committed_to_git": False,
    }
    _write_json_no_overwrite(compact_root / "raw_evidence_manifest.json", raw_manifest)
    summary = {
        "phase": "F3N",
        "status": result.get("status"),
        "counts": copy.deepcopy(result.get("counts", {})),
        "gates": copy.deepcopy(result.get("gates", {})),
        "errors": copy.deepcopy(result.get("errors", [])),
        "terminal_manifest_path": terminal_path,
        "f3n": copy.deepcopy(result["f3n"]),
    }
    _write_json_no_overwrite(compact_root / "f3n_summary.json", summary)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("qualify")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)

    e = sub.add_parser("execute")
    e.add_argument("--config", type=Path, required=True)
    e.add_argument("--qualification", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "qualify":
            value = static_qualify(args.config)
            _write_json_no_overwrite(_resolve(args.output), value)
            print(json.dumps(value, sort_keys=True))
            return 0
        result = execute_f3n(args.config, args.qualification)
        print(json.dumps({
            "status": result.get("status"),
            "terminal_manifest_path": result.get("terminal_manifest_path"),
            "f3n": result.get("f3n"),
        }, sort_keys=True))
        return 0 if result.get("status") == "PASS" else 1
    except Exception as exc:
        if args.cmd == "qualify":
            blocked = {
                "phase": "F3N-static",
                "status": "BLOCKED",
                "authority_commit": AUTHORITY_COMMIT,
                "execution_commit": _repo_head(),
                "config_path": str(_resolve(args.config).resolve()),
                "stage": "static_qualification",
                "reason": f"{type(exc).__name__}: {exc}",
                "legacy_state_replay_consulted": False,
                "policy_or_processor_import_authorized": False,
            }
            try:
                _write_json_no_overwrite(_resolve(args.output), blocked)
            except Exception as publication_exc:
                print(
                    f"F3N qualify evidence publication failed: "
                    f"{type(publication_exc).__name__}: {publication_exc}",
                    file=sys.stderr,
                )
        print(f"F3N {args.cmd} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
