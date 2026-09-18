"""One-shot ReplayVLA-P1 repo-authorized microvalidation harness.

Execution order:
    focused pytest -> MV-P2 CPU camera seam -> MV-P1 DCU explicit-noise select_action -> stop

This is not a scientific rollout and never computes paper estimands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.p1_prefix_reexecution import (
    AgentviewYawController,
    FROZEN_ENVIRONMENT_SEED,
    FROZEN_FLOW_SHAPE,
    ObservationIndexedCameraWrapper,
    generate_paired_flow_noise,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CONFIG = ROOT / "configs" / "m0" / "baseline_a.yaml"
DEFAULT_OUTPUT = ROOT / "runtime" / "replayvla-p1" / "prefix_reexecution_microvalidation.json"
FOCUSED_TESTS = (
    "tests/test_p1_prefix_reexecution.py",
    "tests/test_p1_prefix_reexecution_microvalidate.py",
)
PHASE = "MICROVALIDATION"
ROOT_ID = "libero_spatial-task000-init000-seed2027"
TASK_ID = 0
INIT_STATE_ID = 0


class MicrovalidationError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise MicrovalidationError(f"invalid git HEAD: {value!r}")
    return value


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
    }


def _physics_snapshot(sim: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = getattr(sim, "data", None)
    if data is None:
        raise MicrovalidationError("sim has no data object")
    arrays: dict[str, np.ndarray] = {}
    public: dict[str, Any] = {}
    for name in ("qpos", "qvel", "ctrl"):
        value = getattr(data, name, None)
        if value is None:
            public[name] = {"defined": False}
            continue
        array = np.array(value, copy=True)
        arrays[name] = array
        public[name] = {"defined": True, **_array_identity(array)}
    sim_time = getattr(data, "time", None)
    if sim_time is None:
        raise MicrovalidationError("sim.data.time is unavailable")
    public["sim_time"] = float(sim_time)
    return public, arrays


def _physics_equal(
    reference_public: Mapping[str, Any],
    reference_arrays: Mapping[str, np.ndarray],
    candidate_public: Mapping[str, Any],
    candidate_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    fields: dict[str, bool] = {}
    for name in ("qpos", "qvel", "ctrl"):
        ref_defined = bool(reference_public[name]["defined"])
        got_defined = bool(candidate_public[name]["defined"])
        if ref_defined != got_defined:
            fields[name] = False
        elif not ref_defined:
            fields[name] = True
        else:
            fields[name] = bool(np.array_equal(reference_arrays[name], candidate_arrays[name]))
    time_equal = float(reference_public["sim_time"]) == float(candidate_public["sim_time"])
    return {
        "qpos": fields["qpos"],
        "qvel": fields["qvel"],
        "ctrl": fields["ctrl"],
        "sim_time": time_equal,
        "all": bool(all(fields.values()) and time_equal),
    }


def _image_identity(observation: Mapping[str, Any]) -> dict[str, Any]:
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping):
        raise MicrovalidationError("policy observation has no pixels mapping")
    image = pixels.get("image")
    if image is None:
        raise MicrovalidationError("policy observation has no agentview image key 'image'")
    return _array_identity(image)


def locate_single_libero_subenv(vector_env: Any) -> Any:
    num_envs = getattr(vector_env, "num_envs", None)
    if isinstance(num_envs, bool) or int(num_envs if num_envs is not None else -1) != 1:
        raise MicrovalidationError(f"SyncVectorEnv must expose num_envs=1, got {num_envs!r}")
    envs = getattr(vector_env, "envs", None)
    if not isinstance(envs, Sequence) or isinstance(envs, (str, bytes)) or len(envs) != 1:
        raise MicrovalidationError("single synchronous sub-env is not available through env.envs[0]")
    subenv = envs[0]
    cls = type(subenv)
    if cls.__module__ != "lerobot.envs.libero" or cls.__name__ != "LiberoEnv":
        raise MicrovalidationError(
            f"unexpected single sub-env identity: {cls.__module__}.{cls.__name__}"
        )
    for name in ("reset", "step", "_ensure_env", "_format_raw_obs"):
        if not callable(getattr(subenv, name, None)):
            raise MicrovalidationError(f"LeRobot LiberoEnv seam is missing {name}()")
    return subenv


def _load_configs(baseline_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts import dcu_preflight, m0_baseline_a

    raw = m0_baseline_a.load_config(baseline_path)
    baseline = m0_baseline_a.validate_config(raw)
    if baseline["seed"] != FROZEN_ENVIRONMENT_SEED:
        raise MicrovalidationError("baseline seed drift")
    if baseline["suite"] != "libero_spatial" or TASK_ID not in baseline["task_ids"]:
        raise MicrovalidationError("baseline task0 identity drift")
    if INIT_STATE_ID not in baseline["init_state_ids"]:
        raise MicrovalidationError("baseline init0 identity drift")

    preflight_path = Path(str(baseline["preflight_config"])).resolve()
    if preflight_path.is_symlink() or not preflight_path.is_file():
        raise MicrovalidationError(f"preflight config unavailable: {preflight_path}")
    if _sha256_file(preflight_path) != baseline["preflight_config_sha256"]:
        raise MicrovalidationError("baseline preflight_config SHA drift")
    loader = getattr(dcu_preflight, "_load_yaml_config", None)
    if not callable(loader):
        raise MicrovalidationError("dcu_preflight config loader unavailable")
    preflight = dict(loader(preflight_path))
    cross = getattr(m0_baseline_a, "_cross_check_preflight_identity", None)
    if not callable(cross):
        raise MicrovalidationError("baseline/preflight identity cross-check unavailable")
    identity = dict(cross(raw, preflight))
    task = preflight.get("task")
    if not isinstance(task, Mapping):
        raise MicrovalidationError("preflight task mapping unavailable")
    expected = {
        "suite": "libero_spatial",
        "task_id": TASK_ID,
        "init_state_id": INIT_STATE_ID,
        "seed": FROZEN_ENVIRONMENT_SEED,
    }
    for key, value in expected.items():
        if task.get(key) != value:
            raise MicrovalidationError(f"preflight task.{key} drift: {task.get(key)!r}")
    return baseline, preflight, {
        "baseline_config": str(baseline_path.resolve()),
        "baseline_config_sha256": _sha256_file(baseline_path.resolve()),
        "preflight_config": str(preflight_path),
        "preflight_config_sha256": _sha256_file(preflight_path),
        "cross_identity": identity,
    }


def _install_cpu_runtime_environment(baseline: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    from scripts import dcu_preflight, m0_baseline_a

    baseline_env = m0_baseline_a.build_cpu_child_environment(baseline)
    preflight_env = dcu_preflight.build_cpu_child_environment(preflight, base_environment=baseline_env)
    os.environ.clear()
    os.environ.update(preflight_env)
    return {
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH"),
    }


def run_mv_p2(
    preflight: Mapping[str, Any],
    *,
    runtime_builder: Callable[..., Mapping[str, Any]] | None = None,
    select_init_state: Callable[[Any, int], Mapping[str, Any]] | None = None,
    controller_factory: Callable[[], AgentviewYawController] = AgentviewYawController,
) -> dict[str, Any]:
    """Run camera-only real microvalidation; never creates a policy or calls outer env.step."""

    from scripts import dcu_preflight, m0_baseline_a

    builder = runtime_builder or dcu_preflight.build_cpu_environment_runtime
    selector = select_init_state or m0_baseline_a.select_init_state_before_reset
    runtime: Mapping[str, Any] | None = None
    try:
        runtime = builder(preflight, phase="compare")
        vector_env = runtime.get("env")
        if vector_env is None:
            raise MicrovalidationError("CPU environment runtime did not return env")
        subenv = locate_single_libero_subenv(vector_env)
        init_evidence = dict(selector(vector_env, INIT_STATE_ID))

        ensure_env = getattr(subenv, "_ensure_env")
        ensure_env()
        if getattr(subenv, "_env", None) is None or getattr(subenv._env, "env", None) is None:
            raise MicrovalidationError("LeRobot LiberoEnv did not create OffScreenRenderEnv.env")
        pre_reset_public, _ = _physics_snapshot(subenv._env.env.sim)

        controller = controller_factory()
        wrapped = ObservationIndexedCameraWrapper(subenv, arm="CC", controller=controller)
        clean_obs, reset_info = wrapped.reset(seed=FROZEN_ENVIRONMENT_SEED)
        if controller.sim is None:
            raise MicrovalidationError("camera controller did not bind the current hard-reset sim")
        current_sim = controller.sim

        clean_public, clean_arrays = _physics_snapshot(current_sim)
        clean_image = _image_identity(clean_obs)
        clean_quat = np.asarray(controller.modder.get_quat("agentview"), dtype=np.float64).copy()
        clean_pos = np.asarray(controller.modder.get_pos("agentview"), dtype=np.float64).copy()
        clean_fovy = float(controller.modder.get_fovy("agentview"))

        shifted_request = {
            "observation_index": 0,
            "preceding_action_index": None,
            "requested_camera_mode": "shifted",
        }
        shifted_camera = controller.apply("shifted", request=shifted_request)
        shifted_obs = wrapped._refresh_current_observation()
        shifted_public, shifted_arrays = _physics_snapshot(current_sim)
        shifted_image = _image_identity(shifted_obs)
        shifted_quat = np.asarray(controller.modder.get_quat("agentview"), dtype=np.float64).copy()

        restored_request = {
            "observation_index": 0,
            "preceding_action_index": None,
            "requested_camera_mode": "clean",
        }
        restored_camera = controller.apply("clean", request=restored_request)
        restored_obs = wrapped._refresh_current_observation()
        restored_public, restored_arrays = _physics_snapshot(current_sim)
        restored_image = _image_identity(restored_obs)
        restored_quat = np.asarray(controller.modder.get_quat("agentview"), dtype=np.float64).copy()
        restored_pos = np.asarray(controller.modder.get_pos("agentview"), dtype=np.float64).copy()
        restored_fovy = float(controller.modder.get_fovy("agentview"))

        shifted_physics = _physics_equal(
            clean_public, clean_arrays, shifted_public, shifted_arrays
        )
        restored_physics = _physics_equal(
            clean_public, clean_arrays, restored_public, restored_arrays
        )
        if not shifted_physics["all"] or not restored_physics["all"]:
            raise MicrovalidationError("camera-only mutation changed physics state or sim time")
        if np.allclose(clean_quat, shifted_quat, atol=1e-10, rtol=0.0):
            raise MicrovalidationError("+15 degree camera mutation did not change cam_quat")
        if not np.array_equal(restored_quat, clean_quat):
            raise MicrovalidationError("clean camera quaternion was not exactly restored")
        if not np.array_equal(restored_pos, clean_pos) or restored_fovy != clean_fovy:
            raise MicrovalidationError("camera position or fovy changed during quaternion-only mutation")
        if wrapped.action_index != 0:
            raise MicrovalidationError("MV-P2 must not call policy-indexed environment step")

        return {
            "status": "PASS",
            "phase": "MV-P2",
            "scientific_rollout": False,
            "policy_created": False,
            "outer_env_step_calls": 0,
            "policy_query_count": 0,
            "init_state": init_evidence,
            "pre_normal_reset_physics": pre_reset_public,
            "reset_info": {
                "type": type(reset_info).__name__,
                "keys": sorted(reset_info) if isinstance(reset_info, Mapping) else None,
            },
            "camera": {
                "api": "robosuite.utils.mjmod.CameraModder.set_quat",
                "clean_cam_quat_wxyz": clean_quat.tolist(),
                "shifted_cam_quat_wxyz": shifted_quat.tolist(),
                "restored_cam_quat_wxyz": restored_quat.tolist(),
                "cam_pos_before": clean_pos.tolist(),
                "cam_pos_after": restored_pos.tolist(),
                "cam_fovy_before": clean_fovy,
                "cam_fovy_after": restored_fovy,
                "shifted_evidence": shifted_camera,
                "restored_evidence": restored_camera,
            },
            "physics": {
                "clean": clean_public,
                "shifted": shifted_public,
                "restored": restored_public,
                "clean_to_shifted_exact": shifted_physics,
                "clean_to_restored_exact": restored_physics,
            },
            "observations": {
                "clean_agentview": clean_image,
                "shifted_agentview": shifted_image,
                "restored_agentview": restored_image,
                "branch_local_current_state_refresh": True,
            },
            "settle_steps": {
                "count_as_policy_queries": False,
                "count_toward_switch_index": False,
                "paired_noise_query_index_after_reset": 0,
            },
        }
    finally:
        if runtime:
            close_envs = runtime.get("close_envs")
            envs = runtime.get("envs")
            if callable(close_envs) and envs is not None:
                close_envs(envs)


def run_mv_p1(
    preflight: Mapping[str, Any],
    *,
    preflight_config_path: Path,
    work_dir: Path,
    physical_device: int,
    runtime_builder: Callable[..., Mapping[str, Any]] | None = None,
    worker_starter: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run exactly one real DCU explicit-noise select_action query; never env.step."""

    from scripts import dcu_preflight, m0_baseline_a
    from scripts.dcu_model_worker import FEATURE_SCHEMA, NOISE_SCHEMA

    builder = runtime_builder or dcu_preflight.build_cpu_runtime
    starter = worker_starter or dcu_preflight._start_worker
    runtime: Mapping[str, Any] | None = None
    worker: Mapping[str, Any] | None = None
    try:
        runtime = builder(preflight, include_policy=False, phase="compare")
        env = runtime.get("env")
        if env is None:
            raise MicrovalidationError("CPU processor runtime did not return env")
        locate_single_libero_subenv(env)
        init_evidence = dict(m0_baseline_a.select_init_state_before_reset(env, INIT_STATE_ID))
        observation, reset_info = dcu_preflight._reset_vector_env(env, FROZEN_ENVIRONMENT_SEED)
        features = dcu_preflight._prepare_features(runtime, observation, env)

        noise, noise_evidence = generate_paired_flow_noise(ROOT_ID, 0)
        if tuple(noise.shape) != tuple(FROZEN_FLOW_SHAPE):
            raise MicrovalidationError("paired flow-noise shape drift")

        # _start_worker configures JSONLWorkerServer(response_dir=work_dir / "ipc"),
        # and that server fail-closes every tensor path outside its own IPC root.
        ipc = work_dir / "ipc"
        ipc.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, Any] = {}
        request_index = 0
        noise_index = 0

        def request_writer(bundle: Mapping[str, torch.Tensor]) -> Path:
            nonlocal request_index
            path = ipc / f"features-{request_index:04d}.safetensors"
            request_index += 1
            artifacts["features"] = dcu_preflight._save_bundle(path, bundle, FEATURE_SCHEMA)
            return path

        def noise_writer(bundle: Mapping[str, torch.Tensor]) -> Path:
            nonlocal noise_index
            path = ipc / f"noise-{noise_index:04d}.safetensors"
            noise_index += 1
            artifacts["noise"] = dcu_preflight._save_bundle(path, bundle, NOISE_SCHEMA)
            return path

        worker = starter(
            preflight,
            run_directory=work_dir,
            config_path=preflight_config_path,
            physical_device=physical_device,
        )
        remote = dcu_preflight.FeatureOnlyRemotePolicy(
            worker["client"],
            request_writer=request_writer,
            noise_writer=noise_writer,
        )
        runtime_cfg = preflight["runtime"]
        remote.reset_timeout_seconds = float(runtime_cfg["worker_startup_timeout_seconds"])
        remote.forward_timeout_seconds = float(runtime_cfg["worker_forward_timeout_seconds"])
        remote.reset(seed=FROZEN_ENVIRONMENT_SEED)
        action = remote.select_action(features, noise=noise)
        feature_artifact = artifacts.get("features")
        noise_artifact = artifacts.get("noise")
        if not isinstance(feature_artifact, Mapping) or not isinstance(noise_artifact, Mapping):
            raise MicrovalidationError("remote explicit-noise select_action did not publish both IPC bundles")
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != (1, 7):
            raise MicrovalidationError("real explicit-noise select_action did not return shape (1, 7)")
        if not bool(torch.isfinite(action).all().item()):
            raise MicrovalidationError("real explicit-noise select_action returned non-finite action")

        response = dcu_preflight._response_for(worker, "select_action")
        inputs = response.get("input_dtypes")
        if not isinstance(inputs, Mapping) or "noise" not in inputs:
            raise MicrovalidationError("worker response does not prove explicit noise reached select_action")
        before = response.get("queue_length_before")
        after = response.get("queue_length_after")
        generated = response.get("new_chunk_generated")
        if before != 0 or after != 0 or generated is not True:
            raise MicrovalidationError(
                "n_action_steps=1 queue evidence must be before=0, after=0, generated=true"
            )
        client = worker.get("client")
        if int(getattr(client, "n_action_steps", -1)) != 1:
            raise MicrovalidationError("worker client n_action_steps drift")

        postprocessed = runtime["postprocessor"](action)
        if isinstance(postprocessed, Mapping):
            postprocessed = postprocessed.get("action", postprocessed)
        transition = runtime["env_postprocessor"]({"action": postprocessed})
        if not isinstance(transition, Mapping) or "action" not in transition:
            raise MicrovalidationError("official env postprocessor did not return action transition")
        final_action = transition["action"]
        final_array = np.asarray(
            final_action.detach().cpu().numpy()
            if isinstance(final_action, torch.Tensor)
            else final_action
        )
        if final_array.shape != (1, 7) or not np.isfinite(final_array).all():
            raise MicrovalidationError("official postprocessor chain returned invalid action")

        worker_summary = dcu_preflight._worker_summary(worker)
        compact_worker = {
            "physical_device": worker_summary.get("physical_device"),
            "logical_device": worker_summary.get("logical_device"),
            "torch_device_name": worker_summary.get("torch_device_name"),
            "model_load_success": worker_summary.get("model_load_success"),
            "model": worker_summary.get("model"),
            "torch_runtime": worker_summary.get("torch_runtime"),
            "stderr_tail": str(worker_summary.get("stderr", ""))[-4000:],
        }
        return {
            "status": "PASS",
            "phase": "MV-P1",
            "scientific_rollout": False,
            "outer_env_step_calls": 0,
            "policy_query_count": 1,
            "init_state": init_evidence,
            "reset_info": {
                "type": type(reset_info).__name__,
                "keys": sorted(reset_info) if isinstance(reset_info, Mapping) else None,
            },
            "paired_noise": noise_evidence,
            "artifacts": {
                "features": dict(feature_artifact),
                "noise": dict(noise_artifact),
            },
            "select_action": {
                "cpu_policy_seam": "FeatureOnlyRemotePolicy.select_action(features, noise=noise)",
                "official_worker_command": "select_action",
                "explicit_noise_present_in_worker_inputs": True,
                "action_shape": list(action.shape),
                "action_dtype": str(action.dtype),
                "action_sha256": _sha256_bytes(
                    np.ascontiguousarray(action.detach().cpu().numpy()).tobytes(order="C")
                ),
                "queue_length_before": before,
                "queue_length_after": after,
                "new_chunk_generated": generated,
                "n_action_steps": 1,
                "official_postprocessors_applied": True,
                "postprocessed_action_shape": list(final_array.shape),
                "postprocessed_action_sha256": _sha256_bytes(
                    np.ascontiguousarray(final_array).tobytes(order="C")
                ),
                "response": {
                    "input_dtypes": dict(inputs),
                    "output_dtype": response.get("output_dtype"),
                    "latency_seconds": response.get("latency_seconds"),
                    "peak_memory_bytes": response.get("peak_memory_bytes"),
                },
            },
            "worker": compact_worker,
        }
    finally:
        if worker:
            dcu_preflight._close_worker(worker)
        if runtime:
            close_envs = runtime.get("close_envs")
            envs = runtime.get("envs")
            if callable(close_envs) and envs is not None:
                close_envs(envs)


def run_focused_pytest() -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = completed.stdout or ""
    evidence = {
        "status": "PASS" if completed.returncode == 0 else "BLOCKED",
        "command": command,
        "returncode": int(completed.returncode),
        "output_tail": output[-12000:],
    }
    return evidence


def run_sequence(
    *,
    baseline_config: Path,
    output: Path,
    work_dir: Path,
    physical_device: int,
    pytest_runner: Callable[[], Mapping[str, Any]] = run_focused_pytest,
    p2_runner: Callable[..., Mapping[str, Any]] = run_mv_p2,
    p1_runner: Callable[..., Mapping[str, Any]] = run_mv_p1,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if work_dir.exists():
        raise FileExistsError(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "runtime_phase": PHASE,
        "scientific_rollout": False,
        "pilot_rollout_count": 0,
        "paper_estimands_computed": False,
        "root": {
            "suite": "libero_spatial",
            "task_id": TASK_ID,
            "init_state_id": INIT_STATE_ID,
            "environment_seed": FROZEN_ENVIRONMENT_SEED,
            "root_id": ROOT_ID,
        },
        "execution_commit": None,
        "status": "BLOCKED",
        "focused_pytest": {"status": "NOT_RUN"},
        "MV-P2": {"status": "NOT_RUN"},
        "MV-P1": {"status": "NOT_RUN"},
        "stop_after": "MV-P1",
        "retry": 0,
        "replacement": 0,
    }

    try:
        evidence["execution_commit"] = _git_head()
        pytest_evidence = dict(pytest_runner())
        evidence["focused_pytest"] = pytest_evidence
        if pytest_evidence.get("status") != "PASS":
            raise MicrovalidationError("focused pytest did not PASS")

        baseline, preflight, identity = _load_configs(baseline_config)
        evidence["config_identity"] = identity
        evidence["cpu_environment"] = _install_cpu_runtime_environment(baseline, preflight)

        p2 = dict(p2_runner(preflight))
        evidence["MV-P2"] = p2
        if p2.get("status") != "PASS":
            raise MicrovalidationError("MV-P2 did not PASS")

        preflight_path = Path(identity["preflight_config"])
        expected_device = int(preflight["runtime"]["compare_physical_device"])
        if physical_device != expected_device:
            raise MicrovalidationError(
                f"MV-P1 physical device must equal frozen compare device {expected_device}"
            )
        p1 = dict(
            p1_runner(
                preflight,
                preflight_config_path=preflight_path,
                work_dir=work_dir,
                physical_device=physical_device,
            )
        )
        evidence["MV-P1"] = p1
        if p1.get("status") != "PASS":
            raise MicrovalidationError("MV-P1 did not PASS")
        evidence["status"] = "PASS"
        return evidence
    except BaseException as exc:
        evidence["status"] = "BLOCKED"
        evidence["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return evidence
    finally:
        _json_write_exclusive(output, evidence)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--physical-device", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = args.output.resolve()
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else output.with_suffix(".work")
    )
    result = run_sequence(
        baseline_config=args.baseline_config.resolve(),
        output=output,
        work_dir=work_dir,
        physical_device=args.physical_device,
    )
    print(json.dumps({
        "runtime_phase": result["runtime_phase"],
        "scientific_rollout": result["scientific_rollout"],
        "pilot_rollout_count": result["pilot_rollout_count"],
        "status": result["status"],
        "output": str(output),
    }, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
