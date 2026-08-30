"""Fake-only contract tests for the frozen M0 Baseline-A harness.

These tests deliberately never import LeRobot, LIBERO, MuJoCo, or a policy
implementation.  They exercise only the planning, evidence, and fail-closed
boundaries that are safe to validate without hardware.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
import threading
import time
import types
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "m0" / "baseline_a.yaml"


def _module():
    # Defer import so the first RED run reports the missing feature rather
    # than failing test collection before any test body is selected.
    import scripts.m0_baseline_a as baseline

    return baseline


def _config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "baseline_a",
        "seed": 2027,
        "suite": "libero_spatial",
        "task_ids": [0, 4, 9],
        "init_state_ids": [0, 1, 2, 3, 4],
        "task_names": {
            "0": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
            "4": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
            "9": "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
        },
        "observation": {
            "type": "pixels_agent_pos",
            "height": 360,
            "width": 360,
        },
        "action": {"dim": 7, "chunk_size": 50, "n_action_steps": 1},
        "horizon": 280,
        "checkpoint": {
            "repo_id": "HuggingFaceVLA/smolvla_libero",
            "revision": "checkpoint-revision",
            "path": "/frozen/checkpoint",
            "sha256": "a" * 64,
        },
        "base_model": {
            "repo_id": "HuggingFaceTB/SmolVLM2-500M-Instruct",
            "revision": "base-revision",
            "path": "/frozen/base-model",
            "sha256": "b" * 64,
        },
        "assets": {
            "repo_id": "lerobot/libero-assets",
            "revision": "assets-revision",
            "path": "/frozen/assets",
            "manifest_sha256": "c" * 64,
        },
        "runtime": {
            "cpu_python": "/frozen/cpu/bin/python",
            "dcu_python": "/frozen/dcu/bin/python",
            "cpu_runtime_lock": "/frozen/cpu.lock",
            "dcu_runtime_lock": "/frozen/dcu.lock",
            "physical_devices": [0, 1],
            "logical_device": "cuda:0",
            "worker_startup_timeout_seconds": 300,
            "worker_forward_timeout_seconds": 300,
            "worker_shutdown_timeout_seconds": 30,
            "parent_collection_timeout_seconds": 86400,
            "renderer": {
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": "8",
            },
            "offline": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            },
        },
        "preflight_config": "/frozen/dcu_preflight.yaml",
        "preflight_config_sha256": "3ac615f60a254fe10968f7cf39d51f0d501bbc3b02992ac49fbdcbe2912ba058",
        "output": {"root": "/tmp/baseline-a-runs"},
        "perturbation": False,
        "action_noise": {"source": "native_policy_rng", "explicit": False},
        "provenance": {
            "lerobot_git_sha": "lerobot-sha",
            "libero_git_sha": "libero-sha",
            "python": "3.11",
            "pytorch": "2.x",
            "transformers": "4.x",
            "cuda": "none-under-hip",
            "hip": "6.x",
            "gpu": "K100",
        },
    }


def _episode(module: Any, planned: dict[str, Any], *, success: bool, steps: int | None = None) -> dict[str, Any]:
    if steps is None:
        steps = 3 if success else 280
    action = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32)
    worker_evidence = {
        "physical_device": planned["physical_device"],
        "logical_device": "cuda:0",
        "torch_logical_device": "cuda:0",
        "model_load_success": True,
        "compute_environment": {
            "physical_k100": planned["physical_device"],
            "HIP_VISIBLE_DEVICES": str(planned["physical_device"]),
            "CUDA_VISIBLE_DEVICES": "0",
        },
    }
    gl_evidence = {
        "variables": dict(module.EXPECTED_RENDERER),
        "exact": True,
        "gl_identity": {
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
            "version": "3.1 Mesa 21.1.5",
        },
    }
    return module.make_episode_record(
        planned,
        success=success,
        rewards=[1.0 if success else 0.0] * steps,
        terminated=success,
        truncated=not success,
        steps=steps,
        policy_calls=steps,
        inference_latencies=([0.01, 0.02, 0.03] if success else [0.01] * steps),
        env_step_latencies=([0.04, 0.05, 0.06] if success else [0.04] * steps),
        actions=[action] * steps,
        model_peak_memory_bytes=1234,
        wall_time_seconds=0.5,
        init_state_evidence={
            "selected_init_state_id": planned["init_state_id"],
            "verified_before_reset": True,
            "post_reset_init_state_id": planned["init_state_id"] + 1,
        },
        task_source_evidence={
            "suite": planned["suite"],
            "task_id": planned["task_id"],
            "task_name": planned["task_name"],
            "source_check": "pinned_hf_libero_paths",
        },
        worker_evidence=worker_evidence,
        gl_evidence=gl_evidence,
    )


def _child_provenance(module: Any, config: dict[str, Any], rows: list[dict[str, Any]], sha: str = "project-sha") -> dict[str, Any]:
    return {
        **config["provenance"],
        "git_sha": sha,
        "checkpoint_revision": config["checkpoint"]["revision"],
        "trajectory_ids": [row["trajectory_id"] for row in rows],
    }


def _genuine_worker_metrics(*, completed: int, wall: float, busy: float) -> dict[str, Any]:
    return {
        "wall_time_seconds": wall,
        "busy_time_seconds": busy,
        "completed": completed,
        "throughput_episodes_per_second": completed / wall,
        "utilization": busy / wall,
    }


def _write_child_manifest(module: Any, directory: Path, config: dict[str, Any], rows: list[dict[str, Any]], *, worker: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    aggregate = module.aggregate_results([], planned=rows, worker_metrics={worker: {"wall_time_seconds": 2.0, "busy_time_seconds": 0.0}})
    manifest = module.build_run_manifest(
        config=config,
        config_path=CONFIG_PATH,
        run_directory=directory,
        project_sha="project-sha",
        matrix=rows,
        aggregate=aggregate,
        provenance=_child_provenance(module, config, rows),
        status="BLOCKED",
        scope={"type": "worker", "worker": worker, "physical_device": rows[0]["physical_device"]},
    )
    module.write_json_no_overwrite(directory / "run_manifest.json", manifest)
    return manifest


def test_config_freezes_exact_task_major_matrix_and_runtime_contract() -> None:
    baseline = _module()
    config = _config()

    identity = baseline.validate_config(config, expected_project_sha="sha", actual_project_sha="sha")

    assert identity["seed"] == 2027
    assert identity["suite"] == "libero_spatial"
    assert identity["task_ids"] == [0, 4, 9]
    assert identity["init_state_ids"] == [0, 1, 2, 3, 4]
    assert identity["planned_episodes"] == 15
    assert identity["runtime"]["physical_devices"] == [0, 1]
    assert identity["runtime"]["logical_device"] == "cuda:0"
    assert identity["perturbation"] is False
    assert identity["action_noise"]["explicit"] is False
    assert identity["task_names"] == {
        "0": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
            "4": "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "9": "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    }

    bad = deepcopy(config)
    bad["seed"] = 7
    with pytest.raises(baseline.BaselineConfigError, match="seed"):
        baseline.validate_config(bad, expected_project_sha="sha", actual_project_sha="sha")

    bad_name = deepcopy(config)
    bad_name["task_names"]["4"] = "not-the-pinned-libero-name"
    with pytest.raises(baseline.BaselineConfigError, match="task_names"):
        baseline.validate_config(bad_name, expected_project_sha="sha", actual_project_sha="sha")


def test_preflight_contract_hash_cross_identity_and_existing_gate_seam(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    preflight_path = CONFIG_PATH.parent / "dcu_preflight.yaml"
    config["preflight_config"] = str(preflight_path)
    dcu_config = {
        "schema_version": 1,
        "name": "dcu_preflight",
        "mode": "explicit_phase",
        "seed": 2027,
        "suite": "libero_spatial",
        "obs_type": "pixels_agent_pos",
        "observation_height": 360,
        "observation_width": 360,
        "action_dim": 7,
        "chunk_size": 50,
        "n_action_steps": 1,
        "horizon": 280,
        "task": {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027},
        "checkpoint": {"repo_id": config["checkpoint"]["repo_id"], "revision": config["checkpoint"]["revision"], "path": config["checkpoint"]["path"], "model_sha256": config["checkpoint"]["sha256"]},
        "base_model": {"repo_id": config["base_model"]["repo_id"], "revision": config["base_model"]["revision"], "path": config["base_model"]["path"], "model_sha256": config["base_model"]["sha256"]},
        "assets": {"repo_id": config["assets"]["repo_id"], "revision": config["assets"]["revision"], "path": config["assets"]["path"], "manifest_sha256": config["assets"]["manifest_sha256"]},
        "libero_config": config["libero_config"] if "libero_config" in config else {"path": "/frozen/libero/config.yaml", "sha256": "e" * 64},
        "runtime": {**{key: config["runtime"][key] for key in ("cpu_python", "dcu_python", "cpu_runtime_lock", "dcu_runtime_lock", "physical_devices", "logical_device", "worker_startup_timeout_seconds", "worker_forward_timeout_seconds", "worker_shutdown_timeout_seconds")}, "runtime_lock": config["runtime"]["cpu_runtime_lock"]},
        "offline": config["runtime"]["offline"],
        "renderer": config["runtime"]["renderer"],
    }
    calls: list[dict[str, Any]] = []
    fake_preflight = types.SimpleNamespace(
        _load_yaml_config=lambda path: dcu_config,
        run_preflight_gates=lambda **kwargs: calls.append(kwargs) or {"live": "preflight", "network": {"offline": True}},
    )

    evidence = baseline.validate_preflight_contract(
        config,
        expected_project_sha="project-sha",
        actual_project_sha="project-sha",
        run_directory=tmp_path / "run",
        preflight_module=fake_preflight,
    )

    assert evidence["config_sha256"] == config["preflight_config_sha256"]
    assert evidence["gates"]["live"] == "preflight"
    assert calls[0]["phase"] == "concurrency"
    assert calls[0]["prepare_assets"] is False
    bad = deepcopy(dcu_config)
    bad["checkpoint"]["revision"] = "drift"
    fake_preflight._load_yaml_config = lambda path: bad
    with pytest.raises(baseline.BaselineConfigError, match="checkpoint"):
        baseline.validate_preflight_contract(
            config,
            expected_project_sha="project-sha",
            actual_project_sha="project-sha",
            run_directory=tmp_path / "run2",
            preflight_module=fake_preflight,
            run_gates=False,
        )


def test_bounded_preflight_gate_runner_times_out_and_restores_signal_state() -> None:
    baseline = _module()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    try:
        with pytest.raises(baseline.BaselineRuntimeError, match="timed out"):
            baseline._run_preflight_gates_bounded(
                lambda: (time.sleep(0.05), {"live": "late"})[1],
                timeout_seconds=0.001,
            )
        assert signal.getsignal(signal.SIGALRM) is previous_handler
        restored_timer = signal.getitimer(signal.ITIMER_REAL)
        assert restored_timer[0] == pytest.approx(0.0)
        assert restored_timer[1] == pytest.approx(0.0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def test_bounded_preflight_gate_runner_returns_and_restores_signal_state() -> None:
    baseline = _module()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    try:
        result = baseline._run_preflight_gates_bounded(
            lambda: {"live": "preflight"},
            timeout_seconds=0.2,
        )
        assert result == {"live": "preflight"}
        assert signal.getsignal(signal.SIGALRM) is previous_handler
        restored_timer = signal.getitimer(signal.ITIMER_REAL)
        assert restored_timer[0] == pytest.approx(0.0)
        assert restored_timer[1] == pytest.approx(0.0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def test_bounded_preflight_gate_runner_fails_closed_off_main_thread() -> None:
    baseline = _module()
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            baseline._run_preflight_gates_bounded(lambda: {"live": "preflight"}, timeout_seconds=0.2)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], baseline.BaselineRuntimeError)
    assert "main thread" in str(failures[0])


def test_runtime_task_identity_must_match_the_planned_pinned_name() -> None:
    baseline = _module()
    planned = baseline.build_episode_matrix(_config())[0]

    assert baseline.validate_runtime_task_identity(
        planned,
        {"suite": planned["suite"], "task_id": planned["task_id"], "task_name": planned["task_name"]},
    ) is True
    with pytest.raises(baseline.BaselineRuntimeError, match="task_name"):
        baseline.validate_runtime_task_identity(
            planned,
            {"suite": planned["suite"], "task_id": planned["task_id"], "task_name": "wrong"},
        )


def test_episode_rows_recheck_identity_and_evidence_instead_of_trusting_flags() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    row = _episode(baseline, matrix[0], success=True)

    bad_identity = deepcopy(row)
    bad_identity["physical_device"] = 1
    with pytest.raises(baseline.BaselineSchemaError, match="physical_device"):
        baseline.aggregate_results([bad_identity], planned=matrix)

    bad_action = deepcopy(row)
    bad_action["actions_finite"] = True
    bad_action["action_evidence"][0]["values"][0][0] = "nan"
    with pytest.raises(baseline.BaselineSchemaError, match="action"):
        baseline.aggregate_results([bad_action], planned=matrix)

    bad_offline = deepcopy(row)
    bad_offline["offline"] = False
    with pytest.raises(baseline.BaselineSchemaError, match="offline"):
        baseline.aggregate_results([bad_offline], planned=matrix)


def test_failure_record_is_attempted_and_not_double_counted_with_runtime_failure() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    failure = baseline.make_episode_failure_record(
        matrix[0], reason="worker protocol failed", kind="runtime", wall_time_seconds=0.2
    )

    aggregate = baseline.aggregate_results(
        [failure],
        planned=matrix,
        runtime_failures=[{"episode_id": matrix[0]["episode_id"], "reason": "worker protocol failed"}],
    )

    assert failure["status"] == "crashed"
    assert failure["attempt_count"] == 1
    assert aggregate["attempted"] == 1
    assert aggregate["crashed"] == 1


def test_each_episode_gets_a_unique_ipc_directory_and_peak_memory_is_maximum() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    first = baseline.episode_ipc_directory(Path("/tmp/child-A"), matrix[0])
    second = baseline.episode_ipc_directory(Path("/tmp/child-A"), matrix[2])

    assert first != second
    assert first.name.startswith("0000-")
    assert second.name.startswith("0002-")

    class FakeTransport:
        responses = [
            {"command": "ping", "peak_memory_bytes": 10},
            {"command": "select_action", "peak_memory_bytes": 300},
            {"command": "select_action", "peak_memory_bytes": 200},
        ]

    class FakeClient:
        transport = FakeTransport()

    assert baseline.worker_peak_memory_bytes(FakeClient()) == 300


def test_existing_worker_timeouts_are_applied_and_parent_timeout_is_finite() -> None:
    baseline = _module()
    config = _config()

    class Remote:
        reset_timeout_seconds = None
        forward_timeout_seconds = None

    remote = Remote()
    baseline.configure_remote_timeouts(remote, config["runtime"])

    assert remote.reset_timeout_seconds == 300.0
    assert remote.forward_timeout_seconds == 300.0
    timeout = baseline.parent_collection_timeout_seconds(config)
    assert np.isfinite(timeout)
    assert timeout > 0


def test_runtime_environment_is_closed_for_cleanup() -> None:
    baseline = _module()
    closed: list[Any] = []
    baseline.close_runtime_environment(
        {"envs": "all-envs", "close_envs": lambda envs: closed.append(envs)}
    )
    assert closed == ["all-envs"]


def test_config_rejects_runtime_lock_or_offline_drift() -> None:
    baseline = _module()
    config = _config()

    bad_lock = deepcopy(config)
    bad_lock["runtime"]["cpu_runtime_lock_sha256"] = "not-a-sha"
    with pytest.raises(baseline.BaselineConfigError, match="SHA-256"):
        baseline.validate_config(bad_lock, expected_project_sha="sha", actual_project_sha="sha")

    bad_offline = deepcopy(config)
    bad_offline["runtime"]["offline"]["HF_HUB_OFFLINE"] = "0"
    with pytest.raises(baseline.BaselineConfigError, match="HF_HUB_OFFLINE"):
        baseline.validate_config(bad_offline, expected_project_sha="sha", actual_project_sha="sha")


def test_load_config_reads_project_owned_baseline_file_without_runtime_imports() -> None:
    baseline = _module()

    config = baseline.load_config(CONFIG_PATH)

    assert config["name"] == "baseline_a"
    assert config["task_ids"] == [0, 4, 9]
    assert config["init_state_ids"] == [0, 1, 2, 3, 4]


def test_episode_matrix_is_stable_task_major_and_assigned_by_zero_based_index() -> None:
    baseline = _module()
    config = _config()

    matrix = baseline.build_episode_matrix(config)

    assert len(matrix) == 15
    assert [row["order"] for row in matrix] == list(range(15))
    assert [(row["task_id"], row["init_state_id"]) for row in matrix] == [
        (task_id, init_state_id)
        for task_id in [0, 4, 9]
        for init_state_id in [0, 1, 2, 3, 4]
    ]
    assert [row["worker"] for row in matrix] == ["A", "B"] * 7 + ["A"]
    assert [row["physical_device"] for row in matrix] == [0, 1] * 7 + [0]
    assert all(row["logical_device"] == "cuda:0" for row in matrix)
    assert len({row["episode_id"] for row in matrix}) == 15

    baseline.validate_episode_matrix(matrix, config)
    with pytest.raises(baseline.BaselineSchemaError, match="duplicate"):
        baseline.validate_episode_matrix(matrix + [matrix[0]], config)


def test_worker_environment_separates_cpu_egl_and_nested_compute_visibility() -> None:
    baseline = _module()
    config = _config()
    parent = {
        "HIP_VISIBLE_DEVICES": "99",
        "CUDA_VISIBLE_DEVICES": "99",
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "MUJOCO_EGL_DEVICE_ID": "0",
    }

    cpu = baseline.build_cpu_child_environment(config, parent)
    nested_a = baseline.build_nested_worker_environment(config, 0, parent)
    nested_b = baseline.build_nested_worker_environment(config, 1, parent)

    assert cpu.get("HIP_VISIBLE_DEVICES") is None
    assert cpu.get("CUDA_VISIBLE_DEVICES") is None
    assert cpu["MUJOCO_GL"] == "egl"
    assert cpu["PYOPENGL_PLATFORM"] == "egl"
    assert cpu["MUJOCO_EGL_DEVICE_ID"] == "8"
    assert nested_a["HIP_VISIBLE_DEVICES"] == "0"
    assert nested_b["HIP_VISIBLE_DEVICES"] == "1"
    assert nested_a["CUDA_VISIBLE_DEVICES"] == "0"
    assert nested_b["CUDA_VISIBLE_DEVICES"] == "0"
    assert nested_a["MUJOCO_EGL_DEVICE_ID"] == "8"


def test_action_contract_requires_exact_postprocessed_float32_batch() -> None:
    baseline = _module()
    action = np.asarray([[1, 2, 3, 4, 5, 6, 7]], dtype=np.float32)

    evidence = baseline.validate_action(action)

    assert evidence == {
        "shape": [1, 7],
        "dtype": "float32",
        "finite": True,
        "values": action.tolist(),
    }
    with pytest.raises(baseline.BaselineSchemaError, match="float32"):
        baseline.validate_action(action.astype(np.float64))
    with pytest.raises(baseline.BaselineSchemaError, match="finite"):
        baseline.validate_action(np.asarray([[np.nan] * 7], dtype=np.float32))
    with pytest.raises(baseline.BaselineSchemaError, match="shape"):
        baseline.validate_action(np.zeros((7,), dtype=np.float32))


def test_official_horizon_termination_is_required_when_episode_is_not_done() -> None:
    baseline = _module()

    assert baseline.termination_reason(
        terminated=False, truncated=True, success=False, steps=280, horizon=280
    ) == "official_horizon"
    assert baseline.termination_reason(
        terminated=True, truncated=False, success=True, steps=8, horizon=280
    ) == "environment_termination"
    with pytest.raises(baseline.BaselineSchemaError, match="horizon"):
        baseline.termination_reason(
            terminated=False, truncated=False, success=False, steps=8, horizon=280
        )


def test_episode_record_enforces_one_attempt_no_retry_and_exact_action_evidence() -> None:
    baseline = _module()
    planned = baseline.build_episode_matrix(_config())[0]

    record = _episode(baseline, planned, success=True, steps=3)

    assert record["episode_id"] == planned["episode_id"]
    assert record["attempt_count"] == 1
    assert record["no_retry"] is True
    assert record["actions_finite"] is True
    assert record["actions_dtype"] == "float32"
    assert record["actions_shape"] == [[1, 7], [1, 7], [1, 7]]
    assert record["offline"] is True
    assert record["hub_fallback"] is False
    assert record["policy_calls"] == 3
    assert record["inference_p50_seconds"] == pytest.approx(0.02)
    with pytest.raises(baseline.BaselineSchemaError, match="attempt"):
        baseline.make_episode_record(planned, **{
            "success": True,
            "rewards": [1.0],
            "terminated": True,
            "truncated": False,
            "steps": 1,
            "policy_calls": 1,
            "inference_latencies": [0.1],
            "env_step_latencies": [0.1],
            "actions": [np.zeros((1, 7), dtype=np.float32)],
            "model_peak_memory_bytes": 1,
            "wall_time_seconds": 0.1,
            "attempt_count": 2,
        })


def test_aggregate_math_reports_zero_sr_as_needs_investigation_and_worker_utilization() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    episodes = [_episode(baseline, matrix[0], success=False), _episode(baseline, matrix[1], success=False)]

    aggregate = baseline.aggregate_results(
        episodes,
        planned=matrix,
        total_wall_time_seconds=2.0,
        worker_metrics={
            "A": {"wall_time_seconds": 2.0, "busy_time_seconds": 0.5},
            "B": {"wall_time_seconds": 2.0, "busy_time_seconds": 0.75},
        },
    )

    assert aggregate["total"] == 15
    assert aggregate["planned"] == 15
    assert aggregate["attempted"] == 2
    assert aggregate["completed"] == 2
    assert aggregate["failed"] == 2
    assert aggregate["crashed"] == 0
    assert aggregate["overall_sr"] == 0.0
    assert aggregate["mean_steps_to_success"] is None
    assert aggregate["per_task_sr"]["0"] == 0.0
    assert aggregate["per_worker"]["A"]["utilization"] == pytest.approx(0.25)
    assert "not hardware utilization" in aggregate["per_worker"]["A"]["utilization_definition"]
    assert aggregate["per_worker"]["B"]["throughput_episodes_per_second"] == pytest.approx(0.5)
    # The planned matrix is incomplete, so a partial zero-SR run is blocked;
    # the all-15 zero-SR case below is the NEEDS_INVESTIGATION verdict.
    assert aggregate["final_verdict"] == "BLOCKED"


def test_parent_run_uses_parent_observed_worker_timing_not_forged_child_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    config = baseline.load_config(CONFIG_PATH)
    matrix = baseline.build_episode_matrix(config)
    episodes = [_episode(baseline, row, success=True) for row in matrix]
    children: list[dict[str, Any]] = []
    forged_metrics: dict[str, Any] = {}
    for worker, device, start, finish in (("A", 0, 10.0, 20.0), ("B", 1, 10.0, 30.0)):
        assigned = [row for row in matrix if row["worker"] == worker]
        children.append(
            {
                "worker": worker,
                "physical_device": device,
                "planned": assigned,
                "directory": tmp_path / f"child-{worker}",
                "returncode": 0,
                "parent_observed_start_monotonic": start,
                "parent_observed_finish_monotonic": finish,
            }
        )
        # The forged child report is internally formula-consistent but must not
        # determine the parent aggregate.
        forged_metrics[worker] = _genuine_worker_metrics(
            completed=len(assigned), wall=0.25, busy=0.1
        )
    for child in children:
        Path(child["directory"]).mkdir(parents=True)

    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"live": True})
    monkeypatch.setattr(baseline, "launch_children", lambda *_, **__: children)
    monkeypatch.setattr(baseline, "collect_children_fail_closed", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        baseline,
        "_read_child_artifacts",
        lambda *_args, **_kwargs: (episodes, forged_metrics, []),
    )
    seen_parent_metrics: list[dict[str, Any]] = []
    original_aggregate = baseline.aggregate_results

    def capture_aggregate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("require_worker_metrics"):
            seen_parent_metrics.append(deepcopy(kwargs["worker_metrics"]))
        return original_aggregate(*args, **kwargs)

    monkeypatch.setattr(baseline, "aggregate_results", capture_aggregate)

    assert baseline.run(CONFIG_PATH, expected_project_sha="sha", output_root=tmp_path / "runs") == 0
    assert len(seen_parent_metrics) == 1
    assert seen_parent_metrics[0]["A"]["wall_time_seconds"] == pytest.approx(10.0)
    assert seen_parent_metrics[0]["B"]["wall_time_seconds"] == pytest.approx(20.0)
    assert seen_parent_metrics[0]["A"]["busy_time_seconds"] == pytest.approx(4.0)
    assert seen_parent_metrics[0]["B"]["busy_time_seconds"] == pytest.approx(3.5)
    assert seen_parent_metrics[0]["A"]["throughput_episodes_per_second"] == pytest.approx(0.8)
    assert seen_parent_metrics[0]["B"]["throughput_episodes_per_second"] == pytest.approx(0.35)


def test_parent_worker_timing_requires_both_parent_monotonic_observations() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    episodes = [_episode(baseline, row, success=False) for row in matrix]
    children = [
        {
            "worker": worker,
            "physical_device": device,
            "planned": [row for row in matrix if row["worker"] == worker],
            "directory": Path(f"/tmp/child-{worker}"),
            "parent_observed_start_monotonic": 10.0,
        }
        for worker, device in (("A", 0), ("B", 1))
    ]

    with pytest.raises(baseline.BaselineRuntimeError, match="parent[_ ]observed"):
        baseline._derive_parent_worker_metrics(children, episodes)


def test_runtime_or_numerical_failure_is_blocked_and_unattempted_are_explicit() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    episodes = [_episode(baseline, matrix[0], success=True)]

    aggregate = baseline.aggregate_results(
        episodes,
        planned=matrix,
        runtime_failures=[{"worker": "B", "reason": "worker exited"}],
        numerical_anomalies=[{"episode_id": matrix[1]["episode_id"], "reason": "NaN action"}],
    )

    assert aggregate["crashed"] == 1
    assert aggregate["runtime_failures"][0]["worker"] == "B"
    assert aggregate["numerical_anomalies"][0]["episode_id"] == matrix[1]["episode_id"]
    assert matrix[1]["episode_id"] in aggregate["unattempted_episode_ids"]
    assert aggregate["final_verdict"] == "BLOCKED"


def test_partial_matrix_without_terminal_runtime_evidence_is_blocked() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    episodes = [_episode(baseline, row, success=False) for row in matrix]
    episodes = episodes[:-1]

    aggregate = baseline.aggregate_results(episodes, planned=matrix)

    assert aggregate["attempted"] == 14
    assert aggregate["unattempted_episode_ids"] == [matrix[-1]["episode_id"]]
    assert aggregate["final_verdict"] == "BLOCKED"


def test_all_fifteen_unsuccessful_horizon_episodes_are_needs_investigation() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    episodes = [_episode(baseline, row, success=False) for row in matrix]

    aggregate = baseline.aggregate_results(episodes, planned=matrix)

    assert aggregate["attempted"] == aggregate["completed"] == 15
    assert aggregate["failed"] == 15
    assert aggregate["crashed"] == 0
    assert aggregate["overall_sr"] == 0.0
    assert aggregate["final_verdict"] == "NEEDS_INVESTIGATION"


@pytest.mark.parametrize(
    "worker_metrics",
    [
        {},
        {"A": _genuine_worker_metrics(completed=8, wall=2.0, busy=1.0)},
        {
            "A": {"wall_time_seconds": 2.0, "busy_time_seconds": 1.0},
            "B": _genuine_worker_metrics(completed=7, wall=3.0, busy=1.5),
        },
        {
            "A": _genuine_worker_metrics(completed=8, wall=2.0, busy=1.0),
            "B": {"wall_time_seconds": 3.0, "busy_time_seconds": 1.5},
        },
    ],
    ids=["missing-both", "missing-b", "invalid-a", "invalid-b"],
)
def test_exact_all_zero_matrix_with_missing_or_invalid_explicit_metrics_is_blocked(
    worker_metrics: dict[str, Any],
) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    episodes = [_episode(baseline, row, success=False) for row in matrix]

    aggregate = baseline.aggregate_results(
        episodes,
        planned=matrix,
        total_wall_time_seconds=5.0,
        worker_metrics=worker_metrics,
        require_worker_metrics=True,
    )

    assert aggregate["gates"]["G"] is True
    assert aggregate["gates"]["H"] is False
    assert aggregate["final_verdict"] == "BLOCKED"


def test_exact_all_zero_matrix_with_genuine_explicit_metrics_needs_investigation() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    episodes = [_episode(baseline, row, success=False) for row in matrix]
    worker_metrics = {
        "A": _genuine_worker_metrics(completed=8, wall=2.0, busy=1.0),
        "B": _genuine_worker_metrics(completed=7, wall=3.0, busy=1.5),
    }

    aggregate = baseline.aggregate_results(
        episodes,
        planned=matrix,
        total_wall_time_seconds=5.0,
        worker_metrics=worker_metrics,
        require_worker_metrics=True,
    )

    assert aggregate["gates"]["H"] is True
    assert aggregate["final_verdict"] == "NEEDS_INVESTIGATION"


def test_child_scoped_explicit_metrics_accept_only_the_assigned_worker() -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    assigned = [row for row in matrix if row["worker"] == "A"]
    episodes = [_episode(baseline, row, success=False) for row in assigned]

    aggregate = baseline.aggregate_results(
        episodes,
        planned=assigned,
        total_wall_time_seconds=2.0,
        worker_metrics={"A": _genuine_worker_metrics(completed=8, wall=2.0, busy=1.0)},
        require_worker_metrics=True,
        worker_scope="A",
    )

    assert aggregate["gates"]["H"] is True
    assert aggregate["final_verdict"] == "NEEDS_INVESTIGATION"


def test_init_state_selection_is_verified_before_reset_and_records_post_reset_increment() -> None:
    baseline = _module()

    class FakeVectorEnv:
        def __init__(self) -> None:
            self.init_state_id = 0
            self.events: list[tuple[str, int]] = []

        def set_attr(self, name: str, value: int) -> None:
            assert name == "init_state_id"
            self.init_state_id = value
            self.events.append(("set", value))

        def get_attr(self, name: str) -> list[int]:
            assert name == "init_state_id"
            self.events.append(("get", self.init_state_id))
            return [self.init_state_id]

        def reset(self) -> None:
            self.events.append(("reset", self.init_state_id))
            self.init_state_id += 1

    env = FakeVectorEnv()
    evidence = baseline.select_init_state_before_reset(env, 4)
    env.reset()
    evidence["post_reset_init_state_id"] = env.get_attr("init_state_id")[0]

    assert evidence == {
        "selected_init_state_id": 4,
        "verified_before_reset": True,
        "post_reset_init_state_id": 5,
    }
    assert env.events[:3] == [("set", 4), ("get", 4), ("reset", 4)]


def test_launcher_starts_both_children_before_wait_and_assigns_exact_serial_lists(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    calls: list[tuple[str, Any]] = []

    class FakeProcess:
        _next_pid = 100

        def __init__(self, command: list[str], **_: Any) -> None:
            self.command = command
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.returncode = 0

        def communicate(self) -> tuple[str, str]:
            calls.append(("wait", self.command[ self.command.index("--worker") + 1]))
            return "", ""

        def poll(self) -> int:
            return self.returncode

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append(("launch", command[command.index("--worker") + 1]))
        return FakeProcess(command, **kwargs)

    children = baseline.launch_children(
        config,
        config_path=CONFIG_PATH,
        expected_project_sha="sha",
        run_directory=tmp_path / "run",
        popen=popen,
        project_python="/frozen/cpu/bin/python",
    )

    assert [kind for kind, _ in calls] == ["launch", "launch"]
    assert [item["worker"] for item in children] == ["A", "B"]
    assert [item["physical_device"] for item in children] == [0, 1]
    assert [row["episode_id"] for row in children[0]["planned"]] == [
        row["episode_id"] for row in matrix if row["worker"] == "A"
    ]
    assert [row["episode_id"] for row in children[1]["planned"]] == [
        row["episode_id"] for row in matrix if row["worker"] == "B"
    ]
    assert all(isinstance(item.get("parent_observed_start_monotonic"), float) for item in children)
    assert all(isinstance(item.get("parent_observed_finish_monotonic"), float) for item in children)
    assert all(
        item["parent_observed_finish_monotonic"] >= item["parent_observed_start_monotonic"]
        for item in children
    )


def test_launcher_registers_second_child_before_popen_failure_and_terminalizes_both(
    tmp_path: Path,
) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    launched: list[str] = []

    class FakeProcess:
        _next_pid = 700

        def __init__(self) -> None:
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.returncode = 0

        def poll(self) -> int:
            return self.returncode

    def popen(command: list[str], **_: Any) -> FakeProcess:
        worker = command[command.index("--worker") + 1]
        launched.append(worker)
        if worker == "B":
            raise OSError("injected second-child Popen failure")
        return FakeProcess()

    children: list[dict[str, Any]] = []
    with pytest.raises(baseline.BaselineRuntimeError, match="second-child Popen failure"):
        baseline.launch_children(
            config,
            config_path=CONFIG_PATH,
            expected_project_sha="project-sha",
            run_directory=tmp_path / "run",
            popen=popen,
            project_python="/frozen/cpu/bin/python",
            children_out=children,
        )

    assert launched == ["A", "B"]
    assert [child["worker"] for child in children] == ["A", "B"]
    for child in children:
        assert child["stdout_handle"].closed
        assert child["stderr_handle"].closed
        assert child["log_cleanup"]["closed"] is True
        assert child["ipc_cleanup"]["strict_under_child"] is True
        terminal_path = Path(child["directory"]) / "terminal_manifest.json"
        assert terminal_path.is_file()
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        assigned_ids = [row["episode_id"] for row in matrix if row["worker"] == child["worker"]]
        assert terminal["status"] == "CRASHED"
        assert terminal["worker"] == child["worker"]
        assert terminal["physical_device"] == child["physical_device"]
        assert terminal["planned_episode_ids"] == assigned_ids
        assert terminal["attempted_episode_ids"] == []


def test_production_run_uses_launcher_then_bounded_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline = _module()
    events: list[str] = []
    matrix = baseline.build_episode_matrix(baseline.load_config(CONFIG_PATH))
    children: list[dict[str, Any]] = []

    def fake_preflight(*_: Any, **__: Any) -> dict[str, Any]:
        events.append("preflight")
        return {"live": "preflight"}

    def fake_launch(config: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        del config
        events.append("launch")
        assert kwargs["wait"] is False
        assert kwargs["config_path"] == CONFIG_PATH.resolve()
        for worker, device in zip(("A", "B"), (0, 1), strict=True):
            planned = [row for row in matrix if row["worker"] == worker]
            directory = tmp_path / f"fake-child-{worker}"
            directory.mkdir()
            children.append({"worker": worker, "physical_device": device, "planned": planned, "directory": directory, "returncode": 0})
        return children

    def fake_collect(actual_children: list[dict[str, Any]], *, timeout_seconds: float) -> list[dict[str, Any]]:
        events.append("collect")
        assert actual_children is children
        assert np.isfinite(timeout_seconds) and timeout_seconds > 0
        return []

    monkeypatch.setattr(baseline, "validate_preflight_contract", fake_preflight)
    monkeypatch.setattr(baseline, "launch_children", fake_launch)
    monkeypatch.setattr(baseline, "collect_children_fail_closed", fake_collect)
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")

    result = baseline.run(CONFIG_PATH, expected_project_sha="sha", output_root=tmp_path / "runs")

    assert result == 1  # no episodes were supplied by the fake children
    assert events == ["preflight", "launch", "collect"]
    assert [row["episode_id"] for row in children[0]["planned"]] == [row["episode_id"] for row in matrix if row["worker"] == "A"]
    assert [row["episode_id"] for row in children[1]["planned"]] == [row["episode_id"] for row in matrix if row["worker"] == "B"]


def test_production_run_terminalizes_parent_on_launch_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline = _module()
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"live": "preflight"})

    def fail_launch(*_: Any, **__: Any) -> list[dict[str, Any]]:
        raise baseline.BaselineRuntimeError("second child could not launch")

    monkeypatch.setattr(baseline, "launch_children", fail_launch)

    result = baseline.run(CONFIG_PATH, expected_project_sha="sha", output_root=tmp_path / "runs")

    assert result == 1
    run_directory = next((tmp_path / "runs").iterdir())
    terminal = json.loads((run_directory / "terminal_manifest.json").read_text())
    assert terminal["status"] == "BLOCKED"
    assert "could not launch" in terminal["reason"]
    assert terminal["unattempted_episode_ids"]


def test_production_run_terminalizes_parent_on_collection_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(baseline.load_config(CONFIG_PATH))
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"live": "preflight"})

    children: list[dict[str, Any]] = []

    def fake_launch(*_: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["wait"] is False
        for worker, device in zip(("A", "B"), (0, 1), strict=True):
            children.append({"worker": worker, "physical_device": device, "planned": [row for row in matrix if row["worker"] == worker], "directory": tmp_path / worker, "returncode": -15})
        return children

    monkeypatch.setattr(baseline, "launch_children", fake_launch)
    monkeypatch.setattr(
        baseline,
        "collect_children_fail_closed",
        lambda *_args, **_kwargs: [{"worker": "B", "reason": "collection timeout"}],
    )

    result = baseline.run(CONFIG_PATH, expected_project_sha="sha", output_root=tmp_path / "runs")

    assert result == 1
    run_directory = next((tmp_path / "runs").iterdir())
    terminal = json.loads((run_directory / "terminal_manifest.json").read_text())
    assert terminal["status"] == "BLOCKED"
    assert "collection timeout" in terminal["reason"]


def test_parent_execution_failure_removes_orphan_child_ipc_before_terminalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(baseline.load_config(CONFIG_PATH))
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"live": "preflight"})

    child_directory = tmp_path / "child-A"
    orphan = child_directory / "ipc" / "orphan-request"
    orphan.mkdir(parents=True)
    (orphan / "request.safetensors").write_bytes(b"orphan")

    def fake_launch(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{
            "worker": "A",
            "physical_device": 0,
            "planned": [row for row in matrix if row["worker"] == "A"],
            "directory": child_directory,
        }]

    monkeypatch.setattr(baseline, "launch_children", fake_launch)
    monkeypatch.setattr(
        baseline,
        "collect_children_fail_closed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("collection exploded")),
    )

    assert baseline.run(CONFIG_PATH, expected_project_sha="sha", output_root=tmp_path / "runs") == 1
    assert not orphan.exists()


def test_child_episode_exception_is_one_attempted_failure_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"identity": "verified"})

    class FakeClient:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        baseline,
        "_run_episode_official",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("episode protocol failure")),
    )
    dcu_module = sys.modules.get("scripts.dcu_preflight") or __import__("scripts.dcu_preflight", fromlist=["x"])
    monkeypatch.setattr(dcu_module, "_start_worker", lambda *_args, **_kwargs: {"client": FakeClient(), "ping": {"ok": True}})
    monkeypatch.setattr(
        dcu_module,
        "_worker_summary",
        lambda *_args, **_kwargs: {
            "physical_device": 0,
            "logical_device": "cuda:0",
            "torch_logical_device": "cuda:0",
            "model_load_success": True,
            "compute_environment": {
                "physical_k100": 0,
                "HIP_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        },
    )

    child_directory = tmp_path / "child-A"
    result = baseline._run_child(
        CONFIG_PATH,
        expected_project_sha="sha",
        worker="A",
        physical_device=0,
        child_directory=child_directory,
    )

    assert result == 1
    rows = [json.loads(line) for line in (child_directory / "episodes.jsonl").read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["episode_id"] == matrix[0]["episode_id"]
    assert rows[0]["status"] == "crashed"
    assert rows[0]["attempt_count"] == 1
    aggregate = json.loads((child_directory / "aggregate.json").read_text())
    assert aggregate["attempted"] == 1
    assert aggregate["crashed"] == 1
    assert len(aggregate["runtime_failures"]) == 1
    terminal = json.loads((child_directory / "terminal_manifest.json").read_text())
    assert terminal["attempted_episode_ids"] == [matrix[0]["episode_id"]]
    assert matrix[2]["episode_id"] in terminal["unattempted_episode_ids"]


def test_child_normal_manifest_is_accepted_with_rich_assigned_worker_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    config = baseline.load_config(CONFIG_PATH)
    matrix = baseline.build_episode_matrix(config)
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"identity": "verified"})

    class FakeClient:
        def close(self) -> None:
            pass

    monkeypatch.setattr(
        baseline,
        "_run_episode_official",
        lambda _config, planned, **_kwargs: {
            **_episode(baseline, planned, success=True),
            "wall_time_seconds": 0.001,
        },
    )
    dcu_module = sys.modules.get("scripts.dcu_preflight") or __import__("scripts.dcu_preflight", fromlist=["x"])
    monkeypatch.setattr(dcu_module, "_start_worker", lambda *_args, **_kwargs: {"client": FakeClient(), "ping": {"ok": True}})
    monkeypatch.setattr(
        dcu_module,
        "_worker_summary",
        lambda *_args, **_kwargs: {
            "physical_device": 0,
            "logical_device": "cuda:0",
            "torch_logical_device": "cuda:0",
            "model_load_success": True,
            "compute_environment": {
                "physical_k100": 0,
                "HIP_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        },
    )

    child_directory = tmp_path / "child-A"
    assert baseline._run_child(
        CONFIG_PATH,
        expected_project_sha="sha",
        worker="A",
        physical_device=0,
        child_directory=child_directory,
    ) == 0

    manifest = json.loads((child_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    children = [{
        "worker": "A",
        "physical_device": 0,
        "planned": [row for row in matrix if row["worker"] == "A"],
        "directory": child_directory,
    }]
    rows, metrics, failures = baseline._read_child_artifacts(
        children,
        config=config,
        expected_project_sha="sha",
        config_path=CONFIG_PATH,
    )

    assert len(rows) == 8
    assert metrics["A"] == manifest["aggregate"]["per_worker"]["A"]
    assert failures == []


def test_child_terminalization_marks_unattempted_without_overwriting_terminal_evidence(tmp_path: Path) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    path = tmp_path / "child-A" / "terminal_manifest.json"

    manifest = baseline.terminalize_child(
        path,
        worker="A",
        physical_device=0,
        planned=matrix[:3],
        attempted=[matrix[0]],
        reason="schema failure",
        status="TERMINATED",
    )

    assert manifest["status"] == "TERMINATED"
    assert manifest["attempt_count"] == 1
    assert manifest["unattempted_episode_ids"] == [matrix[1]["episode_id"], matrix[2]["episode_id"]]
    with pytest.raises(FileExistsError):
        baseline.terminalize_child(
            path,
            worker="A",
            physical_device=0,
            planned=matrix[:3],
            attempted=[matrix[0]],
            reason="second write",
            status="TERMINATED",
        )


def test_no_overwrite_json_is_atomic(tmp_path: Path) -> None:
    baseline = _module()
    path = tmp_path / "artifact.json"

    baseline.write_json_no_overwrite(path, {"value": 1})

    assert json.loads(path.read_text()) == {"value": 1}
    with pytest.raises(FileExistsError):
        baseline.write_json_no_overwrite(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_json_safe_normalizes_nested_sets_with_deterministic_order() -> None:
    baseline = _module()
    value = {
        "outer": {"zeta", "alpha", "mu"},
        "nested": [{"numbers": {3, 1, 2}}],
    }

    normalized = baseline._json_safe(value)

    assert normalized == {
        "outer": ["alpha", "mu", "zeta"],
        "nested": [{"numbers": [1, 2, 3]}],
    }
    assert baseline._canonical_json(value) == baseline._canonical_json(value)


def test_parent_run_manifest_publishes_nested_set_preflight_evidence(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    episodes = [_episode(baseline, row, success=True, steps=3) for row in matrix]
    worker_metrics = {
        "A": _genuine_worker_metrics(completed=8, wall=2.0, busy=1.0),
        "B": _genuine_worker_metrics(completed=7, wall=3.0, busy=1.5),
    }
    aggregate = baseline.aggregate_results(
        episodes,
        planned=matrix,
        total_wall_time_seconds=5.0,
        worker_metrics=worker_metrics,
        require_worker_metrics=True,
    )
    provenance = {
        **config["provenance"],
        "git_sha": "project-sha",
        "checkpoint_revision": config["checkpoint"]["revision"],
        "trajectory_ids": [row["trajectory_id"] for row in matrix],
    }
    run_directory = tmp_path / "run"

    baseline._write_run_artifacts(
        run_directory,
        config=config,
        config_path=CONFIG_PATH,
        project_sha="project-sha",
        matrix=matrix,
        episodes=episodes,
        aggregate=aggregate,
        provenance=provenance,
        status="PASS",
        command=["test"],
        preflight_evidence={"patch": {"files": {"b.py", "a.py"}}},
    )

    published = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert published["preflight"]["patch"]["files"] == ["a.py", "b.py"]
    assert baseline.validate_manifest(published) is True


def test_terminal_fallback_publishes_json_safe_evidence(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    run_directory = tmp_path / "run"

    baseline._write_terminal_parent_fallback(
        run_directory=run_directory,
        config=config,
        matrix=matrix,
        episodes=[],
        reason="serialization failure",
        runtime_failures=[{"reason": "serialization failure"}],
        preflight_evidence={"patch": {"files": {"b.py", "a.py"}}},
    )

    published = json.loads((run_directory / "terminal_manifest.json").read_text(encoding="utf-8"))
    assert published["preflight"]["patch"]["files"] == ["a.py", "b.py"]
    assert published["status"] == "BLOCKED"


def test_terminal_fallback_surfaces_publication_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _module()

    def fail_writer(*_: Any, **__: Any) -> Path:
        raise OSError("terminal publication failed")

    monkeypatch.setattr(baseline, "write_json_no_overwrite", fail_writer)
    with pytest.raises(baseline.BaselineRuntimeError, match="terminal publication failed"):
        baseline._write_terminal_parent_fallback(
            run_directory=tmp_path / "run",
            config=_config(),
            matrix=baseline.build_episode_matrix(_config()),
            episodes=[],
            reason="primary failure",
            runtime_failures=[{"reason": "primary failure"}],
            preflight_evidence={},
        )


def test_run_manifest_schema_requires_full_provenance_and_gate_verdict() -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    aggregate = baseline.aggregate_results([], planned=matrix)
    manifest = baseline.build_run_manifest(
        config=config,
        config_path=CONFIG_PATH,
        run_directory=Path("/tmp/baseline-run"),
        project_sha="project-sha",
        matrix=matrix,
        aggregate=aggregate,
        provenance={
            "python": "3.11",
            "pytorch": "2.x",
            "transformers": "4.x",
            "cuda": "none-under-hip",
            "hip": "6.x",
            "gpu": "K100",
            "git_sha": "project-sha",
            "lerobot_git_sha": "lerobot-sha",
            "libero_git_sha": "libero-sha",
            "checkpoint_revision": "checkpoint-revision",
            "trajectory_ids": [],
        },
    )

    assert baseline.validate_manifest(manifest) is True
    assert manifest["schema_version"] == 1
    assert manifest["gate_verdict"] == aggregate["final_verdict"]
    assert manifest["provenance"]["action_noise_source"] == "native_policy_rng"
    assert manifest["provenance"]["perturbation"] is False
    broken = deepcopy(manifest)
    del broken["provenance"]["git_sha"]
    with pytest.raises(baseline.BaselineSchemaError, match="git_sha"):
        baseline.validate_manifest(broken)


def test_validate_only_persists_complete_planned_artifact_set_without_execution(tmp_path: Path) -> None:
    baseline = _module()

    run_directory, manifest = baseline.validate_only(
        CONFIG_PATH,
        expected_project_sha="project-sha",
        actual_project_sha="project-sha",
        output_root=tmp_path,
    )

    assert manifest["status"] == "VALIDATED"
    assert manifest["aggregate"]["planned"] == 15
    assert manifest["aggregate"]["attempted"] == 0
    assert manifest["preflight"]["config_sha256"] == baseline.EXPECTED_PREFLIGHT_CONFIG_SHA256
    assert manifest["preflight"]["cross_identity"]["exact"] is True
    assert sorted(path.name for path in run_directory.iterdir()) == sorted(
        [
            "aggregate.json",
            "command.txt",
            "config_resolved.json",
            "episode_matrix.json",
            "episodes.jsonl",
            "run_manifest.json",
            "stderr.log",
            "stdout.log",
        ]
    )
    assert (run_directory / "episodes.jsonl").read_text() == ""


def test_validate_only_direct_cli_bootstraps_repo_root_without_pythonpath(tmp_path: Path) -> None:
    baseline = _module()
    project_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    output_root = tmp_path / "runs"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "m0_baseline_a.py"),
            "--config",
            str(CONFIG_PATH),
            "--expected-project-sha",
            project_sha,
            "--seed",
            "2027",
            "--validate-only",
            "--output-root",
            str(output_root),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    run_directories = [path for path in output_root.iterdir() if path.is_dir()]
    assert len(run_directories) == 1
    manifest = json.loads((run_directories[0] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "VALIDATED"
    assert manifest["preflight"] == {
        "config_path": str(ROOT / "configs" / "m0" / "dcu_preflight.yaml"),
        "config_sha256": baseline.EXPECTED_PREFLIGHT_CONFIG_SHA256,
        "cross_identity": {
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
        },
        "gates": {"not_run": True},
        "phase": "concurrency",
        "prepare_assets": False,
    }


@pytest.mark.parametrize(
    ("needle", "replacement", "failure_match"),
    [
        (
            "revision: 6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
            "revision: drifted-checkpoint-revision",
            "checkpoint",
        ),
        (
            "revision: 7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            "revision: drifted-base-revision",
            "base_model",
        ),
        (
            "revision: 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
            "revision: drifted-assets-revision",
            "assets",
        ),
        (
            "sha256: 71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f",
            "sha256: " + "d" * 64,
            "checkpoint",
        ),
        (
            "path: /public/home/xuyinghao/workspace/vla/external/artifacts/smolvla_libero/6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
            "path: /frozen/drift-checkpoint",
            "checkpoint",
        ),
        (
            "sha256: b9bfd456c9472c0acd5719d6e514c4b859891af205ee1a736552fd3497b8b0c3",
            "sha256: " + "e" * 64,
            "base_model",
        ),
        (
            "path: /public/home/xuyinghao/workspace/vla/external/artifacts/smolvlm2-500m-instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            "path: /frozen/drift-base-model",
            "base_model",
        ),
        (
            "manifest_sha256: 1a94ebb8cc42614744d8dc9ebdad16f5461006f8806cc7d75869118013198a85",
            "manifest_sha256: " + "f" * 64,
            "assets",
        ),
        (
            "path: /public/home/xuyinghao/workspace/vla/external/artifacts/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
            "path: /frozen/drift-assets",
            "assets",
        ),
        (
            "path: /public/home/xuyinghao/tmp/shiftvla-libero-config/config.yaml",
            "path: /frozen/drift-libero/config.yaml",
            "libero_config",
        ),
        (
            "cpu_python: /public/home/xuyinghao/tmp/shiftvla-libero/bin/python",
            "cpu_python: /frozen/drift-cpu/bin/python",
            "runtime",
        ),
        (
            "dcu_python: /public/home/xuyinghao/tmp/shiftvla-libero-dcu/bin/python",
            "dcu_python: /frozen/drift-dcu/bin/python",
            "runtime",
        ),
        (
            "cpu_runtime_lock: /public/home/xuyinghao/workspace/vla/runtime/locks/shiftvla-libero-runtime.txt",
            "cpu_runtime_lock: /frozen/drift-cpu.lock",
            "runtime",
        ),
        (
            "dcu_runtime_lock: /public/home/xuyinghao/workspace/vla/runtime/locks/shiftvla-libero-dcu-runtime.txt",
            "dcu_runtime_lock: /frozen/drift-dcu.lock",
            "runtime",
        ),
        (
            "cpu_runtime_lock_sha256: 921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc",
            "cpu_runtime_lock_sha256: " + "1" * 64,
            "runtime",
        ),
        (
            "dcu_runtime_lock_sha256: cc507d48c64e72a9d2f6552bc5fa638217f9e6a2addef0067ded3d0adce30ed2",
            "dcu_runtime_lock_sha256: " + "2" * 64,
            "runtime",
        ),
        (
            "sha256: 98d57e0d3d7b70bab5c66a110ecc333c737895e9dcd457c06ec94b4150e0ecc8",
            "sha256: " + "3" * 64,
            "libero_config",
        ),
        (
            "python: 3.11.16",
            "python: 3.12.0",
            "provenance.python",
        ),
    ],
)
def test_validate_only_rejects_drifted_pinned_identity(
    tmp_path: Path, needle: str, replacement: str, failure_match: str
) -> None:
    baseline = _module()
    config_path = tmp_path / "baseline_a.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(needle, replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(baseline.BaselineError, match=failure_match):
        baseline.validate_only(
            config_path,
            expected_project_sha="project-sha",
            actual_project_sha="project-sha",
            output_root=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def test_parent_terminalization_is_blocked_and_preserves_unattempted_matrix(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    episode = _episode(baseline, matrix[0], success=True)

    manifest = baseline.terminalize_parent(
        tmp_path / "run_manifest.json",
        config=config,
        matrix=matrix,
        episodes=[episode],
        reason="worker B crashed",
        runtime_failures=[{"worker": "B", "reason": "worker exited"}],
    )

    assert manifest["status"] == "BLOCKED"
    assert manifest["gate_verdict"] == "BLOCKED"
    assert matrix[1]["episode_id"] in manifest["unattempted_episode_ids"]


def test_task_names_match_pinned_hf_libero_task_map_without_importing_simulator() -> None:
    baseline = _module()
    source = ROOT / "external" / "hf-libero" / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets)
    )
    task_map = ast.literal_eval(assignment.value)
    expected = {str(task_id): task_map["libero_spatial"][task_id] for task_id in (0, 4, 9)}

    assert baseline.EXPECTED_TASK_NAMES == expected
    assert baseline.load_config(CONFIG_PATH)["task_names"] == expected


def test_completed_episode_requires_and_cross_checks_all_runtime_evidence() -> None:
    baseline = _module()
    planned = baseline.build_episode_matrix(_config())[0]
    row = _episode(baseline, planned, success=True)
    assert baseline.aggregate_results([row], planned=[planned])["completed"] == 1

    invalid_fields = {
        "rewards": lambda item: item["rewards"].__setitem__(0, float("nan")),
        "reward_sum": lambda item: item.__setitem__("reward_sum", 0.0),
        "inference_mean_seconds": lambda item: item.__setitem__("inference_mean_seconds", 9.0),
        "env_step_p95_seconds": lambda item: item.__setitem__("env_step_p95_seconds", 9.0),
        "wall_time_seconds": lambda item: item.__setitem__("wall_time_seconds", -1.0),
        "model_peak_memory_bytes": lambda item: item.__setitem__("model_peak_memory_bytes", -1),
        "selected_init_state_id": lambda item: item.__setitem__("selected_init_state_id", 99),
        "task_source_evidence": lambda item: item["task_source_evidence"].__setitem__("task_name", "wrong"),
        "worker_evidence": lambda item: item["worker_evidence"].__setitem__("model_load_success", False),
        "gl_evidence": lambda item: item["gl_evidence"]["gl_identity"].__setitem__("renderer", "wrong"),
    }
    for field, mutate in invalid_fields.items():
        bad = deepcopy(row)
        mutate(bad)
        with pytest.raises(baseline.BaselineSchemaError, match=field.split("_")[0]):
            baseline.aggregate_results([bad], planned=[planned])


def test_worker_peak_memory_can_be_scoped_to_current_episode_responses() -> None:
    baseline = _module()

    class FakeTransport:
        responses = [
            {"command": "select_action", "peak_memory_bytes": 900},
            {"command": "select_action", "peak_memory_bytes": 100},
            {"command": "select_action", "peak_memory_bytes": 300},
        ]

    class FakeClient:
        transport = FakeTransport()

    assert baseline.worker_peak_memory_bytes(FakeClient(), start_index=1) == 300


def test_episode_journal_is_per_event_atomic_and_recovery_turns_started_without_terminal_into_crash(tmp_path: Path) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    journal = tmp_path / "episode_journal"
    baseline.append_episode_journal(journal, matrix[0], event="attempt_start")
    baseline.append_episode_journal(journal, matrix[0], event="terminal", status="completed")
    baseline.append_episode_journal(journal, matrix[1], event="attempt_start")

    events = baseline.read_episode_journal(journal)
    assert [event["event"] for event in events] == ["attempt_start", "terminal", "attempt_start"]
    assert journal.is_dir()
    assert len(list(journal.glob("*.json"))) == 3
    completed = _episode(baseline, matrix[0], success=True)
    recovered = baseline.recover_journal_episodes(journal, planned=matrix[:2], existing=[completed])
    assert [row["episode_id"] for row in recovered] == [matrix[1]["episode_id"]]
    assert recovered[0]["status"] == "crashed"
    assert recovered[0]["attempt_count"] == 1

    with pytest.raises(baseline.BaselineSchemaError, match="terminal"):
        baseline.append_episode_journal(journal, matrix[0], event="terminal", status="completed")


def test_torn_journal_keeps_valid_prefix_available_for_recovery(tmp_path: Path) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(_config())
    journal = tmp_path / "episode_journal"
    baseline.append_episode_journal(journal, matrix[0], event="attempt_start")
    (journal / "9999-torn-terminal.json").write_text('{"event":', encoding="utf-8")

    with pytest.raises(baseline.BaselineSchemaError) as raised:
        baseline.read_episode_journal(journal)
    assert [item["episode_id"] for item in raised.value.partial_events] == [matrix[0]["episode_id"]]


def test_terminal_episode_row_is_persisted_as_its_own_atomic_artifact(tmp_path: Path) -> None:
    baseline = _module()
    planned = baseline.build_episode_matrix(_config())[0]
    row = _episode(baseline, planned, success=True)
    path = baseline.persist_episode_row(tmp_path / "child-A", row)
    assert path.is_file()
    assert path.parent.name == "episode_rows"
    assert json.loads(path.read_text(encoding="utf-8"))["episode_id"] == planned["episode_id"]
    with pytest.raises(FileExistsError):
        baseline.persist_episode_row(tmp_path / "child-A", row)


def test_successful_aggregate_without_genuine_worker_metrics_cannot_pass() -> None:
    baseline = _module()
    planned = baseline.build_episode_matrix(_config())[0]
    row = _episode(baseline, planned, success=True)
    aggregate = baseline.aggregate_results([row], planned=[planned])
    assert aggregate["final_verdict"] != "PASS"
    assert aggregate["gates"]["H"] is False


def test_child_artifact_reader_reports_terminal_manifest_as_runtime_failure(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    assigned = [row for row in matrix if row["worker"] == "A"]
    child = tmp_path / "child-A"
    baseline.terminalize_child(
        child / "terminal_manifest.json",
        worker="A",
        physical_device=0,
        planned=assigned,
        attempted=[],
        reason="worker exited",
        status="CRASHED",
    )
    rows, metrics, failures = baseline._read_child_artifacts(
        [{"worker": "A", "physical_device": 0, "planned": assigned, "directory": child, "returncode": 1}],
        config=config,
        expected_project_sha="project-sha",
        config_path=CONFIG_PATH,
    )
    assert rows == []
    assert metrics == {}
    assert any("terminal" in str(item["reason"]).lower() for item in failures)


def test_child_artifact_reader_reports_missing_assigned_metrics_in_normal_manifest(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    assigned = [row for row in matrix if row["worker"] == "A"]
    episodes = [_episode(baseline, row, success=False) for row in assigned]
    worker_metrics = {"A": _genuine_worker_metrics(completed=8, wall=2.0, busy=1.0)}
    aggregate = baseline.aggregate_results(
        episodes,
        planned=assigned,
        total_wall_time_seconds=2.0,
        worker_metrics=worker_metrics,
        require_worker_metrics=True,
        worker_scope="A",
    )
    child = tmp_path / "child-A"
    manifest = baseline.build_run_manifest(
        config=config,
        config_path=CONFIG_PATH,
        run_directory=child,
        project_sha="project-sha",
        matrix=assigned,
        aggregate=aggregate,
        provenance={
            **_child_provenance(baseline, config, assigned),
            "worker_metrics": worker_metrics,
        },
        status="NEEDS_INVESTIGATION",
        scope={"type": "worker", "worker": "A", "physical_device": 0},
    )
    manifest["aggregate"]["per_worker"].pop("A")
    child.mkdir(parents=True, exist_ok=True)
    baseline.write_json_no_overwrite(child / "run_manifest.json", manifest)
    for row in episodes:
        baseline.persist_episode_row(child, row)
        baseline.append_episode_journal(child / baseline.EPISODE_JOURNAL_NAME, row, event="attempt_start")
        baseline.append_episode_journal(
            child / baseline.EPISODE_JOURNAL_NAME,
            row,
            event="terminal",
            status="completed",
        )
    children = [{"worker": "A", "physical_device": 0, "planned": assigned, "directory": child}]

    rows, metrics, failures = baseline._read_child_artifacts(
        children,
        config=config,
        expected_project_sha="project-sha",
        config_path=CONFIG_PATH,
    )

    assert len(rows) == len(assigned)
    assert metrics == {}
    assert any("worker metrics" in str(item["reason"]).lower() for item in failures)


def test_child_artifact_reader_rejects_metrics_not_derived_from_assigned_rows(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    assigned = [row for row in matrix if row["worker"] == "A"]
    episodes = [_episode(baseline, row, success=False) for row in assigned]
    worker_metrics = {"A": _genuine_worker_metrics(completed=8, wall=10.0, busy=4.0)}
    aggregate = baseline.aggregate_results(
        episodes,
        planned=assigned,
        total_wall_time_seconds=10.0,
        worker_metrics=worker_metrics,
        require_worker_metrics=True,
        worker_scope="A",
    )
    child = tmp_path / "child-A"
    manifest = baseline.build_run_manifest(
        config=config,
        config_path=CONFIG_PATH,
        run_directory=child,
        project_sha="project-sha",
        matrix=assigned,
        aggregate=aggregate,
        provenance={
            **_child_provenance(baseline, config, assigned),
            "worker_metrics": {"A": dict(aggregate["per_worker"]["A"])},
        },
        status="NEEDS_INVESTIGATION",
        scope={"type": "worker", "worker": "A", "physical_device": 0},
    )
    forged = dict(manifest)
    forged["aggregate"] = deepcopy(manifest["aggregate"])
    forged["aggregate"]["completed"] = 999
    forged["aggregate"]["successes"] = 999
    forged["aggregate"]["failed"] = 999
    forged["aggregate"]["per_worker"] = deepcopy(manifest["aggregate"]["per_worker"])
    forged_metric = forged["aggregate"]["per_worker"]["A"]
    forged_metric["completed"] = 999
    forged_metric["throughput_episodes_per_second"] = 999 / 10.0
    forged["provenance"] = deepcopy(manifest["provenance"])
    forged["provenance"]["worker_metrics"] = {"A": dict(forged_metric)}
    child.mkdir(parents=True, exist_ok=True)
    baseline.write_json_no_overwrite(child / "run_manifest.json", forged)
    for row in episodes:
        baseline.persist_episode_row(child, row)
        baseline.append_episode_journal(child / baseline.EPISODE_JOURNAL_NAME, row, event="attempt_start")
        baseline.append_episode_journal(
            child / baseline.EPISODE_JOURNAL_NAME,
            row,
            event="terminal",
            status="completed",
        )

    rows, metrics, failures = baseline._read_child_artifacts(
        [{"worker": "A", "physical_device": 0, "planned": assigned, "directory": child}],
        config=config,
        expected_project_sha="project-sha",
        config_path=CONFIG_PATH,
    )

    assert len(rows) == 8
    assert metrics == {}
    assert any("completed" in str(item["reason"]) or "derived" in str(item["reason"]) for item in failures)


def test_launcher_redirects_child_output_to_exclusive_parent_owned_files(tmp_path: Path) -> None:
    baseline = _module()
    calls: list[dict[str, Any]] = []

    class FakeProcess:
        _next_pid = 500

        def __init__(self, **_: Any) -> None:
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append(kwargs)
        return FakeProcess(**kwargs)

    children = baseline.launch_children(
        _config(),
        config_path=CONFIG_PATH,
        expected_project_sha="sha",
        run_directory=tmp_path / "run",
        popen=popen,
        project_python="/frozen/cpu/bin/python",
    )
    assert len(calls) == 2
    assert all(call["stdout"] is not baseline.subprocess.PIPE for call in calls)
    assert all(call["stderr"] is not baseline.subprocess.PIPE for call in calls)
    assert all((Path(item["directory"]) / "process_stdout.log").is_file() for item in children)
    assert all((Path(item["directory"]) / "process_stderr.log").is_file() for item in children)


def test_orphan_ipc_cleanup_is_strict_and_records_removed_paths(tmp_path: Path) -> None:
    baseline = _module()
    child = tmp_path / "child-A"
    orphan = child / "ipc" / "orphan-request"
    orphan.mkdir(parents=True)
    (orphan / "request.safetensors").write_bytes(b"orphan")
    evidence = baseline.cleanup_child_ipc(child)
    assert not orphan.exists()
    assert evidence["removed_count"] == 1
    assert evidence["strict_under_child"] is True


def test_orphan_ipc_cleanup_rejects_symlink_alias_without_deleting_target(tmp_path: Path) -> None:
    baseline = _module()
    child = tmp_path / "child-A"
    rows = child / "episode_rows"
    rows.mkdir(parents=True)
    keep = rows / "keep.json"
    keep.write_text('{"keep": true}\n', encoding="utf-8")
    alias = child / "ipc" / "alias"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(rows, target_is_directory=True)

    with pytest.raises(baseline.BaselineSchemaError, match="symlink"):
        baseline.cleanup_child_ipc(child)

    assert rows.is_dir()
    assert keep.is_file()


def test_real_run_rejects_drifted_pinned_provenance_before_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    config_path = tmp_path / "baseline_a.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace("python: 3.11.16", "python: 3.12.0", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(
        baseline,
        "validate_preflight_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("drift reached preflight")),
    )

    with pytest.raises(baseline.BaselineConfigError, match="provenance.python"):
        baseline.run(config_path, expected_project_sha="sha", output_root=tmp_path / "runs")

    assert not (tmp_path / "runs").exists()


def test_child_run_rejects_drifted_pinned_provenance_before_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    config_path = tmp_path / "baseline_a.yaml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace("python: 3.11.16", "python: 3.12.0", 1),
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")

    with pytest.raises(baseline.BaselineConfigError, match="provenance.python"):
        baseline._run_child(
            config_path,
            expected_project_sha="sha",
            worker="A",
            physical_device=0,
            child_directory=tmp_path / "child-A",
        )


def test_child_manifest_has_worker_scope_and_matching_counts() -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    assigned = [row for row in matrix if row["worker"] == "A"]
    aggregate = baseline.aggregate_results([], planned=assigned)
    manifest = baseline.build_run_manifest(
        config=config,
        config_path=CONFIG_PATH,
        run_directory=Path("/tmp/child-A"),
        project_sha="project-sha",
        matrix=assigned,
        aggregate=aggregate,
        provenance=_child_provenance(baseline, config, assigned),
        status="BLOCKED",
        scope={"type": "worker", "worker": "A", "physical_device": 0},
    )

    assert manifest["scope"]["planned_count"] == 8
    assert manifest["aggregate"]["planned"] == 8
    assert baseline.validate_manifest(manifest) is True

    bad = deepcopy(manifest)
    bad["aggregate"]["planned"] = 15
    with pytest.raises(baseline.BaselineSchemaError, match="planned"):
        baseline.validate_manifest(bad)

    bad_scope = deepcopy(manifest)
    bad_scope["scope"]["worker"] = "B"
    with pytest.raises(baseline.BaselineSchemaError, match="scope"):
        baseline.validate_manifest(bad_scope)


def test_parent_reads_and_validates_each_child_manifest_before_rows(tmp_path: Path) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    child_a = tmp_path / "child-A"
    assigned = [row for row in matrix if row["worker"] == "A"]
    _write_child_manifest(baseline, child_a, config, assigned, worker="A")
    (child_a / "episodes.jsonl").write_text("", encoding="utf-8")
    children = [{"worker": "A", "physical_device": 0, "planned": assigned, "directory": child_a}]
    with pytest.raises(baseline.BaselineSchemaError, match="non-normal"):
        baseline._read_child_episodes(children)

    manifest = json.loads((child_a / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["scope"]["physical_device"] = 1
    (child_a / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(baseline.BaselineSchemaError, match="physical_device"):
        baseline._read_child_episodes(children)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", 999),
        ("task_name", "tampered task"),
        ("init_state_id", 999),
        ("seed", 1),
        ("horizon", 1),
    ],
)
def test_load_child_manifest_rejects_any_tampered_assigned_matrix_row(
    tmp_path: Path, field: str, value: Any
) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    assigned = [row for row in matrix if row["worker"] == "A"]
    episodes = [_episode(baseline, row, success=False) for row in assigned]
    worker_metrics = {"A": _genuine_worker_metrics(completed=8, wall=10.0, busy=4.0)}
    aggregate = baseline.aggregate_results(
        episodes,
        planned=assigned,
        total_wall_time_seconds=10.0,
        worker_metrics=worker_metrics,
        require_worker_metrics=True,
        worker_scope="A",
    )
    child = tmp_path / "child-A"
    manifest = baseline.build_run_manifest(
        config=config,
        config_path=CONFIG_PATH,
        run_directory=child,
        project_sha="project-sha",
        matrix=assigned,
        aggregate=aggregate,
        provenance=_child_provenance(baseline, config, assigned),
        status="NEEDS_INVESTIGATION",
        scope={"type": "worker", "worker": "A", "physical_device": 0},
    )
    manifest["episode_matrix"]["rows"][0][field] = value
    child.mkdir(parents=True)
    baseline.write_json_no_overwrite(child / "run_manifest.json", manifest)

    with pytest.raises(baseline.BaselineSchemaError, match="rows"):
        baseline._load_child_manifest(
            {"worker": "A", "physical_device": 0, "planned": assigned, "directory": child}
        )


@pytest.mark.parametrize("manifest_name", ["run_manifest.json", "terminal_manifest.json"])
def test_load_child_manifest_rejects_symlinked_manifest(
    tmp_path: Path, manifest_name: str
) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    assigned = [row for row in matrix if row["worker"] == "A"]
    child = tmp_path / "child-A"
    child.mkdir()
    external = tmp_path / f"external-{manifest_name}"
    if manifest_name == "run_manifest.json":
        aggregate = baseline.aggregate_results([], planned=assigned)
        manifest = baseline.build_run_manifest(
            config=config,
            config_path=CONFIG_PATH,
            run_directory=child,
            project_sha="project-sha",
            matrix=assigned,
            aggregate=aggregate,
            provenance=_child_provenance(baseline, config, assigned),
            status="BLOCKED",
            scope={"type": "worker", "worker": "A", "physical_device": 0},
        )
        external.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        baseline.terminalize_child(
            external,
            worker="A",
            physical_device=0,
            planned=assigned,
            attempted=[],
            reason="worker exited",
            status="CRASHED",
        )
    (child / manifest_name).symlink_to(external)

    with pytest.raises(baseline.BaselineSchemaError, match="symlink"):
        baseline._load_child_manifest(
            {"worker": "A", "physical_device": 0, "planned": assigned, "directory": child}
        )


def test_parent_fail_closed_writes_terminal_evidence_for_malformed_child_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(baseline.load_config(CONFIG_PATH))
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"identity": "verified"})

    def fake_launch(*_: Any, **kwargs: Any) -> list[dict[str, Any]]:
        root = tmp_path / "child-A"
        root.mkdir()
        (root / "episodes.jsonl").write_text("{malformed\n", encoding="utf-8")
        return [{"worker": "A", "physical_device": 0, "planned": [row for row in matrix if row["worker"] == "A"], "directory": root, "returncode": 0}]

    monkeypatch.setattr(baseline, "launch_children", fake_launch)
    monkeypatch.setattr(baseline, "collect_children_fail_closed", lambda *_args, **_kwargs: [])
    assert baseline.run(CONFIG_PATH, expected_project_sha="sha", output_root=tmp_path / "runs") == 1
    run_directory = next((tmp_path / "runs").iterdir())
    terminal = json.loads((run_directory / "terminal_manifest.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "BLOCKED"
    assert "malformed" in terminal["reason"] or "JSON" in terminal["reason"]


@pytest.mark.parametrize("failure_point", ["aggregate", "final_write"])
def test_parent_fail_closed_writes_terminal_evidence_for_post_collection_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_point: str
) -> None:
    baseline = _module()
    matrix = baseline.build_episode_matrix(baseline.load_config(CONFIG_PATH))
    monkeypatch.setattr(baseline, "_git_sha", lambda *_: "sha")
    monkeypatch.setattr(baseline, "validate_preflight_contract", lambda *_, **__: {"identity": "verified"})

    def fake_launch(*_: Any, **kwargs: Any) -> list[dict[str, Any]]:
        children = []
        for worker, device in (("A", 0), ("B", 1)):
            directory = tmp_path / f"child-{worker}"
            directory.mkdir()
            children.append({"worker": worker, "physical_device": device, "planned": [row for row in matrix if row["worker"] == worker], "directory": directory, "returncode": 0})
        return children

    monkeypatch.setattr(baseline, "launch_children", fake_launch)
    monkeypatch.setattr(baseline, "collect_children_fail_closed", lambda *_args, **_kwargs: [])
    if failure_point == "aggregate":
        monkeypatch.setattr(baseline, "aggregate_results", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("aggregate exploded")))
    else:
        monkeypatch.setattr(baseline, "_write_run_artifacts", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("final write exploded")))

    assert baseline.run(CONFIG_PATH, expected_project_sha="sha", output_root=tmp_path / "runs") == 1
    run_directory = next((tmp_path / "runs").iterdir())
    terminal = json.loads((run_directory / "terminal_manifest.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "BLOCKED"
    assert failure_point.split("_")[0] in terminal["reason"]


def test_collect_children_does_not_capture_child_output_in_parent_memory() -> None:
    baseline = _module()

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 1234
            self.returncode = 0

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def communicate(self, *args: Any, **kwargs: Any) -> tuple[str, str]:
            raise AssertionError("direct-to-file child output must not be collected")

    process = FakeProcess()
    child = {"worker": "A", "physical_device": 0, "planned": [], "directory": Path("/tmp/A"), "process": process, "pgid": 1234}
    failures = baseline.collect_children_fail_closed([child], timeout_seconds=1.0)
    assert failures == []
    assert "stdout" not in child
    assert "stderr" not in child


def test_terminate_child_escalates_alive_outer_process_without_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _module()

    class StubbornProcess:
        pid = 4242

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.calls: list[tuple[str, float | None]] = []

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            self.calls.append(("wait", timeout))
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake-child", timeout)
            return self.returncode

        def terminate(self) -> None:
            self.calls.append(("terminate", None))

        def kill(self) -> None:
            self.calls.append(("kill", None))
            self.returncode = -9

    process = StubbornProcess()
    monkeypatch.setattr(baseline, "_process_group_descendants", lambda *_args, **_kwargs: [])
    child = {"process": process, "pgid": process.pid}

    baseline._terminate_child_process(child, "test timeout")

    assert process.calls == [
        ("wait", 1.0),
        ("terminate", None),
        ("wait", 1.0),
        ("kill", None),
        ("wait", 1.0),
    ]
    assert process.poll() is not None
    assert baseline._process_group_descendants(process.pid) == []
    assert child["group_cleanup_done"] is True


def test_write_run_artifacts_does_not_publish_pass_manifest_before_late_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _module()
    config = _config()
    matrix = baseline.build_episode_matrix(config)
    episodes = [_episode(baseline, row, success=True) for row in matrix]
    worker_metrics = {
        "A": _genuine_worker_metrics(completed=8, wall=2.0, busy=1.0),
        "B": _genuine_worker_metrics(completed=7, wall=3.0, busy=1.5),
    }
    aggregate = baseline.aggregate_results(
        episodes,
        planned=matrix,
        total_wall_time_seconds=5.0,
        worker_metrics=worker_metrics,
        require_worker_metrics=True,
    )
    provenance = {
        **config["provenance"],
        "git_sha": "project-sha",
        "checkpoint_revision": config["checkpoint"]["revision"],
        "trajectory_ids": [row["trajectory_id"] for row in matrix],
    }
    original_write = baseline._write_bytes_no_overwrite

    def fail_on_stderr(path: str | Path, data: bytes) -> Path:
        if Path(path).name == "stderr.log":
            raise RuntimeError("stderr write exploded")
        return original_write(path, data)

    monkeypatch.setattr(baseline, "_write_bytes_no_overwrite", fail_on_stderr)
    run_directory = tmp_path / "run"
    with pytest.raises(RuntimeError, match="stderr write exploded"):
        baseline._write_run_artifacts(
            run_directory,
            config=config,
            config_path=CONFIG_PATH,
            project_sha="project-sha",
            matrix=matrix,
            episodes=episodes,
            aggregate=aggregate,
            provenance=provenance,
            status="PASS",
            command=["test"],
        )

    assert not (run_directory / "run_manifest.json").exists()
