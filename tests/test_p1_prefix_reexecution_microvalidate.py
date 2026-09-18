from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import p1_prefix_reexecution_microvalidate as mv


class _ExpectedLiberoEnv:
    __module__ = "lerobot.envs.libero"
    __name__ = "LiberoEnv"


# type(...).__name__ ignores an assigned __name__ on a differently named class,
# so construct the exact identity dynamically.
FakeLiberoEnv = type(
    "LiberoEnv",
    (),
    {
        "__module__": "lerobot.envs.libero",
        "reset": lambda self, *args, **kwargs: ({}, {}),
        "step": lambda self, action: ({}, 0.0, False, False, {}),
        "_ensure_env": lambda self: None,
        "_format_raw_obs": lambda self, raw: raw,
    },
)


def test_microvalidation_metadata_is_non_scientific() -> None:
    assert mv.PHASE == "MICROVALIDATION"
    assert mv.TASK_ID == 0
    assert mv.INIT_STATE_ID == 0
    assert mv.FROZEN_ENVIRONMENT_SEED == 2027
    assert mv.ROOT_ID == "libero_spatial-task000-init000-seed2027"


def test_locate_single_libero_subenv_is_fail_closed() -> None:
    subenv = FakeLiberoEnv()
    assert mv.locate_single_libero_subenv(SimpleNamespace(num_envs=1, envs=[subenv])) is subenv
    with pytest.raises(mv.MicrovalidationError, match="num_envs=1"):
        mv.locate_single_libero_subenv(SimpleNamespace(num_envs=2, envs=[subenv, subenv]))
    with pytest.raises(mv.MicrovalidationError, match="env.envs"):
        mv.locate_single_libero_subenv(SimpleNamespace(num_envs=1, envs=[]))
    wrong = type("OtherEnv", (), {"__module__": "lerobot.envs.libero"})()
    with pytest.raises(mv.MicrovalidationError, match="unexpected single sub-env"):
        mv.locate_single_libero_subenv(SimpleNamespace(num_envs=1, envs=[wrong]))


def test_physics_snapshot_detects_exact_changes_and_time_changes() -> None:
    sim = SimpleNamespace(
        data=SimpleNamespace(
            qpos=np.asarray([1.0, 2.0]),
            qvel=np.asarray([3.0, 4.0]),
            ctrl=np.asarray([5.0]),
            time=7.0,
        )
    )
    first_public, first_arrays = mv._physics_snapshot(sim)
    second_public, second_arrays = mv._physics_snapshot(sim)
    assert mv._physics_equal(first_public, first_arrays, second_public, second_arrays)["all"] is True

    sim.data.qvel[0] += 1.0
    changed_public, changed_arrays = mv._physics_snapshot(sim)
    comparison = mv._physics_equal(first_public, first_arrays, changed_public, changed_arrays)
    assert comparison["qvel"] is False
    assert comparison["all"] is False


def test_run_sequence_stops_before_runtime_when_focused_pytest_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mv, "_git_head", lambda: "a" * 40)
    calls = []

    def fail_pytest():
        calls.append("pytest")
        return {"status": "BLOCKED", "returncode": 1}

    def p2(*args, **kwargs):
        calls.append("p2")
        return {"status": "PASS"}

    def p1(*args, **kwargs):
        calls.append("p1")
        return {"status": "PASS"}

    output = tmp_path / "evidence.json"
    result = mv.run_sequence(
        baseline_config=tmp_path / "unused.yaml",
        output=output,
        work_dir=tmp_path / "work",
        physical_device=1,
        pytest_runner=fail_pytest,
        p2_runner=p2,
        p1_runner=p1,
    )
    assert result["status"] == "BLOCKED"
    assert calls == ["pytest"]
    payload = output.read_text(encoding="utf-8")
    assert '"runtime_phase": "MICROVALIDATION"' in payload
    assert '"scientific_rollout": false' in payload
    assert '"pilot_rollout_count": 0' in payload


def test_run_sequence_stops_after_mv_p2_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mv, "_git_head", lambda: "b" * 40)
    monkeypatch.setattr(
        mv,
        "_load_configs",
        lambda _path: (
            {"runtime": {}},
            {"runtime": {"compare_physical_device": 1}},
            {"preflight_config": str(tmp_path / "preflight.yaml")},
        ),
    )
    monkeypatch.setattr(mv, "_install_cpu_runtime_environment", lambda *_args: {"CPU": "fake"})
    calls = []

    def p2(_preflight):
        calls.append("p2")
        return {"status": "BLOCKED"}

    def p1(*args, **kwargs):
        calls.append("p1")
        return {"status": "PASS"}

    result = mv.run_sequence(
        baseline_config=tmp_path / "unused.yaml",
        output=tmp_path / "evidence.json",
        work_dir=tmp_path / "work",
        physical_device=1,
        pytest_runner=lambda: {"status": "PASS"},
        p2_runner=p2,
        p1_runner=p1,
    )
    assert result["status"] == "BLOCKED"
    assert calls == ["p2"]
    assert result["MV-P2"]["status"] == "BLOCKED"
    assert result["MV-P1"]["status"] == "NOT_RUN"


def test_run_sequence_success_writes_compact_non_pilot_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mv, "_git_head", lambda: "c" * 40)
    preflight_path = tmp_path / "preflight.yaml"
    preflight_path.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(
        mv,
        "_load_configs",
        lambda _path: (
            {"runtime": {}},
            {"runtime": {"compare_physical_device": 1}},
            {"preflight_config": str(preflight_path)},
        ),
    )
    monkeypatch.setattr(mv, "_install_cpu_runtime_environment", lambda *_args: {"CPU": "fake"})

    result = mv.run_sequence(
        baseline_config=tmp_path / "baseline.yaml",
        output=tmp_path / "evidence.json",
        work_dir=tmp_path / "work",
        physical_device=1,
        pytest_runner=lambda: {"status": "PASS", "returncode": 0},
        p2_runner=lambda _cfg: {"status": "PASS", "phase": "MV-P2"},
        p1_runner=lambda *_args, **_kwargs: {"status": "PASS", "phase": "MV-P1"},
    )
    assert result["status"] == "PASS"
    assert result["scientific_rollout"] is False
    assert result["pilot_rollout_count"] == 0
    assert result["paper_estimands_computed"] is False
    assert result["retry"] == 0 and result["replacement"] == 0


def test_evidence_writer_is_no_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    mv._json_write_exclusive(target, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        mv._json_write_exclusive(target, {"status": "BLOCKED"})


def test_harness_source_has_no_pilot_estimands_or_env_step_call() -> None:
    import inspect

    source = inspect.getsource(mv)
    assert "G_" + "C" not in source
    assert "G_" + "S" not in source
    assert "paper_estimands_computed" in source
    run_p2_source = inspect.getsource(mv.run_mv_p2)
    assert ".step(" not in run_p2_source
    run_p1_source = inspect.getsource(mv.run_mv_p1)
    assert ".step(" not in run_p1_source
    assert "FeatureOnlyRemotePolicy" in run_p1_source
    assert "remote.select_action(features, noise=noise)" in run_p1_source
    assert 'ipc = work_dir / "ipc"' in run_p1_source
