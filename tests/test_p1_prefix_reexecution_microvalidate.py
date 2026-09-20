from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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
    assert mv.TASK_SOURCE_CONTENT_PINS == {
        "suite": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "bddl_sha256": "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b",
        "init_state_sha256": "cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a",
    }


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

    def p2a(*args, **kwargs):
        calls.append("p2a")
        return {"status": "PASS"}

    def p2b(*args, **kwargs):
        calls.append("p2b")
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
        p2a_runner=p2a,
        p2b_runner=p2b,
        p1_runner=p1,
    )
    assert result["status"] == "BLOCKED"
    assert calls == ["pytest"]
    payload = output.read_text(encoding="utf-8")
    assert '"runtime_phase": "MICROVALIDATION"' in payload
    assert '"scientific_rollout": false' in payload
    assert '"pilot_rollout_count": 0' in payload


def _patch_fake_sequence_identity(monkeypatch, tmp_path: Path) -> None:
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


def test_run_sequence_stops_after_mv_p2a_failure(tmp_path: Path, monkeypatch) -> None:
    _patch_fake_sequence_identity(monkeypatch, tmp_path)
    calls = []

    def p2a(_preflight):
        calls.append("p2a")
        return {"status": "BLOCKED"}

    def p2b(_preflight):
        calls.append("p2b")
        return {"status": "PASS"}

    def p1(*args, **kwargs):
        calls.append("p1")
        return {"status": "PASS"}

    result = mv.run_sequence(
        baseline_config=tmp_path / "unused.yaml",
        output=tmp_path / "evidence.json",
        work_dir=tmp_path / "work",
        physical_device=1,
        pytest_runner=lambda: {"status": "PASS"},
        p2a_runner=p2a,
        p2b_runner=p2b,
        p1_runner=p1,
    )
    assert result["status"] == "BLOCKED"
    assert calls == ["p2a"]
    assert result["MV-P2"]["same_state_camera"]["status"] == "BLOCKED"
    assert result["MV-P2"]["switch_lifecycle"]["status"] == "NOT_RUN"
    assert result["MV-P1"]["status"] == "NOT_RUN"


def test_run_sequence_stops_after_mv_p2b_failure_before_mv_p1(tmp_path: Path, monkeypatch) -> None:
    _patch_fake_sequence_identity(monkeypatch, tmp_path)
    calls = []

    def p2a(_preflight):
        calls.append("p2a")
        return {"status": "PASS"}

    def p2b(_preflight):
        calls.append("p2b")
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
        p2a_runner=p2a,
        p2b_runner=p2b,
        p1_runner=p1,
    )
    assert result["status"] == "BLOCKED"
    assert calls == ["p2a", "p2b"]
    assert result["MV-P2"]["same_state_camera"]["status"] == "PASS"
    assert result["MV-P2"]["switch_lifecycle"]["status"] == "BLOCKED"
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
        p2a_runner=lambda _cfg: {"status": "PASS", "phase": "MV-P2-A"},
        p2b_runner=lambda _cfg: {"status": "PASS", "phase": "MV-P2-B"},
        p1_runner=lambda *_args, **_kwargs: {"status": "PASS", "phase": "MV-P1"},
    )
    assert result["status"] == "PASS"
    assert result["scientific_rollout"] is False
    assert result["pilot_rollout_count"] == 0
    assert result["paper_estimands_computed"] is False
    assert result["retry"] == 0 and result["replacement"] == 0



def _image_evidence(sha: str, *, shape=(360, 360, 3), dtype="uint8"):
    return {"shape": list(shape), "dtype": dtype, "sha256": sha}


def test_same_state_image_gate_blocks_when_clean_and_shifted_sha_match() -> None:
    clean = _image_evidence("a" * 64)
    result = mv._same_state_image_intervention_gate(clean, dict(clean), dict(clean))
    assert result["status"] == "BLOCKED"
    assert result["clean_vs_shifted_different"] is False
    assert result["reason"] == (
        "camera quaternion mutation did not alter the policy-visible agentview observation"
    )


def test_same_state_image_gate_blocks_when_restored_sha_differs() -> None:
    result = mv._same_state_image_intervention_gate(
        _image_evidence("a" * 64),
        _image_evidence("b" * 64),
        _image_evidence("c" * 64),
    )
    assert result["status"] == "BLOCKED"
    assert result["clean_vs_shifted_different"] is True
    assert result["clean_vs_restored_exact"] is False
    assert result["reason"] == (
        "restored clean camera did not reproduce the original policy-visible observation"
    )


def test_same_state_image_gate_passes_only_for_different_shifted_and_exact_restore() -> None:
    result = mv._same_state_image_intervention_gate(
        _image_evidence("a" * 64),
        _image_evidence("b" * 64),
        _image_evidence("a" * 64),
    )
    assert result == {
        "status": "PASS",
        "shape_equal": True,
        "dtype_equal": True,
        "clean_vs_shifted_different": True,
        "clean_vs_restored_exact": True,
        "reason": None,
    }


def test_same_state_image_gate_rejects_schema_shape_or_dtype_drift() -> None:
    shape_result = mv._same_state_image_intervention_gate(
        _image_evidence("a" * 64, shape=(2, 2, 3)),
        _image_evidence("b" * 64, shape=(2, 2, 3)),
        _image_evidence("a" * 64, shape=(2, 2, 3)),
    )
    assert shape_result["status"] == "BLOCKED"
    assert "360x360x3" in shape_result["reason"]

    dtype_result = mv._same_state_image_intervention_gate(
        _image_evidence("a" * 64, dtype="float32"),
        _image_evidence("b" * 64, dtype="float32"),
        _image_evidence("a" * 64, dtype="float32"),
    )
    assert dtype_result["status"] == "BLOCKED"
    assert "uint8" in dtype_result["reason"]


class _FakeP2BController:
    def __init__(self):
        self.sim = None

    def bind(self, sim):
        self.sim = sim
        return {}

    def apply(self, mode, *, request=None):
        return {
            "observation_index": request["observation_index"],
            "preceding_action_index": request["preceding_action_index"],
            "requested_camera_mode": mode,
            "actual_camera_parameters": {"fake": True},
        }


def _make_p2b_runtime(*, terminal_action_index=None):
    class FakeRawTask:
        def __init__(self):
            self.sim = SimpleNamespace(model=object())

        def _get_observations(self, force_update=False):
            index = env.step_calls
            return {
                "pixels": {
                    "image": np.full((2, 2, 3), index % 255, dtype=np.uint8),
                }
            }

    class FakeOffscreen:
        def __init__(self):
            self.env = FakeRawTask()

    class FakeEnv:
        __module__ = "lerobot.envs.libero"

        def __init__(self):
            self._env = FakeOffscreen()
            self.step_calls = 0
            self.reset_calls = 0

        def _ensure_env(self):
            return None

        def _format_raw_obs(self, raw):
            return raw

        def reset(self, seed=None, **kwargs):
            self.reset_calls += 1
            self.step_calls = 0
            return self._format_raw_obs(self._env.env._get_observations(force_update=True)), {}

        def step(self, action):
            action_index = self.step_calls
            self.step_calls += 1
            terminated = terminal_action_index == action_index
            obs = self._format_raw_obs(self._env.env._get_observations(force_update=True))
            return obs, 0.0, terminated, False, {}

    FakeEnv.__name__ = "LiberoEnv"
    env = FakeEnv()
    vector = SimpleNamespace(num_envs=1, envs=[env])
    runtime = {
        "env": vector,
        "envs": {"fake": vector},
        "close_envs": lambda _envs: None,
    }
    return env, runtime


def test_mv_p2b_executes_exactly_50_wrapper_steps_and_switches_obs50() -> None:
    env, runtime = _make_p2b_runtime()
    result = mv.run_mv_p2_b(
        {},
        runtime_builder=lambda *_args, **_kwargs: runtime,
        select_init_state=lambda _env, _index: {"selected_init_state_id": 0},
        dummy_action_factory=lambda: np.zeros(7, dtype=np.float32),
        controller_factory=_FakeP2BController,
    )
    assert result["status"] == "PASS"
    assert env.reset_calls == 1
    assert env.step_calls == 50
    assert result["wrapper_step_calls"] == 50
    assert result["policy_query_count"] == 0
    assert result["model_query_count"] == 0
    assert result["obs49"]["camera_evidence"]["observation_index"] == 49
    assert result["obs49"]["camera_evidence"]["requested_camera_mode"] == "clean"
    assert result["obs50"]["camera_evidence"] == {
        "observation_index": 50,
        "preceding_action_index": 49,
        "requested_camera_mode": "shifted",
        "actual_camera_parameters": {"fake": True},
    }


def test_mv_p2b_early_terminal_blocks_without_replacement_or_continuation() -> None:
    env, runtime = _make_p2b_runtime(terminal_action_index=12)
    result = mv.run_mv_p2_b(
        {},
        runtime_builder=lambda *_args, **_kwargs: runtime,
        select_init_state=lambda _env, _index: {"selected_init_state_id": 0},
        dummy_action_factory=lambda: np.zeros(7, dtype=np.float32),
        controller_factory=_FakeP2BController,
    )
    assert result["status"] == "BLOCKED"
    assert env.reset_calls == 1
    assert env.step_calls == 13
    assert result["wrapper_step_calls"] == 13
    assert result["terminal_before_switch"] == {
        "preceding_action_index": 12,
        "result_observation_index": 13,
        "terminated": True,
        "truncated": False,
    }
    assert result["policy_query_count"] == 0
    assert result["model_query_count"] == 0

def test_evidence_writer_is_no_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    mv._json_write_exclusive(target, {"status": "PASS"})
    with pytest.raises(FileExistsError):
        mv._json_write_exclusive(target, {"status": "BLOCKED"})



def _install_fake_libero_task_sources(
    monkeypatch,
    *,
    bddl_root: Path,
    init_root: Path,
) -> None:
    class FakeTaskSpec:
        problem_folder = "problem"
        bddl_file = "task.bddl"
        init_states_file = "task.init"
        language = "fake task description"
        name = "fake_task"

    class FakeSuite:
        def get_task(self, task_id):
            assert task_id == 0
            return FakeTaskSpec()

    libero_libero = ModuleType("libero.libero")
    libero_libero.benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {"libero_spatial": lambda: FakeSuite()}
    )
    libero_libero.get_libero_path = lambda key: (
        str(bddl_root) if key == "bddl_files" else str(init_root)
    )
    libero_pkg = ModuleType("libero")
    libero_pkg.libero = libero_libero
    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)


def _task_source_test_files(tmp_path: Path):
    runtime_root = tmp_path / "relocated-runtime"
    bddl = runtime_root / "bddl" / "problem" / "task.bddl"
    init_state = runtime_root / "init" / "problem" / "task.init"
    bddl.parent.mkdir(parents=True)
    init_state.parent.mkdir(parents=True)
    bddl.write_bytes(b"runtime-consumed-bddl")
    init_state.write_bytes(b"runtime-consumed-init")
    return runtime_root, bddl, init_state


def _task_source_pins(bddl: Path, init_state: Path):
    return {
        "suite": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
        "init_state_sha256": hashlib.sha256(init_state.read_bytes()).hexdigest(),
    }


def _task_source_task():
    return {
        "suite": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "seed": 2027,
    }


def test_content_pinned_task_source_passes_outside_repo_and_records_consumed_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import dcu_preflight

    runtime_root, bddl, init_state = _task_source_test_files(tmp_path)
    _install_fake_libero_task_sources(
        monkeypatch,
        bddl_root=runtime_root / "bddl",
        init_root=runtime_root / "init",
    )
    pins = _task_source_pins(bddl, init_state)
    evidence = dcu_preflight._task_source_evidence(
        object(),
        _task_source_task(),
        tmp_path / "unrelated-current-repo",
        content_pins=pins,
    )
    assert evidence["source_check"] == "runtime_consumed_content_sha256"
    assert evidence["bddl_path"] == str(bddl.resolve())
    assert evidence["init_state_path"] == str(init_state.resolve())
    assert evidence["bddl_sha256"] == pins["bddl_sha256"]
    assert evidence["init_state_sha256"] == pins["init_state_sha256"]
    assert evidence["content_pins"] == pins
    assert evidence["suite"] == "libero_spatial"
    assert evidence["task_id"] == 0
    assert evidence["init_state_id"] == 0


@pytest.mark.parametrize("which", ["bddl", "init"])
def test_content_pinned_task_source_wrong_hash_blocks(
    tmp_path: Path, monkeypatch, which: str
) -> None:
    from scripts import dcu_preflight

    runtime_root, bddl, init_state = _task_source_test_files(tmp_path)
    _install_fake_libero_task_sources(
        monkeypatch,
        bddl_root=runtime_root / "bddl",
        init_root=runtime_root / "init",
    )
    pins = _task_source_pins(bddl, init_state)
    pins["bddl_sha256" if which == "bddl" else "init_state_sha256"] = "0" * 64
    with pytest.raises(dcu_preflight.DCUPreflightError, match="sha256 mismatch"):
        dcu_preflight._task_source_evidence(
            object(),
            _task_source_task(),
            tmp_path / "repo",
            content_pins=pins,
        )


@pytest.mark.parametrize("kind", ["missing", "non_regular"])
def test_content_pinned_task_source_missing_or_non_regular_blocks(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    from scripts import dcu_preflight

    runtime_root, bddl, init_state = _task_source_test_files(tmp_path)
    pins = _task_source_pins(bddl, init_state)
    if kind == "missing":
        init_state.unlink()
    else:
        init_state.unlink()
        init_state.mkdir()
    _install_fake_libero_task_sources(
        monkeypatch,
        bddl_root=runtime_root / "bddl",
        init_root=runtime_root / "init",
    )
    with pytest.raises(dcu_preflight.DCUPreflightError, match="missing|regular file"):
        dcu_preflight._task_source_evidence(
            object(),
            _task_source_task(),
            tmp_path / "repo",
            content_pins=pins,
        )


def test_content_pinned_task_source_unreadable_resolved_file_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import dcu_preflight

    runtime_root, bddl, init_state = _task_source_test_files(tmp_path)
    _install_fake_libero_task_sources(
        monkeypatch,
        bddl_root=runtime_root / "bddl",
        init_root=runtime_root / "init",
    )
    pins = _task_source_pins(bddl, init_state)
    real_access = dcu_preflight.os.access

    def fake_access(path, mode):
        if Path(path) == init_state.resolve():
            return False
        return real_access(path, mode)

    monkeypatch.setattr(dcu_preflight.os, "access", fake_access)
    with pytest.raises(dcu_preflight.DCUPreflightError, match="not a readable regular file"):
        dcu_preflight._task_source_evidence(
            object(),
            _task_source_task(),
            tmp_path / "repo",
            content_pins=pins,
        )


def test_content_pinned_task_source_hashes_runtime_file_not_reference_copy(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import dcu_preflight

    runtime_root, bddl, init_state = _task_source_test_files(tmp_path)
    reference = tmp_path / "reference-copy.bddl"
    reference.write_bytes(b"different-reference-copy")
    _install_fake_libero_task_sources(
        monkeypatch,
        bddl_root=runtime_root / "bddl",
        init_root=runtime_root / "init",
    )
    pins = _task_source_pins(bddl, init_state)
    evidence = dcu_preflight._task_source_evidence(
        object(), _task_source_task(), tmp_path / "repo", content_pins=pins
    )
    assert evidence["bddl_sha256"] == hashlib.sha256(bddl.read_bytes()).hexdigest()
    assert evidence["bddl_sha256"] != hashlib.sha256(reference.read_bytes()).hexdigest()
    assert evidence["bddl_path"] == str(bddl.resolve())


def test_content_pinned_task_source_rejects_frozen_identity_drift(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import dcu_preflight

    runtime_root, bddl, init_state = _task_source_test_files(tmp_path)
    _install_fake_libero_task_sources(
        monkeypatch,
        bddl_root=runtime_root / "bddl",
        init_root=runtime_root / "init",
    )
    pins = _task_source_pins(bddl, init_state)
    pins["task_id"] = 4
    with pytest.raises(dcu_preflight.DCUPreflightError, match="task_source_content_pins.task_id"):
        dcu_preflight._task_source_evidence(
            object(), _task_source_task(), tmp_path / "repo", content_pins=pins
        )


def test_historical_task_source_mode_keeps_repo_relative_guard(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import dcu_preflight

    runtime_root, _, _ = _task_source_test_files(tmp_path)
    _install_fake_libero_task_sources(
        monkeypatch,
        bddl_root=runtime_root / "bddl",
        init_root=runtime_root / "init",
    )
    with pytest.raises(
        dcu_preflight.DCUPreflightError,
        match="escaped pinned hf-libero tree",
    ):
        dcu_preflight._task_source_evidence(
            object(),
            _task_source_task(),
            tmp_path / "repo",
            content_pins=None,
        )


def test_content_pinned_task_source_seam_does_not_copy_or_rewrite_paths() -> None:
    import inspect
    from scripts import dcu_preflight

    source = inspect.getsource(dcu_preflight._task_source_evidence)
    assert "get_libero_path" in source
    assert "resolve(strict=True)" in source
    assert "_sha256_file" in source
    for forbidden in ("copyfile", "copy2", "write_bytes", "write_text", "symlink_to", "os.symlink"):
        assert forbidden not in source


def test_p1_harness_opt_in_content_pins_reach_all_runtime_builders() -> None:
    import inspect

    expected = "task_source_content_pins=TASK_SOURCE_CONTENT_PINS"
    assert expected in inspect.getsource(mv.run_mv_p2_a)
    assert expected in inspect.getsource(mv.run_mv_p2_b)
    assert expected in inspect.getsource(mv.run_mv_p1)

def test_harness_source_has_no_pilot_estimands_or_env_step_call() -> None:
    import inspect

    source = inspect.getsource(mv)
    assert "G_" + "C" not in source
    assert "G_" + "S" not in source
    assert "paper_estimands_computed" in source
    run_p2a_source = inspect.getsource(mv.run_mv_p2_a)
    assert ".step(" not in run_p2a_source
    run_p2b_source = inspect.getsource(mv.run_mv_p2_b)
    assert "wrapped.step(" in run_p2b_source
    assert "range(50)" in run_p2b_source
    assert "wrapped.action_index =" not in run_p2b_source
    run_p1_source = inspect.getsource(mv.run_mv_p1)
    assert ".step(" not in run_p1_source
    assert "FeatureOnlyRemotePolicy" in run_p1_source
    assert "remote.select_action(features, noise=noise)" in run_p1_source
    assert 'ipc = work_dir / "ipc"' in run_p1_source
