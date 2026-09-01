"""Fake-only contract tests for the frozen M1 exact-state replay harness.

The real LIBERO/MuJoCo runtime is deliberately not imported by this module's
unit tests.  The integration seam is exercised only with small fake objects;
the real 110-step experiment remains an explicit opt-in operation owned by the
main agent after implementation review.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "m1" / "state_replay.yaml"


def _module():
    """Defer import so the first RED run points at missing M1 symbols."""

    import scripts.m1_state_replay as replay

    return replay


def test_action_tape_is_single_pcg64_float32_110_by_7_and_hash_stable() -> None:
    replay = _module()

    tape = replay.generate_action_tape()

    assert tape.shape == (110, 7)
    assert tape.dtype == np.float32
    np.testing.assert_array_equal(tape[10, 3:6], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(tape[50, 3:6], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(tape[100, 3:6], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(tape[:10, 6], np.full(10, -1, dtype=np.float32))
    np.testing.assert_array_equal(tape[10:20, 6], np.full(10, 1, dtype=np.float32))
    assert replay.action_tape_sha256(tape) == replay.action_tape_sha256(
        replay.generate_action_tape()
    )
    assert replay.action_tape_sha256(tape) == hashlib.sha256(tape.tobytes()).hexdigest()
    assert np.all(tape[:, :6] <= np.float32(0.2))
    assert np.all(tape[:, :6] >= np.float32(-0.2))


def test_capture_boundaries_are_post_step_and_reference_windows_are_exact() -> None:
    replay = _module()

    windows = replay.build_capture_windows(
        capture_steps=(10, 50, 100), future_steps=10, total_steps=110
    )

    assert [window.capture_step for window in windows] == [10, 50, 100]
    assert [window.state_steps for window in windows] == [
        tuple(range(10, 21)),
        tuple(range(50, 61)),
        tuple(range(100, 111)),
    ]
    assert [window.action_indices for window in windows] == [
        tuple(range(10, 20)),
        tuple(range(50, 60)),
        tuple(range(100, 110)),
    ]
    assert replay.capture_phase(10, 10) == "after_env_step_before_next_action"
    with pytest.raises(replay.M1SchemaError, match="capture step"):
        replay.build_capture_windows((0, 50, 100), future_steps=10, total_steps=110)


def test_record_bundle_round_trips_losslessly_and_rejects_pickle_nonfinite_or_hash_mismatch(
    tmp_path: Path,
) -> None:
    replay = _module()
    payload = np.asarray([1.0, -2.5, 3.25], dtype=np.float64)
    record = replay.MatchedStateRecord(
        metadata={"capture_step": 10, "kind": "integration"}, payload=payload
    )

    replay.save_record_bundle(record, tmp_path, "capture_10")
    restored = replay.load_record_bundle(tmp_path, "capture_10")
    assert restored.metadata == record.metadata
    assert restored.payload.dtype == np.float64
    assert restored.payload.tobytes() == payload.tobytes()
    assert restored.payload.flags.writeable is False

    bad = tmp_path / "bad.npy"
    np.save(bad, np.asarray([{"not": "pickle"}], dtype=object), allow_pickle=True)
    with pytest.raises(replay.M1SchemaError, match="pickle|object"):
        replay.safe_load_npy(bad)

    with pytest.raises(replay.M1SchemaError, match="finite"):
        replay.MatchedStateRecord(
            metadata={"capture_step": 10}, payload=np.asarray([np.nan], dtype=np.float64)
        )

    metadata_path = tmp_path / "capture_10.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["payload_sha256"] = "0" * 64
    metadata_path.write_text(replay.canonical_json(metadata) + "\n", encoding="utf-8")
    with pytest.raises(replay.M1SchemaError, match="hash"):
        replay.load_record_bundle(tmp_path, "capture_10")


def test_integration_readback_requires_exact_equality_before_forward() -> None:
    replay = _module()

    class FakeData:
        def __init__(self) -> None:
            self.state = np.arange(5, dtype=np.float64)

    class FakeModel:
        nq = 2
        nv = 2
        nu = 1
        na = 0
        nuserdata = 0
        nmocap = 0
        neq = 0
        nplugin = 0

    class FakeMujoco:
        mjSTATE_INTEGRATION = 1

        def __init__(self, data: FakeData) -> None:
            self.data = data
            self.events: list[str] = []

        def mj_setState(self, model: FakeModel, data: FakeData, payload: Any, spec: int) -> None:
            self.events.append("set")
            data.state = np.asarray(payload, dtype=np.float64).copy()

        def mj_getState(self, model: FakeModel, data: FakeData, out: np.ndarray, spec: int) -> None:
            self.events.append("get")
            out[...] = data.state

        def mj_forward(self, model: FakeModel, data: FakeData) -> None:
            self.events.append("forward")

    data = FakeData()
    mujoco = FakeMujoco(data)
    expected = np.arange(5, dtype=np.float64) + 10
    replay.set_integration_state_exact(
        mujoco, FakeModel(), data, expected, require_readback=True
    )
    assert mujoco.events == ["set", "get"]

    class BadMujoco(FakeMujoco):
        def mj_getState(self, model: FakeModel, data: FakeData, out: np.ndarray, spec: int) -> None:
            super().mj_getState(model, data, out, spec)
            out[0] += 1e-13

    bad = BadMujoco(data)
    with pytest.raises(replay.M1RuntimeError, match="readback"):
        replay.set_integration_state_exact(
            bad, FakeModel(), data, expected, require_readback=True
        )
    assert bad.events == ["set", "get"]


def test_contact_pairs_are_canonical_before_exact_comparison() -> None:
    replay = _module()

    left = [("geom_b", "geom_a", -0.1), ("geom_c", "geom_a", -0.2)]
    right = [("geom_a", "geom_c", -0.2), ("geom_a", "geom_b", -0.1)]

    assert replay.canonicalize_contact_pairs(left) == replay.canonicalize_contact_pairs(right)
    assert replay.compare_contact_pairs(left, right).passed
    changed_distance = [("geom_a", "geom_b", -0.1000000000011), ("geom_a", "geom_c", -0.2)]
    assert not replay.compare_contact_pairs(left, changed_distance).passed


def test_comparison_classes_keep_exact_floating_and_diagnostic_results_separate() -> None:
    replay = _module()

    exact = replay.compare_exact({"counter": 2}, {"counter": 2}, path="counter")
    floating = replay.compare_floating(
        np.asarray([1.0]), np.asarray([1.0 + 5e-13]), path="qpos"
    )
    diagnostic = replay.compare_diagnostic({"rgb_sha256": "a"}, {"rgb_sha256": "b"})

    assert exact.comparison_class == replay.ComparisonClass.EXACT
    assert exact.passed
    assert floating.comparison_class == replay.ComparisonClass.FLOATING_PHYSICAL
    assert floating.passed
    assert diagnostic.comparison_class == replay.ComparisonClass.DERIVED_DIAGNOSTIC
    assert not diagnostic.passed
    assert diagnostic.gates_pass is True


def test_float_gate_uses_zero_rtol_and_one_e_minus_twelve_atol() -> None:
    replay = _module()

    assert replay.FLOATING_RTOL == 0.0
    assert replay.FLOATING_ATOL == 1.0e-12
    assert replay.compare_floating([1.0], [1.0 + 5.0e-13]).passed
    assert not replay.compare_floating([1.0], [1.0 + 1.1e-12]).passed
    with pytest.raises(replay.M1ConfigError, match="rtol|atol"):
        replay.compare_floating([1.0], [1.0], rtol=1.0e-9, atol=1.0e-12)


def test_unknown_mutable_state_fails_closed() -> None:
    replay = _module()

    classification = deepcopy(replay.DEFAULT_STATE_CLASSIFICATION)
    with pytest.raises(replay.UnknownMutableStateError, match="unknown|unclassified"):
        replay.validate_state_classification(
            classification, reachable_mutable_fields=("env.new_unknown_counter",)
        )


def test_runtime_audit_rejects_unclassified_reachable_field() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    with pytest.raises(replay.UnknownMutableStateError, match="unknown|unclassified"):
        adapter.audit_runtime_state(("env.unclassified_mutable",))


class _FakeData:
    def __init__(self) -> None:
        self.integration_state = np.zeros(4, dtype=np.float64)
        self.time = 0.0
        self.qpos = np.zeros(2, dtype=np.float64)
        self.qvel = np.zeros(2, dtype=np.float64)
        self.xpos = np.zeros((2, 3), dtype=np.float64)
        self.xquat = np.zeros((2, 4), dtype=np.float64)
        self.contact_pairs: list[tuple[str, str, float]] = []


class _FakeModel:
    nq = 2
    nv = 2
    na = 0
    nu = 1
    nbody = 2
    ngeom = 2
    nsite = 1
    neq = 0
    nmocap = 0
    nuserdata = 0
    npluginstate = 0
    nhistory = 0
    joint_names = ["joint0", "joint1"]
    body_names = ["world", "fixture"]
    body_pos = np.zeros((2, 3), dtype=np.float64)
    body_quat = np.zeros((2, 4), dtype=np.float64)
    site_rgba = np.zeros((1, 4), dtype=np.float32)

    def body_name2id(self, name: str) -> int:
        return {"world": 0, "fixture": 1}[name]


class _FakeMujoco:
    mjSTATE_INTEGRATION = 7

    def __init__(self, data: _FakeData, events: list[str]) -> None:
        self.data = data
        _FAKE_EVENT_RECORDERS[id(self)] = events

    def mj_stateSize(self, model: _FakeModel, spec: int) -> int:
        return int(self.data.integration_state.size)

    def mj_setState(self, model: _FakeModel, data: _FakeData, payload: np.ndarray, spec: int) -> None:
        _fake_record(self, "set_state")
        data.integration_state = np.asarray(payload, dtype=np.float64).copy()

    def mj_getState(self, model: _FakeModel, data: _FakeData, out: np.ndarray, spec: int) -> None:
        _fake_record(self, "get_state")
        out[...] = data.integration_state

    def mj_forward(self, model: _FakeModel, data: _FakeData) -> None:
        _fake_record(self, "forward")
        data.qpos = data.integration_state[:2].copy()
        data.qvel = data.integration_state[2:].copy()


_FAKE_EVENT_RECORDERS: dict[int, list[str]] = {}


def _fake_record(owner: Any, event: str) -> None:
    _FAKE_EVENT_RECORDERS[id(owner)].append(event)


class _FakeController:
    def __init__(self, events: list[str]) -> None:
        _FAKE_EVENT_RECORDERS[id(self)] = events
        self.initial_joint = np.asarray([0.1, 0.2], dtype=np.float64)
        self.goal_pos = np.asarray([0.3, 0.4, 0.5], dtype=np.float64)
        self.goal_ori = np.eye(3, dtype=np.float64)
        self.new_update = True
        self.kp = np.ones(6, dtype=np.float64)
        self.kd = np.ones(6, dtype=np.float64) * 2.0
        self.impedance_mode = "fixed"
        self.use_delta = True
        self.use_ori = True
        self.eef_name = "grip_site"
        self.control_freq = 20
        self.ee_pos = self.goal_pos.copy()
        self.ee_ori_mat = self.goal_ori.copy()
        self.joint_pos = self.initial_joint.copy()
        self.joint_vel = np.zeros(2, dtype=np.float64)

    def update(self, force: bool = False) -> None:
        assert force is True
        _fake_record(self, "controller_update")
        self.new_update = False


class _FakeGripper:
    def __init__(self) -> None:
        self.current_action = np.asarray([0.25, -0.25], dtype=np.float64)


class _FakeEnv:
    """Small official-Libero-shaped env for adapter-only tests."""

    def __init__(self, events: list[str], *, terminate_at: int | None = None) -> None:
        _FAKE_EVENT_RECORDERS[id(self)] = events
        self.sim = SimpleNamespace(model=_FakeModel(), data=_FakeData())
        self.controller = _FakeController(events)
        self.gripper = _FakeGripper()
        self.robots = [SimpleNamespace(controller=self.controller, gripper=self.gripper)]
        self.fixtures_dict = {"fixture": SimpleNamespace(root_body="fixture")}
        self.objects_dict: dict[str, Any] = {}
        self.observables: dict[str, Any] = {}
        self.task = "task-zero"
        self.task_description = "instruction-zero"
        self.task_suite_name = "libero_spatial"
        self.init_state_id = 0
        self.timestep = 0
        self.cur_time = 0.0
        self.done = False
        self.reward_value = 0.0
        self.terminate_at = terminate_at
        self.vis_site_names = {"flat_stove_1": False}
        self.reset_calls = 0
        self.set_init_state_calls = 0
        self.settle_calls = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        self.reset_calls += 1
        _fake_record(self, "construction_reset")
        self.timestep = 0
        self.cur_time = 0.0
        self.done = False
        self.sim.data.integration_state = np.zeros(4, dtype=np.float64)
        return {"step": 0}, {}

    def set_init_state(self, value: Any) -> None:
        self.set_init_state_calls += 1
        raise AssertionError("public set_init_state is forbidden during restore")

    def settle(self) -> None:
        self.settle_calls += 1
        raise AssertionError("dummy settling is forbidden during restore")

    def step(self, action: np.ndarray) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        _fake_record(self, "step_begin")
        self.timestep += 1
        self.cur_time += 0.1
        self.sim.data.integration_state = self.sim.data.integration_state + np.asarray(
            [action[0], action[1], action[2], action[3]], dtype=np.float64
        )
        self.sim.data.qpos = self.sim.data.integration_state[:2].copy()
        self.sim.data.qvel = self.sim.data.integration_state[2:].copy()
        self.sim.data.time = self.cur_time
        self.sim.model.site_rgba[0, 0] = 0.5
        _fake_record(self, "step_return")
        terminated = self.terminate_at is not None and self.timestep >= self.terminate_at
        return {"step": self.timestep}, 0.25, terminated, False, {"is_success": False}

    def check_success(self) -> bool:
        return False

    def _post_process(self) -> None:
        _fake_record(self, "post_process")
        self.sim.model.site_rgba[0, 0] = 0.5

    def _update_observables(self, force: bool = False) -> None:
        assert force is True
        _fake_record(self, "refresh")


class _FakeVector:
    def __init__(self, child: _FakeEnv) -> None:
        self.envs = [child]


def _fake_runtime_builder_factory(
    events: list[str], *, terminate_at: int | None = None
) -> tuple[Any, list[_FakeEnv]]:
    created: list[_FakeEnv] = []

    def builder(config: dict[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events, terminate_at=terminate_at)
        created.append(child)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    return builder, created


class _OuterLiberoLike:
    """Official-shaped outer Libero wrapper whose task lives outside inner."""

    def __init__(self, inner: _FakeEnv) -> None:
        self._env = inner
        self.task = "outer-task-zero"
        self.task_description = "outer-instruction-zero"
        self.task_suite_name = "libero_spatial"
        self.init_state_id = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, int], dict[str, Any]]:
        result = self._env.reset(seed=seed)
        self.init_state_id += 1
        return result

    def step(self, action: np.ndarray) -> Any:
        return self._env.step(action)

    def check_success(self) -> bool:
        return self._env.check_success()

    def close(self) -> None:
        return None


class _InnerWithoutTask(_FakeEnv):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        del self.task
        self.objects_dict = {
            "bowl": SimpleNamespace(root_body="fixture", joints=["joint"])
        }


class _SourceDefinedInner(_FakeEnv):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.source_defined_inner_cache = []


class _SourceDefinedController(_FakeController):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.source_defined_controller_cache = []


class _SourceDefinedGripper(_FakeGripper):
    def __init__(self) -> None:
        super().__init__()
        self.source_defined_gripper_cache = []


class _SourceDefinedRobot:
    def __init__(self, controller: Any, gripper: Any) -> None:
        self.controller = controller
        self.gripper = gripper
        self.source_defined_robot_cache = []


class _CollisionConfiguredNode:
    """Source-shaped fixture/object with robosuite's collision-copy flag."""

    def __init__(self, duplicate_collision_geoms: Any = True) -> None:
        self.root_body = "fixture"
        self.joints: list[str] = ["joint"]
        self.duplicate_collision_geoms = duplicate_collision_geoms


def _fake_config(tmp_path: Path) -> dict[str, Any]:
    replay = _module()
    return {
        "schema_version": 1,
        "name": "state_replay",
        "task": {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027},
        "action_tape": {
            "seed": 12027,
            "shape": [110, 7],
            "dtype": "float32",
            "arm_low": -0.2,
            "arm_high": 0.2,
            "gripper_block_steps": 10,
            "zero_orientation_steps": [11, 51, 101],
            "sha256": replay.action_tape_sha256(replay.generate_action_tape()),
        },
        "capture_steps": [10, 50, 100],
        "future_steps": 10,
        "restores_per_capture": 3,
        "tolerances": {"rtol": 0.0, "atol": 1.0e-12},
        "paths": {
            "action_tape": str(tmp_path / "action_tape.npy"),
            "output_root": str(tmp_path / "runs"),
            "cpu_python": "/public/home/xuyinghao/tmp/shiftvla-libero/bin/python",
            "config": str(CONFIG_PATH),
        },
        "runtime": {"include_policy": False, "offline": True},
        "pins": {},
    }


def test_capture_occurs_only_after_returned_real_step() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    assert adapter.step_count == 0
    adapter.step(np.zeros(7, dtype=np.float32))
    assert events[-1] == "step_return"
    capture = adapter.capture(source_step=1)
    assert capture.identity["source_step"] == 1
    assert adapter.step_count == 1


def test_restore_order_is_model_state_setstate_readback_forward_controller_gripper_counters_refresh() -> None:
    replay = _module()
    events: list[str] = []
    source_builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=source_builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    state = source.capture(source_step=1)

    events.clear()
    fresh_builder, _ = _fake_runtime_builder_factory(events)
    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=fresh_builder, tape_hash="tape")
    fresh.restore(state)

    assert events.index("set_state") < events.index("get_state") < events.index("forward")
    assert events.index("forward") < events.index("controller_update") < events.index("post_process")
    assert events.index("post_process") < events.index("refresh")


def test_restore_never_calls_reset_set_init_state_or_settle() -> None:
    replay = _module()
    events: list[str] = []
    builder, created = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    state = source.capture(source_step=1)
    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    reset_count = created[-1].reset_calls
    fresh.restore(state)
    assert created[-1].reset_calls == reset_count == 1
    assert created[-1].set_init_state_calls == 0
    assert created[-1].settle_calls == 0


def test_model_layout_task_instruction_and_controller_mismatch_fail_closed() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    state = source.capture(source_step=1)
    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    fresh.env.task_description = "different instruction"
    with pytest.raises(replay.M1RuntimeError, match="identity/layout"):
        fresh.restore(state)
    assert "set_state" not in events

    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    fresh.model.nq = 99
    with pytest.raises(replay.M1RuntimeError, match="identity/layout"):
        fresh.restore(state)

    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    fresh.inner.controller.kp[0] += 1.0
    with pytest.raises(replay.M1RuntimeError, match="identity/layout"):
        fresh.restore(state)


def test_fixture_model_state_is_restored_and_verified() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.model.body_pos[1] = [1.0, 2.0, 3.0]
    source.model.body_quat[1] = [1.0, 0.0, 0.0, 0.0]
    state = source.capture(source_step=0)
    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    fresh.model.body_pos[1] = [9.0, 9.0, 9.0]
    fresh.restore(state)
    np.testing.assert_array_equal(fresh.model.body_pos[1], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(fresh.model.body_quat[1], [1.0, 0.0, 0.0, 0.0])


def test_post_process_is_allowed_only_if_runtime_snapshot_changes_visual_allowlist() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    state = source.capture(source_step=1)
    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    fresh.restore(state)

    def bad_post_process() -> None:
        fresh.sim.data.integration_state[0] += 1.0

    fresh.env._post_process = bad_post_process
    with pytest.raises(replay.M1RuntimeError, match="non-visual"):
        fresh.restore(state)


def test_source_termination_before_100_fails_without_regeneration() -> None:
    replay = _module()
    events: list[str] = []
    builder, created = _fake_runtime_builder_factory(events, terminate_at=99)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    with pytest.raises(replay.EarlyTerminationError, match="terminated"):
        replay.run_source_trajectory(
            adapter,
            replay.generate_action_tape(),
            replay.build_capture_windows(),
        )
    assert len(created) == 1


def test_orchestration_uses_one_source_nine_unique_restores_and_terminal_provenance(
    tmp_path: Path,
) -> None:
    replay = _module()
    events: list[str] = []
    builder, created = _fake_runtime_builder_factory(events)
    config = _fake_config(tmp_path)

    manifest = replay.run_experiment(
        config,
        runtime_builder=builder,
        run_directory=tmp_path / "run",
    )

    assert manifest["status"] == "TEST_ONLY"
    assert manifest["authoritative"] is False
    assert manifest["counts"] == {
        "source_trajectories": 1,
        "restore_attempts": 9,
        "dynamic_comparisons": 90,
    }
    assert manifest["attempt_ids"] == [
        f"capture_{capture}_restore_{ordinal}"
        for capture in (10, 50, 100)
        for ordinal in (1, 2, 3)
    ]
    assert len(created) == 10
    assert (tmp_path / "run" / "action_tape.npy").is_file()
    assert (tmp_path / "run" / "run_manifest.json").is_file()
    assert (tmp_path / "run" / "terminal_manifest.json").is_file()
    assert (tmp_path / "run" / "restore_results.json").is_file()


def test_persisted_tape_is_reloaded_and_generation_happens_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(tmp_path)
    generated = 0
    loaded = 0
    original_generate = replay.generate_action_tape
    original_load = replay.load_action_tape

    def counted_generate(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal generated
        generated += 1
        return original_generate(*args, **kwargs)

    def counted_load(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal loaded
        loaded += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(replay, "generate_action_tape", counted_generate)
    monkeypatch.setattr(replay, "load_action_tape", counted_load)
    replay.run_experiment(config, runtime_builder=builder, run_directory=tmp_path / "run")
    assert generated == 1
    assert loaded == 1


def test_terminal_manifest_is_published_on_early_source_failure_without_retry(tmp_path: Path) -> None:
    replay = _module()
    events: list[str] = []
    builder, created = _fake_runtime_builder_factory(events, terminate_at=1)
    run_directory = tmp_path / "failed-run"
    with pytest.raises(replay.EarlyTerminationError):
        replay.run_experiment(
            _fake_config(tmp_path), runtime_builder=builder, run_directory=run_directory
        )
    terminal = json.loads((run_directory / "terminal_manifest.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "BLOCKED"
    assert terminal["counts"]["source_trajectories"] == 0
    assert len(created) == 1


def test_run_directory_creation_refuses_overwrite(tmp_path: Path) -> None:
    replay = _module()
    replay.create_run_directory(tmp_path, "fixed-id")
    with pytest.raises(replay.PublicationError, match="already exists"):
        replay.create_run_directory(tmp_path, "fixed-id")


def test_post_process_guard_rejects_non_allowlisted_derived_mutation() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    state = source.capture(source_step=1)
    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")

    def bad_post_process() -> None:
        fresh.sim.data.qpos[0] += 1.0

    fresh.env._post_process = bad_post_process
    with pytest.raises(replay.M1RuntimeError, match="non-visual"):
        fresh.restore(state)


def test_collected_invariants_include_body_poses_gripper_physical_and_reward() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    adapter.step(np.zeros(7, dtype=np.float32))
    invariants = adapter.collect_invariants()
    assert "body_xpos" in invariants
    assert "body_xquat" in invariants
    assert "gripper_physical" in invariants
    assert invariants["reward"] == 0.25


def test_collected_invariants_expose_all_goal_predicates_as_exact_state() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    adapter.inner.parsed_problem = {
        "goal_state": [("On", "bowl", "plate"), ("Open", "gripper")]
    }
    adapter.inner._eval_predicate = lambda goal: goal[0] == "On"
    invariants = adapter.collect_invariants()
    assert invariants["predicates"] == {
        "available": True,
        "goals": [
            {
                "index": 0,
                "expression": ["On", "bowl", "plate"],
                "value": True,
            },
            {"index": 1, "expression": ["Open", "gripper"], "value": False},
        ],
    }
    changed = dict(invariants)
    changed["predicates"] = {
        **invariants["predicates"],
        "goals": [
            invariants["predicates"]["goals"][0],
            {"index": 1, "expression": ["Open", "gripper"], "value": True},
        ],
    }
    predicate_results = [
        result
        for result in replay.compare_invariants(invariants, changed)
        if result.path.startswith("predicates")
    ]
    assert predicate_results
    assert all(result.comparison_class is replay.ComparisonClass.EXACT for result in predicate_results)
    assert any(not result.passed for result in predicate_results)


def test_missing_invariant_is_a_failed_closed_comparison() -> None:
    replay = _module()
    results = replay.compare_invariants({"qpos": [1.0]}, {})
    assert len(results) == 1
    assert results[0].passed is False


def test_deferred_official_child_is_reset_once_before_simulator_binding() -> None:
    """Construction must not inspect a lazy child's simulator before reset."""

    replay = _module()
    events: list[str] = []
    child = _FakeEnv(events)

    class DeferredChild:
        def __init__(self) -> None:
            self._reset_done = False

        @property
        def sim(self) -> Any:
            events.append("sim_bind")
            if not self._reset_done:
                raise AssertionError("simulator was probed before official reset")
            return child.sim

        def reset(self, *args: Any, **kwargs: Any) -> Any:
            result = child.reset(*args, **kwargs)
            self._reset_done = True
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(child, name)

    deferred = DeferredChild()

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        return {"env": _FakeVector(deferred), "mujoco": _FakeMujoco(child.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    assert adapter.env is deferred
    assert child.reset_calls == 1
    assert events.count("construction_reset") == 1
    assert events.index("construction_reset") < events.index("sim_bind")


def test_enum_only_mujoco_uses_raw_model_and_data_handles() -> None:
    replay = _module()
    events: list[tuple[str, Any, Any, int]] = []

    class RawModel:
        pass

    class RawData:
        def __init__(self) -> None:
            self.value = np.zeros(4, dtype=np.float64)

    raw_model = RawModel()
    raw_data = RawData()

    class Wrapper:
        def __init__(self, raw: Any) -> None:
            self._model = raw if isinstance(raw, RawModel) else None
            self._data = raw if isinstance(raw, RawData) else None

    wrapped_model = Wrapper(raw_model)
    wrapped_data = Wrapper(raw_data)

    class StateEnum:
        mjSTATE_INTEGRATION = 13

    class EnumOnlyMujoco:
        mjtState = StateEnum

        def mj_stateSize(self, model: Any, state_mask: int) -> int:
            assert model is raw_model
            assert state_mask == 13
            return 4

        def mj_setState(self, model: Any, data: Any, payload: np.ndarray, state_mask: int) -> None:
            assert model is raw_model
            assert data is raw_data
            assert state_mask == 13
            events.append(("set", model, data, state_mask))
            data.value = np.asarray(payload, dtype=np.float64).copy()

        def mj_getState(self, model: Any, data: Any, out: np.ndarray, state_mask: int) -> None:
            assert model is raw_model
            assert data is raw_data
            assert state_mask == 13
            events.append(("get", model, data, state_mask))
            out[...] = data.value

    expected = np.arange(4, dtype=np.float64)
    readback = replay.set_integration_state_exact(
        EnumOnlyMujoco(), wrapped_model, wrapped_data, expected, require_readback=True
    )
    np.testing.assert_array_equal(readback, expected)
    assert [event[0] for event in events] == ["set", "get"]


def test_required_runtime_audit_rejects_empty_controller_gripper_and_fixture() -> None:
    replay = _module()
    events: list[str] = []

    class EmptyEnv:
        def __init__(self) -> None:
            self.sim = SimpleNamespace(model=_FakeModel(), data=_FakeData())
            self.task = "task-zero"
            self.task_description = "instruction-zero"
            self.task_suite_name = "libero_spatial"
            self.init_state_id = 0
            self.timestep = 0
            self.cur_time = 0.0
            self.done = False

        def reset(self, **kwargs: Any) -> tuple[dict[str, int], dict[str, Any]]:
            events.append("construction_reset")
            return {}, {}

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = EmptyEnv()
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.M1RuntimeError, match="controller|gripper|fixture|audit"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_config_requires_exact_reachable_classification_and_frozen_flags() -> None:
    replay = _module()
    config = _fake_config(Path("/tmp"))
    config["state_audit"] = {
        "reachable_fields": sorted(
            field
            for fields in replay.DEFAULT_STATE_CLASSIFICATION.values()
            for field in fields
        ),
        "classification": deepcopy(replay.DEFAULT_STATE_CLASSIFICATION),
    }
    # The fake config is deliberately incomplete today; the implementation
    # must reject omission rather than silently substituting defaults.
    with pytest.raises(replay.M1ConfigError, match="state_audit|classification|frozen"):
        replay.validate_config(config)

    config = _fake_config(Path("/tmp"))
    config["action_tape"]["algorithm"] = "RandomState"
    with pytest.raises(replay.M1ConfigError, match="algorithm|PCG64|frozen"):
        replay.validate_config(config)


def test_static_gate_uses_persisted_capture_reference_and_exact_readback(tmp_path: Path) -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(tmp_path)
    # A persisted reference is a required argument to the static gate.  The
    # current implementation has no static row/readback result, so this is a
    # deliberately failing contract test.
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    capture = source.capture(source_step=1)
    reference = source.collect_invariants()
    result = source.restore(capture, static_reference=reference)
    assert result["static"]["gate"]["pass"] is True
    assert result["immediate_readback"]["exact"] is True


def test_corrupt_persisted_capture_or_reference_is_loaded_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(tmp_path)
    original_load = replay.load_record_bundle

    def corrupt_load(directory: Path, stem: str) -> Any:
        record = original_load(directory, stem)
        if stem.startswith("capture_"):
            raise replay.M1SchemaError("corrupt persisted capture/reference")
        return record

    monkeypatch.setattr(replay, "load_record_bundle", corrupt_load)
    with pytest.raises(replay.M1SchemaError, match="corrupt persisted"):
        replay.run_experiment(config, runtime_builder=builder, run_directory=tmp_path / "run")


def test_existing_caller_run_directory_is_rejected_before_publication(tmp_path: Path) -> None:
    replay = _module()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(replay.PublicationError, match="caller|existing|overwrite"):
        replay.run_experiment(
            _fake_config(tmp_path), runtime_builder=lambda _: {}, run_directory=existing
        )


def test_compare_floating_quaternion_sign_is_invariant_per_row_and_reports_metrics() -> None:
    replay = _module()
    expected = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    actual = np.asarray([[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
    result = replay.compare_floating(
        expected, actual, path="body_xquat", quaternion_sign_invariant=True
    )
    assert result.passed is True
    assert result.max_abs == 0.0
    assert result.mean_abs == 0.0
    assert result.tolerance == replay.FLOATING_ATOL


def test_contact_comparison_separates_exact_geom_pairs_from_floating_distances() -> None:
    replay = _module()
    left = [("geom_a", "geom_b", -0.1)]
    right = [("geom_b", "geom_a", -0.1000000000005)]
    result = replay.compare_contact_pairs(left, right)
    assert result.passed is True
    assert result.comparison_class == replay.ComparisonClass.EXACT
    assert result.subresults["geom_pairs"].comparison_class == replay.ComparisonClass.EXACT
    assert result.subresults["distances"].comparison_class == replay.ComparisonClass.FLOATING_PHYSICAL


def test_controller_numeric_and_new_update_and_counter_time_are_separate() -> None:
    replay = _module()
    expected = {
        "controller": {"initial_joint": [0.0], "new_update": True},
        "counters": {"timestep": 4, "done": False, "cur_time": 0.4},
    }
    actual = {
        "controller": {"initial_joint": [5e-13], "new_update": False},
        "counters": {"timestep": 4, "done": False, "cur_time": 0.4000000000005},
    }
    results = {item.path: item for item in replay.compare_invariants(expected, actual)}
    assert results["controller.new_update"].comparison_class == replay.ComparisonClass.EXACT
    assert results["controller.initial_joint"].comparison_class == replay.ComparisonClass.FLOATING_PHYSICAL
    assert results["counters.timestep"].comparison_class == replay.ComparisonClass.EXACT
    assert results["counters.cur_time"].comparison_class == replay.ComparisonClass.FLOATING_PHYSICAL
    assert results["controller.new_update"].passed is False


def test_source_builds_only_capture_states_and_references_at_required_steps() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    calls: list[tuple[str, int]] = []
    original_capture = adapter.capture
    original_collect = adapter.collect_invariants

    def capture(*, source_step: int | None = None) -> Any:
        calls.append(("capture", int(source_step or -1)))
        return original_capture(source_step=source_step)

    def collect() -> Any:
        calls.append(("reference", adapter.step_count))
        return original_collect()

    adapter.capture = capture  # type: ignore[method-assign]
    adapter.collect_invariants = collect  # type: ignore[method-assign]
    replay.run_source_trajectory(adapter, replay.generate_action_tape(), replay.build_capture_windows())
    assert [step for kind, step in calls if kind == "capture"] == [10, 50, 100]
    assert [step for kind, step in calls if kind == "reference"] == list(range(10, 21)) + list(range(50, 61)) + list(range(100, 111))


def test_restore_result_schema_contains_static_one_plus_ten_dynamic_rows_and_outcomes(
    tmp_path: Path,
) -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(tmp_path)
    manifest = replay.run_experiment(config, runtime_builder=builder, run_directory=tmp_path / "run")
    assert manifest["counts"]["static_comparisons"] == 9
    assert manifest["counts"]["dynamic_comparisons"] == 90
    rows = json.loads((tmp_path / "run" / "restore_results.json").read_text(encoding="utf-8"))["attempts"]
    assert all(len(attempt["static"]) == 1 for attempt in rows)
    assert all(len(attempt["rows"]) == 10 for attempt in rows)
    for attempt in rows:
        for row in [attempt["static"][0], *attempt["rows"]]:
            assert {"terminated", "truncated", "done", "reward", "success"} <= set(row)


def test_post_process_nested_visual_allowlist_requires_forced_refresh() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    state = source.capture(source_step=1)
    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    fresh.inner.object_properties = {"fixture": {"vis_site_names": {"x": False}}}
    fresh.env._update_observables = lambda: None
    with pytest.raises(replay.M1RuntimeError, match="force=True|visual|refresh"):
        fresh.restore(state)


def test_experiment_cursor_starts_at_zero_after_official_reset_settle() -> None:
    replay = _module()
    events: list[str] = []

    class SettlingEnv(_FakeEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, int], dict[str, Any]]:
            result = super().reset(seed=seed)
            self.timestep = 10
            self.cur_time = 1.0
            return result

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = SettlingEnv(events)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    assert adapter.experiment_step_count == 0
    assert adapter.inner.timestep == 10
    adapter.step(np.zeros(7, dtype=np.float32))
    assert adapter.experiment_step_count == 1
    assert adapter.inner.timestep == 11


def test_source_survives_110_experiment_actions_when_reset_cursor_is_ten() -> None:
    replay = _module()
    events: list[str] = []

    class SettlingEnv(_FakeEnv):
        def reset(self, seed: int | None = None) -> tuple[dict[str, int], dict[str, Any]]:
            result = super().reset(seed=seed)
            self.timestep = 10
            self.cur_time = 1.0
            return result

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = SettlingEnv(events)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    captures, _ = replay.run_source_trajectory(
        adapter,
        replay.generate_action_tape(),
        replay.build_capture_windows(),
    )
    assert sorted(captures) == [10, 50, 100]
    assert adapter.experiment_step_count == 110
    assert adapter.inner.timestep == 120


def test_injected_runtime_is_test_only_and_never_authoritative_pass(tmp_path: Path) -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    manifest = replay.run_experiment(
        _fake_config(tmp_path), runtime_builder=builder, run_directory=tmp_path / "run"
    )
    assert manifest["status"] == "TEST_ONLY"
    assert manifest["authoritative"] is False
    assert manifest["gates"]["authoritative"]["pass"] is False


def test_runtime_audit_discovers_all_observable_mutable_fields_and_unknowns() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    adapter.inner.observables = {
        f"observable_{index}": SimpleNamespace(
            _timer=0,
            _cache={},
            _delay=0,
            _filter=None,
            _corrupter=None,
            unknown_mutable=[] if index == 30 else None,
        )
        for index in range(31)
    }
    with pytest.raises(replay.UnknownMutableStateError, match="unknown|unclassified|observable"):
        adapter.audit_runtime_state()


def test_runtime_audit_does_not_claim_unobserved_determinism_or_sleeping() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    audit = adapter.audit_runtime_state()
    assert audit["observable_modifiers"] != "deterministic"
    assert audit["sleeping"] is not False


def test_cli_generates_tape_once_and_passes_same_tape_to_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _module()
    original_generate = replay.generate_action_tape
    generated: list[np.ndarray] = []
    received: list[np.ndarray] = []

    def generate_once(*args: Any, **kwargs: Any) -> np.ndarray:
        value = original_generate(*args, **kwargs)
        generated.append(value)
        return value

    def fake_run(config: Mapping[str, Any], *, tape: np.ndarray) -> dict[str, Any]:
        received.append(tape)
        return {"status": "TEST_ONLY", "finished_at": "fake"}

    monkeypatch.setattr(replay, "generate_action_tape", generate_once)
    monkeypatch.setattr(replay, "run_experiment", fake_run)
    assert replay.main(["--config", str(CONFIG_PATH), "--run"]) == 2
    assert len(generated) == 1
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], generated[0])


def test_collect_invariants_include_observables_constraints_eef_velocity_jacobians_and_mass() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    adapter.inner.observables = {f"observable_{index}": np.asarray([index], dtype=np.float64) for index in range(31)}
    adapter.inner.sim.data.efc_force = np.asarray([0.1, 0.2], dtype=np.float64)
    adapter.inner.sim.data.efc_type = np.asarray([1, 2], dtype=np.int32)
    controller = adapter.inner.controller
    controller.ee_pos_vel = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    controller.ee_ori_vel = np.asarray([0.4, 0.5, 0.6], dtype=np.float64)
    controller.J_full = np.eye(2, dtype=np.float64)
    controller.J_pos = np.ones((1, 2), dtype=np.float64)
    controller.J_ori = np.ones((1, 2), dtype=np.float64) * 2
    controller.mass_matrix = np.eye(2, dtype=np.float64) * 3
    invariants = adapter.collect_invariants()
    assert invariants["observables"]["observable_30"] == [30.0]
    np.testing.assert_array_equal(invariants["constraints"]["efc_force"], [0.1, 0.2])
    np.testing.assert_array_equal(invariants["eef_pose"]["linear_velocity"], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(invariants["eef_pose"]["angular_velocity"], [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(invariants["controller_kinematic_caches"]["J_full"], [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(invariants["controller_kinematic_caches"]["mass_matrix"], [[3.0, 0.0], [0.0, 3.0]])


def test_required_invariant_schema_rejects_missing_physical_fields() -> None:
    replay = _module()
    with pytest.raises(replay.M1SchemaError, match="required|missing|eef|constraint"):
        replay.validate_invariant_snapshot({"integration": [0.0]})


def test_required_invariant_schema_rejects_named_object_without_qpos() -> None:
    replay = _module()
    with pytest.raises(replay.M1SchemaError, match=r"objects\.bowl\.joints\.joint\.qpos"):
        replay.validate_invariant_snapshot(
            {
                "objects": {
                    "bowl": {
                        "body_pos": [0.0, 0.0, 0.0],
                        "body_quat": [1.0, 0.0, 0.0, 0.0],
                        "joints": {"joint": {"qvel": 0.0}},
                    }
                }
            }
        )


def test_visual_guard_traverses_fixture_object_properties() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    fixture = adapter.inner.fixtures_dict["fixture"]
    fixture.object_properties = {"vis_site_names": {"fixture_site": False}}
    snapshot = replay._visual_state_snapshot(adapter.inner)
    assert snapshot["vis_site_names"]["fixtures"]["fixture"] == {"fixture_site": False}


def test_strict_provenance_false_and_changed_frozen_identity_are_rejected() -> None:
    replay = _module()
    config = replay.load_config(CONFIG_PATH)
    config["strict_provenance"] = False
    with pytest.raises(replay.M1ConfigError, match="strict_provenance|frozen"):
        replay.validate_config(config)

    config = replay.load_config(CONFIG_PATH)
    config["obs_type"] = "state"
    with pytest.raises(replay.M1ConfigError, match="obs_type|frozen"):
        replay.validate_config(config)


def test_strict_provenance_requires_exact_source_checkout_audit() -> None:
    replay = _module()
    config = replay.load_config(CONFIG_PATH)
    config["source_checkouts"].pop("robosuite")
    with pytest.raises(replay.M1ConfigError, match="checkout|robosuite|source"):
        replay.validate_config(config)


def test_every_frozen_identity_digest_and_execution_field_is_fail_closed() -> None:
    replay = _module()
    mutations = (
        ("obs_type", lambda c: c.__setitem__("obs_type", "state")),
        ("suite", lambda c: c["task"].__setitem__("suite", "libero_object")),
        ("task_id", lambda c: c["task"].__setitem__("task_id", 1)),
        ("init_state_id", lambda c: c["task"].__setitem__("init_state_id", 1)),
        ("seed", lambda c: c["task"].__setitem__("seed", 1)),
        ("trajectory_id", lambda c: c.__setitem__("trajectory_id", "other")),
        ("source_steps", lambda c: c.__setitem__("source_steps", 109)),
        ("tape_algorithm", lambda c: c["action_tape"].__setitem__("algorithm", "RandomState")),
        ("tape_bit_generator", lambda c: c["action_tape"].__setitem__("bit_generator", "MT19937")),
        ("tape_seed", lambda c: c["action_tape"].__setitem__("seed", 1)),
        ("tape_digest", lambda c: c["action_tape"].__setitem__("sha256", "0" * 64)),
        ("captures", lambda c: c.__setitem__("capture_steps", [10, 50, 99])),
        ("future", lambda c: c.__setitem__("future_steps", 9)),
        ("restores", lambda c: c.__setitem__("restores_per_capture", 2)),
        ("rtol", lambda c: c["tolerances"].__setitem__("rtol", 1e-9)),
        ("atol", lambda c: c["tolerances"].__setitem__("atol", 1e-9)),
        ("offline", lambda c: c["runtime"].__setitem__("offline", False)),
        ("renderer", lambda c: c["runtime"]["renderer"].__setitem__("MUJOCO_GL", "osmesa")),
        ("environment", lambda c: c["runtime"]["environment"].__setitem__("HF_HOME", "/tmp/not-frozen")),
        ("cpu_path", lambda c: c["paths"].__setitem__("cpu_python", "/tmp/python")),
        ("source_evidence_digest", lambda c: c["paths"]["source_evidence_sha256"].__setitem__("osc", "0" * 64)),
        ("checkout_sha", lambda c: c["source_checkouts"]["robosuite"].__setitem__("git_sha", "0" * 40)),
        ("bddl_digest", lambda c: c["hashes"]["bddl"].__setitem__("sha256", "0" * 64)),
        ("init_digest", lambda c: c["hashes"]["init_state"].__setitem__("sha256", "0" * 64)),
        ("runtime_digest", lambda c: c["paths"].__setitem__("runtime_lock_sha256", "0" * 64)),
        ("artifact_digest", lambda c: c["paths"].__setitem__("artifact_manifest_sha256", "0" * 64)),
        ("pin", lambda c: c["pins"].__setitem__("mujoco_git_sha", "0" * 40)),
        ("worktree_audit", lambda c: c["worktree_audit"].__setitem__("expected_clean", True)),
        ("publication", lambda c: c["publication"].__setitem__("atomic", False)),
    )
    for label, mutate in mutations:
        config = replay.load_config(CONFIG_PATH)
        mutate(config)
        with pytest.raises(replay.M1ConfigError):
            replay.validate_config(config)


def test_reset_signature_is_validated_without_unseeded_retry() -> None:
    replay = _module()
    events: list[str] = []

    class NoSeedResetEnv(_FakeEnv):
        def reset(self) -> tuple[dict[str, int], dict[str, Any]]:
            self.reset_calls += 1
            events.append("unseeded_reset")
            return {}, {}

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = NoSeedResetEnv(events)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.M1RuntimeError, match="seed|signature|reset"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_reset_provenance_separates_requested_init_state_from_post_reset_cursor() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    assert adapter.reset_provenance["requested_init_state_id"] == 0
    assert adapter.reset_provenance["post_reset_timestep"] == 0


def test_terminal_manifest_survives_source_close_failure(tmp_path: Path) -> None:
    replay = _module()
    events: list[str] = []

    class FailingCloseEnv(_FakeEnv):
        def close(self) -> None:
            raise RuntimeError("source close failed")

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = FailingCloseEnv(events, terminate_at=1)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    run_directory = tmp_path / "failed-close"
    with pytest.raises(replay.EarlyTerminationError, match="terminated"):
        replay.run_experiment(
            _fake_config(tmp_path), runtime_builder=builder, run_directory=run_directory
        )
    terminal = json.loads((run_directory / "terminal_manifest.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "BLOCKED"
    assert "source close failed" in terminal["cleanup_errors"]


@pytest.mark.parametrize("terminate_at", [100, 109])
def test_source_termination_at_100_or_109_is_terminal_failure(terminate_at: int) -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events, terminate_at=terminate_at)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    with pytest.raises(replay.EarlyTerminationError, match="terminated"):
        replay.run_source_trajectory(
            adapter,
            replay.generate_action_tape(),
            replay.build_capture_windows(),
        )


@pytest.mark.parametrize("owner_path", ["inner", "robot", "controller", "gripper", "fixture"])
def test_runtime_audit_rejects_unknown_mutable_field_on_every_owner_root(owner_path: str) -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    if owner_path == "inner":
        owner = adapter.inner
    elif owner_path == "robot":
        owner = adapter.inner.robots[0]
    elif owner_path == "controller":
        owner = adapter.inner.controller
    elif owner_path == "gripper":
        owner = adapter.inner.gripper
    else:
        owner = adapter.inner.fixtures_dict["fixture"]
    owner.unexpected_mutable_state = []
    with pytest.raises(replay.UnknownMutableStateError, match="unknown|unclassified|mutable"):
        adapter.audit_runtime_state()


def test_outer_task_identity_and_object_children_are_audited() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        inner = _InnerWithoutTask(events)
        outer = _OuterLiberoLike(inner)
        return {"env": _FakeVector(outer), "mujoco": _FakeMujoco(inner.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    assert not hasattr(adapter.inner, "task")
    assert adapter._identity(0)["task"] == "outer-task-zero"
    assert "outer" in adapter.runtime_audit["owner_roots"]
    assert "task" in adapter.runtime_audit["outer_identity"]

    adapter.inner.objects_dict["bowl"].unknown_object_cache = []
    with pytest.raises(replay.UnknownMutableStateError, match="object|unknown|unclassified"):
        adapter.audit_runtime_state()


def test_outer_unknown_mutable_field_is_closed_loop_blocked() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        inner = _InnerWithoutTask(events)
        outer = _OuterLiberoLike(inner)
        return {"env": _FakeVector(outer), "mujoco": _FakeMujoco(inner.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    adapter.env.outer_unknown_cache = []
    with pytest.raises(replay.UnknownMutableStateError, match="outer|unknown|unclassified"):
        adapter.audit_runtime_state()


@pytest.mark.parametrize("component", ["inner", "controller", "gripper", "robot"])
def test_source_declared_looking_mutable_fields_still_fail_closed(component: str) -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        if component == "inner":
            child = _SourceDefinedInner(events)
        else:
            child = _FakeEnv(events)
            if component == "controller":
                child.controller = _SourceDefinedController(events)
                child.robots[0].controller = child.controller
            elif component == "gripper":
                child.gripper = _SourceDefinedGripper()
                child.robots[0].gripper = child.gripper
            else:
                child.robots = [_SourceDefinedRobot(child.controller, child.gripper)]
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.UnknownMutableStateError, match="unknown|unclassified|mutable"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_concrete_serialization_coverage_has_explicit_artifact_and_restore_seams() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    audit = adapter.audit_runtime_state()
    coverage = audit["serialization_coverage"]
    templates = coverage["source_design_templates"]
    for path in (
        "mjdata.integration",
        "fixture.model.body_pos",
        "fixture.model.body_quat",
        "env.timestep",
        "env.cur_time",
        "env.done",
        "controller",
        "gripper.current_action",
        "identity",
        "observables",
        "objects.joints",
    ):
        assert path in templates
    classified = set(audit["serialized_paths"])
    concrete = coverage["concrete"]
    assert classified == set(concrete)
    for path in classified:
        row = concrete[path]
        assert row["capture_artifact"]
        assert row["persistence_artifact"]
        assert row["restore_operation"]
        assert row["readback_comparison"]
    assert not any(path in audit["classification"] for path in templates)


def test_named_object_joint_snapshot_uses_mujoco_type_widths() -> None:
    replay = _module()

    class WidthModel:
        jnt_type = np.asarray([0, 1, 2, 3], dtype=np.int32)
        jnt_qposadr = np.asarray([0, 7, 11, 12], dtype=np.int32)
        jnt_dofadr = np.asarray([0, 6, 9, 10], dtype=np.int32)

        def body_name2id(self, name: str) -> int:
            assert name == "fixture"
            return 0

        def joint_name2id(self, name: str) -> int:
            return {"free": 0, "ball": 1, "slide": 2, "hinge": 3}[name]

    class WidthData:
        body_xpos = np.zeros((1, 3), dtype=np.float64)
        body_xquat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        qpos = np.arange(13, dtype=np.float64)
        qvel = np.arange(11, dtype=np.float64) + 100.0

    inner = SimpleNamespace(
        objects_dict={
            "body": SimpleNamespace(
                root_body="fixture",
                joints=["free", "ball", "slide", "hinge"],
            )
        },
        sim=SimpleNamespace(model=WidthModel(), data=WidthData()),
    )
    snapshot = replay._object_snapshot(inner)
    joints = snapshot["body"]["joints"]
    assert len(joints["free"]["qpos"]) == 7
    assert len(joints["free"]["qvel"]) == 6
    assert len(joints["ball"]["qpos"]) == 4
    assert len(joints["ball"]["qvel"]) == 3
    assert len(joints["slide"]["qpos"]) == len(joints["slide"]["qvel"]) == 1
    assert len(joints["hinge"]["qpos"]) == len(joints["hinge"]["qvel"]) == 1
    assert joints["free"]["qpos_width"] == 7
    assert joints["free"]["qvel_width"] == 6


def test_wrong_runtime_fact_publishes_blocked_terminal_gate_without_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = _module()
    config = replay.load_config(CONFIG_PATH)
    original_facts = replay._runtime_facts
    monkeypatch.setattr(
        replay,
        "_worktree_audit",
        lambda _config: {
            "pass": True,
            "implementation": {"unexpected": []},
            "other_dirt": [],
            "errors": [],
        },
    )

    def wrong_facts(current: Mapping[str, Any]) -> dict[str, Any]:
        facts = dict(original_facts(current))
        facts["mujoco"] = "0.0.0-wrong"
        return facts

    monkeypatch.setattr(replay, "_runtime_facts", wrong_facts)
    run_directory = tmp_path / "runtime-mismatch"
    with pytest.raises(replay.M1ConfigError, match="runtime|fact|mismatch"):
        replay.run_experiment(config, run_directory=run_directory)
    terminal = json.loads((run_directory / "terminal_manifest.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "BLOCKED"
    assert terminal["authoritative"] is False
    assert terminal["gates"]["E_provenance"]["pass"] is False
    gate = terminal["provenance"]["runtime_facts_gate"]
    assert gate["pass"] is False
    assert any("mujoco" in str(item) for item in gate["errors"])


def test_runtime_audit_reports_closed_owner_roots_and_handle_traversal() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    audit = adapter.audit_runtime_state()
    assert set(audit["owner_roots"]) >= {
        "inner",
        "robot[0]",
        "robot[0].controller",
        "robot[0].gripper",
        "observables",
        "fixtures",
        "task",
    }
    assert set(audit["handle_classification"]) >= {"inner.sim", "inner.sim.model", "inner.sim.data"}
    assert audit["unknown_paths"] == []


def test_serialized_observable_timer_and_cache_round_trip_through_capture_metadata() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    adapter.inner.observables = {
        "obs": SimpleNamespace(_timer=0.037, _cache={"payload": [1.0, 2.0]})
    }
    state = adapter.capture(source_step=0)
    assert state.serialized_runtime["observables.obs._timer"] == 0.037
    assert state.serialized_runtime["observables.obs._cache"] == {"payload": [1.0, 2.0]}
    record = replay.replay_state_record(state)
    assert record.metadata["serialized_runtime"] == state.serialized_runtime
    restored = replay.replay_state_from_record(record)
    assert restored.serialized_runtime == state.serialized_runtime


def _inject_dirty_implementation_status(
    replay: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_git_command = replay._git_command

    def dirty_git_command(path: Path, *args: str) -> str | None:
        result = original_git_command(path, *args)
        if path == replay.ROOT and args == (
            "status",
            "--porcelain",
            "--untracked-files=all",
        ):
            return " M scripts/m1_state_replay.py"
        return result

    monkeypatch.setattr(replay, "_git_command", dirty_git_command)


def test_dirty_implementation_files_block_authoritative_worktree_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = _module()
    _inject_dirty_implementation_status(replay, monkeypatch)
    config = replay.load_config(CONFIG_PATH)
    audit = replay._worktree_audit(config)
    assert audit["pass"] is False
    assert "scripts/m1_state_replay.py" in audit["implementation"]["unexpected"]


def test_blocked_worktree_audit_is_retained_in_terminal_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay = _module()
    _inject_dirty_implementation_status(replay, monkeypatch)
    config = replay.load_config(CONFIG_PATH)
    run_directory = tmp_path / "blocked-worktree"
    with pytest.raises(replay.M1ConfigError, match="worktree|implementation"):
        replay.run_experiment(config, run_directory=run_directory)
    terminal = json.loads(
        (run_directory / "terminal_manifest.json").read_text(encoding="utf-8")
    )
    audit = terminal["provenance"]["worktree_audit"]
    assert audit["pass"] is False
    assert audit["other_dirt"] == []
    assert "scripts/m1_state_replay.py" in audit["implementation"]["unexpected"]


def test_wrong_runtime_fact_blocks_strict_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = _module()
    config = replay.load_config(CONFIG_PATH)
    original_version = replay._package_version

    def wrong_version(name: str) -> str | None:
        if name in {"torch", "transformers", "lerobot", "robosuite"}:
            return "0.0.0-wrong"
        return original_version(name)

    monkeypatch.setattr(replay, "_package_version", wrong_version)
    with pytest.raises(replay.M1ConfigError, match="runtime|version|fact"):
        replay._provenance(config, tape_hash=config["action_tape"]["sha256"])


def test_missing_init_state_evidence_blocks_strict_provenance() -> None:
    replay = _module()
    config = replay.load_config(CONFIG_PATH)
    config.pop("init_state_id_evidence", None)
    with pytest.raises(replay.M1ConfigError, match="init_state|selection|evidence"):
        replay._provenance(config, tape_hash=config["action_tape"]["sha256"])


def test_reset_provenance_records_selected_init_state_evidence_separately() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    assert adapter.reset_provenance["init_state_id_evidence"] == {
        "requested": 0,
        "selected_pre_reset": 0,
        "post_reset_cursor": 1,
    }


def _collision_runtime_builder(
    events: list[str], duplicate_collision_geoms: Any = True
) -> Any:
    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.fixtures_dict = {
            "fixture": _CollisionConfiguredNode(duplicate_collision_geoms)
        }
        child.objects_dict = {
            "bowl": _CollisionConfiguredNode(duplicate_collision_geoms)
        }
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    return builder


def test_duplicate_collision_geoms_is_explicit_immutable_fixture_object_configuration() -> None:
    replay = _module()
    events: list[str] = []
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")),
        runtime_builder=_collision_runtime_builder(events),
        tape_hash="tape",
    )
    audit = adapter.audit_runtime_state()
    for path in (
        "fixtures.fixture.duplicate_collision_geoms",
        "objects.bowl.duplicate_collision_geoms",
    ):
        assert audit["field_classification"][path] == replay.IMMUTABLE
        evidence = audit["field_evidence"][path]
        assert evidence["source_declared"] is True
        assert evidence["expected_type"] == "bool"
        assert "external/robosuite/robosuite/models/objects/objects.py:57,322" in evidence[
            "source_evidence"
        ]


def test_duplicate_collision_geoms_rejects_non_boolean_content_shape() -> None:
    replay = _module()
    events: list[str] = []
    with pytest.raises(replay.UnknownMutableStateError, match="duplicate_collision_geoms"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")),
            runtime_builder=_collision_runtime_builder(events, duplicate_collision_geoms=[]),
            tape_hash="tape",
        )


@pytest.mark.parametrize(
    "owner_kind",
    ["outer", "inner", "robot", "controller", "gripper", "object"],
)
def test_generic_events_field_on_each_official_owner_is_not_a_production_allowlist(
    owner_kind: str,
) -> None:
    replay = _module()
    events: list[str] = []
    if owner_kind == "outer":
        def builder(config: Mapping[str, Any]) -> dict[str, Any]:
            child = _FakeEnv(events)
            outer = _OuterLiberoLike(child)
            return {"env": _FakeVector(outer), "mujoco": _FakeMujoco(child.sim.data, events)}

        builder_for_test = builder
    else:
        builder_for_test, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder_for_test, tape_hash="tape"
    )
    if owner_kind == "outer":
        owner = adapter.env
    elif owner_kind == "inner":
        owner = adapter.inner
    elif owner_kind == "robot":
        owner = adapter.inner.robots[0]
    elif owner_kind == "controller":
        owner = adapter.inner.controller
    elif owner_kind == "gripper":
        owner = adapter.inner.gripper
    else:
        adapter.inner.objects_dict = {
            "bowl": SimpleNamespace(root_body="fixture", joints=[])
        }
        owner = adapter.inner.objects_dict["bowl"]
    if hasattr(owner, "events"):
        delattr(owner, "events")
    owner.events = []
    with pytest.raises(replay.UnknownMutableStateError, match="events|unknown|unclassified"):
        adapter.audit_runtime_state()


def test_source_declared_mutable_immutable_field_requires_registry_shape_proof() -> None:
    replay = _module()
    events: list[str] = []

    class SourceDeclaredBadFixture:
        def __init__(self) -> None:
            self.root_body = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.fixtures_dict = {"fixture": SourceDeclaredBadFixture()}
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.UnknownMutableStateError, match="root_body|shape|mutable"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_gate_d_requires_exact_frozen_design_template_set() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    audit = adapter.audit_runtime_state()
    coverage = deepcopy(audit["serialization_coverage"])
    del coverage["source_design_templates"]["controller"]
    assert not replay._serialization_coverage_contract_pass(
        coverage, audit["serialized_paths"]
    )


def test_gate_d_rejects_empty_design_template_set() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    audit = adapter.audit_runtime_state()
    coverage = deepcopy(audit["serialization_coverage"])
    coverage["source_design_templates"] = {}
    assert not replay._serialization_coverage_contract_pass(
        coverage, audit["serialized_paths"]
    )


def test_gate_d_requires_exact_integration_template_seams() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    audit = adapter.audit_runtime_state()
    coverage = deepcopy(audit["serialization_coverage"])
    coverage["source_design_templates"]["mjdata.integration"]["restore_operation"] = "wrong"
    assert not replay._serialization_coverage_contract_pass(
        coverage, audit["serialized_paths"]
    )


def test_gate_d_requires_nonempty_complete_observable_expansion() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    adapter.inner.observables = {
        f"observable_{index}": SimpleNamespace(
            _timer=0.037,
            _cache={"payload": [float(index)]},
            _sensor=lambda: 0.0,
            _corrupter=lambda value: value,
            _filter=lambda value: value,
            _delayer=lambda value: value,
        )
        for index in range(31)
    }
    audit = adapter.audit_runtime_state()
    coverage = deepcopy(audit["serialization_coverage"])
    coverage["source_design_templates"]["observables"]["expanded_concrete_paths"] = []
    assert not replay._serialization_coverage_contract_pass(
        coverage, audit["serialized_paths"]
    )


def test_object_joint_registry_is_strict_and_carries_pinned_source_proof() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        inner = _InnerWithoutTask(events)
        outer = _OuterLiberoLike(inner)
        return {"env": _FakeVector(outer), "mujoco": _FakeMujoco(inner.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    path = "objects.bowl.joints"
    audit = adapter.runtime_audit
    assert audit["field_classification"][path] == replay.IMMUTABLE
    evidence = audit["field_evidence"][path]
    assert evidence["expected_type"] == "sequence[str]"
    assert evidence["expected_shape"] == "one-dimensional"
    assert "external/robosuite/robosuite/models/objects/objects.py" in evidence[
        "source_evidence"
    ]


def test_object_joint_registry_reads_mujoco_object_property() -> None:
    replay = _module()
    events: list[str] = []

    class MujocoObjectShaped:
        def __init__(self) -> None:
            self.root_body = "fixture"
            self._joints = ["joint"]

        @property
        def joints(self) -> list[str]:
            return self._joints

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.objects_dict = {"bowl": MujocoObjectShaped()}
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    evidence = adapter.runtime_audit["field_evidence"]["objects.bowl.joints"]
    assert adapter.runtime_audit["field_classification"]["objects.bowl.joints"] == replay.IMMUTABLE
    assert evidence["expected_type"] == "sequence[str]"
    assert evidence["property_registry"] == "robosuite.object.joints"


@pytest.mark.parametrize(
    "joint_value",
    [
        ("joint", 1),
        [""],
        [["joint"]],
        np.asarray(["joint"]),
        {"joint"},
    ],
)
def test_object_joint_registry_rejects_non_string_one_dimensional_values(
    joint_value: Any,
) -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        inner = _InnerWithoutTask(events)
        inner.objects_dict["bowl"].joints = joint_value
        outer = _OuterLiberoLike(inner)
        return {"env": _FakeVector(outer), "mujoco": _FakeMujoco(inner.sim.data, events)}

    with pytest.raises(replay.UnknownMutableStateError, match=r"objects\.bowl\.joints"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_osc_goals_orientation_reference_and_linear_interpolator_state_round_trip() -> None:
    replay = _module()
    events: list[str] = []

    class LinearInterpolator:
        def __init__(self) -> None:
            self.dim = 3
            self.ori_interpolate = None
            self.order = 1
            self.step = 2
            self.total_steps = 4.0
            self.use_delta_goal = False
            self.start = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
            self.goal = np.asarray([0.4, 0.5, 0.6], dtype=np.float64)

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.controller.relative_ori = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)
        child.controller.ori_ref = np.eye(3, dtype=np.float64)
        child.controller.interpolator_pos = LinearInterpolator()
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    state = adapter.capture(source_step=0)
    assert state.controller["relative_ori"] == [0.01, 0.02, 0.03]
    assert state.controller["ori_ref"] == np.eye(3, dtype=np.float64).tolist()
    assert "robot[0].controller.interpolator_pos.step" in state.serialized_runtime
    assert "robot[0].controller.interpolator_pos.start" in state.serialized_runtime

    adapter.inner.controller.relative_ori[...] = 9.0
    adapter.inner.controller.ori_ref[...] = 9.0
    adapter.inner.controller.interpolator_pos.step = 0
    adapter.inner.controller.interpolator_pos.start[...] = 9.0
    adapter.restore(state)
    np.testing.assert_array_equal(adapter.inner.controller.relative_ori, [0.01, 0.02, 0.03])
    np.testing.assert_array_equal(adapter.inner.controller.ori_ref, np.eye(3))
    assert adapter.inner.controller.interpolator_pos.step == 2
    np.testing.assert_array_equal(
        adapter.inner.controller.interpolator_pos.start, [0.1, 0.2, 0.3]
    )


def test_nonempty_unsupported_interpolator_is_a_closed_loop_block() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.controller.interpolator_pos = SimpleNamespace(step=1)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.UnknownMutableStateError, match="interpolator"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_runtime_audit_captures_global_and_env_local_rng_and_marks_accelerators_na() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.np_random = np.random.default_rng(9027)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    audit = adapter.runtime_audit
    assert audit["rng"]["python_global"]["classification"] == replay.SERIALIZED
    assert audit["rng"]["numpy_global"]["classification"] == replay.SERIALIZED
    assert audit["field_classification"]["inner.np_random"] == replay.SERIALIZED
    assert audit["rng"]["accelerators"] == {"cuda": "N/A", "mps": "N/A"}
    state = adapter.capture(source_step=0)
    assert "rng.python_global" in state.serialized_runtime
    assert "rng.numpy_global" in state.serialized_runtime
    assert "inner.np_random" in state.serialized_runtime


def test_nonzero_plugin_state_counter_blocks_even_when_plugin_count_is_zero() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.sim.model.nplugin = 0
        child.sim.model.npluginstate = 1
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.M1RuntimeError, match="npluginstate|plugin"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_strict_plugin_audit_requires_both_named_zero_counters() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    adapter = replay.RuntimeAdapter.construct_fresh(
        _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
    )
    # The focused fake intentionally exposes npluginstate only.  Once strict
    # mode is enabled, that missing nplugin counter must block the audit.
    adapter.config["strict_provenance"] = True
    with pytest.raises(replay.M1RuntimeError, match="nplugin"):
        adapter._runtime_audit()


def test_nonzero_sleeping_island_state_is_a_closed_loop_block() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.sim.data.tree_asleep = np.asarray([0, 1], dtype=np.int32)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.M1RuntimeError, match="sleep|island"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("use_ori", 123456),
        ("initial_joint", "not-a-joint-vector"),
    ],
)
def test_malformed_controller_scalar_or_serialized_shape_fails_closed(
    field_name: str, value: Any
) -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        setattr(child.controller, field_name, value)
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.UnknownMutableStateError, match=field_name):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_unproven_recent_buffer_content_is_not_proven_unused() -> None:
    replay = _module()
    events: list[str] = []

    def builder(config: Mapping[str, Any]) -> dict[str, Any]:
        child = _FakeEnv(events)
        child.robots[0].recent_actions = {"arbitrary": {"mutable": True}}
        return {"env": _FakeVector(child), "mujoco": _FakeMujoco(child.sim.data, events)}

    with pytest.raises(replay.UnknownMutableStateError, match="recent_actions"):
        replay.RuntimeAdapter.construct_fresh(
            _fake_config(Path("/tmp")), runtime_builder=builder, tape_hash="tape"
        )


def test_tagged_mappings_and_duck_typed_rng_payload_round_trip_exactly() -> None:
    replay = _module()
    mapping = {
        "__m1_type__": "ndarray",
        "dtype": "float64",
        "shape": [2],
        "data": [1.0, 2.0],
    }
    encoded = replay._snapshot_serialized_value(mapping)
    assert encoded["__m1_type__"] == "mapping"
    assert replay._exact_equal(replay._restore_snapshot_value(encoded), mapping)

    class DuckRng:
        def __init__(self) -> None:
            self.state = {
                "__m1_type__": "tuple",
                "items": ["cursor", {"value": 3}],
            }

        def get_state(self) -> Any:
            return self.state

        def set_state(self, state: Any) -> None:
            self.state = state

    rng = DuckRng()
    rng_payload = replay._snapshot_rng_object(rng)
    assert rng_payload["state"]["__m1_type__"] == "mapping"
    rng.state = None
    replay._restore_rng_object(rng, rng_payload)
    assert replay._exact_equal(rng.state, {"__m1_type__": "tuple", "items": ["cursor", {"value": 3}]})


def test_worktree_allowed_output_root_is_classified_and_retained(tmp_path: Path) -> None:
    replay = _module()
    output_root = tmp_path / "runs" / "m1_state_replay"
    output_file = output_root / "20260831" / "capture.json"
    output_file.parent.mkdir(parents=True)
    output_file.write_text("{}\n", encoding="utf-8")
    config = {
        "worktree_audit": {
            "timing": "pre_output",
            "allowed_output_root": str(output_root),
            "implementation_paths": (),
        },
        "allowed_dirty_evidence": {},
    }
    # The helper uses the project root for status, so exercise the pure path
    # classifier against a synthetic status entry after resolving the contract.
    original = replay._git_command

    def fake_git(path: Path, *args: str) -> str | None:
        if args == ("rev-parse", "HEAD"):
            return "head"
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return f"?? {output_file}"
        if args == ("symbolic-ref", "-q", "--short", "HEAD"):
            return "branch"
        return original(path, *args)

    # The output path is outside the repository in this focused seam; its
    # configured absolute path is still retained as allowed output evidence.
    config["worktree_audit"]["allowed_output_root"] = str(output_root)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(replay, "_git_command", fake_git)
        audit = replay._worktree_audit(config)
    assert audit["other_dirt"] == []
    assert audit["allowed_output"]["status"]


def test_physics_and_observation_model_fingerprints_have_independent_exact_gates() -> None:
    replay = _module()

    class Options:
        timestep = 0.002
        gravity = np.asarray([0.0, 0.0, -9.81], dtype=np.float64)

    class CompiledModel:
        nq = 2
        nv = 2
        njnt = 1
        nbody = 2
        ngeom = 1
        nsite = 1
        ncam = 1
        nlight = 1
        nactuator = 1
        neq = 0
        ntendon = 0
        nplugin = 0
        npluginstate = 0
        joint_names = ["hinge"]
        body_names = ["world", "arm"]
        geom_names = ["arm_geom"]
        body_pos = np.zeros((2, 3), dtype=np.float64)
        body_quat = np.ones((2, 4), dtype=np.float64)
        geom_size = np.ones((1, 3), dtype=np.float64)
        geom_rgba = np.ones((1, 4), dtype=np.float32)
        jnt_type = np.asarray([3], dtype=np.int32)
        dof_damping = np.asarray([0.2, 0.3], dtype=np.float64)
        actuator_gear = np.ones((1, 6), dtype=np.float64)
        cam_pos = np.zeros((1, 3), dtype=np.float64)
        light_pos = np.zeros((1, 3), dtype=np.float64)
        site_rgba = np.ones((1, 4), dtype=np.float32)
        opt = Options()

    first = CompiledModel()
    second = CompiledModel()
    physics_a = replay.physics_model_fingerprint(first)
    observation_a = replay.observation_model_fingerprint(first, {"width": 64, "height": 64})
    assert physics_a["hash"] == replay.physics_model_fingerprint(second)["hash"]
    assert observation_a["hash"] == replay.observation_model_fingerprint(
        second, {"width": 64, "height": 64}
    )["hash"]
    assert {"dtype", "shape", "sha256"} <= set(physics_a["fields"]["body_pos"])

    second.body_pos[0, 0] += 1.0
    physics_b = replay.physics_model_fingerprint(second)
    observation_b = replay.observation_model_fingerprint(second, {"width": 64, "height": 64})
    assert physics_a["hash"] != physics_b["hash"]
    assert observation_a["hash"] == observation_b["hash"]
    assert replay.exact_model_fingerprint_equal(physics_a, physics_b) is False

    second.body_pos[0, 0] -= 1.0
    second.site_rgba[0, 0] = 0.25
    observation_c = replay.observation_model_fingerprint(second, {"width": 64, "height": 64})
    assert replay.physics_model_fingerprint(second)["hash"] == physics_a["hash"]
    assert observation_a["hash"] != observation_c["hash"]


def test_replay_state_v2_round_trips_python_rng_fingerprints_raw_observation_task_and_tape(
    tmp_path: Path,
) -> None:
    replay = _module()
    state = replay.ReplayState(
        integration_state=np.asarray([1.0, 2.0], dtype=np.float64),
        fixture={},
        counters={"timestep": 2},
        controller={},
        gripper={},
        identity={},
        python_state={"persistent": ("cache", np.asarray([3], dtype=np.int16))},
        rng={"numpy": {"seed": 7}},
        physics_model_fingerprint={"hash": "p", "fields": {}},
        observation_model_fingerprint={"hash": "o", "fields": {}},
        raw_observation={"image": np.asarray([[1, 2]], dtype=np.uint8)},
        task_state={"task_id": 0, "discrete": "open"},
        tape_provenance={"sha256": "tape", "source_step": 2},
    )
    assert state.schema_version == 2
    record = replay.replay_state_record(state)
    assert record.metadata["schema_version"] == 2
    replay.save_record_bundle(record, tmp_path, "capture")
    restored = replay.replay_state_from_record(replay.load_record_bundle(tmp_path, "capture"))
    assert restored.schema_version == 2
    assert replay._exact_equal(restored.python_state, state.python_state)
    assert replay._exact_equal(restored.rng, state.rng)
    assert replay._exact_equal(restored.raw_observation, state.raw_observation)
    assert restored.physics_model_fingerprint == state.physics_model_fingerprint
    assert restored.observation_model_fingerprint == state.observation_model_fingerprint
    assert restored.task_state == state.task_state
    assert restored.tape_provenance == state.tape_provenance


def test_strict_restore_never_writes_compiled_physics_arrays_and_uses_no_update_observation_read() -> None:
    replay = _module()
    events: list[str] = []
    builder, _ = _fake_runtime_builder_factory(events)
    config = _fake_config(Path("/tmp"))
    source = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    source.step(np.zeros(7, dtype=np.float32))
    state = replace(source.capture(source_step=1), raw_observation={"step": 1}, schema_version=2)

    fresh = replay.RuntimeAdapter.construct_fresh(config, runtime_builder=builder, tape_hash="tape")
    fresh.config["strict_provenance"] = True
    fresh.post_process_proof = {"passed": True}

    class WriteTrap(np.ndarray):
        writes = 0

        def __setitem__(self, key: Any, value: Any) -> None:
            type(self).writes += 1
            raise AssertionError("compiled physics model array was written")

    fresh.model.body_pos = np.asarray(fresh.model.body_pos).view(WriteTrap)
    fresh.model.body_quat = np.asarray(fresh.model.body_quat).view(WriteTrap)
    fresh.inner._get_observations = lambda *, force_update=False: events.append(
        f"get_observations:{force_update}"
    ) or {"step": 1}
    fresh.env._format_raw_obs = lambda value: events.append("format_raw_obs") or value
    fresh.collect_invariants = lambda: {}  # type: ignore[method-assign]

    result = fresh.restore(state)
    assert WriteTrap.writes == 0
    assert "refresh" not in events
    assert events.index("get_observations:False") < events.index("format_raw_obs")
    assert result["raw_observation_comparison"]["passed"] is True


def test_build_official_runtime_delegates_to_environment_only_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    replay = _module()
    from scripts import dcu_preflight

    calls: list[tuple[Mapping[str, Any], str]] = []
    sentinel = {"env": object(), "mujoco": object()}

    def environment_only(config: Mapping[str, Any], *, phase: str = "compare") -> Mapping[str, Any]:
        calls.append((config, phase))
        return sentinel

    monkeypatch.setattr(dcu_preflight, "build_cpu_environment_runtime", environment_only)
    config = _fake_config(Path("/tmp"))
    assert replay.build_official_runtime(config) is sentinel
    assert calls == [(config, "compare")]


def test_environment_only_factory_has_no_smolvla_or_policy_processor_imports() -> None:
    """Keep the M1 environment boundary free of policy construction/imports."""

    from scripts import dcu_preflight

    source = inspect.getsource(dcu_preflight.build_cpu_environment_runtime)
    tree = ast.parse(source)
    imported = {
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    called = {
        node.func.id.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any("smolvla" in name or ".policies" in name for name in imported)
    assert not {"make_policy", "make_pre_post_processors"} & called
