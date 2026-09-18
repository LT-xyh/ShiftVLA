from __future__ import annotations

from collections import deque
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.dcu_model_worker import DCUModelWorker, FEATURE_SCHEMA, NOISE_SCHEMA
from scripts.dcu_preflight import FeatureOnlyRemotePolicy
from scripts.dcu_worker import DCUWorkerClient, RESPONSE_SCHEMA, save_tensor_bundle
from scripts.p1_prefix_reexecution import (
    ARMS,
    FROZEN_ENVIRONMENT_SEED,
    FROZEN_SWITCH_INDEX,
    G_P2_SOURCE_TRACE,
    AgentviewYawController,
    ObservationIndexedCameraWrapper,
    PairedNoiseSelectActionPolicy,
    TechnicalMatchingFailure,
    absorbing_terminal_evidence,
    audit_same_prefix,
    build_pilot_schedule,
    camera_mode_for_observation,
    camera_request_before_step,
    camera_request_for_initial_observation,
    derive_world_z_yaw_wxyz,
    generate_paired_flow_noise,
    load_pilot_config,
    paired_noise_spec,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "replayvla" / "p1_prefix_reexecution_pilot.yaml"


def _features(state_value: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.full((1, 8), state_value, dtype=torch.float32),
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.images.image2": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.language.tokens": torch.ones((1, 4), dtype=torch.int64),
        "observation.language.attention_mask": torch.ones((1, 4), dtype=torch.bool),
    }


def test_frozen_pilot_schedule_is_exactly_8_roots_32_main_8_duplicates() -> None:
    config = load_pilot_config(CONFIG)
    rows = build_pilot_schedule(config)
    assert len(rows) == 40
    assert len({row["root_id"] for row in rows}) == 8
    assert sum(not row["clean_duplicate"] for row in rows) == 32
    assert sum(row["clean_duplicate"] for row in rows) == 8
    assert {row["arm"] for row in rows if not row["clean_duplicate"]} == set(ARMS)
    assert config["environment_seed"] == FROZEN_ENVIRONMENT_SEED == 2027
    assert all(row["environment_seed"] == 2027 for row in rows)
    assert all(row["retry"] == 0 and row["replacement"] == 0 for row in rows)


def test_paired_noise_key_excludes_arm_and_is_deterministic_by_root_and_index() -> None:
    first = paired_noise_spec("root-0", 17)
    second = paired_noise_spec("root-0", 17)
    next_index = paired_noise_spec("root-0", 18)
    assert first == second
    assert first.key == second.key
    assert first.seed == second.seed
    assert first.key != next_index.key
    assert first.seed != next_index.seed
    assert all(arm not in first.key for arm in ARMS)


def test_same_root_index_produces_same_noise_bytes_and_hash() -> None:
    first, first_evidence = generate_paired_flow_noise("root-0", 7)
    second, second_evidence = generate_paired_flow_noise("root-0", 7)
    assert torch.equal(first, second)
    assert first_evidence["noise_sha256"] == second_evidence["noise_sha256"]
    assert first_evidence["noise_seed"] == second_evidence["noise_seed"]


def test_paired_noise_wrapper_keeps_action_branch_local() -> None:
    noise = torch.full((1, 50, 32), 0.25, dtype=torch.float32)
    evidence = {
        "root_id": "root",
        "observation_action_index": 0,
        "draw_kind": "flow",
        "draw_slot": 0,
        "noise_key": "key",
        "noise_seed": 1,
        "noise_sha256": "a" * 64,
    }

    class Delegate:
        def reset(self) -> None:
            return None

        def select_action(self, batch, *, noise):
            value = batch["observation.state"][0, 0] + noise[0, 0, 0]
            return torch.full((1, 7), float(value), dtype=torch.float32)

    factory = lambda _root, _index: (noise.clone(), dict(evidence))
    clean = PairedNoiseSelectActionPolicy(Delegate(), root_id="root", noise_factory=factory)
    shifted = PairedNoiseSelectActionPolicy(Delegate(), root_id="root", noise_factory=factory)
    clean_action = clean.select_action(_features(0.0))
    shifted_action = shifted.select_action(_features(1.0))
    assert clean.noise_evidence[0]["noise_sha256"] == shifted.noise_evidence[0]["noise_sha256"]
    assert not torch.equal(clean_action, shifted_action)


def test_remote_select_action_optionally_transports_explicit_noise_without_breaking_legacy() -> None:
    calls = []
    noise_bundles = []

    class Client:
        last_queue_evidence = {
            "queue_length_before": 0,
            "queue_length_after": 0,
            "new_chunk_generated": True,
        }

        def select_action(self, request_path: Path, *, noise_path: Path | None = None) -> torch.Tensor:
            calls.append((request_path, noise_path))
            return torch.zeros((1, 7), dtype=torch.float32)

    policy = FeatureOnlyRemotePolicy(
        Client(),
        request_writer=lambda bundle: Path(bundle["path"]),
        noise_writer=lambda bundle: noise_bundles.append(dict(bundle)) or Path("/tmp/noise.safetensors"),
    )
    policy.select_action({"path": "/tmp/features.safetensors"})
    explicit = torch.zeros((1, 50, 32), dtype=torch.float32)
    policy.select_action({"path": "/tmp/features.safetensors"}, noise=explicit)
    assert calls[0] == (Path("/tmp/features.safetensors"), None)
    assert calls[1] == (Path("/tmp/features.safetensors"), Path("/tmp/noise.safetensors"))
    assert torch.equal(noise_bundles[0]["noise"], explicit)


class _FakeDevice:
    def is_available(self, device: str) -> bool:
        return device == "cuda:0"

    def device_count(self) -> int:
        return 1

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "K100_AI"

    def to_device(self, tensor: torch.Tensor, _device: str) -> torch.Tensor:
        return tensor

    def synchronize(self) -> None:
        return None

    def reset_peak_memory_stats(self) -> None:
        return None

    def max_memory_allocated(self) -> int:
        return 0

    def manual_seed_all(self, _seed: int) -> None:
        return None

    def basic_tensor_ops(self):
        return {
            "fp32": {"shape": [2, 2], "dtype": "torch.float32", "finite": True, "device": "cuda:0"},
            "bf16": {"shape": [2, 2], "dtype": "torch.bfloat16", "finite": True, "device": "cuda:0"},
        }


class _SelectOnlyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, n_action_steps=1)
        self._queues = {"action": deque(maxlen=1)}
        self.calls = []
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def reset(self) -> None:
        self._queues["action"].clear()

    def predict_action_chunk(self, *_args, **_kwargs):
        raise AssertionError("explicit-noise select path must not call predict_action_chunk")

    def select_action(self, batch, **kwargs):
        self.calls.append((batch, dict(kwargs)))
        if not self._queues["action"]:
            noise = kwargs.get("noise")
            base = batch["observation.state"][0, 0]
            delta = torch.tensor(0.0) if noise is None else noise[0, 0, 0]
            chunk = torch.full((1, 50, 7), float(base + delta), dtype=torch.float32)
            self._queues["action"].extend(chunk.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues["action"].popleft()


def test_worker_calls_official_select_action_with_explicit_noise_and_n1_queue(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.safetensors"
    noise_path = tmp_path / "noise.safetensors"
    save_tensor_bundle(feature_path, _features(0.5), schema=FEATURE_SCHEMA)
    noise = torch.full((1, 50, 32), 0.25, dtype=torch.float32)
    save_tensor_bundle(noise_path, {"noise": noise}, schema=NOISE_SCHEMA)

    policy = _SelectOnlyPolicy()
    worker = DCUModelWorker(policy=policy, device_adapter=_FakeDevice(), seed=2027)
    worker.reset()
    first = worker.select_action(
        {"request_path": str(feature_path), "noise_path": str(noise_path)},
        response_path=tmp_path / "action-0.safetensors",
    )
    second = worker.select_action(
        {"request_path": str(feature_path), "noise_path": str(noise_path)},
        response_path=tmp_path / "action-1.safetensors",
    )
    assert len(policy.calls) == 2
    assert torch.equal(policy.calls[0][1]["noise"], noise)
    assert torch.equal(policy.calls[1][1]["noise"], noise)
    assert first["queue_length_before"] == 0 and first["queue_length_after"] == 0
    assert second["queue_length_before"] == 0 and second["queue_length_after"] == 0
    assert first["new_chunk_generated"] is True and second["new_chunk_generated"] is True


def test_worker_historical_no_noise_select_path_is_preserved(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.safetensors"
    save_tensor_bundle(feature_path, _features(), schema=FEATURE_SCHEMA)
    policy = _SelectOnlyPolicy()
    worker = DCUModelWorker(policy=policy, device_adapter=_FakeDevice(), seed=2027)
    worker.reset()
    worker.select_action(
        {"request_path": str(feature_path)},
        response_path=tmp_path / "action.safetensors",
    )
    assert policy.calls[0][1] == {}


def test_dcu_worker_client_forwards_optional_noise_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    class Transport:
        def request(self, command: str, **payload):
            requests.append((command, dict(payload)))
            if command == "reset":
                return {"ok": True, "queue_length_before": 0, "queue_length_after": 0, "queue_length": 0}
            response_path = tmp_path / "action.safetensors"
            save_tensor_bundle(
                response_path,
                {"action": torch.zeros((1, 7), dtype=torch.float32)},
                schema=RESPONSE_SCHEMA,
            )
            return {
                "ok": True,
                "response_path": str(response_path),
                "queue_length_before": 0,
                "queue_length_after": 0,
                "new_chunk_generated": True,
            }

        def close(self) -> None:
            return None

    client = DCUWorkerClient(transport=Transport(), n_action_steps=1)
    client.reset(seed=2027)
    client.select_action(tmp_path / "features.safetensors", noise_path=tmp_path / "noise.safetensors")
    assert requests[-1] == (
        "select_action",
        {
            "request_path": str(tmp_path / "features.safetensors"),
            "noise_path": str(tmp_path / "noise.safetensors"),
        },
    )


@pytest.mark.parametrize(
    ("arm", "prefix", "future"),
    [
        ("CC", "clean", "clean"),
        ("CS", "clean", "shifted"),
        ("SC", "shifted", "clean"),
        ("SS", "shifted", "shifted"),
    ],
)
def test_four_arm_observation_camera_schedule_is_exact(arm: str, prefix: str, future: str) -> None:
    assert all(camera_mode_for_observation(arm, index) == prefix for index in range(0, 50))
    assert all(camera_mode_for_observation(arm, index) == future for index in (50, 51, 279))


def test_same_prefix_audit_is_fail_closed() -> None:
    base = [
        {
            "query_index": index,
            "noise_sha256": f"noise-{index}",
            "camera_mode": "clean",
            "action": [float(index)] * 7,
            "terminal_state": False,
        }
        for index in range(3)
    ]
    assert audit_same_prefix("CC", "CS", base, [dict(row) for row in base])["status"] == "PASS"
    mismatched = [dict(row) for row in base]
    mismatched[1]["action"] = [999.0] * 7
    with pytest.raises(TechnicalMatchingFailure, match="action"):
        audit_same_prefix("CC", "CS", base, mismatched)


def test_absorbing_pre_switch_terminal_disallows_continuation_retry_and_replacement() -> None:
    row = build_pilot_schedule(load_pilot_config(CONFIG))[0]
    evidence = absorbing_terminal_evidence(
        row,
        observation_action_index=12,
        terminal_state={"terminated": True, "success": False},
    )
    assert evidence["absorbing"] is True
    assert evidence["continuation_authorized"] is False
    assert evidence["retry"] == 0
    assert evidence["replacement"] == 0


def test_prefix_module_has_no_exact_state_route_dependency_and_no_manual_chunk_selection() -> None:
    import scripts.p1_prefix_reexecution as module

    source = inspect.getsource(module)
    forbidden = ("m1_" + "state_replay", "f3" + "n", "f3" + "b")
    assert all(name.lower() not in source.lower() for name in forbidden)
    wrapper_source = inspect.getsource(PairedNoiseSelectActionPolicy.select_action)
    assert "predict_" + "action_chunk" not in wrapper_source
    assert ".select_action(" in wrapper_source



def test_paired_noise_wrapper_does_not_own_or_reorder_processors() -> None:
    source = inspect.getsource(PairedNoiseSelectActionPolicy.select_action)
    assert "preprocessor" not in source
    assert "postprocessor" not in source
    assert "env.step" not in source
    assert source.count(".select_action(") == 1



class _FakeTransforms:
    @staticmethod
    def convert_quat(q, to="xyzw"):
        q = np.asarray(q, dtype=np.float64)
        if to == "xyzw":
            return q[[1, 2, 3, 0]]
        if to == "wxyz":
            return q[[3, 0, 1, 2]]
        raise ValueError(to)

    @staticmethod
    def quat2mat(q):
        x, y, z, w = np.asarray(q, dtype=np.float64)
        n = x * x + y * y + z * z + w * w
        s = 2.0 / n
        return np.asarray(
            [
                [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def mat2quat(matrix):
        m = np.asarray(matrix, dtype=np.float64)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return np.asarray([x, y, z, w], dtype=np.float64)


_CLEAN_WXYZ = np.asarray(
    [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
    dtype=np.float64,
)
_CLEAN_WXYZ = _CLEAN_WXYZ / np.linalg.norm(_CLEAN_WXYZ)


def test_world_z_yaw_math_is_normalized_deterministic_and_exactly_plus_15_degrees() -> None:
    assert np.isclose(np.linalg.norm(_CLEAN_WXYZ), 1.0, atol=1e-12)
    first = derive_world_z_yaw_wxyz(_CLEAN_WXYZ, transform_utils=_FakeTransforms)
    second = derive_world_z_yaw_wxyz(_CLEAN_WXYZ, transform_utils=_FakeTransforms)
    assert np.isclose(np.linalg.norm(first), 1.0, atol=1e-12)
    assert np.allclose(first, second, atol=0.0, rtol=0.0)

    clean_rot = _FakeTransforms.quat2mat(_FakeTransforms.convert_quat(_CLEAN_WXYZ, to="xyzw"))
    shifted_rot = _FakeTransforms.quat2mat(_FakeTransforms.convert_quat(first, to="xyzw"))
    theta = math.radians(15.0)
    expected_yaw = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(shifted_rot, expected_yaw @ clean_rot, atol=1e-7, rtol=0.0)


class _FakeSim:
    def __init__(self, *, tag):
        self.tag = tag
        self.model = SimpleNamespace(
            cam_quat=np.asarray([_CLEAN_WXYZ.copy()]),
            cam_pos=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            cam_fovy=np.asarray([45.0], dtype=np.float64),
            object_sentinel=np.asarray([11.0, 12.0]),
            dynamics_sentinel=np.asarray([21.0, 22.0]),
        )
        self.data = SimpleNamespace(
            qpos=np.asarray([1.0, 2.0]),
            qvel=np.asarray([3.0, 4.0]),
            ctrl=np.asarray([5.0, 6.0]),
        )


class _FakeCameraModder:
    def __init__(self, sim, events=None):
        self.sim = sim
        self.events = [] if events is None else events
        self.update_calls = []

    @property
    def model(self):
        return self.sim.model

    def update_sim(self, sim):
        self.events.append(("rebind", sim.tag))
        self.update_calls.append(sim)
        self.sim = sim

    def get_quat(self, name):
        assert name == "agentview"
        return self.model.cam_quat[0]

    def set_quat(self, name, value):
        assert name == "agentview"
        self.events.append(("set_quat", self.sim.tag, tuple(np.asarray(value))))
        self.model.cam_quat[0] = np.asarray(value)

    def get_pos(self, name):
        assert name == "agentview"
        return self.model.cam_pos[0]

    def get_fovy(self, name):
        assert name == "agentview"
        return self.model.cam_fovy[0]


def _fake_controller(events=None):
    holder = {}

    def factory(sim):
        modder = _FakeCameraModder(sim, events=events)
        holder["modder"] = modder
        return modder

    controller = AgentviewYawController(
        modder_factory=factory,
        transform_utils=_FakeTransforms,
    )
    return controller, holder


def test_camera_controller_changes_only_cam_quat_and_restores_clean_identity() -> None:
    sim = _FakeSim(tag="sim0")
    controller, _ = _fake_controller()
    controller.bind(sim)
    sentinels = {
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "ctrl": sim.data.ctrl.copy(),
        "object": sim.model.object_sentinel.copy(),
        "dynamics": sim.model.dynamics_sentinel.copy(),
        "pos": sim.model.cam_pos.copy(),
        "fovy": sim.model.cam_fovy.copy(),
    }
    clean = sim.model.cam_quat.copy()
    controller.apply("shifted")
    shifted = sim.model.cam_quat.copy()
    assert not np.allclose(shifted, clean)
    controller.apply("clean")
    assert np.array_equal(sim.model.cam_quat, clean)
    assert np.array_equal(sim.data.qpos, sentinels["qpos"])
    assert np.array_equal(sim.data.qvel, sentinels["qvel"])
    assert np.array_equal(sim.data.ctrl, sentinels["ctrl"])
    assert np.array_equal(sim.model.object_sentinel, sentinels["object"])
    assert np.array_equal(sim.model.dynamics_sentinel, sentinels["dynamics"])
    assert np.array_equal(sim.model.cam_pos, sentinels["pos"])
    assert np.array_equal(sim.model.cam_fovy, sentinels["fovy"])


class _FakeTaskEnv:
    def __init__(self, sim, events):
        self.sim = sim
        self.events = events
        self.observe_count = 0

    def _get_observations(self, force_update=False):
        self.observe_count += 1
        self.events.append(("observe", self.sim.tag, bool(force_update)))
        return {"camera_quat": self.sim.model.cam_quat[0].copy()}


class _FakeOffscreen:
    def __init__(self, task_env):
        self.env = task_env


class _FakeLeRobotLiberoEnv:
    def __init__(self, events):
        self.events = events
        self.reset_count = 0
        self.step_count = 0
        self.settle_dummy_steps = 0
        self._env = _FakeOffscreen(_FakeTaskEnv(_FakeSim(tag="initial"), events))

    def _format_raw_obs(self, raw_obs):
        return {"policy_camera_quat": np.asarray(raw_obs["camera_quat"]).copy()}

    def reset(self, *, seed):
        self.reset_count += 1
        self.events.append(("delegate_reset", seed))
        current = _FakeSim(tag=f"reset-{self.reset_count}")
        self._env = _FakeOffscreen(_FakeTaskEnv(current, self.events))
        self.settle_dummy_steps += 10
        self.events.append(("settle_dummy_steps", 10))
        raw = self._env.env._get_observations(force_update=True)
        return self._format_raw_obs(raw), {"seed": seed}

    def step(self, action):
        self.step_count += 1
        self.events.append(("delegate_step", self.step_count - 1, action))
        raw = self._env.env._get_observations(force_update=True)
        return self._format_raw_obs(raw), 0.0, False, False, {}


def test_reset_calls_delegate_once_rebinds_current_sim_and_returns_requested_camera_obs0() -> None:
    events = []
    env = _FakeLeRobotLiberoEnv(events)
    old_sim = env._env.env.sim
    controller, holder = _fake_controller(events)
    wrapper = ObservationIndexedCameraWrapper(env, arm="SC", controller=controller)

    observation, info = wrapper.reset(seed=2027)
    assert env.reset_count == 1
    assert env.step_count == 0
    assert info["seed"] == 2027
    assert controller.sim is env._env.env.sim
    assert controller.sim is not old_sim
    assert np.allclose(
        observation["policy_camera_quat"],
        controller.shifted_quaternion_wxyz,
        atol=1e-7,
        rtol=0.0,
    )
    assert events.index(("delegate_reset", 2027)) < next(
        index for index, event in enumerate(events) if event[0] == "set_quat"
    )
    assert holder["modder"].sim is env._env.env.sim


def test_hard_reset_rebinds_existing_modder_to_new_current_sim() -> None:
    events = []
    env = _FakeLeRobotLiberoEnv(events)
    controller, holder = _fake_controller(events)
    wrapper = ObservationIndexedCameraWrapper(env, arm="CC", controller=controller)
    wrapper.reset(seed=2027)
    first_sim = controller.sim
    wrapper.reset(seed=2027)
    assert controller.sim is env._env.env.sim
    assert controller.sim is not first_sim
    assert holder["modder"].update_calls[-1] is controller.sim


def test_step49_sets_future_camera_before_exactly_one_underlying_step() -> None:
    events = []
    env = _FakeLeRobotLiberoEnv(events)
    controller, _ = _fake_controller(events)
    wrapper = ObservationIndexedCameraWrapper(env, arm="CS", controller=controller)
    wrapper.reset(seed=2027)

    for index in range(49):
        wrapper.step(f"a{index}")
    step_count_before = env.step_count
    event_count_before = len(events)
    wrapper.step("a49")
    recent = events[event_count_before:]
    assert env.step_count == step_count_before + 1
    set_index = next(i for i, event in enumerate(recent) if event[0] == "set_quat")
    step_index = next(i for i, event in enumerate(recent) if event[0] == "delegate_step")
    assert set_index < step_index
    assert recent[step_index] == ("delegate_step", 49, "a49")
    assert wrapper.camera_evidence_rows[-1]["observation_index"] == 50
    assert wrapper.camera_evidence_rows[-1]["preceding_action_index"] == 49
    assert wrapper.camera_evidence_rows[-1]["requested_camera_mode"] == "shifted"


def test_shifted_prefix_restores_clean_quaternion_before_step49() -> None:
    events = []
    env = _FakeLeRobotLiberoEnv(events)
    controller, _ = _fake_controller(events)
    wrapper = ObservationIndexedCameraWrapper(env, arm="SC", controller=controller)
    wrapper.reset(seed=2027)
    clean = controller.clean_quaternion_wxyz.copy()
    assert not np.allclose(env._env.env.sim.model.cam_quat[0], clean)
    for index in range(50):
        wrapper.step(f"a{index}")
    assert np.allclose(env._env.env.sim.model.cam_quat[0], clean, atol=1e-12, rtol=0.0)


def test_settle_dummy_steps_do_not_advance_policy_noise_query_index() -> None:
    events = []
    env = _FakeLeRobotLiberoEnv(events)
    controller, _ = _fake_controller(events)
    wrapper = ObservationIndexedCameraWrapper(env, arm="SS", controller=controller)

    class DelegatePolicy:
        def reset(self):
            return None

        def select_action(self, _batch, *, noise):
            assert noise is not None
            return torch.zeros((1, 7))

    noise = torch.zeros((1, 50, 32))
    evidence = {
        "root_id": "root",
        "observation_action_index": 0,
        "draw_kind": "flow",
        "draw_slot": 0,
        "noise_key": "key",
        "noise_seed": 1,
        "noise_sha256": "a" * 64,
    }
    policy = PairedNoiseSelectActionPolicy(
        DelegatePolicy(),
        root_id="root",
        noise_factory=lambda _root, index: (
            noise,
            {**evidence, "observation_action_index": index},
        ),
    )
    wrapper.reset(seed=2027)
    assert env.settle_dummy_steps == 10
    assert policy.query_index == 0
    policy.select_action({"observation": "obs0"})
    assert policy.query_index == 1


def test_g_p2_source_trace_selects_camera_modder_and_keeps_runtime_validation_pending() -> None:
    assert G_P2_SOURCE_TRACE["status"] == "IMPLEMENTED_REPO_ONLY_NOT_RUNTIME_VALIDATED"
    assert G_P2_SOURCE_TRACE["runtime_mutation_api"] == "robosuite.utils.mjmod.CameraModder.set_quat"
    assert G_P2_SOURCE_TRACE["CameraModder"] == "SELECTED_RUNTIME_MODEL_CAMERA_ROUTE"
    assert G_P2_SOURCE_TRACE["CameraMover"].startswith("REJECTED")
    assert G_P2_SOURCE_TRACE["physics_isolation_evidence"]["real_runtime"] == "NOT_VALIDATED_NOT_AUTHORIZED"


def test_camera_controller_source_uses_only_set_quat_camera_mutation_seam() -> None:
    source = inspect.getsource(AgentviewYawController)
    apply_source = inspect.getsource(AgentviewYawController.apply)
    module_source = inspect.getsource(__import__("scripts.p1_prefix_reexecution", fromlist=["*"]))
    assert ".set_quat(" in apply_source
    assert ".set_pos(" not in source
    assert ".set_fovy(" not in source
    assert "CameraMover(" not in module_source
    assert "camera_utils" not in module_source
    assert "m1_" + "state_replay" not in module_source
