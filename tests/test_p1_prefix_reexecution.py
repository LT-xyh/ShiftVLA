from __future__ import annotations

from collections import deque
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.dcu_model_worker import DCUModelWorker, FEATURE_SCHEMA, NOISE_SCHEMA
from scripts.dcu_preflight import FeatureOnlyRemotePolicy
from scripts.dcu_worker import DCUWorkerClient, RESPONSE_SCHEMA, save_tensor_bundle
from scripts.p1_prefix_reexecution import (
    ARMS,
    FROZEN_SWITCH_INDEX,
    G_P2_SOURCE_TRACE,
    PairedNoiseSelectActionPolicy,
    TechnicalMatchingFailure,
    absorbing_terminal_evidence,
    audit_same_prefix,
    build_pilot_schedule,
    camera_mode_for_observation,
    camera_request_before_step,
    camera_request_for_initial_observation,
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


def test_obs0_request_precedes_reset_and_obs50_request_precedes_step49_in_fake_lifecycle() -> None:
    events = []
    initial = camera_request_for_initial_observation("CS")
    events.append(("camera", initial))
    events.append(("reset", None))
    before_49 = camera_request_before_step("CS", 48)
    events.append(("camera", before_49))
    events.append(("step", 48))
    before_50 = camera_request_before_step("CS", 49)
    events.append(("camera", before_50))
    events.append(("step", 49))

    assert events[0][0] == "camera" and events[1][0] == "reset"
    assert initial == {
        "observation_index": 0,
        "preceding_action_index": None,
        "requested_camera_mode": "clean",
    }
    assert before_49["observation_index"] == 49
    assert before_49["requested_camera_mode"] == "clean"
    assert before_50 == {
        "observation_index": 50,
        "preceding_action_index": 49,
        "requested_camera_mode": "shifted",
    }
    assert events.index(("camera", before_50)) < events.index(("step", 49))
    assert [event[0] for event in events].count("reset") == 1
    assert [event[0] for event in events].count("step") == 2


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


def test_g_p2_source_trace_is_blocked_instead_of_inventing_runtime_camera_infrastructure() -> None:
    assert G_P2_SOURCE_TRACE["status"] == "BLOCKED"
    fields = G_P2_SOURCE_TRACE["agentview_model_fields"]
    assert fields["name"] == "agentview"
    assert fields["position"] == [0.5886131746834771, 0.0, 1.4903500240372423]
    assert len(fields["quaternion_wxyz"]) == 4
    assert G_P2_SOURCE_TRACE["runtime_mutation_api"] is None
    assert G_P2_SOURCE_TRACE["yaw_mapping"]["status"] == "BLOCKED"
    assert "hard_reset=True" in G_P2_SOURCE_TRACE["reset_path"]
    assert "env.step(action_49)" in G_P2_SOURCE_TRACE["step_observation_path"]
    assert "CameraMover" in G_P2_SOURCE_TRACE["blocked_reason"]


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
