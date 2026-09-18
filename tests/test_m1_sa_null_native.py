import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from scripts import m1_sa_null_native as native


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/replayvla/p1_sa_null_native.yaml"


def load_config():
    return native._load_yaml(CONFIG)


def test_native_schema_does_not_require_legacy_source_checkouts():
    config = load_config()
    native.validate_native_config(config, check_files=False)
    assert "source_checkouts" not in config
    assert config["runtime_contract"] == native.NATIVE_CONTRACT


@pytest.mark.parametrize("role", ["external/lerobot", "external/robosuite", "external/mujoco"])
def test_absent_legacy_checkout_is_not_a_schema_gate(role):
    config = load_config()
    payload = json.dumps(config, sort_keys=True)
    assert role not in payload
    native.validate_native_config(config, check_files=False)


def test_legacy_state_replay_reference_is_rejected():
    config = load_config()
    config["state_replay_config"] = {"path": "configs/m1/state_replay.yaml"}
    with pytest.raises(native.NativeQualificationError):
        native.validate_native_config(config, check_files=False)


def test_legacy_libero_config_path_is_rejected():
    config = load_config()
    config["runtime"]["environment"]["LIBERO_CONFIG_PATH"] = native.LEGACY_LIBERO_CONFIG
    with pytest.raises(native.NativeQualificationError):
        native.validate_native_config(config, check_files=False)


def test_policy_processor_and_retry_must_remain_disabled():
    for key, bad in [
        ("include_policy", True),
        ("call_policy", True),
        ("call_processors", True),
        ("retry_count", 1),
        ("fresh_processes", False),
    ]:
        config = load_config()
        config["runtime"][key] = bad
        with pytest.raises(native.NativeQualificationError):
            native.validate_native_config(config, check_files=False)


def test_schedule_is_exactly_twenty_pairs():
    config = load_config()
    native.validate_native_config(config, check_files=False)
    assert config["pair_count"] == 20
    assert config["pair_prefix"] == "m1n0-pair-"


def test_runtime_environment_binds_native_libero_config():
    config = load_config()
    native.validate_native_config(config, check_files=False)
    expected = str((ROOT / config["libero_config"]["path"]).resolve().parent)
    actual = str(Path(config["runtime"]["environment"]["LIBERO_CONFIG_PATH"]).resolve())
    assert actual == expected


def test_fail_fast_runner_launches_only_first_worker_after_preconstruction_failure(monkeypatch):
    requests = [
        {"attempt_id": "m1n0-pair-000-A", "pair_id": "m1n0-pair-000", "side": "A"},
        {"attempt_id": "m1n0-pair-000-B", "pair_id": "m1n0-pair-000", "side": "B"},
    ]
    calls = []

    def delegate(request):
        calls.append(request["attempt_id"])
        return {
            "attempt_id": request["attempt_id"],
            "pair_id": request["pair_id"],
            "side": request["side"],
            "status": "failed",
            "error": "setup failed",
            "protocol": {"construction_reset_count": 0},
        }

    runner = native.FailFastProcessRunner(delegate)
    first = runner(requests[0])
    second = runner(requests[1])
    assert first["status"] == "failed"
    assert second["status"] == "failed"
    assert second["not_run_due_to_cohort_block"] is True
    assert calls == ["m1n0-pair-000-A"]
    assert runner.dynamic_launches == 1


def test_fail_fast_does_not_stop_on_postconstruction_failure():
    calls = []

    def delegate(request):
        calls.append(request["attempt_id"])
        return {
            "attempt_id": request["attempt_id"],
            "pair_id": request["pair_id"],
            "side": request["side"],
            "status": "failed",
            "error": "trajectory failure",
            "protocol": {"construction_reset_count": 1},
        }

    runner = native.FailFastProcessRunner(delegate)
    request = {"attempt_id": "m1n0-pair-000-A", "pair_id": "m1n0-pair-000", "side": "A"}
    runner(request)
    assert runner.blocked is False
    assert runner.dynamic_launches == 1


def test_runtime_contract_branch_is_wired_without_legacy_parent():
    from scripts import m1_null_calibration as nullcal

    config = load_config()
    native.validate_native_config(config, check_files=False)
    # This regression only checks routing. File-dependent qualification is
    # separately executed by Codex in the real runtime.
    assert config["runtime_contract"] == "m1_sa_native_v1"
    assert "state_replay_config" not in config
    assert callable(nullcal._construct_adapter)



def test_nullcal_construct_adapter_routes_to_native_without_legacy_parent(monkeypatch):
    from scripts import m1_null_calibration as nullcal

    config = load_config()
    seen = {}

    def fake_construct(value):
        seen["config"] = value
        return object()

    monkeypatch.setattr(native, "construct_native_adapter", fake_construct)
    result = nullcal._construct_adapter(config)

    assert result is not None
    assert seen["config"]["runtime_contract"] == native.NATIVE_CONTRACT
    assert "state_replay_config" not in seen["config"]


def test_execute_attempt_marks_postconstruction_failure_as_constructed(monkeypatch):
    from scripts import m1_null_calibration as nullcal

    class FakeAdapter:
        def step(self, _action):
            raise RuntimeError("trajectory-level failure")

        def close(self):
            return None

    monkeypatch.setattr(nullcal, "_construct_adapter", lambda _config: FakeAdapter())
    monkeypatch.setattr(nullcal, "_window_map_from_registry", lambda _attempt: [])

    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "config": {"strict_runtime_contract": False},
        "tape": np.zeros((82, 7), dtype=np.float32),
    }
    result = nullcal.execute_attempt(attempt)

    assert result["status"] == "failed"
    assert result["protocol"]["construction_reset_count"] == 1
    assert result["close_evidence"]["success"] is True


def test_qualify_failure_publishes_compact_blocked_evidence(tmp_path, monkeypatch):
    config_path = tmp_path / "native.yaml"
    output_path = tmp_path / "static_qualification.json"
    config_path.write_text("runtime_contract: m1_sa_native_v1\n", encoding="utf-8")

    monkeypatch.setattr(
        native,
        "static_qualify",
        lambda _path: (_ for _ in ()).throw(native.NativeQualificationError("missing installed module")),
    )
    monkeypatch.setattr(native, "_repo_head", lambda: "a" * 40)

    code = native.main([
        "qualify",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ])

    assert code == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["stage"] == "static_qualification"
    assert payload["execution_commit"] == "a" * 40
    assert payload["legacy_state_replay_consulted"] is False


def test_construct_native_adapter_checks_libero_config_asset_binding(monkeypatch, tmp_path):
    config = load_config()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    config["assets"] = {"path": str(asset_root), "revision": config["assets"]["revision"]}

    fake_core = types.ModuleType("libero.libero")
    fake_core.get_libero_path = lambda key: str(asset_root) if key == "assets" else None
    fake_pkg = types.ModuleType("libero")
    fake_pkg.libero = fake_core
    monkeypatch.setitem(sys.modules, "libero", fake_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_core)

    monkeypatch.setattr(native, "validate_native_config", lambda *_args, **_kwargs: {"status": "PASS"})

    fake_preflight = types.ModuleType("scripts.dcu_preflight")
    runtime_bundle = {"env": object()}
    fake_preflight.build_cpu_environment_runtime = lambda _config, phase="compare": runtime_bundle
    monkeypatch.setitem(sys.modules, "scripts.dcu_preflight", fake_preflight)

    fake_state = types.ModuleType("scripts.m1_state_replay")

    class FakeRuntimeAdapter:
        @classmethod
        def construct_fresh(cls, _config, *, runtime_builder, tape_hash):
            assert runtime_builder(_config) is runtime_bundle
            return {"tape_hash": tape_hash}

    fake_state.RuntimeAdapter = FakeRuntimeAdapter
    monkeypatch.setitem(sys.modules, "scripts.m1_state_replay", fake_state)

    result = native.construct_native_adapter(config)
    assert result["tape_hash"] == config["action_tape"]["sha256"]


def test_construct_native_adapter_rejects_asset_binding_drift(monkeypatch, tmp_path):
    config = load_config()
    expected = tmp_path / "expected-assets"
    observed = tmp_path / "observed-assets"
    expected.mkdir()
    observed.mkdir()
    config["assets"] = {"path": str(expected), "revision": config["assets"]["revision"]}

    fake_core = types.ModuleType("libero.libero")
    fake_core.get_libero_path = lambda _key: str(observed)
    fake_pkg = types.ModuleType("libero")
    fake_pkg.libero = fake_core
    monkeypatch.setitem(sys.modules, "libero", fake_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_core)
    monkeypatch.setattr(native, "validate_native_config", lambda *_args, **_kwargs: {"status": "PASS"})

    with pytest.raises(native.NativeQualificationError, match="asset binding drift"):
        native.construct_native_adapter(config)
