import copy
import json
from pathlib import Path

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
