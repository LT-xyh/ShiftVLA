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


def schema_config():
    config = load_config()
    config["runtime"]["environment"]["LIBERO_CONFIG_PATH"] = str(
        (ROOT / config["libero_config"]["path"]).resolve().parent
    )
    return config


def test_native_schema_does_not_require_legacy_source_checkouts():
    config = schema_config()
    native.validate_native_config(config, check_files=False)
    assert "source_checkouts" not in config
    assert config["runtime_contract"] == native.NATIVE_CONTRACT


@pytest.mark.parametrize("role", ["external/lerobot", "external/robosuite", "external/mujoco"])
def test_absent_legacy_checkout_is_not_a_schema_gate(role):
    config = schema_config()
    payload = json.dumps(config, sort_keys=True)
    assert role not in payload
    native.validate_native_config(config, check_files=False)


def test_legacy_state_replay_reference_is_rejected():
    config = schema_config()
    config["state_replay_config"] = {"path": "configs/m1/state_replay.yaml"}
    with pytest.raises(native.NativeQualificationError):
        native.validate_native_config(config, check_files=False)


def test_legacy_libero_config_path_is_rejected():
    config = schema_config()
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
        config = schema_config()
        config["runtime"][key] = bad
        with pytest.raises(native.NativeQualificationError):
            native.validate_native_config(config, check_files=False)


def test_hash_domains_are_explicit():
    config = load_config()
    assert config["registry"]["hash_domain"] == "registry_self_hash"
    assert config["action_tape"]["hash_domain"] == "float32_c_order_semantic_bytes"


def test_schedule_is_exactly_twenty_pairs():
    config = schema_config()
    native.validate_native_config(config, check_files=False)
    assert config["pair_count"] == 20
    assert config["pair_prefix"] == "m1n0-pair-"


def test_runtime_environment_binds_native_libero_config_on_locked_host():
    config = load_config()
    assert config["runtime"]["environment"]["LIBERO_CONFIG_PATH"] == (
        "/public/home/xuyinghao/workspace/replayvla-p1/"
        "runtime/replayvla-p1/f2_libero_config"
    )


def test_fail_fast_runner_launches_only_first_worker_after_preconstruction_failure():
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
            "failure_phase": "pre_construction",
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


@pytest.mark.parametrize("failure_phase", ["post_construction", "unknown", None])
def test_fail_fast_does_not_stop_without_explicit_preconstruction_failure(failure_phase):
    calls = []

    def delegate(request):
        calls.append(request["attempt_id"])
        result = {
            "attempt_id": request["attempt_id"],
            "pair_id": request["pair_id"],
            "side": request["side"],
            "status": "failed",
            "error": "trajectory or unclassified failure",
            "protocol": {"construction_reset_count": 0},
        }
        if failure_phase is not None:
            result["failure_phase"] = failure_phase
        return result

    runner = native.FailFastProcessRunner(delegate)
    request = {"attempt_id": "m1n0-pair-000-A", "pair_id": "m1n0-pair-000", "side": "A"}
    runner(request)
    assert runner.blocked is False
    assert runner.dynamic_launches == 1


def test_runtime_contract_routes_to_native_without_legacy_parent(monkeypatch):
    from scripts import m1_null_calibration as nullcal

    sentinel = {"native": True}
    calls = []

    def fake_runtime_config(config):
        calls.append(config)
        return sentinel

    monkeypatch.setattr(native, "native_runtime_config", fake_runtime_config)
    result = nullcal._official_runtime_config({"runtime_contract": native.NATIVE_CONTRACT})
    assert result is sentinel
    assert calls == [{"runtime_contract": native.NATIVE_CONTRACT}]


def test_worker_adapter_routes_to_native_constructor(monkeypatch):
    from scripts import m1_null_calibration as nullcal

    sentinel = object()
    calls = []

    def fake_construct(config):
        calls.append(config)
        return sentinel

    monkeypatch.setattr(native, "construct_native_adapter", fake_construct)
    config = {"runtime_contract": native.NATIVE_CONTRACT}
    assert nullcal._construct_adapter(config) is sentinel
    assert calls == [config]


def _install_fake_native_construction_modules(monkeypatch, *, wrap_error=None):
    fake_libero_package = types.ModuleType("libero")
    fake_libero_package.__path__ = []
    fake_libero_submodule = types.ModuleType("libero.libero")
    fake_libero_package.libero = fake_libero_submodule
    monkeypatch.setitem(sys.modules, "libero", fake_libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_libero_submodule)

    calls = {"build": 0, "close": 0, "wrap": 0}
    envs = object()

    fake_dcu = types.ModuleType("scripts.dcu_preflight")

    def build_cpu_environment_runtime(config, phase):
        calls["build"] += 1
        assert phase == "compare"

        def close_envs(value):
            assert value is envs
            calls["close"] += 1

        return {"envs": envs, "close_envs": close_envs}

    fake_dcu.build_cpu_environment_runtime = build_cpu_environment_runtime
    monkeypatch.setitem(sys.modules, "scripts.dcu_preflight", fake_dcu)

    fake_state = types.ModuleType("scripts.m1_state_replay")

    class FakeRuntimeAdapter:
        @classmethod
        def construct_fresh(cls, config, runtime_builder, tape_hash):
            calls["wrap"] += 1
            bundle = runtime_builder(config)
            assert bundle["envs"] is envs
            if wrap_error is not None:
                raise wrap_error
            return {"adapter": True, "tape_hash": tape_hash}

    fake_state.RuntimeAdapter = FakeRuntimeAdapter
    monkeypatch.setitem(sys.modules, "scripts.m1_state_replay", fake_state)
    return fake_libero_submodule, calls


def test_native_constructor_binds_assets_and_only_environment_factory(monkeypatch):
    config = schema_config()
    monkeypatch.setattr(native, "validate_native_config", lambda *args, **kwargs: {"status": "PASS"})
    libero_submodule, calls = _install_fake_native_construction_modules(monkeypatch)

    result = native.construct_native_adapter(config)

    assert libero_submodule._assets_path_cache == config["assets"]["path"]
    assert calls == {"build": 1, "close": 0, "wrap": 1}
    assert result["adapter"] is True


def test_native_constructor_closes_factory_env_if_adapter_wrap_fails(monkeypatch):
    config = schema_config()
    monkeypatch.setattr(native, "validate_native_config", lambda *args, **kwargs: {"status": "PASS"})
    _, calls = _install_fake_native_construction_modules(
        monkeypatch, wrap_error=RuntimeError("wrap failed")
    )

    with pytest.raises(RuntimeError, match="wrap failed"):
        native.construct_native_adapter(config)

    assert calls == {"build": 1, "close": 1, "wrap": 1}


def test_missing_installed_runtime_identity_blocks(monkeypatch):
    config = schema_config()
    config["python"] = sys.executable
    monkeypatch.setattr(
        native,
        "validate_native_config",
        lambda *args, **kwargs: {"runtime_lock": {"path": "lock", "sha256": "0" * 64}},
    )

    class Result:
        returncode = 1
        stdout = ""
        stderr = "No module named mujoco"

    monkeypatch.setattr(native.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(native.NativeQualificationError, match="trusted-dependency probe failed"):
        native.runtime_identity_audit(config)


def test_static_qualification_failure_publishes_compact_blocked(monkeypatch, tmp_path):
    output = tmp_path / "static_qualification.json"
    monkeypatch.setattr(native, "static_qualify", lambda path: (_ for _ in ()).throw(
        native.NativeQualificationError("missing installed module")
    ))
    monkeypatch.setattr(native, "_repo_head", lambda: "1" * 40)

    rc = native.main([
        "qualify",
        "--config",
        str(CONFIG),
        "--output",
        str(output),
    ])

    assert rc == 2
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "BLOCKED"
    assert evidence["stage"] == "static_qualification"
    assert evidence["execution_commit"] == "1" * 40
    assert "missing installed module" in evidence["reason"]


def test_compact_evidence_is_no_overwrite(tmp_path):
    target = tmp_path / "evidence.json"
    native._write_json_no_overwrite(target, {"status": "BLOCKED"})
    with pytest.raises(native.NativeQualificationError, match="refusing to overwrite"):
        native._write_json_no_overwrite(target, {"status": "PASS"})


def test_prepare_run_freezes_execution_commit_runtime_identity_and_20_pair_schedule(
    monkeypatch, tmp_path
):
    from scripts import m1_null_calibration as nullcal

    config = {
        "runtime_contract": native.NATIVE_CONTRACT,
        "strict_runtime_contract": True,
        "task": copy.deepcopy(nullcal.TASK),
        "pair_count": 20,
        "pair_prefix": "m1n0-pair-",
        "registry": {"path": str(tmp_path / "registry.json"), "sha256": "a" * 64},
        "action_tape": {"sha256": "b" * 64, "shape": [82, 7], "dtype": "float32"},
        "output_root": str(tmp_path / "run"),
        "terminal_contract": copy.deepcopy(nullcal.DEFAULT_TERMINAL_CONTRACT),
        "python": sys.executable,
        "worker_timeout_seconds": 10,
        "runtime": {},
        "obs_type": "pixels_agent_pos",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    trace = {"trace_id": "trace0"}
    registry = {
        "registry_sha256": "a" * 64,
        "traces": [trace],
    }
    tape = np.zeros((82, 7), dtype=np.float32)
    runtime_identity = {"contract": native.NATIVE_CONTRACT, "facts": {"python": "fake"}}

    monkeypatch.setattr(nullcal, "_official_runtime_config", lambda value: copy.deepcopy(value))
    monkeypatch.setattr(nullcal, "_load_verified_registry", lambda path: copy.deepcopy(registry))
    monkeypatch.setattr(nullcal, "_verify_registry_payload", lambda *args, **kwargs: None)
    monkeypatch.setattr(nullcal, "_trace_record", lambda value: trace)
    monkeypatch.setattr(nullcal, "_load_tape", lambda *args, **kwargs: (tape, "b" * 64))
    monkeypatch.setattr(
        nullcal,
        "_windows",
        lambda *args, **kwargs: [
            {
                "window_id": regime,
                "regimes": [regime],
                "capture_offset": 1,
                "continuation_horizon": 1,
                "source_coverage": {"regimes": [regime]},
            }
            for regime in nullcal.REGIME_NAMES
        ],
    )
    monkeypatch.setattr(nullcal, "_validate_strict_null_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(nullcal, "_validate_trace_frozen_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        nullcal,
        "_registry_terminal_contract",
        lambda *args, **kwargs: copy.deepcopy(nullcal.DEFAULT_TERMINAL_CONTRACT),
    )
    monkeypatch.setattr(
        nullcal, "_runtime_identity_audit", lambda *args, **kwargs: copy.deepcopy(runtime_identity)
    )
    monkeypatch.setattr(nullcal, "_repo_head", lambda: "2" * 40)

    prepared = nullcal.prepare_run(config_path=config_path)

    assert prepared.run_spec["execution_commit"] == "2" * 40
    assert prepared.run_spec["runtime_identity_contract"] == runtime_identity
    assert len(prepared.run_spec["pairs"]) == 20
    assert sum(len(pair["attempts"]) for pair in prepared.run_spec["pairs"]) == 40
    assert prepared.run_spec["state_replay_config"] is None
