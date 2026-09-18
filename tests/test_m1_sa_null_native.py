import copy
import hashlib
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
            "failure_stage": "PRE_CONSTRUCTION",
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
            "failure_stage": "POST_CONSTRUCTION",
            "error": "trajectory failure",
            "protocol": {"construction_reset_count": 0},
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


def test_execute_attempt_classifies_postconstruction_failure_without_forging_protocol(monkeypatch):
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
    assert result["failure_stage"] == "POST_CONSTRUCTION"
    assert result["protocol"]["construction_reset_count"] == 0
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
    config_file = (ROOT / config["libero_config"]["path"]).resolve()
    fake_core.libero_config_path = str(config_file.parent)
    fake_core.config_file = str(config_file)
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
    config_file = (ROOT / config["libero_config"]["path"]).resolve()
    fake_core.libero_config_path = str(config_file.parent)
    fake_core.config_file = str(config_file)
    fake_core.get_libero_path = lambda _key: str(observed)
    fake_pkg = types.ModuleType("libero")
    fake_pkg.libero = fake_core
    monkeypatch.setitem(sys.modules, "libero", fake_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_core)
    monkeypatch.setattr(native, "validate_native_config", lambda *_args, **_kwargs: {"status": "PASS"})

    with pytest.raises(native.NativeQualificationError, match="asset binding drift"):
        native.construct_native_adapter(config)



def test_execute_attempt_preconstruction_failure_is_classified_and_failfast_blocks():
    from scripts import m1_null_calibration as nullcal

    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "config": {"strict_runtime_contract": False},
        "tape": np.zeros((82, 7), dtype=np.float32),
    }

    def fail_factory(_config):
        raise RuntimeError("construction failed")

    failure = nullcal.execute_attempt(attempt, adapter_factory=fail_factory)
    assert failure["failure_stage"] == "PRE_CONSTRUCTION"
    assert failure["protocol"]["construction_reset_count"] == 0

    runner = native.FailFastProcessRunner(lambda _request: failure)
    runner(attempt)
    assert runner.blocked is True
    assert runner.dynamic_launches == 1


def test_postconstruction_failure_does_not_trigger_failfast(monkeypatch):
    from scripts import m1_null_calibration as nullcal

    class FakeAdapter:
        def step(self, _action):
            raise RuntimeError("trajectory failed")

        def close(self):
            return None

    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "config": {"strict_runtime_contract": False},
        "tape": np.zeros((82, 7), dtype=np.float32),
    }
    monkeypatch.setattr(nullcal, "_window_map_from_registry", lambda _attempt: [])
    failure = nullcal.execute_attempt(attempt, adapter_factory=lambda _config: FakeAdapter())
    assert failure["failure_stage"] == "POST_CONSTRUCTION"
    assert failure["protocol"]["construction_reset_count"] == 0

    runner = native.FailFastProcessRunner(lambda _request: failure)
    runner(attempt)
    assert runner.blocked is False
    assert runner.dynamic_launches == 1


def test_completed_attempt_keeps_existing_protocol_evidence(monkeypatch):
    from scripts import m1_null_calibration as nullcal

    class FakeAdapter:
        def __init__(self):
            self.calls = 0

        def step(self, _action):
            self.calls += 1
            return ({}, 0.0, self.calls == 82, False, {"termination_reason": "predicate_transition"})

        def close(self):
            return None

    sentinel_protocol = {
        "construction_reset_count": 7,
        "actions_executed": 82,
        "step_calls": 82,
    }
    monkeypatch.setattr(nullcal, "_window_map_from_registry", lambda _attempt: [])
    monkeypatch.setattr(nullcal, "_adapter_snapshot", lambda _adapter: {})
    monkeypatch.setattr(nullcal, "_adapter_rgb", lambda _adapter: None)
    monkeypatch.setattr(nullcal, "_terminal_reason_evidence", lambda *_args, **_kwargs: (
        "predicate_transition", {"source": "fake", "returned_step": True}
    ))
    monkeypatch.setattr(nullcal, "_terminal_success", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(nullcal, "_derive_terminal_reason", lambda reason, **_kwargs: (
        reason, {"source": "fake"}
    ))
    monkeypatch.setattr(nullcal, "_validate_terminal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(nullcal, "_runtime_fingerprint", lambda *_args, **_kwargs: {"fake": True})
    monkeypatch.setattr(nullcal, "_protocol_counters", lambda *_args, **_kwargs: copy.deepcopy(sentinel_protocol))
    monkeypatch.setattr(nullcal, "_proc_start_identity", lambda _pid: "123:456:fake")

    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "trace_id": "fake-trace",
        "config": {"strict_runtime_contract": False},
        "tape": np.zeros((82, 7), dtype=np.float32),
        "terminal_contract": {
            "step": 82,
            "termination_reason": "predicate_transition",
            "success": True,
            "terminated": True,
            "truncated": False,
        },
    }
    result = nullcal.execute_attempt(attempt, adapter_factory=lambda _config: FakeAdapter())
    assert result["status"] == "completed"
    assert "failure_stage" not in result
    assert result["protocol"] == sentinel_protocol


def _write_fake_asset_manifest(manifest_path, asset_root, *, file_bytes=b"asset-bytes", row_sha=None, row_size=None):
    asset_root.mkdir(parents=True, exist_ok=True)
    target = asset_root / "tiny.bin"
    target.write_bytes(file_bytes)
    digest = hashlib.sha256(file_bytes).hexdigest()
    row = {
        "repo_id": native.EXPECTED_ASSET_REPO_ID,
        "revision": native.EXPECTED_ASSET_REVISION,
        "local_path": str(asset_root.resolve()),
        "files": [{
            "path": "tiny.bin",
            "bytes": len(file_bytes) if row_size is None else row_size,
            "sha256": digest if row_sha is None else row_sha,
        }],
        "required_filenames": ["tiny.bin"],
        "inventory": {
            "file_count": 1,
            "total_bytes": len(file_bytes) if row_size is None else row_size,
        },
    }
    manifest_path.write_text(json.dumps({"artifacts": [row]}, sort_keys=True), encoding="utf-8")
    return {
        "repo_id": native.EXPECTED_ASSET_REPO_ID,
        "revision": native.EXPECTED_ASSET_REVISION,
        "path": str(asset_root.resolve()),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
    }


def test_complete_fake_asset_inventory_passes(tmp_path):
    manifest = tmp_path / "manifest.json"
    assets = _write_fake_asset_manifest(manifest, tmp_path / "assets")
    evidence = native._verify_asset_manifest(assets)
    assert evidence["status"] == "PASS"
    assert evidence["file_count"] == 1
    assert evidence["verified_file_count"] == 1
    assert evidence["total_bytes"] == len(b"asset-bytes")
    assert evidence["failures"] == []


def test_asset_manifest_sha_drift_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    assets = _write_fake_asset_manifest(manifest, tmp_path / "assets")
    assets["manifest"]["sha256"] = "0" * 64
    with pytest.raises(native.AssetQualificationError) as caught:
        native._verify_asset_manifest(assets)
    evidence = caught.value.evidence
    assert evidence["status"] == "BLOCKED"
    assert any("manifest SHA drift" in item for item in evidence["failures"])


@pytest.mark.parametrize("drift", ["sha", "size"])
def test_asset_file_sha_or_size_drift_rejected(tmp_path, drift):
    manifest = tmp_path / "manifest.json"
    kwargs = {"row_sha": "0" * 64} if drift == "sha" else {"row_size": 999}
    assets = _write_fake_asset_manifest(manifest, tmp_path / "assets", **kwargs)
    with pytest.raises(native.AssetQualificationError) as caught:
        native._verify_asset_manifest(assets)
    evidence = caught.value.evidence
    assert evidence["status"] == "BLOCKED"
    assert evidence["verified_file_count"] == 0
    assert any(("SHA drift" in item if drift == "sha" else "size drift" in item) for item in evidence["failures"])


def test_construct_native_adapter_rejects_stale_libero_module_config(monkeypatch, tmp_path):
    config = load_config()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    config["assets"]["path"] = str(asset_root)

    fake_core = types.ModuleType("libero.libero")
    fake_core.libero_config_path = str(tmp_path / "stale")
    fake_core.config_file = str(tmp_path / "stale" / "config.yaml")
    fake_core.get_libero_path = lambda _key: str(asset_root)
    fake_pkg = types.ModuleType("libero")
    fake_pkg.libero = fake_core
    monkeypatch.setitem(sys.modules, "libero", fake_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", fake_core)
    monkeypatch.setattr(native, "validate_native_config", lambda *_args, **_kwargs: {"status": "PASS"})

    with pytest.raises(native.NativeQualificationError, match="module-level config path is stale"):
        native.construct_native_adapter(config)


def _write_execute_inputs(tmp_path):
    config_path = tmp_path / "native.yaml"
    config_path.write_text("runtime_contract: m1_sa_native_v1\n", encoding="utf-8")
    qualification_path = tmp_path / "qualification.json"
    qualification = {
        "status": "PASS",
        "config_file_sha256": native._sha256_file(config_path),
        "execution_commit": "b" * 40,
    }
    qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
    return config_path, qualification_path


def test_prepare_run_exception_publishes_blocked_summary_without_workers(tmp_path, monkeypatch):
    from scripts import m1_null_calibration as nullcal

    monkeypatch.setattr(native, "ROOT", tmp_path)
    monkeypatch.setattr(native, "_repo_head", lambda: "b" * 40)
    monkeypatch.setattr(
        nullcal,
        "prepare_run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("prepare failed")),
    )
    config_path, qualification_path = _write_execute_inputs(tmp_path)

    result = native.execute_f3n(config_path, qualification_path)
    summary_path = tmp_path / "runtime/replayvla-p1/f3n/f3n_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert result["status"] == "BLOCKED"
    assert summary["stage"] == "prepare_run"
    assert summary["f3n"]["dynamic_worker_launches"] == 0
    assert summary["f3n"]["schedule_publication"]["runnable"] is False
    assert summary["f3n"]["schedule_publication"]["null_schedule"]["published"] is False
    assert summary["f3n"]["schedule_publication"]["pair_registry"]["published"] is False


def test_prepare_blocked_summary_is_no_overwrite(tmp_path, monkeypatch):
    from scripts import m1_null_calibration as nullcal

    monkeypatch.setattr(native, "ROOT", tmp_path)
    monkeypatch.setattr(native, "_repo_head", lambda: "b" * 40)
    monkeypatch.setattr(
        nullcal,
        "prepare_run",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("prepare failed")),
    )
    config_path, qualification_path = _write_execute_inputs(tmp_path)
    native.execute_f3n(config_path, qualification_path)
    summary_path = tmp_path / "runtime/replayvla-p1/f3n/f3n_summary.json"
    before = summary_path.read_bytes()

    with pytest.raises(native.NativeQualificationError, match="already exists"):
        native.execute_f3n(config_path, qualification_path)
    assert summary_path.read_bytes() == before


def test_partial_schedule_publication_is_blocked_and_not_runnable(tmp_path, monkeypatch):
    from scripts import m1_null_calibration as nullcal

    monkeypatch.setattr(native, "ROOT", tmp_path)
    monkeypatch.setattr(native, "_repo_head", lambda: "b" * 40)
    config_path, qualification_path = _write_execute_inputs(tmp_path)

    run_spec = tmp_path / "run_spec.json"
    pair_registry = tmp_path / "pair_registry.json"
    run_spec.write_text("{}", encoding="utf-8")
    pair_registry.write_text("{}", encoding="utf-8")
    prepared = types.SimpleNamespace(run_spec_path=run_spec, pair_registry_path=pair_registry)
    monkeypatch.setattr(nullcal, "prepare_run", lambda **_kwargs: prepared)

    calls = []

    def copy_once_then_fail(source, target):
        calls.append((source, target))
        if len(calls) == 1:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            return {"path": str(target), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "size": source.stat().st_size}
        raise native.NativeQualificationError("pair registry publication failed")

    monkeypatch.setattr(native, "_copy_bytes_no_overwrite", copy_once_then_fail)
    result = native.execute_f3n(config_path, qualification_path)
    summary = json.loads(
        (tmp_path / "runtime/replayvla-p1/f3n/f3n_summary.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "BLOCKED"
    assert summary["stage"] == "schedule_publication"
    assert summary["f3n"]["dynamic_worker_launches"] == 0
    assert summary["f3n"]["schedule_publication"]["runnable"] is False
    assert summary["f3n"]["schedule_publication"]["null_schedule"]["published"] is True
    assert summary["f3n"]["schedule_publication"]["pair_registry"]["published"] is False
