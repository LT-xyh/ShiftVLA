"""Fake-only contract tests for the fresh-process M1-N0 calibration path."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    import scripts.m1_null_calibration as calibration

    return calibration


def _registry_payload(calibration: Any, *, registry_path: Path, tape_path: Path) -> dict[str, Any]:
    tape = np.zeros((82, 7), dtype=np.float32)
    np.save(tape_path, tape, allow_pickle=False)
    tape_sha = hashlib.sha256(tape.tobytes(order="C")).hexdigest()
    windows = []
    for index, regime in enumerate(calibration.REGIME_NAMES):
        offset = (1, 33, 43, 49, 54, 80, 81)[index]
        horizon = (10, 10, 10, 10, 10, 2, 1)[index]
        coverage = {"regimes": [regime], "capture_offset": offset, "continuation_horizon": horizon}
        if regime in {"release", "predicate_transition"}:
            coverage.update({"terminal_reason": "predicate_transition", "legal_terminal_step": 82})
        windows.append(
            {
                "window_id": regime,
                "regimes": [regime],
                "capture_offset": offset,
                "continuation_horizon": horizon,
                "source_coverage": coverage,
            }
        )
    trace_id = "libero_spatial-task000-init000"
    trace = {
        "trace_id": trace_id,
        "trajectory_id": trace_id,
        "episode_id": trace_id,
        "suite": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "seed": 2027,
        "action_dtype": "float32",
        "action_shape": [82, 7],
        "action_sha256": tape_sha,
        "regimes": list(calibration.REGIME_NAMES),
        "capture_offset": 1,
        "continuation_horizon": 10,
        "source_coverage": {"regimes": list(calibration.REGIME_NAMES)},
        "windows": windows,
    }
    payload = {
        "schema_version": 1,
        "registry_type": "replayvla_m1_frozen_registry",
        "regimes": list(calibration.REGIME_NAMES),
        "traces": [trace],
        "tape_provenance": {
            "algorithm": "persisted_m0_executed_actions",
            "dtype": "float32",
            "shape": [82, 7],
            "sha256": tape_sha,
            "path": str(tape_path),
            "per_trace": {
                trace_id: {
                    "algorithm": "persisted_m0_executed_actions",
                    "dtype": "float32",
                    "shape": [82, 7],
                    "sha256": tape_sha,
                    "path": str(tape_path),
                }
            },
        },
        "selection": {"trace_ids": [trace_id]},
    }
    payload["registry_sha256"] = calibration.sha256_bytes(
        calibration.canonical_json({key: value for key, value in payload.items()}).encode()
    )
    registry_path.write_text(calibration.canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def _config_file(calibration: Any, tmp_path: Path, registry_path: Path, output_root: Path) -> Path:
    tape = np.zeros((82, 7), dtype=np.float32)
    tape_sha = hashlib.sha256(tape.tobytes(order="C")).hexdigest()
    config: dict[str, Any] = {
        "schema_version": 1,
        "name": "m1_null_calibration",
        "pair_count": 20,
        "pair_prefix": "m1n0-pair-",
        "output_root": str(output_root),
        "registry": {"path": str(registry_path)},
        "registry_sha256": calibration.sha256_file(registry_path),
        "action_tape": {
            "sha256": tape_sha,
            "shape": [82, 7],
            "dtype": "float32",
        },
        "task": {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027},
        "python": sys.executable,
        "runtime": {"include_policy": False, "call_policy": False, "call_processors": False},
        "quantity_selection": {"root_patterns": ["qpos", "qvel", "objects", "gripper_physical"]},
        "renderer": {"camera": "agentview", "observation_key": "render_rgb"},
    }
    config["config_sha256"] = calibration.config_contract_sha256(config)
    path = tmp_path / "null_calibration.yaml"
    path.write_text(calibration.canonical_json(config) + "\n", encoding="utf-8")
    return path


def _prepared(calibration: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], Path, Path]:
    registry_path = tmp_path / "registry.json"
    tape_path = tmp_path / "actions.npy"
    payload = _registry_payload(calibration, registry_path=registry_path, tape_path=tape_path)
    config_path = _config_file(calibration, tmp_path, registry_path, tmp_path / "run")
    monkeypatch.setattr(calibration, "_load_verified_registry", lambda path: deepcopy(payload))
    prepared = calibration.prepare_run(config_path=config_path)
    return prepared, payload, config_path, tape_path


def _snapshot(step: int, *, contact: bool = True) -> dict[str, Any]:
    return {
        "invariants": {
            "qpos": np.asarray([step, step + 0.1], dtype=np.float64),
            "qvel": np.asarray([0.1, 0.2], dtype=np.float64),
            "gripper_physical": {"position": np.asarray([0.1], dtype=np.float64)},
            "objects": {
                "target": {
                    "body_pos": np.asarray([step * 0.01, 0.0, 0.0], dtype=np.float64),
                    "body_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    "joints": {},
                }
            },
            "contacts": [("robot_finger", "target_geom", 0.0)] if contact else [],
            "predicates": {
                "available": True,
                "goals": [{"index": 0, "expression": ["On", "target", "plate"], "value": step >= 82}],
            },
            "success": bool(step >= 82),
            "counters": {
                "timestep": step,
                "done": bool(step >= 82),
                "terminated": bool(step >= 82),
                "truncated": False,
            },
            "gripper": {"current_action": np.asarray([-1.0, 1.0], dtype=np.float64)},
        },
        "raw_observation": {
            "pixels": np.full((2, 2, 3), step % 255, dtype=np.uint8),
            "agent_pos": np.asarray([step], dtype=np.float32),
        },
        "renderer": np.full((2, 2, 3), step % 255, dtype=np.uint8),
    }


def _attempt_result(calibration: Any, pair_id: str, side: str, pid: int, *, fail: bool = False) -> dict[str, Any]:
    if fail:
        return {
            "attempt_id": f"{pair_id}-{side}",
            "pair_id": pair_id,
            "side": side,
            "pid": pid,
            "ppid": 1,
            "process_start_identity": f"start-{pid}",
            "status": "failed",
            "error": "synthetic worker failure",
            "protocol": {"retry_count": 0, "restore_count": 0, "policy_calls": 0, "post_terminal_steps": 0},
        }
    windows: dict[str, Any] = {}
    offsets = (1, 33, 43, 49, 54, 80, 81)
    horizons = (10, 10, 10, 10, 10, 2, 1)
    for regime, offset, horizon in zip(calibration.REGIME_NAMES, offsets, horizons):
        windows[regime] = {
            "regime": regime,
            "snapshots": {str(h): _snapshot(offset + h) for h in range(horizon + 1)},
            "action_sha256": "action-sha",
        }
    result = {
        "attempt_id": f"{pair_id}-{side}",
        "pair_id": pair_id,
        "side": side,
        "pid": pid,
        "ppid": 1,
        "process_start_identity": f"start-{pid}",
        "status": "completed",
        "frozen_inputs": {
            "trace_id": "libero_spatial-task000-init000",
            "task": {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027},
            "action_tape_sha256": "tape-sha",
            "action_shape": [82, 7],
            "action_dtype": "float32",
            "physics_model_fingerprint": {"hash": "physics"},
            "observation_model_fingerprint": {"hash": "observation"},
        },
        "windows": windows,
        "terminal": {
            "step": 82,
            "termination_reason": "predicate_transition",
            "success": True,
            "terminated": True,
            "truncated": False,
        },
        "protocol": {
            "construction_reset_count": 1,
            "restore_count": 0,
            "policy_calls": 0,
            "retry_count": 0,
            "post_terminal_steps": 0,
            "actions_executed": 82,
        },
    }
    result["output_sha256"] = calibration.payload_sha256(result)
    return result


def test_prepare_is_source_only_and_preregisters_exactly_twenty_pairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = _module()
    builds: list[Any] = []
    monkeypatch.setattr(calibration, "_construct_adapter", lambda *_args, **_kwargs: builds.append(True))
    prepared, payload, _config_path, _tape_path = _prepared(calibration, tmp_path, monkeypatch)
    assert not builds
    assert prepared.run_spec_path.is_file()
    assert prepared.pair_registry_path.is_file()
    assert calibration.verify_self_hash(prepared.run_spec, "run_spec_sha256")
    assert calibration.verify_self_hash(prepared.pair_registry, "pair_registry_sha256")
    assert [item["pair_id"] for item in prepared.pair_registry["pairs"]] == [
        f"m1n0-pair-{index:03d}" for index in range(20)
    ]
    assert all(len(item["attempts"]) == 2 for item in prepared.pair_registry["pairs"])
    assert payload["registry_sha256"] == prepared.run_spec["registry_sha256"]


def test_run_continues_frozen_schedule_after_failure_and_never_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = _module()
    prepared, _payload, _config, _tape = _prepared(calibration, tmp_path, monkeypatch)
    calls: list[str] = []

    def runner(attempt: dict[str, Any]) -> dict[str, Any]:
        calls.append(attempt["attempt_id"])
        if attempt["attempt_id"] == "m1n0-pair-003-A":
            raise RuntimeError("one-shot failure")
        return _attempt_result(calibration, attempt["pair_id"], attempt["side"], 1000 + len(calls))

    result = calibration.run_calibration(prepared, process_runner=runner)
    assert calls == [
        f"m1n0-pair-{pair:03d}-{side}"
        for pair in range(20)
        for side in ("A", "B")
    ]
    assert len(result["attempts"]) == 40
    failed = [item for item in result["attempts"] if item["attempt_id"] == "m1n0-pair-003-A"]
    assert len(failed) == 1
    assert failed[0]["status"] == "failed"
    assert result["counts"]["scheduled_attempts"] == 40
    assert result["counts"]["valid_pairs"] == 0
    assert result["status"] == "BLOCKED"


def test_pair_validation_requires_distinct_processes_and_zero_forbidden_protocol_activity() -> None:
    calibration = _module()
    pair = {"pair_id": "m1n0-pair-000"}
    good_a = _attempt_result(calibration, pair["pair_id"], "A", 11)
    good_b = _attempt_result(calibration, pair["pair_id"], "B", 12)
    validated = calibration.validate_pair(pair, {"A": good_a, "B": good_b}, parent_pid=1)
    assert validated.valid is True
    same_pid = deepcopy(good_b)
    same_pid["pid"] = good_a["pid"]
    assert calibration.validate_pair(pair, {"A": good_a, "B": same_pid}, parent_pid=1).valid is False
    restored = deepcopy(good_b)
    restored["protocol"]["restore_count"] = 1
    invalid = calibration.validate_pair(pair, {"A": good_a, "B": restored}, parent_pid=1)
    assert invalid.valid is False
    assert any("restore" in reason for reason in invalid.reasons)


def test_worker_executes_once_from_step_one_stops_at_terminal_and_never_restores() -> None:
    calibration = _module()
    class FakeAdapter:
        def __init__(self) -> None:
            self.steps: list[np.ndarray] = []
            self.capture_calls = 0
            self.restore_calls = 0
            self.reset_calls = 1
        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.steps.append(np.asarray(action).copy())
            terminal = len(self.steps) == 82
            return {"step": len(self.steps)}, 0.0, terminal, False, {"termination_reason": "predicate_transition"} if terminal else {}
        def capture(self, *args: Any, **kwargs: Any) -> None:
            self.capture_calls += 1
            raise AssertionError("capture is forbidden")
        def restore(self, *args: Any, **kwargs: Any) -> None:
            self.restore_calls += 1
            raise AssertionError("restore is forbidden")
        def collect_invariants(self) -> dict[str, Any]:
            return _snapshot(len(self.steps))["invariants"]
        def render_rgb(self) -> np.ndarray:
            return _snapshot(len(self.steps))["renderer"]
        def close(self) -> None:
            return None
    adapter = FakeAdapter()
    registry = {
        "registry_sha256": "registry-sha",
        "traces": [{
            "trace_id": "trace",
            "windows": [
                {"window_id": "free_motion", "regimes": ["free_motion"], "capture_offset": 1, "continuation_horizon": 1, "source_coverage": {"regimes": ["free_motion"]}},
                {"window_id": "predicate_transition", "regimes": ["predicate_transition"], "capture_offset": 81, "continuation_horizon": 1, "source_coverage": {"regimes": ["predicate_transition"], "terminal_reason": "predicate_transition", "legal_terminal_step": 82}},
            ],
        }],
        "tape_provenance": {"sha256": "tape-sha", "shape": [82, 7], "dtype": "float32"},
    }
    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "trace_id": "trace",
        "config": {"task": {"suite": "libero_spatial", "task_id": 0, "init_state_id": 0, "seed": 2027}},
        "registry": registry,
        "tape": np.zeros((82, 7), dtype=np.float32),
    }
    result = calibration.execute_attempt(attempt, adapter_factory=lambda _config: adapter)
    assert result["status"] == "completed"
    assert len(adapter.steps) == 82
    assert adapter.capture_calls == 0
    assert adapter.restore_calls == 0
    assert result["protocol"]["post_terminal_steps"] == 0
    assert result["terminal"]["step"] == 82


def test_grouped_envelopes_split_exact_semantics_from_floating_and_renderer_metrics() -> None:
    calibration = _module()
    pair_results = []
    for index in range(20):
        pair_results.append(
            calibration.validate_pair(
                {"pair_id": f"m1n0-pair-{index:03d}", "trace_id": "trace"},
                {
                    "A": _attempt_result(calibration, f"m1n0-pair-{index:03d}", "A", 100 + 2 * index),
                    "B": _attempt_result(calibration, f"m1n0-pair-{index:03d}", "B", 101 + 2 * index),
                },
                parent_pid=1,
            )
        )
    envelopes = calibration.build_envelopes(pair_results, required_pair_ids=[f"m1n0-pair-{i:03d}" for i in range(20)])
    assert envelopes.physics["coverage"]["n_pairs"] == 20
    assert envelopes.renderer["coverage"]["group_count"] > 0
    assert envelopes.renderer["groups"]["agentview"]["render_rgb"]["free_motion"]["0"]["exact_only"] is True
    semantic = envelopes.discrete
    assert semantic["categorical_gate"] is True
    assert semantic["contact_identity_exact"] is True


def test_safe_artifact_creation_is_hash_checked_and_non_overwriting(tmp_path: Path) -> None:
    calibration = _module()
    target = tmp_path / "artifact.json"
    record = calibration.write_json_atomic(target, {"z": 1, "a": 2})
    assert record["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(calibration.PublicationError):
        calibration.write_json_atomic(target, {"different": True})
    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["a"] = 9
    target.write_text(calibration.canonical_json(tampered) + "\n", encoding="utf-8")
    with pytest.raises(calibration.PublicationError, match="hash"):
        calibration.verify_artifact_hash(target, record["sha256"])


def test_config_and_registry_hash_drift_is_refused_before_process_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = _module()
    prepared, _payload, config_path, _tape = _prepared(calibration, tmp_path, monkeypatch)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pair_count"] = 19
    config_path.write_text(calibration.canonical_json(config) + "\n", encoding="utf-8")
    calls: list[Any] = []
    with pytest.raises(calibration.ProvenanceError, match="config|hash"):
        calibration.run_calibration(prepared, process_runner=lambda attempt: calls.append(attempt))
    assert calls == []


def test_direct_script_invocation_has_no_pythonpath_requirement() -> None:
    script = ROOT / "scripts" / "m1_null_calibration.py"
    result = subprocess.run(
        [sys.executable, str(script), "--schema-version"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "1"
    assert "No module named 'scripts'" not in result.stderr


def test_module_keeps_policy_and_environment_imports_lazy() -> None:
    calibration = _module()
    source = ast.parse(__import__("inspect").getsource(calibration))
    eager = "\n".join(ast.unparse(node) for node in source.body if isinstance(node, (ast.Import, ast.ImportFrom)))
    assert "m1_state_replay" not in eager
    assert "dcu_preflight" not in eager
    assert "smolvla" not in eager.lower()


def test_strict_config_requires_obs_type_and_frozen_state_replay_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = _module()
    registry_path = tmp_path / "registry.json"
    tape_path = tmp_path / "actions.npy"
    payload = _registry_payload(calibration, registry_path=registry_path, tape_path=tape_path)
    config_path = _config_file(calibration, tmp_path, registry_path, tmp_path / "run")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["strict_runtime_contract"] = True
    config.pop("obs_type", None)
    config["config_sha256"] = calibration.config_contract_sha256(config)
    config_path.write_text(calibration.canonical_json(config) + "\n", encoding="utf-8")
    monkeypatch.setattr(calibration, "_load_verified_registry", lambda _path: deepcopy(payload))
    with pytest.raises(calibration.ProvenanceError, match="obs_type|state_replay"):
        calibration.prepare_run(config_path=config_path)


def test_pair_validation_rejects_frozen_attempt_id_drift() -> None:
    calibration = _module()
    pair = {"pair_id": "m1n0-pair-000"}
    left = _attempt_result(calibration, pair["pair_id"], "A", 11)
    right = _attempt_result(calibration, pair["pair_id"], "B", 12)
    left["attempt_id"] = "m1n0-pair-001-A"
    validated = calibration.validate_pair(pair, {"A": left, "B": right}, parent_pid=1)
    assert validated.valid is False
    assert any("attempt_id" in reason for reason in validated.reasons)


def test_execute_attempt_records_real_proc_identity_and_recursive_observation_metadata() -> None:
    calibration = _module()

    class FakeAdapter:
        def __init__(self) -> None:
            self.steps = 0

        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.steps += 1
            terminal = self.steps == 82
            observation = {
                "views": {"agentview": {"image": np.full((2, 2, 3), self.steps, dtype=np.uint8)}},
                "agent_pos": np.asarray([self.steps], dtype=np.float32),
            }
            return observation, 0.0, terminal, False, {
                "termination_reason": "predicate_transition" if terminal else None,
                "success": terminal,
            }

        def collect_invariants(self) -> dict[str, Any]:
            return _snapshot(self.steps)

        def render_rgb(self) -> np.ndarray:
            return np.full((2, 2, 3), self.steps, dtype=np.uint8)

        def close(self) -> None:
            return None

    result = calibration.execute_attempt(
        {
            "attempt_id": "m1n0-pair-000-A",
            "pair_id": "m1n0-pair-000",
            "side": "A",
            "trace_id": "trace",
            "config": {"task": dict(calibration.TASK)},
            "registry": {
                "traces": [{
                    "trace_id": "trace",
                    "windows": [{
                        "window_id": "free_motion",
                        "regimes": ["free_motion"],
                        "capture_offset": 1,
                        "continuation_horizon": 1,
                        "source_coverage": {"regimes": ["free_motion"]},
                    }],
                }],
            },
            "tape": np.zeros((82, 7), dtype=np.float32),
        },
        adapter_factory=lambda _config: FakeAdapter(),
    )
    assert result["status"] == "completed"
    pid, start, boot = calibration._parse_proc_start_identity(result["process_start_identity"])
    assert pid == result["pid"]
    assert start and boot
    snapshot = result["windows"]["free_motion"]["snapshots"]["0"]
    assert "observation_metadata" in snapshot
    assert "views.agentview.image" in snapshot["observation_metadata"]["arrays"]


def test_strict_quantity_selection_rejects_missing_required_roots() -> None:
    calibration = _module()
    pair_results = []
    for index in range(20):
        pair_id = f"m1n0-pair-{index:03d}"
        pair_results.append(
            calibration.validate_pair(
                {"pair_id": pair_id, "trace_id": "trace"},
                {
                    "A": _attempt_result(calibration, pair_id, "A", 100 + 2 * index),
                    "B": _attempt_result(calibration, pair_id, "B", 101 + 2 * index),
                },
                parent_pid=1,
            )
        )


def test_quantity_selection_keeps_contact_distances_as_floating_leaves() -> None:
    calibration = _module()
    snapshot = {
        "invariants": {
            "contacts": [("robot_finger", "target_geom", -0.0125)],
        }
    }
    values = calibration._selected_numeric_leaves(snapshot, ("contacts",))
    assert set(values) == {"contacts[0].distance"}
    assert values["contacts[0].distance"].dtype == np.dtype("float64")
    assert values["contacts[0].distance"].tolist() == [-0.0125]


def test_renderer_envelope_uses_configured_camera_and_observation_key() -> None:
    calibration = _module()
    pair_results = []
    for index in range(20):
        pair_id = f"m1n0-pair-{index:03d}"
        pair_results.append(
            calibration.validate_pair(
                {"pair_id": pair_id, "trace_id": "trace"},
                {
                    "A": _attempt_result(calibration, pair_id, "A", 100 + 2 * index),
                    "B": _attempt_result(calibration, pair_id, "B", 101 + 2 * index),
                },
                parent_pid=1,
            )
        )
    envelopes = calibration.build_envelopes(
        pair_results,
        required_pair_ids=[f"m1n0-pair-{index:03d}" for index in range(20)],
        config={"renderer": {"camera": "sideview", "observation_key": "side_rgb"}},
    )
    assert "sideview" in envelopes.renderer["groups"]
    assert "side_rgb" in envelopes.renderer["groups"]["sideview"]
    with pytest.raises(calibration.NullCalibrationError, match="required|root|missing"):
        calibration.build_envelopes(
            pair_results,
            required_pair_ids=[f"m1n0-pair-{index:03d}" for index in range(20)],
            config={
                "strict_runtime_contract": True,
                "quantity_selection": {"root_patterns": list(calibration.STRICT_QUANTITY_ROOTS)},
                "invariants": {"required_roots": list(calibration.STRICT_INVARIANT_ROOTS)},
            },
        )
