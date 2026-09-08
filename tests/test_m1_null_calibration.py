"""Fake-only contract tests for the fresh-process M1-N0 calibration path."""

from __future__ import annotations

import ast
import copy
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
    assert prepared.run_spec["worker_timeout_seconds"] == 600.0
    request = calibration._prepare_attempt(prepared, prepared.pair_registry["pairs"][0], "A")
    assert request["worker_timeout_seconds"] == 600.0
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


def test_run_releases_full_attempt_trajectories_after_pair_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _module()
    prepared, _payload, _config, _tape = _prepared(calibration, tmp_path, monkeypatch)

    def runner(attempt: dict[str, Any]) -> dict[str, Any]:
        return _attempt_result(calibration, attempt["pair_id"], attempt["side"], 1000 + int(attempt["ordinal"]))

    result = calibration.run_calibration(prepared, process_runner=runner)
    assert len(result["attempts"]) == 40
    assert all("windows" not in item for item in result["attempts"])
    measurement_doc = calibration._load_document(result["pair_measurements_path"])
    assert len(measurement_doc["measurement_records"]) == 20
    assert all("windows" not in item for item in measurement_doc["measurement_records"])


def test_strict_completed_worker_missing_output_sha_is_failed_not_synthesized() -> None:
    calibration = _module()
    request = {"attempt_id": "m1n0-pair-000-A", "pair_id": "m1n0-pair-000", "side": "A"}
    result = {
        "attempt_id": request["attempt_id"],
        "pair_id": request["pair_id"],
        "side": request["side"],
        "status": "completed",
    }
    normalized = calibration._finalize_process_result(result, request, strict=True)
    assert normalized["status"] == "failed"
    assert "output_sha256" not in normalized
    assert "output_sha256" in normalized["error"]


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


def test_pair_validation_requires_exact_raw_terminal_reason_and_authoritative_evidence() -> None:
    calibration = _module()
    pair = {"pair_id": "m1n0-pair-000"}
    left = _attempt_result(calibration, pair["pair_id"], "A", 11)
    right = _attempt_result(calibration, pair["pair_id"], "B", 12)
    raw_evidence = {
        "source": "missing",
        "field": None,
        "raw_reason": None,
        "returned_step": True,
        "terminated": True,
        "truncated": False,
    }
    for attempt in (left, right):
        attempt["terminal"]["raw_termination_reason"] = None
        attempt["terminal"]["raw_termination_evidence"] = deepcopy(raw_evidence)
        attempt["terminal"]["official_evidence"] = deepcopy(raw_evidence)
    assert calibration.validate_pair(pair, {"A": left, "B": right}, parent_pid=1).valid

    changed_raw = deepcopy(right)
    changed_raw["terminal"]["raw_termination_reason"] = "environment_termination"
    changed_raw["terminal"]["raw_termination_evidence"]["raw_reason"] = "environment_termination"
    invalid = calibration.validate_pair(pair, {"A": left, "B": changed_raw}, parent_pid=1)
    assert invalid.valid is False
    assert any("raw termination" in reason for reason in invalid.reasons)

    changed_source = deepcopy(right)
    changed_source["terminal"]["raw_termination_evidence"]["source"] = "step_info"
    changed_source["terminal"]["official_evidence"]["source"] = "step_info"
    invalid = calibration.validate_pair(pair, {"A": left, "B": changed_source}, parent_pid=1)
    assert invalid.valid is False
    assert any("authoritative" in reason or "evidence" in reason for reason in invalid.reasons)


def test_snapshot_contacts_preserves_duplicate_topology_and_canonicalizes_distance_order() -> None:
    calibration = _module()
    left = {
        "invariants": {
            "contacts": [
                ("geom_b", "geom_a", -0.1),
                ("geom_a", "geom_b", -0.2),
                ("geom_c", "geom_a", -0.3),
            ]
        }
    }
    right = {
        "invariants": {
            "contacts": [
                ("geom_a", "geom_c", -0.3),
                ("geom_a", "geom_b", -0.2),
                ("geom_b", "geom_a", -0.1),
            ]
        }
    }
    assert calibration._snapshot_contacts(left) == (
        ("geom_a", "geom_b"),
        ("geom_a", "geom_b"),
        ("geom_a", "geom_c"),
    )
    assert calibration._snapshot_contacts(left) == calibration._snapshot_contacts(right)
    assert calibration._contact_distance_leaves(left["invariants"]["contacts"]) == calibration._contact_distance_leaves(
        right["invariants"]["contacts"]
    )


def test_contact_distance_leaves_include_canonical_identity_and_per_identity_occurrence() -> None:
    calibration = _module()
    values = [
        ("geom_b", "geom_a", -0.1),
        ("geom_a", "geom_b", -0.2),
        ("geom_d", "geom_c", -0.3),
    ]
    leaves = calibration._contact_distance_leaves(values)
    assert set(leaves) == {
        'contacts[["geom_a","geom_b"]][0].distance',
        'contacts[["geom_a","geom_b"]][1].distance',
        'contacts[["geom_c","geom_d"]][0].distance',
    }
    assert leaves['contacts[["geom_a","geom_b"]][0].distance'].tolist() == [-0.2]
    assert leaves['contacts[["geom_a","geom_b"]][1].distance'].tolist() == [-0.1]


def test_pair_validation_rejects_contact_distance_when_contact_identity_multiset_differs() -> None:
    calibration = _module()
    pair = {"pair_id": "m1n0-pair-000"}
    left = _attempt_result(calibration, pair["pair_id"], "A", 11)
    right = _attempt_result(calibration, pair["pair_id"], "B", 12)
    left["windows"]["free_motion"]["snapshots"]["0"]["invariants"]["contacts"] = [
        ("robot_finger", "target_geom", -0.1),
        ("robot_finger", "target_geom", -0.2),
    ]
    right["windows"]["free_motion"]["snapshots"]["0"]["invariants"]["contacts"] = [
        ("robot_finger", "target_geom", -0.1),
    ]
    invalid = calibration.validate_pair(pair, {"A": left, "B": right}, parent_pid=1)
    assert invalid.valid is False
    assert invalid.discrete["contact_identity_exact"] is False


def test_protocol_counters_publish_runtime_reset_provenance_and_post_construction_zeroes() -> None:
    calibration = _module()

    class Adapter:
        reset_provenance = {
            "construction": {
                "operations": {
                    "reset": {"allowed": True, "observed": True, "count": 1},
                    "set_init_state": {"allowed": True, "observed": True, "count": 1},
                    "settle": {"allowed": True, "observed": True, "count": 10},
                },
                "post_construction_forbidden": {
                    "reset": {"allowed": False, "observed": True, "count": 0},
                    "set_init_state": {"allowed": False, "observed": True, "count": 0},
                    "settle": {"allowed": False, "observed": True, "count": 0},
                },
            }
        }
        post_construction_operation_counts = {
            "reset": 0,
            "set_init_state": 0,
            "settle": 0,
            "restore": 0,
            "capture": 0,
            "policy_calls": 0,
            "processor_calls": 0,
            "retry": 0,
            "post_terminal_step": 0,
            "dummy_action": 0,
            "autoreset": 0,
        }

    counters = calibration._protocol_counters(
        Adapter(),
        {
            "actions_executed": 82,
            "step_calls": 82,
            "render_calls": 82,
            "invariant_collections": 82,
            "post_terminal_steps": 0,
        },
        82,
    )
    assert counters["construction_reset_count"] == 1
    assert counters["set_init_state_count"] == 0
    assert counters["settle_count"] == 0
    assert counters["post_construction_forbidden"]["reset"]["count"] == 0
    assert counters["reset_provenance"]["construction"]["operations"]["settle"]["count"] == 10


def test_pair_validation_rejects_post_construction_reset_provenance_even_if_flat_count_is_zero() -> None:
    calibration = _module()
    pair = {"pair_id": "m1n0-pair-000"}
    left = _attempt_result(calibration, pair["pair_id"], "A", 11)
    right = _attempt_result(calibration, pair["pair_id"], "B", 12)
    for attempt in (left, right):
        attempt["reset_provenance"] = {
            "construction": {
                "phase": "construction",
                "operations": {
                    "reset": {"allowed": True, "observed": True, "count": 1},
                    "set_init_state": {"allowed": True, "observed": True, "count": 0},
                    "settle": {"allowed": True, "observed": True, "count": 0},
                },
                "post_construction_forbidden": {
                    name: {"allowed": False, "observed": True, "count": 0}
                    for name in calibration.POST_CONSTRUCTION_FORBIDDEN_OPERATIONS
                },
            }
        }
    right["reset_provenance"]["construction"]["post_construction_forbidden"]["reset"]["count"] = 1
    invalid = calibration.validate_pair(pair, {"A": left, "B": right}, parent_pid=1)
    assert invalid.valid is False
    assert any("post-construction" in reason or "reset provenance" in reason for reason in invalid.reasons)


def test_strict_reset_provenance_requires_source_bound_lazy_records_and_complete_inner_monitoring() -> None:
    calibration = _module()
    source = {
        "path": "external/lerobot/src/lerobot/envs/libero.py",
        "sha256": "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71",
        "lines": "external/lerobot/src/lerobot/envs/libero.py:258-270,339-346",
    }
    records = {
        "outer_reset": {"observed": True, "count": 1, "source": source},
        "inner_reset": {"observed": True, "count": 2, "source": source},
        "set_init_state": {"observed": True, "count": 1, "source": source},
        "settle": {"observed": True, "count": 10, "source": source},
        "dummy_action": {"observed": True, "count": 10, "source": source},
        "post_reset_state": {
            "inner_env_exists": True,
            "num_steps_wait": 10,
            "post_reset_timestep": 10,
            "matches_num_steps_wait": True,
        },
        "source_evidence": {"lerobot_libero": source},
    }
    provenance = {
        "construction": {
            "operations": {
                name: {"allowed": True, "observed": True, "count": count}
                for name, count in (("reset", 1), ("set_init_state", 1), ("settle", 10))
            },
            "authoritative_records": records,
            "post_construction_monitoring": {
                "complete": True,
                "observed": True,
                "required_operations": ["reset", "set_init_state", "step"],
                "targets": ["OffScreenRenderEnv"],
                "installed": [
                    {"owner": "OffScreenRenderEnv", "operation": "reset"},
                    {"owner": "OffScreenRenderEnv", "operation": "set_init_state"},
                    {"owner": "OffScreenRenderEnv", "operation": "step"},
                ],
            },
            "post_construction_forbidden": {
                name: {"allowed": False, "observed": True, "count": 0}
                for name in calibration.POST_CONSTRUCTION_FORBIDDEN_OPERATIONS
            },
        }
    }
    assert calibration._reset_provenance_errors(provenance, strict=True) == []

    missing_record = deepcopy(provenance)
    missing_record["construction"]["authoritative_records"]["inner_reset"] = {
        "observed": False,
        "count": None,
    }
    errors = calibration._reset_provenance_errors(missing_record, strict=True)
    assert any("authoritative" in error or "construction record" in error for error in errors)

    incomplete_monitor = deepcopy(provenance)
    incomplete_monitor["construction"]["post_construction_monitoring"]["complete"] = False
    errors = calibration._reset_provenance_errors(incomplete_monitor, strict=True)
    assert any("monitor" in error for error in errors)


def test_protocol_counter_publication_includes_every_forbidden_operation() -> None:
    calibration = _module()

    class Adapter:
        pass

    counters = calibration._protocol_counters(
        Adapter(),
        {
            "actions_executed": 82,
            "step_calls": 82,
            "render_calls": 82,
            "invariant_collections": 82,
            "post_terminal_steps": 0,
        },
        82,
    )
    assert set(calibration.FORBIDDEN_PROTOCOL_FIELDS) <= set(counters)
    assert all(counters[name] == 0 for name in calibration.FORBIDDEN_PROTOCOL_FIELDS)


def test_default_process_runner_reads_dedicated_result_when_stdout_has_runtime_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _module()
    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "output_root": str(tmp_path),
        "python": sys.executable,
        "worker_timeout_seconds": 600,
    }
    worker_payload = {"status": "failed", "error": "preserved worker failure"}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result_path = Path(command[command.index("--result") + 1])
        calibration.write_json_atomic(result_path, worker_payload)
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="Creating LIBERO envs | suites=['libero_spatial']\n",
            stderr="construction diagnostic\n",
        )

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    result = calibration._default_process_runner(attempt)
    assert result["status"] == "failed"
    assert result["error"] == "preserved worker failure"
    assert result["worker_transport"]["result"]["path"].endswith(".result.json")
    assert result["worker_transport"]["stdout"]["size"] > 0
    assert result["worker_transport"]["stderr"]["size"] > 0


def test_default_process_runner_uses_real_worker_result_file_and_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _module()
    prepared, _payload, _config, _tape = _prepared(calibration, tmp_path, monkeypatch)
    pair = prepared.pair_registry["pairs"][0]
    attempt = calibration._prepare_attempt(prepared, pair, "A")
    result = calibration._default_process_runner(attempt)
    assert result["status"] == "failed"
    transport = result["worker_transport"]
    assert transport["protocol"] == "dedicated_result_file_v1"
    assert transport["returncode"] == 1
    assert transport["timed_out"] is False
    assert transport["result"]["exists"] is True
    assert transport["result"]["sha256"]
    stdout = Path(transport["stdout"]["path"]).read_text(encoding="utf-8")
    assert "M1N0_RESULT " in stdout


def test_default_process_runner_reconstructs_sidecar_result_with_nonzero_framed_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _module()
    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "output_root": str(tmp_path),
        "python": sys.executable,
        "worker_timeout_seconds": 600,
    }
    payload = {"status": "failed", "error": "framed worker error", "array": np.arange(6, dtype=np.float32)}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result_path = Path(command[command.index("--result") + 1])
        calibration.write_json_atomic(result_path, payload)
        return subprocess.CompletedProcess(
            command,
            7,
            stdout="worker diagnostic\nM1N0_RESULT sidecar\n",
            stderr="worker stderr\n",
        )

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    result = calibration._default_process_runner(attempt)
    assert result["status"] == "failed"
    assert result["error"] == "framed worker error"
    np.testing.assert_array_equal(result["array"], payload["array"])
    assert result["worker_transport"]["returncode"] == 7
    assert result["worker_transport"]["result"]["sha256"]
    assert result["worker_transport"]["stdout"]["size"] > 0
    assert result["worker_transport"]["stderr"]["size"] > 0


def test_default_process_runner_timeout_persists_partial_logs_and_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _module()
    attempt = {
        "attempt_id": "m1n0-pair-000-A",
        "pair_id": "m1n0-pair-000",
        "side": "A",
        "output_root": str(tmp_path),
        "python": sys.executable,
        "worker_timeout_seconds": 600,
    }
    calls: list[float | None] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    result = calibration._default_process_runner(attempt)
    assert calls == [600.0]
    assert result["status"] == "failed"
    assert "timed out" in result["error"]
    transport = result["worker_transport"]
    assert transport["timed_out"] is True
    assert transport["timeout_seconds"] == 600.0
    assert Path(transport["stdout"]["path"]).read_text(encoding="utf-8") == "partial stdout"
    assert Path(transport["stderr"]["path"]).read_text(encoding="utf-8") == "partial stderr"


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
            return {"step": len(self.steps)}, 0.0, terminal, False, {
                "termination_reason": "predicate_transition",
                "is_success": terminal,
            } if terminal else {}
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


def test_terminal_without_step_reason_derives_frozen_predicate_transition_and_keeps_m0_reason() -> None:
    calibration = _module()

    class FakeAdapter:
        # M0 records the wrapper's generic terminal reason.  The null worker
        # must retain it while deriving the frozen semantic reason.
        last_termination_reason = "environment_termination"

        def __init__(self) -> None:
            self.steps = 0

        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.steps += 1
            return {"observation": {"pixels": {"image": np.zeros((2, 2, 3), dtype=np.uint8)}},
                    "terminated": self.steps == 82,
                    "truncated": False,
                    "info": {"is_success": self.steps == 82}}, 0.0, self.steps == 82, False, {"is_success": self.steps == 82}

        def collect_invariants(self) -> dict[str, Any]:
            return _snapshot(self.steps)["invariants"]

        def render_rgb(self) -> np.ndarray:
            return _snapshot(self.steps)["renderer"]

        def close(self) -> None:
            return None

    result = calibration.execute_attempt(
        {
            "attempt_id": "m1n0-pair-000-A",
            "pair_id": "m1n0-pair-000",
            "side": "A",
            "trace_id": "trace",
            "config": {
                "task": dict(calibration.TASK),
                "terminal_contract": dict(calibration.DEFAULT_TERMINAL_CONTRACT),
            },
            "registry": {
                "traces": [{
                    "trace_id": "trace",
                    "windows": [{
                        "window_id": "predicate_transition",
                        "regimes": ["predicate_transition"],
                        "capture_offset": 81,
                        "continuation_horizon": 1,
                        "source_coverage": {"regimes": ["predicate_transition"]},
                    }],
                }],
            },
            "tape": np.zeros((82, 7), dtype=np.float32),
        },
        adapter_factory=lambda _config: FakeAdapter(),
    )
    assert result["status"] == "completed"
    terminal = result["terminal"]
    assert terminal["termination_reason"] == "predicate_transition"
    assert terminal["raw_termination_reason"] == "environment_termination"
    assert terminal["official_evidence"]["raw_reason"] == "environment_termination"
    assert terminal["official_evidence"]["semantic_derivation"]["predicate_transition"] is True


def test_close_exception_marks_completed_attempt_failed_and_publishes_close_evidence() -> None:
    calibration = _module()

    class FailingCloseAdapter:
        def __init__(self) -> None:
            self.steps = 0

        def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self.steps += 1
            terminal = self.steps == 82
            return _snapshot(self.steps), 0.0, terminal, False, {
                "termination_reason": "predicate_transition" if terminal else None,
                "is_success": terminal,
            }

        def collect_invariants(self) -> dict[str, Any]:
            return _snapshot(self.steps)["invariants"]

        def render_rgb(self) -> np.ndarray:
            return _snapshot(self.steps)["renderer"]

        def close(self) -> None:
            raise RuntimeError("close exploded")

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
        adapter_factory=lambda _config: FailingCloseAdapter(),
    )
    assert result["status"] == "failed"
    assert result["close_evidence"]["attempted"] is True
    assert result["close_evidence"]["success"] is False
    assert result["close_evidence"]["source"] == "adapter.close"
    assert "close exploded" in result["close_evidence"]["error"]
    assert "close exploded" in result["error"]


def test_renderer_rejects_non_agentview_direct_output_and_unknown_official_rgb_key() -> None:
    calibration = _module()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(calibration.NullCalibrationError, match="agentview"):
        calibration._renderer_images(
            {"renderer": image}, camera="sideview", configured_key="side_rgb", strict=True
        )
    with pytest.raises(calibration.NullCalibrationError, match="camera|RGB|unknown"):
        calibration._renderer_images(
            {"raw_observation": {"pixels": {"mystery_rgb": image}}}, strict=True
        )


def test_renderer_maps_official_rgb_keys_to_true_views() -> None:
    calibration = _module()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    images = calibration._renderer_images(
        {
            "renderer": image,
            "raw_observation": {"pixels": {"image": image, "image2": image}},
        },
        camera="agentview",
        configured_key="render_rgb",
        strict=True,
    )
    assert calibration._renderer_camera_for_key("render_rgb") == "agentview"
    assert calibration._renderer_camera_for_key("pixels.image") == "agentview"
    assert calibration._renderer_camera_for_key("pixels.image2") == "robot0_eye_in_hand"


def test_contact_and_carried_evidence_may_begin_at_h1_but_is_required_in_window() -> None:
    calibration = _module()
    contact_h0 = _snapshot(43, contact=False)
    contact_h1 = _snapshot(44, contact=True)
    contact = calibration._independent_evidence(
        "contact", contact_h0, contact_h1, action=np.zeros(7, dtype=np.float32), history=[contact_h0, contact_h1]
    )
    assert contact["contact"]["present"] is True

    carried_h0 = _snapshot(54, contact=True)
    carried_h1 = _snapshot(55, contact=True)
    carried_h1["invariants"]["objects"]["target"]["body_pos"] = np.asarray(
        [0.25, 0.0, 0.0], dtype=np.float64
    )
    carried = calibration._independent_evidence(
        "carried", carried_h0, carried_h1, action=np.zeros(7, dtype=np.float32), history=[carried_h0, carried_h1]
    )
    assert carried["carried"]["present"] is True


def test_contact_evidence_is_bound_to_frozen_target_object() -> None:
    calibration = _module()
    snapshot = _snapshot(44, contact=True)
    snapshot["invariants"]["objects"] = {
        "akita_black_bowl_1": {
            "body_pos": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "body_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "joints": {},
        },
        "other_object": {
            "body_pos": np.asarray([0.1, 0.0, 0.0], dtype=np.float64),
            "body_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "joints": {},
        },
    }
    snapshot["invariants"]["contacts"] = [("robot_finger", "other_object_geom", 0.0)]
    assert calibration._actual_robot_object_contact(snapshot) is False


def test_window_evidence_checks_continuation_for_h1_onset() -> None:
    calibration = _module()
    h0 = _snapshot(43, contact=False)
    h1 = _snapshot(44, contact=True)
    requirements = calibration._window_regime_requirements("contact", [h0, h1])
    assert requirements["contact"] is True


def test_pair_validation_allows_contact_on_h1_while_retaining_window_diagnostics() -> None:
    calibration = _module()
    pair_id = "m1n0-pair-000"
    left = _attempt_result(calibration, pair_id, "A", 11)
    right = _attempt_result(calibration, pair_id, "B", 12)
    for attempt in (left, right):
        window = attempt["windows"]["contact"]
        window["snapshots"]["0"] = _snapshot(43, contact=False)
        window["snapshots"]["1"] = _snapshot(44, contact=True)
        window["evidence"] = calibration._independent_evidence(
            "contact",
            window["snapshots"]["0"],
            window["snapshots"]["1"],
            action=np.ones(7, dtype=np.float32),
            history=[window["snapshots"]["0"], window["snapshots"]["1"]],
        )
    validated = calibration.validate_pair(
        {"pair_id": pair_id, "trace_id": "trace"},
        {"A": left, "B": right},
        parent_pid=1,
    )
    assert validated.valid is True


def test_panda_gripper_positive_command_and_physical_closed_state_are_closed() -> None:
    calibration = _module()
    negative = np.zeros(7, dtype=np.float32)
    negative[-1] = -1.0
    positive = negative.copy()
    positive[-1] = 1.0
    assert calibration._close_command(_snapshot(49), negative) is False
    assert calibration._close_command(_snapshot(49), positive) is True
    assert calibration._close_command(_snapshot(49), None) is True
    opened = _snapshot(49)
    opened["invariants"]["gripper"]["current_action"] = np.asarray([1.0, -1.0])
    assert calibration._close_command(opened, None) is False


def test_strict_terminal_requires_predicate_transition_proof_but_accepts_missing_raw_reason() -> None:
    calibration = _module()
    terminal = dict(calibration.DEFAULT_TERMINAL_CONTRACT)
    terminal["official_evidence"] = {
        "source": "missing",
        "returned_step": True,
        "terminated": True,
        "truncated": False,
        "raw_reason": None,
        "semantic_derivation": {
            "derived": True,
            "predicate_transition": True,
            "frozen_terminal_step": True,
            "success": True,
            "predicate_evidence": {
                "available": True,
                "previous": [False],
                "current": [True],
                "predicate_transition": True,
            },
        },
    }
    calibration._validate_terminal(terminal, calibration.DEFAULT_TERMINAL_CONTRACT, require_official_evidence=True)
    terminal["official_evidence"].pop("semantic_derivation")
    with pytest.raises(calibration.ProtocolError, match="semantic|predicate"):
        calibration._validate_terminal(
            terminal,
            calibration.DEFAULT_TERMINAL_CONTRACT,
            require_official_evidence=True,
        )


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
    assert any(
        quantity.startswith("observation.")
        for regime in envelopes.physics["groups"].values()
        for quantity in regime
    )
    semantic = envelopes.discrete
    assert semantic["categorical_gate"] is True
    assert semantic["contact_identity_exact"] is True


def test_grouped_envelopes_accept_parent_compact_measurements_without_attempt_windows() -> None:
    calibration = _module()
    discrete = {
        name: True
        for name in (
            "categorical_gate",
            "contact_identity_exact",
            "predicate_exact",
            "success_exact",
            "done_termination_exact",
            "terminal_timing_exact",
            "gripper_exact",
            "action_exact",
            "observation_exact",
        )
    }
    evidence = {
        name: {"present": True, "source": "attempt_snapshots_and_frozen_actions"}
        for name in ("contact", "grasp", "carried")
    }
    compact_results = []
    for index in range(20):
        pair_id = f"m1n0-pair-{index:03d}"
        physics_samples = [
            {
                "pair_id": pair_id,
                "trace_id": "trace",
                "regime": regime,
                "quantity": "qpos",
                "horizon": 0,
                "difference": 0.0,
                "independent_evidence": evidence,
            }
            for regime in calibration.REGIME_NAMES
        ]
        renderer_samples = [
            {
                "pair_id": pair_id,
                "trace_id": "trace",
                "camera": "agentview",
                "key": "render_rgb",
                "regime": regime,
                "horizon": 0,
                "duplicate_controls": [
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([0.0], dtype=np.float32),
                ],
                "metrics": {"differing_pixel_count": 0, "max_abs": 0.0, "mean_abs": 0.0},
            }
            for regime in calibration.REGIME_NAMES
        ]
        compact_results.append(
            calibration.PairValidation(
                pair_id,
                "trace",
                True,
                (),
                {},
                discrete,
                {"physics_samples": physics_samples, "renderer_samples": renderer_samples},
            )
        )
    envelopes = calibration.build_envelopes(
        compact_results,
        required_pair_ids=[f"m1n0-pair-{index:03d}" for index in range(20)],
    )
    assert envelopes.physics["coverage"]["n_pairs"] == 20
    assert envelopes.renderer["coverage"]["n_pairs"] == 20


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


def test_worker_rejects_config_drift_before_constructing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration = _module()
    prepared, payload, config_path, _tape = _prepared(calibration, tmp_path, monkeypatch)
    pair = prepared.pair_registry["pairs"][0]
    request = calibration._prepare_attempt(prepared, pair, "A")
    job_path = tmp_path / "job.json"
    result_path = tmp_path / "result.json"
    calibration.write_json_atomic(job_path, request)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pair_count"] = 19
    config_path.write_text(calibration.canonical_json(config) + "\n", encoding="utf-8")
    monkeypatch.setattr(calibration, "_load_verified_registry", lambda _path: deepcopy(payload))
    constructed: list[bool] = []
    monkeypatch.setattr(
        calibration,
        "_construct_adapter",
        lambda _config: constructed.append(True),
    )

    return_code = calibration._worker_main(job_path, result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert return_code != 0
    assert constructed == []
    assert "config" in result["error"].lower()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_run_spec_sha256", "0" * 64, "run specification"),
        ("expected_registry_sha256", "0" * 64, "registry"),
        ("ordinal", 99, "schedule"),
        ("action_tape_sha256", "0" * 64, "tape"),
    ],
)
def test_worker_rejects_run_registry_schedule_and_tape_drift_before_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    calibration = _module()
    prepared, payload, _config_path, _tape = _prepared(calibration, tmp_path, monkeypatch)
    pair = prepared.pair_registry["pairs"][0]
    request = calibration._prepare_attempt(prepared, pair, "A")
    request[field] = value
    job_path = tmp_path / f"{field}.job.json"
    result_path = tmp_path / f"{field}.result.json"
    calibration.write_json_atomic(job_path, request)
    monkeypatch.setattr(calibration, "_load_verified_registry", lambda _path: deepcopy(payload))
    constructed: list[bool] = []
    monkeypatch.setattr(calibration, "_construct_adapter", lambda _config: constructed.append(True))

    return_code = calibration._worker_main(job_path, result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert return_code != 0
    assert constructed == []
    assert message in result["error"].lower()


def test_array_publication_uses_content_addressed_npy_references(tmp_path: Path) -> None:
    calibration = _module()
    payload = {"nested": {"array": np.arange(6, dtype=np.float32).reshape(2, 3)}}
    published, records = calibration._materialize_array_artifacts(payload, tmp_path / "arrays")
    reference = published["nested"]["array"]
    assert reference["__ndarray_ref__"] is True
    assert "data" not in reference
    assert records and records[0]["sha256"] == reference["sha256"]
    artifact = Path(records[0]["path"])
    assert calibration.verify_artifact_hash(artifact, reference["sha256"])
    np.testing.assert_array_equal(np.load(artifact, allow_pickle=False), payload["nested"]["array"])


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
            return _snapshot(self.steps)["invariants"]

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


def test_quantity_selection_keeps_contact_distances_as_floating_leaves() -> None:
    calibration = _module()
    snapshot = {
        "invariants": {
            "contacts": [("robot_finger", "target_geom", -0.0125)],
        }
    }
    values = calibration._selected_numeric_leaves(snapshot, ("contacts",))
    assert set(values) == {'contacts[["robot_finger","target_geom"]][0].distance'}
    assert values['contacts[["robot_finger","target_geom"]][0].distance'].dtype == np.dtype("float64")
    assert values['contacts[["robot_finger","target_geom"]][0].distance'].tolist() == [-0.0125]


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


def _r1_base_config(state_replay: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "state_replay",
        "runtime": {
            "environment": copy.deepcopy(state_replay.FROZEN_RUNTIME_ENVIRONMENT),
            "renderer": {
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": "8",
                "expected_gl": {
                    "vendor": "Mesa/X.org",
                    "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
                    "version": "3.1 Mesa 21.1.5",
                },
            },
        },
    }


def _r1_overlay() -> dict[str, Any]:
    return {
        "policy_compute_device": "not_applicable_no_policy",
        "physical_compute_device_id": None,
        "renderer_backend": "egl",
        "renderer_device_id": "0",
        "runtime": {
            "renderer": {
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "MUJOCO_EGL_DEVICE_ID": "0",
                "expected_gl": {
                    "vendor": "Mesa/X.org",
                    "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
                    "version": "3.1 Mesa 21.1.5",
                },
            }
        },
    }


def test_r1_runtime_config_validates_unchanged_base_then_applies_only_renderer_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    validated: list[dict[str, object]] = []

    def fake_validate(value: dict[str, object]) -> dict[str, object]:
        validated.append(value)
        return value

    monkeypatch.setattr(state_replay, "validate_config", fake_validate)
    overlay = _r1_overlay()
    merged = calibration._r1_runtime_config(base, overlay)
    assert validated == [base]
    assert base["runtime"]["renderer"]["MUJOCO_EGL_DEVICE_ID"] == "8"
    assert merged["runtime"]["renderer"]["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert merged["runtime"]["environment"]["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert merged["policy_compute_device"] == "not_applicable_no_policy"
    assert merged["physical_compute_device_id"] is None


def test_r1_runtime_config_uses_non_none_validator_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    normalized = copy.deepcopy(base)
    normalized["validated_marker"] = "returned_mapping"
    monkeypatch.setattr(state_replay, "validate_config", lambda _value: normalized)
    merged = calibration._r1_runtime_config(base, _r1_overlay())
    assert merged["validated_marker"] == "returned_mapping"
    assert "validated_marker" not in base


def test_r1_runtime_config_uses_base_copy_when_validator_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    monkeypatch.setattr(state_replay, "validate_config", lambda _value: None)
    merged = calibration._r1_runtime_config(base, _r1_overlay())
    assert merged["name"] == base["name"]
    assert merged is not base


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda overlay: overlay.update({"unexpected": True}), "overlay"),
        (lambda overlay: overlay.update({"renderer_backend": "osmesa"}), "renderer_backend"),
        (lambda overlay: overlay.update({"policy_compute_device": "cpu"}), "policy_compute_device"),
        (lambda overlay: overlay.update({"physics": {"timestep": 0.01}}), "overlay"),
        (
            lambda overlay: overlay["runtime"].update({"controller_class": "changed"}),
            "runtime",
        ),
    ],
)
def test_r1_runtime_config_rejects_non_renderer_overlay_changes(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    monkeypatch.setattr(state_replay, "validate_config", lambda value: value)
    overlay = _r1_overlay()
    mutation(overlay)
    with pytest.raises(calibration.ProvenanceError, match=message):
        calibration._r1_runtime_config(base, overlay)


def test_r1_runtime_config_rejects_renderer_fields_inconsistent_with_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    monkeypatch.setattr(state_replay, "validate_config", lambda value: value)
    overlay = _r1_overlay()
    overlay["runtime"]["renderer"]["MUJOCO_EGL_DEVICE_ID"] = "1"
    with pytest.raises(calibration.ProvenanceError, match="renderer_device_id|MUJOCO_EGL_DEVICE_ID"):
        calibration._r1_runtime_config(base, overlay)


@pytest.mark.parametrize("field", ["vendor", "renderer", "version"])
def test_r1_runtime_config_requires_exact_registered_gl_identity(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    monkeypatch.setattr(state_replay, "validate_config", lambda value: value)
    overlay = _r1_overlay()
    overlay["runtime"]["renderer"]["expected_gl"][field] += " changed"
    with pytest.raises(calibration.ProvenanceError, match="expected_gl"):
        calibration._r1_runtime_config(base, overlay)


def test_r1_runtime_config_does_not_mutate_base_overlay_or_frozen_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    overlay = _r1_overlay()
    base_before = copy.deepcopy(base)
    overlay_before = copy.deepcopy(overlay)
    frozen_before = copy.deepcopy(state_replay.FROZEN_RUNTIME_ENVIRONMENT)
    monkeypatch.setattr(state_replay, "validate_config", lambda value: value)
    calibration._r1_runtime_config(base, overlay)
    assert base == base_before
    assert overlay == overlay_before
    assert state_replay.FROZEN_RUNTIME_ENVIRONMENT == frozen_before


def test_official_runtime_config_keeps_legacy_non_strict_behavior() -> None:
    calibration = _module()
    legacy = {"strict_runtime_contract": False, "runtime": {"include_policy": False}}
    resolved = calibration._official_runtime_config(legacy)
    assert resolved == legacy
    assert resolved is not legacy


@pytest.mark.parametrize(
    "mutation",
    [
        lambda merged: merged.update({"task": {"task_id": 1}}),
        lambda merged: merged.update({"physics": {"timestep": 0.01}}),
        lambda merged: merged.update({"paths": {"cpu_python": "changed"}}),
        lambda merged: merged.update({"pins": {"checkpoint": "changed"}}),
        lambda merged: merged["runtime"].update({"offline": False}),
        lambda merged: merged["runtime"].update({"controller_class": "changed"}),
        lambda merged: merged["runtime"]["environment"].update({"HF_HOME": "changed"}),
        lambda merged: merged["runtime"]["environment"].update({"EXTRA_CACHE": "changed"}),
    ],
)
def test_r1_merged_runtime_validator_rejects_every_non_renderer_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    monkeypatch.setattr(state_replay, "validate_config", lambda value: value)
    merged = calibration._r1_runtime_config(base, _r1_overlay())
    mutation(merged)
    with pytest.raises(calibration.ProvenanceError, match="R1 merged|allowed"):
        calibration._validate_r1_merged_runtime(base, merged)


def _mapping_leaf_differences(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[str] = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_mapping_leaf_differences(left[key], right[key], path))
        return differences
    return set() if left == right else {prefix}


def test_r1_runtime_config_real_frozen_base_has_exact_allowlisted_diff() -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = state_replay.load_config(ROOT / "configs/m1/state_replay.yaml")
    assert base["paths"]["config_sha256"] == (
        "730aff4a41fd91fb837102ca5f363a4a142bf7010140f5176940700f1d1fd5f0"
    )
    validated = state_replay.validate_config(base)
    base_before = copy.deepcopy(base)
    merged = calibration._r1_runtime_config(base, _r1_overlay())
    calibration._validate_r1_merged_runtime(validated, merged)
    assert base == base_before
    assert _mapping_leaf_differences(validated, merged) == {
        "policy_compute_device",
        "physical_compute_device_id",
        "renderer_backend",
        "renderer_device_id",
        "runtime.environment.MUJOCO_EGL_DEVICE_ID",
        "runtime.renderer.MUJOCO_EGL_DEVICE_ID",
    }


def test_r1_runtime_config_does_not_apply_environment_or_construct_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = _module()
    import scripts.m1_state_replay as state_replay

    base = _r1_base_config(state_replay)
    monkeypatch.setattr(state_replay, "validate_config", lambda value: value)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("R1 pure overlay path dispatched a runtime side effect")

    monkeypatch.setattr(state_replay, "_apply_runtime_environment", forbidden)
    monkeypatch.setattr(state_replay, "build_official_runtime", forbidden)
    monkeypatch.setattr(state_replay, "RuntimeAdapter", forbidden)
    calibration._r1_runtime_config(base, _r1_overlay())


def test_r1_merged_runtime_sha_is_stable_and_differs_from_legacy() -> None:
    calibration = _module()
    legacy = {
        "runtime": {
            "environment": {"MUJOCO_EGL_DEVICE_ID": "8"},
            "renderer": {"MUJOCO_EGL_DEVICE_ID": "8"},
        }
    }
    r1 = copy.deepcopy(legacy)
    r1["runtime"]["environment"]["MUJOCO_EGL_DEVICE_ID"] = "0"
    r1["runtime"]["renderer"]["MUJOCO_EGL_DEVICE_ID"] = "0"
    assert calibration._r1_merged_runtime_sha256(r1) == calibration._r1_merged_runtime_sha256(copy.deepcopy(r1))
    assert calibration._r1_merged_runtime_sha256(r1) != calibration._r1_merged_runtime_sha256(legacy)
    non_renderer_drift = copy.deepcopy(r1)
    non_renderer_drift["runtime"]["offline"] = False
    assert calibration._r1_merged_runtime_sha256(non_renderer_drift) != calibration._r1_merged_runtime_sha256(r1)
