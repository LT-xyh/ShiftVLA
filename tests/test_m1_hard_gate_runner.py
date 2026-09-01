"""Fake-only contract tests for the bounded M1 hard-gate orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import ast
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest


def _module():
    import scripts.m1_hard_gate_runner as runner

    return runner


def _hard_gate():
    import scripts.m1_hard_gate as hard_gate

    return hard_gate


def _trace(trace_id: str, *, actions: int = 8, root: Path | None = None) -> Any:
    hard_gate = _hard_gate()
    tape = np.zeros((actions, 7), dtype=np.float32)
    digest = hashlib.sha256(tape.tobytes(order="C")).hexdigest()
    artifact_root = root or Path("/tmp") / "m1-hard-gate-runner-tests"
    artifact_root.mkdir(parents=True, exist_ok=True)
    source_file = artifact_root / f"{trace_id}.json"
    parent_file = artifact_root / f"{trace_id}-run.json"
    terminal_file = artifact_root / f"{trace_id}-terminal.json"
    source_file.write_text("{}\n", encoding="utf-8")
    parent_file.write_text("{}\n", encoding="utf-8")
    terminal_file.write_text("{}\n", encoding="utf-8")
    return hard_gate.M0Trace(
        episode_id=trace_id,
        suite="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=2027,
        task_name="frozen-task",
        actions=tape,
        source_file=str(source_file),
        source_file_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        parent_run_manifest=str(parent_file),
        parent_run_manifest_sha256=hashlib.sha256(parent_file.read_bytes()).hexdigest(),
        parent_terminal_manifest=str(terminal_file),
        parent_terminal_manifest_sha256=hashlib.sha256(terminal_file.read_bytes()).hexdigest(),
        action_sha256=digest,
        metadata={"source_coverage": {"regimes": ["free_motion"]}},
    )


def _coverage(regime: str, *, offset: int = 1, horizon: int = 10) -> dict[str, Any]:
    return {
        "regimes": [regime],
        "capture_offset": offset,
        "continuation_horizon": horizon,
        "event": f"source:{regime}",
        "evidence": {"state": "source-only"},
    }


def _primary_and_supplements(runner: Any) -> tuple[list[str], list[dict[str, Any]]]:
    regimes = list(_hard_gate().REGIME_NAMES)
    primary = [f"primary-{regimes[0]}"]
    supplements = [
        {
            "source": f"supplement-{regime}",
            "preregistered": True,
            "source_coverage": _coverage(regime),
        }
        for regime in regimes[1:]
    ]
    return primary, supplements


def test_source_scan_finishes_before_supplements_and_registry_freeze(tmp_path: Path) -> None:
    runner = _module()
    primary, supplements = _primary_and_supplements(runner)
    events: list[tuple[str, str]] = []
    traces = {
        primary[0]: _trace(primary[0], actions=12, root=tmp_path),
        **{item["source"]: _trace(item["source"], actions=12, root=tmp_path) for item in supplements},
    }

    def loader(source: str) -> Any:
        events.append(("load", source))
        return traces[source]

    def classifier(trace: Any) -> dict[str, Any]:
        events.append(("classify", trace.episode_id))
        if trace.episode_id == primary[0]:
            return _coverage("free_motion")
        regime = trace.episode_id.removeprefix("supplement-")
        return _coverage(regime)

    original_write = runner._load_hard_gate().write_frozen_registry

    def write(*args: Any, **kwargs: Any) -> Any:
        events.append(("freeze", str(args[0] if args else kwargs["path"])))
        return original_write(*args, **kwargs)

    runner._load_hard_gate().write_frozen_registry = write
    try:
        prepared = runner.prepare_registry(
            primary_sources=primary,
            preregistered_supplements=supplements,
            registry_path=tmp_path / "registry.json",
            trace_loader=loader,
            source_classifier=classifier,
            tape_provenance={
                "algorithm": "test",
                "dtype": "float32",
                    "shape": [12, 7],
                    "sha256": hashlib.sha256(
                        np.zeros((12, 7), dtype=np.float32).tobytes()
                ).hexdigest(),
            },
        )
    finally:
        runner._load_hard_gate().write_frozen_registry = original_write

    assert prepared.registry_path == tmp_path / "registry.json"
    supplement_loads = [source for kind, source in events if kind == "load" and source.startswith("supplement-")]
    assert supplement_loads == [item["source"] for item in supplements]
    assert max(index for index, item in enumerate(events) if item[0] == "classify" and item[1] == primary[0]) < min(
        index for index, item in enumerate(events) if item[0] == "load" and item[1].startswith("supplement-")
    )
    assert events[-1][0] == "freeze"


@dataclass
class _FakeAdapter:
    events: list[tuple[str, Any]]
    fail_restore: bool = False

    def __post_init__(self) -> None:
        self.restore_calls = 0
        self.step_calls = 0
        self.last_terminated = False
        self.last_truncated = False

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.events.append(("step", self.step_calls))
        self.step_calls += 1
        return ({"obs": self.step_calls}, 0.0, False, False, {})

    def capture(self, *, source_step: int | None = None) -> dict[str, Any]:
        self.events.append(("capture", source_step))
        return {"source_step": source_step}

    def collect_invariants(self) -> dict[str, Any]:
        return {"identity": {"step": self.step_calls}, "observation": {"step": self.step_calls}}

    def render_rgb(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def audit_runtime_state(self) -> dict[str, Any]:
        return {"unknown_paths": [], "pass": True}

    def restore(self, state: Any, *, static_reference: Any = None) -> dict[str, Any]:
        self.events.append(("restore", state["source_step"]))
        self.restore_calls += 1
        if self.fail_restore:
            raise RuntimeError("restore failure")
        return {
            "static": {
                "gate": {"pass": True},
                "observation_gate": {"pass": True},
            },
            "invariants": static_reference or {},
            "pass": True,
        }

    def close(self) -> None:
        self.events.append(("close", None))


def _registry_for_run(tmp_path: Path, *, horizon: int = 10) -> Path:
    runner = _module()
    hard_gate = _hard_gate()
    traces = [_trace(f"trace-{regime}", actions=12, root=tmp_path) for regime in hard_gate.REGIME_NAMES]
    per_trace = {
        trace.episode_id: {
            "regimes": [regime],
            "capture_offset": 1,
            "continuation_horizon": horizon,
            "source_coverage": _coverage(regime, horizon=horizon),
        }
        for trace, regime in zip(traces, hard_gate.REGIME_NAMES)
    }
    tape_path = tmp_path / "tape.npy"
    np.save(tape_path, np.zeros((12, 7), dtype=np.float32), allow_pickle=False)
    return hard_gate.write_frozen_registry(
        tmp_path / "registry.json",
        traces=traces,
        tape_provenance={
            "algorithm": "test",
            "dtype": "float32",
            "shape": [12, 7],
            "sha256": hashlib.sha256(np.zeros((12, 7), dtype=np.float32).tobytes()).hexdigest(),
            "path": str(tape_path),
        },
        per_trace=per_trace,
        selection={"trace_ids": [trace.episode_id for trace in traces]},
    )


def _samples(runner: Any, selected: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hard_gate = _hard_gate()
    physics: list[dict[str, Any]] = []
    renderer: list[dict[str, Any]] = []
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    for regime_index, regime in enumerate(hard_gate.REGIME_NAMES):
        trace_id = selected[regime_index]
        for index in range(20):
            evidence = {
                "contact_evidence": {"source": "contacts", "present": regime == "contact"},
                "grasp_evidence": {"source": "gripper", "present": regime == "grasp"},
                "carried_evidence": {"source": "object-motion", "present": regime == "carried"},
            }
            physics.append(
                {
                    "pair_id": f"{regime}-{index}",
                    "trace_id": trace_id,
                    "regime": regime,
                    "quantity": "qpos",
                    "horizon": 1,
                    "value_a": [0.0],
                    "value_b": [0.0],
                    **evidence,
                }
            )
            for horizon in range(11):
                control = np.zeros(7, dtype=np.float32)
                renderer.append(
                    {
                        "pair_id": f"{trace_id}-renderer-{index}",
                        "trace_id": trace_id,
                        "camera": "agentview",
                        "key": "render_rgb",
                        "regime": regime,
                        "horizon": horizon,
                        "duplicate_controls": [control.copy(), control.copy()],
                        "rgb_a": image.copy(),
                        "rgb_b": image.copy(),
                    }
                )
    return physics, renderer


def test_run_freezes_and_requires_unique_fresh_pid_and_trace(tmp_path: Path) -> None:
    runner = _module()
    registry = _registry_for_run(tmp_path)
    events: list[tuple[str, Any]] = []
    adapters: list[_FakeAdapter] = []

    def builder(config: Any) -> _FakeAdapter:
        events.append(("build", None))
        adapter = _FakeAdapter(events)
        adapters.append(adapter)
        return adapter

    selected = [f"trace-{regime}" for regime in _hard_gate().REGIME_NAMES]
    physics, renderer = _samples(runner, selected)
    pids = iter(range(1001, 1022))

    def process_runner(job: dict[str, Any]) -> dict[str, Any]:
        events.append(("fresh", job["trace_id"]))
        result = {
            "pid": next(pids),
            "trace_id": job["trace_id"],
            "stage_id": job["stage_id"],
            "physics_static_identity": True,
            "observation_static_identity": True,
            "dynamic_pass": True,
            "renderer_pass": True,
            "pass": True,
            "no_retry": True,
            "retry_count": 0,
        }
        result["output_sha256"] = runner._sha256_bytes(
            runner._canonical_json(result).encode("utf-8")
        )
        return result

    result = runner.run_authoritative(
        registry,
        config={},
        trace_loader=lambda source: _trace(Path(str(source)).stem, actions=12, root=tmp_path),
        environment_builder=builder,
        subprocess_runner=process_runner,
        null_physics_samples=physics,
        null_renderer_samples=renderer,
        manifest_path=tmp_path / "manifest.json",
    )

    assert result["gates"]["corpus_regime_coverage"] is True
    assert result["gates"]["fresh_process_dynamic"] is True
    assert len(result["fresh_process"]) == 21
    assert len({item["pid"] for item in result["fresh_process"]}) == 21
    assert events.index(("build", None)) < events.index(("fresh", selected[0]))
    assert (tmp_path / "manifest.json").is_file()


def test_restore_failure_is_not_retried(tmp_path: Path) -> None:
    runner = _module()
    registry = _registry_for_run(tmp_path)
    calls = 0

    def builder(config: Any) -> _FakeAdapter:
        return _FakeAdapter([], fail_restore=True)

    result = runner.run_authoritative(
        registry,
        config={},
        trace_loader=lambda source: _trace(Path(str(source)).stem, actions=12, root=tmp_path),
        environment_builder=builder,
        subprocess_runner=lambda job: {"pid": 1, "trace_id": job["trace_id"], "stage_id": job["stage_id"], "pass": False},
        manifest_path=tmp_path / "manifest.json",
    )
    assert result["status"] in {"BLOCKED", "FAIL"}
    assert result["gates"]["same_process_dynamic"] is False


def test_short_terminal_requires_explicit_legal_source_restore_evidence() -> None:
    runner = _module()
    valid = {
        "steps": 7,
        "nominal_horizon": 10,
        "status": "completed",
        "completed": True,
        "success": True,
        "terminated": True,
        "truncated": False,
        "terminal_observation": {"ok": True},
        "step_returned_terminal": True,
        "autoreset": False,
        "attempt_count": 1,
        "no_retry": True,
        "termination_reason": "predicate_transition",
    }
    assert runner.validate_short_terminal(valid, expected_reason="predicate_transition") is True
    bad = dict(valid)
    bad["termination_reason"] = "different"
    assert runner.validate_short_terminal(bad, expected_reason="predicate_transition") is False


def test_runner_module_has_no_eager_environment_or_model_imports() -> None:
    runner = _module()
    source = inspect.getsource(runner)
    tree = ast.parse(source)
    top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported = " ".join(ast.unparse(node) for node in top_level_imports)
    assert "m1_state_replay" not in imported
    assert "dcu_preflight" not in imported


def test_missing_null_evidence_blocks_before_environment_construction(tmp_path: Path) -> None:
    runner = _module()
    registry = _registry_for_run(tmp_path)
    builds = 0

    def builder(config: Any) -> _FakeAdapter:
        nonlocal builds
        builds += 1
        return _FakeAdapter([])

    result = runner.run_authoritative(
        registry,
        config={},
        trace_loader=lambda source: _trace(Path(str(source)).stem, actions=12, root=tmp_path),
        environment_builder=builder,
        manifest_path=tmp_path / "blocked.json",
    )

    assert result["status"] == "BLOCKED"
    assert builds == 0
    assert result["same_process"] == []
    assert result["fresh_process"] == []
    assert "contact-rich null physics and renderer" in result["errors"][0]


def test_main_passes_manifest_path_to_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _module()
    observed: dict[str, Any] = {}

    def fake_run_authoritative(
        registry_path: str,
        *,
        config: Any,
        manifest_path: str | None,
        output_directory: str | None,
    ) -> dict[str, Any]:
        observed.update(
            registry_path=registry_path,
            config=config,
            manifest_path=manifest_path,
            output_directory=output_directory,
        )
        return {"status": "BLOCKED"}

    monkeypatch.setattr(runner, "run_authoritative", fake_run_authoritative)
    result = runner.main(
        [
            "--run-hard-gate",
            "--registry",
            "registry.json",
            "--manifest",
            "manifest.json",
        ]
    )

    assert result == 1
    assert observed["registry_path"] == "registry.json"
    assert observed["manifest_path"] == "manifest.json"


def test_direct_cli_can_import_repository_scripts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "m1_hard_gate_runner.py"),
            "--run-hard-gate",
            "--registry",
            str(tmp_path / "missing-registry.json"),
            "--manifest",
            str(manifest_path),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "No module named 'scripts'" not in "\n".join(manifest["errors"])


def test_worker_reference_hash_and_exact_gate_are_mandatory() -> None:
    runner = _module()
    references = {2: {"identity": {"step": 2}}}
    job = runner.make_worker_request(
        registry_path="registry.json",
        trace_id="trace",
        stage_id="trace@1",
        worker_index=1,
        horizon=10,
        source_boundary={"identity": {"step": 1}},
        source_references=references,
        physics_envelope={"groups": {}},
        renderer_envelope={"groups": {}},
    )
    assert job["source_references"]["2"] == references[2]
    assert len(job["source_references_sha256"]) == 64
    tampered = dict(job["source_references"])
    tampered["2"] = {"identity": {"step": 9}}
    assert runner._sha256_bytes(runner._canonical_json(tampered).encode("utf-8")) != job[
        "source_references_sha256"
    ]

    incomplete = {"pid": 42, "trace_id": "trace", "stage_id": "trace@1", "pass": True}
    passed, errors = runner._worker_result_gate(incomplete, job)
    assert passed is False
    assert any("physics_static_identity" in error for error in errors)
    assert any("output SHA-256" in error for error in errors)
