"""Contract tests for the standalone ReplayVLA M1 hard-gate primitives.

The tests are deliberately fake/artifact-only.  They must not import the
policy, SmolVLA, LeRobot, LIBERO, or replay runner.  The first TDD run is
expected to fail because ``scripts.m1_hard_gate`` has not been implemented.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    # Defer import so the initial RED run reports a missing feature from the
    # test body instead of failing collection with an import/setup error.
    try:
        import scripts.m1_hard_gate as hard_gate
    except ModuleNotFoundError as exc:  # pragma: no cover - RED-only branch
        pytest.fail(f"m1_hard_gate feature is missing: {exc}")
    return hard_gate


def _action_values(step: int) -> list[float]:
    return [float(step + 1), 0.1, -0.2, 0.3, -0.4, 0.5, -0.6]


def _episode_row(
    episode_id: str = "libero_spatial-task000-init000",
    *,
    task_id: int = 0,
    init_state_id: int = 0,
    seed: int = 2027,
    steps: int = 6,
    success: bool = True,
) -> dict[str, Any]:
    values = [_action_values(i) for i in range(steps)]
    actions = [[item] for item in values]
    action_array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    evidence = [
        {
            "dtype": "float32",
            "finite": True,
            "shape": [1, 7],
            "values": [item],
        }
        for item in values
    ]
    return {
        "episode_id": episode_id,
        "trajectory_id": episode_id,
        "suite": "libero_spatial",
        "task_id": task_id,
        "task_name": "frozen-task",
        "init_state_id": init_state_id,
        "seed": seed,
        "status": "completed",
        "completed": True,
        "success": success,
        "crashed": False,
        "attempt_count": 1,
        "no_retry": True,
        "steps": steps,
        "actions": actions,
        "action_evidence": evidence,
        "actions_dtype": "float32",
        "actions_finite": True,
        "actions_shape": [[1, 7] for _ in range(steps)],
        "action_sha256": hashlib.sha256(action_array.tobytes(order="C")).hexdigest(),
        "task_source_evidence": {
            "suite": "libero_spatial",
            "task_id": task_id,
            "task_name": "frozen-task",
        },
    }


def _write_m0_fixture(
    root: Path,
    *,
    row: dict[str, Any] | None = None,
    terminal_status: str = "PASS",
) -> tuple[Path, Path, Path, dict[str, Any]]:
    run = root / "m0_baseline_a" / "run-001"
    child = run / "children" / "worker-A"
    rows = child / "episode_rows"
    rows.mkdir(parents=True)
    row = deepcopy(row or _episode_row())
    row_path = rows / f"0000-{row['episode_id']}.json"
    row_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    source_sha256 = hashlib.sha256(row_path.read_bytes()).hexdigest()
    matrix = [
        {
            "episode_id": row["episode_id"],
            "trajectory_id": row["trajectory_id"],
            "suite": row["suite"],
            "task_id": row["task_id"],
            "task_name": row["task_name"],
            "init_state_id": row["init_state_id"],
            "seed": row["seed"],
        }
    ]
    (run / "episode_matrix.json").write_text(
        json.dumps(matrix, sort_keys=True) + "\n", encoding="utf-8"
    )
    parent = {
        "schema_version": 1,
        "name": "baseline_a",
        "status": terminal_status,
        "run_directory": str(run),
        "scope": {
            "episode_ids": [row["episode_id"]],
            "trajectory_ids": [row["trajectory_id"]],
        },
        "provenance": {"seed": row["seed"]},
    }
    closure = {
        "source_artifacts": [
            {
                "path": str(row_path),
                "sha256": source_sha256,
                "size_bytes": row_path.stat().st_size,
            }
        ]
    }
    parent["post_run_provenance_closure"] = closure
    terminal = {
        "schema_version": 1,
        "terminal": True,
        "status": terminal_status,
        "run_directory": str(run),
        "post_run_provenance_closure": closure,
    }
    parent_path = run / "run_manifest.json"
    terminal_path = run / "terminal_manifest.json"
    parent_path.write_text(json.dumps(parent, sort_keys=True) + "\n", encoding="utf-8")
    terminal_path.write_text(json.dumps(terminal, sort_keys=True) + "\n", encoding="utf-8")
    return row_path, parent_path, terminal_path, row


def _pinned_extract_kwargs(row_path: Path, parent_path: Path, terminal_path: Path) -> dict[str, Any]:
    return {
        "parent_run_dir": parent_path.parent,
        "expected_source_sha256": hashlib.sha256(row_path.read_bytes()).hexdigest(),
        "expected_parent_run_manifest_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "expected_terminal_manifest_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
    }


def _coverage(regime: str, *, source_sha: str = "a" * 64) -> dict[str, Any]:
    return {
        "regime": regime,
        "source_file_sha256": source_sha,
        "capture_offset": 10,
        "continuation_horizon": 5,
    }


def _candidate(regime: str, episode_id: str, *, role: str = "primary") -> dict[str, Any]:
    import scripts.m1_hard_gate as hard_gate

    action = np.zeros((1, hard_gate.ACTION_DIM), dtype=np.float32)
    return {
        "m0_trace": hard_gate.M0Trace(
            episode_id=episode_id,
            suite="libero_spatial",
            task_id=0,
            init_state_id=0,
            seed=1,
            task_name="frozen-task",
            actions=action,
            source_file="synthetic-row.json",
            source_file_sha256="a" * 64,
            parent_run_manifest="synthetic-run.json",
            parent_run_manifest_sha256="a" * 64,
            parent_terminal_manifest="synthetic-terminal.json",
            parent_terminal_manifest_sha256="a" * 64,
            action_sha256=hashlib.sha256(action.tobytes(order="C")).hexdigest(),
        ),
        "role": role,
        "source_coverage": _coverage(regime),
    }


def _unvalidated_candidate(regime: str, episode_id: str, *, role: str = "primary") -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "role": role,
        "source_coverage": _coverage(regime),
    }


def _registry_trace(module: Any, tmp_path: Path, suffix: str, regime: str) -> Any:
    row = _episode_row(
        episode_id=f"libero_spatial-task{suffix}-init000",
        task_id=int(suffix),
        init_state_id=0,
        steps=32,
    )
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path / suffix, row=row)
    return module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))


def _tape_provenance(root: Path) -> dict[str, Any]:
    tape = np.ascontiguousarray(
        np.asarray([_action_values(index) for index in range(32)], dtype=np.float32)
    )
    path = root / "action_tape.bin"
    path.write_bytes(np.ascontiguousarray(tape).tobytes(order="C"))
    return {
        "path": str(path),
        "algorithm": "fixed-test-tape",
        "seed": 12027,
        "dtype": "float32",
        "shape": [32, 7],
        "sha256": hashlib.sha256(np.ascontiguousarray(tape).tobytes()).hexdigest(),
        "capture_offsets": [10],
    }


def _null_samples(module: Any) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(module.REGIME_NAMES):
        for pair_index in range(20):
            samples.append(
                {
                    "pair_id": f"{regime}-{pair_index:02d}",
                    "trace_id": f"trace-{regime_index}",
                    "regime": regime,
                    "quantity": "qpos",
                    "horizon": 1,
                    "value_a": float(pair_index),
                    "value_b": float(pair_index) + 0.01,
                    # These are deliberately independent evidence fields;
                    # the builder must not infer them from ``regime``.
                    "contact_evidence": {"source": "contact_pairs", "present": regime == "contact"},
                    "grasp_evidence": {"source": "gripper_object", "present": regime == "grasp"},
                    "carried_evidence": {"source": "object_motion", "present": regime == "carried"},
                }
            )
    return samples


def test_extract_m0_trace_validates_parent_hashes_identity_and_exact_actions(tmp_path: Path) -> None:
    module = _module()
    row_path, parent_path, terminal_path, row = _write_m0_fixture(tmp_path)

    trace = module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))

    assert trace.episode_id == row["episode_id"]
    assert trace.task_id == row["task_id"]
    assert trace.init_state_id == row["init_state_id"]
    assert trace.seed == row["seed"]
    assert trace.actions.dtype == np.dtype("float32")
    assert trace.actions.shape == (row["steps"], 7)
    assert trace.actions.flags.c_contiguous
    assert trace.actions.flags.writeable is False
    assert trace.action_sha256 == hashlib.sha256(trace.actions.tobytes(order="C")).hexdigest()
    assert trace.source_file_sha256 == hashlib.sha256(row_path.read_bytes()).hexdigest()
    assert trace.parent_run_manifest_sha256 == hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert trace.parent_terminal_manifest_sha256 == hashlib.sha256(terminal_path.read_bytes()).hexdigest()


def test_extract_m0_trace_requires_explicit_parent_manifest_scope(tmp_path: Path) -> None:
    module = _module()
    row_path, _, _, _ = _write_m0_fixture(tmp_path)
    with pytest.raises(module.ProvenanceError, match="required|implicit"):
        module.extract_m0_trace(row_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"attempt_count": 2}), "attempt"),
        (lambda row: row.update({"no_retry": False}), "retry"),
        (lambda row: row.update({"success": False}), "success"),
        (lambda row: row.update({"action_evidence": []}), "evidence"),
        (lambda row: row.update({"actions": [[[1.0] * 6]]}), "shape"),
        (lambda row: row.update({"actions_dtype": "float64"}), "dtype"),
    ],
)
def test_extract_m0_trace_rejects_contract_violations(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    module = _module()
    row = _episode_row()
    mutation(row)
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path, row=row)

    with pytest.raises(module.ProvenanceError, match=message):
        module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))


def test_extract_m0_trace_rejects_nonfinite_and_action_hash_mismatch(tmp_path: Path) -> None:
    module = _module()
    row = _episode_row()
    row["actions"][0][0][0] = "nan"
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path, row=row)
    with pytest.raises(module.ProvenanceError, match="finite"):
        module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))

    row = _episode_row()
    row["action_sha256"] = "f" * 64
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path / "hash", row=row)
    with pytest.raises(module.ProvenanceError, match="action.*hash"):
        module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))


def test_extract_m0_trace_rejects_parent_terminal_hash_or_identity_mismatch(tmp_path: Path) -> None:
    module = _module()
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["run_directory"] = "/different/run"
    terminal_path.write_text(json.dumps(terminal, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(module.ProvenanceError, match="run_directory"):
        module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))

    row = _episode_row(seed=99)
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path / "identity", row=row)
    with pytest.raises(module.ProvenanceError, match="seed"):
        module.extract_m0_trace(
            row_path,
            **_pinned_extract_kwargs(row_path, parent_path, terminal_path),
            expected_identity={"seed": 2027},
        )


def test_primary_always_minimum_selection_uses_only_source_coverage_and_episode_tie_break(
    tmp_path: Path,
) -> None:
    module = _module()
    primary: dict[str, list[dict[str, Any]]] = {}
    supplements: dict[str, list[dict[str, Any]]] = {}
    for regime in module.REGIME_NAMES:
        primary[regime] = [
            _candidate(regime, f"primary-z-{regime}"),
            _candidate(regime, f"primary-a-{regime}"),
        ]
        supplements[regime] = [
            _candidate(regime, f"supplement-a-{regime}", role="supplement"),
        ]

    selected = module.select_minimum_corpus(
        primary_candidates=primary,
        preregistered_supplements=supplements,
        minimum_per_regime=1,
    )

    assert selected.regimes == module.REGIME_NAMES
    assert all(item.startswith("primary-a-") for item in selected.trace_ids)
    assert selected.source_coverage_only is True
    # The candidate's outcome fields are not consumed by the selector; a
    # forbidden outcome field in the selection input is rejected instead.
    bad = deepcopy(primary)
    bad[module.REGIME_NAMES[0]][0]["success"] = True
    with pytest.raises(module.CorpusSelectionError, match="outcome"):
        module.select_minimum_corpus(
            primary_candidates=bad,
            preregistered_supplements=supplements,
            minimum_per_regime=1,
        )


def test_selection_uses_only_preregistered_supplements_and_fails_when_regime_uncovered() -> None:
    module = _module()
    regime = module.REGIME_NAMES[0]
    primary = {name: [] for name in module.REGIME_NAMES}
    supplements = {name: [] for name in module.REGIME_NAMES}
    supplements[regime] = [_candidate(regime, "not-preregistered", role="other")]
    with pytest.raises(module.CorpusSelectionError, match="preregistered|primary"):
        module.select_minimum_corpus(
            primary_candidates=primary,
            preregistered_supplements=supplements,
            minimum_per_regime=1,
        )


def test_frozen_registry_is_complete_hashed_and_no_overwrite(tmp_path: Path) -> None:
    module = _module()
    traces = [
        _registry_trace(module, tmp_path / "trace", f"{100 + index:03d}", regime)
        for index, regime in enumerate(module.REGIME_NAMES)
    ]
    per_trace = {
        trace.episode_id: {
            "regimes": [regime],
            "capture_offset": 10 + index,
            "continuation_horizon": 5,
            "source_coverage": _coverage(regime),
        }
        for index, (trace, regime) in enumerate(zip(traces, module.REGIME_NAMES))
    }
    path = tmp_path / "registry.json"
    tape_provenance = _tape_provenance(tmp_path)

    written = module.write_frozen_registry(
        path,
        traces=traces,
        tape_provenance=tape_provenance,
        per_trace=per_trace,
    )
    loaded = module.load_frozen_registry(written)

    assert loaded["registry_sha256"]
    assert len(loaded["traces"]) == 7
    assert loaded["tape_provenance"]["sha256"] == tape_provenance["sha256"]
    assert set(loaded["traces"][0]["regimes"]) <= set(module.REGIME_NAMES)
    with pytest.raises(module.RegistryError, match="overwrite|exists"):
        module.write_frozen_registry(
            path,
            traces=traces,
            tape_provenance=tape_provenance,
            per_trace=per_trace,
        )


def test_registry_loader_rejects_tampered_registry_sha(tmp_path: Path) -> None:
    module = _module()
    traces = [
        _registry_trace(module, tmp_path / "trace", f"{200 + index:03d}", regime)
        for index, regime in enumerate(module.REGIME_NAMES)
    ]
    per_trace = {
        trace.episode_id: {
            "regimes": [regime],
            "capture_offset": 10,
            "continuation_horizon": 5,
            "source_coverage": _coverage(regime),
        }
        for trace, regime in zip(traces, module.REGIME_NAMES)
    }
    tape_provenance = _tape_provenance(tmp_path)
    path = module.write_frozen_registry(
        tmp_path / "registry.json",
        traces=traces,
        tape_provenance=tape_provenance,
        per_trace=per_trace,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["traces"][0]["capture_offset"] = 999
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(module.RegistryError, match="registry.*hash|hash.*registry"):
        module.load_frozen_registry(path)


def test_regime_metadata_has_exactly_seven_names_and_rejects_unknown() -> None:
    module = _module()
    assert len(module.REGIME_NAMES) == 7
    assert len(set(module.REGIME_NAMES)) == 7
    assert set(module.REGIME_METADATA) == set(module.REGIME_NAMES)
    for name in module.REGIME_NAMES:
        metadata = module.RegimeMetadata(name=name, event=f"event:{name}", evidence_fields=("state",))
        assert metadata.name == name
    with pytest.raises(module.RegimeError):
        module.RegimeMetadata(name="task_outcome", event="bad", evidence_fields=())


def test_grouped_null_envelope_enforces_pair_and_independent_evidence_coverage() -> None:
    module = _module()
    samples = _null_samples(module)
    selected_trace_ids = sorted({sample["trace_id"] for sample in samples})
    required_groups = {(regime, "qpos", 1) for regime in module.REGIME_NAMES}
    envelope = module.build_grouped_null_envelope(
        samples,
        selected_trace_ids=selected_trace_ids,
        required_groups=required_groups,
    )

    assert set(envelope["groups"]) == set(module.REGIME_NAMES)
    group = envelope["groups"][module.REGIME_NAMES[0]]["qpos"]["1"]
    assert group["n_pairs"] == 20
    assert envelope["coverage"]["minimum_pairs_per_trace"] == 5
    assert envelope["coverage"]["minimum_samples_per_regime"] == 5
    assert set(envelope["coverage"]["independent_evidence"]) == {
        "contact",
        "grasp",
        "carried",
    }

    too_few = _null_samples(module)[:-1]
    with pytest.raises(module.NullCalibrationError, match="20|pairs"):
        module.build_grouped_null_envelope(
            too_few,
            selected_trace_ids=selected_trace_ids,
            required_groups=required_groups,
        )

    missing_evidence = _null_samples(module)
    del missing_evidence[0]["contact_evidence"]
    with pytest.raises(module.NullCalibrationError, match="contact"):
        module.build_grouped_null_envelope(
            missing_evidence,
            selected_trace_ids=selected_trace_ids,
            required_groups=required_groups,
        )


def test_null_envelope_never_uses_global_tolerance_fallback() -> None:
    module = _module()
    samples = _null_samples(module)
    with pytest.raises(module.NullCalibrationError, match="global"):
        module.build_grouped_null_envelope(samples, global_tolerance=1.0)
    envelope = module.build_grouped_null_envelope(
        samples,
        selected_trace_ids=sorted({sample["trace_id"] for sample in samples}),
        required_groups={(regime, "qpos", 1) for regime in module.REGIME_NAMES},
    )
    with pytest.raises(module.NullCalibrationError, match="group|regime"):
        module.check_null_value(envelope, value=0.0, regime="free_motion", quantity="missing", horizon=1)


def test_rgb_metrics_are_pixelwise_and_renderer_envelopes_are_grouped_exact_only() -> None:
    module = _module()
    expected = np.zeros((2, 2, 3), dtype=np.uint8)
    actual = expected.copy()
    actual[0, 1, 0] = 3
    metrics = module.rgb_disagreement_metrics(expected, actual)
    assert metrics == {
        "differing_pixel_count": 1,
        "max_abs": 3.0,
        "mean_abs": 0.25,
    }

    renderer_samples = []
    for regime in module.REGIME_NAMES:
        for index in range(5):
            image = np.full((2, 2, 3), index, dtype=np.uint8)
            renderer_samples.append(
                {
                    "camera": "agentview",
                    "key": "observation.image",
                    "regime": regime,
                    "horizon": 1,
                    "duplicate_controls": [image.copy(), image.copy()],
                    "rgb_a": image,
                    "rgb_b": image.copy(),
                }
            )
    envelope = module.build_renderer_envelopes(renderer_samples)
    group = envelope["groups"]["agentview"]["observation.image"][module.REGIME_NAMES[0]]["1"]
    assert group["exact_only"] is True
    assert group["max_abs"] == 0.0
    assert group["differing_pixel_count_max"] == 0


def test_renderer_envelope_rejects_nonidentical_duplicate_controls_when_exact_only_claimed() -> None:
    module = _module()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    samples = [
        {
            "camera": "agentview",
            "key": "observation.image",
            "regime": regime,
            "horizon": 1,
            "duplicate_controls": [image.copy(), image.copy()],
            "rgb_a": image,
            "rgb_b": image.copy(),
        }
        for regime in module.REGIME_NAMES
        for _ in range(5)
    ]
    samples[0]["duplicate_controls"][1][0, 0, 0] = 1
    with pytest.raises(module.RendererCalibrationError, match="bitwise|duplicate"):
        module.build_renderer_envelopes(samples, exact_only=True)


def test_renderer_envelope_measures_nondeterministic_rgb_under_identical_controls() -> None:
    module = _module()
    control = np.zeros(7, dtype=np.float32)
    baseline = np.zeros((2, 2, 3), dtype=np.uint8)
    samples = []
    for regime in module.REGIME_NAMES:
        for index in range(5):
            observed = baseline.copy()
            observed[0, 0, 0] = index + 1
            samples.append(
                {
                    "camera": "agentview",
                    "key": "observation.image",
                    "regime": regime,
                    "horizon": 1,
                    "duplicate_controls": [control.copy(), control.copy()],
                    "rgb_a": baseline.copy(),
                    "rgb_b": observed,
                }
            )

    envelope = module.build_renderer_envelopes(samples, exact_only=None, min_samples_per_group=5)
    group = envelope["groups"]["agentview"]["observation.image"]["contact"]["1"]
    assert group["duplicate_controls_bitwise_identical"] is True
    assert group["exact_only"] is False
    assert group["differing_pixel_count_max"] == 1
    assert group["max_abs"] == 5.0
    assert module.check_renderer_value(
        envelope,
        camera="agentview",
        key="observation.image",
        regime="contact",
        horizon=1,
        expected=baseline,
        actual=np.pad(np.asarray([[[5, 0, 0]]], dtype=np.uint8), ((0, 1), (0, 1), (0, 0))),
    ) is True


def test_exact_discrete_gate_does_not_accept_tolerance_or_nan_equivalence() -> None:
    module = _module()
    assert module.exact_discrete_equal({"a": [1, 2]}, {"a": [1, 2]}) is True
    assert module.exact_discrete_equal({"a": [1, 2]}, {"a": [1, 3]}) is False
    assert module.exact_discrete_equal(np.array([np.nan]), np.array([np.nan])) is False
    with pytest.raises(module.DiscreteGateError):
        module.require_exact_discrete({"counter": 1}, {"counter": 1.0})
    with pytest.raises(module.DiscreteGateError):
        module.require_exact_discrete({"counter": 1}, {"counter": 1.0000001})


def test_synchronous_legitimate_short_terminal_requires_returned_terminal_without_autoreset() -> None:
    module = _module()
    record = {
        "status": "completed",
        "completed": True,
        "success": True,
        "steps": 7,
        "nominal_horizon": 50,
        "terminated": True,
        "truncated": False,
        "terminal_observation": {"pixels": "present"},
        "step_returned_terminal": True,
        "autoreset": False,
        "attempt_count": 1,
        "no_retry": True,
    }
    assert module.validate_legitimate_short_terminal(record) is True

    for mutation in (
        {"steps": 50},
        {"truncated": True},
        {"step_returned_terminal": False},
        {"autoreset": True},
        {"attempt_count": 2},
    ):
        bad = dict(record)
        bad.update(mutation)
        assert module.validate_legitimate_short_terminal(bad) is False


# Adversarial probes: these are intentionally added before the hardening pass.
# Each probe captures a previously accepted false pass and must be RED against
# the original primitive implementation.


def test_adversarial_extractor_rejects_mapping_source(tmp_path: Path) -> None:
    module = _module()
    row_path, parent_path, _, row = _write_m0_fixture(tmp_path)
    with pytest.raises(module.ProvenanceError):
        module.extract_m0_trace(row, parent_run_dir=parent_path.parent)


def test_adversarial_extractor_requires_caller_pinned_hashes(tmp_path: Path) -> None:
    module = _module()
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path)
    with pytest.raises(module.ProvenanceError):
        module.extract_m0_trace(row_path, parent_run_dir=parent_path.parent)


@pytest.mark.parametrize("representation", ["bool", "int"])
def test_adversarial_extractor_rejects_non_float32_action_representation(
    tmp_path: Path, representation: str
) -> None:
    module = _module()
    row = _episode_row()
    if representation == "bool":
        values = [True] * 7
    else:
        values = list(range(7))
    row["actions"] = [[values]]
    row["action_evidence"] = [
        {"dtype": "float32", "finite": True, "shape": [1, 7], "values": [values]}
    ]
    row["steps"] = 1
    row["actions_shape"] = [[1, 7]]
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path, row=row)
    with pytest.raises(module.ProvenanceError):
        module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))


def test_adversarial_extractor_requires_persisted_action_sha(tmp_path: Path) -> None:
    module = _module()
    row = _episode_row()
    row.pop("action_sha256")
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path, row=row)
    with pytest.raises(module.ProvenanceError):
        module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))


def test_adversarial_extractor_requires_closure_inventory_membership(tmp_path: Path) -> None:
    module = _module()
    row_path, parent_path, terminal_path, _ = _write_m0_fixture(tmp_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["post_run_provenance_closure"] = {"source_artifacts": []}
    terminal_path.write_text(json.dumps(terminal, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(module.ProvenanceError):
        module.extract_m0_trace(row_path, **_pinned_extract_kwargs(row_path, parent_path, terminal_path))


def test_adversarial_extractor_rejects_row_outside_relative_layout(tmp_path: Path) -> None:
    module = _module()
    row_path, parent_path, _, _ = _write_m0_fixture(tmp_path)
    misplaced = parent_path.parent / "row.json"
    misplaced.write_bytes(row_path.read_bytes())
    with pytest.raises(module.ProvenanceError):
        module.extract_m0_trace(misplaced, parent_run_dir=parent_path.parent)


def test_adversarial_selection_accepts_only_validated_m0_traces(tmp_path: Path) -> None:
    module = _module()
    primary = {name: [{"episode_id": f"p-{name}", "source_coverage": _coverage(name)}] for name in module.REGIME_NAMES}
    with pytest.raises(module.CorpusSelectionError):
        module.select_minimum_corpus(primary, minimum_per_regime=1)


@pytest.mark.parametrize("outcome_key", ["success", "rate", "replay", "result", "pass", "fail", "metric"])
def test_adversarial_selection_rejects_all_outcome_like_keys(outcome_key: str) -> None:
    module = _module()
    primary = {name: [_unvalidated_candidate(name, f"p-{name}")] for name in module.REGIME_NAMES}
    primary[module.REGIME_NAMES[0]][0][outcome_key] = 1
    with pytest.raises(module.CorpusSelectionError):
        module.select_minimum_corpus(primary, minimum_per_regime=1)


def test_adversarial_registry_rejects_mapping_trace_and_unbounded_offset(tmp_path: Path) -> None:
    module = _module()
    traces = [
        _registry_trace(module, tmp_path / "trace", f"{300 + index:03d}", regime)
        for index, regime in enumerate(module.REGIME_NAMES)
    ]
    per_trace = {
        trace.episode_id: {
            "regimes": [regime],
            "capture_offset": 999 if index == 0 else 1,
            "continuation_horizon": 5,
            "source_coverage": _coverage(regime),
        }
        for index, (trace, regime) in enumerate(zip(traces, module.REGIME_NAMES))
    }
    with pytest.raises(module.RegistryError):
        module.write_frozen_registry(
            tmp_path / "registry.json",
            traces=[traces[0].provenance(), *traces[1:]],
            tape_provenance=_tape_provenance(tmp_path),
            per_trace=per_trace,
        )


def test_adversarial_registry_requires_persisted_tape_path_and_selection_reconciliation(tmp_path: Path) -> None:
    module = _module()
    traces = [
        _registry_trace(module, tmp_path / "trace", f"{400 + index:03d}", regime)
        for index, regime in enumerate(module.REGIME_NAMES)
    ]
    per_trace = {
        trace.episode_id: {
            "regimes": [regime],
            "capture_offset": 1,
            "continuation_horizon": 5,
            "source_coverage": _coverage(regime),
        }
        for trace, regime in zip(traces, module.REGIME_NAMES)
    }
    with pytest.raises(module.RegistryError):
        module.write_frozen_registry(
            tmp_path / "registry.json",
            traces=traces,
            tape_provenance=_tape_provenance(tmp_path),
            per_trace=per_trace,
            selection={"trace_ids": ["not-selected"]},
        )


def test_adversarial_null_requires_selected_traces_explicit_groups_and_structured_evidence() -> None:
    module = _module()
    with pytest.raises(module.NullCalibrationError):
        module.build_grouped_null_envelope(_null_samples(module))
    samples = _null_samples(module)
    samples[0]["contact_evidence"] = True
    with pytest.raises(module.NullCalibrationError):
        module.build_grouped_null_envelope(
            samples,
            selected_trace_ids={f"trace-{index}" for index in range(7)},
            required_groups={(regime, "qpos", 1) for regime in module.REGIME_NAMES},
        )


def test_adversarial_renderer_requires_duplicate_controls_and_nonzero_nonexact_metric() -> None:
    module = _module()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    samples = [
        {
            "camera": "agentview",
            "key": "observation.image",
            "regime": regime,
            "horizon": 1,
            "rgb_a": image.copy(),
            "rgb_b": image.copy(),
        }
        for regime in module.REGIME_NAMES
        for _ in range(5)
    ]
    with pytest.raises(module.RendererCalibrationError):
        module.build_renderer_envelopes(samples)

    samples = []
    for regime in module.REGIME_NAMES:
        for _ in range(5):
            left = image.copy()
            right = image.copy()
            right[0, 0, 0] = 1
            samples.append(
                {
                    "camera": "agentview",
                    "key": "observation.image",
                    "regime": regime,
                    "horizon": 1,
                    "duplicate_controls": [left.copy(), right.copy()],
                    "rgb_a": image.copy(),
                    "rgb_b": image.copy(),
                }
            )
    with pytest.raises(module.RendererCalibrationError):
        module.build_renderer_envelopes(samples)


def test_adversarial_renderer_gate_checks_mean_abs() -> None:
    module = _module()
    samples = []
    for regime in module.REGIME_NAMES:
        for _ in range(5):
            control_a = np.zeros(7, dtype=np.float32)
            control_b = control_a.copy()
            rgb_a = np.zeros((2, 2, 3), dtype=np.uint8)
            rgb_b = rgb_a.copy()
            rgb_b[0, 0, 0] = 1
            samples.append(
                {
                    "camera": "agentview",
                    "key": "observation.image",
                    "regime": regime,
                    "horizon": 1,
                    "duplicate_controls": [control_a, control_b],
                    "rgb_a": rgb_a,
                    "rgb_b": rgb_b,
                }
            )
    envelope = module.build_renderer_envelopes(samples)
    actual = np.zeros((2, 2, 3), dtype=np.uint8)
    actual[0, 0, :2] = 1
    assert module.check_renderer_value(
        envelope,
        camera="agentview",
        key="observation.image",
        regime=module.REGIME_NAMES[0],
        horizon=1,
        expected=np.zeros((2, 2, 3), dtype=np.uint8),
        actual=actual,
    ) is False


def test_adversarial_terminal_rejects_zero_steps_shortcut_and_mismatched_sync_evidence() -> None:
    module = _module()
    record = {
        "status": "completed",
        "completed": True,
        "success": True,
        "steps": 0,
        "nominal_horizon": 50,
        "terminated": True,
        "truncated": False,
        "terminal_observation": {"present": True},
        "synchronous": True,
        "autoreset": False,
        "attempt_count": 1,
        "no_retry": True,
    }
    assert module.validate_legitimate_short_terminal(record) is False
    record.update({"steps": 7, "terminal_step": 6, "terminal_event": "reset", "termination_reason": "wrong"})
    assert module.validate_legitimate_short_terminal(record) is False


def test_adversarial_exact_discrete_preserves_mapping_key_types_and_bytes() -> None:
    module = _module()
    assert module.exact_discrete_equal({True: "x"}, {1: "x"}) is False
    assert module.exact_discrete_equal(
        np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32)
    ) is False


def test_adversarial_json_rejects_nonfinite_mapping_keys_and_regime_aliases() -> None:
    module = _module()
    with pytest.raises(module.M1HardGateError):
        module.canonical_json({float("nan"): "bad"})
    with pytest.raises(module.RegimeError):
        module.normalize_regime("free motion")
