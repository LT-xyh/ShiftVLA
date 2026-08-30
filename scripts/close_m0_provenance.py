#!/usr/bin/env python3
"""Close parent provenance for an already completed M0-BASELINE-A run.

This utility is intentionally evidence-only.  It never constructs an
environment, starts a policy worker, or changes any episode artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import m0_baseline_a as baseline


ORIGINAL_PROJECT_SHA = "5b7f7623ca6e279b198eff405018af6a5aea28ea"
EXPECTED_RUN_RELATIVE = Path("runs/m0_baseline_a/20260830T023451Z_655986_c2f3caa5")
RECONSTRUCTION_REASON = "parent JSON serialization failure caused by non-JSON-safe set"
EXPECTED_TASKS = {
    0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}


class ClosureError(RuntimeError):
    """Raised when immutable evidence cannot prove the closure contract."""


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ClosureError(f"source artifact must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"could not parse JSON artifact {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ClosureError(f"source artifact must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _artifact_inventory(run_directory: Path) -> list[dict[str, Any]]:
    """Hash every artifact read by the reconstruction and its consistency checks."""

    required: list[Path] = [
        run_directory / "episodes.jsonl",
        run_directory / "aggregate.json",
        run_directory / "episode_matrix.json",
        run_directory / "config_resolved.json",
        ROOT / "configs/m0/baseline_a.yaml",
        ROOT / "configs/m0/dcu_preflight.yaml",
        ROOT / "runtime/locks/shiftvla-libero-runtime.txt",
        ROOT / "runtime/locks/shiftvla-libero-dcu-runtime.txt",
        ROOT / "runtime/manifests/m0_preflight_b_artifacts.json",
        ROOT / "external/pins.yaml",
    ]
    for worker in ("worker-A", "worker-B"):
        child = run_directory / "children" / worker
        required.extend(
            child / name
            for name in (
                "run_manifest.json",
                "aggregate.json",
                "episodes.jsonl",
                "episode_matrix.json",
                "config_resolved.json",
                "command.txt",
                "process_stdout.log",
                "process_stderr.log",
                "stdout.log",
                "stderr.log",
            )
        )
        required.extend(sorted((child / "episode_journal").glob("*.json")))
        required.extend(sorted((child / "episode_rows").glob("*.json")))

    seen: set[Path] = set()
    inventory: list[dict[str, Any]] = []
    for path in required:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        inventory.append(
            {
                "path": _relative_label(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "source": "original_execution" if path.is_relative_to(run_directory) else "pinned_project_input",
            }
        )
    return sorted(inventory, key=lambda item: item["path"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ClosureError(f"episode JSONL must be a regular file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClosureError(f"invalid episode JSONL at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ClosureError(f"episode JSONL line {line_number} is not an object")
        rows.append(value)
    return rows


def _assert_consistency(run_directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    matrix = _load_json(run_directory / "episode_matrix.json")
    if not isinstance(matrix, list):
        raise ClosureError("episode_matrix.json must contain a list")
    rows = _read_jsonl(run_directory / "episodes.jsonl")
    aggregate = _load_json(run_directory / "aggregate.json")
    config = _load_json(run_directory / "config_resolved.json")
    if not isinstance(aggregate, dict) or not isinstance(config, dict):
        raise ClosureError("aggregate/config evidence must be JSON objects")
    if len(matrix) != 15 or len(rows) != 15:
        raise ClosureError(f"expected exactly 15 matrix/episode rows, got {len(matrix)}/{len(rows)}")
    if [item.get("order") for item in matrix] != list(range(15)):
        raise ClosureError("matrix orders are not exactly 0..14")
    if len({item.get("episode_id") for item in matrix}) != 15:
        raise ClosureError("matrix episode IDs are not unique")
    matrix_by_id = {item["episode_id"]: item for item in matrix}
    row_by_id = {item.get("episode_id"): item for item in rows}
    if set(row_by_id) != set(matrix_by_id):
        raise ClosureError("episode rows do not match the frozen matrix IDs")
    for episode_id, planned in matrix_by_id.items():
        row = row_by_id[episode_id]
        for field in ("order", "task_id", "init_state_id", "worker", "physical_device", "trajectory_id"):
            if row.get(field) != planned.get(field):
                raise ClosureError(f"episode {episode_id} changed matrix field {field}")
        if row.get("status") != "completed" or row.get("attempt_count") != 1 or row.get("no_retry") is not True:
            raise ClosureError(f"episode {episode_id} violates completed one-attempt contract")
        if row.get("actions_finite") is not True or row.get("actions_dtype") != "float32":
            raise ClosureError(f"episode {episode_id} has invalid action evidence")
        action_shapes = row.get("actions_shape")
        if (
            not isinstance(action_shapes, list)
            or len(action_shapes) != row.get("steps")
            or any(shape != [1, 7] for shape in action_shapes)
        ):
            raise ClosureError(f"episode {episode_id} has invalid action shape evidence")
        if row.get("offline") is not True or row.get("hub_fallback") is not False:
            raise ClosureError(f"episode {episode_id} does not prove offline execution")
        if row.get("steps") != row.get("policy_calls"):
            raise ClosureError(f"episode {episode_id} policy/env step count mismatch")
    expected_matrix = [
        (task_id, init_state_id)
        for task_id in (0, 4, 9)
        for init_state_id in range(5)
    ]
    actual_matrix = [(item.get("task_id"), item.get("init_state_id")) for item in matrix]
    if actual_matrix != expected_matrix:
        raise ClosureError("task/init matrix does not match the frozen task-major selection")
    for item in matrix:
        if item.get("task_name") != EXPECTED_TASKS.get(item.get("task_id")):
            raise ClosureError(f"task name drifted for task {item.get('task_id')}")
    if sum(row.get("success") is True for row in rows) != 9:
        raise ClosureError("success count is not exactly 9")
    if sum(row.get("termination_reason") == "official_horizon" for row in rows) != 6:
        raise ClosureError("official horizon failure count is not exactly 6")
    if any(row.get("status") == "crashed" for row in rows):
        raise ClosureError("episode crash row present")
    if sum(int(row.get("steps", 0)) for row in rows) != 2751:
        raise ClosureError("total env steps are not exactly 2751")
    if sum(int(row.get("policy_calls", 0)) for row in rows) != 2751:
        raise ClosureError("total policy calls are not exactly 2751")
    if aggregate.get("attempted") != 15 or aggregate.get("completed") != 15 or aggregate.get("crashed") != 0:
        raise ClosureError("aggregate attempt/completion/crash counts drifted")
    if aggregate.get("successes") != 9 or aggregate.get("overall_sr") != 0.6:
        raise ClosureError("aggregate success rate is not exactly 9/15")
    if aggregate.get("per_task_sr") != {"0": 0.6, "4": 0.6, "9": 0.6}:
        raise ClosureError("per-task success rates drifted")
    if aggregate.get("runtime_failures") != [] or aggregate.get("numerical_anomalies") != []:
        raise ClosureError("aggregate contains runtime or numerical anomalies")
    if aggregate.get("final_verdict") != "PASS":
        raise ClosureError("original aggregate is not PASS")
    return matrix, rows, aggregate, config


def _child_runtime_evidence(run_directory: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for worker in ("A", "B"):
        manifest = _load_json(run_directory / "children" / f"worker-{worker}" / "run_manifest.json")
        if not isinstance(manifest, dict) or manifest.get("status") != "PASS":
            raise ClosureError(f"child {worker} manifest is not PASS")
        if manifest.get("project", {}).get("git_sha") != ORIGINAL_PROJECT_SHA:
            raise ClosureError(f"child {worker} project SHA drifted")
        evidence[worker] = {
            "manifest_sha256": _sha256(run_directory / "children" / f"worker-{worker}" / "run_manifest.json"),
            "status": manifest.get("status"),
            "scope": manifest.get("scope"),
            "worker_evidence": manifest.get("provenance", {}).get("worker_evidence", {}),
        }
    return evidence


def _write_no_overwrite(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"closure artifact already exists: {path}")
    baseline.write_json_no_overwrite(path, value)


def reconstruct(run_directory: Path, *, closure_project_sha: str) -> tuple[Path, Path, dict[str, Any]]:
    run_directory = run_directory.resolve(strict=True)
    if run_directory != (ROOT / EXPECTED_RUN_RELATIVE).resolve():
        raise ClosureError(f"unexpected run directory: {run_directory}")
    if closure_project_sha == ORIGINAL_PROJECT_SHA or len(closure_project_sha) != 40:
        raise ClosureError("closure_project_sha must be a new full Git SHA")
    matrix, rows, aggregate, config = _assert_consistency(run_directory)
    inventory = _artifact_inventory(run_directory)
    child_evidence = _child_runtime_evidence(run_directory)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_hashes = {item["path"]: item["sha256"] for item in inventory}
    runtime_evidence = {
        "evidence_class": "RUN-TIME EVIDENCE",
        "source": "original M0-BASELINE-A execution",
        "original_execution_project_sha": ORIGINAL_PROJECT_SHA,
        "parent_manifest_published_during_execution": False,
        "episode_rows_sha256": source_hashes["runs/m0_baseline_a/20260830T023451Z_655986_c2f3caa5/episodes.jsonl"],
        "aggregate_sha256": source_hashes["runs/m0_baseline_a/20260830T023451Z_655986_c2f3caa5/aggregate.json"],
        "episode_matrix_sha256": source_hashes["runs/m0_baseline_a/20260830T023451Z_655986_c2f3caa5/episode_matrix.json"],
        "child_manifests": child_evidence,
        "aggregate_summary": {
            "attempted": aggregate["attempted"],
            "completed": aggregate["completed"],
            "successes": aggregate["successes"],
            "overall_sr": aggregate["overall_sr"],
            "per_task_sr": aggregate["per_task_sr"],
            "total_policy_calls": sum(row["policy_calls"] for row in rows),
            "total_env_steps": sum(row["steps"] for row in rows),
        },
    }
    closure = {
        "evidence_class": "POST-RUN PROVENANCE CLOSURE",
        "provenance_reconstructed_after_run": True,
        "original_execution_project_sha": ORIGINAL_PROJECT_SHA,
        "provenance_closure_project_sha": closure_project_sha,
        "original_run_directory": str(run_directory),
        "reconstruction_timestamp_utc": timestamp,
        "reason": RECONSTRUCTION_REASON,
        "no_episode_rerun": True,
        "source_artifacts": inventory,
    }
    provenance = {
        **dict(config["provenance"]),
        "git_sha": ORIGINAL_PROJECT_SHA,
        "checkpoint_revision": config["checkpoint"]["revision"],
        "trajectory_ids": [row["trajectory_id"] for row in matrix],
    }
    child_preflight = {
        worker: _load_json(run_directory / "children" / f"worker-{worker}" / "run_manifest.json").get("preflight", {})
        for worker in ("A", "B")
    }
    parent = baseline.build_run_manifest(
        config=config,
        config_path=ROOT / "configs/m0/baseline_a.yaml",
        run_directory=run_directory,
        project_sha=ORIGINAL_PROJECT_SHA,
        matrix=matrix,
        aggregate=aggregate,
        provenance=provenance,
        status="PASS",
        preflight_evidence={
            "evidence_class": "RUN-TIME EVIDENCE",
            "source": "child manifests from original execution",
            "parent_live_gate_evidence_available": False,
            "child_preflight": child_preflight,
        },
    )
    parent.update(
        {
            "runtime_evidence": runtime_evidence,
            "post_run_provenance_closure": closure,
            "provenance_reconstructed_after_run": True,
            "original_execution_project_sha": ORIGINAL_PROJECT_SHA,
            "provenance_closure_project_sha": closure_project_sha,
            "original_run_directory": str(run_directory),
            "reconstruction_timestamp_utc": timestamp,
            "reconstruction_reason": RECONSTRUCTION_REASON,
            "no_episode_rerun": True,
        }
    )
    baseline.validate_manifest(parent)
    terminal = {
        "schema_version": baseline.SCHEMA_VERSION,
        "name": baseline.EXPECTED_NAME,
        "status": "PROVENANCE_CLOSED",
        "terminal": True,
        "run_directory": str(run_directory),
        "runtime_verdict": "PASS",
        "runtime_evidence": runtime_evidence,
        "post_run_provenance_closure": closure,
        "provenance_reconstructed_after_run": True,
        "original_execution_project_sha": ORIGINAL_PROJECT_SHA,
        "provenance_closure_project_sha": closure_project_sha,
        "original_run_directory": str(run_directory),
        "reconstruction_timestamp_utc": timestamp,
        "reason": RECONSTRUCTION_REASON,
        "no_episode_rerun": True,
    }
    _write_no_overwrite(run_directory / "run_manifest.json", parent)
    _write_no_overwrite(run_directory / "terminal_manifest.json", terminal)
    return run_directory / "run_manifest.json", run_directory / "terminal_manifest.json", {
        "timestamp": timestamp,
        "source_artifacts": inventory,
        "parent_sha256": _sha256(run_directory / "run_manifest.json"),
        "terminal_sha256": _sha256(run_directory / "terminal_manifest.json"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / EXPECTED_RUN_RELATIVE)
    parser.add_argument("--closure-project-sha", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        run_manifest, terminal_manifest, summary = reconstruct(
            args.run_dir,
            closure_project_sha=args.closure_project_sha,
        )
    except (ClosureError, baseline.BaselineError, OSError, ValueError) as exc:
        print(f"provenance closure failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_manifest": str(run_manifest), "terminal_manifest": str(terminal_manifest), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
