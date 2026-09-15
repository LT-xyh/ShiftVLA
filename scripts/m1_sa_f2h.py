#!/usr/bin/env python3
"""Static-only F2H re-adjudication; never imports LIBERO, MuJoCo, or EGL."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def re_adjudicate(root: Path, package: Path) -> dict[str, Any]:
    if package.exists():
        raise FileExistsError(package)
    package.mkdir(parents=True)
    c5 = root
    adjudication = load(c5 / "adjudication.json")
    effective = load(c5 / "effective_config.json")
    missing: list[str] = []
    workers = adjudication.get("workers", [])
    if len(workers) != 3:
        missing.append("exactly_three_workers")
    for i, worker in enumerate(workers):
        for key in ("returncode", "public_render_count", "close_status", "parent_observed_exit", "forbidden_operation_count", "replacement", "stdout_sha256", "stderr_sha256"):
            if key not in worker:
                missing.append(f"worker_{i}.{key}")
        for key in ("effective_config_sha256", "selected_ordinal"):
            if key not in worker:
                missing.append(f"worker_{i}.{key}")
    repo = Path(__file__).resolve().parents[1]
    trace = repo / "runtime/replayvla-p1/f1t_production_path_trace.md"
    f1t = repo / "runtime/replayvla-p1/f1t_summary.json"
    inputs = {
        "c5_effective_config": {"path": str((c5 / "effective_config.json").resolve()), "sha256": sha256(c5 / "effective_config.json"), "size": (c5 / "effective_config.json").stat().st_size},
        "f1t_summary": {"path": str(f1t.resolve()), "sha256": sha256(f1t)},
        "f1t_trace": {"path": str(trace.resolve()), "sha256": sha256(trace)},
    }
    result = {
        "phase": "F2H",
        "status": "BLOCKED" if missing else "PASS",
        "formal_sa_renderer": "PENDING_EVIDENCE_HARDENING" if missing else "PASS",
        "dynamic_execution": "none; static reads/hashes only",
        "c5_effective_config_sha256": inputs["c5_effective_config"]["sha256"],
        "frozen_selected_ordinal": effective.get("renderer", {}).get("selected_ordinal"),
        "missing_historical_fields": sorted(set(missing)),
        "inputs": inputs,
        "governance_deviation": {
            "prior_cohorts": ["f2-20260915", "f2-20260915-c2", "f2-20260915-c3", "f2-20260915-c4"],
            "no_further_f2_dynamic_cohort_authorized": True,
        },
        "init_state_binding": "SOURCE_ACCOUNTED under trusted-OS/no-concurrent-replacement assumption; historical worker records do not include per-worker byte-use attestations",
        "runtime_lock_sha256": "921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc",
        "evidence_classifications": {"dynamic_render": "OBSERVED", "factory_internal_init_reset_settle": "TRUSTED_DEPENDENCY / NOT_OBSERVED", "source_to_binary": "NOT_INDEPENDENTLY_ATTESTED"},
        "source_adjudication": str((c5 / "adjudication.json").resolve()),
    }
    shutil.copyfile(c5 / "effective_config.json", package / "effective_config.json")
    shutil.copyfile(c5 / "adjudication.json", package / "adjudication.json")
    (package / "authority_erratum.json").write_text(json.dumps({
        "historical_c5_parent_authority": effective.get("parent_authority"),
        "actual_f2_authorization_commit": "ce4e6df13890d6299c4306dbfe96dc00720e3fe7",
        "statement": "c5 exact config is retained; its erroneous parent authority is not rewritten",
        "future_template_only": True,
    }, indent=2, sort_keys=True) + "\n")
    (package / "discovery_summary.json").write_text(json.dumps(adjudication.get("discovery", {}), indent=2, sort_keys=True) + "\n")
    for i, worker in enumerate(workers):
        (package / f"worker_{i}_terminal.json").write_text(json.dumps(worker, indent=2, sort_keys=True) + "\n")
    manifest = []
    for path in sorted(package.iterdir()):
        manifest.append({"path": str(path), "sha256": sha256(path), "size": path.stat().st_size})
    (package / "evidence_manifest.json").write_text(json.dumps({"files": manifest, "raw_evidence_retained_under": str(c5.resolve())}, indent=2, sort_keys=True) + "\n")
    summary_path = package.parent / "f2h_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("re-adjudicate", nargs="?")
    p.add_argument("--c5-root", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    args = p.parse_args()
    result = re_adjudicate(args.c5_root, args.package)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
