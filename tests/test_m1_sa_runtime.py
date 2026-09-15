import json
import subprocess
import sys
from pathlib import Path

from scripts.m1_sa_runtime import qualify


def test_f1_config_declares_new_namespace_and_two_candidates():
    cfg = json.loads(Path("configs/replayvla/p1.yaml").read_text())
    assert cfg["contract"] == "M1-SA-v1"
    assert cfg["phase"] == "F1"
    assert len(cfg["candidates"]) <= 2
    assert cfg["output_namespace"] == "SA-runtime"


def test_qualification_publishes_each_candidate_without_forbidden_runtime_imports(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"contract": "M1-SA-v1", "phase": "F1", "output_namespace": "SA-runtime",
                               "candidates": {"test": sys.executable}}))
    out = tmp_path / "out"
    qualify(cfg, out)
    record = json.loads((out / "test.json").read_text())
    assert record["probe"]["python"]["executable"] == sys.executable
    assert all(v in {"not_available", "present_but_not_imported"} for v in record["probe"]["forbidden_imports"].values())
    assert (out / "manifest.json").exists()
