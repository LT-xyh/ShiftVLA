import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.dcu_worker import REQUEST_SCHEMA, RESPONSE_SCHEMA
from scripts.m1_sa_f1t import deterministic_request, adjudicate, run_step


def test_deterministic_request_matches_exact_frozen_schema():
    bundle = deterministic_request()
    assert set(bundle) == set(REQUEST_SCHEMA)
    for key, (shape, dtypes) in REQUEST_SCHEMA.items():
        actual = bundle[key]
        assert tuple(actual.shape) == tuple(8 if x == -1 else x for x in shape)
        assert actual.dtype in dtypes


def test_response_schema_remains_exact_one_of():
    assert set(RESPONSE_SCHEMA) == {"action", "action_chunk"}
    assert RESPONSE_SCHEMA["action"][0] == (1, 7)
    assert RESPONSE_SCHEMA["action_chunk"][0] == (1, 50, 7)


def test_output_directory_is_no_overwrite(tmp_path):
    trace = tmp_path / "trace.md"
    trace.write_text("DCU-side NumPy conversion required: NO\n")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        adjudicate(out, trace, [])


def test_failed_child_is_terminalized_with_runtime_identity(tmp_path):
    record = run_step([sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(7)"], tmp_path, "bad")
    assert record["returncode"] == 7
    assert record["status"] == "FAIL"
    assert record["runtime_identity"]["executable"] == sys.executable
    assert Path(record["stderr_path"]).read_text().strip() == "boom"


def test_pass_requires_trace_and_never_requires_numpy_bridge(tmp_path):
    missing = tmp_path / "missing.md"
    assert adjudicate(tmp_path / "a", missing, [])["status"] == "BLOCKED"
    trace = tmp_path / "trace.md"
    trace.write_text("DCU-side NumPy conversion required: NO\n")
    assert adjudicate(tmp_path / "b", trace, [{"status": "PASS", "fresh_processes": True, "numpy_bridge_used": False}])["status"] == "BLOCKED"
    result = adjudicate(tmp_path / "c", trace, [
        {"status": "PASS", "fresh_processes": True, "numpy_bridge_used": False},
        {"status": "PASS", "fresh_processes": True, "numpy_bridge_used": False},
    ])
    assert result["status"] == "PASS"
    assert result["required_edges"]["dcu_numpy_to_torch"] is False
