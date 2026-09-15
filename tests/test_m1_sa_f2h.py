import json
from pathlib import Path

import pytest

from scripts.m1_sa_f2h import re_adjudicate, sha256


def test_static_read_detects_missing_worker_bindings(tmp_path):
    c5 = tmp_path / "c5"; c5.mkdir()
    (c5 / "effective_config.json").write_text(json.dumps({"renderer": {"selected_ordinal": 8}}))
    (c5 / "adjudication.json").write_text(json.dumps({"workers": [{"status": "PASS"}] * 3}))
    Path("runtime/replayvla-p1/f1t_summary.json").exists()
    result = re_adjudicate(c5, tmp_path / "package")
    assert result["status"] == "BLOCKED"
    assert "worker_0.effective_config_sha256" in result["missing_historical_fields"]


def test_no_overwrite_package(tmp_path):
    c5 = tmp_path / "c5"; c5.mkdir()
    (c5 / "effective_config.json").write_text("{}")
    (c5 / "adjudication.json").write_text(json.dumps({"workers": []}))
    package = tmp_path / "package"; package.mkdir()
    with pytest.raises(FileExistsError):
        re_adjudicate(c5, package)
