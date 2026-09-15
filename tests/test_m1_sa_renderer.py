import json
from pathlib import Path

import pytest

from scripts.m1_sa_renderer import (
    adjudicate_workers,
    deterministic_select,
    governed_environment,
    validate_f1t_predecessor,
)


def test_missing_or_drifted_f1t_predecessor_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        validate_f1t_predecessor(tmp_path / "missing.json")
    path = tmp_path / "f1t.json"
    path.write_text(json.dumps({"status": "PASS", "authority": "wrong"}))
    with pytest.raises(RuntimeError):
        validate_f1t_predecessor(path)


def test_legacy_renderer_config_cannot_be_selected():
    with pytest.raises(RuntimeError):
        validate_f1t_predecessor(Path("configs/m1/renderer_preflight_r1_egl0.yaml"))


def test_deterministic_selection_requires_exactly_one_candidate():
    assert deterministic_select([{"ordinal": 0, "software": True}]) == 0
    with pytest.raises(RuntimeError, match="zero"):
        deterministic_select([{"ordinal": 0, "software": False}])
    with pytest.raises(RuntimeError, match="multiple"):
        deterministic_select([{"ordinal": 0, "software": True}, {"ordinal": 1, "software": True}])


def test_adjudication_requires_three_terminal_pass_and_matching_identity(tmp_path):
    discovery = {"status": "PASS", "selected_ordinal": 8, "manifest_sha256": "d"}
    effective = {"config_sha256": "c", "effective_config_sha256": "c", "selected_ordinal": 8}
    good = [{"worker_id": f"worker-{i}", "status": "PASS", "public_render_count": 1,
             "egl_identity": {"vendor": "v", "version": "1"}, "gl_identity": {"renderer": "r"},
             "close_status": "PASS", "parent_observed_exit": True, "replacement": False,
                 "forbidden_operation_count": 0, "returncode": 0, "effective_config_sha256": "c", "selected_ordinal": 8,
                 "stdout_sha256": "s", "stderr_sha256": "e", "cleanup_status": "PASS"} for i in range(3)]
    assert adjudicate_workers(discovery, effective, good)["status"] == "PASS"
    bad = [dict(good[0]), dict(good[1]), dict(good[2])]
    bad[2]["gl_identity"] = {"renderer": "other"}
    assert adjudicate_workers(discovery, effective, bad)["status"] == "FAIL"


def test_worker_environment_propagates_frozen_renderer_and_libero_settings():
    env = governed_environment({"environment": {"LIBERO_CONFIG_PATH": "/x", "MUJOCO_GL": "egl"}, "renderer": {"selected_ordinal": 8}})
    assert env["LIBERO_CONFIG_PATH"] == "/x"
    assert env["MUJOCO_GL"] == "egl"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "8"
