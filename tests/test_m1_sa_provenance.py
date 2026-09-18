"""Post-run static audit tests; these were NOT preconditions of the retained v2 run."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.m1_sa_provenance import RecoveryBlocked, verified_bytes, verify_copy, verify_derivation

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runtime/m1/m1_hard_gate_tape_registry.json"
NEW = ROOT / "runtime/replayvla-p1/f3a/m1_hard_gate_tape_registry.relocated.v2.json"


def test_missing_artifact_blocks(tmp_path):
    with pytest.raises(RecoveryBlocked):
        verified_bytes(tmp_path / "absent", "0" * 64)


def test_wrong_sha_blocks(tmp_path):
    p = tmp_path / "artifact"; p.write_bytes(b"artifact")
    with pytest.raises(RecoveryBlocked):
        verified_bytes(p, "0" * 64)


def test_symlink_rejected(tmp_path):
    p = tmp_path / "artifact"; p.write_bytes(b"artifact")
    link = tmp_path / "link"; link.symlink_to(p)
    with pytest.raises(RecoveryBlocked):
        verified_bytes(link, hashlib.sha256(b"artifact").hexdigest())


def test_copy_drift_rejected(tmp_path):
    p = tmp_path / "original"; p.write_bytes(b"original")
    q = tmp_path / "copy"; q.write_bytes(b"drift")
    with pytest.raises(RecoveryBlocked):
        verify_copy(p, q, hashlib.sha256(b"original").hexdigest())


@pytest.mark.parametrize("mutation", ["seed", "missing", "wrong_hash"])
def test_registry_rejects_invalid_derivation(tmp_path, mutation):
    old = json.loads(OLD.read_text()); new = json.loads(NEW.read_text())
    if mutation == "seed":
        new["traces"][0]["seed"] += 1
    else:
        p = tmp_path / "source"
        if mutation == "wrong_hash":
            p.write_bytes(b"incorrect")
        new["traces"][0]["source_file"] = str(p)
    with pytest.raises(RecoveryBlocked):
        verify_derivation(old, new)


def test_old_registry_unchanged():
    frozen = subprocess.check_output(["git", "show", "524109e:runtime/m1/m1_hard_gate_tape_registry.json"], cwd=ROOT)
    assert OLD.read_bytes() == frozen


def test_path_only_and_existing_strict_loader():
    from scripts.m1_hard_gate import load_frozen_registry
    changes = verify_derivation(json.loads(OLD.read_text()), json.loads(NEW.read_text()))
    assert len(changes) == 5
    assert load_frozen_registry(NEW)["registry_sha256"] == json.loads(NEW.read_text())["registry_sha256"]


def test_tape_semantic_identity():
    a = np.load(ROOT / "runtime/m1/tapes/libero_spatial-task000-init000.actions.npy", allow_pickle=False)
    assert a.dtype == np.dtype("float32") and a.shape == (82, 7)
    assert a.flags.c_contiguous and np.isfinite(a).all()
    assert hashlib.sha256(a.tobytes(order="C")).hexdigest() == "c17bc44ad8195fecb42a80b3b272828761a9df6d88dd2bafe45db01a6cb04bbf"


def test_v1_remains_blocked_unchanged():
    name = "runtime/replayvla-p1/f3a/null_schedule.json"
    frozen = subprocess.check_output(["git", "show", "524109e:" + name], cwd=ROOT)
    assert (ROOT / name).read_bytes() == frozen
    assert json.loads(frozen)["status"] == "BLOCKED"


def test_recovery_audit_has_no_dynamic_imports():
    code = '''
import sys, json, importlib.abc
class RejectDynamic(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'libero', 'mujoco', 'OpenGL', 'robosuite', 'lerobot', 'torch'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, RejectDynamic())
from scripts.m1_sa_provenance import verify_derivation
from pathlib import Path
verify_derivation(json.loads(Path('runtime/m1/m1_hard_gate_tape_registry.json').read_text()), json.loads(Path('runtime/replayvla-p1/f3a/m1_hard_gate_tape_registry.relocated.v2.json').read_text()))
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, timeout=30)
