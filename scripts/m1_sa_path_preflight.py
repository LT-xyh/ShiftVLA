"""Static path relocation/preflight. Never imports an environment runtime."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "configs/m1/state_replay.yaml"
REQUIRED = {
    "libero_checkout": REPO / "external/hf-libero",
    "lerobot_checkout": REPO / "external/lerobot",
    "robosuite_checkout": REPO / "external/robosuite",
    "mujoco_checkout": REPO / "external/mujoco",
    "runtime_lock": REPO / "runtime/locks/shiftvla-libero-runtime.txt",
    "bddl": REPO / "external/hf-libero/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl",
    "init_state": REPO / "external/hf-libero/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init",
    "action_tape": REPO / "runtime/m1/tapes/libero_spatial-task000-init000.actions.npy",
    "derived_registry": REPO / "runtime/replayvla-p1/f3a/m1_hard_gate_tape_registry.relocated.v2.json",
    "recovery_manifest": REPO / "runtime/replayvla-p1/f3a/provenance_v2/recovery_manifest.json",
}
CHECKOUTS = {"libero_checkout": "8561c60eea2fb93096146f240194649df73d8b1e", "lerobot_checkout": "7e241bd630a3719a56157a497ce5d08f244784f1", "robosuite_checkout": "fbee5844ff5632f5b5698e204ec5357ca50be0df", "mujoco_checkout": "72cb2b210da666617924de709406d6aadbe60c71"}

def sha256(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink(): return None
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def preflight() -> dict[str, Any]:
    missing = []
    results = {}
    for role, path in REQUIRED.items():
        if role in CHECKOUTS:
            try: ident = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            except Exception: ident = None
            results[role] = {"path": str(path), "exists": path.is_dir(), "checkout_sha": ident, "expected_sha": CHECKOUTS[role], "identity_match": ident == CHECKOUTS[role]}
            if ident != CHECKOUTS[role]: missing.append(role)
        else:
            ident = sha256(path) if path.is_file() else None
            results[role] = {"path": str(path), "exists": path.exists(), "regular_non_symlink": ident is not None, "sha256": ident}
            if ident is None: missing.append(role)
    old_text = OLD.read_text()
    unresolved = "/public/home/xuyinghao/workspace/vla/ShiftVLA/" in old_text
    return {"status": "BLOCKED" if missing or unresolved else "PASS", "missing_roles": missing,
            "unresolved_historical_path_in_old_config": unresolved, "results": results,
            "reason": "required pinned checkout/path binding unavailable" if missing else ""}

if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2, sort_keys=True))
