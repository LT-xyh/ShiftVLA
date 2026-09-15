#!/usr/bin/env python3
"""F1-only runtime qualification for ReplayVLA-P1.

This entrypoint intentionally performs no environment, renderer, policy, or
asset work. Each candidate is inspected in a fresh subprocess and publishes
an immutable JSON record including the first failing bridge boundary.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROBE = r'''
import importlib.metadata as md, json, platform, sys, traceback
out = {"python": {"executable": sys.executable, "version": sys.version,
                   "implementation": platform.python_implementation()},
       "platform": platform.platform(), "modules": {}, "bridges": {},
       "forbidden_imports": {}}
for name in ("numpy", "torch"):
    try:
        m = __import__(name)
        info = {"file": getattr(m, "__file__", None),
                "version": getattr(m, "__version__", None)}
        try: info["distribution"] = md.version(name)
        except Exception as e: info["distribution_error"] = repr(e)
        if name == "numpy": info["abi"] = repr(getattr(m, "__numpy_submodules__", None))
        if name == "torch":
            info["cuda_available"] = bool(m.cuda.is_available())
            info["torch_config"] = m.__config__.show()
            info["device_count"] = int(m.cuda.device_count()) if m.cuda.is_available() else 0
        out["modules"][name] = info
    except Exception:
        out["modules"][name] = {"import_error": traceback.format_exc()}
try:
    import numpy as np
    import torch
    x = np.arange(8, dtype=np.float32)
    t = torch.from_numpy(x)
    y = t.detach().cpu().numpy()
    out["bridges"]["numpy_to_torch_cpu"] = {"status": "PASS", "dtype": str(t.dtype), "equal": bool(np.array_equal(x, y))}
    z = torch.arange(8, dtype=torch.float32)
    back = z.numpy()
    out["bridges"]["torch_to_numpy_cpu"] = {"status": "PASS", "dtype": str(back.dtype), "equal": bool(np.array_equal(back, np.arange(8, dtype=np.float32)))}
    if bool(torch.cuda.is_available()):
        try:
            d = torch.device("cuda")
            dz = torch.from_numpy(x).to(d)
            host = dz.cpu().numpy()
            out["bridges"]["torch_device_to_numpy_host"] = {"status": "PASS", "device": str(dz.device), "equal": bool(np.array_equal(host, x))}
        except Exception:
            out["bridges"]["torch_device_to_numpy_host"] = {"status": "FAIL", "traceback": traceback.format_exc()}
    else:
        out["bridges"]["torch_device_to_numpy_host"] = {"status": "BLOCKED", "reason": "no device reported"}
except Exception:
    out["bridges"]["numpy_torch_import_boundary"] = {"status": "FAIL", "traceback": traceback.format_exc()}
import importlib.util
for name in ("mujoco", "libero", "robosuite", "transformers"):
    out["forbidden_imports"][name] = "present_but_not_imported" if importlib.util.find_spec(name) else "not_available"
print(json.dumps(out, sort_keys=True))
'''


def qualify(config: Path, output_root: Path) -> int:
    spec = json.loads(config.read_text())
    if spec.get("contract") != "M1-SA-v1" or spec.get("phase") != "F1":
        raise ValueError("config must declare M1-SA-v1/F1")
    output_root.mkdir(parents=True, exist_ok=False)
    records = []
    components = spec.get("baseline_components", {})
    candidates = spec.get("compatibility_candidate") or {}
    runs = {**components, **candidates}
    for candidate, executable in runs.items():
        p = subprocess.run([executable, "-c", PROBE], text=True, capture_output=True)
        record = {"candidate": candidate, "executable": executable,
                  "returncode": p.returncode, "stdout": p.stdout,
                  "stderr": p.stderr, "captured_at": datetime.now(timezone.utc).isoformat()}
        try: record["probe"] = json.loads(p.stdout)
        except json.JSONDecodeError: record["probe_parse_error"] = traceback.format_exc()
        (output_root / f"{candidate}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)
    conclusions = {}
    for r in records:
        bridges = r.get("probe", {}).get("bridges", {})
        if r["returncode"] != 0 or any(v.get("status") == "FAIL" for v in bridges.values()):
            conclusions[r["candidate"]] = "FAIL"
        elif any(v.get("status") == "BLOCKED" for v in bridges.values()):
            conclusions[r["candidate"]] = "BLOCKED"
        else:
            conclusions[r["candidate"]] = "PASS"
    manifest = {"contract": "M1-SA-v1", "phase": "F1", "namespace": spec["output_namespace"],
                "baseline_components": list(components),
                "compatibility_candidate": list(candidates),
                "compatibility_budget_used": 1 if candidates else 0,
                "records": [f"{r['candidate']}.json" for r in records],
                "conclusions": conclusions}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("qualify-runtime", nargs="?")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    return qualify(args.config, args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
