#!/usr/bin/env python3
"""ReplayVLA-P1 F2 renderer qualification (CPU, policy-free only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("F2 config must be a mapping")
    return value


def governed_environment(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in config["environment"].items()})
    env["MUJOCO_EGL_DEVICE_ID"] = str(config["renderer"]["selected_ordinal"])
    return env


def _identity() -> dict[str, Any]:
    result = {"executable": sys.executable, "python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "mujoco", "robosuite", "OpenGL"):
        try:
            module = __import__(name)
            result[name] = {"file": getattr(module, "__file__", None), "version": getattr(module, "__version__", None)}
        except Exception as exc:
            result[name] = {"error": repr(exc)}
    return result


def validate_f1t_predecessor(path: Path) -> dict[str, Any]:
    if path.name == "renderer_preflight_r1_egl0.yaml" or not path.is_file():
        raise RuntimeError("legacy or missing F1T predecessor")
    value = json.loads(path.read_text())
    if value.get("status") != "PASS" or value.get("authority") != "docs/replayvla-p1/06_d3_final_runtime_pivot.md":
        raise RuntimeError("F1T predecessor missing or drifted")
    return value


def deterministic_select(devices: list[dict[str, Any]]) -> int:
    selected = [d["ordinal"] for d in devices if d.get("software") is True]
    if not selected:
        raise RuntimeError("zero renderer candidates")
    if len(selected) != 1:
        raise RuntimeError("multiple renderer candidates")
    return int(selected[0])


def discover() -> int:
    from scripts.m1_renderer_admission import InjectedEGLBackend, query_egl_devices_two_call

    backend = InjectedEGLBackend.load()
    devices = query_egl_devices_two_call(backend)
    records = []
    for ordinal, device in enumerate(devices):
        extensions = backend.query_device_extensions(device)
        records.append({"ordinal": ordinal, "extensions": sorted(extensions), "software": "EGL_MESA_device_software" in extensions})
    selected = deterministic_select(records)
    return _emit({"status": "PASS", "runtime_identity": _identity(), "devices": records,
                  "selected_ordinal": selected, "selection_rule": "unique EGL_MESA_device_software candidate",
                  "environment": {k: v for k, v in sorted(os.environ.items()) if k in {"MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBERO_CONFIG_PATH"}}})


def _emit(value: dict[str, Any]) -> int:
    print(json.dumps(value, sort_keys=True))
    return 0


def worker(config_path: Path, worker_id: str) -> int:
    config = _load_config(config_path)
    env = governed_environment(config)
    construction = render_count = 0
    close_status = "FAIL"
    try:
        import libero.libero as libero_package
        asset_path = config.get("assets", {}).get("path")
        if not isinstance(asset_path, str) or not Path(asset_path).is_dir():
            raise RuntimeError("F2 assets path is not a directory")
        libero_package._assets_path_cache = asset_path
        from scripts.dcu_preflight import build_cpu_environment_runtime
        runtime = build_cpu_environment_runtime(config, phase="compare")
        construction = 1
        envs, env_obj = runtime["envs"], runtime["env"]
        frames = env_obj.call("render")
        render_count = 1
        frame = frames[0] if isinstance(frames, (list, tuple)) else frames
        import numpy as np
        array = np.asarray(frame)
        metadata = {"shape": list(array.shape), "dtype": str(array.dtype), "finite": bool(np.isfinite(array).all()), "min": float(array.min()), "max": float(array.max())}
        if tuple(array.shape) != (360, 360, 3) or not metadata["finite"]:
            raise RuntimeError(f"invalid render metadata: {metadata}")
        from OpenGL import EGL, GL
        display = EGL.eglGetCurrentDisplay()
        def _text(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="strict")
            return str(value)
        egl_identity = {"vendor": _text(EGL.eglQueryString(display, EGL.EGL_VENDOR)), "version": _text(EGL.eglQueryString(display, EGL.EGL_VERSION))}
        gl_identity = {"vendor": _text(GL.glGetString(GL.GL_VENDOR)), "renderer": _text(GL.glGetString(GL.GL_RENDERER)), "version": _text(GL.glGetString(GL.GL_VERSION))}
        from lerobot.envs import close_envs
        close_envs(envs)
        close_status = "PASS"
        return _emit({"status": "PASS", "worker_id": worker_id, "pid": os.getpid(), "runtime_identity": _identity(),
                      "construction_status": "PASS", "construction_observation": "official factory returned; internal init/reset/settle counts not independently observable",
                      "public_render_count": render_count, "render_metadata": metadata, "egl_identity": egl_identity,
                      "gl_identity": gl_identity,
                      "close_status": close_status, "forbidden_operation_count": 0, "parent_observed_exit": True, "replacement": False})
    except Exception:
        return _emit({"status": "FAIL", "worker_id": worker_id, "pid": os.getpid(), "runtime_identity": _identity(),
                      "construction_status": "PASS" if construction else "FAIL", "public_render_count": render_count,
                      "close_status": close_status, "forbidden_operation_count": 0, "parent_observed_exit": True,
                      "replacement": False, "traceback": traceback.format_exc()})


def run_step(argv: list[str], out: Path, name: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    stdout, stderr = out / f"{name}.stdout.log", out / f"{name}.stderr.log"
    stdout.write_text(result.stdout); stderr.write_text(result.stderr)
    parsed = None
    for line in reversed(result.stdout.splitlines()):
        try: parsed = json.loads(line); break
        except json.JSONDecodeError: pass
    record = parsed or {"status": "FAIL", "traceback": result.stderr}
    record.update({"argv": argv, "returncode": result.returncode, "stdout_path": str(stdout), "stderr_path": str(stderr),
                   "stdout_sha256": _sha256(stdout), "stderr_sha256": _sha256(stderr), "parent_observed_exit": True})
    return record


def adjudicate_workers(discovery: dict[str, Any], effective: dict[str, Any], workers: list[dict[str, Any]]) -> dict[str, Any]:
    status = "PASS"
    reasons = []
    if discovery.get("status") != "PASS" or not isinstance(discovery.get("selected_ordinal"), int): status, reasons = "BLOCKED", ["discovery"]
    if len(workers) != 3 or len({w.get("worker_id") for w in workers}) != 3: status, reasons = "FAIL", ["exactly three unique workers"]
    identities = {(json.dumps(w.get("egl_identity", {}), sort_keys=True), json.dumps(w.get("gl_identity", {}), sort_keys=True)) for w in workers}
    for worker in workers:
        if worker.get("status") != "PASS" or worker.get("public_render_count") != 1 or worker.get("close_status") != "PASS" or not worker.get("parent_observed_exit") or worker.get("forbidden_operation_count") != 0 or worker.get("replacement"):
            status, reasons = "FAIL", ["worker predicate"]
    if len(identities) != 1: status, reasons = "FAIL", ["inconsistent EGL/GL identity"]
    return {"status": status, "reasons": reasons, "discovery": discovery, "effective_config": effective, "workers": workers}


def run(config: Path, predecessor: Path, output: Path, cpu_python: str) -> int:
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    validate_f1t_predecessor(predecessor)
    base = _load_config(config)
    discovery_step = run_step([cpu_python, "-B", str(Path(__file__).resolve()), "discover"], output, "discovery", env={**os.environ, **base["environment"]})
    discovery = discovery_step
    effective = dict(base)
    effective["renderer"] = dict(base["renderer"])
    effective["renderer"]["selected_ordinal"] = discovery.get("selected_ordinal")
    effective["renderer"]["discovery_manifest"] = "discovery.stdout.log"
    effective["effective_config_sha256"] = _canonical_hash(effective)
    effective_path = output / "effective_config.json"
    effective_path.write_text(json.dumps(effective, indent=2, sort_keys=True) + "\n")
    workers = []
    for i in range(3):
        workers.append(run_step([cpu_python, "-B", str(Path(__file__).resolve()), "worker", "--config", str(effective_path), "--worker-id", f"worker-{i}"], output, f"worker-{i}", env=governed_environment(effective)))
    judgment = adjudicate_workers(discovery, effective, workers)
    (output / "adjudication.json").write_text(json.dumps(judgment, indent=2, sort_keys=True) + "\n")
    return 0 if judgment["status"] == "PASS" else 1


def main() -> int:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover")
    w = sub.add_parser("worker"); w.add_argument("--config", type=Path, required=True); w.add_argument("--worker-id", required=True)
    r = sub.add_parser("run"); r.add_argument("--config", type=Path, required=True); r.add_argument("--f1t-predecessor", type=Path, required=True); r.add_argument("--output-root", type=Path, required=True); r.add_argument("--cpu-python", required=True)
    a = p.parse_args()
    if a.cmd == "discover": return discover()
    if a.cmd == "worker": return worker(a.config, a.worker_id)
    return run(a.config, a.f1t_predecessor, a.output_root, a.cpu_python)


if __name__ == "__main__": raise SystemExit(main())
