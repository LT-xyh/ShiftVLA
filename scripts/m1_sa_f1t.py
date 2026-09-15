#!/usr/bin/env python3
"""F1T qualification for the existing safetensors/Torch transport boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.dcu_worker import (
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    load_tensor_bundle,
    save_tensor_bundle,
    validate_tensor_bundle,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "python": sys.version,
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "torch_config": torch.__config__.show(),
    }


def deterministic_request() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.arange(8, dtype=torch.float32).reshape(1, 8) / 8,
        "observation.images.image": torch.arange(3 * 360 * 360, dtype=torch.float32).reshape(1, 3, 360, 360) / 1_000_000,
        "observation.images.image2": torch.arange(3 * 360 * 360, dtype=torch.float32).reshape(1, 3, 360, 360).flip(-1) / 1_000_000,
        "observation.language.tokens": torch.arange(8, dtype=torch.int64).reshape(1, 8),
        "observation.language.attention_mask": torch.tensor([[True, True, True, True, True, False, False, False]]),
        "noise": torch.arange(50 * 32, dtype=torch.float32).reshape(1, 50, 32) / 10_000,
    }


def _emit(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, sort_keys=True))
    return 0


def create_request(path: Path) -> int:
    bundle = deterministic_request()
    save_tensor_bundle(path, bundle, schema=REQUEST_SCHEMA)
    return _emit({"status": "PASS", "runtime_identity": _identity(), "pid": os.getpid(),
                  "path": str(path.resolve()), "sha256": _sha256(path),
                  "validation": validate_tensor_bundle(bundle, schema=REQUEST_SCHEMA),
                  "numpy_bridge_used": False})


def dcu_transport(request: Path, response: Path) -> int:
    bundle = load_tensor_bundle(request, schema=REQUEST_SCHEMA)
    expected = deterministic_request()
    for key in REQUEST_SCHEMA:
        if not torch.equal(bundle[key], expected[key]):
            raise RuntimeError(f"request content mismatch: {key}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("logical cuda:0 device is unavailable")
    device = torch.device("cuda:0")
    roundtrip = {}
    for key, value in bundle.items():
        if value.is_floating_point():
            returned = value.to(device).cpu()
            if returned.shape != value.shape or returned.dtype != value.dtype or not torch.equal(returned, value):
                raise RuntimeError(f"K100 roundtrip mismatch: {key}")
            roundtrip[key] = {"shape": list(returned.shape), "dtype": str(returned.dtype), "exact": True}
    action = (torch.arange(7, dtype=torch.float32, device=device) / 10).reshape(1, 7)
    save_tensor_bundle(response, {"action": action}, schema=RESPONSE_SCHEMA)
    return _emit({"status": "PASS", "runtime_identity": _identity(), "pid": os.getpid(),
                  "request_sha256": _sha256(request), "response_sha256": _sha256(response),
                  "logical_device": "cuda:0", "device_name": torch.cuda.get_device_name(0),
                  "device_count": torch.cuda.device_count(), "torch_hip": getattr(torch.version, "hip", None),
                  "roundtrip": roundtrip, "numpy_bridge_used": False})


def validate_response(path: Path) -> int:
    bundle = load_tensor_bundle(path, schema=RESPONSE_SCHEMA)
    expected = (torch.arange(7, dtype=torch.float32) / 10).reshape(1, 7)
    if set(bundle) != {"action"} or not torch.equal(bundle["action"], expected):
        raise RuntimeError("response schema/content mismatch")
    return _emit({"status": "PASS", "runtime_identity": _identity(), "pid": os.getpid(),
                  "response_sha256": _sha256(path),
                  "validation": validate_tensor_bundle(bundle, schema=RESPONSE_SCHEMA),
                  "numpy_bridge_used": False})


def run_step(argv: list[str], log_dir: Path, name: str) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = log_dir / f"{name}.stdout", log_dir / f"{name}.stderr"
    result = subprocess.run(argv, text=True, capture_output=True)
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    parsed = None
    for line in reversed(result.stdout.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    record = {"argv": argv, "returncode": result.returncode,
              "status": "PASS" if result.returncode == 0 and parsed and parsed.get("status") == "PASS" else "FAIL",
              "stdout_path": str(stdout_path.resolve()), "stderr_path": str(stderr_path.resolve()),
              "runtime_identity": (parsed or {}).get("runtime_identity", {"executable": argv[0]}),
              "result": parsed}
    return record


def adjudicate(output_root: Path, trace: Path, repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    trace_ok = trace.is_file() and "DCU-side NumPy conversion required: NO" in trace.read_text()
    repetitions_ok = len(repetitions) == 2 and all(
        item.get("status") == "PASS"
        and item.get("fresh_processes") is True
        and not item.get("numpy_bridge_used")
        for item in repetitions
    )
    status = "PASS" if trace_ok and repetitions_ok else "BLOCKED"
    result = {"status": status, "trace_evidence": str(trace.resolve()),
              "trace_present": trace_ok, "repetitions": len(repetitions),
              "required_edges": {"dcu_numpy_to_torch": False}}
    (output_root / "judgment.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_qualification(cpu: str, dcu: str, output_root: Path, trace: Path, repetitions: int) -> int:
    output_root.mkdir(parents=True, exist_ok=False)
    script = str(Path(__file__).resolve())
    completed: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"phase": "F1T", "status": "FAIL", "trace_evidence": str(trace.resolve()), "repetitions": []}
    try:
        for index in range(repetitions):
            rep = output_root / f"repeat-{index + 1}"
            rep.mkdir()
            request, response = rep / "request.safetensors", rep / "response.safetensors"
            steps = [
                run_step([cpu, "-B", script, "create-request", "--path", str(request)], rep, "cpu-create"),
                run_step([dcu, "-B", script, "dcu-transport", "--request", str(request), "--response", str(response)], rep, "dcu-transport"),
                run_step([cpu, "-B", script, "validate-response", "--path", str(response)], rep, "cpu-validate"),
            ]
            rep_status = "PASS" if all(step["status"] == "PASS" for step in steps) else "FAIL"
            pids = [step.get("result", {}).get("pid") for step in steps if step.get("result")]
            record = {"status": rep_status, "steps": steps, "pids": pids,
                      "fresh_processes": len(pids) == 3 and len(set(pids)) == 3,
                      "numpy_bridge_used": any(bool(step.get("result", {}).get("numpy_bridge_used")) for step in steps)}
            if not record["fresh_processes"]:
                record["status"] = "FAIL"
            manifest["repetitions"].append(record)
            completed.append(record)
            if rep_status != "PASS":
                break
        judgment = adjudicate(output_root / "adjudication", trace, completed)
        manifest["status"] = "PASS" if judgment["status"] == "PASS" and len(completed) == repetitions == 2 else "BLOCKED"
    except Exception:
        manifest["traceback"] = traceback.format_exc()
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if manifest["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-request"); create.add_argument("--path", type=Path, required=True)
    dcu = sub.add_parser("dcu-transport"); dcu.add_argument("--request", type=Path, required=True); dcu.add_argument("--response", type=Path, required=True)
    validate = sub.add_parser("validate-response"); validate.add_argument("--path", type=Path, required=True)
    run = sub.add_parser("run"); run.add_argument("--cpu-python", required=True); run.add_argument("--dcu-python", required=True); run.add_argument("--output-root", type=Path, required=True); run.add_argument("--trace-evidence", type=Path, required=True); run.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.command == "create-request": return create_request(args.path)
    if args.command == "dcu-transport": return dcu_transport(args.request, args.response)
    if args.command == "validate-response": return validate_response(args.path)
    return run_qualification(args.cpu_python, args.dcu_python, args.output_root, args.trace_evidence, args.repetitions)


if __name__ == "__main__":
    raise SystemExit(main())
