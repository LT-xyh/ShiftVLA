"""Fake-only contract tests for the DCU-PREFLIGHT harness.

These tests intentionally do not construct LIBERO, import LeRobot runtime
modules, start a worker, or require an accelerator.  They exercise the
project-owned boundaries that make a later real run fail closed.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
import yaml

from scripts import dcu_worker
from scripts.dcu_preflight import (
    DCUPreflightError,
    EXPECTED_PATCH_SHA256,
    EXPECTED_OVERLAY_WHEEL_SHA256,
    FAILURE_MANIFEST_NAME,
    PATCH_PATH,
    build_worker_environment,
    compare_outputs,
    create_run_directory,
    enforce_one_step_boundary,
    generate_flow_noise,
    map_physical_to_logical,
    validate_config_identity,
    validate_forbidden_backends,
    validate_patch,
    write_failure_manifest,
)
from scripts.dcu_worker import (
    DCUWorkerClient,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    SubprocessWorkerTransport,
    decode_json_line,
    encode_json_line,
    load_tensor_bundle,
    save_tensor_bundle,
    validate_tensor_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/m0/dcu_preflight.yaml"


def test_config_identity_is_fail_closed() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["seed"] == 2027
    assert config["task"]["suite"] == "libero_spatial"
    assert config["task"]["task_id"] == 0
    assert config["runtime"]["physical_devices"] == [0, 1]

    validate_config_identity(config, expected_project_sha="project-sha", actual_project_sha="project-sha")
    with pytest.raises(DCUPreflightError, match="project SHA"):
        validate_config_identity(config, expected_project_sha="project-sha", actual_project_sha="other")

    malformed = deepcopy(config)
    del malformed["checkpoint"]["revision"]
    with pytest.raises(DCUPreflightError, match="checkpoint.revision"):
        validate_config_identity(malformed, expected_project_sha="project-sha", actual_project_sha="project-sha")


def test_safetensors_request_response_schema_and_finite_checks(tmp_path: Path) -> None:
    request_path = tmp_path / "request.safetensors"
    request = {
        "observation.state": torch.zeros((1, 8), dtype=torch.float32),
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.images.image2": torch.ones((1, 3, 360, 360), dtype=torch.float32),
        "observation.language.tokens": torch.ones((1, 4), dtype=torch.int64),
        "observation.language.attention_mask": torch.ones((1, 4), dtype=torch.bool),
        "noise": torch.zeros((1, 50, 32), dtype=torch.float32),
    }
    save_tensor_bundle(request_path, request, schema=REQUEST_SCHEMA)
    loaded = load_tensor_bundle(request_path, schema=REQUEST_SCHEMA)
    assert loaded["observation.state"].shape == (1, 8)
    assert loaded["noise"].shape == (1, 50, 32)
    assert validate_tensor_bundle(loaded, schema=REQUEST_SCHEMA)["finite"] is True

    response_path = tmp_path / "response.safetensors"
    response = {"action_chunk": torch.zeros((1, 50, 7), dtype=torch.float32)}
    save_tensor_bundle(response_path, response, schema=RESPONSE_SCHEMA)
    assert load_tensor_bundle(response_path, schema=RESPONSE_SCHEMA)["action_chunk"].shape == (1, 50, 7)

    response["action_chunk"][0, 0, 0] = float("nan")
    with pytest.raises(DCUPreflightError, match="finite"):
        validate_tensor_bundle(response, schema=RESPONSE_SCHEMA)

    with pytest.raises(DCUPreflightError, match="shape"):
        validate_tensor_bundle({"action_chunk": torch.zeros((50, 7))}, schema=RESPONSE_SCHEMA)


def test_request_schema_uses_official_language_keys_and_mask_dtype() -> None:
    assert set(REQUEST_SCHEMA) == {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
        "observation.language.tokens",
        "observation.language.attention_mask",
        "noise",
    }
    assert REQUEST_SCHEMA["observation.language.tokens"][1] == (torch.int64,)
    assert REQUEST_SCHEMA["observation.language.attention_mask"][1] == (torch.bool,)

    with pytest.raises(DCUPreflightError, match="keys"):
        validate_tensor_bundle(
            {
                "observation.state": torch.zeros((1, 8), dtype=torch.float32),
                "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
                "observation.images.image2": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
                "language.tokens": torch.ones((1, 4), dtype=torch.int64),
                "language.attention_mask": torch.ones((1, 4), dtype=torch.bool),
                "noise": torch.zeros((1, 50, 32), dtype=torch.float32),
            },
            schema=REQUEST_SCHEMA,
        )
    with pytest.raises(DCUPreflightError, match="dtype"):
        validate_tensor_bundle(
            {
                "observation.state": torch.zeros((1, 8), dtype=torch.float32),
                "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
                "observation.images.image2": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
                "observation.language.tokens": torch.ones((1, 4), dtype=torch.int64),
                "observation.language.attention_mask": torch.ones((1, 4), dtype=torch.int64),
                "noise": torch.zeros((1, 50, 32), dtype=torch.float32),
            },
            schema=REQUEST_SCHEMA,
        )


def test_worker_and_preflight_export_the_same_error_class() -> None:
    from scripts import dcu_preflight
    from scripts import dcu_worker

    assert dcu_worker.DCUPreflightError is dcu_preflight.DCUPreflightError
    assert not hasattr(dcu_worker, "_error")
    with pytest.raises(dcu_worker.DCUPreflightError, match="command"):
        encode_json_line({"id": "bad", "command": "not-supported"})


def _write_fake_worker(path: Path) -> Path:
    path.write_text(
        """
import json
import os
import sys

mode = sys.argv[1]
count = 0
for raw in sys.stdin:
    request = json.loads(raw)
    count += 1
    expected = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\\n"
    if raw != expected:
        raise SystemExit("request was not canonical JSONL")
    print("fake-worker-stderr", file=sys.stderr, flush=True)
    response = {
        "id": request["id"],
        "ok": True,
        "command": request["command"],
        "env": os.environ.get("FAKE_WORKER_ENV"),
    }
    if mode == "wrong-id" and count == 1:
        response["id"] = "unexpected-id"
    if mode == "noncanonical" and count == 1:
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\\n")
        sys.stdout.flush()
    if request["command"] in {"shutdown", "close"}:
        break
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_subprocess_transport_uses_explicit_argv_env_and_separate_stderr(tmp_path: Path) -> None:
    worker = _write_fake_worker(tmp_path / "fake_worker.py")
    transport = SubprocessWorkerTransport(
        [sys.executable, str(worker), "literal;not-shell"],
        env={"FAKE_WORKER_ENV": "present"},
    )
    response = transport.request("ping", value=3)
    assert response["ok"] is True
    assert response["command"] == "ping"
    assert response["env"] == "present"
    transport.close()
    assert "fake-worker-stderr" in transport.stderr_text
    assert transport.process.returncode == 0
    with pytest.raises(DCUPreflightError, match="closed"):
        transport.request("ping")


@pytest.mark.parametrize("mode,match", [("wrong-id", "id"), ("noncanonical", "canonical")])
def test_subprocess_transport_rejects_unmatched_or_noncanonical_responses(
    tmp_path: Path, mode: str, match: str
) -> None:
    worker = _write_fake_worker(tmp_path / f"fake_worker_{mode}.py")
    transport = SubprocessWorkerTransport([sys.executable, str(worker), mode], env={})
    try:
        with pytest.raises(DCUPreflightError, match=match):
            transport.request("ping")
    finally:
        transport.close()


def test_json_line_worker_protocol_is_strict_and_json_only() -> None:
    line = encode_json_line(
        {"id": "r1", "command": "predict_action_chunk", "request_path": "/tmp/request.safetensors"}
    )
    assert line.endswith("\n")
    decoded = decode_json_line(line)
    assert decoded["id"] == "r1"
    assert decoded["command"] == "predict_action_chunk"
    assert json.loads(line)["request_path"].endswith(".safetensors")
    with pytest.raises(DCUPreflightError, match="JSON object"):
        decode_json_line("[]\n")
    with pytest.raises(DCUPreflightError, match="command"):
        encode_json_line({"id": "r2"})


class _FakeTransport:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[dict[str, object]] = []
        self.reset_count = 0
        self.select_count = 0

    def request(self, command: str, **payload: object) -> dict[str, object]:
        self.requests.append({"command": command, **payload})
        if command == "reset":
            self.reset_count += 1
            return {"ok": True, "queue_length": 0}
        if command == "predict_action_chunk":
            path = self.root / f"chunk-{len(self.requests)}.safetensors"
            save_tensor_bundle(path, {"action_chunk": torch.arange(350, dtype=torch.float32).reshape(1, 50, 7)}, schema=RESPONSE_SCHEMA)
            return {"ok": True, "response_path": str(path), "queue_length_before": 0, "queue_length_after": 50}
        if command == "select_action":
            self.select_count += 1
            path = self.root / f"action-{self.select_count}.safetensors"
            save_tensor_bundle(path, {"action": torch.full((1, 7), 0.25)}, schema=RESPONSE_SCHEMA)
            return {
                "ok": True,
                "response_path": str(path),
                "queue_length_before": 0 if self.select_count == 1 else 49,
                "queue_length_after": 49 if self.select_count == 1 else 48,
                "new_chunk_generated": self.select_count == 1,
            }
        raise AssertionError(command)

    def close(self) -> None:
        pass


def test_client_reset_queue_and_action_semantics_use_persisted_tensors(tmp_path: Path) -> None:
    transport = _FakeTransport(tmp_path)
    client = DCUWorkerClient(transport=transport, action_dim=7, chunk_size=50, n_action_steps=1)
    client.reset()
    chunk = client.predict_action_chunk(tmp_path / "request.safetensors")
    action = client.select_action(tmp_path / "request.safetensors")
    assert chunk.shape == (1, 50, 7)
    assert action.shape == (1, 7)
    assert transport.reset_count == 1
    assert transport.select_count == 1
    assert transport.requests[0]["command"] == "reset"
    assert transport.requests[-1]["command"] == "select_action"
    assert client.last_queue_evidence["new_chunk_generated"] is True
    unreset_client = DCUWorkerClient(transport=transport, action_dim=7, chunk_size=50, n_action_steps=1)
    with pytest.raises(DCUPreflightError, match="reset"):
        unreset_client.select_action(tmp_path / "request.safetensors")


def test_client_reset_forwards_optional_nonnegative_seed() -> None:
    transport = _FakeTransport(Path("/tmp"))
    client = DCUWorkerClient(transport=transport, action_dim=7, chunk_size=50, n_action_steps=1)
    client.reset(seed=2027)
    assert transport.requests[0] == {"command": "reset", "seed": 2027}

    with pytest.raises(DCUPreflightError, match="seed"):
        client.reset(seed=-1)
    with pytest.raises(DCUPreflightError, match="seed"):
        client.reset(seed=True)


def test_explicit_noise_is_shape_checked_and_reproducibly_hashed(tmp_path: Path) -> None:
    first, first_hash = generate_flow_noise(seed=2027, shape=(1, 50, 32))
    second, second_hash = generate_flow_noise(seed=2027, shape=(1, 50, 32))
    assert first.shape == (1, 50, 32)
    assert first_hash == second_hash
    assert torch.equal(first, second)
    assert first_hash == hashlib.sha256(first.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    with pytest.raises(DCUPreflightError, match="noise shape"):
        generate_flow_noise(seed=2027, shape=(1, 49, 32))
    noise_path = tmp_path / "noise.safetensors"
    save_tensor_bundle(noise_path, {"noise": first}, schema={"noise": ((1, 50, 32), (torch.float32,))})
    assert load_tensor_bundle(noise_path, schema={"noise": ((1, 50, 32), (torch.float32,))})["noise"].shape == (1, 50, 32)


def test_comparison_metrics_cover_normalized_and_final_actions() -> None:
    cpu_chunk = torch.zeros((1, 50, 7), dtype=torch.float32)
    dcu_chunk = torch.full((1, 50, 7), 0.5, dtype=torch.float32)
    cpu_action = torch.zeros((1, 7), dtype=torch.float32)
    dcu_action = torch.full((1, 7), 0.25, dtype=torch.float32)
    metrics = compare_outputs(cpu_chunk, dcu_chunk, cpu_action, dcu_action, cpu_latency=0.2, dcu_latency=0.3, dcu_peak_memory=123)
    assert metrics["chunk_shape"] == [1, 50, 7]
    assert metrics["action_shape"] == [1, 7]
    assert metrics["normalized_max_abs_diff"] == pytest.approx(0.5)
    assert metrics["normalized_mean_abs_diff"] == pytest.approx(0.5)
    assert metrics["final_action_max_abs_diff"] == pytest.approx(0.25)
    assert metrics["cpu_latency_seconds"] == pytest.approx(0.2)
    assert metrics["dcu_peak_memory_bytes"] == 123
    with pytest.raises(DCUPreflightError, match="finite"):
        compare_outputs(cpu_chunk, dcu_chunk, cpu_action, torch.full((1, 7), float("inf")))


def test_physical_device_isolated_as_logical_cuda_zero() -> None:
    assert map_physical_to_logical(1) == {"physical_index": 1, "hip_visible_devices": "1", "logical_device": "cuda:0"}
    env = build_worker_environment(1, {"HF_HUB_OFFLINE": "1"})
    assert env["HIP_VISIBLE_DEVICES"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    with pytest.raises(DCUPreflightError, match="physical device"):
        map_physical_to_logical(-1)


def test_forbidden_backend_detection_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert validate_forbidden_backends(imported_modules=[])["forbidden"] == []
    fake = ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", fake)
    with pytest.raises(DCUPreflightError, match="forbidden backend"):
        validate_forbidden_backends(imported_modules=sys.modules)


class _OneStepEnv:
    def __init__(self) -> None:
        self.reset_calls: list[int] = []
        self.actions: list[np.ndarray] = []

    def reset(self, *, seed: int) -> dict[str, object]:
        self.reset_calls.append(seed)
        return {"observation": "fake"}

    def step(self, action: np.ndarray) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        self.actions.append(action.copy())
        return ({"observation": "next"}, 1.0, True, False, {"is_success": True})


class _OneStepClient:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.select_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(self, observation: object) -> np.ndarray:
        self.select_calls += 1
        return np.zeros((1, 7), dtype=np.float32)


def test_concurrency_worker_boundary_is_exactly_reset_one_decision_one_step() -> None:
    env = _OneStepEnv()
    client = _OneStepClient()
    result = enforce_one_step_boundary(env, client, seed=2027)
    assert result["decision_count"] == 1
    assert result["env_step_count"] == 1
    assert env.reset_calls == [2027]
    assert client.reset_calls == 1
    assert client.select_calls == 1
    assert len(env.actions) == 1
    assert result["done"] is True


def test_failure_manifest_and_run_directories_never_overwrite(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path)
    second = create_run_directory(tmp_path)
    assert first != second
    failure = write_failure_manifest(first, RuntimeError("expected"), context={"mode": "compare"})
    assert failure.name == FAILURE_MANIFEST_NAME
    assert json.loads(failure.read_text())["status"] == "FAIL"
    with pytest.raises(FileExistsError):
        write_failure_manifest(first, RuntimeError("second"), context={})


def test_patch_allowlist_hash_and_apply_check_against_pinned_source() -> None:
    assert EXPECTED_PATCH_SHA256 == "3ebaba1e8d305c93d0c758c9c9d3a575215ebe3513e486ae5503da444f8d573d"
    assert EXPECTED_OVERLAY_WHEEL_SHA256 == "d03459fe530b556398bab41ad7de60b47c3354ffa7a5ce44db6249571fcdd2ff"
    source = ROOT / "external/lerobot"
    evidence = validate_patch(PATCH_PATH, source_root=source)
    assert evidence["sha256"] == EXPECTED_PATCH_SHA256
    assert evidence["files"] == {
        "src/lerobot/datasets/aggregate.py",
        "src/lerobot/datasets/streaming_dataset.py",
        "src/lerobot/motors/motors_bus.py",
        "src/lerobot/processor/pipeline.py",
        "src/lerobot/utils/io_utils.py",
    }
    assert evidence["insertions"] == 17
    assert evidence["deletions"] == 13
