"""Fake-only tests for the explicit DCU preflight phase runner.

These tests must not construct a model, simulator, LeRobot environment, or
accelerator.  Real phase entry points are exercised only through injected
dependencies and project-owned validation seams.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from scripts.dcu_preflight import (
    DCUPreflightError,
    EXPECTED_DCU_LOCK_SHA256,
    EXPECTED_RENDERER_ENV,
    ONE_STEP_CHILD_TOKEN,
    FeatureOnlyRemotePolicy,
    build_concurrency_child_command,
    dispatch_phase,
    run_phase,
    validate_config_identity,
    validate_runner_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/m0/dcu_preflight.yaml"


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _complete_concurrency_child_result(physical_device: int) -> dict[str, object]:
    """A complete child evidence fixture for parent-boundary tests."""

    nested = {
        "physical_k100": physical_device,
        "HIP_VISIBLE_DEVICES": str(physical_device),
        "CUDA_VISIBLE_DEVICES": "0",
        "torch_logical_device": "cuda:0",
    }
    return {
        "status": "PASS",
        "physical_device": physical_device,
        "logical_device": "cuda:0",
        "reset_count": 1,
        "decision_count": 2,
        "env_step_count": 2,
        "model_load_success": True,
        "egl": {
            "eglQueryDevicesEXT_device_count": 9,
            "selected_MUJOCO_EGL_DEVICE_ID": "8",
            "test_device": {"returncode": 0},
        },
        "gl": {
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
            "version": "3.1 Mesa 21.1.5",
        },
        "environment": {
            "cpu_child": {
                "HIP_VISIBLE_DEVICES": None,
                "CUDA_VISIBLE_DEVICES": None,
                "MUJOCO_EGL_DEVICE_ID": "8",
            }
        },
        "device": {
            "physical_k100": physical_device,
            "nested_worker_hip_visible_devices": str(physical_device),
            "nested_worker_cuda_visible_devices": "0",
            "torch_logical_device": "cuda:0",
            "torch_device_name": "fake-k100",
        },
        "worker": {
            "physical_device": physical_device,
            "compute_environment": nested,
            "torch_logical_device": "cuda:0",
            "torch_device_name": "fake-k100",
            "model_load_success": True,
        },
        "action_evidence": [
            {"decision_index": 0, "shape": [1, 7], "dtype": "float32", "finite": True},
            {"decision_index": 1, "shape": [1, 7], "dtype": "float32", "finite": True},
        ],
        "network": {"offline": True, "hub_fallback": False},
        "latency": {
            "inference_seconds": [0.1, 0.1],
            "env_step_seconds": [0.1, 0.1],
            "worker_wall_seconds": [0.1, 0.1],
            "render_seconds": [0.1],
        },
        "memory": {"dcu_peak_memory_bytes": 123, "worker_response": {}},
    }


class _FakeEnv:
    def __init__(self) -> None:
        self.reset_count = 0
        self.step_count = 0
        self.render_count = 0

    def reset(self, *, seed: int) -> dict[str, object]:
        self.reset_count += 1
        return {"seed": seed}

    def render(self) -> np.ndarray:
        self.render_count += 1
        return np.zeros((360, 360, 3), dtype=np.uint8)

    def step(self, action: np.ndarray) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        self.step_count += 1
        return {"seed": 2027}, 0.0, True, False, {"is_success": True}


class _FakeClient:
    def __init__(self) -> None:
        self.reset_count = 0
        self.select_count = 0

    def reset(self, *, seed: int) -> dict[str, object]:
        self.reset_count += 1
        return {"queue_length_after": 0, "seed": seed}

    def select_action(self, request_path: Path) -> torch.Tensor:
        self.select_count += 1
        return torch.zeros((1, 7), dtype=torch.float32)

    def close(self) -> None:
        return None


def test_feature_only_remote_policy_rejects_noise_and_preserves_queue_evidence() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def select_action(self, request_path: Path) -> torch.Tensor:
            self.calls.append(request_path)
            return torch.ones((1, 7), dtype=torch.float32)

    client = Client()
    policy = FeatureOnlyRemotePolicy(client, request_writer=lambda bundle: Path(bundle["path"]))
    action = policy.select_action({"path": "/tmp/features.safetensors"})
    assert tuple(action.shape) == (1, 7)
    assert client.calls == [Path("/tmp/features.safetensors")]
    assert policy.last_queue_evidence == {}
    with pytest.raises(DCUPreflightError, match="feature-only"):
        policy.select_action({"path": "/tmp/features.safetensors", "noise": torch.zeros((1, 50, 32))})


def test_feature_only_remote_policy_filters_non_wire_fields_and_checks_queue() -> None:
    features = {
        "observation.state": torch.zeros((1, 8), dtype=torch.float32),
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.images.image2": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.language.tokens": torch.zeros((1, 4), dtype=torch.int64),
        "observation.language.attention_mask": torch.ones((1, 4), dtype=torch.bool),
        "task": ["ignored complementary data"],
    }
    captured: list[dict[str, object]] = []

    class Client:
        last_queue_evidence = {
            "queue_length_before": 0,
            "queue_length_after": 0,
            "new_chunk_generated": True,
        }

        def select_action(self, path: Path) -> torch.Tensor:
            assert path.suffix == ".safetensors"
            return torch.zeros((1, 7), dtype=torch.float32)

    policy = FeatureOnlyRemotePolicy(
        Client(),
        request_writer=lambda bundle: captured.append(dict(bundle)) or Path("/tmp/features.safetensors"),
    )
    policy.select_action(features)
    assert len(captured) == 1
    assert set(captured[0]) == {
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
        "observation.language.tokens",
        "observation.language.attention_mask",
    }
    assert "task" not in captured[0]
    assert policy.last_queue_evidence["new_chunk_generated"] is True


def test_dispatch_phase_does_not_chain_compare_into_closed_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("scripts.dcu_preflight.run_compare", lambda **_: calls.append("compare") or 0)
    monkeypatch.setattr("scripts.dcu_preflight.run_closed_loop", lambda **_: calls.append("closed-loop") or 0)
    config = _config()
    assert dispatch_phase("compare", config=config, expected_project_sha="sha", actual_project_sha="sha") == 0
    assert calls == ["compare"]


def test_config_uses_accepted_locks_directory_semantics_and_bounded_timeouts() -> None:
    config = _config()
    assert config["mode"] == "explicit_phase"
    assert config["runtime"]["cpu_runtime_lock"].endswith("runtime/locks/shiftvla-libero-runtime.txt")
    assert config["runtime_lock_sha256"] == "921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc"
    assert config["dcu_runtime_lock_sha256"] == "cc507d48c64e72a9d2f6552bc5fa638217f9e6a2addef0067ded3d0adce30ed2"
    assert config["libero_config_path"].endswith("/shiftvla-libero-config")
    assert config["libero_config"]["path"].endswith("/shiftvla-libero-config/config.yaml")
    assert config["runtime"]["worker_startup_timeout_seconds"] == 300
    assert config["runtime"]["worker_forward_timeout_seconds"] == 300
    assert config["runtime"]["worker_shutdown_timeout_seconds"] == 30
    assert config["concurrency"]["steps_per_worker"] == 2


def test_dcu_runtime_lock_closure_records_two_worker_evidence() -> None:
    lock_path = ROOT / "runtime" / "locks" / "shiftvla-libero-dcu-runtime.txt"
    archive_path = (
        ROOT
        / "runtime"
        / "locks"
        / "archive"
        / "shiftvla-libero-dcu-runtime.c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a.txt"
    )
    lock_text = lock_path.read_text(encoding="utf-8")
    prefix, _ = lock_text.split("[pip_freeze_all]\n", 1)
    config = _config()
    current_lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert archive_path.is_file()
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == (
        "c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a"
    )
    assert f"validated_input_lock_path: {archive_path}\n" in prefix
    assert current_lock_sha256 == config["dcu_runtime_lock_sha256"]
    assert current_lock_sha256 == EXPECTED_DCU_LOCK_SHA256
    assert "status: VALIDATED_SINGLE_CARD_AND_TWO_INDEPENDENT_WORKERS\n" in prefix
    assert "acceptance: CLOSED_FOR_SINGLE_CARD_AND_TWO_INDEPENDENT_WORKERS\n" in prefix
    assert "validated_input_lock_sha256: c7b912e71c2320ed5312b9844d2284c697b0a38ae34759702b57c062ff04956a\n" in prefix
    assert "validation_run_directory: /public/home/xuyinghao/workspace/vla/runs/dcu_preflight/20260828T024505Z\n" in prefix
    assert "validation_parent_manifest_sha256: 6a59f62c0714a397486d126029d13e3aeb84ecaa450153450baedd7de8d9cf9c\n" in prefix
    assert "validated_workers: 2\n" in prefix
    assert "validated_steps_per_worker: 2\n" in prefix
    assert "m0_baseline: NOT_RUN\n" in prefix
    for evidence_line in (
        "validation_status: PASS\n",
        "validation_project_sha: 38517cbcac9f08154fe93bfe9e25a650c464ee45\n",
        "validation_parent_manifest_sha256: 6a59f62c0714a397486d126029d13e3aeb84ecaa450153450baedd7de8d9cf9c\n",
        "validation_concurrency_result_sha256: a997aefa945e7fca83c8d118d02fd35ca89d71d671353445b110e16fda22c8ae\n",
        "validation_child0_manifest_sha256: 6bf4bce51dd761261f29c4f9528935b1a7f0e694ebdfbc03f5f628a376c043e9\n",
        "validation_child1_manifest_sha256: b92e031b76482f3b231e6d56fe5ce7c28610536e03f88e5624d7e2f8c5e66b87\n",
        "validation_child0_result_sha256: 80b3e0d9962fbab5fc2a25002095aef66079436b54e1e5ba17799363425ecd65\n",
        "validation_child1_result_sha256: add1b7cc3a55600e7dccdc0f7f053304c862810955f35ab520a2aa3e7f1a5632\n",
        "worker_a_physical_k100: 0\n",
        "worker_b_physical_k100: 1\n",
        "worker_a_cpu_hip_visible_devices: UNSET\n",
        "worker_a_cpu_cuda_visible_devices: UNSET\n",
        "worker_b_cpu_hip_visible_devices: UNSET\n",
        "worker_b_cpu_cuda_visible_devices: UNSET\n",
        "worker_a_nested_hip_visible_devices: 0\n",
        "worker_b_nested_hip_visible_devices: 1\n",
        "worker_a_nested_cuda_visible_devices: 0\n",
        "worker_b_nested_cuda_visible_devices: 0\n",
        "worker_a_torch_logical_device: cuda:0\n",
        "worker_b_torch_logical_device: cuda:0\n",
        "eglQueryDevicesEXT_device_count: 9\n",
        "selected_MUJOCO_EGL_DEVICE_ID: 8\n",
        "egl_test_device_returncode: 0\n",
        "gl_vendor: Mesa/X.org\n",
        "gl_renderer: llvmpipe (LLVM 12.0.0, 256 bits)\n",
        "gl_version: 3.1 Mesa 21.1.5\n",
        "worker_a_model_load_success: true\n",
        "worker_b_model_load_success: true\n",
        "worker_a_reset_count: 1\n",
        "worker_b_reset_count: 1\n",
        "worker_a_policy_decisions: 2\n",
        "worker_b_policy_decisions: 2\n",
        "worker_a_env_steps: 2\n",
        "worker_b_env_steps: 2\n",
        "worker_a_actions: finite_float32_shape_1x7\n",
        "worker_b_actions: finite_float32_shape_1x7\n",
        "offline_execution: true\n",
        "hub_fallback: false\n",
        "full_280_step_episode: NOT_RUN\n",
        "m0_baseline: NOT_RUN\n",
    ):
        assert evidence_line in prefix


def test_concurrency_step_boundary_is_fail_closed() -> None:
    config = _config()
    config["concurrency"] = dict(config["concurrency"])
    config["concurrency"]["steps_per_worker"] = 1
    with pytest.raises(DCUPreflightError, match="steps_per_worker"):
        validate_runner_config(
            config,
            phase="concurrency",
            expected_project_sha="sha",
            actual_project_sha="sha",
        )


def test_host_egl_ordinal_rejects_device_zero_and_requires_device_eight() -> None:
    config = _config()
    assert EXPECTED_RENDERER_ENV["MUJOCO_EGL_DEVICE_ID"] == "8"
    assert config["renderer"]["MUJOCO_EGL_DEVICE_ID"] == "8"
    validate_config_identity(
        config,
        expected_project_sha="sha",
        actual_project_sha="sha",
    )

    zero_config = dict(config)
    zero_config["renderer"] = dict(config["renderer"])
    zero_config["renderer"]["MUJOCO_EGL_DEVICE_ID"] = "0"
    with pytest.raises(DCUPreflightError, match="renderer environment"):
        validate_config_identity(
            zero_config,
            expected_project_sha="sha",
            actual_project_sha="sha",
        )


def test_config_mode_is_not_a_phase_selector() -> None:
    config = _config()
    validate_config_identity(
        config,
        expected_project_sha="sha",
        actual_project_sha="sha",
    )
    config["mode"] = "compare"
    with pytest.raises(DCUPreflightError, match="mode"):
        validate_config_identity(
            config,
            expected_project_sha="sha",
            actual_project_sha="sha",
        )


def test_forbidden_backend_gate_includes_mamba_variants() -> None:
    from scripts import dcu_preflight

    assert {"mamba", "mamba_ssm"}.issubset(dcu_preflight.FORBIDDEN_BACKENDS)
    with pytest.raises(DCUPreflightError, match="mamba"):
        dcu_preflight.validate_forbidden_backends(imported_modules={"mamba_ssm"})


def test_dcu_runtime_freeze_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze = b"demo-package==1.0\n"
    freeze_hash = __import__("hashlib").sha256(freeze).hexdigest()
    lock = tmp_path / "dcu-runtime.txt"
    lock.write_text(
        "\n".join(
            [
                "python_executable: /fake/dcu-python",
                "pip_freeze_all_lines: 1",
                f"pip_freeze_all_sha256: {freeze_hash}",
                "[packages]",
                "pip_check:",
                "No broken requirements found.",
                "",
                "[pip_freeze_all]",
                freeze.decode().rstrip("\n"),
                "",
            ]
        ),
        encoding="utf-8",
    )

    class Completed:
        returncode = 0
        stdout = b"other-package==2.0\n"
        stderr = b""

    monkeypatch.setattr("scripts.dcu_preflight.subprocess.run", lambda *_, **__: Completed())
    with pytest.raises(DCUPreflightError, match="freeze"):
        __import__("scripts.dcu_preflight", fromlist=["_verify_live_runtime_lock"])._verify_live_runtime_lock(
            lock, Path("/fake/dcu-python")
        )


def test_hardware_toolchain_probe_requires_successful_read_only_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_command(args: list[str], **_: object) -> dict[str, object]:
        calls.append(args)
        return {"args": args, "returncode": 0, "stdout": "version", "stderr": ""}

    monkeypatch.setattr("scripts.dcu_preflight._run_command", fake_command)
    from scripts import dcu_preflight

    evidence = dcu_preflight._probe_hardware_toolchain()
    assert evidence["hy_smi"]["returncode"] == 0
    assert evidence["hipcc"]["returncode"] == 0
    assert calls == [["hy-smi"], ["hipcc", "--version"]]


def _fake_feature_bundle() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=torch.float32),
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.images.image2": torch.ones((1, 3, 360, 360), dtype=torch.float32),
        "observation.language.tokens": torch.zeros((1, 20), dtype=torch.int64),
        "observation.language.attention_mask": torch.ones((1, 20), dtype=torch.bool),
    }


def test_reconstructed_same_observation_gate_checks_archived_floating_min_max() -> None:
    from scripts import dcu_preflight

    features = _fake_feature_bundle()
    actual = dcu_preflight._feature_metadata(features)
    reference = {"first_decision": {"observation_policy_processor": actual["tensors"]}}
    evidence = dcu_preflight._assert_reference_feature_metadata(features, reference)
    assert evidence["matched"] is True
    assert "not a historic tensor bitwise hash" in evidence["same_initial_observation"]
    reference["first_decision"]["observation_policy_processor"]["observation.state"]["max"] = 7.5
    with pytest.raises(DCUPreflightError, match="feature metadata"):
        dcu_preflight._assert_reference_feature_metadata(features, reference)


def test_gl_identity_gate_uses_the_m0_probe_and_assertion(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import dcu_preflight

    monkeypatch.setattr("scripts.m0_smoke._gl_evidence", lambda: {"vendor": "fake", "renderer": "llvmpipe", "version": "3.1"})
    monkeypatch.setattr("scripts.m0_smoke._assert_gl_identity", lambda value: {**value, "checked": True})
    assert dcu_preflight._gl_identity_after_render()["checked"] is True


def test_egl_probe_evidence_uses_actual_count_and_selected_egl_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    calls: list[list[str]] = []

    class Completed:
        def __init__(self, args: list[str]) -> None:
            self.returncode = 0
            self.stdout = b"9" if args[-1].endswith("query_devices") else b"ok"
            self.stderr = b""

    def fake_run(args: list[str], **_: object) -> Completed:
        calls.append(args)
        return Completed(args)

    monkeypatch.setattr("scripts.dcu_preflight.subprocess.run", fake_run)
    evidence = dcu_preflight._egl_probe_evidence(selected_device="8")
    assert evidence["eglQueryDevicesEXT_device_count"] == 9
    assert evidence["selected_MUJOCO_EGL_DEVICE_ID"] == "8"
    assert calls[0][-1].endswith("/query_devices")
    assert calls[1][-2].endswith("/test_device")
    assert calls[1][-1] == "8"


def test_init_state_evidence_requires_one_exact_zero_without_reset() -> None:
    from scripts import dcu_preflight

    class Env:
        def __init__(self, value: object) -> None:
            self.value = value
            self.reset_count = 0

        def get_attr(self, name: str) -> list[object]:
            assert name == "init_state_id"
            return [self.value]

    env = Env(0)
    evidence = dcu_preflight._init_state_evidence(env)
    assert evidence["value"] == 0
    assert env.reset_count == 0
    with pytest.raises(DCUPreflightError, match="init_state_id"):
        dcu_preflight._init_state_evidence(Env(1))


def test_phase_manifest_records_input_resolved_provenance_and_noise_semantics(tmp_path: Path) -> None:
    config = _config()

    def fake_compare(**_: object) -> dict[str, object]:
        return {"status": "PASS", "comparison": {"noise_sha256": "a" * 64}}

    assert run_phase(
        "compare",
        config=config,
        expected_project_sha="sha",
        actual_project_sha="sha",
        output_root=tmp_path,
        config_path=CONFIG_PATH,
        preflight_gate=lambda **_: {"environment": {"toolchain": {"hy_smi": {"returncode": 0}}}},
        phase_runner=fake_compare,
    ) == 0
    manifest = json.loads(next(tmp_path.iterdir()).joinpath("run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_config"]["sha256"]
    assert manifest["resolved_config_artifact"]["sha256"]
    assert manifest["project"]["actual_sha"] == "sha"
    assert manifest["runtime_locks"]["dcu"]["sha256"] == config["dcu_runtime_lock_sha256"]
    assert manifest["artifacts"]["checkpoint"]["revision"] == config["checkpoint"]["revision"]
    assert manifest["seed"] == 2027
    assert manifest["action_noise"]["mode"] == "explicit_same_noise"
    assert manifest["action_noise"]["sha256"] == "a" * 64
    assert manifest["runtime_hardware_evidence_index"]["preflight"] == "preflight.json"


def test_compare_runner_never_steps_environment(tmp_path: Path) -> None:
    env = _FakeEnv()
    client = _FakeClient()

    def fake_compare(**_: object) -> dict[str, object]:
        env.reset(seed=2027)
        env.render()
        client.reset(seed=2027)
        return {"status": "PASS", "env": env, "client": client}

    result = run_phase(
        "compare",
        config=_config(),
        expected_project_sha="sha",
        actual_project_sha="sha",
        output_root=tmp_path,
        preflight_gate=lambda **_: {"gate": "fake-pass"},
        phase_runner=fake_compare,
    )
    assert result == 0
    assert env.reset_count == 1
    assert env.render_count == 1
    assert env.step_count == 0
    manifest = next(tmp_path.iterdir()) / "run_manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "PASS"


def test_one_step_child_command_maps_physical_device_and_hides_token() -> None:
    command = build_concurrency_child_command(
        config_path=CONFIG_PATH,
        expected_project_sha="sha",
        physical_device=1,
        run_directory=Path("/tmp/child"),
        token=ONE_STEP_CHILD_TOKEN,
    )
    assert command[:3] == [sys.executable, str(ROOT / "scripts" / "dcu_preflight.py"), "one-step-child"]
    assert "--physical-device" in command
    assert command[command.index("--physical-device") + 1] == "1"
    assert "--internal-token" in command
    assert command[command.index("--internal-token") + 1] == ONE_STEP_CHILD_TOKEN


def test_cpu_child_environment_separates_compute_and_egl_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    builder = getattr(dcu_preflight, "build_cpu_child_environment", None)
    assert callable(builder), "concurrency must expose a CPU-child environment builder"
    environment = builder(_config())

    assert environment.get("CUDA_VISIBLE_DEVICES") is None
    assert environment.get("HIP_VISIBLE_DEVICES") is None
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "8"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"


def test_concurrency_keeps_nested_worker_compute_mapping_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    cpu_builder = getattr(dcu_preflight, "build_cpu_child_environment", None)
    assert callable(cpu_builder), "concurrency must expose a CPU-child environment builder"
    cpu_environment = cpu_builder(_config())
    assert cpu_environment.get("CUDA_VISIBLE_DEVICES") is None
    assert cpu_environment.get("HIP_VISIBLE_DEVICES") is None

    for physical_device in (0, 1):
        nested_environment = dcu_preflight._worker_environment(_config(), physical_device)
        assert nested_environment["HIP_VISIBLE_DEVICES"] == str(physical_device)
        assert nested_environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_concurrency_parent_does_not_pass_compute_visibility_to_cpu_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import dcu_preflight

    captured_environments: list[dict[str, str]] = []

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.returncode = 0
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.pid = 3000 + self.physical_device

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "run_manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
            (child_dir / "child_result.json").write_text(
                json.dumps(_complete_concurrency_child_result(self.physical_device)) + "\n",
                encoding="utf-8",
            )
            return "", ""

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def terminate(self) -> None:
            self.returncode = -15

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured_environments.append(dict(kwargs["env"]))
        return FakeProcess(command)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setattr("scripts.dcu_preflight.subprocess.Popen", fake_popen)

    result = dcu_preflight._run_concurrency_impl(
        config=_config(),
        phase="concurrency",
        run_directory=tmp_path / "run",
        expected_project_sha="sha",
        actual_project_sha="sha",
        config_path=CONFIG_PATH,
    )

    assert result["status"] == "PASS"
    assert len(captured_environments) == 2
    assert all(environment.get("CUDA_VISIBLE_DEVICES") is None for environment in captured_environments)
    assert all(environment.get("HIP_VISIBLE_DEVICES") is None for environment in captured_environments)
    assert all(environment["MUJOCO_EGL_DEVICE_ID"] == "8" for environment in captured_environments)


def test_concurrency_marks_killed_sibling_manifest_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import dcu_preflight

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.returncode: int | None = None
            self.pid = 3100 + self.physical_device

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            if self.physical_device == 0:
                self.returncode = 1
                (child_dir / "run_manifest.json").write_text(
                    '{"status":"FAIL"}\n', encoding="utf-8"
                )
            else:
                self.returncode = 0
            return "", ""

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, *, timeout: float) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

    monkeypatch.setattr(
        "scripts.dcu_preflight.subprocess.Popen",
        lambda command, **_: FakeProcess(command),
    )

    with pytest.raises(DCUPreflightError, match="concurrency child 0 returned 1"):
        dcu_preflight._run_concurrency_impl(
            config=_config(),
            phase="concurrency",
            run_directory=tmp_path / "run",
            expected_project_sha="sha",
            actual_project_sha="sha",
            config_path=CONFIG_PATH,
        )

    children_root = tmp_path / "run" / "children"
    child0_manifest = json.loads((children_root / "child0" / "run_manifest.json").read_text())
    child1_manifest_path = children_root / "child1" / "run_manifest.json"
    assert child0_manifest["status"] == "FAIL"
    assert child1_manifest_path.is_file()
    child1_manifest = json.loads(child1_manifest_path.read_text())
    assert child1_manifest["status"] == "TERMINATED"
    assert child1_manifest["returncode"] == -9


@pytest.mark.parametrize(
    "missing_field",
    [
        "model_load_success",
        "egl",
        "gl",
        "action_evidence",
        "network",
        "device",
        "worker",
        "latency",
        "memory",
    ],
)
def test_concurrency_parent_rejects_incomplete_child_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    from scripts import dcu_preflight

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.returncode = 0
            self.pid = 2000 + self.physical_device

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            result = _complete_concurrency_child_result(self.physical_device)
            if self.physical_device == 0:
                result.pop(missing_field)
            (child_dir / "run_manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
            (child_dir / "child_result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
            return "", ""

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def terminate(self) -> None:
            self.returncode = -15

    monkeypatch.setattr(
        "scripts.dcu_preflight.subprocess.Popen",
        lambda command, **_: FakeProcess(command),
    )

    with pytest.raises(DCUPreflightError, match="fixed boundary|evidence"):
        dcu_preflight._run_concurrency_impl(
            config=_config(),
            phase="concurrency",
            run_directory=tmp_path / "run",
            expected_project_sha="sha",
            actual_project_sha="sha",
            config_path=CONFIG_PATH,
        )


def test_concurrency_parent_requires_nested_mapping_and_finite_action_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.returncode = 0
            self.pid = 3300 + self.physical_device

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            result = _complete_concurrency_child_result(self.physical_device)
            if self.physical_device == 0:
                result["device"] = dict(result["device"])
                result["device"]["nested_worker_cuda_visible_devices"] = "1"
                result["action_evidence"] = [
                    {"decision_index": 0, "shape": [1, 7], "dtype": "float32", "finite": True},
                    {"decision_index": 1, "shape": [1, 7], "dtype": "float32", "finite": False},
                ]
            (child_dir / "run_manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
            (child_dir / "child_result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
            return "", ""

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def terminate(self) -> None:
            self.returncode = -15

    monkeypatch.setattr(
        "scripts.dcu_preflight.subprocess.Popen",
        lambda command, **_: FakeProcess(command),
    )
    with pytest.raises(DCUPreflightError, match="evidence|fixed boundary"):
        dcu_preflight._run_concurrency_impl(
            config=_config(),
            phase="concurrency",
            run_directory=tmp_path / "run",
            expected_project_sha="sha",
            actual_project_sha="sha",
            config_path=CONFIG_PATH,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency", None),
        (
            "latency",
            {
                "inference_seconds": [0.1],
                "env_step_seconds": [0.1, 0.1],
                "worker_wall_seconds": [0.1, 0.1],
                "render_seconds": [0.1],
            },
        ),
        (
            "latency",
            {
                "inference_seconds": [0.1, float("nan")],
                "env_step_seconds": [0.1, 0.1],
                "worker_wall_seconds": [0.1, 0.1],
                "render_seconds": [0.1],
            },
        ),
        ("memory", None),
        ("memory", {"dcu_peak_memory_bytes": -1, "worker_response": {}}),
        ("memory", {"dcu_peak_memory_bytes": 1.5, "worker_response": {}}),
        ("memory", {"dcu_peak_memory_bytes": 1, "worker_response": None}),
    ],
)
def test_concurrency_child_rejects_malformed_latency_or_memory_evidence(
    field: str,
    value: object,
) -> None:
    from scripts import dcu_preflight

    result = _complete_concurrency_child_result(0)
    result[field] = value
    with pytest.raises(dcu_preflight.DCUPreflightError, match="latency|memory"):
        dcu_preflight._validate_concurrency_child_result(
            result,
            expected_physical_device=0,
            expected_steps=2,
        )


def test_concurrency_collection_timeout_is_bounded_and_closes_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    class Pipe:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    communication_timeouts: list[float | None] = []
    killpg_calls: list[tuple[int, int]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdin = Pipe()
            self.stdout = Pipe()
            self.stderr = Pipe()

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            assert timeout == pytest.approx(1.0)
            self.returncode = -15
            return self.returncode

        def communicate(self, *, timeout: float | None = None) -> tuple[str, str]:
            communication_timeouts.append(timeout)
            if timeout is None:
                raise AssertionError("communicate must always have a timeout")
            raise subprocess.TimeoutExpired(["fake-child"], timeout)

    process = FakeProcess()

    def fake_killpg(pgid: int, signal_number: int) -> None:
        killpg_calls.append((pgid, signal_number))

    monkeypatch.setattr("scripts.dcu_preflight.os.killpg", fake_killpg)
    metadata = {"process": process, "pgid": 4321}
    stdout, stderr = dcu_preflight._collect_concurrency_process(metadata, timeout=0.25)

    assert (stdout, stderr) == ("", "")
    assert communication_timeouts == [0.25, 1.0]
    assert killpg_calls == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]
    assert metadata["collection_timeout"] is True
    assert process.stdin.closed and process.stdout.closed and process.stderr.closed


def test_concurrency_invalid_pid_reaps_started_child_before_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    calls: list[tuple[str, float | None]] = []

    class InvalidPidProcess:
        pid = 0
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            calls.append(("terminate", None))

        def kill(self) -> None:
            calls.append(("kill", None))
            self.returncode = -9

        def wait(self, *, timeout: float) -> int:
            calls.append(("wait", timeout))
            self.returncode = -15
            return self.returncode

    process = InvalidPidProcess()
    monkeypatch.setattr(
        "scripts.dcu_preflight.subprocess.Popen",
        lambda command, **kwargs: process,
    )

    with pytest.raises(dcu_preflight.DCUPreflightError, match="valid pid"):
        dcu_preflight._run_concurrency_impl(
            config=_config(),
            phase="concurrency",
            run_directory=tmp_path / "run",
            expected_project_sha="sha",
            actual_project_sha="sha",
            config_path=CONFIG_PATH,
        )

    assert calls == [("terminate", None), ("wait", 1.0)]


def test_concurrency_process_group_cleanup_and_start_new_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    popen_kwargs: list[dict[str, object]] = []
    killpg_calls: list[tuple[int, int]] = []
    processes: dict[int, object] = {}

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.pid = 1000 + self.physical_device
            self.returncode: int | None = None
            processes[self.pid] = self

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            if self.physical_device == 0:
                self.returncode = 1
                (child_dir / "run_manifest.json").write_text('{"status":"FAIL"}\n', encoding="utf-8")
            return "", ""

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            del timeout
            return int(self.returncode or 0)

        def kill(self) -> None:
            self.returncode = -9

        def terminate(self) -> None:
            self.returncode = -15

    def fake_killpg(pgid: int, signal_number: int) -> None:
        killpg_calls.append((pgid, signal_number))
        processes[pgid].returncode = -15

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        popen_kwargs.append(dict(kwargs))
        return FakeProcess(command)

    monkeypatch.setattr("scripts.dcu_preflight.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.dcu_preflight.os.killpg", fake_killpg)

    with pytest.raises(DCUPreflightError, match="child 0 returned 1"):
        dcu_preflight._run_concurrency_impl(
            config=_config(),
            phase="concurrency",
            run_directory=tmp_path / "run",
            expected_project_sha="sha",
            actual_project_sha="sha",
            config_path=CONFIG_PATH,
        )

    assert len(popen_kwargs) == 2
    assert all(kwargs["start_new_session"] is True for kwargs in popen_kwargs)
    assert killpg_calls == [
        (1001, signal.SIGTERM),
        (1001, signal.SIGKILL),
        (1000, signal.SIGTERM),
        (1000, signal.SIGKILL),
    ]
    child1_manifest = tmp_path / "run" / "children" / "child1" / "run_manifest.json"
    assert json.loads(child1_manifest.read_text(encoding="utf-8"))["status"] == "TERMINATED"


def test_concurrency_cleans_group_after_outer_child_exits_with_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    killpg_calls: list[tuple[int, int]] = []

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.pid = 5000 + self.physical_device
            # The outer process is already gone, but its nested descendant is
            # intentionally represented as still attached to this PGID.
            self.returncode = 1 if self.physical_device == 0 else 0

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            if self.physical_device == 0:
                (child_dir / "run_manifest.json").write_text('{"status":"RUNNING"}\n', encoding="utf-8")
            else:
                (child_dir / "run_manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
                (child_dir / "child_result.json").write_text(
                    json.dumps(_complete_concurrency_child_result(self.physical_device)) + "\n",
                    encoding="utf-8",
                )
            return "", ""

        def poll(self) -> int:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            del timeout
            return self.returncode

    def fake_popen(command: list[str], **_: object) -> FakeProcess:
        return FakeProcess(command)

    monkeypatch.setattr("scripts.dcu_preflight.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "scripts.dcu_preflight.os.killpg",
        lambda pgid, signal_number: killpg_calls.append((pgid, signal_number)),
    )

    with pytest.raises(DCUPreflightError, match="child 0 returned 1"):
        dcu_preflight._run_concurrency_impl(
            config=_config(),
            phase="concurrency",
            run_directory=tmp_path / "run",
            expected_project_sha="sha",
            actual_project_sha="sha",
            config_path=CONFIG_PATH,
        )

    assert killpg_calls == [(5000, signal.SIGTERM), (5000, signal.SIGKILL)]
    manifest = json.loads(
        (tmp_path / "run" / "children" / "child0" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "FAIL"
    assert manifest["terminal"] is True


def test_concurrency_throughput_reports_four_work_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import dcu_preflight

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.returncode = 0
            self.pid = 3500 + self.physical_device

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "run_manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
            (child_dir / "child_result.json").write_text(
                json.dumps(_complete_concurrency_child_result(self.physical_device)) + "\n",
                encoding="utf-8",
            )
            return "", ""

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def terminate(self) -> None:
            self.returncode = -15

    monkeypatch.setattr(
        "scripts.dcu_preflight.subprocess.Popen",
        lambda command, **_: FakeProcess(command),
    )
    result = dcu_preflight._run_concurrency_impl(
        config=_config(),
        phase="concurrency",
        run_directory=tmp_path / "run",
        expected_project_sha="sha",
        actual_project_sha="sha",
        config_path=CONFIG_PATH,
    )
    assert result["wall_throughput"]["total_env_steps"] == 4
    assert result["wall_throughput"]["total_policy_decisions"] == 4
    assert result["wall_throughput"]["env_steps_per_second"] == pytest.approx(
        4.0 / result["wall_throughput"]["wall_seconds"]
    )
    assert result["wall_throughput"]["policy_decisions_per_second"] == pytest.approx(
        4.0 / result["wall_throughput"]["wall_seconds"]
    )


def test_internal_one_step_child_never_accepts_a_public_call(tmp_path: Path) -> None:
    with pytest.raises(DCUPreflightError, match="private internal token"):
        run_phase(
            "one-step-child",
            config=_config(),
            expected_project_sha="sha",
            actual_project_sha="sha",
            output_root=tmp_path,
        )


def test_run_phase_failure_manifest_is_atomic_and_no_overwrite(tmp_path: Path) -> None:
    def failing_runner(**_: object) -> int:
        raise RuntimeError("fake phase failed")

    assert run_phase(
        "compare",
        config=_config(),
        expected_project_sha="sha",
        actual_project_sha="sha",
        output_root=tmp_path,
        preflight_gate=lambda **_: {"gate": "fake-pass"},
        phase_runner=failing_runner,
    ) == 1
    run_dir = next(tmp_path.iterdir())
    failure = run_dir / "failure_manifest.json"
    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == "FAIL"
    with pytest.raises(FileExistsError):
        run_phase(
            "compare",
            config=_config(),
            expected_project_sha="sha",
            actual_project_sha="sha",
            output_root=tmp_path,
            preflight_gate=lambda **_: {"gate": "fake-pass"},
            phase_runner=failing_runner,
            run_directory=run_dir,
        )


def test_fake_concurrency_launches_both_children_before_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, int]] = []

    class FakeProcess:
        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.returncode = 0
            self.physical_device = int(command[command.index("--physical-device") + 1])
            self.pid = 3400 + self.physical_device

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            events.append(("communicate", int(self.command[self.command.index("--physical-device") + 1])))
            child_dir = Path(self.command[self.command.index("--run-directory") + 1])
            child_dir.mkdir(parents=True, exist_ok=True)
            (child_dir / "run_manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
            (child_dir / "child_result.json").write_text(
                json.dumps(_complete_concurrency_child_result(self.physical_device)) + "\n",
                encoding="utf-8",
            )
            return "child stdout", "child stderr"

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def terminate(self) -> None:
            self.returncode = -15

    def fake_popen(command: list[str], **_: object) -> FakeProcess:
        physical = int(command[command.index("--physical-device") + 1])
        events.append(("popen", physical))
        return FakeProcess(command)

    monkeypatch.setattr("scripts.dcu_preflight.subprocess.Popen", fake_popen)
    monkeypatch.setattr("scripts.dcu_preflight._worker_environment", lambda *_: {})
    from scripts import dcu_preflight

    result = dcu_preflight._run_concurrency_impl(
        config=_config(),
        phase="concurrency",
        run_directory=tmp_path / "run",
        expected_project_sha="sha",
        actual_project_sha="sha",
        config_path=CONFIG_PATH,
    )
    assert result["status"] == "PASS"
    assert events[:2] == [("popen", 0), ("popen", 1)]
    assert [event[1] for event in events[2:]] == [0, 1]


def test_fake_one_step_child_uses_one_reset_decision_and_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import dcu_preflight

    class FakeEnv:
        num_envs = 1
        reset_count = 0
        step_count = 0

        def call(self, name: str) -> list[object]:
            if name == "task_description":
                return ["fake task"]
            raise KeyError(name)

        def reset(self, *, seed: list[int], options: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
            del options
            self.reset_count += 1
            assert seed == [2027]
            return ({"fake": True}, {} )

        def step(self, action: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
            self.step_count += 1
            assert action.shape == (1, 7)
            done = np.asarray([self.step_count >= 2], dtype=bool)
            return ({"fake": True}, np.zeros(1), done, np.zeros(1, dtype=bool), {"is_success": done})

    env = FakeEnv()
    feature_values = {
        "observation.state": torch.zeros((1, 8), dtype=torch.float32),
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.images.image2": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.language.tokens": torch.zeros((1, 4), dtype=torch.int64),
        "observation.language.attention_mask": torch.ones((1, 4), dtype=torch.bool),
    }

    class FakeClient:
        reset_count = 0
        select_count = 0
        last_queue_evidence = {
            "queue_length_before": 0,
            "queue_length_after": 0,
            "new_chunk_generated": True,
        }

        def reset(self, *, seed: int) -> dict[str, object]:
            self.reset_count += 1
            assert seed == 2027
            return {"queue_length_before": 0, "queue_length_after": 0}

        def select_action(self, path: Path) -> torch.Tensor:
            self.select_count += 1
            assert path.suffix == ".safetensors"
            return torch.zeros((1, 7), dtype=torch.float32)

        def close(self) -> None:
            return None

    client = FakeClient()
    runtime = {
        "env": env,
        "envs": {"fake": {0: env}},
        "task": {
            "task_name": "fake-task",
            "task_description": "fake task",
            "bddl_path": "/pinned/bddl/fake.bddl",
            "bddl_file": "fake.bddl",
            "init_state_path": "/pinned/init/fake.xml",
            "init_state_file": "fake.xml",
            "init_state_id_evidence": {"value": 0, "expected": 0},
            "horizon": 280,
        },
        "env_preprocessor": lambda value: value,
        "preprocessor": lambda value: feature_values,
        "postprocessor": lambda value: value,
        "env_postprocessor": lambda value: value,
        "preprocess_observation": lambda value: feature_values,
        "close_envs": lambda value: None,
    }
    monkeypatch.setattr("scripts.dcu_preflight.build_cpu_runtime", lambda *_, **__: runtime)
    monkeypatch.setattr(
        "scripts.dcu_preflight._start_worker",
        lambda *_, **__: {
            "client": client,
            "transport": type("Transport", (), {"responses": [], "stderr_text": ""})(),
            "ping": {
                "ok": True,
                "device": "cuda:0",
                "device_name": "fake-k100",
                "model": {"policy_type": "FakeSmolVLA"},
            },
            "argv": [],
            "physical_device": 0,
            "logical_device": "cuda:0",
            "compute_environment": {
                "physical_k100": 0,
                "HIP_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICES": "0",
                "torch_logical_device": "cuda:0",
            },
            "runtime": _config()["runtime"],
        },
    )
    def fake_render_callback(trace: object) -> object:
        def render(_: object) -> None:
            trace.render_count += 1
            trace.gl = {
                "vendor": "Mesa/X.org",
                "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
                "version": "3.1 Mesa 21.1.5",
            }

        return render

    monkeypatch.setattr("scripts.m0_smoke.make_render_callback", fake_render_callback)
    monkeypatch.setattr(
        "scripts.dcu_preflight._egl_probe_evidence",
        lambda **_: {
            "eglQueryDevicesEXT_device_count": 9,
            "selected_MUJOCO_EGL_DEVICE_ID": "8",
        },
    )
    result = dcu_preflight._run_one_step_child_impl(
        config=_config(),
        phase="one-step-child",
        run_directory=tmp_path / "child",
        config_path=CONFIG_PATH,
        physical_device=0,
    )
    assert result["child_result"]["reset_count"] == 1
    assert result["child_result"]["decision_count"] == 2
    assert result["child_result"]["env_step_count"] == 2
    assert result["child_result"]["task"]["init_state_path"].endswith("fake.xml")
    assert result["child_result"]["task"]["init_state_file"] == "fake.xml"
    assert result["child_result"]["task"]["init_state_id_evidence"]["value"] == 0
    assert "init_state" not in result["child_result"]["task"]
    assert env.reset_count == 1
    assert env.step_count == 2
    assert client.reset_count == 1
    assert client.select_count == 2
