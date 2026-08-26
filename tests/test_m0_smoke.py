"""Focused fake-only tests for the M0 smoke instrumentation seams."""

from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.m0_smoke import (
    EXPECTED_PROJECT_SHA,
    PolicyProxy,
    ProcessorProxy,
    TraceStore,
    VectorEnvProxy,
    _assert_project_head,
    _assert_action_space,
    _assert_renderer_environment,
    _assert_rollout_acceptance,
    _ensure_asset_symlink,
    _git_evidence,
    _load_yaml,
    make_render_callback,
    _manifest_base,
    _parse_runtime_lock,
    _validate_runtime_freeze,
    _verify_runtime_module_paths,
)


class FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, n_action_steps=1)
        self._queues = {"action": deque(maxlen=1)}
        self.reset_calls = 0
        self.select_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self._queues["action"].clear()

    def select_action(self, observation):
        self.select_calls += 1
        action = torch.tensor([[0.25, 0.0, -0.25, 0.1, 0.2, 0.3, -0.4]], dtype=torch.float32)
        self._queues["action"].append(action)
        return self._queues["action"].popleft()


class FakeVectorEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.actions = []

    def step(self, action):
        self.actions.append(action.copy())
        return (
            {"next": np.zeros((1, 1), dtype=np.float32)},
            np.array([1.0], dtype=np.float32),
            np.array([True]),
            np.array([False]),
            {"is_success": np.array([True])},
        )

    def call(self, name, *args, **kwargs):
        assert name == "render"
        return [np.zeros((360, 360, 3), dtype=np.uint8)]


class FakeSpaceEnv:
    num_envs = 1
    action_space = SimpleNamespace(shape=(1, 7), dtype=np.dtype(np.float32))
    single_action_space = SimpleNamespace(shape=(7,), dtype=np.dtype(np.float32))


def test_proxies_preserve_values_and_record_policy_queue_semantics() -> None:
    trace = TraceStore(chunk_size=50, n_action_steps=1)
    original = {"observation.state": torch.ones((1, 8), dtype=torch.float32)}

    env_processor = ProcessorProxy(lambda value: value, "env_processor", trace)
    policy_processor = ProcessorProxy(lambda value: value, "policy_processor", trace)
    postprocessor = ProcessorProxy(lambda value: value, "postprocessor", trace)
    env_postprocessor = ProcessorProxy(lambda value: value, "env_postprocessor", trace)
    policy = PolicyProxy(FakePolicy(), trace)

    env_value = env_processor(original)
    policy_value = policy_processor(env_value)
    normalized_action = policy.select_action(policy_value)
    postprocessed_action = postprocessor(normalized_action)
    env_action = env_postprocessor({"action": postprocessed_action})

    assert env_value is original
    assert policy_value is original
    assert env_action is not None
    assert trace.decisions[0]["observation_env_processor"]["observation.state"]["shape"] == [1, 8]
    assert trace.decisions[0]["observation_policy_processor"]["observation.state"]["dtype"] == "torch.float32"
    assert trace.decisions[0]["normalized_action"]["shape"] == [1, 7]
    assert trace.decisions[0]["postprocessed_action"]["shape"] == [1, 7]
    assert trace.decisions[0]["queue_length_before"] == 0
    assert trace.decisions[0]["queue_length_after"] == 0
    assert trace.decisions[0]["new_chunk_generated"] is True
    assert trace.decisions[0]["normalized_action"]["finite"] is True


def test_vector_proxy_records_exact_action_equality_and_outcome() -> None:
    trace = TraceStore(chunk_size=50, n_action_steps=1)
    trace.begin_decision()
    action = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], dtype=np.float32)
    trace.record("postprocessed_action", action)
    env = FakeVectorEnv()
    proxy = VectorEnvProxy(env, trace)

    result = proxy.step(action)

    assert result[1].tolist() == [1.0]
    decision = trace.decisions[0]
    assert decision["env_step_action"]["shape"] == [1, 7]
    assert decision["env_step_action"]["dtype"] == "float32"
    assert decision["env_step_action"]["finite"] is True
    assert decision["env_step_equals_postprocessed"] is True
    assert decision["reward"] == [1.0]
    assert decision["terminated"] == [True]
    assert decision["truncated"] == [False]
    assert decision["done"] == [True]
    assert decision["success"] == [True]


def test_vector_proxy_rejects_integer_actions() -> None:
    trace = TraceStore(chunk_size=50, n_action_steps=1)
    trace.begin_decision()
    action = np.zeros((1, 7), dtype=np.float32)
    trace.record("postprocessed_action", action)
    with pytest.raises(TypeError, match="float32"):
        VectorEnvProxy(FakeVectorEnv(), trace).step(action.astype(np.int32))


def test_asset_symlink_rejects_wrong_target_and_rebuilds_exact_target(tmp_path) -> None:
    target = tmp_path / "approved-assets"
    wrong = tmp_path / "wrong-assets"
    target.mkdir()
    wrong.mkdir()
    link = tmp_path / "site-packages" / "libero" / "libero" / "assets"
    link.parent.mkdir(parents=True)
    link.symlink_to(wrong, target_is_directory=True)

    with pytest.raises(RuntimeError, match="outside the exact tree"):
        _ensure_asset_symlink(target, link)
    assert link.resolve() == wrong.resolve()

    link.unlink()
    link.symlink_to(target, target_is_directory=True)
    result = _ensure_asset_symlink(target, link)
    assert result["is_symlink"] is True
    assert link.resolve() == target.resolve()


def test_project_head_gate_is_fail_closed() -> None:
    _assert_project_head(EXPECTED_PROJECT_SHA)
    with pytest.raises(RuntimeError, match="project HEAD"):
        _assert_project_head("different")


def test_action_space_gate_requires_batched_float32_and_single_7d() -> None:
    evidence = _assert_action_space(FakeSpaceEnv(), 7)
    assert evidence["batch_action_shape"] == [1, 7]
    assert evidence["single_action_shape"] == [7]
    bad = FakeSpaceEnv()
    bad.action_space = SimpleNamespace(shape=(7,), dtype=np.dtype(np.float32))
    with pytest.raises(RuntimeError, match="vector action_space"):
        _assert_action_space(bad, 7)


def test_rollout_acceptance_gate_checks_counts_actions_queue_and_render_count() -> None:
    trace = TraceStore(chunk_size=50, n_action_steps=1)
    for index in range(2):
        trace.begin_decision()
        action = np.zeros((1, 7), dtype=np.float32)
        trace.record("env_step_action", action)
        trace.update(
            queue_length_before=0,
            queue_length_after=0,
            new_chunk_generated=True,
            env_step_equals_postprocessed=True,
        )
        trace.env_step_latencies.append(0.001)
    policy = SimpleNamespace(call_count=2)
    rollout_data = {
        "action": torch.zeros((1, 2, 7)),
        "done": torch.tensor([[False, True]]),
    }

    _assert_rollout_acceptance(trace, rollout_data, policy, render_count=3)


def test_runtime_module_path_gate_rejects_external_source_tree(tmp_path) -> None:
    purelib = tmp_path / "site-packages"
    purelib.mkdir()
    modules = {
        name: SimpleNamespace(
            __file__=str(purelib / name / "__init__.py"),
            __spec__=SimpleNamespace(origin=str(purelib / name / "__init__.py")),
        )
        for name in ("lerobot", "libero", "robosuite", "mujoco")
    }
    evidence = _verify_runtime_module_paths(tmp_path, modules, purelib=purelib)
    assert set(evidence["modules"]) == {"lerobot", "libero", "robosuite", "mujoco"}

    modules["libero"].__file__ = str(tmp_path / "external" / "libero" / "libero" / "__init__.py")
    with pytest.raises(RuntimeError, match="isolated site-packages"):
        _verify_runtime_module_paths(tmp_path, modules, purelib=purelib)


def test_runtime_lock_parser_and_raw_freeze_validation_are_fail_closed(tmp_path) -> None:
    freeze_bytes = b"demo==1.0\nlocal_pkg @ file:///frozen/local_pkg\n"
    freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
    lock_bytes = (
        "# fake runtime lock\n"
        "python_executable: /venv/bin/python\n"
        "pip_freeze_all_lines: 2\n"
        f"pip_freeze_all_sha256: {freeze_sha256}\n\n"
        "[packages]\n"
        "pip_check:\n"
        "No broken requirements found.\n\n"
        "[pip_freeze_all]\n"
    ).encode() + freeze_bytes
    lock_path = tmp_path / "runtime.lock"
    lock_path.write_bytes(lock_bytes)

    parsed = _parse_runtime_lock(lock_path)
    assert parsed["python_executable"] == "/venv/bin/python"
    assert parsed["pip_freeze_all_lines"] == 2
    assert parsed["pip_freeze_all_sha256"] == freeze_sha256
    assert parsed["pip_freeze_all_bytes"] == freeze_bytes
    assert parsed["pip_check_lines"] == ["No broken requirements found."]

    evidence = _validate_runtime_freeze(freeze_bytes, 2, freeze_sha256, freeze_bytes)
    assert evidence["actual_lines"] == 2
    assert evidence["actual_sha256"] == freeze_sha256
    assert evidence["exact_bytes_match"] is True
    with pytest.raises(RuntimeError, match="pip freeze --all differs"):
        _validate_runtime_freeze(freeze_bytes + b"\n", 2, freeze_sha256, freeze_bytes)

    bad_lock = tmp_path / "bad-runtime.lock"
    bad_lock.write_bytes(lock_bytes.replace(freeze_sha256.encode(), b"0" * 64))
    with pytest.raises(RuntimeError, match="declared line count/hash"):
        _parse_runtime_lock(bad_lock)


def test_manifest_base_initialization_records_runtime_lock_without_undefined_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "m0" / "smoke.yaml"
    config = _load_yaml(config_path)
    run_dir = root / "runs" / "m0_smoke" / "manifest-base-seam"

    manifest = _manifest_base(root, config_path, run_dir, config)

    assert manifest["runtime_lock"]["path"] == str(Path(config["runtime_lock"]).resolve())
    assert len(manifest["runtime_lock"]["lock_sha256"]) == 64


def test_git_evidence_preserves_porcelain_status_columns_for_modified_paths(tmp_path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    tracked = tmp_path / "AGENTS.md"
    tracked.write_text("initial\n")
    subprocess.run(["git", "add", "AGENTS.md"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Smoke Test",
            "-c",
            "user.email=smoke@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    tracked.write_text("modified\n")

    evidence = _git_evidence(tmp_path)

    assert evidence["status"]["stdout"] == " M AGENTS.md"
    assert evidence["dirty_files"] == [
        {
            "path": "AGENTS.md",
            "status": " M",
            "sha256": hashlib.sha256(b"modified\n").hexdigest(),
            "bytes": len(b"modified\n"),
        }
    ]


def test_renderer_environment_gate_requires_exact_egl_settings(monkeypatch) -> None:
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    evidence = _assert_renderer_environment()
    assert evidence["exact"] is True

    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    with pytest.raises(RuntimeError, match="MUJOCO_GL"):
        _assert_renderer_environment()


def test_render_callback_rejects_missing_gl_identity_before_step(monkeypatch) -> None:
    class FakeRenderEnv:
        def __init__(self) -> None:
            self.render_calls = 0

        def call(self, name):
            assert name == "render"
            self.render_calls += 1
            return [np.zeros((360, 360, 3), dtype=np.uint8)]

    monkeypatch.setattr(
        "scripts.m0_smoke._gl_evidence",
        lambda: {"vendor": "Mesa/X.org", "renderer": None, "version": "3.1"},
    )
    trace = TraceStore(chunk_size=50, n_action_steps=1)
    env = FakeRenderEnv()

    with pytest.raises(RuntimeError, match="GL identity"):
        make_render_callback(trace)(env)

    assert env.render_calls == 1
    assert trace.render_count == 0
    assert trace.render_latencies == []
