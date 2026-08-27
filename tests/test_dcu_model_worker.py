"""Fake-only contracts for the isolated DCU SmolVLA model worker."""

from __future__ import annotations

from collections import deque
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.dcu_model_worker import (
    FEATURE_SCHEMA,
    NOISE_SCHEMA,
    PREDICT_REQUEST_SCHEMA,
    SELECT_REQUEST_SCHEMA,
    DCUModelWorker,
    JSONLWorkerServer,
    _enforce_eager_attention,
    build_official_policy,
    load_worker_config,
    validate_device,
    validate_offline_environment,
    validate_worker_config,
)
from scripts.dcu_worker import DCUPreflightError, load_tensor_bundle, save_tensor_bundle


def _feature_bundle() -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.zeros((1, 8), dtype=torch.float32),
        "observation.images.image": torch.zeros((1, 3, 360, 360), dtype=torch.float32),
        "observation.images.image2": torch.ones((1, 3, 360, 360), dtype=torch.float32),
        "observation.language.tokens": torch.ones((1, 4), dtype=torch.int64),
        "observation.language.attention_mask": torch.ones((1, 4), dtype=torch.bool),
    }


def _predict_bundle() -> dict[str, torch.Tensor]:
    return {**_feature_bundle(), "noise": torch.full((1, 50, 32), 0.5, dtype=torch.float32)}


class FakeDevice:
    def __init__(self, name: str = "K100_AI") -> None:
        self.name = name
        self.moves: list[tuple[str, str]] = []
        self.peak_resets = 0
        self.syncs = 0
        self.max_memory_calls = 0
        self.basic_tensor_ops_calls = 0
        self.seed_calls: list[int] = []

    def is_available(self, device: str) -> bool:
        return device == "cuda:0"

    def device_count(self) -> int:
        return 1

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return self.name

    def torch_hip_version(self) -> str:
        return "fake-hip"

    def torch_cuda_version(self) -> str | None:
        return None

    def to_device(self, tensor: torch.Tensor, device: str) -> torch.Tensor:
        self.moves.append((str(tensor.dtype), device))
        return tensor

    def synchronize(self) -> None:
        self.syncs += 1

    def reset_peak_memory_stats(self) -> None:
        self.peak_resets += 1

    def max_memory_allocated(self) -> int:
        self.max_memory_calls += 1
        return 1234

    def manual_seed_all(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def basic_tensor_ops(self) -> dict[str, dict[str, object]]:
        self.basic_tensor_ops_calls += 1
        return {
            "fp32": {
                "shape": [2, 2],
                "dtype": "torch.float32",
                "finite": True,
                "device": "cuda:0",
            },
            "bf16": {
                "shape": [2, 2],
                "dtype": "torch.bfloat16",
                "finite": True,
                "device": "cuda:0",
            },
        }


class FakePolicy(torch.nn.Module):
    def __init__(self, n_action_steps: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=50, n_action_steps=n_action_steps)
        self._queues = {"action": deque(maxlen=n_action_steps)}
        self.reset_calls = 0
        self.predict_calls: list[tuple[dict[str, torch.Tensor], torch.Tensor | None]] = []
        self.select_calls: list[tuple[dict[str, torch.Tensor], dict[str, object]]] = []
        self.eval_calls = 0
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def eval(self) -> "FakePolicy":
        self.eval_calls += 1
        super().eval()
        return self

    def reset(self) -> None:
        self.reset_calls += 1
        self._queues["action"].clear()

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.predict_calls.append((batch, noise))
        assert noise is not None
        return noise[:, :, :7].clone()

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs: object) -> torch.Tensor:
        self.select_calls.append((batch, kwargs))
        if not self._queues["action"]:
            chunk = torch.arange(350, dtype=torch.float32).reshape(1, 50, 7)
            self._queues["action"].extend(chunk.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues["action"].popleft()


class FakeLoadedPolicy(FakePolicy):
    """A fake factory result whose parameter metadata models cuda:0 placement."""

    def parameters(self):  # type: ignore[no-untyped-def]
        return iter(
            (
                SimpleNamespace(numel=lambda: 3, dtype=torch.float32, device="cuda:0"),
                SimpleNamespace(numel=lambda: 2, dtype=torch.bfloat16, device="cuda:0"),
            )
        )


class FakeAttentionConfig:
    def __init__(self, implementation: str = "legacy") -> None:
        self._attn_implementation = implementation
        self._flash_attn_2_enabled = True


class LockedAttentionConfig(FakeAttentionConfig):
    def __init__(self, implementation: str = "legacy") -> None:
        self._implementation = implementation
        self._flash_attn_2_enabled = True

    @property
    def _attn_implementation(self) -> str:
        return self._implementation

    @_attn_implementation.setter
    def _attn_implementation(self, value: str) -> None:
        if value == "eager":
            raise RuntimeError("eager attention is locked")
        self._implementation = value


class FakeAttentionPolicy(FakeLoadedPolicy):
    def __init__(self, config_cls: type[FakeAttentionConfig] = FakeAttentionConfig) -> None:
        super().__init__()
        vlm_config = config_cls()
        vlm_config.text_config = config_cls()
        vlm_config.vision_config = config_cls()
        expert_config = config_cls()
        self.model = SimpleNamespace(
            vlm_with_expert=SimpleNamespace(
                config=vlm_config,
                vlm=SimpleNamespace(config=vlm_config),
                lm_expert=SimpleNamespace(config=expert_config),
            )
        )


class DynamicForbiddenPolicy(FakePolicy):
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        self._monkeypatch = monkeypatch

    def _import_forbidden(self) -> None:
        self._monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace(__name__="flash_attn"))

    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        self._import_forbidden()
        return super().predict_action_chunk(batch, noise)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs: object) -> torch.Tensor:
        self._import_forbidden()
        return super().select_action(batch, **kwargs)


def _write_request(tmp_path: Path, name: str, bundle: dict[str, torch.Tensor], schema: object) -> Path:
    path = tmp_path / name
    save_tensor_bundle(path, bundle, schema=schema)  # type: ignore[arg-type]
    return path


def test_worker_message_flow_preserves_explicit_noise_and_native_queue(tmp_path: Path) -> None:
    device = FakeDevice()
    policy = FakePolicy(n_action_steps=2)
    worker = DCUModelWorker(policy=policy, device_adapter=device, seed=2027)
    server = JSONLWorkerServer(worker, response_dir=tmp_path)
    feature_path = _write_request(tmp_path, "features.safetensors", _feature_bundle(), FEATURE_SCHEMA)
    predict_path = _write_request(tmp_path, "predict.safetensors", _predict_bundle(), PREDICT_REQUEST_SCHEMA)

    reset = server.handle({"id": "r0", "command": "reset", "seed": 2027})
    assert reset["ok"] is True
    assert reset["seed"] == 2027
    assert device.seed_calls[-2:] == [2027, 2027]
    predicted = server.handle(
        {"id": "r1", "command": "predict_action_chunk", "request_path": str(predict_path)}
    )
    assert predicted["ok"] is True
    chunk = load_tensor_bundle(Path(predicted["response_path"]), schema={"action_chunk": ((1, 50, 7), (torch.float32,))})
    assert chunk["action_chunk"].shape == (1, 50, 7)
    assert predicted["input_dtypes"]["noise"] == "torch.float32"
    assert predicted["output_dtype"] == "torch.float32"
    assert predicted["latency_seconds"] >= 0
    assert predicted["peak_memory_bytes"] == 1234
    assert policy.predict_calls[0][1] is not None

    syncs_before_select = device.syncs
    peak_resets_before_select = device.peak_resets
    max_memory_calls_before_select = device.max_memory_calls
    first = server.handle({"id": "r2", "command": "select_action", "request_path": str(feature_path)})
    second = server.handle({"id": "r3", "command": "select_action", "request_path": str(feature_path)})
    assert first["action_shape"] == [1, 7]
    assert first["queue_length_before"] == 0
    assert first["queue_length_after"] == 1
    assert first["new_chunk_generated"] is True
    assert second["queue_length_before"] == 1
    assert second["queue_length_after"] == 0
    assert second["new_chunk_generated"] is False
    assert first["peak_memory_bytes"] == 1234
    assert second["peak_memory_bytes"] == 1234
    assert device.syncs == syncs_before_select + 4
    assert device.peak_resets == peak_resets_before_select + 2
    assert device.max_memory_calls == max_memory_calls_before_select + 2
    assert all("noise" not in kwargs for _, kwargs in policy.select_calls)


@pytest.mark.parametrize("module_name", ["mamba_ssm", "mamba"])
def test_forbidden_mamba_backend_is_rejected_before_policy_use(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    module = SimpleNamespace(__name__=module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    with pytest.raises(DCUPreflightError, match="forbidden backend"):
        DCUModelWorker(policy=FakePolicy(), device_adapter=FakeDevice(), seed=2027)


@pytest.mark.parametrize("command", ["predict_action_chunk", "select_action"])
def test_forbidden_backend_is_rechecked_after_each_action_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    worker = DCUModelWorker(
        policy=DynamicForbiddenPolicy(monkeypatch), device_adapter=FakeDevice(), seed=2027
    )
    server = JSONLWorkerServer(worker, response_dir=tmp_path)
    worker.reset()
    if command == "predict_action_chunk":
        request_path = _write_request(
            tmp_path, "predict.safetensors", _predict_bundle(), PREDICT_REQUEST_SCHEMA
        )
    else:
        request_path = _write_request(
            tmp_path, "features.safetensors", _feature_bundle(), FEATURE_SCHEMA
        )
    with pytest.raises(DCUPreflightError, match="forbidden backend"):
        server.handle({"id": command, "command": command, "request_path": str(request_path)})


def test_startup_and_ping_report_basic_eager_runtime_evidence() -> None:
    device = FakeDevice()
    worker = DCUModelWorker(policy=FakePolicy(), device_adapter=device, seed=2027)
    assert device.basic_tensor_ops_calls == 1
    ping = worker.ping()
    assert device.basic_tensor_ops_calls == 2
    expected = {
        "fp32": {
            "shape": [2, 2],
            "dtype": "torch.float32",
            "finite": True,
            "device": "cuda:0",
        },
        "bf16": {
            "shape": [2, 2],
            "dtype": "torch.bfloat16",
            "finite": True,
            "device": "cuda:0",
        },
    }
    assert ping["evidence"]["basic_tensor_ops"] == expected
    assert ping["evidence"]["startup_basic_tensor_ops"] == expected
    assert ping["basic_tensor_ops"] == expected


def test_startup_fails_closed_without_basic_tensor_ops() -> None:
    class MissingBasicTensorOps(FakeDevice):
        basic_tensor_ops = None

    with pytest.raises(DCUPreflightError, match="basic_tensor_ops"):
        DCUModelWorker(policy=FakePolicy(), device_adapter=MissingBasicTensorOps(), seed=2027)


def test_basic_runtime_evidence_must_be_finite() -> None:
    class NonFiniteBasicTensorOps(FakeDevice):
        def basic_tensor_ops(self) -> dict[str, dict[str, object]]:
            result = super().basic_tensor_ops()
            result["bf16"]["finite"] = False
            return result

    with pytest.raises(DCUPreflightError, match="finite"):
        DCUModelWorker(policy=FakePolicy(), device_adapter=NonFiniteBasicTensorOps(), seed=2027)


def test_select_action_rejects_a_bundle_that_contains_explicit_noise(tmp_path: Path) -> None:
    worker = DCUModelWorker(policy=FakePolicy(), device_adapter=FakeDevice(), seed=2027)
    server = JSONLWorkerServer(worker, response_dir=tmp_path)
    path = _write_request(tmp_path, "bad-select.safetensors", _predict_bundle(), PREDICT_REQUEST_SCHEMA)
    server.handle({"id": "reset", "command": "reset"})
    with pytest.raises(DCUPreflightError, match="feature-only"):
        server.handle({"id": "bad", "command": "select_action", "request_path": str(path)})


def test_jsonl_server_owns_response_path_and_contains_all_ipc_inputs(tmp_path: Path) -> None:
    response_dir = tmp_path / "worker"
    response_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    worker = DCUModelWorker(policy=FakePolicy(), device_adapter=FakeDevice(), seed=2027)
    server = JSONLWorkerServer(worker, response_dir=response_dir)
    inside_features = _write_request(response_dir, "features.safetensors", _feature_bundle(), FEATURE_SCHEMA)
    inside_noise = _write_request(
        response_dir, "noise.safetensors", {"noise": _predict_bundle()["noise"]}, NOISE_SCHEMA
    )
    outside_predict = _write_request(outside_dir, "predict.safetensors", _predict_bundle(), PREDICT_REQUEST_SCHEMA)
    outside_features = _write_request(outside_dir, "features.safetensors", _feature_bundle(), FEATURE_SCHEMA)
    outside_noise = _write_request(
        outside_dir, "noise.safetensors", {"noise": _predict_bundle()["noise"]}, NOISE_SCHEMA
    )
    server.handle({"id": "reset", "command": "reset"})

    with pytest.raises(DCUPreflightError, match="worker-owned"):
        server.handle(
            {
                "id": "client-output",
                "command": "predict_action_chunk",
                "request_path": str(inside_features),
                "noise_path": str(inside_noise),
                "response_path": str(response_dir / "client.safetensors"),
            }
        )
    with pytest.raises(DCUPreflightError, match="response_dir"):
        server.handle(
            {"id": "escape-request", "command": "predict_action_chunk", "request_path": str(outside_predict)}
        )
    with pytest.raises(DCUPreflightError, match="response_dir"):
        server.handle(
            {
                "id": "escape-features",
                "command": "predict_action_chunk",
                "features_path": str(outside_features),
                "noise_path": str(inside_noise),
            }
        )
    with pytest.raises(DCUPreflightError, match="response_dir"):
        server.handle(
            {
                "id": "escape-noise",
                "command": "predict_action_chunk",
                "features_path": str(inside_features),
                "noise_path": str(outside_noise),
            }
        )


def test_jsonl_server_stdout_is_only_canonical_json_and_shutdown_is_clean(tmp_path: Path) -> None:
    worker = DCUModelWorker(policy=FakePolicy(), device_adapter=FakeDevice(), seed=2027)
    server = JSONLWorkerServer(worker, response_dir=tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    server.run(
        io.StringIO(
            '{"id":"p","command":"ping"}\n'
            '{"id":"s","command":"shutdown"}\n'
        ),
        stdout,
        stderr,
    )
    lines = stdout.getvalue().splitlines(keepends=True)
    assert len(lines) == 2
    for line in lines:
        assert line.endswith("\n")
        decoded = json.loads(line)
        assert line == json.dumps(decoded, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert json.loads(lines[0])["evidence"]["device_name"] == "K100_AI"
    assert json.loads(lines[1])["shutdown"] is True
    assert worker.closed is True


def test_policy_runtime_logs_are_redirected_to_stderr(tmp_path: Path) -> None:
    class NoisyPolicy(FakePolicy):
        def predict_action_chunk(
            self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None
        ) -> torch.Tensor:
            print("fake runtime log")
            return super().predict_action_chunk(batch, noise)

    class NoisyDevice(FakeDevice):
        def synchronize(self) -> None:
            print("fake device log")
            super().synchronize()

        def reset_peak_memory_stats(self) -> None:
            print("fake peak reset log")
            super().reset_peak_memory_stats()

        def max_memory_allocated(self) -> int:
            print("fake memory query log")
            return super().max_memory_allocated()

    stdout = io.StringIO()
    stderr = io.StringIO()
    worker = DCUModelWorker(
        policy=NoisyPolicy(),
        device_adapter=NoisyDevice(),
        seed=2027,
        stderr=stderr,
    )
    server = JSONLWorkerServer(worker, response_dir=tmp_path)
    request_path = _write_request(tmp_path, "predict.safetensors", _predict_bundle(), PREDICT_REQUEST_SCHEMA)
    input_text = (
        '{"id":"reset","command":"reset"}\n'
        f'{{"id":"predict","command":"predict_action_chunk","request_path":{json.dumps(str(request_path))}}}\n'
        '{"id":"shutdown","command":"shutdown"}\n'
    )
    with redirect_stdout(stdout):
        server.run(io.StringIO(input_text), stdout, stderr)
    assert "fake runtime log" not in stdout.getvalue()
    assert "fake runtime log" in stderr.getvalue()
    assert "fake device log" not in stdout.getvalue()
    assert "fake device log" in stderr.getvalue()
    assert "fake peak reset log" in stderr.getvalue()
    assert "fake memory query log" in stderr.getvalue()


def test_runtime_gates_are_fail_closed_for_device_and_offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(DCUPreflightError, match="cuda:0"):
        validate_device(FakeDevice(), "cpu")
    with pytest.raises(DCUPreflightError, match="K100"):
        validate_device(FakeDevice(name="software"), "cuda:0")

    expected = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(DCUPreflightError, match="TRANSFORMERS_OFFLINE"):
        validate_offline_environment(expected)


def test_official_loader_uses_pinned_factories_and_strict_reload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    base_model = tmp_path / "base-model"
    checkpoint.mkdir()
    base_model.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"fake")
    (base_model / "config.json").write_text("{}")
    config = {
        "schema_version": 1,
        "name": "dcu_preflight",
        "action_dim": 7,
        "chunk_size": 50,
        "n_action_steps": 1,
        "seed": 2027,
        "task": {
            "suite": "libero_spatial",
            "task_id": 0,
            "observation_height": 360,
            "observation_width": 360,
        },
        "checkpoint": {"path": str(checkpoint), "revision": "checkpoint-rev", "repo_id": "official/checkpoint"},
        "base_model": {"path": str(base_model), "revision": "base-rev", "repo_id": "official/base"},
        "offline": {"HF_HUB_OFFLINE": "1"},
        "runtime": {"logical_device": "cuda:0"},
    }
    validate_worker_config(config)
    env_calls: list[dict[str, object]] = []
    policy_calls: list[dict[str, object]] = []
    reload_calls: list[dict[str, object]] = []

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> "FakeConfig":
            assert path == str(checkpoint)
            assert kwargs["local_files_only"] is True
            assert kwargs["revision"] == "checkpoint-rev"
            return cls()

        def __init__(self) -> None:
            self.device = None
            self.vlm_model_name = ""
            self.pretrained_path = None
            self.pretrained_revision = None
            self.use_peft = True
            self.use_amp = True
            self.compile_model = True
            self.chunk_size = 50
            self.n_action_steps = 1

    class FakePolicyClass:
        @classmethod
        def _load_as_safetensor(cls, policy: FakePolicy, path: str, device: str, strict: bool) -> FakePolicy:
            reload_calls.append({"path": path, "device": device, "strict": strict})
            return policy

    def fake_make_env_config(env_type: str, **kwargs: object) -> object:
        env_calls.append({"env_type": env_type, **kwargs})
        return SimpleNamespace(**kwargs)

    def fake_make_policy(*, cfg: object, env_cfg: object, rename_map: object) -> FakePolicy:
        policy_calls.append({"cfg": cfg, "env_cfg": env_cfg, "rename_map": rename_map})
        return FakeAttentionPolicy()

    policy = build_official_policy(
        config,
        device_adapter=FakeDevice(),
        make_env_config_fn=fake_make_env_config,
        make_policy_fn=fake_make_policy,
        smolvla_config_cls=FakeConfig,
        smolvla_policy_cls=FakePolicyClass,
        environment={"HF_HUB_OFFLINE": "1"},
    )
    assert isinstance(policy, FakePolicy)
    assert env_calls[0]["env_type"] == "libero"
    assert env_calls[0]["task"] == "libero_spatial"
    loaded_cfg = policy_calls[0]["cfg"]
    assert loaded_cfg.device == "cuda:0"
    assert loaded_cfg.vlm_model_name == str(base_model)
    assert loaded_cfg.pretrained_revision == "checkpoint-rev"
    assert loaded_cfg.use_peft is False
    assert loaded_cfg.use_amp is False
    assert loaded_cfg.compile_model is False
    assert reload_calls == [
        {"path": str(checkpoint / "model.safetensors"), "device": "cuda:0", "strict": True}
    ]
    evidence = policy._dcu_model_evidence
    assert evidence["load_latency_seconds"] >= 0
    assert evidence["peak_memory_bytes"] == 1234
    assert evidence["parameter_count"] == 5
    assert evidence["parameter_dtype_counts"] == {"torch.float32": 1, "torch.bfloat16": 1}
    assert evidence["parameter_device_counts"] == {"cuda:0": 2}
    assert evidence["torch_version"] == torch.__version__
    assert evidence["torch_hip_version"] == "fake-hip"
    assert evidence["torch_cuda_version"] is None
    assert evidence["attention"]["requested"] == "eager"
    assert evidence["attention"]["prior"] == {
        "vlm": {
            "_attn_implementation": "legacy",
            "_flash_attn_2_enabled": True,
        },
        "vlm.text": {
            "_attn_implementation": "legacy",
            "_flash_attn_2_enabled": True,
        },
        "vlm.vision": {
            "_attn_implementation": "legacy",
            "_flash_attn_2_enabled": True,
        },
        "action_expert": {
            "_attn_implementation": "legacy",
            "_flash_attn_2_enabled": True,
        },
    }
    assert all(
        entry["_attn_implementation"] == "eager"
        for entry in evidence["attention"]["effective"].values()
    )
    assert evidence["attention_prior"] == evidence["attention"]["prior"]
    assert evidence["attention_effective"] == evidence["attention"]["effective"]


def test_eager_attention_enforcement_fails_closed_when_config_rejects_it() -> None:
    with pytest.raises(DCUPreflightError, match="eager"):
        _enforce_eager_attention(FakeAttentionPolicy(config_cls=LockedAttentionConfig))


def test_load_worker_config_is_local_yaml_only(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("name: dcu_preflight\n")
    assert load_worker_config(path)["name"] == "dcu_preflight"
