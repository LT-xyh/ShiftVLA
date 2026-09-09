"""Contract tests for the additive M1-N0-R1 renderer admission layer.

These tests are intentionally fake-only.  They exercise hashes, manifests,
path contracts, and the pre-import environment transition without importing
EGL, MuJoCo, LIBERO, LeRobot, Torch, or a policy stack.
"""

from __future__ import annotations

import ast
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
OLD_OUTPUT_ROOT = ROOT / "runs/m1_null_calibration/20260901_task000_init000_null20"


def _module():
    import scripts.m1_renderer_admission as admission

    return admission


def test_canonical_config_self_hash_omits_only_config_sha256() -> None:
    admission = _module()
    value = {"schema_version": 1, "name": "renderer_preflight_r1", "config_sha256": ""}
    value["config_sha256"] = admission.config_contract_sha256(value)
    assert admission.verify_config_self_hash(value)
    assert value["config_sha256"] == hashlib.sha256(
        admission.canonical_json({"schema_version": 1, "name": "renderer_preflight_r1"}).encode()
    ).hexdigest()
    changed = copy.deepcopy(value)
    changed["name"] = "changed"
    assert not admission.verify_config_self_hash(changed)


def test_predecessor_manifest_covers_regular_tree_and_semantic_hashes() -> None:
    admission = _module()
    manifest = admission.load_document(ROOT / "runtime/m1/m1_n0_r1_predecessor_manifest.json")
    assert admission.verify_predecessor_manifest(manifest, repo_root=ROOT)
    files = manifest["predecessor_tree"]["files"]
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in OLD_OUTPUT_ROOT.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert {item["path"] for item in files} == expected
    assert len(files) == 205
    assert all(set(item) == {"path", "size", "sha256"} for item in files)
    assert manifest["semantic_hashes"] == {
        "null_config_contract_sha256": "abbda1ee7d347a3c6cafad9f6f79a41a4227b84d063f2eea8ae7d53c2c5fae2b",
        "state_replay_contract_sha256": "730aff4a41fd91fb837102ca5f363a4a142bf7010140f5176940700f1d1fd5f0",
        "run_spec_sha256": "88202837c4dd7c60f0dd968e90d313e8c85a7da479b83ca85f0c32b2685c922c",
        "initial_pair_registry_sha256": "761ce35bb1416e50cbcfc13e7ec8dc1223a2f245a261ddc8c5ac57c06c07f6b9",
        "source_registry_sha256": "092c64910b7530c6d84a98a925124879e9a96196b54d45b284a70ccfb62dd49c",
        "action_bytes_sha256": "c17bc44ad8195fecb42a80b3b272828761a9df6d88dd2bafe45db01a6cb04bbf",
        "final_pair_registry_sha256": "c644ef02171bba0645811e0370a32c947e3065f88adda690c1b1b9fb3066888c",
        "terminal_manifest_sha256": "461c31387aabf0a2bb8db90828fffc50d9181e09e207ab311678facc3df6cd3f",
    }
    assert manifest["terminal_manifest_raw_sha256"] == (
        "6fb307f787eb62e2ef2be20513ee21dbd2852330ba592c6aac4ebc0b8a545a2a"
    )
    assert {item["path"] for item in manifest["anchors"]} == {
        "configs/m1/null_calibration.yaml",
        "configs/m1/state_replay.yaml",
        "runtime/m1/m1_hard_gate_tape_registry.json",
        "runtime/m1/tapes/libero_spatial-task000-init000.actions.npy",
        "runs/m1_null_calibration/20260901_task000_init000_null20/run_spec.json",
        "runs/m1_null_calibration/20260901_task000_init000_null20/pair_registry.json",
        "runs/m1_null_calibration/20260901_task000_init000_null20/final_pair_registry.json",
        "runs/m1_null_calibration/20260901_task000_init000_null20/terminal_manifest.json",
    }


def test_predecessor_manifest_recomputes_semantic_anchors() -> None:
    admission = _module()
    manifest = admission.load_document(ROOT / "runtime/m1/m1_n0_r1_predecessor_manifest.json")
    changed = copy.deepcopy(manifest)
    changed["semantic_hashes"]["run_spec_sha256"] = "0" * 64
    changed["manifest_sha256"] = admission.predecessor_manifest_sha256(changed)
    with pytest.raises(admission.ProvenanceError, match="semantic|run_spec"):
        admission.verify_predecessor_manifest(changed, repo_root=ROOT)


def test_predecessor_manifest_rejects_changed_old_file(tmp_path: Path) -> None:
    admission = _module()
    root = tmp_path / "old"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "record.log").write_bytes(b"original")
    manifest = admission.build_predecessor_manifest(
        root,
        repo_root=tmp_path,
        semantic_hashes={"example": "a" * 64},
    )
    (root / "nested" / "record.log").write_bytes(b"tampered")
    with pytest.raises(admission.ProvenanceError, match="hash|size"):
        admission.verify_predecessor_manifest(manifest, repo_root=tmp_path)


def test_predecessor_manifest_rejects_symlink_entry(tmp_path: Path) -> None:
    admission = _module()
    root = tmp_path / "old"
    root.mkdir()
    (root / "record.log").write_bytes(b"original")
    manifest = admission.build_predecessor_manifest(
        root, repo_root=tmp_path, semantic_hashes={"example": "a" * 64}
    )
    (root / "unexpected-link").symlink_to(root / "record.log")
    with pytest.raises(admission.ProvenanceError, match="symlink|regular"):
        admission.verify_predecessor_manifest(manifest, repo_root=tmp_path)


def test_admission_and_authoritative_paths_are_disjoint() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    admission.validate_admission_config(config, repo_root=ROOT)
    paths = admission.admission_paths(config)
    assert paths["admission_root"] != paths["authoritative_null_root"]
    assert not admission.path_within(paths["admission_root"], paths["authoritative_null_root"])
    assert not admission.path_within(paths["authoritative_null_root"], paths["admission_root"])


def test_admission_schema_has_exact_namespace_and_no_experiment_fields() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    admission.validate_admission_config(config, repo_root=ROOT)
    assert config["policy_compute_device"] == "not_applicable_no_policy"
    assert config["physical_compute_device_id"] is None
    assert config["renderer_backend"] == "egl"
    assert config["renderer_device_id"] == "0"
    forbidden = {
        "action",
        "action_tape",
        "pair",
        "pair_ids",
        "window",
        "windows",
        "envelope",
        "null_threshold",
        "null_thresholds",
    }

    def visit(node: object) -> None:
        if isinstance(node, dict):
            assert not forbidden.intersection(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(config)


def test_policy_sentinel_is_rejected_at_torch_boundary() -> None:
    admission = _module()
    with pytest.raises(admission.AdmissionError, match="policy_compute_device|Torch"):
        admission.torch_device_from_policy_compute_device("not_applicable_no_policy")


def test_governed_environment_transition_is_ordered_and_import_is_last() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "MUJOCO_GL": "old",
        "LD_PRELOAD": "",
        "LD_LIBRARY_PATH": config["governed_environment"]["admitted_ld_library_path"],
        "UNRELATED_USER_VALUE": "preserved",
    }
    events: list[str] = []

    def import_callback() -> None:
        events.append("import")

    result = admission.apply_governed_environment(
        config,
        environ=environment,
        event_sink=events.append,
        import_callback=import_callback,
    )
    assert events.index("record_presence") < events.index("unset")
    assert events.index("unset") < events.index("block_loader")
    assert events.index("block_loader") < events.index("set_cache")
    assert events.index("set_cache") < events.index("set_renderer")
    assert events.index("set_renderer") < events.index("verify_final_mapping")
    assert events[:7] == [
        "record_presence", "unset", "block_loader", "set_cache",
        "set_renderer", "verify_final_mapping", "renderer_import",
    ]
    assert events[-1] == "import"
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert environment["UNRELATED_USER_VALUE"] == "preserved"
    assert result["final_environment"]["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert result["final_environment"]["LD_PRELOAD"] is None
    assert "LD_PRELOAD" not in environment


def test_governed_environment_without_import_callback_does_not_claim_renderer_import() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    environment = {
        "LD_PRELOAD": "",
        "LD_AUDIT": "",
        "LD_LIBRARY_PATH": config["governed_environment"]["admitted_ld_library_path"],
    }
    events: list[str] = []
    admission.apply_governed_environment(config, environ=environment, event_sink=events.append)
    assert events == [
        "record_presence", "unset", "block_loader", "set_cache",
        "set_renderer", "verify_final_mapping",
    ]
    assert "LD_PRELOAD" not in environment
    assert "LD_AUDIT" not in environment


def test_governed_environment_schema_is_exact() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    environment = config["governed_environment"]
    assert environment["blocked_nonempty"] == ["LD_PRELOAD", "LD_AUDIT"]
    assert environment["unset_before_import"] == [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "NVIDIA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "EGL_PLATFORM",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "__EGL_VENDOR_LIBRARY_DIRS",
        "__GLX_VENDOR_LIBRARY_NAME",
        "LIBGL_ALWAYS_INDIRECT",
        "LIBGL_ALWAYS_SOFTWARE",
        "LIBGL_DRIVERS_PATH",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "GALLIUM_DRIVER",
        "DRI_PRIME",
        "MESA_VK_DEVICE_SELECT",
        "VK_ICD_FILENAMES",
    ]
    assert environment["set_cache"] == {
        "LIBERO_CONFIG_PATH": "/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/m1/libero_config",
        "HF_HOME": "/public/home/xuyinghao/tmp/shiftvla-empty-hf-home-g",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    assert environment["set_renderer"] == {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
    }


def test_governed_environment_transition_rejects_loader_injection() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    with pytest.raises(admission.AdmissionError, match="LD_PRELOAD"):
        admission.apply_governed_environment(config, environ={"LD_PRELOAD": "/tmp/inject.so"})


def test_governed_environment_transition_requires_exact_ld_library_path() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    with pytest.raises(admission.AdmissionError, match="LD_LIBRARY_PATH"):
        admission.apply_governed_environment(config, environ={})
    with pytest.raises(admission.AdmissionError, match="LD_LIBRARY_PATH"):
        admission.apply_governed_environment(config, environ={"LD_LIBRARY_PATH": "changed"})


def test_renderer_fingerprint_hashes_are_explicit_and_worker_schema_is_context_free() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    governed = config["governed_environment"]
    governed_names = set(governed["unset_before_import"])
    governed_names.update(governed["blocked_nonempty"])
    governed_names.update(governed["set_cache"])
    governed_names.update(governed["set_renderer"])
    governed_names.add("LD_LIBRARY_PATH")
    governed_values = {name: None for name in governed_names}
    governed_values.update(governed["set_cache"])
    governed_values.update(governed["set_renderer"])
    governed_values["LD_LIBRARY_PATH"] = governed["admitted_ld_library_path"]
    worker = {
        "schema_version": 1,
        "device_count": 1,
        "selected_ordinal": 0,
        "ordered_descriptors": [{
            "ordinal": 0,
            "extensions": ["EGL_EXT_device_base"],
            "drm_device_file": None,
            "drm_render_node_file": None,
        }],
        "supported_drm_nodes": [],
        "namespace_facts": {
            "dev_dri": [],
            "mount_namespace": "mnt:[4026531840]",
            "cgroup": ["0::/"],
            "device_nodes": [],
        },
        "governed_environment": governed_values,
        "static_graphics_libraries": [{
            "path": "/lib/libEGL.so",
            "sha256": "a" * 64,
            "build_id": "deadbeef",
            "package_version": "1.0",
        }],
    }
    full = {
        **worker,
        "egl_identity": {"vendor": "Mesa Project", "version": "1.5"},
        "gl_identity": {"vendor": "Mesa/X.org", "renderer": "llvmpipe", "version": "3.1"},
        "loaded_graphics_libraries": [{
            "path": "/lib/libEGL.so",
            "sha256": "a" * 64,
            "build_id": "deadbeef",
            "package_version": "1.0",
        }],
    }
    full_hash = admission.admission_full_renderer_fingerprint_sha256(full)
    worker_hash = admission.worker_preimport_namespace_fingerprint_sha256(worker)
    assert full_hash == admission.admission_full_renderer_fingerprint_sha256(copy.deepcopy(full))
    assert worker_hash == admission.worker_preimport_namespace_fingerprint_sha256(copy.deepcopy(worker))
    with pytest.raises(admission.AdmissionError, match="context|GL|loaded"):
        admission.worker_preimport_namespace_fingerprint_sha256({**worker, "gl_version": "3.1"})
    with pytest.raises(admission.AdmissionError, match="context|GL|loaded"):
        admission.worker_preimport_namespace_fingerprint_sha256(
            {**worker, "nested": {"post_context": {"loaded_graphics_libraries": []}}}
        )
    with pytest.raises(admission.AdmissionError, match="schema|field"):
        admission.worker_preimport_namespace_fingerprint_sha256(
            {**worker, "ordered_descriptors": [{**worker["ordered_descriptors"][0], "name": "software"}]}
        )


class _FakeEGLBackend:
    EGL_TRUE = 1
    EGL_FALSE = 0
    EGL_SUCCESS = 0x3000
    EGL_BAD_ALLOC = 0x3003
    EGL_EXTENSIONS = "EGL_EXTENSIONS"
    EGL_DRM_DEVICE_FILE_EXT = "EGL_DRM_DEVICE_FILE_EXT"
    EGL_DRM_RENDER_NODE_FILE_EXT = "EGL_DRM_RENDER_NODE_FILE_EXT"
    EGL_NO_DEVICE_EXT = object()

    def __init__(
        self,
        *,
        extensions=None,
        query_results=None,
        query_errors=None,
        reported_counts=None,
    ):
        self.handles = [object(), object()]
        self.extensions = extensions if extensions is not None else [
            [
                "EGL_EXT_device_drm_render_node",
                "EGL_EXT_device_drm",
                "EGL_EXT_device_base",
                "EGL_EXT_device_drm",
            ],
            ["EGL_EXT_device_base"],
        ]
        self.query_results = list(query_results or [self.EGL_TRUE, self.EGL_TRUE])
        self.query_errors = list(query_errors or [self.EGL_SUCCESS, self.EGL_SUCCESS])
        self.reported_counts = list(reported_counts or [len(self.handles), len(self.handles)])
        self.query_calls = []
        self.error_calls = 0
        self.extension_calls = []
        self.drm_calls = []

    def eglQueryDevicesEXT(self, capacity, devices, count):
        call_index = len(self.query_calls)
        self.query_calls.append((capacity, devices is None, len(devices) if devices is not None else None))
        count[0] = self.reported_counts[call_index]
        if capacity:
            for index, handle in enumerate(self.handles[:capacity]):
                devices[index] = handle
        return self.query_results[call_index]

    def eglGetError(self):
        error = self.query_errors[self.error_calls]
        self.error_calls += 1
        return error

    def is_no_device(self, handle):
        return handle is self.EGL_NO_DEVICE_EXT or getattr(handle, "value", 1) == 0

    def query_device_extensions(self, handle):
        self.extension_calls.append(handle)
        return self.extensions[self.handles.index(handle)]

    def query_device_drm_node(self, handle, field):
        self.drm_calls.append((handle, field))
        return {
            "drm_device_file": "/dev/dri/card0",
            "drm_render_node_file": "/dev/dri/renderD128",
        }[field]

def _fake_worker_payload(admission, config, *, device_count=1, selected_ordinal=0):
    governed = config["governed_environment"]
    names = (
        set(governed["unset_before_import"])
        | set(governed["blocked_nonempty"])
        | set(governed["set_cache"])
        | set(governed["set_renderer"])
        | {"LD_LIBRARY_PATH"}
    )
    environment = {name: None for name in names}
    environment.update(governed["set_cache"])
    environment.update(governed["set_renderer"])
    environment["LD_LIBRARY_PATH"] = governed["admitted_ld_library_path"]
    return {
        "schema_version": 1,
        "device_count": device_count,
        "selected_ordinal": selected_ordinal,
        "ordered_descriptors": [{
            "ordinal": index,
            "extensions": ["EGL_EXT_device_base"],
            "drm_device_file": None,
            "drm_render_node_file": None,
        } for index in range(device_count)],
        "supported_drm_nodes": [],
        "namespace_facts": {
            "dev_dri": [],
            "mount_namespace": "mnt:[4026531840]",
            "cgroup": ["0::/"],
            "device_nodes": [],
        },
        "governed_environment": environment,
        "static_graphics_libraries": [{
            "path": "/lib/libEGL.so.1",
            "sha256": "a" * 64,
            "build_id": "build-egl",
            "package_version": "mesa-21.1.5",
        }],
    }


def _fake_environment_transition_audit(config, worker):
    governed = config["governed_environment"]
    names = (
        set(governed["unset_before_import"])
        | set(governed["blocked_nonempty"])
        | set(governed["set_cache"])
        | set(governed["set_renderer"])
        | {"LD_LIBRARY_PATH"}
    )
    initial = {
        name: {"present": False, "nonempty": False}
        for name in sorted(names)
    }
    initial["LD_LIBRARY_PATH"] = {"present": True, "nonempty": True}
    return {
        "events": [
            "record_presence", "unset", "block_loader", "set_cache",
            "set_renderer", "verify_final_mapping", "renderer_import",
        ],
        "initial_presence": initial,
        "final_environment": worker["governed_environment"],
        "renderer_modules_before": [],
        "renderer_modules_after": ["mujoco.egl.egl_ext"],
    }


def _valid_probe_records(admission, config):
    worker = _fake_worker_payload(admission, config)
    worker["ordered_descriptors"][0]["extensions"] = ["EGL_MESA_device_software"]
    worker = admission.build_worker_preimport_fingerprint(
        device_count=1,
        selected_ordinal=0,
        ordered_descriptors=worker["ordered_descriptors"],
        namespace_facts=worker["namespace_facts"],
        governed_environment=worker["governed_environment"],
        static_graphics_libraries=worker["static_graphics_libraries"],
    )
    full = admission.build_full_renderer_fingerprint(
        worker,
        egl_identity={"vendor": "Mesa Project", "version": "1.5"},
        gl_identity={
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
            "version": "3.1 Mesa 21.1.5",
        },
        loaded_libs=worker["static_graphics_libraries"],
    )
    source_sha = admission.probe_source_bindings_sha256(config["probe_contract"]["source_files"])
    return [
        admission.build_renderer_probe_record(
            index=index,
            pid=1000 + index,
            ppid=999,
            worker_preimport_fingerprint=worker,
            full_renderer_fingerprint=full,
            environment_transition_audit=_fake_environment_transition_audit(config, worker),
            config_sha256=config["config_sha256"],
            runtime_lock_sha256=config["runtime"]["runtime_lock"]["sha256"],
            probe_source_bindings_sha256=source_sha,
            execution_commit="a" * 40,
            context_records=[{
                "ordinal": 0,
                "egl_identity": full["egl_identity"],
                "gl_identity": full["gl_identity"],
                "loaded_graphics_libraries": full["loaded_graphics_libraries"],
            }],
            required_selected_extensions=["EGL_MESA_device_software"],
            import_negative={
                "policy_imported": False,
                "processor_imported": False,
                "libero_imported": False,
                "forbidden_modules": [],
            },
            cleanup={
                "status": "PASS",
                "all_contexts_released": True,
                "partial_paths_cleaned": True,
                "egl_thread_released": True,
            },
        )
        for index in range(3)
    ]


def _rehash_probe_payload(admission, record):
    worker = record["worker_preimport_fingerprint"]
    full = record["full_renderer_fingerprint"]
    worker_keys = {
        "schema_version", "device_count", "selected_ordinal", "ordered_descriptors",
        "supported_drm_nodes", "namespace_facts", "governed_environment",
        "static_graphics_libraries",
    }
    for key in worker_keys:
        full[key] = copy.deepcopy(worker[key])
    record["device_count"] = worker["device_count"]
    record["selected_ordinal"] = worker["selected_ordinal"]
    record["ordered_descriptors"] = copy.deepcopy(worker["ordered_descriptors"])
    record["worker_preimport_fingerprint_sha256"] = admission.worker_preimport_namespace_fingerprint_sha256(worker)
    record["full_renderer_fingerprint_sha256"] = admission.admission_full_renderer_fingerprint_sha256(full)
    record["environment_transition_audit"]["final_environment"] = copy.deepcopy(worker["governed_environment"])


def test_checked_egl_query_uses_capacity_zero_then_exact_count_without_fixed_max() -> None:
    admission = _module()
    backend = _FakeEGLBackend()
    devices = admission.query_egl_devices_two_call(backend)
    assert len(devices) == 2
    assert backend.query_calls == [(0, True, None), (2, False, 2)]
    assert backend.error_calls == 2


def test_checked_egl_query_has_no_fixed_ten_device_limit() -> None:
    admission = _module()
    backend = _FakeEGLBackend()
    backend.handles = [object() for _ in range(17)]
    backend.reported_counts = [17, 17]
    devices = admission.query_egl_devices_two_call(backend)
    assert len(devices) == 17
    assert backend.query_calls == [(0, True, None), (17, False, 17)]


class _RawEGLModule:
    EGL_TRUE = 1
    EGL_SUCCESS = 0x3000
    EGLint = ctypes.c_int
    EGLDeviceEXT = ctypes.c_void_p
    EGL_NO_DEVICE_EXT = ctypes.c_void_p()

    def __init__(self, addresses):
        self.addresses = addresses
        self.calls = []
        self._errors = []

        def raw_query(capacity, devices, count_pointer):
            self.calls.append((capacity, devices is None))
            ctypes.cast(count_pointer, ctypes.POINTER(ctypes.c_int))[0] = len(self.addresses)
            if devices is not None:
                for index, address in enumerate(self.addresses):
                    devices[index] = address
            return self.EGL_TRUE

        self._eglQueryDevicesEXT = raw_query

    def eglGetError(self):
        return self._errors.pop(0) if self._errors else self.EGL_SUCCESS


class _UnusedGLModule:
    pass


@pytest.mark.parametrize("addresses", [[], list(range(1, 18))])
def test_lazy_backend_adapts_raw_device_query_with_exact_ctypes_buffer(addresses) -> None:
    admission = _module()
    egl = _RawEGLModule(addresses)
    backend = admission.InjectedEGLBackend(egl, _UnusedGLModule())
    if not addresses:
        with pytest.raises(admission.AdmissionError, match="non-positive"):
            admission.query_egl_devices_two_call(backend)
        assert egl.calls == [(0, True)]
    else:
        devices = admission.query_egl_devices_two_call(backend)
        assert [device.value for device in devices] == addresses
        assert egl.calls == [(0, True), (len(addresses), False)]


@pytest.mark.parametrize(
    "handle",
    [None, 0, ctypes.c_void_p(None), ctypes.POINTER(ctypes.c_int)()],
)
def test_lazy_backend_recognizes_all_null_device_handle_forms(handle) -> None:
    admission = _module()
    backend = admission.InjectedEGLBackend(_RawEGLModule([1]), _UnusedGLModule())
    assert backend.is_no_device(handle)


def test_lazy_backend_loads_renderer_modules_only_when_explicitly_requested(monkeypatch) -> None:
    admission = _module()
    imported = []

    def fake_import(name):
        imported.append(name)
        return object()

    monkeypatch.setattr("importlib.import_module", fake_import)
    backend = admission.InjectedEGLBackend.load()
    assert imported == ["mujoco.egl.egl_ext", "OpenGL.GL"]
    assert backend.egl is not backend.gl


class _RawDeviceStringEGL(_RawEGLModule):
    EGL_EXTENSIONS = 0x3055

    def __init__(self):
        super().__init__([1])
        self.string_calls = []

        def query_string(device, token):
            self.string_calls.append((device.value, token))
            return {
                0x3055: b"EGL_EXT_device_drm EGL_EXT_device_drm_render_node",
                0x3233: b"/dev/dri/card0",
                0x3377: b"/dev/dri/renderD128",
            }[token]

        self.query_string = query_string

    def eglGetProcAddress(self, name):
        assert name in ("eglQueryDeviceStringEXT", b"eglQueryDeviceStringEXT")
        return self.query_string


def test_lazy_backend_queries_raw_device_strings_and_exact_drm_tokens() -> None:
    admission = _module()
    egl = _RawDeviceStringEGL()
    backend = admission.InjectedEGLBackend(egl, _UnusedGLModule())
    descriptors = admission.describe_egl_devices(backend, [ctypes.c_void_p(1)])
    assert descriptors[0]["drm_device_file"] == "/dev/dri/card0"
    assert descriptors[0]["drm_render_node_file"] == "/dev/dri/renderD128"
    assert [token for _device, token in egl.string_calls] == [0x3055, 0x3233, 0x3377]


class _ABISensitiveEGLModule:
    EGL_TRUE = 1
    EGL_SUCCESS = 0x3000
    EGL_BAD_ALLOC = 0x3003
    EGLint = ctypes.c_int
    EGLDeviceEXT = ctypes.c_void_p
    EGLConfig = ctypes.c_int
    EGL_PLATFORM_DEVICE_EXT = 0x313F
    EGL_NO_DISPLAY = ctypes.c_void_p()
    EGL_NO_CONTEXT = ctypes.c_void_p()
    EGL_NO_SURFACE = ctypes.c_void_p()
    EGL_RED_SIZE = 0x3024
    EGL_GREEN_SIZE = 0x3023
    EGL_BLUE_SIZE = 0x3022
    EGL_ALPHA_SIZE = 0x3021
    EGL_DEPTH_SIZE = 0x3025
    EGL_STENCIL_SIZE = 0x3026
    EGL_COLOR_BUFFER_TYPE = 0x303F
    EGL_RGB_BUFFER = 0x308E
    EGL_SURFACE_TYPE = 0x3033
    EGL_PBUFFER_BIT = 0x0001
    EGL_RENDERABLE_TYPE = 0x3040
    EGL_OPENGL_BIT = 0x0008
    EGL_OPENGL_API = 0x30A2
    EGL_NONE = 0x3038

    def __init__(self, *, display_error=EGL_SUCCESS, context_error=EGL_SUCCESS,
                 cleanup_failures=()):
        self.display_error = display_error
        self.context_error = context_error
        self.cleanup_failures = set(cleanup_failures)
        self.calls = []
        self._errors = []
        self.choose_count_types = []

    def _ok(self, name, result=EGL_TRUE):
        self.calls.append(name)
        self._errors.append(self.EGL_BAD_ALLOC if name in self.cleanup_failures else self.EGL_SUCCESS)
        return result

    def eglGetError(self):
        return self._errors.pop(0) if self._errors else self.EGL_SUCCESS

    def eglGetPlatformDisplayEXT(self, platform, device, attributes):
        self.calls.append("display")
        self._errors.append(self.display_error)
        return ctypes.c_void_p(0x1111)

    def eglInitialize(self, display, major, minor):
        return self._ok("initialize")

    def eglChooseConfig(self, display, attributes, config, config_size, count):
        self.calls.append("choose_config")
        self.choose_count_types.append(type(count))
        typed_count = ctypes.cast(count, ctypes.POINTER(self.EGLint))
        typed_count[0] = getattr(self, "reported_count", 1)
        self._errors.append(self.EGL_SUCCESS)
        return self.EGL_TRUE

    def eglBindAPI(self, api):
        return self._ok("bind_api")

    def eglCreateContext(self, display, config, share_context, attributes):
        self.calls.append("create_context")
        self._errors.append(self.context_error)
        return ctypes.c_void_p(0x2222)

    def eglMakeCurrent(self, display, draw, read, context):
        return self._ok("clear_current" if not context else "make_current")

    def eglDestroyContext(self, display, context):
        return self._ok("destroy_context")

    def eglTerminate(self, display):
        return self._ok("terminate")

    def eglReleaseThread(self):
        return self._ok("release_thread")


def test_injected_choose_config_uses_pinned_eglint_pointer_and_rejects_negative_count() -> None:
    admission = _module()
    egl = _ABISensitiveEGLModule()
    backend = admission.InjectedEGLBackend(egl, _UnusedGLModule())
    assert isinstance(backend.choose_config("display"), ctypes.c_int)
    assert egl.choose_count_types == [type(ctypes.byref(ctypes.c_int()))]

    egl.reported_count = -1
    with pytest.raises(admission.AdmissionError, match="choose_config"):
        backend.choose_config("display")


@pytest.mark.parametrize(
    "kwargs, expected_calls, expected_cleanup",
    [
        (
            {"display_error": _ABISensitiveEGLModule.EGL_BAD_ALLOC,
             "cleanup_failures": {"terminate", "release_thread"}},
            ["display", "terminate", "release_thread"],
            ["terminate", "release_thread"],
        ),
        (
            {"context_error": _ABISensitiveEGLModule.EGL_BAD_ALLOC,
             "cleanup_failures": {"clear_current", "destroy_context", "terminate", "release_thread"}},
            ["display", "initialize", "choose_config", "bind_api", "create_context",
             "clear_current", "destroy_context", "terminate", "release_thread"],
            ["clear_current", "destroy_context", "terminate", "release_thread"],
        ),
    ],
)
def test_injected_adapter_hands_off_partial_handles_for_terminal_cleanup(
    kwargs, expected_calls, expected_cleanup
) -> None:
    admission = _module()
    egl = _ABISensitiveEGLModule(**kwargs)
    backend = admission.InjectedEGLBackend(egl, _UnusedGLModule())
    with pytest.raises(admission.AdmissionError) as caught:
        admission.probe_device_context(backend, [ctypes.c_void_p(7)], 0)
    assert egl.calls == expected_calls
    assert caught.value.cleanup["status"] == "FAIL"
    attempted = {
        name for name, item in caught.value.cleanup.items()
        if isinstance(item, dict) and item["attempted"]
    }
    assert attempted == set(expected_cleanup)
    assert all(caught.value.cleanup[name]["success"] is False for name in expected_cleanup)
    assert "0x" not in str(caught.value)
    assert "0x" not in repr(caught.value.cleanup)


def test_checked_egl_query_rejects_no_device_sentinel() -> None:
    admission = _module()
    backend = _FakeEGLBackend()
    backend.handles[1] = backend.EGL_NO_DEVICE_EXT
    with pytest.raises(admission.AdmissionError, match="NO_DEVICE|sentinel"):
        admission.query_egl_devices_two_call(backend)


def test_checked_egl_query_uses_backend_semantic_no_device_predicate() -> None:
    admission = _module()
    backend = _FakeEGLBackend()

    class EquivalentNullHandle:
        value = 0

    backend.handles[1] = EquivalentNullHandle()
    assert backend.handles[1] is not backend.EGL_NO_DEVICE_EXT
    with pytest.raises(admission.AdmissionError, match="NO_DEVICE|sentinel"):
        admission.query_egl_devices_two_call(backend)


def test_checked_egl_query_rejects_an_unfilled_exact_buffer() -> None:
    admission = _module()
    backend = _FakeEGLBackend()
    backend.handles = [object()]
    backend.reported_counts = [2, 2]
    with pytest.raises(admission.AdmissionError, match="fill|buffer"):
        admission.query_egl_devices_two_call(backend)


@pytest.mark.parametrize(
    "query_results, query_errors",
    [
        ([0, 1], [0x3000, 0x3000]),
        ([1, 0], [0x3000, 0x3000]),
        ([1, 1], [0x3003, 0x3000]),
        ([1, 1], [0x3000, 0x3003]),
    ],
)
def test_checked_egl_query_gates_each_boolean_and_egl_error(query_results, query_errors) -> None:
    admission = _module()
    backend = _FakeEGLBackend(query_results=query_results, query_errors=query_errors)
    with pytest.raises(admission.AdmissionError, match="eglQueryDevicesEXT|EGL"):
        admission.query_egl_devices_two_call(backend)


@pytest.mark.parametrize("reported_counts", [[0, 0], [-1, -1], [2, 1], [2, 3]])
def test_checked_egl_query_requires_positive_and_exact_second_count(reported_counts) -> None:
    admission = _module()
    backend = _FakeEGLBackend(reported_counts=reported_counts)
    with pytest.raises(admission.AdmissionError, match="count"):
        admission.query_egl_devices_two_call(backend)


def test_descriptors_are_ordered_and_query_drm_only_when_extension_is_supported() -> None:
    admission = _module()
    backend = _FakeEGLBackend()
    devices = admission.query_egl_devices_two_call(backend)
    descriptors = admission.describe_egl_devices(backend, devices)
    assert [item["ordinal"] for item in descriptors] == [0, 1]
    assert descriptors[0]["drm_device_file"] == "/dev/dri/card0"
    assert descriptors[1]["drm_device_file"] is None
    assert descriptors[0]["extensions"] == [
        "EGL_EXT_device_base", "EGL_EXT_device_drm", "EGL_EXT_device_drm_render_node"
    ]
    assert backend.drm_calls == [
        (devices[0], "drm_device_file"),
        (devices[0], "drm_render_node_file"),
    ]
    assert all(set(item) == {
        "ordinal", "extensions", "drm_device_file", "drm_render_node_file"
    } for item in descriptors)
    assert all(
        isinstance(value, (str, int, type(None), list))
        for item in descriptors
        for value in item.values()
    )


@pytest.mark.parametrize(
    "extensions, expected_primary, expected_render, expected_fields",
    [
        (["EGL_EXT_device_drm"], "/dev/dri/card0", None, ["drm_device_file"]),
        (["EGL_EXT_device_drm_render_node"], None, "/dev/dri/renderD128", ["drm_render_node_file"]),
        (["EGL_EXT_device_base"], None, None, []),
    ],
)
def test_descriptor_drm_fields_have_independent_extension_gates(
    extensions, expected_primary, expected_render, expected_fields
) -> None:
    admission = _module()
    backend = _FakeEGLBackend(extensions=[extensions, ["EGL_EXT_device_base"]])
    devices = admission.query_egl_devices_two_call(backend)
    descriptor = admission.describe_egl_devices(backend, devices)[0]
    assert descriptor["drm_device_file"] == expected_primary
    assert descriptor["drm_render_node_file"] == expected_render
    assert [field for _handle, field in backend.drm_calls] == expected_fields


class _ContextBackend:
    EGL_NO_DISPLAY = None
    EGL_NO_CONTEXT = None

    def __init__(self, *, fail=None, cleanup_fail=()):
        self.fail = fail
        self.cleanup_fail = set(cleanup_fail)
        self.events = []

    def _step(self, name, value=True):
        self.events.append(name)
        if self.fail == name or name in self.cleanup_fail:
            raise RuntimeError(f"{name} failed")
        return value

    def get_platform_display(self, device):
        return self._step("display", "display-handle")

    def initialize(self, display):
        return self._step("initialize")

    def choose_config(self, display):
        return self._step("choose_config", "config-handle")

    def bind_api(self):
        return self._step("bind_api")

    def create_context(self, display, config):
        return self._step("create_context", "context-handle")

    def make_current(self, display, context):
        return self._step("make_current")

    def query_egl_identity(self, display):
        return self._step("read_egl", {"vendor": "Mesa Project", "version": "1.5"})

    def query_gl_identity(self):
        return self._step(
            "read_gl", {"vendor": "Mesa/X.org", "renderer": "llvmpipe", "version": "3.1"}
        )

    def clear_current(self, display):
        return self._step("clear_current")

    def destroy_context(self, display, context):
        return self._step("destroy_context")

    def terminate(self, display):
        return self._step("terminate")

    def release_thread(self):
        return self._step("release_thread")


def test_private_device_context_success_has_exact_lifecycle_and_cleanup_audit() -> None:
    admission = _module()
    backend = _ContextBackend()
    result = admission.probe_device_context(backend, [object()], 0)
    assert result == {
        "ordinal": 0,
        "egl_identity": {"vendor": "Mesa Project", "version": "1.5"},
        "gl_identity": {"vendor": "Mesa/X.org", "renderer": "llvmpipe", "version": "3.1"},
        "cleanup": {
            "status": "PASS",
            "clear_current": {"attempted": True, "success": True, "error": None},
            "destroy_context": {"attempted": True, "success": True, "error": None},
            "terminate": {"attempted": True, "success": True, "error": None},
            "release_thread": {"attempted": True, "success": True, "error": None},
        },
    }
    assert backend.events == [
        "display", "initialize", "choose_config", "bind_api", "create_context",
        "make_current", "read_egl", "read_gl", "clear_current", "destroy_context",
        "terminate", "release_thread",
    ]


def test_private_device_context_rejects_invalid_ordinal_before_backend_calls() -> None:
    admission = _module()
    backend = _ContextBackend()
    with pytest.raises(admission.AdmissionError, match="ordinal|range"):
        admission.probe_device_context(backend, [object()], 1)
    assert backend.events == []


@pytest.mark.parametrize(
    "failed_stage, expected_cleanup",
    [
        ("display", ["release_thread"]),
        ("initialize", ["terminate", "release_thread"]),
        ("choose_config", ["terminate", "release_thread"]),
        ("bind_api", ["terminate", "release_thread"]),
        ("create_context", ["terminate", "release_thread"]),
        ("make_current", ["clear_current", "destroy_context", "terminate", "release_thread"]),
        ("read_egl", ["clear_current", "destroy_context", "terminate", "release_thread"]),
        ("read_gl", ["clear_current", "destroy_context", "terminate", "release_thread"]),
    ],
)
def test_private_context_failure_attempts_every_applicable_cleanup(failed_stage, expected_cleanup) -> None:
    admission = _module()
    backend = _ContextBackend(fail=failed_stage)
    with pytest.raises(admission.AdmissionError, match=failed_stage) as caught:
        admission.probe_device_context(backend, [object()], 0)
    assert backend.events[-len(expected_cleanup):] == expected_cleanup
    assert caught.value.cleanup["status"] == "FAIL"
    assert all(caught.value.cleanup[name]["attempted"] for name in expected_cleanup)
    assert "object at 0x" not in str(caught.value)


def test_private_context_preserves_primary_and_all_cleanup_failures() -> None:
    admission = _module()
    backend = _ContextBackend(
        fail="read_gl",
        cleanup_fail={"clear_current", "destroy_context", "terminate", "release_thread"},
    )
    with pytest.raises(admission.AdmissionError, match="read_gl") as caught:
        admission.probe_device_context(backend, [object()], 0)
    assert backend.events[-4:] == [
        "clear_current", "destroy_context", "terminate", "release_thread"
    ]
    assert caught.value.cleanup["status"] == "FAIL"
    for name in ("clear_current", "destroy_context", "terminate", "release_thread"):
        assert caught.value.cleanup[name] == {
            "attempted": True,
            "success": False,
            "error": {"type": "RuntimeError", "message": f"{name} failed"},
        }


def test_private_context_elides_pointer_repr_from_primary_and_cleanup_errors() -> None:
    admission = _module()

    class _PointerFailureBackend(_ContextBackend):
        def query_gl_identity(self):
            self.events.append("read_gl")
            raise RuntimeError("driver returned object at 0x1234")

    backend = _PointerFailureBackend()
    with pytest.raises(admission.AdmissionError, match="read_gl") as caught:
        admission.probe_device_context(backend, [object()], 0)
    assert "0x" not in str(caught.value)
    assert "0x" not in repr(caught.value.cleanup)
    assert caught.value.cleanup["status"] == "FAIL"
    assert caught.value.cleanup["release_thread"]["success"] is True


@pytest.mark.parametrize("value", [None, [], "not-a-mapping"])
def test_fingerprint_validators_wrap_malformed_top_level_payloads(value) -> None:
    admission = _module()
    with pytest.raises(admission.AdmissionError):
        admission.worker_preimport_namespace_fingerprint_sha256(value)
    with pytest.raises(admission.AdmissionError):
        admission.admission_full_renderer_fingerprint_sha256(value)


def test_full_and_worker_fingerprints_have_disjoint_context_schemas() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    worker = _fake_worker_payload(admission, config)
    built_worker = admission.build_worker_preimport_fingerprint(
        device_count=worker["device_count"],
        selected_ordinal=worker["selected_ordinal"],
        ordered_descriptors=worker["ordered_descriptors"],
        namespace_facts=worker["namespace_facts"],
        governed_environment=worker["governed_environment"],
        static_graphics_libraries=worker["static_graphics_libraries"],
    )
    full = admission.build_full_renderer_fingerprint(
        built_worker,
        egl_identity={"vendor": "Mesa Project", "version": "1.5"},
        gl_identity={"vendor": "Mesa/X.org", "renderer": "llvmpipe", "version": "3.1"},
        loaded_libs=[{"path": "/lib/libEGL.so.1", "sha256": "a" * 64,
                      "build_id": "b", "package_version": "v"}],
    )
    worker_hash = admission.worker_preimport_namespace_fingerprint_sha256(built_worker)
    full_hash = admission.admission_full_renderer_fingerprint_sha256(full)
    assert worker_hash != full_hash
    assert admission.worker_preimport_namespace_fingerprint_sha256(built_worker) == worker_hash
    with pytest.raises(admission.AdmissionError, match="context|GL|loaded"):
        admission.worker_preimport_namespace_fingerprint_sha256(
            {**built_worker, "egl_identity": {"vendor": "Mesa"}}
        )
    assert set(full) == {
        *built_worker, "egl_identity", "gl_identity", "loaded_graphics_libraries",
    }
    assert all(name not in full for name in ("pid", "timestamp", "address", "pointer"))
    assert admission.admission_full_renderer_fingerprint_sha256(copy.deepcopy(full)) == full_hash

    with pytest.raises(admission.AdmissionError, match="schema|field|volatile"):
        admission.admission_full_renderer_fingerprint_sha256({**full, "pid": 123})


def test_worker_builder_normalizes_only_set_like_fields_and_rejects_pointer_values() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    worker = _fake_worker_payload(admission, config, device_count=2)
    worker["ordered_descriptors"] = [
        {
            "ordinal": 0,
            "extensions": [
                "z", "EGL_EXT_device_drm_render_node", "EGL_EXT_device_drm", "a", "z"
            ],
            "drm_device_file": "/dev/dri/card0",
            "drm_render_node_file": "/dev/dri/renderD128",
        },
        {
            "ordinal": 1,
            "extensions": ["b"],
            "drm_device_file": None,
            "drm_render_node_file": None,
        },
    ]
    worker["static_graphics_libraries"] = [
        {"path": "/lib/z.so", "sha256": "b" * 64, "build_id": None, "package_version": None},
        {"path": "/lib/a.so", "sha256": "a" * 64, "build_id": "id", "package_version": "1"},
    ]
    built = admission.build_worker_preimport_fingerprint(
        device_count=2,
        selected_ordinal=0,
        ordered_descriptors=worker["ordered_descriptors"],
        namespace_facts=worker["namespace_facts"],
        governed_environment=worker["governed_environment"],
        static_graphics_libraries=worker["static_graphics_libraries"],
    )
    assert [item["ordinal"] for item in built["ordered_descriptors"]] == [0, 1]
    assert built["ordered_descriptors"][0]["extensions"] == [
        "EGL_EXT_device_drm", "EGL_EXT_device_drm_render_node", "a", "z"
    ]
    assert built["supported_drm_nodes"] == ["/dev/dri/card0", "/dev/dri/renderD128"]
    assert [item["path"] for item in built["static_graphics_libraries"]] == ["/lib/a.so", "/lib/z.so"]
    assert admission.worker_preimport_namespace_fingerprint_sha256(built)
    noncanonical = copy.deepcopy(built)
    noncanonical["static_graphics_libraries"].reverse()
    with pytest.raises(admission.AdmissionError, match="sorted|unique|canonical"):
        admission.worker_preimport_namespace_fingerprint_sha256(noncanonical)

    pointer_descriptor = copy.deepcopy(worker["ordered_descriptors"])
    pointer_descriptor[0]["drm_device_file"] = object()
    with pytest.raises(admission.AdmissionError, match="scalar|pointer|DRM"):
        admission.build_worker_preimport_fingerprint(
            device_count=2,
            selected_ordinal=0,
            ordered_descriptors=pointer_descriptor,
            namespace_facts=worker["namespace_facts"],
            governed_environment=worker["governed_environment"],
            static_graphics_libraries=worker["static_graphics_libraries"],
        )

    unsupported_drm = copy.deepcopy(worker["ordered_descriptors"])
    unsupported_drm[0]["extensions"] = ["EGL_EXT_device_base"]
    with pytest.raises(admission.AdmissionError, match="extension|DRM"):
        admission.build_worker_preimport_fingerprint(
            device_count=2,
            selected_ordinal=0,
            ordered_descriptors=unsupported_drm,
            namespace_facts=worker["namespace_facts"],
            governed_environment=worker["governed_environment"],
            static_graphics_libraries=worker["static_graphics_libraries"],
        )

    missing_render_extension = copy.deepcopy(worker["ordered_descriptors"])
    missing_render_extension[0]["extensions"] = ["EGL_EXT_device_drm"]
    with pytest.raises(admission.AdmissionError, match="render|extension|DRM"):
        admission.build_worker_preimport_fingerprint(
            device_count=2,
            selected_ordinal=0,
            ordered_descriptors=missing_render_extension,
            namespace_facts=worker["namespace_facts"],
            governed_environment=worker["governed_environment"],
            static_graphics_libraries=worker["static_graphics_libraries"],
        )

    invalid_extension = copy.deepcopy(worker["ordered_descriptors"])
    invalid_extension[0]["extensions"] = ["valid", object()]
    with pytest.raises(admission.AdmissionError, match="extension|scalar|string"):
        admission.build_worker_preimport_fingerprint(
            device_count=2,
            selected_ordinal=0,
            ordered_descriptors=invalid_extension,
            namespace_facts=worker["namespace_facts"],
            governed_environment=worker["governed_environment"],
            static_graphics_libraries=worker["static_graphics_libraries"],
        )


def test_library_set_rejects_duplicate_path_conflicts() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    worker = _fake_worker_payload(admission, config)
    libraries = worker["static_graphics_libraries"] * 2
    libraries[1] = {**libraries[1], "sha256": "b" * 64}
    with pytest.raises(admission.AdmissionError, match="duplicate|conflict"):
        admission.build_worker_preimport_fingerprint(
            device_count=1,
            selected_ordinal=0,
            ordered_descriptors=worker["ordered_descriptors"],
            namespace_facts=worker["namespace_facts"],
            governed_environment=worker["governed_environment"],
            static_graphics_libraries=libraries,
        )


@pytest.mark.parametrize(
    "supported_nodes",
    [
        ["/dev/dri/card0"],
        ["/dev/dri/card0", "/dev/dri/renderD128", "/dev/dri/renderD129"],
    ],
)
def test_worker_hash_rejects_forged_supported_drm_node_union(supported_nodes) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    worker = _fake_worker_payload(admission, config)
    worker["ordered_descriptors"][0] = {
        "ordinal": 0,
        "extensions": ["EGL_EXT_device_drm", "EGL_EXT_device_drm_render_node"],
        "drm_device_file": "/dev/dri/card0",
        "drm_render_node_file": "/dev/dri/renderD128",
    }
    worker["supported_drm_nodes"] = supported_nodes
    with pytest.raises(admission.AdmissionError, match="DRM|union|descriptor"):
        admission.worker_preimport_namespace_fingerprint_sha256(worker)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(device_count=0),
        lambda value: value.update(selected_ordinal=1),
        lambda value: value["ordered_descriptors"][0].update(ordinal=1),
        lambda value: value["ordered_descriptors"][0].update(extensions=["z", "a", "a"]),
        lambda value: value["ordered_descriptors"][0].update(drm_device_file="relative/card0"),
        lambda value: value.update(supported_drm_nodes=[{"context": "bad"}]),
        lambda value: value["namespace_facts"].update(dev_dri=[{"renderer": "bad"}]),
        lambda value: value["namespace_facts"].update(extra="bad"),
        lambda value: value["static_graphics_libraries"][0].update(sha256="A" * 64),
        lambda value: value["static_graphics_libraries"][0].update(vendor="Mesa"),
    ],
)
def test_worker_preimport_fingerprint_rejects_nested_schema_bypasses(mutation) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    governed = config["governed_environment"]
    governed_names = (
        set(governed["unset_before_import"])
        | set(governed["blocked_nonempty"])
        | set(governed["set_cache"])
        | set(governed["set_renderer"])
        | {"LD_LIBRARY_PATH"}
    )
    worker = {
        "schema_version": 1,
        "device_count": 1,
        "selected_ordinal": 0,
        "ordered_descriptors": [{
            "ordinal": 0, "extensions": [], "drm_device_file": None,
            "drm_render_node_file": None,
        }],
        "supported_drm_nodes": [],
        "namespace_facts": {
            "dev_dri": [], "mount_namespace": "mnt:[1]", "cgroup": [], "device_nodes": [],
        },
        "governed_environment": {name: None for name in governed_names},
        "static_graphics_libraries": [{
            "path": "/lib/libEGL.so", "sha256": "a" * 64,
            "build_id": None, "package_version": None,
        }],
    }
    mutation(worker)
    with pytest.raises(admission.AdmissionError):
        admission.worker_preimport_namespace_fingerprint_sha256(worker)


def test_admission_module_keeps_renderer_and_policy_imports_lazy() -> None:
    admission = _module()
    source = ast.parse(Path(admission.__file__).read_text(encoding="utf-8"))
    eager = "\n".join(
        ast.unparse(node)
        for node in source.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    assert "mujoco" not in eager.lower()
    assert "egl" not in eager.lower()
    assert "torch" not in eager.lower()
    assert "libero" not in eager.lower()
    assert "smolvla" not in eager.lower()


def test_admission_config_pins_interpreter_and_runtime_lock_without_constructing_environment() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    assert config["runtime"]["python_executable"] == "/public/home/xuyinghao/tmp/shiftvla-libero/bin/python"
    lock = config["runtime"]["runtime_lock"]
    assert lock["sha256"] == "921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc"
    assert config["evidence_role"] == "infrastructure_admission_only"
    assert config["included_in_null_calibration_evidence"] is False
    assert "prepare_run" in config["operations"]["forbidden_dispatch"]
    assert config["operations"]["authoritative_schedule_creation"] is False

    predecessor = config["predecessor"]
    manifest = admission.load_document(ROOT / predecessor["manifest_path"])
    assert predecessor == {
        "manifest_path": "runtime/m1/m1_n0_r1_predecessor_manifest.json",
        "manifest_raw_sha256": hashlib.sha256(
            (ROOT / predecessor["manifest_path"]).read_bytes()
        ).hexdigest(),
        "manifest_self_sha256": manifest["manifest_sha256"],
        "preserved_output_root": "runs/m1_null_calibration/20260901_task000_init000_null20",
    }
    assert all(set(item) == {"path", "sha256"} for item in config["runtime"]["source_files"].values())


def test_admission_config_validator_rejects_nested_contract_drift() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    mutations = [
        ("runtime", "runtime_lock", "sha256"),
        ("runtime", "libero_config_tree", "root"),
        ("governed_environment", "admitted_ld_library_path"),
        ("operations", "authoritative_schedule_creation"),
        ("pass_contract", "public_render_calls"),
        ("publication", "atomic"),
    ]
    for keys in mutations:
        changed = copy.deepcopy(config)
        target = changed
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = "changed"
        changed["config_sha256"] = admission.config_contract_sha256(changed)
        with pytest.raises(admission.AdmissionError):
            admission.validate_admission_config(changed, repo_root=ROOT)


def test_load_document_rejects_symlink_and_non_regular_file(tmp_path: Path) -> None:
    admission = _module()
    regular = tmp_path / "value.json"
    regular.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(regular)
    with pytest.raises(admission.ProvenanceError, match="symlink|regular"):
        admission.load_document(link)
    with pytest.raises(admission.ProvenanceError, match="symlink|regular"):
        admission.load_document(tmp_path)


def test_atomic_no_overwrite_json_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admission = _module()
    destination = tmp_path / "record.json"
    destination.write_bytes(b"preserved\n")
    with pytest.raises(FileExistsError):
        admission._publish_json_no_overwrite(destination, {"new": True})
    assert destination.read_bytes() == b"preserved\n"

    destination.unlink()
    real_link = admission.os.link
    observed: list[bytes] = []

    def checked_link(source: str, target: str) -> None:
        assert not destination.exists()
        observed.append(Path(source).read_bytes())
        real_link(source, target)

    monkeypatch.setattr(admission.os, "link", checked_link)
    admission._publish_json_no_overwrite(destination, {"complete": True})
    assert observed == [b'{\n  "complete": true\n}\n']
    assert destination.read_bytes() == observed[0]


def test_publication_is_confined_to_exact_admission_allowlist(tmp_path: Path) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    assert config["publication"]["artifact_allowlist"] == [
        "probe-000.json", "probe-000.stdout.log", "probe-000.stderr.log",
        "probe-001.json", "probe-001.stdout.log", "probe-001.stderr.log",
        "probe-002.json", "probe-002.stdout.log", "probe-002.stderr.log",
        "preflight.json", "preflight.stdout.log", "preflight.stderr.log",
        "terminal_manifest.json",
    ]
    isolated = copy.deepcopy(config)
    admission_root = tmp_path / "admission"
    authoritative_root = tmp_path / "authoritative"
    isolated["paths"] = {
        "admission_root": str(admission_root),
        "authoritative_null_root": str(authoritative_root),
    }
    isolated["config_sha256"] = admission.config_contract_sha256(isolated)
    for name in ("unknown.json", "nested/probe-000.json", "../authoritative/escape.json"):
        with pytest.raises(admission.AdmissionError):
            admission.resolve_admission_artifact(isolated, name, repo_root=tmp_path)
        assert not admission_root.exists()
        assert not authoritative_root.exists()
    admission_root.symlink_to(authoritative_root, target_is_directory=True)
    with pytest.raises(admission.AdmissionError, match="symlink"):
        admission.resolve_admission_artifact(isolated, "probe-000.json", repo_root=tmp_path)


def test_public_publishers_validate_before_creating_directories(tmp_path: Path) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    invalid = copy.deepcopy(config)
    admission_root = tmp_path / "must-not-exist"
    invalid["paths"]["admission_root"] = str(admission_root)
    invalid["config_sha256"] = "0" * 64
    with pytest.raises(admission.AdmissionError):
        admission.publish_admission_json(invalid, "probe-000.json", {}, repo_root=tmp_path)
    assert not admission_root.exists()
    with pytest.raises(admission.AdmissionError):
        admission.publish_admission_bytes(invalid, "probe-000.stdout.log", b"", repo_root=tmp_path)
    assert not admission_root.exists()


def test_exact_tree_and_empty_cache_reject_extra_entries(tmp_path: Path) -> None:
    admission = _module()
    tree = tmp_path / "tree"
    tree.mkdir()
    file = tree / "config.yaml"
    file.write_bytes(b"frozen")
    expected = [{"path": "tree/config.yaml", "type": "file", "size": 6,
                 "sha256": hashlib.sha256(b"frozen").hexdigest()}]
    admission.verify_exact_regular_tree(tree, expected, repo_root=tmp_path)
    (tree / "extra").write_bytes(b"x")
    with pytest.raises(admission.ProvenanceError, match="tree"):
        admission.verify_exact_regular_tree(tree, expected, repo_root=tmp_path)

    target = tmp_path / "real-tree"
    target.mkdir()
    (target / "config.yaml").write_bytes(b"frozen")
    root_link = tmp_path / "tree-link"
    root_link.symlink_to(target, target_is_directory=True)
    linked_expected = [{**expected[0], "path": "tree-link/config.yaml"}]
    with pytest.raises(admission.ProvenanceError, match="symlink"):
        admission.verify_exact_regular_tree(root_link, linked_expected, repo_root=tmp_path)
    (tree / "extra").unlink()
    (tree / "extra-directory").mkdir()
    with pytest.raises(admission.ProvenanceError, match="tree"):
        admission.verify_exact_regular_tree(tree, expected, repo_root=tmp_path)

    cache = tmp_path / "cache"
    cache.mkdir()
    admission.verify_empty_directory(cache)
    (cache / "unexpected").write_bytes(b"x")
    with pytest.raises(admission.ProvenanceError, match="empty"):
        admission.verify_empty_directory(cache)


def test_probe_contract_freezes_three_children_and_renderer_source_bindings() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    contract = config["probe_contract"]
    assert contract == {
        "schema_version": 1,
        "probe_count": 3,
        "expected_device_count": 1,
        "selected_ordinal": 0,
        "required_selected_extensions": ["EGL_MESA_device_software"],
        "expected_egl_identity": {"vendor": "Mesa Project", "version": "1.5"},
        "expected_gl_identity": {
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
            "version": "3.1 Mesa 21.1.5",
        },
        "source_files": {
            "mujoco_egl_ext": {
                "path": "/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/mujoco/egl/egl_ext.py",
                "sha256": "d3d261c51070482ca749efe14ccb5deb56c6abaf180092f20c8b954a6fa7ed05",
            },
            "robosuite_egl_context": {
                "path": "external/robosuite/robosuite/renderers/context/egl_context.py",
                "sha256": "0919f2232ac4eccde2a1a2e1986b623a99fea109fa86a318cab04537150def49",
            },
            "robosuite_egl_context_installed": {
                "path": "/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/robosuite/renderers/context/egl_context.py",
                "sha256": "0919f2232ac4eccde2a1a2e1986b623a99fea109fa86a318cab04537150def49",
            },
            "pyopengl_egl": {
                "path": "/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/OpenGL/EGL/__init__.py",
                "sha256": "13979115c0873667997e8ba58b985dca3f4f3bb3f5b68c89ce58f4e5505ad7fd",
            },
            "pyopengl_error": {
                "path": "/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/OpenGL/error.py",
                "sha256": "409e1416a583409dbe99fcb78fce96e6ed0fcdfd82fa3492345692efd8e81e86",
            },
        },
    }
    assert admission.verify_admission_probe_contract(contract, repo_root=ROOT)


def test_probe_record_builder_and_three_child_verifier_require_exact_payloads() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    governed = config["governed_environment"]
    worker = _fake_worker_payload(admission, config)
    worker["ordered_descriptors"][0]["extensions"] = ["EGL_MESA_device_software"]
    worker = admission.build_worker_preimport_fingerprint(
        device_count=1,
        selected_ordinal=0,
        ordered_descriptors=worker["ordered_descriptors"],
        namespace_facts=worker["namespace_facts"],
        governed_environment=worker["governed_environment"],
        static_graphics_libraries=worker["static_graphics_libraries"],
    )
    full = admission.build_full_renderer_fingerprint(
        worker,
        egl_identity={"vendor": "Mesa Project", "version": "1.5"},
        gl_identity={
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
            "version": "3.1 Mesa 21.1.5",
        },
        loaded_libs=worker["static_graphics_libraries"],
    )
    records = [
        admission.build_renderer_probe_record(
            index=index,
            pid=1000 + index,
            ppid=999,
            worker_preimport_fingerprint=worker,
            full_renderer_fingerprint=full,
            environment_transition_audit=_fake_environment_transition_audit(config, worker),
            config_sha256=config["config_sha256"],
            runtime_lock_sha256=config["runtime"]["runtime_lock"]["sha256"],
            probe_source_bindings_sha256=admission.probe_source_bindings_sha256(
                config["probe_contract"]["source_files"]
            ),
            execution_commit="a" * 40,
            context_records=[{
                "ordinal": 0,
                "egl_identity": full["egl_identity"],
                "gl_identity": full["gl_identity"],
                "loaded_graphics_libraries": full["loaded_graphics_libraries"],
            }],
            required_selected_extensions=["EGL_MESA_device_software"],
            import_negative={
                "policy_imported": False,
                "processor_imported": False,
                "libero_imported": False,
                "forbidden_modules": [],
            },
            cleanup={
                "status": "PASS",
                "all_contexts_released": True,
                "partial_paths_cleaned": True,
                "egl_thread_released": True,
            },
        )
        for index in range(3)
    ]
    assert admission.verify_three_probe_records(
        records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40,
        repo_root=ROOT,
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r: r[0].update(index="001"), "index"),
        (lambda r: r[1].update(pid=r[0]["pid"]), "PID"),
        (lambda r: r[1].update(ppid=123), "PPID"),
        (lambda r: r[0].update(evidence_role="null_calibration"), "evidence"),
        (lambda r: r[0].update(included_in_null_calibration_evidence=True), "null"),
        (lambda r: r[0].update(exit_code=1), "exit"),
        (lambda r: r[0].update(retry_count=1), "retry"),
        (lambda r: r[0].update(fallback_used=True), "fallback"),
        (lambda r: r[0]["import_negative"].update(policy_imported=True), "policy"),
        (lambda r: r[0]["cleanup"].update(status="FAIL"), "cleanup"),
        (lambda r: r[0]["worker_preimport_fingerprint"].update(device_count=2), "device"),
        (lambda r: r[0]["full_renderer_fingerprint"]["gl_identity"].update(renderer="other"), "GL"),
        (lambda r: r[0].update(full_renderer_fingerprint_sha256="0" * 64), "hash"),
    ],
)
def test_three_child_verifier_rejects_identity_and_protocol_drift(mutation, message) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    worker = _fake_worker_payload(admission, config)
    worker["ordered_descriptors"][0]["extensions"] = ["EGL_MESA_device_software"]
    worker = admission.build_worker_preimport_fingerprint(
        device_count=1,
        selected_ordinal=0,
        ordered_descriptors=worker["ordered_descriptors"],
        namespace_facts=worker["namespace_facts"],
        governed_environment=worker["governed_environment"],
        static_graphics_libraries=worker["static_graphics_libraries"],
    )
    full = admission.build_full_renderer_fingerprint(
        worker,
        egl_identity={"vendor": "Mesa Project", "version": "1.5"},
        gl_identity={
            "vendor": "Mesa/X.org",
            "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
            "version": "3.1 Mesa 21.1.5",
        },
        loaded_libs=worker["static_graphics_libraries"],
    )
    records = [
        admission.build_renderer_probe_record(
            index=index,
            pid=1000 + index,
            ppid=999,
            worker_preimport_fingerprint=worker,
            full_renderer_fingerprint=full,
            environment_transition_audit=_fake_environment_transition_audit(config, worker),
            config_sha256=config["config_sha256"],
            runtime_lock_sha256=config["runtime"]["runtime_lock"]["sha256"],
            probe_source_bindings_sha256=admission.probe_source_bindings_sha256(
                config["probe_contract"]["source_files"]
            ),
            execution_commit="a" * 40,
            context_records=[{
                "ordinal": 0,
                "egl_identity": full["egl_identity"],
                "gl_identity": full["gl_identity"],
                "loaded_graphics_libraries": full["loaded_graphics_libraries"],
            }],
            required_selected_extensions=["EGL_MESA_device_software"],
            import_negative={
                "policy_imported": False,
                "processor_imported": False,
                "libero_imported": False,
                "forbidden_modules": [],
            },
            cleanup={
                "status": "PASS",
                "all_contexts_released": True,
                "partial_paths_cleaned": True,
                "egl_thread_released": True,
            },
        )
        for index in range(3)
    ]
    if message == "hash":
        records[0]["full_renderer_fingerprint_sha256"] = "0" * 64
    else:
        mutation(records)
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_three_probe_records(
            records,
            config=config,
            expected_parent_pid=999,
            expected_execution_commit="a" * 40,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("config_sha256", "b" * 64, "config"),
        ("runtime_lock_sha256", "b" * 64, "runtime"),
        ("probe_source_bindings_sha256", "b" * 64, "source"),
        ("execution_commit", "b" * 40, "execution"),
    ],
)
def test_three_child_verifier_rejects_provenance_binding_drift(field, value, message) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    records[0][field] = value
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_three_probe_records(
            records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r: r[0]["context_records"].pop(), "context"),
        (lambda r: r[0]["context_records"][0].update(ordinal=1), "context"),
        (lambda r: r[0]["context_records"][0]["gl_identity"].update(renderer="other"), "GL"),
        (lambda r: r[0]["context_records"][0].update(pointer=object()), "context"),
    ],
)
def test_three_child_verifier_rejects_context_record_drift(mutation, message) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    mutation(records)
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_three_probe_records(
            records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40,
            repo_root=ROOT,
        )


def test_probe_environment_transition_audit_uses_exact_governed_mapping() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    audit = records[0]["environment_transition_audit"]
    assert audit["events"] == [
        "record_presence", "unset", "block_loader", "set_cache",
        "set_renderer", "verify_final_mapping", "renderer_import",
    ]
    assert audit["final_environment"] == admission.governed_environment_final_mapping(config)
    assert audit["renderer_modules_before"] == []
    assert audit["renderer_modules_after"] == ["mujoco.egl.egl_ext"]


def test_authoritative_three_probe_verifier_requires_repository_root() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    with pytest.raises(TypeError, match="repo_root"):
        admission.verify_three_probe_records(
            records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r: r[0]["environment_transition_audit"]["events"].reverse(), "event"),
        (lambda r: r[0]["environment_transition_audit"]["renderer_modules_before"].append("mujoco"), "before"),
        (lambda r: r[0]["environment_transition_audit"]["renderer_modules_after"].clear(), "after"),
        (lambda r: r[0]["environment_transition_audit"]["initial_presence"].pop("MUJOCO_GL"), "presence"),
        (lambda r: r[0]["environment_transition_audit"]["final_environment"].update(MUJOCO_GL="old"), "environment"),
        (lambda r: r[1]["environment_transition_audit"]["final_environment"].update(LD_PRELOAD="bad"), "environment"),
    ],
)
def test_three_child_verifier_rejects_environment_transition_audit_drift(mutation, message) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    mutation(records)
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_three_probe_records(
            records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40,
            repo_root=ROOT,
        )


def test_three_child_verifier_rejects_valid_schema_but_divergent_transition_audit() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    records[1]["environment_transition_audit"]["renderer_modules_after"] = ["OpenGL"]
    with pytest.raises(admission.AdmissionError, match="transition|audit"):
        admission.verify_three_probe_records(
            records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda r: r[0]["environment_transition_audit"]["initial_presence"].update(
                LD_PRELOAD={"present": True, "nonempty": True}
            ),
            "blocked|loader",
        ),
        (
            lambda r: r[0]["environment_transition_audit"]["initial_presence"].update(
                LD_LIBRARY_PATH={"present": False, "nonempty": False}
            ),
            "LD_LIBRARY_PATH|presence",
        ),
    ],
)
def test_probe_transition_audit_rejects_invalid_initial_loader_presence(mutation, message) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    mutation(records)
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_three_probe_records(
            records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40,
            repo_root=ROOT,
        )


@pytest.mark.parametrize(
    "drift, message",
    [
        (lambda record: record["worker_preimport_fingerprint"]["governed_environment"].update(
            MUJOCO_EGL_DEVICE_ID="1"
        ), "environment"),
        (lambda record: record["worker_preimport_fingerprint"]["governed_environment"].update(
            LD_PRELOAD="injected"
        ), "environment"),
        (lambda record: record["worker_preimport_fingerprint"]["governed_environment"].update(
            HF_HOME="/tmp/changed-cache"
        ), "environment"),
    ],
)
def test_three_child_verifier_rejects_consistent_rehashed_environment_drift(drift, message) -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    for record in records:
        drift(record)
        _rehash_probe_payload(admission, record)
    with pytest.raises(admission.AdmissionError, match=message):
        admission.verify_three_probe_records(
            records, config=config, expected_parent_pid=999, expected_execution_commit="a" * 40,
            repo_root=ROOT,
        )


def test_authoritative_three_probe_verifier_rejects_rehashed_outer_config_drift() -> None:
    admission = _module()
    config = admission.load_admission_config(ROOT / "configs/m1/renderer_preflight_r1_egl0.yaml")
    records = _valid_probe_records(admission, config)
    changed = copy.deepcopy(config)
    changed["renderer_device_id"] = "1"
    changed["config_sha256"] = admission.config_contract_sha256(changed)
    with pytest.raises(admission.AdmissionError, match="renderer namespace|config"):
        admission.verify_three_probe_records(
            records, config=changed, expected_parent_pid=999,
            expected_execution_commit="a" * 40, repo_root=ROOT,
        )
