"""Contract tests for the additive M1-N0-R1 renderer admission layer.

These tests are intentionally fake-only.  They exercise hashes, manifests,
path contracts, and the pre-import environment transition without importing
EGL, MuJoCo, LIBERO, LeRobot, Torch, or a policy stack.
"""

from __future__ import annotations

import ast
import copy
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
    assert events[-1] == "import"
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert environment["UNRELATED_USER_VALUE"] == "preserved"
    assert result["final_environment"]["MUJOCO_EGL_DEVICE_ID"] == "0"


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
    full = {
        "device_count": 1,
        "selected_ordinal": 0,
        "renderer_identity": {"vendor": "Mesa/X.org", "renderer": "llvmpipe"},
        "context_derived": {"gl_version": "3.1"},
        "loaded_graphics_libraries": [{"path": "/lib/libEGL.so", "sha256": "a" * 64}],
    }
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
