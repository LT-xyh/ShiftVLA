#!/usr/bin/env python3
"""Pure contract helpers for the M1-N0-R1 renderer admission layer.

This module deliberately performs no renderer, simulator, environment, Torch,
processor, or policy import.  Runtime probing is implemented in later tasks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping, MutableMapping

import yaml


class AdmissionError(RuntimeError):
    """The frozen renderer-admission contract was violated."""


class ProvenanceError(AdmissionError):
    """Immutable predecessor evidence no longer matches its manifest."""


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def config_contract_sha256(config: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(config))
    value.pop("config_sha256", None)
    return _canonical_sha256(value)


def verify_config_self_hash(config: Mapping[str, Any]) -> bool:
    claimed = config.get("config_sha256")
    return isinstance(claimed, str) and claimed == config_contract_sha256(config)


def load_document(path: Path | str) -> Any:
    source = Path(path)
    try:
        mode = source.lstat().st_mode
    except FileNotFoundError as error:
        raise ProvenanceError(f"document does not exist: {source}") from error
    if source.is_symlink() or not stat.S_ISREG(mode):
        raise ProvenanceError(f"document must be a non-symlink regular file: {source}")
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def _publish_bytes_no_overwrite(path: Path | str, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json_no_overwrite(path: Path | str, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _publish_bytes_no_overwrite(path, payload)


def _regular_tree(root: Path, repo_root: Path) -> list[dict[str, object]]:
    if not root.is_dir():
        raise ProvenanceError(f"predecessor root is not a directory: {root}")
    files = []
    pending = [root]
    paths: list[Path] = []
    while pending:
        directory = pending.pop()
        for entry in os.scandir(directory):
            path = Path(entry.path)
            if entry.is_symlink():
                raise ProvenanceError(f"unexpected symlink in predecessor tree: {path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False):
                paths.append(path)
            else:
                raise ProvenanceError(f"unexpected non-regular predecessor entry: {path}")
    for path in sorted(paths):
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "size": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    return files


def verify_exact_regular_tree(
    root: Path | str, expected: list[Mapping[str, Any]], *, repo_root: Path | str
) -> None:
    tree_root = Path(root)
    try:
        root_mode = tree_root.lstat().st_mode
    except FileNotFoundError as error:
        raise ProvenanceError(f"exact regular tree root does not exist: {tree_root}") from error
    if tree_root.is_symlink() or not stat.S_ISDIR(root_mode):
        raise ProvenanceError(f"exact regular tree root must be a non-symlink directory: {tree_root}")
    for entry in os.scandir(tree_root):
        if entry.is_dir(follow_symlinks=False):
            raise ProvenanceError(f"exact regular tree contains unexpected directory: {entry.path}")
    actual = [{**item, "type": "file"} for item in _regular_tree(Path(root), Path(repo_root).resolve())]
    normalized = [dict(item) for item in expected]
    if actual != normalized:
        raise ProvenanceError("exact regular tree differs from its frozen manifest")


def verify_empty_directory(path: Path | str) -> None:
    directory = Path(path)
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError as error:
        raise ProvenanceError(f"empty directory does not exist: {directory}") from error
    if directory.is_symlink() or not stat.S_ISDIR(mode):
        raise ProvenanceError(f"empty directory must be a non-symlink directory: {directory}")
    if next(os.scandir(directory), None) is not None:
        raise ProvenanceError(f"directory must be empty: {directory}")


def predecessor_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(manifest))
    value.pop("manifest_sha256", None)
    return _canonical_sha256(value)


def build_predecessor_manifest(
    root: Path | str,
    *,
    repo_root: Path | str,
    semantic_hashes: Mapping[str, str],
    terminal_manifest_raw_sha256: str | None = None,
    anchors: list[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    repository = Path(repo_root).resolve()
    predecessor = Path(root).resolve()
    try:
        relative_root = predecessor.relative_to(repository).as_posix()
    except ValueError as error:
        raise ProvenanceError("predecessor root must be inside repo_root") from error
    files = _regular_tree(predecessor, repository)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "predecessor_tree": {
            "root": relative_root,
            "regular_file_count": len(files),
            "files": files,
        },
        "semantic_hashes": dict(semantic_hashes),
        "anchors": [dict(item) for item in (anchors or [])],
    }
    if terminal_manifest_raw_sha256 is not None:
        manifest["terminal_manifest_raw_sha256"] = terminal_manifest_raw_sha256
    manifest["manifest_sha256"] = predecessor_manifest_sha256(manifest)
    return manifest


_PREDECESSOR_SEMANTIC_HASHES = {
    "null_config_contract_sha256": "abbda1ee7d347a3c6cafad9f6f79a41a4227b84d063f2eea8ae7d53c2c5fae2b",
    "state_replay_contract_sha256": "730aff4a41fd91fb837102ca5f363a4a142bf7010140f5176940700f1d1fd5f0",
    "run_spec_sha256": "88202837c4dd7c60f0dd968e90d313e8c85a7da479b83ca85f0c32b2685c922c",
    "initial_pair_registry_sha256": "761ce35bb1416e50cbcfc13e7ec8dc1223a2f245a261ddc8c5ac57c06c07f6b9",
    "source_registry_sha256": "092c64910b7530c6d84a98a925124879e9a96196b54d45b284a70ccfb62dd49c",
    "action_bytes_sha256": "c17bc44ad8195fecb42a80b3b272828761a9df6d88dd2bafe45db01a6cb04bbf",
    "final_pair_registry_sha256": "c644ef02171bba0645811e0370a32c947e3065f88adda690c1b1b9fb3066888c",
    "terminal_manifest_sha256": "461c31387aabf0a2bb8db90828fffc50d9181e09e207ab311678facc3df6cd3f",
}
_PREDECESSOR_ANCHOR_PATHS = [
    "configs/m1/null_calibration.yaml",
    "configs/m1/state_replay.yaml",
    "runtime/m1/m1_hard_gate_tape_registry.json",
    "runtime/m1/tapes/libero_spatial-task000-init000.actions.npy",
    "runs/m1_null_calibration/20260901_task000_init000_null20/run_spec.json",
    "runs/m1_null_calibration/20260901_task000_init000_null20/pair_registry.json",
    "runs/m1_null_calibration/20260901_task000_init000_null20/final_pair_registry.json",
    "runs/m1_null_calibration/20260901_task000_init000_null20/terminal_manifest.json",
]


def _document_self_hash(path: Path, field: str) -> str:
    value = load_document(path)
    if not isinstance(value, dict):
        raise ProvenanceError(f"semantic anchor must be a mapping: {path}")
    payload = dict(value)
    payload.pop(field, None)
    return _canonical_sha256(payload)


def _recompute_predecessor_semantic_hashes(repository: Path) -> dict[str, str]:
    import numpy as np

    tape = np.load(repository / _PREDECESSOR_ANCHOR_PATHS[3], allow_pickle=False)
    if tape.dtype != np.dtype("float32") or tuple(tape.shape) != (82, 7) or not tape.flags.c_contiguous:
        raise ProvenanceError("action semantic anchor is not contiguous float32 shape (82, 7)")
    state_replay = load_document(repository / _PREDECESSOR_ANCHOR_PATHS[1])
    state_paths = dict(state_replay["paths"])
    state_paths.pop("config_sha256", None)
    state_replay = {**state_replay, "paths": state_paths}
    return {
        "null_config_contract_sha256": _document_self_hash(repository / _PREDECESSOR_ANCHOR_PATHS[0], "config_sha256"),
        "state_replay_contract_sha256": _canonical_sha256(state_replay),
        "source_registry_sha256": _document_self_hash(repository / _PREDECESSOR_ANCHOR_PATHS[2], "registry_sha256"),
        "action_bytes_sha256": _sha256_bytes(tape.tobytes(order="C")),
        "run_spec_sha256": _document_self_hash(repository / _PREDECESSOR_ANCHOR_PATHS[4], "run_spec_sha256"),
        "initial_pair_registry_sha256": _document_self_hash(repository / _PREDECESSOR_ANCHOR_PATHS[5], "pair_registry_sha256"),
        "final_pair_registry_sha256": _document_self_hash(repository / _PREDECESSOR_ANCHOR_PATHS[6], "final_pair_registry_sha256"),
        "terminal_manifest_sha256": _document_self_hash(repository / _PREDECESSOR_ANCHOR_PATHS[7], "terminal_manifest_sha256"),
    }


def verify_predecessor_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path | str
) -> bool:
    value = copy.deepcopy(dict(manifest))
    claimed_self_hash = value.pop("manifest_sha256", None)
    if claimed_self_hash != predecessor_manifest_sha256(manifest):
        raise ProvenanceError("predecessor manifest self hash mismatch")
    tree = value.get("predecessor_tree")
    if not isinstance(tree, dict) or set(tree) != {"root", "regular_file_count", "files"}:
        raise ProvenanceError("invalid predecessor tree schema")
    repository = Path(repo_root).resolve()
    unresolved_root = repository / str(tree["root"])
    if unresolved_root.is_symlink():
        raise ProvenanceError("predecessor root must not be a symlink")
    root = unresolved_root.resolve()
    try:
        root.relative_to(repository)
    except ValueError as error:
        raise ProvenanceError("predecessor root escapes repo_root") from error
    actual = _regular_tree(root, repository)
    if tree["regular_file_count"] != len(actual) or tree["files"] != actual:
        raise ProvenanceError("predecessor path, size, or raw hash mismatch")
    semantic = value.get("semantic_hashes")
    if not isinstance(semantic, dict) or not semantic:
        raise ProvenanceError("semantic hashes are missing")
    hash_values = list(semantic.values())
    raw_terminal = value.get("terminal_manifest_raw_sha256")
    if raw_terminal is not None:
        hash_values.append(raw_terminal)
    if any(not isinstance(item, str) or len(item) != 64 for item in hash_values):
        raise ProvenanceError("semantic hash schema is invalid")
    if tree["root"] == "runs/m1_null_calibration/20260901_task000_init000_null20":
        expected_manifest_fields = {
            "schema_version", "predecessor_tree", "semantic_hashes", "anchors",
            "terminal_manifest_raw_sha256",
        }
        if set(value) != expected_manifest_fields or value["schema_version"] != 1:
            raise ProvenanceError("predecessor manifest exact schema mismatch")
        if semantic != _PREDECESSOR_SEMANTIC_HASHES:
            raise ProvenanceError("predecessor semantic hashes differ from the frozen exact values")
        recomputed = _recompute_predecessor_semantic_hashes(repository)
        if recomputed != _PREDECESSOR_SEMANTIC_HASHES:
            raise ProvenanceError("predecessor semantic anchors failed recomputation")
        anchors = value.get("anchors")
        expected_anchors = []
        for relative in _PREDECESSOR_ANCHOR_PATHS:
            path = repository / relative
            expected_anchors.append({"path": relative, "size": path.stat().st_size, "sha256": _raw_file_sha256(path)})
        if anchors != expected_anchors:
            raise ProvenanceError("predecessor raw anchor path, size, or hash mismatch")
        terminal_raw = _raw_file_sha256(repository / _PREDECESSOR_ANCHOR_PATHS[-1])
        if value.get("terminal_manifest_raw_sha256") != terminal_raw:
            raise ProvenanceError("terminal manifest raw SHA-256 mismatch")
    return True


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "config_sha256",
    "evidence_role",
    "included_in_null_calibration_evidence",
    "policy_compute_device",
    "physical_compute_device_id",
    "renderer_backend",
    "renderer_device_id",
    "renderer_identity",
    "probe_contract",
    "task_binding",
    "obs_type",
    "runtime",
    "predecessor",
    "paths",
    "governed_environment",
    "operations",
    "pass_contract",
    "publication",
}


_PROBE_CONTRACT_SOURCE_FILES = {
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
}
_PROBE_CONTRACT = {
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
    "source_files": _PROBE_CONTRACT_SOURCE_FILES,
}

_UNSET_BEFORE_IMPORT = [
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
_BLOCKED_NONEMPTY = ["LD_PRELOAD", "LD_AUDIT"]
_SET_CACHE = {
    "LIBERO_CONFIG_PATH": "/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/m1/libero_config",
    "HF_HOME": "/public/home/xuyinghao/tmp/shiftvla-empty-hf-home-g",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}
_SET_RENDERER = {
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0",
}
_OPERATIONS = {
    "allowed": [
        "official_policy_free_libero_factory_construction",
        "unavoidable_inner_reset_during_construction",
        "one_public_render",
        "render_return_metadata_only",
        "close_destroy_and_exit",
    ],
    "forbidden": [
        "public_reset",
        "env_step",
        "create_or_send_actions",
        "ten_step_settle",
        "runtime_adapter_construction",
        "null_measurement",
        "replay_capture_or_restore",
        "policy_processor_or_inference",
        "success_reward_done_or_predicate_inspection",
        "frame_serialization_or_hashing",
        "retry_replacement_autoreset_or_process_reuse",
    ],
    "forbidden_dispatch": ["prepare_run", "materialize_authoritative_config"],
    "authoritative_schedule_creation": False,
}
_PASS_CONTRACT = {
    "construction_status": "PASS",
    "public_render_calls": 1,
    "render_returned": True,
    "forbidden_operation_count": 0,
    "close_status": "PASS",
    "destroy_environment": True,
    "exit_process": True,
    "failure_status": "BLOCKED",
}
_PUBLICATION = {
    "atomic": True,
    "no_overwrite": True,
    "child_timeout_seconds": 600,
    "persist_rgb_frame": False,
    "artifact_allowlist": [
        "probe-000.json", "probe-000.stdout.log", "probe-000.stderr.log",
        "probe-001.json", "probe-001.stdout.log", "probe-001.stderr.log",
        "probe-002.json", "probe-002.stdout.log", "probe-002.stderr.log",
        "preflight.json", "preflight.stdout.log", "preflight.stderr.log",
        "terminal_manifest.json",
    ],
}


def _raw_file_sha256(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ProvenanceError(f"required file does not exist: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ProvenanceError(f"required path must be a non-symlink regular file: {path}")
    return _sha256_bytes(path.read_bytes())


def _verify_file_binding(repo_root: Path, binding: Mapping[str, Any], label: str) -> None:
    _require_exact_keys(binding, {"path", "sha256"}, label)
    path = repo_root / str(binding["path"])
    if _raw_file_sha256(path) != binding["sha256"]:
        raise ProvenanceError(f"{label} raw SHA-256 mismatch")


def probe_source_bindings_sha256(source_files: Mapping[str, Any]) -> str:
    """Hash the exact source-binding mapping carried by a probe record.

    This helper intentionally hashes only path/SHA bindings.  It never reads
    a source file and therefore remains usable by a policy-free child before
    any renderer import.
    """

    if not isinstance(source_files, Mapping) or not source_files:
        raise AdmissionError("probe source-binding schema mismatch")
    normalized: dict[str, dict[str, str]] = {}
    for name in sorted(source_files):
        binding = source_files[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise AdmissionError(f"probe source binding {name} schema mismatch")
        path = binding["path"]
        digest = binding["sha256"]
        if not isinstance(path, str) or not path:
            raise AdmissionError(f"probe source binding {name} path is invalid")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise AdmissionError(f"probe source binding {name} SHA-256 is invalid")
        normalized[name] = {"path": path, "sha256": digest}
    return _canonical_sha256(normalized)


def verify_admission_probe_contract(
    contract: Mapping[str, Any], *, repo_root: Path | str | None = None
) -> bool:
    """Validate the frozen three-child renderer admission contract.

    ``repo_root`` is optional for pure tests and is required when the caller
    wants the pinned source bytes checked as well.  In either mode the
    complete nested schema is exact and cannot silently grow extra fields.
    """

    if not isinstance(contract, Mapping):
        raise AdmissionError("probe contract must be a mapping")
    expected_keys = {
        "schema_version",
        "probe_count",
        "expected_device_count",
        "selected_ordinal",
        "required_selected_extensions",
        "expected_egl_identity",
        "expected_gl_identity",
        "source_files",
    }
    if set(contract) != expected_keys:
        raise AdmissionError("probe contract schema mismatch")
    integer_fields = ("schema_version", "probe_count", "expected_device_count", "selected_ordinal")
    if any(type(contract[name]) is not int for name in integer_fields):
        raise AdmissionError("probe contract integer field type mismatch")
    if contract["schema_version"] != 1 or contract["probe_count"] != 3:
        raise AdmissionError("probe contract version or probe count mismatch")
    if contract["expected_device_count"] != 1 or contract["selected_ordinal"] != 0:
        raise AdmissionError("probe contract selected device mismatch")
    if contract["required_selected_extensions"] != ["EGL_MESA_device_software"]:
        raise AdmissionError("probe contract required selected extension mismatch")
    expected_egl = {"vendor": "Mesa Project", "version": "1.5"}
    expected_gl = {
        "vendor": "Mesa/X.org",
        "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
        "version": "3.1 Mesa 21.1.5",
    }
    if contract["expected_egl_identity"] != expected_egl:
        raise AdmissionError("probe contract EGL identity mismatch")
    if contract["expected_gl_identity"] != expected_gl:
        raise AdmissionError("probe contract GL identity mismatch")
    if contract["source_files"] != _PROBE_CONTRACT_SOURCE_FILES:
        raise AdmissionError("probe contract source-binding values mismatch")
    source_files = contract["source_files"]
    probe_source_bindings_sha256(source_files)
    if repo_root is not None:
        repository = Path(repo_root).resolve()
        for name in sorted(source_files):
            _verify_file_binding(repository, source_files[name], f"probe source file {name}")
    return True


def load_admission_config(path: Path | str) -> dict[str, Any]:
    value = load_document(path)
    if not isinstance(value, dict):
        raise AdmissionError("admission config must be a mapping")
    return value


def _require_exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise AdmissionError(f"{label} schema mismatch")
    return value


def path_within(path: Path | str, parent: Path | str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def admission_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    return {
        "admission_root": Path(paths["admission_root"]).resolve(),
        "authoritative_null_root": Path(paths["authoritative_null_root"]).resolve(),
    }


def _reject_existing_symlink_components(path: Path, stop: Path) -> None:
    current = path
    while True:
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise AdmissionError(f"publication path contains existing symlink: {current}")
        if current == stop or current.parent == current:
            return
        current = current.parent


def resolve_admission_artifact(
    config: Mapping[str, Any], artifact_name: str, *, repo_root: Path | str
) -> Path:
    publication = config.get("publication")
    if not isinstance(publication, dict) or publication.get("artifact_allowlist") != _PUBLICATION["artifact_allowlist"]:
        raise AdmissionError("publication artifact allowlist mismatch")
    if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name or artifact_name not in publication["artifact_allowlist"]:
        raise AdmissionError("artifact is not an allowed direct child")
    repository = Path(repo_root).resolve()
    raw_paths = config.get("paths")
    if not isinstance(raw_paths, dict):
        raise AdmissionError("publication paths are missing")
    admission_root = Path(str(raw_paths.get("admission_root", "")))
    authoritative_root = Path(str(raw_paths.get("authoritative_null_root", "")))
    _reject_existing_symlink_components(admission_root, repository)
    _reject_existing_symlink_components(authoritative_root, repository)
    resolved_root = admission_root.resolve()
    resolved_authoritative = authoritative_root.resolve()
    target = admission_root / artifact_name
    if target.resolve() != resolved_root / artifact_name:
        raise AdmissionError("resolved publication target differs from the admitted direct child")
    if path_within(target, resolved_authoritative):
        raise AdmissionError("publication target enters the authoritative null root")
    if not path_within(resolved_root, repository):
        raise AdmissionError("admission root escapes repo_root")
    return target


def publish_admission_json(
    config: Mapping[str, Any], artifact_name: str, value: object, *, repo_root: Path | str
) -> Path:
    validate_admission_config(config, repo_root=repo_root)
    target = resolve_admission_artifact(config, artifact_name, repo_root=repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    _publish_json_no_overwrite(target, value)
    return target


def publish_admission_bytes(
    config: Mapping[str, Any], artifact_name: str, payload: bytes, *, repo_root: Path | str
) -> Path:
    validate_admission_config(config, repo_root=repo_root)
    target = resolve_admission_artifact(config, artifact_name, repo_root=repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes_no_overwrite(target, payload)
    return target


def validate_admission_config(config: Mapping[str, Any], *, repo_root: Path | str) -> None:
    _require_exact_keys(config, _TOP_LEVEL_FIELDS, "admission config")
    if config["schema_version"] != 1 or config["name"] != "m1_n0_r1_renderer_preflight":
        raise AdmissionError("admission config identity mismatch")
    if not verify_config_self_hash(config):
        raise AdmissionError("admission config self hash mismatch")
    expected_namespace = {
        "policy_compute_device": "not_applicable_no_policy",
        "physical_compute_device_id": None,
        "renderer_backend": "egl",
        "renderer_device_id": "0",
    }
    if any(config[key] != expected for key, expected in expected_namespace.items()):
        raise AdmissionError("renderer namespace mismatch")
    if config["renderer_identity"] != {
        "vendor": "Mesa/X.org",
        "renderer": "llvmpipe (LLVM 12.0.0, 256 bits)",
        "version": "3.1 Mesa 21.1.5",
    }:
        raise AdmissionError("renderer identity mismatch")
    verify_admission_probe_contract(config["probe_contract"], repo_root=repo_root)
    if config["task_binding"] != {
        "suite": "libero_spatial",
        "task_id": 0,
        "init_state_id": 0,
        "seed": 2027,
    } or config["obs_type"] != "pixels_agent_pos":
        raise AdmissionError("preflight provenance binding mismatch")
    if config["evidence_role"] != "infrastructure_admission_only" or config["included_in_null_calibration_evidence"] is not False:
        raise AdmissionError("admission evidence role mismatch")
    repository = Path(repo_root).resolve()
    runtime = _require_exact_keys(
        config["runtime"],
        {"python_executable", "runtime_lock", "libero_config_tree", "source_checkouts", "source_files"},
        "runtime",
    )
    if runtime["python_executable"] != "/public/home/xuyinghao/tmp/shiftvla-libero/bin/python":
        raise AdmissionError("python executable mismatch")
    expected_lock = {
        "path": "runtime/locks/shiftvla-libero-runtime.txt",
        "sha256": "921ad0d14240e56cbd9297db152f90e167a8d85e690d2010aca6a31348e6a0fc",
    }
    if runtime["runtime_lock"] != expected_lock:
        raise AdmissionError("runtime lock binding mismatch")
    _verify_file_binding(repository, expected_lock, "runtime lock")
    expected_tree = {
        "root": "runtime/m1/libero_config",
        "entries": [{
            "path": "runtime/m1/libero_config/config.yaml",
            "type": "file",
            "size": 502,
            "sha256": "3794964a45a33c0d545118d3c44064bae497931eadd14cb26974008cdd3b55af",
        }],
    }
    if runtime["libero_config_tree"] != expected_tree:
        raise AdmissionError("LIBERO config tree binding mismatch")
    verify_exact_regular_tree(
        repository / expected_tree["root"], expected_tree["entries"], repo_root=repository
    )
    expected_checkouts = {
        "libero": {"path": "external/hf-libero", "git_sha": "8561c60eea2fb93096146f240194649df73d8b1e"},
        "lerobot": {"path": "external/lerobot", "git_sha": "7e241bd630a3719a56157a497ce5d08f244784f1"},
        "robosuite": {"path": "external/robosuite", "git_sha": "fbee5844ff5632f5b5698e204ec5357ca50be0df"},
        "mujoco": {"path": "external/mujoco", "git_sha": "72cb2b210da666617924de709406d6aadbe60c71"},
    }
    if runtime["source_checkouts"] != expected_checkouts:
        raise AdmissionError("source checkout schema mismatch")
    for name, binding in expected_checkouts.items():
        result = subprocess.run(
            ["git", "-C", str(repository / binding["path"]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != binding["git_sha"]:
            raise ProvenanceError(f"{name} checkout SHA mismatch")
    expected_sources = {
        "dcu_preflight": {"path": "scripts/dcu_preflight.py", "sha256": "af2e67fe9cca570694cc5ff84c49cbefffe722cda114f6867792541366731453"},
        "m1_state_replay": {"path": "scripts/m1_state_replay.py", "sha256": "9f3378937f605d7e6216236536f3e7ff038b156ac641044c2623ce9418e0888a"},
        "libero_env_wrapper": {"path": "external/hf-libero/libero/libero/envs/env_wrapper.py", "sha256": "a782fb76c9792268d28979474fe72849e1e98ada49c8e997a65359a8d6b6acd0"},
        "lerobot_libero": {"path": "external/lerobot/src/lerobot/envs/libero.py", "sha256": "97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71"},
        "robosuite_base": {"path": "external/robosuite/robosuite/environments/base.py", "sha256": "16a044bf28cd39623c286d72f8be2f3789d53e215d77bef3b88ca33511c87687"},
    }
    if runtime["source_files"] != expected_sources:
        raise AdmissionError("source file schema mismatch")
    for name, binding in expected_sources.items():
        _verify_file_binding(repository, binding, f"source file {name}")
    predecessor = _require_exact_keys(
        config["predecessor"],
        {"manifest_path", "manifest_raw_sha256", "manifest_self_sha256", "preserved_output_root"},
        "predecessor",
    )
    if predecessor["manifest_path"] != "runtime/m1/m1_n0_r1_predecessor_manifest.json" or predecessor["preserved_output_root"] != "runs/m1_null_calibration/20260901_task000_init000_null20":
        raise AdmissionError("predecessor path binding mismatch")
    manifest_path = repository / predecessor["manifest_path"]
    if _raw_file_sha256(manifest_path) != predecessor["manifest_raw_sha256"]:
        raise ProvenanceError("predecessor manifest raw SHA-256 mismatch")
    manifest = load_document(manifest_path)
    if manifest.get("manifest_sha256") != predecessor["manifest_self_sha256"]:
        raise ProvenanceError("predecessor manifest self SHA-256 binding mismatch")
    verify_predecessor_manifest(manifest, repo_root=repository)
    if manifest["predecessor_tree"]["root"] != predecessor["preserved_output_root"]:
        raise ProvenanceError("preserved predecessor root binding mismatch")
    governed = _require_exact_keys(
        config["governed_environment"],
        {"unset_before_import", "blocked_nonempty", "admitted_ld_library_path", "set_cache", "set_renderer"},
        "governed environment",
    )
    expected_ld_library_path = "/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib/criu/:/opt/hyhal/lib:/opt/dtk-25.04.2/dcc/gcvm/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk-25.04.2/llvm/lib:/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk-25.04.2/opencl/lib:/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib/criu/:/opt/hyhal/lib:/opt/dtk-25.04.2/dcc/gcvm/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk-25.04.2/llvm/lib:/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk-25.04.2/opencl/lib"
    if governed != {
        "unset_before_import": _UNSET_BEFORE_IMPORT,
        "blocked_nonempty": _BLOCKED_NONEMPTY,
        "admitted_ld_library_path": expected_ld_library_path,
        "set_cache": _SET_CACHE,
        "set_renderer": _SET_RENDERER,
    }:
        raise AdmissionError("governed environment contract mismatch")
    verify_empty_directory(Path(str(governed["set_cache"]["HF_HOME"])))
    if config["operations"] != _OPERATIONS:
        raise AdmissionError("operations contract mismatch")
    if config["pass_contract"] != _PASS_CONTRACT:
        raise AdmissionError("PASS contract mismatch")
    if config["publication"] != _PUBLICATION:
        raise AdmissionError("publication contract mismatch")
    paths = admission_paths(config)
    expected_paths = {
        "admission_root": Path("/public/home/xuyinghao/workspace/vla/ShiftVLA/runs/m1_renderer_preflight/20260908_task000_init000_egl0"),
        "authoritative_null_root": Path("/public/home/xuyinghao/workspace/vla/ShiftVLA/runs/m1_null_calibration/20260908_task000_init000_null20_egl0"),
    }
    if paths != expected_paths:
        raise AdmissionError("output path contract mismatch")
    if paths["admission_root"] == paths["authoritative_null_root"] or path_within(paths["admission_root"], paths["authoritative_null_root"]) or path_within(paths["authoritative_null_root"], paths["admission_root"]):
        raise AdmissionError("admission and authoritative null roots must be disjoint")
    if not all(path_within(path, repository) for path in paths.values()):
        raise AdmissionError("output roots must be inside repo_root")


def torch_device_from_policy_compute_device(value: str) -> str:
    if value == "not_applicable_no_policy":
        raise AdmissionError("policy_compute_device sentinel must never reach Torch")
    return value


def apply_governed_environment(
    config: Mapping[str, Any],
    *,
    environ: MutableMapping[str, str],
    event_sink: Callable[[str], None] | None = None,
    import_callback: Callable[[], None] | None = None,
) -> dict[str, object]:
    emit = event_sink or (lambda _event: None)
    governed = config["governed_environment"]
    names = list(dict.fromkeys(governed["unset_before_import"] + governed["blocked_nonempty"] + list(governed["set_cache"]) + list(governed["set_renderer"]) + ["LD_LIBRARY_PATH"]))
    presence = {name: {"present": name in environ, "nonempty": bool(environ.get(name, ""))} for name in names}
    emit("record_presence")
    for name in governed["unset_before_import"]:
        environ.pop(name, None)
    emit("unset")
    for name in governed["blocked_nonempty"]:
        if environ.get(name, ""):
            raise AdmissionError(f"nonempty {name} is forbidden")
        environ.pop(name, None)
    if environ.get("LD_LIBRARY_PATH") != governed["admitted_ld_library_path"]:
        raise AdmissionError("LD_LIBRARY_PATH differs from the admitted value")
    emit("block_loader")
    environ.update({str(key): str(value) for key, value in governed["set_cache"].items()})
    emit("set_cache")
    environ.update({str(key): str(value) for key, value in governed["set_renderer"].items()})
    emit("set_renderer")
    final = {name: environ.get(name) for name in names}
    expected = {**governed["set_cache"], **governed["set_renderer"]}
    if any(final.get(key) != value for key, value in expected.items()):
        raise AdmissionError("final governed environment mapping mismatch")
    if any(name in environ for name in governed["unset_before_import"] if name not in expected):
        raise AdmissionError("an unset-before-import variable remains present")
    if any(environ.get(name, "") for name in governed["blocked_nonempty"]):
        raise AdmissionError("a blocked loader variable is nonempty")
    emit("verify_final_mapping")
    if import_callback is not None:
        emit("renderer_import")
        import_callback()
    return {"initial_presence": presence, "final_environment": final}


def _validate_worker_preimport_fingerprint(value: Mapping[str, Any]) -> None:
    expected_top = {
        "schema_version",
        "device_count",
        "selected_ordinal",
        "ordered_descriptors",
        "supported_drm_nodes",
        "namespace_facts",
        "governed_environment",
        "static_graphics_libraries",
    }
    if set(value) != expected_top:
        raise AdmissionError("worker preimport fingerprint schema forbids context, GL, loaded, or other extra fields")
    if value["schema_version"] != 1:
        raise AdmissionError("worker preimport fingerprint schema version mismatch")
    device_count = value["device_count"]
    selected_ordinal = value["selected_ordinal"]
    if type(device_count) is not int or device_count <= 0 or type(selected_ordinal) is not int:
        raise AdmissionError("worker preimport fingerprint scalar schema mismatch")
    if not 0 <= selected_ordinal < device_count:
        raise AdmissionError("worker selected ordinal is outside the enumerated range")
    descriptors = value["ordered_descriptors"]
    descriptor_keys = {"ordinal", "extensions", "drm_device_file", "drm_render_node_file"}
    if not isinstance(descriptors, list) or len(descriptors) != device_count:
        raise AdmissionError("worker descriptor/device-count schema mismatch")
    descriptor_drm_nodes: set[str] = set()
    for ordinal, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict) or set(descriptor) != descriptor_keys:
            raise AdmissionError("worker descriptor schema contains an unexpected field")
        if type(descriptor["ordinal"]) is not int or descriptor["ordinal"] != ordinal:
            raise AdmissionError("worker descriptor ordinals are not exact enumeration order")
        extensions = descriptor["extensions"]
        if not isinstance(extensions, list) or any(not isinstance(item, str) or not item for item in extensions) or extensions != sorted(set(extensions)):
            raise AdmissionError("worker descriptor extensions must be sorted unique strings")
        if (
            descriptor["drm_device_file"] is not None
            and "EGL_EXT_device_drm" not in extensions
        ):
            raise AdmissionError("worker DRM device path requires EGL_EXT_device_drm extension support")
        if (
            descriptor["drm_render_node_file"] is not None
            and "EGL_EXT_device_drm_render_node" not in extensions
        ):
            raise AdmissionError(
                "worker DRM render-node path requires EGL_EXT_device_drm_render_node extension support"
            )
        for field in ("drm_device_file", "drm_render_node_file"):
            path = descriptor[field]
            if path is not None and (not isinstance(path, str) or not path.startswith("/dev/dri/") or not Path(path).is_absolute()):
                raise AdmissionError("worker descriptor DRM path must be null or an absolute /dev/dri path")
            if path is not None:
                descriptor_drm_nodes.add(path)
    supported_nodes = value["supported_drm_nodes"]
    if not isinstance(supported_nodes, list) or any(
        not isinstance(path, str) or not path.startswith("/dev/dri/") or not Path(path).is_absolute()
        for path in supported_nodes
    ) or supported_nodes != sorted(set(supported_nodes)):
        raise AdmissionError("worker DRM-node schema mismatch")
    if supported_nodes != sorted(descriptor_drm_nodes):
        raise AdmissionError(
            "worker supported DRM nodes must equal the exact descriptor DRM-node union"
        )
    namespace = value["namespace_facts"]
    if not isinstance(namespace, dict) or set(namespace) != {"dev_dri", "mount_namespace", "cgroup", "device_nodes"}:
        raise AdmissionError("worker namespace-facts schema contains an unexpected field")
    if not isinstance(namespace["mount_namespace"], str) or not namespace["mount_namespace"]:
        raise AdmissionError("worker mount namespace must be a nonempty string")
    if not isinstance(namespace["cgroup"], list) or any(not isinstance(item, str) for item in namespace["cgroup"]):
        raise AdmissionError("worker cgroup schema must be a list of strings")

    def validate_device_records(records: object, label: str) -> None:
        record_keys = {"path", "type", "major", "minor", "mode"}
        if not isinstance(records, list):
            raise AdmissionError(f"worker {label} must be a list")
        paths: list[str] = []
        for record in records:
            if not isinstance(record, dict) or set(record) != record_keys:
                raise AdmissionError(f"worker {label} device record schema contains an unexpected field")
            path = record["path"]
            if not isinstance(path, str) or not path.startswith("/dev/dri/") or not Path(path).is_absolute():
                raise AdmissionError(f"worker {label} device path must be absolute under /dev/dri")
            if not isinstance(record["type"], str) or not record["type"]:
                raise AdmissionError(f"worker {label} device type must be a nonempty string")
            if any(type(record[field]) is not int or record[field] < 0 for field in ("major", "minor", "mode")):
                raise AdmissionError(f"worker {label} device number and mode types are invalid")
            paths.append(path)
        if paths != sorted(set(paths)):
            raise AdmissionError(f"worker {label} device records must be sorted and unique")

    validate_device_records(namespace["dev_dri"], "dev_dri")
    validate_device_records(namespace["device_nodes"], "device_nodes")
    libraries = value["static_graphics_libraries"]
    library_keys = {"path", "sha256", "build_id", "package_version"}
    if not isinstance(libraries, list):
        raise AdmissionError("worker static-library schema contains an unexpected field")
    library_paths: list[str] = []
    for library in libraries:
        if not isinstance(library, dict) or set(library) != library_keys:
            raise AdmissionError("worker static-library schema contains an unexpected field")
        path = library["path"]
        digest = library["sha256"]
        build_id = library["build_id"]
        package_version = library["package_version"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise AdmissionError("worker static-library path must be absolute")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise AdmissionError("worker static-library SHA-256 must be lowercase hexadecimal")
        if build_id is not None and (not isinstance(build_id, str) or not build_id):
            raise AdmissionError("worker static-library build ID must be null or nonempty string")
        if package_version is not None and not isinstance(package_version, str):
            raise AdmissionError("worker static-library package version must be null or string")
        library_paths.append(path)
    if library_paths != sorted(set(library_paths)):
        raise AdmissionError("worker static libraries must be a canonical sorted unique set")
    allowed_environment_names = set(_UNSET_BEFORE_IMPORT) | set(_BLOCKED_NONEMPTY) | set(_SET_CACHE) | set(_SET_RENDERER) | {"LD_LIBRARY_PATH"}
    environment = value["governed_environment"]
    if not isinstance(environment, dict) or set(environment) != allowed_environment_names:
        raise AdmissionError("worker governed-environment schema contains an unexpected field")
    if any(item is not None and not isinstance(item, str) for item in environment.values()):
        raise AdmissionError("worker governed-environment values must be null or strings")
    return None


def worker_preimport_namespace_fingerprint_sha256(value: Mapping[str, Any]) -> str:
    _validate_worker_preimport_fingerprint(value)
    return _canonical_sha256(value)


def query_egl_devices_two_call(backend: Any) -> list[Any]:
    """Return every EGL device using the checked count-then-fill contract."""

    count = [0]
    first_result = backend.eglQueryDevicesEXT(0, None, count)
    first_error = backend.eglGetError()
    if first_result != backend.EGL_TRUE or first_error != backend.EGL_SUCCESS:
        raise AdmissionError(
            "first eglQueryDevicesEXT call failed "
            f"(result={first_result!r}, EGL error={first_error!r})"
        )
    requested_count = count[0]
    if type(requested_count) is not int or requested_count <= 0:
        raise AdmissionError("eglQueryDevicesEXT reported a non-positive device count")

    devices: list[Any] = [None] * requested_count
    reported = [0]
    second_result = backend.eglQueryDevicesEXT(requested_count, devices, reported)
    second_error = backend.eglGetError()
    if second_result != backend.EGL_TRUE or second_error != backend.EGL_SUCCESS:
        raise AdmissionError(
            "second eglQueryDevicesEXT call failed "
            f"(result={second_result!r}, EGL error={second_error!r})"
        )
    if type(reported[0]) is not int or reported[0] != requested_count:
        raise AdmissionError(
            "eglQueryDevicesEXT device count changed between count and fill calls"
        )
    if any(device is None for device in devices):
        raise AdmissionError("eglQueryDevicesEXT did not fill the exact device buffer")
    is_no_device = getattr(backend, "is_no_device", None)
    if is_no_device is None:
        is_no_device = lambda device: device is None
    elif not callable(is_no_device):
        raise AdmissionError("EGL backend is_no_device predicate is not callable")
    try:
        contains_no_device = any(bool(is_no_device(device)) for device in devices)
    except Exception as error:
        raise AdmissionError("EGL backend is_no_device predicate failed") from error
    if contains_no_device:
        raise AdmissionError("eglQueryDevicesEXT returned the EGL_NO_DEVICE_EXT sentinel")
    return devices


def describe_egl_devices(backend: Any, devices: list[Any]) -> list[dict[str, object]]:
    """Describe EGL devices in enumeration order without retaining handles."""

    descriptions: list[dict[str, object]] = []
    for ordinal, device in enumerate(devices):
        raw_extensions = backend.query_device_extensions(device)
        if isinstance(raw_extensions, str) or not isinstance(raw_extensions, (list, tuple)):
            raise AdmissionError("EGL device extensions must be a sequence of strings")
        if any(not isinstance(extension, str) or not extension for extension in raw_extensions):
            raise AdmissionError("EGL device extensions must be nonempty strings")
        extensions = sorted(set(raw_extensions))
        drm_device_file = None
        drm_render_node_file = None
        if "EGL_EXT_device_drm" in extensions:
            drm_device_file = backend.query_device_drm_node(device, "drm_device_file")
        if "EGL_EXT_device_drm_render_node" in extensions:
            drm_render_node_file = backend.query_device_drm_node(
                device, "drm_render_node_file"
            )
        description = {
            "ordinal": ordinal,
            "extensions": extensions,
            "drm_device_file": drm_device_file,
            "drm_render_node_file": drm_render_node_file,
        }
        for field in ("drm_device_file", "drm_render_node_file"):
            value = description[field]
            if value is not None and not isinstance(value, str):
                raise AdmissionError("EGL DRM descriptors must be scalar paths, never pointers")
        descriptions.append(description)
    return descriptions


def _normalize_path_record_set(records: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise AdmissionError(f"{label} must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("path"), str):
            raise AdmissionError(f"{label} records must contain scalar paths")
        record = copy.deepcopy(dict(raw))
        path = record["path"]
        previous = by_path.get(path)
        if previous is not None and previous != record:
            raise AdmissionError(f"{label} has a duplicate path conflict: {path}")
        by_path[path] = record
    return [by_path[path] for path in sorted(by_path)]


def _normalize_library_set(libraries: object, *, label: str) -> list[dict[str, Any]]:
    return _normalize_path_record_set(libraries, label=label)


def _validate_graphics_library_records(libraries: object, *, label: str) -> None:
    """Validate a canonical list of stable file metadata without addresses."""

    normalized = _normalize_library_set(libraries, label=label)
    if libraries != normalized:
        raise AdmissionError(f"{label} must be a canonical sorted unique set")
    expected_keys = {"path", "sha256", "build_id", "package_version"}
    for library in normalized:
        if set(library) != expected_keys:
            raise AdmissionError(f"{label} schema contains an unexpected field")
        path = library["path"]
        digest = library["sha256"]
        build_id = library["build_id"]
        package_version = library["package_version"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise AdmissionError(f"{label} path must be absolute")
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise AdmissionError(f"{label} SHA-256 must be lowercase hexadecimal")
        if build_id is not None and (not isinstance(build_id, str) or not build_id):
            raise AdmissionError(f"{label} build ID must be null or nonempty string")
        if package_version is not None and not isinstance(package_version, str):
            raise AdmissionError(f"{label} package version must be null or string")


def build_worker_preimport_fingerprint(
    *,
    device_count: int,
    selected_ordinal: int,
    ordered_descriptors: list[Mapping[str, Any]],
    namespace_facts: Mapping[str, Any],
    governed_environment: Mapping[str, Any],
    static_graphics_libraries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the canonical, context-free worker fingerprint."""

    descriptors: list[dict[str, Any]] = []
    drm_nodes: set[str] = set()
    for descriptor in ordered_descriptors:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "ordinal", "extensions", "drm_device_file", "drm_render_node_file"
        }:
            raise AdmissionError("worker descriptor schema contains an unexpected field")
        raw_extensions = descriptor["extensions"]
        if isinstance(raw_extensions, str) or not isinstance(raw_extensions, (list, tuple)):
            raise AdmissionError("worker descriptor extensions must be a sequence")
        if any(not isinstance(extension, str) or not extension for extension in raw_extensions):
            raise AdmissionError("worker descriptor extensions must be nonempty scalar strings")
        normalized = copy.deepcopy(dict(descriptor))
        normalized["extensions"] = sorted(set(raw_extensions))
        for field in ("drm_device_file", "drm_render_node_file"):
            path = normalized[field]
            if path is not None:
                if not isinstance(path, str):
                    raise AdmissionError("worker DRM path must be a scalar path, never a pointer")
                drm_nodes.add(path)
        descriptors.append(normalized)

    namespace = copy.deepcopy(dict(namespace_facts))
    if set(namespace) == {"dev_dri", "mount_namespace", "cgroup", "device_nodes"}:
        namespace["dev_dri"] = _normalize_path_record_set(namespace["dev_dri"], label="dev_dri")
        namespace["device_nodes"] = _normalize_path_record_set(
            namespace["device_nodes"], label="device_nodes"
        )
    fingerprint = {
        "schema_version": 1,
        "device_count": device_count,
        "selected_ordinal": selected_ordinal,
        "ordered_descriptors": descriptors,
        "supported_drm_nodes": sorted(drm_nodes),
        "namespace_facts": namespace,
        "governed_environment": copy.deepcopy(dict(governed_environment)),
        "static_graphics_libraries": _normalize_library_set(
            static_graphics_libraries, label="static graphics libraries"
        ),
    }
    _validate_worker_preimport_fingerprint(fingerprint)
    return fingerprint


def _validate_identity(value: object, *, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise AdmissionError(f"full renderer {label} schema contains an unexpected field")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise AdmissionError(f"full renderer {label} values must be nonempty scalar strings")


def _validate_full_renderer_fingerprint(value: Mapping[str, Any]) -> None:
    worker_keys = {
        "schema_version", "device_count", "selected_ordinal", "ordered_descriptors",
        "supported_drm_nodes", "namespace_facts", "governed_environment",
        "static_graphics_libraries",
    }
    expected = worker_keys | {"egl_identity", "gl_identity", "loaded_graphics_libraries"}
    if set(value) != expected:
        raise AdmissionError(
            "full renderer fingerprint schema forbids volatile, pointer, or other extra fields"
        )
    _validate_worker_preimport_fingerprint({key: value[key] for key in worker_keys})
    _validate_identity(value["egl_identity"], keys={"vendor", "version"}, label="EGL identity")
    _validate_identity(
        value["gl_identity"], keys={"vendor", "renderer", "version"}, label="GL identity"
    )
    normalized_loaded = _normalize_library_set(
        value["loaded_graphics_libraries"], label="loaded graphics libraries"
    )
    if value["loaded_graphics_libraries"] != normalized_loaded:
        raise AdmissionError("loaded graphics libraries must be a canonical sorted unique set")
    library_check = dict(value)
    library_check["static_graphics_libraries"] = normalized_loaded
    _validate_worker_preimport_fingerprint({key: library_check[key] for key in worker_keys})


def build_full_renderer_fingerprint(
    preimport: Mapping[str, Any],
    egl_identity: Mapping[str, Any],
    gl_identity: Mapping[str, Any],
    loaded_libs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extend a validated pre-import fingerprint with context-only identity."""

    _validate_worker_preimport_fingerprint(preimport)
    fingerprint = copy.deepcopy(dict(preimport))
    fingerprint.update(
        {
            "egl_identity": copy.deepcopy(dict(egl_identity)),
            "gl_identity": copy.deepcopy(dict(gl_identity)),
            "loaded_graphics_libraries": _normalize_library_set(
                loaded_libs, label="loaded graphics libraries"
            ),
        }
    )
    _validate_full_renderer_fingerprint(fingerprint)
    return fingerprint


def admission_full_renderer_fingerprint_sha256(value: Mapping[str, Any]) -> str:
    _validate_full_renderer_fingerprint(value)
    return _canonical_sha256(value)


_PROBE_RECORD_FIELDS = {
    "schema_version",
    "index",
    "pid",
    "ppid",
    "evidence_role",
    "included_in_null_calibration_evidence",
    "config_sha256",
    "runtime_lock_sha256",
    "probe_source_bindings_sha256",
    "execution_commit",
    "device_count",
    "selected_ordinal",
    "required_selected_extensions",
    "ordered_descriptors",
    "worker_preimport_fingerprint",
    "worker_preimport_fingerprint_sha256",
    "full_renderer_fingerprint",
    "full_renderer_fingerprint_sha256",
    "context_records",
    "environment_transition_audit",
    "exit_code",
    "retry_count",
    "fallback_used",
    "import_negative",
    "cleanup",
}
_CONTEXT_RECORD_FIELDS = {
    "ordinal",
    "egl_identity",
    "gl_identity",
    "loaded_graphics_libraries",
}
_IMPORT_NEGATIVE_FIELDS = {
    "policy_imported",
    "processor_imported",
    "libero_imported",
    "forbidden_modules",
}
_CLEANUP_FIELDS = {
    "status",
    "all_contexts_released",
    "partial_paths_cleaned",
    "egl_thread_released",
}
_ENVIRONMENT_TRANSITION_EVENTS = [
    "record_presence",
    "unset",
    "block_loader",
    "set_cache",
    "set_renderer",
    "verify_final_mapping",
    "renderer_import",
]
_ENVIRONMENT_TRANSITION_AUDIT_FIELDS = {
    "events",
    "initial_presence",
    "final_environment",
    "renderer_modules_before",
    "renderer_modules_after",
}


def governed_environment_final_mapping(config_or_governed: Mapping[str, Any]) -> dict[str, str | None]:
    """Return the exact persisted environment view admitted before import."""

    if not isinstance(config_or_governed, Mapping):
        raise AdmissionError("governed environment must be a mapping")
    governed = config_or_governed.get("governed_environment", config_or_governed)
    if not isinstance(governed, Mapping):
        raise AdmissionError("governed environment must be a mapping")
    if set(governed) != {
        "unset_before_import", "blocked_nonempty", "admitted_ld_library_path",
        "set_cache", "set_renderer",
    }:
        raise AdmissionError("governed environment schema mismatch")
    unset = governed["unset_before_import"]
    blocked = governed["blocked_nonempty"]
    set_cache = governed["set_cache"]
    set_renderer = governed["set_renderer"]
    admitted_ld = governed["admitted_ld_library_path"]
    if not isinstance(unset, list) or unset != _UNSET_BEFORE_IMPORT:
        raise AdmissionError("governed unset-before-import list mismatch")
    if not isinstance(blocked, list) or blocked != _BLOCKED_NONEMPTY:
        raise AdmissionError("governed blocked-loader list mismatch")
    if not isinstance(set_cache, Mapping) or dict(set_cache) != _SET_CACHE:
        raise AdmissionError("governed cache mapping mismatch")
    if not isinstance(set_renderer, Mapping) or dict(set_renderer) != _SET_RENDERER:
        raise AdmissionError("governed renderer mapping mismatch")
    expected_ld = "/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib/criu/:/opt/hyhal/lib:/opt/dtk-25.04.2/dcc/gcvm/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk-25.04.2/llvm/lib:/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk-25.04.2/opencl/lib:/opt/hyhal/lib/rocprofiler:/opt/hyhal/lib/criu/:/opt/hyhal/lib:/opt/dtk-25.04.2/dcc/gcvm/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk-25.04.2/llvm/lib:/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk-25.04.2/opencl/lib"
    if admitted_ld != expected_ld:
        raise AdmissionError("governed admitted LD_LIBRARY_PATH mismatch")
    names = list(dict.fromkeys(unset + blocked + list(set_cache) + list(set_renderer) + ["LD_LIBRARY_PATH"]))
    final: dict[str, str | None] = {name: None for name in names}
    final.update({str(key): str(value) for key, value in set_cache.items()})
    final.update({str(key): str(value) for key, value in set_renderer.items()})
    final["LD_LIBRARY_PATH"] = str(admitted_ld)
    return final


def _validate_environment_transition_audit(
    value: object, *, expected_final_environment: Mapping[str, str | None]
) -> None:
    if not isinstance(value, Mapping) or set(value) != _ENVIRONMENT_TRANSITION_AUDIT_FIELDS:
        raise AdmissionError("probe environment-transition audit schema mismatch")
    if value["events"] != _ENVIRONMENT_TRANSITION_EVENTS:
        raise AdmissionError("probe environment-transition audit event order mismatch")
    initial = value["initial_presence"]
    if not isinstance(initial, Mapping) or set(initial) != set(expected_final_environment):
        raise AdmissionError("probe environment-transition initial presence mismatch")
    for name in sorted(initial):
        entry = initial[name]
        if not isinstance(entry, Mapping) or set(entry) != {"present", "nonempty"}:
            raise AdmissionError("probe environment-transition presence schema mismatch")
        if type(entry["present"]) is not bool or type(entry["nonempty"]) is not bool:
            raise AdmissionError("probe environment-transition presence values must be boolean")
        if not entry["present"] and entry["nonempty"]:
            raise AdmissionError("probe environment-transition presence is inconsistent")
        if name in _BLOCKED_NONEMPTY and entry["nonempty"]:
            raise AdmissionError("probe environment-transition blocked loader was initially nonempty")
    loader_presence = initial["LD_LIBRARY_PATH"]
    if not loader_presence["present"] or not loader_presence["nonempty"]:
        raise AdmissionError("probe environment-transition LD_LIBRARY_PATH presence is invalid")
    final = value["final_environment"]
    if not isinstance(final, Mapping) or dict(final) != dict(expected_final_environment):
        raise AdmissionError("probe environment-transition final environment mismatch")
    if any(item is not None and not isinstance(item, str) for item in final.values()):
        raise AdmissionError("probe environment-transition final values must be null or strings")
    if value["renderer_modules_before"] != []:
        raise AdmissionError("renderer modules were imported before the governed transition")
    after = value["renderer_modules_after"]
    if not isinstance(after, list) or not after or after != sorted(set(after)):
        raise AdmissionError("probe renderer_modules_after audit must be a sorted nonempty list")
    if any(
        not isinstance(module, str)
        or not module
        or not (module == "mujoco" or module.startswith("mujoco.") or module == "OpenGL" or module.startswith("OpenGL."))
        or any(token in module.lower() for token in ("policy", "processor", "libero"))
        for module in after
    ):
        raise AdmissionError("probe renderer module audit contains a forbidden or invalid module")


def _require_lower_hex(value: object, length: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise AdmissionError(f"{label} must be lowercase hexadecimal ({length} characters)")
    return value


def _normalize_probe_index(index: object) -> str:
    if type(index) is int:
        if not 0 <= index <= 999:
            raise AdmissionError("probe index is outside the supported range")
        return f"{index:03d}"
    if isinstance(index, str) and len(index) == 3 and index.isdigit():
        return index
    raise AdmissionError("probe index must be an integer or a three-digit decimal string")


def _validate_context_records(
    context_records: object,
    *,
    device_count: int,
    selected_ordinal: int,
    full_renderer_fingerprint: Mapping[str, Any],
) -> None:
    if not isinstance(context_records, list) or len(context_records) != device_count:
        raise AdmissionError("probe context records must contain one record for every ordinal")
    selected: Mapping[str, Any] | None = None
    for ordinal, context in enumerate(context_records):
        if not isinstance(context, Mapping) or set(context) != _CONTEXT_RECORD_FIELDS:
            raise AdmissionError("probe context record schema contains an unexpected field")
        if type(context["ordinal"]) is not int or context["ordinal"] != ordinal:
            raise AdmissionError("probe context records must be complete and ordered by ordinal")
        _validate_identity(context["egl_identity"], keys={"vendor", "version"}, label="EGL identity")
        _validate_identity(
            context["gl_identity"],
            keys={"vendor", "renderer", "version"},
            label="GL identity",
        )
        _validate_graphics_library_records(
            context["loaded_graphics_libraries"], label="context loaded graphics libraries"
        )
        if ordinal == selected_ordinal:
            selected = context
    if selected is None:
        raise AdmissionError("selected probe context record is missing")
    if selected["egl_identity"] != full_renderer_fingerprint["egl_identity"]:
        raise AdmissionError("selected EGL context identity differs from full fingerprint")
    if selected["gl_identity"] != full_renderer_fingerprint["gl_identity"]:
        raise AdmissionError("selected GL context identity differs from full fingerprint")
    if selected["loaded_graphics_libraries"] != full_renderer_fingerprint["loaded_graphics_libraries"]:
        raise AdmissionError("selected context libraries differ from full fingerprint")


def _validate_import_negative(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _IMPORT_NEGATIVE_FIELDS:
        raise AdmissionError("probe import-negative audit schema mismatch")
    for name in ("policy_imported", "processor_imported", "libero_imported"):
        if value[name] is not False:
            raise AdmissionError(f"probe import-negative audit detected {name}")
    modules = value["forbidden_modules"]
    if not isinstance(modules, list) or modules:
        raise AdmissionError("probe import-negative audit detected forbidden modules")


def _validate_cleanup(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _CLEANUP_FIELDS:
        raise AdmissionError("probe cleanup audit schema mismatch")
    if value["status"] != "PASS":
        raise AdmissionError("probe cleanup status is not PASS")
    for name in ("all_contexts_released", "partial_paths_cleaned", "egl_thread_released"):
        if value[name] is not True:
            raise AdmissionError(f"probe cleanup audit failed: {name}")


def _validate_renderer_probe_record(
    record: Mapping[str, Any],
    *,
    expected_parent_pid: int | None = None,
    expected_config_sha256: str | None = None,
    expected_runtime_lock_sha256: str | None = None,
    expected_probe_source_bindings_sha256: str | None = None,
    expected_execution_commit: str | None = None,
    expected_device_count: int | None = None,
    expected_selected_ordinal: int | None = None,
    required_selected_extensions: list[str] | None = None,
    expected_egl_identity: Mapping[str, str] | None = None,
    expected_gl_identity: Mapping[str, str] | None = None,
    expected_final_environment: Mapping[str, str | None] | None = None,
) -> None:
    if not isinstance(record, Mapping) or set(record) != _PROBE_RECORD_FIELDS:
        raise AdmissionError("probe record schema contains an unexpected field")
    if record["schema_version"] != 1:
        raise AdmissionError("probe record schema version mismatch")
    index = _normalize_probe_index(record["index"])
    if record["index"] != index:
        raise AdmissionError("probe record index must use a canonical three-digit form")
    for name in ("pid", "ppid", "device_count", "selected_ordinal", "exit_code", "retry_count"):
        if type(record[name]) is not int:
            raise AdmissionError(f"probe record {name} must be an integer")
    if record["pid"] <= 0 or record["ppid"] <= 0 or record["pid"] == record["ppid"]:
        raise AdmissionError("probe record PID/PPID identity is invalid")
    if expected_parent_pid is not None and record["ppid"] != expected_parent_pid:
        raise AdmissionError("probe record PPID differs from the admission parent")
    if record["evidence_role"] != "infrastructure_admission_only":
        raise AdmissionError("probe record evidence role is not infrastructure admission")
    if record["included_in_null_calibration_evidence"] is not False:
        raise AdmissionError("probe record is incorrectly marked as null-calibration evidence")
    config_sha = _require_lower_hex(record["config_sha256"], 64, "probe config SHA-256")
    runtime_sha = _require_lower_hex(record["runtime_lock_sha256"], 64, "probe runtime lock SHA-256")
    source_sha = _require_lower_hex(record["probe_source_bindings_sha256"], 64, "probe source-binding SHA-256")
    execution_commit = _require_lower_hex(record["execution_commit"], 40, "probe execution commit")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        raise AdmissionError("probe config SHA-256 differs from the frozen config")
    if expected_runtime_lock_sha256 is not None and runtime_sha != expected_runtime_lock_sha256:
        raise AdmissionError("probe runtime lock SHA-256 differs from the frozen runtime")
    if expected_probe_source_bindings_sha256 is not None and source_sha != expected_probe_source_bindings_sha256:
        raise AdmissionError("probe source-binding SHA-256 differs from the frozen sources")
    if expected_execution_commit is not None and execution_commit != expected_execution_commit:
        raise AdmissionError("probe execution commit differs from the expected committed identity")
    if type(record["fallback_used"]) is not bool or record["fallback_used"] is not False:
        raise AdmissionError("probe fallback is forbidden")
    if record["exit_code"] != 0:
        raise AdmissionError("probe child exit code is not zero")
    if record["retry_count"] != 0:
        raise AdmissionError("probe retry count must remain zero")

    worker = record["worker_preimport_fingerprint"]
    full = record["full_renderer_fingerprint"]
    _validate_worker_preimport_fingerprint(worker)
    _validate_full_renderer_fingerprint(full)
    worker_hash = worker_preimport_namespace_fingerprint_sha256(worker)
    full_hash = admission_full_renderer_fingerprint_sha256(full)
    if record["worker_preimport_fingerprint_sha256"] != worker_hash:
        raise AdmissionError("probe device payload or worker fingerprint hash mismatch")
    if record["full_renderer_fingerprint_sha256"] != full_hash:
        raise AdmissionError("probe GL payload or full fingerprint hash mismatch")
    worker_keys = {
        "schema_version", "device_count", "selected_ordinal", "ordered_descriptors",
        "supported_drm_nodes", "namespace_facts", "governed_environment",
        "static_graphics_libraries",
    }
    if {key: full[key] for key in worker_keys} != {key: worker[key] for key in worker_keys}:
        raise AdmissionError("probe full and worker fingerprint payloads differ")
    if record["device_count"] != worker["device_count"] or record["selected_ordinal"] != worker["selected_ordinal"]:
        raise AdmissionError("probe record device count or selected ordinal differs from worker payload")
    if expected_device_count is not None and record["device_count"] != expected_device_count:
        raise AdmissionError("probe device count differs from the frozen count")
    if expected_selected_ordinal is not None and record["selected_ordinal"] != expected_selected_ordinal:
        raise AdmissionError("probe selected ordinal differs from the frozen ordinal")
    descriptors = record["ordered_descriptors"]
    if descriptors != worker["ordered_descriptors"]:
        raise AdmissionError("probe ordered device descriptors differ from worker payload")
    required = record["required_selected_extensions"]
    if not isinstance(required, list) or any(not isinstance(item, str) or not item for item in required) or required != sorted(set(required)):
        raise AdmissionError("probe selected extension list must be canonical strings")
    selected_extensions = worker["ordered_descriptors"][record["selected_ordinal"]]["extensions"]
    if any(extension not in selected_extensions for extension in required):
        raise AdmissionError("probe selected device is missing a required extension")
    if required_selected_extensions is not None and required != required_selected_extensions:
        raise AdmissionError("probe selected extension list differs from the frozen contract")
    if expected_egl_identity is not None and full["egl_identity"] != dict(expected_egl_identity):
        raise AdmissionError("probe EGL identity differs from the frozen renderer")
    if expected_gl_identity is not None and full["gl_identity"] != dict(expected_gl_identity):
        raise AdmissionError("probe GL identity differs from the frozen renderer")
    _validate_environment_transition_audit(
        record["environment_transition_audit"],
        expected_final_environment=(
            worker["governed_environment"]
            if expected_final_environment is None
            else expected_final_environment
        ),
    )
    if expected_final_environment is not None and worker["governed_environment"] != dict(expected_final_environment):
        raise AdmissionError("probe worker governed environment differs from the frozen mapping")
    _validate_context_records(
        record["context_records"],
        device_count=worker["device_count"],
        selected_ordinal=worker["selected_ordinal"],
        full_renderer_fingerprint=full,
    )
    _validate_import_negative(record["import_negative"])
    _validate_cleanup(record["cleanup"])


def build_renderer_probe_record(
    *,
    index: int | str,
    pid: int,
    ppid: int,
    worker_preimport_fingerprint: Mapping[str, Any],
    full_renderer_fingerprint: Mapping[str, Any],
    environment_transition_audit: Mapping[str, Any],
    config_sha256: str,
    runtime_lock_sha256: str,
    probe_source_bindings_sha256: str,
    execution_commit: str,
    context_records: list[Mapping[str, Any]],
    required_selected_extensions: list[str],
    import_negative: Mapping[str, Any],
    cleanup: Mapping[str, Any],
    evidence_role: str = "infrastructure_admission_only",
    included_in_null_calibration_evidence: bool = False,
    exit_code: int = 0,
    retry_count: int = 0,
    fallback_used: bool = False,
) -> dict[str, Any]:
    """Build one immutable, hash-bound renderer probe child record."""

    worker = copy.deepcopy(dict(worker_preimport_fingerprint))
    full = copy.deepcopy(dict(full_renderer_fingerprint))
    record = {
        "schema_version": 1,
        "index": _normalize_probe_index(index),
        "pid": pid,
        "ppid": ppid,
        "evidence_role": evidence_role,
        "included_in_null_calibration_evidence": included_in_null_calibration_evidence,
        "config_sha256": config_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "probe_source_bindings_sha256": probe_source_bindings_sha256,
        "execution_commit": execution_commit,
        "device_count": worker.get("device_count"),
        "selected_ordinal": worker.get("selected_ordinal"),
        "required_selected_extensions": copy.deepcopy(required_selected_extensions),
        "ordered_descriptors": copy.deepcopy(worker.get("ordered_descriptors")),
        "worker_preimport_fingerprint": worker,
        "worker_preimport_fingerprint_sha256": worker_preimport_namespace_fingerprint_sha256(worker),
        "full_renderer_fingerprint": full,
        "full_renderer_fingerprint_sha256": admission_full_renderer_fingerprint_sha256(full),
        "context_records": copy.deepcopy(context_records),
        "environment_transition_audit": copy.deepcopy(dict(environment_transition_audit)),
        "exit_code": exit_code,
        "retry_count": retry_count,
        "fallback_used": fallback_used,
        "import_negative": copy.deepcopy(dict(import_negative)),
        "cleanup": copy.deepcopy(dict(cleanup)),
    }
    _validate_renderer_probe_record(record)
    return record


def verify_three_probe_records(
    records: list[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    expected_parent_pid: int,
    expected_execution_commit: str,
    repo_root: Path | str,
) -> bool:
    """Verify exactly three independent probe payloads against the frozen contract."""

    if not isinstance(records, list) or len(records) != 3:
        raise AdmissionError("renderer admission requires exactly three probe records")
    if type(expected_parent_pid) is not int or expected_parent_pid <= 0:
        raise AdmissionError("renderer admission expected parent PID is invalid")
    validate_admission_config(config, repo_root=repo_root)
    contract = config["probe_contract"]
    expected_config = config.get("config_sha256")
    expected_runtime = config["runtime"]["runtime_lock"]["sha256"]
    expected_sources = probe_source_bindings_sha256(contract["source_files"])
    _require_lower_hex(expected_execution_commit, 40, "expected probe execution commit")
    seen_indices: list[str] = []
    seen_pids: set[int] = set()
    for record in records:
        _validate_renderer_probe_record(
            record,
            expected_parent_pid=expected_parent_pid,
            expected_config_sha256=expected_config,
            expected_runtime_lock_sha256=expected_runtime,
            expected_probe_source_bindings_sha256=expected_sources,
            expected_execution_commit=expected_execution_commit,
            expected_device_count=contract["expected_device_count"],
            expected_selected_ordinal=contract["selected_ordinal"],
            required_selected_extensions=contract["required_selected_extensions"],
            expected_egl_identity=contract["expected_egl_identity"],
            expected_gl_identity=contract["expected_gl_identity"],
            expected_final_environment=governed_environment_final_mapping(config),
        )
        index = record["index"]
        if index in seen_indices:
            raise AdmissionError("renderer admission probe index values are not unique")
        seen_indices.append(index)
        pid = record["pid"]
        if pid in seen_pids:
            raise AdmissionError("renderer admission probe PIDs are not distinct")
        seen_pids.add(pid)
    if seen_indices != ["000", "001", "002"]:
        raise AdmissionError("renderer admission probe indices must be exactly 000, 001, 002")
    reference_worker = records[0]["worker_preimport_fingerprint"]
    reference_full = records[0]["full_renderer_fingerprint"]
    reference_worker_hash = records[0]["worker_preimport_fingerprint_sha256"]
    reference_full_hash = records[0]["full_renderer_fingerprint_sha256"]
    reference_environment_audit = records[0]["environment_transition_audit"]
    for record in records[1:]:
        if record["worker_preimport_fingerprint"] != reference_worker or record["worker_preimport_fingerprint_sha256"] != reference_worker_hash:
            raise AdmissionError("renderer admission worker fingerprints do not match exactly")
        if record["full_renderer_fingerprint"] != reference_full or record["full_renderer_fingerprint_sha256"] != reference_full_hash:
            raise AdmissionError("renderer admission full fingerprints do not match exactly")
        if record["environment_transition_audit"] != reference_environment_audit:
            raise AdmissionError("renderer admission environment transition audits do not match exactly")
    return True
