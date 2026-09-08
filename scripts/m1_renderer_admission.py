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
        import_callback()
    return {"initial_presence": presence, "final_environment": final}


def admission_full_renderer_fingerprint_sha256(value: Mapping[str, Any]) -> str:
    return _canonical_sha256(value)


def worker_preimport_namespace_fingerprint_sha256(value: Mapping[str, Any]) -> str:
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
        raise AdmissionError("worker descriptor schema contains an unexpected field")
    for ordinal, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict) or set(descriptor) != descriptor_keys:
            raise AdmissionError("worker descriptor schema contains an unexpected field")
        if type(descriptor["ordinal"]) is not int or descriptor["ordinal"] != ordinal:
            raise AdmissionError("worker descriptor ordinals are not exact enumeration order")
        extensions = descriptor["extensions"]
        if not isinstance(extensions, list) or any(not isinstance(item, str) or not item for item in extensions) or extensions != sorted(set(extensions)):
            raise AdmissionError("worker descriptor extensions must be sorted unique strings")
        for field in ("drm_device_file", "drm_render_node_file"):
            path = descriptor[field]
            if path is not None and (not isinstance(path, str) or not path.startswith("/dev/dri/") or not Path(path).is_absolute()):
                raise AdmissionError("worker descriptor DRM path must be null or an absolute /dev/dri path")
    supported_nodes = value["supported_drm_nodes"]
    if not isinstance(supported_nodes, list) or any(
        not isinstance(path, str) or not path.startswith("/dev/dri/") or not Path(path).is_absolute()
        for path in supported_nodes
    ) or supported_nodes != sorted(set(supported_nodes)):
        raise AdmissionError("worker DRM-node schema mismatch")
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
    allowed_environment_names = set(_UNSET_BEFORE_IMPORT) | set(_BLOCKED_NONEMPTY) | set(_SET_CACHE) | set(_SET_RENDERER) | {"LD_LIBRARY_PATH"}
    environment = value["governed_environment"]
    if not isinstance(environment, dict) or set(environment) != allowed_environment_names:
        raise AdmissionError("worker governed-environment schema contains an unexpected field")
    if any(item is not None and not isinstance(item, str) for item in environment.values()):
        raise AdmissionError("worker governed-environment values must be null or strings")
    return _canonical_sha256(value)
