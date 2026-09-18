"""Static provenance checks. No environment, renderer or execution entrypoint."""
import copy
import hashlib
import json
from pathlib import Path


class RecoveryBlocked(ValueError):
    pass


def verified_bytes(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RecoveryBlocked(f"not a non-symlink regular file: {path}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RecoveryBlocked(f"byte hash mismatch: {path}")
    return data


def verify_copy(source, target, expected):
    if verified_bytes(source, expected) != verified_bytes(target, expected):
        raise RecoveryBlocked("copy byte drift")


def canonical_registry_hash(value):
    body = {k: v for k, v in value.items() if k != "registry_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def verify_derivation(old, new):
    """Only the four explicitly permitted paths and registry self hash may differ."""
    normalized = copy.deepcopy(new)
    changes = []
    for field in ("source_file", "parent_run_manifest", "parent_terminal_manifest"):
        before, after = old["traces"][0][field], new["traces"][0][field]
        expected = old["traces"][0][field + "_sha256"]
        verified_bytes(after, expected)
        changes.append({"pointer": "/traces/0/" + field, "before": before,
                        "after": after, "target_sha256": expected})
        normalized["traces"][0][field] = before
    trace = old["traces"][0]["trace_id"]
    before = old["tape_provenance"]["per_trace"][trace]["path"]
    after = new["tape_provenance"]["per_trace"][trace]["path"]
    path = Path(after)
    if path.is_symlink() or not path.is_file():
        raise RecoveryBlocked("missing or symlink tape")
    changes.append({"pointer": f"/tape_provenance/per_trace/{trace}/path",
                    "before": before, "after": after,
                    "container_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    normalized["tape_provenance"]["per_trace"][trace]["path"] = before
    normalized["registry_sha256"] = old["registry_sha256"]
    if normalized != old:
        raise RecoveryBlocked("non-path scientific mutation")
    if canonical_registry_hash(new) != new["registry_sha256"]:
        raise RecoveryBlocked("registry self hash mismatch")
    changes.append({"pointer": "/registry_sha256", "before": old["registry_sha256"],
                    "after": new["registry_sha256"]})
    return changes
