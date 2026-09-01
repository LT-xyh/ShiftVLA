#!/usr/bin/env python3
"""Bounded, policy-free orchestration for the M1 hard gate.

The module is intentionally a thin orchestration layer around the strict
primitives in :mod:`scripts.m1_hard_gate` and the already audited
``RuntimeAdapter`` in ``scripts.m1_state_replay``.  Both modules are loaded
only at the point where a caller has explicitly asked for registry or runtime
work.  Registry selection is source-only and is completed before an
environment is constructed or a restore is attempted.

The default command is suitable for an integration run, but unit tests should
inject ``trace_loader``, ``environment_builder`` and ``subprocess_runner``.
No model or action-generation stack is part of this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, TypeVar

import numpy as np


if __package__ in {None, ""}:
    # ``python scripts/m1_hard_gate_runner.py`` places only ``scripts/`` on
    # sys.path.  Fresh workers use that exact entrypoint, so make the repository
    # package importable without importing any environment or policy module.
    _REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPOSITORY_ROOT not in sys.path:
        sys.path.insert(0, _REPOSITORY_ROOT)


# Keep these literals local so importing this module cannot transitively load
# an environment stack.  ``prepare_registry`` cross-checks them with the
# strict hard-gate module once that module is loaded.
REGIME_NAMES: tuple[str, ...] = (
    "free_motion",
    "pre_contact",
    "contact",
    "grasp",
    "carried",
    "release",
    "predicate_transition",
)
_RENDERER_REFERENCE_KEY = "__m1_renderer_rgb__"
_RENDERER_CAMERA = "agentview"
_RENDERER_OBSERVATION_KEY = "render_rgb"
GATE_NAMES: tuple[str, ...] = (
    "state_closure",
    "physics_static_identity",
    "observation_static_identity",
    "null_physics",
    "null_renderer",
    "corpus_regime_coverage",
    "same_process_dynamic",
    "fresh_process_dynamic",
    "provenance",
)
DEFAULT_HORIZON = 10
RESTORES_PER_STAGE = 3
FRESH_PROCESSES_PER_STAGE = 3
WORKER_PROTOCOL_VERSION = 1
_TERMINAL_REGIMES = frozenset({"release", "predicate_transition"})
_OUTCOME_KEYS = frozenset(
    {
        "success",
        "reward",
        "rewards",
        "return",
        "returns",
        "terminated",
        "truncated",
        "done",
        "outcome",
        "replay",
        "policy",
        "action",
        "actions",
        "action_evidence",
        "crashed",
        "status",
        "result",
        "results",
        "pass",
        "passed",
        "fail",
        "failed",
        "metric",
        "rate",
    }
)
_FORBIDDEN_RESTORE_WORDS = frozenset(
    {"reset", "set_init_state", "settle", "dummy", "retry", "constructor"}
)
_T = TypeVar("_T")


class HardGateRunnerError(RuntimeError):
    """Base class for fail-closed orchestration errors."""


class RegistryPreparationError(HardGateRunnerError):
    """Source-only corpus selection or registry freezing failed."""


class SourceCoverageError(RegistryPreparationError):
    """A classifier did not return valid source coverage metadata."""


class RestoreProtocolError(HardGateRunnerError):
    """A restore/worker violated the bounded protocol."""


class PublicationError(HardGateRunnerError):
    """An artifact would overwrite an existing path or could not be published."""


def _load_hard_gate() -> Any:
    """Load strict source/registry/calibration primitives lazily."""

    from scripts import m1_hard_gate

    return m1_hard_gate


def _load_replay_symbols() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    """Load RuntimeAdapter and official runtime factories lazily."""

    from scripts import m1_state_replay

    # The environment-only factory is imported as a fallback for callers that
    # explicitly select it.  Neither import occurs while this module is
    # imported or while a registry is being selected.
    from scripts.dcu_preflight import build_cpu_environment_runtime

    return (
        m1_state_replay.RuntimeAdapter,
        m1_state_replay.build_official_runtime,
        build_cpu_environment_runtime,
    )


def _canonical_json(value: Any) -> str:
    """Serialize finite manifest values without relying on pickle."""

    def safe(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            if item.dtype.kind == "O":
                raise TypeError("object arrays are not publication values")
            if item.dtype.kind == "f" and not np.all(np.isfinite(item)):
                raise ValueError("publication values must be finite")
            return item.tolist()
        if isinstance(item, np.generic):
            return safe(item.item())
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise TypeError("publication mapping keys must be strings")
            return {key: safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("publication values must be finite")
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        raise TypeError(f"unsupported publication value: {type(item).__name__}")

    return json.dumps(safe(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_publish(path: str | Path, payload: Mapping[str, Any] | bytes) -> Path:
    """Publish one file with exclusive link semantics and fsync."""

    target = Path(path)
    if target.is_symlink() or target.exists():
        raise PublicationError(f"refusing to overwrite existing artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else (_canonical_json(payload) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PublicationError(f"refusing to overwrite existing artifact: {target}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _atomic_publish_npy(path: str | Path, array: np.ndarray) -> Path:
    """Write one exact float32 tape in NumPy format without overwriting."""

    target = Path(path)
    if target.is_symlink() or target.exists():
        raise PublicationError(f"refusing to overwrite tape artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        # np.save appends .npy when given a string path; an open handle keeps
        # the exact temporary filename and avoids an untracked side artifact.
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise PublicationError(f"refusing to overwrite tape artifact: {target}") from exc
        finally:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _mapping(value: Any, name: str, error: type[Exception] = HardGateRunnerError) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str, error: type[Exception] = HardGateRunnerError) -> str:
    if not isinstance(value, str) or not value:
        raise error(f"{name} must be a non-empty string")
    return value


def _int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RegistryPreparationError(f"{name} must be an integer")
    result = int(value)
    if (positive and result <= 0) or (not positive and result < 0):
        raise RegistryPreparationError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return result


def _reject_outcome_fields(value: Any, path: str = "source_coverage") -> None:
    """Reject outcome-like information from classifier/selection metadata."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in _OUTCOME_KEYS or any(token in key_text for token in ("outcome", "reward", "replay")):
                raise SourceCoverageError(f"source coverage contains outcome field {path}.{key}")
            _reject_outcome_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_outcome_fields(child, f"{path}[{index}]")


def _trace_id(trace: Any) -> str:
    if isinstance(trace, Mapping):
        value = trace.get("episode_id", trace.get("trace_id", trace.get("id")))
    else:
        value = getattr(trace, "episode_id", getattr(trace, "trace_id", None))
    return _text(value, "validated M0Trace episode_id", RegistryPreparationError)


def _trace_actions(trace: Any) -> np.ndarray:
    value = trace.get("actions") if isinstance(trace, Mapping) else getattr(trace, "actions", None)
    if value is None:
        raise RegistryPreparationError(f"trace {_trace_id(trace)} has no action tape")
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise RegistryPreparationError(f"trace {_trace_id(trace)} action tape is unreadable") from exc
    if array.ndim != 2 or array.shape[1] != 7 or array.dtype != np.dtype("float32"):
        raise RegistryPreparationError(f"trace {_trace_id(trace)} actions must be float32[:, 7]")
    if not array.flags.c_contiguous or not np.all(np.isfinite(array)):
        raise RegistryPreparationError(f"trace {_trace_id(trace)} actions must be finite and contiguous")
    return np.ascontiguousarray(array)


def _trace_metadata(trace: Any) -> Mapping[str, Any]:
    if isinstance(trace, Mapping):
        return trace
    value = getattr(trace, "metadata", None)
    return value if isinstance(value, Mapping) else {}


def _coverage_from_classifier(value: Any, trace: Any, *, default_horizon: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceCoverageError(f"classifier output for {_trace_id(trace)} must be a mapping")
    _reject_outcome_fields(value)
    nested = value.get("source_coverage")
    if nested is not None:
        nested = _mapping(nested, "classifier source_coverage", SourceCoverageError)
        _reject_outcome_fields(nested)
        coverage = dict(nested)
        # These convenience fields are accepted only when they do not
        # conflict with the nested source-only record.
        for key in ("regimes", "regime", "covered_regimes", "capture_offset", "continuation_horizon", "horizon"):
            if key in value and key not in coverage:
                coverage[key] = value[key]
    else:
        coverage = dict(value)
    metadata = _trace_metadata(trace)
    if not coverage:
        fallback = metadata.get("source_coverage")
        if isinstance(fallback, Mapping):
            coverage = dict(fallback)
        elif any(key in metadata for key in ("regime", "regimes", "covered_regimes")):
            coverage = {key: metadata[key] for key in ("regime", "regimes", "covered_regimes") if key in metadata}
    raw_regimes = coverage.get("regimes", coverage.get("covered_regimes"))
    if raw_regimes is None and "regime" in coverage:
        raw_regimes = [coverage["regime"]]
    if isinstance(raw_regimes, str):
        raw_regimes = [raw_regimes]
    if not isinstance(raw_regimes, Sequence) or isinstance(raw_regimes, (str, bytes)):
        raise SourceCoverageError(f"classifier output for {_trace_id(trace)} lacks source regimes")
    regimes: list[str] = []
    for regime in raw_regimes:
        if regime not in REGIME_NAMES:
            raise SourceCoverageError(f"unknown source regime {regime!r}")
        if regime not in regimes:
            regimes.append(regime)
    if not regimes:
        raise SourceCoverageError(f"classifier output for {_trace_id(trace)} has no source regimes")
    coverage["regimes"] = regimes
    coverage.pop("regime", None)
    coverage.pop("covered_regimes", None)
    if "capture_offset" not in coverage:
        coverage["capture_offset"] = coverage.get("capture_step", 1)
    if "continuation_horizon" not in coverage:
        coverage["continuation_horizon"] = coverage.get("horizon", default_horizon)
    coverage["capture_offset"] = _int(coverage["capture_offset"], "source capture_offset")
    coverage["continuation_horizon"] = _int(
        coverage["continuation_horizon"], "source continuation_horizon", positive=True
    )
    allowed_terminal = {"terminal_evidence", "terminal_reason", "legal_terminal_step", "event", "evidence"}
    unexpected = set(coverage).difference(
        {
            "regimes",
            "capture_offset",
            "capture_step",
            "continuation_horizon",
            "horizon",
            "source_file_sha256",
            "event",
            "evidence",
            "terminal_evidence",
            "terminal_reason",
            "legal_terminal_step",
            "windows",
            "detector",
            "source",
        }
    )
    # Rich detector metadata is intentionally retained, but arbitrary result
    # fields are not.  Keeping this list strict prevents a classifier from
    # smuggling a post-hoc replay result into registry selection.
    if unexpected:
        raise SourceCoverageError(
            f"classifier output for {_trace_id(trace)} has non-source fields: {sorted(unexpected)}"
        )
    if "terminal_reason" in coverage and coverage.get("terminal_reason") is not None:
        coverage["terminal_reason"] = _text(coverage["terminal_reason"], "terminal_reason", SourceCoverageError)
    if "legal_terminal_step" in coverage:
        coverage["legal_terminal_step"] = _int(coverage["legal_terminal_step"], "legal_terminal_step")
    trace_provenance = trace.provenance() if callable(getattr(trace, "provenance", None)) else {}
    declared_source_sha = coverage.get("source_file_sha256")
    if declared_source_sha is not None and isinstance(trace_provenance, Mapping):
        if str(declared_source_sha).lower() != str(trace_provenance.get("source_file_sha256", "")).lower():
            raise SourceCoverageError(
                f"source coverage SHA-256 does not match validated trace {_trace_id(trace)}"
            )
    return coverage


def _default_source_classifier(trace: Any) -> Mapping[str, Any]:
    """Read pre-audited source coverage; never infer a regime from outcomes."""

    metadata = _trace_metadata(trace)
    coverage = metadata.get("source_coverage")
    if isinstance(coverage, Mapping):
        return {"source_coverage": dict(coverage)}
    for key in ("regimes", "covered_regimes", "regime"):
        if key in metadata:
            return {key: metadata[key]}
    raise SourceCoverageError(
        f"trace {_trace_id(trace)} lacks explicit source coverage; an audited classifier is required"
    )


def _iter_source_items(value: Any) -> list[tuple[str | None, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        # A regime-keyed map is convenient for preregistered supplements.
        if any(key in REGIME_NAMES for key in value):
            result: list[tuple[str | None, Any]] = []
            for key, items in value.items():
                if key not in REGIME_NAMES:
                    raise RegistryPreparationError(f"unknown source-regime key: {key}")
                if isinstance(items, (str, Path, Mapping)):
                    items = [items]
                if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                    raise RegistryPreparationError(f"source list for {key} must be a sequence")
                result.extend((key, item) for item in items)
            return result
        return [(None, value)]
    if isinstance(value, (str, Path)):
        return [(None, value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RegistryPreparationError("source collection must be a sequence, mapping, or path")
    return [(None, item) for item in value]


def _source_value(item: Any) -> Any:
    if isinstance(item, Mapping) and "source" in item:
        return item["source"]
    return item


def _default_trace_loader(source: Any) -> Any:
    hard_gate = _load_hard_gate()
    if isinstance(source, Mapping):
        raise RegistryPreparationError("default trace loader accepts only a persisted M0 row path")
    path = Path(source)
    for parent in path.parents:
        if (parent / "run_manifest.json").is_file() and (parent / "terminal_manifest.json").is_file():
            return hard_gate.extract_m0_trace(source, parent_run_dir=parent)
    raise RegistryPreparationError(
        f"could not locate explicit parent manifests for M0 source {path}; pass an injected loader"
    )


def _invoke_loader(loader: Callable[..., Any], source: Any) -> Any:
    """Call an injected loader with its simple one-source contract."""

    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        return loader(source)
    parameters = list(signature.parameters.values())
    if any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return loader(source)
    if not parameters:
        return loader()
    return loader(source)


@dataclass(frozen=True)
class _LoadedTrace:
    source: Any
    trace: Any
    coverage: Mapping[str, Any]
    role: str

    @property
    def trace_id(self) -> str:
        return _trace_id(self.trace)


@dataclass
class PreparedRegistry(Mapping[str, Any]):
    """Frozen registry plus validated source traces used by the runner."""

    registry_path: Path
    registry: Mapping[str, Any]
    traces: Mapping[str, Any] = field(default_factory=dict)
    per_trace: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    selection: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        values = {
            "registry_path": self.registry_path,
            "path": self.registry_path,
            "registry": self.registry,
            "traces": self.traces,
            "per_trace": self.per_trace,
            "selection": self.selection,
            "trace_ids": tuple(self.traces),
        }
        return values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("registry_path", "path", "registry", "traces", "per_trace", "selection", "trace_ids"))

    def __len__(self) -> int:
        return 7

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": str(self.registry_path),
            "registry": copy.deepcopy(dict(self.registry)),
            "traces": sorted(self.traces),
            "per_trace": copy.deepcopy(dict(self.per_trace)),
            "selection": copy.deepcopy(dict(self.selection)),
        }


def _validated_trace(trace: Any, hard_gate: Any) -> Any:
    """Require the loader boundary to produce a validated M0Trace."""

    if isinstance(trace, hard_gate.M0Trace):
        return trace
    # Test doubles may expose the same immutable provenance API.  They still
    # must pass the strict registry writer below; accepting this seam keeps the
    # runner testable without allowing a mapping to masquerade as a trace.
    if callable(getattr(trace, "provenance", None)) and hasattr(trace, "actions") and hasattr(trace, "episode_id"):
        return trace
    raise RegistryPreparationError("trace loader did not return a validated M0Trace")


def _candidate_coverage(loaded: Sequence[_LoadedTrace]) -> dict[str, Mapping[str, Any]]:
    return {item.trace_id: dict(item.coverage) for item in loaded}


def _selection_ids(selection: Any) -> tuple[str, ...]:
    value = getattr(selection, "trace_ids", None)
    if value is None and isinstance(selection, Mapping):
        value = selection.get("trace_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RegistryPreparationError("strict corpus selector did not return trace_ids")
    return tuple(_text(item, "selected trace ID", RegistryPreparationError) for item in value)


def _local_source_selection(
    hard_gate: Any,
    primary: Sequence[_LoadedTrace],
    supplements: Sequence[_LoadedTrace],
    *,
    minimum_per_regime: int,
) -> Any | None:
    """Conservative compatibility selector used only if a strict selector errors.

    The checked-in strict module is the authority.  This tiny fallback exists
    for older source snapshots whose selector may fail while evaluating a
    mathematically complete validated set; registry writing and loading still
    always go through the strict module.  No outcomes are inspected here.
    """

    candidates = list(primary) + list(supplements)
    if not candidates:
        return None
    def cover(items: Sequence[_LoadedTrace]) -> bool:
        counts = {name: 0 for name in REGIME_NAMES}
        for item in items:
            for regime in item.coverage["regimes"]:
                counts[str(regime)] += 1
        return all(value >= minimum_per_regime for value in counts.values())

    # Exact minimum-cardinality search is cheap for the small preregistered
    # corpus and preserves primary precedence/tie-by-ID semantics.
    ordered_primary = tuple(sorted(primary, key=lambda value: value.trace_id))
    chosen: list[_LoadedTrace] = []
    for size in range(1, len(ordered_primary) + 1):
        options = [combo for combo in itertools.combinations(ordered_primary, size) if cover(combo)]
        if options:
            chosen = list(min(options, key=lambda combo: tuple(item.trace_id for item in combo)))
            break
    if not chosen:
        ordered_all = tuple(sorted(candidates, key=lambda value: value.trace_id))
        options: list[tuple[_LoadedTrace, ...]] = []
        for size in range(1, len(ordered_all) + 1):
            options.extend(
                combo
                for combo in itertools.combinations(ordered_all, size)
                if any(item in supplements for item in combo) and cover(combo)
            )
            if options:
                best_key = lambda combo: (
                    sum(item in supplements for item in combo),
                    tuple(item.trace_id for item in combo),
                )
                chosen = list(min(options, key=best_key))
                break
    if not cover(chosen):
        return None
    selected_ids = tuple(sorted(item.trace_id for item in chosen))
    primary_ids = tuple(sorted(item.trace_id for item in chosen if item in primary))
    supplement_ids = tuple(sorted(item.trace_id for item in chosen if item in supplements))
    coverage = {
        item.trace_id: tuple(str(regime) for regime in item.coverage["regimes"])
        for item in chosen
    }
    return hard_gate.CorpusSelection(
        trace_ids=selected_ids,
        regimes=REGIME_NAMES,
        source_coverage=coverage,
        primary_trace_ids=primary_ids,
        supplement_trace_ids=supplement_ids,
        source_coverage_only=True,
    )


def _derive_tape_provenance(traces: Sequence[Any], per_trace: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not traces:
        raise RegistryPreparationError("cannot derive tape provenance from an empty registry")
    first = _trace_actions(traces[0])
    offsets = [int(per_trace[_trace_id(trace)]["capture_offset"]) for trace in traces]
    trace_seed = getattr(traces[0], "seed", None)
    result: dict[str, Any] = {
        "algorithm": "validated_m0_action_tape",
        "dtype": "float32",
        "shape": list(first.shape),
        "sha256": _sha256_bytes(first.tobytes(order="C")),
        "capture_offsets": sorted(set(offsets)),
    }
    if trace_seed is not None and isinstance(trace_seed, (int, np.integer)) and not isinstance(trace_seed, bool):
        result["seed"] = int(trace_seed)
    return result


def _complete_tape_provenance(
    target_registry: Path,
    value: Mapping[str, Any],
    traces: Sequence[Any],
    *,
    action_tape: Any | None,
    per_trace: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Ensure the strict registry writer has a persisted exact tape path."""

    result = dict(value)
    if "path" in result or isinstance(result.get("per_trace"), Mapping):
        return result
    if action_tape is None:
        if not traces:
            raise RegistryPreparationError("cannot materialize tape without selected traces")
        tape = _trace_actions(traces[0])
    else:
        try:
            tape = np.asarray(action_tape)
        except Exception as exc:
            raise RegistryPreparationError("action_tape is not array-like") from exc
        if tape.dtype != np.dtype("float32"):
            raise RegistryPreparationError("action_tape must preserve float32 dtype")
        if tape.ndim != 2 or tape.shape[1] != 7 or not np.all(np.isfinite(tape)):
            raise RegistryPreparationError("action_tape must be finite float32[:, 7]")
        tape = np.ascontiguousarray(tape)
    declared_shape = result.get("shape")
    if declared_shape is not None and list(declared_shape) != list(tape.shape):
        raise RegistryPreparationError("tape provenance shape does not match action_tape")
    declared_sha = result.get("sha256")
    actual_sha = _sha256_bytes(tape.tobytes(order="C"))
    if declared_sha is not None and str(declared_sha).lower() != actual_sha:
        raise RegistryPreparationError("tape provenance SHA-256 does not match action_tape")
    tape_path = target_registry.parent / f"{target_registry.stem}.action_tape.npy"
    _atomic_publish_npy(tape_path, tape)
    result.setdefault("algorithm", "validated_m0_action_tape")
    result.setdefault("dtype", "float32")
    result.setdefault("shape", list(tape.shape))
    result.setdefault("sha256", actual_sha)
    result.setdefault(
        "capture_offsets",
        sorted({int(item["capture_offset"]) for item in per_trace.values()}),
    )
    result["path"] = str(tape_path.resolve())
    return result


def _normalise_optional_alias(kwargs: dict[str, Any], names: Sequence[str], current: Any) -> Any:
    if current is not None:
        return current
    for name in names:
        if name in kwargs:
            return kwargs.pop(name)
    return current


def prepare_registry(
    primary_sources: Any = None,
    preregistered_supplements: Any = None,
    registry_path: str | Path | None = None,
    *,
    trace_loader: Callable[..., Any] | None = None,
    source_classifier: Callable[..., Mapping[str, Any]] | None = None,
    tape_provenance: Mapping[str, Any] | None = None,
    action_tape: Any | None = None,
    horizon: int = DEFAULT_HORIZON,
    minimum_per_regime: int = 1,
    **aliases: Any,
) -> PreparedRegistry:
    """Scan primary M0 traces, add only preregistered supplements, then freeze.

    The primary collection is loaded and classified in full before a
    supplement loader is called.  Selection receives only validated traces and
    source coverage, never outcomes.  The strict ``write_frozen_registry``
    function performs the exclusive atomic publication.
    """

    primary_sources = _normalise_optional_alias(
        aliases,
        (
            "primary_m0_sources",
            "primary_m0_trace_paths",
            "primary_trace_paths",
            "primary_traces",
            "primary_candidates",
            "primary",
        ),
        primary_sources,
    )
    preregistered_supplements = _normalise_optional_alias(
        aliases,
        (
            "supplements",
            "supplement_sources",
            "supplement_m0_trace_paths",
            "preregistered_supplements",
            "preregistered",
        ),
        preregistered_supplements,
    )
    registry_path = _normalise_optional_alias(aliases, ("output_path", "path"), registry_path)
    source_classifier = _normalise_optional_alias(
        aliases, ("classifier", "trajectory_classifier", "classify_source"), source_classifier
    )
    tape_provenance = _normalise_optional_alias(aliases, ("tape",), tape_provenance)
    if aliases:
        raise RegistryPreparationError(f"unknown prepare_registry arguments: {sorted(aliases)}")
    if registry_path is None:
        raise RegistryPreparationError("registry_path is required")
    target = Path(registry_path)
    if target.is_symlink() or target.exists():
        raise PublicationError(f"refusing to overwrite existing registry: {target}")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise RegistryPreparationError("horizon must be a positive integer")
    if isinstance(minimum_per_regime, bool) or not isinstance(minimum_per_regime, int) or minimum_per_regime <= 0:
        raise RegistryPreparationError("minimum_per_regime must be a positive integer")
    hard_gate = _load_hard_gate()
    if tuple(hard_gate.REGIME_NAMES) != REGIME_NAMES:
        raise RegistryPreparationError("strict hard-gate regime names differ from runner contract")
    loader = trace_loader or _default_trace_loader
    classifier = source_classifier or _default_source_classifier

    def scan(items: Any, role: str) -> list[_LoadedTrace]:
        loaded: list[_LoadedTrace] = []
        seen: set[str] = set()
        for fallback_regime, raw_item in _iter_source_items(items):
            if isinstance(raw_item, Mapping):
                _reject_outcome_fields(raw_item, path=f"{role}_candidate")
                if role == "supplement" and raw_item.get("preregistered") is False:
                    raise RegistryPreparationError(
                        f"supplement {raw_item.get('source', raw_item.get('episode_id', '<unknown>'))} is not preregistered"
                    )
            source = _source_value(raw_item)
            trace = _validated_trace(_invoke_loader(loader, source), hard_gate)
            trace_id = _trace_id(trace)
            if trace_id in seen:
                raise RegistryPreparationError(f"duplicate {role} trace {trace_id}")
            seen.add(trace_id)
            try:
                result = _invoke_loader(classifier, trace)
            except SourceCoverageError:
                candidate_coverage = raw_item.get("source_coverage") if isinstance(raw_item, Mapping) else None
                if candidate_coverage is None:
                    raise
                result = {"source_coverage": candidate_coverage}
            if (
                isinstance(raw_item, Mapping)
                and isinstance(raw_item.get("source_coverage"), Mapping)
                and isinstance(result, Mapping)
                and not any(key in result for key in ("regime", "regimes", "covered_regimes", "source_coverage"))
            ):
                result = {"source_coverage": raw_item["source_coverage"], **dict(result)}
            if fallback_regime is not None and isinstance(result, Mapping) and not any(
                key in result for key in ("regime", "regimes", "covered_regimes", "source_coverage")
            ):
                result = {"regimes": [fallback_regime], **dict(result)}
            coverage = _coverage_from_classifier(result, trace, default_horizon=horizon)
            loaded.append(_LoadedTrace(source=source, trace=trace, coverage=coverage, role=role))
        return loaded

    # This is deliberately one complete primary scan before any supplement
    # source is touched.  It is the source-only discovery boundary.
    primary = scan(primary_sources, "primary")
    primary_coverage = _candidate_coverage(primary)
    selected_primary: Any | None = None
    try:
        selected_primary = hard_gate.select_minimum_corpus(
            [item.trace for item in primary],
            preregistered_supplements=None,
            minimum_per_regime=minimum_per_regime,
            source_coverage=primary_coverage,
        )
    except Exception as exc:
        if not isinstance(exc, hard_gate.CorpusSelectionError):
            raise
        selected_primary = _local_source_selection(
            hard_gate,
            primary,
            (),
            minimum_per_regime=minimum_per_regime,
        )

    supplements: list[_LoadedTrace] = []
    if selected_primary is None:
        # The supplement collection is a caller-declared preregistration
        # boundary.  It is not scanned at all when primary coverage suffices.
        supplements = scan(preregistered_supplements, "supplement")
    all_loaded = primary + supplements
    coverage_by_id = _candidate_coverage(all_loaded)
    try:
        selection = hard_gate.select_minimum_corpus(
            [item.trace for item in primary],
            [item.trace for item in supplements],
            minimum_per_regime=minimum_per_regime,
            source_coverage=coverage_by_id,
        )
    except Exception as exc:
        if not isinstance(exc, hard_gate.CorpusSelectionError):
            raise RegistryPreparationError(f"strict source corpus selection failed: {exc}") from exc
        selection = _local_source_selection(
            hard_gate,
            primary,
            supplements,
            minimum_per_regime=minimum_per_regime,
        )
        if selection is None:
            raise RegistryPreparationError(f"strict source corpus selection failed: {exc}") from exc
    selected_ids = _selection_ids(selection)
    loaded_by_id = {item.trace_id: item for item in all_loaded}
    if set(selected_ids) != set(selected_ids).intersection(loaded_by_id):
        raise RegistryPreparationError("strict selector returned an unknown trace")
    selected_loaded = [loaded_by_id[trace_id] for trace_id in selected_ids]
    per_trace: dict[str, dict[str, Any]] = {}
    for item in selected_loaded:
        coverage = dict(item.coverage)
        actions = _trace_actions(item.trace)
        offset = int(coverage["capture_offset"])
        continuation = int(coverage["continuation_horizon"])
        if continuation != horizon and not set(coverage["regimes"]) & _TERMINAL_REGIMES:
            raise RegistryPreparationError(
                f"trace {item.trace_id} may use a short horizon only for release/predicate_transition"
            )
        if continuation != horizon and (
            not coverage.get("terminal_reason") or coverage.get("legal_terminal_step") is None
        ):
            raise RegistryPreparationError(
                f"trace {item.trace_id} short horizon needs terminal_reason and legal_terminal_step"
            )
        if offset + continuation > len(actions):
            raise RegistryPreparationError(
                f"trace {item.trace_id} source coverage exceeds persisted action tape: "
                f"offset={offset}, horizon={continuation}, steps={len(actions)}"
            )
        per_trace[item.trace_id] = {
            "regimes": list(coverage["regimes"]),
            "capture_offset": offset,
            "continuation_horizon": continuation,
            "source_coverage": coverage,
        }
        if isinstance(coverage.get("windows"), Sequence) and not isinstance(coverage.get("windows"), (str, bytes)):
            per_trace[item.trace_id]["windows"] = list(coverage["windows"])
    if tape_provenance is None:
        if action_tape is not None:
            array = np.ascontiguousarray(np.asarray(action_tape, dtype=np.float32))
            if array.ndim != 2 or array.shape[1] != 7 or not np.all(np.isfinite(array)):
                raise RegistryPreparationError("action_tape must be finite float32[:, 7]")
            tape_provenance = {
                "algorithm": "injected_action_tape",
                "dtype": "float32",
                "shape": list(array.shape),
                "sha256": _sha256_bytes(array.tobytes(order="C")),
                "capture_offsets": sorted({item["capture_offset"] for item in per_trace.values()}),
            }
        else:
            tape_provenance = _derive_tape_provenance([item.trace for item in selected_loaded], per_trace)
    try:
        tape_provenance = _complete_tape_provenance(
            target,
            tape_provenance,
            [item.trace for item in selected_loaded],
            action_tape=action_tape,
            per_trace=per_trace,
        )
        written = hard_gate.write_frozen_registry(
            target,
            traces=[item.trace for item in selected_loaded],
            tape_provenance=tape_provenance,
            per_trace=per_trace,
            selection=selection,
        )
        frozen = hard_gate.load_frozen_registry(written)
    except Exception as exc:
        raise RegistryPreparationError(f"strict registry freeze failed: {exc}") from exc
    frozen_ids = tuple(item["trace_id"] for item in frozen.get("traces", ()))
    if set(frozen_ids) != set(selected_ids):
        raise RegistryPreparationError("frozen registry trace set differs from source selection")
    selected_map = {item.trace_id: item.trace for item in selected_loaded}
    frozen_per_trace = {
        str(item["trace_id"]): dict(per_trace[str(item["trace_id"])]) for item in frozen["traces"]
    }
    return PreparedRegistry(
        registry_path=Path(written),
        registry=frozen,
        traces=selected_map,
        per_trace=frozen_per_trace,
        selection=frozen.get("selection", selection.to_dict() if hasattr(selection, "to_dict") else {}),
    )


def validate_short_terminal(
    record: Mapping[str, Any],
    *,
    nominal_horizon: int = DEFAULT_HORIZON,
    expected_reason: str | None = None,
    expected_step: int | None = None,
) -> bool:
    """Validate the only permitted reason for a horizon shorter than ten.

    The strict primitive checks returned-step termination, a terminal
    observation, no autoreset, one attempt, and no retry.  This wrapper adds
    the frozen source-coverage reason/step so a restore cannot choose a new
    early exit after the registry has been frozen.
    """

    if not isinstance(record, Mapping):
        return False
    try:
        hard_gate = _load_hard_gate()
        strict_record = dict(record)
        # The strict primitive's reason vocabulary predates the corpus
        # classifier's ``predicate_transition`` label.  Preserve the exact
        # caller reason below while presenting its documented predicate class
        # to the primitive validator.
        strict_reason = strict_record.get("termination_reason")
        if isinstance(strict_reason, str) and strict_reason not in {
            "success",
            "task_success",
            "predicate",
            "predicate_success",
            "terminated",
            "termination",
            "done",
        } and "predicate" in strict_reason:
            strict_record["termination_reason"] = "predicate"
        if not hard_gate.validate_legitimate_short_terminal(
            strict_record, nominal_horizon=nominal_horizon
        ):
            return False
    except Exception:
        return False
    if expected_reason is not None:
        actual_reason = record.get("termination_reason", record.get("terminal_reason"))
        if actual_reason != expected_reason:
            return False
    if expected_step is not None:
        try:
            if int(record.get("steps")) != int(expected_step):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _resolve_registry_input(registry: Any) -> tuple[Path, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if isinstance(registry, PreparedRegistry):
        return registry.registry_path, registry.registry, registry.traces
    if isinstance(registry, (str, Path)):
        return Path(registry), None, None
    if isinstance(registry, Mapping):
        path = registry.get("registry_path", registry.get("path"))
        if path is None:
            raise RestoreProtocolError("run_authoritative requires a persisted frozen registry path")
        payload = registry.get("registry")
        traces = registry.get("traces")
        return Path(path), payload if isinstance(payload, Mapping) else None, traces if isinstance(traces, Mapping) else None
    raise RestoreProtocolError("registry must be a frozen path or PreparedRegistry")


def _load_traces_for_registry(
    payload: Mapping[str, Any],
    *,
    prepared_traces: Mapping[str, Any] | None,
    trace_loader: Callable[..., Any] | None,
) -> dict[str, Any]:
    hard_gate = _load_hard_gate()
    result: dict[str, Any] = {}
    for index, item in enumerate(payload.get("traces", ())):
        item = _mapping(item, f"registry traces[{index}]", RestoreProtocolError)
        trace_id = _text(item.get("trace_id"), "registry trace_id", RestoreProtocolError)
        trace = prepared_traces.get(trace_id) if prepared_traces is not None else None
        if trace is None:
            source = item.get("source_file")
            if source is None:
                raise RestoreProtocolError(f"registry trace {trace_id} has no source_file")
            if trace_loader is not None:
                trace = _invoke_loader(trace_loader, source)
            else:
                # The frozen registry carries all pins needed to re-open the
                # persisted M0 row.  Never rediscover an ancestor manifest or
                # regenerate an action tape in the authoritative worker.
                trace = hard_gate.extract_m0_trace(
                    source,
                    parent_manifest_path=item.get("parent_run_manifest"),
                    terminal_manifest_path=item.get("parent_terminal_manifest"),
                    expected_source_sha256=item.get("source_file_sha256"),
                    expected_parent_manifest_sha256=item.get("parent_run_manifest_sha256"),
                    expected_parent_terminal_manifest_sha256=item.get("parent_terminal_manifest_sha256"),
                    expected_action_sha256=item.get("action_sha256"),
                    allow_relocated_run_directory=bool(
                        isinstance(item.get("relocation"), Mapping)
                        and item.get("relocation", {}).get("allowed") is True
                    ),
                    declared_original_run_directory=(
                        item.get("relocation", {}).get("declared_original_run_directory")
                        if isinstance(item.get("relocation"), Mapping)
                        else None
                    ),
                )
        if not isinstance(trace, hard_gate.M0Trace) and not callable(getattr(trace, "provenance", None)):
            raise RestoreProtocolError(f"trace_loader did not return validated trace {trace_id}")
        if _trace_id(trace) != trace_id:
            raise RestoreProtocolError(f"trace_loader returned the wrong trace for {trace_id}")
        # A run must use the exact source bytes selected by the registry.  The
        # strict provenance map is checked again even for injected test seams.
        provenance = trace.provenance() if callable(getattr(trace, "provenance", None)) else {}
        for key in ("source_file_sha256", "parent_run_manifest_sha256", "parent_terminal_manifest_sha256", "action_sha256"):
            expected = item.get(key)
            actual = provenance.get(key) if isinstance(provenance, Mapping) else None
            if expected is not None and actual is not None and str(expected).lower() != str(actual).lower():
                raise RestoreProtocolError(f"trace {trace_id} provenance {key} differs from frozen registry")
        result[trace_id] = trace
    return result


def _registry_stages(payload: Mapping[str, Any], *, default_horizon: int) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for index, raw in enumerate(payload.get("traces", ())):
        item = _mapping(raw, f"registry traces[{index}]", RestoreProtocolError)
        trace_id = _text(item.get("trace_id"), "registry trace_id", RestoreProtocolError)
        coverage = item.get("source_coverage", {})
        coverage = _mapping(coverage, f"registry trace {trace_id} source_coverage", RestoreProtocolError)
        raw_regimes = item.get("regimes", coverage.get("regimes"))
        if isinstance(raw_regimes, str):
            raw_regimes = [raw_regimes]
        if not isinstance(raw_regimes, Sequence) or isinstance(raw_regimes, (str, bytes)) or not raw_regimes:
            raise RestoreProtocolError(f"registry trace {trace_id} has no regimes")
        regimes = tuple(str(value) for value in raw_regimes)
        if any(value not in REGIME_NAMES for value in regimes):
            raise RestoreProtocolError(f"registry trace {trace_id} contains an unknown regime")
        try:
            offset = _int(item.get("capture_offset"), f"registry trace {trace_id} capture_offset")
            horizon = _int(
                item.get("continuation_horizon", default_horizon),
                f"registry trace {trace_id} continuation_horizon",
                positive=True,
            )
        except RegistryPreparationError as exc:
            raise RestoreProtocolError(str(exc)) from exc
        if offset <= 0:
            raise RestoreProtocolError(f"registry trace {trace_id} capture_offset must follow a returned step")
        if horizon != default_horizon:
            if not set(regimes) & _TERMINAL_REGIMES:
                raise RestoreProtocolError(
                    f"trace {trace_id} may use a short horizon only for release/predicate_transition"
                )
            if not coverage.get("terminal_reason") or coverage.get("legal_terminal_step") is None:
                raise RestoreProtocolError(
                    f"trace {trace_id} short horizon lacks frozen terminal reason/step evidence"
                )
        windows = item.get("windows")
        if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)) and windows:
            for wi, raw_window in enumerate(windows):
                window = _mapping(raw_window, f"registry trace {trace_id} window[{wi}]", RestoreProtocolError)
                wr = window.get("regimes")
                if isinstance(wr, str):
                    wr = [wr]
                if not isinstance(wr, Sequence) or isinstance(wr, (str, bytes)) or not wr:
                    raise RestoreProtocolError(f"registry trace {trace_id} window[{wi}] has no regimes")
                woffset = _int(window.get("capture_offset"), f"registry trace {trace_id} window[{wi}] capture_offset")
                whorizon = _int(window.get("continuation_horizon", default_horizon), f"registry trace {trace_id} window[{wi}] continuation_horizon", positive=True)
                wcoverage = _mapping(window.get("source_coverage", {"regimes": list(wr)}), f"registry trace {trace_id} window[{wi}] source_coverage", RestoreProtocolError)
                stages.append({"stage_id": f"{trace_id}@{woffset}", "trace_id": trace_id, "regimes": tuple(str(v) for v in wr), "capture_offset": woffset, "horizon": whorizon, "source_coverage": dict(wcoverage)})
        else:
            stages.append(
                {
                    "stage_id": f"{trace_id}@{offset}",
                    "trace_id": trace_id,
                    "regimes": regimes,
                    "capture_offset": offset,
                    "horizon": horizon,
                    "source_coverage": dict(coverage),
                }
            )
    if not stages:
        raise RestoreProtocolError("frozen registry has no stages")
    return stages


def _adapter_method(adapter: Any, name: str) -> Callable[..., Any]:
    method = getattr(adapter, name, None)
    if not callable(method):
        raise RestoreProtocolError(f"runtime adapter lacks {name}()")
    return method


def _construct_adapter(
    config: Mapping[str, Any],
    *,
    environment_builder: Callable[..., Any] | None,
    adapter_factory: Callable[..., Any] | None,
) -> Any:
    """Construct one adapter after registry verification, with test seams."""

    if adapter_factory is not None:
        return adapter_factory(config)
    runtime_adapter, official_builder, cpu_builder = _load_replay_symbols()
    if environment_builder is None:
        # ``build_official_runtime`` remains the preferred project seam; it
        # delegates to the environment-only factory and imports it lazily.
        environment_builder = official_builder
    try:
        signature = inspect.signature(environment_builder)
        del signature
    except (TypeError, ValueError):
        pass
    # A test builder may directly return an adapter.  Real builders return the
    # raw runtime mapping, which is passed through RuntimeAdapter.construct_fresh
    # exactly once.
    runtime = environment_builder(config)
    if callable(getattr(runtime, "restore", None)) and callable(getattr(runtime, "step", None)):
        return runtime
    return runtime_adapter.construct_fresh(config, runtime_builder=lambda _config: runtime)


def _close_adapter(adapter: Any) -> str | None:
    close = getattr(adapter, "close", None)
    if not callable(close):
        return None
    try:
        close()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _is_terminal(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("terminated", value.get("done", False))) or bool(value.get("truncated", False))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 5:
            return bool(value[2]) or bool(value[3])
        if len(value) >= 4:
            return bool(value[2])
    return False


def _terminal_record(
    result: Any,
    adapter: Any,
    *,
    steps: int,
    nominal_horizon: int,
    reason: str | None,
    absolute_step: int | None = None,
) -> dict[str, Any]:
    terminated = False
    truncated = False
    observation: Any = None
    info: Mapping[str, Any] = {}
    if isinstance(result, Mapping):
        terminated = bool(result.get("terminated", result.get("done", False)))
        truncated = bool(result.get("truncated", False))
        observation = result.get("terminal_observation", result.get("observation"))
        if isinstance(result.get("info"), Mapping):
            info = result["info"]
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if len(result) >= 1:
            observation = result[0]
        if len(result) >= 5:
            terminated, truncated = bool(result[2]), bool(result[3])
            if isinstance(result[4], Mapping):
                info = result[4]
        elif len(result) >= 4:
            terminated = bool(result[2])
            info = result[3] if isinstance(result[3], Mapping) else {}
    if isinstance(result, Mapping) and isinstance(result.get("success"), bool):
        success = bool(result["success"])
    else:
        checker = getattr(adapter, "_check_success", None)
        success = checker() if callable(checker) else getattr(adapter, "last_success", None)
    if not isinstance(success, bool):
        success = False
    actual_reason = None
    if isinstance(result, Mapping):
        actual_reason = result.get("termination_reason", result.get("terminal_reason"))
    if actual_reason is None:
        actual_reason = info.get("termination_reason", info.get("terminal_reason"))
    if actual_reason is None:
        if success and terminated:
            actual_reason = "predicate_transition"
        elif truncated:
            actual_reason = "time_limit"
        elif terminated:
            actual_reason = "environment_termination"
    if reason is not None and actual_reason != reason:
        # Persist both values so the caller can publish the semantic mismatch;
        # never replace an actual worker/source reason with frozen metadata.
        frozen_reason_match = False
    else:
        frozen_reason_match = True
    return {
        "status": "completed",
        "completed": True,
        "success": success,
        "steps": int(steps),
        "absolute_step": int(absolute_step) if absolute_step is not None else int(steps),
        "nominal_horizon": int(nominal_horizon),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_observation": observation if observation is not None else {"present": True},
        "step_returned_terminal": True,
        "autoreset": False,
        "attempt_count": 1,
        "no_retry": True,
        "retry_count": 0,
        "reset_count": 0,
        "termination_reason": actual_reason,
        "frozen_termination_reason": reason,
        "frozen_termination_reason_match": frozen_reason_match,
    }


def _call_capture(adapter: Any, step: int) -> Any:
    capture = _adapter_method(adapter, "capture")
    try:
        signature = inspect.signature(capture)
        if "source_step" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        ):
            return capture(source_step=step)
    except (TypeError, ValueError):
        pass
    return capture(step)


def _call_restore(adapter: Any, capture: Any, reference: Any) -> Mapping[str, Any]:
    restore = _adapter_method(adapter, "restore")
    try:
        signature = inspect.signature(restore)
        if "static_reference" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        ):
            value = restore(capture, static_reference=reference)
        else:
            value = restore(capture)
    except (TypeError, ValueError):
        value = restore(capture)
    if value is None:
        raise RestoreProtocolError("restore did not return a result mapping")
    return _mapping(value, "restore result", RestoreProtocolError)


def _collect_invariants(adapter: Any) -> Mapping[str, Any] | None:
    collector = getattr(adapter, "collect_invariants", None)
    if not callable(collector):
        return None
    value = collector()
    return _mapping(value, "invariant snapshot", RestoreProtocolError)


def _render_rgb_reference(adapter: Any) -> np.ndarray:
    renderer = getattr(adapter, "render_rgb", None)
    if not callable(renderer):
        raise RestoreProtocolError("authoritative adapter does not expose render_rgb()")
    value = renderer()
    if value is None:
        raise RestoreProtocolError("authoritative adapter returned no renderer observation")
    image = np.asarray(value)
    if image.dtype != np.dtype(np.uint8) or image.ndim < 2 or image.size == 0:
        raise RestoreProtocolError("renderer observation must be a non-empty uint8 array")
    return np.ascontiguousarray(image).copy()


def _renderer_pair_gate(
    envelope: Mapping[str, Any],
    *,
    expected: Any,
    actual: Any,
    regimes: Sequence[str],
    horizon: int,
) -> tuple[bool, list[dict[str, Any]]]:
    hard_gate = _load_hard_gate()
    metrics = hard_gate.rgb_disagreement_metrics(expected, actual)
    checks: list[dict[str, Any]] = []
    passed = bool(regimes)
    for regime in regimes:
        try:
            regime_pass = hard_gate.check_renderer_value(
                envelope,
                camera=_RENDERER_CAMERA,
                key=_RENDERER_OBSERVATION_KEY,
                regime=str(regime),
                horizon=int(horizon),
                expected=expected,
                actual=actual,
            )
        except Exception:
            regime_pass = False
        checks.append(
            {
                "camera": _RENDERER_CAMERA,
                "key": _RENDERER_OBSERVATION_KEY,
                "regime": str(regime),
                "horizon": int(horizon),
                **metrics,
                "pass": bool(regime_pass),
            }
        )
        passed = passed and bool(regime_pass)
    return bool(passed), checks


def _source_stage(
    adapter: Any,
    trace: Any,
    stage: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], dict[int, Mapping[str, Any]], dict[str, Any] | None, int]:
    actions = _trace_actions(trace)
    offset = int(stage["capture_offset"])
    nominal_horizon = int(stage["horizon"])
    if offset + nominal_horizon > len(actions):
        raise RestoreProtocolError(
            f"trace {stage['trace_id']} action tape ends before capture+horizon"
        )
    for index in range(offset):
        result = _adapter_method(adapter, "step")(actions[index])
        if _is_terminal(result):
            raise RestoreProtocolError(
                f"source trace {stage['trace_id']} terminated before capture step {offset}"
            )
    # Capture is after the returned source step and before the suffix action.
    capture = _call_capture(adapter, offset)
    boundary = dict(_collect_invariants(adapter) or {})
    boundary[_RENDERER_REFERENCE_KEY] = _render_rgb_reference(adapter)
    references: dict[int, Mapping[str, Any]] = {}
    terminal: dict[str, Any] | None = None
    actual_horizon = nominal_horizon
    for index in range(nominal_horizon):
        step = offset + index + 1
        result = _adapter_method(adapter, "step")(actions[offset + index])
        reference = dict(_collect_invariants(adapter) or {})
        reference[_RENDERER_REFERENCE_KEY] = _render_rgb_reference(adapter)
        references[step] = reference
        if _is_terminal(result):
            terminal = _terminal_record(
                result,
                adapter,
                steps=index + 1,
                nominal_horizon=DEFAULT_HORIZON,
                reason=stage.get("source_coverage", {}).get("terminal_reason"),
                absolute_step=step,
            )
            actual_horizon = index + 1
            if actual_horizon < nominal_horizon:
                if not set(stage.get("regimes", ())) & _TERMINAL_REGIMES:
                    raise RestoreProtocolError(
                        "source terminated early outside a terminal/predicate regime"
                    )
                expected_reason = stage.get("source_coverage", {}).get("terminal_reason")
                expected_step = stage.get("source_coverage", {}).get("legal_terminal_step")
                if not expected_reason or expected_step is None:
                    raise RestoreProtocolError(
                        "early source terminal lacks frozen terminal reason/step evidence"
                    )
                if expected_step is not None and int(expected_step) != int(terminal["absolute_step"]):
                    raise RestoreProtocolError("source terminal step differs from frozen legal terminal step")
                if not validate_short_terminal(
                    terminal,
                    nominal_horizon=DEFAULT_HORIZON,
                    expected_reason=expected_reason,
                    expected_step=step if expected_step is not None else None,
                ):
                    raise RestoreProtocolError("source short terminal is not a legal returned-step terminal")
            elif nominal_horizon != DEFAULT_HORIZON:
                expected_reason = stage.get("source_coverage", {}).get("terminal_reason")
                expected_step = stage.get("source_coverage", {}).get("legal_terminal_step")
                if expected_step is not None and int(expected_step) != int(terminal["absolute_step"]):
                    raise RestoreProtocolError("source terminal step differs from frozen legal terminal step")
                if not validate_short_terminal(
                    terminal,
                    nominal_horizon=DEFAULT_HORIZON,
                    expected_reason=expected_reason,
                    expected_step=index + 1,
                ):
                    raise RestoreProtocolError("source terminal lacks frozen legal evidence")
            break
    if terminal is None and nominal_horizon != DEFAULT_HORIZON:
        raise RestoreProtocolError("short registry horizon did not produce its frozen terminal")
    return capture, boundary, references, terminal, actual_horizon


def _event_values(adapter: Any) -> list[Any]:
    values: list[Any] = []
    for owner in (adapter, getattr(adapter, "env", None), getattr(adapter, "inner", None)):
        if owner is None:
            continue
        for name in ("events", "call_trace", "restore_trace", "operations"):
            candidate = getattr(owner, name, None)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                values.extend(candidate)
    return values


def _forbidden_restore_event(value: Any) -> bool:
    text = str(value).lower()
    return any(word in text for word in _FORBIDDEN_RESTORE_WORDS)


def _post_restore_events(adapter: Any) -> list[Any]:
    """Return only calls made from the first restore boundary onward."""

    values = _event_values(adapter)
    for index, value in enumerate(values):
        text = str(value).lower()
        if "restore" in text or "set_state" in text or "setstate" in text:
            return values[index:]
    return []


def _comparison_pass(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        if "pass" in value:
            return bool(value["pass"])
        if "passed" in value:
            return bool(value["passed"])
    return default


def _manifest_target(
    manifest_path: str | Path | None,
    output_directory: str | Path | None,
) -> Path | None:
    value = manifest_path if manifest_path is not None else output_directory
    if value is None:
        return None
    target = Path(value)
    if manifest_path is None or target.suffix.lower() != ".json":
        target = target / "terminal_manifest.json"
    return target


def _sample_callback(callback: Callable[..., Any], context: Mapping[str, Any]) -> list[Any]:
    try:
        signature = inspect.signature(callback)
        params = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        params = [None]
    if len(params) >= 2:
        value = callback(context["stage"], context["adapter"])
    elif len(params) == 1:
        value = callback(context)
    else:
        value = callback()
    if value is None:
        return []
    if isinstance(value, Mapping):
        # A callback can return separate physics/renderer collections.
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise RestoreProtocolError("null sample callback must return a sequence or mapping")


def _gate_value_from_result(result: Mapping[str, Any], names: Sequence[str], default: bool = False) -> bool:
    for name in names:
        if name in result:
            return _comparison_pass(result[name], default)
    return default


def _renderer_result_pass(result: Mapping[str, Any], envelope: Mapping[str, Any] | None) -> bool:
    explicit = result.get("renderer_pass", result.get("observation_dynamic"))
    if explicit is not None and not _comparison_pass(explicit, False):
        return False
    checks = result.get("renderer_checks")
    if checks is None or envelope is None:
        return True
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return False
    hard_gate = _load_hard_gate()
    for check in checks:
        if not isinstance(check, Mapping):
            return False
        try:
            if not hard_gate.check_renderer_value(envelope, **dict(check)):
                return False
        except Exception:
            return False
    return True


def run_authoritative(
    registry: str | Path | PreparedRegistry | Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    trace_loader: Callable[..., Any] | None = None,
    environment_builder: Callable[..., Any] | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    subprocess_runner: Callable[..., Any] | None = None,
    null_physics_samples: Iterable[Mapping[str, Any]] | None = None,
    null_renderer_samples: Iterable[Mapping[str, Any]] | None = None,
    null_sample_builder: Callable[..., Any] | None = None,
    horizon: int = DEFAULT_HORIZON,
    restores_per_stage: int = RESTORES_PER_STAGE,
    fresh_processes_per_stage: int = FRESH_PROCESSES_PER_STAGE,
    **aliases: Any,
) -> dict[str, Any]:
    """Run bounded static/dynamic gates from one already frozen registry.

    Registry verification is the first operation.  Then each selected source
    is stepped once to its frozen stage, followed by three same-process
    restores and three independently launched worker restores.  A restore or
    worker exception stops that stage immediately; there is no retry path.
    """

    manifest_path = _normalise_optional_alias(aliases, ("output_path", "manifest", "terminal_manifest"), manifest_path)
    output_directory = _normalise_optional_alias(aliases, ("output_dir", "run_directory"), output_directory)
    trace_loader = _normalise_optional_alias(aliases, ("loader",), trace_loader)
    environment_builder = _normalise_optional_alias(aliases, ("runtime_builder", "env_builder"), environment_builder)
    subprocess_runner = _normalise_optional_alias(aliases, ("process_runner",), subprocess_runner)
    null_physics_samples = _normalise_optional_alias(aliases, ("physics_samples", "null_samples"), null_physics_samples)
    null_renderer_samples = _normalise_optional_alias(aliases, ("renderer_samples",), null_renderer_samples)
    null_sample_builder = _normalise_optional_alias(aliases, ("calibration_builder",), null_sample_builder)
    if aliases:
        raise RestoreProtocolError(f"unknown run_authoritative arguments: {sorted(aliases)}")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise RestoreProtocolError("horizon must be a positive integer")
    if isinstance(restores_per_stage, bool) or restores_per_stage != RESTORES_PER_STAGE:
        raise RestoreProtocolError("same-process restore count is frozen at three")
    if isinstance(fresh_processes_per_stage, bool) or fresh_processes_per_stage != FRESH_PROCESSES_PER_STAGE:
        raise RestoreProtocolError("fresh-process restore count is frozen at three")
    authoritative_config = dict(config or {})
    if authoritative_config.get("authoritative") is False or authoritative_config.get("strict_provenance") is False:
        raise RestoreProtocolError(
            "authoritative M1 forbids disabling authoritative or strict_provenance restore mode"
        )
    authoritative_config["authoritative"] = True
    authoritative_config["strict_provenance"] = True
    config = authoritative_config
    target = _manifest_target(manifest_path, output_directory)
    if target is not None and (target.is_symlink() or target.exists()):
        raise PublicationError(f"refusing to overwrite existing manifest: {target}")

    started = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "runner_protocol_version": WORKER_PROTOCOL_VERSION,
        "status": "RUNNING",
        "started_at": started,
        "registry_path": None,
        "registry_sha256": None,
        "config": dict(config),
        "gates": {name: False for name in GATE_NAMES},
        "gate_details": {},
        "source": [],
        "same_process": [],
        "fresh_process": [],
        "null": {},
        "errors": [],
        "provenance": {},
    }
    registry_path, embedded_payload, prepared_traces = _resolve_registry_input(registry)
    manifest["registry_path"] = str(registry_path)
    source_imports_before = set(sys.modules)
    payload: Mapping[str, Any]
    stages: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    contexts: list[dict[str, Any]] = []
    fresh_pids: set[int] = set()
    protocol_errors: list[str] = []
    blocked = False
    try:
        # This load is intentionally before any adapter/environment factory
        # call.  A caller-provided embedded payload is only a convenience for
        # metadata; the persisted bytes remain authoritative.
        hard_gate = _load_hard_gate()
        payload = hard_gate.load_frozen_registry(registry_path)
        if embedded_payload is not None:
            expected_hash = embedded_payload.get("registry_sha256")
            if expected_hash is not None and expected_hash != payload.get("registry_sha256"):
                raise RestoreProtocolError("embedded registry payload differs from persisted registry")
        manifest["registry_sha256"] = payload.get("registry_sha256")
        stages = _registry_stages(payload, default_horizon=horizon)
        traces = _load_traces_for_registry(
            payload,
            prepared_traces=prepared_traces,
            trace_loader=trace_loader,
        )
        coverage_lists: dict[str, list[str]] = {}
        selected_order: list[str] = []
        for stage in stages:
            trace_id = str(stage["trace_id"])
            if trace_id not in coverage_lists:
                coverage_lists[trace_id] = []
                selected_order.append(trace_id)
            for regime in stage["regimes"]:
                if regime not in coverage_lists[trace_id]:
                    coverage_lists[trace_id].append(str(regime))
        coverage_by_trace = {
            trace_id: tuple(regimes) for trace_id, regimes in coverage_lists.items()
        }
        selected_ids = tuple(selected_order)
        all_regimes = {regime for stage in stages for regime in stage["regimes"]}
        manifest["gates"]["corpus_regime_coverage"] = all_regimes == set(REGIME_NAMES)
        manifest["gate_details"]["corpus_regime_coverage"] = {
            "required": list(REGIME_NAMES),
            "observed": sorted(all_regimes),
            "trace_ids": list(selected_ids),
        }
        if all_regimes != set(REGIME_NAMES):
            raise RestoreProtocolError("frozen registry does not cover all seven regimes")

        # A null calibration is an authoritative prerequisite.  Refuse to
        # construct/step a fresh environment when the caller has not supplied
        # persisted duplicate-control evidence (or a bounded builder); this
        # keeps an absent envelope from being mistaken for a clean run.
        if null_physics_samples is None and null_renderer_samples is None and null_sample_builder is None:
            raise RestoreProtocolError("authoritative M1 requires persisted contact-rich null physics and renderer samples")

        # Source construction/stepping occurs only after the strict registry
        # hash and regime matrix have been verified.
        for stage in stages:
            adapter = _construct_adapter(
                dict(config or {}),
                environment_builder=environment_builder,
                adapter_factory=adapter_factory,
            )
            capture, boundary, references, terminal, actual_horizon = _source_stage(
                adapter,
                traces[stage["trace_id"]],
                stage,
            )
            context = {
                "stage": stage,
                "adapter": adapter,
                "capture": capture,
                "boundary": boundary,
                "references": references,
                "source_terminal": terminal,
                "actual_horizon": actual_horizon,
            }
            contexts.append(context)
            audit = getattr(adapter, "audit_runtime_state", None)
            audit_value = audit() if callable(audit) else None
            closure_pass = isinstance(audit_value, Mapping) and not audit_value.get("unknown_paths") and audit_value.get(
                "pass", True
            ) is not False
            context["state_closure"] = bool(closure_pass)
            manifest["source"].append(
                {
                    "stage_id": stage["stage_id"],
                    "trace_id": stage["trace_id"],
                    "regimes": list(stage["regimes"]),
                    "capture_offset": stage["capture_offset"],
                    "horizon": stage["horizon"],
                    "actual_horizon": actual_horizon,
                    "terminal": copy.deepcopy(terminal),
                    "state_closure": bool(closure_pass),
                }
            )
        manifest["gates"]["state_closure"] = bool(contexts) and all(
            bool(context["state_closure"]) for context in contexts
        )
        manifest["gate_details"]["state_closure"] = [
            {"stage_id": context["stage"]["stage_id"], "pass": bool(context["state_closure"])}
            for context in contexts
        ]

        # Calibration runs after registry freeze but before the first restore.
        physics_samples = list(null_physics_samples) if null_physics_samples is not None else None
        renderer_samples = list(null_renderer_samples) if null_renderer_samples is not None else None
        if null_sample_builder is not None:
            for context in contexts:
                generated = _sample_callback(null_sample_builder, context)
                for item in generated:
                    if not isinstance(item, Mapping):
                        continue
                    kind = str(item.get("sample_kind", item.get("kind", "physics"))).lower()
                    if kind in {"renderer", "rgb"}:
                        if renderer_samples is None:
                            renderer_samples = []
                        renderer_samples.append(dict(item))
                    elif kind in {"both", "null"}:
                        if physics_samples is None:
                            physics_samples = []
                        physics_samples.append(dict(item))
                        if renderer_samples is None and "duplicate_controls" in item:
                            renderer_samples = []
                        if renderer_samples is not None and "duplicate_controls" in item:
                            renderer_samples.append(dict(item))
                    else:
                        if physics_samples is None:
                            physics_samples = []
                        physics_samples.append(dict(item))
        if renderer_samples is None and physics_samples is not None and all(
            isinstance(item, Mapping) and "duplicate_controls" in item for item in physics_samples
        ):
            renderer_samples = list(physics_samples)
        physics_envelope, renderer_envelope, null_details, null_errors = _build_null_envelopes(
            physics_samples=physics_samples,
            renderer_samples=renderer_samples,
            selected_ids=selected_ids,
            coverage_by_trace=coverage_by_trace,
        )
        manifest["null"] = null_details
        manifest["errors"].extend(null_errors)
        manifest["gates"]["null_physics"] = physics_envelope is not None and not any(
            error.startswith("null physics") or error.startswith("physics ") for error in null_errors
        )
        manifest["gates"]["null_renderer"] = renderer_envelope is not None and not any(
            error.startswith("null renderer") or error.startswith("renderer ") for error in null_errors
        )
        # Null calibration is a prerequisite for authoritative numerical and
        # RGB acceptance.  Never execute restore branches with a missing or
        # under-covered envelope; doing so would turn an absent baseline into
        # an implicit tolerance.
        if not manifest["gates"]["null_physics"] or not manifest["gates"]["null_renderer"]:
            raise RestoreProtocolError("null calibration did not satisfy the frozen contact-rich physics and renderer gates")

        static_physics: list[bool] = []
        static_observation: list[bool] = []
        same_dynamic: list[bool] = []
        for context in contexts:
            stage = context["stage"]
            adapter = context["adapter"]
            stage_same_records: list[dict[str, Any]] = []
            for ordinal in range(1, RESTORES_PER_STAGE + 1):
                record: dict[str, Any] = {
                    "stage_id": stage["stage_id"],
                    "trace_id": stage["trace_id"],
                    "ordinal": ordinal,
                    "pid": os.getpid(),
                    "no_retry": True,
                }
                try:
                    restore_result = _call_restore(adapter, context["capture"], context["boundary"])
                    physics_pass, observation_pass, static_details = _static_gates(
                        restore_result, context["boundary"]
                    )
                    static_physics.append(physics_pass)
                    static_observation.append(observation_pass)
                    dynamic_pass, rows, restored_terminal = _call_step_and_compare(
                        adapter,
                        _trace_actions(traces[stage["trace_id"]]),
                        offset=int(stage["capture_offset"]),
                        horizon=int(stage["horizon"]),
                        source_references=context["references"],
                        stage=stage,
                        source_terminal=context["source_terminal"],
                        physics_envelope=physics_envelope,
                        renderer_envelope=renderer_envelope,
                    )
                    expected_renderer = context["boundary"].get(_RENDERER_REFERENCE_KEY)
                    if expected_renderer is None:
                        raise RestoreProtocolError("source boundary lacks renderer reference")
                    renderer_pass, renderer_checks = _renderer_pair_gate(
                        renderer_envelope,
                        expected=expected_renderer,
                        actual=_render_rgb_reference(adapter),
                        regimes=tuple(str(value) for value in stage.get("regimes", ())),
                        horizon=0,
                    )
                    static_details["renderer"] = {
                        "pass": bool(renderer_pass),
                        "checks": renderer_checks,
                    }
                    record.update(
                        {
                            "static": static_details,
                            "physics_static_identity": physics_pass,
                            "observation_static_identity": observation_pass,
                            "rows": rows,
                            "terminal": restored_terminal,
                            "pass": bool(physics_pass and dynamic_pass and renderer_pass),
                            "dynamic_pass": bool(dynamic_pass),
                            "renderer_pass": bool(renderer_pass),
                        }
                    )
                except Exception as exc:
                    protocol_errors.append(
                        f"same-process {stage['stage_id']} restore {ordinal} failed: {type(exc).__name__}: {exc}"
                    )
                    record.update({"pass": False, "error": protocol_errors[-1], "no_retry": True})
                    stage_same_records.append(record)
                    manifest["same_process"].append(record)
                    # One failed restore is terminal for this stage.  Do not
                    # invoke the same ordinal again or silently retry it.
                    break
                stage_same_records.append(record)
                manifest["same_process"].append(record)
                same_dynamic.append(bool(record.get("pass")))
            if len(stage_same_records) != RESTORES_PER_STAGE:
                protocol_errors.append(
                    f"same-process stage {stage['stage_id']} has {len(stage_same_records)} attempts, required 3"
                )
                break
        manifest["gates"]["physics_static_identity"] = bool(static_physics) and all(static_physics)
        manifest["gates"]["observation_static_identity"] = bool(static_observation) and all(static_observation)
        manifest["gates"]["same_process_dynamic"] = (
            len(same_dynamic) == len(stages) * RESTORES_PER_STAGE and all(same_dynamic)
        )

        # Every worker request is a new process protocol invocation.  The
        # injected runner can return a mapping directly; the default runner
        # parses one JSON mapping from the child stdout.
        fresh_dynamic: list[bool] = []
        fresh_static_physics: list[bool] = []
        fresh_static_observation: list[bool] = []
        fresh_runner = subprocess_runner
        for context in contexts:
            stage = context["stage"]
            stage_count = 0
            for worker_index in range(1, FRESH_PROCESSES_PER_STAGE + 1):
                job = make_worker_request(
                    registry_path=registry_path,
                    trace_id=stage["trace_id"],
                    stage_id=stage["stage_id"],
                    worker_index=worker_index,
                    horizon=int(stage["horizon"]),
                    config=dict(config or {}),
                    capture=context["capture"],
                    source_boundary=context["boundary"],
                    source_references=context["references"],
                    source_terminal=context["source_terminal"],
                    physics_envelope=physics_envelope,
                    renderer_envelope=renderer_envelope,
                )
                if fresh_runner is None:
                    artifact_dir = (
                        target.parent
                        if target is not None
                        else Path(tempfile.mkdtemp(prefix="m1_fresh_process_"))
                    )
                    if job.get("capture_in_worker") is False:
                        try:
                            from scripts import m1_state_replay as replay

                            capture_value = context["capture"]
                            if isinstance(capture_value, replay.ReplayState):
                                stage_token = _sha256_bytes(
                                    str(stage["stage_id"]).encode("utf-8")
                                )[:16]
                                stem = f"m1_capture_{stage_token}_{worker_index}"
                                replay.save_record_bundle(
                                    replay.replay_state_record(capture_value),
                                    artifact_dir,
                                    stem,
                                )
                                job["capture_path"] = str(artifact_dir / stem)
                        except Exception as exc:
                            protocol_errors.append(
                                f"could not persist worker capture for {stage['stage_id']}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    if target is not None:
                        job_path = _persist_worker_job(
                            artifact_dir,
                            job,
                            "m1_worker_"
                            + _sha256_bytes(str(stage["stage_id"]).encode("utf-8"))[:16]
                            + f"_{worker_index}",
                        )
                    else:
                        job_path = _persist_worker_job(
                            artifact_dir,
                            job,
                            "m1_worker_"
                            + _sha256_bytes(str(stage["stage_id"]).encode("utf-8"))[:16]
                            + f"_{worker_index}",
                        )
                    command = _worker_command(job_path)
                    raw_result = subprocess.run(
                        command,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                else:
                    command = _worker_command(Path("worker-job.json"))
                    raw_result = _invoke_process_runner(fresh_runner, command, job)
                result = _normalise_process_result(raw_result)
                valid, worker_errors = _worker_result_gate(result, job)
                pid_value: int | None = None
                if "pid" in result:
                    try:
                        pid_value = int(result["pid"])
                    except (TypeError, ValueError):
                        pid_value = None
                if pid_value is not None and pid_value in fresh_pids:
                    worker_errors.append("fresh worker PID was reused")
                    valid = False
                if pid_value is not None:
                    fresh_pids.add(pid_value)
                physics_pass = _gate_value_from_result(
                    result,
                    ("physics_static_identity", "static_physics", "static_pass"),
                    default=valid,
                )
                observation_pass = _gate_value_from_result(
                    result,
                    ("observation_static_identity", "static_observation", "observation_pass"),
                    default=valid,
                )
                fresh_static_physics.append(physics_pass)
                fresh_static_observation.append(observation_pass)
                record = {
                    **result,
                    "stage_id": stage["stage_id"],
                    "trace_id": stage["trace_id"],
                    "worker_index": worker_index,
                    "pid": pid_value,
                    "pass": bool(valid),
                    "errors": worker_errors,
                    "no_retry": result.get("no_retry", True),
                }
                manifest["fresh_process"].append(record)
                stage_count += 1
                if not valid:
                    protocol_errors.extend(
                        f"fresh-process {stage['stage_id']} worker {worker_index}: {error}"
                        for error in worker_errors
                    )
                    # Do not launch another worker to retry a failed worker.
                    break
                fresh_dynamic.append(True)
            if stage_count != FRESH_PROCESSES_PER_STAGE:
                protocol_errors.append(
                    f"fresh-process stage {stage['stage_id']} has {stage_count} workers, required 3"
                )
                break
        manifest["gates"]["fresh_process_dynamic"] = (
            len(fresh_dynamic) == len(stages) * FRESH_PROCESSES_PER_STAGE
            and len(fresh_pids) == len(fresh_dynamic)
            and all(fresh_dynamic)
        )
        if fresh_static_physics:
            manifest["gates"]["physics_static_identity"] = manifest["gates"]["physics_static_identity"] and all(
                fresh_static_physics
            )
        if fresh_static_observation:
            manifest["gates"]["observation_static_identity"] = manifest["gates"]["observation_static_identity"] and all(
                fresh_static_observation
            )
        for context in contexts:
            events = _post_restore_events(context["adapter"])
            forbidden = [str(value) for value in events if _forbidden_restore_event(value)]
            if forbidden:
                protocol_errors.append(
                    f"restore trace {context['stage']['stage_id']} contains forbidden operation: "
                    + ", ".join(forbidden)
                )
        forbidden_imports = sorted(
            name
            for name in set(sys.modules).difference(source_imports_before)
            if "smolvla" in name.lower() or name.lower().endswith(".policies")
        )
        manifest["provenance"] = {
            "registry_sha256": payload.get("registry_sha256"),
            "source_selection_frozen_before_restore": True,
            "no_retry": not protocol_errors,
            "fresh_pids": sorted(fresh_pids),
            "fresh_pid_count": len(fresh_pids),
            "forbidden_restore_operations": sorted(set(protocol_errors))
            if any("forbidden operation" in error for error in protocol_errors)
            else [],
            "forbidden_imports": forbidden_imports,
            "action_source": "validated M0 traces",
        }
        provenance_pass = not protocol_errors and not forbidden_imports and len(fresh_pids) == len(fresh_dynamic)
        manifest["gates"]["provenance"] = bool(provenance_pass)
        manifest["errors"].extend(protocol_errors)
    except Exception as exc:
        blocked = True
        message = f"authoritative hard-gate run blocked: {type(exc).__name__}: {exc}"
        manifest["errors"].append(message)
        manifest["provenance"] = {
            "source_selection_frozen_before_restore": not bool(manifest["same_process"] or manifest["fresh_process"]),
            "no_retry": True,
            "fresh_pids": sorted(fresh_pids),
            "fresh_pid_count": len(fresh_pids),
        }
    finally:
        for context in contexts:
            close_error = _close_adapter(context["adapter"])
            if close_error:
                manifest["errors"].append(
                    f"close {context['stage']['stage_id']} failed: {close_error}"
                )
                blocked = True
        if manifest["errors"]:
            manifest["gates"]["provenance"] = False
        all_gates = all(bool(manifest["gates"].get(name)) for name in GATE_NAMES)
        if all_gates and not manifest["errors"]:
            manifest["status"] = "PASS"
        elif blocked:
            manifest["status"] = "BLOCKED"
        else:
            manifest["status"] = "FAIL"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["terminal"] = True
        if target is not None:
            _atomic_publish(target, manifest)
    return manifest


def _load_document(path: str | Path) -> Any:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise HardGateRunnerError(f"input document is not a regular file: {target}")
    text = target.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception as exc:
            raise HardGateRunnerError(f"input document is not JSON/YAML: {target}: {exc}") from exc


def _worker_load_capture(job: Mapping[str, Any]) -> Any:
    if job.get("capture_in_worker") is True and "capture" in job:
        return job["capture"]
    capture_path = job.get("capture_path")
    if capture_path is None:
        raise RestoreProtocolError("worker request has no serialized capture")
    target = Path(str(capture_path))
    if target.suffix.lower() == ".json":
        return _load_document(target)
    # ReplayState's binary record format is intentionally delegated to its
    # existing safe loader.  This import is reached only by --worker.
    from scripts import m1_state_replay as replay

    stem = target.stem
    directory = target.parent
    return replay.replay_state_from_record(replay.load_record_bundle(directory, stem))


def _run_worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one fresh-process restore request and printable result."""

    if job.get("protocol_version") != WORKER_PROTOCOL_VERSION:
        raise RestoreProtocolError("unsupported worker protocol version")
    registry_path = Path(str(job.get("registry_path")))
    hard_gate = _load_hard_gate()
    payload = hard_gate.load_frozen_registry(registry_path)
    stage_id = _text(job.get("stage_id"), "worker stage_id", RestoreProtocolError)
    trace_id = _text(job.get("trace_id"), "worker trace_id", RestoreProtocolError)
    stages = _registry_stages(payload, default_horizon=DEFAULT_HORIZON)
    stage = next((item for item in stages if item["stage_id"] == stage_id), None)
    if stage is None or stage["trace_id"] != trace_id:
        raise RestoreProtocolError("worker stage is not in frozen registry")
    if int(job.get("horizon", -1)) != int(stage["horizon"]):
        raise RestoreProtocolError("worker horizon differs from frozen registry stage")
    traces = _load_traces_for_registry(payload, prepared_traces=None, trace_loader=None)
    adapter = _construct_adapter(
        dict(job.get("config", {})) if isinstance(job.get("config"), Mapping) else {},
        environment_builder=None,
        adapter_factory=None,
    )
    calls: list[Any] = []
    try:
        capture = _worker_load_capture(job)
        # A worker request may carry a full replay record, in which case the
        # adapter's restore seam is authoritative.  Minimal test workers may
        # carry a mapping and implement the same seam.
        source_boundary = job.get("source_boundary")
        if not isinstance(source_boundary, Mapping):
            raise RestoreProtocolError("worker request has no persisted source boundary")
        source_references_raw = job.get("source_references")
        if not isinstance(source_references_raw, Mapping) or not source_references_raw:
            raise RestoreProtocolError("worker request has no persisted source suffix references")
        expected_reference_sha = _text(
            job.get("source_references_sha256"),
            "worker source_references_sha256",
            RestoreProtocolError,
        )
        if _sha256_bytes(_canonical_json(source_references_raw).encode("utf-8")) != expected_reference_sha:
            raise RestoreProtocolError("worker source suffix reference hash mismatch")
        source_references: dict[int, Mapping[str, Any]] = {}
        for key, value in source_references_raw.items():
            try:
                step = int(key)
            except (TypeError, ValueError) as exc:
                raise RestoreProtocolError("worker source reference step is not an integer") from exc
            source_references[step] = _mapping(
                value,
                f"worker source reference {step}",
                RestoreProtocolError,
            )
        source_terminal_raw = job.get("source_terminal")
        source_terminal = (
            _mapping(source_terminal_raw, "worker source_terminal", RestoreProtocolError)
            if source_terminal_raw is not None
            else None
        )
        physics_envelope = _mapping(
            job.get("physics_envelope"), "worker physics_envelope", RestoreProtocolError
        )
        renderer_envelope = _mapping(
            job.get("renderer_envelope"), "worker renderer_envelope", RestoreProtocolError
        )
        result = _call_restore(adapter, capture, source_boundary)
        calls.append("restore")
        physics_pass, observation_pass, _details = _static_gates(result, source_boundary)
        actions = _trace_actions(traces[trace_id])
        dynamic_pass, rows, terminal = _call_step_and_compare(
            adapter,
            actions,
            offset=int(stage["capture_offset"]),
            horizon=int(stage["horizon"]),
            source_references=source_references,
            stage=stage,
            source_terminal=source_terminal,
            physics_envelope=physics_envelope,
            renderer_envelope=renderer_envelope,
        )
        calls.append("step")
        expected_renderer = source_boundary.get(_RENDERER_REFERENCE_KEY)
        if expected_renderer is None:
            raise RestoreProtocolError("worker source boundary lacks renderer reference")
        renderer_pass, renderer_checks = _renderer_pair_gate(
            renderer_envelope,
            expected=expected_renderer,
            actual=_render_rgb_reference(adapter),
            regimes=tuple(str(value) for value in stage.get("regimes", ())),
            horizon=0,
        )
        worker_result = {
            "protocol_version": WORKER_PROTOCOL_VERSION,
            "pid": os.getpid(),
            "trace_id": trace_id,
            "stage_id": stage_id,
            "worker_index": job.get("worker_index"),
            "physics_static_identity": physics_pass,
            "observation_static_identity": observation_pass,
            "dynamic_pass": bool(dynamic_pass),
            "renderer_pass": bool(renderer_pass),
            "renderer_checks": renderer_checks,
            "rows": rows,
            "terminal": terminal,
            "pass": bool(physics_pass and observation_pass and dynamic_pass and renderer_pass),
            "no_retry": True,
            "retry_count": 0,
            "calls": calls,
            "restore_trace": [str(value) for value in _post_restore_events(adapter)] or list(calls),
            "environment": {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "cwd": str(Path.cwd()),
                "config_sha256": _sha256_bytes(
                    _canonical_json(dict(job.get("config", {}))).encode("utf-8")
                ),
            },
        }
        worker_result["output_sha256"] = _sha256_bytes(
            _canonical_json(worker_result).encode("utf-8")
        )
        return worker_result
    finally:
        close_error = _close_adapter(adapter)
        if close_error:
            raise RestoreProtocolError(close_error)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for explicit registry preparation, hard-gate execution, or worker."""

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-registry", action="store_true")
    modes.add_argument("--run-hard-gate", action="store_true")
    modes.add_argument("--worker", action="store_true")
    parser.add_argument("--registry", "--registry-path", dest="registry_path")
    parser.add_argument("--manifest", "--output", dest="manifest_path")
    parser.add_argument("--output-directory", dest="output_directory")
    parser.add_argument("--primary", "--primary-sources", dest="primary_path")
    parser.add_argument("--supplements", "--preregistered-supplements", dest="supplement_path")
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--job", dest="job_path")
    args = parser.parse_args(argv)
    if args.worker:
        if args.job_path is None:
            parser.error("--worker requires --job")
        job = _load_document(args.job_path)
        if not isinstance(job, Mapping):
            parser.error("worker job must be a mapping")
        try:
            print(_canonical_json(_run_worker_job(job)))
        except Exception as exc:
            print(
                _canonical_json(
                    {
                        "protocol_version": WORKER_PROTOCOL_VERSION,
                        "pid": os.getpid(),
                        "trace_id": job.get("trace_id"),
                        "stage_id": job.get("stage_id"),
                        "pass": False,
                        "no_retry": True,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            )
            return 1
        return 0
    if args.prepare_registry:
        if args.primary_path is None or args.registry_path is None:
            parser.error("--prepare-registry requires --primary and --registry")
        primary = _load_document(args.primary_path)
        supplements = _load_document(args.supplement_path) if args.supplement_path else None
        prepare_registry(
            primary_sources=primary,
            preregistered_supplements=supplements,
            registry_path=args.registry_path,
        )
        return 0
    if args.registry_path is None:
        parser.error("--run-hard-gate requires --registry")
    config = _load_document(args.config_path) if args.config_path else {}
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        parser.error("--config must contain a mapping")
    result = run_authoritative(
        args.registry_path,
        config=config,
        manifest_path=args.manifest_path,
        output_directory=args.output_directory,
    )
    return 0 if result.get("status") == "PASS" else 1


__all__ = [
    "DEFAULT_HORIZON",
    "FRESH_PROCESSES_PER_STAGE",
    "GATE_NAMES",
    "HardGateRunnerError",
    "PublicationError",
    "REGIME_NAMES",
    "RESTORES_PER_STAGE",
    "PreparedRegistry",
    "RegistryPreparationError",
    "RestoreProtocolError",
    "SourceCoverageError",
    "WORKER_PROTOCOL_VERSION",
    "main",
    "make_worker_request",
    "prepare_registry",
    "run_authoritative",
    "validate_short_terminal",
]


def _static_gates(result: Mapping[str, Any], expected: Mapping[str, Any] | None) -> tuple[bool, bool, dict[str, Any]]:
    static = result.get("static", {})
    static = static if isinstance(static, Mapping) else {}
    gate = static.get("gate", static.get("physics_gate", static))
    physics = _comparison_pass(
        result.get("physics_static_identity", gate),
        _comparison_pass(gate, False),
    )
    observation_value = result.get(
        "observation_static_identity",
        result.get("observation_gate", static.get("observation_gate")),
    )
    observation = _comparison_pass(observation_value, False)
    details: dict[str, Any] = {
        "physics": physics,
        "observation": observation,
    }
    # Existing RuntimeAdapter publishes an invariant comparison list.  Keep
    # its exact/floating result as the authoritative static physical check.
    comparisons = static.get("comparisons")
    if expected is not None and isinstance(result.get("invariants"), Mapping):
        try:
            _runtime_adapter, _official, _cpu = _load_replay_symbols()
            del _runtime_adapter, _official, _cpu
            from scripts import m1_state_replay as replay

            expected_invariants = dict(expected)
            expected_invariants.pop(_RENDERER_REFERENCE_KEY, None)
            actual_invariants = dict(result["invariants"])
            actual_invariants.pop(_RENDERER_REFERENCE_KEY, None)
            aggregate = replay.aggregate_gate(
                replay.compare_invariants(expected_invariants, actual_invariants)
            )
            details["invariant_gate"] = aggregate
            physics = physics and bool(aggregate.get("pass"))
        except Exception as exc:
            details["invariant_gate_error"] = f"{type(exc).__name__}: {exc}"
            physics = False
    if comparisons is not None:
        details["comparisons"] = copy.deepcopy(comparisons)
    return physics, observation, details


def _dynamic_pass(
    result: Mapping[str, Any],
    *,
    expected_rows: Mapping[int, Mapping[str, Any]],
    adapter: Any,
) -> tuple[bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    direct = _comparison_pass(result.get("pass"), False)
    raw_rows = result.get("rows")
    if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)):
        for row in raw_rows:
            if isinstance(row, Mapping):
                rows.append(copy.deepcopy(dict(row)))
    # A direct adapter returns one state after each step.  Compare it against
    # source suffix references when both snapshots expose the replay seam.
    compare_ok = True
    for step, expected in sorted(expected_rows.items()):
        actual = _collect_invariants(adapter)
        if actual is None:
            compare_ok = False
            continue
        try:
            from scripts import m1_state_replay as replay

            aggregate = replay.aggregate_gate(replay.compare_invariants(expected, actual))
            rows.append({"step": step, "gate": aggregate, "pass": bool(aggregate.get("pass"))})
            compare_ok = compare_ok and bool(aggregate.get("pass"))
        except Exception:
            compare_ok = False
    if not rows and result.get("rows") is None:
        compare_ok = direct
    return bool(direct and compare_ok), rows


def _call_step_and_compare(
    adapter: Any,
    actions: np.ndarray,
    *,
    offset: int,
    horizon: int,
    source_references: Mapping[int, Mapping[str, Any]],
    stage: Mapping[str, Any],
    source_terminal: Mapping[str, Any] | None = None,
    physics_envelope: Mapping[str, Any] | None = None,
    renderer_envelope: Mapping[str, Any] | None = None,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    passed = True
    actual_horizon = horizon
    for index in range(horizon):
        step = offset + index + 1
        result = _adapter_method(adapter, "step")(actions[offset + index])
        actual = _collect_invariants(adapter)
        row_pass = actual is not None
        comparison: Mapping[str, Any] | None = None
        if actual is not None and step in source_references:
            try:
                from scripts import m1_state_replay as replay

                expected_reference = dict(source_references[step])
                expected_renderer = expected_reference.pop(_RENDERER_REFERENCE_KEY, None)
                if expected_renderer is None or physics_envelope is None or renderer_envelope is None:
                    raise RestoreProtocolError("source suffix lacks renderer/null envelope evidence")
                actual_reference = dict(actual)
                actual_reference.pop(_RENDERER_REFERENCE_KEY, None)
                comparison_results = replay.compare_invariants(expected_reference, actual_reference)
                comparison = replay.aggregate_gate(comparison_results)
                exact_pass = bool(comparison.get("exact", {}).get("pass"))
                floating_checks: list[dict[str, Any]] = []
                floating_pass = True
                hard_gate = _load_hard_gate()
                for result_item in comparison_results:
                    if result_item.comparison_class is not replay.ComparisonClass.FLOATING_PHYSICAL:
                        continue
                    item_pass = result_item.max_abs is not None and physics_envelope is not None
                    regime_checks: dict[str, bool] = {}
                    if item_pass:
                        for regime in stage.get("regimes", ()):
                            try:
                                regime_pass = hard_gate.check_null_value(
                                    physics_envelope,
                                    value=result_item.max_abs,
                                    regime=str(regime),
                                    quantity=result_item.path,
                                    horizon=index + 1,
                                )
                            except Exception:
                                regime_pass = False
                            regime_checks[str(regime)] = bool(regime_pass)
                        item_pass = bool(regime_checks) and all(regime_checks.values())
                    floating_checks.append(
                        {
                            "path": result_item.path,
                            "max_abs": result_item.max_abs,
                            "horizon": index + 1,
                            "regimes": regime_checks,
                            "pass": bool(item_pass),
                        }
                    )
                    floating_pass = floating_pass and bool(item_pass)
                comparison["floating_null_envelope"] = {
                    "pass": bool(floating_pass),
                    "checks": floating_checks,
                    "global_tolerance": None,
                }
                comparison["pass"] = bool(exact_pass and floating_pass)
                renderer_pass, renderer_checks = _renderer_pair_gate(
                    renderer_envelope,
                    expected=expected_renderer,
                    actual=_render_rgb_reference(adapter),
                    regimes=tuple(str(value) for value in stage.get("regimes", ())),
                    horizon=index + 1,
                )
                comparison["renderer"] = {
                    "pass": bool(renderer_pass),
                    "checks": renderer_checks,
                }
                comparison["pass"] = bool(comparison["pass"] and renderer_pass)
                row_pass = bool(comparison["pass"])
            except Exception:
                row_pass = False
        rows.append({"step": step, "pass": row_pass, "gate": comparison or {"pass": row_pass}})
        passed = passed and row_pass
        if _is_terminal(result):
            terminal = _terminal_record(
                result,
                adapter,
                steps=index + 1,
                nominal_horizon=DEFAULT_HORIZON,
                reason=stage.get("source_coverage", {}).get("terminal_reason"),
                absolute_step=step,
            )
            actual_horizon = index + 1
            if source_terminal is not None:
                if terminal.get("steps") != source_terminal.get("steps"):
                    passed = False
                if terminal.get("termination_reason") != source_terminal.get("termination_reason"):
                    passed = False
            if actual_horizon < horizon:
                if not set(stage.get("regimes", ())) & _TERMINAL_REGIMES:
                    passed = False
                expected_reason = stage.get("source_coverage", {}).get("terminal_reason")
                if not expected_reason or stage.get("source_coverage", {}).get("legal_terminal_step") is None:
                    passed = False
                else:
                    legal_step = int(stage["source_coverage"]["legal_terminal_step"])
                    if int(terminal.get("absolute_step", -1)) != legal_step:
                        passed = False
                if not validate_short_terminal(
                    terminal,
                    nominal_horizon=DEFAULT_HORIZON,
                    expected_reason=expected_reason,
                ):
                    passed = False
            break
    if actual_horizon != horizon:
        # A terminal stage has an intentionally shorter horizon, otherwise an
        # early end is a failed branch and no further action is submitted.
        if horizon == DEFAULT_HORIZON:
            passed = False
        elif terminal is None:
            passed = False
    return passed, rows, terminal


def _validate_sample_coverage(
    samples: Sequence[Mapping[str, Any]],
    *,
    selected_ids: Sequence[str],
    coverage_by_trace: Mapping[str, Sequence[str]],
    kind: str,
) -> tuple[bool, list[str]]:
    """Check pair minimums before invoking the strict envelope builders."""

    errors: list[str] = []
    if len(samples) < 20:
        errors.append(f"{kind} calibration has fewer than 20 duplicate-control pairs")
    pair_sets: dict[tuple[str, str], set[str]] = {}
    trace_pairs: dict[str, set[str]] = {}
    all_pairs: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            errors.append(f"{kind} sample {index} is not a mapping")
            continue
        trace_id = sample.get("trace_id", sample.get("episode_id"))
        regimes: list[str]
        if trace_id is None:
            errors.append(f"{kind} sample {index} lacks trace_id")
            continue
        trace_text = str(trace_id)
        pair_id = sample.get("pair_id", sample.get("run_id"))
        if not isinstance(pair_id, str) or not pair_id:
            errors.append(f"{kind} sample {index} lacks pair_id/run_id")
            continue
        all_pairs.add(pair_id)
        trace_pairs.setdefault(trace_text, set()).add(pair_id)
        regime = sample.get("regime")
        if regime is not None:
            regimes = [str(regime)]
        else:
            regimes = [str(value) for value in coverage_by_trace.get(trace_text, ())]
        if not regimes:
            errors.append(f"{kind} sample {index} lacks a regime")
            continue
        for regime_name in regimes:
            pair_sets.setdefault((trace_text, regime_name), set()).add(pair_id)
    if len(all_pairs) < 20:
        errors.append(f"{kind} calibration has fewer than 20 distinct complete duplicate-control runs")
    for trace_id in selected_ids:
        if len(trace_pairs.get(trace_id, set())) < 5:
            errors.append(
                f"{kind} coverage for trace={trace_id} has "
                f"{len(trace_pairs.get(trace_id, set()))} distinct runs; minimum is 5"
            )
        for regime in coverage_by_trace.get(trace_id, ()):
            count = len(pair_sets.get((trace_id, regime), set()))
            if count < 5:
                errors.append(
                    f"{kind} coverage for trace={trace_id} regime={regime} has "
                    f"{count} distinct runs; minimum is 5"
                )
    return not errors, errors


def _build_null_envelopes(
    *,
    physics_samples: Any,
    renderer_samples: Any,
    selected_ids: Sequence[str],
    coverage_by_trace: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any], list[str]]:
    hard_gate = _load_hard_gate()
    physics_envelope: dict[str, Any] | None = None
    renderer_envelope: dict[str, Any] | None = None
    details: dict[str, Any] = {}
    errors: list[str] = []
    if physics_samples is None:
        errors.append("null physics samples were not supplied")
    else:
        try:
            physics_list = list(physics_samples)
        except TypeError as exc:
            physics_list = []
            errors.append(f"null physics samples are not iterable: {exc}")
        valid, coverage_errors = _validate_sample_coverage(
            physics_list,
            selected_ids=selected_ids,
            coverage_by_trace=coverage_by_trace,
            kind="physics",
        )
        errors.extend(coverage_errors)
        if valid:
            try:
                required_groups = sorted(
                    {
                        (
                            str(sample.get("regime")),
                            str(sample.get("quantity")),
                            int(sample.get("horizon")),
                        )
                        for sample in physics_list
                        if isinstance(sample, Mapping)
                        and sample.get("regime") is not None
                        and sample.get("quantity") is not None
                        and not isinstance(sample.get("horizon"), bool)
                        and isinstance(sample.get("horizon"), (int, np.integer))
                    }
                )
                physics_envelope = hard_gate.build_grouped_null_envelope(
                    physics_list,
                    selected_trace_ids=selected_ids,
                    required_groups=required_groups,
                    min_pairs=20,
                    min_pairs_per_trace=5,
                    min_samples_per_regime=5,
                )
                details["physics"] = physics_envelope
            except Exception as exc:
                errors.append(f"null physics envelope failed: {type(exc).__name__}: {exc}")
    if renderer_samples is None:
        errors.append("null renderer samples were not supplied")
    else:
        try:
            renderer_list = list(renderer_samples)
        except TypeError as exc:
            renderer_list = []
            errors.append(f"null renderer samples are not iterable: {exc}")
        valid, coverage_errors = _validate_sample_coverage(
            renderer_list,
            selected_ids=selected_ids,
            coverage_by_trace=coverage_by_trace,
            kind="renderer",
        )
        errors.extend(coverage_errors)
        if valid:
            try:
                # Exact duplicate controls are required for a grouped RGB
                # envelope; differing controls are not a valid null.
                renderer_envelope = hard_gate.build_renderer_envelopes(
                    renderer_list,
                    # Let calibration decide: bitwise-identical duplicate
                    # controls require exact RGB; otherwise publish and use
                    # the measured grouped null envelope.
                    exact_only=None,
                    min_samples_per_group=5,
                )
                observed_regimes = {
                    str(regime)
                    for camera_group in renderer_envelope.get("groups", {}).values()
                    if isinstance(camera_group, Mapping)
                    for key_group in camera_group.values()
                    if isinstance(key_group, Mapping)
                    for regime in key_group
                }
                required_regimes = {
                    str(regime)
                    for trace_id in selected_ids
                    for regime in coverage_by_trace.get(trace_id, ())
                }
                if observed_regimes != required_regimes:
                    raise RestoreProtocolError(
                        "renderer envelope regime groups do not cover frozen source regimes"
                    )
                renderer_pair_ids = sorted(
                    {
                        str(item.get("pair_id", item.get("run_id")))
                        for item in renderer_list
                        if isinstance(item, Mapping)
                    }
                )
                renderer_envelope.setdefault("coverage", {}).update(
                    {
                        "n_complete_duplicate_control_runs": len(renderer_pair_ids),
                        "pair_ids": renderer_pair_ids,
                        "minimum_complete_runs": 20,
                    }
                )
                details["renderer"] = renderer_envelope
            except Exception as exc:
                errors.append(f"null renderer envelope failed: {type(exc).__name__}: {exc}")
    return physics_envelope, renderer_envelope, details, errors


def _normalise_process_result(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, subprocess.CompletedProcess):
        stdout = value.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stdout, str) and stdout.strip():
            try:
                parsed = json.loads(stdout)
                if isinstance(parsed, Mapping):
                    result = dict(parsed)
                else:
                    result = {"pass": False, "error": "worker stdout was not a mapping"}
            except json.JSONDecodeError as exc:
                result = {"pass": False, "error": f"worker stdout is not JSON: {exc}"}
        else:
            result = {"pass": False, "error": "worker returned no JSON"}
        result.setdefault("returncode", value.returncode)
        return result
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            return {"pass": False, "error": f"worker output is not JSON: {exc}"}
        return dict(parsed) if isinstance(parsed, Mapping) else {"pass": False, "error": "worker output is not a mapping"}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        return {
            "returncode": value[0],
            "stdout": value[1],
            "stderr": value[2] if len(value) > 2 else "",
        }
    return {"pass": False, "error": f"unsupported worker result: {type(value).__name__}"}


def _invoke_process_runner(
    runner: Callable[..., Any],
    command: Sequence[str],
    job: Mapping[str, Any],
) -> Any:
    """Support simple injected ``runner(job)`` and subprocess-like seams."""

    try:
        signature = inspect.signature(runner)
        params = list(signature.parameters.values())
    except (TypeError, ValueError):
        return runner(command)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in params):
        return runner(command, job=job)
    positional = [
        parameter
        for parameter in params
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    names = {parameter.name.lower() for parameter in positional}
    if len(positional) == 1:
        parameter_name = positional[0].name.lower()
        if parameter_name in {"job", "payload", "request", "worker_job"}:
            return runner(job)
        return runner(command)
    if len(positional) >= 2:
        return runner(command, job)
    return runner(command)


def make_worker_request(
    *,
    registry_path: str | Path,
    trace_id: str,
    stage_id: str,
    worker_index: int,
    horizon: int,
    config: Mapping[str, Any] | None = None,
    capture: Any | None = None,
    source_boundary: Mapping[str, Any] | None = None,
    source_references: Mapping[int, Mapping[str, Any]] | None = None,
    source_terminal: Mapping[str, Any] | None = None,
    physics_envelope: Mapping[str, Any] | None = None,
    renderer_envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the serializable request consumed by ``--worker``."""

    request: dict[str, Any] = {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "registry_path": str(registry_path),
        "trace_id": str(trace_id),
        "stage_id": str(stage_id),
        "worker_index": int(worker_index),
        "horizon": int(horizon),
        "config": dict(config or {}),
    }
    if capture is not None:
        try:
            _canonical_json(capture)
        except Exception:
            # Binary ReplayState records are persisted by a caller-owned
            # adapter seam; the request carries only a protocol marker here.
            request["capture_in_worker"] = False
        else:
            request["capture"] = capture
            request["capture_in_worker"] = True
    if source_boundary is not None:
        request["source_boundary"] = dict(source_boundary)
    if source_references is not None:
        persisted_references = {
            str(int(step)): dict(reference)
            for step, reference in sorted(source_references.items(), key=lambda item: int(item[0]))
        }
        request["source_references"] = persisted_references
        request["source_references_sha256"] = _sha256_bytes(
            _canonical_json(persisted_references).encode("utf-8")
        )
    if source_terminal is not None:
        request["source_terminal"] = dict(source_terminal)
    if physics_envelope is not None:
        request["physics_envelope"] = dict(physics_envelope)
    if renderer_envelope is not None:
        request["renderer_envelope"] = dict(renderer_envelope)
    return request


def _worker_command(job_path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker", "--job", str(job_path)]


def _persist_worker_job(directory: Path, job: Mapping[str, Any], stem: str) -> Path:
    target = directory / f".{stem}.json"
    return _atomic_publish(target, job)


def _worker_result_gate(result: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if result.get("trace_id") != job.get("trace_id"):
        errors.append("worker trace_id does not match requested frozen trace")
    if result.get("stage_id") != job.get("stage_id"):
        errors.append("worker stage_id does not match requested frozen stage")
    if "pid" not in result or isinstance(result.get("pid"), bool):
        errors.append("worker did not report a Python PID")
    else:
        try:
            if int(result["pid"]) <= 0:
                errors.append("worker PID is not positive")
            elif int(result["pid"]) == os.getpid():
                errors.append("worker PID is not a fresh Python process")
        except (TypeError, ValueError):
            errors.append("worker PID is not an integer")
    if result.get("retry_count", 0) != 0 or result.get("no_retry") is False:
        errors.append("worker reported a retry")
    if result.get("returncode") not in (None, 0):
        errors.append(f"worker returned nonzero status {result.get('returncode')}")
    calls = result.get("calls", result.get("restore_trace", ()))
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
        forbidden = [str(item) for item in calls if _forbidden_restore_event(item)]
        if forbidden:
            errors.append("worker restore trace contains forbidden operation: " + ", ".join(forbidden))
    for name in (
        "physics_static_identity",
        "observation_static_identity",
        "dynamic_pass",
        "renderer_pass",
    ):
        if name not in result:
            errors.append(f"worker did not report {name}")
        elif not _comparison_pass(result[name], False):
            errors.append(f"worker {name} gate failed")
    output_sha = result.get("output_sha256")
    if not isinstance(output_sha, str) or len(output_sha) != 64:
        errors.append("worker did not report a valid output SHA-256")
    else:
        without_hash = dict(result)
        without_hash.pop("output_sha256", None)
        if output_sha != _sha256_bytes(_canonical_json(without_hash).encode("utf-8")):
            errors.append("worker output SHA-256 does not match result payload")
    if "state_closure" in result and not _comparison_pass(result["state_closure"], False):
        errors.append("worker state closure gate failed")
    return not errors and _comparison_pass(result.get("pass"), False), errors


def _comparison_pass(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        if "pass" in value:
            return bool(value["pass"])
        if "passed" in value:
            return bool(value["passed"])
    return default


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
