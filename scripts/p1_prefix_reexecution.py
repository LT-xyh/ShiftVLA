"""Repo-only ReplayVLA-P1 prefix-reexecution schedule and evidence skeleton.

This module contains no environment construction, rendering, simulator stepping, or
policy inference.  It freezes the pilot matrix, paired-noise identity, observation
indexing, and same-prefix evidence rules authorized by the current Paper-1 route.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ARMS = ("CC", "CS", "SC", "SS")
PREFIX_FUTURE = {
    "CC": ("clean", "clean"),
    "CS": ("clean", "shifted"),
    "SC": ("shifted", "clean"),
    "SS": ("shifted", "shifted"),
}
FROZEN_SUITE = "libero_spatial"
FROZEN_TASKS = (0, 4)
FROZEN_INIT_STATE_IDS = (0, 1, 2, 3)
FROZEN_SWITCH_INDEX = 50
FROZEN_HORIZON = 280
FROZEN_CAMERA_NAME = "agentview"
FROZEN_YAW_DEGREES = 15
FROZEN_FLOW_SHAPE = (1, 50, 32)
DRAW_KIND = "flow"
DRAW_SLOT = 0

# Source-accounted gate status.  The pinned stack exposes an XML-construction
# camera setter, while its supported CameraMover resets/rebuilds the environment.
# With hard_reset=True, no narrow external hook was found between model rebuild
# and reset-time camera observation generation.  Runtime camera mutation is
# intentionally not implemented here.
G_P2_SOURCE_TRACE: dict[str, Any] = {
    "status": "BLOCKED",
    "hf_lerobot_revision": "7e241bd630a3719a56157a497ce5d08f244784f1",
    "hf_libero_revision": "8561c60eea2fb93096146f240194649df73d8b1e",
    "robosuite_revision": "fbee5844ff5632f5b5698e204ec5357ca50be0df",
    "agentview_construction": (
        "hf-LIBERO bddl_base_domain.py::_setup_camera -> "
        "mujoco_arena.set_camera(name='agentview', pos=..., quat=...)"
    ),
    "agentview_model_fields": {
        "name": "agentview",
        "position": [0.5886131746834771, 0.0, 1.4903500240372423],
        "quaternion_wxyz": [
            0.6380177736282349,
            0.3048497438430786,
            0.30484986305236816,
            0.6380177736282349,
        ],
        "source_api": "robosuite.models.arenas.Arena.set_camera (XML/model construction)",
    },
    "yaw_mapping": {
        "status": "BLOCKED",
        "reason": (
            "no supported pinned runtime API was found that can apply +15 degree "
            "agentview yaw after hard-reset model rebuild and before obs_0 rendering "
            "without reset/XML rewrite/state restoration or deep internal monkeypatching"
        ),
    },
    "observation_path": (
        "robosuite RobotEnv._create_camera_sensors.camera_rgb -> "
        "sim.render(camera_name='agentview', ...)"
    ),
    "reset_path": (
        "LeRobot defaults hard_reset=True; robosuite MujocoEnv.reset rebuilds "
        "model/sim, recreates camera observables, then returns "
        "_get_observations(force_update=True)"
    ),
    "step_observation_path": (
        "robosuite MujocoEnv.step performs simulation updates, updates observables, "
        "then returns _get_observations(); a valid obs_50 switch would therefore "
        "have to be installed before env.step(action_49)"
    ),
    "runtime_mutation_api": None,
    "physics_isolation_status": "UNPROVEN_BECAUSE_NO_AUTHORIZED_RUNTIME_CAMERA_MUTATION_SEAM",
    "blocked_reason": (
        "no narrow supported external API was found to install shifted agentview "
        "after hard-reset model rebuild but before obs_0 rendering; CameraMover "
        "rewrites XML and resets/restores the environment"
    ),
}


class PrefixReexecutionConfigError(ValueError):
    pass


class TechnicalMatchingFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class PairedNoiseSpec:
    root_id: str
    observation_action_index: int
    draw_kind: str
    draw_slot: int
    key: str
    seed: int


def paired_noise_spec(
    root_id: str,
    observation_action_index: int,
    *,
    draw_kind: str = DRAW_KIND,
    draw_slot: int = DRAW_SLOT,
) -> PairedNoiseSpec:
    """Derive an arm-independent deterministic flow-noise identity."""

    if not isinstance(root_id, str) or not root_id:
        raise PrefixReexecutionConfigError("root_id must be a non-empty string")
    if (
        isinstance(observation_action_index, bool)
        or not isinstance(observation_action_index, int)
        or observation_action_index < 0
    ):
        raise PrefixReexecutionConfigError("observation_action_index must be non-negative")
    if draw_kind != DRAW_KIND or draw_slot != DRAW_SLOT:
        raise PrefixReexecutionConfigError("Paper-1 paired noise is frozen to flow/draw_slot=0")
    payload = json.dumps(
        {
            "draw_kind": draw_kind,
            "draw_slot": draw_slot,
            "observation_action_index": observation_action_index,
            "root_id": root_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return PairedNoiseSpec(
        root_id=root_id,
        observation_action_index=observation_action_index,
        draw_kind=draw_kind,
        draw_slot=draw_slot,
        key=payload,
        seed=seed,
    )


def generate_paired_flow_noise(
    root_id: str,
    observation_action_index: int,
) -> tuple[Any, dict[str, Any]]:
    """Reuse the existing deterministic flow-noise generator."""

    from scripts.dcu_preflight import generate_flow_noise

    spec = paired_noise_spec(root_id, observation_action_index)
    noise, digest = generate_flow_noise(seed=spec.seed, shape=FROZEN_FLOW_SHAPE)
    return noise, {
        "root_id": spec.root_id,
        "observation_action_index": spec.observation_action_index,
        "draw_kind": spec.draw_kind,
        "draw_slot": spec.draw_slot,
        "noise_key": spec.key,
        "noise_seed": spec.seed,
        "noise_sha256": digest,
    }


class PairedNoiseSelectActionPolicy:
    """Arm-independent noise wrapper around the existing remote select_action seam.

    Official rollout still owns processor ordering.  This wrapper only derives
    one paired flow-noise tensor per observation/action query and forwards it as
    a keyword to the existing remote policy select_action method.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        root_id: str,
        noise_factory: Callable[[str, int], tuple[Any, Mapping[str, Any]]] = generate_paired_flow_noise,
    ) -> None:
        if not callable(getattr(delegate, "select_action", None)):
            raise PrefixReexecutionConfigError("delegate must provide select_action")
        self.delegate = delegate
        self.root_id = paired_noise_spec(root_id, 0).root_id
        self.noise_factory = noise_factory
        self.query_index = 0
        self.noise_evidence: list[dict[str, Any]] = []

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self.query_index = 0
        self.noise_evidence.clear()
        return self.delegate.reset(*args, **kwargs)

    def select_action(self, batch: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            raise PrefixReexecutionConfigError(
                "paired-noise wrapper does not accept caller action-selection overrides"
            )
        index = self.query_index
        noise, evidence = self.noise_factory(self.root_id, index)
        action = self.delegate.select_action(batch, noise=noise)
        row = dict(evidence)
        if row.get("observation_action_index") != index:
            raise PrefixReexecutionConfigError("noise factory returned mismatched query index")
        self.noise_evidence.append(row)
        self.query_index += 1
        return action

    def eval(self) -> "PairedNoiseSelectActionPolicy":
        method = getattr(self.delegate, "eval", None)
        if callable(method):
            method()
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def camera_mode_for_observation(
    arm: str,
    observation_index: int,
    *,
    switch_index: int = FROZEN_SWITCH_INDEX,
) -> str:
    if arm not in PREFIX_FUTURE:
        raise PrefixReexecutionConfigError(f"unsupported arm: {arm!r}")
    if isinstance(observation_index, bool) or not isinstance(observation_index, int) or observation_index < 0:
        raise PrefixReexecutionConfigError("observation_index must be non-negative")
    if switch_index != FROZEN_SWITCH_INDEX:
        raise PrefixReexecutionConfigError("switch_index must remain frozen at 50")
    prefix, future = PREFIX_FUTURE[arm]
    return prefix if observation_index < switch_index else future


def camera_request_for_initial_observation(arm: str) -> dict[str, Any]:
    return {
        "observation_index": 0,
        "preceding_action_index": None,
        "requested_camera_mode": camera_mode_for_observation(arm, 0),
    }


def camera_request_before_step(arm: str, action_index: int) -> dict[str, Any]:
    if isinstance(action_index, bool) or not isinstance(action_index, int) or action_index < 0:
        raise PrefixReexecutionConfigError("action_index must be non-negative")
    observation_index = action_index + 1
    return {
        "observation_index": observation_index,
        "preceding_action_index": action_index,
        "requested_camera_mode": camera_mode_for_observation(arm, observation_index),
    }


def camera_evidence(
    request: Mapping[str, Any],
    *,
    actual_camera_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(actual_camera_parameters, Mapping) or not actual_camera_parameters:
        raise PrefixReexecutionConfigError("actual_camera_parameters must be a non-empty mapping")
    return {
        "observation_index": int(request["observation_index"]),
        "preceding_action_index": request.get("preceding_action_index"),
        "requested_camera_mode": str(request["requested_camera_mode"]),
        "actual_camera_parameters": dict(actual_camera_parameters),
    }


def validate_pilot_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "suite": FROZEN_SUITE,
        "tasks": list(FROZEN_TASKS),
        "init_state_ids": list(FROZEN_INIT_STATE_IDS),
        "switch_index": FROZEN_SWITCH_INDEX,
        "arms": list(ARMS),
        "clean_duplicate_per_root": 1,
        "horizon": FROZEN_HORIZON,
        "retry": 0,
        "replacement": 0,
        "runtime_authorized": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise PrefixReexecutionConfigError(f"{key} must equal frozen value {value!r}")
    camera = config.get("camera")
    if not isinstance(camera, Mapping):
        raise PrefixReexecutionConfigError("camera must be a mapping")
    if camera.get("name") != FROZEN_CAMERA_NAME or camera.get("yaw_degrees") != FROZEN_YAW_DEGREES:
        raise PrefixReexecutionConfigError("camera must be agentview yaw +15 degrees")
    return dict(config)


def build_pilot_schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_pilot_config(config)
    rows: list[dict[str, Any]] = []
    order = 0
    for task_id in FROZEN_TASKS:
        for init_state_id in FROZEN_INIT_STATE_IDS:
            root_id = f"{FROZEN_SUITE}-task{task_id:03d}-init{init_state_id:03d}"
            for arm in ARMS:
                rows.append(
                    {
                        "order": order,
                        "run_id": f"{root_id}-{arm}",
                        "root_id": root_id,
                        "task_id": task_id,
                        "init_state_id": init_state_id,
                        "arm": arm,
                        "clean_duplicate": False,
                        "switch_index": FROZEN_SWITCH_INDEX,
                        "horizon": FROZEN_HORIZON,
                        "retry": 0,
                        "replacement": 0,
                        "status": "PLANNED",
                    }
                )
                order += 1
            rows.append(
                {
                    "order": order,
                    "run_id": f"{root_id}-CC-DUP0",
                    "root_id": root_id,
                    "task_id": task_id,
                    "init_state_id": init_state_id,
                    "arm": "CC",
                    "clean_duplicate": True,
                    "switch_index": FROZEN_SWITCH_INDEX,
                    "horizon": FROZEN_HORIZON,
                    "retry": 0,
                    "replacement": 0,
                    "status": "PLANNED",
                }
            )
            order += 1
    if len(rows) != 40 or sum(not row["clean_duplicate"] for row in rows) != 32:
        raise PrefixReexecutionConfigError("frozen pilot schedule must contain 32 main + 8 duplicate rows")
    return rows


def absorbing_terminal_evidence(
    planned_row: Mapping[str, Any],
    *,
    observation_action_index: int,
    terminal_state: Mapping[str, Any],
) -> dict[str, Any]:
    if observation_action_index >= FROZEN_SWITCH_INDEX:
        raise PrefixReexecutionConfigError("absorbing pre-switch terminal must occur before index 50")
    return {
        "run_id": planned_row["run_id"],
        "root_id": planned_row["root_id"],
        "arm": planned_row["arm"],
        "terminal_index": observation_action_index,
        "terminal_state": dict(terminal_state),
        "absorbing": True,
        "continuation_authorized": False,
        "retry": 0,
        "replacement": 0,
    }


_PREFIX_AUDIT_FIELDS = (
    "noise_sha256",
    "query_index",
    "camera_mode",
    "action",
    "terminal_state",
)


def audit_same_prefix(
    left_arm: str,
    right_arm: str,
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    *,
    switch_index: int = FROZEN_SWITCH_INDEX,
) -> dict[str, Any]:
    allowed = {("CC", "CS"), ("SC", "SS")}
    if (left_arm, right_arm) not in allowed:
        raise PrefixReexecutionConfigError("same-prefix audit supports CC/CS or SC/SS only")
    if switch_index != FROZEN_SWITCH_INDEX:
        raise PrefixReexecutionConfigError("same-prefix audit switch_index must remain 50")

    def prefix(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [row for row in rows if int(row["query_index"]) < switch_index]

    lhs = prefix(left_rows)
    rhs = prefix(right_rows)
    if len(lhs) != len(rhs):
        raise TechnicalMatchingFailure("same-prefix evidence length mismatch")
    for position, (left, right) in enumerate(zip(lhs, rhs)):
        for field in _PREFIX_AUDIT_FIELDS:
            if left.get(field) != right.get(field):
                raise TechnicalMatchingFailure(
                    f"same-prefix disagreement at position {position} field {field}"
                )
    return {
        "status": "PASS",
        "pair": [left_arm, right_arm],
        "checked_rows": len(lhs),
        "switch_index": switch_index,
    }


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - project dependency invariant
        raise PrefixReexecutionConfigError("PyYAML is required to load the pilot config") from exc
    candidate = Path(path)
    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise PrefixReexecutionConfigError("pilot config root must be a mapping")
    return validate_pilot_config(payload)
