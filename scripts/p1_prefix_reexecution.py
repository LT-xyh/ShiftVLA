"""Repo-only ReplayVLA-P1 prefix-reexecution schedule and evidence skeleton.

This module contains no environment construction, rendering, simulator stepping, or
policy inference.  It freezes the pilot matrix, paired-noise identity, observation
indexing, and same-prefix evidence rules authorized by the current Paper-1 route.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


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
FROZEN_ENVIRONMENT_SEED = 2027
FROZEN_CAMERA_NAME = "agentview"
FROZEN_YAW_DEGREES = 15
FROZEN_FLOW_SHAPE = (1, 50, 32)
DRAW_KIND = "flow"
DRAW_SLOT = 0

# Source-accounted G-P2 trace.  This is repo-only evidence, not an observed
# LIBERO runtime qualification.
G_P2_SOURCE_TRACE: dict[str, Any] = {
    "status": "IMPLEMENTED_REPO_ONLY_NOT_RUNTIME_VALIDATED",
    "hf_lerobot_revision": "7e241bd630a3719a56157a497ce5d08f244784f1",
    "hf_libero_revision": "8561c60eea2fb93096146f240194649df73d8b1e",
    "robosuite_revision": "fbee5844ff5632f5b5698e204ec5357ca50be0df",
    "runtime_mutation_api": "robosuite.utils.mjmod.CameraModder.set_quat",
    "CameraMover": "REJECTED_XML_RESET_STATE_RESTORE_ROUTE",
    "CameraModder": "SELECTED_RUNTIME_MODEL_CAMERA_ROUTE",
    "DomainRandomizationWrapper": "SELECTED_LIFECYCLE_PRECEDENT",
    "agentview_construction": (
        "hf-LIBERO bddl_base_domain.py::_setup_camera -> "
        "mujoco_arena.set_camera(name='agentview', pos=..., quat=...)"
    ),
    "agentview_clean_model_fields": {
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
    "camera_modder_source_account": (
        "CameraModder.get_quat returns model.cam_quat[camid]; "
        "CameraModder.set_quat writes only model.cam_quat[camid]"
    ),
    "yaw_definition": (
        "world-frame +Z extrinsic yaw; R_shifted = R_yaw(+15deg) @ R_clean; "
        "CameraModder boundary uses wxyz while robosuite transform_utils matrix helpers use xyzw"
    ),
    "lerobot_insertion": (
        "lerobot.envs.libero.LiberoEnv owns self._env: OffScreenRenderEnv; "
        "OffScreenRenderEnv -> ControlEnv -> self.env = robosuite task env -> sim"
    ),
    "observation_path": (
        "robosuite RobotEnv._create_camera_sensors.camera_rgb -> "
        "sim.render(camera_name='agentview', ...)"
    ),
    "reset_lifecycle_precedent": (
        "DomainRandomizationWrapper.reset: underlying reset -> save defaults -> "
        "modder.update_sim(current sim) -> camera mutation -> current-state _get_observations"
    ),
    "project_reset_lifecycle": (
        "delegate LiberoEnv.reset (including init-state + settle dummy steps) exactly once -> "
        "rebind CameraModder to current sim -> set requested agentview quaternion -> "
        "force-refresh current-state observation -> return that as the only policy obs_0"
    ),
    "step_lifecycle_precedent": (
        "DomainRandomizationWrapper.step performs randomization update before delegated env.step(action)"
    ),
    "project_step_lifecycle": (
        "obs_t -> action_t -> apply camera request for obs_{t+1} -> "
        "delegate env.step(action_t) -> obs_{t+1}; action_49 therefore switches before its one step"
    ),
    "settle_step_treatment": (
        "LiberoEnv.reset internal num_steps_wait dummy steps occur before policy obs_0; "
        "they are not policy queries, do not consume paired-noise indices, and do not count toward t_switch"
    ),
    "physics_isolation_evidence": {
        "source_accounted": "set_quat writes model.cam_quat[camera_id] only",
        "fake_unit": (
            "sentinel qpos/qvel/ctrl/object/dynamics fields plus cam_pos/cam_fovy "
            "are checked unchanged across clean/shifted/clean mutation"
        ),
        "real_runtime": "NOT_VALIDATED_NOT_AUTHORIZED",
    },
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


def _load_robosuite_transform_utils() -> Any:
    from robosuite.utils import transform_utils

    return transform_utils


def _camera_modder_for_sim(sim: Any) -> Any:
    from robosuite.utils.mjmod import CameraModder

    return CameraModder(
        sim=sim,
        camera_names=[FROZEN_CAMERA_NAME],
        randomize_position=False,
        randomize_rotation=True,
        randomize_fovy=False,
    )


def derive_world_z_yaw_wxyz(
    clean_quaternion_wxyz: Sequence[float],
    *,
    yaw_degrees: float = FROZEN_YAW_DEGREES,
    transform_utils: Any | None = None,
) -> np.ndarray:
    """Compose a frozen world-frame +Z extrinsic yaw using robosuite conventions."""

    if float(yaw_degrees) != float(FROZEN_YAW_DEGREES):
        raise PrefixReexecutionConfigError("Paper-1 camera yaw must remain +15 degrees")
    clean = np.asarray(clean_quaternion_wxyz, dtype=np.float64)
    if clean.shape != (4,) or not np.isfinite(clean).all():
        raise PrefixReexecutionConfigError("clean camera quaternion must be finite wxyz length 4")
    norm = float(np.linalg.norm(clean))
    if not np.isclose(norm, 1.0, atol=1e-6, rtol=0.0):
        raise PrefixReexecutionConfigError("clean camera quaternion must be normalized")

    transforms = transform_utils or _load_robosuite_transform_utils()
    clean_xyzw = np.asarray(transforms.convert_quat(clean, to="xyzw"), dtype=np.float64)
    clean_rotation = np.asarray(transforms.quat2mat(clean_xyzw), dtype=np.float64)
    theta = math.radians(float(yaw_degrees))
    yaw_rotation = np.asarray(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    shifted_rotation = yaw_rotation @ clean_rotation
    shifted_xyzw = np.asarray(transforms.mat2quat(shifted_rotation), dtype=np.float64)
    shifted_wxyz = np.asarray(transforms.convert_quat(shifted_xyzw, to="wxyz"), dtype=np.float64)
    shifted_norm = float(np.linalg.norm(shifted_wxyz))
    if shifted_norm <= 0.0 or not np.isfinite(shifted_norm):
        raise PrefixReexecutionConfigError("derived shifted camera quaternion is invalid")
    return shifted_wxyz / shifted_norm


class AgentviewYawController:
    """Thin deterministic agentview controller using only CameraModder.set_quat."""

    def __init__(
        self,
        *,
        camera_name: str = FROZEN_CAMERA_NAME,
        yaw_degrees: float = FROZEN_YAW_DEGREES,
        modder_factory: Callable[[Any], Any] = _camera_modder_for_sim,
        transform_utils: Any | None = None,
    ) -> None:
        if camera_name != FROZEN_CAMERA_NAME:
            raise PrefixReexecutionConfigError("only agentview is authorized")
        if float(yaw_degrees) != float(FROZEN_YAW_DEGREES):
            raise PrefixReexecutionConfigError("camera yaw must remain +15 degrees")
        self.camera_name = camera_name
        self.yaw_degrees = float(yaw_degrees)
        self.modder_factory = modder_factory
        self.transform_utils = transform_utils
        self.modder: Any | None = None
        self.sim: Any | None = None
        self.clean_quaternion_wxyz: np.ndarray | None = None
        self.shifted_quaternion_wxyz: np.ndarray | None = None
        self.clean_position: np.ndarray | None = None
        self.clean_fovy: float | None = None

    def bind(self, sim: Any) -> dict[str, Any]:
        if sim is None or getattr(sim, "model", None) is None:
            raise PrefixReexecutionConfigError("camera controller requires a current sim/model")
        if self.modder is None:
            self.modder = self.modder_factory(sim)
        else:
            update_sim = getattr(self.modder, "update_sim", None)
            if not callable(update_sim):
                raise PrefixReexecutionConfigError("CameraModder seam must provide update_sim")
            update_sim(sim)
        self.sim = sim
        clean_quat = np.asarray(self.modder.get_quat(self.camera_name), dtype=np.float64).copy()
        clean_pos = np.asarray(self.modder.get_pos(self.camera_name), dtype=np.float64).copy()
        clean_fovy = float(self.modder.get_fovy(self.camera_name))
        self.clean_quaternion_wxyz = clean_quat
        self.clean_position = clean_pos
        self.clean_fovy = clean_fovy
        self.shifted_quaternion_wxyz = derive_world_z_yaw_wxyz(
            clean_quat,
            yaw_degrees=self.yaw_degrees,
            transform_utils=self.transform_utils,
        )
        return {
            "camera_name": self.camera_name,
            "clean_quaternion_wxyz": clean_quat.tolist(),
            "shifted_quaternion_wxyz": self.shifted_quaternion_wxyz.tolist(),
            "position": clean_pos.tolist(),
            "fovy": clean_fovy,
        }

    def apply(self, mode: str, *, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.modder is None or self.sim is None:
            raise PrefixReexecutionConfigError("camera controller must be bound before apply")
        if mode not in {"clean", "shifted"}:
            raise PrefixReexecutionConfigError("camera mode must be clean or shifted")
        assert self.clean_quaternion_wxyz is not None
        assert self.shifted_quaternion_wxyz is not None
        assert self.clean_position is not None
        assert self.clean_fovy is not None

        target = (
            self.clean_quaternion_wxyz
            if mode == "clean"
            else self.shifted_quaternion_wxyz
        )
        self.modder.set_quat(self.camera_name, target.copy())
        actual_quat = np.asarray(self.modder.get_quat(self.camera_name), dtype=np.float64).copy()
        actual_pos = np.asarray(self.modder.get_pos(self.camera_name), dtype=np.float64).copy()
        actual_fovy = float(self.modder.get_fovy(self.camera_name))
        if not np.array_equal(actual_pos, self.clean_position):
            raise PrefixReexecutionConfigError("camera position changed outside the authorized seam")
        if actual_fovy != self.clean_fovy:
            raise PrefixReexecutionConfigError("camera fovy changed outside the authorized seam")
        if not np.allclose(actual_quat, target, atol=1e-7, rtol=0.0):
            raise PrefixReexecutionConfigError("camera quaternion readback does not match requested mode")
        return camera_evidence(
            request
            or {
                "observation_index": -1,
                "preceding_action_index": None,
                "requested_camera_mode": mode,
            },
            actual_camera_parameters={
                "camera_name": self.camera_name,
                "quaternion_wxyz": actual_quat.tolist(),
                "position": actual_pos.tolist(),
                "fovy": actual_fovy,
            },
        )


class ObservationIndexedCameraWrapper:
    """Project-owned LeRobot/LIBERO integration seam with no state restore or extra step."""

    def __init__(
        self,
        delegate: Any,
        *,
        arm: str,
        controller: AgentviewYawController,
        switch_index: int = FROZEN_SWITCH_INDEX,
    ) -> None:
        if arm not in ARMS:
            raise PrefixReexecutionConfigError("camera wrapper requires a scientific arm")
        if switch_index != FROZEN_SWITCH_INDEX:
            raise PrefixReexecutionConfigError("switch_index must remain frozen at 50")
        self.delegate = delegate
        self.arm = arm
        self.controller = controller
        self.switch_index = switch_index
        self.action_index = 0
        self.camera_evidence_rows: list[dict[str, Any]] = []

    def _current_robosuite_env(self) -> Any:
        offscreen = getattr(self.delegate, "_env", None)
        task_env = getattr(offscreen, "env", None)
        if task_env is None or getattr(task_env, "sim", None) is None:
            raise PrefixReexecutionConfigError(
                "expected LeRobot LiberoEnv._env -> OffScreenRenderEnv.env -> robosuite task env"
            )
        return task_env

    def _refresh_current_observation(self) -> Any:
        task_env = self._current_robosuite_env()
        get_observations = getattr(task_env, "_get_observations", None)
        formatter = getattr(self.delegate, "_format_raw_obs", None)
        if not callable(get_observations) or not callable(formatter):
            raise PrefixReexecutionConfigError("narrow LIBERO observation refresh seam is unavailable")
        raw_obs = get_observations(force_update=True)
        return formatter(raw_obs)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self.delegate.reset(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise PrefixReexecutionConfigError("LeRobot reset must return (observation, info)")
        _, info = result
        task_env = self._current_robosuite_env()
        self.controller.bind(task_env.sim)
        request = camera_request_for_initial_observation(self.arm)
        evidence = self.controller.apply(request["requested_camera_mode"], request=request)
        self.camera_evidence_rows = [evidence]
        self.action_index = 0
        observation = self._refresh_current_observation()
        return observation, info

    def step(self, action: Any) -> Any:
        request = camera_request_before_step(self.arm, self.action_index)
        evidence = self.controller.apply(request["requested_camera_mode"], request=request)
        self.camera_evidence_rows.append(evidence)
        result = self.delegate.step(action)
        self.action_index += 1
        return result

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
        "environment_seed": FROZEN_ENVIRONMENT_SEED,
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
                        "environment_seed": FROZEN_ENVIRONMENT_SEED,
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
                    "environment_seed": FROZEN_ENVIRONMENT_SEED,
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
