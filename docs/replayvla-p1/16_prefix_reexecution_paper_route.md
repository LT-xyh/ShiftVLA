# Prefix reexecution Paper-1 route authority

**Status: CONDITIONAL GO — REPO-ONLY IMPLEMENTATION OF G-P1 / G-P2 AUTHORIZED.**

Authority branch: `xyh/replayvla-p1`.

Reviewed proposal commit:

`afff8b3653fdda5314f301d2bdd4eb4e26517af0`

Reviewed proposal:

`docs/replayvla-p1/proposals/prefix_reexecution_candidate_design.md`

Predecessor route authority:

`docs/replayvla-p1/15_f3n_route_level_pivot.md`

## 1. Paper-1 scientific target

For a frozen closed-loop VLA, matched fresh reexecution measures whether observation corruption
leaves persistent behavioral aftereffects after the corruption is removed, and whether accumulated
corrupted history changes sensitivity to subsequent corruption.

This is an operational sequential-intervention study of closed-loop persistence / hysteresis.

It is not an exact-state counterfactual study.

## 2. Branch construction

Every arm is a fresh full forward execution from the same frozen root.

No simulator capture/restore is used.

For a fixed switch index `t_switch`:

| arm | observations before switch | observations from switch onward |
|---|---|---|
| CC | clean | clean |
| CS | clean | shifted |
| SC | shifted | clean |
| SS | shifted | shifted |

All arms bind the same:
- task and init state;
- environment seed;
- policy/checkpoint identity;
- processor identity;
- renderer identity;
- horizon;
- paired policy-noise namespace;
- execution/config identity.

No image, action, simulator state, policy queue, or hidden state is copied between arms.

## 3. Frozen timing convention

The primary closed-loop indexing convention is:

`obs_t -> policy -> action_t -> env.step(action_t) -> obs_{t+1}`

`t_switch` is the first policy observation/action index using the future condition.

For the pilot, `t_switch = 50` means:
- `obs_0 ... obs_49`: prefix condition;
- `obs_50 ...`: future condition;
- `action_49` is selected from prefix `obs_49`;
- `action_50` is the first action selected from future-condition `obs_50`.

The camera mode for `obs_0` must be installed before reset returns the initial observation.

For `obs_50`, the future camera mode must be installed before `env.step(action_49)`.

The switch must not add an environment step, policy query, reset, horizon change, copied image, or
copied action.

## 4. Paired policy randomness

Fixed policy randomness is not a fixed action sequence.

The paired key excludes arm identity and binds root/query identity, e.g.:

`(root_id, observation_action_index, draw_kind="flow", draw_slot=0)`

The preferred implementation preserves the pinned official LeRobot API:

`SmolVLAPolicy.select_action(batch, noise=explicit_noise)`

Pinned LeRobot revision:

`7e241bd630a3719a56157a497ce5d08f244784f1`

Frozen `n_action_steps = 1`.

Do not manually call `predict_action_chunk` and select an action outside official
`select_action` semantics.

The repo change must be limited to transporting explicit paired noise through the existing remote
select-action boundary while preserving official queue semantics and official pre/postprocessors.

## 5. Pre-pilot implementation gates

### G-P1 — official select-action explicit-noise preservation

PASS requires static/fake/unit evidence that:
- the final model call is the same official `SmolVLAPolicy.select_action`;
- explicit noise reaches it as `noise=...`;
- official env/policy pre/postprocessors are unchanged;
- frozen `n_action_steps=1` queue semantics remain intact;
- matched noise bytes can be reproduced from the root/query key;
- arm identity does not enter the paired-noise key;
- actions are never copied between arms.

If satisfying G-P1 requires replacing official select-action semantics, G-P1 fails.

### G-P2 — observation-only camera switch timing

PASS requires source-accounted plus fake/unit evidence that:
- the selected intervention changes only the intended agentview camera observation path;
- `obs_0` condition is installed before reset returns;
- `obs_50` future condition is installed before `env.step(action_49)`;
- the recorded observation index, preceding action index, requested camera mode, and actual camera
  parameters agree;
- no extra `env.step`, policy query, reset, action copy, image copy, or horizon change occurs;
- branch images come from that branch's current simulator state;
- camera mutation does not change physics-relevant model/state identity.

If a narrow observation-only camera seam cannot be established from the pinned LIBERO/robosuite
source/API without invasive runtime machinery, G-P2 fails.

## 6. Same-prefix audit

Before `t_switch`:
- CC and CS must match under the same clean prefix;
- SC and SS must match under the same shifted prefix.

Audit at least:
- paired-noise hash;
- observation/action query index;
- requested/actual camera mode;
- action;
- terminal state.

Any unexplained same-prefix disagreement is a technical matching/reexecution failure, not a
scientific history effect.

The extra clean duplicate per root is only a same-condition discrepancy / reexecution-noise-floor
check. It is not a formal variance estimate.

## 7. Pilot matrix

The pilot is frozen to:

- suite: `libero_spatial`;
- tasks: 0 and 4;
- init-state IDs: 0, 1, 2, 3 for each task;
- roots: 8 total;
- switch: `t_switch = 50`;
- corruption: clean vs agentview yaw `+15 degrees`;
- four scientific arms per root: CC / CS / SC / SS;
- one extra clean duplicate per root;
- 32 main rollouts + 8 duplicate controls = 40 total rollouts;
- original horizon retained;
- absorbing early terminal retained;
- no retry or replacement.

This pilot is NOT yet runtime-authorized.

## 8. Primary estimands and scientific boundary

Let `Y` denote final success.

`G_C = E[Y_CC - Y_CS]`

`G_S = E[Y_SC - Y_SS]`

`L = E[Y_CC - Y_SC]`

`I = (Y_SC - Y_SS) - (Y_CC - Y_CS)`

Interpretation:
- `G_C`: future-corruption susceptibility after clean history;
- `G_S`: future-corruption susceptibility after shifted history;
- `L`: persistent total aftereffect / operational hysteresis under clean future;
- `I`: history-dependent future-corruption susceptibility.

Allowed claims concern matched fresh-reexecution effects and persistent total aftereffects.

Forbidden claims include:
- identical physical state at the switch;
- pure physical drift;
- exact-state counterfactual;
- identical-state history-only mediation;
- exact simulator-state decomposition.

## 9. Paper-signal gate

A nonzero `G_C` or `G_S` is necessary evidence that the intervention is behaviorally active, but
is not sufficient to continue to a full study.

A full study may be considered only if the pilot shows history-specific persistent structure:
- through `L` and/or `I`;
- with corresponding post-switch trajectory persistence;
- as a repeated qualitative pattern in both task 0 and task 4;
- beyond the same-condition clean duplicate discrepancy;
- not only as a one-step action difference at the switching boundary.

If camera shift changes success but `L ≈ 0`, `I ≈ 0`, and there is no persistent post-switch
trajectory signature, the route pivots rather than scaling into a generic robustness paper.

No pilot p-value significance threshold is required.

## 10. Engineering budget

The new route must remain a thin extension of the already working ordinary baseline path.

Authorized repo-only implementation classes:
1. explicit-noise support on the existing remote `select_action` transport;
2. an observation-indexed camera controller;
3. a thin four-arm fresh-reexecution schedule/evidence layer.

Do not copy or depend on:
- `m1_state_replay.py` capture/restore;
- F3N orchestration;
- F3N null oracle;
- F3b physical replay;
- exact-state owner closure;
- fixed action tape execution.

Contact/grasp diagnostics are optional only if obtainable at low engineering cost.

## 11. Current permissions

AUTHORIZED:
- repo/source analysis;
- G-P1 implementation;
- G-P2 implementation;
- thin pilot config/schema/orchestrator skeleton;
- fake/unit/static tests;
- source-accounted camera API trace;
- deterministic schedule/evidence generation tests.

NOT AUTHORIZED:
- real LIBERO environment construction;
- EGL/runtime experiment;
- render;
- `env.step`;
- policy inference on CPU/DCU;
- pilot rollout;
- micro-validation;
- Codex runtime execution;
- F3N/F3b/exact-state repair;
- full-study execution.

After G-P1 and G-P2 implementation, an independent reviewer must approve the candidate before any
real micro-validation.
