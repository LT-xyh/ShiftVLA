# ReplayVLA-P1 CURRENT EXECUTION

Status: PREFIX-REEXECUTION G-P1 / G-P2 REPO-ONLY IMPLEMENTATION AUTHORIZED
Branch: xyh/replayvla-p1

Latest authority:
- docs/replayvla-p1/16_prefix_reexecution_paper_route.md

Reviewed proposal commit:
- afff8b3653fdda5314f301d2bdd4eb4e26517af0

Historical exact-state route:
- paused for Paper-1;
- F3N/F3b/exact-state repair remain unauthorized.

## Current scientific route

Paper-1 target:

**prefix reexecution / matched-history closed-loop persistence**

Central claim target:

Matched fresh reexecution asks whether observation corruption leaves persistent behavioral
aftereffects after corruption is removed, and whether accumulated corrupted history changes
subsequent corruption susceptibility.

This route does not claim exact physical-state branching.

## Frozen pilot design

- suite: libero_spatial
- tasks: 0 and 4
- init-state IDs: 0, 1, 2, 3
- roots: 8
- switch: t_switch = 50
- corruption: clean vs agentview yaw +15 degrees
- main arms: CC / CS / SC / SS
- one clean duplicate per root
- total planned pilot rollouts: 40
- original horizon retained
- no retry/replacement

The pilot is NOT runtime-authorized yet.

## Frozen timing

`obs_t -> policy -> action_t -> env.step(action_t) -> obs_{t+1}`

For t_switch=50:
- obs_0..obs_49 use prefix condition;
- obs_50 onward use future condition;
- action_49 is chosen from prefix obs_49;
- camera future condition is installed before env.step(action_49) so returned obs_50 is future-condition;
- action_50 is the first action chosen from future-condition obs_50.

No extra env.step or policy query is allowed.

## G-P1

Implement narrow explicit-noise support through the existing remote select-action path.

The final model call must remain:

`SmolVLAPolicy.select_action(batch, noise=explicit_noise)`

Pinned LeRobot:
`7e241bd630a3719a56157a497ce5d08f244784f1`

Preserve:
- official select_action;
- official queue semantics;
- n_action_steps=1;
- official env/policy processors;
- action generation from each branch's own observation.

Do not replace this with manual predict_action_chunk selection.

Paired-noise key must exclude arm identity and bind root/query identity.

## G-P2

Implement/source-account a narrow observation-indexed agentview-yaw controller.

Must prove with source trace + fake/unit tests:
- obs_0 mode installed before reset returns;
- obs_50 future mode installed before env.step(action_49);
- no extra env.step;
- no extra policy query;
- no reset/horizon change;
- no action/image copy;
- branch-local observation rendering;
- physics-relevant model/state identity unchanged by camera mutation.

If the pinned source/API does not support a narrow observation-only seam without invasive runtime
machinery, stop and report G-P2 BLOCKED rather than building another infrastructure stack.

## Same-prefix audit

Before the switch:
- CC vs CS must match;
- SC vs SS must match.

Audit at least:
- noise hash;
- observation/action query index;
- requested/actual camera mode;
- action;
- terminal state.

Any unexplained same-prefix disagreement is technical failure.

## Repo-only implementation scope

Authorized change classes:
1. explicit-noise support on existing remote select_action transport;
2. observation-indexed camera controller;
3. thin four-arm schedule/evidence/orchestration layer;
4. config/schema and fake/unit/static tests needed to prove G-P1/G-P2.

Likely files may include:
- scripts/dcu_model_worker.py
- scripts/dcu_preflight.py
- configs/replayvla/p1_prefix_reexecution_pilot.yaml
- scripts/p1_prefix_reexecution.py
- tests/test_p1_prefix_reexecution.py
- focused existing worker/preflight tests if needed

Keep changes minimal.

Do not import or copy exact-state machinery.

## Explicitly forbidden in this phase

Do not run:
- real LIBERO environment construction;
- EGL discovery;
- render;
- env.step;
- policy inference;
- CPU/DCU real rollout;
- camera runtime experiment;
- pilot;
- micro-validation;
- Codex runtime execution.

Do not:
- repair F3N/F3b;
- modify numbered authority 01-16;
- modify historical evidence;
- install/change packages;
- create a second experimental runtime;
- add broad contact/state instrumentation.

## Required repo-only acceptance tests

At minimum cover:
- 4-arm schedule;
- paired-noise key excludes arm;
- matched key produces same noise bytes;
- explicit noise transported through select_action;
- worker calls official select_action with noise;
- n_action_steps=1 queue semantics preserved in fakes;
- no action copying;
- obs[0:50] / obs[50:] branch conditions exactly match authority;
- obs_0 camera condition before reset return;
- obs_50 future condition before env.step(action_49);
- no extra env.step/query at switch;
- absorbing pre-switch terminal;
- same-prefix audit failure is fail-closed;
- no retry/replacement;
- no import/dependency on m1_state_replay/F3N/F3b.

## After implementation

Produce one minimal candidate commit.

Do not advance to runtime automatically.

Independent reviewer must inspect:
- source-accounted camera API;
- G-P1 semantics;
- G-P2 timing;
- test coverage;
- scope.

Only after that review may a tiny real micro-validation be considered.

## Codex state

Codex runtime work: NOT AUTHORIZED.

Use Web Implementer for this repo-only implementation.
