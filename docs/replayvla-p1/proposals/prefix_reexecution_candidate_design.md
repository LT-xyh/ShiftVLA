# Prefix reexecution / matched-history candidate design

**Status: CANDIDATE PROPOSAL — NOT AUTHORITY.**

Repository state reviewed: `bea4e32272db6ec50b734bb778bfacefff27c84d` on `xyh/replayvla-p1`.

This document is a repo-only design artifact produced after the exact-state Paper-1 route was paused by
`docs/replayvla-p1/15_f3n_route_level_pivot.md`. It does not authorize runtime execution and does not
modify phase permissions.

## 1. Decision summary

### One-sentence scientific claim

For a frozen closed-loop VLA, matched fresh reexecution measures whether **observation corruption
leaves persistent behavioral aftereffects after the corruption is removed**, and whether accumulated
corrupted history changes sensitivity to subsequent corruption, without claiming an exact
same-physical-state counterfactual.

### Recommendation

**CONDITIONAL GO.**

The route is substantially simpler than exact-state replay and is supported by an already working
ordinary closed-loop rollout path. Two narrow pre-pilot implementation gates must close before even
a very small real micro-validation is worth authorizing:

- **G-P1 — official select-action explicit-noise preservation.** PASS must prove that the future
  transport extension still uses the same official `SmolVLAPolicy.select_action`, the same official
  processors, the same frozen `n_action_steps=1` queue semantics, and changes only the explicit
  matched flow-noise input.
- **G-P2 — observation-only camera switch timing.** PASS must prove the frozen observation/action
  index convention, no extra `env.step`, no horizon change, no action/image copying, branch-local
  rendering, and unchanged physics-relevant model/state identity.

No exact simulator restore, capture, replay, or physical-state branching is needed.

## 2. Static feasibility trace

### 2.1 Existing ordinary closed-loop rollout entry

The usable production path already exists in `scripts/m0_baseline_a.py`.

Call graph:

```text
scripts/m0_baseline_a.py
  _run_child(...)
    -> _run_episode_official(...)
       -> _run_episode_official_impl(...)
          -> dcu_preflight.build_cpu_runtime(
                 episode_config,
                 include_policy=False,
                 phase="concurrency")
          -> select_init_state_before_reset(env, planned.init_state_id)
          -> FeatureOnlyRemotePolicy(worker_client, ...)
          -> m0_smoke.PolicyProxy(...)
          -> m0_smoke.ProcessorProxy(env_preprocessor / preprocessor /
                                     postprocessor / env_postprocessor)
          -> lerobot.scripts.lerobot_eval.rollout(...)
             -> policy.select_action(current processed observation)
                -> FeatureOnlyRemotePolicy.select_action(...)
                   -> worker_client.select_action(...)
                      -> dcu_model_worker.DCUModelWorker.select_action(...)
                         -> official SmolVLA policy.select_action(batch)
             -> official policy postprocessor
             -> official environment postprocessor
             -> env.step(action)
```

This is an ordinary forward closed-loop path. It does not require
`scripts/m1_state_replay.py`, capture/restore, F3N null execution, or a fixed action tape.

Historical evidence at
`runs/m0_baseline_a/20260830T023451Z_655986_c2f3caa5/run_manifest.json`
shows that this route completed 15/15 frozen episodes for tasks 0/4/9 and init-state IDs 0..4,
with 9 successes and no runtime failures. The two serial workers took about 1801 s wall time for
15 episodes total.

### 2.2 Reconstructing the same root

The baseline route already binds:

- task suite / task ID;
- init-state ID;
- seed;
- horizon;
- frozen checkpoint and base model;
- official environment and policy processors;
- renderer environment;
- action contract `dim=7, chunk_size=50, n_action_steps=1`.

Immediately before the official reset,
`select_init_state_before_reset(...)` writes and verifies the requested `init_state_id`.
The official rollout receives the frozen seed list. A fresh environment is constructed per episode.

Therefore repeated trajectory construction from the same
`(task_id, init_state_id, seed, policy identity, renderer identity)` is source-supported.

What is **not** yet proven is that repeated full trajectories are numerically identical under every
runtime stochastic source. The pilot must measure this rather than assume it.

### 2.3 Policy state and queue semantics

The model worker exposes explicit reset and queue evidence.

`DCUModelWorker.reset(seed=...)`:

- seeds Torch and the accelerator RNG;
- calls official `policy.reset()`;
- seeds Torch again after reset;
- verifies that the action queue is empty.

`DCUModelWorker.select_action(...)` currently calls official
`policy.select_action(batch)` under inference mode and records queue length before/after and whether
a new chunk was generated. Reviewer verification of the pinned LeRobot implementation
(`huggingface/lerobot@7e241bd630a3719a56157a497ce5d08f244784f1`) establishes that official
`SmolVLAPolicy.select_action(batch, noise=...)` itself accepts explicit flow noise. With frozen
`n_action_steps=1`, the official action queue has length one: when empty it generates a chunk,
enqueues only `actions[:1]`, then returns it via `popleft()`. Therefore each environment action
query regenerates a chunk through the official `select_action` path.

The CPU-side `FeatureOnlyRemotePolicy` mirrors this queue state, and `m0_smoke.PolicyProxy`
records queue length around each call.

Thus queue/cache-like policy state that is owned by the official policy is naturally rebuilt by
starting each branch from a fresh policy reset and then reexecuting the whole prefix. No capture or
restore of that state is required.

This does **not** prove that no other hidden mutable state exists. The safe claim is narrower:
fresh branch execution reconstructs state through the same official initialization and prefix
execution path rather than serializing internal state.

### 2.4 RNG / stochasticity sources visible in repo

Relevant sources found in the checked-in path:

1. **Environment seed**: the episode root carries a fixed seed and official rollout receives it.
2. **Python / NumPy / Torch seeding**: the historical smoke path explicitly seeds Python, NumPy and
   Torch; the production model worker explicitly seeds Torch/accelerator RNG at reset.
3. **Native policy RNG**: the historical baseline declares
   `action_noise.source = native_policy_rng` and `explicit = false`.
4. **Explicit flow noise support and narrow missing transport**:
   - `dcu_preflight.generate_flow_noise(seed, shape)` creates deterministic CPU flow noise with a
     dedicated `torch.Generator`;
   - pinned official `SmolVLAPolicy.select_action(batch, noise=...)` accepts explicit flow noise;
   - current repo transport is the missing seam: `FeatureOnlyRemotePolicy.select_action` rejects
     extra args, worker `_request_bundle(..., command="select_action")` rejects `noise_path`, and
     the worker currently invokes `policy.select_action(batch)` without a noise argument.

Therefore the future Paper-1 implementation should **not** call `predict_action_chunk` manually or
reimplement action selection. It only needs a narrow transport extension carrying
`features + explicit_noise` to the same official
`policy.select_action(batch, noise=explicit_noise)`.

The current baseline is seeded but is **not** a sufficient paired-noise protocol for four-arm
scientific comparison: sharing execution order or one mutable global RNG stream is not acceptable
evidence of matched stochasticity.

### 2.5 Renderer/environment feasibility

Ordinary policy rollout has already executed on the frozen CPU-simulator / DCU-policy split.
The exact-state route later failed because of its new F3N runtime binding, not because the historical
ordinary baseline path had never worked.

The prefix-reexecution design should therefore start from the ordinary baseline execution seam, not
from F3N/F3a/F3b.

### 2.6 Camera intervention trace

At this HEAD the repository contains protocol definitions for camera perturbation, including the
Paper-1 suggestion of agentview yaw and development severities 5° / 15°, plus observation-model
fingerprinting that separates camera/rendering identity from physics identity.

However, the checked-in `scripts/` and `tests/` contain no executable branch-study camera-yaw
intervention that can be reused as-is.

This is the primary implementation gap. The candidate therefore does **not** pretend that a camera
route is already executable.

The implementation should add one narrow observation-side intervention seam to the ordinary rollout
path. It must:

- affect only the selected agentview camera observation;
- never copy an image from another rollout;
- render the current branch's own current simulator state;
- apply before the policy sees the current observation;
- be switchable by absolute environment action index;
- not advance physics, consume an action, reset the env, or alter remaining horizon;
- record clean/shifted camera parameters per step;
- demonstrate that physics-side model identity is unchanged by toggling the intervention.

A reviewer must approve the exact camera field/API before runtime use.

## 3. Components to reuse and components to bypass

### Reuse

- `m0_baseline_a._run_episode_official_impl` structure for fresh task/init/seed construction.
- `dcu_preflight.build_cpu_runtime` and the CPU/DCU split.
- `FeatureOnlyRemotePolicy` / worker transport.
- official env and policy pre/postprocessors.
- `m0_smoke.TraceStore`, `PolicyProxy`, and environment action capture ideas.
- existing task/init selection and terminal/success handling.
- existing failure/no-overwrite and root-level evidence conventions.
- explicit flow-noise generation plus the pinned official
  `SmolVLAPolicy.select_action(batch, noise=...)` capability; the future repo change is limited to
  transporting that explicit noise through the existing remote select-action boundary.
- task-equal aggregation, root clustering, technical-missing ledger and leave-one-task-out analysis
  from the approved Paper-1 protocol.

### Bypass completely for this route

- `scripts/m1_state_replay.py` state capture/restore machinery.
- restore owner closure and exact-state replay guards.
- F3N null cohort orchestration and F3b physical replay.
- recovered fixed action tape as an execution input.
- replay-specific camera sham/cache restoration machinery.
- any claim requiring identical simulator state at the switch.

Existing exact-state code may remain historical evidence, but it is not a dependency of the new
branch runner.

## 4. Proposed root schema

One scientific root should bind at least:

```text
root_id
suite
task_id
task_name
init_state_id
environment_seed
policy_noise_key_namespace
policy_checkpoint:
  repo_id
  revision
  model_sha256
base_model:
  repo_id
  revision
  model_sha256
processor_identity:
  env_preprocessor configuration/hash
  env_postprocessor configuration/hash
  policy_preprocessor configuration/hash
  policy_postprocessor configuration/hash
renderer_identity:
  MUJOCO_GL
  PYOPENGL_PLATFORM
  selected EGL device/renderer identity
camera_intervention:
  camera = agentview
  axis = yaw
  severity_degrees
  clean_definition
  shifted_definition
switch_index  # first policy observation/action index using the future condition
horizon
action_contract:
  dim = 7
  chunk_size = 50
  n_action_steps = 1
runtime_identity:
  CPU interpreter/runtime lock
  DCU interpreter/runtime lock
execution_commit
config_sha256
```

Also record per-root task/init asset identity and every terminal/technical disposition.

Branch name must **not** participate in the paired policy-noise key.

## 5. Fresh-reexecution branch construction

Freeze the closed-loop indexing convention as:

```text
obs_t
 -> policy
 -> action_t
 -> env.step(action_t)
 -> obs_{t+1}
```

`t_switch` is the first **policy observation/action index** using the future condition. For
`t_switch = 50`, `obs_0 ... obs_49` use the prefix condition and `obs_50 ...` use the future
condition. Consequently `action_49` is still chosen from prefix `obs_49`, while `action_50`
is the first action chosen from future-condition `obs_50`.

Every arm starts from a newly constructed official environment and freshly reset policy.

| Arm | observations `[0:t_switch]` | observations `[t_switch:]` |
|---|---|---|
| CC | clean | clean |
| CS | clean | shifted |
| SC | shifted | clean |
| SS | shifted | shifted |

Here Python-style `[0:50]` means observation indices 0 through 49.

Rules common to all four arms:

1. same root fields and same horizon;
2. fresh environment and fresh policy construction;
3. same task/init/seed;
4. same paired policy-noise key at each corresponding policy query;
5. policy generates its own action from the branch's current observation every step;
6. no fixed action tape;
7. no image copying;
8. no simulator state copying;
9. no artificial state alignment at the switch;
10. if an arm terminates before the switch, retain it as an absorbing outcome.

### Observation/action switch mechanics

For `obs_0`, the branch camera condition must be installed **before** `env.reset()` produces the
initial observation.

For `obs_t` where `t > 0`, the observation is returned by the preceding
`env.step(action_{t-1})`. Therefore, to make `obs_50` the first future-condition observation,
the camera must switch to the future condition **before** `env.step(action_49)`.

This convention guarantees:

- `action_49` is still determined by prefix-condition `obs_49`;
- `action_50` first observes future-condition `obs_50`;
- no extra environment step or policy query is introduced;
- the original horizon is unchanged;
- no image or simulator state is copied or restored.

The future camera seam must record at each transition:

```text
observation_index
preceding_action_index
requested_camera_mode
actual_camera_parameters
```

and must verify that camera-parameter mutation does not change physics-relevant model/state identity.

### Prefix matching evidence

Although branches are separate reexecutions, CC and CS receive the same prefix treatment and paired
noise before `t_switch`; SC and SS do likewise.

For CC vs CS and for SC vs SS, every observation/action index strictly before `t_switch` has the
same root, camera condition, paired flow-noise bytes, policy identity, processor identity and runtime
identity. Therefore any unexplained same-prefix disagreement is a **technical
matching/reexecution failure**, not a scientific treatment effect.

The runner must audit at least:

- paired noise hash;
- policy-query / observation-action index;
- requested and actual camera mode;
- action vector;
- terminal state.

Optional low-cost task-progress evidence may be retained, but it is not required for the matching
gate.

The one extra clean duplicate per root is a **same-condition duplicate discrepancy /
reexecution-noise floor** check. With only one extra duplicate per root it must not be described as a
formal variance estimate.

## 6. Paired policy randomness

### Required scientific rule

```text
fixed policy randomness != fixed action sequence
```

For each policy query use a deterministic key such as:

```text
(root_id, global_env_action_index, draw_kind="flow", draw_slot=0)
```

The arm name is excluded.

The key deterministically derives the flow-noise seed/tensor. The policy still receives each
branch's own current processed observation, so its resulting action is allowed to differ.

### Preferred implementation seam

Preserve the official select-action semantics end to end:

```text
official rollout
 -> policy.select_action(current observation)
 -> prefix-reexecution wrapper derives paired flow noise
 -> remote client select_action(features, explicit_noise)
 -> DCU worker
 -> official SmolVLAPolicy.select_action(batch, noise=explicit_noise)
 -> official postprocessor
 -> env.step
```

The implementation must **not** manually call `predict_action_chunk` and then choose an action.
With frozen `n_action_steps=1`, the official select-action queue semantics already provide one
environment action per query while generating a new chunk whenever the queue is empty.

Thus the only intended policy-path change is explicit matched noise transported through the existing
remote select-action seam. The action remains a function of the branch's current observation and its
paired noise.

**G-P1 PASS** requires static/fake evidence that the path still uses:

- the same official `SmolVLAPolicy.select_action`;
- the same official env/policy pre/postprocessors;
- the same `n_action_steps=1` queue semantics;
- explicit matched noise as the only action-generation change.

If this cannot be preserved, stop and redesign the RNG seam; do not substitute episode-level native
seeding and call it matched noise.

### Other RNGs

Fresh construction should also freeze and record:

- Python seed;
- NumPy seed;
- Torch CPU seed;
- Torch/DCU seed;
- environment seed.

These are secondary to the explicit per-query flow-noise key and must not be used as a substitute
for it.

## 7. Switch-index decision

### Fixed temporal switches

Advantages:

- no event detector;
- same intervention schedule across arms;
- no post-treatment switch selection;
- straightforward root matching and bootstrap clustering.

Disadvantage: the same absolute time can correspond to different manipulation phases across tasks.

### Semantic/event switches

Contact onset, grasp, carried and release are scientifically meaningful, but their timing is itself
changed by the prefix treatment. Triggering the switch on an arm-specific event would therefore
condition intervention time on a post-treatment variable and contaminate the primary comparison.

### Paper-1 choice

Use **fixed absolute temporal switches** for primary inference.

The old `{20, 50, 80}` set is an approved historical candidate, not something this new route
should inherit automatically. For the **minimal pilot**, use only:

```text
t_switch = 50
```

Reason: it is an existing predeclared candidate, is far enough from initialization to permit a
nontrivial prefix, and keeps the pilot to one decomposition cell per root. The pilot is not allowed
to select a more favorable switch after inspecting outcomes.

If the pilot succeeds, the full-study freeze may adopt `{20, 50, 80}` or another fixed set, but
that choice must be made from development evidence before held-out results are visible.

Semantic contact/grasp/carried annotations remain diagnostic covariates only.

## 8. Camera intervention choice

The first implementation should support only:

```text
clean agentview
agentview yaw = +15 degrees
```

No second axis and no corruption suite.

Why 15° for the pilot:

- 5° risks producing an underpowered feasibility result;
- 15° is already inside the approved development suggestion;
- one strength is sufficient to ask whether future corruption and history burden produce separable
  closed-loop structure.

If 15° causes immediate near-universal failure before meaningful closed-loop history develops, the
pilot fails the useful-intervention criterion. Do not automatically add many severities. A single
predeclared downgrade to 5° would require a new reviewed pilot version.

"Clean future" always means a clean render of **that branch's current physical state**. It never
means the image from CC or another rollout.

**G-P2 PASS** requires static/fake evidence that the observation-index convention above is exact,
the camera condition for `obs_0` is established before reset returns the initial observation, the
future condition for `obs_50` is established before `env.step(action_49)`, and the switch causes
no extra `env.step`, policy query, horizon change, action/image copying, or physics-relevant
identity change.

## 9. Estimands and metrics

Let `Y_arm` be final task success within the original horizon.

### Primary estimands

Use the sign convention from the approved protocol:

```text
G_C = E[Y_CC - Y_CS]
G_S = E[Y_SC - Y_SS]
L   = E[Y_CC - Y_SC]
I   = G_S - G_C
```

Interpretation:

- `G_C`: future observation corruption effect after a clean prefix.
- `G_S`: future observation corruption effect after a shifted prefix.
- `L`: total accumulated shifted-history aftereffect under a clean future.
- `I`: whether accumulated history changes sensitivity to future corruption.

For the reexecution route, `L` is explicitly **not** a pure physical-state-mediated effect.

### Primary paper quantities and diagnostics

**Primary**

1. final success `Y`;
2. `L = E[Y_CC - Y_SC]`: persistent total aftereffect / operational hysteresis under clean future;
3. `I = (Y_SC - Y_SS) - (Y_CC - Y_CS)`: history × future-corruption interaction.

`G_C` and `G_S` remain necessary intervention/future-effect estimands, but they cannot by
themselves satisfy the Paper-1 signal gate.

**Diagnostics**

- terminal step / remaining-budget survival;
- post-switch action divergence;
- task-progress / predicate evidence.

Contact/grasp trajectory evidence is optional only if the ordinary runner can expose it at low
engineering cost. Do not rebuild exact-state contact instrumentation merely to reproduce the old
route.

Do not promote every available state field to a paper metric.

## 10. Causal claim boundary

### Allowed

- Under a clean prefix, CC vs CS estimates the closed-loop effect of future camera corruption under
  matched root and policy randomness.
- Under a shifted prefix, SC vs SS estimates the analogous future-camera effect.
- CC vs SC estimates the **persistent total aftereffect / history burden** accumulated under
  different prefix observation histories when both subsequently run clean. Operationally, this is
  closed-loop observation-corruption **hysteresis**: persistence after the corruption is removed.
- The difference-of-differences `I` asks whether shifted history changes future-corruption
  susceptibility.
- Fresh reexecution trades exact-state branching for a simpler matched-history decomposition.

"Hysteresis" here is strictly an operational sequential-intervention term. It does not imply
identical physical state, identical-state mediation, or an exact-state hysteresis experiment.

### Forbidden

Do not call CC vs SC:

- pure physical drift;
- a history-only causal effect at identical physical state;
- an exact-state counterfactual;
- a simulator-state-matched mediation effect.

Do not claim that the four branches share the same physical state at `t_switch`.

Do not claim that a fixed noise tensor is a fixed action sequence.

Do not claim camera robustness alone as the contribution.

## 11. Minimal falsifiable pilot

### Objective

The pilot asks whether this route is worth a full Paper-1 study, not whether the final hypothesis is
already significant.

It must test four things:

1. same-root fresh reexecution is stable enough for matched comparisons;
2. the +15° camera intervention changes closed-loop behavior without degenerating into universal
   immediate failure;
3. corruption removal leaves a repeated **history-specific persistent aftereffect** rather than
   only an immediate robustness drop;
4. accumulated shifted history changes subsequent susceptibility and/or leaves post-switch
   trajectory persistence beyond one-step action disagreement.

### Frozen pilot matrix

Development tasks: **0 and 4**.

Roots: init-state IDs **0,1,2,3** for each task = **8 roots**.

Switch: **50** only.

Corruption strength: **+15° agentview yaw** only.

Main branches: **4 per root** = **32 main rollouts**.

Reexecution control: one extra clean duplicate per root = **8 control rollouts**, used only to
measure the same-condition duplicate discrepancy / reexecution-noise floor, not a formal variance
estimate.

Total: **40 rollouts**.

All arms retain the original maximum horizon 280 and absorbing early terminals.

### Runtime estimate

The historical ordinary baseline completed 15 episodes in about 1801 s wall time using two serial
workers. Linear scaling gives roughly 4800 s (~80 min) for 40 similarly distributed rollouts before
additional logging/intervention overhead.

For planning, reserve **2–3 hours wall time** for the pilot execution itself after qualification,
not days. This estimate is evidence-based but not a promise; camera intervention and explicit-noise
plumbing may add overhead.

### Pilot pass requirements

All are required to continue:

1. **Reexecution stability**
   - every root has its planned technical disposition;
   - no unexplained prefix mismatch in CC/CS or SC/SS before the switch;
   - clean duplicate outcomes do not show route-level instability large enough to dominate arm
     differences.
2. **Intervention validity**
   - per-step evidence proves clean/shifted camera mode switched at exactly index 50;
   - no extra env step/action/query is introduced by switching;
   - branch renders come from their own current state.
3. **Matched randomness**
   - every matched policy query records the same noise key/hash across arms;
   - different observations may produce different actions.
4. **Intervention gate — necessary but not sufficient for a paper**
   - at least one of `G_C`, `G_S`, or a corresponding trajectory response demonstrates that the
     +15° shift is not behaviorally inert;
   - shifted arms are not near-universally terminal before useful post-switch observation.
5. **Paper-signal gate — required to authorize a full study**
   - a history-specific persistent structure is visible through `L` and/or `I`, together with
     corresponding post-switch trajectory persistence;
   - the structure is observed as a repeated qualitative pattern in **both task 0 and task 4**;
   - it is not plausibly explained by the clean same-condition duplicate discrepancy;
   - it appears after corruption removal / after the switch, rather than only as a one-step action
     difference at the switching boundary.

The pilot is descriptive; no p-value significance threshold is required.

If `G_C` and/or `G_S` are clearly nonzero but `L ≈ 0`, `I ≈ 0`, and there is no persistent
post-switch trajectory signature, the route is **PIVOT**, even if camera corruption clearly reduces
success. Do not scale that outcome into a generic robustness paper.

### Kill criteria

Stop or redesign before a full study if any holds:

- same-condition duplicate discrepancy / reexecution-noise floor is comparable to or larger than
  the history-specific contrasts;
- paired flow-noise semantics cannot be implemented without changing the frozen policy action
  contract;
- camera shift cannot be applied as observation-only intervention without unintended physics/time
  side effects;
- the shift has no measurable behavioral/trajectory effect on the development roots;
- the shift causes near-universal immediate failure and therefore cannot expose history/future
  decomposition;
- task 0 and task 4 do not both show repeated post-switch history-specific persistence beyond the
  clean duplicate discrepancy, including the case where `G_C/G_S` are nonzero but `L` and `I`
  are approximately zero with no persistent trajectory signature;
- technical completion is insufficient for root-level matched estimates;
- implementation/runtime cost grows back toward exact-state-infrastructure scale.

## 12. Full study projection — contingent, not authorized

Only if the pilot passes:

- retain task-level development/held-out separation;
- choose held-out tasks before looking at their results;
- freeze switch set and severity from development evidence;
- use 20/30/40 roots per task only after a development precision calculation;
- keep all arms/severities/switches from one root in the same statistical cluster;
- estimate task means first, then task-equal aggregate;
- clustered bootstrap over roots within task;
- report leave-one-task-out;
- retain full denominators and technical missingness;
- do not replace failed roots.

A plausible full study remains the approved six held-out tasks with 30 roots/task if compute and
pilot precision support it, but that scale is not a current execution requirement.

## 13. Novelty / rejection-risk review

### Likely reviewer objections

1. **"This is only camera-corruption robustness."**
   - Valid risk. A paper that only reports success degradation is weak.
2. **"There is no exact counterfactual."**
   - Correct. This route intentionally gives up exact physical-state branching.
3. **"CC vs SC confounds state divergence."**
   - Correct; therefore it is named total history aftereffect, not pure physical drift.
4. **"Fresh reexecution is obvious."**
   - The execution mechanism itself is not novelty.
5. **"Success-rate decomposition is too shallow."**
   - Also valid unless repeated trajectory-level history/future structure is observed.

### Where paper value can still come from

The potentially publishable object is not reexecution. It is the empirical characterization of
**closed-loop observation-corruption hysteresis / persistent aftereffect** and its interaction with
subsequent observation corruption under a matched, root-clustered protocol:

- future corruption conditional on prefix history;
- total aftereffect of corrupted history;
- history × future-corruption interaction;
- trajectory-level failure onset/contact/task-progress diagnostics;
- contact-rich manipulation rather than one-step action sensitivity.

Novelty must be presented conservatively as a diagnostic experimental framework/result, not a new
causal-identification theorem.

### Venue-risk judgment

The pivot is **not yet strong enough for an unconditional CCF-B claim**. If the pilot only shows
"15° camera yaw lowers success", the route should be pivoted again rather than scaled.

If the pilot shows a stable, task-repeated separation between future corruption and accumulated
history burden, with a meaningful interaction or trajectory diagnostic not visible in immediate
action error alone, a CCF-B / Zone-3-class paper remains plausible within the current scope.

## 14. Minimal implementation plan after reviewer approval

No code is proposed in this commit.

If this design is approved, future implementation is limited to three narrow change classes:

1. **explicit-noise support on the existing remote `select_action` transport**, carrying
   features + paired noise into official
   `SmolVLAPolicy.select_action(batch, noise=explicit_noise)` without changing queue semantics;
2. **an observation-indexed camera controller** implementing the frozen
   `obs_t -> action_t -> env.step -> obs_{t+1}` timing convention;
3. **a thin four-arm fresh-reexecution orchestrator/evidence layer** reusing the ordinary baseline
   runtime, official processors, worker client, terminal handling and evidence conventions.

A likely file shape remains:

```text
configs/replayvla/p1_prefix_reexecution_pilot.yaml
scripts/p1_prefix_reexecution.py
tests/test_p1_prefix_reexecution.py
```

Do not copy or adapt the F3N orchestration, `m1_state_replay`, null oracle, or exact-state owner
closure into the new route.

Focused fake/unit tests should cover:

- exact 4-arm branch schedule;
- fresh construction per arm;
- exact off-by-one observation semantics:
  - CC/CS: `obs[0:50]` clean;
  - SC/SS: `obs[0:50]` shifted;
  - CC/SC: `obs[50:]` clean;
  - CS/SS: `obs[50:]` shifted;
- camera mode for `obs_0` is installed before reset produces the initial observation;
- future camera mode for `obs_50` is installed before `env.step(action_49)`;
- absorbing pre-switch terminal;
- paired noise key excludes arm and includes root/step;
- matched key -> same noise bytes, while action is never copied;
- official `select_action` and `n_action_steps=1` queue semantics are preserved under explicit
  noise transport;
- camera mode changes observation-side path only;
- no extra policy query/env step at switch;
- CC/CS and SC/SS same-prefix audit checks noise hash, query index, camera mode, action and terminal
  state;
- technical failure preserved with no retry;
- root-level clustering metadata;
- held-out schedule cannot be changed after freeze.

No exact-state capture/restore helper should be imported by the new runner.

## 15. Answers to the required candidate questions

1. **Scientific claim:** matched fresh reexecution measures whether observation corruption leaves
   persistent behavioral aftereffects after the corruption is removed, and whether accumulated
   corrupted history changes sensitivity to subsequent corruption.
2. **Why no exact restore:** all branch-specific hidden/policy/controller state is regenerated by
   ordinary initialization plus full prefix execution.
3. **Branch construction:** four complete fresh rollouts with clean/shifted mode selected by prefix
   and future relative to a fixed switch.
4. **Allowed causal claims:** conditional future-corruption effects and total history aftereffect.
5. **Forbidden claims:** identical-state counterfactual, pure physical drift/history-only mediation.
6. **Policy stochasticity:** deterministic root/step explicit flow-noise keys transported into the
   official `SmolVLAPolicy.select_action(batch, noise=...)`; never fixed actions and never manual
   `predict_action_chunk` selection.
7. **Camera intervention:** one observation-only agentview yaw seam; +15° in the pilot.
8. **Switch:** fixed policy observation/action index; pilot uses 50, meaning `obs_0..obs_49` are
   prefix observations and `obs_50` is the first future-condition observation.
9. **Pilot size:** 2 tasks × 4 roots/task × 4 arms + 8 clean repeats.
10. **Pilot branch count:** 32 main + 8 controls = **40 rollouts**.
11. **PASS/KILL:** intervention activity alone is insufficient; full-study continuation requires
    repeated history-specific persistent structure in both development tasks beyond the clean
    duplicate discrepancy.
12. **Full study:** task-equal root-clustered held-out study only after pilot success.
13. **Reusable code:** ordinary baseline runner, CPU/DCU policy seam, processors, explicit-noise API,
    evidence/statistics conventions.
14. **Bypass:** all exact-state restore/null/F3b machinery.
15. **Largest rejection risk:** result collapses to ordinary camera-robustness curves.
16. **1–2 month feasibility:** plausible if the two narrow implementation gates close quickly and
    the pilot produces history-specific persistent structure; otherwise stop early.
17. **Pre-pilot gates:** G-P1 official select-action explicit-noise preservation and G-P2 exact
    observation-indexed camera switching must both pass static/fake review before any small real
    micro-validation is worth authorizing.
18. **Recommendation:** **CONDITIONAL GO**.
