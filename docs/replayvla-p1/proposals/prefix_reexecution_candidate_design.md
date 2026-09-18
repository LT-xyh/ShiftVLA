# Prefix reexecution / matched-history candidate design

**Status: CANDIDATE PROPOSAL — NOT AUTHORITY.**

Repository state reviewed: `bea4e32272db6ec50b734bb778bfacefff27c84d` on `xyh/replayvla-p1`.

This document is a repo-only design artifact produced after the exact-state Paper-1 route was paused by
`docs/replayvla-p1/15_f3n_route_level_pivot.md`. It does not authorize runtime execution and does not
modify phase permissions.

## 1. Decision summary

### One-sentence scientific claim

For a frozen closed-loop VLA, matched fresh reexecution from the same initial condition can separate
the effect of **future observation corruption conditional on a fixed prefix treatment** from the
**total aftereffect of previously corrupted closed-loop history**, without claiming an exact
same-physical-state counterfactual.

### Recommendation

**CONDITIONAL GO.**

The route is substantially simpler than exact-state replay and is supported by an already working
ordinary closed-loop rollout path. Two narrow items must be closed before a pilot is authorized:

1. an executable, observation-only camera intervention seam does not currently exist in this
   repository; only protocol-level camera/yaw definitions exist;
2. paired policy stochasticity should use the repository's explicit-flow-noise policy API rather
   than relying on mutable global RNG, and the one-action-per-query semantics must be validated
   against the frozen `n_action_steps=1` contract before real pilot execution.

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

`DCUModelWorker.select_action(...)` calls official
`policy.select_action(batch)` under inference mode and records queue length before/after and whether
a new chunk was generated.

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
4. **Explicit flow noise API**:
   - `dcu_preflight.generate_flow_noise(seed, shape)` creates deterministic CPU flow noise with a
     dedicated `torch.Generator`;
   - `DCUModelWorker.predict_action_chunk(...)` accepts explicit noise and calls the official
     `policy.predict_action_chunk(batch, noise=noise)`;
   - `select_action(...)` intentionally rejects caller-supplied noise and uses native policy RNG.

The current baseline is therefore seeded but is **not** a sufficient paired-noise protocol for
four-arm scientific comparison: sharing execution order or one mutable global RNG stream is not
acceptable evidence of matched stochasticity.

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
- explicit flow-noise generation and the model worker's
  `predict_action_chunk(batch, noise=...)` seam.
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
switch_index
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

Let environment action indices be zero-based. `t_switch` is the first index executed under the
future condition.

Every arm starts from a newly constructed official environment and freshly reset policy.

| Arm | steps `< t_switch` | steps `>= t_switch` |
|---|---|---|
| CC | clean | clean |
| CS | clean | shifted |
| SC | shifted | clean |
| SS | shifted | shifted |

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

### Prefix matching evidence

Although branches are separate reexecutions, CC and CS receive the same prefix treatment and paired
noise before `t_switch`; SC and SS do likewise.

The runner should therefore retain a prefix audit:

- terminal status before switch;
- per-step policy-query index;
- paired-noise key/hash;
- action vector;
- selected proprio/task progress diagnostics.

A disagreement before the switch is a reexecution/matching failure to be reported, not silently
treated as history effect.

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

Use the already checked-in explicit path:

```text
current branch observation
 -> official preprocessors
 -> deterministic flow noise for (root, step)
 -> policy.predict_action_chunk(batch, noise=xi_root_step)
 -> take the action required by frozen n_action_steps=1 semantics
 -> official postprocessors
 -> env.step
```

This preserves closed-loop dependence on the current observation while matching stochastic input.

Before pilot runtime, one bounded validation must show that the explicit one-action path is
compatible with the frozen `n_action_steps=1` policy contract. If it materially changes action
semantics relative to the official closed-loop path, stop and redesign the RNG seam; do not fall
back to claiming that episode-level native seeding is matched noise.

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

### Trajectory metrics

Keep four metrics only.

**Primary**

1. final success `Y`.

**Diagnostic**

2. terminal step / remaining-budget survival;
3. action disagreement relative to the matched same-prefix comparator, summarized per root and
   split pre/post switch;
4. task-progress/contact state trajectory using already available success/contact/grasp/carried
   annotations where the ordinary runner can obtain them without exact-state instrumentation.

Do not promote every available state field to a paper metric.

## 10. Causal claim boundary

### Allowed

- Under a clean prefix, CC vs CS estimates the closed-loop effect of future camera corruption under
  matched root and policy randomness.
- Under a shifted prefix, SC vs SS estimates the analogous future-camera effect.
- CC vs SC estimates the **total aftereffect / history burden** accumulated under different prefix
  observation histories when both subsequently run clean.
- The difference-of-differences `I` asks whether shifted history changes future-corruption
  sensitivity.
- Fresh reexecution trades exact-state branching for a simpler matched-history decomposition.

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
3. CC/CS/SC/SS yields a nontrivial history/future structure on at least some roots;
4. trajectory diagnostics expose something beyond a flat success-rate drop.

### Frozen pilot matrix

Development tasks: **0 and 4**.

Roots: init-state IDs **0,1,2,3** for each task = **8 roots**.

Switch: **50** only.

Corruption strength: **+15° agentview yaw** only.

Main branches: **4 per root** = **32 main rollouts**.

Reexecution control: one extra clean duplicate per root = **8 control rollouts**.

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
4. **Nondegenerate signal**
   - not all 32 main branches have identical trajectories/outcomes;
   - shifted arms are not near-universally terminal before useful post-switch observation;
   - at least one of `G_C`, `G_S`, `L`, or `I` shows a root-level pattern large enough to
     justify precision estimation, or trajectory diagnostics show a clear repeated divergence
     pattern tied to history/future condition.

The pilot is descriptive; no p-value threshold is a pass criterion.

### Kill criteria

Stop or redesign before a full study if any holds:

- same-condition reexecution variance is comparable to or larger than the treatment contrasts;
- paired flow-noise semantics cannot be implemented without changing the frozen policy action
  contract;
- camera shift cannot be applied as observation-only intervention without unintended physics/time
  side effects;
- the shift has no measurable behavioral/trajectory effect on the development roots;
- the shift causes near-universal immediate failure and therefore cannot expose history/future
  decomposition;
- CC/SC history burden is indistinguishable from ordinary reexecution noise on development data and
  diagnostics add no repeated structure;
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
**closed-loop history burden** and its interaction with subsequent observation corruption under a
matched, root-clustered protocol:

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

If this design is approved, implementation should be a thin branch-study layer rather than a copy
of the exact-state stack:

```text
configs/replayvla/p1_prefix_reexecution_pilot.yaml
scripts/p1_prefix_reexecution.py
tests/test_p1_prefix_reexecution.py
```

The runner should reuse the ordinary baseline runtime builder, processor path, worker client,
terminal handling and evidence conventions.

Focused fake/unit tests should cover:

- exact 4-arm branch schedule;
- fresh construction per arm;
- switch boundary `<50` vs `>=50`;
- absorbing pre-switch terminal;
- paired noise key excludes arm and includes root/step;
- matched key -> same noise bytes, while action is never copied;
- camera mode changes observation-side path only;
- no extra policy query/env step at switch;
- CC/CS and SC/SS prefix audit;
- technical failure preserved with no retry;
- root-level clustering metadata;
- held-out schedule cannot be changed after freeze.

No exact-state capture/restore helper should be imported by the new runner.

## 15. Answers to the required candidate questions

1. **Scientific claim:** accumulated closed-loop observation history and future observation
   corruption can be separately characterized by matched fresh reexecution.
2. **Why no exact restore:** all branch-specific hidden/policy/controller state is regenerated by
   ordinary initialization plus full prefix execution.
3. **Branch construction:** four complete fresh rollouts with clean/shifted mode selected by prefix
   and future relative to a fixed switch.
4. **Allowed causal claims:** conditional future-corruption effects and total history aftereffect.
5. **Forbidden claims:** identical-state counterfactual, pure physical drift/history-only mediation.
6. **Policy stochasticity:** deterministic root/step explicit flow-noise keys, never fixed actions.
7. **Camera intervention:** one observation-only agentview yaw seam; +15° in the pilot.
8. **Switch:** fixed absolute index; pilot uses 50.
9. **Pilot size:** 2 tasks × 4 roots/task × 4 arms + 8 clean repeats.
10. **Pilot branch count:** 32 main + 8 controls = **40 rollouts**.
11. **PASS/KILL:** defined in section 11.
12. **Full study:** task-equal root-clustered held-out study only after pilot success.
13. **Reusable code:** ordinary baseline runner, CPU/DCU policy seam, processors, explicit-noise API,
    evidence/statistics conventions.
14. **Bypass:** all exact-state restore/null/F3b machinery.
15. **Largest rejection risk:** result collapses to ordinary camera-robustness curves.
16. **1–2 month feasibility:** plausible if the two narrow implementation gates close quickly and
    the pilot produces nontrivial structure; otherwise stop early.
17. **Recommendation:** **CONDITIONAL GO**.
