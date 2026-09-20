# M0-native prefix execution feasibility

**Status: CANDIDATE FEASIBILITY PROPOSAL — NOT AUTHORITY.**

Base reviewed:

`3cdda0b2ffd652ef00daa3598f5c6416933f2165`

Current authority:

- `docs/replayvla-p1/16_prefix_reexecution_paper_route.md`
- `docs/replayvla-p1/20_prefix_runtime_route_pivot.md`

Historical runtime authorities/evidence remain immutable. This document does not authorize runtime
execution, microvalidation, pilot execution, or changes to installed packages.

## 1. Decision summary

**Recommendation: CONDITIONAL GO.**

Prefix reexecution still has a thin M0-native route, but only under one hard architectural
condition:

> P1 must enter through the existing M0 phase/bootstrap owner
> `scripts/dcu_preflight.py::run_phase(...)`, so the existing preflight gates and
> `_verify_assets_binding(..., create=True)` execute before any P1 environment construction.

P1 must not continue the direct-builder microvalidation architecture and must not own asset setup,
environment bootstrap, renderer setup, package patching, or worker-process management logic.

This is not a plain **GO** because `m0_baseline_a.run()` by itself does **not** create the asset
binding: its parent live gate calls `run_preflight_gates(..., prepare_assets=False)`. The viable
composition therefore needs two small, explicit integration hooks around the historical ordinary
episode path.

This is not a **PIVOT** because the required setup, environment, worker, processor, rollout and
episode-freshness seams already exist and have historical runtime evidence.

If implementation requires a new asset path repair, new environment factory, global monkeypatch,
new worker manager, copied M0 runner, or another provenance/bootstrap framework, this decision
automatically becomes **PIVOT**.

## 2. Existing M0 call graph

There are two related historical execution layers. They must not be conflated.

### 2.1 Asset-owning M0 native phase lifecycle

This is the correct outer owner for future P1 execution.

```text
scripts/dcu_preflight.py::main
  -> _load_yaml_config
  -> dispatch_phase
  -> run_closed_loop / run_compare / run_concurrency
  -> run_phase
       -> validate_runner_config
       -> run_preflight_gates(
            ...,
            prepare_assets=True     # every normal phase, not one-step-child
          )
            -> runtime/config/artifact/reference gates
            -> _verify_assets_binding(config, create=True)
                 -> scripts.m0_smoke._safe_asset_symlink(...)
                 -> package-local libero/libero/assets binding
       -> phase_runner(...)
```

For the historically accepted ordinary closed-loop phase, the runner path is:

```text
run_phase
  -> _run_closed_loop_impl
       -> build_cpu_runtime(config, include_policy=False)
            -> build_cpu_environment_runtime
            -> LeRobot make_env_config / make_env
            -> official env + policy/env processors
       -> _start_worker
            -> existing DCUWorkerClient / subprocess transport
            -> installed SmolVLAPolicy
       -> FeatureOnlyRemotePolicy
       -> m0_smoke.PolicyProxy
       -> m0_smoke.ProcessorProxy(env_preprocessor)
       -> m0_smoke.ProcessorProxy(policy preprocessor)
       -> m0_smoke.ProcessorProxy(policy postprocessor)
       -> m0_smoke.ProcessorProxy(env postprocessor)
       -> m0_smoke.VectorEnvProxy
       -> lerobot.scripts.lerobot_eval.rollout
            -> policy.reset()
            -> env.reset(seed=[2027], NEW_ROLLOUT_OPTION)
            -> preprocess_observation
            -> env_preprocessor
            -> policy preprocessor
            -> policy.select_action
            -> policy postprocessor
            -> env postprocessor
            -> env.step
            -> terminal / horizon
       -> result + worker + trace evidence
       -> _close_worker
       -> close_envs
```

Historical evidence in `docs/dcu_preflight.md` records an accepted closed-loop execution on this
route with one reset, 82 policy calls, 82 environment steps, 83 renders, official
`n_action_steps=1` queue semantics, success observed, and no processor/action-boundary bypass.

### 2.2 M0 baseline parent/child lifecycle

`scripts/m0_baseline_a.py` is the reusable multi-episode execution layer, but it is not the
asset-binding creator.

Parent:

```text
m0_baseline_a.py::main
  -> run
       -> load_config / validate_config
       -> build_episode_matrix
       -> validate_preflight_contract
            -> run_preflight_gates(..., prepare_assets=False)
            -> verify an already-established binding
       -> launch_children
            -> build_child_command
            -> build_cpu_child_environment
            -> subprocess.Popen(two CPU child processes)
       -> collect_children_fail_closed
       -> aggregate / immutable evidence
```

Child:

```text
m0_baseline_a.py --child
  -> _run_child
       -> load / validate config
       -> validate_preflight_contract(..., run_gates=False)
       -> dcu_preflight._start_worker(...)      # once per child
       -> for each assigned planned episode:
            _run_episode_official
              -> _run_episode_official_impl
                   -> fresh build_cpu_runtime(..., include_policy=False)
                   -> select_init_state_before_reset
                   -> fresh per-episode IPC
                   -> FeatureOnlyRemotePolicy(existing worker_client)
                   -> PolicyProxy + four ProcessorProxy instances
                   -> VectorEnvProxy
                   -> official lerobot_eval.rollout
                   -> make_episode_record
              -> close_runtime_environment      # every episode
              -> cleanup per-episode IPC
       -> _close_worker_bounded                 # after serial episode list
       -> child evidence
```

The historical baseline run completed 15/15 planned episodes with no runtime failures. Its child
evidence explicitly records `prepare_assets=false`: this proves the baseline execution path can
consume an established M0 runtime, but does not prove that `m0_baseline_a.run()` itself creates
the asset binding.

## 3. Existing runtime setup ownership

### Asset binding owner

The source-accounted creator is the existing M0 setup layer:

```text
dcu_preflight.run_phase
 -> run_preflight_gates(prepare_assets=True)
 -> _verify_assets_binding(create=True)
 -> m0_smoke._safe_asset_symlink
```

This happens before the phase runner and therefore before
`build_cpu_runtime/make_env`.

The historical M0 smoke manifest independently records the resulting package-local binding:

```text
<cpu-venv>/site-packages/libero/libero/assets
  -> frozen lerobot/libero-assets revision
```

The accepted `dcu_preflight` closed-loop run demonstrates that the same phase-level setup can be
followed by a complete ordinary policy/environment rollout.

### Baseline parent

`m0_baseline_a.validate_preflight_contract` deliberately passes
`prepare_assets=False`. It is a verifier/consumer, not the binding creator.

Therefore future P1 must not simply invoke `m0_baseline_a.run()` from an arbitrary clean runtime
and assume the package-local asset binding already exists.

### Environment construction owner

Environment construction remains the existing
`dcu_preflight.build_cpu_runtime -> build_cpu_environment_runtime -> LeRobot make_env` path.

P1 must not add another environment factory.

### Worker owner

Worker process creation, reset, select-action transport and shutdown remain:

- `dcu_preflight._start_worker`;
- `DCUWorkerClient` / existing subprocess transport;
- `FeatureOnlyRemotePolicy`;
- `dcu_preflight._close_worker` or baseline bounded wrapper.

P1 must call these existing seams rather than implement another worker manager.

## 4. Camera insertion point

A source-accounted thin insertion point exists after ordinary environment construction and before
the official rollout reset.

For one M0 episode:

```text
fresh build_cpu_runtime(...)
 -> SyncVectorEnv, num_envs == 1
 -> select_init_state_before_reset(...)
 -> locate env.envs[0]
 -> require exact LeRobot LiberoEnv lifecycle
 -> replace only that sub-env reference with:
      ObservationIndexedCameraWrapper(
          original LiberoEnv,
          arm=...,
          controller=AgentviewYawController(...)
      )
 -> official lerobot_eval.rollout(...)
      -> wrapper.reset(...)
      -> wrapper.step(...)
```

Why this is thin:

- pinned Gymnasium 1.3.0 `SyncVectorEnv` stores its live sub-environments in public
  `self.envs`;
- reset, step, call/get-attr and close operate through that list;
- `ObservationIndexedCameraWrapper.__getattr__` delegates the existing LiberoEnv interface;
- no vector environment is rebuilt;
- no second reset is introduced;
- no installed class is monkeypatched;
- no package source is changed;
- no asset/bootstrap path is touched.

The safest order is to perform the existing M0
`select_init_state_before_reset(env, init_state_id)` first, then substitute
`env.envs[0]` immediately before `rollout`. The official rollout still owns the single episode
reset.

The existing P1 camera helper remains reusable scientific helper logic. The direct-builder
microvalidation harness is not the future runtime owner.

## 5. Explicit-noise insertion point

The existing M0 processor order must remain unchanged:

```text
raw M0 observation
 -> LeRobot preprocess_observation
 -> existing env_preprocessor
 -> existing policy preprocessor
 -> PolicyProxy
 -> paired-noise adapter
 -> FeatureOnlyRemotePolicy.select_action(features, noise=noise)
 -> existing DCU transport
 -> official SmolVLAPolicy.select_action(batch, noise=noise)
 -> existing policy postprocessor
 -> existing env postprocessor
 -> VectorEnvProxy
 -> env.step
```

The existing G-P1 transport already supports the needed explicit-noise path. No change to
`dcu_worker.py`, `dcu_model_worker.py`, or the official policy is indicated by this trace.

The minimal episode-level change is:

1. give the existing per-episode IPC boundary a `noise_writer` beside its existing feature
   writer;
2. construct `FeatureOnlyRemotePolicy(..., noise_writer=noise_writer)`;
3. wrap that remote object with the existing
   `PairedNoiseSelectActionPolicy(root_id=...)`;
4. leave the existing outer `m0_smoke.PolicyProxy` and all processors in place.

`PairedNoiseSelectActionPolicy` resets its query index at official `policy.reset()`, derives
noise from root/query identity, and delegates the branch's current observation plus that noise to
the same remote select-action seam.

No manual `predict_action_chunk`, fixed action tape, copied action, or processor bypass is needed.

## 6. Fresh-root semantics

The historical baseline episode lifecycle already provides the important freshness boundaries.

### Environment

`_run_episode_official_impl` calls `build_cpu_runtime` for every episode.
`_run_episode_official` closes that runtime in `finally`.

Therefore every P1 arm can receive a newly constructed CPU environment rather than a reused
simulator.

### Init state and environment seed

Before rollout, M0 explicitly selects and verifies the planned init-state ID.

The official LeRobot rollout then calls:

```text
policy.reset()
env.reset(seed=[planned seed], NEW_ROLLOUT_OPTION)
```

Pinned `LiberoEnv.reset` seeds the underlying LIBERO environment, performs its normal reset,
applies the selected init state, and executes its normal internal settle steps. Those settle steps
remain reset internals and are not P1 policy-query indices.

### Policy / worker state

The model worker may remain persistent across serial episodes. This is not a state-copy operation.

At each official rollout start:

```text
PolicyProxy.reset
 -> paired-noise wrapper reset
 -> FeatureOnlyRemotePolicy.reset(seed=2027)
 -> DCUWorkerClient.reset
 -> official policy reset / empty queue / worker RNG reset
```

With `n_action_steps=1` and explicit per-query flow noise, each arm begins with a cleared action
queue and the same root/query noise namespace.

A new DCU model process per arm is therefore not required.

### Residual freshness uncertainty

The baseline source does not explicitly reseed every process-global Python/NumPy/Torch CPU RNG at
the beginning of each serial episode. The currently relevant environment seed and remote policy
reset are explicit, and P1 action-generation noise is explicit, but this leaves a bounded empirical
risk that an unobserved process-global consumer could differ across serial arms.

Do not build a new bootstrap layer to eliminate this speculative risk. The first qualification and
later clean-duplicate/same-prefix audits must fail closed on unexplained disagreement. If such
disagreement appears and cannot be fixed with one narrow episode-local seed hook, return to
route-level review rather than beginning another runtime repair sequence.

## 7. Branch construction on the M0 lifecycle

A future P1 row can be treated as a planned M0-style episode with additional P1 metadata:

```text
root_id
arm
switch_index
task/init/seed
worker/runtime identity
paired-noise namespace
```

For CC/CS/SC/SS:

- construct a fresh ordinary CPU environment for each arm;
- use the same task/init/seed/horizon;
- use one frozen worker identity, preferably physical device 1 for qualification and the initial
  implementation;
- official rollout resets the worker/policy before each arm;
- paired-noise key excludes arm;
- only the camera condition schedule differs at intervention level;
- subsequent action/state divergence is allowed to emerge naturally.

Do not use:

- state restore;
- image copy;
- action copy;
- fixed action tape;
- queue splice;
- exact-state owner closure.

For the first implementation, keeping all arms for a root serial on one existing worker is simpler
and scientifically cleaner than reusing the historical alternating A/B matrix assignment.

## 8. Minimal required modifications

### A. New P1 M0-native phase runner

Likely new file:

`scripts/p1_m0_native.py`

Responsibilities only:

- enter through `dcu_preflight.run_phase` as a supplied phase runner / M0-owned phase boundary;
- construct the very small P1 qualification schedule, later the frozen pilot schedule if separately
  authorized;
- start one worker through the existing dcu_preflight worker seam;
- call the reusable M0 episode executor for each row;
- add P1 evidence: arm, switch, camera rows, paired-noise key/hash, same-prefix disposition;
- close through existing worker/environment cleanup;
- never create or repair an asset binding itself.

It must not implement environment construction, subprocess transport, renderer selection, package
setup, or universal provenance.

### B. Two narrow default-off episode hooks in `m0_baseline_a.py`

A minimal implementation should add optional/default-`None` seams around
`_run_episode_official_impl` rather than copy the function.

**Environment/sub-env adapter hook**

Placement:

```text
build_cpu_runtime
 -> select_init_state_before_reset
 -> optional P1 sub-env adapter
 -> ordinary rollout
```

Historical callers with no hook remain byte-for-byte semantically unchanged.

**Remote-policy adapter/factory hook**

Placement:

```text
existing feature/noise IPC writers
 -> FeatureOnlyRemotePolicy
 -> optional P1 paired-noise wrapper
 -> existing PolicyProxy
 -> existing processors / rollout
```

Again, the default historical path remains the current no-explicit-noise baseline.

### C. Focused fake/static tests

Likely new:

`tests/test_p1_m0_native.py`

Required tests should prove:

- phase-level asset setup is owned by `run_phase`, not P1;
- P1 code contains no `_safe_asset_symlink` or asset write;
- env adapter is called after runtime construction/init selection and before rollout reset;
- `SyncVectorEnv.num_envs == 1` and `env.envs[0]` identity fail closed;
- paired noise is inserted between policy preprocessing and official select-action, not before
  processors or after action generation;
- default M0 episode path is unchanged when hooks are absent;
- each planned arm creates/closes a separate CPU runtime;
- one persistent worker can be reset across multiple arms;
- no state/image/action copy;
- no retry/replacement;
- same-prefix audit is technical-fail-closed.

## 9. Files likely modified

First implementation should fit in:

1. `scripts/p1_m0_native.py` — new, thin phase runner / evidence layer;
2. `scripts/m0_baseline_a.py` — two small default-off episode hooks;
3. `tests/test_p1_m0_native.py` — fake/static integration tests.

Existing reusable helper file:

- `scripts/p1_prefix_reexecution.py` — camera controller, camera schedule, paired-noise derivation and
  audit logic. Prefer reuse without changing it unless a tiny interface extraction is necessary.

No change is currently indicated for:

- `scripts/dcu_worker.py`;
- `scripts/dcu_model_worker.py`;
- G-P1 transport in `scripts/dcu_preflight.py`.

The existing P1 pilot config can remain the scientific matrix source after runtime qualification;
do not introduce another bootstrap config.

## 10. Components explicitly NOT copied

The M0-native route must not copy or reimplement:

- `dcu_preflight.run_preflight_gates`;
- `_verify_assets_binding`;
- `m0_smoke._safe_asset_symlink`;
- `build_cpu_runtime` / `build_cpu_environment_runtime`;
- LeRobot environment factory;
- renderer/EGL setup;
- `_start_worker` / worker transport / shutdown logic;
- `m0_baseline_a._run_episode_official_impl`;
- M0 processor proxies;
- LeRobot rollout engine;
- M0 evidence/terminal semantics;
- F3N;
- `m1_state_replay`;
- null calibration;
- exact-state owner closure.

The direct-builder
`scripts/p1_prefix_reexecution_microvalidate.py` may remain as historical source/reference, but its
runtime-bootstrap path is abandoned.

## 11. First real qualification design

This document does not authorize it.

If reviewer later authorizes one qualification, the smallest useful **M0-native** check is one
non-scientific CS episode:

```text
outer owner:
  dcu_preflight.run_phase
    -> normal preflight / asset binding / renderer setup

root:
  suite = libero_spatial
  task = 0
  init = 0
  seed = 2027

arm:
  CS
  switch = 50

worker:
  existing physical device 1 owner/seam

execution:
  fresh M0 CPU runtime
  -> select init 0
  -> install ObservationIndexedCameraWrapper
  -> official rollout reset
  -> clean obs_0..obs_49
  -> paired explicit noise on each ordinary policy query
  -> camera switches before action_49 step produces obs_50
  -> continue through ordinary terminal/horizon
  -> normal M0 evidence/cleanup
```

Qualification PASS should require, at minimum:

- M0 phase preflight/bootstrap PASS and existing asset binding evidence;
- no P1 asset/bootstrap operation;
- exactly one fresh environment construction/close for the arm;
- ordinary init-state and seed evidence;
- camera evidence reaches obs 50 with the frozen timing contract;
- paired-noise evidence reaches the existing official `select_action`;
- official pre/postprocessor ordering remains intact;
- queue semantics remain `n_action_steps=1`;
- no state/image/action copying;
- no retry;
- no paper estimands.

If the episode terminates before the switch, qualification is BLOCKED and is not replaced.

A later reviewer may choose a two-arm CC/CS qualification to add same-prefix evidence before pilot,
but this is not required to prove the runtime composition itself.

## 12. Engineering cost estimate

### Qualification implementation

Expected production change:

- new thin P1 runner: approximately 150–250 lines;
- M0 default-off hooks: approximately 30–60 lines;
- no new runtime/bootstrap module.

Expected focused tests:

- approximately 200–350 lines in one test file.

Expected files touched: **3**, possibly 4 only if a tiny helper interface in
`p1_prefix_reexecution.py` is needed.

This is consistent with a thin extension.

### Path from qualification to 40-rollout pilot

If the qualification passes without another bootstrap/path defect:

- reuse the already frozen P1 schedule;
- extend the same thin P1 runner from one row to 40 planned rows;
- retain one persistent worker and fresh CPU runtime per row;
- add root/same-prefix evidence aggregation, not another execution stack.

Expected additional production code after qualification: roughly 100–200 lines plus focused tests.
No additional runtime/bootstrap file should be necessary.

For execution-time planning only, the historical baseline recorded approximately 3,384 seconds of
summed worker busy time across 15 episodes, about 226 seconds per episode. A serial 40-rollout
pilot is therefore on the order of 2.5 hours of episode busy time before modest P1 evidence
overhead. This is not a runtime promise.

If multi-worker throughput optimization becomes necessary before scientific feasibility is shown,
defer it; do not complicate the first P1 implementation with a new scheduler.

## 13. Evidence ownership

Reuse existing M0 evidence for:

- task/runtime source identity;
- selected and post-reset init-state evidence;
- worker/model identity;
- processor/action boundary trace;
- queue evidence;
- terminal/success/horizon semantics;
- GL/runtime identity;
- exact environment actions and per-step latency.

P1 adds only:

- root ID and arm;
- switch index;
- requested/actual camera evidence;
- paired-noise key/seed/hash per query;
- same-prefix audit outcome;
- duplicate discrepancy metadata when later authorized.

No new universal evidence framework is justified.

## 14. Engineering risks

### Largest runtime risk

The proposed route composes two historically proven layers that were not previously exercised as
one P1 invocation:

1. `dcu_preflight.run_phase` as the active asset/bootstrap owner;
2. `m0_baseline_a`'s fresh-per-episode execution logic and reusable persistent worker pattern.

Both are source-compatible, but the exact composition is not yet observed. The two failed
direct-builder microvalidations show that LIBERO path/bootstrap assumptions are brittle.

Therefore the first M0-native qualification is a kill gate:

> if another pre-environment asset/path/bootstrap defect appears while P1 is using the unmodified
> normal M0 phase owner, do not begin a new repair sequence; return to route-level review / PIVOT.

A secondary risk is that substituting `SyncVectorEnv.envs[0]` with the project wrapper, although
source-accounted under Gymnasium 1.3.0, has not yet been observed in the historical M0 rollout.

### Freshness risk

The current source explicitly provides fresh env construction, environment seed/init selection and
worker/policy reset. It does not explicitly reseed every process-global CPU RNG at every serial
episode boundary. If clean duplicate or same-prefix evidence later reveals unexplained divergence,
treat it as a technical matching failure. Do not hide it as a scientific history effect.

## 15. Scientific risks

The runtime route can be thin and still fail scientifically.

The largest paper risk remains the authority-16 kill condition:

- camera corruption may clearly change behavior (`G_C/G_S`);
- but prior corruption may leave no persistent post-removal effect (`L ≈ 0`);
- and no history-dependent susceptibility may appear (`I ≈ 0`);
- or any apparent persistence may be comparable to same-condition reexecution discrepancy.

In that case the result reduces to ordinary camera robustness and the route should PIVOT rather than
scale the pilot.

Runtime success must not weaken the same-prefix technical-failure rule or the paper-signal gate.

## 16. Decisive answers

1. **Does normal M0 runtime establish asset binding itself?**
   **Yes for the asset-owning `dcu_preflight.run_phase` / M0 smoke lifecycle; no for
   `m0_baseline_a.run()` in isolation.** Normal dcu_preflight phases call
   `run_preflight_gates(..., prepare_assets=True)` before their runner. Baseline-A only verifies
   the already established binding with `prepare_assets=False`.

2. **Can P1 fully reuse that lifecycle?**
   **Yes, conditionally:** P1 must be a phase-runner/episode extension under
   `dcu_preflight.run_phase`, not a direct-builder program.

3. **Where does the camera wrapper insert?**
   After fresh `build_cpu_runtime` and init-state selection, before official rollout reset, by
   replacing the single `SyncVectorEnv.envs[0]` reference with
   `ObservationIndexedCameraWrapper`.

4. **Where does paired noise insert?**
   At the existing remote policy query boundary, after the existing policy preprocessor and before
   the official remote `SmolVLAPolicy.select_action`, using
   `PairedNoiseSelectActionPolicy -> FeatureOnlyRemotePolicy(..., noise_writer=...)`.

5. **Is environment bootstrap modification required?**
   **NO.**

6. **Is new asset-setup code required?**
   **NO.**

7. **Is a new worker manager required?**
   **NO.**

8. **Must the M0 runner be copied?**
   **NO.** Add narrow hooks and call the existing episode executor.

9. **Can each arm start from a clean fresh root?**
   **Source-accounted YES for fresh env + selected init + environment seed + policy/queue reset.**
   One persistent model worker may be reused because official episode reset clears its queue/RNG
   state and P1 policy noise is explicit. Residual process-global CPU RNG is an empirical matching
   risk, not a reason to create a new runtime.

10. **Can P1 preserve existing processors/policy semantics?**
    **YES.** The camera seam is below the observation producer, and paired noise enters only inside
    the existing remote select-action layer. Official pre/postprocessors and LeRobot rollout remain
    unchanged.

11. **What is the minimum real qualification?**
    One non-scientific CS episode at task0/init0/seed2027 under
    `dcu_preflight.run_phase`, using the existing M0 worker/runtime, camera wrapper, switch 50 and
    paired explicit noise; no estimands and no retry.

12. **How much new code/files from here to the 40-rollout pilot?**
    Approximately 250–450 production lines total plus 250–400 focused-test lines, centered on one
    new runner, two small default-off M0 hooks, and one test file. The pilot should not require a
    new bootstrap/runtime/config stack.

13. **Largest runtime risk?**
    The exact composition of phase-level M0 bootstrap with baseline-style fresh episodes has not
    yet been observed; another inherited path/bootstrap blocker at that boundary is an immediate
    route-level stop condition.

14. **Largest scientific risk?**
    Intervention susceptibility without persistent history-specific aftereffect/hysteresis.

## 17. Recommendation

**CONDITIONAL GO**

Conditions:

1. implementation remains within one small P1 phase runner plus narrow default-off M0 hooks;
2. `dcu_preflight.run_phase` remains the sole asset/bootstrap owner;
3. no P1 code creates or repairs package asset bindings;
4. existing env factory, worker manager, processors, rollout and evidence semantics remain owners of
   their current responsibilities;
5. first real qualification is a bounded non-scientific M0-native episode;
6. any new pre-environment bootstrap/path defect under this composition triggers route-level review
   rather than MV-R2-style infrastructure repair.

Under those conditions, prefix reexecution remains a thin M0 extension rather than a third LIBERO
runtime stack.
