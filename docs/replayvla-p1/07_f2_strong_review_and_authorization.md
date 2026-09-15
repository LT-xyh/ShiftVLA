# F2 strong review and bounded renderer authorization

**Status: APPROVED FOR F2 ONLY**

Authority branch: `xyh/replayvla-p1`.
Reviewed predecessor: `57666bf5e713602e75229be42da24400aaa583ff`.

This document authorizes the bounded M1-SA-v1 F2 renderer qualification defined below. It does not authorize F3, null calibration, replay, policy/processor execution, scientific env steps, or any legacy G1/E3-E6/G2 work.

## 1. F1T adjudication

Repository evidence at `57666bf5e713602e75229be42da24400aaa583ff` is accepted for the following narrow claim:

- `F1T = PASS`.
- `SA-runtime = tensor-transport-qualified` for the traced production transport path.
- The production CPU/DCU boundary is CPU Torch tensors -> safetensors -> DCU CPU Torch tensors -> K100 Torch tensors -> safetensors -> CPU Torch tensors.
- The known DCU NumPy ABI warning remains recorded and unresolved as a general NumPy/Torch bridge issue; it is not promoted to PASS.
- Static production-path tracing plus two fresh-process transport repetitions support that the failed DCU-side NumPy -> Torch bridge is not a required edge of the qualified policy transport path.

No runtime/package change was made. The rejected/unavailable NumPy 1.26.4 candidate remains historical evidence and is not reopened.

D3 is therefore recorded as `TRIGGERED / RESOLVED BY BOUNDED ROUTE PIVOT`: the generic DCU NumPy bridge requirement was removed because it is not a production edge, not because it passed. No additional runtime candidate is authorized.

## 2. Strong-review findings for F2

The M1-SA-v1 contract already makes third-party renderer/native internals trusted dependencies while retaining project-owned entrypoint, phase, failure-publication, no-overwrite, identity and scientific-boundary obligations.

The existing `scripts/m1_renderer_admission.py` may be reused only for pure validation/provenance/no-overwrite helpers that remain semantically applicable. It is not itself authority for the new route.

The legacy config `configs/m1/renderer_preflight_r1_egl0.yaml` MUST NOT be executed as the F2 config. It is historical R1 evidence and binds old workspace paths, old source hashes and a selected EGL ordinal. It may be read as an implementation reference only.

F2 must create a new versioned config/entrypoint under the ReplayVLA-P1 namespace and bind actual current bytes and paths.

## 3. F2 execution boundary

F2 may use only the CPU simulator runtime. The DCU policy runtime is not needed and policy/checkpoint/processor execution remains forbidden.

F2 is infrastructure admission only. The only dynamic environment operations permitted per qualification worker are:

1. official policy-free LIBERO environment construction;
2. the unavoidable construction-time initialization/reset/init-state/settle behavior of that official path, recorded honestly;
3. exactly one explicit/public render requested by the F2 worker after construction completes;
4. close/destroy;
5. process termination observed by the parent.

No research `env.step` is permitted. No action is created or sent. No reward/success/predicate study is permitted. No replay capture/restore and no null schedule is permitted.

## 4. Deterministic renderer discovery and freezing

F2 MUST NOT inherit legacy EGL ordinal `8` or `0` as an accepted value, and MUST NOT loop over ordinals until a worker succeeds.

Before any LIBERO environment construction, run exactly one dedicated renderer-discovery process that:

- uses the F2 CPU interpreter;
- performs read-only EGL device enumeration/identity inspection sufficient to define the new cohort;
- records the ordered device enumeration, relevant device extensions/identity, environment variables, interpreter/module origins and stdout/stderr;
- does not construct LIBERO/robosuite environment objects;
- does not run policy/processor/checkpoint code;
- publishes one terminal discovery record with no-overwrite semantics.

The implementation must define a deterministic selection rule before applying it to the observed enumeration. Preferred rule: select the unique enumerated device satisfying the declared software-renderer criterion used by the official CPU path. If the rule yields zero or more than one candidate, F2 is `BLOCKED`; do not try candidates sequentially.

The selected ordinal and discovery identity are then frozen into one new effective F2 config before qualification workers start. All three workers use exactly that same frozen ordinal/config.

The discovery result is cohort definition, not renderer PASS.

## 5. New F2 config and identity bindings

Create a new ReplayVLA-P1 F2 config rather than editing a legacy M1 config. It must bind at least:

- contract `M1-SA-v1`, phase `F2`, namespace `SA-renderer`;
- execution commit / parent authority;
- CPU Python executable and version;
- actual NumPy/Torch/MuJoCo/robosuite/LIBERO/PyOpenGL module origins relevant to the environment path;
- actual current runtime-lock bytes/hash or an explicit new F2 identity manifest if the old lock does not describe the current interpreter exactly;
- actual source/install file hashes for project-owned entrypoint and the fixed third-party files used for source-accounted statements;
- task suite/task id/init-state id/seed used only for renderer qualification identity;
- actual init-state asset path and byte hash consumed by the official factory;
- renderer backend, frozen selected EGL ordinal and discovery record hash;
- governed environment variables and effective config hash;
- a fresh output root under `runs/replayvla-p1/`;
- explicit `TRUSTED_DEPENDENCY` / `SOURCE_ACCOUNTED` / `OBSERVED` classifications where applicable.

A hash identifies bytes; source-to-binary build provenance that was not independently established remains `NOT_INDEPENDENTLY_ATTESTED`.

Do not modify old runtime locks merely to make them match. If a new identity record is necessary, create a new ReplayVLA-P1-specific record and preserve the old lock unchanged.

## 6. Entrypoint and fail-closed behavior

Implement a thin F2 entrypoint, preferably `scripts/m1_sa_renderer.py` or the F2 subcommand of the planned `scripts/m1_sa_workflow.py`.

Before dynamic execution it must mechanically verify:

- F1T PASS summary and its bound authority/production trace exist and match expected bytes;
- current execution commit/config identity is recorded;
- output paths do not already exist;
- legacy configs/evidence are not output targets;
- only F2 operations are reachable;
- the frozen discovery/effective-config record is present before worker construction;
- failure, timeout or exception produces a terminal FAIL/BLOCKED record rather than disappearance;
- parent owns and terminalizes only its own worker processes/process groups.

Do not add a general-purpose operation/native guard. This route relies on the M1-SA trusted-dependency boundary.

## 7. Qualification schedule

After discovery and config freeze, launch exactly three predetermined fresh qualification workers. No worker reuse and no replacement attempts.

Each worker must record at least:

- unique worker/attempt id;
- argv and governed environment;
- interpreter and relevant module/install identity;
- effective-config hash and selected ordinal;
- official construction start/end status;
- construction-time reset/init-state/settle behavior to the extent directly observable or source-accounted; do not record unknown internals as zero;
- one explicit/public render call count = 1;
- render return metadata sufficient to establish a valid observation image (shape, dtype, finite/range metadata as appropriate) without committing raw frame bytes;
- observed EGL/GL identity after context creation;
- close/destroy outcome;
- return code, timeout status, cleanup result and parent-observed process exit;
- stdout/stderr paths and hashes.

All three planned attempts are retained whether they pass or fail. Do not rerun a failed worker under a different ordinal/config and substitute the new result.

## 8. F2 PASS criteria

`SA-renderer = PASS` only if all of the following hold:

1. discovery produced exactly one deterministic selected renderer device under the frozen rule;
2. one immutable effective F2 config was used by all three workers;
3. all three workers are fresh processes and each completes the official policy-free construction;
4. each worker makes exactly one explicit/public render and returns a valid image metadata result;
5. each worker closes/destroys successfully and the parent observes normal termination;
6. the observed EGL/GL identity and selected ordinal are identical across all three workers;
7. no forbidden F2 operation occurred;
8. no attempt was replaced, retried under a different renderer choice, or silently dropped;
9. identity/evidence records are hash-bound and published with no-overwrite semantics;
10. known unknowns/native internals remain classified as trusted/unobserved rather than `observed_zero`.

Any required-condition failure yields `F2 = FAIL` or `BLOCKED` with the exact failing predicate. Do not change renderer selection, dependency versions, task semantics, oracle or contract inside the same cohort to obtain PASS.

## 9. Tests before real F2 execution

Focused synthetic/fake tests must pass before the first real renderer-discovery or environment worker. Cover at least:

- F1T predecessor missing/mismatched -> reject;
- legacy config cannot be selected as F2 output config;
- deterministic discovery rule: zero/multiple candidates -> BLOCKED;
- effective config cannot change after freeze;
- output/no-overwrite and duplicate worker ids -> reject;
- exactly three worker slots, no replacement slot;
- public render count must equal exactly one;
- env step/action/policy/processor/replay/null command paths -> reject;
- worker failure/timeout -> terminal record;
- inconsistent EGL/GL identity across workers -> F2 not PASS;
- unknown native/source-accounted facts cannot be serialized as observed zero;
- PASS adjudication requires all three terminal PASS records plus discovery/effective-config evidence.

Synthetic/fake tests do not count as renderer evidence.

## 10. Stop and escalation rules

F2 is authorized under this document. F3 remains `NOT AUTHORIZED` even after a local F2 PASS until the resulting commit/evidence is pushed and reviewed.

Stop and return for review if any of these occurs:

- deterministic discovery cannot select exactly one device;
- official environment construction requires policy/checkpoint/processor execution;
- actual init-state byte-use cannot be bound to the recorded asset;
- renderer identity differs among the three workers;
- a worker requires a different ordinal/config to pass;
- package/runtime changes appear necessary;
- environment/task/action semantics must change;
- project-owned phase/no-overwrite/failure-publication contract cannot be enforced;
- any forbidden operation is reached.

Do not continue to F3, create a null schedule, run env steps, capture/restore replay state, or load policy/checkpoint under this F2 authorization.

## 11. State after this authorization

- F0 = PASS.
- F1 generic baseline NumPy bridge issue = historical unresolved/blocked evidence.
- F1T = PASS.
- SA-runtime = `tensor-transport-qualified` for the traced production edge.
- D3 = `TRIGGERED / RESOLVED BY BOUNDED ROUTE PIVOT`.
- F2 = `AUTHORIZED` under this document, not yet PASS.
- F3-F6 = `NOT AUTHORIZED`.
- old G1 remains `BLOCKED`; old operation-guard E3-E6/G2 remain paused/not started.
