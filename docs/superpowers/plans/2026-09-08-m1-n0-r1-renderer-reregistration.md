# M1-N0-R1 renderer re-registration implementation plan

> Execute this plan with subagent-driven development.  Each implementation
> task is test-first, reviewed against the design before moving to the next,
> and committed separately.  The authoritative empirical steps are strictly
> sequential and stop immediately on a failed admission gate.

**Goal:** Re-register the current policy-free LIBERO EGL renderer without
altering the preserved M1-N0 failure bundle, admit it through one construction
and single-render preflight, and only then prepare and execute one new frozen
20-pair null calibration.

**Design authority:**
`docs/superpowers/specs/2026-09-08-m1-n0-r1-renderer-reregistration-design.md`

**Hard stop:** Do not modify `ReplayState`, run source-vs-restored replay, or
begin CC/CS/SC/SS or M2.

## Task 1: Freeze the R1 configuration and immutable predecessor contract

**Files**

- Create: `tests/test_m1_renderer_admission.py`
- Create: `scripts/m1_renderer_admission.py`
- Create: `runtime/m1/m1_n0_r1_predecessor_manifest.json`
- Create: `configs/m1/renderer_preflight_r1_egl0.yaml`
- Modify: `tests/test_m1_null_calibration.py`
- Modify: `scripts/m1_null_calibration.py`

### 1. Write failing contract tests

Add tests proving:

- the admission config uses canonical JSON self-hashing with
  `config_sha256` omitted;
- the predecessor manifest covers every regular file below the old output
  root and binds raw file size/SHA plus the eight frozen semantic hashes;
- old config, state-replay config, run-spec, pair registries, action bytes,
  40 attempts, logs, and terminal manifest all verify unchanged;
- admission paths and null-calibration paths must be disjoint;
- the three namespace fields have exact values and
  `not_applicable_no_policy` can never reach Torch;
- the base ordinal-8 state-replay contract is validated first and remains
  unchanged, while a dedicated R1 copy receives only the renderer overlay;
- the merged runtime SHA is stable and differs from the legacy runtime;
- legacy `configs/m1/null_calibration.yaml` still validates exactly as before.

Run the narrow tests and confirm that the new expectations fail for missing
R1 implementation, not for unrelated fixtures:

```bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest \
  tests/test_m1_renderer_admission.py \
  tests/test_m1_null_calibration.py -q
```

### 2. Implement the minimal contracts

Implement in `scripts/m1_renderer_admission.py`:

- canonical hashing and strict no-overwrite JSON publication;
- complete regular-file predecessor tree manifest generation/verification;
- admission config loading and exact schema validation;
- path-containment and artifact-allowlist checks;
- exact governed environment transition with the frozen unset/block/set/import
  order;
- two explicit hashes:
  `admission_full_renderer_fingerprint_sha256` and
  `worker_preimport_namespace_fingerprint_sha256`.

Implement in `scripts/m1_null_calibration.py`:

- `_r1_runtime_config()` that calls the legacy state-replay validator on the
  unchanged base bytes first, copies the validated mapping, applies only the
  admitted renderer/environment overlay, and validates it with an R1-only
  contract;
- no mutation of `m1_state_replay.FROZEN_RUNTIME_ENVIRONMENT` and no call to
  its ordinal-8 environment application on the R1 path;
- stable canonical merged-runtime hashing.

Materialize the predecessor manifest and admission config only after their
tests pass.  The admission config must contain no action tape, pair IDs,
regime windows, envelopes, or null thresholds.

### 3. Verify and commit

Run the narrow suite, `py_compile` for both scripts, JSON/YAML parse checks,
old-bundle hash verification, and `git diff --check`.  Commit only Task 1
files.

## Task 2: Implement context-safe EGL admission probes

**Files**

- Modify: `tests/test_m1_renderer_admission.py`
- Modify: `scripts/m1_renderer_admission.py`

### 1. Write failing probe tests

Use fake EGL and subprocess seams to prove:

- a checked two-call `eglQueryDevicesEXT` obtains the complete count and
  ordered device list without a fixed maximum;
- both query return values and EGL errors gate admission;
- only in-range ordinals are initialized;
- device descriptors query extension/DRM fields only when supported and never
  persist pointer values;
- full and pre-import fingerprints have disjoint schemas: GL strings and
  post-context loaded libraries cannot appear in the worker schema;
- three fresh probe children are mandatory, have distinct PIDs, and produce
  exact canonical fingerprints after only documented volatile exclusions;
- ambiguity, count/order drift, ordinal drift, GL identity drift, timeout,
  cleanup failure, policy-bearing imports, or any retry produces `BLOCKED`;
- probe output is confined to the admission root allowlist and marked
  `infrastructure_admission_only` / not null evidence.

### 2. Implement probe parent and child modes

Add explicit CLI modes that cannot dispatch null preparation:

- `renderer-probe-worker`: apply the governed environment before renderer
  imports, enumerate devices, initialize/destroy one bounded context per
  in-range ordinal, collect stable EGL/GL/library metadata, publish one
  no-overwrite child record, and exit;
- `renderer-admit`: launch exactly three fresh probe workers once each, verify
  PID independence and exact fingerprints, resolve exactly one frozen renderer
  identity at ordinal 0, and retain probe diagnostics for the later preflight
  manifest.

Do not construct LIBERO, import policy/processor modules, or read any action
tape in these modes.  Do not execute an invalid-ordinal negative-control child;
out-of-range rejection is established from enumeration and pinned selector
source, avoiding unsafe helper indexing.

### 3. Verify and commit

Run fake-only probe tests and `py_compile`.  Inspect the import graph to ensure
the new module has no top-level EGL/MuJoCo/policy imports.  Commit Task 2.

## Task 3: Implement the single-render LIBERO preflight

**Files**

- Modify: `tests/test_m1_renderer_admission.py`
- Modify: `scripts/m1_renderer_admission.py`
- Modify: `docs/m1_null_calibration.md`

### 1. Write failing preflight tests

Build strict fakes mirroring the lazy LeRobot wrapper and assert:

- exactly one official policy-free factory construction;
- exactly one public `render()` dispatch;
- exactly one permitted inner reset;
- zero public reset, inner step, action creation/transmission, settle,
  RuntimeAdapter, null measurement, capture/restore, success/predicate/reward/
  done inspection, policy/processor calls, retry, and process reuse;
- transparent probes are installed before lazy construction, delegate calls
  unchanged, restore originals, and record cleanup;
- only render-returned/type/shape/dtype metadata survives; frame bytes, hash,
  pixel statistics, observation content, and physics state never enter output;
- init-state 0 and seed 2027 are marked provenance-only/not applied;
- the child environment closes in `finally`, exits, and cannot be reused;
- failure creates terminal `BLOCKED` and no authoritative config/root/schedule.

### 2. Implement preflight parent and child modes

After the three probes pass, launch one new `renderer-preflight-worker` child.
It must:

1. reapply and verify the exact governed environment;
2. build one lazy environment with
   `dcu_preflight.build_cpu_environment_runtime()` only;
3. unwrap the single synchronous `LiberoEnv` without constructing a
   `RuntimeAdapter`;
4. install count-only instrumentation on factory construction, wrapper
   render/reset/step, and inner reset/step before lazy initialization;
5. invoke the wrapper's public `render()` once;
6. retain only permitted return metadata and live GL identity;
7. restore instrumentation, close the environment, and exit.

The `renderer-admit` parent validates the child and writes the terminal
preflight manifest.  It must not call `prepare_run`, create the null output
root, or materialize the R1 null config.

### 3. Verify and commit admission implementation

Run the complete fake-only admission suite, relevant policy-import-negative
tests, `py_compile`, schema/self-hash checks, and `git diff --check`.  Update
the documentation with the exact admission command and artifact allowlist.
Commit Task 3 as an intermediate implementation checkpoint; it is not yet the
frozen admission source identity.

## Task 4: Gate and materialize the new authoritative null schedule

**Files**

- Modify: `tests/test_m1_null_calibration.py`
- Modify: `scripts/m1_null_calibration.py`
- Modify: `scripts/m1_renderer_admission.py`
- Modify: `docs/m1_null_calibration.md`
- Create after empirical preflight PASS only:
  `configs/m1/null_calibration_r1_egl0.yaml`
- Create after empirical preflight PASS only:
  `runs/m1_null_calibration/20260908_task000_init000_null20_egl0/run_spec.json`
- Create after empirical preflight PASS only:
  `runs/m1_null_calibration/20260908_task000_init000_null20_egl0/pair_registry.json`

### 1. Write failing post-PASS and schedule tests

Prove:

- materialization rejects a missing, malformed, hash-invalid, non-PASS, wrong
  evidence-role, null-evidence-marked, wrong-PID, or wrong-renderer manifest;
- materialization is the only path that creates the R1 config and is atomic,
  self-hashed, and no-overwrite;
- prepare rejects an absent/drifted preflight binding before creating its
  output root;
- R1 pair IDs are exactly `m1n0-r1-pair-000` through `-019`, with 40 unique
  A/B attempt IDs; legacy pair IDs remain valid only for the old config;
- run-spec records base state SHA, admission config/manifest raw+self hashes,
  merged-runtime SHA, `MUJOCO_EGL_DEVICE_ID=0`, all three namespace fields,
  no retry/replacement, and the unchanged source/action/terminal contracts;
- every worker recomputes only the context-free pre-import fingerprint before
  renderer import, never creates a preliminary context, and fails closed on
  drift without fallback;
- all jobs/results record one execution commit `A` and verify committed
  scientific-scope blobs while preserving unrelated allowlisted user changes;
- preflight PID/environment/frame cannot appear in job or result payloads;
- `run` cannot launch before the new schedule is committed and verified.

### 2. Implement materializer and R1 schedule support

Add a `materialize-r1-config` mode that consumes only the terminal-PASS
preflight manifest and creates the new config.  Generalize pair validation to
derive the exact expected IDs from the frozen prefix without weakening the
legacy contract.  On R1 construction, apply the validated renderer overlay,
call the policy-free factory, and pass its runtime mapping to
`RuntimeAdapter.construct_fresh()` through the supplied builder seam so the
legacy ordinal-8 environment application is never invoked.

Before each authoritative environment construction, recompute and compare the
context-free admitted namespace fingerprint.  After construction, retain the
existing exact live-GL identity gate.  No thresholds or measurement semantics
change.

### 3. Verify and commit code, not empirical outputs

Run the focused M1-N0/R1 tests, all related M1 tests, `py_compile`, import
negative scans, JSON/YAML/schema/hash audits, and `git diff --check`.  Obtain
independent specification and code-quality reviews.  Commit Task 4 code/docs.
This final reviewed implementation commit is the one and only admission
source identity `P`; Task 3's intermediate commit is not `P`.

## Task 5: Execute renderer admission exactly once

### 1. Pre-execution integrity audit

Verify:

- old bundle tree and semantic hashes still match Task 1;
- admission config/self hash and code paths match committed source identity
  `P`, and `HEAD == P` before the single-shot admission command;
- the admission output root does not exist;
- the authoritative R1 config and null output root do not exist;
- the exact CPU interpreter is available;
- no policy/processor module is imported.

### 2. Run the bounded admission command

Run exactly one parent admission command.  It performs three EGL-only fresh
probes followed by one fresh LIBERO construction/render preflight.  There is
no retry or replacement.

### 3. Evaluate the terminal manifest

If any probe or preflight gate fails:

- retain and commit the immutable admission diagnostics;
- publish `Renderer Preflight = BLOCKED` and `M1-N0 = BLOCKED`;
- assert that no authoritative config, output root, run-spec, pair registry,
  job, or attempt exists;
- stop the plan.

If and only if the manifest is terminal `PASS`, verify its self/raw hashes,
commit the admission evidence under a distinct evidence commit identity `E`,
and proceed.  `E` never replaces or redefines the executed source identity
`P` recorded by the manifest.

## Task 6: Freeze and execute the single R1 authoritative calibration

### 1. Materialize and freeze after PASS

Use the post-PASS materializer to create
`configs/m1/null_calibration_r1_egl0.yaml`.  Verify its self hash and exact
preflight bindings.  Run `prepare` once to create the new output root,
run-spec, pair registry, 20 pair IDs, and 40 attempt IDs without environment
construction.

Verify no old path changed, then commit the new config and frozen schedule.
Record this committed scientific identity as `A`; require all authoritative
workers to verify the same committed blobs.

### 2. Execute once

Launch the frozen schedule exactly once.  Preserve all 40 scheduled outcomes;
never retry or replace a failed worker.  Do not change renderer ordinal,
thresholds, source tape, regime windows, or terminal semantics after launch.

### 3. Publish and stop

Run the terminal artifact/hash/schema audit and independent result review.
Publish:

- `M1-N0 infrastructure/protocol = READY` or its precise blocker;
- `Renderer Preflight = PASS/BLOCKED`;
- `M1-N0 empirical calibration = PASS/BLOCKED`;
- grouped physics and renderer envelopes only if 20 valid pairs exist;
- every nonzero renderer discrepancy explicitly.

Commit terminal evidence.  Stop without starting authoritative M1 replay or
any later experimental condition.
