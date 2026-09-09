# M1-N0 fresh-process null calibration

M1-N0 measures the reproducibility floor of the pinned policy-free
LIBERO/robosuite/MuJoCo runtime.  It is not an authoritative replay test and
does not load SmolVLA, a checkpoint, processors, or policy actions.

## Frozen inputs

The checked-in configuration is
`configs/m1/null_calibration.yaml`.  It fixes task 0, init state 0, seed
2027, the existing hard-gate registry, and its persisted `(82, 7)` float32
action tape.  The registry and action-byte SHA-256 values are checked before
preparation and again before workers are launched.  The output root is
`runs/m1_null_calibration/20260901_task000_init000_null20`.
The config freezes `worker_timeout_seconds: 600` for every child.  A timeout
publishes partial stdout/stderr and a failed attempt record; it never retries
or replaces that scheduled attempt.

## Protocol

Run preparation first.  Preparation is source-only: it verifies the frozen
registry/tape, writes a canonical self-hashed `run_spec.json`, and atomically
preregisters exactly `m1n0-pair-000` through `m1n0-pair-019`, each with one A
and one B attempt.  It never constructs an environment.

```bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python \
  scripts/m1_null_calibration.py prepare \
  --config configs/m1/null_calibration.yaml
```

Run consumes only that immutable schedule.  A and B are fresh Python
processes, each constructs one official policy-free environment, performs its
single construction reset, executes the tape once from step 1, collects every
frozen regime window, and stops on the returned terminal at step 82.  There is
no replay restore, capture, mid-trajectory reset, settling, dummy action,
autoreset, policy call, action generation, retry, or post-terminal step.

```bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python \
  scripts/m1_null_calibration.py run \
  --config configs/m1/null_calibration.yaml
```

Workers are also available as an explicit mode for the parent process:

```bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python \
  scripts/m1_null_calibration.py worker --job <persisted-job.json>
```

Every scheduled attempt is single-shot.  A failure is persisted and the
remaining frozen schedule continues; failed or semantically divergent pairs
never count toward the 20 valid pairs.

## Measurements and gates

Attempt artifacts retain process identity, frozen inputs, model/runtime
fingerprints, full invariant snapshots, official observation-tree images,
direct renderer RGB, terminal semantics, and protocol counters.  They also
publish RuntimeAdapter construction provenance separately from the live
post-construction forbidden-operation counters: the one seeded reset and any
construction-time `set_init_state`/settle calls are allowed, while reset,
set-init, settle, restore, capture, policy/processor, retry, dummy,
autoreset, and post-terminal calls after construction must all be observed as
zero.  Pair validation checks this provenance on both sides in addition to
the raw terminal reason/source evidence and semantic terminal reason.  A
missing raw wrapper reason is retained as explicit `null` evidence and is
accepted only when both sides match and the independent predicate proof is
valid.  Pair validation keeps predicates, success, done/termination timing,
discrete gripper state, regime coordinates, and contact geom identity as
exact gates.
Contact, grasp, and carried evidence is derived from snapshots and frozen
controls rather than regime labels.

Contact topology is canonicalized by unordered geometry identity while
retaining duplicate rows.  Contact-distance quantities are canonicalized in
the same order and are never compared when the A/B contact identity
multisets differ.

For the lazy official LeRobot wrapper, construction provenance also contains
the pinned source path/SHA, exact inner-reset/init/settle/dummy counts, the
post-reset inner-environment state, and complete post-construction probe
installation evidence. Strict validation rejects missing or merely
nonnegative counter values; an unobserved construction call is not published
as zero.

Floating physics variability is grouped by regime × quantity × horizon using
the existing hard-gate envelope builder.  This includes every non-RGB numeric
leaf in the official observation tree; the observation keys, dtypes, and
shapes remain exact structural gates.  RGB leaves are renderer-only.  Renderer
output is grouped by camera × observation key × regime × horizon.
Bitwise-identical duplicate controls publish exact-only RGB groups; nonzero
RGB differences publish their differing-pixel, maximum, and mean metrics
without changing physics gates.

The contact and grasp requirements are continuation-aware: a capture window
may first satisfy its semantic predicate at horizon 1 (the frozen contact
window begins at step 43 and first contact is at step 44), while every
horizon's diagnostics remains persisted.  Panda gripper commands use the
official `-1 = open`, `+1 = closed` convention.  A predicate-transition
terminal is accepted only with an independent successful false-to-true
predicate proof; any generic or missing raw wrapper reason is retained as raw
evidence rather than treated as semantic proof.

Before constructing an environment, each worker verifies the config contract,
source-registry canonical SHA, run-spec and pair-registry self-hashes, exact
scheduled attempt binding, and action-tape path/SHA binding.  Full invariant
and official-observation arrays are published once as atomic, no-pickle,
content-addressed `.npy` sidecars.  JSON registries and measurements retain
references and hashes so the evidence can be independently reconstructed.
The parent validates each A/B pair immediately and retains only compact scalar
physics/RGB measurements plus artifact references in `pair_measurements.json`;
full trajectories remain in the immutable attempt artifacts and are not
accumulated in the parent schedule.

The immutable output bundle contains the run specification, pair registry,
one artifact per attempt, pair measurements, grouped physics and renderer
envelopes, and `terminal_manifest.json`.  A terminal status is `PASS` only
when all 20 pairs are complete, independent, semantically equal, contact-rich
coverage is present, all groups are covered, and every artifact hash verifies;
otherwise it is `BLOCKED` with the causal failure retained.

## R1 backend checkpoint review (2026-09-09)

The main agent reviewed the current EGL-backend code and tests against base
commit `5f3b7409938e701fdef3fc3138bfa7930bafb97c`. This is a backend
implementation checkpoint, **not** a completed LIBERO preflight implementation
or an empirical admission result.

> independent quality-review subagents repeatedly failed to return; no review conclusion was fabricated or assumed.

The bounded backend specification review returned ACCEPT. The main-agent
diff review accepts the backend slice: raw enumeration uses the EGL integer
ABI, failed non-null handles retain cleanup ownership, and each applicable
cleanup is attempted once. Independent quality review remains unavailable,
not PASS; its unavailability alone is not an experimental blocker.

The requested preflight boundary audit found:

| Boundary | Current evidence |
| --- | --- |
| Official construction plus one public `render()` | Not implemented in the admission module yet. |
| No step, outer reset, settle/dummy action, RuntimeAdapter, policy/processor/checkpoint, null measurement, or experimental frame persistence | No such execution is introduced by the backend diff; this does not prove the absent preflight execution path. |
| Environment close and fresh-process termination | Private EGL cleanup is implemented and fake-tested; LIBERO cleanup and a fresh-child launcher remain unimplemented. |
| Preserved M1-N0 bundle | Complete predecessor tree and semantic hashes reverified unchanged. |
| Renderer versus compute namespace | Ordinal remains an EGL enumeration index; the R1 config separately fixes renderer ordinal 0 and `not_applicable_no_policy`. No K100/DCU affinity mapping is introduced. |

The admission CLI, metadata collectors, three-fresh-child launch path, and
the instrumented single-render LIBERO worker still need implementation before
the one-shot command can run. Existing passing backend/contract tests cannot
stand in for tests of that missing path. The frozen admission root, R1 null
configuration, and new null output root are absent at this checkpoint.

Thus `Renderer Preflight = NOT RUN` and `M1-N0 = BLOCKED` remain unchanged.
No authoritative schedule may be created before a committed implementation
produces terminal preflight PASS. Future review unavailability must be recorded
explicitly for a main-agent decision, not retried indefinitely.

Verification at this checkpoint used the frozen CPU interpreter:
`pytest tests/test_m1_renderer_admission.py tests/test_m1_null_calibration.py -q`
returned **189 passed in 34.79s**; `py_compile` and `git diff --check` passed.
Only the admission script, its tests, and this review note are in scope for
the checkpoint commit. `AGENTS.md`, M0 evidence, and unrelated untracked
artifacts remain excluded. The resulting commit is a backend checkpoint,
not the final admission execution source identity `P`.
