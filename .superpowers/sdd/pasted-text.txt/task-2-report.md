# Task 2 report — M1-N0 fresh-process null calibration

## Status

Implementation complete and locally verified.  The real 40-process
calibration was intentionally not run in this task; empirical M1-N0 remains
an explicit runtime operation.

## Red/green evidence

The initial focused run produced the expected RED state: the new module and
predicate root were absent (`10 failed, 111 passed`).  After the production
patch, a transient worker-window and tuple-expression mismatch was fixed with
small scoped edits.  The final focused run is green:

```text
tests/test_m1_null_calibration.py
tests/test_m1_state_replay.py::test_collected_invariants_expose_all_goal_predicates_as_exact_state
10 passed in 4.55s
```

The complete existing replay-focused suite is also green:

```text
tests/test_m1_state_replay.py
112 passed in 19.29s
```

Additional checks:

```text
py_compile scripts/m1_null_calibration.py scripts/m1_state_replay.py: PASS
null-calibration config self-hash: PASS
git diff --check (scoped tracked edits): PASS
direct --schema-version invocation: PASS via focused test
```

## Implementation

`scripts/m1_null_calibration.py` provides lazy environment/runtime imports,
source-only preparation, canonical self-hashed immutable run and pair
registries, an exact 20-pair/40-attempt schedule, one-shot failure retention,
fresh worker execution, legal terminal handling, exact discrete pair gates,
contact/grasp/carried evidence, grouped physics and renderer envelopes via
the existing hard-gate primitives, safe JSON artifact publication, and
prepare/run/worker CLI modes.

`scripts/m1_state_replay.py` now persists every concrete goal predicate as a
JSON-like exact invariant and classifies the predicate root as exact during
comparison.  The new config freezes the requested task, source registry and
action-byte SHA, output root, runtime policy prohibitions, quantity policy,
and renderer identity.  The documentation records the protocol and gates.

## Changed files

- `scripts/m1_null_calibration.py`
- `scripts/m1_state_replay.py`
- `tests/test_m1_null_calibration.py`
- `tests/test_m1_state_replay.py`
- `configs/m1/null_calibration.yaml`
- `docs/m1_null_calibration.md`

## Self-review, assumptions, and concerns

- The focused tests use fake adapters/process runners only; no live
  environment, policy, checkpoint, or 40-process calibration was used.
- Preparation does not call `_construct_adapter`; source registry/tape/config
  hashes are rechecked before the first worker callback.
- Pair-level floating differences are retained for grouped envelopes while
  predicates, terminal semantics, gripper commands, and contact topology are
  exact gates.
- Runtime-specific fingerprint and observation details are delegated to the
  existing `RuntimeAdapter` when a real worker is used.  The live calibration
  must still verify its pinned runtime and artifact paths on the target
  device.
- Existing user-owned `AGENTS.md`, M0 evidence, prior manifests, and old run
  outputs were not staged or modified.

