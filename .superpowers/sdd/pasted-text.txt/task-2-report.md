# Task 2 report — M1-N0 null-calibration contract revision

## Status

The M1-N0 implementation contract is updated and locally verified at the
fake-artifact/test seam.  The authoritative 40-process calibration remains
intentionally unrun; empirical M1-N0 is still a separate runtime operation.

## Red/green evidence

The review-driven focused baseline was:

```text
6 failed, 23 passed in 15.83s
```

The failures covered the contact-window offset, Panda gripper sign, semantic
terminal proof, observation physics leaves, worker provenance ordering, and
array publication.  After the revision:

```text
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest -q \
  tests/test_m1_null_calibration.py
33 passed in 19.94s
```

The empty strict quantity-selection test now executes its missing-root
assertion instead of silently passing.

## Implementation

- Contact/grasp/carried semantic gates are derived over the complete frozen
  continuation window while horizon-local evidence remains available for
  diagnostics.
- Panda gripper semantics follow the pinned source: `-1 = open`, `+1 =
  closed`, including the two-channel `current_action` representation.
- Strict predicate-transition terminals require an independent successful
  false-to-true predicate proof, while preserving missing/generic raw wrapper
  evidence.
- Workers verify config, source registry, run-spec, pair-registry, schedule,
  and tape bindings before constructing an adapter.
- Non-RGB official observation numeric leaves enter the physics
  regime × quantity × horizon envelope; RGB remains renderer-only and
  observation keys/dtypes/shapes remain exact structural gates.
- Arrays and RGB are published through atomic, no-pickle,
  content-addressed NumPy sidecars.  Final registries and pair measurements
  reference immutable attempt artifacts instead of duplicating raw arrays.

## Scope and verification

All tests use fake adapters/process runners only.  No live environment,
policy/checkpoint, renderer calibration, replay restore, retry, tape
regeneration, or 40-process run was performed.  The frozen source registry,
action tape, runtime pins, user `AGENTS.md`, M0 evidence, and prior run outputs
were preserved.  The next task, after an empirical calibration pass, is the
separately controlled M1 authoritative source-vs-restored replay.
