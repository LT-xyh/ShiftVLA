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
array publication.  After the revision and bounded-memory/transport hardening:

```text
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest -q tests/test_m1_null_calibration.py
46 passed in 30.27s

/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest -q tests/test_m1*.py
216 passed in 55.60s
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
- Contact-distance leaves are keyed by canonical unordered geom-pair identity
  and per-identity occurrence; contact topology mismatches fail before any
  numeric distance comparison.
- The parent validates and reduces each pair before scheduling the next one.
  `pair_measurements.json` publishes compact scalar physics/RGB records and
  immutable artifact references, while raw trajectories remain on disk once.
- `worker_timeout_seconds: 600` is frozen in both config and run spec. A
  timeout preserves partial stdout/stderr and records one failed attempt; the
  schedule continues without retry. A strict completed result missing
  `output_sha256` is rejected instead of synthesized.
- Dedicated result-file transport is isolated from runtime stdout/stderr and
  records return code, timeout state, sidecar/hash metadata, and partial logs.

## Scope and verification

All tests use fake adapters/process runners only at the runtime seam. No live environment,
policy/checkpoint, renderer calibration, replay restore, retry, tape
regeneration, or 40-process run was performed.  The frozen source registry,
action tape, runtime pins, user `AGENTS.md`, M0 evidence, and prior run outputs
were preserved.  The next task, after an empirical calibration pass, is the
separately controlled M1 authoritative source-vs-restored replay.

The checked-in null-calibration config self-hash is
`abbda1ee7d347a3c6cafad9f6f79a41a4227b84d063f2eea8ae7d53c2c5fae2b`; the
frozen source registry self-hash remains
`092c64910b7530c6d84a98a925124879e9a96196b54d45b284a70ccfb62dd49c`.

## Scoped files and self-review

The follow-up scope is limited to `scripts/m1_null_calibration.py`,
`scripts/m1_hard_gate.py`, `tests/test_m1_null_calibration.py`,
`configs/m1/null_calibration.yaml`, this report, and
`docs/m1_null_calibration.md`.  `AGENTS.md`, M0 evidence, prior manifests,
and existing run directories were not staged.  `py_compile`, YAML/config
self-hash, registry self-hash, and `git diff --check` pass.  No authoritative
prepare, renderer calibration, or 40-attempt runtime was run.
The follow-up implementation is committed as
`c0cc431669046054cd506e0a5a658fc846a08996`.

The remaining empirical concern is intentionally unchanged: M1-N0 still needs
the separately scheduled live duplicate-control calibration before its physics
and renderer envelopes can be accepted.
