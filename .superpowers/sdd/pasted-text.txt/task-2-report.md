# Task 2 report — M1-N0 fresh-process null calibration

## Status

Implementation is complete and locally verified at the code/test seam. The
authoritative 40-process calibration was intentionally not run, as required;
empirical M1-N0 remains a separate runtime operation.

## Red/green evidence

The first focused run was `13 passed, 1 failed`: strict envelope aggregation
accepted a snapshot missing configured invariant roots. The strict root check
was added and the focused suite became `14 passed`.

Two targeted tests then exposed the remaining aggregation gaps:

```text
2 failed: contact tuples were discarded as numeric leaves; renderer groups
ignored configured camera/observation_key
```

Contact distances are now explicit float64 leaves (`contacts[i].distance`),
strict action evidence is checked from every frozen window, and renderer
aggregation routes the configured direct key/camera. The final focused result:

```text
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest -q \
  tests/test_m1_null_calibration.py
15 passed in 13.47s
```

Relevant replay/predicate verification:

```text
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest -q \
  tests/test_m1_state_replay.py -k 'predicate or replay'
112 passed in 19.22s
```

Additional checks:

```text
py_compile scripts/m1_null_calibration.py scripts/m1_state_replay.py \
  tests/test_m1_null_calibration.py tests/test_m1_state_replay.py: PASS
strict null-calibration config validation and self-hash: PASS
git diff --check: PASS
direct --schema-version invocation: PASS via focused test
```

## Implementation

The strict path keeps configured-interpreter subprocess probes, pinned
runtime facts and live GL identity, `/proc` PID start identities, observed
protocol counters, exact 20-pair/40-attempt schedule and terminal contract,
canonical no-pickle artifact hashes, reconstructible final pair registry and
terminal manifest. Strict workers use canonical compiled-model physics and
observation fingerprints, one official policy-free construction, the exact
float32 tape, recursive official observation metadata, actual frozen actions,
and no retry/restore/policy/processor/post-terminal calls.

Envelope aggregation retains full configured quantity roots, contact topology
as exact semantics, contact distances as floating physics samples, and
snapshot/action-derived contact, sustained grasp, and carried evidence. RGB
groups retain camera × observation key × regime × horizon structure and
publish exact/non-exact metrics without changing physics tolerances.

## Changed files in this handoff

- `scripts/m1_null_calibration.py`
- `tests/test_m1_null_calibration.py`
- `configs/m1/null_calibration.yaml`
- `.superpowers/sdd/pasted-text.txt/task-2-report.md`

The preceding M1 implementation commit already contains the minimal
state-replay predicate exposure and its documentation/tests; they were not
rewritten in this handoff.

## Self-review, assumptions, and concerns

- All tests use fake adapters/process runners only. No live environment,
  policy/checkpoint, renderer calibration, or 40-process run was performed.
- The frozen source registry/action tape and state-replay runtime remain the
  authoritative inputs; no tape regeneration or threshold widening occurred.
- Strict pair validation now requires real `/proc` identity provenance,
  complete observed counters, runtime facts/GL identity, exact action-window
  evidence, and actual artifact/output hashes. Failed/divergent attempts stay
  registered and are never retried.
- Empirical M1-N0 is still unrun, so no PASS claim about runtime
  reproducibility is made here. The live run must verify its device-visible
  GL identity and all 20 valid contact-rich pairs before acceptance.
- Existing user-owned `AGENTS.md`, M0 evidence, prior manifests, and old run
  outputs were preserved and excluded from staging.
