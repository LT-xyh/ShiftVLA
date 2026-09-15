# ReplayVLA-P1 D3 runtime route review

**Status: APPROVED F1-ONLY CONTINUATION**

This review adjudicates the D3 kill-gate result recorded at commit
`90953e50338d1cbe6148a0c0a0fa7bc97e02564b` without rewriting that historical
evidence.

## Repository/evidence status reviewed

- F0 is accepted as PASS within the recorded local scope, pending no contradiction from later evidence.
- F1 baseline qualification is BLOCKED.
- F2-F6 remain NOT AUTHORIZED.
- Old G1 remains BLOCKED; old E3-E6 and G2 are not resumed.

## Candidate-budget clarification

`docs/replayvla-p1/02_execution_plan.md` defines the F1 candidate budget as:

> current runtime baseline + one minimal compatibility candidate

The CPU simulator runtime and the DCU policy runtime are two components of the
single current baseline architecture. They are not two alternative compatibility
candidates. Therefore the baseline qualification did **not** consume the one
allowed compatibility-candidate slot.

The committed `runtime/replayvla-p1/f1_summary.json` field
`candidate_budget_used: 2` is retained as historical evidence of the original
adjudication, but it is superseded for route control by this review.

Likewise, lack of an accelerator in the CPU simulator interpreter is not by
itself an F1 failure. The CPU runtime is responsible for simulator-side host
operations. The host/device bridge requirement belongs to the DCU policy
runtime/candidate.

## F1 baseline conclusion

The baseline provides a sufficiently specific failing boundary to justify the
single compatibility candidate:

- CPU runtime NumPy->Torch and Torch->NumPy host bridges pass.
- DCU runtime uses Python 3.11.16, NumPy 2.2.6, Torch 2.7.1.
- DCU Torch import emits `_ARRAY_API not found`.
- DCU NumPy->Torch reaches `RuntimeError: Numpy is not available`.

This is a bounded compatibility problem, not evidence that ReplayVLA or
exact-state replay is scientifically infeasible.

## Authorized compatibility candidate

Exactly one compatibility candidate is authorized under F1:

- keep Python 3.11.16 unchanged;
- keep Torch 2.7.1 unchanged;
- keep Transformers, checkpoint, task/runtime semantics and the remaining
  scientific stack unchanged;
- change only NumPy to `1.26.4`;
- materialize it in a new isolated environment or an equivalently auditable
  isolated package overlay;
- do not mutate the original DCU virtual environment.

Before materialization, record the baseline package/runtime identity. If package
metadata/constraints establish incompatibility before installation, publish
BLOCKED and stop rather than force-installing.

## Required F1-only validation

The candidate must record and pass all applicable checks:

1. exact Python executable/version;
2. NumPy/Torch versions and module origins;
3. install/package origin and dependency-integrity check;
4. no NumPy ABI / `_ARRAY_API` failure on Torch import;
5. float32 NumPy -> Torch CPU tensor;
6. Torch CPU tensor -> NumPy;
7. K100 visibility in the DCU runtime;
8. CPU host tensor -> DCU tensor -> CPU tensor roundtrip;
9. exact dtype/shape/value equality for the synthetic bridge;
10. repeated fresh-process qualification with stdout/stderr/traceback retained;
11. frozen candidate package/runtime lock and manifest.

F1 remains synthetic-only. Environment construction, init-state loading, EGL,
render, processor/checkpoint/policy execution, rollout, null calibration and
replay remain forbidden.

## Stop condition

If this single candidate is FAIL, BLOCKED, unavailable, or requires changing
software beyond the authorized NumPy-only compatibility delta, stop F1. Do not
try another NumPy version, Torch version, Python version, or a third runtime
candidate. Return to route-level PIVOT review.

If it passes, freeze its identity and commit/push the evidence, but do not enter
F2 until a separate F2 strong review is completed.

## Git transport note

The repository-specific SSH invocation that bypasses the broken system include
and selects the dedicated key may be used for fetch/push without modifying
`/etc/ssh` or global/shared Git configuration. Authentication repair is not a
scientific runtime change.
