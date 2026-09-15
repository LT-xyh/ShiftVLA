# F2 route adjudication and bounded F3a null authorization

**Status: ROUTE AMENDMENT / F3a AUTHORIZED WITH PRECONDITIONS**

Reviewed execution commit: `5bea98918a331dff21a1670c84e5dc150c65dd5e`.
This document is an explicit amendment to the mechanical F2->F3 dependency in M1-SA-v1. It does not rewrite the historical F2/F2H/F2R outcomes and does not declare strict `SA-renderer = PASS`.

## 1. F2R adjudication

The final F2R run is retained as `FAIL / NONCONFORMANT` under its strict evidence contract.

The observed renderer/environment behavior itself did not fail:

- deterministic discovery again selected the unique `EGL_MESA_device_software` device at ordinal `8`;
- all three predetermined fresh workers used ordinal `8`;
- all three returned process code `0`;
- all three completed official policy-free construction;
- all three made exactly one public render;
- all three returned the expected software EGL/GL identity;
- all three recorded `close_status = PASS`;
- all three were observed to exit by the parent;
- no replacement worker was run.

The strict F2R FAIL is caused by project-owned evidence instrumentation defects plus an execution-governance violation:

1. Parent and worker config hashes used different canonicalization domains. The parent computed `effective_config_sha256` before inserting that field, while the worker hashed the loaded mapping including the field. The resulting mismatch is deterministic and does not establish different renderer configuration bytes or behavior.
2. `cleanup_status` was defaulted to `UNKNOWN`; workers did record `close_status = PASS`, but the required explicit cleanup predicate was not implemented.
3. The dynamic F2R run started before every required regression test, cleanup implementation and input-byte gate had been completed. This violates the authorization preconditions and prevents strict F2R acceptance even though the observed renderer behavior was successful.

No additional F2 dynamic cohort is authorized. Do not create c6/c7/F2R2 or rerun a renderer worker to repair evidence.

## 2. Route-level decision

Continuing to block the scientific program on another renderer-admission rerun would recreate the guard-engineering critical path that ReplayVLA-P1 was explicitly created to avoid.

Therefore:

- strict `F2 = FAIL / NONCONFORMANT` is preserved;
- strict `SA-renderer = PASS` is **not** claimed;
- renderer operational evidence is accepted only for the narrower purpose of entering an independent duplicate-control null calibration;
- operational renderer status is `QUALIFIED_FOR_F3A_NULL_ONLY_WITH_LIMITATION`;
- the limitation must remain visible in future manifests and paper/reproducibility notes if the route survives later gates.

This is a scientific-route governance decision, not a reinterpretation of F2R as PASS.

The rationale is that F3a null is itself an independent empirical test of the actual official environment + renderer path over complete fixed-action trajectories. It is higher-value evidence for downstream replay correctness than another one-render infrastructure cohort.

## 3. State after this adjudication

- F0 = PASS.
- F1T = PASS / tensor-transport-qualified.
- F2 strict admission = `FAIL / NONCONFORMANT`.
- F2 renderer behavior = `OPERATIONALLY_OBSERVED` on c5 and F2R, with evidence-contract limitations.
- No further F2 dynamic execution is authorized.
- **F3a null calibration = AUTHORIZED**, subject to the preconditions below.
- F3b physical replay = NOT AUTHORIZED until F3a is pushed and reviewed.
- F4-F6 = NOT AUTHORIZED.
- old G1 remains BLOCKED; old E3-E6/G2 remain paused/not started.

## 4. Mandatory static preconditions before the first F3a env step

Codex must first make and test the following project-owned corrections without any F2 dynamic execution:

1. Define one self-hash convention for config documents (e.g. hash the canonical mapping with the self-hash field omitted) and use it consistently for any new F3 evidence.
2. Add the omitted F2 regression tests that can be tested statically/fake: hash-domain consistency, nonzero return code, ordinal mismatch, missing hashes, timeout terminalization, cleanup-record semantics, no-overwrite.
3. Record the F2 execution-governance deviation in the phase summary; do not hide it.
4. Ensure the new F3a entrypoint/config does not inherit the broken F2 parent/worker hash comparison.

These corrections are code/evidence hygiene only. They do not authorize a renderer rerun.

## 5. F3a scope

F3a is **null calibration only**. No capture/restore replay and no policy/checkpoint/processor execution.

Use the official CPU environment path and the frozen persisted action tape. Prefer the existing task-0/init-0/seed-2027 source if its actual bytes and identities still match the historical authority; re-verify rather than trust constants.

The null experiment must consist of at least 20 complete pre-registered duplicate-control pairs. For each pair A/B:

- fresh official environment construction for A and B;
- same fixed init-state identity and same fixed action tape bytes;
- no policy call and no regenerated actions;
- complete trajectory according to the frozen action budget/terminal semantics;
- retain both attempts whether they pass, fail, terminate early, time out or encounter a technical error;
- no replacement seed/pair to make the schedule look successful.

The renderer must remain the deterministic software path discovered in F2/F2R (unique `EGL_MESA_device_software`, ordinal 8) unless discovery at entry no longer matches. If it no longer matches, F3a is `BLOCKED`; do not choose another ordinal.

## 6. Schedule freeze before execution

Before the first dynamic null attempt, materialize a new ReplayVLA-P1 null schedule and immutable config under a fresh namespace, e.g.:

- `configs/replayvla/p1_null.yaml`
- `runtime/replayvla-p1/f3a/null_schedule.json`

The schedule/config must bind at least:

- this authority commit / contract amendment;
- execution commit;
- CPU interpreter/runtime identity;
- current task BDDL and init-state actual byte hashes;
- actual action-tape path, dtype, shape and byte hash;
- renderer rule and expected ordinal 8;
- current project-owned null/replay source hashes used by the entrypoint;
- exactly 20 (or a pre-declared larger fixed number of) pair IDs;
- attempt A/B identity for every pair;
- no replacement/retry policy;
- no-overwrite output root;
- trajectory budget and terminal semantics;
- planned trace/regime coverage accounting.

Do not inspect dynamic null results and then change the schedule in place.

## 7. Scientific null evidence requirements

Preserve the original M1 scientific standards that matter downstream:

- at least 20 complete duplicate-control pairs;
- every selected replay trace/regime used later must have at least 5 independent null pairs in its support group;
- contact/grasp/carried regimes require positive contact-rich evidence;
- discrete action bytes, contact IDs/sets as defined by the frozen oracle, task predicates, termination semantics, wrapper counters and gripper discrete state remain exact where the existing contract requires exactness;
- floating physics quantities use measured group-wise null envelopes rather than a new global epsilon;
- RGB/observation comparisons remain grouped by camera/key/regime/horizon; do not inflate physics tolerances because pixels differ;
- unknown/missing measurements are not zeros;
- every attempt publishes a terminal record.

If the existing null harness cannot produce these without changing scientific semantics, stop and return for route review rather than simplifying the oracle after seeing data.

## 8. F3a implementation policy

Prefer reusing the established scientific comparison/oracle code from `scripts/m1_null_calibration.py`, `scripts/m1_hard_gate.py` and `scripts/m1_state_replay.py` where semantically valid, wrapped by a thin ReplayVLA-P1 entrypoint/config.

Do not revive the old operation/native guard.

The F3a entrypoint must enforce:

- predecessor/authority and actual input-byte binding;
- no-overwrite;
- schedule immutability;
- exactly the registered attempts;
- no policy/checkpoint/processor import/call path;
- no capture/restore replay path;
- failure/timeout terminalization;
- parent-owned process cleanup;
- evidence hashes and runtime identity.

Synthetic/fake tests establish harness logic only; they are not null evidence.

## 9. F3a stop rules

Stop immediately and report without entering F3b if any of the following occurs:

- renderer discovery no longer uniquely selects ordinal 8;
- actual init-state/action-tape bytes differ from the frozen source and cannot be versioned without changing the approved cohort;
- official null execution requires policy/checkpoint/processor;
- the null harness must alter task/action/terminal semantics;
- required attempt records are lost/overwritten;
- unknown scientific owner or comparison quantity prevents a valid null oracle;
- the pre-registered 20-pair schedule cannot be completed into an interpretable terminal dataset within the bounded project budget.

A nonzero measured null envelope is not itself a failure. The purpose of F3a is to measure the platform's empirical duplicate-control variability and determine whether a defensible replay oracle exists.

## 10. F3a completion states

After the fixed schedule:

- `PASS`: complete evidence supports valid grouped physical/renderer null envelopes and required exact discrete invariants;
- `FAIL`: duplicate controls violate exact invariants or variability makes the intended replay claim scientifically unusable;
- `BLOCKED`: technical/evidence incompleteness prevents adjudication.

Even on F3a PASS, F3b physical replay remains NOT AUTHORIZED until the commit/evidence is pushed and reviewed.

## 11. Publication interpretation

Do not claim that F2 strict renderer admission passed. If the project reaches a paper, the reproducibility statement should instead distinguish:

- operational renderer observations (multiple fresh construction/render/close workers on the frozen software EGL path);
- F2 evidence-instrumentation nonconformance;
- independent F3a duplicate-control calibration as the empirical basis for replay tolerances.

The paper's scientific credibility must ultimately rest on null calibration, replay closure and branch controls, not on a claim of exhaustive renderer/native admission.
