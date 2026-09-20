# Prefix reexecution microvalidation provenance correction

**Status: MV-R1 REPO-ONLY CORRECTION AUTHORIZED.**

Branch: `xyh/replayvla-p1`

Historical one-shot microvalidation authority:
- `docs/replayvla-p1/17_prefix_reexecution_microvalidation_authorization.md`

Historical execution evidence:
- execution HEAD: `32cbe471c4cc9987b64fbff2ac4c98717d664721`
- evidence commit: `33f0e2cb1f699254f92fb7c46df93ec653b8f864`
- result: BLOCKED before MV-P2-A

Scientific route authority:
- `docs/replayvla-p1/16_prefix_reexecution_paper_route.md`

## 1. Adjudication

The doc17 one-shot run is complete and immutable.

Its blocker:

`DCUPreflightError: LIBERO BDDL/init state escaped pinned hf-libero tree`

occurred after focused pytest PASS and before any MV-P2-A environment qualification,
MV-P2-B switch-lifecycle validation, DCU worker launch, policy query, or scientific rollout.

This is not evidence against:
- the CameraModder observation-only seam;
- the prefix-reexecution switch semantics;
- explicit-noise select_action;
- the paper hypothesis.

It is a provenance-admission defect inherited from the old `dcu_preflight` runtime.

## 2. Root cause

`scripts/dcu_preflight.py::_task_source_evidence()` resolves the BDDL and init-state actually
selected by the installed LIBERO runtime, but then additionally requires those resolved files to be
located under:

`<current repository root>/external/hf-libero/...`

The currently authorized ordinary runtime is intentionally relocated and still consumes frozen
artifacts rooted outside the current worktree. The M0 config itself binds multiple immutable runtime
artifacts under `/public/home/xuyinghao/workspace/vla/...`.

Therefore repo-relative location is not a valid scientific identity criterion for this pivoted
Paper-1 route.

The correct identity is the exact consumed task bytes plus frozen task identity.

## 3. Frozen task0 input identities

For `libero_spatial`, task 0, init 0:

BDDL expected SHA256:

`9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b`

Init-state expected SHA256:

`cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`

These values were already frozen in the accepted F3N input bindings and may be reused only as
content identities. This does not re-open the exact-state/F3N route.

## 4. Authorized correction

Implement one narrow content-pinned task-source admission seam.

The runtime must still:
1. resolve the BDDL and init-state through the installed LIBERO API actually used by the runtime;
2. resolve each path strictly;
3. require each to be a regular readable file;
4. hash the exact consumed bytes;
5. require the BDDL SHA256 to equal the frozen BDDL hash;
6. require the init-state SHA256 to equal the frozen init-state hash;
7. record the actual resolved paths and hashes in evidence.

Do not require the files to be descendants of the current Git worktree.

Do not copy files into the repository merely to satisfy admission.

Do not redirect the runtime to a different BDDL/init-state path.

## 5. Scope

The correction may touch only the minimum files necessary to:
- express frozen BDDL/init-state hashes in the P1 microvalidation contract;
- validate exact consumed content;
- test fail-closed behavior;
- propagate task-source evidence into the microvalidation compact evidence where practical.

Preferred scope:
- `scripts/dcu_preflight.py` only if a generic optional content-pinned seam can preserve historical callers;
- `scripts/p1_prefix_reexecution_microvalidate.py`;
- `configs/replayvla/p1_prefix_reexecution_pilot.yaml` or a P1 microvalidation-specific constant/config binding;
- focused tests.

Do not weaken historical M0 semantics for unrelated callers.

If changing `_task_source_evidence()` globally would silently relax old callers, introduce an explicit
optional expected-content binding used only by the P1 microvalidation route.

## 6. Required fail-closed cases

Repo-only tests must prove:
- exact expected BDDL/init hashes PASS independent of repository-relative location;
- wrong BDDL hash BLOCKS;
- wrong init-state hash BLOCKS;
- missing file BLOCKS;
- unreadable/non-regular source BLOCKS where testable;
- runtime-resolved path is recorded;
- runtime-resolved bytes, not a copied/reference file, are hashed;
- task/suite/init identity remains frozen;
- no path rewriting/copying occurs;
- no package/runtime modification occurs.

## 7. Forbidden

Do not:
- delete or overwrite the historical BLOCKED evidence;
- rerun the microvalidation under the old implementation;
- modify camera/noise/switch scientific criteria;
- modify installed LIBERO/LeRobot/robosuite packages;
- create symlinks or copy source files to satisfy a path guard;
- repair F3N/F3b;
- run the 40-rollout pilot;
- run real LIBERO/DCU during this correction;
- start Codex runtime validation.

## 8. Requalification policy

This authority does NOT itself authorize another real microvalidation.

After a minimal repo-only correction candidate:
1. independent reviewer inspects the diff and focused tests;
2. if the correction is genuinely limited to provenance admission, reviewer may authorize exactly one
   MV-R1 requalification;
3. historical doc17 evidence remains immutable.

If MV-R1 requalification again fails before reaching MV-P2-A because of another inherited
provenance/path plumbing requirement, do not continue serial infrastructure repair. Return to
route-level review.

## 9. Current phase

- prefix-reexecution scientific route: CONDITIONAL GO;
- focused pytest from historical doc17 run: observed PASS (40 passed);
- camera/runtime microvalidation: NOT YET OBSERVED;
- MV-R1 repo-only provenance correction: AUTHORIZED;
- MV-R1 runtime requalification: NOT YET AUTHORIZED;
- 40-rollout pilot: NOT AUTHORIZED.
