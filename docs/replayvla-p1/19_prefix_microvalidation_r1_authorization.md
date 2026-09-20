# Prefix reexecution MV-R1 runtime requalification

**Status: MV-R1 RUNTIME REQUALIFICATION AUTHORIZED EXACTLY ONCE.**

Branch: `xyh/replayvla-p1`

Reviewed correction candidate:

`65d2ee3c5fbe58381039c06ed24f2d73ffcc320c`

Correction authority:

`docs/replayvla-p1/18_prefix_microvalidation_provenance_correction.md`

Scientific route authority:

`docs/replayvla-p1/16_prefix_reexecution_paper_route.md`

Historical doc17 microvalidation evidence commit:

`33f0e2cb1f699254f92fb7c46df93ec653b8f864`

The historical BLOCKED evidence remains immutable.

## 1. Review verdict

The candidate is approved as a bounded provenance correction.

The historical default semantics remain unchanged:
- callers that do not pass `task_source_content_pins` retain the repo-relative hf-libero ancestry guard.

Only the P1 microvalidation path explicitly opts into content-pinned admission.

The content-pinned path still resolves BDDL/init-state through the installed LIBERO API actually used
by runtime, resolves the consumed files strictly, requires readable regular files, hashes those exact
bytes, checks exact frozen task identity, and checks exact frozen SHA256 identities.

No task file is copied, rewritten, redirected, or symlinked by the new seam.

## 2. Frozen task content identities

Suite/task/init:

- suite: `libero_spatial`
- task_id: `0`
- init_state_id: `0`
- environment seed: `2027`

Expected BDDL SHA256:

`9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b`

Expected init-state SHA256:

`cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`

## 3. Requalification sequence

Exactly one new harness invocation is authorized.

Sequence:

1. focused pytest;
2. MV-P2-A same-current-state camera intervention;
3. MV-P2-B switch-index lifecycle;
4. MV-P1 one-query DCU explicit-noise select_action;
5. compact evidence;
6. STOP.

No phase may run unless every earlier phase PASSes.

## 4. New immutable evidence identity

Do not touch the historical evidence:

`runtime/replayvla-p1/prefix_reexecution_microvalidation.json`

MV-R1 must write to:

`runtime/replayvla-p1/prefix_reexecution_microvalidation_r1.json`

Recommended transient work directory:

`runs/replayvla-p1/prefix_reexecution_microvalidation_r1.work`

The new compact evidence is no-overwrite.

## 5. Focused tests

The harness owns focused pytest.

Tests must include:
- the existing prefix-reexecution contract tests;
- microvalidation harness tests;
- relocated task-source exact-content PASS;
- wrong BDDL hash BLOCK;
- wrong init-state hash BLOCK;
- missing/non-regular/unreadable source BLOCK;
- frozen task identity drift BLOCK;
- historical no-pin repo-relative guard preserved;
- P1 content pins reaching MV-P2-A, MV-P2-B and MV-P1.

If focused pytest does not PASS, no real runtime phase may start.

## 6. Provenance evidence requirement

For every real runtime phase that constructs the environment, task-source evidence must identify the
actual consumed runtime paths and exact content identities.

At minimum record:
- source_check = runtime_consumed_content_sha256;
- actual resolved BDDL path;
- actual BDDL SHA256;
- expected BDDL SHA256 through the content-pins evidence;
- actual resolved init-state path;
- actual init-state SHA256;
- expected init-state SHA256 through the content-pins evidence;
- suite/task/init identity.

A hash mismatch is BLOCKED. Do not redirect the runtime to another file.

## 7. MV-P2 and MV-P1 gates

All doc17/doc16 scientific and implementation gates remain unchanged.

No changes are authorized to:
- CameraModder intervention semantics;
- +15 degree yaw definition;
- image exactness gates;
- qpos/qvel/ctrl/sim-time invariance gates;
- switch index 50 semantics;
- dummy-action lifecycle;
- paired policy-noise key;
- explicit-noise select_action transport;
- one-query MV-P1 limit;
- queue semantics.

## 8. Stop rule

This is the only authorized MV-R1 runtime requalification.

Once the new compact evidence exists:
- do not rerun;
- do not overwrite/delete/rename it;
- do not modify implementation/config/runtime and retry.

If MV-R1 again BLOCKS before MV-P2-A because of another inherited provenance/path plumbing
requirement, do not continue serial infrastructure repair. Return to route-level review.

If MV-R1 reaches MV-P2 or MV-P1 and BLOCKS there, preserve that observed result and return to
reviewer.

If MV-R1 PASSes, preserve evidence and return to reviewer.

## 9. Forbidden

Do not:
- modify packages/runtime;
- modify implementation/tests/config during execution;
- repair F3N/F3b;
- run the 40-rollout pilot;
- compute paper estimands;
- retry/replacement;
- use git reset --hard;
- use git clean;
- use git add -A.

## 10. Next phase

Even after MV-R1 PASS:
- pilot remains NOT AUTHORIZED;
- reviewer adjudication is required before scientific rollout.
