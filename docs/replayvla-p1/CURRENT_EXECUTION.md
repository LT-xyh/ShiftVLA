# ReplayVLA-P1 CURRENT EXECUTION

Status: MV-R1 REPO-ONLY PROVENANCE CORRECTION AUTHORIZED
Branch: xyh/replayvla-p1

Latest authority:
- docs/replayvla-p1/18_prefix_microvalidation_provenance_correction.md

Scientific route authority:
- docs/replayvla-p1/16_prefix_reexecution_paper_route.md

Historical microvalidation authority:
- docs/replayvla-p1/17_prefix_reexecution_microvalidation_authorization.md

Historical microvalidation evidence:
- execution HEAD: 32cbe471c4cc9987b64fbff2ac4c98717d664721
- evidence commit: 33f0e2cb1f699254f92fb7c46df93ec653b8f864
- focused pytest: PASS (40 passed)
- harness: BLOCKED before MV-P2-A
- blocker: DCUPreflightError: LIBERO BDDL/init state escaped pinned hf-libero tree
- MV-P2-A: NOT_RUN
- MV-P2-B: NOT_RUN
- MV-P1: NOT_RUN
- dynamic/DCU worker launches: 0
- scientific rollout: false
- retry/replacement: 0/0

Historical evidence is immutable.

## Adjudication

The blocker is an inherited provenance-location defect, not a camera/prefix/noise scientific result.

Current dcu_preflight resolves the actual installed-LIBERO BDDL/init files and then requires them to
live under:

`<current repo>/external/hf-libero/...`

The authorized ordinary runtime is relocated and already binds immutable artifacts under
`/public/home/xuyinghao/workspace/vla/...`.

For the pivoted prefix-reexecution Paper-1 route, the task-source admission criterion is the exact
consumed bytes plus frozen task identity, not current-worktree ancestry.

## Frozen task0 content identities

BDDL SHA256:

`9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b`

Init-state SHA256:

`cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`

These hashes are content identities only. This does not reopen F3N/exact-state work.

## Current authorized work

Repo-only implementation correction named MV-R1.

Implement a narrow content-pinned task-source admission seam that:

1. resolves BDDL/init through the installed LIBERO API actually used by runtime;
2. strictly resolves the resulting files;
3. verifies regular/readable files;
4. hashes the exact consumed bytes;
5. requires exact frozen BDDL/init SHA256;
6. records actual resolved paths and hashes;
7. does not require current-repo ancestry;
8. does not copy, rewrite, redirect, or symlink the files.

Preserve task/suite/init identity.

Do not weaken unrelated historical M0 callers. If necessary use an explicit optional expected-content
binding only for the P1 microvalidation path.

## Frozen scientific/runtime parts

Do not modify:
- CameraModder intervention semantics;
- +15 degree yaw definition;
- obs/action switch timing;
- paired-noise semantics;
- G-P1 remote select_action path;
- MV-P2-A/P2-B scientific gates;
- MV-P1 query semantics;
- installed packages/runtime.

## Repo-only acceptance

Tests must include:
- exact content match PASS outside current repo path;
- wrong BDDL hash BLOCK;
- wrong init hash BLOCK;
- missing/non-regular source BLOCK where testable;
- actual consumed path/hash evidence;
- no path rewrite/copy;
- frozen task/suite/init preserved.

## Runtime state

MV-R1 real requalification: NOT AUTHORIZED YET.

Do not:
- rerun historical harness;
- run LIBERO/DCU;
- start Codex runtime validation;
- delete/overwrite historical compact evidence;
- run pilot;
- repair F3N/F3b.

After one minimal correction candidate, return to independent reviewer.
