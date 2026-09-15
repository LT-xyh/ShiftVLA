# F3a provenance recovery and bounded null-resume authorization

**Status: APPROVED FOR ONE BOUNDED STATIC PROVENANCE RECOVERY, THEN F3a NULL ONLY IF ALL EXACT-BYTE GATES PASS**

Authority branch: `xyh/replayvla-p1`.
Reviewed predecessor: `f7c19a0fad15238894ce05e7d4f6017231c77af1`.

This document does not authorize F3b replay, policy/processor/checkpoint execution, camera intervention, or any new F2 renderer cohort.

## 1. Adjudication of the F3a block

The F3a preparation at `f7c19a0` is accepted as a fail-closed static result:

- `F3a = BLOCKED` for schedule v1;
- dynamic execution did not start;
- zero pairs and zero attempts were run;
- the old F2R governance deviation remains recorded;
- no null/replay result exists yet.

The blocking predicate is not a null-calibration scientific failure. The frozen M1 registry contains absolute historical artifact paths. In particular, its trace binds a source episode row, parent run manifest and parent terminal manifest by absolute path plus SHA-256. `scripts/m1_hard_gate.load_frozen_registry()` requires those paths to resolve to actual non-symlink regular files and verifies their bytes. The registry also contains an absolute tape-provenance path. In the ReplayVLA-P1 worktree, at least the historical `source_file` path is no longer reachable, so schedule preparation correctly stopped before any dynamic operation.

The registry itself already records content-addressed identities and a relocation history. Therefore one bounded recovery of the exact historical bytes is scientifically legitimate if and only if the recovered files match the pre-existing frozen SHA-256 values exactly. Path substitution without byte identity is not allowed.

## 2. Exact historical identities that must be recovered

For trace `libero_spatial-task000-init000`, recover the exact bytes already declared by the frozen registry:

- source episode row SHA-256: `772dccb25c1692c5ce4b4d791f82c736e2a1921d9541a07e40dd2ed92873306e`;
- parent run manifest SHA-256: `fd8a0ae67a5f2c295b7fd84a71b2ff34319b7a2aa64c84374ab788fb774e399f`;
- parent terminal manifest SHA-256: `338588dedaad3e929ca9023b984d509549fd0743b77a7cce579c70712d4edef1`;
- action semantic bytes SHA-256: `c17bc44ad8195fecb42a80b3b272828761a9df6d88dd2ed92873306e`;
- original registry self-hash: `092c64910b7530c6d84a98a925124879e9a96196b54d45b284a70ccfb62dd49c`.

The recovered source artifacts must be regular files, not symlinks. Their bytes must match these hashes before they may be copied or rebound.

## 3. Bounded search scope

The recovery may read only the known project/history roots needed to locate these exact artifacts, including:

- `/public/home/xuyinghao/workspace/vla`;
- `/public/home/xuyinghao/workspace/vla/ShiftVLA` if it exists;
- `/public/home/xuyinghao/workspace/replayvla-p1`;
- the exact historical relative run directory `runs/m0_baseline_a/20260830T023451Z_655986_c2f3caa5` beneath those roots.

A filename/path match is not sufficient. Hash every candidate before acceptance. Do not modify, clean, stage, or normalize the historical/main worktree.

Do not perform an unbounded filesystem scan outside these project roots. Do not use network retrieval as a replacement for missing local historical bytes.

If any of the three required historical JSON artifacts cannot be recovered with its exact frozen SHA-256, recovery is `BLOCKED` and ReplayVLA-P1 returns for route-level PIVOT review. Do not reconstruct the missing episode row from the action tape, logs, registry metadata, or a new rollout.

## 4. Controlled recovery bundle

If all exact artifacts are found, copy their bytes without parse/re-serialization into a new controlled, versioned bundle, for example:

`runtime/replayvla-p1/f3a/provenance_v2/`

The bundle should contain compact names for:

- the source episode row;
- parent run manifest;
- parent terminal manifest;
- a `recovery_manifest.json` recording original declared path, located path, copied path, size and SHA-256 for each artifact.

The copy must be byte-identical. Verify the copied file hash again after writing. Refuse overwrite and reject symlinks.

The action tape need not be duplicated if the current tracked tape exists and its loaded contiguous float32 `[82,7]` action bytes still hash to `c17bc44...`. It may be referenced through a new relative path. If copied instead, the same no-overwrite and exact-byte rules apply.

## 5. Derived registry v2

Do not edit or replace `runtime/m1/m1_hard_gate_tape_registry.json`. It remains immutable historical authority and its schedule-v1 block remains preserved.

Create a new ReplayVLA-P1 derived registry, for example:

`runtime/replayvla-p1/f3a/m1_hard_gate_tape_registry.relocated.v2.json`

It must be derived mechanically from the old registry. Permitted semantic changes are limited to path-bearing provenance fields required to point at the exact recovered bytes/current exact tape, plus the resulting new registry self-hash. The frozen task identity, trace identity, seven windows/regimes, source-coverage content, action dtype/shape/hash, terminal evidence, capture offsets and continuation horizons must remain exactly equal.

At minimum rebind:

- `traces[0].source_file`;
- `traces[0].parent_run_manifest`;
- `traces[0].parent_terminal_manifest`;
- `tape_provenance.per_trace[libero_spatial-task000-init000].path`.

Use paths that the existing loader resolves deterministically from the derived registry location, preferably relative paths into the controlled recovery bundle/current tracked tape. No symlink indirection.

Publish a `registry_derivation.json` containing:

- old registry path and self-hash;
- new registry path and self-hash;
- exact JSON pointers/fields changed;
- old and new path values;
- byte hashes proving every rebound target is the same frozen artifact;
- a semantic-equivalence check showing no non-path scientific field changed.

The existing `scripts.m1_hard_gate.load_frozen_registry()` must successfully load the new registry without weakening its validation code.

## 6. Tests before any dynamic F3a work

Add focused regression coverage for at least:

- missing historical artifact -> recovery BLOCKED;
- wrong candidate hash -> recovery BLOCKED;
- symlink candidate -> reject;
- copy changes bytes -> reject;
- derived registry changes any non-permitted scientific field -> reject;
- derived registry target missing/hash mismatch -> reject;
- old registry remains byte-identical;
- derived registry loads under the existing strict loader;
- current action tape semantic bytes remain exact float32 `[82,7]` with SHA `c17bc44...`;
- schedule-v1 remains preserved as BLOCKED;
- no dynamic import/environment/renderer operation occurs during recovery preparation.

Run the relevant focused test suite and `git diff --check` before continuing.

## 7. Schedule v2

Only after recovery and derived-registry validation PASS may a new null schedule be frozen. Do not overwrite schedule v1 or `p1_null.yaml`.

Create versioned artifacts, for example:

- `configs/replayvla/p1_null_v2.yaml`;
- `runtime/replayvla-p1/f3a/null_schedule_v2.json`;
- a fresh output root such as `runs/replayvla-p1/f3a-v2-20260915`.

Schedule v2 must bind:

- this authority commit;
- the derived registry path and new self-hash;
- original registry self-hash as `derived_from` provenance;
- recovery-manifest hash;
- actual init-state/BDDL/action-tape bytes and hashes;
- the same task/init/seed, renderer identity, 20-pair/40-attempt budget, no-retry semantics and scientific null oracle already approved for F3a.

The fact that schedule v1 was blocked is retained; schedule v2 is a new cohort created before any dynamic null attempt, not a replacement of a failed empirical pair.

## 8. Automatic resume permission

If and only if Sections 2-7 all PASS mechanically, Codex may continue directly into the already-approved F3a null experiment without another checkpoint.

The dynamic scope remains exactly the F3a scope from `10_f2_route_adjudication_and_f3a_authorization.md`:

- 20 predetermined duplicate-control pairs / 40 attempts;
- fresh official environments;
- same exact init-state and fixed action tape;
- policy/checkpoint/processor disabled;
- no replay capture/restore;
- deterministic renderer entry check must still uniquely select ordinal 8;
- all planned attempts retained; no replacement/retry;
- exact discrete invariants plus grouped physical and renderer null envelopes;
- contact/grasp/carried coverage requirements preserved.

F3b remains `NOT AUTHORIZED` even if F3a PASS.

## 9. Stop rules

Stop with `F3a = BLOCKED` and return for route review if:

- any exact historical provenance artifact cannot be recovered;
- any recovered hash differs;
- creating the derived registry would require changing a non-path scientific field;
- the existing strict registry loader cannot accept the derived registry without weakening validation;
- actual init-state, BDDL or action tape bytes drift;
- renderer entry identity no longer uniquely selects ordinal 8;
- a package/runtime/scientific semantic change is needed.

Once dynamic schedule v2 begins, retain the original F3a rules: no replacement pair, no post-hoc threshold changes, no replay, and no F3b continuation without review.

## 10. State after this authorization

- F0 = PASS.
- F1T = PASS.
- F2 strict admission = FAIL / NONCONFORMANT; no further F2 dynamic cohort is authorized.
- F2 renderer behavior remains `OPERATIONALLY_OBSERVED` only.
- F3a schedule v1 = BLOCKED before dynamic execution.
- F3a provenance recovery v2 = AUTHORIZED under this document.
- F3a dynamic null v2 = CONDITIONALLY AUTHORIZED only after all exact-byte/static gates above PASS.
- F3b-F6 = NOT AUTHORIZED.
- old G1 remains BLOCKED; old operation-guard E3-E6/G2 remain paused/not started.
