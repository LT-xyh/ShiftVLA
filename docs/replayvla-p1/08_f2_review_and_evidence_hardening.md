# F2 review: positive dynamic cohort, evidence hardening required

**Status: F2H AUTHORIZED; F3 NOT AUTHORIZED**

Reviewed execution commit: `2a826864bed34942244ea6dcda297b22eb8394c0`.
Reviewed F2 authority: `docs/replayvla-p1/07_f2_strong_review_and_authorization.md` from commit `ce4e6df13890d6299c4306dbfe96dc00720e3fe7`.

This review does **not** reject the observed renderer behavior in cohort `f2-20260915-c5`. It withholds formal `SA-renderer = PASS` until the existing c5 evidence is statically hardened and re-adjudicated. No additional renderer cohort is authorized by this document.

## 1. Positive findings accepted from the pushed commit

The pushed implementation and summary support the following narrow observations:

- one deterministic discovery rule was implemented: the unique device exposing `EGL_MESA_device_software`;
- the reported enumeration contained nine EGL devices and selected ordinal `8`;
- the F2 runner schedules exactly three worker ids in one cohort and does not contain a replacement-worker loop;
- the committed summary reports 3/3 construction, exactly one public render per worker, 3/3 close, identical EGL/GL identity and no forbidden operation;
- the F2 code remains policy/checkpoint/replay/null free;
- no package/runtime or legacy config was overwritten in the reviewed commit.

These are useful positive observations. They are not, by themselves, sufficient to accept the formal evidence gate below.

## 2. Blocking evidence/contract findings

### F2-R1 — wrong authority binding

`configs/replayvla/p1_renderer.yaml` records:

`parent_authority: 411320d01cdb983b81d1f04a9b378ebe4f634cfe`

but F2 was actually authorized by commit:

`ce4e6df13890d6299c4306dbfe96dc00720e3fe7`

The historical c5 input must not be silently rewritten. Record an explicit erratum and preserve the exact executed config bytes/hash. Any active template used in the future may be corrected, but the c5 evidence must continue to identify the bytes actually executed.

### F2-R2 — predecessor and input byte bindings are not mechanically enforced

`validate_f1t_predecessor()` currently accepts a JSON file based on semantic fields (`status` and `authority`) rather than a frozen expected SHA256 / exact-byte identity. The F2 config also contains claimed hashes for the runtime lock and init-state asset, but the F2 entrypoint does not mechanically recompute and compare those bytes before construction.

The F2 authority required actual predecessor/config/input binding rather than caller-provided hash claims. Under the M1-SA trusted-OS/no-concurrent-replacement assumption, source-accounted path binding is acceptable only if the exact bytes/path/checkout identity used by c5 can be reconstructed from retained evidence and pinned source state.

### F2-R3 — GitHub does not contain enough compact c5 evidence for independent review

`runtime/replayvla-p1/f2_summary.json` points to local files under `runs/replayvla-p1/f2-20260915-c5/`, but the reviewed GitHub tree does not contain the c5 effective config, discovery terminal record, three worker terminal records, adjudication record, or raw stdout/stderr hashes.

Raw logs do not need to be committed. A compact immutable evidence manifest must, however, contain their paths and SHA256 values plus the terminal fields required for adjudication. The exact c5 `effective_config.json` and adjudication/terminal summaries should be committed in compact form.

### F2-R4 — adjudication omits required mechanical checks

The current `adjudicate_workers()` checks the number of workers, worker status, render count, close status, parent-observed exit, forbidden count, replacement flag and EGL/GL identity. It does not mechanically require, for each worker:

- `returncode == 0`;
- the same frozen effective-config SHA256;
- the same selected ordinal as discovery/effective config;
- stdout/stderr SHA256 presence;
- the expected execution/runtime identity fields.

`run_step()` also has no timeout. The worker exception path does not attempt best-effort close in a `finally` block after successful construction. These are harness defects, not evidence that c5 renderer behavior failed, but they must be fixed before this harness is treated as the F2 authority implementation.

### F2-R5 — required regression coverage is incomplete

The reviewed F2 test file covers five useful cases but does not cover several tests explicitly required by the F2 authorization, including frozen effective-config identity, no-overwrite/duplicate output, timeout/failure terminalization, worker return code, selected-ordinal/config-hash agreement, observed-vs-trusted classification, and exact predecessor/input byte mismatch.

### F2-R6 — cohort-budget governance deviation

The report preserves four failed engineering cohorts before c5 (`f2-20260915`, `-c2`, `-c3`, `-c4`). This exceeds the original execution plan's one replacement-cohort budget. The failures are reported as entry/config/path issues rather than renderer-selection cherry-picking, so this review does not discard c5 solely for that reason. The deviation must be recorded explicitly and **no further dynamic F2 cohort is authorized** under the current plan.

## 3. F2H authorization: static/evidence hardening only

F2H may modify project-owned F2 code/tests/config templates and may read/hash the already existing c5 evidence. It may not construct an environment or create a new renderer cohort.

Allowed work:

1. Preserve the exact c5 input config bytes and SHA256; add an erratum for the incorrect parent-authority field. Do not rewrite c5 history.
2. Correct the active source template for future provenance bookkeeping, while making clear that c5 used the historical bytes.
3. Add exact SHA256 binding for the accepted F1T predecessor summary/production trace and for required F2 input files.
4. Recompute/verify the runtime-lock, init-state and BDDL bytes plus pinned checkout/path identity from the retained c5 context. Record whether the M1-SA trusted-OS assumption is sufficient to bind the official construction path to those bytes. If it is not, stop with `F2H = BLOCKED`.
5. Commit a compact c5 evidence package under `runtime/replayvla-p1/f2/` containing at least:
   - discovery terminal summary and raw-log SHA256s;
   - exact effective-config bytes (or an exact canonical copy) and SHA256;
   - three worker terminal summaries including argv, return code, selected ordinal, config hash, render metadata, EGL/GL identity, close status, parent exit and stdout/stderr SHA256s;
   - c5 adjudication record;
   - a manifest hashing all compact evidence records and referencing retained raw paths;
   - explicit evidence classifications (`OBSERVED`, `SOURCE_ACCOUNTED`, `TRUSTED_DEPENDENCY`, `NOT_OBSERVED/UNRESOLVED`).
6. Harden adjudication so PASS requires return code zero, exact effective-config hash/ordinal agreement and the required terminal/hash fields.
7. Add bounded timeout/failure terminalization and best-effort close for future F2 harness use. Do not dynamically rerun it under this authorization.
8. Add the missing focused regression tests.
9. Implement a **static re-adjudication command** that reads only the retained c5 records and the pinned repository/source identities. It must not import/construct LIBERO/MuJoCo/EGL for dynamic behavior. It may hash files and inspect source/config/evidence.

## 4. F2H PASS criteria

`F2H = PASS` and formal `F2 / SA-renderer = PASS` may be proposed only if the static re-adjudication can establish all of the following from existing c5 evidence without a new dynamic attempt:

1. the exact executed c5 config/effective-config bytes and hashes are preserved;
2. the authority-field error is explicitly recorded as an erratum rather than silently rewritten;
3. F1T predecessor bytes/trace and required F2 input bytes are bound to expected identities;
4. discovery selected one deterministic ordinal and its terminal/log hashes are preserved;
5. exactly the original three c5 workers are represented; no replacement is introduced;
6. every worker has return code 0, one public render, matching selected ordinal/config hash, valid render metadata, matching EGL/GL identity, close PASS and parent-observed exit;
7. stdout/stderr hashes and terminal summaries are retained;
8. forbidden operations remain zero and unknown/trusted facts are not encoded as observed zero;
9. the c5 execution can be source-accounted to the declared init-state/runtime paths under the approved trusted-OS assumption;
10. focused hardening tests and `git diff --check` pass.

If any required historical field was never recorded and cannot be established without another environment/EGL/render execution, report `F2H = BLOCKED`. Do **not** create c6 and do not infer the missing field from expected behavior.

## 5. Phase state

Until F2H is pushed and reviewed:

- F0 = PASS.
- F1T = PASS.
- F2 dynamic cohort c5 = positive observed result, **formal gate pending F2H**.
- `SA-renderer` = `PENDING_EVIDENCE_HARDENING`.
- F3-F6 = `NOT AUTHORIZED`.
- no new F2 dynamic cohort is authorized.
- old G1 remains `BLOCKED`; old E3-E6/G2 remain paused/not started.

This bounded repair is intended to preserve the successful c5 observation while preventing an evidence-contract shortcut from propagating into null/replay work.
