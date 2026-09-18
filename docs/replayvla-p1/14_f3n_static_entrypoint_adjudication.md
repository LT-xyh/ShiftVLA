# F3N static entrypoint adjudication

**Status: APPROVED BOUNDED STATIC REQUALIFICATION.**

Authority branch: `xyh/replayvla-p1`.

Predecessor authority: `docs/replayvla-p1/13_f3_native_sa_runtime_pivot.md`.

Reviewed blocked evidence commit:

`8912b5cc19e2fd780204977a402ccb1dbb9c5633`

Reviewed execution commit:

`ee8aed4b9d40218e320e5de892ee4623828203a9`

## Adjudication

The first F3N static qualification remains historically **BLOCKED**.

Its recorded reason is:

`ModuleNotFoundError: No module named 'scripts'`.

This is not evidence that the M1-SA trusted installed runtime, LIBERO environment,
renderer, or null dynamics failed. No F3N dynamic cohort started and no EGL
discovery, official environment construction, render, or `env.step` occurred.

The blocker is a Python entrypoint/package-resolution defect in the invocation:

`python scripts/m1_sa_null_native.py ...`

When a script under `scripts/` is executed by file path, Python places that
directory, rather than the repository root, at the front of `sys.path`.
The F3N static implementation later imports repository modules using the
package form `scripts.*`, so the direct-file invocation can fail before the
intended repository package import is reachable.

The existing implementation can be entered without changing runtime semantics
by invoking the same checked-in module from the repository root:

`python -m scripts.m1_sa_null_native ...`

This changes only Python package entrypoint resolution. It does not change:
- the CPU interpreter/runtime lock;
- installed dependency identities;
- LIBERO config or checkout;
- BDDL/init-state/assets;
- recovered registry or action tape bytes;
- renderer rule;
- null schedule;
- scientific oracle;
- policy/replay permissions.

Therefore this repair is **not** a new F3N runtime generation and does not
trigger the route-level stop rule by itself.

## Historical evidence

Keep immutable:

`runtime/replayvla-p1/f3n/static_qualification.json`

with status `BLOCKED`.

Do not overwrite, delete, rename, or reinterpret it.

## Authorized requalification

Exactly one static requalification is authorized, named **F3N-S1**.

Use the current authoritative branch HEAD and the same config:

`configs/replayvla/p1_sa_null_native.yaml`

Invoke only by module entry:

`/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B -m scripts.m1_sa_null_native qualify ...`

Publish to a new immutable path:

`runtime/replayvla-p1/f3n/static_qualification_s1.json`

No implementation/config/package changes are authorized before this attempt.

## Dynamic authorization

If and only if F3N-S1 static qualification is `PASS`, the single F3N dynamic
cohort already authorized by doc13 remains authorized in the same unchanged
execution commit.

Run it using module entry and the F3N-S1 qualification file.

Do not commit the PASS static evidence before dynamic execution, because the
qualified execution commit must remain identical through dynamic startup.

The dynamic cohort remains exactly:
- 20 predetermined pairs / 40 attempts;
- fresh official environment per attempt;
- fixed action tape;
- zero retry/replacement;
- no policy/processor/checkpoint inference;
- no replay/capture/restore.

## Stop rule

If F3N-S1 static qualification is again `BLOCKED` or `FAIL` for any reason:

- do not repair it in Codex;
- do not create F3N-S2;
- do not change the entrypoint again;
- do not run dynamic;
- return to route-level PIVOT review under doc13.

If F3N-S1 PASSes but the authorized dynamic cohort cannot construct the
official environment or cannot produce a complete interpretable null dataset,
the doc13 route-level stop rule applies.

## Phase state

- original F3N static: BLOCKED, immutable history;
- F3N-S1 static: AUTHORIZED exactly once;
- F3N dynamic: CONDITIONALLY AUTHORIZED only after F3N-S1 PASS;
- physical replay / policy replay / F3b: NOT AUTHORIZED.
