# ReplayVLA-P1 CURRENT EXECUTION

Status: F3N-S1 STATIC REQUALIFICATION AUTHORIZED
Branch: xyh/replayvla-p1

Primary route authority:
- docs/replayvla-p1/13_f3_native_sa_runtime_pivot.md
- authority commit: 3902dea1879a4bdf3fbb1f76ad60aa46295e8ed8

Latest adjudication:
- docs/replayvla-p1/14_f3n_static_entrypoint_adjudication.md

## Historical blocked evidence

The first F3N static invocation is immutable history:

- evidence commit: 8912b5cc19e2fd780204977a402ccb1dbb9c5633
- evidence: runtime/replayvla-p1/f3n/static_qualification.json
- status: BLOCKED
- reason: ModuleNotFoundError: No module named 'scripts'
- dynamic cohort: NOT STARTED

Do not overwrite or delete this evidence.

The blocker was adjudicated as a Python direct-file entrypoint/package-resolution defect,
not evidence that the trusted installed runtime, renderer, official environment, or null
dynamics failed.

## Validated implementation

Validated implementation commit:

`f2331a203b2a8abaefc57b5ef684706bcdf83a4b`

Runtime validation already completed:

- py_compile: PASS
- tests/test_m1_sa_null_native.py: 35 passed
- tests/test_m1_null_calibration.py: 71 passed in 29.64s
- renderer/F1T regression: 10 passed
- committed-range git diff --check: PASS
- no tracked worktree modifications

Do not repeat those validation suites unless tracked implementation/config files change.

## Goal

Run exactly one bounded static requalification, F3N-S1, using module entry.

If and only if F3N-S1 is PASS, execute the already-authorized single F3N dynamic cohort
in the same unchanged execution commit.

If F3N-S1 is BLOCKED/FAIL, stop. No F3N-S2 is authorized.

## Start

~~~bash
cd /public/home/xuyinghao/workspace/replayvla-p1

GIT_SSH_COMMAND='ssh -F /dev/null -i /public/home/xuyinghao/.ssh/shiftvla_github -o IdentitiesOnly=yes' \
git fetch origin

git checkout xyh/replayvla-p1
git merge --ff-only origin/xyh/replayvla-p1

git rev-parse HEAD
git status --short
~~~

Known pre-existing untracked `runs/replayvla-p1/` is allowed.

If any tracked file is unexpectedly modified, STOP without reset/clean.

Record the current HEAD. It must remain unchanged between F3N-S1 static PASS and
dynamic startup.

## Gate 1: F3N-S1 static qualification

Run exactly once, from repository root, using module entry:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  -m scripts.m1_sa_null_native qualify \
  --config configs/replayvla/p1_sa_null_native.yaml \
  --output runtime/replayvla-p1/f3n/static_qualification_s1.json
~~~

Do NOT use:

`python scripts/m1_sa_null_native.py ...`

Static qualification may inspect/import the pinned CPU runtime and hash frozen assets.
It must not construct the LIBERO environment or create an EGL context.

If F3N-S1 returns BLOCKED/FAIL:

- STOP immediately;
- do not run dynamic;
- do not repair in Codex;
- do not modify implementation/config/packages;
- do not create F3N-S2;
- preserve both static qualification evidence files;
- commit/push only new compact evidence;
- return to reviewer for route-level PIVOT review.

## Gate 2: single F3N dynamic cohort

Only after F3N-S1 PASS, and without committing/changing HEAD in between:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  -m scripts.m1_sa_null_native execute \
  --config configs/replayvla/p1_sa_null_native.yaml \
  --qualification runtime/replayvla-p1/f3n/static_qualification_s1.json
~~~

Exactly one invocation is authorized.

The cohort remains:
- 20 predetermined pairs / 40 attempts;
- fresh official environment per attempt;
- fixed 82x7 float32 action tape;
- unique software-EGL ordinal-8 entry check;
- zero retry/replacement;
- no policy/checkpoint inference;
- no processor calls;
- no replay/capture/restore.

PRE_CONSTRUCTION or UNKNOWN setup/transport failure stops further worker launches.
POST_CONSTRUCTION scientific/trajectory outcomes remain part of the frozen cohort and
do not authorize replacement.

## Scientific criteria

Unchanged:
- action semantic bytes exact;
- contact identity/set exact;
- predicates/terminal/counters/gripper discrete exact;
- grouped empirical floating physics envelopes;
- separate RGB envelopes;
- seven-regime support;
- positive contact/grasp/carried evidence;
- no global epsilon;
- no post-hoc multiplier.

## Forbidden

Do not:
- install/upgrade/downgrade packages;
- create missing third-party source checkouts;
- edit implementation/config/tests;
- alter historical evidence;
- run camera intervention;
- enter physical replay/F3b;
- run another static requalification after F3N-S1;
- run a second dynamic cohort;
- reset/clean unrelated files;
- use git add -A.

## Evidence

Preserve:
- runtime/replayvla-p1/f3n/static_qualification.json

New:
- runtime/replayvla-p1/f3n/static_qualification_s1.json

If dynamic starts, expected additional compact evidence:
- runtime/replayvla-p1/f3n/null_schedule.json
- runtime/replayvla-p1/f3n/pair_registry.json
- runtime/replayvla-p1/f3n/raw_evidence_manifest.json
- runtime/replayvla-p1/f3n/f3n_summary.json

Large raw evidence remains under runs/replayvla-p1/ and is not staged.

## Commit/push after execution stops

~~~bash
git status --short
git diff --check
~~~

Stage only newly generated compact F3N evidence that exists.

Commit a factual PASS/FAIL/BLOCKED evidence message, then push:

~~~bash
GIT_SSH_COMMAND='ssh -F /dev/null -i /public/home/xuyinghao/.ssh/shiftvla_github -o IdentitiesOnly=yes' \
git push origin xyh/replayvla-p1
~~~

## Return to reviewer

Report:
- execution commit SHA;
- evidence commit SHA;
- original static evidence preserved yes/no;
- F3N-S1 PASS/BLOCKED;
- installed dependency identities;
- asset qualification status/file count/verified bytes;
- renderer entry result if dynamic started;
- F3N PASS/FAIL/BLOCKED;
- completed pairs/attempts;
- dynamic worker launch count;
- fail-fast triggered/reason;
- technical failures;
- contact/grasp/carried evidence;
- exact discrete verdict;
- physics envelope summary;
- RGB envelope summary;
- compact evidence paths;
- remote HEAD.

Even if F3N PASS: STOP. Physical replay/F3b remains unauthorized.
