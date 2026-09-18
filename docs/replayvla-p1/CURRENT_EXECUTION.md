# ReplayVLA-P1 CURRENT EXECUTION

Status: F3N EXECUTION AUTHORIZED
Branch: xyh/replayvla-p1
Scientific authority: docs/replayvla-p1/13_f3_native_sa_runtime_pivot.md
Authority commit: 3902dea1879a4bdf3fbb1f76ad60aa46295e8ed8

## Validated implementation

Validated implementation commit:

`f2331a203b2a8abaefc57b5ef684706bcdf83a4b`

Runtime validation on the real repository/runtime completed with:

- Python compile: PASS
- `tests/test_m1_sa_null_native.py`: 35 passed
- `tests/test_m1_null_calibration.py`: 71 passed in 29.64s
- renderer/F1T regression: 10 passed
- committed-range `git diff --check`: PASS
- no tracked worktree modifications
- pre-existing untracked `runs/replayvla-p1/` preserved

The validated implementation was fast-forwarded into `xyh/replayvla-p1`.
This handoff update is documentation-only and does not alter the validated runtime code.

Do not rerun the validation suites unless a tracked implementation/config file changes.

## Goal

Execute the single authorized M1-SA-native null qualification (F3N).

Sequence:

1. synchronize to current `origin/xyh/replayvla-p1`;
2. verify no unexpected tracked modifications;
3. run F3N static qualification exactly once;
4. if and only if static qualification is PASS, run the single F3N dynamic cohort;
5. publish/commit compact evidence;
6. stop. F3b remains unauthorized.

## Immutable route rules

F3N does NOT execute `configs/m1/state_replay.yaml`.

Missing source checkouts for installed:
- LeRobot
- robosuite
- MuJoCo

are not gates. These installed packages are `TRUSTED_DEPENDENCY` with
source-to-binary provenance `NOT_INDEPENDENTLY_ATTESTED`.

The F3N static qualifier DOES bind:
- exact CPU interpreter/runtime lock;
- installed dependency versions/origins;
- pinned LIBERO checkout;
- exact ReplayVLA-P1 LIBERO config;
- exact BDDL/init-state;
- exact recovered registry/action semantic bytes;
- exact frozen LIBERO asset manifest and all frozen asset file bytes.

## Forbidden

Do not:
- modify/install/uninstall packages;
- create missing third-party source checkouts;
- modify implementation/config/tests;
- use policy/checkpoint inference;
- construct/call processors;
- use replay capture/restore;
- run camera intervention;
- enter F3b;
- retry or replace failed F3N attempts;
- create another F3N runtime/cohort;
- reset/clean unrelated user files;
- use `git add -A`.

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

Record the current HEAD. That HEAD is the F3N execution commit and must not change between static qualification and dynamic execution.

## Gate 1: F3N static qualification

Run exactly once:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  scripts/m1_sa_null_native.py qualify \
  --config configs/replayvla/p1_sa_null_native.yaml \
  --output runtime/replayvla-p1/f3n/static_qualification.json
~~~

This is allowed to inspect/import the pinned installed CPU runtime and hash frozen assets.
It must not construct the LIBERO environment or create an EGL context.

If static qualification returns BLOCKED/FAIL:

- STOP;
- do not run `execute`;
- do not repair in Codex;
- preserve the generated compact evidence;
- commit only the generated compact F3N evidence;
- push and return to Web GPT.

## Gate 2: single F3N cohort

Only if Gate 1 is PASS, execute exactly once:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  scripts/m1_sa_null_native.py execute \
  --config configs/replayvla/p1_sa_null_native.yaml \
  --qualification runtime/replayvla-p1/f3n/static_qualification.json
~~~

The runner owns:
- frozen 20-pair / 40-attempt schedule;
- compact schedule publication;
- unique software-EGL ordinal-8 entry check;
- fresh official environment per attempt;
- fixed 82x7 float32 action tape;
- existing null oracle;
- PRE_CONSTRUCTION / POST_CONSTRUCTION / UNKNOWN failure classification;
- fail-closed cohort stop for PRE_CONSTRUCTION or UNKNOWN setup/transport failures;
- no retry/replacement.

POST_CONSTRUCTION trajectory/scientific failures remain part of the frozen cohort and do not authorize cherry-picking or replacement.

No second invocation is authorized.

## Scientific null criteria

Unchanged:
- action semantic bytes exact;
- contact identities/sets exact;
- predicates/terminal semantics/counters/gripper discrete state exact;
- grouped empirical floating physics envelopes;
- separate RGB envelopes;
- seven-regime support;
- positive contact/grasp/carried evidence;
- no global epsilon;
- no post-hoc multiplier.

F3N PASS requires the complete frozen cohort and all frozen gates.

## Evidence

Compact evidence lives under:

`runtime/replayvla-p1/f3n/`

Expected depending on outcome:
- `static_qualification.json`
- `null_schedule.json`
- `pair_registry.json`
- `raw_evidence_manifest.json`
- `f3n_summary.json`

Large raw artifacts remain under `runs/replayvla-p1/` and must not be deleted or committed unless already part of the compact-evidence design.

## Commit and push

After execution stops:

~~~bash
git status --short
git diff --check
~~~

Stage only newly generated compact F3N evidence files that actually exist.
Do not stage unrelated untracked raw runs.

Commit a factual PASS/FAIL/BLOCKED evidence message, then:

~~~bash
GIT_SSH_COMMAND='ssh -F /dev/null -i /public/home/xuyinghao/.ssh/shiftvla_github -o IdentitiesOnly=yes' \
git push origin xyh/replayvla-p1
~~~

## Return to reviewer

Report:

- execution commit SHA;
- evidence commit SHA;
- F3N-static PASS/BLOCKED;
- installed dependency identities;
- asset qualification status/file count/verified bytes;
- renderer entry result if dynamic started;
- F3N PASS/FAIL/BLOCKED;
- completed pairs/attempts;
- dynamic worker launch count;
- fail-fast triggered and reason;
- technical failures;
- contact/grasp/carried evidence;
- exact discrete verdict;
- physics envelope summary;
- RGB envelope summary;
- compact evidence paths;
- remote HEAD.

Even if F3N PASS: STOP. F3b is not authorized.
