# ReplayVLA-P1 CURRENT EXECUTION

Status: EXECUTION-READY FOR F3N
Branch: xyh/replayvla-p1
Scientific authority: docs/replayvla-p1/13_f3_native_sa_runtime_pivot.md
Authority commit: 3902dea1879a4bdf3fbb1f76ad60aa46295e8ed8

This file is the only operational handoff Codex needs for the next session.
Always fast-forward to the latest origin/xyh/replayvla-p1 containing this file.

## Goal

Run exactly one M1-SA-native null qualification (F3N) using the real CPU runtime.

Codex is the environment executor only. Do not redesign the route, refactor broadly,
change packages, or invent another runtime if this execution fails.

## What Web GPT already prepared

- configs/replayvla/p1_sa_null_native.yaml
- scripts/m1_sa_null_native.py
- native branches in scripts/m1_null_calibration.py
- tests/test_m1_sa_null_native.py
- runtime/replayvla-p1/f3n/source_trace.json
- compact evidence publication
- pre-construction cohort fail-fast

The F3N path does NOT execute configs/m1/state_replay.yaml and does NOT require
external/lerobot, external/robosuite, or external/mujoco source checkouts.

## Allowed real-environment work

- import/identify pinned installed CPU dependencies
- verify actual filesystem inputs
- deterministic EGL software-device discovery
- official policy-free LIBERO environment construction
- fixed-action null trajectories
- render/observation/invariant collection required by the frozen null protocol
- write local raw evidence and compact Git evidence

## Forbidden

- package install/uninstall
- NumPy/Torch/Python changes
- creating missing third-party source checkouts
- policy/checkpoint inference
- processor construction/calls
- replay capture/restore
- camera intervention
- F3b physical replay
- replacement/retry cohort
- F3a-v4/v5
- git add -A
- cleanup/reset of unrelated dirty/untracked user files

## Start

Use the dedicated SSH command already established for this repository.

~~~bash
cd /public/home/xuyinghao/workspace/replayvla-p1

GIT_SSH_COMMAND='ssh -F /dev/null -i /public/home/xuyinghao/.ssh/shiftvla_github -o IdentitiesOnly=yes' \
git fetch origin

git merge --ff-only origin/xyh/replayvla-p1
git status --short
git rev-parse HEAD
~~~

Do not proceed if the branch cannot be fast-forwarded or tracked files are unexpectedly modified.

## Gate 1: focused static tests

Run:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest -q \
  tests/test_m1_sa_null_native.py \
  tests/test_m1_sa_renderer.py \
  tests/test_m1_sa_f1t.py
~~~

Then run the existing null suite with a normal cluster timeout, not the previous 30-second wrapper:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -m pytest -q \
  tests/test_m1_null_calibration.py
~~~

Also:

~~~bash
git diff --check
~~~

If any required test fails: STOP. Do not repair in Codex. Report the exact failure to Web GPT.

## Gate 2: F3N static qualification

Run exactly once:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  scripts/m1_sa_null_native.py qualify \
  --config configs/replayvla/p1_sa_null_native.yaml \
  --output runtime/replayvla-p1/f3n/static_qualification.json
~~~

PASS requires:
- current CPU interpreter and runtime lock
- installed dependency versions/origins
- installed modules under the isolated purelib
- pinned LIBERO checkout
- exact ReplayVLA-P1 LIBERO config bytes
- exact BDDL/init-state
- exact derived registry
- exact action semantic bytes
- fixed assets
- no legacy state_replay execution parent

If qualification BLOCKS/FAILS: STOP and push the compact static evidence only.
Do not construct an environment.

## Gate 3: execute one F3N cohort

Only after Gate 2 PASS:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  scripts/m1_sa_null_native.py execute \
  --config configs/replayvla/p1_sa_null_native.yaml \
  --qualification runtime/replayvla-p1/f3n/static_qualification.json
~~~

The runner will:
1. freeze the 20-pair / 40-attempt schedule;
2. publish compact schedule copies;
3. perform one deterministic renderer entry check;
4. execute the existing null oracle through the native runtime branch;
5. stop launching workers after the first pre-construction setup failure;
6. retain post-construction scientific/technical outcomes normally;
7. publish runtime/replayvla-p1/f3n/f3n_summary.json.

No rerun is authorized.

## Evidence expected in Git

Depending on where execution stops:

- runtime/replayvla-p1/f3n/static_qualification.json
- runtime/replayvla-p1/f3n/null_schedule.json
- runtime/replayvla-p1/f3n/pair_registry.json
- runtime/replayvla-p1/f3n/raw_evidence_manifest.json
- runtime/replayvla-p1/f3n/f3n_summary.json

Large raw attempt/trajectory evidence stays under runs/replayvla-p1/ and is not committed.
Do not delete it.

## Commit/push

Inspect first:

~~~bash
git status --short
git diff --check
~~~

Stage only generated F3N compact evidence that exists. Omit nonexistent files instead of creating placeholders.

Then commit with a factual message reflecting PASS/FAIL/BLOCKED, and push with:

~~~bash
GIT_SSH_COMMAND='ssh -F /dev/null -i /public/home/xuyinghao/.ssh/shiftvla_github -o IdentitiesOnly=yes' \
git push origin xyh/replayvla-p1
~~~

## Return to Web GPT

Report only:

- execution commit SHA before qualification
- evidence commit SHA
- static tests
- F3N-static PASS/BLOCKED
- installed dependency identities
- renderer entry result
- F3N PASS/FAIL/BLOCKED
- completed pairs / attempts
- dynamic worker launch count
- fail-fast triggered yes/no
- technical failures
- contact/grasp/carried coverage
- exact discrete verdict
- main physics envelopes
- RGB envelope summary
- compact evidence paths
- remote HEAD

Do not enter F3b even if F3N PASS.
