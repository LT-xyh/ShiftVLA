# ReplayVLA-P1 CURRENT EXECUTION

Status: PREFIX-REEXECUTION MICROVALIDATION AUTHORIZED EXACTLY ONCE
Branch: xyh/replayvla-p1

Latest authority:
- docs/replayvla-p1/17_prefix_reexecution_microvalidation_authorization.md

Reviewed implementation commit:
- 3669a1b7c2bd05438b40351bc4cf7e066286a6f8

Scientific route authority:
- docs/replayvla-p1/16_prefix_reexecution_paper_route.md

Historical exact-state route:
- paused for Paper-1;
- F3N/F3b/exact-state repair remain unauthorized.

## Current permission

One non-scientific runtime microvalidation is authorized.

The sequence is fixed:

1. focused pytest;
2. MV-P2-A same-current-state camera intervention;
3. MV-P2-B switch-index lifecycle;
4. MV-P1 one-query real DCU explicit-noise select_action;
5. compact evidence;
6. STOP.

The 40-rollout pilot is NOT authorized.

## Execution identity

Implementation under validation:

`3669a1b7c2bd05438b40351bc4cf7e066286a6f8`

The branch may contain later docs-only authorization commits.
Do not modify tracked implementation/config/tests before or during execution.

## Harness

Entry:

`scripts/p1_prefix_reexecution_microvalidate.py`

Compact evidence:

`runtime/replayvla-p1/prefix_reexecution_microvalidation.json`

Recommended transient work directory:

`runs/replayvla-p1/prefix_reexecution_microvalidation.work`

The compact evidence path is no-overwrite and is absent from the reviewed repository state.

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

Existing unrelated untracked `runs/replayvla-p1/` is allowed.

If there are unexpected tracked modifications, STOP without reset/clean.

Before invoking the harness, verify:

~~~bash
test ! -e runtime/replayvla-p1/prefix_reexecution_microvalidation.json
test ! -e runs/replayvla-p1/prefix_reexecution_microvalidation.work
~~~

If either exists, STOP and do not delete it.

## Execute exactly once

Use the frozen CPU interpreter:

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  -m scripts.p1_prefix_reexecution_microvalidate \
  --baseline-config configs/m0/baseline_a.yaml \
  --output runtime/replayvla-p1/prefix_reexecution_microvalidation.json \
  --work-dir runs/replayvla-p1/prefix_reexecution_microvalidation.work \
  --physical-device 1
~~~

Do not separately rerun focused pytest: the harness owns the gate and records it in the terminal evidence.

Ordinary shell/cwd invocation mistakes may be corrected only if the harness has not created compact evidence.

Once terminal compact evidence exists, do not rerun.

## Fixed runtime gates

Focused pytest must PASS before any real runtime stage.

MV-P2-A must prove:
- 360x360x3 uint8 agentview;
- clean image SHA != shifted image SHA;
- restored-clean SHA == clean SHA;
- qpos/qvel/ctrl/sim time unchanged during camera-only mutation;
- camera position/FOV unchanged;
- clean quaternion exactly restored;
- no policy/model query.

MV-P2-B must use a fresh CPU env and:
- arm CS;
- t_switch 50;
- one normal reset;
- 50 wrapper dummy-action steps;
- obs_0..obs_49 clean;
- action_49 requests shifted camera for obs_50;
- final evidence observation_index 50 / preceding_action_index 49 / shifted;
- policy/model queries 0;
- early terminal => BLOCKED.

MV-P1 runs only after both P2 gates PASS and must:
- use physical device 1;
- create official real feature batch;
- use paired explicit flow noise for task0/init0/seed2027/query0;
- execute exactly one FeatureOnlyRemotePolicy.select_action(features, noise=noise);
- reach official SmolVLAPolicy.select_action(batch, noise=noise);
- queue before=0, after=0, new_chunk_generated=true;
- n_action_steps=1;
- no env.step.

## Evidence handling

Regardless of PASS/BLOCKED:

~~~bash
git status --short
git diff --check
~~~

Do not stage transient work directory.

Stage only:

`runtime/replayvla-p1/prefix_reexecution_microvalidation.json`

if it exists.

Do not use `git add -A`.

Commit a factual evidence-only message and push branch.

If the harness itself fails before compact evidence can be produced because of a filesystem-level condition,
report it and do not manufacture evidence.

## Forbidden

Do not:
- modify implementation/config/tests;
- install/change packages;
- repair runtime;
- retry the harness after terminal evidence;
- run a second microvalidation;
- run the 40-rollout pilot;
- compute paper estimands;
- run F3N/F3b/exact-state work;
- git reset --hard;
- git clean;
- git add -A.

## Return to reviewer

Report:
- execution branch HEAD before harness;
- implementation commit under validation;
- evidence commit SHA;
- remote HEAD;
- harness final PASS/BLOCKED;
- focused pytest result;
- MV-P2-A result and image SHA relations;
- MV-P2-A physics invariance;
- MV-P2-B result, reset count, wrapper step count, obs50 camera evidence, terminal-before-switch;
- MV-P1 result;
- worker/device identity;
- explicit-noise presence;
- queue before/after/new_chunk_generated;
- policy/model query counts;
- compact evidence path;
- final git status.

Even on PASS: STOP.
Pilot remains NOT AUTHORIZED.
