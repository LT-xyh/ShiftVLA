# ReplayVLA-P1 CURRENT EXECUTION

Status: MV-R1 RUNTIME REQUALIFICATION AUTHORIZED EXACTLY ONCE
Branch: xyh/replayvla-p1

Latest authority:
- docs/replayvla-p1/19_prefix_microvalidation_r1_authorization.md

Scientific route authority:
- docs/replayvla-p1/16_prefix_reexecution_paper_route.md

Correction authority:
- docs/replayvla-p1/18_prefix_microvalidation_provenance_correction.md

Reviewed correction implementation:
- 65d2ee3c5fbe58381039c06ed24f2d73ffcc320c

Historical microvalidation evidence:
- execution HEAD: 32cbe471c4cc9987b64fbff2ac4c98717d664721
- evidence commit: 33f0e2cb1f699254f92fb7c46df93ec653b8f864
- result: BLOCKED before MV-P2-A
- reason: old repo-relative BDDL/init source-location guard
- focused pytest: PASS (40 passed)
- historical compact evidence is immutable

## Reviewed MV-R1 correction

Historical callers remain unchanged by default.

Without explicit content pins:
- dcu_preflight retains its original current-worktree hf-libero ancestry check.

P1 microvalidation explicitly opts into:
- runtime-resolved BDDL/init paths;
- strict readable regular files;
- SHA256 of exact consumed bytes;
- frozen task identity;
- frozen BDDL/init hashes.

Frozen identities:

BDDL:
`9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b`

Init state:
`cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`

## MV-R1 permission

One new non-scientific microvalidation invocation is authorized.

Fixed sequence:

1. focused pytest;
2. MV-P2-A;
3. MV-P2-B;
4. MV-P1;
5. compact evidence;
6. STOP.

The 40-rollout pilot remains NOT AUTHORIZED.

## Evidence identity

Do NOT modify:

`runtime/replayvla-p1/prefix_reexecution_microvalidation.json`

MV-R1 output:

`runtime/replayvla-p1/prefix_reexecution_microvalidation_r1.json`

Transient work:

`runs/replayvla-p1/prefix_reexecution_microvalidation_r1.work`

Both MV-R1 paths must be absent before execution.

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

Unexpected tracked modifications => STOP.

Do not reset/clean.

Before harness:

~~~bash
test -f runtime/replayvla-p1/prefix_reexecution_microvalidation.json
test ! -e runtime/replayvla-p1/prefix_reexecution_microvalidation_r1.json
test ! -e runs/replayvla-p1/prefix_reexecution_microvalidation_r1.work
~~~

Historical evidence must exist; new paths must not exist.

## Execute exactly once

~~~bash
/public/home/xuyinghao/tmp/shiftvla-libero/bin/python -B \
  -m scripts.p1_prefix_reexecution_microvalidate \
  --baseline-config configs/m0/baseline_a.yaml \
  --output runtime/replayvla-p1/prefix_reexecution_microvalidation_r1.json \
  --work-dir runs/replayvla-p1/prefix_reexecution_microvalidation_r1.work \
  --physical-device 1
~~~

Do not separately run pytest before or after the harness.

Once the new compact evidence exists, never invoke the harness again.

## MV-R1 task-source evidence

Every real runtime phase that constructs the environment must use:
- source_check = runtime_consumed_content_sha256;
- runtime-resolved BDDL path and actual SHA;
- runtime-resolved init-state path and actual SHA;
- content pins showing expected SHA/task identity.

Do not redirect/copy/symlink source files.

## Runtime gates

All existing MV-P2-A, MV-P2-B and MV-P1 gates remain unchanged.

If MV-R1 BLOCKS before MV-P2-A because of another inherited path/provenance requirement:
- do not repair again;
- return to route-level review.

If it reaches MV-P2/MV-P1 and blocks:
- preserve observed evidence;
- no retry.

## Evidence closure

After the one harness invocation:

~~~bash
git status --short
git diff --check
cat runtime/replayvla-p1/prefix_reexecution_microvalidation_r1.json
~~~

Stage only:

`runtime/replayvla-p1/prefix_reexecution_microvalidation_r1.json`

Never use `git add -A`.

Commit factual PASS/BLOCKED evidence and push the branch.

## Forbidden

Do not:
- edit implementation/config/tests during runtime validation;
- install/change packages;
- rerun after compact evidence exists;
- run pilot;
- compute paper estimands;
- repair exact-state/F3N/F3b;
- git reset --hard;
- git clean.

## Return

Report:
- branch HEAD before harness;
- implementation commit;
- evidence commit / remote HEAD;
- focused pytest;
- task-source actual paths/hashes and matches;
- MV-P2-A;
- MV-P2-B;
- MV-P1;
- final status;
- final git status.

Pilot remains NOT AUTHORIZED.
