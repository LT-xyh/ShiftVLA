# ReplayVLA-P1 CURRENT EXECUTION

Status: DIRECT-BUILDER MICROVALIDATION PAUSED / M0-NATIVE FEASIBILITY ONLY
Branch: xyh/replayvla-p1

Latest authority:
- docs/replayvla-p1/20_prefix_runtime_route_pivot.md

Scientific route authority:
- docs/replayvla-p1/16_prefix_reexecution_paper_route.md

Historical microvalidation evidence:
- first evidence commit: 33f0e2cb1f699254f92fb7c46df93ec653b8f864
- MV-R1 evidence commit: 5e8e63a104909ff299672e509f3f2567f1e2a1c1
- focused pytest in MV-R1: PASS (51 passed)
- MV-R1: BLOCKED before MV-P2-A
- blocker: missing installed LIBERO package-local asset path
- MV-P2-A: NOT_RUN
- MV-P2-B: NOT_RUN
- MV-P1: NOT_RUN
- model/DCU queries: 0
- scientific rollout: false
- retry/replacement: 0/0

Historical compact evidence files are immutable.

## Adjudication

The direct-builder P1 microvalidation path is paused.

Do not create MV-R2.
Do not manually create or repair the package-local asset symlink.
Do not modify installed packages/runtime to rerun the same harness.

The observed blocker is the same package-local LIBERO asset-binding defect previously seen in F3N.

The frozen runtime lock records the expected historical binding:

`libero/libero/assets -> assets_path`

The existing ordinary M0 phase runner owns a setup seam for that binding before its normal execution.

The custom P1 harness bypassed that phase-level setup by calling low-level environment builders directly.

## Scientific status

Prefix reexecution / matched-history closed-loop persistence:

`CONDITIONAL GO`

No real camera/switch/noise microvalidation has yet been observed.

This is not a scientific failure.

## Current authorized work

Repo-only static feasibility for one remaining live-LIBERO route:

**M0-native prefix execution integration**

The route must reuse the already-existing ordinary M0 phase/child lifecycle end-to-end.

Static work must determine whether:
- existing M0 preflight owns all environment/bootstrap setup;
- ObservationIndexedCameraWrapper can be inserted after ordinary env construction;
- paired explicit-noise select_action can be inserted without changing processor/worker semantics;
- fresh CC/CS/SC/SS executions can be scheduled without a new runtime stack.

## Hard scope boundary

A viable candidate must not require:
- new environment bootstrap;
- new asset-binding code;
- manual symlink management;
- package mutation;
- alternate LIBERO config/runtime;
- new renderer discovery;
- another provenance framework;
- exact-state infrastructure.

If a thin M0-native integration is not statically supportable, live-LIBERO prefix reexecution is paused for Paper-1.

## Current permissions

AUTHORIZED:
- repo/source analysis;
- M0 parent/child call tracing;
- repo-only candidate design;
- fake/unit/static tests if needed to prove insertion seams.

NOT AUTHORIZED:
- Codex runtime;
- real LIBERO construction;
- EGL/render/env.step;
- policy inference;
- asset/runtime repair;
- another microvalidation;
- 40-rollout pilot;
- paper estimands.

## Pilot

40-rollout pilot remains NOT AUTHORIZED.

## Next reviewer gate

Return one minimal M0-native feasibility candidate for independent review.

If it is not a thin extension of the already-working M0 path, stop live-LIBERO infrastructure work for Paper-1 and return to research-direction selection.
