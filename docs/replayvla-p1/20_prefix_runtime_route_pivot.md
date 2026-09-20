# Prefix reexecution route-level runtime pivot after MV-R1

**Status: DIRECT-BUILDER MICROVALIDATION PAUSED / M0-NATIVE INTEGRATION FEASIBILITY ONLY.**

Branch: `xyh/replayvla-p1`

Scientific route authority:
- `docs/replayvla-p1/16_prefix_reexecution_paper_route.md`

Historical microvalidation authorities:
- `docs/replayvla-p1/17_prefix_reexecution_microvalidation_authorization.md`
- `docs/replayvla-p1/19_prefix_microvalidation_r1_authorization.md`

Historical evidence:
- first microvalidation evidence commit: `33f0e2cb1f699254f92fb7c46df93ec653b8f864`
- MV-R1 evidence commit: `5e8e63a104909ff299672e509f3f2567f1e2a1c1`

## 1. Adjudication

MV-R1 is retained as BLOCKED.

Focused tests PASSed:

`51 passed`

The runtime then blocked before MV-P2-A with:

`FileNotFoundError: /public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/libero/libero/assets/scenes/libero_tabletop_base_style.xml`

No MV-P2-A, MV-P2-B, MV-P1, DCU model query, scientific rollout, retry, replacement, or
paper-estimand computation occurred.

This is not evidence against the prefix-reexecution scientific hypothesis.

## 2. Same infrastructure defect as F3N

The missing package-local LIBERO asset path is the same runtime-binding defect previously observed
in the F3N exact-state route.

The frozen runtime lock explicitly records:

`site_packages_asset_binding: libero/libero/assets -> assets_path (environment-local reversible symlink)`

The existing M0/dcu_preflight runtime owns a historical setup seam:

`_verify_assets_binding(..., create=True)`

which delegates to the existing safe asset-binding helper before normal phase execution.

The custom P1 microvalidation instead called `build_cpu_environment_runtime()` directly and therefore
bypassed that normal phase-level setup.

## 3. Stop rule

Do NOT create MV-R2 by adding a symlink/setup call to the current direct-builder microvalidation.

Do NOT:
- repair the installed environment by hand;
- create the package-local symlink manually;
- call the existing asset-binding helper merely to rerun the same harness;
- patch LIBERO packages;
- create another direct-builder microvalidation;
- mutate or overwrite either historical compact evidence file.

The direct-builder microvalidation route has consumed its engineering budget.

## 4. Scientific status

Prefix reexecution / matched-history closed-loop persistence remains scientifically plausible.

Not yet observed:
- real CameraModder intervention;
- real switch-index lifecycle;
- real explicit-noise select_action on the P1 route;
- any paper signal.

Therefore scientific route status remains:

`CONDITIONAL GO`

not PASS and not scientific FAIL.

## 5. One remaining live-LIBERO route candidate

The only remaining live-LIBERO Paper-1 route that may be considered is:

**M0-native prefix execution integration**

This is a route pivot, not MV-R2.

It must reuse the historical ordinary M0 execution lifecycle end-to-end, including its existing
preflight/setup ownership, rather than calling low-level environment builders directly.

The goal is to insert only the already-reviewed P1 scientific seams into a runtime path that has
previously demonstrated ordinary reset/render/policy execution.

## 6. Repo-only feasibility questions

Before any implementation or runtime authorization, statically establish:

1. the exact M0 parent/child call path that runs phase preflight with existing asset binding setup;
2. the exact point after ordinary environment construction where the single LeRobot LIBERO sub-env is
   available;
3. whether `ObservationIndexedCameraWrapper` can be inserted there without changing M0 preflight,
   environment construction, package paths, reset counts, or worker lifecycle;
4. whether the existing paired-noise `FeatureOnlyRemotePolicy.select_action(features, noise=...)`
   seam can replace only the action query while preserving the M0 processor order and worker boundary;
5. whether four fresh full rollouts can be scheduled using existing M0 child/runtime ownership without
   copying F3N/direct-builder infrastructure.

## 7. Admission budget for the M0-native route

A candidate is admissible only if the static design shows a thin extension of the existing M0 runner.

Preferred implementation budget:
- one small P1 runner/orchestrator;
- focused extension points in existing M0 child execution if unavoidable;
- existing camera/noise helpers;
- tests.

It must not require:
- a new environment bootstrap stack;
- a new asset-binding implementation;
- package mutation;
- manual symlink management;
- alternate LIBERO config layout;
- new renderer discovery;
- another provenance framework;
- exact-state machinery.

If any of those are required, live-LIBERO prefix reexecution is PAUSED for Paper-1.

## 8. No runtime yet

Current authorized work is repo-only static feasibility/design.

Do not:
- start Codex runtime validation;
- construct LIBERO;
- render;
- call env.step;
- run policy inference;
- repair runtime;
- run pilot.

## 9. Pilot remains unauthorized

The frozen 40-rollout pilot remains NOT AUTHORIZED.

No CC/CS/SC/SS scientific rollout may begin until:
- an M0-native candidate is independently reviewed;
- one bounded real qualification of that route is explicitly authorized and PASSes.

## 10. Paper-1 stop criterion

If the M0-native static feasibility review shows that P1 cannot be integrated as a thin extension of
the already-working ordinary M0 path, stop live-LIBERO infrastructure work for Paper-1.

Do not create a third runtime route.

Return instead to research-direction selection optimized for the shortest credible path to the
publication target.
