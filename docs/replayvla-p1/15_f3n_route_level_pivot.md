# F3N route-level adjudication: pause exact-state infrastructure for paper 1

**Status: ROUTE PIVOT / EXACT-STATE INFRASTRUCTURE PAUSED FOR PAPER 1.**

Authority branch: `xyh/replayvla-p1`.

Predecessor authorities:
- `docs/replayvla-p1/13_f3_native_sa_runtime_pivot.md`
- `docs/replayvla-p1/14_f3n_static_entrypoint_adjudication.md`

Reviewed evidence commit:

`48f3d60f5fa4ba5c42e4ed886371b85cb5715833`

Reviewed execution commit:

`15a2176ac0ea08884a48d135b1b8fba519d52c75`

## Evidence judgment

F3N-S1 static qualification is retained as **PASS** for the checks it actually performed.

It established, among other items:
- pinned CPU interpreter/runtime lock;
- installed dependency identities;
- pinned LIBERO checkout and config;
- exact BDDL/init-state;
- exact action semantic bytes and registry identity;
- frozen LIBERO asset manifest identity;
- 586/586 frozen asset files verified;
- 422,320,936 frozen asset bytes verified.

The single authorized F3N dynamic cohort then started.

Renderer entry qualification passed:
- unique software EGL candidate;
- selected ordinal 8.

Exactly one dynamic worker was launched.

That worker failed **PRE_CONSTRUCTION** before an official environment was returned:

`FileNotFoundError: .../site-packages/libero/libero/assets/scenes/libero_tabletop_base_style.xml`

The cohort fail-fast rule triggered:
- planned pairs: 20;
- planned attempts: 40;
- completed pairs: 0;
- completed attempts: 0;
- dynamic worker launches: 1;
- retry: 0;
- replacement: 0.

No scientific null trajectory was obtained. Physics/RGB envelopes and
contact/grasp/carried scientific coverage therefore remain UNKNOWN.

## Root-cause interpretation

This failure is not a scientific null instability result.

It exposes an incomplete runtime-binding closure.

The successful historical CPU runtime lock explicitly recorded:

`site_packages_asset_binding: libero/libero/assets -> assets_path (environment-local reversible symlink)`

The dynamic failure shows that the installed LIBERO execution path still dereferences a
package-local asset path under `site-packages/libero/libero/assets`.

F3N-S1 static qualification verified the external frozen asset tree itself, but did not
verify that this package-local consumption path was bound to the frozen tree.

Therefore the F3N-S1 static PASS must not be interpreted as complete proof that every
asset path actually consumed by official environment construction was closed.

This does not invalidate its successful byte-identity checks. It limits their scope.

## Stop-rule application

Doc13 states that:
- the first pre-construction setup/path defect stops the cohort;
- there is no further F3N infrastructure generation;
- if the direct trusted-install runtime cannot construct the official environment, the
  exact-state ReplayVLA infrastructure route is pivoted/paused for the first paper.

Those conditions are now met.

Accordingly:

- do not create F3N-S2;
- do not create a second F3N dynamic cohort;
- do not add/recreate the package-local asset symlink merely to rerun F3N;
- do not mutate the installed package;
- do not return to F3a-v4/v5;
- do not resume old G1/E3-E6;
- do not enter physical replay or policy replay.

The historical failed and blocked evidence remains immutable.

## Scientific interpretation

Current evidence does **not** show that exact-state ReplayVLA is scientifically false.

It shows that the current exact-state execution infrastructure has exceeded the allowed
engineering budget before obtaining the first complete null dataset.

For the first-paper objective, further work on:
- provenance guards;
- source checkout closure;
- package-local asset plumbing;
- exact restore admission;
- additional runtime generations

has lower expected value than moving to a decomposition that does not require exact
simulator restore.

## Paper-1 pivot

The next paper route should preserve the central scientific question while removing the
exact-state restore dependency.

Preferred next design target:

**prefix reexecution / matched-history failure decomposition**

Core idea:
- construct clean and shifted histories by ordinary forward execution from the same
  frozen initial condition;
- at predetermined switch indices, change only future observation condition;
- compare clean-prefix/clean-future, clean-prefix/shifted-future,
  shifted-prefix/clean-future, and shifted-prefix/shifted-future where feasible through
  forward reexecution rather than simulator state restore;
- treat differences as closed-loop history burden / future-observation effects, without
  claiming exact physical-state branching.

This route may reuse:
- frozen task/init/action/policy identities;
- camera intervention definitions;
- switch-index protocol;
- paired seeds/noise controls;
- task-equal statistical analysis;
- contact/grasp/carried regime annotations where obtainable.

It must not reuse exact-state claims that require restore equivalence.

A new authority/spec is required before any runtime experiment on the pivoted route.

## Current phase state

- ReplayVLA scientific direction: **CONDITIONAL GO**.
- F3N-S1 static: PASS with scope limitation described above.
- F3N dynamic: BLOCKED pre-construction.
- exact-state infrastructure for paper 1: **PAUSED / PIVOTED AWAY**.
- F3b physical replay: NOT AUTHORIZED.
- policy replay on exact-state route: NOT AUTHORIZED.
- next authorized work: **repo-only design of the prefix-reexecution paper route**.
- next Codex runtime execution: NOT AUTHORIZED until that new design is reviewed.
