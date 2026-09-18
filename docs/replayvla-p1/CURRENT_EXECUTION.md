# ReplayVLA-P1 CURRENT EXECUTION

Status: EXACT-STATE PAPER-1 ROUTE PAUSED / PIVOTED
Branch: xyh/replayvla-p1

Latest authority:
- docs/replayvla-p1/15_f3n_route_level_pivot.md

Reviewed evidence commit:
- 48f3d60f5fa4ba5c42e4ed886371b85cb5715833

## Final F3N state

Historical original F3N static:
- BLOCKED
- preserved at runtime/replayvla-p1/f3n/static_qualification.json
- reason: direct-file Python entrypoint could not resolve repository package

F3N-S1 static:
- PASS
- preserved at runtime/replayvla-p1/f3n/static_qualification_s1.json
- installed dependencies qualified
- frozen asset manifest PASS
- 586/586 asset files verified
- 422,320,936 asset bytes verified

F3N dynamic:
- BLOCKED PRE_CONSTRUCTION
- renderer entry PASS
- unique software EGL ordinal 8
- planned: 20 pairs / 40 attempts
- completed: 0 pairs / 0 attempts
- dynamic worker launches: 1
- retry: 0
- replacement: 0
- fail-fast: triggered

Blocking path:
`/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/libero/libero/assets/scenes/libero_tabletop_base_style.xml`

The historical CPU runtime lock records a package-local asset binding:
`libero/libero/assets -> frozen assets root`.

F3N static verified the frozen external asset bytes but did not close this actual
package-local consumption binding. The dynamic failure exposed that missing runtime
binding before official environment construction returned.

## Adjudication

Do NOT repair the exact-state runtime for paper 1.

Do NOT:
- create F3N-S2;
- create another F3N dynamic cohort;
- add/recreate a site-packages asset symlink for rerun;
- modify installed packages;
- return to F3a-v4/v5;
- return to old G1/E3-E6;
- run physical replay/F3b;
- run policy replay on the exact-state route.

The exact-state infrastructure route is paused for the first-paper objective.

This is not a scientific falsification of ReplayVLA.
It is an engineering-budget stop after the direct runtime failed before the first
complete null trajectory.

## Next authorized work

Repo-only work only.

Design a new paper route that preserves the scientific question while removing the
exact simulator restore dependency.

Preferred target:

**prefix reexecution / matched-history failure decomposition**

High-level target:
- forward execute from the same frozen initial condition;
- produce clean and shifted prefixes by ordinary execution;
- at frozen switch indices, vary only future observation condition;
- construct CC / CS / SC / SS through matched forward reexecution where feasible;
- interpret differences as future-observation effect and accumulated closed-loop
  history burden;
- do not claim exact physical-state branching or exact restore equivalence.

Before any new runtime experiment:
1. write a new scientific/design authority;
2. define branch construction semantics;
3. define paired RNG/noise policy;
4. define switch indices and task split;
5. define success/failure and trajectory-level metrics;
6. define minimum pilot needed to validate the hypothesis;
7. independent reviewer approval.

## Codex state

No Codex runtime work is currently authorized.

Do not spend Codex quota on exact-state infrastructure repair.

Next work should be performed by Web Implementer + Reviewer until the pivoted route is
execution-ready.
