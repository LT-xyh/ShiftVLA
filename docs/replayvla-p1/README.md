# ReplayVLA-P1

**Status: APPROVED EXECUTION ROUTE**

Approved by the user on 2026-09-15. This route starts from repository baseline `e37dfa93dfcc5793340829f8c685a4f5ec8c172b` and is executed on branch `xyh/replayvla-p1`.

## Decision

ReplayVLA remains **CONDITIONAL GO**. The old operation-guard route is **PAUSED** as the default path to the first paper.

Historical state is preserved:

- E2 = `COVERED / PASS (E2-local)`.
- Old G1 = `BLOCKED`.
- Old E3-E6 workflow = `NOT STARTED`; known predicates remain unresolved where previously recorded.
- Old G2 = `NOT AUTHORIZED`.
- Old Renderer Preflight / M1-N0 results are not rewritten by this route.

A PASS under ReplayVLA-P1 / M1-SA-v1 never retroactively means that old G1, old Renderer Preflight, or old M1-N0 passed.

## Goal

Use a bounded scientific-admission contract to obtain trustworthy dynamic replay evidence quickly, while preserving exact-state scientific requirements that matter for the paper.

The investment gates are:

1. **D3 runtime gate**: identify the concrete runtime problem and freeze a qualified candidate. No blind dependency churn.
2. **D10 replay gate**: obtain a real physical replay judgment and a minimal policy-state replay judgment. Physical replay alone is insufficient for paper branching.
3. **Week-3 science gate**: only scale if the question, controls, precision and cost support a meaningful held-out study.

Target: a submission-ready first draft in roughly 6 weeks, with weeks 7-8 only for bounded evidence/writing buffer. This is a project target, not a publication guarantee.

## Authority files

- `01_scientific_admission_contract.md` — M1-SA-v1 trust boundary, retained obligations and phase permissions.
- `02_execution_plan.md` — D1-D10 and 6-week execution plan.
- `03_paper_protocol.md` — paper question, four-branch design, estimands, controls and minimum experiment package.
- `04_codex_handoff.md` — execution instructions for Codex.

## Collaboration model

GitHub is the source of truth. Codex performs local/runtime work and pushes commits to this branch. Web GPT reviews repository commits and makes route/gate decisions. Do not depend on hidden chat context when repository evidence exists.

Routine work may proceed without per-checkpoint user approval as long as the current phase permission and preceding real PASS are satisfied. Stop and escalate only when a kill gate fires, scientific definitions/trust boundaries must change, new runtime permissions are needed, evidence is contaminated, or the planned budget is exceeded.
