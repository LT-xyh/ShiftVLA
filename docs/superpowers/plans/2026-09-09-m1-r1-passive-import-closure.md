# R1 passive-import closure implementation plan

> Execute with subagent-driven development, test-first. Main-agent review is
> authorized if independent reviewers are unavailable; do not restart them
> indefinitely. All execution subagents use GPT-5.6-luna / max, not speed.

**Goal:** Freeze and enforce the user-approved exact passive processor-module
import closure, then finish the policy-free renderer-preflight entrypoint.

**Architecture:** A dedicated import guard enforces a self-hashed, source-bound
exact module closure and records forbidden-operation attempts separately from
passive imports. The existing admission script consumes this contract and
owns the three fresh EGL probes and one separate single-render LIBERO worker.

**Stack:** Pinned Python 3.12 CPU environment, standard-library instrumentation,
existing YAML/JSON contracts, pytest fakes and isolated import-only checks.

## 1. Audit the official dependency graph

- [ ] Read installed `lerobot.envs` / `configs` / `factory` and each eagerly
  reachable processor-related source. Record exact module names, dependency
  edges, paths and SHA-256; distinguish declarations from instances/calls.
- [ ] Flag unconditional prohibited effects before any dynamic audit.
- [ ] Keep source-derived candidates distinct from the frozen runtime set.
  Import-only discovery must reject unknown related modules, not grant them.

## 2. Implement and test the guard

Files: create `scripts/m1_import_guard.py` and `tests/test_m1_import_guard.py`.

- [ ] Write failing tests for exact set/schema/self-hash validation, wildcard
  rejection, path/source drift, unexpected related import, preloaded modules,
  and duplicate/missing expected modules.
- [ ] Write failing tests for processor construction (including inherited and
  dataclass constructors), pipelines, `__call__`, processing functions, policy
  imports/calls, model/tokenizer construction, checkpoint loads and Hub/network
  attempts. Catching a blocked exception must not erase the failed evidence.
- [ ] Implement a scoped guard with explicit installation/cleanup evidence.
  Permit only audited passive module/class declarations and registration
  decorators; do not count declarations as processor instances or operations.
- [ ] Tests must prove an unknown module is rejected before its body executes,
  and that guards exist before processor imports, not only after them.
- [ ] Run the focused tests using the pinned interpreter; observe RED then GREEN.

## 3. Verify and freeze the closure

Files: create `runtime/m1/processor_import_closure_r1.json`; update
`configs/m1/renderer_preflight_r1_egl0.yaml`, admission validation/tests and docs.

- [ ] Run a guarded import-only check of the audited official factory import
  seam, without calling environment construction/render/reset/step, and without
  creating an EGL context. Persist no experimental evidence.
- [ ] Source-review any blocked new module before revising the draft candidate
  contract. Never automatically allow a module because a probe encountered it.
- [ ] Freeze only the exact observed and audited related-module set and source
  hashes. Require exact equality, not a superset match, in subsequent checks.
- [ ] Bind the closure hash into the R1 admission config and recompute its
  canonical self hash; preserve legacy configs and the predecessor bundle.
- [ ] Main agent reviews the diff, tests and positive/negative guard coverage;
  commit only scoped implementation, tests, contract and documentation files.

## 4. Complete the original admission implementation

Continue original plan Task 2 metadata collectors and three-child launcher,
then Task 3 instrumented LIBERO preflight, using the amended import guard.
The original construction/render/cleanup restrictions remain unchanged.
The new worker must install the guard before official imports, preserve all
blocked-attempt evidence through close, and never reuse its process or frame.

- [ ] Complete fake-only metadata, subprocess, operation-count and cleanup tests.
- [ ] Integrate exact import closure and zero-use evidence into worker and
  terminal schema validation; reject missing, malformed or drifted evidence.
- [ ] Re-run admission/null regression tests, compile checks, source/hash audits,
  import-negative tests and `git diff --check`; main agent reviews final diff.
- [ ] Commit completed implementation with explicit provenance. Report actual
  implementation/review status without claiming empirical admission.

**Stop boundary for this amendment:** Do not run real renderer preflight or
create a new M1-N0 schedule before implementation is complete and reviewed.
No replay, ReplayState changes, policy execution, or later M1/M2 experiments.
