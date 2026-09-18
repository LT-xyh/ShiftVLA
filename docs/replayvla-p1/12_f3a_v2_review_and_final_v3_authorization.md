# F3a-v2 route adjudication and final path-relocated F3a-v3 authorization

**Status: APPROVED FOR ONE FINAL F3a-v3 COHORT ONLY**

Authority branch: `xyh/replayvla-p1`.

Reviewed predecessor: `c3c6d7a871c48fa449b9cc7ee4812d971421801a`.

## 1. Adjudication of F3a-v2

The retained F3a-v2 cohort is `BLOCKED`, not PASS and not a scientific null-calibration FAIL.

All 40 scheduled attempts failed before official environment construction because the frozen legacy state-replay runtime still points `LIBERO_CONFIG_PATH` at:

`/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/m1/libero_config`

which is unavailable in the ReplayVLA-P1 worktree.

Therefore:

- completed scientific trajectories = 0;
- completed duplicate-control attempts = 0;
- exact/discrete null behavior = UNKNOWN;
- physics/RGB envelopes = UNKNOWN;
- contact/grasp/carried coverage = UNKNOWN.

The 40 terminal technical-failure records remain immutable historical evidence. They are not deleted, rewritten, or counted as successful null attempts.

## 2. Why one final v3 cohort is allowed

The failure occurred before environment construction and before any action, render trajectory, null measurement, replay, policy, or scientific state transition.

Repository evidence also shows that F2 successfully used a ReplayVLA-P1-specific LIBERO config rooted in the current worktree. The legacy state-replay config is path-bound to the historical worktree. This is a portability/provenance binding defect rather than evidence that the null experiment itself is unstable.

A final v3 cohort is therefore authorized only if a static relocation proof closes **all** runtime paths before the first dynamic attempt.

There will be no F3a-v4. If v3 encounters another path/provenance/setup defect after dynamic execution begins, the route returns to PIVOT review.

## 3. Immutable history

Do not modify or reinterpret:

- old `configs/m1/state_replay.yaml`;
- F3a v1 BLOCKED evidence;
- F3a v2 BLOCKED evidence and all 40 technical-failure terminal records;
- old/derived registry history;
- F2/F2R history.

The v2 statement `dynamic_scope_exhausted=true` remains true for v2. This document grants a new route-level exception for exactly one v3 cohort because v2 produced zero constructed environments and zero scientific trajectories.

## 4. Static relocation phase (F3a-RP)

Before any environment construction, create a ReplayVLA-P1-specific path-relocated runtime/config derived from the legacy state-replay config.

Allowed changes are only path/provenance binding fields needed to point at byte-identical or content-identical current resources. Scientific values must remain unchanged.

At minimum audit every absolute or repo-relative file/directory that the official CPU factory can read, including:

- LIBERO_CONFIG_PATH and its config tree;
- BDDL;
- init-state asset;
- LIBERO/LeRobot/robosuite/MuJoCo source checkouts;
- libero assets;
- runtime lock;
- artifact manifests if still used;
- any checkpoint/base-model identity path that the policy-free factory reads for configuration identity;
- action tape;
- recovered provenance bundle and derived registry.

Do not assume a path is unused merely because policy calls are disabled; source-trace or actual factory configuration logic must justify omissions.

## 5. Relocation requirements

For each relocated file:

- regular non-symlink file;
- exact expected SHA256 or exact frozen semantic hash as appropriate;
- record old path, new path, size and hash.

For each relocated directory / checkout:

- record actual resolved path;
- verify expected git commit when a git checkout is required;
- verify required files/hashes relevant to the factory;
- do not modify third-party source.

For LIBERO config, the current ReplayVLA-P1 config tree may be reused only if its semantics are shown equivalent to the required official task construction inputs. The known working file is:

`runtime/replayvla-p1/f2_libero_config/config.yaml`

Do not simply replace a path string without comparing the config content/meaning.

## 6. Derived state-replay runtime config

Create a new file, e.g.:

`configs/replayvla/p1_state_runtime_v3.yaml`

It must be mechanically derived from `configs/m1/state_replay.yaml`.

Generate a derivation manifest listing every changed JSON/YAML pointer.

Allowed changes:

- path locations;
- authority/execution/provenance metadata required by the new route;
- self-hash values consequent to those changes.

Forbidden changes include task, seed, observation type, controller/gripper semantics, renderer identity/ordinal, action/state semantics, tolerances/oracles, source checkout commits, policy-free flags, replay/null scientific definitions, or any numeric scientific parameter.

If semantic equivalence cannot be established, F3a-RP = BLOCKED.

## 7. Mandatory path preflight

Implement a pure-static command/test that walks the complete v3 runtime config and validates every required path before schedule execution.

It must fail if any required path is:

- missing;
- symlink when a regular file is required;
- wrong hash;
- wrong checkout commit;
- inconsistent with its declared role.

The first dynamic worker may launch only if this preflight is PASS.

Add regression coverage for the exact v2 failure: a missing `LIBERO_CONFIG_PATH` must be rejected during static preflight, before schedule publication/execution.

## 8. v3 schedule

Only after F3a-RP PASS, freeze a new v3 config/schedule/output root.

Suggested names:

- `configs/replayvla/p1_null_v3.yaml`
- `runtime/replayvla-p1/f3a/null_schedule_v3.json`
- `runs/replayvla-p1/f3a-v3-20260918`

Bind:

- this authority commit;
- v3 relocated runtime config + hash;
- derived registry v2 + hash;
- recovery manifest + hash;
- actual BDDL/init-state/action-tape hashes;
- CPU interpreter/runtime lock;
- task0/init0/seed2027;
- deterministic EGL software rule and ordinal 8;
- exactly 20 pairs / 40 attempts;
- zero retry/replacement.

## 9. Dynamic F3a-v3 authorization

After all static gates PASS, execute exactly one v3 null cohort.

Each pair remains:

- A fresh official env;
- B fresh official env;
- same exact init-state;
- same exact action tape;
- same runtime/config;
- no policy/checkpoint/processor;
- no replay/capture/restore;
- no replacement or retry.

All terminal outcomes remain published.

## 10. Stop rule

This is the final infrastructure-repair allowance for F3a.

If any dynamic v3 attempt fails because of another missing path, asset binding, construction-precondition error, or other setup defect that should have been caught statically:

- retain all evidence;
- F3a = BLOCKED;
- do not create v4;
- do not repair-and-rerun;
- return to route-level PIVOT review.

A genuine simulator/renderer/null behavior failure also stops the phase normally.

## 11. Scientific PASS criteria remain unchanged

F3a / SA-null can PASS only from the fixed v3 cohort with complete evidence for:

- 20 complete duplicate-control pairs;
- exact discrete invariants;
- grouped physics envelopes;
- grouped renderer/RGB envelopes;
- required contact/grasp/carried and regime support;
- terminal semantics;
- no replacement/retry.

No global epsilon or post-hoc multiplier may be introduced.

## 12. State after this authorization

- F0 = PASS.
- F1T = PASS.
- F2 strict admission = FAIL / NONCONFORMANT.
- renderer behavior = OPERATIONALLY_OBSERVED.
- F3a v1 = BLOCKED.
- F3a provenance recovery = PASS.
- F3a v2 = BLOCKED before environment construction.
- F3a-RP static relocation = AUTHORIZED.
- F3a-v3 dynamic null = CONDITIONALLY AUTHORIZED after F3a-RP PASS.
- no F3a-v4 is authorized.
- F3b-F6 = NOT AUTHORIZED.
