# F3 route pivot: native M1-SA null runtime

**Status: APPROVED ROUTE PIVOT.**

Authority branch: `xyh/replayvla-p1`.

Reviewed predecessor: `3b0717ab68b3b955f257d3872a7be75ef9621b11`.

## Decision

Do not create F3a-v4 and do not continue deriving executable null configs from legacy `configs/m1/state_replay.yaml`.

The F3a-v3 static preflight is retained as `BLOCKED`. Its blocker is not a scientific null failure: it requires source checkouts for LeRobot, robosuite and MuJoCo that are absent in the ReplayVLA-P1 worktree.

That requirement conflicts with M1-SA-v1 section 2.1, which explicitly trusts fixed installed third-party runtime artifacts and does not require complete source-to-binary provenance as a prerequisite for the new route.

Repository F2 evidence already demonstrated successful official policy-free LIBERO construction/render using the installed runtime path. Therefore the next route is a direct M1-SA-native null runtime, not another relocation of the legacy state-replay config.

## Historical state remains immutable

- F2 strict admission remains FAIL / NONCONFORMANT.
- F2 renderer behavior remains operationally observed.
- F3a v1 remains BLOCKED.
- F3a provenance recovery remains PASS.
- F3a v2 remains BLOCKED before environment construction.
- F3a v3 / F3a-RP remains BLOCKED statically.
- no F3a-v4 is authorized.

## New phase

The replacement phase is named **F3N** (M1-SA native null qualification).

F3N is not a retry of v1-v3. It uses the approved M1-SA trust boundary directly.

F3N may use:
- the pinned CPU interpreter/runtime lock;
- installed NumPy, MuJoCo, robosuite, LeRobot and PyOpenGL package/module identities;
- the current pinned LIBERO checkout and fixed task assets;
- the proven ReplayVLA-P1 LIBERO config layout;
- the recovered historical action tape/provenance registry;
- the deterministic software EGL ordinal 8 observed in F2.

F3N must not require missing source checkouts for installed third-party packages merely to establish source provenance. Those packages are `TRUSTED_DEPENDENCY`, with source-to-binary provenance marked `NOT_INDEPENDENTLY_ATTESTED`.

## F3N static runtime qualification

Before any environment construction, create a new direct config, for example:

`configs/replayvla/p1_sa_null_native.yaml`

Do not inherit or validate against legacy `configs/m1/state_replay.yaml`.

Bind only inputs actually required by the policy-free null path:
- contract/authority/execution commit;
- CPU Python exact path/version;
- runtime lock bytes/hash;
- installed module origins/versions for NumPy, MuJoCo, robosuite, LeRobot, PyOpenGL;
- current LIBERO checkout identity;
- ReplayVLA-P1 LIBERO config directory and exact config bytes;
- fixed BDDL/init-state bytes;
- fixed assets revision/path;
- recovered source evidence / derived registry;
- fixed action tape semantic bytes;
- task0/init0/seed2027;
- renderer rule/ordinal/identity;
- 20 pairs / 40 attempts;
- policy/processor/checkpoint calls disabled.

Checkpoint/base-model paths are not null-path requirements unless static source tracing of the actual factory proves they are read by this policy-free construction path.

## Installed dependency identity

For trusted installed dependencies, record:
- module `__file__`;
- package/distribution version;
- relevant distribution metadata/location;
- actual imported module file hashes where practical;
- critical DSO identity where already available;
- classification = `TRUSTED_DEPENDENCY`;
- source-to-binary = `NOT_INDEPENDENTLY_ATTESTED`.

Do not require absent git checkouts for these packages.

## Static source trace

Trace the actual call path used by F2 / `build_cpu_environment_runtime` and identify the exact configuration keys and paths consumed for policy-free environment construction.

The static F3N preflight must reject any required missing path before schedule execution.

It must also prove that no legacy `/public/home/xuyinghao/workspace/vla/ShiftVLA/runtime/m1/libero_config` path remains in the effective F3N runtime environment.

## Dynamic authorization

If F3N static qualification and focused tests PASS, one F3N null cohort is authorized automatically.

Schedule:
- exactly 20 predetermined duplicate-control pairs;
- exactly 40 total attempts;
- fresh official environment per attempt;
- same init state and fixed action tape;
- no policy/checkpoint inference/processor;
- no replay/capture/restore;
- zero retry/replacement.

Perform one deterministic renderer consistency check before the cohort. It must still select unique software ordinal 8.

If the first attempted worker fails before environment construction due to a setup/path/preflight defect, stop the cohort immediately and return to route-level PIVOT review. Do not generate another full set of identical technical failures.

## Scientific null criteria

Unchanged from M1-SA-v1:
- action bytes exact;
- contact identity/set exact;
- predicates/terminal semantics/counters/gripper discrete state exact;
- grouped empirical floating physics envelopes;
- separate RGB envelopes;
- contact/grasp/carried and seven-regime support;
- no post-hoc epsilon/multiplier.

F3N PASS authorizes review for physical replay, not automatic F3b execution.

## Stop rule

There is no further infrastructure generation after F3N.

If F3N cannot construct the official environment with the direct trusted-install runtime, or cannot produce a complete interpretable null dataset, the current exact-state ReplayVLA infrastructure route is PIVOTED/PAUSED for the first paper.

Do not return to old G1/E3-E6, do not recreate missing source checkouts merely to satisfy legacy provenance, and do not create F3a-v4/v5.

## State after this decision

- ReplayVLA scientific direction: CONDITIONAL GO.
- legacy exact-state runtime-config derivation route: PIVOTED AWAY.
- F3N static qualification: AUTHORIZED.
- F3N dynamic null: CONDITIONALLY AUTHORIZED after static PASS.
- physical replay / policy replay / paper pilot remain NOT AUTHORIZED.
