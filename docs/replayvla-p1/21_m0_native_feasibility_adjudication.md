# M0-native prefix execution feasibility adjudication

**Status: PIVOT — LIVE-LIBERO PREFIX-REEXECUTION RUNTIME WORK PAUSED FOR PAPER-1.**

Branch: `xyh/replayvla-p1`

Reviewed feasibility candidate:

`2f2f9aac75c1d50ea9b7dba6a48a7fd311d0b1a2`

Predecessor authority:

`docs/replayvla-p1/20_prefix_runtime_route_pivot.md`

Scientific route authority:

`docs/replayvla-p1/16_prefix_reexecution_paper_route.md`

## 1. Decision

The candidate's proposed M0-native runtime route is not admitted.

The paper hypothesis is not scientifically falsified. The runtime route is stopped because the proposed
"thin reuse of the existing M0 lifecycle" does not actually remain thin in the current ReplayVLA-P1
worktree.

No further live-LIBERO infrastructure repair is authorized for Paper-1 under this route.

## 2. Decisive blocker: normal M0 phase preflight cannot run unchanged in the current worktree

The candidate correctly identified that normal `dcu_preflight.run_phase()` owns the historical asset
binding through:

`run_preflight_gates(... prepare_assets=True) -> _verify_assets_binding(create=True)`

However, source inspection shows that `run_preflight_gates()` executes, in order:

1. `m0_smoke._verify_source_checkouts(_ROOT, preflight_manifest)`;
2. runtime/environment/artifact/reference gates;
3. `_verify_assets_binding(...)`.

The source-checkout gate requires all of:

- `external/lerobot`;
- `external/hf-libero`;
- `external/robosuite`;
- `external/mujoco`;

to exist below the current repository root as clean detached Git checkouts matching the approved
manifest and `external/pins.yaml`.

The reviewed ReplayVLA-P1 repository state contains `external/pins.yaml` but not those checkout
directories in the authoritative Git tree.

This is the same class of repo-local source-provenance requirement that earlier authority explicitly
moved away from for the pivoted runtime.

Therefore an unmodified normal M0 phase owner cannot be assumed to reach the asset-binding step in
the current P1 worktree.

Making it run would require at least one of:

- recreating repo-local third-party source checkouts;
- weakening or parameterizing the historical source-checkout preflight;
- creating a P1-specific preflight wrapper/variant.

All three exceed the admitted "no new bootstrap/provenance infrastructure" condition for this final
M0-native feasibility route.

## 3. Second blocker: existing closed-loop provenance misstates P1 action-noise semantics

`dcu_preflight.run_phase("closed-loop", phase_runner=...)` uses the existing phase provenance
builder after the custom runner returns.

For `closed-loop`, `_phase_noise_manifest()` currently records:

- mode = `native_worker_torch_rng`;
- explicit = `false`.

P1 requires explicit paired flow noise passed through official:

`SmolVLAPolicy.select_action(batch, noise=...)`.

Therefore using the existing closed-loop phase envelope unchanged would publish incorrect provenance.

Correcting this would require another default-off provenance/schema extension in `dcu_preflight`
or a new phase identity. That is additional infrastructure/provenance work, not merely a camera/noise
episode hook.

## 4. Consequence

The reviewed feasibility proposal is valuable static analysis, but its recommendation is changed from:

`CONDITIONAL GO`

to:

`PIVOT`

for Paper-1 live-LIBERO execution.

The route has now encountered, across bounded attempts:

- repo-relative BDDL/init provenance admission;
- package-local LIBERO asset binding;
- current-worktree source-checkout ownership;
- closed-loop phase provenance incompatibility with explicit paired noise.

Continuing would amount to serially rebuilding the runtime/provenance boundary rather than testing the
scientific hypothesis.

That violates the project's publication-time objective and the stop rule established in authority 20.

## 5. Scientific interpretation

Do not interpret this as evidence that closed-loop observation-corruption hysteresis is false.

Not observed on the admitted P1 route:

- a real CameraModder intervention;
- real switch-index behavior;
- a real paired-noise P1 policy query;
- CC/CS/SC/SS scientific rollouts;
- G_C, G_S, L, or I.

The failure is engineering/runtime admission, not scientific falsification.

## 6. Frozen historical evidence

Retain without modification:

- `runtime/replayvla-p1/prefix_reexecution_microvalidation.json`;
- `runtime/replayvla-p1/prefix_reexecution_microvalidation_r1.json`;
- all F3N historical evidence.

Do not create another microvalidation cohort for this route.

## 7. Forbidden next work for Paper-1

Do not:

- create MV-R2;
- recreate the missing third-party source checkouts solely to satisfy M0 preflight;
- weaken historical source-checkout gates for another retry;
- add another P1-specific preflight;
- add a new action-noise provenance framework to rescue this route;
- manually manage LIBERO asset symlinks;
- patch installed packages;
- create another environment/bootstrap runner;
- run the 40-rollout pilot;
- run CC/CS/SC/SS scientific execution;
- repair F3N/F3b/exact-state infrastructure.

## 8. Paper-1 state

For the publication objective, the live-LIBERO ReplayVLA-P1 execution direction is now:

`PIVOTED / PAUSED`.

The next authorized activity is research-direction selection from scratch, using the user's constraints:

- fastest credible paper path;
- ideally 1–2 months, 2–3 months acceptable;
- CCF B+ or CAS Zone 3+ target;
- no low-level engineering-only contribution;
- compatible with 2 x Hygon DCU K100 64 GB;
- no dependency on CUDA-specific FlashAttention/Mamba kernels;
- preference for a direction extensible to stronger venues / PhD research.

Existing ReplayVLA design/code may be reused only if it materially shortens a newly selected route.
Sunk cost is not a decision criterion.

## 9. Final route verdict

- exact-state ReplayVLA Paper-1 runtime: PIVOTED / PAUSED;
- direct-builder prefix microvalidation: PIVOTED / PAUSED;
- M0-native prefix runtime integration: PIVOTED / PAUSED;
- 40-rollout prefix pilot: NOT AUTHORIZED;
- next step: independent research-direction reselection.
