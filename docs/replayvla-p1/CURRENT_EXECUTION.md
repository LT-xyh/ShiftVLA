# ReplayVLA-P1 CURRENT EXECUTION

Status: PAPER-1 LIVE-LIBERO ROUTE PIVOTED / PAUSED
Branch: xyh/replayvla-p1

Latest authority:
- docs/replayvla-p1/21_m0_native_feasibility_adjudication.md

Scientific route history:
- docs/replayvla-p1/16_prefix_reexecution_paper_route.md
- docs/replayvla-p1/20_prefix_runtime_route_pivot.md

Reviewed M0-native feasibility candidate:
- 2f2f9aac75c1d50ea9b7dba6a48a7fd311d0b1a2

Historical runtime evidence:
- first prefix microvalidation evidence commit: 33f0e2cb1f699254f92fb7c46df93ec653b8f864
- MV-R1 evidence commit: 5e8e63a104909ff299672e509f3f2567f1e2a1c1
- historical compact evidence files are immutable

## Current adjudication

The proposed M0-native prefix route is not admitted.

Reason 1:
normal dcu_preflight.run_phase preflight executes the current-worktree source-checkout gate before
asset binding. That gate requires repo-local detached clean checkouts for lerobot / hf-libero /
robosuite / mujoco. The authoritative P1 Git tree contains external/pins.yaml but not those checkout
directories.

Therefore the unmodified M0 phase owner cannot be assumed to reach its historical asset-binding seam
in the current worktree.

Making it run would require source-checkout recreation, preflight relaxation/parameterization, or a
P1-specific preflight variant. Those exceed the final thin-extension budget.

Reason 2:
existing dcu_preflight closed-loop provenance records action noise as native worker RNG with
explicit=false. P1 uses explicit paired flow noise. Reusing the closed-loop phase envelope unchanged
would therefore record incorrect provenance and would require another provenance/schema extension.

The route is stopped rather than serially repairing more runtime/provenance infrastructure.

## Scientific interpretation

This is an engineering/runtime-admission stop, not scientific falsification.

Not observed:
- real P1 CameraModder intervention;
- real switch-index lifecycle;
- real P1 explicit-noise policy query;
- CC / CS / SC / SS scientific rollouts;
- G_C / G_S / L / I.

## Current permissions

AUTHORIZED:
- research-direction reselection from scratch;
- repo/source analysis needed to assess alternative directions;
- reuse of existing code/evidence only when it shortens a newly selected route.

NOT AUTHORIZED:
- MV-R2;
- another P1 runtime/bootstrap route;
- third-party source checkout recreation solely for P1 preflight;
- preflight/provenance relaxation to rescue P1;
- manual asset-binding repair;
- exact-state/F3N/F3b repair;
- P1 40-rollout pilot;
- CC/CS/SC/SS scientific execution;
- Codex runtime for ReplayVLA-P1.

## Paper-1 objective

Optimize from scratch for:
- fastest credible publication path;
- 1–2 months ideal, 2–3 months acceptable;
- CCF B or above OR CAS Zone 3 or above;
- not merely engineering;
- compatible with 2 x Hygon DCU K100 64 GB;
- standard PyTorch Transformer paths;
- no dependence on CUDA-specific FlashAttention/Mamba kernels;
- extension potential to stronger venues / PhD-level Embodied AI/VLA work.

Sunk cost in ReplayVLA is not a reason to continue it.

## Next reviewer task

Perform independent research-direction selection.

ReplayVLA-P1 live-LIBERO implementation work is paused until a new authority explicitly reopens it.
