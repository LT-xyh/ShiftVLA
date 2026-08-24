# ShiftVLA Experiment Protocol

## Status

Initial protocol draft derived from the frozen project plan. The
architecture-specific seams, exact environment APIs, and executable
configuration remain pending the read-only architecture audit.

This protocol operationalizes docs/research_spec.md. If an implementation
choice conflicts with that file, the research specification takes
precedence and human approval is required.

## Global rules

- Use a frozen SmolVLA checkpoint.
- Use official LeRobot preprocessing and environment processors.
- Keep simulator state, instruction, proprioception, checkpoint,
  preprocessing, and explicit action-generation noise matched.
- Change only the observation perturbation in an ID/OOD pair.
- Split discovery and validation by trajectory.
- Produce a run manifest for every experiment.
- Record the command used, configuration, seed, repository revisions, and
  unverified assumptions.
- Run the relevant tests and the smallest end-to-end smoke experiment
  before claiming a milestone complete.

## Milestones

| Milestone | Scope | Pass condition |
| --- | --- | --- |
| M0 | Environment audit and frozen baseline | Official policy evaluation runs reproducibly; smoke output and manifest are inspectable. |
| M1 | Simulator state replay | A serialized state can be restored with equivalent physical state and observations. |
| M2 | Matched ID/OOD camera rendering | Physical state, instruction, and proprioception match; only camera settings differ. |
| M3 | Deterministic matched sampling | Same observation plus same explicit noise produces the same action within tolerance. |
| M4 | Paired activation exchange | Disabled instrumentation is equivalent to the original model within tolerance. |
| M5 | Camera dependency discovery | State-level effects and hierarchical-bootstrap estimates are produced using discovery trajectories only. |
| M6 | Held-out restoration | A frozen Top-Delta-K selection is evaluated against Random-K on held-out trajectories. |

Do not implement a later milestone until the preceding milestone has
passed its verification gate.

## M0: baseline

- Do not train or fine-tune.
- Load the official frozen SmolVLA LIBERO checkpoint.
- Use the official preprocessing and environment processors.
- Provide separate smoke and full configurations.
- Save per-episode success, task ID, seed, rollout length, and
  run_manifest.json.
- The smoke configuration should use only 1-2 tasks and one episode.
- First inspect the pinned APIs before implementing the harness.

## M1: state replay

- Define a serializable matched-state record.
- Preserve task identity and instruction.
- Inspect the underlying LIBERO/MuJoCo state representation.
- Verify robot and object state before save and after restore.
- Test repeated restoration.
- Distinguish initialization-state replay from arbitrary mid-trajectory
  replay; do not assume an initialization reset API is sufficient.

## M2: matched camera pairs

For every pair, automatically check:

- physical_state_ID equals physical_state_OOD;
- instruction_ID equals instruction_OOD;
- proprioception_ID equals proprioception_OOD.

Record camera perturbation parameters and seeds. The initial smoke run
should use one task, 20 states, and one camera severity.

## M3: matched sampling

Expose an explicit noise input to the policy runner. Verify both:

- repeated ID runs with the same observation and noise are equivalent;
- ID and OOD runs use the same noise tensor.

Do not add activation intervention at this milestone.

## M4: activation exchange

Use paired clean/OOD activation exchange as the primary intervention.
Keep zero/mean ablation as a robustness or sanity check only.

The first acceptance test is no-op equivalence:

\[
a_{\mathrm{original}}
\approx
a_{\mathrm{instrumented,no\text{-}op}}.
\]

If no-op equivalence fails, stop and do not run paper experiments.

## M5: discovery

Use discovery trajectories only. For every matched state and component
group, compute the quantities in docs/research_spec.md and save a tidy
state-level table preserving:

- task_id;
- trajectory_id;
- state_id;
- component;
- condition.

Use hierarchical bootstrap over task, trajectory, and state. Report
component-wise \(\Delta\), 95% confidence intervals, heterogeneity
diagnostics, and a diagnostic plot.

## M6: held-out restoration

Freeze the component ranking and selected Top-K using discovery data
before loading validation results. On held-out trajectories compare:

- Top-Delta-K;
- Random-K;
- ID-important-K.

Compute normalized restoration exactly as defined in
docs/research_spec.md. The primary comparison is
\(R(\mathrm{Top\text{-}Delta\text{-}K}) -
R(\mathrm{Random\text{-}K})\).

Do not add Lighting, Noise, LoRA, or a second model in this pilot.

## Required run record

Every run should record:

- project and dependency Git revisions;
- checkpoint identifier and revision;
- Python, PyTorch, Transformers, CUDA, and GPU information;
- seed and action-generation noise seed;
- task, trajectory, and state identifiers;
- perturbation settings;
- configuration path;
- executed verification commands;
- output artifact paths.
