# ShiftVLA Engineering Instructions

## Project mission

Implement the frozen-policy ShiftVLA pilot for testing whether
observation distribution shifts reconfigure internal dependencies
of Vision-Language-Action policies.

The scientific design is defined in:
- docs/research_spec.md
- docs/experiment_protocol.md

Do not alter scientific definitions without explicit human approval.

## Pilot scope

The pilot uses:

- Model: frozen SmolVLA
- Benchmark: LIBERO / LIBERO-Plus
- First perturbation: Camera
- Intervention granularity: 6-8 architecture-aware groups
- No model training in the pilot
- No LoRA
- No World Model
- No SAE
- No exhaustive head-level analysis

## Scientific invariants

Matched comparisons MUST keep fixed:

- underlying simulator state
- task instruction
- proprioceptive state
- model checkpoint
- preprocessing
- action-generation noise / flow initial condition

Only the observation perturbation may change.

Discovery and validation data MUST be split by trajectory,
never by random frames.

## Intervention rules

Primary intervention:
paired clean/OOD activation exchange.

Zero/mean ablation is only a robustness/sanity check.

Counterfactual restoration is an analysis tool,
not a deployable repair algorithm.

## Engineering rules

1. Never silently modify third-party behavior.
2. Prefer wrappers and explicit instrumentation.
3. Any necessary LeRobot modification must be minimal,
   backward-compatible, disabled by default, and tested.
4. Do not bypass official LeRobot preprocessors.
5. Every experiment must produce a run manifest.
6. Every script must accept an explicit seed and config path.
7. Do not hard-code task IDs, paths, devices, or checkpoints.
8. Never use discovery data to evaluate held-out restoration.
9. Do not report a result until its verification command has run.
10. Keep changes small and commit after each verified milestone.

## Reproducibility

Every run manifest must include:

- git SHA
- LeRobot git SHA
- LIBERO / LIBERO-Plus git SHA
- checkpoint identifier and revision
- Python version
- PyTorch version
- transformers version
- CUDA version
- GPU model
- seed
- task IDs
- trajectory IDs
- perturbation settings
- action-generation noise seed
- config file
