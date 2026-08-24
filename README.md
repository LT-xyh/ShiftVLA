# ShiftVLA

This repository is the engineering workspace for a frozen-policy ShiftVLA
pilot studying whether observation distribution shifts reconfigure
internal dependencies of Vision-Language-Action policies.

## Start here

1. Read AGENTS.md.
2. Read docs/research_spec.md.
3. Read docs/experiment_protocol.md.
4. Complete the read-only audit recorded in docs/architecture_map.md.
5. Begin with the M0 baseline only after the audit design is approved.

The full engineering rationale and staged Codex prompts are in
docs/shiftvla_codex_engineering_plan.md.

## Pilot scope

- Frozen SmolVLA.
- LIBERO or LIBERO-Plus.
- Camera perturbation as the first shift.
- Matched simulator state, instruction, proprioception, preprocessing,
  checkpoint, and action-generation noise.
- Paired activation exchange across 6-8 architecture-aware groups.
- No training, fine-tuning, LoRA, World Model, SAE, or exhaustive
  head-level analysis in the pilot.

## Repository status

The project currently contains planning and protocol documents only.
Runtime code, pinned external source trees, configurations, tests, and
experiment outputs must be added milestone by milestone with verification.

Outputs are experiment artifacts and should not be committed to Git.
