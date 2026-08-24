# ShiftVLA Architecture Map

## Status

Pending read-only architecture and environment reconnaissance.

This file is intentionally a template. The current project directory does
not yet contain the pinned LeRobot or LIBERO/LIBERO-Plus source trees, so
exact call paths, tensor shapes, hook behavior, and state-replay seams
must not be guessed.

## Audit scope

Inspect the pinned implementations and record evidence for:

1. official SmolVLA LIBERO checkpoint loading;
2. observation preprocessing and environment processors;
3. the public policy action-selection API and final action chunk;
4. flow-matching noise creation and explicit-noise injection;
5. vision encoder, connector, VLM layers, action expert, and projections;
6. the exact transformer execution loop and forward-hook behavior;
7. LIBERO reset and MuJoCo state serialization/restoration APIs;
8. LIBERO-Plus camera perturbation representation.

## Required evidence

### Source inventory

| Subsystem | Exact source path | Relevant class/function | Revision |
| --- | --- | --- | --- |
| SmolVLA policy | To be filled | To be filled | To be filled |
| Flow matching | To be filled | To be filled | To be filled |
| VLM and action expert | To be filled | To be filled | To be filled |
| Observation processor | To be filled | To be filled | To be filled |
| LIBERO environment | To be filled | To be filled | To be filled |
| LIBERO-Plus perturbation | To be filled | To be filled | To be filled |

### Call graph

To be filled from source inspection:

\`\`\`text
public policy action API
  -> observation preprocessing
  -> vision/VLM prefix computation
  -> action expert
  -> flow-matching integration
  -> final action chunk
\`\`\`

### Tensor shapes

Record concrete shapes at every proposed intervention location:

- image tensors;
- vision features;
- connector output;
- VLM hidden states;
- action-expert hidden states;
- action projections;
- flow noise and denoising states;
- final action chunk.

### State replay seam

To be filled after inspecting the underlying simulator state API.

The audit must distinguish initialization-state replay from arbitrary
mid-trajectory replay and must include a testable restoration invariant.

### Activation-intervention seam

To be filled after tracing the actual execution loop.

The audit must establish:

- whether ordinary PyTorch forward hooks fire at the intended boundaries;
- whether the intervention is applied during prefix computation,
  denoising, or both;
- exact stream and layer identifiers;
- hidden-state shapes;
- disabled-controller no-op equivalence strategy.

### Preliminary component grouping

Proposed grouping to verify against the implementation:

1. vision encoder;
2. multimodal connector/projector;
3. early VLM layers;
4. late VLM layers;
5. VLM-to-action interface;
6. early action-expert layers;
7. late action-expert layers;
8. action projection/output.

This is a proposal, not an experimental result.

## M0 design checkpoint

After the audit, propose the M0 baseline implementation only. Include
the public APIs used, official preprocessing path, smoke configuration,
manifest fields, tests, and the exact verification command. Do not modify
runtime code until the design is approved.
