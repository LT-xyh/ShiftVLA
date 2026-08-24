# ShiftVLA Research Specification

## Status and authority

This file defines the frozen scientific quantities for the initial
ShiftVLA pilot. It is not an implementation plan. Scientific definitions
must not be changed without explicit human approval.

The initial pilot studies a frozen SmolVLA policy under a camera-induced
observation shift, using matched clean/OOD comparisons and paired
activation exchange.

## Matched setting

For each matched state \(s\), keep the following fixed between the
in-distribution (ID) and shifted condition \(k\):

- simulator state;
- task instruction;
- proprioceptive state;
- policy checkpoint;
- official preprocessing;
- action-generation noise or flow initial condition \(\xi\).

Only the observation perturbation may change.

Let \(o_s^{ID}\) denote the ID observation and \(o_s^k\) the observation
under shifted condition \(k\). The corresponding policy outputs are

\[
a_{ID} = \pi(o_s^{ID}; \xi),
\qquad
a_k = \pi(o_s^k; \xi).
\]

The explicit noise \(\xi\) is part of the matched condition. It must be
held fixed when comparing ID and OOD outputs.

## Component intervention effect

Let \(j\) index an architecture-aware component group. The intervention
effect under condition \(k\) is

\[
I_j^k(s)
=
D\!\left(
  \pi^{do(j)}(o_s^k; \xi), a_{ID}
\right)
-
D\!\left(
  \pi(o_s^k; \xi), a_{ID}
\right).
\]

The shift in the component effect relative to ID is

\[
\delta_j^k(s) = I_j^k(s) - I_j^{ID}(s).
\]

The aggregate dependency-shift quantity is

\[
\Delta_j^k = \mathbb{E}_s\left[\delta_j^k(s)\right].
\]

The distance \(D\) must be specified and held fixed for an experiment.
Do not silently replace it with a demonstration-action MSE.

## Restoration

Let \(a_k^{patch(j)}\) be the action produced under condition \(k\)
after applying the component-\(j\) counterfactual patch. Define the
normalized restoration as

\[
R_j^k(s)
=
\frac{
  D(a_k, a_{ID}) - D(a_k^{patch(j)}, a_{ID})
}{
  D(a_k, a_{ID}) + \epsilon
}.
\]

Here \(\epsilon > 0\) prevents division by zero and must be recorded in
the experiment configuration.

Counterfactual restoration is an analysis tool, not a deployable repair
algorithm.

Demo action distance is an auxiliary indicator. Closed-loop success is
the final behavioral validation.

## Data separation

Discovery and validation splits are made by trajectory, never by random
frames. Component ranking, thresholds, and the final selected Top-K set
must be frozen using discovery data before held-out restoration is loaded.

## Scope boundaries

The initial pilot uses:

- frozen SmolVLA;
- LIBERO or LIBERO-Plus;
- camera perturbation as the first shift;
- 6-8 architecture-aware component groups;
- no training, fine-tuning, LoRA, World Model, SAE, or exhaustive
  head-level analysis.
