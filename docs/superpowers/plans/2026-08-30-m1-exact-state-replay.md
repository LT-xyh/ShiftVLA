# M1 exact mid-trajectory replay implementation plan

This is the execution checklist for the reviewed M1 amendment. The detailed
contract and evidence are in `docs/m1_state_replay.md`.

- [x] Verify the persisted M0 action artifacts and exact source/parent/action
  hashes. Do not invoke SmolVLA.
- [x] Close the concrete mutable-state owner graph, RNG state, OSC/interpolator
  state, plugin counters, sleeping markers, object joint property, observables,
  task predicates, queues, and caches. Unknown state fails closed.
- [x] Implement ReplayState schema v2 and exact integration-state readback.
- [x] Split physics and observation model fingerprints and forbid authoritative
  compiled-physics patching.
- [x] Implement a policy-free environment factory and import-negative audit.
- [x] Run source-only regime detection on the primary 82-step M0 trace.
- [x] Select the minimum source-only corpus and freeze the seven-window
  content-addressed registry before any authoritative restore.
- [x] Implement grouped contact-rich physics null and renderer null contracts,
  including nondeterministic RGB envelopes and no global fallback.
- [x] Implement three same-process and three fresh-process restores per stage,
  persisted source references, distinct PIDs, output hashes, exact predicates,
  actual terminal reasons, synchronous short terminal handling, and no retry.
- [x] Add fake-only TDD coverage and real task-0 construction/fingerprint/state
  closure checks.
- [ ] Supply at least 20 persisted, contact-rich complete duplicate-control
  physics and renderer run pairs for the frozen corpus.
- [ ] If null evidence passes, execute the bounded same/fresh restore matrix and
  publish M1 PASS/FAIL. If it is absent or incomplete, publish M1 BLOCKED before
  restore.
- [ ] Stop after the M1 terminal judgment. Do not start CC/CS/SC/SS or M2.
