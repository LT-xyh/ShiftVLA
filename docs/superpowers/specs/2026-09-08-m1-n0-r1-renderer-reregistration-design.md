# M1-N0-R1 renderer device re-registration design

## Status and scope

### Operation-guard design revision (2026-09-10)

The user approved a source-bound explicit operation catalog, `sys.monitoring`,
and native-boundary fail-closed enforcement. The written revision is in
`2026-09-10-m1-r1-operation-guard-design.md` in this directory. It supersedes
the guard mechanism and early child-side restoration wording below, without
changing the experimental scope or counts. Production monitoring remains
active until process death; only the parent can finalize admission after exit.
The monitor is not proof of complete native/C-internal execution coverage.
Guard design remains REVISE pending the revision's review and implementation
gates. Renderer Preflight is NOT RUN; M1-N0 is BLOCKED; no new schedule exists.

### Approved passive-import clarification (2026-09-09)

The user explicitly approved a narrow exception to the processor import
prohibition below. The official `LiberoEnv` factory may passively import only
processor-related modules required by its audited dependency graph and named
individually in a frozen exact-module allowlist. No wildcard or prefix grants
import permission. A new or unexpected related module fails closed before
execution until separately reviewed; the runtime never learns or expands the
allowlist automatically.

This exception permits imports only. Processor instances, pipeline creation,
processor calls (including `__call__`), pre/postprocessing, policy or SmolVLA
loading/invocation, checkpoint loading, model/tokenizer/processor instance
creation, Hub/network fallback, and action generation remain prohibited.
Class definitions, type declarations, and the audited registration decorators
needed to import the official modules are passive module initialization, not
permission to instantiate or execute the registered classes.

The exact related-module closure will bind each installed source path and raw
SHA-256, its dependency provenance, and the audit entrypoint. An isolated
import-only audit verifies the static source-derived candidate set under
runtime guards; it does not construct an environment, create an EGL context,
render, or write to either empirical output root. Any newly encountered module
is rejected and must be source-reviewed before a draft audit can be repeated.
Only the exact successfully audited set is frozen for later execution.

Runtime guards must be installed before official factory imports and remain
active through cleanup. Evidence must separately report the exact allowlisted
passive imports, guard installation/restoration, zero processor instances,
pipeline creations and calls, zero policy imports/calls, zero checkpoint/model
loads, zero tokenizer/model instances, and zero Hub/network accesses. A blocked
attempt must remain visible even if application code catches its exception;
zero counters cannot be substituted for missing instrumentation. Source/hash
drift and pre-existing related modules invalidate a fresh-child audit.

Use a dedicated `scripts/m1_import_guard.py` with focused tests to keep this
cross-cutting boundary separate from the existing EGL adapter. The existing
admission module will verify the frozen contract and include its evidence in
the single-render worker record. No third-party sources or old bundle change.

This is a clarification of policy-free construction, not permission to use
the processor stack. Implementation and review must finish before any real
preflight or new M1-N0 schedule. Independent reviewer unavailability is recorded
as unavailable, never PASS; the user authorized the main agent to perform the
implementation checkpoint review without indefinite reviewer restarts.

This design covers only `M1-N0-R1 — Renderer Device Re-registration` and the
single replacement M1-N0 null-calibration attempt that it may admit.  It does
not change `ReplayState`, execute source-vs-restored replay, or begin
CC/CS/SC/SS or M2.

The preserved run at
`runs/m1_null_calibration/20260901_task000_init000_null20` remains immutable.
Its terminal `BLOCKED` judgment is valid evidence that all 40 workers rejected
the then-frozen EGL ordinal before construction, stepping, rendering, restore,
or policy use.  Nothing in R1 repairs, retries, or overwrites that bundle.

## Audited cause

The value `MUJOCO_EGL_DEVICE_ID=8` was a valid, host-specific EGL ordinal in
the earlier DCU preflight environment.  It selected Mesa llvmpipe after that
environment enumerated nine EGL devices.  M1 later copied that ordinal into
its CPU runtime contract even though the CPU runtime lock recorded EGL
ordinal 0.

The preserved failed workers directly prove only that their EGL enumeration
accepted the range 0 through 0 and rejected ordinal 8 before the first frame.
They do not prove why the visible enumeration changed.  Separate read-only
reconnaissance on 2026-09-08 provisionally observed one software EGL device:
ordinal 0 initialized as `EGL_MESA_device_software` and reported:

- vendor: `Mesa/X.org`
- renderer: `llvmpipe (LLVM 12.0.0, 256 bits)`
- version: `3.1 Mesa 21.1.5`

That current observation is not authoritative until persisted by the R1
admission probe with the exact interpreter, environment, sources, and Git
identity.  What is already established is that `8` was not a K100/DCU
physical index and that the failed run produced no physical-dynamics or
renderer-nondeterminism evidence.  EGL ordinals, physical accelerator
affinity, and policy compute devices are separate namespaces.

## Considered implementation strategies

### Selected: additive R1 admission contract and renderer overlay

Add an R1-only admission path to `scripts/m1_null_calibration.py`.  It first
validates the old state-replay file, SHA, and ordinal-8 contract unchanged.
It then makes an in-memory copy and applies only the independently registered
renderer overlay.  A dedicated R1 environment validator applies that merged
contract; it does not call the legacy ordinal-8 environment validator or
mutate `m1_state_replay.FROZEN_RUNTIME_ENVIRONMENT`.  The old state-replay and
null-calibration configurations are not edited.

This is the narrowest implementation that preserves old hashes, avoids a
second copy of the complete M1 configuration, and leaves `ReplayState`
untouched.  Legacy M1-N0 validation remains backward compatible so the old
bundle can still be audited byte-for-byte.

### Rejected: duplicate the complete state-replay configuration

An EGL-0-specific copy of the full state-replay configuration would avoid an
overlay seam, but would duplicate dynamics-critical settings and create a
future drift surface unrelated to renderer registration.

### Rejected: edit the old configurations or schedule

Changing `8` to `0` in the existing files would invalidate the frozen config,
run-spec, schedule, and terminal evidence hashes.  It would also turn a clean
historical failure into an untraceable in-place retry.

## Configuration namespaces

R1 records these independent fields:

```yaml
policy_compute_device: not_applicable_no_policy
renderer_backend: egl
renderer_device_id: "0"
```

`renderer_backend=egl` maps to `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl`.  `renderer_device_id` alone maps to
`MUJOCO_EGL_DEVICE_ID`.  No code derives it from CUDA, HIP, HCU, DCU, worker
affinity, or physical-device identifiers.  R1 child processes explicitly
remove inherited compute-visibility variables before importing the renderer
runtime, and publish that absence.

`not_applicable_no_policy` is a validation sentinel and must never be passed
to Torch.  Physical affinity, when present in another protocol, is separate
provenance such as `physical_compute_device_id`; R1 requires it to be null.
At minimum, children remove and record the absence of
`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
`GPU_DEVICE_ORDINAL`, and `NVIDIA_VISIBLE_DEVICES`.

The admission spec freezes the complete environment transition before any
EGL, MuJoCo, robosuite, or LIBERO import.  In order, the child:

1. records presence/absence for every governed variable without publishing
   unrelated environment content;
2. removes the exact `unset_before_import` list:
   `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
   `GPU_DEVICE_ORDINAL`, `NVIDIA_VISIBLE_DEVICES`, `MUJOCO_GL`,
   `PYOPENGL_PLATFORM`, `MUJOCO_EGL_DEVICE_ID`, `EGL_PLATFORM`,
   `__EGL_VENDOR_LIBRARY_FILENAMES`, `__EGL_VENDOR_LIBRARY_DIRS`,
   `__GLX_VENDOR_LIBRARY_NAME`, `LIBGL_ALWAYS_INDIRECT`,
   `LIBGL_ALWAYS_SOFTWARE`, `LIBGL_DRIVERS_PATH`,
   `MESA_LOADER_DRIVER_OVERRIDE`, `GALLIUM_DRIVER`, `DRI_PRIME`,
   `MESA_VK_DEVICE_SELECT`, and `VK_ICD_FILENAMES`;
3. rejects nonempty `LD_PRELOAD` or `LD_AUDIT` rather than silently changing
   loader injection, and freezes the admitted `LD_LIBRARY_PATH` value;
4. sets the exact offline/cache mapping from the validated base contract;
5. sets only `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and
   `MUJOCO_EGL_DEVICE_ID=0` as renderer selection;
6. verifies the final governed mapping and only then imports renderer code.

The R1 run specification records the unchanged base state-replay contract
SHA, renderer-admission spec and manifest hashes, the canonical SHA of the
merged runtime mapping, and all three namespace fields.

## Two distinct preregistration layers

R1 has an infrastructure admission layer and, only after it passes, a null
calibration layer.

### Renderer admission specification

A checked-in, self-hashed R1 renderer-admission specification at
`configs/m1/renderer_preflight_r1_egl0.yaml` freezes:

- the selected backend, renderer identity, and candidate ordinal 0;
- the exact CPU interpreter and runtime/source identities;
- task 0, init state 0, seed 2027, and `pixels_agent_pos` environment shape as
  provenance bindings only;
- the only allowed preflight operations and all forbidden operations;
- no-overwrite artifact paths and a bounded child-process timeout;
- exact PASS/BLOCKED rules;
- the predecessor bundle manifest described below.

The pinned runtime bindings include the exact interpreter
`/public/home/xuyinghao/tmp/shiftvla-libero/bin/python`, the raw SHA-256 of
`runtime/locks/shiftvla-libero-runtime.txt`, and a deterministic path/type/
size/SHA manifest of the actual M1 LIBERO configuration directory
`runtime/m1/libero_config`.  The legacy CPU lock's historical config path is
not substituted for the M1 path.  The admission spec also freezes the empty
HF cache path and verifies its emptiness without adding content.

This is not an authoritative M1-N0 configuration and contains no pair IDs,
action tape, regime windows, numerical envelopes, or null thresholds.  Its
contract SHA is canonical JSON of the parsed mapping with
`config_sha256` omitted.  The preflight terminal manifest at
`runs/m1_renderer_preflight/20260908_task000_init000_egl0/terminal_manifest.json`
uses the same canonical rule with `terminal_manifest_sha256` omitted; raw file
SHA-256 is recorded separately.

The predecessor manifest binds explicit paths, byte sizes, and raw file
SHA-256 values for every preserved config, run-spec, registry, attempt JSON,
sidecar, and log.  It also binds these semantic self/contract hashes:

- null config contract: `abbda1ee7d347a3c6cafad9f6f79a41a4227b84d063f2eea8ae7d53c2c5fae2b`;
- state-replay contract: `730aff4a41fd91fb837102ca5f363a4a142bf7010140f5176940700f1d1fd5f0`;
- run-spec self hash: `88202837c4dd7c60f0dd968e90d313e8c85a7da479b83ca85f0c32b2685c922c`;
- initial pair-registry self hash: `761ce35bb1416e50cbcfc13e7ec8dc1223a2f245a261ddc8c5ac57c06c07f6b9`;
- source-registry self hash: `092c64910b7530c6d84a98a925124879e9a96196b54d45b284a70ccfb62dd49c`;
- exact action-byte hash: `c17bc44ad8195fecb42a80b3b272828761a9df6d88dd2bafe45db01a6cb04bbf`;
- final pair-registry self hash: `c644ef02171bba0645811e0370a32c947e3065f88adda690c1b1b9fb3066888c`;
- terminal-manifest self hash: `461c31387aabf0a2bb8db90828fffc50d9181e09e207ab311678facc3df6cd3f`;
- terminal-manifest raw file hash: `6fb307f787eb62e2ef2be20513ee21dbd2852330ba592c6aac4ebc0b8a545a2a`.

### Authoritative null-calibration configuration

The authoritative file `configs/m1/null_calibration_r1_egl0.yaml` does not
exist until renderer admission is terminal `PASS`.  It then
freezes a new output root,
`runs/m1_null_calibration/20260908_task000_init000_null20_egl0`, and a new
`m1n0-r1-pair-` namespace.  It binds the complete renderer-admission manifest
by path, file SHA-256, and self SHA-256.  It retains the existing frozen
action tape, source registry, task, regimes, terminal contract, quantities,
pair count, timeouts, no-retry rule, and physics/renderer gates.

Preparation refuses to create an output root, run specification, or pair
registry unless the bound admission manifest is hash-valid and terminal
`PASS`.  No ordinal fallback or scan occurs during preparation or workers.

The admission output root and authoritative null output root are disjoint and
checked with resolved-path containment.  Admission writes only a fixed
allowlist of probe records and its terminal manifest.  Its command cannot call
`prepare_run` or create pair, attempt, job, tape, window, schedule, or null
measurement artifacts.  A separate post-PASS materializer is the only code
allowed to create the authoritative configuration; `prepare` is the only code
allowed to create its output root and schedule.

## Read-only EGL enumeration

The admission command first launches fresh, policy-free probe children in the
same CPU environment intended for preflight.  Each child:

1. removes compute-visibility and driver-override variables;
2. performs a checked two-call `eglQueryDevicesEXT` to obtain and then fill the
   complete device list without a fixed truncation limit;
3. checks only ordinals proven to be in range;
4. records device count, ordered stable descriptors, supported DRM node paths,
   EGL/GL identity, loaded graphics-library identities, runtime fingerprints,
   `/dev/dri`, mount/cgroup/device-namespace facts, and exit status;
5. destroys its temporary graphics context and exits.

Volatile values such as PID, time, and pointer addresses are excluded from the
canonical renderer identity.  Three fresh probe children must agree exactly
on two separately hashed schemas:

- `admission_full_renderer_fingerprint` includes the complete pre-import
  namespace fields plus context-derived EGL/GL identity and the graphics
  libraries actually loaded after context creation;
- `worker_preimport_namespace_fingerprint` contains only context-free device
  count/order/descriptors, supported DRM nodes, `/dev/dri`, mount/cgroup/device
  namespace facts, selected ordinal, exact governed environment mapping, and
  statically resolved graphics-library paths, file hashes, and build IDs.

The full fingerprint must resolve exactly one match for the preregistered Mesa
llvmpipe renderer, and its resolved ordinal must be 0.  A different count,
order, identity, or ambiguous match is `BLOCKED`; there is no automatic
fallback.

The authoritative worker later recomputes only the non-context-creating
`worker_preimport_namespace_fingerprint` before renderer import and compares
it exactly with that schema in the admission manifest.  It never compares
against context-derived fields and must not create a preliminary GL context,
because that would contaminate the official environment's renderer lifecycle.

These records are infrastructure reconnaissance, not null-calibration
evidence.

## Frozen M1-N0 Renderer Preflight

After EGL admission succeeds, one new child process performs exactly one
construction/render preflight through the official policy-free `LiberoEnv`
factory.

Task 0 is used to choose the BDDL/model at lazy construction.  Init-state 0
and seed 2027 are recorded bindings only: the approved seam does not call
public `reset()`, seed the inner environment, apply the persisted init state,
or settle it.  The preflight therefore proves renderer admission for this
task model, not equivalence to authoritative M1-N0 construction or state.

Allowed operations are:

- create the official lazy `LiberoEnv` wrapper for task 0;
- allow the unavoidable inner reset performed while constructing its
  `OffScreenRenderEnv`;
- call the public `render()` method exactly once;
- read only non-experimental return metadata: whether the call returned,
  return type, shape, and dtype;
- close/destroy the environment and exit the child process.

Forbidden operations are:

- public `reset()`;
- `env.step()` or any action creation/transmission;
- the 10-step settle path;
- `RuntimeAdapter` construction;
- null measurement, invariant capture, or frame persistence;
- replay capture/restore;
- SmolVLA, policy, checkpoint, processor, or inference imports/calls;
- task `check_success`, predicates, reward, done, or termination inspection;
- observation/state content, RGB bytes or hash, pixel metrics, or any other
  proxy for experimental output;
- retry, replacement, autoreset, or process reuse.

The returned RGB array is never serialized, hashed, retained as experimental
data, or reused.  Only its successful return, shape, and dtype are recorded.
The environment is closed in `finally`; the preflight process then exits.  A
later authoritative worker always constructs a new environment in a new
process.

Before the public render dispatch, transparent count-only instrumentation is
installed on the lazy wrapper and its `OffScreenRenderEnv` construction seam.
It wraps factory construction, public render, public reset, inner reset, and
inner step; delegates each call unchanged; records only counts; and restores
the original callables during cleanup.  It neither calls an operation nor
reads its return value beyond the permitted render metadata.  This provides
runtime evidence for the PASS counters rather than inferring them solely from
source text.

The preflight manifest records only:

- process PID and parent PID;
- EGL backend, selected ordinal, and enumerated device count;
- actual EGL/GL renderer identity;
- construction operation status and whether the one render call returned;
- returned shape/dtype when successful;
- exact allowed/forbidden operation counters;
- policy/processor import-negative audit;
- runtime, config, source, and Git identities;
- cleanup result and child exit status;
- explicit `evidence_role: infrastructure_admission_only` and
  `included_in_null_calibration_evidence: false`.

`Renderer Preflight = PASS` requires successful construction, exactly one
public render that returns normally, exactly the permitted inner reset, zero
forbidden operations, successful close, restored instrumentation, a fresh
child PID, the exact preregistered renderer identity, and a hash-valid
manifest.  Any deviation is terminal `BLOCKED` and prevents creation of the
authoritative 20-pair schedule.

The preflight runs at a committed admission source identity `P`.  After PASS,
the materialized authoritative config and prepared schedule are committed at
a later execution identity `A`.  The config binds `P` and the admission
manifest; before launch the parent requires all scientific-scope paths to
match committed `A`, records `A` in every in-memory job and result, and every
worker independently verifies the same committed blobs.  Unrelated user-owned
dirty paths remain untouched and explicitly allowlisted.  The future commit
`A` is not embedded into a pre-commit file, avoiding a circular Git identity.

## Schedule freeze and single authoritative run

Only after preflight `PASS`:

1. generate and self-hash the new null-calibration configuration;
2. prepare the new no-overwrite output root, run specification, pair registry,
   20 pair IDs, and 40 attempt IDs;
3. embed `MUJOCO_EGL_DEVICE_ID=0`, the three namespace fields, and the bound
   preflight hashes in the run specification;
4. commit the configuration and frozen schedule before launching workers;
5. execute the frozen schedule exactly once.

The authoritative protocol remains 20 pairs / 40 independent workers, with
no retry, replacement, SmolVLA, processor, replay restore, threshold change,
or reuse of the preflight process/environment/frame.  The existing grouped
physics and renderer null-envelope evaluation remains unchanged.

## Failure handling and stopping rule

All artifacts are atomic and no-overwrite.  A probe/preflight failure retains
its immutable diagnostics, publishes `BLOCKED`, and creates no authoritative
schedule.  An admitted authoritative run publishes the existing M1-N0
terminal contract: `PASS` only with 20 valid duplicate pairs and all required
coverage/gates; otherwise `BLOCKED` with the exact failure evidence.

After publishing the R1/M1-N0 judgment, execution stops.  It does not modify
`ReplayState` or begin authoritative source-vs-restored replay, CC/CS/SC/SS,
Camera/Lighting interventions, SensorNoise, or M2.
