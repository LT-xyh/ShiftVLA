# G1 — Factory deserialization and necessary native closure

## Verdict

**BLOCKED.** This is a feasibility result, not an environment or renderer
execution result. G2 and all implementation tasks remain unauthorized.

The gate is blocked because the fixed simulator init-state deserialization
reaches native-backed NumPy/Torch reconstruction whose complete reducer,
binary/build, and callback provenance cannot be closed from the reviewed
sources and static artifacts alone. The required native closure is therefore
unresolved; no conditional PASS is claimed.

## Authority and inputs

- Accepted implementation-plan commit: `16f27942825b078d1794a4c5f9610272f932d6cf`.
- Accepted design authority: `a511e05034c3c65a0a034226c625b091e0b2e319`.
- Interpreter named by the plan: CPython 3.12.6 at
  `/public/home/xuyinghao/tmp/shiftvla-libero/bin/python`.
- Frozen binding: `libero_spatial`, task `0`, init-state id `0`, seed `2027`,
  from `configs/m1/renderer_preflight_r1_egl0.yaml`.
- Path configuration: `runtime/m1/libero_config/config.yaml` maps `init_states`
  to `external/hf-libero/libero/libero/init_files`.
- Source revisions (read from the governing configuration/lock): LeRobot
  `7e241bd630a3719a56157a497ce5d08f244784f1`, hf-libero
  `8561c60eea2fb93096146f240194649df73d8b1e`, robosuite
  `fbee5844ff5632f5b5698e204ec5357ca50be0df`, MuJoCo
  `72cb2b210da666617924de709406d6aadbe60c71`.

The worktree already contained user-owned dirty/untracked files. None were
modified or staged by this gate.

## Source-to-factory trace

Static reading established the following chain without importing the official
runtime or constructing an object:

1. `scripts/dcu_preflight.py:2233-2320`
   `build_cpu_environment_runtime` imports the environment packages and calls
   `make_env_config(..., init_states=True)` followed by `make_env(...)`.
2. `external/lerobot/src/lerobot/envs/factory.py:58-160` dispatches the local
   `EnvConfig` path to the registered LIBERO factory and constructs synchronous
   vector environments from environment callables.
3. `external/lerobot/src/lerobot/envs/libero.py:107-181` constructs `LiberoEnv`;
   when `init_states` is true its constructor calls `get_task_init_states`.
4. `external/lerobot/src/lerobot/envs/libero.py:59-75` resolves the task init
   file and calls `torch.load(path, weights_only=False)`.
5. The task map at
   `external/hf-libero/libero/libero/benchmark/libero_suite_task_map.py:2-12`
   makes task 0 the first `libero_spatial` task,
   `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`.
   The selected file is therefore:
   `external/hf-libero/libero/libero/init_files/libero_spatial/`
   `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init`.
6. `external/robosuite/robosuite/environments/base.py:110-140,216-303,689-700`
   statically shows simulator construction, `mj_forward`, renderer-context
   setup during internal reset, and close/destroy paths. This source was not
   executed.

The installed copies of the reviewed LeRobot, hf-libero, and robosuite Python
files were byte-identical to the corresponding pinned checkout files for the
factory, LIBERO environment, vector utility, wrapper, base environment,
binding utility, and EGL context. This establishes source identity for those
Python files only; it does not establish native binary/build identity.

## Safe static asset inspection

No `torch`, NumPy, LIBERO, or environment module was imported. No unpickling,
`torch.load`, reducer invocation, tensor reconstruction, numeric state read, or
environment construction occurred.

Commands executed:

```text
sha256sum <selected .pruned_init>
python3 (stdlib zipfile + pickletools.genops only)
```

Observed selected container evidence:

- outer SHA-256:
  `cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`
- size: `44271` bytes; ZIP members: `archive/data.pkl` (43859 bytes) and
  `archive/version` (2 bytes)
- `archive/data.pkl` SHA-256:
  `19a1f03ed94cf5bff88906b2b3742245a64443e5cc623e6ffc6aed6773f4fef4`
- safe opcode counts included one `PROTO`, four `GLOBAL`, four `REDUCE`, two
  `BUILD`, and one `STOP` (plus ordinary container opcodes).
- global references were exactly:
  `numpy.core.multiarray._reconstruct`, `numpy.ndarray`, `_codecs.encode`,
  and `numpy.dtype`.

The scan proves the selected file is not a plain inert scalar and identifies a
small visible reducer surface. It does **not** prove the resulting object
graph, reducer arguments, NumPy C implementation behavior, Torch storage
handling, or all transitive callbacks. Inspecting those by unpickling would
violate this gate. The `torch.load` source also has separate ZIP/TorchScript
dispatch, `_load`/legacy branches, persistent storage loading, and an
`UnpicklerWrapper`; `weights_only=False` selects the unrestricted pickle path
and is not a blanket permission.

## Native, callback, and provenance findings

Static source review found unresolved necessary native boundaries:

- The installed interpreter dependency contains native Torch (`torch/_C`),
  NumPy multiarray, and MuJoCo extension/`libmujoco` DSOs. Static hashes were
  recorded for representative files (`torch/_C`:
  `812f4d4f89eac01fd902d2aa055941861c4313a113534598f90c04a2b9ed5b39`,
  NumPy multiarray:
  `e6241082eaab01d0f68b2b661522ecb9d5d2160915104df8b112cc161dfa18b4`,
  MuJoCo `_structs`:
  `7adae30fdcfdbecde496551942ba08d08e1dfa75918a4c227bf194be26dc4b5b`).
  These are observations of candidate binaries, not a claim that they were
  loaded during this gate.
- `external/mujoco/python/mujoco/callbacks.cc:161-221` defines Python-backed
  MuJoCo callback bridges (`mju_user_warning`, `mjcb_passive`, `mjcb_control`,
  contact-filter, sensor, time, actuator and related callbacks). Their native
  reachability, registration state, and source-to-built-binary correspondence
  are not closed by the Python source SHA.
- `external/robosuite/robosuite/renderers/context/egl_context.py:20-30,111-155`
  imports native EGL bindings, registers `atexit` termination, and exposes
  `__del__` cleanup. `binding_utils.py:63-115,196-210` adds native MuJoCo render
  context creation/destruction and finalizer behavior. These are necessary for
  the eventual renderer path but their native dossier is not present.
- `external/robosuite/robosuite/utils/observables.py:210-266` invokes sensor,
  corrupter, filter, and delay callables from observation updates. Their exact
  callback closure is not established by the factory source alone.
- `external/lerobot/src/lerobot/envs/libero.py:393-399` and
  `external/lerobot/src/lerobot/envs/utils.py:312-335` close the environment;
  Gymnasium vector cleanup at the installed `sync_vector_env.py:374-377`
  iterates child `close()` calls. Destruction/finalization edges remain part of
  the required callback/lifetime dossier.

The runtime lock records package versions and source revisions, but does not
provide reproducible native build inputs, compiler/linker provenance, complete
DSO closure, or callback coverage dossiers. A package SHA/version and a DSO
hash alone are expressly insufficient under the accepted Layer 3 contract.

## Required counterexamples and rejection points

The following must remain rejected before entry in a future implementation:

1. A different `.pruned_init` file, a renamed/alternate task asset, or an
   asset outside the pinned `init_files` tree: reject during source/path binding.
2. An executable reducer or global outside the four observed fixed-asset
   globals, including a reducer reached through changed pickle bytes: reject
   before unpickling; do not widen `torch.load` permission.
3. A TorchScript ZIP or legacy/non-ZIP serialization branch: reject unless its
   complete source/native dossier is independently admitted.
4. A model/checkpoint passed to the same `torch.load` API: classify as forbidden
   ML loading; the simulator-asset exception is path- and graph-specific.
5. A changed NumPy/Torch/MuJoCo/EGL binary, unresolved transitive DSO, native
   callback, finalizer, `atexit` action, or build/source mismatch: reject before
   factory entry.

No counterexample was executed; these are required future rejection cases from
the accepted contract, not test results.

## Unresolved items

- Complete fixed-asset reducer graph cannot be proved without executing
  reducer semantics; safe static inspection intentionally stopped at opcode and
  global identities.
- Source-to-binary/build provenance for NumPy, Torch, MuJoCo, EGL/OpenGL and
  CPython/libc termination dependencies is absent from the governing evidence.
- Native callback registration/reachability and callback-suppressed behavior
  are unresolved, including MuJoCo callback bridges, EGL `atexit`, object
  finalizers, and observation callbacks.
- Complete transitive DSO closure and loader/build correspondence are not
  frozen; observed binary hashes are candidate evidence only.
- No required G1 native dossier or reviewed reducer whitelist exists.

Because these unresolved items are necessary coverage, G1 cannot PASS.

## Files and checks

Changed file (the only file authorized by G1):

`docs/superpowers/feasibility/m1-r1/G1-factory-native.md`

Checks actually run were read-only/static only: numbered source reads and
`rg` searches; SHA-256 hashing of listed source/config/asset/binary files;
byte-identity `cmp` checks between seven installed and pinned Python source
files; `readelf -d` on one MuJoCo extension; stdlib ZIP member inspection and
`pickletools.genops` opcode/global counting. No pytest, package import,
environment construction, EGL context, render, asset deserialization, policy,
processor, replay, schedule, or empirical command was run.

## Review/commit status

This report is newly written and has not yet been committed. Per the accepted
plan, the main agent must review this report, run `git diff --check`, stage only
this report, and commit it if the BLOCKED evidence is verified. G2 is not to be
started automatically.
