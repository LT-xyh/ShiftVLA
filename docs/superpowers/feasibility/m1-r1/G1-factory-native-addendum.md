# G1 addendum — bounded E1–E6 evidence closure

## Authority and scope

This addendum supplements, and does not overwrite, the historical G1 report
`34978f9511177e71ef10fc94b842175d63f410fb`. It is bound to accepted design
`a511e05034c3c65a0a034226c625b091e0b2e319` and accepted implementation plan
`16f27942825b078d1794a4c5f9610272f932d6cf`. Only non-executing source,
metadata, hashing, ELF-header, ZIP, and pickle-instruction inspection was
performed. The official runtime was not imported.

The historical statement that the reconstruction graph “cannot be proved
without executing reducer semantics” is corrected here: the pickle instruction
stream and symbolic reducer targets can be inspected without executing them.
The remaining blocker is native-effect and provenance closure, not a requirement
to execute reducers.

## E1 — Fixed asset and load dispatch

**Verdict: COVERED for fixed-byte identification and branch selection;
UNRESOLVED for validation/use immutability at an eventual execution boundary.**

The frozen configuration binds `libero_spatial`, task `0`, init-state `0`, seed
`2027`. The pinned task map makes task 0 the first spatial task, whose selected
asset is:

```text
external/hf-libero/libero/libero/init_files/libero_spatial/
pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init
```

Evidence:

- outer file: 44,271 bytes, SHA-256
  `cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`;
- ZIP members are exactly `archive/data.pkl` (43,859 bytes) and
  `archive/version` (2 bytes), both stored (uncompressed), with no
  `constants.pkl`, `code/`, or other TorchScript records;
- `archive/data.pkl` SHA-256 is
  `19a1f03ed94cf5bff88906b2b3742245a64443e5cc623e6ffc6aed6773f4fef4`;
- `scripts/dcu_preflight.py:2233-2320` constructs the environment with
  `init_states=True`;
- `external/lerobot/src/lerobot/envs/libero.py:59-75,107-181` resolves the
  task file and calls `torch.load(path, weights_only=False)`;
- matching pinned/installed Python sources were byte-identical for the reviewed
  factory, LIBERO, wrapper, robosuite base/binding/EGL files.

The pinned `torch/serialization.py:1316-1604` selects the ZIP reader for this
container. `_is_torchscript_zip` at `2200-2201` is a pure member-name test for
`constants.pkl`; that member is absent. The `weights_only=False` call selects
the unrestricted pickle-module path. The ZIP structure therefore excludes the
legacy non-ZIP branch and TorchScript dispatch for these fixed bytes. The
remaining `_load` persistent-storage path is not a license: it is part of the
selected implementation and requires the E3 native dossier. Source validation
and execution-time byte immutability are not yet bound, so those dispositions
remain unresolved.

## E2 — Symbolic reconstruction graph

**Verdict: PARTIAL COVERAGE; overall E2 = UNRESOLVED.**

The complete `archive/data.pkl` instruction stream was traversed with
`pickletools.genops` without unpickling. It contains one `PROTO`, four
`GLOBAL`, four `REDUCE`, two `BUILD`, and one `STOP`, in addition to ordinary
tuple/list/scalar/memo instructions. The four global targets are exactly:

```text
numpy.core.multiarray._reconstruct
numpy.ndarray
_codecs.encode
numpy.dtype
```

The stream is a Torch ZIP pickle and therefore has no pickle `PERSID` or
`BINPERSID` instructions in the inspected data stream; tensor storage records
are not present as separate `data/<key>` ZIP members. This excludes a storage
record edge in the fixed bytes, but not the implementation's persistent-load
code path before the stream is consumed.

The symbolic scan establishes every visible GLOBAL/REDUCE/BUILD opcode and
their byte offsets (REDUCE at offsets 104, 110, 152, and 43851; BUILD at 184
and 43857). It intentionally does not print or retain physical state values.
However, a complete stack-and-memo type derivation for every intermediate
opaque ndarray argument was not produced by the bounded audit. Consequently
the exact final constructed Python type, ndarray subclass condition, dtype
object identity, and BUILD state shape remain **UNRESOLVED**, rather than being
inferred from familiar NumPy serialization patterns.

Required rejection point: any changed opcode/global, additional reducer,
unexpected memo target, or non-fixed member must be rejected before reducer
execution. No runtime observation may auto-expand this graph.

## E3 — Reconstruction native effects

**Verdict: UNRESOLVED.**

Only the source needed by the observed graph was reviewed. NumPy's
`numpy.core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype`, and
`_codecs.encode` are native-backed or dispatch into native-backed object
construction. Torch's unrestricted `serialization.py` calls an unpickler,
installs a `persistent_load`, may create storage mappings, and validates loaded
storages after reconstruction (`/public/home/xuyinghao/tmp/shiftvla-libero/`
`lib/python3.12/site-packages/torch/serialization.py:1945-2190`).

The fixed stream has no storage member or persistent opcode, which excludes
that concrete storage-record edge for these bytes. It does not prove that
NumPy allocation, dtype construction, ndarray subclass checks, object
references, or codec allocation are inert. The required actual native
implementations are present as candidate binaries, including NumPy
`_multiarray_umath` (SHA-256
`e6241082eaab01d0f68b2b661522ecb9d5d2160915104df8b112cc161dfa18b4`),
Torch `_C` (SHA-256
`812f4d4f89eac01fd902d2aa055941861c4313a113534598f90c04a2b9ed5b39`), and
the MuJoCo extension used by the factory. These hashes identify candidates only;
they do not establish source/build correspondence, allocator behavior, or
callback reachability. No reducers or native routines were executed.

## E4 — Factory/render/callback lifetime

**Verdict: UNRESOLVED.**

Static sources establish the following necessary boundaries:

- `robosuite/environments/base.py:110-140,216-303` constructs the simulator,
  calls `mj_forward`, creates the offscreen render context during internal
  reset, and later destroys simulator/renderer state at `689-700`;
- `robosuite/renderers/context/egl_context.py:20-30,111-155` imports EGL,
  creates the display/context, registers `atexit(EGL.eglTerminate, ...)`, and
  frees context state in `__del__`;
- `robosuite/utils/binding_utils.py:63-115,196-210` creates native render
  contexts and frees them from a finalizer;
- `robosuite/utils/observables.py:210-266` invokes installed sensor,
  corrupter, filter, and delay callbacks;
- `external/mujoco/python/mujoco/callbacks.cc:161-221` exposes Python-backed
  MuJoCo warning, passive, control, contact-filter, sensor, time, actuator and
  related callback bridges;
- LeRobot/Gymnasium close paths (`lerobot/envs/libero.py:393-399`,
  `lerobot/envs/utils.py:312-335`, installed Gymnasium SyncVectorEnv
  `close_extras:374-377`) destroy child environments.

The frozen configuration establishes an offscreen `mujoco` renderer and pixel
observations, but no execution was allowed to discover which optional
observables, callbacks, plugins, or finalizers are actually instantiated.
Registration state, mutation authority, callback re-entry, asynchronous
delivery, and whether the registered EGL `atexit` handler runs before the
controlled terminal boundary are therefore unresolved. Routes not reachable
from the exact model can be excluded only after a source/model closure dossier;
the current source inventory alone is insufficient.

## E5 — Native artifact and loader provenance

**Verdict: UNRESOLVED.**

Candidate native components on the frozen path include NumPy multiarray,
Torch `_C`/CPU libraries, MuJoCo Python extensions and `libmujoco.so.3.7.0`,
PyOpenGL/EGL bindings, Mesa/EGL/GL libraries, CPython/libc, and native callback
bridges. Representative static hashes were collected for NumPy, Torch, and
MuJoCo `_structs`; `readelf -d` on MuJoCo `_structs` showed a `$ORIGIN` runpath
and dependencies including `libmujoco`, libc, libm, librt, libpthread,
libdl, and the dynamic loader.

The pinned lock supplies package versions and checkout SHAs, while the
installed Python files are byte-identical to the reviewed source files. It does
not bind each loaded DSO to its reviewed implementation/build inputs, complete
transitive loader closure, environment interposition, or MuJoCo plugin
selection. A package version, DSO hash, or source SHA alone is insufficient
under Layer 3. Dynamic-loader and plugin dispositions therefore remain
unresolved, as do source-to-binary correspondence and callback registration
provenance.

## E6 — Closure and rejection index

**Verdict: UNRESOLVED.** The following index records exactly one disposition for
each necessary edge currently examined:

| Necessary edge | Disposition | Evidence / required rejection point |
|---|---|---|
| Fixed asset path and bytes | COVERED | Config, task map, ZIP listing, SHA-256; reject path/hash drift before load |
| ZIP vs legacy/TorchScript dispatch | EXCLUDED_WITH_EVIDENCE | ZIP member set lacks `constants.pkl`; reject alternate container before load |
| Official load call and `weights_only=False` | COVERED | LeRobot source lines 59–75; reject changed call/options |
| Full symbolic reducer stack/memo/type graph | UNRESOLVED | `genops` targets/offsets recorded; incomplete opaque intermediate typing; reject any graph drift before reducer |
| NumPy/codecs reducer effects | UNRESOLVED | Native-backed targets; reject unsupported type/subclass/dtype/allocation path before reducer |
| Torch persistent-storage behavior | EXCLUDED_WITH_EVIDENCE for fixed bytes; UNRESOLVED for implementation closure | No `data/*` members or persistent opcodes; reject changed bytes or storage path |
| MuJoCo callback bridges | UNRESOLVED | `callbacks.cc:161–221`; reject un-dossiered registration/re-entry |
| Observation callbacks | UNRESOLVED | `observables.py:210–266`; reject unknown callable or callback path |
| EGL setup/atexit/finalizers | UNRESOLVED | EGL context source and `atexit`/`__del__`; reject unreviewed terminal/finalizer behavior |
| Native DSO/build correspondence | UNRESOLVED | Candidate hashes/readelf only; reject binary/build drift before entry |
| Loader/interposition/plugin closure | UNRESOLVED | `$ORIGIN` and native dependencies observed; reject unexpected loader/DSO/plugin |
| Changed reducer/global | EXCLUDED_WITH_EVIDENCE as a required counterexample | Fixed opcode/global set is hashed; reject before reducer execution |
| Unsupported dtype/object/subclass | UNRESOLVED | Exact opaque state typing/native conditions not closed; reject before reconstruction |
| Unsupported finalizer/terminal behavior | UNRESOLVED | No runtime lifecycle execution permitted; reject before terminal sealing |

Counterexamples map to pre-entry rejection as follows: changed asset or reducer
global → source/payload hash and opcode contract; unsupported dtype/object →
symbolic type contract; binary/build drift → native dossier identity; loader
substitution → loader/source check before body; unexpected callback → callback
registration/dispatch dossier; unexpected DSO/plugin/interposition → loader
and ELF dependency contract; unsupported finalizer/terminal behavior → lifetime
dossier before sealing. None may be converted into an ALLOW by runtime learning.

## Overall G1 candidate verdict

**BLOCKED.** E1 has fixed-byte/dispatch coverage, but E2 is incomplete at the
opaque type-state level and E3–E6 contain necessary unresolved native,
callback, loader, and source/build provenance edges. This satisfies the
fail-closed rule. G1 is not a PASS candidate; G2 remains not authorized.

## Evidence inspected and commands executed

Exact source/configs inspected:

```text
docs/superpowers/specs/2026-09-10-m1-r1-operation-guard-design.md
docs/superpowers/plans/2026-09-10-m1-r1-operation-guard-implementation.md
configs/m1/renderer_preflight_r1_egl0.yaml
runtime/m1/libero_config/config.yaml
runtime/locks/shiftvla-libero-runtime.txt
scripts/dcu_preflight.py
external/lerobot/src/lerobot/envs/{factory.py,libero.py,utils.py}
external/hf-libero/libero/libero/{benchmark/libero_suite_task_map.py,envs/env_wrapper.py}
external/robosuite/robosuite/{environments/base.py,utils/binding_utils.py,renderers/context/egl_context.py,utils/observables.py}
external/mujoco/python/mujoco/callbacks.cc
/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages/torch/serialization.py
```

Commands actually executed were bounded `sed`, `rg`, `sha256sum`, `cmp`,
`readelf -d`, `find`, and a stdlib-only Python script using `zipfile` and
`pickletools.genops`. A second stdlib script tracked ZIP member metadata and
opcode/global counts. Report validation used `git diff --check` and a standard
library text assertion. No pytest or package imports were run.

## Prohibited-execution confirmation

No official runtime import, environment construction, EGL context creation,
render, policy/processor/replay invocation, init-state deserialization,
reducer execution, numerical state inspection, new schedule, or G2 operation
occurred. The original G1 report remains unchanged.

## Commit status

This addendum is the only file changed by this continuation and is not yet
committed. The main agent must independently verify it, stage only this file,
and commit it if the evidence remains accurate. No G2 transition is implied.
