# M1-N0-R1 Operation Guard Implementation Plan — execution revision

> **For agentic workers:** Use `subagent-driven-development` or
> `executing-plans`, plus `test-driven-development` and
> `verification-before-completion`, after user acceptance of this plan.
> All delegated substantive work uses GPT-5.6-luna / max. If unavailable,
> the main agent may continue without subagents under the user's latest routing
> instruction. Never substitute a speed-oriented execution model.

**Goal:** Translate the accepted operation-guard design into bounded,
test-first work without authorizing empirical execution.

**Architecture:** An independent source/payload/import authority and exact
semantic/infrastructure resolver feed CPython monitoring. A previously verified
native fatal primitive enforces nonreturning rejection. An independently
validated terminal channel and parent verdict complete the evidence chain.

**Tech stack:** Pinned CPython 3.12.6, standard library, pytest, the existing
CPU runtime, and one small project-owned C extension. No third-party rewriting.

## 0. Authority, current status, and execution boundary

Repository: `/public/home/xuyinghao/workspace/vla/ShiftVLA`.
Branch: `xyh/m1-state-replay`.
Accepted specification commit:
`a511e05034c3c65a0a034226c625b091e0b2e319`.
Specification:
`docs/superpowers/specs/2026-09-10-m1-r1-operation-guard-design.md`.

The user supplied the ACCEPT decision after the immutable specification commit;
historical REVISE wording in that blob is not a reason to reopen design. Preserve
the blob and attach the user decision to this plan's authority record.

At this revision:

| Item | State |
|---|---|
| Guard Design | ACCEPT |
| Implementation Plan | REVISE — revised execution draft awaiting user review |
| Phase 0 gates | NOT RUN |
| Guard Implementation | NOT STARTED |
| Renderer Preflight | NOT RUN |
| M1-N0 | BLOCKED |
| New schedule | NOT CREATED |

Only this plan document is written during plan revision. Code blocks below are
future instructions, not executed implementation. Neither a test name nor an
expected GREEN is evidence that a test was run.

No official environment construction, EGL context creation, public render,
schedule preparation, replay, policy invocation, or empirical command is allowed
in Phase 0 or synthetic implementation verification. Official import-only audit
is a separate later checkpoint with its own source/native admission prerequisite.

**Phase 0 is a hard dependency, not a progress suggestion.** A failed or
unresolved gate changes the main-agent judgment to IMPLEMENTATION PLAN BLOCKED.
Do not begin C1 or later work, patch a third-party package, alter the factory,
or expand permission to obtain a PASS. Missing evidence is not a conditional PASS.

### 0.1 Verified repository facts

- HEAD was verified equal to the accepted commit before writing this revision.
- `scripts/m1_renderer_admission.py` has pure validators, no-overwrite writers,
  `InjectedEGLBackend`, fingerprint builders, and `verify_three_probe_records`.
  It has no complete parent/worker preflight CLI.
- `scripts/m1_import_guard.py` and `tests/test_m1_import_guard.py` are untracked
  rejected work, not an accepted committed guard. Its AST rewrite, name
  classifiers, mutable class instrumentation, finder fall-through, and early
  restoration must not enter the replacement.
- `scripts/m1_null_calibration.py::_default_process_runner` uses subprocess
  completion plus a dedicated result file; retain it for the old null workflow,
  not as proof of protected terminal streaming.
- `scripts/dcu_preflight.py::build_cpu_environment_runtime` requires the pinned
  official factory, `init_states=True`, synchronous one-env operation, and no
  policy. `lerobot/envs/libero.py::get_task_init_states` calls
  `torch.load(..., weights_only=False)` for the simulator asset.
- CPython `-I -S -B` currently gives base prefix `/usr/local/python3.12`, not the
  venv prefix; blindly enabling it breaks the factory's purelib check.
- robosuite EGL registers an `atexit` termination callback and implements
  `__del__ -> free`. These cannot disappear from the coverage argument.
- No repository implementation of the accepted `sys.monitoring` mechanism or
  protected evidence pipe was found. Prior transient probes are feasibility
  observations, not persistent regression tests or acceptance evidence.

### 0.2 Common task conventions

Every command below runs from the repository root. Set these shell variables
once in the future execution session; do not change HOME or CODEX_HOME:

```bash
cd /public/home/xuyinghao/workspace/vla/ShiftVLA
GUARD_PY=/public/home/xuyinghao/tmp/shiftvla-libero/bin/python
```

Tests use `"$GUARD_PY" -B -m pytest`. Child probes must launch the same executable
with `-I -S -B`; no pytest plugin or developer-shell environment becomes trusted
production bootstrap. RED caused by a typo or missing fixture is not an accepted
behavioral RED. When an API is new, initial import failure is a scaffold RED;
add a rejecting/empty stub and observe the specified behavioral assertion before
implementing the positive path. Never weaken the assertion to obtain GREEN.

Each numbered task ends with exact staging and a bounded commit. Test failure
means do not commit that task as complete and do not execute a dependent task.
Rollback means disable the new admission entrypoint and retain diagnostic
evidence; never reset the worktree, erase evidence, or restore the rejected guard
as an accepted fallback. Preserve user changes in AGENTS.md and all run files.

New source files use `apply_patch`; future fixture writes inside pytest may use
`tmp_path`. Do not run the commands in this document while merely revising it.

## 1. Dependency graph and migration

```text
Plan review
  -> G1 factory/native feasibility
  -> G2 startup feasibility
  -> G3 payload/loader feasibility
  -> G4 fatal/no-return feasibility
  -> main-agent Phase-0 PASS (all four, no unresolved coverage)
  -> C1 schemas -> C2 identity records
  -> P1 source snapshot -> P2 nested authority -> P3 payload dispositions
  -> I1 exact resolver -> I2 pre-exec loader -> I3 alternate-execution rejection
  -> F1 promote verified fatal primitive (same source as G4)
  -> R1 callable resolver -> R2 infrastructure boundaries
  -> M1 event observation -> M2 fatal rejection -> M3 supervised failure cleanup
  -> M4 monitor integrity
  -> B1 production bootstrap -> N1 dossier enforcement -> A1 callbacks
  -> T1 framed channel -> T2 no-return sealing -> T3 parent supervision
  -> V1 evidence validator -> V2 v2 publication
  -> L1 synthetic worker -> L2 synthetic parent
  -> Q1 audit/freeze/review (no empirical run)
```

G1-G4 may fail before any production implementation. F1 precedes M2; there is
no Python exception-only or temporary `os._exit` substitute used to make M2 GREEN.
T2 adds successful terminal publication to the already verified fatal mechanism;
it does not replace M2's failure semantics.

| Existing component | Decision | Reason |
|---|---|---|
| Admission canonical hashing, predecessor verification, exclusive publication | RETAIN | Useful pure behavior; revalidate before reuse |
| EGL backend and fingerprint validators | RETAIN | Existing tested adapter, not native absence proof |
| Admission config and new worker evidence | REFACTOR | Versioned guard bindings and stream evidence required |
| Old ImportUseGuard / AST transformer / operation classifiers | REPLACE | Rejected semantic and lifecycle model |
| Old guard positive/negative scenarios | REFACTOR tests only | Keep adversarial intent, discard invalid PASS assertions |
| dcu_preflight official environment-only factory | RETAIN unchanged | Frozen workload seam |
| Null result-file runner and old null records | RETAIN unchanged | Not the new admission transport |
| RuntimeAdapter / replay / schedule dispatch from admission | DEAD PATH | Must be unreachable |

New modules live in `scripts/m1_guard/`; do not introduce a compatibility wrapper
around the old guard. Old untracked files are not deleted by this plan. Record
their hashes and make the new dependency graph unable to import them.

## 2. Phase 0: four feasibility gates before implementation

Phase 0 artifacts are design evidence, never empirical admission evidence.
Future gate reports live under `docs/superpowers/feasibility/m1-r1/`. Scratch
sources/binaries use newly created temporary directories outside run roots.
No gate creates `runs/m1_renderer_preflight`, a null config, or a schedule.

Each report has exact fields: `gate_id`, `spec_commit`, `input_bindings`,
`method`, `observations`, `negative_cases`, `coverage_limits`, `unresolved`,
`verdict`, `reviewer`. A PASS requires `unresolved=[]` and cited evidence for
every required edge. A synthetic test cannot manufacture real source provenance.

### G1 — Factory deserialization and necessary native closure

**Depends on:** user plan review. **Inspect:**
`scripts/dcu_preflight.py:2233`, installed and pinned-source
`lerobot/envs/libero.py`, `external/hf-libero/libero/libero/envs/env_wrapper.py`,
`external/robosuite/robosuite/renderers/context/egl_context.py`, the runtime lock,
interpreter/native package metadata. **Output:**
`docs/superpowers/feasibility/m1-r1/G1-factory-native.md`.

- [ ] Step 1: Read the exact factory and transitive construction/import/render/
  cleanup entry sources. Enumerate source path/hash, loaded binary candidate,
  deserialization reducer, implicit callback and dynamic dispatch boundaries.
  Follow conditional branches relevant to the frozen task without calling them.
- [ ] Step 2: Trace `init_states=True` to `torch.load(weights_only=False)`.
  Review the exact fixed asset's container/reducer structure using static
  inspection only, without unpickling or emitting numerical state content.
  Hashes and reducer identities may be recorded; no state values are recorded.
  If safe inspection itself needs prohibited deserialization, stop BLOCKED.
- [ ] Step 3: Distinguish simulator asset deserialization from forbidden ML
  checkpoint/model loading. Bind the exact asset and allowed reducer graph.
  Neither a `.pruned_init` suffix nor `weights_only=False` is permission. No
  blanket `torch.load` exemption and no `init_states=False` replacement.
- [ ] Step 4: Establish a reviewable provenance/coverage chain for all necessary
  native entries, bootstrap and termination dependencies, including transitive
  callbacks. A binary hash or matching version alone is insufficient. Unknown
  extensions, opaque reducers, or unavailable necessary build correspondence
  make this gate BLOCKED before C1.
- [ ] Step 5: Record counterexamples: different asset, executable reducer outside
  the fixed graph, model file at the same API, native dependency drift, unknown
  callback. For each, show the required rejection point before entry.
- [ ] Step 6: Main-agent review the report; do not label possible future audit
  work as current coverage. Verify and stage only this report:

```bash
git diff --check
git add docs/superpowers/feasibility/m1-r1/G1-factory-native.md
git commit -m "docs(m1): record factory native feasibility gate"
```

Expected RED: necessary source/binary/reducer coverage is incomplete, so verdict
is BLOCKED. GREEN means the evidence actually closes these paths, not merely that
an inventory exists. On BLOCKED, do not advance to production tasks.

### G2 — Isolated startup without site execution

**Depends on:** G1 PASS. **Inspect:** actual `pyvenv.cfg`, interpreter/build,
startup/frozen modules and loader configuration. **Output:**
`docs/superpowers/feasibility/m1-r1/G2-bootstrap.md`.

- [ ] Step 1: Run the following pure-stdlib baseline, which intentionally
  demonstrates the venv mismatch rather than pretending `-S` solves startup:

```bash
"$GUARD_PY" -I -S -B -c 'import sys,sysconfig; assert sys.version_info[:3] == (3,12,6); assert sys.flags.isolated and sys.flags.no_site; print(sys.prefix); print(sysconfig.get_paths()["purelib"])'
```

Expected baseline: `/usr/local/python3.12` and its base site-packages, not the
admitted venv. This command imports no official experiment packages.

- [ ] Step 2: In a fresh synthetic child, set only the verified venv
  `sys.prefix/sys.exec_prefix` before importing sysconfig; construct sys.path
  from the frozen stdlib, lib-dynload and venv purelib roots. Do not call site,
  process `.pth`, or inherit PYTHONPATH. Assert sysconfig's purelib is exactly
  `/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages`.
- [ ] Step 3: Plant inert synthetic sitecustomize/usercustomize/.pth sentinels
  in a temporary test directory. Launch with polluted PYTHONPATH and assert
  no sentinel runs. Also verify unknown preload/audit environment is rejected
  by the proposed parent before spawning. Do not test by loading an unknown DSO.
- [ ] Step 4: Account for executable work before this Python script: interpreter,
  dynamic loader, libraries, frozen bootstrap and supported execution hooks.
  A clean sys.modules snapshot is not that proof. Unresolved provenance blocks G2.
- [ ] Step 5: Report exact argv, environment disposition and path bindings;
  review and commit only the gate report:

```bash
git diff --check
git add docs/superpowers/feasibility/m1-r1/G2-bootstrap.md
git commit -m "docs(m1): record earliest startup feasibility gate"
```

GREEN requires both sentinel tests and provenance closure. No `-S`-only claim.

### G3 — Direct source payload and loader bypass spike

**Depends on:** G1-G2 PASS. **Output:**
`docs/superpowers/feasibility/m1-r1/G3-payload.md`.

- [ ] Step 1: In a pure synthetic child compile reviewed bytes once. Retain the
  root and all nested CodeType references; execute that same root. Test skeleton:

```python
import types

def test_retained_payload_and_nested_identity():
    source = b"def outer():\n    def inner():\n        return 7\n    return inner\n"
    root = compile(source, "fixture.py", "exec", flags=0,
                   dont_inherit=True, optimize=0)
    admitted = []
    def collect(code):
        admitted.append(code)
        for item in code.co_consts:
            if type(item) is types.CodeType:
                collect(item)
    collect(root)
    namespace = {}
    exec(root, namespace, namespace)
    actual = namespace["outer"]().__code__
    assert any(actual is code for code in admitted)
    forged = actual.replace()
    assert not any(forged is code for code in admitted)
```

- [ ] Step 2: Add synthetic counterexamples for mismatched cache, same metadata
  with different executable, nested replacement, source replacement after read,
  custom finder/loader, direct loader execution, sourceless/frozen/generated
  paths without provenance. Assert before-body rejection or execution of the
  retained verified bytes, never execution of the substituted payload.
- [ ] Step 3: Compare the selected retained-bytes loader with default cached
  SourceFileLoader behavior. Do not claim this spike attests real frozen modules.
  Their provenance must already be accounted for by G2.
- [ ] Step 4: Record the precise compile conditions and identify required
  generated-code contracts (including dataclass/namedtuple). If unavoidable
  generated code cannot be bound before execution, G3 is BLOCKED.
- [ ] Step 5: Commit only reviewed design evidence:

```bash
git diff --check
git add docs/superpowers/feasibility/m1-r1/G3-payload.md
git commit -m "docs(m1): record executable payload feasibility gate"
```

RED is acceptance of a forged executable or inability to close an unavoidable
path. GREEN is exact provenance plus rejection, not matching metadata.

### G4 — Minimal native fatal primitive and nonreturn proof

**Depends on:** G1-G3 PASS. **Output:**
`docs/superpowers/feasibility/m1-r1/G4-fatal.md` and the reviewed source listing
inside that report. Compile/run only in a new temporary directory.

- [ ] Step 1: Use this complete minimal extension source for the spike. This
  same source, without semantic substitutions, is promoted by F1:

```c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <unistd.h>

static PyObject *fatal_blocked(PyObject *self, PyObject *unused) {
    (void)self;
    (void)unused;
    _exit(70);
}
static PyMethodDef methods[] = {
    {"fatal_blocked", fatal_blocked, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_terminal", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};
PyMODINIT_FUNC PyInit__terminal(void) { return PyModule_Create(&module); }
```

- [ ] Step 2: Bind actual compiler/header/libc identities. Verified discovery
  points are `/usr/bin/cc`, `/usr/local/python3.12/include/python3.12`, and
  extension suffix `.cpython-312-x86_64-linux-gnu.so`; recheck before build.
  Compile the scratch source with `-shared -fPIC -O2 -std=c11` and the pinned
  include directory. Do not use setuptools downloads or environment-provided
  extra compiler/linker flags. Record actual linked dependencies.
- [ ] Step 3: In subprocesses, invoke fatal directly and from a monitoring
  callback. Surround the call by repeated `except BaseException`, `finally`,
  generator/coroutine and registered atexit sentinels. Assert return code 70 and
  absence of every target/continuation/finalization marker. Use a pipe for
  sentinels, not empirical files.
- [ ] Step 4: Test an absent/unloadable/wrong-hash primitive: admission must not
  start. Test callback failure: no exception-only fallback is permitted. Record
  why no Python cleanup is attempted inside a suppressed monitoring callback.
- [ ] Step 5: Bind the exact scratch source/binary/build hashes and review
  no-return semantics, signal/native callback assumptions and OS resource
  cleanup limits. Synthetic success without the native provenance is BLOCKED.
- [ ] Step 6: Commit the gate report, not the scratch binary:

```bash
git diff --check
git add docs/superpowers/feasibility/m1-r1/G4-fatal.md
git commit -m "docs(m1): record fatal nonreturn feasibility gate"
```

The gate's source is already the future failure primitive. F1 cannot replace it
with an exception, monkeypatchable Python-only terminator, or different C body
without repeating G4 review. Full success-marker transport is not needed here.

### Phase-0 main-agent decision

- [ ] Verify all four reports bind the accepted spec and actual inputs.
- [ ] Require PASS for all four, no unresolved source/native/loader/startup
  boundaries, and a reviewed G4 implementation/build recipe.
- [ ] If any gate fails, record IMPLEMENTATION PLAN BLOCKED with exact missing
  evidence and stop. Do not start C1 or quietly defer native feasibility to N1.

## 3. Frozen interface choices for the implementation tasks

These are new interfaces, not existing repository symbols. Test skeletons below
use only standard pytest fixtures or helpers explicitly introduced by a task.

### 3.1 Contracts

`scripts/m1_guard/contracts.py` defines:

- `ContractError(ValueError)`.
- `canonical_bytes(value: object) -> bytes`: strict JSON primitives only,
  recursively sorted object keys, UTF-8, separators `(',', ':')`, ensure_ascii
  false, no NaN/Infinity, no arbitrary object conversion.
- `validate_gate_results(value: dict[str, str]) -> None`: exact keys G1-G4 and
  every value PASS; no `all([])` success.
- `validate_operation(value: dict) -> dict`: exact fields `id`, `authority_id`,
  `locator`, `disposition`, `phases`, `category`, `count_limit`; dispositions
  ALLOW/FORBID, nonempty phase list, nonnegative exact-int count limit or null.
- `validate_dossier(value: dict) -> dict`: exact fields `id`, `kind`,
  `entry_ids`, `member_ids`, `semantic_exits`, `conditions`, `evidence`,
  `limitations`, `verdict`; kind infrastructure/native/bootstrap/terminal;
  PASS requires nonempty evidence and no unresolved limitation.
- `validate_bundle(value: dict) -> dict`: schema_version 2, spec_commit,
  interpreter, compilation, modules, operations, dossiers, launch, transport,
  gate_report_hashes, bundle_sha256. Reject unknown/missing keys and conflicts.

IDs are printable ASCII identifiers, not Python expressions. A canonical SHA
omits only its own self-hash field. Duplicate JSON keys are rejected at parse.
Validated data is copied into private immutable tuples/read-only maps; callers
cannot retain mutable aliases. Operation counts and completion states are not
inferred from missing entries.

### 3.2 Payload and imports

`payload.py` defines frozen `SourceBinding(module, path, sha256)`,
`SourceSnapshot(binding, raw)`, `Payload(snapshot, root, compilation_id)`,
`Authority(payload, nodes)`, and these functions:

```python
def read_snapshot(binding):
    """Return SourceSnapshot from one verified regular-file descriptor."""

def compile_snapshot(snapshot):
    """Compile retained bytes with exec/flags=0/dont_inherit=True/optimize=0."""

def authorize(payload):
    """Retain root and nested CodeType objects; return Authority."""

def require_code(authority, code):
    """Return the co_consts index-tuple locator only for the identical object."""

def execute(authority, payload, namespace):
    """Require identical retained payload/root; execute it, return root."""
```

Functions above are signatures/contracts in this document, not stub production
implementations to commit. Errors are `SourceBindingError`, `PayloadError`, and
`UnsupportedPayloadError`, all subclasses of ContractError.

`imports.py` defines `ImportDenied`, `ImportTable(bindings)`,
`AdmittedLoader(authority)`, `resolve_exact(table, fullname)`,
`validate_loader(spec, binding)`, `reject_unregistered_execution(authority, code)`.
The new loader never delegates `create_module/get_code/exec_module` to an unknown
original loader. Module names/paths are exact; all supported package metadata
must be source-bound. Namespace/sourceless/extension/frozen/generated paths need
their distinct G1-G3-reviewed contracts; direct-source permission never covers them.

### 3.3 Runtime, transport, and termination

- `resolver.py`: `Decision(kind, stable_id, disposition)`, `Registry`,
  `register_python(authority, function, operation)`, `resolve_callable(target)`.
- `monitor.py`: `GuardState`, `install_guard(bundle, registry, sink)`,
  `run_supervised(work, cleanup, runtime)`. There is no workload-facing reset,
  finalize, monitoring disable, or permission-grant API.
- `transport.py`: `encode_frame(record)`, `FrameReader.feed(bytes)`,
  `validate_terminal(observation)`, `supervise_child(request)`.
- `_terminal.fatal_blocked()` never returns and exits 70. T2 adds
  `_terminal.seal_and_exit(fd, exact_bytes, code)`; only exact bytes/int inputs,
  no coercion, formatting, GIL release, Python callback or post-marker return.
- `bootstrap.py`: `build_launch(config, bindings) -> LaunchRequest`,
  `verify_bootstrap(request) -> BootstrapEvidence`.

Trace records have exact keys `version`, `run_id`, `seq`, `kind`, `phase`,
`state`, `authority_id`, `operation_id`, `disposition`, `completion`.
Use null for not-applicable identifiers, not omitted keys. `completion` is
attempted/completed/raised/unknown. Separate record kinds cover startup, module
admission, operation, phase, violation, seal and terminal. They have explicit
schemas; they do not permit arbitrary error strings or arbitrary payload dicts.

Maximum frame bytes: 4096 and no greater than actual PIPE_BUF. Maximum trace:
64 MiB per worker. Timeout: existing 600 seconds. Any overflow or partial frame
is BLOCKED; never silently truncate. Parent owns all files. Raw stdout/stderr
is drained and discarded, with byte counts and approved diagnostic IDs only.

Bootstrap uses `-I -S -B`, the reviewed venv-prefix reconstruction, exact
sys.path, and a parent-built allowlisted environment. No `site.main`, `.pth`,
unknown preload, unreviewed alternate evaluator or extra tracing tool.

All construction/render restrictions remain those of the accepted spec. Renderer
metadata is limited to returned/type/shape/dtype and approved renderer identity.
No tensor, RGB, state, local, argument, result content, or pointer repr enters
trace, error records or the gate reports.

## 4. Contracts, payload and imports: execution-sized tasks

### C1 — Canonical primitives and nonempty gate validation

**Depends on:** reviewed G1-G4 PASS. **Create:**
`scripts/m1_guard/__init__.py`, `scripts/m1_guard/contracts.py`,
`tests/test_m1_guard_contracts.py`. **Inspect:** existing
`scripts/m1_renderer_admission.py::canonical_json`.

- [ ] Step 1: Add these first tests:

```python
import pytest
from scripts.m1_guard.contracts import (
    ContractError, canonical_bytes, validate_gate_results,
)

def test_canonical_json_and_empty_gate_rejection():
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ContractError):
        validate_gate_results({})
    validate_gate_results(dict.fromkeys(("G1", "G2", "G3", "G4"), "PASS"))

@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_noncanonical_values_are_not_serialized(value):
    with pytest.raises(ContractError):
        canonical_bytes({"value": value})
```

- [ ] Step 2: RED command:

```bash
"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_contracts.py
```

Expected initial missing-module error; with empty stubs, wrong canonical bytes
and failure to reject the empty gate set must produce assertion failures.

- [ ] Step 3: Implement recursive exact JSON-type checking, then the existing
  canonical JSON convention. Gate validation is deliberately small:

```python
def validate_gate_results(value):
    if type(value) is not dict or set(value) != {"G1", "G2", "G3", "G4"}:
        raise ContractError("gate-set")
    if any(type(v) is not str or v != "PASS" for v in value.values()):
        raise ContractError("gate-not-pass")
```

- [ ] Step 4: GREEN: rerun the same test command; require all assertions pass.
  Evidence impact: canonical contract bytes only, no empirical artifacts.
- [ ] Step 5: Verify and commit:

```bash
git diff --check
git add scripts/m1_guard/__init__.py scripts/m1_guard/contracts.py tests/test_m1_guard_contracts.py
git commit -m "feat(m1): add canonical guard contract primitives"
```

### C2 — Operation and dossier schemas with immutable bindings

**Depends on:** C1. **Modify:** `scripts/m1_guard/contracts.py`,
`tests/test_m1_guard_contracts.py`. **Inspect:** accepted spec identity, coverage
domain and native dossier clauses; existing exact-key admission validators.

- [ ] Step 1: Add a complete minimal operation fixture and mutation tests:

```python
from copy import deepcopy
from scripts.m1_guard.contracts import validate_operation, validate_dossier

def operation_record():
    return {"id": "render", "authority_id": "payload.fixture", "locator": [1],
            "disposition": "ALLOW", "phases": ["RENDER"],
            "category": "public_render", "count_limit": 1}

def test_operation_validation_detaches_input():
    raw = operation_record()
    checked = validate_operation(raw)
    raw["phases"].append("CLEANUP_ONLY")
    assert tuple(checked["phases"]) == ("RENDER",)

@pytest.mark.parametrize("patch", [
    {"count_limit": True}, {"count_limit": -1}, {"disposition": "MAYBE"},
    {"phases": []}, {"extra": 0}, {"authority_id": ""},
])
def test_invalid_operation_is_rejected(patch):
    raw = deepcopy(operation_record())
    raw.update(patch)
    with pytest.raises(ContractError):
        validate_operation(raw)

def test_empty_native_evidence_cannot_pass():
    raw = {"id": "native.fixture", "kind": "native", "entry_ids": ["f"],
           "member_ids": [], "semantic_exits": [], "conditions": {},
           "evidence": [], "limitations": [], "verdict": "PASS"}
    with pytest.raises(ContractError):
        validate_dossier(raw)
```

- [ ] Step 2: Run `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_contracts.py`.
  Behavioral RED: aliases mutate validated data or malformed records are accepted.
- [ ] Step 3: Implement Section 3 exact schemas, private immutable copies,
  duplicate/conflict rejection and strict JSON-key parsing. Bundle self-hash
  excludes only `bundle_sha256`; duplicate operation IDs never merge.
- [ ] Step 4: Run the same command GREEN. Add duplicate ID, FORBID/ALLOW conflict,
  missing semantic-exit coverage and self-hash mutation cases before committing.
  Evidence: validated schema v2 bindings, never actual audit PASS values.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/contracts.py tests/test_m1_guard_contracts.py
git commit -m "feat(m1): validate immutable catalog and dossier bindings"
```

### P1 — Retained source bytes and deterministic compile conditions

**Depends on:** C2. **Create:** `scripts/m1_guard/payload.py`,
`tests/test_m1_guard_payload.py`. **Inspect:** old `_GuardedLoader` only as a
negative reference; do not import or reuse its AST transformer.

- [ ] Step 1: Add the self-contained snapshot tests:

```python
import hashlib
import pytest
from scripts.m1_guard.payload import (
    SourceBinding, SourceBindingError, read_snapshot, compile_snapshot,
)

def bind(path):
    return SourceBinding("fixture", path, hashlib.sha256(path.read_bytes()).hexdigest())

def test_compile_uses_snapshot_not_reopened_path(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_bytes(b"VALUE = 7\n")
    snap = read_snapshot(bind(path))
    path.write_bytes(b"raise RuntimeError('substituted')\n")
    result = compile_snapshot(snap)
    namespace = {}
    exec(result.root, namespace, namespace)
    assert namespace["VALUE"] == 7

def test_source_drift_before_read_is_rejected(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_bytes(b"VALUE = 7\n")
    binding = bind(path)
    path.write_bytes(b"VALUE = 8\n")
    with pytest.raises(SourceBindingError):
        read_snapshot(binding)
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_payload.py`.
  The behavioral failure is executing the replacement bytes or accepting drift.
- [ ] Step 3: Read using one canonical regular-file descriptor; reject symlinks
  and mismatched hash. Retain exact bytes. Compile with
  `compile(snapshot.raw, str(binding.path), 'exec', flags=0,
  dont_inherit=True, optimize=0)`. Record interpreter/build and compiler contract.
  Stat metadata is diagnostic, not an immutability substitute. No AST rewrite.
- [ ] Step 4: Same command GREEN; add symlink/nonregular/encoding-cookie cases.
  Evidence: source SHA and compile binding; path replacement never supplies code.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/payload.py tests/test_m1_guard_payload.py
git commit -m "feat(m1): compile retained reviewed source bytes"
```

### P2 — Nested authority and same-object execution

**Depends on:** P1. **Modify:** payload module and its test file.

- [ ] Step 1: Add the actual collision test, using `bind` defined in P1:

```python
from scripts.m1_guard.payload import authorize, require_code, execute, PayloadError

def test_metadata_collision_does_not_authorize_code(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_bytes(b"def outer():\n    def inner():\n        return 7\n    return inner\n")
    payload = compile_snapshot(read_snapshot(bind(path)))
    authority = authorize(payload)
    namespace = {}
    assert execute(authority, payload, namespace) is payload.root
    inner = namespace["outer"]().__code__
    assert require_code(authority, inner) != ()
    with pytest.raises(PayloadError):
        require_code(authority, inner.replace())
    other = compile_snapshot(read_snapshot(bind(path)))
    with pytest.raises(PayloadError):
        execute(authority, other, {})
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_payload.py`.
  Equality-based registries fail because separately allocated colliding code
  can compare equal. The test must require identity, not merely a differing hash.
- [ ] Step 3: Recursively walk CodeType entries in `co_consts`, retain strong
  references and integer-index tuple locators. Lookup uses `is`, or id followed
  by `is`; no user equality/hash. `execute` requires the identical Payload and
  root then calls `exec` on that retained root.
- [ ] Step 4: Same command GREEN; test nested replacement before any execution.
  Evidence digest uses a typed canonical code tree, not marshal equality as
  authority: include bytecode, exception/line tables, names, vars/free/cell vars,
  flags/counts and typed constants. Sort frozen-set encodings; preserve float
  bit patterns; unknown constant types BLOCKED. Runtime authority remains `is`.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/payload.py tests/test_m1_guard_payload.py
git commit -m "feat(m1): bind nested execution to retained payload authority"
```

### P3 — Cache and non-direct-source disposition

**Depends on:** P2. **Modify:** payload module/tests. **Inspect:** pinned
importlib cache behavior and G2/G3 frozen/generated provenance findings.

- [ ] Step 1: Add explicit unsupported-kind rejection:

```python
from scripts.m1_guard.payload import require_payload_contract, UnsupportedPayloadError

@pytest.mark.parametrize("kind", ["sourceless", "frozen", "generated", "extension"])
def test_non_source_kind_without_proof_is_blocked(kind):
    with pytest.raises(UnsupportedPayloadError):
        require_payload_contract(kind, None)

def test_unrelated_cached_code_is_not_an_authority(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_bytes(b"VALUE = 7\n")
    payload = compile_snapshot(read_snapshot(bind(path)))
    authority = authorize(payload)
    malicious = compile("VALUE = 99", str(path), "exec")
    with pytest.raises(PayloadError):
        require_code(authority, malicious)
```

- [ ] Step 2: RED/GREEN command:
  `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_payload.py`.
  RED is implicit acceptance based on source hash, kind or code metadata.
- [ ] Step 3: Define `require_payload_contract(kind: str, proof: dict | None)`.
  Direct source uses P1/P2. Frozen, generated and extension paths require exact
  reviewed proofs from G1-G3, not a generic allow flag. Python caches are never
  executed. Generated dataclass/namedtuple code must match its reviewed generator,
  conditions and resulting source/payload binding before execution.
- [ ] Step 4: Add a real synthetic `.pyc` fixture with mismatched body and matching
  cache metadata; imports must execute verified source or reject, never cached
  body. Validate frozen/generated positive fixtures independently; no production
  path is allowed solely because synthetic proof passed.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/payload.py tests/test_m1_guard_payload.py
git commit -m "feat(m1): enforce explicit non-source payload dispositions"
```

### I1 — Exact module table, no unchecked finder fallback

**Depends on:** P3. **Create:** `scripts/m1_guard/imports.py`,
`tests/test_m1_guard_imports.py`. **Inspect:** rejected finder fall-through and
accepted exact-closure requirements.

- [ ] Step 1:

```python
import hashlib
import pytest
from scripts.m1_guard.payload import SourceBinding
from scripts.m1_guard.imports import ImportTable, ImportDenied, resolve_exact

def test_unknown_module_does_not_fall_through(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_bytes(b"VALUE = 1\n")
    binding = SourceBinding("fixture", path, hashlib.sha256(path.read_bytes()).hexdigest())
    table = ImportTable({"fixture": binding})
    assert resolve_exact(table, "fixture") is binding
    with pytest.raises(ImportDenied):
        resolve_exact(table, "fixture.unreviewed")
    with pytest.raises(ImportDenied):
        resolve_exact(table, "other")
```

- [ ] Step 2: Run `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_imports.py`.
  RED: unknown returns None or calls another finder.
- [ ] Step 3: Build an exact immutable table; no prefix grants. Finder decisions
  are either a fully admitted spec or a rejection. Existing bootstrap modules
  are checked against bootstrap provenance; all unexpected preloads reject.
- [ ] Step 4: Same command GREEN; add duplicate name, preload, missing origin,
  undeclared package/submodule and drifted package metadata cases.
  Evidence: attempted module IDs and pre-execution disposition only.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/imports.py tests/test_m1_guard_imports.py
git commit -m "feat(m1): resolve imports from an exact admitted module table"
```

### I2 — Loader executes only its retained payload

**Depends on:** I1. **Modify:** imports module/tests.

- [ ] Step 1:

```python
import types
from importlib.machinery import ModuleSpec
from scripts.m1_guard.payload import read_snapshot, compile_snapshot, authorize
from scripts.m1_guard.imports import AdmittedLoader, validate_loader

def test_custom_loader_cannot_execute_before_validation(tmp_path):
    touched = []
    class OtherLoader:
        def create_module(self, spec):
            touched.append("create")
        def exec_module(self, module):
            touched.append("exec")
    path = tmp_path / "fixture.py"
    path.write_bytes(b"VALUE = 1\n")
    binding = SourceBinding("fixture", path, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ImportDenied):
        validate_loader(ModuleSpec("fixture", OtherLoader(), origin=str(path)), binding)
    assert touched == []

def test_loader_never_reopens_execution_source(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_bytes(b"VALUE = 1\n")
    binding = SourceBinding("fixture", path, hashlib.sha256(path.read_bytes()).hexdigest())
    authority = authorize(compile_snapshot(read_snapshot(binding)))
    loader = AdmittedLoader(authority)
    path.write_bytes(b"VALUE = 99\n")
    module = types.ModuleType("fixture")
    loader.exec_module(module)
    assert module.VALUE == 1
```

- [ ] Step 2: Run imports tests RED: delegation executes OtherLoader or replaced
  path. Source drift may additionally invalidate final admission, but cannot
  change which retained payload is executed.
- [ ] Step 3: Check loader implementation identity, origin and source binding
  before calling any loader hook. Direct-source adapter compiles through P1/P2;
  create_module returns normal default allocation and exec_module uses only the
  retained authority. No original-loader execution, AST injection or class proxy.
- [ ] Step 4: Run imports and payload suites GREEN. Add a custom SourceFileLoader
  subclass regression: same class name/interface is not exact provenance.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/imports.py tests/test_m1_guard_imports.py
git commit -m "feat(m1): execute admitted source through a nondelegating loader"
```

### I3 — Direct exec, dynamic loader and generated-code entry checks

**Depends on:** I2. **Modify:** imports module/tests.

- [ ] Step 1:

```python
from scripts.m1_guard.imports import reject_unregistered_execution

def test_direct_execution_cannot_reuse_source_metadata(tmp_path):
    path = tmp_path / "fixture.py"
    path.write_bytes(b"VALUE = 1\n")
    binding = SourceBinding("fixture", path, hashlib.sha256(path.read_bytes()).hexdigest())
    authority = authorize(compile_snapshot(read_snapshot(binding)))
    forged = compile("VALUE = 2", str(path), "exec")
    with pytest.raises(ImportDenied):
        reject_unregistered_execution(authority, forged)
    reject_unregistered_execution(authority, authority.payload.root)
```

- [ ] Step 2: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_imports.py` must
  fail if matching filename/source metadata bypasses identity.
- [ ] Step 3: Add execution-authority checks at reviewed compile/exec/audit
  seams and bind native import/create entry contracts. A Python audit hook is
  supporting evidence, not a sandbox. Before monitoring exists, these are pure
  admission checks, not a claim to protect arbitrary live workload execution.
- [ ] Step 4: GREEN the focused suite; add custom finder, direct loader call,
  marshal payload and code.replace cases. End-to-end nonexecution moves to M2
  using the already validated fatal primitive, not an exception-only surrogate.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/imports.py tests/test_m1_guard_imports.py
git commit -m "feat(m1): bind alternate execution paths to payload authority"
```

### F1 — Promote G4 fatal primitive before monitoring enforcement

**Depends on:** C2, G4 PASS; complete before M1/M2. **Create:**
`scripts/m1_guard/terminal.c`, `scripts/build_m1_guard_terminal.py`,
`tests/test_m1_guard_fatal.py`. **Inspect:** G4 exact source/build/provenance.

- [ ] Step 1: Put the G4 source verbatim in terminal.c. First write this test:

```python
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_native_fatal_cannot_be_caught_or_finalize(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/build_m1_guard_terminal.py"),
                    "--output", str(tmp_path)], check=True, capture_output=True)
    source = (
        "import sys,atexit\n"
        f"sys.path.insert(0,{str(tmp_path)!r})\n"
        "import _terminal\n"
        "atexit.register(lambda: print('ATEXIT',flush=True))\n"
        "try:\n"
        "    try: _terminal.fatal_blocked()\n"
        "    except BaseException: print('CAUGHT',flush=True)\n"
        "finally: print('FINALLY',flush=True)\n"
        "print('AFTER',flush=True)\n"
    )
    result = subprocess.run([sys.executable,"-I","-S","-B","-c",source],
                            capture_output=True, timeout=10)
    assert result.returncode == 70
    assert result.stdout == b""
    assert result.stderr == b""
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_fatal.py`.
  Initial missing builder is scaffold RED; temporary test-only returning C body
  must demonstrate AFTER/FINALLY failure before restoring the reviewed G4 body.
- [ ] Step 3: Builder accepts only `--output`, requires an existing empty regular
  directory, pins `/usr/bin/cc`, include directory and G4 flags, and publishes
  one extension plus build.json without overwrite. It must reject extra compiler
  flags from environment and record source/compiler/header/dependency identities.
  Production startup loads only a previously built, hash-bound artifact; it
  never compiles. The fatal entry remains G4's no-argument `_exit(70)` function.
- [ ] Step 4: Same command GREEN, plus wrong binary hash/unloadable binary tests.
  Test outputs stay in tmp_path; no binary enters an empirical root or Git.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/terminal.c scripts/build_m1_guard_terminal.py tests/test_m1_guard_fatal.py
git commit -m "feat(m1): promote verified nonreturning fatal primitive"
```

## 5. Resolution and monitoring tasks

### R1 — Exact callable identity without user dispatch

**Depends on:** P2, F1. **Create:** `scripts/m1_guard/resolver.py`,
`tests/test_m1_guard_resolver.py`. **Inspect:** processor converters, pipeline
constructors and rejected `_call_category` to identify regressions, not rules.

- [ ] Step 1: Add a pure identity test:

```python
import pytest
from scripts.m1_guard.resolver import Registry, ResolutionError

def test_alias_and_bound_method_share_authority():
    class Item:
        def method(self):
            return 1
    instance = Item()
    registry = Registry()
    registry.bind_exact(Item.method, "forbidden.method", "FORBID")
    alias = instance.method
    assert registry.resolve_callable(alias).stable_id == "forbidden.method"
    assert registry.resolve_callable(alias).disposition == "FORBID"
    with pytest.raises(ResolutionError):
        registry.resolve_callable(lambda: 1)

def test_resolution_never_calls_custom_attribute_access():
    touched = []
    class Trap:
        def __getattribute__(self, name):
            touched.append(name)
            raise AssertionError("descriptor dispatch")
    with pytest.raises(ResolutionError):
        Registry().resolve_callable(Trap())
    assert touched == []
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_resolver.py`.
  Behavioral RED: aliases change judgment or resolver introspection executes Trap.
- [ ] Step 3: Add `Registry.bind_exact(target, stable_id, disposition)` as a
  private initialization operation; production construction additionally requires
  source/payload/dossier authority. Normalize only exact builtin MethodType
  through its builtin `__func__`; do not introspect arbitrary callable objects.
  Retain references and resolve with identity. Python function registration checks
  P2 authority before exposing the registry to the runtime.
- [ ] Step 4: Same command GREEN; add renamed __name__/__module__, separately
  compiled code and callable-object dispatch tests. No name-based permissions.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/resolver.py tests/test_m1_guard_resolver.py
git commit -m "feat(m1): normalize admitted callable identities without dispatch"
```

### R2 — Infrastructure membership cannot hide semantic targets

**Depends on:** R1, C2. **Modify:** resolver module/tests.

- [ ] Step 1:

```python
def test_outer_permission_cannot_override_forbidden_target():
    def outer():
        return 1
    def target():
        return 2
    registry = Registry()
    registry.bind_exact(outer, "infra.outer", "ALLOW")
    registry.bind_exact(target, "processor.convert", "FORBID")
    assert registry.resolve_callable(outer).disposition == "ALLOW"
    assert registry.resolve_callable(target).disposition == "FORBID"

def test_unregistered_stdlib_callable_has_no_implicit_permission():
    with pytest.raises(ResolutionError):
        Registry().resolve_callable(len)
```

- [ ] Step 2: Same focused command RED: a stdlib/module/outer-call shortcut grants
  permission. Add duplicate/conflicting binding rejection before implementation.
- [ ] Step 3: Implement exact infrastructure membership from dossiers. Each
  binding identifies dispatch conditions, semantic exits and supported phase;
  FORBID cannot be shadowed by infrastructure. Descriptor/partial/decorator
  outer entries never grant the target. Unknown native callback dispatch blocks
  the outer entry before execution if its dossier cannot bound the target.
- [ ] Step 4: GREEN resolver tests; add property getter, callable instance,
  functools.partial, decorated forbidden function and C-mediated callback
  identity tests. Their end-to-end nonexecution is verified in M2, not inferred
  from these resolver unit tests. Evidence: decision references, no arguments.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/resolver.py tests/test_m1_guard_resolver.py
git commit -m "feat(m1): enforce bounded infrastructure membership"
```

### M1 — Event adapter and attempt/completion pairing only

**Depends on:** F1, R2, I3. **Create:** `scripts/m1_guard/monitor.py`,
`tests/test_m1_guard_events.py`. **Inspect:** CPython 3.12 event signatures and
suppression semantics; no official dependency imports.

- [ ] Step 1: Define the narrow ledger interface and test it:

```python
from scripts.m1_guard.monitor import EventLedger

def test_call_and_python_entry_are_one_attempt():
    ledger = EventLedger()
    attempt = ledger.call("op.render", caller="payload.worker", offset=24)
    ledger.python_start("op.render", paired_attempt=attempt)
    ledger.python_return(attempt)
    assert ledger.counts("op.render") == {"attempted": 1, "completed": 1,
                                          "raised": 0, "unknown": 0}

def test_entry_without_python_call_is_not_zero_use():
    ledger = EventLedger()
    attempt = ledger.python_start("op.forbidden", paired_attempt=None)
    assert ledger.counts("op.forbidden")["attempted"] == 1
    assert ledger.counts("op.forbidden")["completed"] == 0
    assert ledger.counts("op.forbidden")["unknown"] == 1
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_events.py`.
  Behavioral RED: double counting or missing entry interpreted as zero attempts.
- [ ] Step 3: Implement `EventLedger` with a single-thread invocation stack,
  monotonic attempt IDs and separate attempted/completed/raised/unknown states.
  CALL/PY_START pairing requires the actual target authority and the matching
  dispatch relation, not merely adjacent IDs. Recursion, class construction,
  nested C-mediated entry and generator resumption receive distinct cases.
  Never inspect return content or persist arg0. Unmatched events cannot PASS.
- [ ] Step 4: GREEN events suite; add recursion, PY_UNWIND, native C_RAISE,
  missing completion and resume/yield cases. C_RETURN is not native safety proof.
  This commit alone does not enable workload execution.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/monitor.py tests/test_m1_guard_events.py
git commit -m "feat(m1): pair monitored operation attempts and completion"
```

### M2 — Real nonreturning violation from CALL and PY_START

**Depends on:** M1 and verified F1 binary. **Modify:** monitor module.
**Create:** `tests/test_m1_guard_violation.py`.

Internal adapter API: `make_rejecting_callbacks(resolve_call, resolve_code,
state, sink, fatal)` returns CALL/PY_START/PY_RESUME/PY_THROW callbacks. The
production installer supplies only verified resolver, state, sink and F1 fatal
identities. Dependency injection here is for unit tests, never a CLI bypass.
`ViolationState` exposes read-only state/evidence, not a reset method.

- [ ] Step 1: Add the following fully isolated native rejection test. The resolver
  is deliberately tiny test scaffolding; L1 later tests actual catalog admission.

```python
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("invocation", ["target()", "alias()", "partial(target)()"])
def test_rejection_cannot_resume_handlers(tmp_path, invocation):
    subprocess.run([sys.executable, str(ROOT / "scripts/build_m1_guard_terminal.py"),
                    "--output", str(tmp_path)], check=True, capture_output=True)
    source = f'''
import sys, os
sys.path[:0] = [{str(tmp_path)!r}, {str(ROOT)!r}]
from functools import partial
import _terminal
from scripts.m1_guard.monitor import make_rejecting_callbacks, ViolationState
def target():
    os.write(1, b"TARGET\\n")
alias = target
def by_call(value):
    return "FORBID" if value is target else "ALLOW"
def by_code(value):
    return "FORBID" if value is target.__code__ else "ALLOW"
callbacks = make_rejecting_callbacks(by_call, by_code, ViolationState(),
                                    lambda record: None, _terminal.fatal_blocked)
m = sys.monitoring
m.use_tool_id(5, "synthetic-guard")
for event, callback in callbacks.items():
    m.register_callback(5, event, callback)
m.set_events(5, m.events.CALL | m.events.PY_START | m.events.PY_RESUME | m.events.PY_THROW)
try:
    try:
        {invocation}
    except BaseException:
        os.write(1, b"CAUGHT\\n")
finally:
    os.write(1, b"FINALLY\\n")
os.write(1, b"AFTER\\n")
'''
    result = subprocess.run([sys.executable,"-I","-S","-B","-c",source],
                            capture_output=True, timeout=10)
    assert result.returncode == 70
    assert result.stdout == b""
    assert result.stderr == b""
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_violation.py`.
  Test an exception-only adapter first in the test harness: it must produce
  CAUGHT/FINALLY or non-70 exit and fail. Do not commit that adapter as production.
- [ ] Step 3: On unknown/FORBID, latch first; attempt only bounded safe diagnostic
  writing, then invoke F1 fatal. Diagnostic failure still invokes fatal. Do not
  throw back to workload; do not cleanup in the callback. PY_START handles tested
  C-mediated Python entries; unsupported native-before-entry effects are blocked
  by the outer dossier, not excused by this test.
- [ ] Step 4: GREEN violation suite, then add bound-method/decorator/property,
  map/operator.call, generator/coroutine resume, repeated catching and inline
  STORE side-effect variants. For every case assert target and continuation
  sentinels absent, not merely final BLOCKED. Callback corruption/no fatal
  binding prevents startup; it never selects an exception-only fallback.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/monitor.py tests/test_m1_guard_violation.py
git commit -m "feat(m1): enforce nonreturning guarded rejection"
```

### M3 — Supervisor-only cleanup transition

**Depends on:** M2. **Modify:** monitor module. **Create:**
`tests/test_m1_guard_supervisor.py`. **Inspect:** accepted conservative failure
choice; cleanup may run only after control is already safely in the supervisor.

- [ ] Step 1: Test the immutable transition validator separately from authority:

```python
import pytest
from scripts.m1_guard.monitor import check_transition, TransitionError

@pytest.mark.parametrize("edge", [
    ("VIOLATION_LATCHED", "RUNNING"), ("CLEANUP_ONLY", "RUNNING"),
    ("EXIT", "CLEANUP_ONLY"),
])
def test_reverse_transition_is_impossible(edge):
    with pytest.raises(TransitionError):
        check_transition(*edge)

def test_failure_transition_shapes():
    check_transition("RUNNING", "VIOLATION_LATCHED")
    check_transition("VIOLATION_LATCHED", "CLEANUP_ONLY")
    check_transition("VIOLATION_LATCHED", "EXIT")
    check_transition("CLEANUP_ONLY", "EXIT")
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_supervisor.py`.
  Then add actual workload-invoked supervisor and forged-context subprocess
  tests; merely validating an enum is not authority enforcement.
- [ ] Step 3: Require a registered supervisor code identity, exact legal call
  site, private monotonic call ledger and no active unproven workload frame.
  No caller-supplied boolean/token grants cleanup. Guard state remains private
  to the installed runtime; introspection/mutation entry paths are unadmitted.
  Unsupported attempts to reach state are fatal, not silently ignored.
  `run_supervised` may cleanup ordinary workload errors only after verified
  unwinding; a guard violation from M2 does not return to this function.
- [ ] Step 4: GREEN supervisor and violation suites. INSTRUCTION checks protect
  the frozen supervisor transition/sealing sites; PY_RESUME/PY_THROW cannot
  restore permissions. No general exception-stack recovery is implemented.
  Evidence: monotonic transitions and successful/failed cleanup, never re-PASS.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/monitor.py tests/test_m1_guard_supervisor.py
git commit -m "feat(m1): restrict cleanup authority to the admitted supervisor"
```

### M4 — Tool ownership and suppression-domain integrity

**Depends on:** M3. **Modify:** monitor module. **Create:**
`tests/test_m1_guard_integrity.py`.

- [ ] Step 1:

```python
import pytest
from scripts.m1_guard.monitor import validate_hook_snapshot, IntegrityError

def test_other_tool_or_trace_hook_blocks_admission():
    clean = {"tools": [None] * 6, "trace": False, "profile": False,
             "alternate_execution": False, "stale_events": False}
    validate_hook_snapshot(clean)
    for key in ("trace", "profile", "alternate_execution", "stale_events"):
        candidate = dict(clean, **{key: True})
        with pytest.raises(IntegrityError):
            validate_hook_snapshot(candidate)
    candidate = dict(clean, tools=[None, "another-tool", None, None, None, None])
    with pytest.raises(IntegrityError):
        validate_hook_snapshot(candidate)
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_integrity.py`.
- [ ] Step 3: Implement exact snapshot validation plus live checks. Tool 5 must
  be vacant and no unreviewed other hook present. After installation require
  owned ID, exact callback identities and event masks at boundaries. Never return
  DISABLE. Tampering operations reject before mutation. Bootstrap evidence, not
  Python introspection alone, must exclude hidden evaluator/native hooks.
- [ ] Step 4: GREEN, then subprocess tests for stale callback registration,
  freed-but-active tool state, second monitor, sys.settrace/sys.setprofile,
  callback failure and suppressed-callee dispatch. Suppressed callback callees
  are independently audited; absence of their events cannot satisfy zero-use.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/monitor.py tests/test_m1_guard_integrity.py
git commit -m "feat(m1): enforce monitoring ownership and suppression integrity"
```

## 6. Bootstrap, native dossiers and implicit callbacks

### B1 — Production launch from G2's reviewed startup contract

**Depends on:** M4, G2. **Create:** `scripts/m1_guard/bootstrap.py`,
`tests/test_m1_guard_bootstrap.py`.

- [ ] Step 1:

```python
import pytest
from scripts.m1_guard.bootstrap import validate_startup, BootstrapError

def test_startup_rejects_site_and_base_purelib():
    record = {"isolated": True, "no_site": True, "dont_write_bytecode": True,
              "prefix": "/public/home/xuyinghao/tmp/shiftvla-libero",
              "purelib": "/public/home/xuyinghao/tmp/shiftvla-libero/lib/python3.12/site-packages"}
    validate_startup(record)
    for patch in ({"no_site": False}, {"prefix": "/usr/local/python3.12"},
                  {"purelib": "/usr/local/python3.12/lib/python3.12/site-packages"}):
        with pytest.raises(BootstrapError):
            validate_startup(dict(record, **patch))
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_bootstrap.py`.
- [ ] Step 3: Implement G2's exact launch/prefix/path recipe and validate the
  immutable launch dossier before spawn. Parent rejects LD_PRELOAD/LD_AUDIT,
  unknown startup variables and mismatched loader libraries before execution.
  Retain the governed renderer/cache mapping and freeze any additional required
  environment keys explicitly. The first executable bootstrap bytes must be
  bound, not loaded from an unchecked path after the fact.
- [ ] Step 4: GREEN plus G2 sentinel suite translated into persistent tests.
  Assert no official package import in the parent/bootstrap before installation.
  Evidence: launch argv/env disposition, runtime binary/build and payload IDs.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/bootstrap.py tests/test_m1_guard_bootstrap.py
git commit -m "feat(m1): implement the reviewed isolated bootstrap contract"
```

### N1 — Enforce already-reviewed native dossiers, not defer feasibility

**Depends on:** B1, G1 PASS. **Create:** `scripts/m1_guard/native.py`,
`tests/test_m1_guard_native.py`. **Inspect:** actual G1 source/build dossiers.

- [ ] Step 1:

```python
import pytest
from scripts.m1_guard.native import check_binary_binding, NativeCoverageError

def test_binary_identity_is_required_but_not_sufficient():
    digest = "1" * 64
    with pytest.raises(NativeCoverageError):
        check_binary_binding(expected=digest, actual=digest, coverage=None)
    proof = {"verdict": "PASS", "evidence": ["reviewed-dossier-id"], "unresolved": []}
    with pytest.raises(NativeCoverageError):
        check_binary_binding(expected=digest, actual="2" * 64, coverage=proof)
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_native.py`.
- [ ] Step 3: Implement `check_binary_binding` and integrate exact native entry
  resolution with the reviewed transitive/build evidence. Real evidence IDs must
  resolve and hash-validate; the string in this negative test is not production
  proof. No file-extension, version or C_RETURN-based allow shortcut.
- [ ] Step 4: GREEN missing/broad/unresolved-callback/build-drift cases. Revalidate
  G1 input hashes; any drift sends the implementation back to the gate, not an
  automatic dossier update. Evidence: validated actual native identities.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/native.py tests/test_m1_guard_native.py
git commit -m "feat(m1): enforce frozen native coverage dossiers"
```

### A1 — Callback quiescence and reviewed cleanup entries

**Depends on:** N1. **Create:** `scripts/m1_guard/callbacks.py`,
`tests/test_m1_guard_callbacks.py`. **Inspect:** robosuite EGL atexit/__del__,
G1 callback graph and G4 termination assumptions.

- [ ] Step 1:

```python
import pytest
from scripts.m1_guard.callbacks import require_quiescence, CallbackCoverageError

@pytest.mark.parametrize("pending", ["weakref", "finalizer", "gc", "signal", "native"])
def test_unknown_pending_callback_blocks_terminal(pending):
    record = {"covered": [], "excluded": [], "unknown": [pending]}
    with pytest.raises(CallbackCoverageError):
        require_quiescence(record)

def test_empty_record_is_not_a_quiescence_proof():
    with pytest.raises(CallbackCoverageError):
        require_quiescence({})
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_callbacks.py`.
- [ ] Step 3: Validate callback absence/coverage against bound G1 dossiers and
  actual cleanup completion. Reviewed finalizers may run only in their permitted
  guarded phases; no arbitrary collection callback is admitted. Do not equate
  gc.disable with disabling reference-release finalizers. Audit held references,
  signals and pending native work through the terminal handoff.
- [ ] Step 4: GREEN plus real synthetic weakref/finalizer/deferred-signal tests
  using pipes. Native-callback coverage tests use synthetic mocks unless the
  native fixture itself has G1/G4 provenance; never import actual renderer here.
  Evidence: quiescence contract, unresolved callbacks always BLOCKED.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/callbacks.py tests/test_m1_guard_callbacks.py
git commit -m "feat(m1): gate terminal handoff on callback coverage"
```

## 7. Terminal transport and parent validation

### T1 — Bounded framing independent of environment code

**Depends on:** A1. **Create:** `scripts/m1_guard/transport.py`,
`tests/test_m1_guard_transport.py`. **Inspect:** existing no-overwrite writers;
do not reuse null result-file transport as a stream.

- [ ] Step 1: Introduce a self-contained trace record fixture:

```python
import pytest
from scripts.m1_guard.transport import encode_frame, FrameReader, TransportError

def trace_record(seq=0):
    return {"version": 2, "run_id": "synthetic-1", "seq": seq,
            "kind": "operation", "phase": "RENDER", "state": "RUNNING",
            "authority_id": "payload.fixture", "operation_id": "render",
            "disposition": "ALLOW", "completion": "attempted"}

def test_frame_can_be_read_in_arbitrary_chunks():
    record = trace_record()
    raw = encode_frame(record)
    reader = FrameReader(max_frame=4096, max_total=67108864)
    assert reader.feed(raw[:3]) == []
    assert reader.feed(raw[3:]) == [record]
    reader.finish()

def test_partial_frame_is_not_a_complete_empty_trace():
    reader = FrameReader(max_frame=4096, max_total=67108864)
    reader.feed(encode_frame(trace_record())[:-1])
    with pytest.raises(TransportError):
        reader.finish()
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_transport.py`.
  Behavioral RED: partial input is silently accepted or returned as an empty run.
- [ ] Step 3: Use a four-byte unsigned big-endian payload length followed by
  canonical UTF-8 JSON. Total frame, including prefix, is at most 4096 and
  validated PIPE_BUF. Reject duplicate keys, unknown schema, oversized declared
  length, total-byte overflow and trailing incomplete input. Writer uses one
  nonblocking os.write per frame; short/EAGAIN/EPIPE is fatal, not a retry.
- [ ] Step 4: GREEN plus duplicate sequence, bad length, malformed UTF-8,
  malicious field, integer overflow and max-total boundary tests. Evidence:
  parent hash is over exact accepted frame bytes; no arbitrary object repr.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/transport.py tests/test_m1_guard_transport.py
git commit -m "feat(m1): add bounded canonical guard framing"
```

### T2 — Native seal-and-exit, preserving F1 failure semantics

**Depends on:** T1, F1, A1. **Modify:** `scripts/m1_guard/terminal.c`,
builder manifest handling. **Create:** `tests/test_m1_guard_terminal.py`.

- [ ] Step 1: Add a subprocess test using the F1 builder, an inherited pipe and
  explicit sentinel. This test isolates the primitive; it does not replace the
  later validation of an actual terminal record:

```python
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_marker_write_has_no_python_return(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts/build_m1_guard_terminal.py"),
                    "--output", str(tmp_path)], check=True, capture_output=True)
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    source = (f"import sys,os; sys.path.insert(0,{str(tmp_path)!r}); "
              f"import _terminal; _terminal.seal_and_exit({write_fd},b'MARKER',0); "
              "os.write(1,b'AFTER')")
    try:
        result = subprocess.run([sys.executable,"-I","-S","-B","-c",source],
                                pass_fds=(write_fd,), capture_output=True, timeout=10)
        os.close(write_fd)
        write_fd = -1
        assert os.read(read_fd, 4096) == b"MARKER"
        assert os.read(read_fd, 1) == b""
        assert result.returncode == 0
        assert result.stdout == b""
        assert result.stderr == b""
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_terminal.py`.
  Behavioral RED: a returning implementation reaches AFTER. Missing API alone
  does not demonstrate the nonreturn guarantee.
- [ ] Step 3: Keep `fatal_blocked` unchanged. Add this terminal body and register
  it as METH_VARARGS; include `<limits.h>`. No Python API after the final write:

```c
static PyObject *seal_and_exit(PyObject *self, PyObject *args) {
    (void)self;
    if (PyTuple_GET_SIZE(args) != 3) _exit(71);
    PyObject *fd_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *data = PyTuple_GET_ITEM(args, 1);
    PyObject *code_obj = PyTuple_GET_ITEM(args, 2);
    if (!PyLong_CheckExact(fd_obj) || !PyBytes_CheckExact(data) ||
        !PyLong_CheckExact(code_obj)) _exit(71);
    long fd = PyLong_AsLong(fd_obj);
    long code = PyLong_AsLong(code_obj);
    if (PyErr_Occurred() || fd < 0 || fd > INT_MAX ||
        (code != 0 && code != 70 && code != 71)) _exit(71);
    Py_ssize_t length = PyBytes_GET_SIZE(data);
    if (length <= 0 || length > 4096) _exit(71);
    ssize_t written = write((int)fd, PyBytes_AS_STRING(data), (size_t)length);
    _exit(written == length ? (int)code : 71);
}
```

Production binding proves FD identity, nonblocking status and PIPE_BUF before
entry. An invalid invocation terminates 71; it does not call argument coercion
or return a catchable error. The C path does not release the GIL or run handlers.
Native signal/work assumptions still require A1/G4 coverage.

- [ ] Step 4: GREEN terminal and fatal suites. Add full-pipe, closed-reader,
  wrong types, oversize, zero bytes and post-marker atexit/finalizer tests.
  Rebuild and independently review the new binary/dossier; the old binary hash
  is not silently reused. Re-run every F1 fatal assertion against the new binary.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/terminal.c scripts/build_m1_guard_terminal.py tests/test_m1_guard_terminal.py
git commit -m "feat(m1): seal terminal evidence through a no-return primitive"
```

### T3 — Parent drains channel and reaps exactly its child

**Depends on:** T2. **Modify:** transport module/tests.
**Inspect:** null runner only for differences; its result file is not adopted.

- [ ] Step 1: Test the necessary terminal observation relation:

```python
from scripts.m1_guard.transport import validate_process_end

@pytest.mark.parametrize("reaped,eof,code", [
    (False, True, 0), (True, False, 0), (False, False, 0), (True, True, 70),
])
def test_process_end_cannot_pass_without_reap_and_eof(reaped, eof, code):
    with pytest.raises(TransportError):
        validate_process_end(reaped=reaped, eof=eof, returncode=code)

def test_zero_exit_and_eof_are_necessary_not_sufficient():
    assert validate_process_end(reaped=True, eof=True, returncode=0) is None
```

- [ ] Step 2: RED/GREEN command:
  `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_transport.py`.
  This helper's None return is not an AdmissionVerdict.
- [ ] Step 3: Implement `supervise_child(request)` with one Popen, `close_fds`,
  only required pass_fds, exact argv/env and PID/start identity capture. Parent
  closes its write end immediately, uses selectors to drain evidence/stdout/
  stderr concurrently and waits for its child. Deadline is monotonic and bounded;
  timeout terminates/reaps the child and yields BLOCKED, with no replacement.
  Descendant FD leakage, death without EOF or EOF without legal death blocks.
- [ ] Step 4: Add actual subprocess tests: early writer close, inherited writer
  kept open by a child, large trace, timeout, pipe break and exit 0 without marker.
  No test uses real factory/import/render. GREEN requires no deadlock and no PASS
  on incomplete transport. Raw console bytes are not persisted as diagnostics.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/transport.py tests/test_m1_guard_transport.py
git commit -m "feat(m1): supervise guard channel through child death and EOF"
```

### V1 — Parent derives verdict from complete evidence

**Depends on:** T3, C2. **Create:** `scripts/m1_guard/evidence.py`,
`tests/test_m1_guard_evidence.py`.

API: `validate_observation(expected: dict, records: list[dict], terminal:
dict | None, process: dict) -> dict`. Result exact keys: `status`, `reason_ids`,
`verified_hashes`, `counts`, `process`. Only this parent API can return final
PASS; child record type never contains a final PASS verdict.

- [ ] Step 1: Add complete negative baseline input:

```python
import pytest
from scripts.m1_guard.evidence import validate_observation

def expected_contract():
    return {"run_id": "synthetic-1", "bundle_sha256": "1" * 64,
            "pid": 1234, "start_identity": "synthetic-start",
            "required_counts": {"factory": 1, "lazy_construction": 1,
                                "inner_reset": 1, "public_render": 1, "close": 1},
            "forbidden_counts": {"step": 0, "processor": 0, "policy": 0}}

@pytest.mark.parametrize("terminal", [None, {}, {"status": "PASS"}])
def test_empty_evidence_never_becomes_pass(terminal):
    verdict = validate_observation(expected_contract(), [], terminal,
                                  {"reaped": True, "eof": True, "returncode": 0})
    assert verdict["status"] == "BLOCKED"
    assert verdict["reason_ids"]
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_evidence.py`.
  Behavioral RED: all([]), child status, or exit=0 produces PASS.
- [ ] Step 3: Validate schema, expected PID/start/run, exact module closure,
  catalog/payload/dossier/bootstrap bindings, ordered phases, every attempted
  and completed count, latch state, cleanup, terminal sequence/digest, death and
  EOF. Recompute counters from trace; compare with summary. Required missing
  completion remains unknown and blocks. Semantic forbidden set is the full
  accepted catalog; the three-entry fixture is only a minimal negative case.
- [ ] Step 4: Construct the positive fixture by collecting a real synthetic
  guarded worker through T3, not by hand-setting summary PASS. Mutate one field
  at a time: marker sequence/digest, PID/start, hash, dossier, phase, unknown
  completion, missing/duplicate event, overflow and unresolved operation. Every
  mutant must BLOCK. Run evidence and transport suites GREEN.
  Evidence: parent-owned decision and exact input/trace hashes.
- [ ] Step 5:

```bash
git diff --check
git add scripts/m1_guard/evidence.py tests/test_m1_guard_evidence.py
git commit -m "feat(m1): derive final admission verdict in the parent"
```

### V2 — Versioned config and exact no-overwrite publication

**Depends on:** V1. **Modify:** `scripts/m1_renderer_admission.py`,
`tests/test_m1_renderer_admission.py`. **Create after real bindings exist:**
`configs/m1/renderer_preflight_r1_guard_v2_egl0.yaml`.
**Inspect:** old config, `_PUBLICATION`, `validate_admission_config`, predecessor
manifest and governed environment validators.

- [ ] Step 1: Add new execution-gate test without modifying the v1 reader:

```python
import pytest

def test_v1_contract_is_not_authorized_for_guarded_execution():
    admission = _module()
    with pytest.raises(admission.AdmissionError):
        admission.require_guarded_execution_contract({"schema_version": 1})
```

`_module()` already exists in this test file; do not invent a second loader.

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_renderer_admission.py`.
- [ ] Step 3: Add the explicit v2 execution gate. Retain v1 historical reading
  and tests unchanged. v2 includes guard bundle/hash, launch/terminal bindings,
  all gate-report hashes and compiled primitive identity. Required missing
  bindings reject; never create a config filled with placeholder hashes.
  Preserve all task/renderer/count/predecessor values from the accepted workload.
- [ ] Step 4: Extend the artifact allowlist by exactly these eight names:
  `probe-000.guard.json`, `probe-000.trace.jsonl`, `probe-001.guard.json`,
  `probe-001.trace.jsonl`, `probe-002.guard.json`, `probe-002.trace.jsonl`,
  `preflight.guard.json`, `preflight.trace.jsonl`. Retain existing names.
  Trace JSONL is a parent rendering of validated frames; record both raw-wire
  digest and rendered-file digest, never substitute one for the other.
- [ ] Step 5: GREEN admission suite: unlisted filename, nested/symlink target,
  existing output, self-hash mismatch, missing gate and invalid artifact schema
  must reject before directories/publications are created. v2 config is committed
  only when all real reviewed bindings have been computed and verified.
- [ ] Step 6:

```bash
git diff --check
git add scripts/m1_renderer_admission.py tests/test_m1_renderer_admission.py
git commit -m "feat(m1): bind guarded admission to an exact v2 contract"
```

V2 commits schema support only. Its tests use explicitly synthetic bindings in
tmp_path. Q1 creates and commits the real v2 configuration after auditing and
freezing the actual manifests. L1-L2 use synthetic configuration; they must not
depend on an invented production catalog hash. Missing real bindings blocks Q1,
not these synthetic schema tests.

## 8. Synthetic integration and final review

### L1 — One synthetic worker through the full guard

**Depends on:** V2. **Create:** `scripts/m1_renderer_preflight.py`,
`tests/test_m1_renderer_preflight.py`.
**Inspect:** accepted factory seam, LiberoEnv lazy initialization/render/close,
existing metadata-only contract and backend resource ownership.

New internal API: `execute_single_render(factory, metadata, runtime)`.
Production factory is bound to the unchanged
`dcu_preflight.build_cpu_environment_runtime`, not supplied from CLI. Test
dependency injection is restricted to the test harness and cannot be selected
by an empirical argument/config field.

- [ ] Step 1: Add a test that proves call ordering without any environment:

```python
from scripts.m1_renderer_preflight import execute_single_render

def test_worker_has_one_render_and_closes_before_seal():
    events = []
    class Env:
        def render(self):
            events.append("render")
            return object()
        def close(self):
            events.append("close")
    class Runtime:
        def enter(self, phase):
            events.append(phase)
        def seal(self, record):
            events.append("seal")
            return record
    def factory():
        events.append("factory")
        return Env()
    def metadata(value):
        return {"returned": True, "type": "synthetic", "shape": [], "dtype": None}
    execute_single_render(factory, metadata, Runtime())
    assert events == ["FACTORY", "factory", "RENDER", "render",
                      "TEARDOWN", "close", "TERMINAL", "seal"]
```

- [ ] Step 2: RED: `"$GUARD_PY" -B -m pytest -q tests/test_m1_renderer_preflight.py`.
  Behavioral RED: retry, extra render or seal before close changes the sequence.
- [ ] Step 3: Implement only this narrow worker sequence and failure routing.
  Test Runtime.seal returning is a unit stub only: the production runtime seal
  invokes T2 and cannot return. Guard installs before official imports/factory;
  factory result unwraps exactly one synchronous LiberoEnv. Preserve original
  factory parameters; never instantiate RuntimeAdapter or call public reset.
- [ ] Step 4: Add full subprocess tests using source-bound fake modules,
  generated fake catalog/dossiers and the actual bootstrap/guard/native terminal.
  The fake source has the same logical lazy-construction/inner-reset call graph.
  Test forbidden step/action/settle/reset/processor/policy/replay paths, cleanup
  failure, missing metadata and unsupported callbacks. No real dependencies.
- [ ] Step 5: GREEN preflight, events, violation, evidence and transport suites.
  Every full integration record must be marked synthetic and written to tmp_path.
  Evidence: synthetic counts and terminal proof, never empirical PASS.
- [ ] Step 6:

```bash
git diff --check
git add scripts/m1_renderer_preflight.py tests/test_m1_renderer_preflight.py
git commit -m "feat(m1): implement the guarded single-render worker seam"
```

### L2 — Synthetic parent sequencing and unreachable schedule path

**Depends on:** L1. **Modify:** preflight entrypoint/tests.
**Inspect:** `verify_three_probe_records`, existing probe count contract and
forbidden null dispatch names.

- [ ] Step 1: Add parent sequencing test. Interface:
  `run_admission_sequence(run_probe, verify_probes, run_preflight)`.

```python
import pytest
from scripts.m1_renderer_preflight import run_admission_sequence

def test_failed_probe_verification_never_runs_preflight():
    events = []
    def probe(index):
        events.append(index)
        return {"index": index}
    def verify(records):
        raise ValueError("synthetic invalid probe")
    def preflight():
        events.append("preflight")
    with pytest.raises(ValueError):
        run_admission_sequence(probe, verify, preflight)
    assert events == ["000", "001", "002"]
```

- [ ] Step 2: RED/GREEN command:
  `"$GUARD_PY" -B -m pytest -q tests/test_m1_renderer_preflight.py`.
  RED is preflight dispatch despite invalid probes, reordered probes or retries.
- [ ] Step 3: Production adapters use B1/T3 for three distinct fresh probe
  children, existing backend functions and exact three-probe verification, then
  one distinct fresh single-render child. No synthetic callback injection is
  accepted from CLI. Define only parent admission and private worker roles;
  private role requires parent channel/run identity, not a reusable text token.
- [ ] Step 4: Add success-order, probe failure, worker timeout, reused PID/start
  identity, zero/multiple render and forbidden CLI-mode tests. Parse the entrypoint
  source/dependency graph to assert no `prepare_run`, materializer, calibration
  runner, RuntimeAdapter or replay dispatch. CLI help/config validation must not
  spawn children or create empirical roots.
- [ ] Step 5: Run preflight and existing admission suites GREEN. Completion of
  this task does not execute the parent against a real environment.
- [ ] Step 6:

```bash
git diff --check
git add scripts/m1_renderer_preflight.py tests/test_m1_renderer_preflight.py
git commit -m "feat(m1): sequence guarded admission without schedule dispatch"
```

### Q1 — Freeze actual evidence, review, and stop before empirical execution

**Depends on:** L2 and every prior gate/task. **Inspect:** accepted spec, all
changed source/test files, gate reports, actual G1-G4 bindings, predecessor tree.
**Future files:** `runtime/m1/guard_r1/catalog.json`, `payloads.json`,
`imports.json`, `infrastructure.json`, `native.json`, `bootstrap.json`,
`terminal.json`, `bundle.json`; update `docs/m1_null_calibration.md` only now.

- [ ] Step 1: Review static source-derived candidate closure and source/native
  dossiers. The old 55-module candidate inventory is not a runtime closure.
  Missing dynamic/generated/native paths are blockers, not empty evidence.
- [ ] Step 2: Only after candidate admission review, perform a separately scoped
  guarded official import-only audit. No environment construction, EGL context,
  renderer, schedule or empirical output. Reject new modules before their bodies;
  any newly required permission returns to source review. No auto-learning.
- [ ] Step 3: Freeze the exact audited/observed closure, actual operation catalog
  and all real dossier/source/binary hashes. Runtime object addresses never enter
  the files. Commit source-bound manifests without self-referential commit hashes;
  bind the final execution commit from the parent launch attestation after commit.
- [ ] Step 4: Run focused and existing regressions:

```bash
"$GUARD_PY" -B -m pytest -q tests/test_m1_guard_contracts.py tests/test_m1_guard_payload.py tests/test_m1_guard_imports.py tests/test_m1_guard_fatal.py tests/test_m1_guard_resolver.py tests/test_m1_guard_events.py tests/test_m1_guard_violation.py tests/test_m1_guard_supervisor.py tests/test_m1_guard_integrity.py tests/test_m1_guard_bootstrap.py tests/test_m1_guard_native.py tests/test_m1_guard_callbacks.py tests/test_m1_guard_transport.py tests/test_m1_guard_terminal.py tests/test_m1_guard_evidence.py tests/test_m1_renderer_preflight.py
"$GUARD_PY" -B -m pytest -q tests/test_m1_renderer_admission.py tests/test_m1_null_calibration.py tests/test_m1_state_replay.py
git diff --check
```

- [ ] Step 5: Revalidate predecessor raw/semantic hashes through existing
  `verify_predecessor_manifest`; verify no empirical roots/config/schedule were
  created. Require an independent implementation/spec review when available;
  unavailable is recorded as unavailable, never PASS. Main agent owns the final
  judgment and must not silently waive a required acceptance gate.
- [ ] Step 6: Stage only the eight verified manifests and status document:

  Before staging, generate `configs/m1/renderer_preflight_r1_guard_v2_egl0.yaml`
  from the preserved v1 workload values and the now-frozen guard manifest hashes;
  recompute its self-hash and verify through the V2 validator. Include this exact
  additional file in the commit below, never placeholder values:

```bash
git add configs/m1/renderer_preflight_r1_guard_v2_egl0.yaml
```

```bash
git add runtime/m1/guard_r1/catalog.json runtime/m1/guard_r1/payloads.json runtime/m1/guard_r1/imports.json runtime/m1/guard_r1/infrastructure.json runtime/m1/guard_r1/native.json runtime/m1/guard_r1/bootstrap.json runtime/m1/guard_r1/terminal.json runtime/m1/guard_r1/bundle.json docs/m1_null_calibration.md
git commit -m "docs(m1): freeze reviewed guard contracts and implementation evidence"
```

- [ ] Step 7: Bind final reviewed execution commit and bundle SHA to the original
  R1 provenance protocol. Verify all protocol prerequisites independently; do
  not call this intermediate guard work execution identity P if other required
  admission implementation is still missing. Stop and hand off. No real Renderer
  Preflight command, null materialization, schedule creation or push is executed
  automatically by this plan.

## 9. Acceptance matrix, stop conditions and review ledger

| Normative obligation | Component / tasks | Evidence | Acceptance / failure |
|---|---|---|---|
| Exact semantics and audited infrastructure | C1-C2, R1-R2 | catalog + dossier IDs | Unknown, conflict or outer inheritance blocks |
| Source -> compiler -> actual executable | G3, P1-P3 | retained-byte and nested-authority records | Cache/metadata/TOCTOU forgery never executes |
| Before-body module/loader check | I1-I3 | module, loader and execution authority | No unchecked fallback or direct-exec bypass |
| CALL/PY_START and resumption | M1-M2 | paired attempts/completions | No target body or ordinary continuation after rejection |
| Irreversible failure + supervisor cleanup | G4, F1, M2-M3 | fatal exit + state ledger | No catch-to-resume or forged cleanup privilege |
| Monitoring suppression and coexisting hooks | G2, M4 | startup and callback coverage | Tool ownership alone is insufficient |
| Native source/build/transitive coverage | G1, N1 | actual binary/build dossiers | Missing required coverage blocks before C1/entry |
| Earliest startup authority | G2, B1 | launch/interpreter/bootstrap provenance | Fresh PID/inventory cannot replace prior-execution proof |
| Implicit/asynchronous entries | A1 | callback/quiescence dossiers | Unknown finalizer/signal/native work blocks |
| Controlled terminal | T1-T3 | marker, raw-wire digest, EOF/death | No post-marker Python/workload return |
| Parent-only final judgment | V1-V2 | recomputed counts/hash-valid observation | Child status/zero exit/empty set cannot PASS |
| Narrow official factory and one render | G1, L1-L2 | source map + synthetic integration | No changed factory/reset/render/observation scope |
| Immutable predecessor / no-overwrite | C1, V2, Q1 | original manifests/tree hashes | No mutation, retry, replacement or new null schedule |

### 9.1 Mandatory adversarial coverage checklist

Each checkbox refers to the owning task's test file, not a promise to invent
tests after implementation. A task remains incomplete until its listed cases
have actual assertions and observed RED/GREEN evidence.

- [ ] P1-P3: mismatched cached payload; copied filename/qualname/location;
  code.replace/nested substitution; validation/use substitution; unknown frozen,
  sourceless and generated provenance.
- [ ] R1-R2/M2: alias; renamed reference; bound method; decorator; descriptor;
  partial; map/operator.call; C-mediated Python entry; unbound native target.
- [ ] I1-I3/M2: alternate loader; custom finder; preload; opaque loader;
  create_module side effect; manual exec_module; execution before validation.
- [ ] F1/M2-M3: BaseException catch; inline mutation; repeated catch; finally;
  generator/coroutine resume and throw; workload cleanup call; supervisor forgery;
  no fatal binding; callback sink failure; no temporary exception-only GREEN.
- [ ] M4/B1: callback failure/tampering; stale ID/events; another monitoring tool;
  tracing/profiling; alternative evaluator; callback-suppressed callees; executable
  startup hooks before application bootstrap.
- [ ] G1/N1/A1: missing native dossier; broad helper; unresolved callback;
  source/build/binary drift; weakref/finalizer/GC/deferred signal/pending native work.
- [ ] T1-T3/V1: early/duplicate/missing marker; post-marker workload; partial/full
  pipe; channel loss; timeout; death without EOF; EOF without valid death; zero
  exit with incomplete evidence; incomplete cleanup; child forged summary.
- [ ] C1/V1-V2: empty aggregation; missing completion; trace overflow/truncation;
  inconsistent hashes; missing dossier; unresolved operation; forbidden artifact
  content; overwrite and symlink publication attempts.

### 9.2 Stop conditions

Immediately mark IMPLEMENTATION PLAN BLOCKED and stop dependent work if:

1. Any G1-G4 mandatory edge remains unknown, fails a counterexample or lacks
   required provenance. Four report files alone do not satisfy the gate.
2. Executable payload/source or validation/use correspondence cannot be
   established before execution, including necessary generated/frozen startup code.
3. The actual fatal primitive can return, be caught, invoke workload callbacks,
   or cannot be bound to its reviewed build. Do not patch monitoring around it.
4. The official factory needs prohibited semantic behavior; necessary asset
   deserialization cannot be bounded; or a proposed proof requires forbidden
   state/observation inspection. Do not change init_states or substitute a factory.
5. Startup, native callback, suppression or terminal-tail authority cannot be
   closed without weakening accepted constraints.
6. A prerequisite test fails, evidence is incomplete, or a reviewer identifies
   an accepted-spec contradiction. Document the precise issue; no silent redesign.

### 9.3 Plan-review checklist (main agent)

- [ ] Accepted commit and original factory remain authoritative.
- [ ] Four Phase-0 gates precede production tasks; none are marked run by this plan.
- [ ] F1 precedes M2; T2 extends but never substitutes failure semantics.
- [ ] Each task has explicit files, interfaces, a first test, RED/GREEN command,
  minimal implementation responsibility and exact staging/commit command.
- [ ] Test snippets introduce helpers before use and preserve real failures;
  no undefined fixture or xfail silently fills a proof gap.
- [ ] All normative clauses map to tasks/evidence/failure and acceptance.
- [ ] No empirical command, file cleanup or implicit Git push is included.

### Main-agent judgment for this persisted revision

**Implementation Plan = REVISE, awaiting the user's plan-level review.**

The document is now persisted as a revision rather than claiming READY based on
an in-chat roadmap. No Phase-0 gate has been executed or accepted by writing it.
After plan review, Phase 0 is the only authorized first execution block. A failed
gate yields **IMPLEMENTATION PLAN BLOCKED** with exact evidence gaps; passing
all gates is required before beginning the main implementation tasks.

Guard Design remains ACCEPT. Guard Implementation remains NOT STARTED.
Renderer Preflight remains NOT RUN; M1-N0 remains BLOCKED; new schedule remains
NOT CREATED. This plan neither reopens design nor grants empirical authorization.
