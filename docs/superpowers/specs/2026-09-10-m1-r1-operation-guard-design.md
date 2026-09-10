# M1-N0-R1 source-bound operation guard revision

## Authority, status, and scope

The user approved this direction on 2026-09-10: a source-bound explicit
operation catalog, external `sys.monitoring` observation, and a fail-closed
native boundary. This document materializes that approval; it does not approve
the existing guard implementation or authorize an empirical run.

Current statuses remain:

- Guard design = REVISE.
- Renderer Preflight = NOT RUN.
- M1-N0 = BLOCKED.
- New schedule = NOT CREATED.

Written-spec checkpoint: the user's written review returned REVISE, not
NOT_ACCEPTED. The three amendments below address coverage-domain scope,
irreversible enforcement, and controlled termination. Revised written-spec
acceptance is pending; no implementation plan or empirical execution is
authorized by this amendment. The earlier unavailable independent review
is not counted as PASS.

The subsequent executable-payload, monitoring-suppression, earliest-startup,
and implicit/asynchronous callback amendments below are mandatory design
constraints. Their inclusion is not evidence of implementation or user ACCEPT.

The 2026-09-08 renderer design remains authoritative for task, environment,
operation counts, renderer identities, predecessor immutability, and single-shot
execution. This revision supersedes the 2026-09-09 guard mechanism wherever it
relies on module permission or name heuristics as evidence of allowed use.
It also supersedes child-side early guard restoration: production protection
must cover the workload through the controlled terminal boundary defined below,
not merely until render returns. Parent-observed death remains required for PASS.

No third-party source or bytecode rewriting is selected. No ReplayState,
replay, CC/CS/SC/SS, M2, physics tolerance, or old bundle changes are in scope.
All execution and review subagents use GPT-5.6-luna / max, not speed routing.

## Evidence motivating the revision

The prior synthetic suite passed 26 tests, but independent specification
review was NOT_ACCEPTED. Confirmed gaps were unclassified processor operations,
unlisted import-time effects, alternate-loader execution before instrumentation,
custom `__new__` running before an `__init__` guard, and early `finalize()`
disarming an otherwise active scope. These are harness defects, not renderer
or null-calibration observations. Existing tests are retained as regressions,
not as proof of complete coverage.

## Layer 1: explicit source-bound operation catalog

### Identity and rules

Start with the legal Renderer Preflight execution path in the pinned installed
sources, not with a list learned from successful runtime calls. Each operation
has a stable ID and, at minimum, an exact module, qualified callable identity,
and source-file SHA-256. A binding also records the canonical installed path,
owning pinned revision, callable kind, allowed lifecycle phase, disposition,
operation category, and any required count or boundary evidence.

Python code identities additionally bind the exact interpreter and deterministic
code location. Qualname alone is insufficient: distinguish same-line lambdas,
nested definitions, decorators, and generated code using an unambiguous
source/code locator and validated call-site offsets where needed. Do not hash
memory addresses or use mutable display names as authoritative identities.
Unbound dynamically generated code is BLOCKED; generation needs its own
source-bound contract before it can be admitted.
Source/code locations are locators, not executable-payload attestation. Every
source-derived binding must also satisfy Executable-Payload Correspondence below.

Runtime object identities may be used privately to resolve aliases and bound
methods to catalog entries. Persist only stable IDs. Renaming an alias must not
change the disposition of its target. Resolve wrapper/decorator provenance
explicitly; do not trust a copied `__name__` or `__module__` attribute.

The catalog has exact, disjoint ALLOW and FORBID entries. Duplicate, ambiguous,
unresolved, conflicting, or source-drifted entries invalidate admission.
An observed semantic call with no matching valid ALLOW entry is BLOCKED. FORBID wins
over any wrapper or outer-entry permission. No wildcard, module-prefix grant,
name-substring classifier, automatic catalog growth, or blanket permission for
all calls beneath an allowed entry is permitted.

### Guard Coverage Domain and Infrastructure Boundaries

The coverage domain is the frozen preflight workload and its reachable semantic
boundaries, not a capability sandbox for every callable in CPython. Every
delivered call/entry event must resolve to either a semantic operation or a
reviewed infrastructure boundary; an unresolved or unknown semantic boundary
is BLOCKED. There is no third category of silently ignored calls.

Semantic operations require individual catalog entries: factory, reset, render,
step, processor, policy, model/checkpoint loading, RuntimeAdapter, replay, and
network/Hub operations, including indirect entry to those behaviors. Their
dispositions and counts cannot be absorbed into infrastructure permission.

Pure utility primitives and stable stdlib/control-plane helpers may instead
resolve to a frozen source/binary boundary dossier. The dossier defines exact
membership resolution, source/build identities, supported caller/type/dispatch
conditions, lifecycle permissions, reachable effects, and semantic exits that
remain independently guarded. It may cover a reviewed family of primitives
without enumerating each invocation as a scientific operation. A module prefix,
the label "stdlib", or ancestry beneath an allowed call is never sufficient.
Overloaded methods, descriptors, callbacks, or higher-order dispatch cannot
inherit utility permission: prove their dispatch restrictions or guard their
semantic exits; otherwise BLOCKED. Boundary drift and ambiguous membership
also block admission. No runtime trace expands a dossier automatically.

Global event delivery does not imply a global per-callable scientific whitelist.
The resolver records the semantic operation ID or infrastructure dossier ID
supporting each decision. Native internals and callback-suppressed work still
require independent coverage under Layer 3; a dossier is not an exemption from
FORBID, loader checks, phase restrictions, or the violation state machine.

### Legal path and prohibited operations

ALLOW entries cover only the reviewed path: necessary passive imports and
declarations; exact audited registration helpers; official environment factory
construction; simulator model/data and renderer construction; exactly one
public render; and necessary teardown, bounded diagnostics, and process exit.
Observable helpers need semantic permission or audited infrastructure coverage;
unobservable work needs explicit boundary coverage. Neither inherits permission
from the factory.
Simulator model/data construction is distinct
from prohibited ML model/checkpoint loading.

Retain the original preflight counts: one official lazy construction, one
unavoidable construction-time inner reset, one successful public render, and
successful environment teardown. Init-state 0 and seed 2027 remain recorded
provenance only, not applied through a new reset, seed, or settle operation.

FORBID includes processor/pipeline construction and invocation, converters and
pre/postprocessing, policy construction/invocation, ML model/tokenizer/checkpoint
loading or construction, Hub/network paths, action generation, public step,
explicit outer reset, settle/dummy actions, RuntimeAdapter, replay restore,
null measurement, and the other experimental inspections prohibited by the
original renderer design. Catching a rejection never changes this judgment.

### Imports remain independently guarded

An operation catalog does not itself close loader bypasses. Retain an exact
module/source/loader contract, source checks before execution, and fresh-child
preload rejection. Validate the actual loader and source origin before allowing
the body to run. Missing, alternate, indirect, opaque, or dynamically selected
loaders are BLOCKED unless a separately frozen loader contract proves their
before-execution enforcement and coverage. Never fall through to an unchecked
finder and validate only `__file__` after execution.

Static dependency candidates and observed module closure remain distinct.
The earlier 55-module LeRobot-side inventory is not a frozen runtime closure;
external dependencies still need source and boundary review. Unknown modules
are rejected before execution and require source review before a draft audit
can be repeated. No runtime result automatically adds permission.

### Executable-Payload Correspondence

Approval of Python source does not by itself authorize an executable payload.
For every source-backed Python module or callable admitted by the guard, the
loader and code-identity contract must establish, before execution, that the
executable module code and all reachable nested code admitted under that source
binding correspond to the reviewed source bytes under the pinned interpreter
and declared compilation conditions.

The authoritative relation is reviewed source bytes -> declared compilation
semantics -> admitted executable payload, not reviewed source path/hash ->
loader with matching metadata. A source-file SHA, loader identity, filename,
qualified name, source location, line/offset metadata, bytecode-freshness header,
or source-hash field is not by itself sufficient executable-payload attestation.

Cached bytecode, frozen modules, sourceless modules, generated code, or other
non-direct-source execution paths require an explicit executable-payload/source-
provenance contract. It must establish correspondence between the executable
code admitted to the interpreter and the reviewed source or other independently
reviewed payload provenance. Without that correspondence before execution, the
path is BLOCKED. This also applies to Python infrastructure and bootstrap code;
their boundary dossiers cannot substitute metadata for payload provenance.

Validation and execution must refer to the same admitted source/payload identity.
A time-of-check/time-of-use substitution between validation and execution is
forbidden unless an independently established immutability guarantee proves
that substitution cannot occur. The contract covers nested code objects whose
authority derives from the reviewed module. Metadata collisions, copied
filenames/qualnames, or equivalent source-location metadata do not establish
identity. No matching outer module metadata authorizes unbound nested code.

Evidence binds, where applicable, reviewed source identity, loader identity,
executable-payload/code provenance, pinned interpreter and declared compilation
conditions, cache/frozen/generated-code disposition, validation/use correspondence
or immutability guarantee, and the operation-catalog identity relying on that
code. Missing, ambiguous, drifted, or inconsistent correspondence is BLOCKED
and cannot be interpreted as an unused execution path. This amendment freezes
the proof obligation, not a particular compiler/cache-handling implementation.

## Layer 2: external runtime observation and enforcement

The execution target remains the pinned CPython 3.12.6 CPU interpreter. Consult
the matching [Python 3.12 monitoring documentation](https://docs.python.org/3.12/library/sys.monitoring.html),
not a silent interpreter upgrade. The user's linked
[Python 3.13 documentation](https://docs.python.org/zh-cn/3.13/library/sys.monitoring.html)
does not change this pin.

`CALL` observes a Python-code call before dispatch and supplies its callable.
`C_RETURN`/`C_RAISE` describe non-Python-callable exit events associated with
monitored calls; they do not reveal all native internal execution. `PY_START`
is an additional code-bound guard at Python function entry, after call
initiation, not a replacement for pre-call protection. These API boundaries
are specified in the version-matched documentation above.

Pure-stdlib feasibility checks in the pinned interpreter showed why both
identity and coverage matter: direct aliases and class construction could be
rejected before their bodies, but C-mediated `map`, `partial`, and
`operator.call` paths could omit a target CALL event. A code-bound PY_START
check caught the tested Python targets. This is a feasibility observation,
not universal coverage of higher-order dispatch or native call targets.

Every delivered call/entry event is resolved under the Guard Coverage Domain.
FORBID, missing permission, or missing boundary evidence triggers the irreversible
transition below. No callback returns DISABLE to optimize away future checking.

The callback and bootstrap are source-bound control-plane components. For the
registering tool, events are suspended in its callbacks and their callees
([PEP 669, callback events](https://peps.python.org/pep-0669/#events-in-callback-functions)).
This is a trusted coverage boundary, not self-monitoring evidence. Its frozen
dossier must bound identity resolution, latch updates, redacted evidence writes,
and rejection machinery. No workload callback, arbitrary formatting/descriptor
dispatch, import, or environment cleanup may be invoked from that suppressed
region. Cleanup runs after callback unwinding with protection active. An
uncovered callback callee blocks admission, even if its outer callback is trusted.

Reserve a configured monitoring tool ID without stealing another tool's state.
Missing APIs, ownership conflicts, stale callbacks/events, callback failure,
monitor tampering, unsupported concurrency, or lost event coverage are BLOCKED.
Any callbacks/threads that the native path can initiate need explicit coverage;
do not assume the main-thread trace represents them. Validate installation
before factory imports and check integrity at lifecycle boundaries. A callback
exception must not permit silent continued execution with monitoring disabled.

### Monitoring-Suppression Domain Clarification

Monitoring-tool ownership alone does not imply isolation from other tracing,
profiling, or monitoring mechanisms. The trusted-bootstrap and runtime-integrity
contract must either establish that coexisting monitoring/tracing/profiling hooks
and alternative execution hooks are absent under the admitted launch, or
independently cover their execution and suppression effects. Tool-ID ownership
and the guard's own event configuration are not sufficient evidence.

Execution within any monitoring-suppressed region is never evidence of absence
merely because this guard did not receive an event. Missing suppression-domain
coverage is BLOCKED, including when the guard's own callbacks appear intact.

### Irreversible Violation State Machine

The normal failure path is `RUNNING -> VIOLATION_LATCHED -> CLEANUP_ONLY -> EXIT`;
unsafe cleanup permits only `VIOLATION_LATCHED -> EXIT` with BLOCKED preserved.
No reverse transition is legal. Normal completion takes a separate guarded
teardown/terminal path; it cannot overwrite a latched violation.

On violation, the guard first irrevocably records BLOCKED, revokes all normal
work permissions, and rejects dispatch by raising a dedicated guard exception
at CALL, or rejects the Python body at PY_START where that complementary
coverage applies. Pre-entry native side effects still require Layer 3 coverage.
Raising alone is not the enforcement proof. Each subsequent decision reads the
latched state: only frozen failure-unwind, cleanup, and diagnostic permissions
can succeed, including infrastructure permissions narrowed to that failure path.
Only the pinned outer supervisor may enter CLEANUP_ONLY; calling an allowed
cleanup function from workload code does not grant that control-plane context.
Supervisor authority must be protected from workload forgery or mutation under
the frozen supported path; a writable phase flag or reusable workload token is
not sufficient. Failure to establish that boundary blocks admission.

CALL/PY_START alone cannot prevent inline bytecode continuation in an already
active frame after `except BaseException`. The enforcement contract therefore
also requires a source-bound rejection-transfer gate before any post-rejection
workload continuation (including exception handlers, finally blocks, and
generator/coroutine resumption). The implementation plan must select and test
an instruction/resumption-level gate or equivalently proven narrow transfer
mechanism for the pinned interpreter. Only audited failure-unwind/control-plane
locations may proceed; ordinary continuation must be rejected before effects.
If safe transfer to the supervisor cannot be proved, terminate BLOCKED through
the audited failure termination path; do not resume work to obtain cleanup.
Parent timeout/termination is a failure fallback, not proof that continuation
was prevented. Cleanup is best-effort under protection, never permission to
execute arbitrary finally handlers. Missing this transfer proof blocks admission.

The latch has no workload reset API; child reports cannot clear it or rewrite
parent-held evidence. Caught exceptions never restore permissions. Callback
failure, recursion, or lost monitoring integrity likewise cannot resume normal
work; an independently audited failure path must terminate BLOCKED if guarded
cleanup is unsafe. No cleanup executes inside the callback's suppressed region.
The implementation must prove both target nonexecution and continuation
prevention, not merely that the final verdict stayed BLOCKED.

## Layer 3: native boundary coverage

The invariant is **not observable does not mean proven absent**.

Permission for a native entry is necessary but not sufficient. Each allowed
native/C-extension entry must bind the actual binary identity and a reviewed
coverage dossier: pinned-source audit, a narrow external wrapper/instrumentation,
or independent evidence establishing that its allowed path cannot perform the
prohibited operations. The dossier records the entry, reachable behavior under
the frozen conditions, assumptions, evidence hashes, limitations, and review
decision. Bind source-to-binary/build provenance when the argument relies on
source; a Python package SHA alone does not identify its native implementation.

An overly broad entry, unresolved transitive native behavior, unchecked callback,
or unproven source/binary correspondence required by its dossier remains BLOCKED
before entry. A wrapper
must actually narrow/check the behavior; an outer counter is insufficient.
If a required coverage assumption cannot be established without inspecting
forbidden experimental content, the path remains BLOCKED rather than expanding
the permitted measurements. No large third-party source mutation is authorized.

Neither an entry ALLOW, C_RETURN, absence of a trace event, nor zero counters
proves absence of unauthorized native behavior. The monitor is not a complete
C/C++ execution monitor or a general-purpose hostile-code sandbox.

## Lifetime, failure, and parent-owned final judgment

### Trusted Bootstrap Start Boundary

The source-bound bootstrap contract begins at the earliest executable startup
boundary relevant to the admitted child process, not merely when the application-
level guard bootstrap starts. Interpreter startup configuration and executable
startup hooks, including applicable `site`, `.pth`, customization, tracing,
profiling, preload, or equivalent startup behavior, must either be disabled by
the frozen launch contract or included in reviewed bootstrap provenance.

A fresh PID or post-start module inventory does not by itself prove that no
executable startup work occurred earlier. Reviewed launch/bootstrap provenance
must cover the interval before runtime monitoring is installed; no retrospective
zero-use claim can replace it. Source-backed startup payloads obey Executable-
Payload Correspondence; other startup behavior needs its independently reviewed
source/native provenance. Unknown startup behavior blocks admission.

The production child installs protection before official imports/factory entry.
That installation is a runtime milestone, not the start of bootstrap authority.
Its guarded phases are passive import, factory construction, the single render,
environment teardown, and exit. Only the pinned control plane advances phase;
a callable cannot grant itself a new phase or permission.

Rendering success does not disarm the monitor. `close()`/`finalize()` exposed
to the running workload cannot disable hooks or create a reusable terminal PASS.
Any early-disarm attempt is latched BLOCKED. On failure, the audited supervisor's
guarded finally path attempts only approved cleanup when safe transfer is proven.
Failure to clean up remains BLOCKED.

### Implicit and Asynchronous Callback Coverage

Callback and terminal-boundary proofs include explicit calls and implicit or
asynchronous executable entries. Where applicable these include reference-release
finalizers, weak-reference callbacks, collection/finalization callbacks, deferred
signal handling, native-triggered callbacks, and pending native work. Such
behavior must be absent under the frozen execution contract or covered by the
corresponding source/native/infrastructure dossier.

In particular, the interval from final evidence publication through the no-return
termination primitive must not admit uncontrolled workload callbacks or other
mutable application execution. Releasing references or completing a write cannot
be assumed inert merely because no explicit workload call appears in the source.
Unknown implicit or asynchronous behavior remains BLOCKED. The same obligation
applies to callback-suppressed control-plane work and failure cleanup.

### Controlled Terminal Boundary

The protected sequence is guarded application teardown, guarded final evidence
flush, parent-visible terminal marker, then minimal independently audited
process termination. Scientific guard lifetime ends only at this frozen
controlled terminal boundary. It is not a claim to observe arbitrary CPython
finalization internals. For a PASS candidate, application cleanup must already
be complete; an incomplete-cleanup failure may terminate only as BLOCKED. There is
no return to workload, third-party exit handler, or mutable child-owned final
evidence after crossing the boundary. No early-disarm/resume window is allowed.

The terminal contract binds the final trace sequence and evidence digest to a
single marker on the parent-owned bounded channel, along with the termination
primitive's source/binary identity, no-return semantics, and permitted effects.
The child emits the marker after the guarded flush, bound to its fresh PID/run
identity; the parent cannot synthesize a missing child marker.
The parent drains through child death and EOF and seals the received evidence;
a child cannot replace that record with a later report. Missing/duplicate
markers, unexpected post-marker application records, lost bytes, termination
failure, or a still-running child yield BLOCKED. Exit code zero alone is not PASS.

The implementation plan must compare a minimal termination primitive (including
`os._exit()` as a candidate, not a selected implementation) against a bounded
normal-exit path. Either requires independent termination semantics evidence.
Arbitrary third-party atexit handlers or unbounded interpreter shutdown are not
silently covered. A normal-exit alternative is admissible only if it proves an
equally bounded no-workload terminal path; otherwise it remains BLOCKED.

A child report or terminal marker is provisional, never PASS. Only after child
death plus EOF may the parent validate the complete trace, successful teardown,
terminal contract, and absence of violations. A failure marker preserves BLOCKED
even after clean termination. No retry, replacement, child/frame reuse, or extra
render is permitted. Transport loss or unsupported termination remains BLOCKED.

Synthetic test-harness cleanup is separate from production lifetime. Tests may
release their own monitoring tool only after the tested workload has ended,
checking event disablement, callback unregistration, then ID release. Test-only
cleanup cannot be reachable from an empirical worker. Do not claim production
early-restoration success from these unit tests.

## Evidence and validator contract

Bind interpreter/runtime, harness Git identity, source and loader identities,
module closure, canonical operation-catalog SHA, infrastructure/native dossier hashes,
phase transitions, fresh PID/parent identity, installation/integrity checks,
attempted versus completed counts, latched violations, teardown outcome, and
terminal marker/sequence/digest and parent-observed death/EOF/exit status.
Also bind executable-payload correspondence and nested-code authority,
compilation/cache dispositions, validation/use guarantees, earliest-startup
provenance, hook/suppression-domain disposition, and implicit/asynchronous
callback absence or coverage dossiers. Missing evidence in any of these domains
is BLOCKED, not an unused path or an inferred zero count.
Use strict schemas and deterministic canonical
hashing; absence, drift, or inconsistent bindings invalidate the record.

An ordered bounded trace contains only event sequence, phase, event kind,
stable operation or infrastructure-dossier and source/call-site IDs,
disposition, violation-state transitions, and completion status
where observable. No argument values, locals, return contents, pointer reprs,
RGB bytes/hashes, state, success/reward inspection, or pixel metrics are allowed.
The public-render return retains only the metadata authorized by the original
design. Do not double-count CALL and PY_START as two operation attempts.
If completion is unobservable, record it as unknown, not completed zero.

Trace truncation, overflow, callback failure, unresolved identity, or missing
native evidence is BLOCKED, never an abbreviated PASS. Separately hashed trace
and coverage evidence must reconcile with summary counters. Control-plane
publication and teardown are themselves governed; artifact limits must be
explicitly updated and reviewed before any empirical command, not discovered
by writing outside the existing admission artifact allowlist.

PASS requires the unchanged exact preflight operation counts, no forbidden or
unknown attempts, complete required coverage, an intact guard throughout the
workload lifetime through the controlled terminal boundary, successful teardown,
and parent-confirmed death plus EOF with hash-valid
evidence. No module-import audit or unit-test result alone satisfies this gate.

## Required regression and adversarial tests

- Catch actual converter/processor/factory invocation, not only imports or
  familiar names; test aliases, renamed references, bound methods, indirect
  helpers, partials, decorators, and C-mediated Python dispatch.
- Preserve legal passive type/dataclass/registration behavior without rewriting
  third-party code or replacing class identities.
- Reject unknown import-time operations and alternate/indirect loaders before
  their bodies; reject source, code-offset, binary, and catalog drift.
- Reject approved source with a mismatched cached executable payload; reject
  executable code with colliding filename, qualname, or source-location metadata.
- Reject source validation followed by executable-payload substitution; reject
  sourceless/frozen/generated execution without explicit payload provenance.
- Reject nested executable code not traceable to its admitted source/payload
  authority, even when the outer module's source and loader metadata match.
- Reject uncovered coexisting tracing/profiling/monitoring or alternative
  execution hooks and their suppression effects; absence of guard events is
  not acceptable evidence that these paths did not execute.
- Reject executable startup hooks outside the disabled-or-reviewed launch
  contract, including startup before application bootstrap; a fresh PID and
  clean post-start inventory cannot rescue admission.
- Reject uncovered implicit/asynchronous callbacks and pending native work,
  including finalizers, weakref/collection callbacks, and deferred signals
  between final evidence publication and no-return termination.
- Block forbidden class construction before custom `__new__` side effects;
  test inherited and generated dataclass constructors and native-boundary gaps.
- Prove unknown native behavior remains BLOCKED despite an allowed outer entry;
  test missing dossiers, broad helpers, and unsatisfied boundary assumptions.
- Prove caught violations cannot resume normal work or erase failed evidence;
  include inline side effects after `except BaseException`, repeated catches,
  finally/resumption paths, cleanup-only permission narrowing, and callback
  callees that attempt reentrant workload dispatch; a BLOCKED flag alone fails.
  Test early finalize/close, monitor tampering, stale tool state, callback errors,
  phase spoofing, trace overflow, unsupported threads/callbacks, and exit failure.
- Prove protection is active during factory, render, teardown, and exit paths;
  a provisional child success cannot become parent PASS before process exit.
- Prove infrastructure dossiers cannot hide semantic calls, overloaded dispatch,
  or loader bypass; unknown boundaries remain BLOCKED without per-primitive
  scientific enumeration.
- Prove terminal-marker integrity, no workload return after the boundary,
  missing/duplicate markers, post-marker records, channel loss, and death without
  EOF cannot yield PASS; arbitrary finalization is not assumed covered.
- Prove exact operation counts and redacted evidence; no forbidden observation
  content, null data, or experimental frame survives into artifacts.
- Revalidate the untouched predecessor and absence of new empirical artifacts.

All initial execution tests are synthetic and isolated. Real import-only audits
require a source-reviewed candidate catalog and native coverage first, and may
not construct an environment, create a renderer, or write empirical artifacts.

## Milestone gates

1. Review this written revision, then prepare the implementation plan.
2. Replace the rejected mechanism and pass unit/adversarial regressions.
3. Complete source/native coverage and independently review implementation,
   catalog, evidence schemas, and integration with the full preflight path.
4. Commit the reviewed implementation and frozen catalog SHA before admission;
   bind them to the original R1 execution-provenance protocol.
5. Only after all gates permit it may the previously specified single real
   Renderer Preflight occur. This design-document checkpoint does not run it.

No actual catalog SHA or native coverage is claimed by this document. Missing
audit results remain admission blockers, not invented frozen values. Document
review, implementation acceptance, and empirical PASS are separate judgments.
