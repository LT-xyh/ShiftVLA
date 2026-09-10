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

Written-spec checkpoint: main-agent self-review completed. The independent
written-spec review did not return a usable verdict in this session and is
not counted as PASS. User review of this written document is pending. None of
these document-level states replaces the required implementation/specification
acceptance before empirical execution.

The 2026-09-08 renderer design remains authoritative for task, environment,
operation counts, renderer identities, predecessor immutability, and single-shot
execution. This revision supersedes the 2026-09-09 guard mechanism wherever it
relies on module permission or name heuristics as evidence of allowed use.
It also supersedes child-side early guard restoration: production protection
must remain active until process death, not merely until render returns.

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

Runtime object identities may be used privately to resolve aliases and bound
methods to catalog entries. Persist only stable IDs. Renaming an alias must not
change the disposition of its target. Resolve wrapper/decorator provenance
explicitly; do not trust a copied `__name__` or `__module__` attribute.

The catalog has exact, disjoint ALLOW and FORBID entries. Duplicate, ambiguous,
unresolved, conflicting, or source-drifted entries invalidate admission.
An observed call with no matching valid ALLOW entry is BLOCKED. FORBID wins
over any wrapper or outer-entry permission. No wildcard, module-prefix grant,
name-substring classifier, automatic catalog growth, or blanket permission for
all calls beneath an allowed entry is permitted.

### Legal path and prohibited operations

ALLOW entries cover only the reviewed path: necessary passive imports and
declarations; exact audited registration helpers; official environment factory
construction; simulator model/data and renderer construction; exactly one
public render; and necessary teardown, bounded diagnostics, and process exit.
Every observable helper needs its own catalog entry; unobservable work needs
explicit boundary coverage. Neither inherits permission from the factory.
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

Every delivered call/entry event must resolve to the frozen catalog. A target
FORBID or missing permission latches BLOCKED and prevents further normal
execution. Explicitly catalogued failure teardown and diagnostic operations
may still run; all other work remains prohibited even if an exception was
caught. No callback returns DISABLE to optimize away future checking.

Interpreter helpers, standard-library calls, and harness/control-plane calls
are not silently exempted by module prefix. Observable calls require catalog
entries even if a broader source audit exists. Unobservable work requires
independently audited boundary coverage. The callback implementation
and bootstrap are trusted, source-bound control-plane components; event
suppression inside monitoring callbacks is a coverage boundary, not evidence
that nothing executes there. Unsupported paths cannot contribute zero-use PASS.

Reserve a configured monitoring tool ID without stealing another tool's state.
Missing APIs, ownership conflicts, stale callbacks/events, callback failure,
monitor tampering, unsupported concurrency, or lost event coverage are BLOCKED.
Any callbacks/threads that the native path can initiate need explicit coverage;
do not assume the main-thread trace represents them. Validate installation
before factory imports and check integrity at lifecycle boundaries. A callback
exception must not permit silent continued execution with monitoring disabled.

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

The production child installs protection before official imports/factory entry.
Its guarded phases are passive import, factory construction, the single render,
environment teardown, and exit. Only the pinned control plane advances phase;
a callable cannot grant itself a new phase or permission.

Rendering success does not disarm the monitor. `close()`/`finalize()` exposed
to the running workload cannot disable hooks or create a reusable terminal PASS.
Any early-disarm attempt is latched BLOCKED. On failure, a guarded finally path
attempts only the approved cleanup. Failure to clean up remains BLOCKED.

The empirical child retains its monitor through teardown, diagnostics, exit
handlers, and process termination; OS process death ends its ownership. It
does not free its monitoring slot and then resume an unguarded exit tail.
The shutdown path itself needs the same source/native coverage. A child report
is provisional: only the parent, after observing the fresh child's terminal
exit, can validate complete evidence and publish PASS. Missing reports, abnormal
exit, timeouts, trace loss, unsupported shutdown, or a still-running child are
BLOCKED. No retry, replacement, child/frame reuse, or extra render is permitted.

The parent owns a bounded trace channel and drains it through child death/EOF;
it materializes the final trace hash after reaping the child. A provisional
child-report file is not the last-event boundary. Post-report teardown or exit
violations must reach that channel or invalidate the run. The exact transport
and shutdown tail need an audited control-plane/native contract; a lost channel
or unsupported tail is BLOCKED even with exit code zero. Do not solve the tail
problem by disabling monitoring before exit or by dropping final events.

Synthetic test-harness cleanup is separate from production lifetime. Tests may
release their own monitoring tool only after the tested workload has ended,
checking event disablement, callback unregistration, then ID release. Test-only
cleanup cannot be reachable from an empirical worker. Do not claim production
early-restoration success from these unit tests.

## Evidence and validator contract

Bind interpreter/runtime, harness Git identity, source and loader identities,
module closure, canonical operation-catalog SHA, native coverage dossier hashes,
phase transitions, fresh PID/parent identity, installation/integrity checks,
attempted versus completed counts, latched violations, teardown outcome, and
parent-observed exit status. Use strict schemas and deterministic canonical
hashing; absence, drift, or inconsistent bindings invalidate the record.

An ordered bounded trace contains only event sequence, phase, event kind,
stable operation and source/call-site IDs, disposition, and completion status
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
unknown attempts, complete required coverage, an intact guard through the full
lifetime, successful teardown, and parent-confirmed exit with hash-valid
evidence. No module-import audit or unit-test result alone satisfies this gate.

## Required regression and adversarial tests

- Catch actual converter/processor/factory invocation, not only imports or
  familiar names; test aliases, renamed references, bound methods, indirect
  helpers, partials, decorators, and C-mediated Python dispatch.
- Preserve legal passive type/dataclass/registration behavior without rewriting
  third-party code or replacing class identities.
- Reject unknown import-time operations and alternate/indirect loaders before
  their bodies; reject source, code-offset, binary, and catalog drift.
- Block forbidden class construction before custom `__new__` side effects;
  test inherited and generated dataclass constructors and native-boundary gaps.
- Prove unknown native behavior remains BLOCKED despite an allowed outer entry;
  test missing dossiers, broad helpers, and unsatisfied boundary assumptions.
- Prove caught violations cannot resume normal work or erase failed evidence;
  test early finalize/close, monitor tampering, stale tool state, callback errors,
  phase spoofing, trace overflow, unsupported threads/callbacks, and exit failure.
- Prove protection is active during factory, render, teardown, and exit paths;
  a provisional child success cannot become parent PASS before process exit.
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
