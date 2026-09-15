# F2H final review and one-time F2R authorization

**Status: APPROVED FOR F2R ONLY**

Authority branch: `xyh/replayvla-p1`.
Reviewed commit: `fd3456b1d23c44f1e6f3a8cbe25fd00a9d010ec6`.

## 1. F2H adjudication

F2H is accepted as `BLOCKED`, not `FAIL`.

The retained c5 evidence is sufficient to support that the following were directly observed in that historical cohort: deterministic discovery selected EGL ordinal 8 under the unique `EGL_MESA_device_software` rule; all three predetermined workers returned code 0; all three completed official policy-free construction, exactly one public render, close/destroy, and parent-observed normal exit; all three reported the same EGL/GL identity; raw stdout/stderr hashes are retained; no replacement worker was used.

The compact c5 evidence package and static hardening also bind the retained effective config bytes, runtime lock, F1T predecessor bytes, and the authority erratum. The F2H summary correctly classifies construction internals as trusted/not observed and source-to-binary provenance as not independently attested.

F2H cannot formally promote c5 to `SA-renderer = PASS` because the three historical worker terminal records did not directly record two predicates required by the approved F2 contract:

- the effective-config SHA256 used by that worker;
- the selected EGL ordinal used by that worker.

Those facts may be strongly source-accounted from the parent process and retained config, but the approved F2 contract required them in each worker record. They cannot be retroactively converted into historical observations.

Therefore:

- c5 remains a positive historical dynamic cohort;
- F2H remains `BLOCKED`;
- c5 is not rewritten as formal F2 PASS;
- F3 remains `NOT AUTHORIZED`.

## 2. Governance exception

The previous `NO_FURTHER_F2_DYNAMIC_COHORT_AUTHORIZED` rule is superseded only by this document and only for one final evidence-complete requalification, named **F2R**.

This exception is justified because the blocking predicates are instrumentation/evidence fields, not renderer behavior failures, runtime incompatibility, or renderer-selection uncertainty. It is not permission to search for another working renderer configuration.

Exactly one F2R cohort is authorized. No second F2R cohort, c7, replacement worker, alternate ordinal, package change, task change, asset change, runtime change, or renderer-relevant config change is authorized.

If F2R does not PASS under the frozen constraints below, F2 is closed as `BLOCKED/FAIL` and returns for route-level review. Do not create another renderer cohort.

## 3. Frozen renderer-relevant behavior

F2R must preserve the renderer-relevant behavior of the successful c5 cohort:

- CPU interpreter: `/public/home/xuyinghao/tmp/shiftvla-libero/bin/python`;
- task: `libero_spatial`, task 0, init-state 0, seed 2027;
- observation type: `pixels_agent_pos`;
- renderer backend: EGL;
- selection rule: unique device exposing `EGL_MESA_device_software`;
- expected selected ordinal from the retained c5 discovery: 8;
- expected EGL identity: Mesa Project / 1.5;
- expected GL identity: Mesa/X.org / llvmpipe (LLVM 12.0.0, 256 bits) / 3.1 Mesa 21.1.5;
- runtime-lock bytes and hash as bound by F2H;
- same BDDL/init-state/assets identity as the c5 qualified path;
- no policy, processor, checkpoint, env research step, null or replay.

A new F2R config is allowed only to correct authority/provenance metadata and to add the missing evidence instrumentation. It must not change runtime behavior. The old c5 config bytes remain immutable historical evidence.

The new config must bind this authorization commit as its parent authority and must explicitly list the retained c5 effective-config SHA256 plus the semantic-delta statement: only authority/provenance/instrumentation fields differ.

## 4. Discovery rule

F2R may perform exactly one new discovery process before worker construction.

The deterministic selection rule is frozen before execution: select the unique device exposing `EGL_MESA_device_software`.

For this requalification, the discovery result must also equal ordinal 8. If the rule returns zero candidates, multiple candidates, or a unique ordinal other than 8, F2R is `BLOCKED`; stop without launching workers. Do not adapt the expected ordinal.

This is a stability check, not a renderer search.

## 5. Required worker evidence

Launch exactly three predetermined fresh workers and no replacements.

Each worker terminal record must directly include and the adjudicator must verify:

- worker id;
- return code == 0;
- status == PASS;
- selected ordinal == frozen discovery ordinal == 8;
- effective-config SHA256 == frozen F2R effective-config SHA256;
- public render count == 1;
- valid render metadata;
- construction status == PASS;
- close/destroy status == PASS;
- parent-observed normal exit == true;
- forbidden operation count == 0;
- replacement == false;
- EGL identity and GL identity;
- interpreter/runtime identity;
- stdout path/hash and stderr path/hash;
- timeout/cleanup fields.

The worker must derive the recorded selected ordinal and config hash from the actual config it loads, not from caller-provided expected values alone.

The parent adjudicator must reject any mismatch among the three workers or between workers, discovery and effective config.

## 6. Fail-closed and cleanup requirements

Use a bounded subprocess timeout.

If construction succeeds and a later operation raises, run best-effort close in `finally` and record the cleanup result. Cleanup failure is not PASS.

All output publication remains no-overwrite. F2R must have a fresh versioned output root. Old c1-c5 and F2H evidence are immutable.

Before dynamic execution, focused tests must cover the hardened predicates, including config hash mismatch, selected-ordinal mismatch, nonzero return code, missing stdout/stderr hash, timeout terminalization, duplicate worker id, and post-construction cleanup.

## 7. F2R PASS criteria

F2R passes only if:

1. one discovery process selects exactly ordinal 8 under the frozen rule;
2. one immutable F2R effective config is frozen before workers;
3. exactly three predetermined fresh workers run, with no replacements;
4. all three directly record the same frozen effective-config SHA256 and ordinal 8;
5. all three satisfy the full worker predicates in section 5;
6. all three observe the expected identical EGL/GL identity;
7. input byte bindings and evidence classifications remain valid;
8. no forbidden operation or runtime/package/scientific-semantics change occurs.

If PASS, publish a compact GitHub-reviewable evidence package and `f2r_summary.json`, but do not enter F3 until review.

If FAIL/BLOCKED, stop. No further F2 dynamic run is authorized.

## 8. State after this authorization

- F0 = PASS.
- F1T = PASS.
- F2 c5 = positive historical dynamic evidence, not formal PASS.
- F2H = BLOCKED due missing historical per-worker config-hash and ordinal fields.
- F2R = AUTHORIZED exactly once under this document.
- formal `SA-renderer` = PENDING.
- F3-F6 = NOT AUTHORIZED.
- old G1 remains BLOCKED; old E3-E6/G2 remain paused/not started.
