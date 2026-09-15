# D3 final runtime-route review: pivot to the actual tensor transport boundary

**Status: APPROVED ROUTE AMENDMENT.** This review follows commit `3fb7e6dd7470bec2262e36eb99cc08a56ff49d33` and supersedes the assumption that a working NumPy C-API bridge inside the DCU policy process is itself required for ReplayVLA-P1.

## Decision

The NumPy-1.26.4 package-compatibility route is **CLOSED / PIVOTED AWAY**. The single allowed compatibility candidate was unavailable because the package could not be materialized from the host's available package sources. Do not try another NumPy, Torch, Python, package index, wheel source, or runtime candidate under this route.

ReplayVLA itself remains **CONDITIONAL GO**. F2 remains **NOT AUTHORIZED**.

Before deciding whether the runtime is genuinely blocked, perform one bounded F1 amendment, `F1T`, that qualifies the **actual project-owned tensor transport used by the existing CPU-simulator / DCU-policy architecture**. This is not a third runtime candidate and does not change either runtime. It changes the admission predicate from an unused generic bridge to the concrete execution boundary.

## Why the previous predicate was over-broad

The current project transport is defined in `scripts/dcu_worker.py` and `scripts/dcu_model_worker.py`:

- CPU/DCU payloads are persisted as `.safetensors` files.
- Request and response schemas are `torch.Tensor` schemas.
- The DCU model worker loads tensor bundles with `safetensors.torch`, moves tensors to the logical accelerator, runs the policy, and returns tensor bundles.
- The model worker explicitly does not own simulator or observation preprocessing; callers provide the feature tensors expected by the pinned policy.

Therefore, a DCU-side `numpy -> torch` C-API roundtrip is not automatically a necessary edge of this architecture. Requiring it as a universal F1 PASS predicate can reject a runtime even when the actual transport and policy execution path does not traverse that bridge.

Historical evidence also matters, but is not current acceptance evidence: `docs/dcu_concurrency_fix.md` records that the same general DCU stack loaded the frozen model on two K100 workers, produced finite `[1,7]` actions, and completed real environment steps despite the known NumPy-2.x `_ARRAY_API` warning. This proves only that the warning was not sufficient to prevent that historical tensor/policy path; current F1T must re-qualify the concrete boundary on the current branch/runtime.

## F1T scope

Use the existing baseline runtimes only:

- CPU simulator runtime: `/public/home/xuyinghao/tmp/shiftvla-libero/bin/python`
- DCU policy runtime: `/public/home/xuyinghao/tmp/shiftvla-libero-dcu/bin/python`

No package installation, environment construction, EGL, renderer, LIBERO reset/step, checkpoint loading, processor invocation, or policy inference is authorized in F1T.

F1T may exercise only synthetic tensors and the existing project-owned transport modules.

### Required transport proof

For at least two independent fresh-process repetitions:

1. In the CPU runtime, create deterministic synthetic tensors matching the frozen request schema, including state, both images, language tensors, and explicit noise.
2. Save the request through the existing `scripts.dcu_worker.save_tensor_bundle` path.
3. In a fresh DCU runtime process, load the exact bundle through the existing `load_tensor_bundle` path.
4. Verify exact keys, shapes, dtypes, finite status, and content identity on CPU before device transfer.
5. Move the loaded floating tensors through the real logical K100 device path (`cuda:0` under the DCU/HIP stack), then return them to CPU, verifying shape/dtype/value identity for this synthetic transport qualification.
6. Create a deterministic synthetic response tensor in the DCU process and save it through the existing response schema.
7. In a fresh CPU runtime process, load the response through the existing transport path and verify exact schema/content identity.
8. Record K100 visibility, Torch build/HIP identity, executable paths, module origins, command lines, stdout/stderr, file hashes, and no-overwrite evidence.
9. The DCU process may emit the already-known NumPy ABI warning if NumPy is imported indirectly, but F1T must demonstrate that the concrete transport path completes without invoking a required failing NumPy<->Torch conversion. If the actual transport itself hits `RuntimeError: Numpy is not available`, F1T FAILS.

### Static dependency check

Codex must also trace the concrete ReplayVLA policy boundary from CPU observation/preprocessing to DCU request and from DCU response to CPU action application. Record whether any production-relevant step actually requires DCU-side NumPy conversion. Do not infer this solely from the synthetic probe.

If an unavoidable production edge does require DCU-side NumPy conversion, F1T is BLOCKED and the D3 final verdict is PIVOT. Do not redesign the bridge in the same phase.

## F1T PASS condition

F1T may PASS only when both are true:

1. the real safetensors/Torch transport boundary succeeds in repeated fresh processes with K100 transfer; and
2. source/runtime tracing shows the intended production policy path does not require the failed DCU NumPy C-API bridge.

A PASS means only `SA-runtime / tensor-transport-qualified`. It does not authorize F2 automatically. Return for the planned F2 strong review.

## F1T FAIL/BLOCKED condition

Any of the following ends F1 and returns a final PIVOT decision:

- the concrete safetensors/Torch path fails;
- K100 tensor transfer fails;
- actual production tracing reveals an unavoidable DCU NumPy conversion;
- qualification would require changing packages or scientific semantics;
- evidence cannot be made complete and version-bound.

No further runtime/package candidate is allowed after F1T.

## Historical status preservation

- Old G1 remains `BLOCKED`.
- Old E3-E6 and G2 remain outside this route.
- Commit `3fb7e6d` remains valid historical evidence that the generic DCU NumPy bridge failed and the NumPy-1.26.4 candidate was unavailable.
- F1T does not rewrite that history; it narrows the runtime admission predicate to the actual execution boundary.
