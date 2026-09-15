# G1-E2 — Closure adjudication

## Verdict

**E2 = COVERED / PASS (E2-local).** The fixed-payload semantic predicates are
closed by the accepted E2.3 canonical authority graph and E2.4 role-bound
metadata evidence. `G1` remains **BLOCKED** because later validation/use
evidence is still required. E3–E6 remain **NOT STARTED** and G2 remains
**NOT AUTHORIZED**.

## Evidence supersession

E2.3 supersedes the contradictory hand-maintained callable interpretation in
the earlier E2.1/E2.2 reports: the machine-generated graph is the sole
authority, with REDUCE offsets 104, 110, 152, and 43851 and BUILD offsets 184
and 43857. E2.4 supersedes the earlier unresolved literal/schema conclusions
for the fixed roles only; the generic VM remains fully opaque for raw literals.

| E2 predicate | adjudication |
|---|---|
| fixed reducer callable authority | CLOSED |
| exact ndarray subtype | CLOSED (`numpy ndarray`) |
| dynamic callable/type substitution | CLOSED for fixed graph |
| exact BUILD targets | CLOSED |
| dtype REDUCE metadata | CLOSED (`f8`, `False`, `True`) |
| dtype BUILD schema | CLOSED (8 fields) |
| ndarray BUILD schema | CLOSED (5 fields) |
| simple vs structured dtype | CLOSED: simple scalar `f8` / `float64` |
| object-bearing and nested-object condition | CLOSED: false / absent |
| physical ndarray rawdata | preserved as `PHYSICAL_OPAQUE` |
| validation/use consistency | `CROSS_PACKAGE_DEPENDENCY_ON_FUTURE_ADMISSION` |

The final row is intentionally not reclassified as an E2-local failure: it is
an admission/use dependency outside this static structural analysis. No claim
of independent outer-byte authentication is made; the extractor checks the
caller-provided fixed asset binding.

## Verification

The E2.4 focused suite is **4 passed** in the pinned runtime. The broader M1
suite remains environment-blocked by the pinned PyTorch/NumPy ABI
(`RuntimeError: Numpy is not available`) in runtime replay tests; this is not a
counterexample to the E2 semantic evidence. No E3, E4, E5, E6, or G2 action was
performed.

## Transition boundary

E2 may transition to the next stage only through an explicit authorization
decision after reviewing this adjudication. This commit performs no automatic
transition and does not imply that G1 is unblocked.
