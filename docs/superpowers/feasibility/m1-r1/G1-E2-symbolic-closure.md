# E2 — Symbolic reconstruction graph closure

## Verdict

**E2 = UNRESOLVED.**

This result is deliberately limited to E2. E3–E6 and G2 were not started.
The fixed pickle stream was inspected as a virtual-machine program with
`pickletools.genops`; no reducer, BUILD handler, dtype constructor, codec,
Torch, NumPy, LIBERO, MuJoCo, or official-runtime code was executed.

## Fixed input and symbolic scan

Input: `archive/data.pkl` inside the fixed task-0 `.pruned_init` ZIP. The
outer asset SHA-256 is
`cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a`; the
member SHA-256 is
`19a1f03ed94cf5bff88906b2b3742245a64443e5cc623e6ffc6aed6773f4fef4`.

The scan records one `PROTO`, four `GLOBAL`, four `REDUCE`, two `BUILD`, and
one `STOP`, with 21 memo writes/reads represented in the stream. No
`STACK_GLOBAL`, `EXT1/EXT2/EXT4`, `NEWOBJ`, `NEWOBJ_EX`, `INST`, `OBJ`,
`PERSID`, or `BINPERSID` opcode was observed. The exact GLOBAL targets are:

| offset | opcode | exact target |
|---:|---|---|
| 2 | GLOBAL | `numpy.core.multiarray _reconstruct` |
| 40 | GLOBAL | `numpy ndarray` |
| 62 | GLOBAL | `_codecs encode` |
| 123 | GLOBAL | `numpy dtype` |

## Reducer and BUILD authority table

The following is the complete authority-opcode table obtained from the fixed
instruction stream. `arg structure` is intentionally symbolic; scalar/array
payloads are opaque and are never printed.

| offset | opcode | symbolic callable / target | symbolic argument structure | symbolic result / state |
|---:|---|---|---|---|
| 104 | REDUCE | `GLOBAL numpy.ndarray` | tuple containing the immediately preceding `_codecs.encode` result | `REDUCE_RESULT(numpy.ndarray, opaque-codec-result)` |
| 110 | REDUCE | `GLOBAL numpy.core.multiarray._reconstruct` | tuple containing the prior ndarray-call result and opaque tuple payload | `REDUCE_RESULT(_reconstruct, opaque tuple)` |
| 152 | REDUCE | prior `_reconstruct` result | tuple containing `GLOBAL numpy.dtype` and opaque dtype descriptor | `REDUCE_RESULT(prior-result, dtype-result)` |
| 184 | BUILD | current reconstruction target | opaque state tuple/dict payload | target state mutation is symbolic only |
| 43851 | REDUCE | `GLOBAL _codecs.encode` | tuple containing opaque byte/string payload and opaque encoding argument | `REDUCE_RESULT(_codecs.encode, opaque)` |
| 43857 | BUILD | current reconstruction target | opaque state tuple/dict payload | target state mutation is symbolic only |

The stack/memo-aware prototype (`scripts/m1_e2_symbolic.py`) emits an event
record for every opcode with offset, opcode, stack-kind snapshot, memo reads,
and memo writes. Synthetic tests cover REDUCE/BUILD presence and rejection of
PERSID. The prototype intentionally never calls `pickle.loads`; its output
keeps physical values opaque.

For each authority opcode, the VM event snapshots were:

| offset | opcode | stack before → after | memo reads / writes |
|---:|---|---|---|
| 2 | GLOBAL | `()` → `(GLOBAL)` | `[] / []` |
| 40 | GLOBAL | `(GLOBAL)` → `(GLOBAL, GLOBAL)` | `[] / []` |
| 62 | GLOBAL | `(GLOBAL, TUPLE)` → `(GLOBAL, TUPLE, GLOBAL)` | `[] / []` |
| 104 | REDUCE | `(GLOBAL, TUPLE)` → `(REDUCE_RESULT)` | `[] / []` |
| 110 | REDUCE | `(TUPLE)` → `(REDUCE_RESULT)` | `[] / []` |
| 123 | GLOBAL | `(TUPLE)` → `(TUPLE, GLOBAL)` | `[] / []` |
| 152 | REDUCE | `(TUPLE)` → `(REDUCE_RESULT)` | `[] / []` |
| 184 | BUILD | `(REDUCE_RESULT, TUPLE)` → `(BUILT)` | `[] / []` |
| 43851 | REDUCE | `(BUILT, TUPLE)` → `(REDUCE_RESULT)` | `[] / []` |
| 43857 | BUILD | `(REDUCE_RESULT, TUPLE)` → `(BUILT)` | `[] / []` |

Memo writes occur at non-authority `MEMOIZE`/put instructions and are retained
in the event stream; no authority opcode itself reads or writes a memo entry.

## Required predicates and current result

The fixed stream provides strong negative evidence: no extension opcode,
persistent-ID opcode, `STACK_GLOBAL`, `NEWOBJ*`, or executable Python class or
function GLOBAL outside the four observed names occurs. There is also no
separate Torch storage member in the ZIP. These predicates are
`EXCLUDED_WITH_EVIDENCE` for this fixed byte stream, subject to pre-entry hash
checking.

The following predicates remain **UNRESOLVED**, so E2 cannot be COVERED:

- complete stack and memo transition semantics through MARK/POP/tuple and
  memo operations, including the exact symbolic argument tuple at every
  REDUCE;
- exact final constructed Python type (plain `numpy.ndarray` versus a
  subclass) without executing or independently proving the relevant reducer
  type checks;
- exact BUILD target identity and complete state schema at offsets 184 and
  43857;
- exact `numpy.dtype` descriptor semantics, including whether the opaque
  descriptor denotes an object-bearing dtype or nested object field;
- proof that no reducer callable is dynamically supplied through opaque stack
  values beyond the four visible GLOBALs;
- validation/use consistency and source-byte immutability between the static
  scan and any future execution.

These are precise missing predicates, not a claim that symbolic analysis is
impossible. Materializing the simulator state or executing reducers would be
required to discharge some of them, and that is forbidden in E2.

## Files changed

- `scripts/m1_e2_symbolic.py` — stdlib-only symbolic VM inspection helper;
- `tests/test_m1_e2_symbolic.py` — two synthetic tests;
- `docs/superpowers/feasibility/m1-r1/G1-E2-symbolic-closure.md` — this report.

The historical G1 reports and accepted design/plan were not modified.

## Commands executed

```text
sed / rg                         # fixed source and call-site inspection
sha256sum                        # fixed asset/member/source hashes
python3 stdlib zipfile + pickletools.genops  # ZIP and opcode scan
PYTHONPATH=. ... pytest -q tests/test_m1_e2_symbolic.py  # 2 passed
PYTHONPATH=. ... scripts/m1_e2_symbolic.py               # fixed stream scan
git diff --check                 # whitespace validation
```

No Torch/NumPy/LIBERO/MuJoCo/official-runtime import occurred. No
`pickle.loads`, `torch.load`, unpickling, deserialization, REDUCE, BUILD,
`__setstate__`, dtype construction, codec execution, or reducer execution
occurred. No E3, E4, E5, E6, G2, environment, EGL, render, policy, processor,
replay, or schedule operation occurred.

## Commit status

This E2 closure work is not yet committed. The required next step is review of
the symbolic interpreter and this report; E3 must not begin automatically.
