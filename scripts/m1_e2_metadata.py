"""Role-bound, non-executing structural metadata extraction for E2.4.

This module deliberately sits beside (and never changes) the generic symbolic
VM. Only literals at paths proven by the canonical authority graph are read.
"""
from __future__ import annotations
import pickletools
from typing import Any, Mapping

FIXED_ASSET_SHA256 = "cbbc73792ce546c9bec181fd328a411d3183074840b282671dee481511381d0a"

# These are symbolic paths in the machine-generated E2.3 graph. They are an
# allowlist, not a content heuristic: changing an offset, node id, node kind,
# or authority string disables extraction.
_APPROVED_PATHS = frozenset({
    "REDUCE@152.args.TUPLE#13.BINUNICODE#14",
    "REDUCE@152.args.TUPLE#13.NEWFALSE#15",
    "REDUCE@152.args.TUPLE#13.NEWTRUE#16",
    *(f"BUILD@184.state.TUPLE#21.field[{i}]" for i in range(8)),
    "BUILD@43857.state.TUPLE#30.version",
    "BUILD@43857.state.TUPLE#30.shape",
    "BUILD@43857.state.TUPLE#30.dtype",
    "BUILD@43857.state.TUPLE#30.order",
    "BUILD@43857.state.TUPLE#30.data",
    "REDUCE@43851.result",
})

_EXPECTED_AUTHORITIES = {
    104: "_codecs encode",
    110: "numpy.core.multiarray _reconstruct",
    152: "numpy dtype",
    43851: "_codecs encode",
}
_EXPECTED_OFFSETS = frozenset({104, 110, 152, 184, 43851, 43857})


def _node_map(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(node["id"]): node for node in graph.get("nodes", ())}

def _ops(graph: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(x["offset"]): x for x in graph.get("operations", ())}


def _literal_map(data: bytes, wanted: Mapping[int, str]) -> dict[int, tuple[str, Any]]:
    """Capture only approved metadata literals; never retain rawdata bytes."""
    out: dict[int, tuple[str, Any]] = {}
    for op, arg, pos in pickletools.genops(data):
        expected = wanted.get(pos)
        if expected is not None and op.name == expected:
            out[pos] = (op.name, arg)
    return out


def _graph_is_canonical(graph: Mapping[str, Any]) -> bool:
    """Validate the exact E2.3 authority topology before reading literals."""
    ops = _ops(graph)
    nodes = _node_map(graph)
    if set(ops) != _EXPECTED_OFFSETS:
        return False
    if set(_EXPECTED_AUTHORITIES) - set(ops):
        return False
    for offset, authority in _EXPECTED_AUTHORITIES.items():
        op = ops[offset]
        if op.get("opcode") != "REDUCE":
            return False
        ref = op.get("callable", {}).get("ref")
        node = nodes.get(ref)
        if not node or node.get("kind") != "GLOBAL" or node.get("authority") != authority:
            return False

    def ref_kind(ref: Any, kind: str) -> bool:
        node = nodes.get(ref)
        return bool(node and node.get("kind") == kind)

    dtype_args = ops[152].get("args", {}).get("ref")
    dtype_node = nodes.get(dtype_args)
    if not dtype_node or dtype_node.get("id") != "TUPLE#13":
        return False
    children = dtype_node.get("children", [])
    if [x.get("ref") for x in children] != ["BINUNICODE#14", "NEWFALSE#15", "NEWTRUE#16"]:
        return False
    if not all(ref_kind(x.get("ref"), k) for x, k in zip(children, ("BINUNICODE", "NEWFALSE", "NEWTRUE"))):
        return False

    build_dtype = ops.get(184)
    if not build_dtype or build_dtype.get("opcode") != "BUILD":
        return False
    if build_dtype.get("target", {}).get("ref") != "REDUCE_RESULT#17" or build_dtype.get("state", {}).get("ref") != "TUPLE#21":
        return False
    dtype_state = nodes.get("TUPLE#21", {})
    dtype_children = dtype_state.get("children", [])
    if len(dtype_children) != 8 or [nodes.get(x.get("ref"), {}).get("kind") for x in dtype_children] != [
        "BININT1", "BINUNICODE", "NONE", "NONE", "NONE", "BININT", "BININT", "BININT1"
    ]:
        return False

    build_array = ops.get(43857)
    if not build_array or build_array.get("opcode") != "BUILD":
        return False
    if build_array.get("target", {}).get("ref") != "REDUCE_RESULT#11" or build_array.get("state", {}).get("ref") != "TUPLE#30":
        return False
    array_children = nodes.get("TUPLE#30", {}).get("children", [])
    if len(array_children) != 5:
        return False
    expected_kinds = ["BININT1", "TUPLE", "BUILT", "NEWFALSE", "REDUCE_RESULT"]
    if [nodes.get(x.get("ref"), {}).get("kind") for x in array_children] != expected_kinds:
        return False
    return True


def _safe_graph_is_canonical(graph: Mapping[str, Any]) -> bool:
    try:
        return _graph_is_canonical(graph)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _literal(literals: Mapping[int, tuple[str, Any]], offset: int, opcode: str) -> Any:
    """Read one literal only after its exact opcode has been checked."""
    actual = literals.get(offset)
    if actual is None or actual[0] != opcode:
        raise ValueError(f"literal drift at offset {offset}: expected {opcode}")
    if opcode == "NEWFALSE":
        return False
    if opcode == "NEWTRUE":
        return True
    return actual[1]


def extract_structural_metadata(data: bytes, *, asset_sha256: str,
                                authority_graph: Mapping[str, Any],
                                allow_paths: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Extract approved dtype/ndarray metadata, failing closed on drift."""
    if asset_sha256 != FIXED_ASSET_SHA256:
        raise ValueError("fixed asset hash mismatch; structural decoding disabled")
    requested = _APPROVED_PATHS if allow_paths is None else frozenset(allow_paths)
    if not requested or not requested.issubset(_APPROVED_PATHS):
        return {"status": "UNRESOLVED", "fields": []}
    if not _safe_graph_is_canonical(authority_graph):
        return {"status": "UNRESOLVED", "fields": []}
    literal_opcodes = {
        138: "BINUNICODE", 147: "NEWFALSE", 148: "NEWTRUE",
        156: "BININT1", 158: "BINUNICODE", 166: "NONE", 167: "NONE",
        168: "NONE", 169: "BININT", 174: "BININT", 179: "BININT1",
        114: "BININT1", 116: "BININT1", 118: "BININT1", 185: "NEWFALSE",
    }
    try:
        literals = _literal_map(data, literal_opcodes)
    except (ValueError, IndexError, KeyError):
        return {"status": "UNRESOLVED", "fields": []}

    def field(path: str, offset: int | None, classification: str, value: Any = None, reason: str = ""):
        return {"path": path, "offset": offset, "classification": classification, "value": value, "reason": reason}

    try:
        values = {
            "descriptor": _literal(literals, 138, "BINUNICODE"),
            "align": _literal(literals, 147, "NEWFALSE"),
            "copy": _literal(literals, 148, "NEWTRUE"),
            "dtype_state": [_literal(literals, off, opcode) for off, opcode in (
                (156, "BININT1"), (158, "BINUNICODE"), (166, "NONE"), (167, "NONE"),
                (168, "NONE"), (169, "BININT"), (174, "BININT"), (179, "BININT1"),
            )],
            "version": _literal(literals, 114, "BININT1"),
            "shape0": _literal(literals, 116, "BININT1"),
            "shape1": _literal(literals, 118, "BININT1"),
            "order": _literal(literals, 185, "NEWFALSE"),
        }
    except (ValueError, IndexError, KeyError):
        return {"status": "UNRESOLVED", "fields": []}

    fields = [
        field("REDUCE@152.args.TUPLE#13.BINUNICODE#14", 138, "STRUCTURAL_METADATA", values["descriptor"], "dtype descriptor/type code; NumPy legacy dtype constructor input"),
        field("REDUCE@152.args.TUPLE#13.NEWFALSE#15", 147, "STRUCTURAL_METADATA", values["align"], "dtype alignment flag"),
        field("REDUCE@152.args.TUPLE#13.NEWTRUE#16", 148, "STRUCTURAL_METADATA", values["copy"], "dtype copy flag"),
    ]
    state_offsets = (156, 158, 166, 167, 168, 169, 174, 179)
    state_roles = ("version", "byteorder", "subarray", "fields", "names", "itemsize", "alignment", "flags")
    state = [field(f"BUILD@184.state.TUPLE#21.field[{i}]", off, "STRUCTURAL_METADATA", value, f"dtype __setstate__ {role}") for i, (off, value, role) in enumerate(zip(state_offsets, values["dtype_state"], state_roles))]
    fields.extend(state)
    array_fields = [
        field("BUILD@43857.state.TUPLE#30.version", 114, "STRUCTURAL_METADATA", values["version"], "ndarray __setstate__ version"),
        field("BUILD@43857.state.TUPLE#30.shape", 120, "STRUCTURAL_METADATA", {"class": "tuple", "arity": 2, "elements": [values["shape0"], values["shape1"]]}, "ndarray shape structure; dimensions are metadata, not data"),
        field("BUILD@43857.state.TUPLE#30.dtype", 184, "AUTHORITY_REFERENCE", "BUILD@184 -> REDUCE_RESULT#17", "dtype object identity"),
        field("BUILD@43857.state.TUPLE#30.order", 185, "STRUCTURAL_METADATA", values["order"], "ndarray isFortran flag"),
        field("REDUCE@43851.result", 43851, "PHYSICAL_OPAQUE", None, "codec result used as ndarray rawdata; never decode"),
        field("BUILD@43857.state.TUPLE#30.data", 43851, "PHYSICAL_OPAQUE", None, "ndarray rawdata field is codec result"),
    ]
    fields.extend(array_fields)
    selected = [item for item in fields if item["path"] in requested]
    return {"status": "COVERED", "asset_sha256": asset_sha256,
            "dtype": {"descriptor": fields[0], "align": fields[1], "copy": fields[2],
                      "hasobject": False, "nested_object_fields": False,
                      "kind": "exact scalar/simple dtype (f8 = float64; non-object)"},
            "dtype_build": {"arity": 8, "fields": state},
            "ndarray_build": {"arity": 5, "data": array_fields[-1], "fields": array_fields},
            "fields": selected}
