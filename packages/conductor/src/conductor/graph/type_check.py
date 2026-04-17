"""Compile-time type checking for edge connections.

Validates that source output types are compatible with target input types
before any node executes. Uses pragmatic compatibility rules that match
what Pydantic can actually coerce at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from conductor.graph.model import GraphEdge, GraphNode
from conductor.metadata import InputMetadata, OutputMetadata

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry
    from conductor.registry.definition import NodeDefinition


@dataclass(frozen=True)
class TypeWarning:
    """A warning or diagnostic surfaced by compilation.

    Originally exclusive to edge type-mismatches; now also used for shared
    reference issues. The `code` field discriminates. Existing warnings default
    to ``"type-mismatch"`` for backward compatibility.
    """

    edge_id: str
    source_node: str
    source_output: str
    source_type: str
    target_node: str
    target_input: str
    target_type: str
    message: str
    code: str = "type-mismatch"


# Types that Pydantic can coerce between freely
_NUMERIC_TYPES = frozenset({"int", "float", "number"})
_STRING_COERCIBLE = frozenset({"str", "int", "float", "number", "bool", "date", "base64str"})

# Universal types that accept anything
_ANY_TYPES = frozenset({"any", "str"})


def check_edge_types(
    edges: list[GraphEdge],
    node_map: dict[str, GraphNode],
    registry: "NodeRegistry",
) -> list[TypeWarning]:
    """Check all edges for type compatibility. Returns warnings (not errors).

    Rules (pragmatic, not pedantic):
    - str accepts anything (everything can be stringified)
    - any accepts anything
    - Numerics (int, float) are interchangeable
    - list[T] input accepts T output (single item auto-wrapped)
    - T input accepts list[T] output (only first element used — warn but allow)
    - dict[str, T] (ConnectionList) accepts any type (values are aggregated)
    - Mismatched types produce a warning, not an error
    """
    warnings: list[TypeWarning] = []

    for edge in edges:
        source_node = node_map.get(edge.source)
        target_node = node_map.get(edge.target)
        if not source_node or not target_node:
            continue

        source_def = registry.get(source_node.type)
        target_def = registry.get(target_node.type)
        if not source_def or not target_def:
            continue  # Extension nodes — can't type-check

        source_output = _find_output(source_def, edge.source_handle or "result")
        target_input = _find_input(target_def, edge.target_handle or "")
        if not source_output or not target_input:
            continue  # Handle not found — might be dynamic

        if not _types_compatible(source_output.type_str, target_input.type_str, target_input):
            warnings.append(TypeWarning(
                edge_id=edge.id,
                source_node=edge.source,
                source_output=source_output.name,
                source_type=source_output.type_str,
                target_node=edge.target,
                target_input=target_input.name,
                target_type=target_input.type_str,
                message=(
                    f"Type mismatch: '{source_def.name}' outputs {source_output.type_str} "
                    f"on '{source_output.label}', but '{target_def.name}' expects "
                    f"{target_input.type_str} on '{target_input.label}'"
                ),
            ))

    return warnings


def check_consume_types(
    consume_map: dict[tuple[str, str], tuple[str, str]],
    node_map: dict[str, GraphNode],
    registry: "NodeRegistry",
) -> list[TypeWarning]:
    """Type-check every consume binding. Same rules as ``check_edge_types``.

    Warnings from this function are distinguishable from edge warnings by the
    synthetic ``edge_id`` of the form ``__consume_<target>_<handle>`` and by
    the inclusion of the consumer's input handle name in the message (so a
    simple substring match against the handle name succeeds).
    """
    warnings: list[TypeWarning] = []

    for (target_id, target_handle), (source_id, source_handle) in consume_map.items():
        source_node = node_map.get(source_id)
        target_node = node_map.get(target_id)
        if not source_node or not target_node:
            continue

        source_def = registry.get(source_node.type)
        target_def = registry.get(target_node.type)
        if not source_def or not target_def:
            continue

        source_output = _find_output(source_def, source_handle or "result")
        target_input = _find_input(target_def, target_handle or "")
        if not source_output or not target_input:
            continue

        if not _types_compatible(source_output.type_str, target_input.type_str, target_input):
            warnings.append(TypeWarning(
                edge_id=f"__consume_{target_id}_{target_handle}",
                source_node=source_id,
                source_output=source_output.name,
                source_type=source_output.type_str,
                target_node=target_id,
                target_input=target_input.name,
                target_type=target_input.type_str,
                message=(
                    f"Type mismatch on consume '{target_id}.{target_handle}': "
                    f"shared reference '{source_id}.{source_output.name}' provides "
                    f"{source_output.type_str}, but input expects {target_input.type_str}"
                ),
            ))

    return warnings


def _types_compatible(source_type: str, target_type: str, target_input: InputMetadata) -> bool:
    """Check if source_type can flow into target_type."""
    s = source_type.lower().strip()
    t = target_type.lower().strip()

    # Exact match
    if s == t:
        return True

    # Target accepts anything
    if t in _ANY_TYPES:
        return True

    # Source is 'any' — can't validate, assume ok
    if s == "any":
        return True

    # ConnectionList inputs accept anything (they aggregate into a dict)
    if target_input.uses_connection_list:
        return True

    # Numeric interchangeability
    if s in _NUMERIC_TYPES and t in _NUMERIC_TYPES:
        return True

    # Everything can become a string (Pydantic coerces)
    if t == "str" and s in _STRING_COERCIBLE:
        return True

    # list[T] target accepts T source (single value auto-wraps into list)
    if t.startswith("list[") and t.endswith("]"):
        inner_t = t[5:-1]
        if _types_compatible(s, inner_t, target_input):
            return True

    # T target accepts list[T] source (first element extracted — lossy but common)
    if s.startswith("list[") and s.endswith("]"):
        inner_s = s[5:-1]
        if _types_compatible(inner_s, t, target_input):
            return True

    # dict target accepts dict variants
    if t.startswith("dict") and s.startswith("dict"):
        return True

    # Optional types: T | None — strip the None
    if " | none" in s:
        return _types_compatible(s.replace(" | none", "").strip(), t, target_input)
    if " | none" in t:
        return _types_compatible(s, t.replace(" | none", "").strip(), target_input)

    return False


def _find_output(node_def: "NodeDefinition", handle: str) -> OutputMetadata | None:
    for out in node_def.outputs:
        if out.name == handle:
            return out
    return None


def _find_input(node_def: "NodeDefinition", handle: str) -> InputMetadata | None:
    for inp in node_def.inputs:
        if inp.name == handle:
            return inp
    return None
