"""Validation and wiring for shared references (produce/consume).

See ``docs/shared-references.md`` for the design. This module is only called
from ``graph/compiler.py``; it is kept separate so the compiler stays focused
on the overall pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor.errors import CompilationError
from conductor.graph.model import GraphEdge, GraphNode
from conductor.graph.type_check import TypeWarning
from conductor.metadata import OutputMetadata

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry


ConsumeMap = dict[tuple[str, str], tuple[str, str]]
# (target_id, target_handle) -> (source_id, source_handle)


def validate_and_build_consume_map(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    node_map: dict[str, GraphNode],
    managed_ids: frozenset[str],
    registry: "NodeRegistry",
    node_outputs: dict[str, tuple[OutputMetadata, ...]] | None = None,
) -> tuple[ConsumeMap, list[TypeWarning]]:
    """Validate produces/consumes decorations and return the consume map.

    Raises ``CompilationError`` on structural violations (§6.1, §6.2 of the
    design). Returns a list of non-fatal warnings (e.g. duplicate producer
    labels) alongside the map.

    ``node_outputs`` is the post-``compute_outputs`` map from the compiler.
    When provided, producer-handle existence is checked against the
    resolved outputs in preference to the static schema, so a hook that
    introduces ``output_3`` enables ``produces={"output_3": "..."}``.
    """
    warnings: list[TypeWarning] = []

    _validate_producers(nodes, managed_ids, registry, warnings, node_outputs)
    consume_map = _validate_consumers_and_build_map(nodes, edges, node_map, registry)
    return consume_map, warnings


# ---------------------------------------------------------------------------
# Producer validation (§6.1)
# ---------------------------------------------------------------------------


def _validate_producers(
    nodes: list[GraphNode],
    managed_ids: frozenset[str],
    registry: "NodeRegistry",
    warnings: list[TypeWarning],
    node_outputs: dict[str, tuple[OutputMetadata, ...]] | None = None,
) -> None:
    label_to_producers: dict[str, list[tuple[str, str]]] = {}

    for node in nodes:
        if not node.produces:
            continue

        # §6.1.1 — producers cannot sit inside compound regions in v1
        if node.id in managed_ids:
            raise CompilationError(
                f"Node '{node.id}' cannot produce a shared reference from "
                f"inside a compound region (v1 limitation — see "
                f"docs/shared-references.md §8)"
            )

        node_def = registry.get(node.type)

        # When this node owns a ``compute_outputs`` hook and we don't yet
        # have the resolved map, defer handle-existence to the second pass
        # — the hook may legitimately introduce the handle being published.
        defer_handle_check = (
            node_outputs is None
            and node_def is not None
            and getattr(node_def, "compute_outputs", None) is not None
        )

        for output_handle, label in node.produces.items():
            # §6.1.2 — output handle must exist on the node type
            if node_def is not None and not defer_handle_check:
                # Prefer resolved (post-compute_outputs) handles when
                # available so dynamic outputs participate in shared-ref
                # validation without a separate code path.
                resolved = (
                    node_outputs.get(node.id) if node_outputs is not None else None
                )
                pool = resolved if resolved is not None else node_def.outputs
                known_outputs = {o.name for o in pool}
                if output_handle not in known_outputs:
                    raise CompilationError(
                        f"Node '{node.id}' produces unknown handle "
                        f"'{output_handle}' — not declared on node type "
                        f"'{node.type}'. Available outputs: "
                        f"{sorted(known_outputs) or '(none)'}"
                    )

            label_to_producers.setdefault(label, []).append((node.id, output_handle))

    # §6.1.3 — duplicate labels are a warning, not an error
    for label, producers in label_to_producers.items():
        if len(producers) > 1:
            producer_list = ", ".join(f"{nid}.{h}" for nid, h in producers)
            warnings.append(TypeWarning(
                edge_id="",
                source_node="",
                source_output="",
                source_type="",
                target_node="",
                target_input="",
                target_type="",
                message=(
                    f"Multiple producers share the display label '{label}': "
                    f"{producer_list}. Labels are for UI only; references are "
                    f"bound by identity, so this is non-fatal."
                ),
                code="shared-label-collision",
            ))


# ---------------------------------------------------------------------------
# Consumer validation + map construction (§6.2)
# ---------------------------------------------------------------------------


def _validate_consumers_and_build_map(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    node_map: dict[str, GraphNode],
    registry: "NodeRegistry",
) -> ConsumeMap:
    # Pre-compute: which (target, handle) pairs already have an explicit edge
    edge_targets: set[tuple[str, str]] = {
        (e.target, e.target_handle or "") for e in edges
    }

    consume_map: ConsumeMap = {}

    for node in nodes:
        if not node.consumes:
            continue

        node_def = registry.get(node.type)

        for input_handle, ref in node.consumes.items():
            producer_id, output_handle = ref

            # §6.2.1.1 — producer must exist
            producer_node = node_map.get(producer_id)
            if producer_node is None:
                raise CompilationError(
                    f"Node '{node.id}' consumes from unknown producer "
                    f"'{producer_id}' (input '{input_handle}')"
                )

            # §6.2.1.2 — producer must explicitly publish this handle
            producer_published = producer_node.produces or {}
            if output_handle not in producer_published:
                raise CompilationError(
                    f"Node '{node.id}' consumes '{producer_id}.{output_handle}' "
                    f"but that output is not produced as a shared reference. "
                    f"Add '{output_handle}' to '{producer_id}'.produces."
                )

            # §6.2.1.3 — input handle must exist on the consumer's node type
            if node_def is not None:
                known_inputs = {i.name for i in node_def.inputs}
                if input_handle not in known_inputs:
                    raise CompilationError(
                        f"Node '{node.id}' consumes into unknown input "
                        f"'{input_handle}' — not declared on node type "
                        f"'{node.type}'. Available inputs: "
                        f"{sorted(known_inputs) or '(none)'}"
                    )

            # §6.2.1.4 — cannot also be the target of an explicit edge
            if (node.id, input_handle) in edge_targets:
                raise CompilationError(
                    f"Input '{node.id}.{input_handle}' is both consumed from "
                    f"'{producer_id}.{output_handle}' and connected by an edge — "
                    f"choose one."
                )

            consume_map[(node.id, input_handle)] = (producer_id, output_handle)

    return consume_map
