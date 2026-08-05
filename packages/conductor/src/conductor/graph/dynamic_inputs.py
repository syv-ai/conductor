"""Compile-time resolution of dynamic node inputs.

Mirrors :mod:`conductor.graph.dynamic_outputs`, minus the topological walk:
an input roster depends only on the node's own ``data``, so each node
resolves independently. See :mod:`conductor.registry.dynamic_inputs` for
why there is no ``incoming``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from conductor.errors import CompilationError
from conductor.metadata import InputMetadata
from conductor.registry.dynamic_inputs import ComputeInputsContext

if TYPE_CHECKING:
    from conductor.graph.model import GraphNode
    from conductor.registry.definition import NodeDefinition

__all__ = ["resolve_node_inputs", "resolve_graph_inputs"]


def resolve_node_inputs(
    *,
    node: GraphNode,
    node_def: NodeDefinition | None,
) -> tuple[InputMetadata, ...]:
    """Return the input roster this node instance actually exposes.

    Nodes without a hook get a verbatim copy of their static
    ``NodeDefinition.inputs``. Extension nodes (no definition) resolve to an
    empty tuple, matching ``resolve_node_outputs``.
    """
    if node_def is None:
        return ()

    hook = getattr(node_def, "compute_inputs", None)
    static_inputs = tuple(node_def.inputs)
    if hook is None:
        return static_inputs

    raw_data = dict(node.data or {})

    validated_data: dict[str, Any] | None = None
    validation_model = getattr(node_def, "validation_model", None)
    if validation_model is not None:
        try:
            validated_data = validation_model(**raw_data).model_dump()
        except Exception:  # noqa: BLE001 — validation failures are expected
            validated_data = None

    ctx = ComputeInputsContext(
        data=raw_data,
        node_id=node.id,
        defaults=static_inputs,
        validated_data=validated_data,
    )

    try:
        result = hook(ctx)
    except CompilationError:
        raise
    except Exception as exc:  # noqa: BLE001 — any error becomes a CompilationError
        raise CompilationError(
            f"compute_inputs failed for node {node.id} ({node.type}): {exc}"
        ) from exc

    if not isinstance(result, list):
        raise CompilationError(
            f"compute_inputs failed for node {node.id} ({node.type}): "
            f"hook must return list[InputMetadata], got {type(result).__name__}"
        )

    seen: set[str] = set()
    for idx, item in enumerate(result):
        if not isinstance(item, InputMetadata):
            raise CompilationError(
                f"compute_inputs failed for node {node.id} ({node.type}): "
                f"item {idx} is not an InputMetadata, got {type(item).__name__}"
            )
        if item.name in seen:
            raise CompilationError(
                f"compute_inputs failed for node {node.id} ({node.type}): "
                f"duplicate input name {item.name!r}"
            )
        seen.add(item.name)

    if not getattr(node_def, "dynamic_handles", False):
        static_names = {i.name for i in static_inputs}
        missing = static_names - seen
        if missing:
            raise CompilationError(
                f"compute_inputs failed for node {node.id} ({node.type}): "
                f"hook dropped statically declared input(s) {sorted(missing)!r}; "
                f"declare ``dynamic_handles=True`` to opt out of static-handle "
                f"preservation."
            )

    return tuple(result)


def resolve_graph_inputs(
    nodes: list[GraphNode],
    definitions: dict[str, NodeDefinition | None],
) -> dict[str, tuple[InputMetadata, ...]]:
    """Resolve every node's input roster.

    The public counterpart to :func:`conductor.resolve_graph_outputs`, for
    hosts that need the resolved shape without compiling.
    """
    return {
        node.id: resolve_node_inputs(node=node, node_def=definitions.get(node.type))
        for node in nodes
    }
