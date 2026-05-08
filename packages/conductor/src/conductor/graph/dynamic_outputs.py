"""Compile-time resolver for the ``compute_outputs`` hook.

Walks each node in topological order and, when a hook is registered, calls
it with the resolved upstream outputs. The result is validated and stored
on ``CompiledGraph.node_outputs``; downstream stages (type-check,
shared-refs, compound runtimes) consult that map in preference to the
static ``NodeDefinition.outputs``.

The hook is purely a compile-time feature. The runtime contract — node
function returns a dict, ``extract_output`` keys by handle name — is
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor.errors import CompilationError
from conductor.metadata import OutputMetadata
from conductor.registry.dynamic_outputs import (
    ComputeOutputsContext,
    IncomingBinding,
)

if TYPE_CHECKING:
    from conductor.graph.model import GraphNode
    from conductor.registry import NodeRegistry


def resolve_node_outputs(
    node: "GraphNode",
    node_def,
    incoming_edges: list[tuple[str, str, str, str]],
    resolved_outputs: dict[str, tuple[OutputMetadata, ...]],
    node_map: dict[str, "GraphNode"],
    registry: "NodeRegistry",
) -> tuple[OutputMetadata, ...]:
    """Resolve a node's outputs, invoking ``compute_outputs`` if present.

    Args:
        node: The graph instance.
        node_def: Its registered ``NodeDefinition`` (or ``None`` for
            extension nodes).
        incoming_edges: ``incoming_map[node.id]`` — list of
            ``(target_handle, source_id, source_handle, edge_id)``. May be
            empty.
        resolved_outputs: Already-resolved producers' outputs, keyed by
            node id. Producers are guaranteed to appear here because the
            compiler iterates topologically.
        node_map: All graph nodes by id, for source lookups.
        registry: The active registry, for source-side ``NodeDefinition``
            lookups when the producer has no hook.

    Returns:
        A tuple of ``OutputMetadata``. When the node has no hook this is
        simply ``node_def.outputs`` (or an empty tuple for extension nodes).

    Raises:
        CompilationError: When the hook returns a non-list, has duplicate
            output names, drops a statically declared name without
            ``dynamic_handles=True``, or raises any exception. The original
            exception is chained via ``__cause__`` for debugging.
    """
    # Extension nodes (or genuinely unknown types) — nothing to resolve.
    if node_def is None:
        return ()

    hook = getattr(node_def, "compute_outputs", None)
    static_outputs = tuple(node_def.outputs)
    if hook is None:
        return static_outputs

    # Build the IncomingBinding tuple. Each binding carries the producer's
    # *resolved* output metadata so a hook can read upstream type-strs that
    # were themselves computed dynamically.
    bindings: list[IncomingBinding] = []
    for target_handle, source_id, source_handle, _edge_id in incoming_edges:
        source_outputs = resolved_outputs.get(source_id)
        if source_outputs is None:
            # Producer is an extension node or hasn't been resolved yet
            # (shouldn't happen given topological order, but be defensive).
            source_node = node_map.get(source_id)
            source_def = (
                registry.get(source_node.type) if source_node is not None else None
            )
            source_outputs = tuple(source_def.outputs) if source_def else ()

        match = next(
            (o for o in source_outputs if o.name == source_handle),
            None,
        )
        if match is None:
            # Handle not declared on the producer (likely a dynamic-handles
            # compound emitting ``output_3``, etc.). Synthesize a permissive
            # placeholder so the hook still gets a complete picture.
            match = OutputMetadata(
                name=source_handle, type_str="any", label=source_handle,
            )

        bindings.append(IncomingBinding(
            target_handle=target_handle,
            source_node_id=source_id,
            source_handle=source_handle,
            source_output=match,
        ))

    ctx = ComputeOutputsContext(
        data=dict(node.data or {}),
        incoming=tuple(bindings),
        node_id=node.id,
        defaults=static_outputs,
    )

    try:
        result = hook(ctx)
    except CompilationError:
        raise
    except Exception as exc:  # noqa: BLE001 — any error becomes a CompilationError
        raise CompilationError(
            f"compute_outputs failed for node {node.id} ({node.type}): {exc}"
        ) from exc

    if not isinstance(result, list):
        raise CompilationError(
            f"compute_outputs failed for node {node.id} ({node.type}): "
            f"hook must return list[OutputMetadata], got {type(result).__name__}"
        )

    # Validate every entry is OutputMetadata and names are unique.
    seen: set[str] = set()
    for idx, item in enumerate(result):
        if not isinstance(item, OutputMetadata):
            raise CompilationError(
                f"compute_outputs failed for node {node.id} ({node.type}): "
                f"item {idx} is not an OutputMetadata, got "
                f"{type(item).__name__}"
            )
        if item.name in seen:
            raise CompilationError(
                f"compute_outputs failed for node {node.id} ({node.type}): "
                f"duplicate output name {item.name!r}"
            )
        seen.add(item.name)

    # Unless the node opts into dynamic handles, every statically declared
    # output name must still be present — otherwise type-checked edges and
    # shared-reference produce decorations would silently dangle.
    if not getattr(node_def, "dynamic_handles", False):
        static_names = {o.name for o in static_outputs}
        missing = static_names - seen
        if missing:
            raise CompilationError(
                f"compute_outputs failed for node {node.id} ({node.type}): "
                f"hook dropped statically declared output(s) "
                f"{sorted(missing)!r}; declare ``dynamic_handles=True`` to "
                f"opt out of static-handle preservation."
            )

    return tuple(result)
