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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from conductor.errors import CompilationError
from conductor.graph.topology import build_incoming_map, topological_sort
from conductor.metadata import Output
from conductor.registry.dynamic_outputs import (
    ComputeOutputsContext,
    IncomingBinding,
)

if TYPE_CHECKING:
    from conductor.graph.model import GraphEdge, GraphNode
    from conductor.registry import NodeDefinition


class _DefinitionGet(Protocol):
    """The one lookup shape the resolution engine needs: ``get`` by type.

    Structurally satisfied by ``NodeRegistry``, ``RegistryView`` and the
    mapping adapter below — the engine neither knows nor cares which.
    """

    def get(self, node_type: str) -> "NodeDefinition | None": ...


def resolve_node_outputs(
    node: "GraphNode",
    node_def,
    incoming_edges: list[tuple[str, str, str, str]],
    resolved_outputs: dict[str, tuple[Output, ...]],
    node_map: dict[str, "GraphNode"],
    registry: _DefinitionGet,
) -> tuple[Output, ...]:
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
        registry: Any ``_DefinitionGet`` lookup (registry, view, or
            mapping adapter), for source-side ``NodeDefinition`` lookups
            when the producer has no hook.

    Returns:
        A tuple of ``Output``. When the node has no hook this is
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
            match = Output(
                name=source_handle, type_str="any", label=source_handle,
            )

        bindings.append(IncomingBinding(
            target_handle=target_handle,
            source_node_id=source_id,
            source_handle=source_handle,
            source_output=match,
        ))

    raw_data = dict(node.data or {})

    # Run the node's validation model if it has one. Hooks for nodes with
    # non-trivial widget config (SchemaBuilder, ConnectionList) often want
    # the coerced shape rather than the raw payload, and re-implementing
    # the coercion in every hook is plumbing debt. ``validated_data`` may
    # be ``None`` when the node has no model or when validation fails —
    # the latter is expected during in-progress editing where ``data`` is
    # incomplete. Failures must not crash the resolver.
    validated_data: dict[str, Any] | None = None
    validation_model = getattr(node_def, "validation_model", None)
    if validation_model is not None:
        try:
            validated_data = validation_model(**raw_data).model_dump()
        except Exception:  # noqa: BLE001 — validation failures are expected
            validated_data = None

    ctx = ComputeOutputsContext(
        data=raw_data,
        incoming=tuple(bindings),
        node_id=node.id,
        defaults=static_outputs,
        validated_data=validated_data,
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
            f"hook must return list[Output], got {type(result).__name__}"
        )

    # Validate every entry is Output and names are unique.
    seen: set[str] = set()
    for idx, item in enumerate(result):
        if not isinstance(item, Output):
            raise CompilationError(
                f"compute_outputs failed for node {node.id} ({node.type}): "
                f"item {idx} is not an Output, got "
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


class _MappingLookup:
    """``_DefinitionGet`` over a host-supplied definitions mapping."""

    def __init__(self, definitions: Mapping[str, "NodeDefinition | None"]) -> None:
        self._definitions = definitions

    def get(self, node_type: str) -> "NodeDefinition | None":
        return self._definitions.get(node_type)


def _resolve_in_order(
    order: list[str],
    node_map: "dict[str, GraphNode]",
    incoming_map: dict[str, list[tuple[str, str, str, str]]],
    lookup: _DefinitionGet,
) -> dict[str, tuple[Output, ...]]:
    """The ONE resolution walk — both ``compile()`` (step 8b) and
    :func:`resolve_graph_outputs` are callers.

    ``order`` must be topological over the drawn edges; a superset
    ordering (compile passes one that also honours consume dependencies)
    resolves identically, because bindings are built from drawn edges
    only — every producer a binding reads is resolved before its
    consumer either way.
    """
    resolved: dict[str, tuple[Output, ...]] = {}
    for node_id in order:
        node = node_map[node_id]
        resolved[node_id] = resolve_node_outputs(
            node=node,
            node_def=lookup.get(node.type),
            incoming_edges=incoming_map.get(node_id, []),
            resolved_outputs=resolved,
            node_map=node_map,
            registry=lookup,
        )
    return resolved


def resolve_graph_outputs(
    nodes: "list[GraphNode]",
    edges: "list[GraphEdge]",
    definitions: Mapping[str, "NodeDefinition | None"],
) -> dict[str, tuple[Output, ...]]:
    """Resolve every node's effective outputs in topological order.

    The ahead-of-compile entry to the same engine ``compile()`` runs at
    step 8b (:func:`_resolve_in_order` — one walk, two callers): a hook's
    ``ctx.incoming[*].source_output`` carries the producer's post-hook
    metadata. ``compile()`` answers *executability* (decision gates,
    compound-region completeness, strict types); this answers the weaker
    "what handles does each node expose right now" — well-defined on
    graphs that are not yet executable (a draft mid-edit, an incomplete
    loop region), which is exactly when hosts need it (schema derivation,
    label seeding, palettes).

    ``definitions`` is **required** and keyed by node *type*: the host
    resolves each type however it wants (a plain registry lookup, or a
    synthesized definition for an embedded sub-flow the registry doesn't
    know). A ``None`` value means "known but definition-less" (extension
    semantics — resolves to ``()``); a *missing key* is a host bug and
    raises. For compile-time overlay semantics use
    :class:`~conductor.registry.view.RegistryView`; for this API,
    materialize the mapping from whatever registry-ish holder you have:
    ``{n.type: view.get(n.type) for n in nodes}``.

    Gates (the same structural checks ``compile()`` applies before its
    resolution pass):

    * every ``node.type`` present in ``definitions`` — else
      ``CompilationError`` ("Unknown node type"),
    * every edge endpoint references an existing node — else
      ``CompilationError``,
    * acyclic — else ``CycleDetectionError``,
    * hook results validated by :func:`resolve_node_outputs` (non-list,
      non-``Output`` items, duplicate names, dropped static
      handles without ``dynamic_handles=True``, hook exceptions) — all
      ``CompilationError``.

    On this full-graph walk every producer resolves before its consumers,
    so the per-node engine's defensive source-side fallback never fires —
    the mapping's ``.get`` cannot miss after the definitions gate.
    ``produces``/``consumes`` shared-reference dependencies do not
    participate in the ordering here — they are execution-plumbing that
    ``compile()`` owns.
    """
    node_map = {n.id: n for n in nodes}

    for node in nodes:
        if node.type not in definitions:
            raise CompilationError(f"Unknown node type: '{node.type}'")

    for edge in edges:
        if edge.source not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent source node: "
                f"'{edge.source}'"
            )
        if edge.target not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent target node: "
                f"'{edge.target}'"
            )

    return _resolve_in_order(
        order=topological_sort(nodes, edges),
        node_map=node_map,
        incoming_map=build_incoming_map(edges),
        lookup=_MappingLookup(definitions),
    )
