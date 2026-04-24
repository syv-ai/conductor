"""Graph compilation — validate and produce an immutable execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from conductor.errors import CompilationError
from conductor.expr import ExpressionError
from conductor.expr import parse as parse_expr
from conductor.graph.model import Flow, FlowDependency, GraphEdge, GraphNode
from conductor.graph.shared_refs import validate_and_build_consume_map
from conductor.graph.topology import build_edge_map, build_incoming_map, topological_sort
from conductor.graph.type_check import TypeWarning, check_consume_types, check_edge_types

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry


class ExtensionResolver(Protocol):
    """Implemented by host applications for custom node types."""

    def is_known_type(self, node_type: str) -> bool: ...
    def create_executor(self, node_type: str) -> Any: ...


# ---------------------------------------------------------------------------
# Decision information (populated when a decision node is present)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionGuard:
    """One outgoing edge of a decision node with its parsed guard."""

    edge_id: str
    source_handle: str | None
    target_id: str
    target_handle: str | None
    when: Any  # parsed Expression or None for else
    priority: int


@dataclass(frozen=True)
class CompiledGraph:
    """Immutable, validated, ready-to-execute graph."""

    execution_order: tuple[str, ...]
    edge_map: dict[tuple[str, str], list[tuple[str, str, str]]]
    node_map: dict[str, GraphNode]
    registry: Any  # NodeRegistry
    extension_resolver: ExtensionResolver | None = None
    compound_nodes: dict[str, Any] = field(default_factory=dict)
    managed_ids: frozenset[str] = field(default_factory=frozenset)
    type_warnings: tuple[TypeWarning, ...] = field(default_factory=tuple)
    # (target_id, target_handle) -> (producer_id, output_handle)
    consume_map: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    # managed_node_id -> its region's start_id (for scheduling redirection)
    managed_to_region_start: dict[str, str] = field(default_factory=dict)
    # Flat tuple of edges so the engine can reach back to when/priority.
    edges: tuple[GraphEdge, ...] = ()
    # decision_node_id -> sorted list of DecisionGuard (highest priority first)
    decision_guards: dict[str, tuple[DecisionGuard, ...]] = field(default_factory=dict)
    # Flow-level metadata
    flow: Flow | None = None
    # Parsed CEL expressions: (node_id, "idempotency_key") -> Expression
    compiled_expressions: dict[tuple[str, str], Any] = field(default_factory=dict)
    # Nodes that are only ever dispatched during compensation (they're the
    # target of a node's ``compensation=`` field). Excluded from normal
    # scheduling.
    compensation_node_ids: frozenset[str] = field(default_factory=frozenset)
    # target_id -> [(target_handle, source_id, source_handle, edge_id), ...]
    # Inverted edge view — faster than scanning edge_map per node.
    incoming_map: dict[str, list[tuple[str, str, str, str]]] = field(default_factory=dict)


def compile(
    nodes: list[GraphNode] | None = None,
    edges: list[GraphEdge] | None = None,
    registry: "NodeRegistry" = None,
    *,
    compound_types: list[Any] | None = None,
    extension_resolver: ExtensionResolver | None = None,
    strict_types: bool = False,
    flow: Flow | None = None,
    subprocess_registry: Any = None,
) -> CompiledGraph:
    """Validate and compile a graph into an immutable execution plan.

    Accepts either a ``Flow`` via ``flow=`` or the traditional ``nodes``
    + ``edges`` args. ``subprocess_registry`` is forwarded to subprocess
    nodes so they can look up their target flow by id.

    Args:
        strict_types: If True, type mismatches raise CompilationError.
                      If False (default), they're returned as warnings
                      on CompiledGraph.type_warnings.
    """
    if flow is not None:
        nodes = flow.nodes
        edges = flow.edges
    if nodes is None or edges is None:
        raise TypeError("compile() needs either `flow=` or both `nodes=` and `edges=`")
    if registry is None:
        raise TypeError("compile() needs a `registry`")

    node_map = {n.id: n for n in nodes}

    # 1. Validate node types
    for node in nodes:
        known = registry.contains(node.type)
        if not known and extension_resolver:
            known = extension_resolver.is_known_type(node.type)
        if not known:
            raise CompilationError(f"Unknown node type: '{node.type}'")

    # 2. Validate edges reference existing nodes
    for edge in edges:
        if edge.source not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent source node: '{edge.source}'"
            )
        if edge.target not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent target node: '{edge.target}'"
            )

    # 3. Validate dependencies (uses: lists against flow.dependencies).
    #    Always run when a flow is provided so nodes with `uses` that reference
    #    non-existent deps are caught even if the flow has no manifest yet.
    if flow is not None:
        _validate_dependency_usage(nodes, registry, flow.dependencies)

    # 4. Validate compensation references
    _validate_compensation(nodes, node_map)

    # 5. Discover compound regions first, so shared-ref validation can know
    #    which nodes are inside them (producers inside regions are rejected).
    compound_nodes: dict[str, Any] = {}
    managed_ids: set[str] = set()

    if compound_types:
        from conductor.graph.regions import discover_regions

        for ct, region in discover_regions(nodes, edges, compound_types):
            compound_nodes[region.start_id] = ct  # placeholder; rebuilt below
            managed_ids.update(region.body_ids)
            # Only mark end as managed when it's a separate node — a single
            # node subprocess region uses start_id == end_id and must remain
            # schedulable.
            if region.end_id != region.start_id:
                managed_ids.add(region.end_id)

    # 6. Validate shared references (produce/consume), build consume map
    consume_map, shared_warnings = validate_and_build_consume_map(
        nodes, edges, node_map, frozenset(managed_ids), registry,
    )

    # 7. Topological sort — edges + consume dependencies participate equally
    extra_deps = [
        (producer_id, target_id)
        for (target_id, _), (producer_id, _) in consume_map.items()
    ]
    order = topological_sort(nodes, edges, extra_dependencies=extra_deps)

    # 8. Build edge maps — forward (for resolver) and inverted (for fast
    #    per-node incoming lookup).
    edge_map = build_edge_map(edges)
    incoming_map = build_incoming_map(edges)

    # 9. Now that we have the topological order, rebuild the compound node
    #    executors with the proper order (matching pre-refactor behavior) and
    #    build the managed-node → region-start lookup used for scheduling.
    compound_nodes = {}
    managed_to_region_start: dict[str, str] = {}
    if compound_types:
        from conductor.graph.regions import discover_regions

        for ct, region in discover_regions(nodes, edges, compound_types):
            executor = ct.factory(region, tuple(order))
            # Allow the factory to set the subprocess registry if supported.
            if subprocess_registry is not None and hasattr(executor, "set_subprocess_registry"):
                executor.set_subprocess_registry(subprocess_registry)
            compound_nodes[region.start_id] = executor
            for body_id in region.body_ids:
                managed_to_region_start[body_id] = region.start_id
            if region.end_id != region.start_id:
                managed_to_region_start[region.end_id] = region.start_id

    # 10. Type-check edges and consume bindings
    edge_warnings = check_edge_types(edges, node_map, registry)
    consume_warnings = check_consume_types(consume_map, node_map, registry)
    type_warnings = [*edge_warnings, *consume_warnings, *shared_warnings]

    # 11. Validate and pre-parse decision guards (and edge ``when`` in general)
    decision_guards, expr_warnings = _compile_decisions(nodes, edges, registry, type_warnings)
    type_warnings.extend(expr_warnings)

    # 12. Pre-parse other CEL expressions: idempotency keys, timeout info lives
    #     in node definitions rather than on instances.
    compiled_expressions = _compile_idempotency_expressions(nodes, registry)

    # Strict mode promotes only real mismatches (not informational warnings
    # like label collisions) to an error.
    strict_fatal = [w for w in type_warnings if w.code == "type-mismatch"]
    if strict_types and strict_fatal:
        messages = [w.message for w in strict_fatal]
        raise CompilationError(
            f"Type errors in {len(strict_fatal)} connection(s):\n"
            + "\n".join(f"  - {m}" for m in messages)
        )

    # Compensation nodes should never run as regular nodes — the engine
    # only dispatches them via ``_run_compensation``.
    compensation_node_ids = frozenset(
        n.compensation for n in nodes if n.compensation is not None
    )

    return CompiledGraph(
        execution_order=tuple(order),
        edge_map=edge_map,
        node_map=node_map,
        registry=registry,
        extension_resolver=extension_resolver,
        compound_nodes=compound_nodes,
        managed_ids=frozenset(managed_ids),
        type_warnings=tuple(type_warnings),
        consume_map=consume_map,
        managed_to_region_start=managed_to_region_start,
        edges=tuple(edges),
        decision_guards=decision_guards,
        flow=flow,
        compiled_expressions=compiled_expressions,
        compensation_node_ids=compensation_node_ids,
        incoming_map=incoming_map,
    )


# ---------------------------------------------------------------------------
# Decision node validation + guard parsing
# ---------------------------------------------------------------------------


def _compile_decisions(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    registry: "NodeRegistry",
    existing_warnings: list[TypeWarning],
) -> tuple[dict[str, tuple[DecisionGuard, ...]], list[TypeWarning]]:
    """Validate every decision node and pre-parse its outgoing guards."""
    warnings: list[TypeWarning] = []
    decision_guards: dict[str, tuple[DecisionGuard, ...]] = {}

    # Group edges by source (one pass)
    by_source: dict[str, list[GraphEdge]] = {}
    for e in edges:
        by_source.setdefault(e.source, []).append(e)

    for node in nodes:
        node_def = registry.get(node.type)
        is_decision = node_def is not None and node_def.is_decision
        outgoing = by_source.get(node.id, [])

        if not is_decision:
            # Non-decision nodes: edges with `when` are disallowed.
            for e in outgoing:
                if e.when is not None:
                    raise CompilationError(
                        f"Edge '{e.id}' has a `when` guard but its source "
                        f"'{node.id}' is not a decision node. Guards are only "
                        f"allowed on outgoing edges of decision nodes."
                    )
            continue

        # Decision node: validate exactly one else edge + at least one guard
        else_edges = [e for e in outgoing if e.when is None]
        guarded_edges = [e for e in outgoing if e.when is not None]

        if not outgoing:
            raise CompilationError(
                f"Decision node '{node.id}' has no outgoing edges — a "
                f"decision node must have at least one guarded edge and "
                f"exactly one else edge."
            )
        if len(else_edges) != 1:
            raise CompilationError(
                f"Decision node '{node.id}' must have exactly one else edge "
                f"(no `when`), got {len(else_edges)}."
            )
        if not guarded_edges:
            raise CompilationError(
                f"Decision node '{node.id}' has only an else edge. Add at "
                f"least one guarded edge with a `when` expression."
            )

        parsed_guards: list[DecisionGuard] = []
        for e in outgoing:
            if e.when is None:
                parsed_guards.append(DecisionGuard(
                    edge_id=e.id,
                    source_handle=e.source_handle,
                    target_id=e.target,
                    target_handle=e.target_handle,
                    when=None,
                    priority=e.priority,
                ))
            else:
                try:
                    expr = parse_expr(e.when)
                except ExpressionError as exc:
                    raise CompilationError(
                        f"Edge '{e.id}' has an invalid `when` expression "
                        f"{e.when!r}: {exc}"
                    ) from exc
                parsed_guards.append(DecisionGuard(
                    edge_id=e.id,
                    source_handle=e.source_handle,
                    target_id=e.target,
                    target_handle=e.target_handle,
                    when=expr,
                    priority=e.priority,
                ))

        # Order: guards by priority desc (else pushed to the end)
        parsed_guards.sort(
            key=lambda g: (g.when is None, -g.priority),
        )
        decision_guards[node.id] = tuple(parsed_guards)

    return decision_guards, warnings


def _compile_idempotency_expressions(
    nodes: list[GraphNode],
    registry: "NodeRegistry",
) -> dict[tuple[str, str], Any]:
    """Pre-parse idempotency_key CEL expressions for quick runtime lookup."""
    out: dict[tuple[str, str], Any] = {}
    for node in nodes:
        node_def = registry.get(node.type)
        if node_def is None or not node_def.idempotency_key:
            continue
        try:
            out[(node.id, "idempotency_key")] = parse_expr(node_def.idempotency_key)
        except ExpressionError as e:
            raise CompilationError(
                f"Node '{node.id}' ({node.type}) has invalid idempotency_key "
                f"expression {node_def.idempotency_key!r}: {e}"
            ) from e
    return out


# ---------------------------------------------------------------------------
# Dependency + compensation validation
# ---------------------------------------------------------------------------


def _validate_dependency_usage(
    nodes: list[GraphNode],
    registry: "NodeRegistry",
    dependencies: tuple[FlowDependency, ...],
) -> None:
    """Every node `uses:` entry must reference a declared top-level dependency."""
    declared = {d.id for d in dependencies}
    for node in nodes:
        node_def = registry.get(node.type)
        if node_def is None or not node_def.uses:
            continue
        for dep_id in node_def.uses:
            if dep_id not in declared:
                raise CompilationError(
                    f"Node '{node.id}' ({node.type}) declares it uses "
                    f"dependency '{dep_id}' but it's not in the flow's "
                    f"`dependencies` manifest. Declared: {sorted(declared) or '(none)'}."
                )


def _validate_compensation(
    nodes: list[GraphNode],
    node_map: dict[str, GraphNode],
) -> None:
    """Every ``compensation=`` reference must point at an existing node."""
    for node in nodes:
        if node.compensation is None:
            continue
        if node.compensation not in node_map:
            raise CompilationError(
                f"Node '{node.id}' declares compensation='{node.compensation}' "
                f"but no such node exists in the flow."
            )
        if node.compensation == node.id:
            raise CompilationError(
                f"Node '{node.id}' cannot be its own compensation."
            )
        if node.on_error and node.on_error not in ("fail", "continue", "compensate"):
            raise CompilationError(
                f"Node '{node.id}' has invalid on_error='{node.on_error}'. "
                f"Valid values: fail, continue, compensate."
            )

    # Validate on_error on nodes without compensation as well
    for node in nodes:
        if node.on_error and node.on_error not in ("fail", "continue", "compensate"):
            raise CompilationError(
                f"Node '{node.id}' has invalid on_error='{node.on_error}'. "
                f"Valid values: fail, continue, compensate."
            )
