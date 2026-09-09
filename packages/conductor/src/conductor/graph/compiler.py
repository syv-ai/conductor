"""``compile_graph`` — from a ``Graph`` and a registry to a ``CompiledGraph``.

Pure: no session, no I/O, no loading. Every definition the graph names
must already be in the registry; a host that had to load one built it
and called ``NodeRegistry.extended_with`` first.

Everything wrong with the graph comes back as a ``Problem`` on the
result rather than raising, so an editor can show a half-finished graph
with its faults marked. What raises is a caller asking something no
graph can produce — an input a node does not have, a node compile could
not resolve.

The passes below each read the one before. A node that fails a pass
carries a fatal ``Problem`` and drops out of the passes after it, so
nothing is guessed about a node downstream of a fault.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import TypeAdapter, ValidationError

from conductor.dtype import DType
from conductor.graph.binding import Edges, many, static_values
from conductor.graph.compiled import CompiledGraph
from conductor.graph.conditions import conditions_of
from conductor.graph.expand import expand, surfaced
from conductor.graph.lifting import derive
from conductor.graph.model import Graph, GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.topology import dependencies_of, order_of
from conductor.graph.views import derive_interface, lock_problems
from conductor.metadata import Input, Roster
from conductor.node import GraphVersion, NodeVersion
from conductor.registry import NodeRegistry
from conductor.series import Series
from conductor.widgets import ConnectionList


def compile_graph(graph: Graph, registry: NodeRegistry) -> CompiledGraph:
    """Compile ``graph`` against ``registry``.

    In order: the nodes by id (two with one id is a problem); the version
    each node uses; an execution order over the graph as authored (a cycle
    is a problem); embedded flows inlined as their inner nodes (``expand``);
    each node's inputs and outputs as its hooks answer, with the values
    the author typed read through their declared types; the stored
    bindings checked against those inputs; one walk over the edges that
    types every field, decides which nodes run once per row and completes
    the outputs (``lifting.derive``); the condition under which each output
    appears; and finally what the graph takes and returns. A problem found
    inside an embedded flow is reported on the node the author placed.
    """
    problems: list[Problem] = []
    authored = _nodes_by_id(graph, problems)
    pinned = _pinned_versions(authored, registry, problems)
    authored_dependencies = dependencies_of(authored.values())
    authored_order, cyclic = order_of(authored_dependencies)
    problems.extend(problem("cycle", node_id) for node_id in sorted(cyclic))
    expansion = expand(authored, authored_order, pinned, registry, problems)
    nodes, versions = expansion.nodes, expansion.versions
    rosters, statics = _rosters(nodes, versions, registry, problems)
    broken = _check_bindings(nodes, rosters, problems)
    dependencies = dependencies_of(nodes.values())
    lifting = derive(
        [nodes[node_id] for node_id in expansion.order if node_id in rosters and node_id not in broken],
        rosters, versions, registry, statics,
        placement_of=expansion.placement_of, members=expansion.members,
    )
    problems.extend(lifting.problems)
    rosters = {**rosters, **lifting.rosters}
    problems.extend(_roster_rules(rosters))
    conditions = conditions_of(
        [nodes[node_id] for node_id in expansion.order if node_id in lifting.lifted], rosters, lifting.lifted
    )
    placements = frozenset(expansion.placement_versions)
    for placement, version in expansion.placement_versions.items():
        rosters[placement] = Roster(inputs=version.interface.inputs, outputs=version.interface.outputs)
    interface = derive_interface(graph, {n: rosters[n] for n in authored if n in rosters}, pinned, authored_dependencies)
    return CompiledGraph(
        _nodes=nodes,
        _registry=registry,
        _versions={**versions, **expansion.placement_versions},
        _rosters=rosters,
        _statics=statics,
        _dependencies=dependencies,
        _order=expansion.order,
        _lifted=lifting.lifted,
        _carried=lifting.carried,
        _conditions=conditions,
        _placements=placements,
        _placement_of=expansion.placement_of,
        interface=interface,
        _problems=tuple(surfaced(problem, nodes) for problem in problems),
    )


def _nodes_by_id(graph: Graph, problems: list[Problem]) -> dict[str, GraphNode]:
    """Every node the author placed, by id. Two nodes with one id is a fatal
    problem: the second would silently shadow the first everywhere else."""
    nodes: dict[str, GraphNode] = {}
    for node in graph.nodes:
        if node.id in nodes:
            problems.append(problem("duplicate_node_id", node.id))
            continue
        nodes[node.id] = node
    return nodes


def _pinned_versions(
    nodes: dict[str, GraphNode], registry: NodeRegistry, problems: list[Problem]
) -> dict[str, NodeVersion | GraphVersion]:
    """The version each node uses, looked up once, here.

    A node stores a ``type`` and a ``version`` number; ``registry.get(type)``
    gives the definition and ``definition.versions[version]`` the version
    record. A stored graph can name a type the catalog has since lost or a
    version the class has since dropped, so either miss is a problem on
    the node rather than an error. A version may be a ``GraphVersion`` —
    an embedded flow — which ``expand`` inlines.
    """
    versions: dict[str, NodeVersion | GraphVersion] = {}
    for node in nodes.values():
        definition = registry.get(node.type)
        if definition is None:
            problems.append(problem("unknown_node_type", node.id, node_type=node.type))
            continue
        version = definition.versions.get(node.version)
        if version is None:
            problems.append(problem("unknown_node_version", node.id, node_type=node.type, version=node.version))
            continue
        versions[node.id] = version
    return versions


def _rosters(
    nodes: dict[str, GraphNode],
    versions: dict[str, NodeVersion],
    registry: NodeRegistry,
    problems: list[Problem],
) -> tuple[dict[str, Roster], dict[str, dict[str, Any]]]:
    """Ask each node which inputs it has, once, on a fresh instance.

    A node's inputs may depend on the values the author typed into it (a
    mode dropdown that adds fields), so ``compute_inputs`` is called with
    those values, each read through its declared type by ``_typed_statics``
    and laid over the declaration's defaults; a hook that reads a table's
    columns or a schema's fields off a value parses nothing. Only the
    inputs are asked here. The outputs stay as declared until the walk over
    the edges (``lifting.derive``), which can tell ``compute_outputs`` what
    type arrives on each connected input.

    On a version that takes ``**inputs``, every connected name that is not
    a declared parameter becomes an ``Input`` of its own: typed ``Any`` or
    ``Series[Any]`` until the edge walk types it, titled by its name.

    Nothing else is asked of the node. A value its type cannot read is the
    ``invalid_static`` problem ``_typed_statics`` reports; two fields
    sharing a name is ``duplicate_field_name``, checked once every node's
    inputs and outputs are complete.
    """
    rosters: dict[str, Roster] = {}
    statics: dict[str, dict[str, Any]] = {}
    for node_id, version in versions.items():
        node = nodes[node_id]
        instance = registry.get(node.type)()
        defaults = {i.name: i.default for i in version.interface.inputs if i.optional}
        values = {**defaults, **_typed_statics(version.interface.inputs, node, problems)}
        inputs = instance.compute_inputs(version.interface.inputs, values)
        added = tuple(i for i in inputs if i.name not in {d.name for d in version.interface.inputs})
        if added:
            values = {**values, **_typed_statics(added, node, problems)}
        if version.interface.open is not None:
            # One Input per edge: `Series[Any]` when the node reduces each
            # ("series"), `Any` when it receives each whole ("single"). The
            # edge walk types it from what arrives.
            shape = Series[Any] if version.interface.open == "series" else Any
            named = {i.name for i in inputs}
            inputs = (*inputs, *(
                Input(name=name, dtype=shape, title=name, widget=ConnectionList(title=name))
                for name, binding in node.bindings.items()
                if name not in named and isinstance(binding, Edges)
            ))
        rosters[node_id] = Roster(inputs=inputs, outputs=version.interface.outputs)
        statics[node_id] = values
    return rosters, statics


@cache
def _adapter(dtype: type) -> TypeAdapter[Any]:
    return TypeAdapter(dtype)


def _typed_statics(
    inputs: tuple[Any, ...], node: GraphNode, problems: list[Problem]
) -> dict[str, Any]:
    """Every value the author typed into ``node``, read through its field's declared type.

    A stored ``Static`` holds JSON — a file comes back as a dict, an
    authored schema as a list — and every reader downstream wants the
    value, not its JSON form, so each is converted once here. A sequence
    the declared type cannot read as one value is read as a sequence of
    values, which keeps a list-shaped scalar (a schema) one value while
    three uploaded files are three rows. A value the type cannot read at
    all is a fatal ``invalid_static``, carrying whatever the type's
    constructor said. A value on a parameter typed ``Any`` is skipped:
    only an edge can give that parameter a type, and ``_check_bindings``
    reports it as unbound.

    The invariant every reader relies on: the converted value is a
    ``list`` exactly when the author typed many values.
    """
    declared = {i.name: i for i in inputs}
    typed: dict[str, Any] = {}
    for name, value in static_values(node.bindings).items():
        inp = declared.get(name)
        if inp is None or inp.dtype is Any:
            continue  # a field the node lacks, or one only an edge can type; _check_bindings reports both
        try:
            typed[name] = _typed_static(inp, value)
        except (ValidationError, TypeError, ValueError) as invalid:
            said = _what_the_type_said(invalid)
            problems.append(problem("invalid_static", node.id, name, **({"reason": said} if said else {})))
    return typed


def _what_the_type_said(invalid: Exception) -> str:
    """The sentence the type's own constructor raised, if it raised one.

    A type's constraints live in its constructor, and a constructor
    written for people says something specific — "the field must not be
    empty", "every row needs three cells". That sentence is the one thing
    that tells an author what to change, so it is appended to the generic
    ``invalid_static`` message. pydantic keeps a validator's ``ValueError``
    under ``ctx["error"]``; anything pydantic says on its own is about
    JSON shape and is not shown.
    """
    if isinstance(invalid, ValidationError):
        errors = invalid.errors()
        cause = errors[0].get("ctx", {}).get("error") if errors else None
        return str(cause) if isinstance(cause, ValueError) else ""
    return str(invalid) if isinstance(invalid, ValueError) else ""


def _typed_static(inp: Any, value: Any) -> Any:
    element = getattr(inp.dtype, "element", None)
    if element is not None:
        # A Series[X] input takes the whole sequence, each element typed.
        if not many(value):
            raise TypeError("a Series input takes a sequence")
        return [_adapter(element).validate_python(v) for v in value]
    try:
        return _adapter(inp.dtype).validate_python(value)
    except ValidationError:
        if many(value):
            return [_adapter(inp.dtype).validate_python(v) for v in value]
        raise


def _check_bindings(
    nodes: dict[str, GraphNode], rosters: dict[str, Roster], problems: list[Problem]
) -> frozenset[str]:
    """Check the stored bindings against the inputs each node actually has.

    Reports a lock on a field the node does not have (``unknown_locked_field``,
    not fatal), a binding on a field the node does not have (``stale_binding``,
    not fatal), an edge into an input that cannot be connected
    (``show_handle=False``), an edge from a node that does not exist, and a
    required input nothing binds (``unbound_required``). A parameter typed
    ``Any`` gets its type from its edge and nothing else, so with no edge
    it is ``unbound_required`` as well. Returns the ids of the nodes whose
    edges are wrong; the edge walk leaves them out, since nothing can be
    derived from a broken edge.
    """
    broken: set[str] = set()
    problems.extend(lock_problems(nodes, rosters))
    for node_id, roster in rosters.items():
        node = nodes[node_id]
        declared = {i.name: i for i in roster.inputs}

        for name, binding in node.bindings.items():
            if name not in declared:
                problems.append(problem("stale_binding", node_id, name))
                continue
            if not isinstance(binding, Edges):
                continue
            target = declared[name]
            if not target.show_handle:
                broken.add(node_id)
                problems.append(problem("edge_into_closed_handle", node_id, name))
            for ref in binding.refs:
                if ref.node_id not in nodes:
                    broken.add(node_id)
                    problems.append(problem("unknown_ref_node", node_id, name, source_node=ref.node_id))

        for inp in roster.inputs:
            if inp.dtype is Any:
                if not isinstance(node.bindings.get(inp.name), Edges):
                    broken.add(node_id)
                    problems.append(problem("unbound_required", node_id, inp.name))
            elif not inp.optional and inp.name not in node.bindings:
                broken.add(node_id)
                problems.append(problem("unbound_required", node_id, inp.name))
    return frozenset(broken)


def _roster_rules(rosters: dict[str, Roster]) -> list[Problem]:
    """Two rules checked on every node's completed inputs and outputs, after the edge walk.

    A field name is unique within a node across inputs and outputs, because
    a ``Ref(node, field)`` must name one field; a declaration is checked
    for this when the class is defined, but a hook can compute a roster
    that breaks it (``duplicate_field_name``). And every field that can be
    connected — every output, and every input not closed with
    ``show_handle=False`` — must carry a ``DType`` or ``Any``, or nothing
    could connect it (``handle_needs_dtype``). An ``Any`` still untyped is
    not a violation here: unconnected, it is ``unbound_required``, already
    reported.
    """
    found: list[Problem] = []
    for node_id, roster in rosters.items():
        seen: set[str] = set()
        for declared in (*roster.inputs, *roster.outputs):
            if declared.name in seen:
                found.append(problem("duplicate_field_name", node_id, declared.name))
            seen.add(declared.name)
        handles = (*(inp for inp in roster.inputs if inp.show_handle), *roster.outputs)
        for declared in handles:
            if declared.dtype is not Any and not (
                isinstance(declared.dtype, type) and issubclass(declared.dtype, DType)
            ):
                found.append(problem("handle_needs_dtype", node_id, declared.name))
    return found
