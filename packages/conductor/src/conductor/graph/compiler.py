"""``compile_graph`` — from a ``Graph`` and a registry to a ``CompiledGraph``.

Pure: no session, no I/O, no loading. Every definition the graph names
must already be in the registry; a host that had to load one built it
and called ``NodeRegistry.extended_with`` first.

Everything wrong with the graph comes back as a ``Problem`` on the
result rather than raising, so an editor can show a half-finished graph
with its faults marked. What raises is a caller asking something no
graph can produce — an input a node does not have, a node that did not
resolve.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

from pydantic import TypeAdapter, ValidationError

from conductor.dtype import DType
from conductor.graph.binding import Edges, many, static_values
from conductor.graph.compiled import CompiledGraph
from conductor.graph.lifting import derive
from conductor.graph.model import Graph, GraphNode
from conductor.graph.problem import (
    Problem,
    cycle,
    stale_binding,
    unknown_node_type,
    unknown_node_version,
)
from conductor.graph.topology import dependencies_of, order_of
from conductor.graph.views import derive_interface, lock_problems
from conductor.metadata import Input, Roster
from conductor.node import GraphVersion, NodeVersion
from conductor.registry import NodeRegistry
from conductor.series import Series
from conductor.widgets import ConnectionList


def compile_graph(graph: Graph, registry: NodeRegistry) -> CompiledGraph:
    """Compile ``graph`` against ``registry``.

    Six passes, each reading the one before: the nodes (duplicate ids),
    pins (which version each node uses), rosters (what each node's hooks
    say its inputs are, with the statics typed), bindings (do the stored
    bindings fit the rosters), order (cycles), and edges (what arrives on
    every field, lifting, outputs completed); then the two roster rules
    on every completed roster, and the graph's interface, read off them.
    A node that fails a pass carries a fatal ``Problem`` and drops out of
    the passes after it.
    """
    problems: list[Problem] = []
    nodes = _placements(graph, problems)
    versions = _pins(nodes, registry, problems)
    rosters, statics = _rosters(nodes, versions, registry, problems)
    broken = _check_bindings(nodes, rosters, problems)
    dependencies = dependencies_of(nodes.values())
    order, cyclic = order_of(dependencies)
    problems.extend(cycle(node_id) for node_id in sorted(cyclic))
    order = tuple(node_id for node_id in order if node_id in versions)
    lifting = derive(
        [nodes[node_id] for node_id in order if node_id in rosters and node_id not in broken],
        rosters, versions, registry, statics,
    )
    problems.extend(lifting.problems)
    rosters = {**rosters, **lifting.rosters}
    problems.extend(_roster_rules(rosters))
    interface = derive_interface(graph, rosters, versions, dependencies)
    return CompiledGraph(
        _nodes=nodes,
        _registry=registry,
        _versions=versions,
        _rosters=rosters,
        _statics=statics,
        _dependencies=dependencies,
        _order=order,
        _lifted=lifting.lifted,
        _carried=lifting.carried,
        _conditions={},
        _placements=frozenset(),
        _placement_of={node_id: None for node_id in nodes},
        interface=interface,
        _problems=tuple(problems),
    )


def _placements(graph: Graph, problems: list[Problem]) -> dict[str, GraphNode]:
    """Every authored node by id. Two nodes with one id is a fatal problem:
    the second would silently shadow the first everywhere else."""
    nodes: dict[str, GraphNode] = {}
    for node in graph.nodes:
        if node.id in nodes:
            problems.append(Problem(
                code="duplicate_node_id",
                message=f"Two nodes have the id '{node.id}'.",
                fatal=True,
                node_id=node.id,
            ))
            continue
        nodes[node.id] = node
    return nodes


def _pins(
    nodes: dict[str, GraphNode], registry: NodeRegistry, problems: list[Problem]
) -> dict[str, NodeVersion | GraphVersion]:
    """The version each node pins, resolved here and nowhere else.

    ``registry.get(node.type)`` gives the definition and
    ``versions[node.version]`` the version; reading ``versions[cls.current]``
    instead would silently re-point every stored graph at the newest
    signature. Both misses are graph states — a catalog can lose a type, a
    class can drop a version — and become problems. A version may be a
    ``GraphVersion``; expansion inlines it.
    """
    versions: dict[str, NodeVersion | GraphVersion] = {}
    for node in nodes.values():
        definition = registry.get(node.type)
        if definition is None:
            problems.append(unknown_node_type(node.id, node.type))
            continue
        version = definition.versions.get(node.version)
        if version is None:
            problems.append(unknown_node_version(node.id, node.type, node.version))
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

    ``values`` are the author's statics, typed by ``_typed_statics`` over
    the declaration's defaults, so a hook that reads a table's columns or
    a schema's fields off a value parses nothing. Only the inputs are
    asked here; the outputs stay as declared until the edges pass, which
    can tell ``compute_outputs`` what arrives.

    On a version whose interface is open (``**inputs``), every connected name
    that is not a declared parameter becomes an ``Input``: typed ``Any`` or
    ``Series[Any]`` until the edges pass binds it, edited by edges,
    titled by its name.

    Nothing else is asked of the node. A static its type cannot read is
    the ``invalid_static`` problem ``_typed_statics`` reports; edges
    problems are compile's own; two fields sharing a name is
    ``duplicate_field_name``, checked once every roster is complete.
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
            # One Input per edge: `Series[Any]` when each edge is a reduction
            # ("series"), `Any` when each is received whole ("single"). The
            # edges pass types it from what arrives.
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
    """Every static on ``node``, read through its field's declared type.

    A stored ``Static`` holds JSON — a file comes back as a dict, an
    authored schema as a list — and every reader downstream wants the
    value, not its JSON form, so each static is typed once here. A
    sequence the declared type cannot read as one value is read as a
    sequence of values, which keeps a list-shaped scalar (a schema) one
    value while three uploaded files are three rows. A value the type
    cannot read at all is a fatal ``invalid_static``, carrying whatever
    the type's constructor said. A static on a parameter typed ``Any`` is
    skipped: only an edge can type it, and the bindings pass reports it.

    The invariant every reader relies on: a typed static is a ``list``
    exactly when it carries many values.
    """
    declared = {i.name: i for i in inputs}
    typed: dict[str, Any] = {}
    for name, value in static_values(node.bindings).items():
        inp = declared.get(name)
        if inp is None or inp.dtype is Any:
            continue  # a stale binding, or a static where only an edge binds; the bindings pass reports it
        try:
            typed[name] = _typed_static(inp, value)
        except (ValidationError, TypeError, ValueError) as invalid:
            said = _what_the_type_said(invalid)
            problems.append(Problem(
                code="invalid_static",
                message=f"The value in '{name}' cannot be read as the field's type."
                + (f" {said}" if said else ""),
                fatal=True,
                node_id=node.id,
                field=name,
                details={"reason": said} if said else {},
            ))
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
    """Check the stored bindings against the rosters.

    Reports a lock on a field the node does not have (``unknown_locked_field``,
    not fatal), a binding on a field the node does not have (``stale_binding``,
    not fatal), an edge into an input without a handle, an edge from a node
    that does not exist, and a required input nothing binds
    (``unbound_required``). A parameter typed ``Any`` is typed by its edge
    and nothing else, so with no connect it is ``unbound_required`` as well.
    Returns the ids of the nodes whose edges are wrong; nothing about
    their shape is derived afterwards.
    """
    broken: set[str] = set()
    problems.extend(lock_problems(nodes, rosters))
    for node_id, roster in rosters.items():
        node = nodes[node_id]
        declared = {i.name: i for i in roster.inputs}

        for name, binding in node.bindings.items():
            if name not in declared:
                problems.append(stale_binding(node_id, name))
                continue
            if not isinstance(binding, Edges):
                continue
            target = declared[name]
            if not target.show_handle:
                broken.add(node_id)
                problems.append(_at(node_id, name, code="edge_into_closed_handle",
                    message=f"Field '{name}' has no handle, so nothing can be connected to it."))
            for ref in binding.refs:
                if ref.node_id not in nodes:
                    broken.add(node_id)
                    problems.append(_at(node_id, name, code="unknown_ref_node",
                        message=f"Field '{name}' is connected to '{ref.node_id}', which is not in the flow.",
                        details={"source_node": ref.node_id}))

        for inp in roster.inputs:
            if inp.dtype is Any:
                if not isinstance(node.bindings.get(inp.name), Edges):
                    broken.add(node_id)
                    problems.append(_at(node_id, inp.name, code="unbound_required",
                        message="Nothing is connected to the field."))
            elif not inp.optional and inp.name not in node.bindings:
                broken.add(node_id)
                problems.append(_at(node_id, inp.name, code="unbound_required",
                    message="Nothing is connected to the field."))
    return frozenset(broken)


def _at(node_id: str, field: str, *, code: str, message: str, fatal: bool = True, details: Mapping[str, Any] | None = None) -> Problem:
    return Problem(code=code, message=message, fatal=fatal, node_id=node_id, field=field, details=details or {})


def _roster_rules(rosters: dict[str, Roster]) -> list[Problem]:
    """Two rules checked on every completed roster, after the edges pass.

    A field name is unique within a node across inputs and outputs: a
    ``Ref`` must name one field, and a roster a hook computed can break
    that just as a declaration could (``Interface.of`` refuses the
    latter). And every field with a handle — every output, and every input
    not closed with ``show_handle=False`` — must carry a ``DType`` or
    ``Any``, or nothing could connect it (``handle_needs_dtype``). An ``Any``
    still untyped is not a violation: unconnected, it is ``unbound_required``,
    already reported.
    """
    found: list[Problem] = []
    for node_id, roster in rosters.items():
        seen: set[str] = set()
        for declared in (*roster.inputs, *roster.outputs):
            if declared.name in seen:
                found.append(_at(node_id, declared.name, code="duplicate_field_name",
                    message=f"The node has two fields named '{declared.name}'."))
            seen.add(declared.name)
        handles = (*(inp for inp in roster.inputs if inp.show_handle), *roster.outputs)
        for declared in handles:
            if declared.dtype is not Any and not (
                isinstance(declared.dtype, type) and issubclass(declared.dtype, DType)
            ):
                found.append(_at(node_id, declared.name, code="handle_needs_dtype",
                    message=f"Field '{declared.name}' has a handle but no type that can travel on an edge."))
    return found
