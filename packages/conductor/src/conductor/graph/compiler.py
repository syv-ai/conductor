"""How a ``CompiledGraph`` is built: one graph compiled, pass by pass.

``CompiledGraph.from_graph(graph, registry)`` is the door; it builds a
``_Compilation`` and asks it to ``build``. Pure: no session, no I/O, no
loading. Every definition the graph names must already be in the
registry; a host that had to load one built it and called
``NodeRegistry.extended_with`` first.

Everything wrong with the graph comes back as a ``Problem`` on the
result rather than raising, so an editor can show a half-finished graph
with its faults marked. What raises is a caller asking something no
graph can produce — an input a node does not have, a node compile could
not resolve.

The passes each read the one before, through the state ``_Compilation``
holds. A node that fails a pass carries a fatal ``Problem`` and drops out
of the passes after it, so nothing is guessed about a node downstream of
a fault.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import TypeAdapter, ValidationError

from conductor.dtype import DType
from conductor.graph.binding import Edges, many, static_values
from conductor.graph.compiled import CompiledGraph
from conductor.graph.conditions import conditions_of
from conductor.graph.expand import Expansion, expand, surfaced
from conductor.graph.lifting import Lifting, derive
from conductor.graph.model import Graph, GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.topology import dependencies_of, order_of
from conductor.graph.views import derive_interface, lock_problems
from conductor.metadata import Input, Roster
from conductor.node import GraphVersion, NodeVersion
from conductor.registry import NodeRegistry
from conductor.series import Series
from conductor.widgets import ConnectionList


class _Compilation:
    """One graph being compiled: the state every pass reads and writes.

    ``build`` runs the passes in order and returns the ``CompiledGraph``.
    Each pass is a method that reads what the passes before it left on
    ``self`` and appends what it finds wrong to ``problems``. Nothing here
    outlives ``build``; the result is the immutable record.
    """

    def __init__(self, graph: Graph, registry: NodeRegistry) -> None:
        self.graph = graph
        self.registry = registry
        self.problems: list[Problem] = []
        #: The nodes the author placed, by id (pass 1).
        self.authored: dict[str, GraphNode] = {}
        #: The version each authored node uses (pass 2).
        self.pinned: dict[str, NodeVersion | GraphVersion] = {}
        #: Who waits for whom, and an execution order, over the authored graph (pass 3).
        self.authored_dependencies: dict[str, frozenset[str]] = {}
        self.authored_order: tuple[str, ...] = ()
        #: The expanded graph: every embedded graph inlined (pass 4).
        self.expansion: Expansion
        #: Each node's inputs and outputs, and the values the author typed, by
        #: expanded id (pass 5; the walk over the edges completes them).
        self.rosters: dict[str, Roster] = {}
        self.statics: dict[str, dict[str, Any]] = {}
        #: Nodes whose stored edges are wrong; the edge walk leaves them out (pass 6).
        self.broken: frozenset[str] = frozenset()
        #: What the walk over the edges decided (pass 7).
        self.lifting: Lifting

    def build(self) -> CompiledGraph:
        """The passes, in order. Each reads the one before:

        1. ``place`` — the nodes by id; two with one id is a problem.
        2. ``pin`` — the version each node uses; an unknown type or version
           is a problem on the node.
        3. ``order`` — who waits for whom and an execution order, over the
           graph as authored; a cycle is a problem.
        4. ``expand`` — every node whose version is a graph is inlined as
           its inner nodes.
        5. ``ask_inputs`` — each node's inputs as its hook answers, with the
           values the author typed read through their declared types.
        6. ``check_bindings`` — the stored bindings against those inputs.
        7. ``walk_edges`` — one walk over the edges that types every field,
           decides which nodes run once per row and completes the outputs.
        8. ``field_rules`` — every completed node's fields have unique names
           and a type an edge can carry.
        9. ``conditions`` — the condition under which each output appears.
        10. ``interface`` — what the graph takes and returns.

        A problem found inside an embedded graph is reported on the node
        the author placed.
        """
        self.place()
        self.pin()
        self.order()
        self.expand()
        self.ask_inputs()
        self.check_bindings()
        self.walk_edges()
        self.field_rules()
        conditions = self.conditions()
        interface = self.interface()
        expansion, lifting = self.expansion, self.lifting
        return CompiledGraph(
            _nodes=expansion.nodes,
            _registry=self.registry,
            _versions={**expansion.versions, **expansion.placement_versions},
            _rosters=self.rosters,
            _statics=self.statics,
            _dependencies=dependencies_of(expansion.nodes.values()),
            _order=expansion.order,
            _lifted=lifting.lifted,
            _carried=lifting.carried,
            _conditions=conditions,
            _placements=frozenset(expansion.placement_versions),
            _placement_of=expansion.placement_of,
            interface=interface,
            _problems=tuple(surfaced(found, expansion.nodes) for found in self.problems),
        )

    # -- the passes -------------------------------------------------------------

    def place(self) -> None:
        """Every node the author placed, by id. Two nodes with one id is a
        fatal problem: the second would silently shadow the first everywhere
        else."""
        for node in self.graph.nodes:
            if node.id in self.authored:
                self.problems.append(problem("duplicate_node_id", node.id))
                continue
            self.authored[node.id] = node

    def pin(self) -> None:
        """The version each node uses, looked up once, here.

        A node stores a ``type`` and a ``version`` number; ``registry.get(type)``
        gives the definition and ``definition.versions[version]`` the version
        record. A stored graph can name a type the catalog has since lost or a
        version the class has since dropped, so either miss is a problem on
        the node rather than an error. A version may be a ``GraphVersion`` —
        an embedded graph — which ``expand`` inlines.
        """
        for node in self.authored.values():
            definition = self.registry.get(node.type)
            if definition is None:
                self.problems.append(problem("unknown_node_type", node.id, node_type=node.type))
                continue
            version = definition.versions.get(node.version)
            if version is None:
                self.problems.append(
                    problem("unknown_node_version", node.id, node_type=node.type, version=node.version)
                )
                continue
            self.pinned[node.id] = version

    def order(self) -> None:
        """Who waits for whom, read off the authored edges, and an execution
        order over it. A node on a cycle is left out of the order and gets a
        fatal problem."""
        self.authored_dependencies = dependencies_of(self.authored.values())
        self.authored_order, cyclic = order_of(self.authored_dependencies)
        self.problems.extend(problem("cycle", node_id) for node_id in sorted(cyclic))

    def expand(self) -> None:
        """Every node whose version is a graph, inlined as its inner nodes
        under its name (``expand``). From here on the passes see the
        expanded graph."""
        self.expansion = expand(self.authored, self.authored_order, self.pinned, self.registry)
        self.problems.extend(self.expansion.problems)

    def ask_inputs(self) -> None:
        """Ask each node which inputs it has, once, on a fresh instance.

        A node's inputs may depend on the values the author typed into it (a
        mode dropdown that adds fields), so ``compute_inputs`` is called with
        those values, each read through its declared type by ``_typed_statics``
        and laid over the declaration's defaults; a hook that reads a table's
        columns or a schema's fields off a value parses nothing. Only the
        inputs are asked here. The outputs stay as declared until the walk over
        the edges, which can tell ``compute_outputs`` what type arrives on each
        connected input.

        On a version that takes ``**inputs``, every connected name that is not
        a declared parameter becomes an ``Input`` of its own: typed ``Any`` or
        ``Series[Any]`` until the edge walk types it, titled by its name.

        Nothing else is asked of the node. A value its type cannot read is the
        ``invalid_static`` problem ``_typed_statics`` reports; two fields
        sharing a name is ``duplicate_field_name``, checked once every node's
        inputs and outputs are complete (``field_rules``).
        """
        for node_id, version in self.expansion.versions.items():
            node = self.expansion.nodes[node_id]
            instance = self.registry.get(node.type)()
            defaults = {i.name: i.default for i in version.interface.inputs if i.optional}
            values = {**defaults, **self._typed_statics(version.interface.inputs, node)}
            inputs = instance.compute_inputs(version.interface.inputs, values)
            added = tuple(i for i in inputs if i.name not in {d.name for d in version.interface.inputs})
            if added:
                values = {**values, **self._typed_statics(added, node)}
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
            self.rosters[node_id] = Roster(inputs=inputs, outputs=version.interface.outputs)
            self.statics[node_id] = values

    def check_bindings(self) -> None:
        """Check the stored bindings against the inputs each node actually has.

        Reports a lock on a field the node does not have (``unknown_locked_field``,
        not fatal), a binding on a field the node does not have (``stale_binding``,
        not fatal), an edge into an input that cannot be connected
        (``show_handle=False``), an edge from a node that does not exist, and a
        required input nothing binds (``unbound_required``). A parameter typed
        ``Any`` gets its type from its edge and nothing else, so with no edge
        it is ``unbound_required`` as well. Leaves in ``broken`` the ids of the
        nodes whose edges are wrong; the edge walk leaves them out, since
        nothing can be derived from a broken edge.
        """
        nodes = self.expansion.nodes
        broken: set[str] = set()
        self.problems.extend(lock_problems(nodes, self.rosters))
        for node_id, roster in self.rosters.items():
            node = nodes[node_id]
            declared = {i.name: i for i in roster.inputs}

            for name, binding in node.bindings.items():
                if name not in declared:
                    self.problems.append(problem("stale_binding", node_id, name))
                    continue
                if not isinstance(binding, Edges):
                    continue
                target = declared[name]
                if not target.show_handle:
                    broken.add(node_id)
                    self.problems.append(problem("edge_into_closed_handle", node_id, name))
                for ref in binding.refs:
                    if ref.node_id not in nodes:
                        broken.add(node_id)
                        self.problems.append(problem("unknown_ref_node", node_id, name, source_node=ref.node_id))

            for inp in roster.inputs:
                if inp.dtype is Any:
                    if not isinstance(node.bindings.get(inp.name), Edges):
                        broken.add(node_id)
                        self.problems.append(problem("unbound_required", node_id, inp.name))
                elif not inp.optional and inp.name not in node.bindings:
                    broken.add(node_id)
                    self.problems.append(problem("unbound_required", node_id, inp.name))
        self.broken = frozenset(broken)

    def walk_edges(self) -> None:
        """One walk over the edges in expanded order (``lifting.derive``):
        what type every field carries, which nodes run once per row and on
        which index, and each node's completed outputs. Nodes whose edges
        are broken are left out."""
        expansion = self.expansion
        self.lifting = derive(
            [
                expansion.nodes[node_id]
                for node_id in expansion.order
                if node_id in self.rosters and node_id not in self.broken
            ],
            self.rosters, expansion.versions, self.registry, self.statics,
            placement_of=expansion.placement_of, members=expansion.members,
        )
        self.problems.extend(self.lifting.problems)
        self.rosters = {**self.rosters, **self.lifting.rosters}

    def field_rules(self) -> None:
        """Two rules checked on every node's completed inputs and outputs, after the edge walk.

        A field name is unique within a node across inputs and outputs, because
        a ``Ref(node, field)`` must name one field; a declaration is checked
        for this when the class is defined, but a hook can compute a set of
        fields that breaks it (``duplicate_field_name``). And every field that
        can be connected — every output, and every input not closed with
        ``show_handle=False`` — must carry a ``DType`` or ``Any``, or nothing
        could connect it (``handle_needs_dtype``). An ``Any`` still untyped is
        not a violation here: unconnected, it is ``unbound_required``, already
        reported.
        """
        for node_id, roster in self.rosters.items():
            seen: set[str] = set()
            for declared in (*roster.inputs, *roster.outputs):
                if declared.name in seen:
                    self.problems.append(problem("duplicate_field_name", node_id, declared.name))
                seen.add(declared.name)
            handles = (*(inp for inp in roster.inputs if inp.show_handle), *roster.outputs)
            for declared in handles:
                if declared.dtype is not Any and not (
                    isinstance(declared.dtype, type) and issubclass(declared.dtype, DType)
                ):
                    self.problems.append(problem("handle_needs_dtype", node_id, declared.name))

    def conditions(self) -> Any:
        """The condition under which each output appears, from the ``choice``
        groups of the nodes that run once (``conditions_of``)."""
        expansion, lifting = self.expansion, self.lifting
        return conditions_of(
            [expansion.nodes[node_id] for node_id in expansion.order if node_id in lifting.lifted],
            self.rosters, lifting.lifted,
        )

    def interface(self) -> Any:
        """What the graph takes and returns, read off the authored graph
        (``derive_interface``). An embedded graph shows its version's
        interface as its fields from here on."""
        for placement, version in self.expansion.placement_versions.items():
            self.rosters[placement] = Roster(inputs=version.interface.inputs, outputs=version.interface.outputs)
        return derive_interface(
            self.graph,
            {n: self.rosters[n] for n in self.authored if n in self.rosters},
            self.pinned,
            self.authored_dependencies,
        )

    # -- the values the author typed --------------------------------------------

    def _typed_statics(self, inputs: tuple[Any, ...], node: GraphNode) -> dict[str, Any]:
        """Every value the author typed into ``node``, read through its field's declared type.

        A stored ``Static`` holds JSON — a file comes back as a dict, an
        authored schema as a list — and every reader downstream wants the
        value, not its JSON form, so each is converted once here. A sequence
        the declared type cannot read as one value is read as a sequence of
        values, which keeps a list-shaped scalar (a schema) one value while
        three uploaded files are three rows. A value the type cannot read at
        all is a fatal ``invalid_static``, carrying whatever the type's
        constructor said. A value on a parameter typed ``Any`` is skipped:
        only an edge can give that parameter a type, and ``check_bindings``
        reports it as unbound.

        The invariant every reader relies on: the converted value is a
        ``list`` exactly when the author typed many values.
        """
        declared = {i.name: i for i in inputs}
        typed: dict[str, Any] = {}
        for name, value in static_values(node.bindings).items():
            inp = declared.get(name)
            if inp is None or inp.dtype is Any:
                continue  # a field the node lacks, or one only an edge can type; check_bindings reports both
            try:
                typed[name] = _typed_static(inp, value)
            except (ValidationError, TypeError, ValueError) as invalid:
                said = _what_the_type_said(invalid)
                self.problems.append(problem("invalid_static", node.id, name, **({"reason": said} if said else {})))
        return typed


@cache
def _adapter(dtype: type) -> TypeAdapter[Any]:
    return TypeAdapter(dtype)


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
    """``value`` read through ``inp``'s declared type: a sequence element by
    element for a ``Series[X]`` input; one value, or a sequence of values
    when the type refuses the whole, for anything else."""
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
