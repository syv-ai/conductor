"""``CompiledGraph`` — everything the compiler learned about a graph, as one immutable value.

``CompiledGraph.from_graph`` builds it from a ``Graph`` and a ``NodeRegistry``.
Everyone else asks it questions and never reads the nodes' bindings
themselves: which version each node uses, which inputs and outputs it
actually has, where each input's value comes from and how each unit
receives it, what type travels on every field, which nodes run once per
row and in what order, under which condition each output appears, and
what is wrong. The engine is one more caller.

The questions come at three scales. The graph itself answers what it
takes and returns (``interface``), what is wrong with it (``problems``,
``is_runnable``), in what order its nodes run (``execution_order``) and
which decisions a run makes (``decisions``). ``node(node_id)`` returns a
``CompiledNode``, everything the compiler knows about one node; ``field(ref)``
returns a ``CompiledField``, everything it knows about one input or
output. Compile builds each once, holding its own answers, and these two
hand back the same value on every call. The graph stores a ``GraphNode``
(a node as the author saved it) and addresses fields by ``Ref``; what
compile learned about that node or field lives on its ``CompiledNode`` and
``CompiledField``, and a ``CompiledNode`` hands back the stored node as
``graph_node``.

The words this module uses throughout, each defined once here.

A node's interface is the list of inputs and outputs it actually has.
Usually that is what its version declares, but a node may add or drop
fields depending on the values it holds and the types connected to it.

A series is a value with many rows, and an index names where those rows
come from: ten uploaded documents are a series of ten rows on the index
of the node that uploaded them. A scalar input is an input that takes one
value. A node that receives a series on a scalar input runs once per row
of the series; we say the node iterates on that index.

Any node placed in a graph is a placement of its definition (see
``GraphNode``). Some nodes have a version that is itself a graph; that
graph is an embedded graph, and below "the placement" means the node
that embeds it. The compiler inlines the embedded graph's nodes in place
of that node, so the graph the author drew (the authored graph) differs
from the graph that runs (the expanded graph). In the authored graph the
embedded graph is one node, ``approve``, and its fields are addressed
through it, ``Ref("approve", "check.amount")``. In the expanded graph
its inner nodes are nodes of the run, named ``approve/check``.

``execution_order`` and ``decisions`` speak of the expanded graph;
``problems`` and ``interface`` speak of the authored one. ``node`` answers
for both graphs: an inner node by its expanded id (``approve/check``) and
the placement by its own (``approve``). A question about a field may use either address:
``field(Ref("approve", "check.amount"))`` and
``field(Ref("approve/check", "amount"))`` are the same field.

It is a plain value: immutable, no I/O, no session. The same graph and
the same registry always give the same ``CompiledGraph``, so it can be
cached, and "compile this and assert what it says" is a complete test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from conductor.codec import to_wire
from conductor.errors import InputNotOffered
from conductor.graph.binding import Binding, Static
from conductor.graph.compiler import Compilation
from conductor.graph.expand import SEPARATOR
from conductor.graph.problem import Problem
from conductor.graph.receive import Iterate
from conductor.interface import model_of
from conductor.node import GraphVersion
from conductor.ref import Ref
from conductor.series import Series

if TYPE_CHECKING:
    from conductor.graph.conditions import Condition
    from conductor.graph.model import Graph, GraphNode
    from conductor.graph.receive import Receive
    from conductor.interface import Interface
    from conductor.node import NodeVersion
    from conductor.registry import NodeRegistry
    from conductor.series import Index


#: What a ``CompiledNode`` or ``CompiledField`` holds where compile has no
#: answer — a node the edge walk did not derive, a placement's missing
#: ``graph_node``, an output's ``binding``. Reading it raises ``KeyError``.
_MISSING: Any = object()


def _written(value: Any, dtype: Any, listed: bool) -> Any:
    """A static as its declared type writes it, for ``CompiledNode.fingerprint``:
    each of many typed-in values (``listed``) through the scalar type, a
    series through its element type, one value through its own."""
    element = getattr(dtype, "element", None)
    if element is not None:
        return [to_wire(item, element) for item in (value.values if isinstance(value, Series) else value)]
    if listed:
        return [to_wire(item, dtype) for item in value]
    return to_wire(value, dtype)


@dataclass(frozen=True, repr=False)
class CompiledGraph:
    """The result of compiling one graph. Ask it; do not read through it.

    Built by ``from_graph`` and read by the engine (``execution_order``,
    then ``node`` and ``field`` for each node it runs), an editor's
    compile endpoint (``problems``, ``interface``, and ``node`` and
    ``field`` for everything it draws), and anything deciding whether a
    run may start (``is_runnable``).

    Three attributes are public, ``graph``, ``interface`` and ``problems``; every other
    attribute is private and read through the methods here and through
    ``node`` and ``field``. A node compile could not resolve — unknown type
    or version — or could not order — on a cycle — has a fatal ``Problem``
    and no interface, version, index or types; asking ``node`` for it
    raises, and an editor paints it from ``problems`` alone.
    """

    #: The registry the graph was compiled against, kept so a compiled
    #: graph can produce another (``with_inputs``).
    _registry: NodeRegistry
    #: Every node compile resolved, by expanded id, and every node whose
    #: version is a graph, by its own: what ``node`` hands back.
    _nodes: Mapping[str, CompiledNode]
    #: What ``execution_order`` answers.
    _order: tuple[str, ...]
    #: What this graph takes and returns, in the same record a node version
    #: declares: ``inputs`` are the unlocked, connectable fields of the nodes
    #: nothing feeds into (each declaration whole, titled as the author
    #: titled it on that node), ``outputs`` every field of the nodes nothing
    #: reads from, both named by address (``"node.field"``); ``returns`` is
    #: ``Mapping``.
    interface: Interface
    #: The graph as the author saved it — with the values ``with_inputs``
    #: filled, on a copy it made — which a host stores as what ran. Not read
    #: back for what compile learned: ask ``node`` and ``field`` for that.
    graph: Graph
    #: Everything wrong with the graph, fatal or not, each on the node and
    #: field it is about. Anchored on the authored graph: a problem found
    #: inside an embedded graph sits on the node the author placed, with the
    #: inner address as the field.
    problems: tuple[Problem, ...]

    @classmethod
    def from_graph(cls, graph: Graph, registry: NodeRegistry) -> CompiledGraph:
        """Compile ``graph`` against ``registry``: the one way to get a ``CompiledGraph``.

        Pure — the same graph and registry always give the same result.
        Every definition the graph names must already be in the registry.
        Nothing raises for a fault in the graph; read ``problems`` and
        ``is_runnable``. The passes are ``compiler.Compilation.run``'s;
        this builds every ``CompiledNode`` and ``CompiledField`` once from
        what they left.
        """
        passes = Compilation(graph, registry)
        passes.run()
        expansion, iteration = passes.expansion, passes.iteration
        versions = {**expansion.versions, **expansion.placement_versions}
        nodes: dict[str, CompiledNode] = {}
        for node_id, interface in passes.interfaces.items():
            placed = expansion.nodes.get(node_id, _MISSING)
            inputs = {inp.name for inp in interface.inputs}
            fields: dict[str, CompiledField] = {}
            # A placement holds no fields of its own: ``field`` reads every
            # address on it through to the inner node that runs.
            declared_fields = (*interface.inputs, *interface.outputs) if placed is not _MISSING else ()
            for declared in declared_fields:
                ref = Ref(node_id, declared.name)
                fields[declared.name] = CompiledField(
                    ref=ref,
                    _type=iteration.types.get(ref, _MISSING),
                    _index=iteration.indexes.get(ref, _MISSING),
                    _binding=(placed.bindings.get(declared.name)
                              if placed is not _MISSING and declared.name in inputs else _MISSING),
                    _receives=iteration.receives.get(ref, _MISSING),
                    _condition=passes.output_conditions.get(ref, _MISSING),
                )
            derived = placed is not _MISSING and node_id in iteration.iterated
            nodes[node_id] = CompiledNode(
                id=node_id,
                version=versions[node_id],
                interface=interface,
                embedded_in=expansion.placement_of[node_id],
                _graph_node=placed,
                _statics=passes.statics.get(node_id, _MISSING),
                _iterates_on=iteration.iterated.get(node_id, _MISSING),
                _call_model=model_of(interface.inputs) if derived else _MISSING,
                _fields=fields,
                _registry=registry,
            )
        return cls(
            _registry=registry,
            _nodes=nodes,
            _order=expansion.order,
            interface=passes.graph_interface,
            graph=graph,
            problems=tuple(passes.problems),
        )

    # -- one node, one field ---------------------------------------------------

    def node(self, node_id: str) -> CompiledNode:
        """Everything the compiler knows about one node, by expanded id —
        or, for a node whose version is a graph, by its own id. Raises for
        an id compile has nothing on: not in the graph, or a node it could
        not resolve or order, whose ``Problem`` in ``problems`` says why."""
        if node_id not in self._nodes:
            raise KeyError(f"no resolved node {node_id!r}: not in the graph, or its problem says why")
        return self._nodes[node_id]

    def field(self, ref: Ref) -> CompiledField:
        """Everything the compiler knows about one input or output, by
        either address: an address on the authored graph reads through to
        the field that runs. Raises for a field the node does not have —
        a programming error, not a state of the graph."""
        at = self.expanded(ref)
        if at.node_id not in self._nodes:
            if at.node_id != ref.node_id:
                raise KeyError(f"{ref.node_id!r} has no field {ref.field!r} on this node")
            raise KeyError(f"no resolved node {at.node_id!r}: not in the graph, or its problem says why")
        fields = self._nodes[at.node_id]._fields
        if at.field not in fields:
            raise KeyError(f"{at.node_id!r} has no field {at.field!r} on this node")
        return fields[at.field]

    # -- the two graphs ------------------------------------------------------

    def expanded(self, ref: Ref) -> Ref:
        """An address on the authored graph, read through to the node that
        runs: ``Ref("emb", "all.result")`` becomes ``Ref("emb/all", "result")``.
        A host reads engine results by the expanded address, since the
        engine knows only the expanded graph."""
        node_id, name = ref.node_id, ref.field
        while "." in name and node_id in self._nodes and isinstance(self._nodes[node_id].version, GraphVersion):
            inner, name = name.split(".", 1)
            node_id = f"{node_id}{SEPARATOR}{inner}"
        return Ref(node_id, name)

    # -- the interface, from each side --------------------------------------------

    def with_inputs(self, **inputs: Any) -> CompiledGraph:
        """This graph with some of its inputs filled, compiled again: what ``run`` takes to run it with those values.

        Each keyword names an input of ``interface.inputs``: by its bare
        field name (``text=...``) when no other input has that name, else
        by its address (``**{"a.text": ...}``); an input inside an embedded
        graph has a dotted field name and is named by address only. A name
        the interface does not offer — unknown, locked, fed by an edge, or
        shared by several inputs — is an ``InputNotOffered`` that lists the names
        it does offer. Each value becomes a ``Static`` on its input in a
        copy of the authored graph, which compiles against the same
        registry, so compile is what reads it: a list on an input for one
        value runs the graph once per item, a value the type cannot read is
        ``invalid_static``. The copy still offers every input: a static is a
        value a caller may answer over, so filling an input again replaces it.

        Not a run, and it validates nothing itself. ``Ledger.inject`` is the
        different act of recording a node's *outputs* from ``cache``.
        """
        offered = [inp.name for inp in self.interface.inputs]
        filled: dict[str, dict[str, Static]] = {}
        for name, value in inputs.items():
            ref = self._offered(name, offered)
            filled.setdefault(ref.node_id, {})[ref.field] = Static(value)
        graph = self.graph.model_copy(update={"nodes": tuple(
            node.model_copy(update={"bindings": {**node.bindings, **filled.get(node.id, {})}})
            for node in self.graph.nodes
        )})
        return CompiledGraph.from_graph(graph, self._registry)

    def _offered(self, name: str, offered: list[Ref]) -> Ref:
        """The input a keyword to ``with_inputs`` names, or an ``InputNotOffered`` listing what is offered."""
        listing = ", ".join(str(ref) for ref in offered) if offered else "none"
        if "." in name:
            ref = Ref(name)
            if ref in offered:
                return ref
            raise InputNotOffered(f"{name!r} is not an input this graph offers; it offers {listing}"
                            if offered else f"{name!r}: this graph takes no inputs")
        matches = [ref for ref in offered if ref.field == name]
        if len(matches) == 1:
            return matches[0]
        if matches:
            both = ", ".join(str(ref) for ref in matches)
            raise InputNotOffered(f"{name!r} is an input of several nodes ({both}); name one by its address")
        raise InputNotOffered(f"{name!r} is not an input this graph offers; it offers {listing}"
                        if offered else f"{name!r}: this graph takes no inputs")

    # -- the run ------------------------------------------------------------------

    def __repr__(self) -> str:
        """One line: the nodes the author placed, whether it runs, how many problems."""
        placed = tuple(node.id for node in self.graph.nodes)
        return f"CompiledGraph(nodes={placed!r}, is_runnable={self.is_runnable}, problems={len(self.problems)})"

    @property
    def execution_order(self) -> tuple[str, ...]:
        """Expanded node ids in an order where every edge's source precedes
        its target. A node in a cycle is not in it; it has a fatal
        ``Problem`` instead."""
        return self._order

    @property
    def decisions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every decision a caller could observe: for each node that runs
        once and declares a ``choice`` group, the group's alternatives in
        field order, keyed by expanded node id. A node that runs per row is
        left out: its decision picks rows and gates nothing downstream."""
        found: dict[str, dict[str, tuple[str, ...]]] = {}
        for node_id in self._order:
            node = self._nodes[node_id]
            if node._iterates_on is not None and node._iterates_on is not _MISSING:
                continue
            groups: dict[str, list[str]] = {}
            for out in node.interface.outputs:
                if out.choice is not None:
                    groups.setdefault(out.choice, []).append(out.name)
            if groups:
                found[node_id] = {choice: tuple(names) for choice, names in groups.items()}
        return found

    # -- what is wrong -------------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run refuses otherwise."""
        return not any(p.fatal for p in self.problems)


@dataclass(frozen=True, eq=False)
class CompiledNode:
    """One node as the compiler left it: what it has, what it holds, how it runs.

    Built once by ``CompiledGraph.from_graph`` and handed back by
    ``CompiledGraph.node(node_id)`` on every call, so it is equal only to
    itself: the same node compiled twice is two values. The engine reads
    ``interface``, ``validate``, ``statics``, ``runner`` and
    ``iterates_on`` for each node it runs, and ``version`` and
    ``graph_node`` for the policy and the type it reports; an editor reads
    ``interface`` and ``iterates_on`` for each node it draws, and filters
    ``CompiledGraph.problems`` by node id for its marks. Its sibling is
    ``CompiledField``, the same for one input or output.

    An attribute a node cannot answer raises: a node the edge walk could
    not derive — its own edges wrong, or a fault upstream of it — has an
    interface but no ``iterates_on`` or ``validate``, and a node whose
    version is a graph has an interface, a version and ``embedded_in`` but
    no ``graph_node`` or ``statics`` — its inner nodes run in its place.
    The ``Problem`` on it in ``CompiledGraph.problems`` says why.
    """

    #: The expanded id — ``"approve/check"`` for an inner node — or, for a
    #: node whose version is a graph, its own.
    id: str
    #: The version this node uses: its ``run``, interface and policy — or,
    #: for a node whose version is a graph, that version's interface and
    #: its graph.
    version: NodeVersion | GraphVersion = field(repr=False)
    #: The inputs and outputs this node actually has, with every type the
    #: edges gave it — not merely what its version declared. The one place
    #: anything asks what a node has: the engine validates a call against
    #: it and an editor draws the fields from it. For a node whose version
    #: is a graph, the interface read off its inner nodes.
    interface: Interface = field(repr=False)
    #: The id of the node whose embedded graph this node belongs to —
    #: ``"approve"`` for ``"approve/check"`` — or ``None`` for a node the
    #: author placed.
    embedded_in: str | None = field(repr=False)
    _graph_node: GraphNode = field(repr=False)
    _statics: Mapping[str, Any] = field(repr=False)
    _iterates_on: Index | None = field(repr=False)
    #: The pydantic model that validates a call against ``interface``,
    #: built once at compile, not once per unit.
    _call_model: type[BaseModel] = field(repr=False)
    #: Every input and output by name: what ``CompiledGraph.field`` hands back.
    _fields: Mapping[str, CompiledField] = field(repr=False)
    #: Where ``runner`` looks the callable up, on each read.
    _registry: NodeRegistry = field(repr=False)

    @property
    def graph_node(self) -> GraphNode:
        """The node as the author stored it: its type, version number, title
        and bindings. Only for a node the engine runs; the placement of an
        embedded graph is not one, its inner nodes are."""
        if self._graph_node is _MISSING:
            raise KeyError(self.id)
        return self._graph_node

    @property
    def statics(self) -> Mapping[str, Any]:
        """The values the author typed into this node, by field.

        Each read through the field's declared type — the value, never its
        JSON form. Only what the author typed: an input left to its
        declared default is absent. Where the author typed many values for
        a scalar input the value is a list of them, and the field's
        ``receives`` is ``Iterate`` on the input's own index.
        """
        if self._statics is _MISSING:
            raise KeyError(self.id)
        return self._statics

    @property
    def runner(self) -> Callable[..., Any]:
        """The callable that runs this node, on a fresh instance per call."""
        node = self.graph_node
        return self._registry.runner_for(node.type, node.version)

    @property
    def fingerprint(self) -> str:
        """A hash of how the graph places this node: its type, version and
        bindings. A run's state stores one per node, and a leg restored into
        a graph whose fingerprint differs runs the node again — a static
        edited, a version bumped, an edge moved all change it; a title or a
        position does not."""
        node = self.graph_node
        declared = {inp.name: inp.dtype for inp in self.interface.inputs}
        bindings: dict[str, Any] = {}
        for name, binding in node.bindings.items():
            if isinstance(binding, Static) and name in self.statics:
                # The value as its type writes it, never as the author spelled
                # it: ``2`` and ``2.0`` on a number are one value, and a graph
                # built in Python with the typed value hashes like the stored
                # graph with its JSON. The author typed many values exactly
                # when the input is received one per row of its own index.
                # A static for an input the node no longer has is hashed as
                # spelled, since no type reads it.
                listed = isinstance(self._fields[name].receives, Iterate)
                bindings[name] = {"static": _written(self.statics[name], declared[name], listed)}
            else:
                bindings[name] = binding.model_dump()
        placed = {"type": node.type, "version": node.version, "bindings": bindings}
        return hashlib.sha256(json.dumps(to_jsonable_python(placed), sort_keys=True).encode("utf-8")).hexdigest()

    def validate(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """The keyword arguments a call of this node runs with: ``inputs``
        checked against the interface, one value per input, a default
        filled in where the call gave none. Raises pydantic's
        ``ValidationError`` when a value is not its input's type or a
        required input is missing; the engine turns that into the node's
        failure. The check runs through a model built once when the graph
        was compiled, not once per unit. Only for a node the walk over the
        edges derived — a node with a fault upstream has none, and asking
        raises."""
        if self._call_model is _MISSING:
            raise KeyError(self.id)
        validated = self._call_model(**inputs)
        return {info.alias or field: getattr(validated, field) for field, info in type(validated).model_fields.items()}

    @property
    def iterates_on(self) -> Index | None:
        """The index this node runs once per row of, or ``None`` when it runs
        once. Read off its edges at compile — a series arriving on a scalar
        input is what makes a node run per row — and never written on the
        ``GraphNode``. For a
        node whose version is a graph, the index its inner nodes run per row
        of, where a series entered it."""
        if self._iterates_on is _MISSING:
            raise KeyError(self.id)
        return self._iterates_on


@dataclass(frozen=True, eq=False)
class CompiledField:
    """One input or output as the compiler left it: its type, its rows, where its value comes from, how it is received.

    Built once by ``CompiledGraph.from_graph``, held by its node, and
    handed back by ``CompiledGraph.field(ref)`` on every call; like its
    node, it is equal only to itself. The engine
    reads ``index``, ``binding`` and ``receives`` to lay values out per
    row, to find them and to hand each unit what it takes; an editor reads
    ``type`` and ``index`` to draw the edge and ``receives`` to label it,
    and filters ``CompiledGraph.problems`` by node and field for its marks.
    Its sibling is ``CompiledNode``, the same for one node.

    An attribute a field cannot answer raises: an input has no
    ``condition`` and an output has no ``binding`` or ``receives``; and no
    field of a node the edge walk could not derive has a ``type``,
    ``index``, ``receives`` or ``condition``, since nothing downstream of
    a fault is guessed at. The ``Problem`` on it says why.
    """

    #: The expanded address of the field — ``Ref("approve/check", "amount")``
    #: however it was asked for.
    ref: Ref
    _type: Any = field(repr=False)
    _index: Index | None = field(repr=False)
    _binding: Binding | None = field(repr=False)
    _receives: Receive = field(repr=False)
    _condition: Condition = field(repr=False)

    @property
    def type(self) -> Any:
        """The type that travels on this field: what an edge from it or into
        it carries. An output of a node that runs once per row carries
        ``Series[X]`` even where its declaration says ``X``; an input fed a
        series carries that series, before the engine slices it per row."""
        if self._type is _MISSING:
            raise KeyError(self.ref)
        return self._type

    @property
    def index(self) -> Index | None:
        """Where the rows of the series on this field come from, or ``None``
        for a field that carries one value. For an input fed several series
        gathered together, the fresh index they were gathered onto. Read
        by the engine to lay values out per row."""
        if self._index is _MISSING:
            raise KeyError(self.ref)
        return self._index

    @property
    def binding(self) -> Binding | None:
        """Where this input's value comes from: ``From`` from other nodes'
        outputs, ``Static`` for a value the author typed, or ``None`` when
        nothing binds it and its declared default applies. Only an input
        has one; asking on an output raises."""
        if self._binding is _MISSING:
            raise KeyError(f"{self.ref.node_id!r} has no input {self.ref.field!r} on this node")
        return self._binding

    @property
    def receives(self) -> Receive:
        """How a unit of this input's node receives the value: the value at
        its own row (``Iterate``), the one value there is (``Broadcast``),
        everything on the field (``Whole``), the rows under its row
        (``Group``) or its unrelated sources collected (``Gather``) — see
        ``conductor.graph.receive``. Decided once by the
        walk over the edges; the engine's ledger reads it to tell when a
        unit is ready and what to hand it, and an editor may label the
        edge from it. Only an input has one; asking on an output raises."""
        if self._receives is _MISSING:
            raise KeyError(self.ref)
        return self._receives

    @property
    def condition(self) -> Condition:
        """Under which condition this output appears: a boolean formula over
        the decisions upstream (see ``conductor.graph.conditions``),
        ``ALWAYS`` when nothing gates it. Derived from ``choice`` groups
        and edges; the engine never reads it. Only an output has one."""
        if self._condition is _MISSING:
            raise KeyError(self.ref)
        return self._condition
