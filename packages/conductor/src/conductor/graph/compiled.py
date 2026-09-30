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
``CompiledNode``, everything the compiler knows about one node, whether it
runs as one unit or its version is a graph (its ``kind``), and each says
its ``state``: how far compile got with it. ``field(ref)`` returns a
``CompiledField``, everything it knows about one input or output; both
live in ``conductor.graph.compiled_node``.
Compile builds each once, holding its own answers, and these two
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

Some nodes have a version that is itself a graph; that graph is an
embedded graph, and the node is a ``graph`` (its ``kind``). The compiler
inlines the embedded graph's nodes in place of that node, so the graph the author drew (the authored graph) differs
from the graph that runs (the expanded graph). In the authored graph the
embedded graph is one node, ``approve``, and its fields are addressed
through it, ``Ref("approve", "check.amount")``. In the expanded graph
its inner nodes are nodes of the run, named ``approve/check``.

``execution_order`` and ``decisions`` speak of the expanded graph;
``problems`` and ``interface`` speak of the authored one. ``node`` answers
for both graphs: an inner node by its expanded id (``approve/check``) and
the node that embeds it by its own (``approve``). A question about a field may use either address:
``field(Ref("approve", "check.amount"))`` and
``field(Ref("approve/check", "amount"))`` are the same field.

It is a plain value: immutable, no I/O, no session. The same graph and
the same registry always give the same ``CompiledGraph``, so it can be
cached, and "compile this and assert what it says" is a complete test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from conductor.errors import InputNotOffered
from conductor.graph.binding import From, Static
from conductor.graph.compiled_node import CompiledField, CompiledNode, NodeState, _gate
from conductor.graph.compiler import Compilation
from conductor.graph.expand import SEPARATOR, authored_address, embedded_in, expanded_ref
from conductor.graph.problem import Problem
from conductor.ref import Ref

if TYPE_CHECKING:
    from conductor.graph.model import Graph, GraphNode
    from conductor.graph.receive import Receive
    from conductor.interface import Interface
    from conductor.registry import NodeRegistry
    from conductor.series import Index


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
    ``node`` and ``field``. ``node`` answers for every node compile met, in
    one of three states (``NodeState``), and raises ``KeyError`` only for
    an id the graph does not have; an editor asks each node its ``state``
    and paints what that state has.
    """

    #: The registry the graph was compiled against, kept so a compiled
    #: graph can produce another (``with_inputs``).
    _registry: NodeRegistry
    #: Every node compile met, by expanded id, and every node whose version
    #: is a graph, by its own: what ``node`` hands back.
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
    #: The finished compile this was folded from. A graph that places this
    #: one reads its records from here and lifts them (``embedding``);
    #: nothing else does.
    _compilation: Compilation = field(repr=False)

    @classmethod
    def from_graph(cls, graph: Graph, registry: NodeRegistry) -> CompiledGraph:
        """Compile ``graph`` against ``registry``: the one way to get a ``CompiledGraph``.

        Pure — the same graph and registry always give the same result.
        Every definition the graph names must already be in the registry.
        Nothing raises for a fault in the graph; read ``problems`` and
        ``is_runnable``. The steps of compile are ``compiler.Compilation.run``'s;
        this builds every ``CompiledNode`` and ``CompiledField`` once from
        what they recorded (``_Fold``).
        """
        compilation = Compilation(graph, registry)
        compilation.run()
        return cls(
            _registry=registry,
            _nodes=_Fold(compilation).nodes(),
            _order=compilation.expansion.order,
            interface=compilation.graph_interface,
            graph=graph,
            problems=tuple(compilation.problems),
            _compilation=compilation,
        )

    # -- one node, one field ---------------------------------------------------

    def node(self, node_id: str) -> CompiledNode:
        """Look up one compiled node by its expanded id.

        Every node compile saw has an entry, broken or not: each node in the
        graph and each node inside an embedded graph. An id the graph does
        not have raises ``KeyError``."""
        if node_id not in self._nodes:
            raise KeyError(f"{node_id!r} is not a node of this graph")
        return self._nodes[node_id]

    def field(self, ref: Ref) -> CompiledField:
        """Look up one compiled input or output by its address.

        The address can be written against the graph as the author built it
        (``Ref("emb", "all.result")``) or as it runs (``Ref("emb/all",
        "result")``). A node or field the graph does not have raises
        ``KeyError``, since asking for one is a programming error. A node
        compile could not resolve raises ``NodeResolutionError``: nobody
        knows its fields."""
        node_id, name, node = self._reached(ref)
        if node is None:
            if node_id != ref.node_id:
                raise KeyError(f"{ref.node_id!r} has no field {ref.field!r} on this node")
            raise KeyError(f"{node_id!r} is not a node of this graph")
        if node._kind == "graph":
            raise KeyError(f"{node_id!r} has no field {name!r} on this node")
        _gate(node, f"field {name!r}", needs_wiring=False)
        compiled_field = node._fields.get(name)
        if compiled_field is None:
            raise KeyError(f"{node_id!r} has no field {name!r} on this node")
        return compiled_field

    # -- the two graphs ------------------------------------------------------

    def expanded(self, ref: Ref) -> Ref:
        """An address on the authored graph, read through to the node that
        runs: ``Ref("emb", "all.result")`` becomes ``Ref("emb/all", "result")``.
        A host reads engine results by the expanded address, since the
        engine knows only the expanded graph."""
        node_id, name, _ = self._reached(ref)
        return Ref(node_id, name)

    def _reached(self, ref: Ref) -> tuple[str, str, CompiledNode | None]:
        """Where ``ref`` lands once read through every graph it names:
        the expanded node id, the field name there, and the node (``None``
        for an id the graph does not have). Splits the address once, since
        the engine asks ``field`` for every value it reads or writes."""
        node_id, _, name = ref.partition(".")
        node = self._nodes.get(node_id)
        while "." in name and node is not None and node._kind == "graph":
            inner, name = name.split(".", 1)
            node_id = f"{node_id}{SEPARATOR}{inner}"
            node = self._nodes.get(node_id)
        return node_id, name, node

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
        left out: its decision picks rows and gates nothing downstream. So
        is a node that is not ready: whether it runs per row is not
        known."""
        found: dict[str, dict[str, tuple[str, ...]]] = {}
        for node_id in self._order:
            node = self._nodes[node_id]
            if node.state != "ready" or node.iterates_on is not None:
                continue
            groups: dict[str, list[str]] = {}
            for out in node.interface.outputs:
                if out.choice is not None:
                    groups.setdefault(out.choice, []).append(out.name)
            if groups:
                found[node_id] = {choice: tuple(names) for choice, names in groups.items()}
        return found

    def _downstream(self, node_ids: Iterable[str]) -> frozenset[str]:
        """Every node that reads an output of one of ``node_ids``, directly or through other nodes, by expanded id.

        A graph question a run asks: a restore leaves out, with each node
        that changed, everything its values reached, so those units run
        again. A node of ``node_ids`` is in the answer only when another of
        them reads it. A walk over ``CompiledField._read_by`` on each call,
        not a stored closure: storing every node's readers costs memory
        quadratic in a long chain.
        """
        found: set[str] = set()
        frontier = list(node_ids)
        while frontier:
            node = self._nodes[frontier.pop()]
            for out in node.interface.outputs:
                for reader, _ in self.field(Ref(node.id, out.name))._read_by:
                    if reader.node_id not in found:
                        found.add(reader.node_id)
                        frontier.append(reader.node_id)
        return frozenset(found)

    # -- what is wrong -------------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run refuses otherwise."""
        return not any(p.fatal for p in self.problems)


class _Fold:
    """What the steps of compile recorded, read into one stored value per node compile met.

    Used once, by ``CompiledGraph.from_graph``, and then dropped. The ids
    it answers for are every id the author wrote (the first of two that
    share one), every expanded id, and every inner id of every embedded
    graph compile entered, resolved or not. Each gets a ``NodeState`` read
    off what the steps recorded — derived by the walk, or with an
    interface, or neither — and every one that is not ready gets the fatal
    ``Problem`` that explains it, found in this order: a fatal problem on
    the node or anything inside it; for a ``graph``, the cause of its
    first inner node that is not ready; the cause of its first source that
    is not ready; a fatal problem on an embedded graph around it, the
    innermost first. Finding none is a bug in compile, and raises.
    """

    def __init__(self, compilation: Compilation) -> None:
        self.compilation = compilation
        self.expansion = compilation.expansion
        self.iteration = compilation.iteration
        #: The graph nodes compile entered, with the version each uses: which
        #: nodes get ``kind`` ``graph``.
        self.graphs = self.expansion.graphs
        #: Every id compile met: the node as stored and the graph node it
        #: sits in — ``None`` for every id the author wrote, a refused one
        #: holding a ``/`` included; read off the id for every other.
        self.met: dict[str, tuple[GraphNode, str | None]] = {}
        for node in compilation.graph.nodes:
            self.met.setdefault(node.id, (node, None))
        # An inner node answers for its expanded id even where the author
        # wrote that id too: the authored one is refused (``invalid_node_id``).
        inner_nodes: dict[str, tuple[GraphNode, str | None]] = {}
        for outer, version in self.graphs.items():
            for inner in (version.graph.nodes if isinstance(version, CompiledGraph) else version.graph):
                inner_id = f"{outer}{SEPARATOR}{inner.id}"
                inner_nodes.setdefault(inner_id, (inner.model_copy(update={"id": inner_id}), outer))
        self.met.update(inner_nodes)
        for node_id, node in self.expansion.nodes.items():
            self.met[node_id] = (node, embedded_in(node_id))
        for outer in self.graphs:
            self.met[outer] = (self.met[outer][0], embedded_in(outer))
        #: The graph nodes with a node inside them, at any depth, that the
        #: walk did not derive: each reads ``wiring_failed``, never ``ready``.
        self.broken_inside: set[str] = set()
        for node_id, (_, outer) in self.met.items():
            if node_id in self.iteration.iterated:
                continue
            while outer is not None:
                self.broken_inside.add(outer)
                outer = self.met[outer][1]
        self.causes: dict[str, Problem] = {}
        self._plan()

    def _plan(self) -> None:
        """The read plan: the walk's decisions inverted once, over the nodes it
        derived in execution order. Who reads each output (``read_by``);
        per index, who runs once per row of it (``iterated_by``), which
        outputs sit on it (``carried_by``) and which typed-in lists are born
        under it (``typed_lists``). Each index's entries go to what births
        its rows: a node's series outputs, or an input's typed-in list. An
        index with entries and no such owner is a bug in compile, and
        raises."""
        iteration, listed = self.iteration, self.compilation.listed
        self.read_by: dict[Ref, list[tuple[Ref, Receive]]] = {}
        self.reads: dict[str, list[tuple[Ref, Receive]]] = {}
        self.iterated_by: dict[str, list[str]] = {}
        self.carried_by: dict[str, list[Ref]] = {}
        self.typed_lists: dict[str, list[Ref]] = {}
        #: Each index's owner: the node whose series outputs birth it, or the listed input.
        self.births: dict[str, Index] = {}
        self.listed: dict[Ref, Index] = {}
        for node_id in self.expansion.order:
            if node_id not in iteration.iterated:
                continue
            node, interface = self.expansion.nodes[node_id], self.compilation.interfaces[node_id]
            for inp in interface.inputs:
                ref = Ref(node_id, inp.name)
                binding = node.bindings.get(inp.name)
                if isinstance(binding, From):
                    for source in binding.refs:
                        self.read_by.setdefault(source, []).append((ref, iteration.receives[ref]))
                        self.reads.setdefault(node_id, []).append((source, iteration.receives[ref]))
                if inp.name in listed[node_id]:
                    own = iteration.indexes[ref]
                    self.listed[ref] = own
                    if own.parent is not None:
                        self.typed_lists.setdefault(own.parent.id, []).append(ref)
            if (index := iteration.iterated[node_id]) is not None:
                self.iterated_by.setdefault(index.id, []).append(node_id)
            for out in interface.outputs:
                ref = Ref(node_id, out.name)
                if (index := iteration.indexes[ref]) is not None:
                    self.carried_by.setdefault(index.id, []).append(ref)
                    if out.dtype.element is not None:
                        self.births.setdefault(node_id, index)
        owned = {index.id for index in (*self.births.values(), *self.listed.values())}
        for planned in (self.iterated_by, self.carried_by, self.typed_lists):
            for index_id in planned.keys() - owned:
                raise RuntimeError(f"compile planned index {index_id!r}, which no node or typed-in list owns")

    def _share(self, index: Index | None) -> dict[str, Any]:
        """What the read plan says about ``index``, for the value that owns it."""
        key = None if index is None else index.id
        return {
            "_iterated_by": tuple(self.iterated_by.get(key, ())),
            "_carried_by": tuple(self.carried_by.get(key, ())),
            "_typed_lists": tuple(self.typed_lists.get(key, ())),
        }

    def nodes(self) -> dict[str, CompiledNode]:
        built: dict[str, CompiledNode] = {}
        for node_id, (placed, outer) in self.met.items():
            state = self.state(node_id)
            cause = None if state == "ready" else self.cause(node_id, frozenset())
            problems = self.problems_on(self.address(node_id))
            interface = self.compilation.interfaces.get(node_id)
            graph = self.graphs.get(node_id)
            if graph is not None:
                built[node_id] = CompiledNode(
                    id=node_id,
                    state=state,
                    graph_node=placed,
                    embedded_in=outer,
                    problems=problems,
                    _cause=cause,
                    _kind="graph",
                    _version=graph,
                    _interface=interface,
                    _statics=(
                        self.compilation.statics.get(node_id, {}) if isinstance(graph, CompiledGraph)
                        else self.typed_on(node_id, placed)
                    ),
                    _definition=self.expansion.definitions[node_id],
                    _iterates_on=self.iteration.iterated.get(node_id),
                    _births=None,
                    _reads=(),
                    **self._share(None),
                    _fields={},
                )
                continue
            built[node_id] = CompiledNode(
                id=node_id,
                state=state,
                graph_node=placed,
                embedded_in=outer,
                problems=problems,
                _cause=cause,
                _kind=None if interface is None else "node",
                _definition=self.expansion.definitions.get(node_id),
                _version=self.expansion.versions.get(node_id),
                _interface=interface,
                _statics=self.compilation.statics.get(node_id),
                _iterates_on=self.iteration.iterated.get(node_id),
                _births=self.births.get(node_id),
                _reads=tuple(self.reads.get(node_id, ())),
                **self._share(self.births.get(node_id)),
                _fields={} if interface is None else self.fields(node_id, placed, interface, cause, problems),
            )
        return built

    def typed_on(self, node_id: str, placed: GraphNode) -> dict[str, Any]:
        """The values the author typed on a node whose version is a graph, by
        inner address: each read where it landed, on the inner field its
        address names, through that field's type. A value on a field no
        inner node has, or on an inner node compile could not resolve, has
        no reading and is left out; its problem says why."""
        graphs = self.graphs.keys()
        typed: dict[str, Any] = {}
        for name, binding in placed.bindings.items():
            if not isinstance(binding, Static):
                continue
            inner_ref = expanded_ref(Ref(node_id, name), graphs)
            held = self.compilation.statics.get(inner_ref.node_id, {})
            if inner_ref.field in held:
                typed[name] = held[inner_ref.field]
        return typed

    def fields(
        self, node_id: str, placed: GraphNode, interface: Interface, cause: Problem | None,
        problems: tuple[Problem, ...],
    ) -> dict[str, CompiledField]:
        """One ``CompiledField`` per name the node has. A name that is both
        an input and an output is one field: its ``type`` is the output's,
        its ``binding`` and ``receives`` the input's."""
        inputs = {inp.name for inp in interface.inputs}
        outputs = {out.name for out in interface.outputs}
        ready = cause is None
        iteration, conditions = self.iteration, self.compilation.output_conditions
        fields: dict[str, CompiledField] = {}
        for name in (declared.name for declared in (*interface.inputs, *interface.outputs)):
            if name in fields:
                continue
            ref = Ref(node_id, name)
            address = f"{self.address(node_id)}.{name}"
            fields[name] = CompiledField(
                ref=ref,
                problems=tuple(p for p in problems if _address(p) == address),
                _cause=cause,
                _input=name in inputs,
                _output=name in outputs,
                _binding=placed.bindings.get(name) if name in inputs else None,
                _type=iteration.types[ref] if ready else None,
                _index=iteration.indexes[ref] if ready else None,
                _receives=iteration.receives[ref] if ready and name in inputs else None,
                _condition=conditions[ref] if ready and name in outputs else None,
                _read_by=tuple(self.read_by.get(ref, ())),
                _listed=name in self.compilation.listed.get(node_id, ()),
                **self._share(self.listed.get(ref)),
            )
        return fields

    def state(self, node_id: str) -> NodeState:
        """Membership, read once here: derived by the walk, or given an
        interface, or neither. A ``graph`` is ready only when every node
        inside it is."""
        if node_id in self.iteration.iterated and node_id not in self.broken_inside:
            return "ready"
        if node_id in self.compilation.interfaces:
            return "wiring_failed"
        return "resolution_failed"

    def address(self, node_id: str) -> str:
        """Where the author sees a node, and so where its problems sit:
        its own id at the top level, ``approve.check`` for ``approve/check``."""
        return node_id if self.met[node_id][1] is None else authored_address(node_id)

    def problems_on(self, address: str) -> tuple[Problem, ...]:
        """Every problem at ``address`` or inside it."""
        return tuple(
            p for p in self.compilation.problems
            if _address(p) == address or _address(p).startswith(address + ".")
        )

    def cause(self, node_id: str, visiting: frozenset[str]) -> Problem:
        """The fatal problem that explains why ``node_id`` is not ready (see the class docstring)."""
        if node_id in self.causes:
            return self.causes[node_id]
        visiting = visiting | {node_id}
        found = next((p for p in self.problems_on(self.address(node_id)) if p.fatal), None)
        placed, outer = self.met[node_id]
        graph = node_id in self.graphs
        if found is None and graph:
            inner = self.expansion.members.get(node_id, ())
            found = self._first_cause(inner, visiting)
        if found is None and not graph:
            sources = [
                ref.node_id for binding in placed.bindings.values() if isinstance(binding, From)
                for ref in binding.refs
            ]
            found = self._first_cause(sources, visiting)
        while found is None and outer is not None:
            found = next((p for p in self.compilation.problems if p.fatal and _address(p) == self.address(outer)), None)
            outer = self.met[outer][1]
        if found is None:
            raise AssertionError(f"compile left {node_id!r} {self.state(node_id)} with nothing fatal to say why")
        self.causes[node_id] = found
        return found

    def _first_cause(self, ids: Iterable[str], visiting: frozenset[str]) -> Problem | None:
        """The cause of the first of ``ids`` that compile met and that is not ready."""
        for other in ids:
            if other in self.met and other not in visiting and self.state(other) != "ready":
                return self.cause(other, visiting)
        return None


def _address(found: Problem) -> str:
    """Where a problem sits: its node, and its field when it has one (``approve.check.amount``)."""
    return found.node_id if found.field is None else f"{found.node_id}.{found.field}"
