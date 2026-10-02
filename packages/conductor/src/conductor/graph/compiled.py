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

Some nodes have a version that is itself a graph, which the host compiled
on its own; that graph is an embedded graph, and the node is a ``graph``
(its ``kind``). Compile places the embedded graph's nodes in place of that
node (``conductor.graph.embedding``), so the graph the author drew (the
authored graph) differs from the graph that runs (the expanded graph). In
the authored graph the embedded graph is one node, ``approve``, and its
fields are addressed through it, ``Ref("approve", "check.amount")``. In the
expanded graph its inner nodes are nodes of the run, named ``approve/check``.
Only the fields the embedded graph offers can be addressed from outside.

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
from functools import cached_property
from typing import TYPE_CHECKING, Any

from conductor.errors import InputNotOffered
from conductor.graph.binding import Static
from conductor.graph.compiled_node import CompiledField, CompiledNode, _gate
from conductor.graph.compiler import Compilation
from conductor.graph.embedding import as_drawn
from conductor.graph.problem import Problem
from conductor.ref import Ref

if TYPE_CHECKING:
    from conductor.graph.model import Graph
    from conductor.interface import Interface
    from conductor.registry import NodeRegistry


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
    #: field it is about, all on nodes of the authored graph. An embedded
    #: graph that cannot be placed is one ``embedded_graph_broken`` on the
    #: node that places it; its own problems are on it (``node(id).version.problems``).
    problems: tuple[Problem, ...]
    #: The finished compile, whose last step builds ``node`` and ``field``'s
    #: values on the first question. A graph that places this
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
        every ``CompiledNode`` and ``CompiledField`` is built once from what
        they recorded, on the first question about a node (``_nodes``).
        """
        compilation = Compilation(graph, registry)
        compilation.run()
        return cls(
            _registry=registry,
            _order=compilation.expansion.order,
            interface=compilation.graph_interface,
            graph=graph,
            problems=tuple(compilation.problems),
            _compilation=compilation,
        )

    # -- one node, one field ---------------------------------------------------

    @cached_property
    def _nodes(self) -> Mapping[str, CompiledNode]:
        """Every node compile met, by expanded id, and every node whose version
        is a graph, by its own: what ``node`` hands back. Folded from the
        compile's last step on first read (``Compilation.compiled_nodes``),
        since a graph compiled only to be embedded in another is read through
        its compile and never asked about a node."""
        return self._compilation.compiled_nodes()

    @cached_property
    def _index_owners(self) -> Mapping[str, CompiledNode | CompiledField]:
        """Who owns each index's share of the read plan, by index id: a node,
        for the index named after it, or the input whose typed-in list
        creates one. The ledger's way back from an index to its owner, read
        on its first question; compile names the index, so nobody parses
        the id to find which of the two it is."""
        owners: dict[str, CompiledNode | CompiledField] = {}
        for node_id, node in self._nodes.items():
            if node._kind == "graph":
                continue
            owners[node_id] = node
            for compiled_field in node._fields.values():
                if compiled_field._listed:
                    owners[compiled_field.index.id] = compiled_field
        return owners

    def node(self, node_id: str) -> CompiledNode:
        """Look up one compiled node by its expanded id.

        Every node compile saw has an entry, broken or not: each node in the
        graph and each node inside an embedded graph placed in it. An id the
        graph does not have raises ``KeyError``."""
        if node_id not in self._nodes:
            raise KeyError(f"{node_id!r} is not a node of this graph")
        return self._nodes[node_id]

    def field(self, ref: Ref) -> CompiledField:
        """Look up one compiled input or output by its address.

        A field inside an embedded graph is at its path: ``Ref("emb/all",
        "result")``. The node ``emb`` itself is a graph and has no fields, so
        the author's spelling, ``Ref("emb", "all.result")``, raises ``KeyError``
        like any node or field the graph does not have, since asking for one
        is a programming error. A node compile could not resolve raises
        ``NodeResolutionError``: nobody knows its fields."""
        node = self._nodes.get(ref.node_id)
        if node is None:
            raise KeyError(f"{ref.node_id!r} is not a node of this graph")
        if node._kind == "graph":
            raise KeyError(f"{ref.node_id!r} is a graph; its fields are on the nodes inside it")
        _gate(node, f"field {ref.field!r}", needs_wiring=False)
        compiled_field = node._fields.get(ref.field)
        if compiled_field is None:
            raise KeyError(f"{ref.node_id!r} has no field {ref.field!r} on this node")
        return compiled_field

    # -- the interface, from each side --------------------------------------------

    def with_inputs(self, **inputs: Any) -> CompiledGraph:
        """This graph with some of its inputs filled, compiled again: what ``run`` takes to run it with those values.

        Each keyword names an input of ``interface.inputs``: by its address
        (``**{"a.text": ...}``, ``**{"emb/up.text": ...}`` inside an embedded
        graph), or by its bare field name (``text=...``) when no other input
        has that name. A name
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
        return self._refilled(inputs, ())

    def _refilled(self, values: Mapping[str, Any], cleared: Iterable[str]) -> CompiledGraph:
        """This graph compiled again with ``values`` typed on the inputs they
        name and the values its author typed on the ``cleared`` inputs taken
        off, so they arrive at run time. ``with_inputs`` for a caller; a
        graph that places this one asks for the copy one placement fills
        (``Compilation._as_placed``). Names are read as ``with_inputs`` reads them."""
        offered = [inp.name for inp in self.interface.inputs]
        filled: dict[str, dict[str, Static]] = {}
        for name, value in values.items():
            node_id, key = as_drawn(self._offered(name, offered), self._compilation.expansion.graphs)
            filled.setdefault(node_id, {})[key] = Static(value)
        taken: dict[str, set[str]] = {}
        for name in cleared:
            node_id, key = as_drawn(self._offered(name, offered), self._compilation.expansion.graphs)
            taken.setdefault(node_id, set()).add(key)
        graph = self.graph.model_copy(update={"nodes": tuple(
            node.model_copy(update={"bindings": {
                **{field: binding for field, binding in node.bindings.items() if field not in taken.get(node.id, ())},
                **filled.get(node.id, {}),
            }})
            for node in self.graph.nodes
        )})
        return CompiledGraph.from_graph(graph, self._registry)

    def _offered(self, name: str, offered: list[Ref]) -> Ref:
        """The input a keyword to ``with_inputs`` names, or an ``InputNotOffered`` listing what is offered."""
        listing = ", ".join(str(ref) for ref in offered) if offered else "none"
        matches = [ref for ref in offered if ref == name] or [ref for ref in offered if ref.field == name]
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

    # -- what is wrong -------------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run refuses otherwise."""
        return not any(p.fatal for p in self.problems)
