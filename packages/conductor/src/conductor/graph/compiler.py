"""How a graph is compiled: a fixed list of steps, one after another, over one graph.

``CompiledGraph.from_graph(graph, registry)`` is the door; it runs a
``Compilation`` and builds the ``CompiledGraph`` from what the steps
recorded. Pure: no session, no I/O, no loading. Every definition the graph
names must already be in the registry; a host that had to load one built
it and called ``NodeRegistry.extended_with`` first.

Everything wrong with the graph comes back as a ``Problem`` rather than
raising, so an editor can show a half-finished graph with its faults
marked.

The steps each read the one before, through the state ``Compilation``
holds. A node that fails a step carries a fatal ``Problem`` and drops out
of the steps after it, so nothing is guessed about a node downstream of
a fault. This module knows nothing of ``CompiledGraph``: the result is
built on the other side, in ``conductor.graph.compiled``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError

from conductor.errors import Refuses
from conductor.graph.binding import From, Static, static_values
from conductor.graph.conditions import Condition, conditions_of
from conductor.graph.embedding import (
    SEPARATOR,
    Crossing,
    expanded_ref,
    graph_as_placed,
    inside,
    received_renamed,
    renamed,
)
from conductor.graph.iteration import Iteration, derive
from conductor.graph.model import Graph, GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.topology import dependencies_of, order_of
from conductor.graph.views import derive_interface, field_problems, lock_problems
from conductor.interface import Interface
from conductor.metadata import Input
from conductor.node import NodeDefinition, NodeVersion
from conductor.ref import Ref
from conductor.registry import NodeRegistry
from conductor.series import Index, Series

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph


@dataclass(frozen=True)
class Expansion:
    """The graph that runs: every node compile works with, by id, and the order the engine runs them in.

    ``Compilation.order`` lays it out from the nodes the author placed, a
    node whose version is a compiled graph among them as one node, and
    ``Compilation.place_graphs`` puts each such graph's nodes in its place
    in the order (``approve/check``). A placed graph node stays in
    ``nodes``, a node the engine doesn't run. ``versions`` and
    ``definitions`` hold what each node resolved to (``resolved``).
    ``CompiledGraph``'s fold reads it all; which graph a node sits in is not
    stored, since its id says (``embedding.embedded_in``).
    """

    nodes: dict[str, GraphNode]
    order: tuple[str, ...]
    #: The version each node uses: a ``NodeVersion``, or a compiled graph.
    versions: dict[str, Any]
    #: The class each node resolved to: what its hooks and its runner are made from.
    definitions: dict[str, type[NodeDefinition]]
    #: Every node whose version is a compiled graph, placed or not, nested
    #: ones included: the one answer to "is this a graph node?", read by
    #: every step after ``order`` and by ``CompiledGraph``'s fold.
    graphs: frozenset[str]


def resolved(node: GraphNode, registry: NodeRegistry) -> tuple[type[NodeDefinition], Any] | Problem:
    """The class a placed node names and the version it pins, or the problem saying which is missing.

    Compile's one lookup of a node (``Compilation.pin``). A stored graph can
    name a type the registry has since lost or a version the class has
    since dropped; either is a fatal problem on the node. Everything after
    compile reads the class and version off the compiled node.
    """
    if node.type not in registry:
        return problem("unknown_node_type", node.id, node_type=node.type)
    definition = registry[node.type]
    version = definition.versions.get(node.version)
    if version is None:
        return problem("unknown_node_version", node.id, node_type=node.type, version=node.version)
    return definition, version


class Compilation:
    """One graph being compiled: the state every step reads and writes.

    ``run`` runs the steps in order. Each step is a method that reads what
    the steps before it left on ``self`` and records what it finds wrong in
    ``problems``.
    ``CompiledGraph.from_graph`` makes one, runs it and keeps it: the stored
    ``CompiledNode`` and ``CompiledField`` values are read from it on first
    question, and a graph that places that one reads its records from it.
    """

    def __init__(self, graph: Graph, registry: NodeRegistry) -> None:
        self.graph = graph
        self.registry = registry
        self.problems: list[Problem] = []
        #: Every id the author wrote, refused ones included: an edge from one
        #: of these reads a node the author wrote, not a missing one.
        self.graph_ids: frozenset[str] = frozenset(node.id for node in graph.nodes)
        #: The nodes the author placed, by id (set by ``place``).
        self.authored: dict[str, GraphNode] = {}
        #: The class each authored node resolved to and the version it uses (set by ``pin``).
        self.pinned: dict[str, tuple[type[NodeDefinition], Any]] = {}
        #: Who waits for whom, and an execution order, over the authored graph (set by ``order``).
        self.authored_dependencies: dict[str, frozenset[str]] = {}
        self.authored_order: tuple[str, ...] = ()
        #: The graph that runs (laid out by ``order``, filled in by ``place_graphs``).
        self.expansion: Expansion
        #: Each node's inputs and outputs, and the values the author typed, by
        #: expanded id (set by ``ask_inputs``; ``walk_edges`` completes them).
        self.interfaces: dict[str, Interface] = {}
        self.statics: dict[str, dict[str, Any]] = {}
        #: Per node, the scalar inputs where the author typed many values —
        #: three files, a list of texts — which the node runs once per value of.
        self.listed: dict[str, frozenset[str]] = {}
        #: Nodes whose version is a compiled graph that compile could not fully
        #: work out, so it cannot be placed (set by ``pin``).
        self.unplaceable: set[str] = set()
        #: Nodes whose stored edges are wrong; the edge walk leaves them out (set by ``check_bindings``).
        self.broken: frozenset[str] = frozenset()
        #: What the walk over the edges decided (set by ``walk_edges``).
        self.iteration: Iteration
        #: The condition under which each output appears (set by ``conditions``).
        self.output_conditions: dict[Ref, Condition] = {}
        #: What the graph takes and returns (set by ``interface``).
        self.graph_interface: Interface

    def run(self) -> None:
        """The steps, in order. Each reads the one before:

        1. ``place`` — the nodes by id; two with one id is a problem.
        2. ``pin`` — the version each node uses; an unknown type or version
           is a problem on the node.
        3. ``order`` — who waits for whom and an execution order; a cycle is
           a problem. A node whose version is a compiled graph is one node here.
        4. ``ask_inputs`` — each node's inputs as its hook answers, with the
           values the author typed read through their declared types.
        5. ``check_bindings`` — the stored bindings against those inputs, and
           the inputs themselves against the two field rules.
        6. ``walk_edges`` — one walk over the edges that types every field,
           decides which nodes run once per row and completes the outputs,
           checking each node's outputs against the field rules as it does.
        7. ``interface`` — what the graph takes and returns.
        8. ``place_graphs`` — every node whose version is a compiled graph
           swapped for that graph's records, lifted under the row it runs on.
        9. ``conditions`` — the condition under which each output appears,
           over the graph with every placed graph's nodes in their place.

        Every problem is about a node the author placed, in this graph: a
        placed graph's own problems stay on it.
        """
        self.place()
        self.pin()
        self.order()
        self.ask_inputs()
        self.check_bindings()
        self.walk_edges()
        self.interface()
        self.place_graphs()
        self.conditions()

    # -- the steps --------------------------------------------------------------

    def place(self) -> None:
        """Every node the author placed, by id. Two nodes with one id is a
        fatal problem: the second would silently shadow the first everywhere
        else. An id holding ``/`` is one too: compile names an embedded
        graph's nodes with it, and an authored ``e/holder`` beside a graph
        node ``e`` would be read in the inner node's place."""
        for node in self.graph.nodes:
            if SEPARATOR in node.id:
                self.problems.append(problem("invalid_node_id", node.id))
                continue
            if node.id in self.authored:
                self.problems.append(problem("duplicate_node_id", node.id))
                continue
            self.authored[node.id] = node

    def pin(self) -> None:
        """The class and version each authored node uses, resolved once, here (``resolved``).

        A node stores a ``type`` and a ``version`` number; the registry gives
        the class and the class its version record. A stored graph can name
        a type the registry has since lost or a version the class has since
        dropped, so either miss is a problem on the node rather than an
        error. A version may be a graph the host compiled on its own, once
        per version; the walk sees it as one node and ``place_graphs`` lifts it.
        """
        for node in self.authored.values():
            found = resolved(node, self.registry)
            if isinstance(found, Problem):
                self.problems.append(found)
                continue
            self.pinned[node.id] = found

    def _refuse_if_broken(self, node_id: str, definition: type[NodeDefinition], graph: CompiledGraph) -> None:
        """A compiled graph can be placed only when nothing in it is fatal but
        an input it offers left empty: that value arrives from outside, where
        the graph is placed. An empty input the graph doesn't offer (locked,
        or on a node with an edge in) can never be filled, so it is a fault
        like any other. Otherwise the author sees one problem on the placed
        node, ``embedded_graph_broken``, counting the graph's faults, and the
        graph's own problems stay on the graph, in its own words
        (``compiled.node(id).version.problems``). A node compile couldn't
        work out always carries a fatal problem of its own or upstream, so
        the faults are the whole answer."""
        offered = {str(inp.name) for inp in graph.interface.inputs}
        faults = [
            found for found in graph.problems
            if found.fatal and not (found.code == "unbound_required" and f"{found.node_id}.{found.field}" in offered)
        ]
        if faults:
            self.unplaceable.add(node_id)
            self.problems.append(problem("embedded_graph_broken", node_id, graph=definition.title, problems=len(faults)))

    def order(self) -> None:
        """Who waits for whom, read off the authored edges, and an execution
        order over it; a node on a cycle is left out of the order and gets a
        fatal problem. The graph that runs is laid out from it: every node
        compile could resolve, in that order, each with its class and version."""
        self.authored_dependencies = dependencies_of(self.authored.values())
        self.authored_order, cyclic = order_of(self.authored_dependencies)
        self.problems.extend(problem("cycle", node_id) for node_id in sorted(cyclic))
        runs = [node_id for node_id in self.authored_order if node_id in self.pinned]
        versions = {node_id: self.pinned[node_id][1] for node_id in runs}
        self.expansion = Expansion(
            nodes={node_id: self.authored[node_id] for node_id in runs},
            order=tuple(runs),
            versions=versions,
            definitions={node_id: self.pinned[node_id][0] for node_id in runs},
            # A version is a NodeVersion or a compiled graph, which this module
            # may not import; this is the one place compile tells them apart.
            graphs=frozenset(node_id for node_id, version in versions.items() if not isinstance(version, NodeVersion)),
        )

    def ask_inputs(self) -> None:
        """Ask each node which inputs it has, once, on a fresh instance.

        A node's inputs may depend on the values the author typed into it (a
        mode dropdown that adds fields), so ``compute_inputs`` is called with
        those values, each read through its declared type by ``_typed_statics``
        and laid over the declaration's defaults, so a hook that reads a table's
        columns or a schema's fields gets the typed value and never parses JSON
        itself. That reading is the hook's argument and nothing more: once the
        hook has answered, every value is read once, against the inputs the
        hook returned — a hook may retype an input, and the value the author
        typed has to be read as that type — and that is what ``statics`` and
        ``listed`` keep, and where ``invalid_static`` is reported. Both hold
        only what the author typed: a default is the declaration's and applies
        wherever nothing is bound. Only the inputs are asked here. The outputs
        stay as declared until the walk over the edges, which can tell
        ``compute_outputs`` what type arrives on each connected input.

        A hook that cannot answer for these values raises ``Refuses``; its
        code and message are the node's one fatal ``Problem``, and the node is
        left out of every later pass, as a node whose type is unknown is.

        On a version that takes ``**inputs``, every connected name that is not
        a declared parameter becomes an ``Input`` of its own: typed ``Any`` or
        ``Series[Any]`` until the edge walk types it, titled by its name.

        Nothing else is asked of the node. Two inputs sharing a name, or a
        handle without a type an edge can carry, is ``check_bindings``'s to
        report.
        """
        for node_id, node in self.expansion.nodes.items():
            version = self.expansion.versions[node_id]
            if node_id not in self.expansion.graphs:
                instance = self.expansion.definitions[node_id]()
                defaults = {i.name: i.default for i in version.interface.inputs if i.optional}
                for_hook, _, _ = self._typed_statics(version.interface.inputs, node)
                try:
                    inputs = instance.compute_inputs(version.interface.inputs, {**defaults, **for_hook})
                except Refuses as refusal:
                    self.problems.append(Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node_id))
                    continue
            else:
                inputs = version.interface.inputs  # a compiled graph placed as a node: it has no hooks
            typed, listed, invalid = self._typed_statics(inputs, node)
            self.problems.extend(invalid)
            if node_id in self.expansion.graphs:
                version = self._as_placed(node_id, node, version, typed, listed)
                listed = frozenset()
            if version.interface.open is not None:
                # One Input per edge: `Series[Any]` when the node reduces each
                # ("series"), `Any` when it receives each whole ("single"). The
                # edge walk types it from what arrives.
                shape = Series[Any] if version.interface.open == "series" else Any
                named = {i.name for i in inputs}
                inputs = (*inputs, *(
                    Input(name=name, dtype=shape, title=name)
                    for name, binding in node.bindings.items()
                    if name not in named and isinstance(binding, From)
                ))
            self.interfaces[node_id] = replace(version.interface, inputs=inputs)
            self.statics[node_id] = typed
            self.listed[node_id] = listed

    def _as_placed(
        self, node_id: str, node: GraphNode, graph: CompiledGraph, typed: dict[str, Any], listed: frozenset[str],
    ) -> CompiledGraph:
        """The compiled graph as this placement fills it, and whether it can be placed.

        Values typed on a placed graph's inputs belong to the nodes inside,
        exactly as if they had been typed there, and an outer edge into an
        input the graph's own author filled replaces that value. Mostly the
        lift just puts the placement's value or edge on the inner field. But
        a value can decide something when compile runs: a list makes the node
        that holds it run once per item, and a hook reads the values typed on
        its node. When the placement's value, or the inner value an edge
        replaces, is one of those, what compile decided inside no longer
        holds, so the graph is compiled again with the placement's values
        (``CompiledGraph._refilled``). The copy replaces the version for this
        placement only.
        """
        refill = {name: value for name, value in typed.items() if name in listed or self._read_by_a_hook(graph, name)}
        cleared = [
            name for name, binding in node.bindings.items()
            if isinstance(binding, From) and self._filled_inside(graph, name)
            and (self._listed_inside(graph, name) or self._read_by_a_hook(graph, name))
        ]
        if refill or cleared:
            graph = graph._refilled(refill, cleared)
            self.expansion.versions[node_id] = graph
        self._refuse_if_broken(node_id, self.expansion.definitions[node_id], graph)
        return graph

    @staticmethod
    def _read_by_a_hook(graph: CompiledGraph, name: str) -> bool:
        """Does the node inside ``graph`` that input ``name`` lands on have a hook, which reads the values typed on it?"""
        definition = graph._compilation.expansion.definitions[inside(graph, name).node_id]
        return (
            definition.compute_inputs is not NodeDefinition.compute_inputs
            or definition.compute_outputs is not NodeDefinition.compute_outputs
        )

    @staticmethod
    def _listed_inside(graph: CompiledGraph, name: str) -> bool:
        """Did ``graph``'s own author type a list on input ``name``, so the node holding it runs once per item?"""
        ref = inside(graph, name)
        return ref.field in graph._compilation.listed.get(ref.node_id, ())

    def check_bindings(self) -> None:
        """Check the stored bindings against the inputs each node actually has, and those inputs against the field rules.

        Reports a lock on a field the node does not have (``unknown_locked_field``,
        not fatal), a binding on a field the node does not have (``stale_binding``:
        fatal, a typo that would otherwise run with the default, unless the
        node's ``compute_inputs`` makes its fields come and go), an edge into an input that cannot be connected
        (``show_handle=False``), an edge from a node that does not exist, and a required
        input nothing binds (``unbound_required``). A parameter typed ``Any``,
        or ``Series[Any]``, gets its type from its edge and nothing else, so
        with no edge it is ``unbound_required`` as well, and its node can't be
        worked out. Two inputs sharing a name, or a handle
        without a type an edge can carry (``field_problems``), and an input
        named with a leading underscore (``parameter_name_invalid``), are
        checked here on the inputs the hook answered, before the walk reads
        them.

        An edge from an id the author wrote but compile could not resolve —
        an unknown type, a cycle, a refused id — is left alone: that node
        carries its own fatal problem, and "not in the graph" would be
        untrue. Leaves in ``broken``
        the ids of the nodes whose edges or inputs are wrong; the edge walk
        leaves them out, since nothing can be derived from a broken edge.
        A typed required input left empty is fatal but does not break its
        node: the walk types it from its declaration, so the node and
        everything after it are still worked out.
        """
        nodes = self.expansion.nodes
        broken: set[str] = set(self.unplaceable)
        self.problems.extend(lock_problems(nodes, self.interfaces))
        for node_id, interface in self.interfaces.items():
            node = nodes[node_id]
            declared = {i.name: i for i in interface.inputs}
            faults = field_problems(node_id, interface.inputs, untyped_ok=True)
            # A leading underscore is private by convention and never a field
            # a call can carry: a declared parameter is refused at class
            # definition, and a ``**inputs`` edge or a hook's input here.
            faults += [problem("parameter_name_invalid", node_id, i.name) for i in interface.inputs if i.name.startswith("_")]
            if faults:
                broken.add(node_id)
                self.problems.extend(faults)

            for name, binding in node.bindings.items():
                if name not in declared:
                    dead = problem("stale_binding", node_id, name, inputs=", ".join(sorted(declared)) or "none")
                    if self.expansion.definitions[node_id].compute_inputs is not NodeDefinition.compute_inputs:
                        dead = dead.model_copy(update={"fatal": False})
                    self.problems.append(dead)
                    continue
                if not isinstance(binding, From):
                    continue
                target = declared[name]
                if not target.show_handle:
                    broken.add(node_id)
                    self.problems.append(problem("edge_into_closed_handle", node_id, name))
                for ref in binding.refs:
                    if ref.node_id in nodes:
                        continue  # a node of the run; the walk checks that it has the output
                    broken.add(node_id)
                    if ref.node_id not in self.graph_ids:
                        self.problems.append(problem("unknown_ref_node", node_id, name, source_node=ref.node_id))
                    # else: an id the author wrote that compile could not resolve; it carries its own fatal problem

            for inp in interface.inputs:
                if inp.dtype is Any or inp.dtype is Series[Any]:
                    if not isinstance(node.bindings.get(inp.name), From):
                        broken.add(node_id)
                        self.problems.append(problem("unbound_required", node_id, inp.name))
                elif not inp.optional and inp.name not in node.bindings and not (
                    node_id in self.expansion.graphs and self._filled_inside(self.expansion.versions[node_id], inp.name)
                ):
                    # Not broken: its type is declared, so the walk still works the node out, as a value that arrives at run time.
                    self.problems.append(problem("unbound_required", node_id, inp.name))
        self.broken = frozenset(broken)

    @staticmethod
    def _filled_inside(graph: CompiledGraph, name: str) -> bool:
        """Is ``name`` an input of ``graph`` that the graph's own author already
        filled in? Then nothing needs connecting where it is placed: the value
        inside holds unless the outer graph gives another."""
        ref = inside(graph, name)
        return isinstance(graph._compilation.expansion.nodes[ref.node_id].bindings.get(ref.field), Static)

    def walk_edges(self) -> None:
        """One walk over the edges in expanded order (``iteration.derive``):
        what type every field carries, which nodes run once per row and on
        which index, and each node's completed outputs. Nodes whose edges
        are broken are left out."""
        expansion = self.expansion
        self.iteration = derive(
            [
                expansion.nodes[node_id]
                for node_id in expansion.order
                if node_id in self.interfaces and node_id not in self.broken
            ],
            self.interfaces, expansion.versions, expansion.definitions, self.statics, self.listed, expansion.graphs,
        )
        self.problems.extend(self.iteration.problems)
        self.interfaces = {**self.interfaces, **self.iteration.interfaces}

    def place_graphs(self) -> None:
        """Every node whose version is a compiled graph and that the walk derived, filled in with that graph's records (``embedding.graph_as_placed``).

        The walk saw each as one node with the compiled graph's interface and
        decided the row it runs on; here its nodes take its place in the
        order, its fields' types, rows and receipts are lifted under that
        row, and every outer edge from one of its outputs is re-pointed at
        the inner field (``approve.check.amount`` at ``approve/check.amount``),
        as if the nodes had been drawn there. An index the walk named after
        one of its inputs is renamed after the inner field. The placed node
        itself stays, a ``graph`` whose version is the compiled graph. A
        graph the walk left out (it can't be placed, or an edge into it is
        broken) stays one node, with nothing under it.
        """
        expansion, iteration = self.expansion, self.iteration
        placed_ids = [node_id for node_id in expansion.order if node_id in expansion.graphs and node_id in iteration.iterated]
        if not placed_ids:
            return
        nodes, order = dict(expansion.nodes), list(expansion.order)
        versions, definitions, graphs = dict(expansion.versions), dict(expansion.definitions), set(expansion.graphs)
        iterated, types = dict(iteration.iterated), dict(iteration.types)
        indexes, receives = dict(iteration.indexes), dict(iteration.receives)
        renames: dict[str, Index] = {}
        for node_id in placed_ids:
            node, interface = nodes[node_id], self.interfaces[node_id]
            # Only an input the graph offers crosses into it; an edge into any
            # other name is already a ``stale_binding`` and reaches nothing.
            offered = {str(inp.name) for inp in versions[node_id].interface.inputs}
            edges = {name: binding for name, binding in node.bindings.items() if isinstance(binding, From) and name in offered}
            own = {Ref(node_id, str(field.name)) for field in (*interface.inputs, *interface.outputs)}
            # An index the walk named after one of this node's inputs (a
            # gathering fed several edges) is named after the inner field it reaches.
            for ref in own:
                index = indexes.get(ref)
                if index is not None and index.id == str(ref):
                    renames[index.id] = Index(str(self._lifted_ref(node_id, ref.field)), parent=index.parent)
            crossings = {
                name: Crossing(
                    type=types[Ref(node_id, name)],
                    index=renamed(indexes[Ref(node_id, name)], renames),
                    receives=received_renamed(receives[Ref(node_id, name)], renames),
                )
                for name in edges
            }
            lifted = graph_as_placed(versions[node_id], node_id, edges, self.statics[node_id], renamed(iterated[node_id], renames), crossings)
            for ref in own:
                types.pop(ref, None)
                indexes.pop(ref, None)
                receives.pop(ref, None)
            at = order.index(node_id)
            order[at:at + 1] = lifted.order
            nodes.update(lifted.nodes)
            versions.update(lifted.versions)
            definitions.update(lifted.definitions)
            graphs |= lifted.graphs
            self.interfaces.update(lifted.interfaces)
            self.statics.update(lifted.statics)
            self.listed.update(lifted.listed)
            iterated.update(lifted.iterated)
            types.update(lifted.types)
            indexes.update(lifted.indexes)
            receives.update(lifted.receives)
        # Every graph whose nodes are in place now: the ones placed here and
        # every one nested inside them. An edge from one of them is re-pointed.
        filled = graphs - (expansion.graphs - set(placed_ids))
        for node_id, node in nodes.items():
            if node_id in graphs:
                continue  # a graph node's own edges stay as its author wrote them
            if any(isinstance(b, From) and any(r.node_id in filled for r in b.refs) for b in node.bindings.values()):
                nodes[node_id] = node.model_copy(update={"bindings": {
                    name: From(*(expanded_ref(ref, filled) for ref in binding.refs)) if isinstance(binding, From) else binding
                    for name, binding in node.bindings.items()
                }})
        self.expansion = replace(
            expansion, nodes=nodes, order=tuple(order), versions=versions, definitions=definitions, graphs=frozenset(graphs),
        )
        self.iteration = replace(
            iteration,
            iterated={node_id: renamed(index, renames) for node_id, index in iterated.items()},
            types=types,
            indexes={ref: renamed(index, renames) for ref, index in indexes.items()},
            receives={ref: received_renamed(receipt, renames) for ref, receipt in receives.items()},
        )

    def conditions(self) -> None:
        """The condition under which each output appears, from the ``choice``
        groups of the nodes that run once (``conditions_of``)."""
        expansion, iteration = self.expansion, self.iteration
        self.output_conditions = conditions_of(
            [expansion.nodes[node_id] for node_id in expansion.order if node_id in iteration.iterated],
            self.interfaces, iteration.iterated,
        )

    def _lifted_ref(self, node_id: str, address: str) -> Ref:
        """The inner field an address on a placed node reaches, by lifted ref: ``emb`` + ``mid.join.result`` → ``emb/mid/join.result``."""
        inner = inside(self.expansion.versions[node_id], address)
        return Ref(f"{node_id}{SEPARATOR}{inner.node_id}", inner.field)

    def interface(self) -> None:
        """What the graph takes and returns, read off its nodes (``derive_interface``).
        A node whose version is a compiled graph counts with that graph's
        interface, as the walk completed it."""
        self.graph_interface = derive_interface(
            self.graph,
            {n: self.interfaces[n] for n in self.authored if n in self.interfaces},
            self.authored_dependencies,
        )

    # -- the values the author typed --------------------------------------------

    def _typed_statics(self, inputs: tuple[Any, ...], node: GraphNode) -> tuple[dict[str, Any], frozenset[str], list[Problem]]:
        """Every value the author typed into ``node``, read through its field's
        type as ``inputs`` states it; which of the scalar inputs hold many
        values; and the values no type could read.

        A stored ``Static`` holds JSON — a file comes back as a dict, an
        authored schema as a list — and every reader downstream wants the
        value, not its JSON form, so each is converted here. A sequence the
        type cannot read as one value is read as a sequence of values, which
        keeps a list-shaped scalar (a schema, a list of tags) one value while
        three uploaded files are three rows. Which of the two happened is
        decided by which reading succeeded, and returned as the set of
        inputs holding many — never again from the shape of the value. A
        value the type cannot read at all is a fatal ``invalid_static``,
        carrying whatever the type's constructor said; the caller decides
        whether to report it (``ask_inputs`` reads once for the hook's
        argument and once, reported, against the hook's answer). A value on
        a parameter typed ``Any`` is skipped: only an edge can give that
        parameter a type, and ``check_bindings`` reports it as unbound.
        """
        declared = {i.name: i for i in inputs}
        typed: dict[str, Any] = {}
        listed: set[str] = set()
        invalid: list[Problem] = []
        for name, value in static_values(node.bindings).items():
            inp = declared.get(name)
            if inp is None or inp.dtype is Any:
                continue  # a field the node lacks, or one only an edge can type; check_bindings reports both
            try:
                typed[name], many = self._typed_static(inp, value)
            except (ValidationError, TypeError, ValueError) as unreadable:
                said = _what_the_type_said(unreadable)
                invalid.append(problem("invalid_static", node.id, name, **({"reason": said} if said else {})))
                continue
            if many:
                listed.add(name)
        return typed, frozenset(listed), invalid

    @staticmethod
    def _typed_static(inp: Any, value: Any) -> tuple[Any, bool]:
        """``value`` read through ``inp``'s declared type, and whether it is
        many values: a sequence element by element for a ``Series[X]`` input
        (one series, not many); one value, or — when the type refuses the
        whole — a sequence of values, for anything else."""
        element = getattr(inp.dtype, "element", None)
        if element is not None:
            # A Series[X] input takes the whole sequence, each element typed.
            if not _sequence(value):
                raise TypeError("a Series input takes a sequence")
            return [_adapter(element).validate_python(v) for v in value], False
        try:
            return _adapter(inp.dtype).validate_python(value), False
        except ValidationError:
            if _sequence(value):
                return [_adapter(inp.dtype).validate_python(v) for v in value], True
            raise


def _sequence(value: Any) -> bool:
    """Is ``value`` a sequence of values, as a list widget or a multi-upload holds one? Text and bytes are one value each."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


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
