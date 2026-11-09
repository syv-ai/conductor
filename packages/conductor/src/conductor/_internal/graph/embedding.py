"""A compiled graph embedded as one node: what it offers, and how its nodes are placed.

A host compiles each graph version once and hands the ``CompiledGraph`` to
conductor as the node's version. The outer compile wires the graph node as a
plain node with the graph's interface, which decides the row it runs on, then
compiles each node of the graph where it stands, under that row (``runs_under``
of ``compile_one_node``): the same walk, so an embedded graph behaves as its
nodes drawn there. Under a row that can differ from the graph run alone, since
there is then a level above its columns: a reduction that ran per row of a
series the graph makes runs per row of the graph's row instead. What each node
reads is what its own run holds, so a run never sees another row's values.
Its author's own problems were reported when
it was compiled; a problem only this graph node's edges or values cause is the
graph node's. Its inner nodes' ids take the graph node's path (``emb/lines``).

A graph that embeds itself is the host's to catch, as it builds the
versions: a compiled graph can't contain itself, but a host that swaps a
version in place can close a loop through a static on an embedded
graph, which compile then follows until Python's recursion limit.

``/`` separates the graphs inside a node id, and ``.`` stays the address
separator, so a place in the run is a path and a field: ``approve/check.amount``.
Only embedding writes a ``/``: an id holding one is the problem ``invalid_node_id``.

An embedded graph has an interface of its own. Each input and output it offers
goes by its bare name where that is unique on its side, by its address on a
clash (``offered_names``), and from outside those names are all there is: a
binding, a ``From``, an entry in an editor's list of nodes. A rename inside it breaks no caller.
Inner addresses stay for reading a run (``approve/check``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor._internal.graph.binding import Binding, From, Static
from conductor._internal.graph.compiled_node import CompiledNode
from conductor._internal.node_signature import Interface
from conductor._internal.ref import NODE_PATH_SEPARATOR, Ref
from conductor._internal.table import columns_of

if TYPE_CHECKING:
    from conductor._internal.errors import Problem
    from conductor._internal.field_records import Input
    from conductor._internal.graph.compiled_graph import CompiledGraph
    from conductor._internal.graph.graph_records import GraphNode


def offered_names(addresses: Iterable[Ref]) -> dict[Ref, str]:
    """The name each of these fields goes by: its bare field name where that is unique among them, its address on a clash.

    The one rule for naming what a graph offers, used for wiring an embedded
    graph (``a.text`` is ``text``) and for ``with_inputs``."""
    addresses = list(addresses)
    bare = Counter(ref.field for ref in addresses)
    return {ref: ref.field if bare[ref.field] == 1 else str(ref) for ref in addresses}


def parent_graph_of_node_id(node_id: str) -> str | None:
    """The graph a node inside sits in, read off its id: ``approve/check`` for
    ``approve/check/amount``; ``None`` for an id with no ``/``."""
    outer, _, _ = node_id.rpartition(NODE_PATH_SEPARATOR)
    return outer or None


@dataclass(frozen=True)
class EmbeddableGraph:
    """What a compiled graph tells a graph that embeds it (``CompiledGraph._embeddable``)."""

    #: The interface the embedding graph sees: the same fields under ``offered_names``.
    interface: Interface
    #: The inputs it offers, by their offered names: the node its author drew and the field there.
    inputs: Mapping[str, Ref]
    #: The outputs it offers and each column of a table output, by offered name, each with where it is in the run.
    outputs: Mapping[str, Ref]
    #: Every fatal problem but an input it offers left empty, which a value from outside fills.
    problems: tuple[Problem, ...]


def embeddable_view_of(graph: CompiledGraph) -> EmbeddableGraph:
    """The record a graph that embeds ``graph`` asks."""
    nodes = graph._nodes

    def output_address_in_the_run(ref: Ref) -> Ref:
        """Where an output of ``graph``'s interface is in the run: on a node that is itself a graph, the place inside it."""
        embedded = nodes[ref.node_id].embedded_graph
        if embedded is None:
            return ref
        inside = embedded._embeddable.outputs[ref.field]
        return Ref(f"{ref.node_id}{NODE_PATH_SEPARATOR}{inside.node_id}", inside.field)

    # One namespace for both sides: a field name is unique within the node this graph becomes.
    names = offered_names(Ref(field.name) for field in (*graph.interface.inputs, *graph.interface.outputs))
    inputs = {names[Ref(input_.name)]: Ref(input_.name) for input_ in graph.interface.inputs}
    outputs: dict[str, Ref] = {}
    for out in graph.interface.outputs:
        name, at = names[Ref(out.name)], output_address_in_the_run(Ref(out.name))
        outputs[name] = at
        outputs.update({f"{name}.{column}": Ref(at.node_id, f"{at.field}.{column}") for column, _ in columns_of(out.dtype)})
    interface = Interface(
        inputs=tuple(_as_offered(input_, names[Ref(input_.name)], nodes[Ref(input_.name).node_id]) for input_ in graph.interface.inputs),
        outputs=tuple(out.model_copy(update={"name": names[Ref(out.name)]}) for out in graph.interface.outputs),
        needs=graph.interface.needs,
    )
    return EmbeddableGraph(interface=interface, inputs=inputs, outputs=outputs, problems=tuple(_problems_blocking_embedding(graph, nodes, inputs)))


def _as_offered(input_: Input, name: str, node: CompiledNode) -> Input:
    """``input_`` as a graph node offers it, under ``name``: one its author filled inside is optional there."""
    update: dict[str, Any] = {"name": name}
    if isinstance(node.graph_node.bindings.get(Ref(input_.name).field), Static):
        update["required"] = False
    return input_.model_copy(update=update)


def _problems_blocking_embedding(graph: CompiledGraph, nodes: Mapping[str, CompiledNode], offered: Mapping[str, Ref]) -> list[Problem]:
    """A graph can be embedded only when nothing in it is fatal but an input it
    offers left empty, which a value from outside fills. An empty input the
    graph doesn't offer can never be filled, and one with no type of its own
    leaves its node unworked-out, so each stays a problem."""
    offered_addresses = set(offered.values())

    def a_value_from_outside_fills_it(problem: Problem) -> bool:
        if problem.type != "unbound_required":
            return False
        node = nodes.get(str(problem.loc[0]))
        return node is not None and node.wiring is not None and Ref(str(problem.loc[0]), str(problem.loc[1])) in offered_addresses

    return [problem for problem in graph.problems if problem.fatal and not a_value_from_outside_fills_it(problem)]


def inner_node_placed_at(inner: GraphNode, prefix: str, filled_from_outside: Mapping[str, Binding]) -> GraphNode:
    """``inner`` as the node it is once embedded at ``prefix``: its id and edges under that path, and any edge or static the graph node set on it in place."""
    bindings: dict[str, Binding] = {
        name: From(*(Ref._from_checked_parts(prefix + ref.node_id, ref.field) for ref in binding.refs)) if isinstance(binding, From) else binding
        for name, binding in inner.bindings.items()
    }
    bindings.update(filled_from_outside)
    return inner.model_copy(update={"id": prefix + inner.id, "bindings": bindings, "locked": ()})
