"""``CheckedGraph``: a stored graph checked against the nodes it uses.

A table is one value. An embedded graph is one node: its inputs are the inner
graph's inputs, its outputs the inner graph's outputs. It loads once, alone,
however many graphs embed it, and nothing of it is copied into the graph
around it. An author and a host see that one node, and an edge names its
output, never a node inside it. Its nodes as the loader places them are in
``placed_node.py``, and what a run walks of it is in ``tree.py``.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from typing import Any

from conductor._internal.address import NodePath
from conductor._internal.canonical_json import canonical_json
from conductor._internal.checked_graph.describe import (
    GraphDescription,
    GraphInput,
    GraphNodeDescription,
    PlacedFieldDescription,
    RunsOn,
    feed_of,
    runs_on,
)
from conductor._internal.checked_graph.placed_node import (
    PlacedEmbeddedGraph,
    PlacedFunction,
    PlacedNode,
)
from conductor._internal.checked_graph.tree import FunctionAtPath, NodeAtPath, nodes_at_paths
from conductor._internal.fields import Field
from conductor._internal.graph_record import GraphRecord
from conductor._internal.node_definition import FunctionDefinition
from conductor._internal.problem import Problem


@dataclass(frozen=True, slots=True)
class InputField:
    """What a run keeps of a graph input.

    Attributes:
        name: Its name in ``graph.inputs``: the input's own, or its address where two inputs share a name.
        address: Where the caller's value is kept, ``node.input``.
        field: Its field, whose default is what it holds when the caller gives nothing: the record's default, else
            the node's, and ``REQUIRED`` when the caller must give one.
    """

    name: str
    address: str
    field: Field


@dataclass(frozen=True, eq=False)
class CheckedGraph:
    """A graph record checked against the nodes it uses, made by the loader: its problems, its open inputs and its
    nodes placed. The ``Graph`` a host holds is a checked graph that can start a run; a stored graph used as a node
    is checked alone as one (``PlacedEmbeddedGraph.inner``) and never run alone.

    ``record``, ``problems`` and ``inputs`` are what a host reads of one, and ``Graph`` says what each holds.
    """

    record: GraphRecord
    problems: list[Problem]
    inputs: dict[str, GraphInput]
    #: Each node whose fields are known, by id, each after the nodes its inputs read. A graph with problems holds
    #: the nodes they are on too, with ``InputNotFed`` for an input that gets nothing, and never runs.
    _nodes: dict[str, PlacedNode]
    #: Each graph input as a run keeps it, by address.
    _input_fields: dict[str, InputField]
    #: The nodes that read each node.
    _readers: dict[str, list[PlacedNode]]
    #: The nodes with a ``FromRun`` parameter, an embedded graph's too, as ``from_run_values`` reads them.
    _needs_from_run: list[tuple[str, FunctionDefinition[..., Any]]]

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}"
        return f"<{type(self).__name__} of {len(self.record.nodes)} node{'s' if len(self.record.nodes) != 1 else ''}, {state}>"

    # -- what a host reads ------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """Whether the graph has no problems, so it runs."""
        return not self.problems

    @functools.cached_property
    def outputs(self) -> list[str]:
        """The outputs of the nodes nothing reads, as addresses."""
        return [f"{p.id}.{o}" for p in self._nodes.values() if not self._readers[p.id] for o in p.outputs]

    @property
    def runs_on(self) -> dict[str, RunsOn]:
        """Where each node runs: ``once``, or ``per_item`` of the innermost Series of its scope."""
        return {placed.id: runs_on(placed.scope) for placed in self._nodes.values()}

    @property
    def exclusive_groups(self) -> dict[str, dict[str, list[str]]]:
        """Each node's exclusive groups and their outputs, for a host to say which way a run went."""
        found: dict[str, dict[str, list[str]]] = {}
        for placed in self._nodes.values():
            for name, spec in placed.outputs.items():
                if spec.param.exclusive_group:
                    found.setdefault(placed.id, {}).setdefault(spec.param.exclusive_group, []).append(name)
        return found

    def describe(self) -> GraphDescription:
        """The graph as an editor draws it: every node of the record, with its fields and where it runs.

        Each node lists its own ``problems``, and ``problems`` beside the nodes lists every one. A node with a
        problem is drawn with its fields all the same, an input whose edge is refused fed ``unset``, and so is a
        node that reads it. A ``type_mismatch`` names the input in its ``loc``; a ``cycle`` and
        ``series_not_nested`` name the node. Only a node whose fields aren't known is there with ``loaded``
        false and no fields, its problems saying why: an unknown node or version, an id no address can say, a
        graph that embeds itself, a hook that failed or wasn't called for a value it reads, an embedded graph
        with problems of its own. An embedded graph is one node, with the fields it offers; one with problems
        isn't placed, and its problems are this graph's, each ``loc`` the embedding node's id and then the
        place inside it.
        """
        nodes: dict[str, GraphNodeDescription] = {}
        for record in self.record.nodes:
            own = [p for p in self.problems if p.loc[:1] == (record.id,)]
            placed = self._nodes.get(record.id)
            if placed is None:
                nodes[record.id] = GraphNodeDescription(node=record.node, version=record.version, loaded=False, problems=own)
                continue
            nodes[record.id] = GraphNodeDescription(
                node=record.node, version=record.version, loaded=True, problems=own, runs=runs_on(placed.scope),
                inputs=[PlacedFieldDescription(field=plan.field.description, feed=feed_of(placed.scope, plan)) for plan in placed.inputs.values()],
                outputs=[spec.description for spec in placed.outputs.values()],
            )
        return GraphDescription(nodes=nodes, inputs=dict(self.inputs), outputs=self.outputs, problems=list(self.problems))

    # -- what a run walks -------------------------------------------------------

    @functools.cached_property
    def _at_paths(self) -> tuple[list[FunctionAtPath], dict[str, NodeAtPath]]:
        """The graph's nodes and those of each graph embedded in it, where they stand (``nodes_at_paths``),
        made when first read."""
        return nodes_at_paths(self)

    @property
    def _tree(self) -> list[FunctionAtPath]:
        """Every function of the graph and of each graph embedded in it, at any depth, each after the
        functions it reads: the order a leg tries them in."""
        return self._at_paths[0]

    @property
    def _node_at(self) -> dict[str, NodeAtPath]:
        """Each node of the graph and of each graph embedded in it, a function or an embedded graph, by its
        name (``e/ask``)."""
        return self._at_paths[1]

    @functools.cached_property
    def _functions(self) -> dict[NodePath, PlacedFunction]:
        """Every function node of ``_tree``, by its path."""
        return {node.path: node.placed for node in self._tree}

    @functools.cached_property
    def _threads(self) -> int:
        """How many calls a leg over the graph can have on threads at once: the ``concurrency`` of every node
        called on a thread, those of each embedded graph with them."""
        return sum(policy.concurrency for placed in self._functions.values() if (policy := placed.definition.policy).threaded)

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of what a run of this graph rests on: the record, but for each node's ``display``, which
        conductor never reads, and the fingerprint of every graph embedded in it.

        Keys are sorted first, so a record a JSON store gives back in another key order has the same fingerprint.
        A saved run names it, and loading a save into a graph whose fingerprint differs is refused.
        """
        own = self.record.model_dump(mode="json", exclude={"nodes": {"__all__": {"display"}}})
        embedded = {p.id: p.inner.fingerprint for p in self._nodes.values() if isinstance(p, PlacedEmbeddedGraph)}
        return hashlib.blake2b(canonical_json([own, embedded]), digest_size=8).hexdigest()
