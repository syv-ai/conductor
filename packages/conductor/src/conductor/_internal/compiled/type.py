"""``CompiledGraph``: a graph record compiled against the node definitions it uses.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

Compiled, it is ``<Graph of 3 nodes, runnable>``. Its ``inputs`` has one graph input, ``text`` (``lines.text``),
its ``outputs`` are ``["join.result"]``, and ``runs_on["shout"]`` says ``shout`` is called once per item of
``lines.result``. ``Graph``, the class a host uses, is a ``CompiledGraph`` that can also start a run.

A subgraph used as a node is one node of the graph. Its inputs and outputs are those of the subgraph's own
compiled graph, which is compiled once, on its own, and none of its nodes is copied into this graph's nodes. An
edge names one of its outputs (``loud.shout.result``), never a node inside it. The nodes of a compiled graph are
in ``nodes.py``. The flat graph, which a run works on and which has the function nodes inside subgraphs too, is
in ``flat_nodes.py``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

from conductor._internal.address import NodePath
from conductor._internal.compiled.describe import (
    GraphDescription,
    GraphInput,
    GraphNodeDescription,
    NodeInputDescription,
    RunsOn,
    input_source_of,
    runs_on,
)
from conductor._internal.compiled.flat_nodes import FlatFunctionNode, FlatNode, flat_nodes
from conductor._internal.compiled.nodes import (
    CompiledFunctionNode,
    CompiledNode,
    CompiledSubgraphNode,
)
from conductor._internal.field import Field
from conductor._internal.graph_record import GraphRecord
from conductor._internal.hashing import canonical_json, short_hash
from conductor._internal.node_definition import FunctionDefinition
from conductor._internal.problem import Problem
from conductor._internal.text import plural


@dataclass(frozen=True, slots=True)
class InputField:
    """A graph input, as a run reads the caller's value for it. In the example,
    ``InputField("text", "lines.text", field)`` with the field ``text: str``.

    Attributes:
        name: Its key in ``graph.inputs``: the input's own name, or its address when two graph inputs share a name.
        address: The key a run keeps the caller's value under, ``node.input``.
        field: The input. Its default is the value used when the caller gives none: the graph record's default,
            else the node's, and ``REQUIRED`` when the caller must give one.
    """

    name: str
    address: str
    field: Field


@dataclass(frozen=True, eq=False)
class CompiledGraph:
    """A graph record compiled against the node definitions it uses: its problems, its graph inputs and its
    nodes. ``compile_graph`` makes it. ``Graph``, the class a host uses, is a compiled graph that can start a run.
    A subgraph's own graph record is compiled as one too (``CompiledSubgraphNode.inner``), and never run on its
    own.

    ``record``, ``problems`` and ``inputs`` are what a host reads; ``Graph``'s docstring says what each is.
    """

    record: GraphRecord
    problems: list[Problem]
    inputs: dict[str, GraphInput]
    #: Each node that was compiled, by id, each after the nodes upstream of it. A node with a problem on one of its
    #: inputs is here too, with ``InputMissing`` for that input; a graph with problems never runs.
    _nodes: dict[str, CompiledNode]
    #: Each graph input as a run reads it, by address.
    _input_fields: dict[str, InputField]
    #: For each node id, the nodes that an edge from it leads to.
    _downstream: dict[str, list[CompiledNode]]
    #: The id and definition of every function node with a ``Context`` parameter, those inside subgraphs too. A run
    #: reads it (``context_values``) to check it was given every ``Context`` value.
    _needs_context: list[tuple[str, FunctionDefinition[..., Any]]]

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else plural(len(self.problems), "problem")
        return f"<{type(self).__name__} of {plural(len(self.record.nodes), 'node')}, {state}>"

    # -- what a host reads ------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """Whether the graph has no problems, so it runs."""
        return not self.problems

    @functools.cached_property
    def outputs(self) -> list[str]:
        """The graph outputs, as addresses: every output of every node that no edge leaves. In the example,
        ``["join.result"]``."""
        return [f"{p.id}.{o}" for p in self._nodes.values() if not self._downstream[p.id] for o in p.outputs]

    @property
    def runs_on(self) -> dict[str, RunsOn]:
        """For each node id, whether the node is called ``once`` or ``per_item`` of the list of the innermost loop
        it is inside. In the example, ``shout`` is ``per_item`` of ``lines.result``."""
        return {compiled_node.id: runs_on(compiled_node.loops) for compiled_node in self._nodes.values()}

    @property
    def exclusive_groups(self) -> dict[str, dict[str, list[str]]]:
        """The outputs in each exclusive group, by node id and then group name, for a host to show which
        output each call chose."""
        found: dict[str, dict[str, list[str]]] = {}
        for compiled_node in self._nodes.values():
            for name, field in compiled_node.outputs.items():
                if field.param.exclusive_group:
                    found.setdefault(compiled_node.id, {}).setdefault(field.param.exclusive_group, []).append(name)
        return found

    def describe(self) -> GraphDescription:
        """Returns the graph as an editor draws it: every node of the record, with its inputs and outputs and
        whether it is called once or once per item.

        Each node lists its own ``problems``, and the description's ``problems`` lists every problem of the graph.
        A node with a problem is described with its inputs and outputs all the same; an input that gets no value
        because of a problem has the source ``missing``. A ``type_mismatch`` names the input in its ``loc``; a
        ``cycle`` and ``series_not_nested`` name the node.

        Only a node that could not be compiled has ``found`` false and no inputs or outputs, and its problems say
        why: its definition or version is not in the registry, its id cannot be written in an address, a hook
        failed or needs a value it was not given, or it is a subgraph that contains itself or has problems of its
        own. A subgraph node is one node, with the inputs and outputs of its subgraph. Each problem of a subgraph
        is a problem of this graph, with a ``loc`` that starts with the subgraph node's id and goes on with the
        place inside the subgraph.
        """
        nodes: dict[str, GraphNodeDescription] = {}
        for record in self.record.nodes:
            own = [p for p in self.problems if p.loc[:1] == (record.id,)]
            compiled_node = self._nodes.get(record.id)
            if compiled_node is None:
                nodes[record.id] = GraphNodeDescription(node=record.node, version=record.version, found=False, problems=own)
                continue
            nodes[record.id] = GraphNodeDescription(
                node=record.node, version=record.version, found=True, problems=own, runs=runs_on(compiled_node.loops),
                inputs=[NodeInputDescription(field=plan.field.description, source=input_source_of(compiled_node.loops, plan)) for plan in compiled_node.inputs.values()],
                outputs=compiled_node.described_outputs(),
            )
        return GraphDescription(nodes=nodes, inputs=dict(self.inputs), outputs=self.outputs, problems=list(self.problems))

    # -- what a run uses --------------------------------------------------------

    @functools.cached_property
    def _flat_nodes(self) -> tuple[dict[str, FlatFunctionNode], dict[str, FlatNode]]:
        """The flat graph (``flat_nodes``), made the first time it is read."""
        return flat_nodes(self)

    @functools.cached_property
    def _flat_function_nodes(self) -> list[FlatFunctionNode]:
        """Every function node of the graph and of its subgraphs, at any depth, in the order a run tries to call
        them: each after the nodes upstream of it."""
        return list(self._flat_nodes[0].values())

    @property
    def _function_nodes_by_name(self) -> dict[str, FlatFunctionNode]:
        """Every function node of ``_flat_function_nodes``, by its name (``"loud/shout"``)."""
        return self._flat_nodes[0]

    @property
    def _nodes_by_name(self) -> dict[str, FlatNode]:
        """Every node of the graph and of its subgraphs, a function or a subgraph, by its name
        (``"loud"``, ``"loud/shout"``)."""
        return self._flat_nodes[1]

    @functools.cached_property
    def _functions(self) -> dict[NodePath, CompiledFunctionNode]:
        """Every function node of ``_flat_function_nodes`` as its own graph compiled it, by its node path
        (``("loud", "shout")``)."""
        return {node.node_path: node.compiled_node for node in self._flat_function_nodes}

    @functools.cached_property
    def _threads(self) -> int:
        """How many calls can run on threads at once during one call of ``resume()``: the sum of the
        ``concurrency`` of every function node called on a thread, those inside subgraphs too."""
        return sum(policy.concurrency for compiled_node in self._functions.values() if (policy := compiled_node.definition.policy).threaded)

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of what a run of this graph depends on: the graph record without each node's
        ``display`` (which conductor never reads), and the fingerprint of every subgraph used in it.

        Keys are sorted first, so a record that a JSON store gives back with its keys in another order has the
        same fingerprint. A saved run contains it, and loading a saved run into a graph whose fingerprint differs
        is refused.
        """
        own = self.record.model_dump(mode="json", exclude={"nodes": {"__all__": {"display"}}})
        subgraph_fingerprints = {p.id: p.inner.fingerprint for p in self._nodes.values() if isinstance(p, CompiledSubgraphNode)}
        return short_hash(canonical_json([own, subgraph_fingerprints]), 8)
