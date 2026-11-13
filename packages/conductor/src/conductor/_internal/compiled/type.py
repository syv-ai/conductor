"""``CompiledGraph``: a graph record compiled against the node definitions it uses.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

Its record declares the graph input ``text`` on ``lines.text`` and the graph output ``result``, the output of
``join``. Compiled, it is ``<Graph of 3 nodes, runnable>``. Its ``inputs`` has the one graph input, ``text``, its
``outputs`` are ``["result"]``, and ``runs_on["shout"]`` says ``shout`` is called once per item of
``lines.result``. ``Graph``, the class a host uses, is a ``CompiledGraph`` that can also start a run.

A subgraph used as a node is one node of the graph. Its inputs and outputs are the graph inputs and outputs its
own record declares, compiled once, on its own, and none of its nodes is copied into this graph's nodes. An edge
names one of its outputs (``loud.result``), never a node inside it. The nodes of a compiled graph are in
``node.py``.

The names with an underscore (``_nodes``, ``_inputs``, ``_inputs_door``, ...) are what the engine reads:
the compiler, a subgraph node and a run. They keep the underscore because ``Graph`` is a ``CompiledGraph``, and a
host that holds a ``Graph`` must not see them.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Annotated

from conductor._internal.base import Address, Loc, NodePath, json_hash, plural
from conductor._internal.context import ContextParameter
from conductor._internal.field import Field
from conductor._internal.gate import ContextDoor, InputsDoor, RefusedError
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.record import GraphRecord
from conductor._internal.types import Param

from .describe import (
    GraphDescription,
    GraphInputDescription,
    GraphNodeDescription,
    NodeInputDescription,
    RunsOn,
    input_source_of,
    runs_on,
)
from .input import Edge
from .node import (
    CompiledNode,
    CompiledSubgraphNode,
    RequiredBranch,
)


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
    #: Each node that was compiled, by id, each after the nodes upstream of it. A node with a problem on one of its
    #: inputs is here too, with ``MissingInput`` for that input; a graph with problems never runs.
    _nodes: dict[str, CompiledNode]
    #: Each graph input, by name, as the caller gives a value for it: its type, and the value used when the caller
    #: gives none (``REQUIRED`` when the caller must give one).
    _inputs: dict[str, Field]
    #: Each graph output, by name, as what an edge from it reads: ``{"result": Edge(Address("join", "result"), ...)}``.
    #: A graph output whose source the graph cannot read is left out, and the graph has the problem.
    _outputs: dict[str, Edge]
    #: The class the registry names with ``context_type``, which every node's ``Context[X]`` was checked against;
    #: the ``context=`` of a run is an instance of it. ``None`` when the registry names none.
    _context_type: type | None

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else plural(len(self.problems), "problem")
        return f"<{type(self).__name__} of {plural(len(self.record.nodes), 'node')}, {state}>"

    # -- what a host reads ------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """Whether the graph has no problems, so it runs."""
        return not self.problems

    @property
    def inputs(self) -> dict[str, GraphInputDescription]:
        """The graph inputs, by name: what a caller gives a value for. In the example, ``text``."""
        return {name: GraphInputDescription(field=field.description) for name, field in self._inputs.items()}

    @property
    def outputs(self) -> list[str]:
        """The names of the graph outputs the record declares, sorted. In the example, ``["result"]``."""
        return sorted(self.record.outputs)

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
        (``compute_added_inputs`` or ``compute_outputs``) failed or needs a constant it was not given, or it is a
        subgraph that contains itself or has problems of its own. A subgraph node is one node, with the inputs and
        outputs of its subgraph. Each problem of a subgraph is a problem of this graph, with a ``loc`` that starts
        with the subgraph node's id and goes on with the place inside the subgraph.
        """
        problems_of: dict[Loc, list[Problem]] = {}  # by the start of each problem's loc: its node's id
        for graph_problem in self.problems:
            problems_of.setdefault(graph_problem.loc[:1], []).append(graph_problem)
        nodes: dict[str, GraphNodeDescription] = {}
        for node_id, record in sorted(self.record.nodes.items()):
            own = problems_of.get((node_id,), [])
            compiled_node = self._nodes.get(node_id)
            if compiled_node is None:
                nodes[node_id] = GraphNodeDescription(node=record.node, version=record.version, found=False, problems=own)
                continue
            nodes[node_id] = GraphNodeDescription(
                node=record.node, version=record.version, found=True, problems=own, runs=runs_on(compiled_node.loops),
                inputs=[NodeInputDescription(field=node_input.field.description, source=input_source_of(compiled_node.loops, node_input)) for node_input in compiled_node.inputs.values()],
                outputs=compiled_node.described_outputs(),
            )
        return GraphDescription(nodes=nodes, inputs=dict(self.inputs), outputs=self.outputs, problems=list(self.problems))

    # -- what a subgraph node that uses this graph reads ------------------------

    def _output_field(self, output: str) -> Field:
        """The graph output ``output`` as the output of a subgraph node that uses this graph: the field of the node's
        output it is, under the graph output's name.

        An exclusive group is named by the node here (``route.length``), so the groups of two nodes stay apart.
        """
        source = self._outputs[output].source
        field = self._nodes[source.node].outputs[source.output]
        group = field.param.exclusive_group
        hint = Annotated[field.hint, Param(exclusive_group=f"{source.node}.{group}")] if group else field.hint
        return Field(output, hint, field.default)

    def _loops_around(self, output: str) -> tuple[Address, ...]:
        """Returns the loops inside this graph that the graph output ``output`` has one value per item of, outermost
        first."""
        source = self._outputs[output].source
        return self._nodes[source.node].loops_around(source.output)

    def _branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches inside this graph that must be chosen for the graph output ``output`` to have a
        value."""
        source = self._outputs[output].source
        return self._nodes[source.node].branches_required_for(source.output)

    # -- what a run uses --------------------------------------------------------

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
        return json_hash([own, subgraph_fingerprints], 8, keys_sorted=True)

    @functools.cached_property
    def _context_door(self) -> ContextDoor:
        """Checks the ``context=`` of each call that starts or resumes a run: built once, from the registry's
        ``context_type`` and every node that takes the context, those inside subgraphs too."""
        return ContextDoor(self._context_type, self._nodes_taking_context())

    def _nodes_taking_context(self) -> list[tuple[NodePath, ContextParameter]]:
        """Every function node of the graph and of its subgraphs that takes the run's context, as its node path
        (``("loud", "shout")``) and its ``Context`` parameter, in the order a run tries them."""
        takers: list[tuple[NodePath, ContextParameter]] = []
        for compiled_node in self._nodes.values():
            if isinstance(compiled_node, CompiledSubgraphNode):
                takers += [((compiled_node.id, *path), parameter) for path, parameter in compiled_node.inner._nodes_taking_context()]
            elif (parameter := compiled_node.definition.context) is not None:
                takers.append(((compiled_node.id,), parameter))
        return takers

    def _checked_context[C](self, context: C | None) -> C | None:
        """``context`` as the host passed it, once the context door let it in.

        Raises:
            InvalidCallError: The door refused it: of another class (``invalid_context``), or none for a graph with a
                node that takes it (``context_missing``).
        """
        try:
            checked: C | None = self._context_door(context)
        except RefusedError as refused:
            raise InvalidCallError(refused.problems) from None
        return checked

    @functools.cached_property
    def _inputs_door(self) -> InputsDoor:
        """Reads the graph inputs of each call that makes a run: built when the first run is made."""
        return InputsDoor(self._inputs)
