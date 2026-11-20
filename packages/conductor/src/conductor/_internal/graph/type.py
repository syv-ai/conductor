"""``Graph``, a graph record that has been compiled and is ready to run.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

``lines`` splits a text into lines. ``shout`` is called once for each line, and ``join`` gets all their results
back as one list. The record makes ``lines.text`` the graph input ``text``, which the caller gives, and ``join``'s
output the graph output ``result``. Compiled, it shows as ``<Graph of 3 nodes, runnable>``.

A subgraph used as a node counts as one node of the graph. Its inputs and outputs are the graph inputs and outputs
its own record declares. Its record is compiled once, on its own, and none of its nodes is copied into this graph.
An edge reads one of its outputs, like ``loud.result``, and never a node inside it.

This module and ``run/`` need each other. ``run/`` sits above this package and imports it when it loads. So this
module imports from ``run/`` only inside the methods that need it (``with_inputs``, ``_blueprint`` and
``_codec``), and under ``TYPE_CHECKING`` for its annotations. The layer contract in ``pyproject.toml`` allows
that. ``compile.py`` and this module import each other too, so ``Graph.compile`` imports it inside its body.

Names with an underscore, like ``_nodes`` and ``_inputs``, are for the engine: the compiler, subgraph nodes and
runs. A host never uses them.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import MISSING

from conductor._internal.ask import AnswerAsJsonValidator
from conductor._internal.base import Loc, json_hash, place_text
from conductor._internal.context import ContextValidator
from conductor._internal.node import NodeDefinition
from conductor._internal.port import Port, PortDescription
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.record import GraphRecord, RecordValidator
from conductor._internal.utils import plural

from .describe import (
    GraphDescription,
    GraphNodeDescription,
    NodeInputDescription,
    innermost_series,
    input_source_of,
)
from .input import Edge
from .node import (
    CompiledNode,
    CompiledSubgraphNode,
)
from .validator import InputsValidator, LimitsValidator

if TYPE_CHECKING:
    from conductor._internal.run import Run, RunBlueprint, SavedRun


class Definitions[C](Protocol):
    """The node definitions a graph is compiled against.

    In practice this is a ``NodeRegistry``. The registry lives in a package above this one, so this protocol names
    just what the compiler uses. ``definitions["shout"]`` gives the newest version of ``shout``, and raises
    ``KeyError`` for an id it doesn't know. Iterating gives every definition. ``context_type`` is the class of the
    run's context that the registry names, or ``None``.
    """

    def __getitem__(self, node_id: str, /) -> NodeDefinition: ...

    def __iter__(self) -> Iterator[NodeDefinition]: ...

    @property
    def context_type(self) -> type[C] | None: ...


@dataclass(frozen=True, eq=False)
class Graph[C]:
    """A graph record that has been checked against a registry's nodes and can start runs.

    Make one with ``Graph.compile(record, registry)``. Compiling never fails on a wrong graph. Whatever is wrong
    ends up in ``problems``, and only a graph with no problems runs: ``graph.run(inputs)``.

    ``C`` is the type of the run's context. A registry made with ``NodeRegistry(context_type=HostContext)``
    compiles a ``Graph[HostContext]``. With no ``context_type``, ``C`` is ``None``.

    A subgraph's own record is compiled into a ``Graph`` too (``CompiledSubgraphNode.inner``). That graph is never
    run on its own.

    Attributes:
        record: The graph record it was compiled from.
        problems: Everything wrong with the graph. Empty when it can run.
        inputs: The graph inputs the record offers (``GraphRecord.inputs_offered``), by name. These are what a
            caller gives.
        outputs: The graph outputs the record offers (``GraphRecord.outputs_offered``) whose node compiled, by name.
            These are what a complete run gives. One whose node didn't compile isn't here, as with ``inputs``.
        asks: Whether a node of the graph, or of a subgraph in it at any depth, may stop to ask a person.
    """

    record: GraphRecord
    problems: list[Problem]
    #: Every node that compiled, by id. Each comes after the nodes it reads from. A node with a problem on one of
    #: its inputs is here too, with a ``MissingInput`` for that input. A graph with problems never runs anyway.
    _nodes: dict[str, CompiledNode]
    #: Each graph input, by name, as a port: its type, and the value used when the caller gives none
    #: (``REQUIRED`` when the caller must give one).
    _inputs: dict[str, Port]
    #: Each graph output whose node compiled, by name, sorted: the port of the node output behind it, under the
    #: graph output's name, with no default.
    _outputs: dict[str, Port]
    #: The ``context_type`` the registry names, or ``None``. Every node's ``Context[X]`` was checked against it,
    #: and a run's ``context=`` is an instance of it.
    _context_type: type[C] | None

    @staticmethod
    def compile[R](
        record: GraphRecord | str | bytes | Mapping[str, Any], registry: Definitions[R]
    ) -> Graph[R]:
        """Compiles a graph record against the node definitions in ``registry``, a ``NodeRegistry``.

        The record can be a ``GraphRecord``, its JSON text, or the plain data ``read_json`` returns.

        A graph that is wrong or half finished doesn't raise. What's wrong goes in ``graph.problems``, and a graph
        with problems won't run. The graph keeps its own copy of the definitions it used, as they are now, and the
        registry's ``context_type``.

        Raises:
            InvalidCallError: ``record`` doesn't have the shape of a graph record (``invalid_record``).
        """
        # compile.py imports this module when it loads, so this module imports it here, not at the top
        from .compile import Compiler

        return Compiler(
            RecordValidator()(record), registry, inside_subgraphs=(), compiled_subgraphs={}
        ).compile()

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else plural(len(self.problems), "problem")
        return f"<{type(self).__name__} of {plural(len(self.record.nodes), 'node')}, {state}>"

    # -- what a host uses -------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """Whether the graph has no problems, so it can run."""
        return not self.problems

    @functools.cached_property
    def inputs(self) -> Mapping[str, PortDescription]:
        """The graph inputs, by name: what a caller gives values for. In the example, that's ``text``.

        They come in the order the nodes run in: a node's after those of the nodes it reads from, by node id where
        that leaves a choice, and a node's own in the order it declares them. The record doesn't decide it, since it
        keeps no order.

        A graph input that only nodes switched off take isn't here, and a caller who gives it is refused.

        The mapping is read-only, since every caller shares it: copy it with ``dict(graph.inputs)`` to change it.
        """
        return MappingProxyType({name: port.description for name, port in self._inputs.items()})

    @functools.cached_property
    def outputs(self) -> Mapping[str, PortDescription]:
        """The graph outputs, by name, sorted: what a complete run gives (``run.outputs``).

        Each is described as the output of the node behind it, under the graph output's name. In the example,
        ``result`` is ``join``'s ``result``, a ``str``.

        An output that needs a branch may be skipped on a run, so it is described as one that may be skipped:
        ``"required": false``, with no ``default``.

        An output whose node is inside loops is described as a list (``is_series``), with one level per entry of
        ``loops``. If ``join`` were called once per document, ``result`` would have ``is_series`` true and the
        ``loops`` ``["documents.result"]``, a ``str`` per document. An output that's a ``Series`` itself adds one
        more, innermost level: the list it returns.

        A graph output whose node didn't compile isn't here. The graph's problems say why. Nor is one whose node is
        switched off. One read from a node downstream of it is here, and a run skips it.

        The mapping is read-only, since every caller shares it, ``describe()`` and a subgraph node that uses this
        graph included: copy it with ``dict(graph.outputs)`` to change it.
        """
        described: dict[str, PortDescription] = {}
        for name, port in self._outputs.items():
            node, output = self._node_returning(name)
            description = port.description
            if node.branches_required_for(output):
                description = description.model_copy(update={"required": False, "default": MISSING})
            edge = node.edge_reading(output)
            loops = [place_text(*loop) for loop in edge.loops]
            if loops and port.is_series:
                loops.append(place_text(edge.node, edge.output))
            if loops:
                description = description.model_copy(update={"is_series": True, "loops": loops})
            described[name] = description
        return MappingProxyType(described)

    @functools.cached_property
    def asks(self) -> bool:
        """Whether any function node may stop to ask a person, in this graph or in a subgraph at any depth.

        A node says it may with ``asks=True``. A node switched off never asks, since it is never called. Only the
        nodes that compiled are read, so it is certain only for a graph with no problems.
        """
        return any(node.asks for node in self._nodes.values() if not node.off)

    def describe(self) -> GraphDescription:
        """Returns the graph as an editor draws it.

        Every node of the record is in it, with its inputs and outputs, and whether it's called once or once per
        item. Each node lists its own problems, and the description's ``problems`` lists all of them.

        A node with a problem is still described with its inputs and outputs. An input that gets no value because
        of a problem has the source ``missing``.

        Only a node that couldn't be compiled at all has ``found`` false, and no inputs or outputs. Its problems say
        why. The reasons are listed in ``python -m conductor.about compiling``.

        A subgraph node is one node, with its subgraph's inputs and outputs. A problem inside the subgraph is a
        problem of this graph too. Its ``loc`` starts with the subgraph node's id, followed by the place inside.
        """
        problems_of: dict[
            Loc, list[Problem]
        ] = {}  # keyed by the first part of each problem's loc, its node's id
        for graph_problem in self.problems:
            problems_of.setdefault(graph_problem.loc[:1], []).append(graph_problem)
        nodes: dict[str, GraphNodeDescription] = {}
        for node_id, node_record in sorted(self.record.nodes.items()):
            own = problems_of.get((node_id,), [])
            compiled_node = self._nodes.get(node_id)
            if compiled_node is None:
                nodes[node_id] = GraphNodeDescription(
                    node=node_record.node, version=node_record.version, found=False, problems=own
                )
                continue
            nodes[node_id] = GraphNodeDescription(
                node=node_record.node,
                version=node_record.version,
                found=True,
                problems=own,
                per_item_of=innermost_series(compiled_node.loops),
                inputs=[
                    NodeInputDescription(
                        port=node_input.port.description,
                        source=input_source_of(compiled_node.loops, node_input),
                    )
                    for node_input in compiled_node.inputs.values()
                ],
                outputs=compiled_node.described_outputs(),
            )
        return GraphDescription(
            nodes=nodes,
            inputs=dict(self.inputs),
            outputs=dict(self.outputs),
            problems=list(self.problems),
        )

    # -- what a subgraph node that uses this graph needs ------------------------

    def _node_returning(self, output: str) -> tuple[CompiledNode, str]:
        """Returns the node that returns the graph output ``output``, and that output's name on the node.

        If the record declares the graph output ``loud`` as ``shout.result``, this returns the node ``shout`` and
        ``"result"``. The node may be a subgraph node. Then its own graph knows which node inside returns it.
        """
        source = self.record.outputs[output]
        return self._nodes[source.node], source.output

    def _edge_reading(self, output: str) -> Edge:
        """Returns the edge that reads the whole of the graph output ``output``.

        It's the ``edge_reading`` of the node behind the graph output, so its ``node`` names the function node that
        returns the value, from this graph.
        """
        node, name = self._node_returning(output)
        return node.edge_reading(name)

    # -- what a run uses --------------------------------------------------------

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of the graph's shape.

        It covers the graph record, without each node's ``display`` (which conductor never reads), and the
        fingerprint of every subgraph used in it. Keys are sorted first. So a record that comes back from a JSON
        store with its keys in another order has the same fingerprint.

        A saved run stores the fingerprint. Loading a saved run into a graph with a different one is refused.

        The code of the nodes' functions isn't part of it. A run checks that per call instead, and reuses an
        earlier call's outcome only when the node's code is the same.
        """
        own = self.record.model_dump(mode="json", exclude={"nodes": {"__all__": {"display"}}})
        subgraph_fingerprints = {
            p.id: p.inner.fingerprint
            for p in self._nodes.values()
            if isinstance(p, CompiledSubgraphNode)
        }
        return json_hash([own, subgraph_fingerprints], 8)

    @functools.cached_property
    def _context_validator(self) -> ContextValidator:
        """The validator for the ``context=`` passed when a run starts or resumes.

        It's built once, from the registry's ``context_type`` and every function node that takes a context,
        including those inside subgraphs.
        """
        contexts = {
            path: node.compiled_node.definition.context
            for path, node in self._blueprint.function_nodes.items()
        }
        return ContextValidator(
            self._context_type,
            [(path, context) for path, context in contexts.items() if context is not None],
        )

    @functools.cached_property
    def _inputs_validator(self) -> InputsValidator:
        """The validator for the graph inputs a caller gives. Built when the first run is made."""
        return InputsValidator(self._inputs)

    @functools.cached_property
    def _blueprint(self) -> RunBlueprint:
        """The run blueprint every run of this graph works from. Built and kept when the first run is made or read.

        ``graph.with_inputs`` makes a run, and ``Run.from_json`` reads one. If two threads do that at once, each may
        build a blueprint. They're equal, and one is kept.
        """
        from conductor._internal.run import RunBlueprint

        return RunBlueprint.of(self)

    @functools.cached_property
    def _codec(self) -> type[SavedRun]:
        """The model that converts this graph's runs to and from JSON.

        Built the first time a run is saved or loaded, and kept.
        """
        from conductor._internal.run import codec_of

        return codec_of(self)

    # -- what a host calls to run it ---------------------------------------------

    def answers_as_json(self, answers: Mapping[str, Any]) -> dict[str, Any]:
        """Turns the answers a host queues for a run of this graph into the JSON data the run will be given.

        A server can't give a running run an answer, so it queues it as JSON until the run stops. Each answer
        becomes the JSON data ``run.with_answers`` reads it from: a ``date`` its ISO text, a dataclass or a model a
        mapping of its fields, and a ``polars.DataFrame`` its columns. The node a waiting id names is looked up in
        the graph, whether its call waits yet or not, and ``SKIPPED`` is read as the run reads it: a branch not
        chosen is left out, and ``SKIPPED`` on an output that isn't a branch is refused at its question. So a
        queued answer reads back as the answer itself. What JSON can't write is refused as a whole, since the
        call's question isn't read here (``run.with_answers`` refuses the same value at its question). What depends
        on the question, such as an answer of the wrong type, is checked when the run is given the data.

        It is provisional: the name and shape may change before 2.0.

        Args:
            answers: Answers by waiting id, as ``run.with_answers`` takes them.

        Returns:
            The answers by waiting id, as the run is to be given them. For example ``{"approve~3fa2c0d1":
            "2026-11-15"}`` for an answer of ``date(2026, 11, 15)``.

        Raises:
            InvalidCallError: The graph's own problems, if it has any, before any answer is read. Then
                ``invalid_argument`` if a waiting id isn't text, with every such key in ``ctx.errors``. Else every
                refusal at once, each an ``invalid_answer`` at the call its waiting id names: for ``SKIPPED`` on an
                output that isn't a branch, and for an answer JSON can't write, such as a ``str`` with a lone
                surrogate. An id that names no node of the graph has its answer written as JSON as it is, and the
                run refuses it as ``not_waiting``.
        """
        if self.problems:
            raise InvalidCallError(self.problems)
        # run/ imports graph/, so it is imported here
        from conductor._internal.run import WaitingIdValidator

        parts = WaitingIdValidator(
            function="graph.answers_as_json", argument="answers", keyed=True
        )(answers)
        problems: list[Problem] = []
        queued: dict[str, Any] = {}
        for waiting_id, answer in answers.items():
            loc, node_path, _, _ = parts[waiting_id]
            # None when the id names a subgraph, a node the graph doesn't have, or no node at all
            node = self._blueprint.function_nodes.get(node_path)
            try:
                queued[waiting_id] = AnswerAsJsonValidator(
                    node.compiled_node.outputs if node is not None else None,
                    loc=loc,
                    waiting_id=waiting_id,
                )(answer)
            except InvalidCallError as refused:
                problems.extend(refused.problems)
        if problems:
            raise InvalidCallError(problems)
        return queued

    def with_inputs(
        self,
        inputs: Mapping[str, Any] = MappingProxyType({}),
        *,
        max_items: int | None = None,
        previous: Run[Any] | None = None,
    ) -> Run[C]:
        """Returns a run with the caller's inputs checked, but not started. ``run.resume()`` starts it.

        Args:
            inputs: The graph's inputs by name, as one mapping (see ``graph.inputs``).
            max_items: The most items any one ``Series`` that a node loops over may have. Past that, the run stops.
            previous: An earlier run of this graph, or of an older version of it, to reuse work from, as
                ``run.again`` does. A node isn't called again where it gets the same arguments it got there, as
                long as its id, version, code, ports and loops are the same. So after an edit, only the nodes the
                edit changed, and what they feed, run again.

        Returns:
            The run, with the status ``ready``.

        Raises:
            InvalidCallError: The graph has problems. Or ``max_items`` isn't a whole number of at least 1
                (``invalid_limit``). Or an input is missing, unknown or not of its type. Every fault in the inputs
                is listed.
        """
        from conductor._internal.run import make_run

        if self.problems:
            raise InvalidCallError(self.problems)
        limits = LimitsValidator()({"max_items": max_items})
        return make_run(self, self._inputs_validator({**inputs}), limits["max_items"], previous)

    def run(
        self,
        inputs: Mapping[str, Any] = MappingProxyType({}),
        *,
        max_items: int | None = None,
        previous: Run[Any] | None = None,
        context: C | None = None,
        timeout: float | None = None,
    ) -> Run[C]:
        """Runs the graph with these inputs until it's done, waits for a person, fails or stops.

        It's the same as ``graph.with_inputs(inputs, max_items=max_items, previous=previous).resume(context=context,
        timeout=timeout)``.
        """
        return self.with_inputs(inputs, max_items=max_items, previous=previous).resume(
            context=context, timeout=timeout
        )
