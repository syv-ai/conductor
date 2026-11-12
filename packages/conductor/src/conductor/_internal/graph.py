"""``Graph``: the compiled graph a host holds, with the calls that make a run of it and how its runs are saved."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from conductor._internal.compiled import CompiledGraph, compile_graph
from conductor._internal.gate import (
    ContextDoor,
    InputsDoor,
    RefusedError,
    check_limits,
    read_record,
)
from conductor._internal.graph_record import GraphRecord
from conductor._internal.problem import InvalidCallError
from conductor._internal.run import Run, no_outcomes
from conductor._internal.run.json import RunCodec, run_codec

if TYPE_CHECKING:
    from conductor._internal.registry import NodeRegistry


class Graph[C](CompiledGraph):
    """A graph record checked against a registry's nodes, which can start a run: ``Graph.compile(record, registry)`` makes one.

    ``problems`` is everything wrong with it, and a graph with none runs: ``graph.run(**inputs)``. ``C`` is the class
    of the run's context the registry names (``NodeRegistry(context_type=HostContext)`` compiles a
    ``Graph[HostContext]``), and ``None`` for a registry that names none.

    Attributes:
        record: The graph record it was compiled from.
        problems: Everything wrong with it; empty for a graph that runs.
        inputs: The open inputs of its first nodes, by name (``node.input`` where two share a name): what a
            caller fills.
    """

    @staticmethod
    def compile[R](record: GraphRecord | str | bytes, registry: NodeRegistry[R]) -> Graph[R]:
        """Compile a graph record, given as a ``GraphRecord`` or as its JSON text, against the node definitions of ``registry``.

        Compiling never raises for a wrong graph, a half-finished one included: what is wrong is in
        ``graph.problems``, and a graph with problems does not run. The graph keeps its own copy of the
        definitions as they are when it is compiled, and the registry's ``context_type``.
        """
        try:
            read = read_record(record)
        except RefusedError as refused:
            raise InvalidCallError(refused.problems) from None
        return compile_graph(read, registry._nodes, registry.context_type, Graph)

    def with_inputs(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, previous: Run[Any] | None = None, **by_name: Any,
    ) -> Run[C]:
        """A run with the caller's inputs checked, not started; ``run.resume()`` starts it.

        Args:
            inputs: The inputs by name (``graph.inputs``) or by address (``node.input``), for names Python can't
                take as keywords (``{"a.text": ...}``).
            max_items: The most items any one Series a node works through may hold before the run stops.
            previous: A run of this graph, or of an earlier version of it, that lends its work as ``run.again``
                does: a node with the same id, version, code, fields and scope is not called where it receives
                what it received there, so after an edit only the nodes it changes, and what they feed, run again.
            **by_name: The inputs, as keywords.

        Returns:
            The run, ``ready``.

        Raises:
            InvalidCallError: The graph has problems; ``max_items`` is not a whole number of at least 1
                (``invalid_max_items``); or an input is missing, unknown, given twice or not of its type (the inputs
                door's problems, every one).
        """
        if self.problems:
            raise InvalidCallError(self.problems)
        try:
            check_limits(max_items=max_items)
            values = self._inputs_door({**inputs, **by_name})
        except RefusedError as refused:
            raise InvalidCallError(refused.problems) from None
        return Run(self, values, max_items, no_outcomes(self), previous._reusable_outcomes(into=self) if previous is not None else {})

    def run(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None,
        context: C | None = None, timeout: float | None = None, **by_name: Any,
    ) -> Run[C]:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops:
        ``graph.with_inputs(...).resume(context=context, timeout=timeout)``."""
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(context=context, timeout=timeout)

    @functools.cached_property
    def _context_door(self) -> ContextDoor:
        """Checks the ``context=`` of each call that starts or resumes a run: built once, from the registry's
        ``context_type`` and every node that takes the context."""
        takers = [(node.node_path, context) for node in self._flat_function_nodes if (context := node.compiled_node.definition.context) is not None]
        return ContextDoor(self._context_type, takers)

    def _checked_context(self, context: C | None) -> C | None:
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
        return InputsDoor({address: (graph_input.name, graph_input.field) for address, graph_input in self._input_fields.items()})

    @functools.cached_property
    def _run_codec(self) -> RunCodec:
        """How this graph's runs are read from JSON and written as it, made when the first one is."""
        return run_codec(
            self.fingerprint,
            inputs={address: graph_input.field.whole for address, graph_input in self._input_fields.items()},
            outputs={node.name: node.compiled_node.outputs for node in self._flat_function_nodes},
            asked={node.name: asked.field for node in self._flat_function_nodes if (asked := node.compiled_node.definition.asked) is not None},
            nodes=[node.name for node in self._flat_function_nodes],
        )
