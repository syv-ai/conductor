"""``Graph``: the compiled graph a host holds, with the calls that make a run of it and how its runs are saved."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self

from conductor._internal.compiled import CompiledGraph, compile_graph
from conductor._internal.graph_record import GraphRecord, as_record
from conductor._internal.markers import REQUIRED
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.run import Run, check_run_limits, no_outcomes
from conductor._internal.run.json import RunCodec, run_codec
from conductor._internal.value_type import FieldValueError

if TYPE_CHECKING:
    from conductor._internal.registry import NodeRegistry


class Graph(CompiledGraph):
    """A graph record checked against a registry's nodes, which can start a run: ``Graph.compile(record, registry)`` makes one.

    ``problems`` is everything wrong with it, and a graph with none runs: ``graph.run(**inputs)``.

    Attributes:
        record: The graph record it was compiled from.
        problems: Everything wrong with it; empty for a graph that runs.
        inputs: The open inputs of its first nodes, by name (``node.input`` where two share a name): what a
            caller fills.
    """

    @classmethod
    def compile(cls, record: GraphRecord | str | bytes, registry: NodeRegistry) -> Self:
        """Compile a graph record, given as a ``GraphRecord`` or as its JSON text, against the node definitions of ``registry``.

        Compiling never raises for a wrong graph, a half-finished one included: what is wrong is in
        ``graph.problems``, and a graph with problems does not run. The graph keeps its own copy of the
        definitions as they are when it is compiled.
        """
        return compile_graph(as_record(record), registry._nodes, cls)

    def with_inputs(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, previous: Run | None = None, **by_name: Any,
    ) -> Run:
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
            InvalidCallError: The graph has problems, or an input is missing, unknown or not of its type.
        """
        if self.problems:
            raise InvalidCallError(self.problems)
        check_run_limits(max_items=max_items)
        checked = self._checked_inputs(*self._given_by_address({**inputs, **by_name}))
        return Run(self, checked, max_items, no_outcomes(self), previous._reusable_outcomes(into=self) if previous is not None else {})

    def run(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None,
        context: Mapping[type, Any] = MappingProxyType({}), timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops."""
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(context=context, timeout=timeout)

    @functools.cached_property
    def _run_codec(self) -> RunCodec:
        """How this graph's runs are read from JSON and written as it, made when the first one is."""
        return run_codec(
            self.fingerprint,
            inputs={address: graph_input.field.whole for address, graph_input in self._input_fields.items()},
            outputs={node.name: node.compiled_node.outputs for node in self._flat_function_nodes},
            nodes=[node.name for node in self._flat_function_nodes],
        )

    def _given_by_address(self, given: Mapping[str, Any]) -> tuple[dict[str, Any], list[Problem]]:
        """What a caller gave by name or by address, by address; with a problem for each name the graph doesn't
        offer and each input given by both its name and its address."""
        fields = {**{f.address: f for f in self._input_fields.values()}, **{f.name: f for f in self._input_fields.values()}}
        by_address: dict[str, Any] = {}
        problems: list[Problem] = []
        for name, value in given.items():
            graph_input = fields.get(name)
            if graph_input is None:
                problems.append(problem.not_offered("inputs", name, name=name, inputs=list(self.inputs)))
            elif graph_input.address in by_address:
                problems.append(problem.input_given_twice("inputs", graph_input.name, name=graph_input.name, address=graph_input.address))
            else:
                by_address[graph_input.address] = value
        return by_address, problems

    def _checked_inputs(self, given: Mapping[str, Any], problems: list[Problem]) -> dict[str, Any]:
        """The value of every graph input, by address: the caller's, read by the input's field, else its default.

        Raises:
            InvalidCallError: With ``problems`` and then every input that is missing or not of its type.
        """
        problems, values = list(problems), dict[str, Any]()
        for address, graph_input in self._input_fields.items():
            if address not in given:
                if graph_input.field.default is REQUIRED:
                    problems.append(problem.missing_input("inputs", graph_input.name, name=graph_input.name))
                values[address] = graph_input.field.default
                continue
            value, field = given[address], graph_input.field
            try:
                values[address] = field.whole.read(value)
            except FieldValueError as refused:
                whole = all(not error["loc"] for error in refused.errors or ())  # the list itself was refused, not an item of it
                if not field.is_series and type(value) is list and whole:  # the one mistake whose fix is in the graph
                    problems.append(problem.invalid_input_make_a_series("inputs", graph_input.name, name=graph_input.name, expected=field.whole.id, length=len(value)))
                else:
                    problems.append(problem.invalid_input("inputs", graph_input.name, name=graph_input.name, reason=refused.reason, errors=refused.errors, column=refused.column))
        if problems:
            raise InvalidCallError(problems)
        return values

