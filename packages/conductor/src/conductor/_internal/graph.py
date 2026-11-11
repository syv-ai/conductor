"""``Graph``: the checked graph a host holds, with the verbs that start a run."""

from __future__ import annotations

import functools
import types
from collections.abc import Mapping
from typing import Any

from conductor._internal import problem
from conductor._internal.address import NodePath
from conductor._internal.fields import REQUIRED, FieldValueError
from conductor._internal.loaded_graph import (
    CheckedGraph,
    FunctionAtPath,
    NodeAtPath,
    PlacedFunction,
    nodes_at_paths,
)
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.run import Run, checked_leg, no_units, run_from_json
from conductor._internal.saved_run import SaveForm, save_form


class Graph(CheckedGraph):
    """A graph record checked against a registry's nodes, which can start a run: ``registry.load(record)`` makes one.

    ``problems`` is everything wrong with it, and a graph with none runs: ``graph.run(**inputs)``.

    Attributes:
        record: The stored graph it was loaded from.
        problems: Everything wrong with it; empty for a graph that runs.
        inputs: The open inputs of its first nodes, by name (``node.input`` where two share a name): what a
            caller fills.
    """

    @functools.cached_property
    def _at_paths(self) -> tuple[list[FunctionAtPath], dict[str, NodeAtPath]]:
        """The graph's nodes and those of each graph embedded in it, where they stand (``nodes_at_paths``),
        made when a run of the graph is first made."""
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

    def with_inputs(
        self, inputs: Mapping[str, Any] = types.MappingProxyType({}), /, *, max_items: int | None = None, earlier: Run | None = None, **by_name: Any,
    ) -> Run:
        """A run with the caller's inputs checked, not started; ``run.resume()`` starts it.

        Args:
            inputs: The inputs by name (``graph.inputs``) or by address (``node.input``), for names Python can't
                take as keywords (``{"a.text": ...}``).
            max_items: The most items any one Series a node works through may hold before the run stops.
            earlier: A run of this graph, or of an earlier version of it, that lends its work as ``run.again``
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
        checked_leg(max_items=max_items)
        checked = self._checked_inputs(*self._given_by_address({**inputs, **by_name}))
        return Run(self, checked, max_items, no_units(self), earlier._carried(into=self) if earlier is not None else {})

    def run(
        self, inputs: Mapping[str, Any] = types.MappingProxyType({}), /, *, max_items: int | None = None,
        from_run: Mapping[type, Any] = types.MappingProxyType({}), timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops."""
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(from_run=from_run, timeout=timeout)

    def load_run(self, saved: str | bytes | Mapping[str, Any]) -> Run:
        """A run of this graph from what ``run.to_json()`` wrote; a graph with problems refuses it with them.

        A save is read from its JSON text, a mapping written as JSON first, so each value is read as its type
        reads JSON.
        """
        return run_from_json(self, saved)

    @functools.cached_property
    def _save_form(self) -> SaveForm:
        """How this graph's runs are read from JSON and written as it, made when the first one is."""
        return save_form(
            self.fingerprint,
            inputs={address: offered.field.whole for address, offered in self._input_fields.items()},
            outputs={node.name: node.placed.outputs for node in self._tree},
            nodes=[node.name for node in self._tree],
        )

    def _given_by_address(self, given: Mapping[str, Any]) -> tuple[dict[str, Any], list[Problem]]:
        """What a caller gave by name or by address, by address; with a problem for each name the graph doesn't
        offer and each input given by both its name and its address."""
        fields = {**{f.address: f for f in self._input_fields.values()}, **{f.name: f for f in self._input_fields.values()}}
        by_address: dict[str, Any] = {}
        problems: list[Problem] = []
        for name, value in given.items():
            offered = fields.get(name)
            if offered is None:
                problems.append(problem.not_offered("inputs", name, name=name, inputs=list(self.inputs)))
            elif offered.address in by_address:
                problems.append(problem.input_given_twice("inputs", offered.name, name=offered.name, address=offered.address))
            else:
                by_address[offered.address] = value
        return by_address, problems

    def _checked_inputs(self, given: Mapping[str, Any], problems: list[Problem]) -> dict[str, Any]:
        """The value of every graph input, by address: the caller's, read by the input's field, else its default.

        Raises:
            InvalidCallError: With ``problems`` and then every input that is missing or not of its type.
        """
        problems, values = list(problems), dict[str, Any]()
        for address, offered in self._input_fields.items():
            if address not in given:
                if offered.field.default is REQUIRED:
                    problems.append(problem.missing_input("inputs", offered.name, name=offered.name))
                values[address] = offered.field.default
                continue
            value, spec = given[address], offered.field
            try:
                values[address] = spec.whole.read(value)
            except FieldValueError as refused:
                whole = all(not error["loc"] for error in refused.errors or ())  # the list itself was refused, not an item of it
                if not spec.many and type(value) is list and whole:  # the one mistake whose fix is in the graph
                    problems.append(problem.invalid_input_make_a_series("inputs", offered.name, name=offered.name, expected=spec.whole.id, length=len(value)))
                else:
                    problems.append(problem.invalid_input("inputs", offered.name, name=offered.name, reason=refused.reason, errors=refused.errors, column=refused.column))
        if problems:
            raise InvalidCallError(problems)
        return values

