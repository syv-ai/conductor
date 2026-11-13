"""``Graph``: the compiled graph a host uses, with the calls that make a run of it and how its runs are saved."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from conductor._internal.compiled import CompiledGraph, compile_graph
from conductor._internal.gate import RefusedError, check_limits
from conductor._internal.gate import read_json as read_json_text
from conductor._internal.problem import InvalidCallError
from conductor._internal.record import GraphRecord
from conductor._internal.registry import NodeRegistry
from conductor._internal.run import FlatGraph, Run, SavedRun, codec_of, make_run


def read_json(text: str | bytes) -> Any:
    """Returns JSON text as plain data, the way conductor reads a graph record's text: with every object's keys seen,
    so a key written twice is refused where a ``dict`` would keep the last. A host's route that takes a body holding
    a graph record reads the body with it, ``body = read_json(await request.body())``, and hands the record on to
    ``Graph.compile`` as data.

    Raises:
        InvalidCallError: The text is not JSON, or writes a key twice in one object: ``invalid_record``.
    """
    try:
        return read_json_text(text)
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None


class Graph[C](CompiledGraph):
    """A graph record checked against a registry's nodes, which can start a run: ``Graph.compile(record, registry)`` makes one.

    ``problems`` is everything wrong with it, and a graph with none runs: ``graph.run(**inputs)``. ``C`` is the class
    of the run's context the registry names (``NodeRegistry(context_type=HostContext)`` compiles a
    ``Graph[HostContext]``), and ``None`` for a registry that names none.

    Attributes:
        record: The graph record it was compiled from.
        problems: Everything wrong with it; empty for a graph that runs.
        inputs: The graph inputs its record declares, by name: what a caller gives.
    """

    @staticmethod
    def compile[R](record: GraphRecord | str | bytes | Mapping[str, Any], registry: NodeRegistry[R]) -> Graph[R]:
        """Compile a graph record, given as a ``GraphRecord``, its JSON text, or the plain data ``read_json`` reads,
        against the node definitions of ``registry``.

        Compiling never raises for a wrong graph, a half-finished one included: what is wrong is in
        ``graph.problems``, and a graph with problems does not run. The graph keeps its own copy of the
        definitions as they are when it is compiled, and the registry's ``context_type``.
        """
        try:
            return compile_graph(record, registry, registry.context_type, Graph)
        except RefusedError as refused:
            raise InvalidCallError(refused.problems) from None

    @functools.cached_property
    def _flat(self) -> FlatGraph:
        """The flat graph every run of this graph works on: built on the first run and kept. A first run on two threads
        at once may build two equal flat graphs; one is kept."""
        return FlatGraph.of(self)

    @functools.cached_property
    def _codec(self) -> type[SavedRun]:
        """The model this graph's runs are read from JSON and written as JSON by: built the first time a run is
        saved or read, and kept."""
        return codec_of(self, self._flat)

    def with_inputs(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, previous: Run[Any] | None = None, **by_name: Any,
    ) -> Run[C]:
        """A run with the caller's inputs checked, not started; ``run.resume()`` starts it.

        Args:
            inputs: The inputs by name (``graph.inputs``), for a name Python can't take as a keyword
                (``{"class": ...}``).
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
        return make_run(self, values, max_items, previous)

    def run(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None,
        context: C | None = None, timeout: float | None = None, **by_name: Any,
    ) -> Run[C]:
        """Run the graph with these inputs until it is done, waits for a person, fails or stops:
        ``graph.with_inputs(...).resume(context=context, timeout=timeout)``."""
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(context=context, timeout=timeout)
