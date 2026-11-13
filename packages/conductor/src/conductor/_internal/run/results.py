"""What a host is shown of a run's outputs: ``run.results`` and ``run.outputs``, read whatever the run's status.

For the README graph after ``graph.run(text="a\\nb")``, ``run.results["shout"]["result"]`` is ``["A!", "B!"]`` and
``run.outputs`` is ``{"result": "A!+B!"}``. A host reads an output the way a node wired to it would, with the
inputs ``display_inputs_of`` gives (``flat.py``), by a ``DisplayReader``, which reads a value that is not
there yet as ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from conductor._internal.base import SKIPPED, ItemIndexes
from conductor._internal.compiled import CompiledGraph, Edge
from conductor._internal.series import StoredSeries
from conductor._internal.types import NoSuchFieldError

from .arguments import ResultReader
from .flat import (
    CollectingArgument,
    EdgeArgument,
    FlatGraph,
    display_inputs_of,
)
from .outcome import NotReady, Outcome


def results_of(
    graph: CompiledGraph, flat: FlatGraph, outcomes: dict[str, dict[ItemIndexes, Outcome]], graph_inputs: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """``run.results``: every node's outputs so far, by node id and output name. ``flat`` is ``graph``'s flat graph."""
    reader = DisplayReader(outcomes, graph_inputs)
    return {n: {o: reader.display_value(display_inputs_of(*flat.nodes[n].source_of(o))) for o in compiled_node.outputs} for n, compiled_node in graph._nodes.items()}


def outputs_of(
    graph: CompiledGraph, flat: FlatGraph, outcomes: dict[str, dict[ItemIndexes, Outcome]], graph_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """``run.outputs`` of a complete run: the graph outputs, by name, each read as ``run.results`` reads the output
    it is. ``flat`` is ``graph``'s flat graph."""
    reader = DisplayReader(outcomes, graph_inputs)
    return {name: reader.display_value(display_inputs_of(*flat.nodes[edge.source[0]].source_of(edge.source[1]))) for name, edge in graph._outputs.items()}


class DisplayReader(ResultReader):
    """Reads the outputs of a run for a host (``run.results``, ``run.outputs``), whatever the run's status.

    In the example, while ``shout`` has returned for the first line only, ``results["shout"]["result"]`` is
    ``["A!", None]``. Where a ``ResultReader`` returns ``NOT_READY`` (a call that has not returned, or that
    waits for a person), this reader returns ``None``, so a host is shown the values so far.

    An edge can name one field of a record (``give.result.name``). When the record has no such field, the
    node the edge leads to fails and the run ends ``failed`` with that as its cause. Reading the run
    afterwards must not raise again, so here the missing field is read as ``None``.
    """

    __slots__ = ()

    def read_output(self, source: Edge, item_indexes: ItemIndexes) -> Any:
        """Returns what ``ResultReader.read_output`` returns, as a host is shown it: ``None`` when the node has
        not returned yet, or when the edge names a field the record does not have, and the value of a
        ``Series`` output as a plain list of its items."""
        try:
            value = ResultReader.read_output(self, source, item_indexes)
        except NoSuchFieldError:
            return None
        if value is NotReady.NOT_READY:
            return None
        return value.as_list() if isinstance(value, StoredSeries) else value

    def _list_looped_over(self, argument: CollectingArgument, item_indexes: ItemIndexes) -> Any:
        """Returns what ``ResultReader._list_looped_over`` returns, with ``None`` when the list is not made yet."""
        looped_over = ResultReader._list_looped_over(self, argument, item_indexes)
        return None if looped_over is NotReady.NOT_READY else looped_over

    def collect(self, argument: CollectingArgument, item_indexes: ItemIndexes) -> Any:
        """The list a ``Series`` input with ``argument`` would be called with for the item at ``item_indexes``;
        ``None`` when the edge names a field that a record in the list doesn't have."""
        try:
            return ResultReader.collect(self, argument, item_indexes)
        except NoSuchFieldError:
            return None

    def display_value(self, node_inputs: tuple[EdgeArgument | CollectingArgument, ...], item_indexes: ItemIndexes = ()) -> Any:
        """One output of a node as ``run.results`` shows it.

        Example: ``documents -> lines -> shout`` with three documents, the second empty, so ``lines`` was
        never called for it. ``shout`` is called once per line of each document, and its output is shown as
        ``[["A", "B"], None, ["C"]]``.

        Args:
            node_inputs: The output's inputs from ``display_inputs_of``: one for each loop the node is called in.
            item_indexes: The index of the item in each loop already entered; empty for the whole output.

        Returns:
            The return value, for a node called once. For a node called per item, a list with one entry per
            item, ``None`` for an item that has no value; for a node called in a loop inside another loop, a
            list of such lists. ``None`` instead of a list when the list being looped over was never made,
            or when no item of it could have a value because of a skipped branch or a ``None`` outside the
            loop.
        """
        argument = node_inputs[len(item_indexes)]
        if len(item_indexes) == len(node_inputs) - 1:
            value = self.read_argument(argument, item_indexes)
            return None if value is SKIPPED else value
        looped_over = self._list_looped_over(cast("CollectingArgument", argument), item_indexes)  # every input before the last lists one loop
        if looped_over is None or looped_over is SKIPPED:
            return None
        return [self.display_value(node_inputs, (*item_indexes, index)) for index in range(len(looped_over))]
