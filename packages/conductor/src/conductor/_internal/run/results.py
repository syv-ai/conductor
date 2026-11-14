"""What a host is shown of a run's outputs: ``run.results`` and ``run.outputs``, read whatever the run's status.

For the README graph after ``graph.run(text="a\\nb")``, ``run.results["shout"]["result"]`` is ``["A!", "B!"]`` and
``run.outputs`` is ``{"result": "A!+B!"}``. A host reads an output the way a node wired to it would, by a
``DisplayReader``, which reads a value that is not there yet as ``None``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from conductor._internal.base import SKIPPED, ItemIndexes, NodePath
from conductor._internal.graph import CollectingInput
from conductor._internal.series import StoredSeries
from conductor._internal.types import NoSuchFieldError

from .arguments import ResultReader
from .blueprint import BlueprintFunctionNode, function_node_in_run
from .outcome import NotReady

if TYPE_CHECKING:
    from .type import Run


def results_of(run: Run[Any]) -> dict[str, dict[str, Any]]:
    """``run.results``: every node's outputs so far, by node id and output name."""
    reader = DisplayReader(run._blueprint.function_nodes, run._outcomes, run.inputs)
    return {n: {o: reader.display_value(*compiled_node.returned_by(o)) for o in compiled_node.outputs} for n, compiled_node in run.graph._nodes.items()}


def outputs_of(run: Run[Any]) -> dict[str, Any]:
    """``run.outputs`` of a complete run: the graph outputs, by name, each read as ``run.results`` reads the output
    it is."""
    reader = DisplayReader(run._blueprint.function_nodes, run._outcomes, run.inputs)
    return {name: reader.display_value(*node.returned_by(output)) for name in run.graph.outputs for node, output in [run.graph._node_returning(name)]}


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

    def read_output(
        self, node: BlueprintFunctionNode, output: str, item_indexes: ItemIndexes, record_field: str | None = None, *, one_item: bool = False,
    ) -> Any:
        """Returns what ``ResultReader.read_output`` returns, as a host is shown it: ``None`` when the node has
        not returned yet, or when the edge names a field the record does not have, and the value of a
        ``Series`` output as a plain list of its items."""
        try:
            value = ResultReader.read_output(self, node, output, item_indexes, record_field, one_item=one_item)
        except NoSuchFieldError:
            return None
        if value is NotReady.NOT_READY:
            return None
        return value.as_list() if isinstance(value, StoredSeries) else value

    def _list_looped_over(self, graph_path: NodePath, node_input: CollectingInput, item_indexes: ItemIndexes) -> Any:
        """Returns what ``ResultReader._list_looped_over`` returns, with ``None`` when the list is not made yet."""
        looped_over = ResultReader._list_looped_over(self, graph_path, node_input, item_indexes)
        return None if looped_over is NotReady.NOT_READY else looped_over

    def collect(self, graph_path: NodePath, node_input: CollectingInput, item_indexes: ItemIndexes) -> Any:
        """The list a ``Series`` input with ``node_input`` would be called with for the item at ``item_indexes``;
        ``None`` when the edge names a field that a record in the list doesn't have."""
        try:
            return ResultReader.collect(self, graph_path, node_input, item_indexes)
        except NoSuchFieldError:
            return None

    def display_value(self, node_path: NodePath, output: str, item_indexes: ItemIndexes = ()) -> Any:
        """One output of a function node as ``run.results`` shows it.

        Example: ``documents -> lines -> shout`` with three documents, the second empty, so ``lines`` was
        never called for it. ``shout`` is called once per line of each document, and its output is shown as
        ``[["A", "B"], None, ["C"]]``.

        Args:
            node_path: The path of the function node that returns the output, from the graph a run is of.
            output: The output's name.
            item_indexes: The index of the item in each loop of the node already entered; empty for the whole output.

        Returns:
            The return value, for a node called once. For a node called per item, a list with one entry per
            item, ``None`` for an item that has no value; for a node called in a loop inside another loop, a
            list of such lists. ``None`` instead of a list when the list being looped over was never made,
            or when no item of it could have a value because of a skipped branch outside the loop.
        """
        node, depth = function_node_in_run(self.function_nodes, (), node_path), len(item_indexes)
        if depth == len(node.loops):  # one call: its return value
            value = self.read_output(node, output, item_indexes)
            return None if value is SKIPPED else value
        if self._for_any_item_of(node.loops[depth], [node], item_indexes).read_output(node, output, item_indexes) is SKIPPED:  # asked for any item of the loop
            return None
        series_node, series_output = node.loops[depth]
        outcome = self.outcomes[series_node].get(item_indexes)
        looped_over = outcome.outputs[series_output] if outcome is not None and outcome.has_outputs else None
        if looped_over is None or looped_over is SKIPPED:
            return None
        return [self.display_value(node_path, output, (*item_indexes, index)) for index in range(len(looped_over))]
