"""What a host is shown of a run's outputs: ``run.results`` and ``run.outputs``.

Take the README's graph after ``graph.run(text="a\\nb")``. ``run.results["shout"]["result"]`` is
``["A!", "B!"]``, and ``run.outputs`` is ``{"result": "A!+B!"}``.

A host is shown an output the way a node wired to it would get it. The difference is that a value that isn't
there yet is shown as ``None``, so ``run.results`` works whatever the run's status. ``DisplayReader`` does this.
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
    """Returns ``run.results``: every node's outputs so far, by node id and output name."""
    reader = DisplayReader(run._blueprint.function_nodes, run._outcomes, run.inputs)
    return {
        n: {o: reader.display_value(*compiled_node.returned_by(o)) for o in compiled_node.outputs}
        for n, compiled_node in run.graph._nodes.items()
    }


def outputs_of(run: Run[Any]) -> dict[str, Any]:
    """Returns ``run.outputs`` of a complete run: the graph outputs by name, each shown as ``run.results`` shows it."""
    reader = DisplayReader(run._blueprint.function_nodes, run._outcomes, run.inputs)
    return {
        name: reader.display_value(*node.returned_by(output))
        for name in run.graph.outputs
        for node, output in [run.graph._node_returning(name)]
    }


class DisplayReader(ResultReader):
    """Reads a run's outputs for a host, for ``run.results`` and ``run.outputs``, whatever the run's status.

    It is a ``ResultReader`` that never says "not ready". Where a ``ResultReader`` returns ``NOT_READY``, because a
    call hasn't returned or waits for a person, this reader returns ``None``. So a host is shown the values so far.
    In the example, while ``shout`` has only returned for the first line, ``results["shout"]["result"]`` is
    ``["A!", None]``.

    An edge can name one field of a record (``give.result.name``). If the record has no such field, the node the
    edge leads to fails, and the run ends ``failed`` with that as its cause. Showing the run afterwards mustn't
    raise the same error again, so here the missing field is shown as ``None``.
    """

    __slots__ = ()

    def read_output(
        self,
        node: BlueprintFunctionNode,
        output: str,
        item_indexes: ItemIndexes,
        record_field: str | None,
        *,
        one_item: bool,
    ) -> Any:
        """Returns what ``ResultReader.read_output`` returns, as a host is shown it.

        A value that isn't there yet is ``None``, and so is a field the record doesn't have. A ``Series`` output is a
        plain list of its items.
        """
        try:
            value = ResultReader.read_output(
                self, node, output, item_indexes, record_field, one_item=one_item
            )
        except NoSuchFieldError:
            return None
        if value is NotReady.NOT_READY:
            return None
        return value.as_list() if isinstance(value, StoredSeries) else value

    def _list_looped_over(
        self,
        loop: tuple[NodePath, str],
        for_any_item: Any,
        keeps_none_items: bool,
        item_indexes: ItemIndexes,
    ) -> Any:
        """Returns what ``ResultReader._list_looped_over`` returns, but ``None`` while the list hasn't been made."""
        looped_over = ResultReader._list_looped_over(
            self, loop, for_any_item, keeps_none_items, item_indexes
        )
        return None if looped_over is NotReady.NOT_READY else looped_over

    def collect(
        self, graph_path: NodePath, node_input: CollectingInput, item_indexes: ItemIndexes
    ) -> Any:
        """Returns the list a collecting input would get for the call at ``item_indexes``.

        It is ``None`` when the edge names a field that a record in the list doesn't have.
        """
        try:
            return ResultReader.collect(self, graph_path, node_input, item_indexes)
        except NoSuchFieldError:
            return None

    def display_value(
        self, node_path: NodePath, output: str, item_indexes: ItemIndexes = ()
    ) -> Any:
        """Returns one output of a function node, as ``run.results`` shows it.

        Say ``documents -> lines -> shout`` has three documents, and the second is empty, so ``lines`` gave it no
        lines. ``shout`` is called once per line of each document, and its output is shown as
        ``[["A", "B"], None, ["C"]]``.

        Args:
            node_path: The path of the function node that returns the output, from the outer graph.
            output: The output's name.
            item_indexes: The item index in each loop of the node already chosen. Empty for the whole output.

        Returns:
            For a node called once, what it returned. For a node called per item, a list with one entry per item, and
            ``None`` for an item with no value. For a node in a loop inside another loop, a list of such lists.
            ``None`` in place of a list when the list being looped over was never made, or when a skipped branch
            outside the loop means no item could have a value.
        """
        node, depth = function_node_in_run(self.function_nodes, (), node_path), len(item_indexes)
        if depth == len(node.loops):  # one call, so show what it returned
            value = self.read_output(node, output, item_indexes, None, one_item=False)
            return None if value is SKIPPED else value
        loop = node.loops[depth]
        for_any_item = self._for_any_item_of(loop, [node], item_indexes).read_output(
            node, output, item_indexes, None, one_item=False
        )
        looped_over = self._list_looped_over(
            loop, for_any_item, True, item_indexes
        )  # a host is shown None for an item with no value
        if looped_over is None or looped_over is SKIPPED:
            return None
        return [
            self.display_value(node_path, output, (*item_indexes, index))
            for index in range(len(looped_over))
        ]
