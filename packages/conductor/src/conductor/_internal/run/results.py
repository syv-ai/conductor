"""What a host is shown of a run's outputs: ``run.results`` and ``run.outputs``.

Take the README's graph after ``graph.run({"text": "a\\nb"})``. ``run.results["shout"]["result"]`` is
``["A!", "B!"]``, and ``run.outputs`` is ``{"result": "A!+B!"}``.

A host is shown an output the way a node wired to it would get it. The difference is that a value that isn't
there yet is shown as ``None``, so ``run.results`` works whatever the run's status. ``DisplayReader`` does this.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.base import SKIPPED, ItemIndexes, NodePath
from conductor._internal.graph import CollectingInput, Graph
from conductor._internal.series import StoredSeries
from conductor._internal.types import UnreadableFieldError

from .blueprint import BlueprintFunctionNode
from .outcome import NotReady
from .outcomes import MAY_HAVE_VALUE, RunOutcomes


def results_of(graph: Graph[Any], outcomes: RunOutcomes) -> dict[str, dict[str, Any]]:
    """Returns ``run.results`` of a run of ``graph`` with these outcomes: every node's outputs so far, by node id and
    output name."""
    reader = DisplayReader.sharing(outcomes)
    return {
        n: {
            o: reader.display_value(edge.node, edge.output)
            for o in compiled_node.outputs
            for edge in [compiled_node.edge_reading(o)]
        }
        for n, compiled_node in graph._nodes.items()
    }


def outputs_of(graph: Graph[Any], outcomes: RunOutcomes) -> dict[str, Any]:
    """Returns ``run.outputs`` of a complete run of ``graph`` with these outcomes: the graph outputs by name, each
    shown as ``run.results`` shows it."""
    reader = DisplayReader.sharing(outcomes)
    return {
        name: reader.display_value(edge.node, edge.output)
        for name in graph._outputs
        for edge in [graph._edge_reading(name)]
    }


class DisplayReader(RunOutcomes):
    """Reads a run's outputs for a host, for ``run.results`` and ``run.outputs``, whatever the run's status.

    It reads a run's outcomes as ``RunOutcomes`` reads them for a call, but its ``display_value`` never says "not
    ready". Where a call hasn't returned or waits for a person, it shows ``None``. So a host is shown the values so
    far. In the example, while ``shout`` has only returned for the first line, ``results["shout"]["result"]`` is
    ``["A!", None]``.

    An edge can name one field of a value (``give.result.name``). If the value has no such field, or the field is
    ``SKIPPED``, the node the edge leads to fails, and the run ends ``failed`` with that as its cause. Showing the
    run afterwards mustn't raise the same error again, so here that field is shown as ``None``.
    """

    __slots__ = ()

    def read_output(
        self,
        node: BlueprintFunctionNode,
        output: str,
        item_indexes: ItemIndexes,
        field: str | None,
        *,
        one_item: bool,
    ) -> Any:
        """Returns what ``RunOutcomes.read_output`` returns, as a host is shown it.

        A field the value doesn't have, or that is ``SKIPPED``, is ``None``. A ``Series`` output is a plain list of its items. A value that
        isn't there yet stays ``NOT_READY``, so that working out whether any item of a loop may have a value goes as
        it does for a call: a question still waiting upstream is not a ``None``. ``display_value`` shows it as
        ``None``.
        """
        try:
            value = RunOutcomes.read_output(
                self, node, output, item_indexes, field, one_item=one_item
            )
        except UnreadableFieldError:
            return None
        return value.as_list() if isinstance(value, StoredSeries) else value

    def collect(self, node_input: CollectingInput, item_indexes: ItemIndexes) -> Any:
        """Returns the list a collecting input would get for the call at ``item_indexes``.

        It is ``None`` when the edge names a field that an item of the list doesn't have, or that is ``SKIPPED``.
        """
        try:
            return RunOutcomes.collect(self, node_input, item_indexes)
        except UnreadableFieldError:
            return None

    def display_value(
        self, node_path: NodePath, output: str, item_indexes: ItemIndexes = ()
    ) -> Any:
        """Returns one output of a function node, as ``run.results`` shows it.

        Say ``documents -> lines -> shout`` has three documents, and the second is ``None``, so ``lines`` isn't
        called for it. ``shout`` is called once per line of each document, and its output is shown as
        ``[["A", "B"], None, ["C"]]``. Had the second document been empty, it would show ``[]`` there.

        Args:
            node_path: The path of the function node that returns the output, from the outer graph (the ``node`` of
                its ``edge_reading``).
            output: The output's name.
            item_indexes: The item index in each loop of the node already chosen. Empty for the whole output.

        Returns:
            For a node called once, what it returned. For a node called per item, a list with one entry per item, and
            ``None`` for an item with no value. For a node in a loop inside another loop, a list of such lists.
            ``None`` in place of a list when the list being looped over was never made, or when a skipped branch
            outside the loop means no item could have a value.
        """
        node, depth = self.blueprint.function_nodes[node_path], len(item_indexes)
        if depth == len(node.loops):  # one call, so show what it returned
            value = self.read_output(node, output, item_indexes, None, one_item=False)
            return None if value is SKIPPED or value is NotReady.NOT_READY else value
        loop = node.loops[depth]
        for_any_item = self._for_any_item_of(loop, [node], item_indexes).read_output(
            node, output, item_indexes, None, one_item=False
        )
        # Not known yet, as while a question upstream waits: the list made so far is shown, one None per item.
        if for_any_item is NotReady.NOT_READY:
            for_any_item = MAY_HAVE_VALUE
        # a host is shown None for an item with no value
        looped_over = self._list_looped_over(loop, for_any_item, True, item_indexes)
        if looped_over is None or looped_over is SKIPPED or looped_over is NotReady.NOT_READY:
            return None
        return [
            self.display_value(node_path, output, (*item_indexes, index))
            for index in range(len(looped_over))
        ]
