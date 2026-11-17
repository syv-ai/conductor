"""Works out the arguments of one call of a node, from the outcomes a run has so far.

The example graph in this file::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b"]
    shout(line: str, mark: str) -> str       # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!"]

``shout`` is called once per line. Its call for the second line has the item indexes ``(1,)``, one index per
loop the node is inside. ``ResultReader.arguments_for(shout, (1,))`` returns its arguments,
``{"line": "b", "mark": "!"}``.

Sometimes a node can't be called, and ``arguments_for`` says why instead. It checks these in order:

1. An input gets only ``SKIPPED``, because every edge into it comes from a skipped branch. The call is
   ``Skipped``, whatever the other inputs get.
2. A value an input needs doesn't exist yet. The call is ``NOT_READY``, and the run tries it again later.
3. An edge reads a field of a record (``lines.result.name``) that the record doesn't have. The call fails with
   ``invalid_edge_value``.
4. An input gets ``None``, and its parameter doesn't take ``None``. The call is a ``NoneInput``, and the
   function isn't called.

``join.texts`` is a collecting input: it gets a list with one value per item of a loop. Before it looks at the
items, it asks whether any item could have a value at all. Say ``mark`` comes from a skipped branch.
Then every call of ``shout`` is skipped. With two lines, ``join`` would get ``[None, None]``, and with no lines
``[]``. Instead, ``join`` is skipped either way, so the answer doesn't depend on how many lines there are.

To ask that question, the reader calls ``arguments_for`` again for ``shout``, with item indexes one index short:
``()`` instead of ``(0,)`` or ``(1,)``. ``mark`` is read as usual, since its node is inside no loop. ``line`` is
one item of a list, and no index says which one. So once the list exists, ``line`` reads as
``MAY_HAVE_VALUE``. The result is ``Skipped`` when ``mark`` is ``SKIPPED``, and arguments when some call of
``shout`` may be made. The rule is written out in ``python -m conductor.about one-or-many``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, assert_never

import polars as pl
from typing_extensions import sentinel

from conductor._internal.base import SKIPPED, ItemIndexes, NodePath
from conductor._internal.graph import CollectingInput, Edge, EdgeInput
from conductor._internal.problem import InvalidCallError, run_problems
from conductor._internal.types import NoSuchFieldError, field_of

from .blueprint import BlueprintFunctionNode, loop_body_of
from .outcome import NoneInput, NotReady, Outcome, Skipped

#: Returned in place of a value when the reader is asked about any item of a loop, and some item may have a
#: value. When no item can have one, the reader returns what every item would get: ``SKIPPED`` or ``None``.
MAY_HAVE_VALUE = sentinel("MAY_HAVE_VALUE")


class ResultReader:
    """Works out the arguments of a call from a run's outcomes.

    The run makes one reader for its outcomes, and the scheduler asks it for each call's arguments with
    ``arguments_for``.

    Attributes:
        function_nodes: The blueprint's function nodes, by node path.
        outcomes: The run's outcomes, by node path and then by item indexes.
        graph_inputs: The values the caller gave for the graph inputs, by name (``"text"``).
        for_any_item: Empty in the reader a run uses for its calls. A reader made to ask "could any item of this
            loop have a value?" (``_for_any_item_of``) has here what ``arguments_for`` returned for each node
            inside the loop.
    """

    __slots__ = ("for_any_item", "function_nodes", "graph_inputs", "outcomes")

    def __init__(
        self,
        function_nodes: Mapping[NodePath, BlueprintFunctionNode],
        outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
        graph_inputs: Mapping[str, Any],
    ) -> None:
        self.function_nodes, self.outcomes, self.graph_inputs = (
            function_nodes,
            outcomes,
            graph_inputs,
        )
        self.for_any_item: dict[
            BlueprintFunctionNode,
            dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY],
        ] = {}

    def arguments_for(
        self, node: BlueprintFunctionNode, item_indexes: ItemIndexes
    ) -> dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]:
        """Returns the arguments of the call of ``node`` at ``item_indexes``, or why the node isn't called.

        In the example, ``arguments_for(shout, (1,))`` returns ``{"line": "b", "mark": "!"}``.

        Every input is read before anything is decided, so the order of the parameters never changes the result. The
        checks then go in the order the module docstring lists. When two inputs fail the same check, the first in the
        signature is the one named.

        ``_for_any_item_of`` calls this with item indexes one index shorter than the node's own. The result then says
        what a call would be for any item. It is ``Skipped`` or ``NoneInput`` when every item would be, and arguments,
        some of them ``MAY_HAVE_VALUE``, when an item may be called.

        Returns:
            The arguments by parameter name. Or ``Skipped`` when an input gets ``SKIPPED``. Or ``NOT_READY`` when a
            value an input needs doesn't exist yet. Or ``NoneInput`` when an input gets ``None`` and its parameter
            doesn't take it.

        Raises:
            InvalidCallError: ``invalid_edge_value`` if an edge reads a field of a record (``lines.result.name``) and
                the record has no such field.
        """
        arguments = node.constants.copy()
        graph_inputs = self.graph_inputs
        for name, graph_input in node.graph_inputs:
            arguments[name] = graph_inputs[graph_input]
        none_input: str | None = None
        fault: InvalidCallError | None = None
        ready = True
        for name, node_input in node.edge_inputs.items():
            try:
                value = self.read_input(node_input, item_indexes)
            # Keep going: a later input may get SKIPPED, which skips the call.
            except NoSuchFieldError as missing:
                if fault is None:
                    reason = f"it has no field '{missing.field}'"
                    fault = InvalidCallError(
                        [
                            run_problems.invalid_edge_value(
                                name=name, reason=reason, record_field=missing.field
                            )
                        ]
                    )
                continue
            if value is SKIPPED:
                return Skipped(node.compiled_node.outputs_when_skipped)
            # Keep going: a later input may get SKIPPED, which skips the call.
            if value is NotReady.NOT_READY:
                ready = False
            elif value is None and none_input is None and not node_input.field.whole.optional:
                none_input = name
            arguments[name] = value
        if not ready:
            return NotReady.NOT_READY
        if fault is not None:
            raise fault
        if none_input is not None:
            return NoneInput(node.compiled_node.outputs_when_not_called, none_input)
        return self.align_lists(node, arguments) if node.lists_from_same_loop else arguments

    def read_input(self, node_input: EdgeInput | CollectingInput, item_indexes: ItemIndexes) -> Any:
        """Returns what one input of a node gets for the call at ``item_indexes``.

        ``shout.line`` gets ``"b"`` at ``(1,)``, and ``join.texts`` gets ``["A!", "B!"]`` at ``()``.
        """
        match node_input:
            case EdgeInput():
                return self.read_edges(node_input.sources, item_indexes)
            case CollectingInput():
                return self.collect(node_input, item_indexes)
            case _:
                assert_never(node_input)

    def read_edges(self, edges: tuple[Edge, ...], item_indexes: ItemIndexes) -> Any:
        """Returns what the edges into one input give for the call at ``item_indexes``.

        With one edge, that is what ``read_output`` reads from the output the edge comes from. Several edges into one
        input come from different branches of one node, so for one call no two of them have a value. When the
        reader is asked about any item of a loop, several may have one.

        Returns:
            The first value an edge gives. Otherwise ``NOT_READY`` if any edge gave that. Otherwise ``None`` if any
            edge gave that. ``SKIPPED`` when every edge comes from a skipped branch.
        """
        best: Any = SKIPPED
        for edge in edges:
            value = self.read_output(
                self.function_nodes[edge.node],
                edge.output,
                item_indexes,
                edge.wired_to.field,
                one_item=edge.per_item,
            )
            if value is NotReady.NOT_READY:
                best = NotReady.NOT_READY
            elif value is None:
                if best is SKIPPED:
                    best = None
            elif value is not SKIPPED:
                return value
        return best

    def read_output(
        self,
        node: BlueprintFunctionNode,
        output: str,
        item_indexes: ItemIndexes,
        record_field: str | None,
        *,
        one_item: bool,
    ) -> Any:
        """Returns what an edge reads from the output ``output`` of ``node``, for the call at ``item_indexes``.

        In the example, the edge into ``shout.line`` reads one item of ``lines.result``. At ``(1,)`` it reads ``"b"``.
        The edge into ``join.texts`` reads ``shout.result``. At ``(1,)`` it reads ``"B!"``.

        Args:
            node: The function node whose output it is.
            output: The output's name.
            item_indexes: The item indexes of the call that reads it.
            record_field: The field of the output's record that the edge reads. ``None`` for the whole value.
            one_item: Whether the output is a ``Series`` and the edge reads one item of it.

        Returns:
            The value, which may be ``None`` or ``SKIPPED``. ``NOT_READY`` when ``node`` has no outcome yet for these
            item indexes, or is waiting for a person. When asked about any item, because ``item_indexes`` lacks an
            index for a loop it needs, it returns ``MAY_HAVE_VALUE`` if an item may have a value. If none can, it
            returns the ``None`` or ``SKIPPED`` that every item would get.

        Raises:
            NoSuchFieldError: The edge reads a field of a record, and the record has no such field.
        """
        depth = len(node.loops)
        outcome = self.outcomes[node.node_path].get(item_indexes[:depth])
        if outcome is None or not outcome.has_outputs:
            if (
                len(item_indexes) < depth
            ):  # asked about any item of a loop that the source node is inside
                would_be = self.for_any_item[node]
                if isinstance(would_be, dict):
                    return MAY_HAVE_VALUE
                return would_be if would_be is NotReady.NOT_READY else would_be.outputs[output]
            return NotReady.NOT_READY
        value = outcome.outputs[output]
        if value is None or value is SKIPPED:
            return value
        if not one_item:
            return value if record_field is None else field_of(value, record_field)
        try:  # ``value`` is a stored Series: the edge reads one item of it, or one field of that item
            index = item_indexes[depth]
        except IndexError:  # asked about any item of this list, which exists
            return MAY_HAVE_VALUE
        if record_field is None:
            return value[index]
        return value.cell(record_field, index)

    def collect(self, node_input: CollectingInput, item_indexes: ItemIndexes) -> Any:
        """Returns the list a collecting input gets for the call at ``item_indexes``.

        In the example, ``join.texts`` at ``()`` gets ``["A!", "B!"]``: what the edge from ``shout.result`` reads for
        ``(0,)`` and for ``(1,)``.

        Returns:
            The list, with one entry per item of the loop, and ``None`` for an item with no value because its call was
            skipped or not made. ``SKIPPED`` or ``None`` in place of the list when no item can have a value, however
            many items there are (``_list_looped_over``). ``NOT_READY`` until every item can be read. When asked about
            any item of an outer loop: ``MAY_HAVE_VALUE``, or the ``SKIPPED``, ``NOT_READY`` or ``None`` that every
            item gets.
        """
        series_node = self.function_nodes[node_input.series[0]]
        depth, keeps_none_items = len(series_node.loops), node_input.field.keeps_none_items
        if len(item_indexes) < depth:
            return _may_have_value(
                self.read_edges(node_input.sources, item_indexes), keeps_none_items
            )
        item_indexes = item_indexes[:depth]
        loop = node_input.series
        sources = [self.function_nodes[edge.node] for edge in node_input.sources]
        for_any_item = self._for_any_item_of(loop, sources, item_indexes).read_edges(
            node_input.sources, item_indexes
        )
        series = self._list_looped_over(loop, for_any_item, keeps_none_items, item_indexes)
        if series is None or series is SKIPPED or series is NotReady.NOT_READY:
            return series
        edges = node_input.sources
        if (
            len(edges) == 1 and edges[0].per_item
        ):  # the input collects the list itself: its items, or one field of each item
            record_field = edges[0].wired_to.field
            if record_field is not None:
                return series.field_as_list(record_field)
            return (
                series.as_table_input()
                if node_input.field.whole.row is not None
                else series.as_list()
            )
        items = []
        for index in range(len(series)):
            value = self.read_edges(edges, (*item_indexes, index))
            if value is NotReady.NOT_READY:
                return NotReady.NOT_READY
            items.append(None if value is SKIPPED else value)
        return items

    def _list_looped_over(
        self,
        loop: tuple[NodePath, str],
        for_any_item: Any,
        keeps_none_items: bool,
        item_indexes: ItemIndexes,
    ) -> Any:
        """Returns the list being looped over, if any item of it may have a value.

        When no item can have one, it returns what every item gets instead, however many items the list has.

        In the example, ``join.texts`` collects the loop over ``lines.result``, and ``shout`` is the one node inside
        that loop. ``for_any_item`` is what ``shout.result`` gives for any line (``_for_any_item_of``). If
        ``shout.mark`` gets ``SKIPPED``, every call of ``shout`` is skipped. Then ``for_any_item`` is ``SKIPPED``, and so
        is ``join.texts``, with two lines just as with none.

        Args:
            loop: The list looped over: the path of the function node that returns it, and the output's name.
            for_any_item: What any item of the loop gives: ``MAY_HAVE_VALUE``, or the ``SKIPPED``, ``NOT_READY`` or
                ``None`` that every item gets.
            keeps_none_items: Whether a ``None`` item counts as a value, as in ``Series[str | None]``.
            item_indexes: The item indexes of the call that returned the list.

        Returns:
            The stored list, which has ``len()``. Or ``None`` or ``SKIPPED``, when the list itself is that, or when a
            ``None`` or a skipped branch outside the loop means no item can have a value. Or ``NOT_READY``.
        """
        value = _may_have_value(for_any_item, keeps_none_items)
        if value is not MAY_HAVE_VALUE:
            return value
        node_path, output = loop
        outcome = self.outcomes[node_path].get(item_indexes)
        return (
            outcome.outputs[output]
            if outcome is not None and outcome.has_outputs
            else NotReady.NOT_READY
        )

    def _for_any_item_of(
        self,
        loop: tuple[NodePath, str],
        sources: list[BlueprintFunctionNode],
        item_indexes: ItemIndexes,
    ) -> ResultReader:
        """Returns a new reader that answers for any item of ``loop``.

        Its ``for_any_item`` has what ``arguments_for`` returns at ``item_indexes`` for each node inside the loop that
        ``sources`` depend on. Each node is worked out after the nodes upstream of it (``loop_body_of``).

        Args:
            loop: The list looped over: the path of the function node that returns it, and the output's name.
            sources: The function nodes a value is read from.
            item_indexes: The item indexes of the call that returned the list.
        """
        any_item = type(self)(self.function_nodes, self.outcomes, self.graph_inputs)
        for node in loop_body_of(loop, sources):
            any_item.for_any_item[node] = any_item.arguments_for(node, item_indexes)
        return any_item

    def align_lists(self, node: BlueprintFunctionNode, arguments: dict[str, Any]) -> dict[str, Any]:
        """Returns ``arguments`` with the lists collected from one loop cut down to the same items.

        Say ``pair(names: Series[str], ages: Series[int])`` collects two outputs of one loop, and item 1 has no age.
        Item 1 is dropped from both lists, so ``names[i]`` and ``ages[i]`` are still the same item. An input that takes
        ``None`` items, like ``Series[int | None]``, drops nothing: it keeps the ``None``.
        """
        for inputs in node.lists_from_same_loop:
            dropped: set[int] = (
                set()
            )  # the indexes where an input that takes no None item has a None
            for name, keeps_none_items in inputs:
                if not keeps_none_items and arguments[name] is not None:
                    dropped.update(_indexes_of_none(arguments[name]))
            if dropped:
                for name, _ in inputs:
                    if arguments[name] is not None:
                        arguments[name] = _without_indexes(arguments[name], dropped)
        return arguments


def _may_have_value(value: Any, keeps_none_items: bool) -> Any:
    """Returns ``MAY_HAVE_VALUE`` when ``value`` can be an item of an input's list, and ``value`` itself otherwise.

    ``value`` is what the input's edges gave. ``SKIPPED``, ``NOT_READY`` and ``None`` are returned as they are,
    since the input gets them in place of a list. A ``None`` counts as an item when the items take ``None``, as in
    ``Series[str | None]``. When asked about any item of a loop, the result says whether some item may have a
    value.
    """
    if value is SKIPPED or value is NotReady.NOT_READY or (value is None and not keeps_none_items):
        return value
    return MAY_HAVE_VALUE


def _indexes_of_none(value: Any) -> list[int]:
    """Returns the indexes of the ``None`` items in the value of a ``Series`` or ``Table`` input.

    For example, ``_indexes_of_none(["a", None])`` is ``[1]``. A frame has no ``None`` rows, so for a frame the
    result is ``[]``.
    """
    return (
        [index for index, item in enumerate(value) if item is None] if type(value) is list else []
    )


def _without_indexes(
    value: list[Any] | pl.DataFrame, indexes: set[int]
) -> list[Any] | pl.DataFrame:
    """Returns the value of a ``Series`` or ``Table`` input with the items at ``indexes`` left out.

    For example, ``_without_indexes(["a", "b", "c"], {1})`` is ``["a", "c"]``. For a frame of three rows, ``{1}``
    gives a frame of the first and third rows.
    """
    match value:
        case list():
            return [item for index, item in enumerate(value) if index not in indexes]
        case pl.DataFrame():
            return value.filter([index not in indexes for index in range(value.height)])
        case _:
            assert_never(value)
