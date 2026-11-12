"""Works out the arguments of one call of a node from the outcomes a run has so far.

The example graph used in this file::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b"]
    shout(line: str, mark: str) -> str       # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!"]

``ResultReader.arguments_for(shout, (1,))`` returns ``{"line": "b", "mark": "!"}``: the arguments of the call
of ``shout`` for the second line. ``(1,)`` is the call's item indexes: one index per loop the node is inside.

When a node cannot be called, ``arguments_for`` returns why instead. The checks are made in this order:

1. An input has only ``SKIPPED`` to read, because every edge into it comes from a branch that was not chosen.
   The node is skipped, whatever its other inputs are: the result is a ``Skipped`` outcome.
2. A value an input reads does not exist yet. The result is ``NOT_READY``, and the node is tried again later.
3. An input reads ``None`` and its parameter does not accept ``None``. The node is not called: the result is a
   ``NotCalled`` outcome.

``join.texts`` is a collecting input: it gets one value per item of a loop, as a list. Before it reads the
items it asks one question: can any item have a value here? Suppose ``mark`` comes from a branch that was not
chosen. Then every call of ``shout`` is skipped. With two lines ``join`` could be called with ``[None, None]``
and with no lines with ``[]``. The engine gives one answer for any number of lines instead: ``join`` is
skipped.

That question is ``arguments_for`` again, asked for ``shout`` with item indexes one index short: ``()``, where
a call of ``shout`` has ``(0,)`` or ``(1,)``. ``mark`` is read as for any call, since its node is inside no
loop. ``line`` is an item of a list, and no index says which: when the list exists it is read as
``MAY_HAVE_VALUE``. So ``arguments_for(shout, ())`` returns ``Skipped`` when ``mark`` is ``SKIPPED``, and
arguments when some call of ``shout`` may be made.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from conductor._internal.compiled.input_plans import (
    EdgeSource,
    InputCollectsLoopResults,
    InputReadsOneValue,
)
from conductor._internal.markers import SKIPPED
from conductor._internal.outcomes import ItemIndexes, NotCalled, NotReady, Outcome, Skipped
from conductor._internal.problem import CallRejectedError
from conductor._internal.problem import catalogue as problem
from conductor._internal.record_field import NoSuchFieldError, field_of
from conductor._internal.series import StoredSeries, places_of_none, without_places

if TYPE_CHECKING:
    from conductor._internal.compiled.flat_nodes import FlatFunctionNode


class _MayHaveValue:
    def __repr__(self) -> str:
        return "MAY_HAVE_VALUE"


#: What a read returns in place of a value when it is asked for any item of a loop and some item may have a
#: value. When no item can have one, the read returns what every item would read: ``SKIPPED`` or ``None``.
MAY_HAVE_VALUE: Final = _MayHaveValue()


class ResultReader:
    """Reads what a call needs from the outcomes of a run, and returns the call's arguments.

    Attributes:
        outcomes: The run's outcomes, by node name and then by item indexes.
        graph_inputs: The values the caller gave for the graph inputs, by address (``"lines.text"``).
        for_any_item: Empty in the reader a run calls its nodes with. A reader made to answer "can any item of
            this loop have a value?" (``_list_looped_over``) keeps here what ``arguments_for`` returned for each
            node inside the loop, by node name.
    """

    __slots__ = ("for_any_item", "graph_inputs", "outcomes")

    def __init__(self, outcomes: dict[str, dict[ItemIndexes, Outcome]], graph_inputs: Mapping[str, Any]) -> None:
        self.outcomes, self.graph_inputs = outcomes, graph_inputs
        self.for_any_item: dict[str, dict[str, Any] | Skipped | NotCalled | Literal[NotReady.NOT_READY]] = {}

    def arguments_for(self, node: FlatFunctionNode, item_indexes: ItemIndexes) -> dict[str, Any] | Skipped | NotCalled | Literal[NotReady.NOT_READY]:
        """Returns the arguments of the call of ``node`` for ``item_indexes``, or why the node is not called.

        In the example, ``arguments_for(shout, (1,))`` returns ``{"line": "b", "mark": "!"}``.

        Every input is read before ``NOT_READY`` or ``NotCalled`` is returned, because a later input may read
        ``SKIPPED``, and that skips the node whatever the earlier inputs are. So the order of a node's
        parameters never changes the result.

        ``_list_looped_over`` calls this with item indexes one index shorter than the node's own. The result
        then says what a call of the node would be for any item: ``Skipped`` or ``NotCalled`` when every item
        would be, and arguments (some of them ``MAY_HAVE_VALUE``) when an item may be called.

        Returns:
            The arguments by parameter name. Or the node's ``Skipped`` outcome, when an input reads ``SKIPPED``.
            Or ``NOT_READY``, when a value an input reads does not exist yet. Or a ``NotCalled`` outcome, when an
            input reads ``None`` and its parameter does not accept ``None``.

        Raises:
            CallRejectedError: An edge reads one field of a record (``lines.result.name``) and the record has no
                such field (``invalid_edge_value``).
        """
        arguments = node.constants.copy()
        graph_inputs = self.graph_inputs
        for name, address in node.graph_inputs:
            arguments[name] = graph_inputs[address]
        none_input: str | None = None
        ready = True
        for name, plan, accepts_none in node.edge_inputs:
            try:
                value = plan.value(self, item_indexes)
            except NoSuchFieldError as missing:
                reason = f"it has no field '{missing.field}'"
                raise CallRejectedError(problem.invalid_edge_value(name=name, reason=reason, record_field=missing.field)) from None
            if value is SKIPPED:
                return node.compiled_node.skipped
            if value is NotReady.NOT_READY:  # keep reading: a later input may read SKIPPED, which skips the node
                ready = False
            elif value is None and none_input is None and not accepts_none:
                none_input = name
            arguments[name] = value
        if not ready:
            return NotReady.NOT_READY
        if none_input is not None:
            return NotCalled(node.compiled_node.outputs_when_not_called, none_input)
        return self.align_lists(node, arguments) if node.lists_from_same_loop else arguments

    def read_alternatives(self, sources: tuple[EdgeSource, ...], item_indexes: ItemIndexes) -> Any:
        """Returns what an input with several edges into it reads for ``item_indexes``.

        The edges come from different branches of one decision, so for one call at most one of them has a
        value. Asked for any item, several may have one.

        Returns:
            The first value an edge reads; else ``NOT_READY`` if an edge read that; else ``None`` if an edge read
            ``None``; else ``SKIPPED``, when every edge comes from a branch that was not chosen.
        """
        best: Any = SKIPPED
        for source in sources:
            value = self.read_output(source, item_indexes)
            if value is NotReady.NOT_READY:
                best = NotReady.NOT_READY
            elif value is None:
                if best is SKIPPED:
                    best = None
            elif value is not SKIPPED:
                return value
        return best

    def read_output(self, source: EdgeSource, item_indexes: ItemIndexes) -> Any:
        """Returns what one edge reads from the output ``source`` for the call at ``item_indexes``.

        In the example, the edge into ``shout.line`` reads one item of ``lines.result``: for ``(1,)`` it reads
        ``"b"``. The edge into ``join.texts`` reads ``shout.result``: for ``(1,)`` it reads ``"B!"``.

        Returns:
            The value. ``None`` or ``SKIPPED`` when the output is that. ``NOT_READY`` when the node that returns
            the output has no outcome yet for these item indexes, or is waiting for a person. Asked for any
            item (``item_indexes`` has no index for a loop the read needs one for): ``MAY_HAVE_VALUE`` when an
            item may have a value, else the ``None`` or ``SKIPPED`` every item would read.

        Raises:
            NoSuchFieldError: The edge reads one field of a record, and the record has no such field.
        """
        depth = source.depth
        outcome = self.outcomes[source.node_id].get(item_indexes[:depth])
        if outcome is None or not outcome.has_outputs:
            if len(item_indexes) < depth:  # asked for any item of a loop the source's node is inside
                would_be = self.for_any_item[source.node_id]
                if isinstance(would_be, dict):
                    return MAY_HAVE_VALUE
                return would_be if would_be is NotReady.NOT_READY else would_be.outputs[source.output]
            return NotReady.NOT_READY
        value = outcome.outputs[source.output]
        if value is None or value is SKIPPED:
            return value
        record_field = source.record_field
        if not source.item:
            return value if record_field is None else field_of(value, record_field)
        try:  # ``value`` is a stored Series: one item of it, or one field of that item
            index = item_indexes[depth]
        except IndexError:  # asked for any item of this list, which exists
            return MAY_HAVE_VALUE
        if record_field is None:
            return value[index]
        cell = value.columns[record_field][index]
        if type(cell) is NoSuchFieldError:
            raise NoSuchFieldError(record_field)
        return cell

    def collect(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """Returns the list a collecting input gets for the call at ``item_indexes``.

        In the example, ``collect`` for ``join.texts`` at ``()`` returns ``["A!", "B!"]``: what the edge from
        ``shout.result`` reads for ``(0,)`` and for ``(1,)``.

        Returns:
            The list, with one entry per item of the loop and ``None`` for an item with no value (its call was
            skipped or not called). ``SKIPPED`` or ``None`` instead of a list when no item can have a value,
            however many items there are (``_list_looped_over``). ``NOT_READY`` until every item has been
            read. Asked for any item of a loop this input's loop is inside: what ``_edges_for_any_item``
            returns.
        """
        depth = plan.series.depth
        if len(item_indexes) < depth:
            return self._edges_for_any_item(plan, item_indexes)
        item_indexes = item_indexes[:depth]
        series = self._list_looped_over(plan, item_indexes)
        if series is None or series is SKIPPED or series is NotReady.NOT_READY:
            return series
        edges = plan.sources
        if len(edges) == 1 and edges[0].item:  # the input collects the list itself: its items, or one field of each
            record_field = edges[0].record_field
            if record_field is not None:
                return series.field_as_list(record_field)
            return series.as_table_input() if plan.field.whole.record is not None else series.as_list()
        items = []
        one_edge = edges[0] if len(edges) == 1 else None
        for index in range(len(series)):
            indexes_of_item = (*item_indexes, index)
            value = self.read_output(one_edge, indexes_of_item) if one_edge is not None else self.read_alternatives(edges, indexes_of_item)
            if value is NotReady.NOT_READY:
                return NotReady.NOT_READY
            items.append(None if value is SKIPPED else value)
        return items

    def _list_looped_over(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """Returns the list whose items the input of ``plan`` collects, when an item of it may have a value for
        the input. When no item can, returns what the input is instead, however many items the list has.

        In the example, ``join.texts`` collects the loop over ``lines.result``, and ``shout`` is the one node
        inside that loop. ``arguments_for(shout, ())`` is asked first: ``()`` is the item indexes of the list,
        one index short of a call of ``shout``. If ``shout.mark`` reads ``SKIPPED``, the answer is ``Skipped``:
        every call of ``shout`` would be skipped, so ``join.texts`` is ``SKIPPED``, with two lines as with none.

        Args:
            plan: The collecting input's plan.
            item_indexes: The item indexes of the list.

        Returns:
            The stored list (it has ``len()``); or ``None`` or ``SKIPPED`` when the list itself is that, or when
            no item can have a value because of a ``None`` or a branch not chosen outside the loop; or
            ``NOT_READY``.
        """
        if plan.loop_body:  # speed only: with no node inside the loop, the lines below return what reading the list returns. Worth 12% of a run that collects 2,000 lists.
            any_item = type(self)(self.outcomes, self.graph_inputs)
            for node in plan.loop_body:  # each after the nodes it reads
                any_item.for_any_item[node.name] = any_item.arguments_for(node, item_indexes)
            value = any_item._edges_for_any_item(plan, item_indexes)
            if value is not MAY_HAVE_VALUE:
                return value
        outcome = self.outcomes[plan.series.node_id].get(item_indexes)
        return outcome.outputs[plan.series.output] if outcome is not None and outcome.has_outputs else NotReady.NOT_READY

    def _edges_for_any_item(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """Returns what the edges into a collecting input read for any item of its loop: ``MAY_HAVE_VALUE``, or
        the ``SKIPPED`` or ``None`` that every item would read, or ``NOT_READY``. A ``None`` that the input
        keeps as an item (``Series[str | None]``) counts as a value."""
        edges = plan.sources
        value = self.read_output(edges[0], item_indexes) if len(edges) == 1 else self.read_alternatives(edges, item_indexes)
        if value is SKIPPED or value is NotReady.NOT_READY or (value is None and not plan.field.accepts_none_items):
            return value
        return MAY_HAVE_VALUE

    def align_lists(self, node: FlatFunctionNode, arguments: dict[str, Any]) -> dict[str, Any]:
        """Returns ``arguments`` with the lists collected from one loop cut to the same items.

        Example: ``pair(names: Series[str], ages: Series[int])`` collects two outputs of one loop, and item 1
        has no age. Item 1 is dropped from both lists, so ``names[i]`` and ``ages[i]`` are still the same item.
        An input that accepts ``None`` items (``Series[int | None]``) drops nothing: it keeps the ``None``.
        """
        for inputs in node.lists_from_same_loop:
            dropped: set[int] = set()  # the indexes at which an input that accepts no None item has one
            for name, accepts_none_items in inputs:
                if not accepts_none_items and arguments[name] is not None:
                    dropped.update(places_of_none(arguments[name]))
            if dropped:
                for name, _ in inputs:
                    if arguments[name] is not None:
                        arguments[name] = without_places(arguments[name], dropped)
        return arguments


class DisplayReader(ResultReader):
    """Reads what the units of a run hold as they stand, for a host that reads a run in any state: an output that
    isn't settled reads as a blank, where a call waits for it. So a host is shown the items so far, with a
    blank in the place of one that hasn't run or waits for a person.

    An edge can name one field of a record (``give.result.name``). When the record has no such field, the
    node the edge leads to fails and the run ends ``failed`` with that as its cause. Reading the run
    afterwards must not raise again, so here the missing field is read as ``None``.
    """

    __slots__ = ()

    def read_output(self, source: EdgeSource, item_indexes: ItemIndexes) -> Any:
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

    def _list_looped_over(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """Returns what ``ResultReader._list_looped_over`` returns, with ``None`` when the list is not made yet."""
        looped_over = ResultReader._list_looped_over(self, plan, item_indexes)
        return None if looped_over is NotReady.NOT_READY else looped_over

    def collect(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """The list a ``Series`` input with ``plan`` would be called with for the item at ``item_indexes``;
        ``None`` when the edge names a field that a record in the list doesn't have."""
        try:
            return ResultReader.collect(self, plan, item_indexes)
        except NoSuchFieldError:
            return None

    def display_value(self, plans: tuple[InputReadsOneValue | InputCollectsLoopResults, ...], item_indexes: ItemIndexes = ()) -> Any:
        """One output of a node as ``run.results`` shows it.

        Example: ``documents -> lines -> shout`` with three documents, the second empty, so ``lines`` was
        never called for it. ``shout`` is called once per line of each document, and its output is shown as
        ``[["A", "B"], None, ["C"]]``.

        Args:
            plans: The output's plans from ``display_plans_of``: one for each loop the node is called in.
            item_indexes: The index of the item in each loop already entered; empty for the whole output.

        Returns:
            The return value, for a node called once. For a node called per item, a list with one entry per
            item, ``None`` for an item that has no value; for a node called in a loop inside another loop, a
            list of such lists. ``None`` instead of a list when the list being looped over was never made,
            or when no item of it could have a value because of a skipped branch or a ``None`` outside the
            loop.
        """
        plan = plans[len(item_indexes)]
        if len(item_indexes) == len(plans) - 1:
            value = plan.value(self, item_indexes)
            return None if value is SKIPPED else value
        looped_over = self._list_looped_over(cast("InputCollectsLoopResults", plan), item_indexes)  # every plan before the last lists one loop
        if looped_over is None or looped_over is SKIPPED:
            return None
        return [self.display_value(plans, (*item_indexes, index)) for index in range(len(looped_over))]
