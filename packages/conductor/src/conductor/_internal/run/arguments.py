"""What a call receives: the rule that says what a node comes to for one item, stated once.

**The node rule.** A node needs every one of its inputs. If one comes only from
branches not taken, the node is skipped, whatever the others hold. Else, while
one isn't settled, nothing is known yet. Else, if one that takes no blank got a
blank, the node isn't called. Else it is called with what its inputs hold
(``Reader.arguments_for``).

**One input.** Several edges into one input are alternatives, on opposite
branches of one decision: the input holds the best of what they bring, a value
before what isn't settled, before a blank, before a branch not taken
(``Reader.read_alternatives``). A gathering input (a ``Series``, or a ``Table`` fed
records) receives the items of one Series, and first asks what was settled
outside the items one thing: could any item bring a value here? When none
could, for a branch not taken or for a blank, that is what the input holds,
with two items as with none. When one could, each item answers for itself.
Its node needs the Series itself as well, and reads it as one more input, so
the node rule is what weighs a blank Series against a branch not taken.

**The readers.** The rule reads outputs, and there are two ways to read one:

- ``ResultReader`` reads what the units of a run hold for one item: the reader a
  leg calls a node with. ``DisplayReader`` is the same for a host, whose
  ``results`` are read in any state: what isn't settled reads as a blank.
- ``LoopDryRun`` reads for an item of one Series that nobody knows, which is
  how "could any item bring a value" is answered. An output made outside the items
  is read as it stands. An output made per item is what its node comes to for any
  item, by the node rule again: a node that comes to skipped or blank for any
  item holds what such a unit holds, and one that could be called could hold a
  value, ``MAY_HAVE_VALUE``.

So the answers are an order, skipped before blank before could bring a value,
and no rule is written for "any item" that isn't the rule for one.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from conductor._internal.compiled.input_plans import (
    InputCollectsLoopResults,
    InputReadsOneValue,
    ItemOfSeries,
    OutputInLoops,
    SeriesAsList,
)
from conductor._internal.outcomes import ItemIndexes, NotCalled, NotReady, Outcome, Skipped, Waiting
from conductor._internal.problem import CallRejectedError
from conductor._internal.problem import catalogue as problem
from conductor._internal.record_field import NoSuchFieldError, field_of
from conductor._internal.sentinels import SKIPPED
from conductor._internal.series import places_of_none, without_places

if TYPE_CHECKING:
    from conductor._internal.compiled.flat_nodes import FlatFunctionNode


class _MayHaveValue:
    def __repr__(self) -> str:
        return "MAY_HAVE_VALUE"


#: What an output holds for an item nobody knows, when it could hold a value. Beside it an output holds ``SKIPPED`` (no
#: item could bring a value, for a branch not taken) or ``None`` (none could, for a blank).
MAY_HAVE_VALUE: Final = _MayHaveValue()


class Reader(abc.ABC):
    """Reads what a node's inputs hold, and says what the node comes to: the node rule, over a way to read a
    output.

    Attributes:
        graph_inputs: The caller's inputs, by address.
    """

    __slots__ = ("graph_inputs",)

    graph_inputs: Mapping[str, Any]

    @abc.abstractmethod
    def read_output(self, source: OutputInLoops, item_indexes: ItemIndexes) -> Any:
        """What the output ``source`` holds for ``item_indexes`` as an edge brings it: a value, ``None`` for a blank, ``SKIPPED``
        for a branch not taken, or ``NOT_READY``."""

    @abc.abstractmethod
    def collect(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """What the gathering input of ``plan`` holds for ``item_indexes``: its items, ``None`` for a blank,
        ``SKIPPED`` for a branch not taken, or ``NOT_READY``."""

    @abc.abstractmethod
    def align_lists(self, node: FlatFunctionNode, arguments: dict[str, Any]) -> dict[str, Any]:
        """``arguments`` with the gathering inputs of ``node`` that gather one Series holding the same items."""

    def arguments_for(self, node: FlatFunctionNode, item_indexes: ItemIndexes) -> dict[str, Any] | Skipped | NotCalled | Literal[NotReady.NOT_READY]:
        """What ``node`` comes to for ``item_indexes``: the node rule.

        Every input is read before not settled or a blank is the answer, so the order of a node's parameters
        decides nothing.

        Returns:
            The arguments of its call; or the unit it comes to uncalled: ``Skipped`` when an input comes only
            from branches not taken, whatever the others hold, else ``NotCalled`` when an input that takes no blank
            got one; or ``NOT_READY`` while something it reads is.

        Raises:
            CallRejectedError: An edge names a field the record it brings doesn't have (``invalid_edge_value``).
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
            if value is NotReady.NOT_READY:  # a later input may be skipped, which settles the node whatever this one comes to
                ready = False
            elif value is None and none_input is None and not accepts_none:
                none_input = name
            arguments[name] = value
        if not ready:
            return NotReady.NOT_READY
        if none_input is not None:
            return NotCalled(node.compiled_node.outputs_when_not_called, none_input)
        return self.align_lists(node, arguments) if node.lists_from_same_loop else arguments

    def read_alternatives(self, reads: tuple[OutputInLoops, ...], item_indexes: ItemIndexes) -> Any:
        """What alternative edges bring one input: a value when one of them holds one, else ``NOT_READY``
        while one isn't settled, else a blank, and ``SKIPPED`` when every branch wasn't taken.

        For one item at most one alternative holds a value, as they sit on opposite branches of one decision.
        For an item nobody knows several could, which is why this is the best of them and not the first.
        """
        best: Any = SKIPPED
        for source in reads:
            value = self.read_output(source, item_indexes)
            if value is NotReady.NOT_READY:
                best = NotReady.NOT_READY
            elif value is None:
                if best is SKIPPED:
                    best = None
            elif value is not SKIPPED:
                return value
        return best


class ResultReader(Reader):
    """Reads what the units of a run hold for one item: what a call of a node receives.

    Attributes:
        units: Each function's units.
    """

    __slots__ = ("outcomes",)

    def __init__(self, outcomes: dict[str, dict[ItemIndexes, Outcome]], graph_inputs: Mapping[str, Any]) -> None:
        self.outcomes, self.graph_inputs = outcomes, graph_inputs

    def read_output(self, source: OutputInLoops, item_indexes: ItemIndexes) -> Any:
        """What the output ``source`` holds for ``item_indexes``: an ``ItemOfSeries``'s is the item of ``item_indexes``, and a field named
        on the edge is read from it.

        Raises:
            NoSuchFieldError: The edge names a field the record doesn't have.
        """
        depth = source.depth
        outcome = self.outcomes[source.node_id].get(item_indexes[:depth])
        if outcome is None or type(outcome) is Waiting:
            return NotReady.NOT_READY
        value = outcome.outputs[source.output]
        if value is None or value is SKIPPED or type(source) is OutputInLoops:
            return value
        if type(source) is ItemOfSeries:  # ``value`` is a stored Series: one item of it, or one field of that item
            record_field = source.record_field
            if record_field is None:
                return value[item_indexes[depth]]
            cell = value.columns[record_field][item_indexes[depth]]
            if type(cell) is NoSuchFieldError:
                raise NoSuchFieldError(record_field)
            return cell
        if type(source) is SeriesAsList:
            return value.as_list()
        field = source.record_field  # type: ignore[attr-defined]  # an edge brings a WholeOutput, an ItemOfSeries or a SeriesAsList, each with its record_field
        return value if field is None else field_of(value, field)

    def collect(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """The items the gathering input of ``plan`` gets for ``item_indexes``, with ``None`` for an item that has
        no value, for a blank or a branch not taken.

        What is outside the items is asked first whether any item could bring a value. When none could, that
        is the answer for any number of items, none included. When one could, each item answers for itself.
        The input's node needs the Series too, and reads it as an input of its own (``FlatFunctionNode.fed``),
        so the node rule weighs the two; here a Series that holds no list is only handed back.

        Returns:
            The list, one place per item of the Series; ``NOT_READY`` until what is outside and every item
            is settled; ``SKIPPED`` when no item could bring a value for a branch not taken outside the items;
            ``None`` when none could for a blank there.
        """
        item_indexes = item_indexes[: plan.reads[0].depth]
        series = self._list_looped_over(plan, item_indexes)
        if series is None or series is SKIPPED or series is NotReady.NOT_READY:
            return series
        edges = plan.sources
        if len(edges) == 1 and type(edges[0]) is ItemOfSeries:  # the Series itself: its items, or one field of each
            record_field = edges[0].record_field
            if record_field is not None:
                return series.field_as_list(record_field)
            return series.as_table_input() if plan.field.whole.record is not None else series.as_list()
        items = []
        one_edge = edges[0] if len(edges) == 1 else None
        for index in range(len(series)):
            prefix = (*item_indexes, index)
            value = self.read_output(one_edge, prefix) if one_edge is not None else self.read_alternatives(edges, prefix)
            if value is NotReady.NOT_READY:
                return NotReady.NOT_READY
            items.append(None if value is SKIPPED else value)
        return items

    def _list_looped_over(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """The Series the input of ``plan`` gathers, where an item of it could bring a value; else what the
        input holds whatever its items: ``SKIPPED``, ``None`` for a blank, or ``NOT_READY``. A Series that
        was never made, or isn't settled, is handed back as that too."""
        series = plan.reads[0]
        brought = LoopDryRun(self, series, item_indexes).result_for_every_item(plan)
        return brought if brought is not MAY_HAVE_VALUE else self.read_output(series, item_indexes)

    def align_lists(self, node: FlatFunctionNode, arguments: dict[str, Any]) -> dict[str, Any]:
        """``arguments`` with each gathering input holding the items that have a value in every input of the
        node that gathers the same Series, so those inputs stay in step; an input whose items may be blank
        keeps a blank in its place."""
        for inputs in node.lists_from_same_loop:
            dropped: set[int] = set()  # the indexes at which an input that takes no None has one
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

    def read_output(self, source: OutputInLoops, item_indexes: ItemIndexes) -> Any:
        """The value of the output ``source`` for the item at ``item_indexes``; ``None`` when the node hasn't
        returned yet, or when the edge names a field the record doesn't have."""
        try:
            value = ResultReader.read_output(self, source, item_indexes)
        except NoSuchFieldError:
            return None
        return None if value is NotReady.NOT_READY else value

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


class LoopDryRun(Reader):
    """Reads for an item nobody knows of one Series, under the item indexes its gathering input gathers at.

    Attributes:
        items: Reads what is made outside the items, as it stands.
        series: The Series whose items nobody knows, where it was made.
        item_indexes: The item indexes the Series is gathered at.
        dry_outcomes: What each function made per item of the Series comes to for any item, by its name, as far as
            asked: a function not in it is made outside the items.
    """

    __slots__ = ("dry_outcomes", "item_indexes", "items", "series")

    def __init__(self, items: ResultReader, series: OutputInLoops, item_indexes: ItemIndexes) -> None:
        self.items, self.series, self.item_indexes, self.graph_inputs = items, series, item_indexes, items.graph_inputs
        self.dry_outcomes: dict[str, dict[str, Any] | Skipped | NotCalled | Literal[NotReady.NOT_READY]] = {}

    def result_for_every_item(self, plan: InputCollectsLoopResults) -> Any:
        """Whether any item of the Series could bring the gathering input of ``plan`` a value, going by what
        is outside the items. Each function the input gathers through is asked what it comes to for any item,
        in order, and then what the input's edges bring.

        Returns:
            ``MAY_HAVE_VALUE``; ``SKIPPED`` when none could for a branch not taken; ``None`` when none could for a
            blank; ``NOT_READY`` while something outside the items isn't settled.
        """
        for node in plan.loop_body:
            self.dry_outcomes[node.name] = self.arguments_for(node, self.item_indexes)
        return self._read_edges(plan)

    def read_output(self, source: OutputInLoops, item_indexes: ItemIndexes) -> Any:
        """What the output ``source`` holds for an item nobody knows: what its node comes to for any item when it is made
        per item, and as it stands when it is made outside the items, ``MAY_HAVE_VALUE`` for a value."""
        dry_outcomes = self.dry_outcomes.get(source.node_id)
        if dry_outcomes is None:  # made outside the items
            series = self.series
            if type(source) is ItemOfSeries and source.node_id == series.node_id and source.output == series.output:
                source = series  # an item of the Series itself is what the Series is: one never made has none
            value = self.items.read_output(source, self.item_indexes)
            return value if value is NotReady.NOT_READY or value is SKIPPED or value is None else MAY_HAVE_VALUE
        if dry_outcomes is NotReady.NOT_READY:
            return NotReady.NOT_READY
        return MAY_HAVE_VALUE if type(dry_outcomes) is dict else dry_outcomes.outputs[source.output]  # type: ignore[union-attr]  # mypy doesn't narrow a union by ``type(x) is dict``

    def collect(self, plan: InputCollectsLoopResults, item_indexes: ItemIndexes) -> Any:
        """What a gathering input met on the way holds for an item nobody knows. One whose Series is made per
        item is asked what its edges bring for the same items; one whose Series is made outside them is an output
        outside like any other: its own, whole answer, read as it stands."""
        if plan.reads[0].node_id in self.dry_outcomes:
            return self._read_edges(plan)
        value = self.items.collect(plan, self.item_indexes)
        return value if value is NotReady.NOT_READY or value is SKIPPED or value is None else MAY_HAVE_VALUE

    def align_lists(self, node: FlatFunctionNode, arguments: dict[str, Any]) -> dict[str, Any]:
        """``arguments`` as it is: nothing is known of the items to keep in step."""
        return arguments

    def _read_edges(self, plan: InputCollectsLoopResults) -> Any:
        """What the edges of a gathering input bring for an item nobody knows; an input whose items take a
        blank receives the blank an edge brings, so it could hold one."""
        edges = plan.sources
        brought = self.read_output(edges[0], self.item_indexes) if len(edges) == 1 else self.read_alternatives(edges, self.item_indexes)
        return MAY_HAVE_VALUE if brought is None and plan.field.accepts_none_items else brought
