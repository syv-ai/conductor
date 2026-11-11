"""What a call receives: the rule that says what a node comes to for one item, stated once.

**The node rule.** A node needs every one of its inputs. If one comes only from
branches not taken, the node is skipped, whatever the others hold. Else, while
one isn't settled, nothing is known yet. Else, if one that takes no blank got a
blank, the node isn't called. Else it is called with what its inputs hold
(``Reader.comes_to``).

**One input.** Several edges into one input are alternatives, on opposite
branches of one decision: the input holds the best of what they bring, a value
before what isn't settled, before a blank, before a branch not taken
(``Reader.best_of``). A gathering input (a ``Series``, or a ``Table`` fed
records) receives the items of one Series, and first asks what was settled
outside the items one thing: could any item bring a value here? When none
could, for a branch not taken or for a blank, that is what the input holds,
with two items as with none. When one could, each item answers for itself.
Its node needs the Series itself as well, and reads it as one more input, so
the node rule is what weighs a blank Series against a branch not taken.

**The readers.** The rule reads outputs, and there are two ways to read one:

- ``ItemReader`` reads what the units of a run hold for one item: the reader a
  leg calls a node with. ``AsItStandsReader`` is the same for a host, whose
  ``results`` are read in any state: what isn't settled reads as a blank.
- ``AnyItemReader`` reads for an item of one Series that nobody knows, which is
  how "could any item bring a value" is answered. An output made outside the items
  is read as it stands. An output made per item is what its node comes to for any
  item, by the node rule again: a node that comes to skipped or blank for any
  item holds what such a unit holds, and one that could be called could hold a
  value, ``COULD``.

So the answers are an order, skipped before blank before could bring a value,
and no rule is written for "any item" that isn't the rule for one.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from conductor._internal.checked_graph.input_plans import (
    InputGathersItems,
    InputReadsOneValue,
    ItemOfSeries,
    OutputAtScope,
    SeriesAsList,
)
from conductor._internal.problem import UnitFailedError
from conductor._internal.problem import catalogue as problem
from conductor._internal.record_field import NoSuchFieldError, field_of
from conductor._internal.sentinels import SKIPPED
from conductor._internal.series import places_of_none, without_places
from conductor._internal.units import (
    NOT_SETTLED,
    Asking,
    Blank,
    ItemPath,
    NotReady,
    Skipped,
    Units,
    length,
)

if TYPE_CHECKING:
    from conductor._internal.checked_graph.tree import FunctionAtPath

#: The arguments of one call of a node, by input name.
CallInputs = dict[str, Any]


class _Could:
    def __repr__(self) -> str:
        return "COULD"


#: What an output holds for an item nobody knows, when it could hold a value. Beside it an output holds ``SKIPPED`` (no
#: item could bring a value, for a branch not taken) or ``None`` (none could, for a blank).
COULD: Final = _Could()


class Reader(abc.ABC):
    """Reads what a node's inputs hold, and says what the node comes to: the node rule, over a way to read a
    output.

    Attributes:
        caller_inputs: The caller's inputs, by address.
    """

    __slots__ = ("caller_inputs",)

    caller_inputs: Mapping[str, Any]

    @abc.abstractmethod
    def output(self, at: OutputAtScope, item_path: ItemPath) -> Any:
        """What the output ``at`` holds for ``item_path`` as an edge brings it: a value, ``None`` for a blank, ``SKIPPED``
        for a branch not taken, or ``NOT_SETTLED``."""

    @abc.abstractmethod
    def gathered(self, plan: InputGathersItems, item_path: ItemPath) -> Any:
        """What the gathering input of ``plan`` holds for ``item_path``: its items, ``None`` for a blank,
        ``SKIPPED`` for a branch not taken, or ``NOT_SETTLED``."""

    @abc.abstractmethod
    def kept_in_step(self, node: FunctionAtPath, received: CallInputs) -> CallInputs:
        """``received`` with the gathering inputs of ``node`` that gather one Series holding the same items."""

    def comes_to(self, node: FunctionAtPath, item_path: ItemPath) -> CallInputs | Skipped | Blank | Literal[NotReady.NOT_SETTLED]:
        """What ``node`` comes to for ``item_path``: the node rule.

        Every input is read before not settled or a blank is the answer, so the order of a node's parameters
        decides nothing.

        Returns:
            The arguments of its call; or the unit it comes to uncalled: ``Skipped`` when an input comes only
            from branches not taken, whatever the others hold, else ``Blank`` when an input that takes no blank
            got one; or ``NOT_SETTLED`` while something it reads is.

        Raises:
            UnitFailedError: An edge names a field the record it brings doesn't have (``invalid_edge_value``).
        """
        received = node.call_template.copy()
        caller_inputs = self.caller_inputs
        for name, address in node.from_caller:
            received[name] = caller_inputs[address]
        blank: str | None = None
        settled = True
        for name, plan, takes_a_blank in node.fed:
            try:
                value = plan.value(self, item_path)
            except NoSuchFieldError as missing:
                reason = f"it has no field '{missing.field}'"
                raise UnitFailedError(problem.invalid_edge_value(name=name, reason=reason, record_field=missing.field)) from None
            if value is SKIPPED:
                return node.placed.skipped
            if value is NOT_SETTLED:  # a later input may be skipped, which settles the node whatever this one comes to
                settled = False
            elif value is None and blank is None and not takes_a_blank:
                blank = name
            received[name] = value
        if not settled:
            return NOT_SETTLED
        if blank is not None:
            return Blank(node.placed.outputs_not_called, blank)
        return self.kept_in_step(node, received) if node.gathered_in_step else received

    def best_of(self, reads: tuple[OutputAtScope, ...], item_path: ItemPath) -> Any:
        """What alternative edges bring one input: a value when one of them holds one, else ``NOT_SETTLED``
        while one isn't settled, else a blank, and ``SKIPPED`` when every branch wasn't taken.

        For one item at most one alternative holds a value, as they sit on opposite branches of one decision.
        For an item nobody knows several could, which is why this is the best of them and not the first.
        """
        best: Any = SKIPPED
        for at in reads:
            value = self.output(at, item_path)
            if value is NOT_SETTLED:
                best = NOT_SETTLED
            elif value is None:
                if best is SKIPPED:
                    best = None
            elif value is not SKIPPED:
                return value
        return best


class ItemReader(Reader):
    """Reads what the units of a run hold for one item: what a call of a node receives.

    Attributes:
        units: Each function's units.
    """

    __slots__ = ("units",)

    def __init__(self, units: Units, caller_inputs: Mapping[str, Any]) -> None:
        self.units, self.caller_inputs = units, caller_inputs

    def output(self, at: OutputAtScope, item_path: ItemPath) -> Any:
        """What the output ``at`` holds for ``item_path``: an ``ItemOfSeries``'s is the item of ``item_path``, and a field named
        on the edge is read from it.

        Raises:
            NoSuchFieldError: The edge names a field the record doesn't have.
        """
        depth = at.depth
        unit = self.units[at.node_id].get(item_path[:depth])
        if unit is None or type(unit) is Asking:
            return NOT_SETTLED
        value = unit.out[at.output]
        if value is None or value is SKIPPED or type(at) is OutputAtScope:
            return value
        if type(at) is ItemOfSeries:  # ``value`` is a stored Series: one item of it, or one field of that item
            wanted = at.record_field
            if wanted is None:
                return value[item_path[depth]]
            cell = value.columns[wanted][item_path[depth]]
            if type(cell) is NoSuchFieldError:
                raise NoSuchFieldError(wanted)
            return cell
        if type(at) is SeriesAsList:
            return value.as_list()
        field = at.record_field  # type: ignore[attr-defined]  # an edge brings a WholeOutput, an ItemOfSeries or a SeriesAsList, each with its record_field
        return value if field is None else field_of(value, field)

    def gathered(self, plan: InputGathersItems, item_path: ItemPath) -> Any:
        """The items the gathering input of ``plan`` gets for ``item_path``, with ``None`` for an item that has
        no value, for a blank or a branch not taken.

        What is outside the items is asked first whether any item could bring a value. When none could, that
        is the answer for any number of items, none included. When one could, each item answers for itself.
        The input's node needs the Series too, and reads it as an input of its own (``FunctionAtPath.fed``),
        so the node rule weighs the two; here a Series that holds no list is only handed back.

        Returns:
            The list, one place per item of the Series; ``NOT_SETTLED`` until what is outside and every item
            is settled; ``SKIPPED`` when no item could bring a value for a branch not taken outside the items;
            ``None`` when none could for a blank there.
        """
        item_path = item_path[: plan.reads[0].depth]
        series = self._series_gathered(plan, item_path)
        if series is None or series is SKIPPED or series is NOT_SETTLED:
            return series
        edges = plan.sources
        if len(edges) == 1 and type(edges[0]) is ItemOfSeries:  # the Series itself: its items, or one field of each
            wanted = edges[0].record_field
            if wanted is not None:
                return series.field_as_list(wanted)
            return series.as_table_input() if plan.field.whole.record is not None else series.as_list()
        item_paths = self._item_paths(series, plan.reads[1 : plan.levels], item_path)
        if item_paths is NOT_SETTLED:
            return NOT_SETTLED
        items = []
        one_edge = edges[0] if len(edges) == 1 else None
        for under in item_paths:
            value = self.output(one_edge, under) if one_edge is not None else self.best_of(edges, under)
            if value is NOT_SETTLED:
                return NOT_SETTLED
            items.append(None if value is SKIPPED else value)
        return items

    def _series_gathered(self, plan: InputGathersItems, item_path: ItemPath) -> Any:
        """The Series the input of ``plan`` gathers, where an item of it could bring a value; else what the
        input holds whatever its items: ``SKIPPED``, ``None`` for a blank, or ``NOT_SETTLED``. A Series that
        was never made, or isn't settled, is handed back as that too."""
        made = plan.reads[0]
        brought = AnyItemReader(self, made, item_path).could_bring(plan)
        return brought if brought is not COULD else self.output(made, item_path)

    def _item_paths(self, series: Any, inner: tuple[OutputAtScope, ...], item_path: ItemPath) -> list[ItemPath] | Literal[NotReady.NOT_SETTLED]:
        """The path of each item of ``series`` under ``item_path``, and under each of those of each item of the
        Series ``inner`` names, outermost first; ``NOT_SETTLED`` while one of those isn't settled."""
        item_paths = [(*item_path, place) for place in range(len(series))]
        for made in inner:
            deeper: list[ItemPath] = []
            for under in item_paths:
                held = self.output(made, under)
                if held is NOT_SETTLED:
                    return NOT_SETTLED
                deeper.extend((*under, place) for place in range(length(held)))
            item_paths = deeper
        return item_paths

    def kept_in_step(self, node: FunctionAtPath, received: CallInputs) -> CallInputs:
        """``received`` with each gathering input holding the items that have a value in every input of the
        node that gathers the same Series, so those inputs stay in step; an input whose items may be blank
        keeps a blank in its place."""
        for inputs in node.gathered_in_step:
            dropped: set[int] = set()  # the indexes at which an input that takes no None has one
            for name, keeps_blanks in inputs:
                if not keeps_blanks and received[name] is not None:
                    dropped.update(places_of_none(received[name]))
            if dropped:
                for name, _ in inputs:
                    if received[name] is not None:
                        received[name] = without_places(received[name], dropped)
        return received


class AsItStandsReader(ItemReader):
    """Reads what the units of a run hold as they stand, for a host that reads a run in any state: an output that
    isn't settled reads as a blank, where a call waits for it. So a host is shown the items so far, with a
    blank in the place of one that hasn't run or waits for a person.

    An edge can name one field of a record (``give.result.name``). When the record has no such field, the
    node the edge leads to fails and the run ends ``failed`` with that as its cause. Reading the run
    afterwards must not raise again, so here the missing field is read as ``None``.
    """

    __slots__ = ()

    def output(self, at: OutputAtScope, item_path: ItemPath) -> Any:
        """The value of the output ``at`` for the item at ``item_path``; ``None`` when the node hasn't
        returned yet, or when the edge names a field the record doesn't have."""
        try:
            value = ItemReader.output(self, at, item_path)
        except NoSuchFieldError:
            return None
        return None if value is NOT_SETTLED else value

    def gathered(self, plan: InputGathersItems, item_path: ItemPath) -> Any:
        """The list a ``Series`` input with ``plan`` would be called with for the item at ``item_path``;
        ``None`` when the edge names a field that a record in the list doesn't have."""
        try:
            return ItemReader.gathered(self, plan, item_path)
        except NoSuchFieldError:
            return None

    def shown(self, plans: tuple[InputReadsOneValue | InputGathersItems, ...], item_path: ItemPath = ()) -> Any:
        """One output of a node as ``run.results`` shows it.

        Example: ``documents -> lines -> shout`` with three documents, the second empty, so ``lines`` was
        never called for it. ``shout`` is called once per line of each document, and its output is shown as
        ``[["A", "B"], None, ["C"]]``.

        Args:
            plans: The output's plans from ``host_reads``: one for each loop the node is called in.
            item_path: The index of the item in each loop already entered; empty for the whole output.

        Returns:
            The return value, for a node called once. For a node called per item, a list with one entry per
            item, ``None`` for an item that has no value; for a node called in a loop inside another loop, a
            list of such lists. ``None`` instead of a list when the list being looped over was never made,
            or when no item of it could have a value because of a skipped branch or a ``None`` outside the
            loop.
        """
        plan = plans[len(item_path)]
        if len(item_path) == len(plans) - 1:
            value = plan.value(self, item_path)
            return None if value is SKIPPED else value
        looped_over = self._series_gathered(cast("InputGathersItems", plan), item_path)  # every plan before the last lists one loop
        if looped_over is None or looped_over is SKIPPED:
            return None
        return [self.shown(plans, (*item_path, index)) for index in range(len(looped_over))]


class AnyItemReader(Reader):
    """Reads for an item nobody knows of one Series, under the item path its gathering input gathers at.

    Attributes:
        items: Reads what is made outside the items, as it stands.
        series: The Series whose items nobody knows, where it was made.
        item_path: The item path the Series is gathered at.
        came_to: What each function made per item of the Series comes to for any item, by its name, as far as
            asked: a function not in it is made outside the items.
    """

    __slots__ = ("came_to", "item_path", "items", "series")

    def __init__(self, items: ItemReader, series: OutputAtScope, item_path: ItemPath) -> None:
        self.items, self.series, self.item_path, self.caller_inputs = items, series, item_path, items.caller_inputs
        self.came_to: dict[str, CallInputs | Skipped | Blank | Literal[NotReady.NOT_SETTLED]] = {}

    def could_bring(self, plan: InputGathersItems) -> Any:
        """Whether any item of the Series could bring the gathering input of ``plan`` a value, going by what
        is outside the items. Each function the input gathers through is asked what it comes to for any item,
        in order, and then what the input's edges bring.

        Returns:
            ``COULD``; ``SKIPPED`` when none could for a branch not taken; ``None`` when none could for a
            blank; ``NOT_SETTLED`` while something outside the items isn't settled.
        """
        for node in plan.asked_for_any_item:
            self.came_to[node.name] = self.comes_to(node, self.item_path)
        return self._brought(plan)

    def output(self, at: OutputAtScope, item_path: ItemPath) -> Any:
        """What the output ``at`` holds for an item nobody knows: what its node comes to for any item when it is made
        per item, and as it stands when it is made outside the items, ``COULD`` for a value."""
        came_to = self.came_to.get(at.node_id)
        if came_to is None:  # made outside the items
            series = self.series
            if type(at) is ItemOfSeries and at.node_id == series.node_id and at.output == series.output:
                at = series  # an item of the Series itself is what the Series is: one never made has none
            value = self.items.output(at, self.item_path)
            return value if value is NOT_SETTLED or value is SKIPPED or value is None else COULD
        if came_to is NOT_SETTLED:
            return NOT_SETTLED
        return COULD if type(came_to) is dict else came_to.out[at.output]  # type: ignore[union-attr]  # mypy doesn't narrow a union by ``type(x) is dict``

    def gathered(self, plan: InputGathersItems, item_path: ItemPath) -> Any:
        """What a gathering input met on the way holds for an item nobody knows. One whose Series is made per
        item is asked what its edges bring for the same items; one whose Series is made outside them is an output
        outside like any other: its own, whole answer, read as it stands."""
        if plan.reads[0].node_id in self.came_to:
            return self._brought(plan)
        value = self.items.gathered(plan, self.item_path)
        return value if value is NOT_SETTLED or value is SKIPPED or value is None else COULD

    def kept_in_step(self, node: FunctionAtPath, received: CallInputs) -> CallInputs:
        """``received`` as it is: nothing is known of the items to keep in step."""
        return received

    def _brought(self, plan: InputGathersItems) -> Any:
        """What the edges of a gathering input bring for an item nobody knows; an input whose items take a
        blank receives the blank an edge brings, so it could hold one."""
        edges = plan.sources
        brought = self.output(edges[0], self.item_path) if len(edges) == 1 else self.best_of(edges, self.item_path)
        return COULD if brought is None and plan.field.keeps_blanks else brought
