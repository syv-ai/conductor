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

**Two readers.** The rule reads cells, and there are two ways to read one:

- ``ItemReader`` reads what the units of a run hold for one item: the reader a
  leg calls a node with, and the one a host's ``results`` are read through.
- ``AnyItemReader`` reads for an item of one Series that nobody knows, which is
  how "could any item bring a value" is answered. A cell made outside the items
  is read as it stands. A cell made per item is what its node comes to for any
  item, by the node rule again: a node that comes to skipped or blank for any
  item holds what such a unit holds, and one that could be called could hold a
  value, ``COULD``.

So the answers are an order, skipped before blank before could bring a value,
and no rule is written for "any item" that isn't the rule for one.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal

from conductor._internal.fields import SKIPPED, FieldValueError, Records
from conductor._internal.loaded_graph.input_plans import Cell, Gather, Item, WholeSeries
from conductor._internal.units import (
    NOT_SETTLED,
    Asking,
    Blank,
    ItemPath,
    NotReady,
    Skipped,
    Units,
    instances,
)

if TYPE_CHECKING:
    from conductor._internal.loaded_graph.tree import FunctionAtPath, NodeAtPath

#: The arguments of one call of a node, by input name.
CallInputs = dict[str, Any]


class _Could:
    def __repr__(self) -> str:
        return "COULD"


#: What a cell holds for an item nobody knows, when it could hold a value. Beside it a cell holds ``SKIPPED`` (no
#: item could bring a value, for a branch not taken) or ``None`` (none could, for a blank).
COULD: Final = _Could()


class Reader(abc.ABC):
    """Reads what a node's inputs hold, and says what the node comes to: the node rule, over a way to read a
    cell.

    Attributes:
        caller_inputs: The caller's inputs, by address.
    """

    __slots__ = ("caller_inputs",)

    caller_inputs: Mapping[str, Any]

    @abc.abstractmethod
    def cell(self, cell: Cell, item_path: ItemPath) -> Any:
        """What ``cell`` holds for ``item_path`` as an edge brings it: a value, ``None`` for a blank, ``SKIPPED``
        for a branch not taken, or ``NOT_SETTLED``."""

    @abc.abstractmethod
    def gathered(self, plan: Gather, item_path: ItemPath) -> Any:
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
            FieldValueError: An edge names a field the record it brings doesn't have (``invalid_edge_value``);
                its ``ctx`` names the input.
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
            except FieldValueError as missing:
                missing.ctx = {"name": name, **missing.ctx}
                raise
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

    def best_of(self, reads: tuple[Cell, ...], item_path: ItemPath) -> Any:
        """What alternative edges bring one input: a value when one of them holds one, else ``NOT_SETTLED``
        while one isn't settled, else a blank, and ``SKIPPED`` when every branch wasn't taken.

        For one item at most one alternative holds a value, as they sit on opposite branches of one decision.
        For an item nobody knows several could, which is why this is the best of them and not the first.
        """
        best: Any = SKIPPED
        for cell in reads:
            value = self.cell(cell, item_path)
            if value is NOT_SETTLED:
                best = NOT_SETTLED
            elif value is None:
                if best is SKIPPED:
                    best = None
            elif value is not SKIPPED:
                return value
        return best


class ItemReader(Reader):
    """Reads what the units of a run hold for one item.

    Attributes:
        units: Each function's units.
        node_at: Each node of the run's graph, by its name.
        waits: Whether a gathering input waits until what is outside its items and every item is settled, as a
            call must. A host that reads a run as it stands doesn't: it is shown the items so far, with a blank
            where one isn't settled.
    """

    __slots__ = ("node_at", "units", "waits")

    def __init__(self, units: Units, caller_inputs: Mapping[str, Any], node_at: Mapping[str, NodeAtPath], *, waits: bool = True) -> None:
        self.units, self.caller_inputs, self.node_at, self.waits = units, caller_inputs, node_at, waits

    def cell(self, cell: Cell, item_path: ItemPath) -> Any:
        """What ``cell`` holds for ``item_path``: an ``Item``'s is the item of ``item_path``, and a field named
        on the edge is read from it.

        Raises:
            FieldValueError: The edge names a field the record doesn't have (``invalid_edge_value``).
        """
        depth = cell.depth
        unit = self.units[cell.node_id].get(item_path[:depth])
        if unit is None or type(unit) is Asking:
            return NOT_SETTLED
        value = unit.out[cell.output]
        if value is None or value is SKIPPED or type(cell) is Cell:
            return value
        if type(cell) is Item:
            if type(value) is Records:  # held by column: the cell an edge names, with no record built
                wanted = cell.field
                return value[item_path[depth]] if wanted is None else field_of(value.columns, wanted)[item_path[depth]]
            value = value[item_path[depth]]
        elif type(cell) is WholeSeries:  # the list as a reader receives one: no mark of a branch not taken in it
            return value.records() if type(value) is Records else [None if item is SKIPPED else item for item in value]
        field = cell.field  # type: ignore[attr-defined]  # an edge brings a Value, an Item or a WholeSeries, each with its field
        return value if field is None or value is None or value is SKIPPED else field_of(value, field)

    def gathered(self, plan: Gather, item_path: ItemPath) -> Any:
        """The items the gathering input of ``plan`` gets for ``item_path``, with ``None`` for an item that has
        no value, for a blank or a branch not taken.

        What is outside the items is asked first whether any item could bring a value. When none could, that
        is the answer for any number of items, none included. When one could, each item answers for itself.

        Returns:
            The list, one place per item of the Series; ``NOT_SETTLED`` until what is outside and every item
            is settled; ``SKIPPED`` when no item could bring a value for a branch not taken outside the items;
            ``None`` when none could for a blank there.
        """
        series_made = plan.reads[0]
        item_path = item_path[: series_made.depth]
        certain = AnyItemReader(self, series_made, item_path).could_bring(plan)
        if certain is not COULD and (self.waits or certain is not NOT_SETTLED):
            return certain
        series = self.cell(series_made, item_path)
        if series is NOT_SETTLED:  # a host reads the run as it stands: no item so far
            return []
        if series is None or series is SKIPPED:  # and what is outside isn't settled: the Series' own blank or branch not taken, so far
            return series
        edges = plan.sources
        if len(edges) == 1 and type(edges[0]) is Item:  # the Series itself: its items, or one field of each
            wanted = edges[0].field
            if type(series) is Records:
                return series.records() if wanted is None else field_of(series.columns, wanted)
            if wanted is None:
                return [None if item is SKIPPED else item for item in series] if any(item is SKIPPED for item in series) else series
            return [None if item is SKIPPED or item is None else field_of(item, wanted) for item in series]
        if plan.levels == 1:
            item_paths = [(*item_path, place) for place in range(len(series))]
        else:  # what an embedded graph offers of an output made several maps deep inside it: every item of every one
            item_paths, known = instances(self.units, self.node_at[edges[0].node_id].maps, item_path)
            if not known and self.waits:
                return NOT_SETTLED
        items = []
        one_edge = edges[0] if len(edges) == 1 else None
        for under in item_paths:
            value = self.cell(one_edge, under) if one_edge is not None else self.best_of(edges, under)
            if value is NOT_SETTLED:
                if self.waits:
                    return NOT_SETTLED
                value = None
            items.append(None if value is SKIPPED else value)
        return items

    def kept_in_step(self, node: FunctionAtPath, received: CallInputs) -> CallInputs:
        """``received`` with each gathering input holding the items that have a value in every input of the
        node that gathers the same Series, so those inputs stay in step; an input whose items may be blank
        keeps a blank in its place."""
        for inputs in node.gathered_in_step:
            together = [(name, received[name], keeps_blanks) for name, keeps_blanks in inputs if received[name] is not None]
            for name, items, keeps_blanks in together:
                if len(together) == 1:  # by identity: an item may be a frame, which has no plain ==
                    received[name] = items if keeps_blanks else [item for item in items if item is not None]
                else:
                    received[name] = [item for place, item in enumerate(items) if all(keeps or others[place] is not None for _, others, keeps in together)]
        return received


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

    def __init__(self, items: ItemReader, series: Cell, item_path: ItemPath) -> None:
        self.items, self.series, self.item_path, self.caller_inputs = items, series, item_path, items.caller_inputs
        self.came_to: dict[str, CallInputs | Skipped | Blank | Literal[NotReady.NOT_SETTLED]] = {}

    def could_bring(self, plan: Gather) -> Any:
        """Whether any item of the Series could bring the gathering input of ``plan`` a value, going by what
        is outside the items. Each function the input gathers through is asked what it comes to for any item,
        in order, and then the input itself.

        Returns:
            ``COULD``; ``SKIPPED`` when none could for a branch not taken; ``None`` when none could for a
            blank; ``NOT_SETTLED`` while something outside the items isn't settled.
        """
        for node in plan.through:
            self.came_to[node.name] = self.comes_to(node, self.item_path)
        return self._could_gather(plan)

    def cell(self, cell: Cell, item_path: ItemPath) -> Any:
        """What ``cell`` holds for an item nobody knows: what its node comes to for any item when it is made
        per item, and as it stands when it is made outside the items, ``COULD`` for a value."""
        came_to = self.came_to.get(cell.node_id)
        if came_to is None:  # made outside the items
            series = self.series
            if type(cell) is Item and cell.node_id == series.node_id and cell.output == series.output:
                cell = series  # an item of the Series itself is what the Series is: one never made has none
            value = self.items.cell(cell, self.item_path)
            return value if value is NOT_SETTLED or value is SKIPPED or value is None else COULD
        if came_to is NOT_SETTLED:
            return NOT_SETTLED
        return COULD if type(came_to) is dict else came_to.out[cell.output]  # type: ignore[union-attr]  # mypy doesn't narrow a union by ``type(x) is dict``

    def gathered(self, plan: Gather, item_path: ItemPath) -> Any:
        """What a gathering input met on the way holds for an item nobody knows. One whose Series is made per
        item is asked the same of the same items; one whose Series is made outside them is a cell outside like
        any other: its own, whole answer, read as it stands."""
        if plan.reads[0].node_id in self.came_to:
            return self._could_gather(plan)
        value = self.items.gathered(plan, self.item_path)
        return value if value is NOT_SETTLED or value is SKIPPED or value is None else COULD

    def kept_in_step(self, node: FunctionAtPath, received: CallInputs) -> CallInputs:
        """``received`` as it is: nothing is known of the items to keep in step."""
        return received

    def _could_gather(self, plan: Gather) -> Any:
        """What a gathering input holds for an item nobody knows. It needs its Series and what its edges bring
        for an item, so it holds the worst of them, in the order the node rule has for a node's inputs. An
        input whose items take a blank receives the blank an edge brings."""
        edges = plan.sources
        brought = self.cell(edges[0], self.item_path) if len(edges) == 1 else self.best_of(edges, self.item_path)
        if brought is None and plan.spec.keeps_blanks:
            brought = COULD
        needed = [*[self.cell(series, self.item_path) for series in plan.reads[: plan.levels]], brought]
        for worst in (SKIPPED, NOT_SETTLED, None):
            if any(answer is worst for answer in needed):
                return worst
        return COULD


def field_of(value: Any, name: str) -> Any:
    """One field of a record: a mapping's key, else an attribute; of records held by column, that column.

    Raises:
        FieldValueError: The record has no such field (``invalid_edge_value``). Only a record whose type
            declares no fields can lack one: a field a type declares is checked when the graph loads.
    """
    try:
        return value[name] if type(value) is dict or isinstance(value, Mapping) else getattr(value, name)
    except (KeyError, AttributeError):
        raise FieldValueError(f"it has no field '{name}'", code="invalid_edge_value", field=name) from None


