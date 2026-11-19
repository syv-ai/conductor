"""The outcomes of a run: the outcome of every call so far, and the arguments of a call worked out from them.

The example graph in this file::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b"]
    shout(line: str, mark: str) -> str       # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!"]

A run stores one outcome per call (``outcome.py`` has the four kinds), by the node's path and then by the call's
item indexes, one index per loop the node is inside. After the run, ``RunOutcomes`` has::

    ("lines",): {():   Done({"result": ["a", "b"]})}
    ("shout",): {(0,): Done({"result": "A!"}), (1,): Done({"result": "B!"})}
    ("join",):  {():   Done({"result": "A!+B!"})}

``RunOutcomes.arguments_for(shout, (1,))`` returns the arguments of ``shout``'s call for the second line,
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

To ask that question, ``arguments_for`` is called again for ``shout``, with item indexes one index short:
``()`` instead of ``(0,)`` or ``(1,)``. ``mark`` is read as usual, since its node is inside no loop. ``line`` is
one item of a list, and no index says which one. So once the list exists, ``line`` reads as
``MAY_HAVE_VALUE``. The result is ``Skipped`` when ``mark`` is ``SKIPPED``, and arguments when some call of
``shout`` may be made. The rule is written out in ``python -m conductor.about one-or-many``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal, Self, assert_never

import polars as pl
from typing_extensions import sentinel

from conductor._internal.base import SKIPPED, ItemIndexes, NodePath
from conductor._internal.graph import CollectingInput, Edge, EdgeInput
from conductor._internal.problem import InvalidCallError, run_problems
from conductor._internal.types import NoSuchFieldError, field_of

from .blueprint import BlueprintFunctionNode, RunBlueprint, loop_body_of
from .outcome import NoneInput, NotReady, Outcome, Skipped

#: Returned in place of a value when the outcomes are asked about any item of a loop, and some item may have a
#: value. When no item can have one, what every item would get is returned instead: ``SKIPPED`` or ``None``.
MAY_HAVE_VALUE = sentinel("MAY_HAVE_VALUE")


class RunOutcomes:
    """The outcome of every call of a run so far, and the arguments of each call worked out from them.

    A run holds one, and never changes it. ``resume()`` gives the scheduler a ``copy()`` to ``record`` the new
    outcomes in, and the run it returns holds a copy of that. ``with_answers`` and ``with_decline`` make theirs with
    ``replacing``. A copy shares each node's outcomes with the table it was copied from, until one of the two
    records an outcome of that node: then it gets a dict of its own first. So a copy costs one step per node.

    Every copy has each node's outcomes in item order. Calls on threads come back in any order, so the table
    being recorded into may hold them out of order until it is copied.

    Attributes:
        blueprint: The blueprint of the run's graph: every function node, by node path.
        graph_inputs: The values the caller gave for the graph inputs, by name (``"text"``). It is read-only.
        for_any_item: Empty in the outcomes a run holds. The outcomes made to ask "could any item of this loop
            have a value?" (``_for_any_item_of``) have here what ``arguments_for`` returned for each node inside
            the loop.
    """

    __slots__ = ("_by_node", "_out_of_order", "_own", "blueprint", "for_any_item", "graph_inputs")

    def __init__(self, blueprint: RunBlueprint, graph_inputs: Mapping[str, Any]) -> None:
        """Makes the outcomes of a run that has made no call yet."""
        self._shaped(
            blueprint,
            MappingProxyType(dict(graph_inputs)),
            {path: {} for path in blueprint.function_nodes},
            set(blueprint.function_nodes),
        )

    def _shaped(
        self,
        blueprint: RunBlueprint,
        graph_inputs: Mapping[str, Any],
        by_node: dict[NodePath, dict[ItemIndexes, Outcome]],
        own: set[NodePath],
    ) -> None:
        """Sets every attribute. ``__init__``, ``copy`` and ``sharing`` make their tables with it.

        Args:
            blueprint: The blueprint of the run's graph.
            graph_inputs: The graph inputs, read-only.
            by_node: The outcomes of each function node, by node path and then by item indexes.
            own: The nodes whose dict of outcomes no other table shares, so ``record`` may write into it.
        """
        self.blueprint, self.graph_inputs, self._by_node, self._own = (
            blueprint,
            graph_inputs,
            by_node,
            own,
        )
        #: The nodes whose outcomes ``record`` wrote out of item order. ``copy`` sorts them.
        self._out_of_order: set[NodePath] = set()
        self.for_any_item: dict[
            BlueprintFunctionNode,
            dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY],
        ] = {}

    @classmethod
    def sharing(cls, outcomes: RunOutcomes) -> Self:
        """Returns a table of the same outcomes as ``outcomes``, sharing every node's dict, for reading only.

        ``results.py`` reads a run's outcomes as a host is shown them this way, and ``_for_any_item_of`` reads them
        for any item of a loop.
        """
        shared = object.__new__(cls)
        shared._shaped(outcomes.blueprint, outcomes.graph_inputs, outcomes._by_node, set())
        return shared

    # -- the outcomes ------------------------------------------------------------------------------------------

    def of(self, path: NodePath) -> Mapping[ItemIndexes, Outcome]:
        """Returns the outcomes of the function node at ``path``, by item indexes.

        ``of(("shout",))`` is ``{(0,): Done({"result": "A!"}), (1,): Done({"result": "B!"})}`` in the example. A call
        with no outcome has no key. The result is in item order, except in a table being recorded into.
        """
        return self._by_node[path]

    def __len__(self) -> int:
        """Returns how many calls have an outcome."""
        return sum(len(of_node) for of_node in self._by_node.values())

    def calls[O](self, kind: type[O]) -> Iterator[tuple[BlueprintFunctionNode, ItemIndexes, O]]:
        """Yields every call whose outcome is of ``kind``, with its node and item indexes.

        They come in node order, and then in item order. ``calls(Unanswered)`` yields every call waiting for a
        person. A call inside a subgraph comes where its subgraph node is.
        """
        for node in self.blueprint.function_nodes.values():
            for item_indexes, outcome in self._by_node[node.node_path].items():
                if isinstance(outcome, kind):
                    yield node, item_indexes, outcome

    def record(
        self, node: BlueprintFunctionNode, item_indexes: ItemIndexes, outcome: Outcome
    ) -> None:
        """Stores the outcome of the call of ``node`` at ``item_indexes``, in place of any it had."""
        path = node.node_path
        of_node = self._own_dict(path)
        if item_indexes not in of_node and of_node and item_indexes < next(reversed(of_node)):
            self._out_of_order.add(path)
        of_node[item_indexes] = outcome

    def copy(self) -> RunOutcomes:
        """Returns a copy to record into, with every node's outcomes in item order. This table is left as it is."""
        by_node = dict(self._by_node)
        for path in self._out_of_order:
            by_node[path] = dict(sorted(by_node[path].items()))
        copied = object.__new__(RunOutcomes)
        copied._shaped(self.blueprint, self.graph_inputs, by_node, set(self._out_of_order))
        # From here on, neither this table nor the copy writes into a dict the other has.
        self._own = set()
        return copied

    def replacing(
        self, changes: Mapping[tuple[BlueprintFunctionNode, ItemIndexes], Outcome | None]
    ) -> RunOutcomes:
        """Returns a copy in which each call in ``changes`` has its new outcome. ``None`` takes a call's outcome away.

        ``with_answers`` replaces a waiting call's ``Unanswered`` with ``Done`` this way, and ``with_decline`` takes
        every ``Unanswered`` away.
        """
        copied = self.copy()
        for (node, item_indexes), outcome in changes.items():
            if outcome is not None:
                copied.record(node, item_indexes, outcome)
                continue
            del copied._own_dict(node.node_path)[item_indexes]
        return copied

    def _own_dict(self, path: NodePath) -> dict[ItemIndexes, Outcome]:
        """Returns the dict of outcomes of the node at ``path`` to write into, copying it first if another table shares it."""
        if path not in self._own:
            self._by_node[path] = dict(self._by_node[path])
            self._own.add(path)
        return self._by_node[path]

    def item_indexes(
        self, loops: tuple[tuple[NodePath, str], ...], prefix: ItemIndexes = ()
    ) -> tuple[list[ItemIndexes], bool]:
        """Returns the item indexes of every call a node inside ``loops`` will have, and whether that list is final yet.

        ``shout`` is inside one loop, over ``lines.result``. Once ``lines`` has returned two lines,
        ``item_indexes(shout.loops)`` returns ``([(0,), (1,)], True)``. Before ``lines`` has returned, it returns
        ``([], False)``: no call is known yet, and more will come. ``lines`` is inside no loop, so for it the result
        is always ``([()], True)``, one call.

        Args:
            loops: The loops the node is inside, outermost first. Each is the path of the node that returns the
                series, and that output's name.
            prefix: Only return the item indexes that start with these. ``()`` returns all of them. With two nested
                loops, ``(1,)`` returns the calls for item 1 of the outer loop: ``(1, 0)``, ``(1, 1)``, and so on.

        Returns:
            The item indexes of each call known so far, and ``True`` once every series the node loops over exists.
        """
        every_item_indexes, complete = [prefix], True
        for depth in range(len(prefix), len(loops)):
            series_node, output = loops[depth]
            of_series_node = self._by_node[series_node]
            one_loop_deeper: list[ItemIndexes] = []
            for item_indexes in every_item_indexes:
                outcome = of_series_node.get(item_indexes)
                if outcome is None or not outcome.has_outputs:
                    complete = False
                    continue
                one_loop_deeper.extend(
                    (*item_indexes, index) for index in range(length(outcome.outputs[output]))
                )
            every_item_indexes = one_loop_deeper
        return every_item_indexes, complete

    # -- the arguments of a call -------------------------------------------------------------------------------

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
                                input=name, reason=reason, record_field=missing.field
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
            return NoneInput(node.compiled_node.outputs_when_none_input, none_input)
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
        outcomes are asked about any item of a loop, several may have one.

        Returns:
            The first value an edge gives. Otherwise ``NOT_READY`` if any edge gave that. Otherwise ``None`` if any
            edge gave that. ``SKIPPED`` when every edge comes from a skipped branch.
        """
        best: Any = SKIPPED
        for edge in edges:
            value = self.read_output(
                self.blueprint.function_nodes[edge.node],
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
        outcome = self._by_node[node.node_path].get(item_indexes[:depth])
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
        series_node = self.blueprint.function_nodes[node_input.series[0]]
        depth, keeps_none_items = len(series_node.loops), node_input.field.keeps_none_items
        if len(item_indexes) < depth:
            return _may_have_value(
                self.read_edges(node_input.sources, item_indexes), keeps_none_items
            )
        item_indexes = item_indexes[:depth]
        loop = node_input.series
        sources = [self.blueprint.function_nodes[edge.node] for edge in node_input.sources]
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
        outcome = self._by_node[node_path].get(item_indexes)
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
    ) -> Self:
        """Returns outcomes that answer for any item of ``loop``: the same outcomes, read for any item.

        Its ``for_any_item`` has what ``arguments_for`` returns at ``item_indexes`` for each node inside the loop that
        ``sources`` depend on. Each node is worked out after the nodes upstream of it (``loop_body_of``).

        Args:
            loop: The list looped over: the path of the function node that returns it, and the output's name.
            sources: The function nodes a value is read from.
            item_indexes: The item indexes of the call that returned the list.
        """
        any_item = type(self).sharing(self)
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


def length(value: Any) -> int:
    """Returns how many items the value of a ``Series`` output has. A value of ``None`` or ``SKIPPED`` has 0."""
    return 0 if value is SKIPPED or value is None else len(value)
