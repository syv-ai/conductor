"""``CompiledNode``, ``Wiring`` and ``CompiledField``: what compile learned about one node, and about one of its inputs or outputs.

A node compile could not fully work out says so by what it lacks, never by
raising: ``interface`` is ``None`` when compile doesn't know what fields it
has, and ``wiring`` is ``None`` when it couldn't work out its connections.
``cause`` then names the one problem behind it.

This module knows nothing of the compiler or of ``CompiledGraph``: it holds
the answers, and compile puts them in. The terms it uses (interface, series,
embedded graph) are defined at the top of ``conductor._internal.graph.compiled_graph``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from pydantic_core import to_jsonable_python

from conductor._internal.codec import cell_validator_for, json_schema_of, to_wire
from conductor._internal.execution.run_state import Question
from conductor._internal.graph.run_plan import NodeRunPlan, plan_of_node
from conductor._internal.node_definition import NodeVersion
from conductor._internal.node_signature import build_call_validator
from conductor._internal.series import Series
from conductor._internal.table import is_table_or_series_of_tables

if TYPE_CHECKING:
    from conductor._internal.errors import Problem
    from conductor._internal.field_records import Input
    from conductor._internal.graph.binding import Binding
    from conductor._internal.graph.compiled_graph import CompiledGraph
    from conductor._internal.graph.graph_records import GraphNode
    from conductor._internal.graph.output_conditions import Condition
    from conductor._internal.graph.receive_modes import ReceiveMode
    from conductor._internal.node_definition import NodeDefinition
    from conductor._internal.node_signature import Interface
    from conductor._internal.ref import Ref
    from conductor._internal.series import Index

#: How far compile got with a node, read off what it holds, for the wire.
#:
#: ``ready``: compile worked out its connections (``wiring``). A graph can
#: run only when every node in it is ready.
#:
#: ``wiring_failed``: compile knows its inputs and outputs (``interface``) but
#: not its connections: one of its edges is broken, a node upstream of it is,
#: or the embedded graph it stands for can't be embedded. An editor draws the
#: node with its fields and marks the problem that ``cause`` names.
#:
#: ``resolution_failed``: compile doesn't know what fields it has: its type or
#: version is unknown, it sits on a cycle, its id can't name a node or its
#: ``compute_inputs`` raised on the statics set on it. An editor draws a bare
#: box with its problem.
NodeState = Literal["ready", "wiring_failed", "resolution_failed"]


def _static_as_json(value: Any, dtype: Any) -> Any:
    """A static as its field's type writes it, for ``CompiledNode.fingerprint``:
    each of many values through the element type, one value through its own."""
    if value is None:
        return None
    if dtype.element is not None:
        return [to_wire(item, dtype.element) for item in (value.values if isinstance(value, Series) else value)]
    return to_wire(value, dtype)


def _nullable_dtype(dtype: Any) -> Any:
    """The type a call reads an input that takes a blank as: a series' cells may be blank, one value may."""
    return Series[dtype.element | None] if dtype.element is not None else dtype | None


@dataclass(frozen=True, eq=False, slots=True)
class Wiring:
    """What a node's connections decided. Only a node compile fully worked out has one."""

    #: Every input and output by name, and each column a table output offers
    #: (``rows.email``). Empty on an embedded graph: its fields are the nodes' inside it.
    fields: Mapping[str, CompiledField]
    #: The rows the node runs once per row of, ``None`` when it runs once. The
    #: engine's own; a host reads ``CompiledGraph.run_plan``.
    _runs_per_row_of: Index | None = field(repr=False)


@dataclass(frozen=True, eq=False, init=False)
class CompiledNode:
    """What compile worked out about one node in a graph.

    ``CompiledGraph.node(id)`` hands back the same object every time. The
    engine reads it to run the node, an editor to draw it: ``problems`` and
    ``cause`` for the marks, ``interface`` for the fields where compile knows
    them, and ``CompiledGraph.run_plan`` for where it runs.
    """

    #: Its id in the graph that runs: its path, ``"approve/check"``, for an inner node.
    id: str
    #: The node as the author stored it: its type, version number, title and
    #: bindings. An inner node's has its path as its id.
    graph_node: GraphNode = field(repr=False)
    #: The id of the graph node this node sits inside, or ``None`` for a node the author drew.
    embedded_in: str | None = field(repr=False)
    #: Every problem about this node or one of its fields, in ``CompiledGraph.problems`` order.
    problems: tuple[Problem, ...] = field(repr=False)
    #: The fatal problem that explains a node with no ``wiring``: its own, or the one upstream of it.
    cause: Problem | None = field(repr=False)
    #: The definition its type resolved to, and the version of it it uses; ``None`` when neither could be found.
    definition: NodeDefinition[..., Any] | None = field(repr=False)
    version: NodeVersion | CompiledGraph | None = field(repr=False)
    #: The inputs and outputs this node actually has, with the types its
    #: edges gave them; ``None`` when compile doesn't know them.
    interface: Interface | None = field(repr=False)
    #: The values set directly on this node, each read through its field's
    #: declared type. An input left to its default is absent. Empty when
    #: compile couldn't read them. On a graph: by inner address (``check.amount``).
    statics: Mapping[str, Any] = field(repr=False)
    #: What its connections decided; ``None`` when compile couldn't work them out.
    wiring: Wiring | None = field(repr=False)

    def __init__(
        self, id: str, graph_node: GraphNode, embedded_in: str | None, problems: tuple[Problem, ...], cause: Problem | None,
        definition: NodeDefinition[..., Any] | None, version: NodeVersion | CompiledGraph | None, interface: Interface | None,
        statics: Mapping[str, Any], wiring: Wiring | None,
    ) -> None:
        self.__dict__.update(
            id=id, graph_node=graph_node, embedded_in=embedded_in, problems=problems, cause=cause, definition=definition,
            version=version, interface=interface, statics=statics, wiring=wiring,
        )

    @property
    def state(self) -> NodeState:
        """How far compile got with this node (``NodeState``)."""
        if self.wiring is not None:
            return "ready"
        return "wiring_failed" if self.interface is not None else "resolution_failed"

    @property
    def embedded_graph(self) -> CompiledGraph | None:
        """The compiled graph this node stands for, ``None`` on a node that is a function."""
        return None if self.version is None or isinstance(self.version, NodeVersion) else self.version

    @property
    def _inputs(self) -> tuple[Input, ...]:
        assert self.interface is not None
        return self.interface.inputs

    @cached_property
    def _run_plan(self) -> NodeRunPlan:
        return plan_of_node(self)

    @cached_property
    def fingerprint(self) -> str | None:
        """A hash of how the graph uses this node: its type, version and
        bindings, its run plan and each output's type. A run's state stores
        one per node, and a leg restored into a graph whose fingerprint
        differs runs the node again: a static edited, a version bumped, an
        edge moved or a signature that moves the node to other rows all
        change it; a title or a position does not. ``None`` on a node that
        doesn't run as one unit."""
        if self.wiring is None or self.embedded_graph is not None:
            return None
        node, interface = self.graph_node, self.interface
        assert interface is not None
        fields = self.wiring.fields
        bindings: dict[str, Any] = {}
        for name, binding in node.bindings.items():
            if name in self.statics:
                # The value as its type writes it, never as the author spelled
                # it: ``2`` and ``2.0`` on a number are one value, and a graph
                # built in Python hashes like the stored graph with its JSON.
                bindings[name] = {"static": _static_as_json(self.statics[name], fields[name].type)}
            else:
                bindings[name] = binding.model_dump()
        used = {
            "type": node.type,
            "version": node.version,
            "bindings": bindings,
            "plan": self._run_plan,
            "outputs": {out.name: out.dtype.describe() for out in interface.outputs},
        }
        return hashlib.sha256(json.dumps(to_jsonable_python(used), sort_keys=True).encode("utf-8")).hexdigest()

    @cached_property
    def _questions(self) -> tuple[Question, ...]:
        """What a person is asked when this node asks, one record per output:
        built once, shared by every unit, and never written to."""
        assert self.interface is not None
        return tuple(
            Question(
                name=out.name, title=self._title_of_a_question(out), description=out.description, dtype=out.dtype.describe(),
                json_schema=json_schema_of(out.dtype), optional=out.takes_blank and out.choice is None, choice=out.choice,
            )
            for out in self.interface.outputs
        )

    def _title_of_a_question(self, out: Any) -> str:
        """An output nobody titled, on a node with one output, reads as the node's own title rather than ``result``."""
        untitled = out.title == out.name and len(self.interface.outputs) == 1  # type: ignore[union-attr]
        return (self.graph_node.title or self.definition.title) if untitled and self.definition is not None else out.title

    @cached_property
    def _validate_call_inputs(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """The check a call goes through (``build_call_validator``), built on first use. An
        input that takes a blank is read as its type or ``None``."""
        return build_call_validator(tuple(
            inp.model_copy(update={"dtype": _nullable_dtype(inp.dtype)}) if inp.takes_blank else inp for inp in self._inputs
        ))

    @cached_property
    def _inputs_rejecting_a_blank(self) -> tuple[str, ...]:
        """The one-value inputs on which a blank means this node isn't called."""
        return tuple(inp.name for inp in self._inputs if inp.dtype.element is None and not inp.takes_blank)

    @cached_property
    def _series_inputs_leaving_blanks_out(self) -> tuple[str, ...]:
        """The series inputs whose blank cells, row by row, are left out of the call."""
        return tuple(inp.name for inp in self._inputs if inp.dtype.element is not None and not inp.takes_blank)


@dataclass(frozen=True, eq=False, init=False)
class CompiledField:
    """One input or output of a node whose connections compile worked out.

    Built once, while compile walks the node, and handed back by
    ``CompiledGraph.field(ref)`` on every call.
    """

    #: The field's address: ``Ref("approve/check", "amount")`` for a field inside an embedded graph.
    ref: Ref
    #: The type that travels on this field. An output of a node that runs once
    #: per row carries ``Series[X]`` even where its declaration says ``X``.
    type: Any
    #: Where this input's value comes from: ``From``, ``Static``, or ``None``
    #: when its declared default applies. Always ``None`` on an output.
    binding: Binding | None
    #: Under which condition this output appears (``ALWAYS`` when no decision holds it back). Always ``None`` on an input.
    condition: Condition | None
    #: Every problem about this field.
    problems: tuple[Problem, ...] = field(repr=False)
    #: Where its rows come from, for a field that carries a series; ``None`` otherwise.
    _index: Index | None = field(repr=False)
    #: How a unit of its node receives it; ``None`` exactly on an output.
    _receive_mode: ReceiveMode | None = field(repr=False)
    #: An input the author set a list on, so its node runs once per
    #: value; an edge may read the values as a column of those rows.
    _is_list_on_node: bool = field(repr=False)
    #: The rows this output makes: a series output or a table output. ``None`` on any other field.
    _makes_rows: Index | None = field(repr=False)
    #: For a column of a table output offered as a field: the table output and
    #: the column. Its values are read from the table, never stored one by one.
    _column_of_table: tuple[Ref, str] | None = field(repr=False)
    #: For an output: ``None`` is a value it may be given.
    _takes_blank: bool = field(repr=False)
    #: What a person reads it as, and the name it comes into a merged table under:
    #: an output's or a list's title, a table column's own name.
    _title: str = field(repr=False)
    #: For an output or a list set on a node: the indexes its rows can trace back to, ``None`` when
    #: that can't be known (``conductor._internal.graph.row_origins``).
    _row_origins: frozenset[str] | None = field(repr=False)

    def __init__(
        self, ref: Ref, type: Any, binding: Binding | None, condition: Condition | None, problems: tuple[Problem, ...], *,
        _index: Index | None, _receive_mode: ReceiveMode | None, _is_list_on_node: bool, _makes_rows: Index | None, _column_of_table: tuple[Ref, str] | None,
        _takes_blank: bool, _title: str, _row_origins: frozenset[str] | None,
    ) -> None:
        """Compile builds one per field of a graph, so this sets them in one step, not one at a time."""
        self.__dict__.update(
            ref=ref, type=type, binding=binding, condition=condition, problems=problems, _index=_index, _receive_mode=_receive_mode,
            _is_list_on_node=_is_list_on_node, _makes_rows=_makes_rows, _column_of_table=_column_of_table, _takes_blank=_takes_blank, _title=_title, _row_origins=_row_origins,
        )

    @property
    def _is_output(self) -> bool:
        return self._receive_mode is None

    @cached_property
    def _cell_type(self) -> Any:
        """The type of one value of this field: the element of a series on rows, the field's own type otherwise."""
        return self.type if self._index is None or self.type.element is None else self.type.element

    @cached_property
    def _validate_cell(self) -> Callable[[Any], Any]:
        """The one check a value of this output passes to enter a run's state
        (``conductor._internal.codec.cell_validator_for``), returned by a node, answered by a
        person or read back from a stored state. A blank passes where the
        output's type says ``| None``."""
        read = cell_validator_for(self._cell_type)
        return (lambda value: None if value is None else read(value)) if self._takes_blank else read

    @cached_property
    def _unit_writes_many_cells(self) -> bool:
        """One unit writes many cells here, one per row it makes: a series output, not a table."""
        return self._makes_rows is not None and not is_table_or_series_of_tables(self.type)
