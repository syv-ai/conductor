"""``CompiledNode`` and ``CompiledField``: what compile learned about one node, and about one of its inputs or outputs.

A node compile could not fully work out says so by what it lacks, never by
raising: ``interface`` is ``None`` when compile doesn't know what fields it
has, and ``state`` says whether it worked out its connections.
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

from conductor._internal.codec import cell_validator_for, json_schema_of, to_jsonable
from conductor._internal.execution.run_state import Question
from conductor._internal.field_records import was_titled_by_its_author
from conductor._internal.graph.output_conditions import ALWAYS, run_once_condition
from conductor._internal.graph.receive_modes import FixedValue
from conductor._internal.graph.run_plan import NodeRunPlan, plan_of_node
from conductor._internal.node_definition import NodeVersion
from conductor._internal.node_signature import build_call_validator

if TYPE_CHECKING:
    from conductor._internal.errors import Problem
    from conductor._internal.execution.values_of_run import HeldAs
    from conductor._internal.field_records import Input
    from conductor._internal.graph.binding import Binding
    from conductor._internal.graph.compiled_graph import CompiledGraph
    from conductor._internal.graph.graph_records import GraphNode
    from conductor._internal.graph.output_conditions import Condition
    from conductor._internal.graph.receive_modes import ReceiveMode
    from conductor._internal.index import Index
    from conductor._internal.node_definition import NodeDefinition
    from conductor._internal.node_signature import Interface
    from conductor._internal.ref import Ref

#: How far compile got with a node, read off what it holds, for the wire.
#:
#: ``ready``: compile worked out its connections. A graph can
#: run only when every node in it is ready.
#:
#: ``wiring_failed``: compile knows its inputs and outputs (``interface``) but
#: not its connections: one of its edges is broken, a node upstream of it is,
#: or the embedded graph it stands for can't be embedded. An editor draws the
#: node with its fields and marks the problem that ``cause`` names.
#:
#: ``resolution_failed``: compile doesn't know what fields it has: its type or
#: version is unknown, it sits on a cycle, its id can't name a node or its
#: ``compute_added_inputs`` raised on the statics set on it. An editor draws a bare
#: box with its problem.
NodeState = Literal["ready", "wiring_failed", "resolution_failed"]




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
    #: The fatal problem that explains a node that isn't ``ready``: its own, or the one upstream of it.
    cause: Problem | None = field(repr=False)
    #: The definition its type resolved to, and the version of it it uses; ``None`` when neither could be found.
    definition: NodeDefinition[..., Any] | None = field(repr=False)
    version: NodeVersion | None = field(repr=False)
    #: The inputs and outputs this node actually has, with the types its
    #: edges gave them; ``None`` when compile doesn't know them.
    interface: Interface | None = field(repr=False)
    #: The values set directly on this node, each read through its field's
    #: declared type. An input left to its default is absent. Empty when
    #: compile couldn't read them. On a graph: by inner address (``check.amount``).
    statics: Mapping[str, Any] = field(repr=False)
    #: Every input and output by name, and each column a table output offers (``rows.email``);
    #: ``None`` when compile couldn't work out its connections. Empty on an embedded graph: its fields are the nodes' inside it.
    _fields: Mapping[str, CompiledField] | None = field(repr=False)
    #: The rows the node runs once per row of, ``None`` when it runs once or isn't ready. A host reads ``CompiledGraph.run_plan``.
    _runs_per_row_of: Index | None = field(repr=False)

    def __init__(
        self, id: str, graph_node: GraphNode, embedded_in: str | None, problems: tuple[Problem, ...], cause: Problem | None,
        definition: NodeDefinition[..., Any] | None, version: NodeVersion | None, interface: Interface | None,
        statics: Mapping[str, Any], fields: Mapping[str, CompiledField] | None, runs_per_row_of: Index | None,
    ) -> None:
        self.__dict__.update(
            id=id, graph_node=graph_node, embedded_in=embedded_in, problems=problems, cause=cause, definition=definition,
            version=version, interface=interface, statics=statics, _fields=fields, _runs_per_row_of=runs_per_row_of,
        )

    @property
    def state(self) -> NodeState:
        """How far compile got with this node (``NodeState``)."""
        if self._fields is not None:
            return "ready"
        return "wiring_failed" if self.interface is not None else "resolution_failed"

    @property
    def embedded_graph(self) -> CompiledGraph | None:
        """The compiled graph this node stands for, ``None`` on a node that is a function."""
        return None if self.version is None else self.version.embedded_graph

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
        bindings, its run plan, which inputs take a blank and each output's
        type. A run's state stores one per node, and a leg restored into a
        graph whose fingerprint differs runs the node again: a static edited,
        a version bumped, an edge moved or a signature that moves the node to
        other rows or lets an input take a blank all change it; a title or a
        position does not. ``None`` on a node that doesn't run as one unit."""
        if self._fields is None or self.embedded_graph is not None:
            return None
        node, interface = self.graph_node, self.interface
        assert interface is not None
        bindings: dict[str, Any] = {}
        for name, binding in node.bindings.items():
            if name in self.statics:
                # The value as its type writes it, never as the author spelled
                # it: ``2`` and ``2.0`` on a number are one value, and a graph
                # built in Python hashes like the stored graph with its JSON.
                bindings[name] = {"static": to_jsonable(self.statics[name])}
            else:
                bindings[name] = binding.model_dump()
        used = {
            "type": node.type,
            "version": node.version,
            "bindings": bindings,
            "plan": self._run_plan,
            "takes_blank": {input_.name: input_.takes_blank for input_ in interface.inputs},
            "outputs": self._outputs_described,
        }
        return hashlib.sha256(json.dumps(to_jsonable_python(used), sort_keys=True).encode("utf-8")).hexdigest()

    @cached_property
    def _outputs_described(self) -> dict[str, Any]:
        """Each output's type, as a state's fingerprint and a unit's input hash both name it."""
        assert self.interface is not None
        return {out.name: {**out.dtype.describe(), "many": out.many} for out in self.interface.outputs}

    @cached_property
    def _input_hash_seed(self) -> bytes:
        """What a unit's input hash starts from: what the node *is*, apart from what
        it receives. A different function behind the same inputs is different work;
        how its inputs get there is not (the fingerprint holds that)."""
        assert self.interface is not None
        node = self.graph_node
        identity = {"type": node.type, "version": node.version, "inputs": [input_.name for input_ in self.interface.inputs], "outputs": self._outputs_described}
        return hashlib.blake2b(json.dumps(to_jsonable_python(identity), sort_keys=True).encode("utf-8"), digest_size=16).digest()

    @cached_property
    def _questions(self) -> tuple[Question, ...]:
        """What a person is asked when this node asks, one record per output:
        built once, shared by every unit, and never written to."""
        assert self.interface is not None
        return tuple(
            Question(
                name=out.name, title=self._title_of_a_question(out), description=out.description, dtype=out.dtype.describe(), many=out.many,
                json_schema=json_schema_of(out.dtype, out.many), optional=out.takes_blank and out.exclusive_group is None, exclusive_group=out.exclusive_group,
            )
            for out in self.interface.outputs
        )

    def _title_of_a_question(self, out: Any) -> str:
        """An output nobody titled, on a node with one output, reads as the node's own title rather than ``result``."""
        untitled = not was_titled_by_its_author(out) and len(self.interface.outputs) == 1  # type: ignore[union-attr]
        return (self.graph_node.title or self.definition.title) if untitled and self.definition is not None else out.title

    @cached_property
    def _validate_call_inputs(self) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        """The check a call goes through (``build_call_validator``), built on first use.

        An input left to a static or a default is not in it: compile read that value once."""
        fixed = {input_.name for input_ in self._inputs if self._fields[input_.name]._receive_mode.__class__ is FixedValue}  # type: ignore[union-attr]  # only a ready node is called
        check = build_call_validator(tuple(input_ for input_ in self._inputs if input_.name not in fixed))
        if not fixed:
            return check
        return lambda values: {**values, **check(values)}

    @cached_property
    def _inputs_rejecting_a_blank(self) -> tuple[str, ...]:
        """The one-value inputs on which a blank means this node isn't called."""
        return tuple(input_.name for input_ in self._inputs if not input_.many and not input_.takes_blank)

    @cached_property
    def _series_inputs_leaving_blanks_out(self) -> tuple[str, ...]:
        """The series inputs whose blank cells, row by row, are left out of the call."""
        return tuple(input_.name for input_ in self._inputs if input_.many and not input_.takes_blank)


@dataclass(frozen=True, eq=False, init=False)
class CompiledField:
    """One input or output of a node whose connections compile worked out.

    Built once, while compile walks the node, and handed back by
    ``CompiledGraph.field(ref)`` on every call.
    """

    #: The field's address: ``Ref("approve/check", "amount")`` for a field inside an embedded graph.
    ref: Ref
    #: The type of one value on this field. Whether it holds many is ``many``.
    dtype: Any
    #: Where this input's value comes from: ``From``, ``Static``, or ``None``
    #: when its declared default applies. Always ``None`` on an output.
    binding: Binding | None
    #: Under which condition this output appears: the decisions on nodes that run once
    #: (``ALWAYS`` when none holds it back). Always ``None`` on an input.
    condition: Condition | None
    #: Every problem about this field.
    problems: tuple[Problem, ...] = field(repr=False)
    #: Where its rows come from, for a field that holds many values; ``None`` otherwise.
    _index: Index | None = field(repr=False)
    #: How a unit of its node receives it; ``None`` exactly on an output.
    _receive_mode: ReceiveMode | None = field(repr=False)
    #: How a run holds this field's values (``conductor._internal.execution.values_of_run``).
    _held_as: HeldAs = field(repr=False)
    #: For an output: ``None`` is a value it may be given.
    _takes_blank: bool = field(repr=False)
    #: ``condition`` with the per-row decisions as well, each naming the rows it is decided on: what can keep two fields off one row.
    _condition_with_rows: Condition | None = field(repr=False)
    #: For an output or a list set on a node: the indexes its rows can trace back to, ``None`` when
    #: that can't be known (``conductor._internal.graph.row_origins``).
    _row_origins: frozenset[str] | None = field(repr=False)

    def __init__(
        self, ref: Ref, dtype: Any, binding: Binding | None, problems: tuple[Problem, ...], *,
        _condition_with_rows: Condition | None, _index: Index | None, _receive_mode: ReceiveMode | None, _held_as: HeldAs,
        _takes_blank: bool, _row_origins: frozenset[str] | None,
    ) -> None:
        """Compile builds one per field of a graph, so this sets them in one step, not one at a time."""
        self.__dict__.update(
            ref=ref, dtype=dtype, binding=binding, condition=_condition_with_rows if _condition_with_rows is None or _condition_with_rows is ALWAYS else run_once_condition(_condition_with_rows), problems=problems,
            _condition_with_rows=_condition_with_rows, _index=_index, _receive_mode=_receive_mode, _held_as=_held_as, _takes_blank=_takes_blank, _row_origins=_row_origins,
        )

    @property
    def _is_output(self) -> bool:
        return self._receive_mode is None

    @property
    def many(self) -> bool:
        """Does this field hold many values, one per row? An output of a node that runs once per row does, even where its declaration says one; an input does when it is collected or holds a whole list."""
        return self._index is not None or (self._receive_mode is not None and self._receive_mode.receive_mode in ("collected", "entire_value"))

    @cached_property
    def _validate_cell(self) -> Callable[[Any], Any]:
        """The one check a value of this output passes to enter a run's state
        (``conductor._internal.codec.cell_validator_for``), returned by a node, answered by a
        person or read back from a stored state. A blank passes where the
        output's type says ``| None``."""
        read = cell_validator_for(self.dtype)
        return (lambda value: None if value is None else read(value)) if self._takes_blank else read
