"""``CompiledNode`` and ``CompiledField``: what compile learned about one node, and about one of its inputs or outputs.

``CompiledGraph`` (in ``conductor.graph.compiled``) builds each once and
hands the same value back from ``node(node_id)`` and ``field(ref)``. They
live here, apart from the graph, so the values can be read and changed
without the walk that fills them in. The words they use — interface,
series, index, embedded graph — are defined at the top of
``conductor.graph.compiled``.

This module knows nothing of the compiler or of ``CompiledGraph``: it holds
the answers, and the graph's compile puts them in.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from conductor.codec import to_wire
from conductor.errors import NodeKindError, NodeResolutionError, NodeWiringError
from conductor.graph.binding import Static
from conductor.graph.conditions import substituted
from conductor.interface import model_of
from conductor.node import runner_of
from conductor.series import Series

if TYPE_CHECKING:
    from conductor.graph.binding import Binding
    from conductor.graph.conditions import Condition
    from conductor.graph.model import GraphNode
    from conductor.graph.problem import Problem
    from conductor.graph.receive import Receive
    from conductor.interface import Interface
    from conductor.node import GraphVersion, NodeDefinition, NodeVersion
    from conductor.ref import Ref
    from conductor.series import Index

#: How far compile got with a node, and so what an editor can show for it.
#:
#: ``ready``: compile worked out everything about the node — its inputs and
#: outputs, and what its connections decide: the type that arrives on each
#: field and whether it runs once per row. A graph can run only when every
#: node in it is ready.
#:
#: ``wiring_failed``: compile knows the node itself — its inputs and outputs,
#: its version, the values typed into it — but could not work out its
#: connections. One of its own edges is broken, a node upstream of it is
#: broken, or the embedded graph around it does not line up. An editor draws
#: the node with its fields and marks the broken connection; the author fixes
#: the edge, or the node upstream, that its problem names.
#:
#: ``resolution_failed``: compile could not make sense of the node at all,
#: so it does not know what inputs or outputs it has. Its type or version is
#: unknown, it sits on a cycle, its id was refused, or its ``compute_inputs``
#: refused the values typed into it. An editor draws it as a bare box with
#: its problem; the author replaces the node, breaks the cycle, or changes
#: the values its problem names.
NodeState = Literal["ready", "wiring_failed", "resolution_failed"]

#: What a node is. ``node``: it runs as one unit, one call of its version's
#: ``run`` per row. ``graph``: its version is a graph, which compile
#: inlines, so its inner nodes run in its place and an address on it reads
#: through to theirs. The fold sets it from the version. It is read where
#: the two kinds answer differently: ``_gate``, for the reads only a unit
#: answers, and the address walk (``CompiledGraph.expanded`` and ``field``),
#: where an address on a graph reads through. Which nodes are graphs is
#: expand's to say (``Expansion.graphs``), since inlining is what differs.
NodeKind = Literal["node", "graph"]


def _written(value: Any, dtype: Any, listed: bool) -> Any:
    """A static as its declared type writes it, for ``CompiledNode.fingerprint``:
    each of many typed-in values (``listed``) through the scalar type, a
    series through its element type, one value through its own."""
    element = getattr(dtype, "element", None)
    if element is not None:
        return [to_wire(item, element) for item in (value.values if isinstance(value, Series) else value)]
    if listed:
        return [to_wire(item, dtype) for item in value]
    return to_wire(value, dtype)


def _gate(node: CompiledNode, asked: str, *, needs_wiring: bool, needs_one_unit: bool = False) -> None:
    """Raise when ``node`` cannot answer ``asked``.

    Everything gated needs a node compile could resolve. ``needs_wiring``
    also asks for a ready node, whose connections compile worked out;
    ``needs_one_unit`` asks for a node that runs as one unit rather than as
    an embedded graph. Only a ready node has no ``_cause``."""
    cause = node._cause
    if cause is not None and node.state == "resolution_failed":
        raise NodeResolutionError(node.id, asked, cause)
    if needs_one_unit and node._kind != "node":
        raise NodeKindError(node.id, asked, node._kind)
    if cause is not None and needs_wiring:
        raise NodeWiringError(node.id, asked, cause)


@dataclass(frozen=True, eq=False)
class CompiledNode:
    """One node as the compiler left it: what it has, what it holds, how it runs.

    Built once by ``CompiledGraph.from_graph`` and handed back by
    ``CompiledGraph.node(node_id)`` on every call, so it is equal only to
    itself: the same node compiled twice is two values. The engine reads
    ``interface``, ``validate``, ``statics``, ``runner`` and
    ``iterates_on`` for each node it runs, and ``version`` and
    ``graph_node`` for the policy and the type it reports; an editor reads
    ``state`` and ``kind``, then ``interface`` and ``iterates_on`` for each
    node it draws — a node whose ``kind`` is ``graph`` as a box around its
    inner nodes, from ``version.graph`` — and ``problems`` for its marks.
    Its sibling is ``CompiledField``, the same for one input or output.

    What it answers depends on its ``state``. ``id``, ``state``,
    ``graph_node``, ``embedded_in`` and ``problems`` answer always.
    ``kind``, ``definition``, ``version``, ``interface`` and ``statics``
    need a node compile could resolve, and raise ``NodeResolutionError``
    otherwise. ``iterates_on`` needs a ready one, and raises
    ``NodeWiringError`` for a node whose connections compile could not work
    out. Both errors hold, in ``problems``, the one ``Problem`` that
    explains the state. ``runner``, ``validate`` and ``fingerprint`` are the
    run's, and only a node that runs as one unit answers them: on a
    ``graph`` they raise ``NodeKindError``, since its inner nodes run in its
    place. A ``graph`` never fails to resolve, is ready only when every node
    inside it is, and has no fields of its own — an address on it
    (``Ref("approve", "check.amount")``) reads through to the inner field.
    """

    #: The expanded id: ``"approve/check"`` for an inner node.
    id: str
    #: How far compile got with it: ``ready``, ``wiring_failed`` or
    #: ``resolution_failed`` (see ``NodeState``).
    state: NodeState
    #: The node as the author stored it: its type, version number, title
    #: and bindings. An inner node's has its expanded id; a ``graph``'s
    #: bindings name inner addresses (``check.amount``).
    graph_node: GraphNode = field(repr=False)
    #: The id of the node whose embedded graph this node belongs to —
    #: ``"approve"`` for ``"approve/check"`` — or ``None`` for a node the
    #: author placed.
    embedded_in: str | None = field(repr=False)
    #: Every problem about this node or one of its fields, in
    #: ``CompiledGraph.problems`` order: what an editor marks on it.
    problems: tuple[Problem, ...] = field(repr=False)
    #: The fatal problem that explains a state other than ``ready``: the
    #: node's own, the one upstream of it, or the one on the embedded graph
    #: around it. ``None`` exactly when ready.
    _cause: Problem | None = field(repr=False)
    #: The answers a resolved node has; ``None`` on one whose resolution failed.
    _kind: NodeKind = field(repr=False)
    _definition: type[NodeDefinition] = field(repr=False)
    _version: NodeVersion | GraphVersion = field(repr=False)
    _interface: Interface = field(repr=False)
    _statics: Mapping[str, Any] = field(repr=False)
    #: What the walk decided; ``None`` also on a node that is not ready.
    _iterates_on: Index | None = field(repr=False)
    #: The read plan, the engine's alone: the walk's decisions inverted once
    #: at compile, so the ledger looks each up instead of searching the graph
    #: on every run. ``_reads``: every output an edge into this node reads,
    #: one per edge ref, with how the input it feeds receives it. An index's
    #: share sits on what births its rows — here, the index this node's
    #: series outputs birth (``_births``, ``None`` for a node with none): the
    #: nodes that run once per row of it (``_iterated_by``), the outputs that
    #: sit on it (``_carried_by``) and the inputs holding a typed-in list
    #: born under each of its rows (``_typed_lists``), each in execution
    #: order. Empty on a node the walk did not derive, and on a ``graph``.
    _births: Index | None = field(repr=False)
    _reads: tuple[tuple[Ref, Receive], ...] = field(repr=False)
    _iterated_by: tuple[str, ...] = field(repr=False)
    _carried_by: tuple[Ref, ...] = field(repr=False)
    _typed_lists: tuple[Ref, ...] = field(repr=False)
    #: Every input and output by name: what ``CompiledGraph.field`` hands back.
    _fields: Mapping[str, CompiledField] = field(repr=False)

    @property
    def definition(self) -> type[NodeDefinition]:
        """The class this node's type resolved to in the registry compile was
        given; ``version`` is one of its ``versions``. Its hooks and its
        runner are made from it, each on a fresh instance."""
        _gate(self, "definition", needs_wiring=False)
        return self._definition

    @property
    def kind(self) -> NodeKind:
        """What this node is: ``node``, run as one unit, or ``graph``, a node
        whose version is a graph, run as its inner nodes (``NodeKind``)."""
        _gate(self, "kind", needs_wiring=False)
        return self._kind

    @property
    def version(self) -> NodeVersion | GraphVersion:
        """The version this node uses: a ``NodeVersion`` — its ``run``,
        interface and policy — for a ``node``; a ``GraphVersion`` — its
        graph and declared interface — for a ``graph``."""
        _gate(self, "version", needs_wiring=False)
        return self._version

    @property
    def interface(self) -> Interface:
        """The inputs and outputs this node actually has, with every type the
        edges gave it — not merely what its version declared. The one place
        anything asks what a node has: the engine validates a call against
        it and an editor draws the fields from it."""
        _gate(self, "interface", needs_wiring=False)
        return self._interface

    @property
    def statics(self) -> Mapping[str, Any]:
        """The values set directly on this node, by field.

        Each is read through the field's declared type, so it is the value
        itself, not its JSON form. An input left to its declared default is
        absent. When a single-value input is given several values, its entry
        is the list of them, and the node runs once per value. For a
        ``graph``, the values set on it, by inner address (``check.amount``),
        each read by the inner node that holds it.
        """
        _gate(self, "statics", needs_wiring=False)
        return self._statics

    @property
    def runner(self) -> Callable[..., Any]:
        """The callable that runs this node, on a fresh instance of
        ``definition`` per call. Made on each read, not stored."""
        _gate(self, "runner", needs_wiring=False, needs_one_unit=True)
        return runner_of(self._definition, self._version)

    @property
    def fingerprint(self) -> str:
        """A hash of how the graph places this node: its type, version and
        bindings. A run's state stores one per node, and a leg restored into
        a graph whose fingerprint differs runs the node again — a static
        edited, a version bumped, an edge moved all change it; a title or a
        position does not."""
        _gate(self, "fingerprint", needs_wiring=True, needs_one_unit=True)
        node = self.graph_node
        declared = {inp.name: inp.dtype for inp in self._interface.inputs}
        bindings: dict[str, Any] = {}
        for name, binding in node.bindings.items():
            if isinstance(binding, Static) and name in self._statics:
                # The value as its type writes it, never as the author spelled
                # it: ``2`` and ``2.0`` on a number are one value, and a graph
                # built in Python with the typed value hashes like the stored
                # graph with its JSON. The author typed many values exactly
                # when the input is received one per row of its own index.
                # A static for an input the node no longer has is hashed as
                # spelled, since no type reads it.
                bindings[name] = {"static": _written(self._statics[name], declared[name], self._fields[name].listed)}
            else:
                bindings[name] = binding.model_dump()
        placed = {"type": node.type, "version": node.version, "bindings": bindings}
        return hashlib.sha256(json.dumps(to_jsonable_python(placed), sort_keys=True).encode("utf-8")).hexdigest()

    def validate(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """The keyword arguments a call of this node runs with: ``inputs``
        checked against the interface, one value per input, a default
        filled in where the call gave none. Raises pydantic's
        ``ValidationError`` when a value is not its input's type or a
        required input is missing; the engine turns that into the node's
        failure. The check runs through a model built on the first call and
        kept, not built once per unit."""
        _gate(self, "validate", needs_wiring=True, needs_one_unit=True)
        validated = self._call_model(**inputs)
        return {info.alias or field: getattr(validated, field) for field, info in type(validated).model_fields.items()}

    @cached_property
    def _call_model(self) -> type[BaseModel]:
        """The pydantic model a call validates through, built on first use.
        Compile has already refused every input name it could not carry
        (``parameter_name_invalid``), so building it cannot fail here."""
        return model_of(self._interface.inputs)

    @property
    def iterates_on(self) -> Index | None:
        """The index this node runs once per row of, or ``None`` when it runs
        once. Read off its edges at compile — a series arriving on a scalar
        input is what makes a node run per row — and never written on the
        ``GraphNode``. For a ``graph``, the index its inner nodes run once
        per row of, where a series entered it."""
        _gate(self, "iterates_on", needs_wiring=True)
        return self._iterates_on


@dataclass(frozen=True, eq=False)
class CompiledField:
    """One input or output as the compiler left it: its type, its rows, where its value comes from, how it is received.

    Built once by ``CompiledGraph.from_graph``, held by its node, and
    handed back by ``CompiledGraph.field(ref)`` on every call; like its
    node, it is equal only to itself. The engine reads ``index``,
    ``binding`` and ``receives`` to lay values out per row, to find them
    and to hand each unit what it takes; an editor reads ``type`` and
    ``index`` to draw the edge, ``receives`` to label it and ``problems``
    for its marks. Its sibling is ``CompiledNode``, the same for one node.

    Only a resolved node has fields. ``ref``, ``problems`` and ``binding``
    answer always; ``type``, ``index``, ``receives`` and ``condition``
    are decided by the node's connections, and raise ``NodeWiringError``
    on a node whose connections compile could not work out.
    Asking a field for what its kind does not have is a ``KeyError``: an
    input has no ``condition``, an output has no ``binding``, ``receives``
    or ``listed``.
    """

    #: The expanded address of the field — ``Ref("approve/check", "amount")``
    #: however it was asked for.
    ref: Ref
    #: Every problem about this field, in ``CompiledGraph.problems`` order.
    problems: tuple[Problem, ...] = field(repr=False)
    #: Its node's ``_cause``: set exactly when the node is not ready.
    _cause: Problem | None = field(repr=False)
    #: Which kinds this name is: an input, an output, or both.
    _input: bool = field(repr=False)
    _output: bool = field(repr=False)
    _binding: Binding | None = field(repr=False)
    #: What the walk decided; ``None`` on a node that is not ready, or for a kind the field is not.
    #: ``_condition`` is over the graph's inputs (``conductor.graph.conditions.INPUT``).
    _type: Any = field(repr=False)
    _index: Index | None = field(repr=False)
    _receives: Receive = field(repr=False)
    _condition: Condition = field(repr=False)
    #: Its share of the read plan (see ``CompiledNode._reads``): every input
    #: an edge from this output feeds, in execution order, with how it
    #: receives the value (``_read_by``); and, for a ``listed`` input, the
    #: plan of the index its typed-in list births.
    _read_by: tuple[tuple[Ref, Receive], ...] = field(repr=False)
    _listed: bool = field(repr=False)
    _iterated_by: tuple[str, ...] = field(repr=False)
    _carried_by: tuple[Ref, ...] = field(repr=False)
    _typed_lists: tuple[Ref, ...] = field(repr=False)

    def _gate(self, asked: str) -> None:
        if self._cause is not None:
            raise NodeWiringError(self.ref.node_id, asked, self._cause)

    def _only(self, kind: Literal["input", "output"]) -> None:
        """Raise the ``KeyError`` for a read only an input, or only an output, has."""
        if not (self._input if kind == "input" else self._output):
            raise KeyError(f"{self.ref.node_id!r} has no {kind} {self.ref.field!r}")

    @property
    def type(self) -> Any:
        """The type that travels on this field: what an edge from it or into
        it carries. An output of a node that runs once per row carries
        ``Series[X]`` even where its declaration says ``X``; an input fed a
        series carries that series, before the engine slices it per row."""
        self._gate("type")
        return self._type

    @property
    def index(self) -> Index | None:
        """Where the rows of the series on this field come from, or ``None``
        for a field that carries one value. For an input fed several series
        gathered together, the fresh index they were gathered onto. Read
        by the engine to lay values out per row."""
        self._gate("index")
        return self._index

    @property
    def binding(self) -> Binding | None:
        """Where this input's value comes from: ``From`` when it is wired to
        other nodes' outputs, ``Static`` when a value is set directly on the
        node, or ``None`` when neither is and its declared default applies.
        Only an input has one; asking on an output raises."""
        self._only("input")
        return self._binding

    @property
    def receives(self) -> Receive:
        """How a unit of this input's node receives the value: the value at
        its own row (``Iterate``), the one value there is (``Broadcast``),
        everything on the field (``Whole``), the rows under its row
        (``Group``) or its unrelated sources collected (``Gather``) — see
        ``conductor.graph.receive``. Decided once by the
        walk over the edges; the engine's ledger reads it to tell when a
        unit is ready and what to hand it, and an editor may label the
        edge from it. Only an input has one; asking on an output raises."""
        self._only("input")
        self._gate("receives")
        return self._receives

    @property
    def listed(self) -> bool:
        """The author typed many values into this scalar input — three files,
        a list of texts — so its node runs once per value, on an index of
        the input's own. Only an input is typed into."""
        self._only("input")
        return self._listed

    @property
    def condition(self) -> Condition:
        """Under which condition this output appears: a boolean formula over
        the decisions upstream (see ``conductor.graph.conditions``),
        ``ALWAYS`` when nothing gates it. Derived from ``choice`` groups
        and edges; the engine never reads it. Only an output has one.
        Compile records it over the graph's inputs, for a graph that places
        this one; here every input holds, worked out on the first read."""
        self._only("output")
        self._gate("condition")
        return self._condition_with_inputs_holding

    @cached_property
    def _condition_with_inputs_holding(self) -> Condition:
        return substituted(self._condition, {}, "", per_row=False)
