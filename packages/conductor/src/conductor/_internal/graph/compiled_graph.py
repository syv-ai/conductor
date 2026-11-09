"""``CompiledGraph`` — everything the compiler learned about a graph, as one immutable value, and every verb that runs it.

``CompiledGraph.from_graph`` builds it from a ``Graph`` and a ``NodeRegistry``.
Everyone else asks it questions and never reads the nodes' bindings
themselves: which version each node uses, which inputs and outputs it
actually has, where each input's value comes from and how each unit
receives it, what type travels on every field, which nodes run once per
row and in what order, under which condition each output appears, and
what is wrong. The engine is one more caller.

It also owns every step and question of a run, and holds no run itself:
``run``, ``run_sync`` and ``stream`` take a ``RunState`` and return or end on
one, ``answer`` writes a person's answers into one, and ``outputs``,
``results`` and ``nodes_changed_since`` read one.

The questions come at three scales. The graph itself answers what it
takes and returns (``interface``), what is wrong with it (``problems``,
``is_runnable``), in what order its nodes run (``execution_order``), where
each runs and what its inputs receive (``run_plan``) and which decisions a run
makes (``decisions``). ``node(node_id)`` returns a ``CompiledNode``, everything
compile knows about one node, and ``field(ref)`` a ``CompiledField``, the same
for one input or output of a node compile fully worked out. Both hand back the
same value on every call. A node compile couldn't fully work out says so by
what it lacks: its ``interface`` or its ``wiring`` is ``None``, and its
``cause`` says why. The graph stores a ``GraphNode`` (a node as the author
saved it) and addresses fields by ``Ref``; what compile learned about that
node or field lives on its ``CompiledNode`` and ``CompiledField``.

The words this module uses throughout, each defined once here.

A node's interface is the list of inputs and outputs it actually has.
Usually that is what its version declares, but a node may add or drop
fields depending on the values it holds and the types connected to it.

A series is a value with many rows, and an index names where those rows
come from: ten uploaded documents are a series of ten rows on the index
of the node that uploaded them. A scalar input is an input that takes one
value. A node that receives a series on a scalar input runs once per row
of the series; we say the node runs per row of that index.

Some nodes have a version that is itself a graph, which the host compiled
on its own; that graph is an embedded graph, the node's ``embedded_graph``. Compile embeds the graph's nodes in place of that
node (``conductor._internal.graph.embedding``), so the graph the author drew (the
authored graph) differs from the graph that runs. In the authored graph the
embedded graph is one node, ``approve``; in the graph that runs its inner
nodes are nodes of the run, at their path, ``approve/check``. An embedded
graph is reached from outside only by the names it offers (``embedding``);
``interface`` names its fields ``approve.amount``, as any node's.
``execution_order`` and ``decisions`` speak of the graph that runs;
``problems`` speak of the authored one, on the node and field the author
drew. ``node`` answers for both graphs: an inner node by its path
(``approve/check``) and the node that embeds it by its own id (``approve``).

It is a plain value: immutable, no I/O, no session. The same graph and
the same registry always give the same ``CompiledGraph``, so it can be
cached, and "compile this and assert what it says" is a complete test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from conductor._internal.errors import InvalidCallError, Problem, make_problem
from conductor._internal.execution.events import Event, RunEnd
from conductor._internal.execution.run_ledger import node_result, read_output_columns
from conductor._internal.execution.run_state import Answers, RunState
from conductor._internal.execution.run_until_idle import Leg
from conductor._internal.execution.waiting_units import apply_answers
from conductor._internal.graph.binding import Static
from conductor._internal.graph.compile_graph import compile_graph
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
from conductor._internal.graph.embedding import EmbeddableGraph, embeddable_view_of, offered_names
from conductor._internal.graph.engine_lookups import EngineLookups, build_engine_lookups
from conductor._internal.graph.run_plan import RunPlan
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.graph.graph_records import Graph
    from conductor._internal.node_signature import Interface
    from conductor._internal.registry import NodeRegistry


@dataclass(frozen=True, repr=False)
class CompiledGraph:
    """The result of compiling one graph. Ask it; do not read through it.

    Built by ``from_graph`` and read by the engine (``execution_order``,
    then ``node`` and ``field`` for each node it runs), an editor's
    compile endpoint (``problems``, ``interface``, and ``node`` and
    ``field`` for everything it draws), and anything deciding whether a
    run may start (``is_runnable``).

    Three attributes are public, ``graph``, ``interface`` and ``problems``; every other
    attribute is private and read through the methods here and through
    ``node`` and ``field``. ``node`` answers for every node compile met and
    raises ``KeyError`` only for an id the graph does not have.
    """

    #: The registry the graph was compiled against, kept so a compiled
    #: graph can produce another (``with_inputs``).
    _registry: NodeRegistry
    #: What ``execution_order`` answers.
    _execution_order: tuple[str, ...]
    #: What this graph takes and returns, in the same record a node version
    #: declares: ``inputs`` are the unlocked, connectable fields of the nodes
    #: nothing feeds into (each declaration whole, titled as the author
    #: titled it on that node), ``outputs`` every field of the nodes nothing
    #: reads from, both named by address (``"node.field"``).
    interface: Interface
    #: The graph as the author saved it — with the values ``with_inputs``
    #: filled, on a copy it made — which a host stores as what ran. Not read
    #: back for what compile learned: ask ``node`` and ``field`` for that.
    graph: Graph
    #: Everything wrong with the graph, fatal or not, each on the node and
    #: field it is about, all on nodes of the authored graph. An embedded
    #: graph that cannot be embedded is one ``embedded_graph_broken`` on the
    #: graph node; its own problems are on it (``node(id).version.problems``).
    problems: tuple[Problem, ...]
    #: Every node compile met, by id: what ``node`` hands back.
    _nodes: Mapping[str, CompiledNode] = field(repr=False)
    #: The tables a merge matches against what they were built from.
    _row_ids_must_survive: frozenset[Ref] = field(repr=False)

    @classmethod
    def from_graph(cls, graph: Graph, registry: NodeRegistry) -> CompiledGraph:
        """Compile ``graph`` against ``registry``: the one way to get a ``CompiledGraph``.

        Pure: the same graph and registry always give the same result. Every
        definition the graph names must already be in the registry. Nothing
        raises for a fault in the graph; read ``problems`` and ``is_runnable``.
        A node's hook that raises anything but ``ComputeHookError`` is a bug in
        the node, and that exception leaves here as it is.
        """
        records = compile_graph(graph, registry)
        return cls(
            _registry=registry, _execution_order=records.order, interface=records.interface, graph=graph, problems=records.problems,
            _nodes=records.nodes, _row_ids_must_survive=records.row_ids_must_survive,
        )

    # -- one node, one field ---------------------------------------------------

    @cached_property
    def _engine_lookups(self) -> EngineLookups:
        """What the engine looks up on every step, built on the first run: compiling alone never pays for it."""
        return build_engine_lookups((self._nodes[node_id] for node_id in self._execution_order), self._row_ids_must_survive)

    @cached_property
    def _embeddable(self) -> EmbeddableGraph:
        """What a graph that embeds this one asks of it (``conductor._internal.graph.embedding``)."""
        return embeddable_view_of(self)

    @cached_property
    def _where_each_output_is_in_the_run(self) -> dict[Ref, Ref]:
        names = offered_names(Ref(out.name) for out in self.interface.outputs)
        return {output_address: self._embeddable.outputs[name] for output_address, name in names.items()}

    @cached_property
    def run_plan(self) -> RunPlan:
        """Where every ready node runs and what each of its inputs receives,
        in execution order: the shape compile decided, for a host to show and
        compare (``conductor._internal.graph.run_plan``). ``str(compiled.run_plan)`` is
        that as text. A node that isn't ready is absent."""
        return RunPlan.model_construct(nodes={
            node_id: node._run_plan for node_id in self._execution_order if (node := self._nodes[node_id]).wiring is not None
        })

    def node(self, node_id: str) -> CompiledNode:
        """One compiled node by its id: its path inside an embedded graph
        (``approve/check``). An id the graph does not have raises ``KeyError``."""
        if node_id not in self._nodes:
            raise KeyError(f"{node_id!r} is not a node of this graph")
        return self._nodes[node_id]

    def field(self, ref: Ref | str) -> CompiledField:
        """One compiled input or output by its address, a ``Ref`` or the string.

        A field inside an embedded graph is at its path: ``Ref("emb/all",
        "result")``. Anything the graph has no such field for raises
        ``KeyError``, naming the problem behind it when a node compile
        couldn't work out is why."""
        ref = ref if isinstance(ref, Ref) else Ref(ref)
        node = self.node(ref.node_id)
        if node.wiring is None:
            raise KeyError(f"{ref.node_id!r} has no fields compile could work out: {node.cause.msg if node.cause else ''}")
        if node.embedded_graph is not None:
            raise KeyError(f"{ref.node_id!r} is a graph; its fields are on the nodes inside it")
        compiled_field = node.wiring.fields.get(ref.field)
        if compiled_field is None:
            raise KeyError(f"{ref.node_id!r} has no field {ref.field!r} on this node")
        return compiled_field

    # -- the interface, from each side --------------------------------------------

    def with_inputs(self, inputs: Mapping[str, Any] | None = None, /, **by_name: Any) -> CompiledGraph:
        """This graph with some of its inputs filled, compiled again, to run with those values::

            compiled.with_inputs({"a.text": "hello"}, count=3)

        ``inputs`` is keyed by address (``a.text``, ``emb.text`` on a node that
        is a graph); a keyword is a field name when no other input has that name. A name the interface does not offer is refused with
        ``InvalidCallError``, code ``not_offered``, every such name at once.
        Each value becomes a ``Static`` on its input in a copy of the graph,
        which compiles against the same registry: a list on an input for one
        value runs the graph once per item. A value its input's type can't
        read is an ``invalid_static``, raised with any names it doesn't offer.
        The copy still offers every input. Not a run.
        """
        return self._recompiled_with({**(inputs or {}), **by_name}, (), raise_on_unreadable_values=True)

    def _recompiled_with(self, values: Mapping[str, Any], input_names_to_clear: Iterable[str], *, raise_on_unreadable_values: bool = False) -> CompiledGraph:
        """This graph compiled again with ``values`` typed on the inputs they
        name and the values its author typed on the ``statics_to_remove`` inputs taken
        off, so they arrive at run time. ``with_inputs`` for a caller; a
        graph that embeds this one asks for the copy one graph node fills. Names are read as ``with_inputs`` reads them.
        A graph that embeds this one wants a value its input can't read as
        a problem of the compile, so it says so with ``raise_on_unreadable_values=False``; a caller
        gets it raised."""
        offered = [inp.name for inp in self.interface.inputs]
        problems: list[Problem] = []
        statics_to_set: dict[str, dict[str, Static]] = {}
        written_as: dict[tuple[str, str], str] = {}
        for name, value in values.items():
            ref = self._offered_input_named(name, offered, problems)
            if ref is not None:
                statics_to_set.setdefault(ref.node_id, {})[ref.field] = Static(value)
                written_as[ref.node_id, ref.field] = name
        statics_to_remove: dict[str, set[str]] = {}
        for name in input_names_to_clear:
            ref = self._offered_input_named(name, offered, problems)
            if ref is not None:
                statics_to_remove.setdefault(ref.node_id, set()).add(ref.field)
        graph = self.graph.model_copy(update={"nodes": tuple(
            node.model_copy(update={"bindings": {
                **{field: binding for field, binding in node.bindings.items() if field not in statics_to_remove.get(node.id, ())},
                **statics_to_set.get(node.id, {}),
            }})
            for node in self.graph.nodes
        )})
        refilled = CompiledGraph.from_graph(graph, self._registry)
        if raise_on_unreadable_values:
            problems += [
                problem.model_copy(update={"loc": ("inputs", written_as[problem.loc])})  # type: ignore[index]
                for problem in refilled.problems if problem.type in ("invalid_static", "table_column_missing") and problem.loc in written_as
            ]
        if problems:
            raise InvalidCallError(problems)
        return refilled

    def _offered_input_named(self, name: str, offered: list[Ref], problems: list[Problem]) -> Ref | None:
        """The input a keyword to ``with_inputs`` names; or ``None``, and the fault that lists what is offered."""
        names = offered_names(offered)
        matches = [ref for ref in offered if ref == name or names[ref] == name]
        if matches:
            return matches[0]
        clashing = [ref for ref in offered if ref.field == name]
        if clashing:
            both = ", ".join(str(ref) for ref in clashing)
            reason = f"{name!r} is an input of several nodes ({both}); name one by its address"
        elif offered:
            reason = f"{name!r} is not an input this graph offers; it offers {', '.join(str(ref) for ref in offered)}"
        else:
            reason = f"{name!r}: this graph takes no inputs"
        problems.append(make_problem("not_offered", "inputs", name, reason=reason, offered=[str(ref) for ref in offered]))
        return None

    # -- the run ------------------------------------------------------------------

    def __repr__(self) -> str:
        """One line: the nodes the author drew, whether it runs, how many problems."""
        drawn = tuple(node.id for node in self.graph.nodes)
        return f"CompiledGraph(nodes={drawn!r}, is_runnable={self.is_runnable}, problems={len(self.problems)})"

    @property
    def execution_order(self) -> tuple[str, ...]:
        """The ids of the nodes that run, in an order where every edge's source precedes
        its target. A node in a cycle is not in it; it has a fatal
        ``Problem`` instead."""
        return self._execution_order

    @property
    def decisions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every decision a caller could observe: for each node that runs
        once and declares a ``choice`` group, the group's alternatives in
        field order, keyed by node id. A node that runs per row is
        left out: its decision picks rows and gates nothing downstream. So
        is a node that is not ready: whether it runs per row is not
        known."""
        groups_by_node: dict[str, dict[str, tuple[str, ...]]] = {}
        for node_id in self._execution_order:
            node = self._nodes[node_id]
            if node.wiring is None or node.wiring._runs_per_row_of is not None:
                continue
            groups: dict[str, list[str]] = {}
            for out in node.interface.outputs:
                if out.choice is not None:
                    groups.setdefault(out.choice, []).append(out.name)
            if groups:
                groups_by_node[node_id] = {choice: tuple(names) for choice, names in groups.items()}
        return groups_by_node

    # -- running it ----------------------------------------------------------------

    def stream(
        self,
        state: RunState | None = None,
        /,
        *,
        from_run: Mapping[type, Any] | None = None,
        timeout: float | None = None,
        cancel: asyncio.Event | None = None,
        checkpoint_every: float | None = None,
    ) -> AbstractAsyncContextManager[AsyncIterator[Event]]:
        """One leg, as a stream of events; leaving the block stops every unit::

            async with compiled.stream(state, timeout=300) as events:
                async for event in events:
                    ...

        The first event is ``RunStart`` and the last ``RunEnd``, whose ``state``
        says how the leg ended. Every check is made at this call, before the
        block is entered, so a refused start raises here (``InvalidCallError``).

        ``state`` is where to go on from, ``None`` for a fresh run: a node the
        graph has changed since, and whatever reads it, runs again. ``from_run``
        supplies values by type to parameters annotated
        ``FromRun[X]``; a type no one supplied, or a key that is no type, is an ``InvalidCallError``. A type no node needs is left alone, so a host can hand every graph the same values.
        ``timeout`` bounds the leg in seconds and ends it ``timed_out``;
        setting ``cancel`` ends it ``cancelled`` at once.

        ``checkpoint_every`` asks for a ``RunCheckpoint`` at most this many
        seconds apart, taken when a unit finishes, so a crash loses only the units
        since the last one saved. ``0`` saves at every finished unit, a pass over
        every value so far each time, so its cost is the square of the units:
        for a test, not for a large run. ``None``, the
        default, costs nothing. A checkpoint is a state with status ``running``:
        ``run`` and ``answer`` go on from it, a finished unit is not called again
        and one in flight is, so a unit runs at least once and a node with side
        effects must allow for that. Nothing in the state says whether its leg is
        still alive; a host resuming from one holds a lease of its own, or two
        workers will run it. Building the state costs a pass over every value so
        far, on the event loop; dump it on another thread, it is a copy.
        """
        return Leg(self, state=state, from_run=from_run or {}, timeout=timeout, cancel=cancel or asyncio.Event(), checkpoint_every=checkpoint_every)

    async def run(
        self,
        state: RunState | None = None,
        /,
        *,
        from_run: Mapping[type, Any] | None = None,
        timeout: float | None = None,
        cancel: asyncio.Event | None = None,
    ) -> RunState:
        """One leg to its end, and the state it ended on: ``stream``, drained.

        A pause or a failure is a ``status``, not an exception. For code with
        no event loop, ``run_sync``.
        """
        ended: Event | None = None
        async with self.stream(state, from_run=from_run, timeout=timeout, cancel=cancel) as events:
            async for ended in events:
                pass
        if not isinstance(ended, RunEnd):
            raise RuntimeError(f"the leg ended on {ended!r}, not a RunEnd — an engine bug")
        return ended.state

    def run_sync(
        self,
        state: RunState | None = None,
        /,
        *,
        from_run: Mapping[type, Any] | None = None,
        timeout: float | None = None,
        cancel: asyncio.Event | None = None,
    ) -> RunState:
        """``run`` for a script or a test. Inside a running event loop it
        refuses, since ``asyncio.run`` cannot nest: ``await compiled.run(...)``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(state, from_run=from_run, timeout=timeout, cancel=cancel))
        raise RuntimeError("a loop is running; use await compiled.run(...)")

    def answer(self, state: RunState, answers: Answers, /) -> RunState:
        """``state`` with these waiting units answered; no node runs.

        ``answers`` maps ``WaitingUnit.id`` to a value per output, or to
        ``None`` for no value on that row. Every answer is checked here, with a
        wrong one refused from this call (``InvalidCallError``, every fault at
        once) and ``state`` untouched. A unit left out goes on waiting, so
        people can answer rows days apart. ``run`` the result to go on.
        """
        return apply_answers(self, state, answers)

    def results(self, state: RunState, /) -> dict[str, dict[str, Any]]:
        """Every node's outputs so far in ``state``, typed, by node id
        (``emb/up`` inside an embedded graph), whatever way the leg ended.

        A node with nothing finished is absent, and so is an output a node
        skipped (the branch not taken). One that runs per row is a ``Series``
        of the rows that have values so far, so a host can show a table filling
        in as people answer. A node the graph has changed since, and whatever
        reads it, is left out.
        """
        dropped = set(self.nodes_changed_since(state))
        values, faults = read_output_columns(self, state, dropped)
        if faults:
            raise InvalidCallError(faults)
        produced: dict[str, dict[str, Any]] = {}
        for node_id in self._execution_order:
            if node_id in state.finished_rows and node_id not in dropped:
                result = node_result(self._engine_lookups, node_id, values)
                if result is not None:
                    produced[node_id] = result
        return produced

    def outputs(self, state: RunState, /) -> dict[str, Any]:
        """What the graph returned: each output of ``interface.outputs`` by its address.

        Only a ``complete`` state of this graph has any. On another state, or
        one where this graph has changed an output's node since, the call
        raises ``InvalidCallError`` with the code ``run_not_complete``, its
        ``ctx`` holding the status (``stale`` for the changed graph) and the
        failure's cause, so a host never reads an empty answer for a run that
        was not this one. An output whose node skipped it is absent. ``results``
        reads what a stopped run did produce.
        """
        if state.status != "complete":
            failure = state.failure
            raise InvalidCallError([make_problem(
                "run_not_complete", "state", "status", status=state.status,
                reason="" if failure is None else failure.msg,
                failure=None if failure is None else failure.model_dump(mode="json"),
            )])
        where = self._where_each_output_is_in_the_run
        changed = set(self.nodes_changed_since(state)).intersection(output_address.node_id for output_address in where.values())
        if changed:
            raise InvalidCallError([make_problem(
                "run_not_complete", "state", "status", status="stale",
                reason=f"These have changed since it ran: {', '.join(sorted(changed))}.", failure=None,
            )])
        results = self.results(state)
        returned: dict[str, Any] = {}
        for output in self.interface.outputs:
            output_address = where[Ref(output.name)]
            if output_address.field in results.get(output_address.node_id, {}):
                returned[str(output.name)] = results[output_address.node_id][output_address.field]
        return returned

    def nodes_changed_since(self, state: RunState, /) -> list[str]:
        """Every node ``state`` holds nothing current for, in execution order,
        then the nodes it names that the graph no longer has.

        Changed means its fingerprint differs or is missing: a static edited,
        a version bumped, an edge moved. Whatever reads such a node is in the
        answer too, directly or through other nodes. The one place that decides
        what is stale; a host wanting a strict pin refuses when the list is
        not empty.
        """
        self._raise_unless_runnable()
        fingerprints = state.node_fingerprints
        changed = {node_id for node_id in self._execution_order if fingerprints.get(node_id) != self._nodes[node_id].fingerprint}
        stale = changed | self._readers_of(changed)
        named = {*fingerprints, *state.finished_rows, *(ref.node_id for ref in state.output_columns), *(unit.node_id for unit in state.waiting)}
        return [node_id for node_id in self._execution_order if node_id in stale] + sorted(named - set(self._execution_order))

    def _readers_of(self, node_ids: Iterable[str]) -> set[str]:
        """Every node that reads a field of one of ``node_ids``, directly or
        through other nodes: an output, or a column of a table output. Walked
        over the read plan on each call rather than stored, since every node's
        readers stored would cost memory quadratic in a long chain."""
        read_by = self._engine_lookups.read_by
        found: set[str] = set()
        frontier = list(node_ids)
        while frontier:
            for compiled_field in self._nodes[frontier.pop()].wiring.fields.values():
                for reader, _ in read_by[compiled_field.ref]:
                    if reader.node_id not in found:
                        found.add(reader.node_id)
                        frontier.append(reader.node_id)
        return found

    def _raise_unless_runnable(self) -> None:
        """Refuse any verb on a graph with a fatal problem, naming every one."""
        if not self.is_runnable:
            raise InvalidCallError(problem for problem in self.problems if problem.fatal)

    # -- what is wrong -------------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run refuses otherwise."""
        return not any(problem.fatal for problem in self.problems)
