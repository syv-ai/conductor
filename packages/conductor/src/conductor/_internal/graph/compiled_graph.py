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
one, ``answer`` writes a person's answers into one, and ``outputs`` and
``results`` read one.

The questions come at three scales. The graph itself answers what it
takes and returns (``interface``), what is wrong with it (``problems``,
``is_runnable``), in what order its nodes run (``execution_order``), where
each runs and what its inputs receive (``run_plan``) and which decisions a run
makes (``decisions``). ``describe()`` is all of that as one JSON-ready record,
for an editor. ``node(node_id)`` returns a ``CompiledNode``, everything
compile knows about one node, and ``field(ref)`` a ``CompiledField``, the same
for one input or output of a node compile fully worked out. Both hand back the
same value on every call. A node compile couldn't fully work out says so by
its ``state``, and its ``cause`` says why. The graph stores a ``GraphNode`` (a node as the author
saved it) and addresses fields by ``Ref``; what compile learned about that
node or field lives on its ``CompiledNode`` and ``CompiledField``.

The terms this module uses throughout, each defined once here.

A node's interface is the list of inputs and outputs it actually has.
Usually that is what its version declares, but a node may add or drop
fields depending on the values it holds and the types connected to it.

A field that holds many values has a row for each, and an index names where those
rows come from: ten uploaded documents are ten rows on the index
of the node that uploaded them. A scalar input is an input that takes one
value. A node that receives many values on an input for one runs once per row
of them; we say the node runs per row of that index.

Some nodes have a version that is itself a graph, which the host compiled
on its own; that graph is an embedded graph, the node's ``embedded_graph``. Compile embeds the graph's nodes in place of that
node (``conductor._internal.graph.embedding``), so the graph the author drew (the
authored graph) differs from the graph that runs. In the authored graph the
embedded graph is one node, ``approve``; in the graph that runs its inner
nodes are nodes of the run, at their path, ``approve/check``. An embedded
graph is reached from outside only by the names it offers (``embedding``);
``interface`` names its fields ``approve.amount``, as any node's, and ``field``
reads that address through to the field inside.
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
from conductor._internal.execution.run_state import Answers, RunState
from conductor._internal.execution.running_leg import RunningLeg
from conductor._internal.execution.values_of_run import ValuesOfRun
from conductor._internal.execution.waiting_units import apply_answers
from conductor._internal.graph.binding import Static
from conductor._internal.graph.compile_graph import compile_graph
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
from conductor._internal.graph.embedding import (
    EmbeddableGraph,
    embeddable_view_of,
    offered_names,
    place_under,
)
from conductor._internal.graph.engine_lookups import EngineLookups, build_engine_lookups
from conductor._internal.graph.graph_description import GraphDescription, describe_graph
from conductor._internal.graph.run_plan import RunPlan
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.graph.graph_records import Graph
    from conductor._internal.node_signature import Interface
    from conductor._internal.registry import NodeRegistry


def where_it_failed(failure: Problem) -> str:
    """A failure in a sentence: the node, the row and the kind of exception it was, for an error read beside no state."""
    if not failure.loc:
        return failure.msg
    node_id, *row = failure.loc
    where = f"It stopped at node {node_id!r}" + (f", row {'.'.join(map(str, row))}" if row else "")
    kind = failure.ctx.get("exception")
    return f"{where}{f' ({kind})' if kind else ''}: {failure.msg} See state.failure and the 'conductor' log."


def _place_of_offered_field(handle: Ref, graph: CompiledGraph) -> Ref:
    """Where the field ``handle`` names on a node that is ``graph`` is in the run: inside, at the node's path."""
    inside = graph._embeddable.inputs.get(handle.field) or graph._embeddable.outputs.get(handle.field)
    if inside is None:
        raise KeyError(f"{handle.node_id!r} offers no field {handle.field!r}")
    return place_under(handle.node_id, inside)


@dataclass(frozen=True, repr=False)
class CompiledGraph:
    """The result of compiling one graph. Ask it; do not read through it.

    Built by ``from_graph`` and read by the engine (``_engine_lookups``, on
    every step of a run), an editor's compile endpoint (``describe``), and
    anything deciding whether a run may start (``is_runnable``).

    Four attributes are public, ``graph``, ``interface``, ``problems`` and ``execution_order``; every other
    attribute is private and read through the methods here and through
    ``node`` and ``field``. ``node`` answers for every node compile met and
    raises ``KeyError`` only for an id the graph does not have.
    """

    #: The registry the graph was compiled against, kept so a compiled
    #: graph can produce another (``with_inputs``).
    _registry: NodeRegistry
    #: The ids of the nodes that run, in an order where every edge's source precedes
    #: its target. A node in a cycle is not in it; it has a fatal ``Problem`` instead.
    execution_order: tuple[str, ...]
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
        raises for a problem in the graph; read ``problems`` and ``is_runnable``.
        A node's hook that raises anything but ``ComputeHookError`` is a bug in
        the node, and that exception leaves here as it is.
        """
        records = compile_graph(graph, registry)
        return cls(
            _registry=registry, execution_order=records.order, interface=records.interface, graph=graph, problems=records.problems,
            _nodes=records.nodes, _row_ids_must_survive=records.row_ids_must_survive,
        )

    # -- one node, one field ---------------------------------------------------

    @cached_property
    def _engine_lookups(self) -> EngineLookups:
        """What the engine looks up on every step, built on the first run: compiling alone never pays for it."""
        return build_engine_lookups(self._nodes[node_id] for node_id in self.execution_order)

    @cached_property
    def _embeddable(self) -> EmbeddableGraph:
        """What a graph that embeds this one asks of it (``conductor._internal.graph.embedding``)."""
        return embeddable_view_of(self)

    @cached_property
    def run_plan(self) -> RunPlan:
        """Where every ready node runs and what each of its inputs receives,
        in execution order: the shape compile decided, for a host to show and
        compare (``conductor._internal.graph.run_plan``). ``str(compiled.run_plan)`` is
        that as text. A node that isn't ready is absent."""
        return RunPlan.model_construct(nodes={
            node_id: node._run_plan for node_id in self.execution_order if (node := self._nodes[node_id])._fields is not None
        })

    def node(self, node_id: str) -> CompiledNode:
        """One compiled node by its id: its path inside an embedded graph
        (``approve/check``). An id the graph does not have raises ``KeyError``."""
        if node_id not in self._nodes:
            raise KeyError(f"{node_id!r} is not a node of this graph")
        return self._nodes[node_id]

    def field(self, ref: Ref | str) -> CompiledField:
        """One compiled input or output by its address, a ``Ref`` or the string.

        An address on a node that is a graph reads the field that name offers,
        ``Ref("emb", "result")``, the same field as its place in the run,
        ``Ref("emb/all", "result")``. Anything the graph has no such field for
        raises ``KeyError``, naming the problem behind it when a node compile
        couldn't work out is why."""
        ref = ref if isinstance(ref, Ref) else Ref(ref)
        node = self.node(ref.node_id)
        if node._fields is None:
            raise KeyError(f"{ref.node_id!r} has no fields compile could work out: {node.cause.msg if node.cause else ''}")
        if node.embedded_graph is not None:
            return self.field(_place_of_offered_field(ref, node.embedded_graph))
        compiled_field = node._fields.get(ref.field)
        if compiled_field is None:
            raise KeyError(f"{ref.node_id!r} has no field {ref.field!r} on this node")
        return compiled_field

    # -- the interface, from each side --------------------------------------------

    def with_inputs(self, inputs: Mapping[str, Any] | None = None, /, **by_name: Any) -> CompiledGraph:
        """This graph with some of its inputs filled, compiled again, to run with those values::

            compiled.with_inputs({"a.text": "hello"}, count=3)

        ``inputs`` is keyed by address (``a.text``, ``emb.text`` on a node that
        is a graph); a keyword is a field name when no other input has that name. A name the interface does not offer raises
        ``InvalidCallError``, code ``not_offered``, every such name at once.
        Each value becomes a ``Static`` on its input in a copy of the graph,
        which compiles against the same registry: a list on an input for one
        value runs the graph once per item. Only what the interface offers: a
        caller's body can go here, and ``Graph.with_values`` is an author's. A value its input's type can't
        read is an ``invalid_static``, raised with any names it doesn't offer.
        The copy offers every input as this one does. Not a run.
        """
        values = {**(inputs or {}), **by_name}
        offered = [input_.name for input_ in self.interface.inputs]
        problems: list[Problem] = []
        statics_to_set: dict[str, dict[str, Static]] = {}
        written_as: dict[tuple[str, str], str] = {}
        for name, value in values.items():
            ref = self._offered_input_named(name, offered, problems)
            if ref is not None:
                statics_to_set.setdefault(ref.node_id, {})[ref.field] = Static(value)
                written_as[ref.node_id, ref.field] = name
        refilled = CompiledGraph.from_graph(self.graph._with_statics(statics_to_set), self._registry)
        problems += [
            problem.model_copy(update={"loc": ("inputs", written_as[problem.loc])})  # type: ignore[index]
            for problem in refilled.problems if problem.type in ("invalid_static", "table_column_missing") and problem.loc in written_as
        ]
        if problems:
            raise InvalidCallError(problems)
        return refilled

    def _offered_input_named(self, name: str, offered: list[Ref], problems: list[Problem]) -> Ref | None:
        """The input a keyword to ``with_inputs`` names; or ``None``, and the problem that lists what is offered."""
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

    def describe(self) -> GraphDescription:
        """This compiled graph as an editor reads it: whether it runs, what is
        wrong, what it takes and returns, where each node runs, and per node
        the node's fields with the type on each (``GraphDescription``)."""
        return describe_graph(self)

    @property
    def decisions(self) -> dict[str, dict[str, tuple[str, ...]]]:
        """Every decision a caller could observe: for each node that runs
        once and declares an ``exclusive_group`` group, the group's alternatives in
        field order, keyed by node id. A node that runs per row is
        left out: its decision picks rows and holds back nothing downstream. So
        is a node that is not ready: whether it runs per row is not
        known."""
        groups_by_node: dict[str, dict[str, tuple[str, ...]]] = {}
        for node_id in self.execution_order:
            node = self._nodes[node_id]
            if node._fields is None or node._runs_per_row_of is not None:
                continue
            groups: dict[str, list[str]] = {}
            for out in node.interface.outputs:
                if out.exclusive_group is not None:
                    groups.setdefault(out.exclusive_group, []).append(out.name)
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
        block is entered, so a start that can't go raises here (``InvalidCallError``).

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
        return RunningLeg(self, state=state, from_run=from_run or {}, timeout=timeout, cancel=cancel or asyncio.Event(), checkpoint_every=checkpoint_every)

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
        raises, since ``asyncio.run`` cannot nest: ``await compiled.run(...)``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(state, from_run=from_run, timeout=timeout, cancel=cancel))
        raise RuntimeError("a loop is running; use await compiled.run(...)")

    def answer(self, state: RunState, answers: Answers, /) -> RunState:
        """``state`` with these waiting units answered; no node runs.

        ``answers`` maps ``WaitingUnit.id`` to a value per output, or to
        ``None`` for no value on that row. Every answer is checked here, with a
        wrong one raises from this call (``InvalidCallError``, every problem at
        once) and ``state`` untouched. A unit left out goes on waiting, so
        people can answer rows days apart. ``run`` the result to go on.
        """
        return apply_answers(self, state, answers)

    def results(self, state: RunState, /) -> dict[str, dict[str, Any]]:
        """Every node's outputs so far in ``state``, typed, by node id
        (``emb/up`` inside an embedded graph), whatever way the leg ended.

        A node with nothing finished is absent, and so is an output a node
        skipped (the branch not taken). One that holds many is
        ``{"rows": [...], "values": [...]}`` (``ValuesOnRows``), the rows that have
        values so far and a value each, so a host can show a table filling
        in as people answer. A node the graph has changed since, and whatever
        reads it, is left out.
        """
        dropped = set(self._nodes_changed_since(state))
        values = ValuesOfRun(self._engine_lookups)
        problems = values.read_state(self, state, dropped)
        if problems:
            raise InvalidCallError(problems)
        produced: dict[str, dict[str, Any]] = {}
        for node_id in self.execution_order:
            work = state.nodes.get(node_id)
            if work is not None and work.finished_units is not None and node_id not in dropped:
                result = values.result_of(node_id)
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
                reason="" if failure is None else where_it_failed(failure),
                failure=None if failure is None else failure.model_dump(mode="json"),
            )])
        where = self._embeddable.place_of_output
        changed = set(self._nodes_changed_since(state)).intersection(output_address.node_id for output_address in where.values())
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

    def _nodes_changed_since(self, state: RunState, /) -> list[str]:
        """Every node ``state`` holds nothing current for, in execution order,
        then the nodes it names that the graph doesn't have.

        Changed means its fingerprint differs or is missing: a static edited,
        a version bumped, an edge moved. Whatever reads such a node is in the
        answer too, directly or through other nodes. The one place that decides
        what is stale; a host wanting a strict pin rejects the run when the list is
        not empty.
        """
        self._raise_unless_runnable()
        work_of_node = state.nodes
        changed = {node_id for node_id in self.execution_order if (work := work_of_node.get(node_id)) is None or work.fingerprint != self._nodes[node_id].fingerprint}
        return self._with_their_readers(changed, state)

    def _nodes_a_leg_runs_again(self, state: RunState, /) -> list[str]:
        """What a leg starting from ``state`` runs again: every node changed since, and from a ``complete`` state
        every node that reads the world (``Policy(reuse_when_inputs_match=False)``), each with what reads it.

        A ``complete`` state is finished, so a leg from it is a new one. A leg from any other state
        goes on with the same run, so a person's answer doesn't read a file again."""
        changed = self._nodes_changed_since(state)
        if state.status != "complete":
            return changed
        reading_the_world = {node_id for node_id in self.execution_order if not self._nodes[node_id].version.policy.reuse_when_inputs_match}  # type: ignore[union-attr]  # a node that runs is a function
        return self._with_their_readers(set(changed) | reading_the_world, state)

    def _with_their_readers(self, node_ids: set[str], state: RunState) -> list[str]:
        """``node_ids`` and whatever reads them, in execution order, then the nodes ``state`` names that the graph doesn't have."""
        stale = node_ids | self._readers_of(node_ids & set(self.execution_order))
        return [node_id for node_id in self.execution_order if node_id in stale] + sorted(state.nodes.keys() - set(self.execution_order))

    def _readers_of(self, node_ids: Iterable[str]) -> set[str]:
        """Every node that reads a field of one of ``node_ids``, directly or
        through other nodes: an output, a column of a table output, or the
        rows such a node makes, which a node inside a graph run per row runs
        on with no edge to them. Walked over the lookups on each call rather
        than stored, since every node's readers stored would cost memory
        quadratic in a long chain."""
        lookups = self._engine_lookups
        found: set[str] = set()
        frontier = list(node_ids)

        def add(node_id: str) -> None:
            if node_id not in found:
                found.add(node_id)
                frontier.append(node_id)

        while frontier:
            for compiled_field in self._nodes[frontier.pop()]._fields.values():
                for reader in lookups.read_by[compiled_field.ref]:
                    add(reader.node_id)
                rows_made = compiled_field._held_as.rows_made
                if rows_made is not None:
                    for runs_on_them in lookups.nodes_running_per_row_of_index[rows_made.id]:
                        add(runs_on_them)
                    for list_under_them in lookups.lists_on_nodes_under_index[rows_made.id]:
                        add(list_under_them.node_id)
        return found

    def _raise_unless_runnable(self) -> None:
        """Raise for any verb on a graph with a fatal problem, naming every one."""
        if not self.is_runnable:
            raise InvalidCallError(problem for problem in self.problems if problem.fatal)

    # -- what is wrong -------------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """True when nothing fatal was found. A run raises otherwise."""
        return not any(problem.fatal for problem in self.problems)
