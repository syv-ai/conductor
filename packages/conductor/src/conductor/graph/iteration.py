"""One walk over the edges: what arrives on every field, whether each edge is allowed, and how each input receives.

Called by the compiler once every node's inputs are known. It walks the
nodes in execution order and, for each edge into each input, answers
three questions.

May this edge land? ``target.accepts(source)`` decides; a series is
judged by its element type. A refusal is the fatal ``type_mismatch``.

Does the node run once, or once per row? A series is a value with many
rows, and an index names where those rows come from. A scalar input is
an input that takes one value. A series arriving on a scalar input means
the node runs once per row of that index; we say the node iterates on
the index. Its other scalar inputs are the same value every row, every
output becomes a series on the same index, and nodes downstream receive
a series and iterate in turn. Nothing is stored or marked to make this
happen; "receives a series" is the whole rule.

Do its inputs agree? A node fed series on several inputs needs their
indexes to be related: one must be the other, or a child of it (a child
index is made when a node splits each row into several). The node runs
once per row of the deepest one; a value on a shallower, parent index is
the same for every child row under it. Two indexes that are not related
are the fatal ``misaligned``.

Before those questions, inputs with no type of their own are typed from
their edges: a parameter declared ``Any`` takes the element type of what
arrives, and the node runs once per row if a series arrives; a
``**inputs`` parameter takes what arrives per edge — ``**inputs: Single``
receives each value whole, series and all, and never makes the node run
per row; ``**inputs: Series`` treats each as a series to reduce. Once
every input is typed, ``compute_outputs`` is asked with the arriving
types and the node's outputs are complete.

A ``Series[X]`` input receives a whole series; what it does depends on
the index that series is on. One series on a child index is a reduction:
the node runs once per parent row and receives the child rows under it.
One series on a root index is received whole, once. Anything else
feeding the input — several scalars, several series on different
indexes, a typed-in list, a default — is gathered onto a fresh index
that belongs to the input, and the node runs once.

Every one of those answers is written down as the input's receive record
(``conductor.graph.receive``): one row at a time on which index, broadcast,
whole, grouped at which depth, or gathered. The engine reads the record and decides
nothing of this again.

A scalar input holding a typed-in sequence (a multi-file upload, a list
typed by hand) is a series on an index of its own, and the node runs once
per row of it, exactly as it would for a series arriving on an edge.

Indexes are only named here; the engine adds rows to them later. A node's
series output sits on ``Index(node_id)`` — a root when the node runs
once, a child of the node's own index when it runs per row; a gathering
``Series[X]`` input sits on ``Index(ref)``. The compiler reasons about
which index, never about how many rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from conductor.dtype import DType
from conductor.dtype_ref import description_of, name_of
from conductor.errors import Refuses
from conductor.graph.binding import From
from conductor.graph.embedding import SEPARATOR, address, as_drawn, under
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.receive import Broadcast, Gather, Group, Iterate, Receive, Whole
from conductor.graph.views import field_problems
from conductor.interface import Interface
from conductor.metadata import Input, Output
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph


@dataclass(frozen=True)
class Iteration:
    """What ``derive`` returns.

    ``iterated``: for each node, the index it runs once per row of, or
    ``None`` when it runs once; for each embedded graph whose entering
    series agree, the index its inner nodes run per row of. A node has an
    entry exactly when the walk derived it. ``types``: the type that travels on every
    field; ``indexes``: for a field carrying a series, where its rows come
    from (``None`` otherwise). ``receives``: for every input, how a unit
    receives its value. ``interfaces``: each node's inputs and outputs
    with the types the edges gave them — what ``CompiledGraph.node(...).interface``
    serves from then on. ``problems``: what went wrong.
    """

    iterated: Mapping[str, Index | None]
    types: Mapping[Ref, Any]
    indexes: Mapping[Ref, Index | None]
    receives: Mapping[Ref, Receive]
    interfaces: Mapping[str, Interface]
    problems: tuple[Problem, ...]


def derive(
    nodes: Sequence[GraphNode],
    interfaces: Mapping[str, Interface],
    versions: Mapping[str, Any],
    definitions: Mapping[str, type[NodeDefinition]],
    statics: Mapping[str, Mapping[str, Any]],
    listed: Mapping[str, frozenset[str]],
    graphs: frozenset[str],
) -> Iteration:
    """Walk ``nodes`` in execution order and record, for each, whether it
    runs once per row, what type every field carries, how every input
    receives, and its completed inputs and outputs.

    ``nodes`` holds only nodes whose edges all point at existing nodes
    (the compiler leaves the rest out). A node fed by a node that was
    left out or could not be resolved gets no entry in ``iterated``,
    ``types``, ``indexes`` or ``receives`` — the broken source carries the
    problem, and nothing downstream of a fault is guessed at — but its
    inputs and outputs are still completed from what did arrive, so an
    editor can draw its handles.

    ``statics`` holds the values the author typed, by node and input, each
    read through its declared type; ``listed`` names, per node, the scalar
    inputs where the author typed many values (three files, a list of
    texts), which the node runs once per value of.

    ``compute_outputs`` is asked once per node, here, with ``arriving``
    (the type each connected input receives per call) and the values the
    author typed laid over the declaration's defaults. A hook that raises
    ``Refuses`` makes its code and message the node's one fatal problem.

    A node in ``graphs``, whose version is a compiled graph, is walked as
    one node with that graph's interface; ``Compilation.place_graphs`` lifts
    the graph under the row the walk decides for it.
    """
    walk = _Walk(nodes, interfaces, versions, definitions, statics, listed, graphs)
    for node in nodes:
        walk.visit(node)
    return walk.result()


@dataclass(frozen=True, slots=True)
class _Carried:
    """What one field carries, while the walk works: its type, and the index
    its rows come from when it is a series (``None`` otherwise). Split into
    ``types`` and ``indexes`` when the walk ends."""

    dtype: Any
    index: Index | None


@dataclass
class _Arrivals:
    """What one node's inputs receive, gathered while its edges are read.

    ``arrivals``: per input, by address, what it carries — the type, and the index for
    a series. ``receives``: per input, how a unit receives it. ``arriving``:
    per connected input, the type the node sees per call (an element,
    where a series is sliced per row) — what ``compute_outputs`` is told.
    ``demands``: the indexes the node must run per row of, each with the
    input that demands it. ``bound``: the types the edges gave to inputs
    that had none of their own. ``broken`` is set when a source cannot be
    read and the walk stopped at that input.
    """

    arrivals: dict[Ref, _Carried] = dataclasses.field(default_factory=dict)
    receives: dict[Ref, Receive] = dataclasses.field(default_factory=dict)
    arriving: dict[str, Any] = dataclasses.field(default_factory=dict)
    demands: list[tuple[Index, Ref]] = dataclasses.field(default_factory=list)
    bound: dict[str, Any] = dataclasses.field(default_factory=dict)
    broken: bool = False


class _Walk:
    """The walk over the edges, one node at a time in execution order.

    ``visit`` reads what arrives on a node's inputs, decides whether it
    runs once per row, completes its inputs and outputs and records what
    every field carries; later nodes read those records as their sources.
    ``result`` freezes what was found into an ``Iteration``.
    """

    def __init__(
        self,
        nodes: Sequence[GraphNode],
        interfaces: Mapping[str, Interface],
        versions: Mapping[str, Any],
        definitions: Mapping[str, type[NodeDefinition]],
        statics: Mapping[str, Mapping[str, Any]],
        listed: Mapping[str, frozenset[str]],
        graphs: frozenset[str],
    ) -> None:
        self.nodes = {node.id: node for node in nodes}
        #: Each node's inputs and outputs as the input hook answered them,
        #: before the walk types them.
        self.asked = interfaces
        self.versions = versions
        #: The class each node resolved to, which ``compute_outputs`` is asked of.
        self.definitions = definitions
        self.statics = statics
        self.listed = listed
        #: The nodes whose version is a compiled graph (``Expansion.graphs``).
        self.graphs = graphs
        #: The index each visited node runs once per row of (``None``: once).
        self.iterated: dict[str, Index | None] = {}
        #: What every output of every visited node carries: the sources a
        #: later node's edges may name. Only outputs are sources.
        self.carried: dict[Ref, _Carried] = {}
        #: What every input of every visited node carries.
        self.arrived: dict[Ref, _Carried] = {}
        #: How every input of every visited node receives its value.
        self.receives: dict[Ref, Receive] = {}
        #: Each visited node's inputs and outputs, completed: every type the
        #: edges gave them, and the outputs as ``compute_outputs`` answered.
        #: What ``Iteration.interfaces`` serves.
        self.completed: dict[str, Interface] = {}
        self.problems: list[Problem] = []

    def result(self) -> Iteration:
        fields = {**self.arrived, **self.carried}
        return Iteration(
            iterated=self.iterated,
            types={ref: c.dtype for ref, c in fields.items()},
            indexes={ref: c.index for ref, c in fields.items()},
            receives=self.receives,
            interfaces=self.completed,
            problems=tuple(self.problems),
        )

    def visit(self, node: GraphNode) -> None:
        """Read ``node``'s inputs, then either derive everything about it or,
        when a source could not be read, complete its fields from what did
        arrive and derive nothing."""
        arrived = self._read_inputs(node)
        if arrived.broken:
            self._complete_without_deriving(node, arrived)
        else:
            self._derive(node, arrived)

    def _read_inputs(self, node: GraphNode) -> _Arrivals:
        """What arrives on each input of ``node``, how each is received, and whether each edge is allowed.

        An input nothing feeds carries what the author typed (a scalar, or a
        series on an index of its own). A connected input is checked: every
        source must have been visited and have that output; a ``**inputs``
        parameter takes one edge whole; an input with no type of its own
        takes its first edge's; ``accepts`` judges every edge; a scalar input
        fed a series demands its index, a ``Series[X]`` input fed one series
        reduces on the parent index or receives the series whole, and fed
        anything else gathers onto a fresh index. Stops at the first input
        that cannot be read and says so in ``broken``.
        """
        interface = self.asked[node.id]
        found = _Arrivals()

        graphs = self.graphs if node.id in self.graphs else ()
        for inp in interface.inputs:
            ref = address(node.id, inp.name, graphs) if graphs else Ref(node.id, inp.name)
            binding = node.bindings.get(inp.name)
            if not isinstance(binding, From):
                carried, receive = self._originates(inp, ref, inp.name in self.listed[node.id])
                found.arrivals[ref], found.receives[ref] = carried, receive
                if isinstance(receive, Iterate):
                    found.demands.append((receive.index, ref))
                continue
            missing = [source for source in binding.refs if source not in self.carried]
            if missing:
                # A source that was resolved but has no such output: the ref names
                # an output it does not have, or an input. A source that was never
                # resolved carries its own problem and is not reported again here.
                self.problems.extend(
                    problem("unknown_ref_output", node.id, inp.name, source=str(source))
                    for source in missing
                    if source.node_id in self.iterated or (self.graphs and as_drawn(source, self.graphs)[0] in self.iterated)
                )
                found.broken = True
                return found
            sources = [self.carried[source] for source in binding.refs]
            if self._whole(node.id, inp):
                # A `**inputs` parameter takes what arrives, whole. It is passed
                # as a keyword argument, so its name must be an identifier.
                if not inp.name.isidentifier():
                    self.problems.append(problem("parameter_name_invalid", node.id, inp.name))
                    found.broken = True
                    return found
                if len(sources) > 1:
                    self.problems.append(problem("one_edge_per_parameter", node.id, inp.name))
                    found.broken = True
                    return found
                refused = self._refuses_whole(node.id, inp.name, sources[0].dtype)
                if refused is not None:
                    self.problems.append(refused)
                    found.broken = True
                    return found
                found.bound[inp.name] = sources[0].dtype
                found.arriving[inp.name] = sources[0].dtype
                found.arrivals[ref] = sources[0]
                found.receives[ref] = Whole()
                continue
            target = inp.dtype
            if target is Any or target.element is Any:
                # An input with no type of its own takes the type of its first
                # edge; the other edges are checked against it.
                element = sources[0].dtype.element or sources[0].dtype
                target = Series[element] if getattr(inp.dtype, "element", None) is not None else element
                found.bound[inp.name] = target
            for source_ref, source in zip(binding.refs, sources, strict=True):
                self.problems.extend(self._admit(target, ref, source_ref, source))
            if target.element is None:
                if len(sources) > 1 and not self._one_index(sources):
                    self.problems.append(problem("union_needs_one_index", node.id, inp.name))
                    found.broken = True
                    return found
                arriving = sources[0]  # one ref, or a union of several on one index
                found.arriving[inp.name] = arriving.dtype.element or arriving.dtype
                if arriving.index is None:
                    found.receives[ref] = Broadcast()
                else:
                    found.receives[ref] = Iterate(arriving.index)
                    found.demands.append((arriving.index, ref))
            elif self._one_index(sources):
                arriving = sources[0]  # one series, or a union of several on one index
                found.arriving[inp.name] = arriving.dtype
                if arriving.index.parent is not None:
                    found.receives[ref] = Group(arriving.index, arriving.index.parent.depth)
                    found.demands.append((arriving.index.parent, ref))
                else:
                    found.receives[ref] = Whole()
            else:
                arriving = _Carried(target, Index(ref))
                found.arriving[inp.name] = target
                found.receives[ref] = Gather(arriving.index)
            found.arrivals[ref] = arriving
        return found

    def _derive(self, node: GraphNode, arrived: _Arrivals) -> None:
        """Every input read: complete the node's fields, ask its outputs,
        decide the index it runs per row of, and record what each field
        carries and how each input receives."""
        interface = self.asked[node.id]
        inputs = tuple(self._typed(inp, arrived.bound) for inp in interface.inputs)
        answered = self._outputs(node, arrived.arriving)
        if isinstance(answered, Problem):
            # The refusal is the node's problem: fatal, with no `no_outputs`
            # beside it, and nothing is derived past it.
            self.problems.append(answered)
            self.completed[node.id] = replace(self.versions[node.id].interface, inputs=inputs, outputs=())
            return
        outputs = answered
        self.completed[node.id] = replace(self.versions[node.id].interface, inputs=inputs, outputs=outputs)
        faults = field_problems(node.id, outputs, frozenset(i.name for i in inputs), untyped_ok=False)
        if faults:
            # An output no edge could carry, or a name used twice: the node has
            # its interface, so an editor draws it, and nothing is derived
            # from it, so no reader is told the output is missing.
            self.problems.extend(faults)
            return
        if not outputs:
            self.problems.append(problem("no_outputs", node.id))
        iteration_index, disagreeing = self._iteration_index(arrived.demands)
        if disagreeing is not None:
            self.problems.append(self._misaligned(node.id, *disagreeing))
            return
        self.iterated[node.id] = iteration_index
        self.arrived.update(arrived.arrivals)
        self.receives.update(arrived.receives)
        node_index: Index | None = None
        version = self.versions[node.id]
        for out in outputs:
            ref = address(node.id, out.name, self.graphs)
            if node.id in self.graphs:
                self.carried[ref] = self._carried_out_of_graph(node.id, version, out, iteration_index)
            elif out.dtype.element is not None:
                node_index = node_index or Index(node.id, parent=iteration_index)
                self.carried[ref] = _Carried(out.dtype, node_index)
            elif iteration_index is not None:
                self.carried[ref] = _Carried(Series[out.dtype], iteration_index)
            else:
                self.carried[ref] = _Carried(out.dtype, None)

    @staticmethod
    def _carried_out_of_graph(node_id: str, graph: CompiledGraph, out: Output, row: Index | None) -> _Carried:
        """What an output of a compiled graph placed as one node carries, as it
        will once the graph's nodes are in its place: the type and rows the
        output has inside the graph, its rows named
        under the placed node and hung under the ``row`` it runs on
        (``embedding.under``). Two outputs share rows only when a node inside
        births both; an output that runs once inside is one value per
        ``row``, or one value."""
        ref, iteration = Ref(out.name), graph._compilation.iteration
        dtype, index = iteration.types[ref], iteration.indexes[ref]
        if index is None:
            return _Carried(Series[dtype], row) if row is not None else _Carried(dtype, None)
        return _Carried(dtype, under(index, f"{node_id}{SEPARATOR}", row))

    def _complete_without_deriving(self, node: GraphNode, arrived: _Arrivals) -> None:
        """A source was broken: complete the inputs and outputs from what
        arrived so an editor can draw the node, and derive nothing."""
        answered = self._outputs(node, arrived.arriving)
        if isinstance(answered, Problem):
            # The broken source carries the fault; a refusal about its missing
            # arrival would report the same fact twice.
            answered = ()
        inputs = tuple(self._typed(inp, arrived.bound) for inp in self.asked[node.id].inputs)
        self.completed[node.id] = replace(self.versions[node.id].interface, inputs=inputs, outputs=answered)
        # An output still ``Any`` here is one an edge would have typed; only a
        # name used twice or a type no edge can carry is the node's own fault.
        self.problems.extend(field_problems(node.id, answered, frozenset(i.name for i in inputs), untyped_ok=True))

    def _outputs(self, node: GraphNode, arriving: Mapping[str, Any]) -> tuple[Output, ...] | Problem:
        """Ask the node's ``compute_outputs`` now that it can be told what
        arrives, with the values the author typed laid over the
        declaration's defaults. An optional input is always there; a
        required one the author left empty, or connected, is not.

        A hook that cannot answer for these values and arrivals raises
        ``Refuses(code, message)``, and the refusal comes back as the node's
        one fatal ``Problem`` — the hook chose the code and the message,
        compile only records which node it belongs to.
        """
        version = self.versions[node.id]
        if node.id in self.graphs:
            return version.interface.outputs  # a compiled graph placed as a node: it has no hooks
        declared = (*version.interface.inputs, *self.asked[node.id].inputs)
        values = {**{i.name: i.default for i in declared if i.optional}, **self.statics[node.id]}
        try:
            return self.definitions[node.id]().compute_outputs(version.interface.outputs, values, arriving)
        except Refuses as refusal:
            return Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node.id)

    def _whole(self, node_id: str, inp: Input) -> bool:
        """Is ``inp`` a ``**inputs: Single`` parameter — a connected name the
        version did not declare, received whole?"""
        version = self.versions[node_id]
        return version.interface.open == "single" and inp.name not in {i.name for i in version.interface.inputs}

    @staticmethod
    def _typed(field: Any, bound: Mapping[str, Any]) -> Any:
        """``field`` with the type its edge gave it, if an edge gave one."""
        if field.name in bound:
            return field.model_copy(update={"dtype": bound[field.name]})
        return field

    @staticmethod
    def _one_index(sources: Sequence[_Carried]) -> bool:
        """Do all sources carry a series on one and the same index? Then the edges together are one set of rows."""
        return all(s.index is not None for s in sources) and len({s.index for s in sources}) == 1

    @staticmethod
    def _refuses_whole(node_id: str, field: str, dtype: type[DType]) -> Problem | None:
        """A source type that says it cannot be handed over whole — a table whose
        columns nobody stated — is refused on the field, fatal, with the type's
        own message. Asked only where a node will *read* the value, a
        ``**inputs`` parameter; an ``Any`` input only passes the value on, and
        nothing is asked.
        """
        asked = dtype.element or dtype
        try:
            answered = asked.refuses_whole()
        except Refuses as refusal:
            return Problem(
                code=refusal.code, message=f"Field '{field}': {refusal.message}", fatal=True, node_id=node_id,
                field=field, details={"inner_message": refusal.message},
            )
        if answered is not None:
            raise TypeError(
                f"{asked.__name__}.refuses_whole() returned {answered!r}; a type refuses by raising Refuses "
                "and returns None otherwise"
            )
        return None

    @staticmethod
    def _originates(inp: Input, ref: Ref, listed: bool) -> tuple[_Carried, Receive]:
        """What a field carries when no edge feeds it, and how it is received:
        a scalar, broadcast; or a series on an index of the field's own —
        always for a ``Series[X]`` input, received whole; and for a scalar
        input when the author typed many values (``listed``), received one
        per row, so the node runs once per value.
        """
        if getattr(inp.dtype, "element", None) is not None:  # only a Series type has an element
            return _Carried(inp.dtype, Index(ref)), Whole()
        if listed:
            own = Index(ref)
            return _Carried(Series[inp.dtype], own), Iterate(own)
        return _Carried(inp.dtype, None), Broadcast()

    @staticmethod
    def _admit(target: Any, ref: Ref, source_ref: Ref, source: _Carried) -> list[Problem]:
        """May what arrives land on ``ref``? ``target.accepts`` decides."""
        if not target.accepts(source.dtype):
            return [problem(
                "type_mismatch", ref.node_id, ref.field,
                source=str(source_ref),
                source_type=description_of(source.dtype),
                target_type=description_of(target),
                source_said=name_of(source.dtype),
                target_said=name_of(target),
            )]
        return []

    @staticmethod
    def _iteration_index(demands: list[tuple[Index, Ref]]) -> tuple[Index | None, tuple[Ref, Ref] | None]:
        """The index a node runs once per row of, given the indexes its inputs
        demand — or the two fields whose indexes disagree.

        Every demanded index must lie on one line of descent: an index and the
        indexes opened from its rows. The deepest is the answer; a shallower
        one is an ancestor whose rows repeat down to it.
        """
        if not demands:
            return None, None
        deepest, deepest_ref = max(demands, key=lambda demand: demand[0].depth)
        lineage = set(deepest.lineage)
        for index, ref in demands:
            if index not in lineage:
                return None, (deepest_ref, ref)
        return deepest, None

    @staticmethod
    def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
        return problem("misaligned", node_id, a=str(a), b=str(b))
