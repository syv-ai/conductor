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

An embedded graph (a node whose version is a graph, inlined by ``expand``)
has a boundary. Where a series enters it through a scalar field, the
whole inner graph runs once per row of that index: every inner node runs
at least once per row of it, every index opened inside it is a child of
it, and an inner reduction over the entering index receives one row at a
time — so the inner graph behaves exactly as it would standalone, once
per outer row. That index is found once, when the walk reaches the first
inner node, from the edges crossing in (``_Walk._entering_index``), and
is one more demand on every inner node: an inner node fed rows from an
index unrelated to it is ``misaligned``, as it would be on a flat node.

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
from typing import Any

from conductor.dtype import DType
from conductor.dtype_ref import description_of, name_of
from conductor.errors import Refuses
from conductor.graph.binding import From
from conductor.graph.expand import authored_ref
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem, problem
from conductor.graph.receive import Broadcast, Gather, Group, Iterate, Receive, Whole
from conductor.graph.views import field_problems
from conductor.interface import Interface
from conductor.metadata import Input, Output
from conductor.node import NodeVersion
from conductor.ref import Ref
from conductor.registry import NodeRegistry
from conductor.series import Index, Series


@dataclass(frozen=True)
class Iteration:
    """What ``derive`` returns.

    ``iterated``: for each node, the index it runs once per row of, or
    ``None`` when it runs once; for each embedded graph, the index its
    inner nodes run per row of. ``types``: the type that travels on every
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
    versions: Mapping[str, NodeVersion],
    registry: NodeRegistry,
    statics: Mapping[str, Mapping[str, Any]],
    listed: Mapping[str, frozenset[str]],
    *,
    placement_of: Mapping[str, str | None] = {},
    members: Mapping[str, Sequence[str]] = {},
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

    ``placement_of`` and ``members`` come from ``expand``: which embedded
    graph each node belongs to, and which nodes each embedded graph holds.
    The index an embedded graph runs per row of is found when its first
    node is reached.
    """
    walk = _Walk(nodes, interfaces, versions, registry, statics, listed, placement_of, members)
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

    ``arrivals``: per input, what it carries — the type, and the index for
    a series. ``receives``: per input, how a unit receives it. ``arriving``:
    per connected input, the type the node sees per call (an element,
    where a series is sliced per row) — what ``compute_outputs`` is told.
    ``demands``: the indexes the node must run per row of, each with the
    input that demands it. ``bound``: the types the edges gave to inputs
    that had none of their own. ``broken`` is set when a source cannot be
    read and the walk stopped at that input.
    """

    arrivals: dict[str, _Carried] = dataclasses.field(default_factory=dict)
    receives: dict[str, Receive] = dataclasses.field(default_factory=dict)
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
        versions: Mapping[str, NodeVersion],
        registry: NodeRegistry,
        statics: Mapping[str, Mapping[str, Any]],
        listed: Mapping[str, frozenset[str]],
        placement_of: Mapping[str, str | None],
        members: Mapping[str, Sequence[str]],
    ) -> None:
        self.nodes = {node.id: node for node in nodes}
        #: Each node's inputs and outputs as the input hook answered them,
        #: before the walk types them.
        self.asked = interfaces
        self.versions = versions
        self.registry = registry
        self.statics = statics
        self.listed = listed
        self.placement_of = placement_of
        self.members = members
        #: The index each visited node runs once per row of (``None``: once);
        #: each embedded graph's, under its placement id.
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
        #: The index each embedded graph's inner nodes run per row of, found
        #: when the walk reaches its first inner node, with the inner field
        #: the series entered through; ``None`` when no series enters.
        self.scopes: dict[str, tuple[Index, Ref] | None] = {}
        #: Embedded graphs whose entering series disagree; their inner nodes
        #: are not derived.
        self.misaligned_placements: set[str] = set()

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
        scope = self._scope_of(node)
        arrived = self._read_inputs(node, scope)
        if arrived.broken or self.placement_of.get(node.id) in self.misaligned_placements:
            self._complete_without_deriving(node, arrived)
        else:
            self._derive(node, arrived, scope)

    def _scope_of(self, node: GraphNode) -> tuple[Index, Ref] | None:
        """The index that the embedded graph containing ``node`` runs once per
        row of, and the inner field it entered through; ``None`` when ``node``
        is at the top level or that graph runs once. Found on the first inner
        node visited and recorded under the id of the node that embeds the
        graph. An embedded graph whose entering series disagree is recorded
        in ``misaligned_placements``: its inner nodes are completed but not
        derived, so the one ``misaligned`` on the placement is the only report."""
        placement = self.placement_of.get(node.id)
        if placement is None:
            return None
        if placement not in self.scopes:
            entering = self._entering_index(placement)
            if isinstance(entering, Problem):
                self.problems.append(entering)
                self.misaligned_placements.add(placement)
                entering = None
            self.scopes[placement] = entering
            self.iterated[placement] = None if entering is None else entering[0]
        return self.scopes[placement]

    def _read_inputs(self, node: GraphNode, scope: tuple[Index, Ref] | None) -> _Arrivals:
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
        scope_index = None if scope is None else scope[0]
        found = _Arrivals()

        for inp in interface.inputs:
            ref = Ref(node.id, inp.name)
            binding = node.bindings.get(inp.name)
            if not isinstance(binding, From):
                carried, receive = self._originates(inp, ref, inp.name in self.listed[node.id], scope_index)
                found.arrivals[inp.name], found.receives[inp.name] = carried, receive
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
                    for source in missing if source.node_id in self.iterated
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
                found.arrivals[inp.name] = sources[0]
                found.receives[inp.name] = Whole()
                continue
            target = inp.dtype
            if target is Any or getattr(target, "element", None) is Any:
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
                    found.receives[inp.name] = Broadcast()
                else:
                    found.receives[inp.name] = Iterate(arriving.index)
                    found.demands.append((arriving.index, ref))
            elif self._one_index(sources):
                arriving = sources[0]  # one series, or a union of several on one index
                found.arriving[inp.name] = arriving.dtype
                if scope_index is not None and arriving.index == scope_index:
                    # The series entered this embedded graph from outside, so the
                    # reduction runs once per outer row and receives the one row
                    # under it.
                    found.receives[inp.name] = Group(arriving.index, arriving.index.depth)
                    found.demands.append((arriving.index, ref))
                elif arriving.index.parent is not None:
                    found.receives[inp.name] = Group(arriving.index, arriving.index.parent.depth)
                    found.demands.append((arriving.index.parent, ref))
                else:
                    found.receives[inp.name] = Whole()
            else:
                arriving = _Carried(target, Index(ref, parent=scope_index))
                found.arriving[inp.name] = target
                found.receives[inp.name] = Gather(arriving.index)
            found.arrivals[inp.name] = arriving
        return found

    def _derive(self, node: GraphNode, arrived: _Arrivals, scope: tuple[Index, Ref] | None) -> None:
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
        # Inside an embedded graph that runs per row, every node runs at least
        # once per outer row: that index is one more demand, named by the inner
        # field the series entered through, on the same line of descent as the
        # node's own. A node nothing from outside reaches runs per row of it; a
        # node fed from a shallower index repeats its value down to it; a node
        # fed rows unrelated to it is misaligned.
        demands = arrived.demands if scope is None else [*arrived.demands, scope]
        iteration_index, disagreeing = self._iteration_index(demands)
        if disagreeing is not None:
            self.problems.append(self._misaligned(node.id, *disagreeing))
            return
        self.iterated[node.id] = iteration_index
        self.arrived.update({Ref(node.id, name): c for name, c in arrived.arrivals.items()})
        self.receives.update({Ref(node.id, name): r for name, r in arrived.receives.items()})
        scope_index = None if scope is None else scope[0]
        node_index: Index | None = None
        for out in outputs:
            ref = Ref(node.id, out.name)
            if getattr(out.dtype, "element", None) is not None:
                node_index = node_index or Index(node.id, parent=iteration_index or scope_index)
                self.carried[ref] = _Carried(out.dtype, node_index)
            elif iteration_index is not None:
                self.carried[ref] = _Carried(Series[out.dtype], iteration_index)
            else:
                self.carried[ref] = _Carried(out.dtype, None)

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
        declaration's defaults (so a hook indexes ``values[...]`` without a
        guard).

        A hook that cannot answer for these values and arrivals raises
        ``Refuses(code, message)``, and the refusal comes back as the node's
        one fatal ``Problem`` — the hook chose the code and the message,
        compile only records which node it belongs to.
        """
        version = self.versions[node.id]
        declared = (*version.interface.inputs, *self.asked[node.id].inputs)
        values = {**{i.name: i.default for i in declared if i.optional}, **self.statics[node.id]}
        try:
            return self.registry[node.type]().compute_outputs(version.interface.outputs, values, arriving)
        except Refuses as refusal:
            return Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node.id)

    def _entering_index(self, placement: str) -> tuple[Index, Ref] | None | Problem:
        """The index an embedded graph's inner nodes run once per row of, and
        the inner field it entered through: the deepest index among the
        series that enter it from outside through a scalar field, or ``None``
        when no series enters that way (the inner nodes then run once each,
        as if the graph were flat). A ``**inputs`` parameter receives its
        edge whole and is not a way in. Two entering series on unrelated
        indexes are the ``misaligned`` problem returned, on the embedded
        graph's node with the two fields in the author's addresses, exactly
        as they would be on a single node.
        """
        block = self.members[placement]
        inside = set(block)
        demands: list[tuple[Index, Ref]] = []
        for node_id in block:
            if node_id not in self.nodes:
                continue  # a broken edge; the node carries its own problem and was left out of the walk
            node = self.nodes[node_id]
            for inp in self.asked[node_id].inputs:
                binding = node.bindings.get(inp.name)
                if not isinstance(binding, From) or getattr(inp.dtype, "element", None) is not None or self._whole(node_id, inp):
                    continue
                entering = {
                    self.carried[source].index
                    for source in binding.refs
                    if source.node_id not in inside and source in self.carried and self.carried[source].index is not None
                }
                # One input, one index: several edges on different indexes into
                # one scalar input is the node's own fault (``union_needs_one_index``),
                # reported when the node is read, not a disagreement between fields.
                if len(entering) == 1:
                    demands.append((entering.pop(), Ref(node_id, inp.name)))
        index, disagreeing = self._iteration_index(demands)
        if disagreeing is not None:
            a, b = disagreeing
            return self._misaligned(placement, authored_ref(a), authored_ref(b))
        if index is None:
            return None
        entered_through = next(ref for demanded, ref in demands if demanded == index)
        return index, entered_through

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
    def _originates(inp: Input, ref: Ref, listed: bool, scope: Index | None) -> tuple[_Carried, Receive]:
        """What a field carries when no edge feeds it, and how it is received:
        a scalar, broadcast; or a series on an index of the field's own —
        always for a ``Series[X]`` input, received whole; and for a scalar
        input when the author typed many values (``listed``), received one
        per row, so the node runs once per value. Inside an embedded graph
        that runs per row, that index is a child of the entering one
        (``scope``).
        """
        if getattr(inp.dtype, "element", None) is not None:  # only a Series type has an element
            return _Carried(inp.dtype, Index(ref, parent=scope)), Whole()
        if listed:
            own = Index(ref, parent=scope)
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
        deepest, deepest_ref = max(demands, key=lambda demand: len(_lineage(demand[0])))
        lineage = set(_lineage(deepest))
        for index, ref in demands:
            if index not in lineage:
                return None, (deepest_ref, ref)
        return deepest, None

    @staticmethod
    def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
        return problem("misaligned", node_id, a=str(a), b=str(b))


def _lineage(index: Index) -> list[Index]:
    """This index and every index it was opened from, nearest first."""
    found: list[Index] = []
    current: Index | None = index
    while current is not None:
        found.append(current)
        current = current.parent
    return found
