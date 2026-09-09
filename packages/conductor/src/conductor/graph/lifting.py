"""One walk over the edges: what arrives on every field, and whether it may.

Called by compile once every node's inputs are known. It walks
the nodes in execution order and, for each edge into each input, answers
three questions that turn out to be one:

* **May this edge land?** ``target.accepts(source)`` decides; a series is
  judged by its element type. A refusal is the fatal ``type_mismatch``.
* **Does the node run once, or once per row?** A *series* is a value with
  many rows, and an *index* names where those rows come from. A series
  arriving on a scalar input means the node runs once per row of that
  index — we say the node is *lifted* on the index. Its other scalar
  inputs are the same value every row, every output becomes a series on
  the same index, and nodes downstream receive a series and are lifted
  in turn. Nothing is stored or marked to make this happen; "receives a
  series" is the whole rule.
* **Do its inputs agree?** A node fed series on several inputs needs their
  indexes on one line of descent — an index and the indexes opened from
  its rows. The deepest is the one the node runs per row of; a shallower
  one is an ancestor whose rows repeat down. Two indexes that are not
  related are the fatal ``misaligned``.

Before those questions, inputs with no type of their own are typed from
their edges: a parameter declared ``Any`` takes the element type of what
arrives, and is lifted if a series arrives; a ``**inputs`` parameter
takes what arrives per edge — ``**inputs: Single`` receives each value
whole, series and all, and is never lifted; ``**inputs: Series`` treats
each as a series to reduce. Once every input is typed, ``compute_outputs``
is asked with the arriving types and the node's outputs are complete.

A ``Series[X]`` input receives a whole series and is read by the index
that arrives. One series on an index opened from another index's rows is
a *reduction*: the node runs once per parent row and receives the child
rows under it. One series on a root index is received whole, once.
Anything else feeding the input — several scalars, several series on
different indexes, a typed-in list, a default — is *gathered* onto a
fresh index that belongs to the input, and the node runs once.

An embedded flow (a node whose version is a graph, inlined by ``expand``)
has a boundary. Where a series enters it through a scalar field, the
whole inner graph runs once per row of that index: every inner node is
lifted on at least that index, every index opened inside it is a child
of it, and an inner reduction over the entering index receives one row
at a time — so the inner flow behaves exactly as it would standalone,
once per outer row. That index is found once, when the walk reaches the
first inner node, from the edges crossing in (``_Walk._entering_index``).

A scalar input holding a typed-in sequence (a multi-file upload, a list
typed by hand) is a series on an index of its own and lifts the node as
a connected series would.

Indexes are only named here; the engine adds rows to them later. A node's
series output sits on ``Index(node_id)`` — a root when the node runs
once, a child of the node's own index when it runs per row; a gathering
``Series[X]`` input sits on ``Index(ref)``. Compile reasons about *which*
index, never about how many rows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from conductor.dtype import DType
from conductor.dtype_ref import description_of
from conductor.graph.binding import Edges, many
from conductor.graph.compiled import Carried
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem, problem
from conductor.metadata import Input, Output, Roster
from conductor.node import NodeVersion, Refuses
from conductor.ref import Ref
from conductor.registry import NodeRegistry
from conductor.series import Index, Series


@dataclass(frozen=True)
class Lifting:
    """What ``derive`` returns.

    ``lifted``: for each node, the index it runs once per row of, or
    ``None`` when it runs once; for each embedded graph, the index its
    inner nodes run per row of. ``carried``: the type on every field and,
    for a series, its index. ``rosters``: each node's inputs and outputs
    with the types the edges gave them — what ``CompiledGraph.roster``
    serves from then on. ``problems``: what went wrong.
    """

    lifted: Mapping[str, Index | None]
    carried: Mapping[Ref, Carried]
    rosters: Mapping[str, Roster]
    problems: tuple[Problem, ...]


def derive(
    nodes: Sequence[GraphNode],
    rosters: Mapping[str, Roster],
    versions: Mapping[str, NodeVersion],
    registry: NodeRegistry,
    statics: Mapping[str, Mapping[str, Any]],
    *,
    placement_of: Mapping[str, str | None] = {},
    members: Mapping[str, Sequence[str]] = {},
) -> Lifting:
    """Walk ``nodes`` in execution order and record, for each, whether it
    runs once per row, what type every field carries, and its completed
    inputs and outputs.

    ``nodes`` holds only nodes whose edges all point at existing nodes
    (compile leaves the rest out). A node fed by a node that was
    left out or could not be resolved gets no entry in ``lifted`` or
    ``carried`` — the broken source carries the problem, and nothing
    downstream of a fault is guessed at — but its inputs and outputs are
    still completed from what did arrive, so an editor can draw its
    handles.

    ``compute_outputs`` is asked once per node, here, with ``arriving``
    (the type each connected input receives per call) and the values the
    author typed laid over the declaration's defaults. A hook that raises
    ``Refuses`` makes its code and message the node's one fatal problem.

    ``placement_of`` and ``members`` come from ``expand``: which embedded
    graph each node belongs to, and which nodes each embedded graph holds.
    The index an embedded graph runs per row of is found when its first
    node is reached.
    """
    walk = _Walk(nodes, rosters, versions, registry, statics, placement_of, members)
    for node in nodes:
        walk.visit(node)
    return walk.result()


@dataclass
class _Arrivals:
    """What one node's inputs receive, gathered while its edges are read.

    ``arrivals``: per input, what it carries — the type, and the index for
    a series. ``receives``: per connected input, the type the node sees
    per call (an element, where a series is sliced per row). ``demands``:
    the indexes the node must run per row of, each with the input that
    demands it. ``bound``: the types the edges gave to inputs that had
    none of their own. ``broken`` is set when a source cannot be read and
    the walk stopped at that input.
    """

    arrivals: dict[str, Carried] = dataclasses.field(default_factory=dict)
    receives: dict[str, Any] = dataclasses.field(default_factory=dict)
    demands: list[tuple[Index, Ref]] = dataclasses.field(default_factory=list)
    bound: dict[str, Any] = dataclasses.field(default_factory=dict)
    broken: bool = False


class _Walk:
    """The walk over the edges, one node at a time in execution order.

    ``visit`` reads what arrives on a node's inputs, decides whether it
    runs once per row, completes its inputs and outputs and records what
    every field carries; later nodes read those records as their sources.
    ``result`` freezes what was found into a ``Lifting``.
    """

    def __init__(
        self,
        nodes: Sequence[GraphNode],
        rosters: Mapping[str, Roster],
        versions: Mapping[str, NodeVersion],
        registry: NodeRegistry,
        statics: Mapping[str, Mapping[str, Any]],
        placement_of: Mapping[str, str | None],
        members: Mapping[str, Sequence[str]],
    ) -> None:
        self.nodes = {node.id: node for node in nodes}
        self.rosters = rosters
        self.versions = versions
        self.registry = registry
        self.statics = statics
        self.placement_of = placement_of
        self.members = members
        #: The index each visited node runs once per row of (``None``: once);
        #: each embedded graph's, under its placement id.
        self.lifted: dict[str, Index | None] = {}
        #: What every field of every visited node carries.
        self.carried: dict[Ref, Carried] = {}
        #: Each visited node's inputs and outputs, completed.
        self.completed: dict[str, Roster] = {}
        self.problems: list[Problem] = []
        #: The index each embedded graph's inner nodes run per row of, found
        #: when the walk reaches its first inner node.
        self.scopes: dict[str, Index | None] = {}

    def result(self) -> Lifting:
        return Lifting(
            lifted=self.lifted, carried=self.carried, rosters=self.completed, problems=tuple(self.problems)
        )

    def visit(self, node: GraphNode) -> None:
        """Read ``node``'s inputs, then either derive everything about it or,
        when a source could not be read, complete its fields from what did
        arrive and derive nothing."""
        scope = self._scope_of(node)
        arrived = self._read_inputs(node, scope)
        if arrived.broken:
            self._complete_without_deriving(node, arrived)
        else:
            self._derive(node, arrived, scope)

    def _scope_of(self, node: GraphNode) -> Index | None:
        """The index the embedded graph ``node`` sits in runs per row of, or
        ``None`` when it sits at the top or the graph runs once. Found on
        the first inner node visited and recorded for the placement."""
        placement = self.placement_of.get(node.id)
        if placement is None:
            return None
        if placement not in self.scopes:
            self.scopes[placement] = self._entering_index(placement)
            self.lifted[placement] = self.scopes[placement]
        return self.scopes[placement]

    def _read_inputs(self, node: GraphNode, scope: Index | None) -> _Arrivals:
        """What arrives on each input of ``node``, and whether it may.

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
        roster = self.rosters[node.id]
        version = self.versions[node.id]
        declared_names = {i.name for i in version.interface.inputs}
        found = _Arrivals()

        for inp in roster.inputs:
            ref = Ref(node.id, inp.name)
            binding = node.bindings.get(inp.name)
            whole = version.interface.open == "single" and inp.name not in declared_names
            if not isinstance(binding, Edges):
                found.arrivals[inp.name] = _originates(inp, ref, self.statics[node.id].get(inp.name), scope)
                if getattr(inp.dtype, "element", None) is None and found.arrivals[inp.name].index is not None:
                    found.demands.append((found.arrivals[inp.name].index, ref))
                continue
            missing = [source for source in binding.refs if source not in self.carried]
            if missing:
                # A source that was resolved but has no such field: the ref names
                # an output it does not have. A source that was never resolved
                # carries its own problem and is not reported again here.
                self.problems.extend(
                    problem("unknown_ref_output", node.id, inp.name, source=str(source))
                    for source in missing if source.node_id in self.lifted
                )
                found.broken = True
                return found
            sources = [self.carried[source] for source in binding.refs]
            if whole:
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
                refused = _refuses_whole(node.id, inp.name, sources[0].dtype)
                if refused is not None:
                    self.problems.append(refused)
                    found.broken = True
                    return found
                found.bound[inp.name] = sources[0].dtype
                found.receives[inp.name] = sources[0].dtype
                found.arrivals[inp.name] = sources[0]
                continue
            target = inp.dtype
            if target is Any or getattr(target, "element", None) is Any:
                # An input with no type of its own takes the type of its first
                # edge; the other edges are checked against it.
                element = sources[0].dtype.element or sources[0].dtype
                target = Series[element] if getattr(inp.dtype, "element", None) is not None else element
                found.bound[inp.name] = target
            for source_ref, source in zip(binding.refs, sources, strict=True):
                self.problems.extend(_admit(target, ref, source_ref, source))
            if target.element is None:
                if len(sources) > 1 and not _one_index(sources):
                    self.problems.append(problem("union_needs_one_index", node.id, inp.name))
                    found.broken = True
                    return found
                arriving = sources[0]  # one ref, or a union of several on one index
                found.receives[inp.name] = arriving.dtype.element or arriving.dtype
                if arriving.index is not None:
                    found.demands.append((arriving.index, ref))
            elif _one_index(sources):
                arriving = sources[0]  # one series, or a union of several on one index
                found.receives[inp.name] = arriving.dtype
                if scope is not None and arriving.index == scope:
                    # The series entered this embedded graph from outside, so the
                    # reduction runs once per outer row and receives the one row
                    # under it.
                    found.demands.append((arriving.index, ref))
                elif arriving.index.parent is not None:
                    found.demands.append((arriving.index.parent, ref))
            else:
                arriving = Carried(target, Index(ref, parent=scope))
                found.receives[inp.name] = target
            found.arrivals[inp.name] = arriving
        return found

    def _derive(self, node: GraphNode, arrived: _Arrivals, scope: Index | None) -> None:
        """Every input read: complete the node's fields, ask its outputs,
        decide the index it runs per row of, and record what each output
        carries."""
        roster = self.rosters[node.id]
        inputs = tuple(_typed(inp, arrived.bound) for inp in roster.inputs)
        answered = self._outputs(node, arrived.receives)
        if isinstance(answered, Problem):
            # The refusal is the node's problem: fatal, with no `no_outputs`
            # beside it, and nothing is derived past it.
            self.problems.append(answered)
            self.completed[node.id] = Roster(inputs=inputs, outputs=())
            return
        outputs = answered
        self.completed[node.id] = Roster(inputs=inputs, outputs=outputs)
        if not outputs:
            self.problems.append(problem("no_outputs", node.id))
        lift_index, disagreeing = _lift_index(arrived.demands)
        if disagreeing is not None:
            self.problems.append(_misaligned(node.id, *disagreeing))
            return
        # Inside an embedded graph that runs per row, every node runs at least
        # once per outer row: that index is one more demand on the same line
        # of descent. A node nothing from outside reaches runs per row of it;
        # a node fed from a shallower index repeats its value down to it.
        if scope is not None and (lift_index is None or scope not in _lineage(lift_index)):
            lift_index = scope
        self.lifted[node.id] = lift_index
        self.carried.update({Ref(node.id, name): c for name, c in arrived.arrivals.items()})
        node_index: Index | None = None
        for out in outputs:
            ref = Ref(node.id, out.name)
            if getattr(out.dtype, "element", None) is not None:
                node_index = node_index or Index(node.id, parent=lift_index or scope)
                self.carried[ref] = Carried(out.dtype, node_index)
            elif lift_index is not None:
                self.carried[ref] = Carried(Series[out.dtype], lift_index)
            else:
                self.carried[ref] = Carried(out.dtype, None)

    def _complete_without_deriving(self, node: GraphNode, arrived: _Arrivals) -> None:
        """A source was broken: complete the inputs and outputs from what
        arrived so an editor can draw the node, and derive nothing."""
        answered = self._outputs(node, arrived.receives)
        if isinstance(answered, Problem):
            # The broken source carries the fault; a refusal about its missing
            # arrival would report the same fact twice.
            answered = ()
        self.completed[node.id] = Roster(
            inputs=tuple(_typed(inp, arrived.bound) for inp in self.rosters[node.id].inputs),
            outputs=answered,
        )

    def _outputs(self, node: GraphNode, receives: Mapping[str, Any]) -> tuple[Output, ...] | Problem:
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
        declared = (*version.interface.inputs, *self.rosters[node.id].inputs)
        values = {**{i.name: i.default for i in declared if i.optional}, **self.statics[node.id]}
        try:
            return self.registry.get(node.type)().compute_outputs(version.interface.outputs, values, receives)
        except Refuses as refusal:
            return Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node.id)

    def _entering_index(self, placement: str) -> Index | None:
        """The index an embedded graph's inner nodes run once per row of: the
        deepest index among the series that enter it from outside through a
        scalar field, or ``None`` when no series enters that way (the inner
        nodes then run once each, as if the graph were flat). Two entering
        series on unrelated indexes are reported as ``misaligned`` on the
        embedded graph's node, exactly as they would be on a single node.
        """
        block = self.members[placement]
        inside = set(block)
        demands: list[tuple[Index, Ref]] = []
        for node_id in block:
            if node_id not in self.nodes:
                continue  # a broken edge; the node carries its own problem and was left out of the walk
            node = self.nodes[node_id]
            for inp in self.rosters[node_id].inputs:
                binding = node.bindings.get(inp.name)
                if not isinstance(binding, Edges) or getattr(inp.dtype, "element", None) is not None:
                    continue
                for source in binding.refs:
                    if source.node_id not in inside and source in self.carried and self.carried[source].index is not None:
                        demands.append((self.carried[source].index, Ref(node_id, inp.name)))
        index, disagreeing = _lift_index(demands)
        if disagreeing is not None:
            self.problems.append(_misaligned(placement, *disagreeing))
            return None
        return index


def _typed(field: Any, bound: Mapping[str, Any]) -> Any:
    """``field`` with the type its edge gave it, if an edge gave one."""
    if field.name in bound:
        return replace(field, dtype=bound[field.name])
    return field


def _one_index(sources: Sequence[Carried]) -> bool:
    """Do all sources carry a series on one and the same index? Then the edges are a union of rows."""
    return all(s.index is not None for s in sources) and len({s.index for s in sources}) == 1


def _refuses_whole(node_id: str, field: str, dtype: type[DType]) -> Problem | None:
    """A source type that says it cannot be handed over whole — a table whose
    columns nobody stated — is refused on the field, fatal, with the type's
    own message. Asked only where a node will *read* the value, a
    ``**inputs`` parameter; an ``Any`` input only passes the value on, and
    nothing is asked.
    """
    refusal = (dtype.element or dtype).refuses_whole()
    if refusal is None:
        return None
    code, message = refusal
    return Problem(
        code=code, message=f"Field '{field}': {message}", fatal=True, node_id=node_id, field=field,
        details={"inner_message": message},
    )


def _originates(inp: Input, ref: Ref, static: Any, scope: Index | None) -> Carried:
    """What a field carries when no edge feeds it: a scalar, or a series on
    an index of the field's own — always for a ``Series[X]`` input, and for
    a scalar input when the author typed many values. Inside an embedded
    flow that runs per row, that index is a child of the entering one
    (``scope``).

    ``static`` is the converted value from ``compiler._typed_statics`` — a
    ``list`` exactly when the author typed many — so a scalar type whose
    own JSON form happens to be a list never makes the node run per row.
    """
    if getattr(inp.dtype, "element", None) is not None:  # only a Series type has an element
        return Carried(inp.dtype, Index(ref, parent=scope))
    if many(static):
        return Carried(Series[inp.dtype], Index(ref, parent=scope))
    return Carried(inp.dtype, None)


def _admit(target: Any, ref: Ref, source_ref: Ref, source: Carried) -> list[Problem]:
    """May what arrives land on ``ref``? ``target.accepts`` decides."""
    if not target.accepts(source.dtype):
        return [problem(
            "type_mismatch", ref.node_id, ref.field,
            source=str(source_ref),
            source_type=description_of(source.dtype),
            target_type=description_of(target),
            source_said=_say(source.dtype),
            target_said=_say(target),
        )]
    return []


def _say(dtype: Any) -> str:
    """A type named by its id, for the English message: ``text``, ``a series of text``."""
    if dtype.element is not None:
        return f"a series of {dtype.element.id}"
    return dtype.id


def _lift_index(demands: list[tuple[Index, Ref]]) -> tuple[Index | None, tuple[Ref, Ref] | None]:
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


def _lineage(index: Index) -> list[Index]:
    """This index and every index it was opened from, nearest first."""
    found: list[Index] = []
    current: Index | None = index
    while current is not None:
        found.append(current)
        current = current.parent
    return found


def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
    return problem("misaligned", node_id, a=str(a), b=str(b))
