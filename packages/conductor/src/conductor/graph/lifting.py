"""The edges pass: what arrives on every field, and whether it may.

Walks the expanded graph in execution order and answers, per edge, three
questions that are really one:

* **Admission** — may this edge land? ``target.accepts(source)``, a series
  judged by its element. A refusal is the fatal ``type_mismatch``.
* **Lifting** — a series into a scalar input means the node runs once per
  row: its scalar inputs broadcast, every output becomes a series on the
  same index, and nodes downstream receive a series and lift in turn.
  Nothing is stored or marked; "receives a series" is the whole rule.
* **Alignment** — a node lifted by several inputs needs their indexes on
  one lineage: the deepest is the lift index and the others broadcast
  down to it. Two unrelated indexes are the fatal ``misaligned``.

Before any of them, unconstrained inputs are typed from their edges: a
parameter declared ``Any`` takes the element of what arrives, and lifts
if a series arrives; an open roster's parameters take what arrives per
edge — ``**inputs: Single`` each value whole, series and all, never
lifting; ``**inputs: Series`` each as a reduction. Once every input is
typed, ``compute_outputs`` is asked with ``arriving`` and the outputs
roster is completed.

A ``Series[X]`` input is read by the index that arrives. One series on a
child index is a reduction: it lifts the node on the parent, and per
parent row the input receives the child rows under it. One series on a
root index is received whole, once. Anything else feeding the input —
several scalars, several series on different indexes, a typed list, a
default — gathers onto a fresh index born at the input, and lifts
nothing.

An embedded flow's boundary is an index scope. Where a series enters a
placement through a scalar field, the placement is lifted on that index;
every node inside lifts on at least that index; every index born inside
is a child of it; and an inner reduction over the entering index is a
"fold of one" — per outer row it receives the one row under it — so the
inner flow behaves exactly as it would standalone, once per outer row.
The scope is read when the block starts, from what crossed into it.

A scalar input holding a static sequence (a multi-file upload, a list
typed by hand) is a series on the input's own index and lifts the node
as a connected series would.

Indexes are named here and rows are added to them by the engine: a node's
series output sits on ``Index(node_id)`` — a root when the node is not
lifted, a child of the lift index when it is; a gathering ``Series[X]``
input sits on ``Index(ref)``. Compile reasons about *which* index, never
about how many rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from conductor.dtype import DType
from conductor.dtype_ref import description_of
from conductor.graph.binding import Edges, many
from conductor.graph.compiled import Carried
from conductor.graph.model import GraphNode
from conductor.graph.problem import Problem
from conductor.metadata import Input, Output, Roster
from conductor.node import NodeVersion, Refuses
from conductor.ref import Ref
from conductor.registry import NodeRegistry
from conductor.series import Index, Series


@dataclass(frozen=True)
class Lifting:
    """What ``derive`` returns.

    ``lifted``: each node's lift index (``None`` when it runs once), and
    each placement's scope. ``carried``: what travels on every field.
    ``rosters``: each node's roster completed with what arrives — what
    ``CompiledGraph.roster`` serves from then on. ``problems``: what went
    wrong. Its sibling is ``Expansion``, the other pass result.
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
    """Walk ``nodes`` (expanded order, sound edges only) and derive lifting,
    carried types and completed rosters.

    A node one of whose sources has no shape — the source was broken and
    carries the problem — gets no shape either; nothing downstream of a
    fault is guessed at. Its roster is still completed from what did
    arrive, so an editor can draw its handles.

    ``compute_outputs`` is asked once per node, here, with ``arriving``
    (the type each connected input receives per unit) and the typed statics
    over the declaration's defaults. A hook that raises ``Refuses`` makes
    its code and message the node's one fatal problem.

    ``placement_of`` and ``members`` come from the expansion: which block
    each node is in, and what each block holds. A block's scope is read
    when its first node is reached.
    """
    lifted: dict[str, Index | None] = {}
    carried: dict[Ref, Carried] = {}
    completed: dict[str, Roster] = {}
    problems: list[Problem] = []
    scopes: dict[str, Index | None] = {}

    for node in nodes:
        placement = placement_of.get(node.id)
        if placement is not None and placement not in scopes:
            scopes[placement] = _scope(placement, members[placement], rosters, nodes, carried, problems)
            lifted[placement] = scopes[placement]
        scope = scopes.get(placement) if placement is not None else None

        roster = rosters[node.id]
        declared_names = {i.name for i in versions[node.id].interface.inputs}
        arrivals: dict[str, Carried] = {}
        receives: dict[str, Any] = {}
        demands: list[tuple[Index, Ref]] = []
        bound: dict[str, Any] = {}

        for inp in roster.inputs:
            ref = Ref(node.id, inp.name)
            binding = node.bindings.get(inp.name)
            whole = versions[node.id].interface.open == "single" and inp.name not in declared_names
            if not isinstance(binding, Edges):
                arrivals[inp.name] = _originates(inp, ref, statics[node.id].get(inp.name), scope)
                if getattr(inp.dtype, "element", None) is None and arrivals[inp.name].index is not None:
                    demands.append((arrivals[inp.name].index, ref))
                continue
            missing = [source for source in binding.refs if source not in carried]
            if missing:
                # A source that has a shape but not this field: the ref names an
                # output it does not have. A source that never got a shape
                # carries its own problem and is not reported again here.
                problems.extend(_unknown_output(node.id, inp.name, source) for source in missing if source.node_id in lifted)
                break
            sources = [carried[source] for source in binding.refs]
            if whole:
                # An open roster's parameter takes what arrives, whole. It is
                # passed as a keyword argument, so its name must be an identifier.
                if not inp.name.isidentifier():
                    problems.append(_not_a_parameter_name(node.id, inp.name))
                    break
                if len(sources) > 1:
                    problems.append(_one_edge(node.id, inp.name))
                    break
                refused = _refuses_whole(node.id, inp.name, sources[0].dtype)
                if refused is not None:
                    problems.append(refused)
                    break
                bound[inp.name] = sources[0].dtype
                receives[inp.name] = sources[0].dtype
                arrivals[inp.name] = sources[0]
                continue
            target = inp.dtype
            if target is Any or getattr(target, "element", None) is Any:
                # An unconstrained input types from its first edge; the
                # rest are admitted against it.
                element = sources[0].dtype.element or sources[0].dtype
                target = Series[element] if getattr(inp.dtype, "element", None) is not None else element
                bound[inp.name] = target
            for source_ref, source in zip(binding.refs, sources, strict=True):
                problems.extend(_admit(target, ref, source_ref, source))
            if target.element is None:
                if len(sources) > 1 and not _one_index(sources):
                    problems.append(_union_needs_one_index(node.id, inp.name))
                    break
                arriving = sources[0]  # one ref, or a union of several on one index
                receives[inp.name] = arriving.dtype.element or arriving.dtype
                if arriving.index is not None:
                    demands.append((arriving.index, ref))
            elif _one_index(sources):
                arriving = sources[0]  # one series, or a union of several on one index
                receives[inp.name] = arriving.dtype
                if scope is not None and arriving.index == scope:
                    # A fold of one: the series entered the block, so the
                    # reduction runs per outer row and receives the row under it.
                    demands.append((arriving.index, ref))
                elif arriving.index.parent is not None:
                    demands.append((arriving.index.parent, ref))
            else:
                arriving = Carried(target, Index(ref, parent=scope))
                receives[inp.name] = target
            arrivals[inp.name] = arriving
        else:
            inputs = tuple(_typed(inp, bound) for inp in roster.inputs)
            answered = _outputs(node, versions[node.id].interface.outputs, registry, receives, _hook_values((*versions[node.id].interface.inputs, *roster.inputs), statics[node.id]))
            if isinstance(answered, Problem):
                # The refusal is the node's problem: fatal, no `no_outputs`
                # beside it, and nothing derived past it.
                problems.append(answered)
                completed[node.id] = Roster(inputs=inputs, outputs=())
                continue
            outputs = answered
            completed[node.id] = Roster(inputs=inputs, outputs=outputs)
            if not outputs:
                problems.append(_no_outputs(node.id))
            lift_index, disagreeing = _lift_index(demands)
            if disagreeing is not None:
                problems.append(_misaligned(node.id, *disagreeing))
                continue
            # Inside a lifted block every node runs at least once per outer row:
            # the block's scope is one more demand on the same lineage. A node
            # nothing from outside reaches lifts on it; a node fed from above
            # it broadcasts down.
            if scope is not None and (lift_index is None or scope not in _lineage(lift_index)):
                lift_index = scope
            lifted[node.id] = lift_index
            carried.update({Ref(node.id, name): c for name, c in arrivals.items()})
            node_index: Index | None = None
            for out in outputs:
                ref = Ref(node.id, out.name)
                if getattr(out.dtype, "element", None) is not None:
                    node_index = node_index or Index(node.id, parent=lift_index or scope)
                    carried[ref] = Carried(out.dtype, node_index)
                elif lift_index is not None:
                    carried[ref] = Carried(Series[out.dtype], lift_index)
                else:
                    carried[ref] = Carried(out.dtype, None)
            continue
        # A source had no shape: complete the roster from what arrived, derive nothing.
        answered = _outputs(
            node, versions[node.id].interface.outputs,
            registry, receives, _hook_values((*versions[node.id].interface.inputs, *roster.inputs), statics[node.id]),
        )
        if isinstance(answered, Problem):
            # The broken source carries the fault; a refusal about its missing
            # arrival would report the same fact twice.
            answered = ()
        completed[node.id] = Roster(
            inputs=tuple(_typed(inp, bound) for inp in roster.inputs),
            outputs=answered,
        )

    return Lifting(lifted=lifted, carried=carried, rosters=completed, problems=tuple(problems))


def _typed(field: Any, bound: Mapping[str, Any]) -> Any:
    """``field`` with the type its edge gave it, if an edge gave one."""
    if field.name in bound:
        return replace(field, dtype=bound[field.name])
    return field


def _scope(
    placement: str,
    block: Sequence[str],
    rosters: Mapping[str, Roster],
    nodes: Sequence[GraphNode],
    carried: Mapping[Ref, Carried],
    problems: list[Problem],
) -> Index | None:
    """The index a placement's block is lifted on: the deepest index among
    the series that cross into it through a scalar field, or ``None`` when
    none does (the block then expands flat). Two unrelated crossings are
    the placement's ``misaligned``, as they would be a node's.
    """
    inside = set(block)
    by_id = {node.id: node for node in nodes}
    demands: list[tuple[Index, Ref]] = []
    for node_id in block:
        if node_id not in by_id:
            continue  # broken edges; the node carries its own problem and has no shape
        node = by_id[node_id]
        for inp in rosters[node_id].inputs:
            binding = node.bindings.get(inp.name)
            if not isinstance(binding, Edges) or getattr(inp.dtype, "element", None) is not None:
                continue
            for source in binding.refs:
                if source.node_id not in inside and source in carried and carried[source].index is not None:
                    demands.append((carried[source].index, Ref(node_id, inp.name)))
    index, disagreeing = _lift_index(demands)
    if disagreeing is not None:
        problems.append(_misaligned(placement, *disagreeing))
        return None
    return index


def _hook_values(inputs: Sequence[Input], statics: Mapping[str, Any]) -> dict[str, Any]:
    """What ``compute_outputs`` reads: the typed statics over the declaration's
    defaults. An input nothing binds falls back to what the declaration
    states, so a hook indexes ``values[...]`` without a guard.
    """
    return {**{i.name: i.default for i in inputs if i.optional}, **statics}


def _outputs(
    node: GraphNode, declared: tuple[Output, ...], registry: NodeRegistry, receives: Mapping[str, Any], values: Mapping[str, Any]
) -> tuple[Output, ...] | Problem:
    """Ask the node's ``compute_outputs`` now that it can be told what arrives.

    A hook that cannot answer for these values and arrivals raises
    ``Refuses(code, message)``, and the refusal comes back as the node's
    one fatal ``Problem`` — the hook chose the code and the message,
    compile only anchors it.
    """
    try:
        return registry.get(node.type)().compute_outputs(declared, values, receives)
    except Refuses as refusal:
        return Problem(code=refusal.code, message=refusal.message, fatal=True, node_id=node.id)


def _one_index(sources: Sequence[Carried]) -> bool:
    """Every source carries a series, all on one index — a union of rows."""
    return all(s.index is not None for s in sources) and len({s.index for s in sources}) == 1


def _refuses_whole(node_id: str, field: str, dtype: type[DType]) -> Problem | None:
    """A source type that says it cannot be received whole — a table whose
    columns nobody stated — is refused on the field, fatal, with the type's
    own message. Asked only where a node will *read* the value: the open
    roster. On an ``Any`` input the value is only routed, and nothing is asked.
    """
    refusal = (dtype.element or dtype).refuses_whole()
    if refusal is None:
        return None
    code, message = refusal
    return Problem(
        code=code, message=f"Field '{field}': {message}", fatal=True, node_id=node_id, field=field,
        details={"inner_message": message},
    )


def _union_needs_one_index(node_id: str, field: str) -> Problem:
    return Problem(
        code="union_needs_one_index",
        message=f"Field '{field}' has several edges; that only works when they are all rows of one table.",
        fatal=True,
        node_id=node_id,
        field=field,
    )


def _not_a_parameter_name(node_id: str, field: str) -> Problem:
    return Problem(
        code="parameter_name_invalid",
        message=f"'{field}' cannot be a parameter name; use letters, digits and underscores, and start with a letter.",
        fatal=True,
        node_id=node_id,
        field=field,
    )


def _one_edge(node_id: str, field: str) -> Problem:
    return Problem(
        code="one_edge_per_parameter",
        message=f"Parameter '{field}' takes one edge; an extra edge is an extra parameter.",
        fatal=True,
        node_id=node_id,
        field=field,
    )


def _unknown_output(node_id: str, field: str, source: Ref) -> Problem:
    return Problem(
        code="unknown_ref_output",
        message=f"Field '{field}' is connected to '{source}', which is not an output of that node.",
        fatal=True,
        node_id=node_id,
        field=field,
        details={"source": str(source)},
    )


def _no_outputs(node_id: str) -> Problem:
    """The ordinary mid-edit state of a node whose outputs depend on what the
    author has not yet given — not fatal: the node runs, produces nothing,
    and nothing can hang off it yet.
    """
    return Problem(
        code="no_outputs",
        message="The node has no fields to pass on, so nothing can be connected from it.",
        fatal=False,
        node_id=node_id,
    )


def _originates(inp: Input, ref: Ref, static: Any, scope: Index | None) -> Carried:
    """What a field holds when no edge feeds it: a scalar, or a series on an
    index born at this very field — a ``Series[X]`` input's own, or a
    scalar input's when the author typed many. Born inside a lifted block,
    the index is a child of the entering one.

    ``static`` is the *typed* value from the statics pass — a ``list``
    exactly when it carries many values — so a scalar type whose own JSON
    form is list-shaped never lifts the node holding it.
    """
    if getattr(inp.dtype, "element", None) is not None:  # a closed static type has no element
        return Carried(inp.dtype, Index(ref, parent=scope))
    if many(static):
        return Carried(Series[inp.dtype], Index(ref, parent=scope))
    return Carried(inp.dtype, None)


def _admit(target: Any, ref: Ref, source_ref: Ref, source: Carried) -> list[Problem]:
    """May what arrives land on ``ref``? ``target.accepts`` decides."""
    if not target.accepts(source.dtype):
        return [Problem(
            code="type_mismatch",
            message=f"'{source_ref}' is {_say(source.dtype)}, but '{ref}' takes {_say(target)}.",
            fatal=True,
            node_id=ref.node_id,
            field=ref.field,
            details={
                "source": str(source_ref),
                "source_type": description_of(source.dtype),
                "target_type": description_of(target),
            },
        )]
    return []


def _say(dtype: Any) -> str:
    """A type named by its id, for the English message: ``text``, ``a series of text``."""
    if dtype.element is not None:
        return f"a series of {dtype.element.id}"
    return dtype.id


def _lift_index(demands: list[tuple[Index, Ref]]) -> tuple[Index | None, tuple[Ref, Ref] | None]:
    """The index a node runs once per row of, or the two fields whose
    indexes disagree.

    Every demanded index must lie on one lineage. The deepest is the lift
    index; a shallower one is an ancestor whose rows broadcast down.
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
    """This index and every ancestor, nearest first."""
    found: list[Index] = []
    current: Index | None = index
    while current is not None:
        found.append(current)
        current = current.parent
    return found


def _misaligned(node_id: str, a: Ref, b: Ref) -> Problem:
    return Problem(
        code="misaligned",
        message=f"'{a}' and '{b}' get their rows from different sources, so the node cannot run per row.",
        fatal=True,
        node_id=node_id,
        details={"a": str(a), "b": str(b)},
    )
