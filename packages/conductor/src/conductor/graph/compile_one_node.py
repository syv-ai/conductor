"""One node, compiled from the nodes compiled above it.

Find its definition and version; read what the author typed; ask
``compute_inputs``; check its bindings; wire each input from the nodes above;
ask ``compute_outputs``; build its fields. It returns at the first stage that
can't go on, with what it has, and the node says so by what it lacks
(``interface``, ``wiring``). A node a graph stands for gets its inner nodes
placed under the row it runs on, as if they had been drawn there.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from conductor.dtype import is_untyped
from conductor.errors import CompileProblem, Problem, problem
from conductor.graph.binding import From
from conductor.graph.compiled_node import CompiledField, CompiledNode, Wiring
from conductor.graph.conditions import ALWAYS, Condition, condition_at_node, output_condition
from conductor.graph.embedding import (
    SEPARATOR,
    Crossing,
    EmbeddableGraph,
    as_drawn,
    embedded_in,
    lift_embedded_nodes,
    table_rows_inside,
)
from conductor.graph.model import GraphNode
from conductor.graph.row_origins import origins_taken_in
from conductor.graph.typed_in_values import read_typed_in_values
from conductor.graph.views import field_problems
from conductor.graph.wire_inputs import InputsRead, decide_rows, read_inputs
from conductor.interface import Interface
from conductor.metadata import Input, Output
from conductor.node_decorator import NodeVersion
from conductor.ref import Ref
from conductor.registry import NodeRegistry
from conductor.series import Index, Series
from conductor.table import columns_of, holds_tables

if TYPE_CHECKING:
    from conductor.graph.compile_graph import WalkSoFar
    from conductor.graph.compiled import CompiledGraph


def compile_one_node(node: GraphNode, walk: WalkSoFar, registry: NodeRegistry) -> None:
    """Compile ``node`` and add it to ``walk``, with its inner nodes if a graph stands for it."""
    problems = walk.found_before.pop(node.id, [])
    if node.type not in registry:
        problems.append(problem("unknown_node_type", node.id, node_type=node.type, choices=", ".join(sorted(registry)) or "none"))
        return _add(walk, node, problems)
    definition = registry[node.type]
    version: Any = definition.versions.get(node.version)
    if version is None:
        problems.append(problem("unknown_node_version", node.id, node_type=node.type, version=node.version))
        return _add(walk, node, problems)
    if walk.graph_nodes:
        node = _with_edges_out_of_embedded_graphs_resolved(node, walk)
    is_graph = not isinstance(version, NodeVersion)
    if is_graph:
        walk.graph_nodes.add(node.id)

    def finish(interface: Interface | None, statics: dict[str, Any], wiring: Wiring | None) -> None:
        _add(walk, node, problems, definition=definition, version=version, interface=interface, statics=statics, wiring=wiring)

    declared = version.interface
    typed, listed, invalid = read_typed_in_values(declared.inputs, node)
    inputs = declared.inputs
    if not is_graph and version.compute_inputs is not None:
        defaults = {i.name: i.default for i in inputs if not i.required}
        try:
            added = version.compute_inputs.answer({**defaults, **typed}, {}, typed) or ()
        except CompileProblem as refusal:
            problems.append(refusal.at(node.id))
            return finish(None, {}, None)
        inputs = (*inputs, *added)
        typed_added, listed_added, invalid_added = read_typed_in_values(added, node)
        typed, listed, invalid = {**typed, **typed_added}, listed | listed_added, [*invalid, *invalid_added]
    problems.extend(invalid)

    embeddable = None
    can_be_wired = True
    if is_graph:
        version, can_be_wired = _graph_as_this_node_fills_it(node, version, definition.title, typed, listed, problems)
        embeddable = version._for_embedding
        listed = frozenset()
    elif version.open_input is not None:
        named = {i.name for i in inputs}
        inputs = (*inputs, *(
            version.open_input.model_copy(update={"name": name, "title": name})
            for name, binding in node.bindings.items()
            if name not in named and isinstance(binding, From)
        ))

    can_be_wired = _check_bindings(node, inputs, version, embeddable, walk, problems) and can_be_wired
    if not can_be_wired:
        return finish(replace(version.interface, inputs=inputs), typed, None)

    found = read_inputs(node, inputs, listed, walk, problems)
    if found.broken:
        return finish(_interface_without_wiring(node, version, inputs, found, typed, problems), typed, None)
    typed_inputs = tuple(_with_type_from_its_edge(inp, found.bound) for inp in inputs)
    try:
        outputs = _ask_outputs(version, found.arriving, typed)
    except CompileProblem as refusal:
        problems.append(refusal.at(node.id))
        return finish(replace(version.interface, inputs=typed_inputs, outputs=()), typed, None)
    interface = replace(version.interface, inputs=typed_inputs, outputs=outputs)
    faults = field_problems(node.id, outputs, frozenset(i.name for i in typed_inputs), untyped_ok=False)
    faults += [problem("series_of_tables", node.id, out.name) for out in outputs if out.dtype.element is not None and holds_tables(out.dtype.element)]
    if faults:
        problems.extend(faults)
        return finish(interface, typed, None)
    if not outputs and (is_graph or version.returns is not None):
        problems.append(problem("no_outputs", node.id))
    settled, row = decide_rows(node.id, found, problems, walk.row_origins)
    if not settled:
        return finish(interface, typed, None)
    if is_graph:
        _place_embedded_graph(node, definition, version, embeddable, interface, typed, row, found, walk, problems)
        return
    wiring = _wiring_of_function_node(node, interface, found, listed, row, walk, problems)
    finish(interface, typed, wiring)


def _add(
    walk: WalkSoFar, node: GraphNode, problems: list[Problem], *, definition: Any = None, version: Any = None,
    interface: Interface | None = None, statics: dict[str, Any] | None = None, wiring: Wiring | None = None,
) -> None:
    """One finished record, with the cause of a node that isn't wired."""
    walk.compiled[node.id] = CompiledNode(
        id=node.id, graph_node=node, embedded_in=embedded_in(node.id), problems=tuple(problems),
        cause=None if wiring is not None else _cause_of(node, problems, walk),
        definition=definition, version=version, interface=interface, statics=statics or {}, wiring=wiring,
    )
    walk.problems.extend(problems)


def _cause_of(node: GraphNode, problems: list[Problem], walk: WalkSoFar) -> Problem:
    """Why a node isn't wired: its own first fatal problem, or else the cause of the first source that isn't."""
    own = next((p for p in problems if p.fatal), None)
    if own is not None:
        return own
    for binding in node.bindings.values():
        if not isinstance(binding, From):
            continue
        for ref in binding.refs:
            if ref.node_id in walk.refused:
                return walk.refused[ref.node_id]
            source = walk.compiled.get(ref.node_id) or walk.compiled.get(as_drawn(ref, walk.graph_nodes)[0])
            if source is not None and source.wiring is None:
                return source.cause
    raise AssertionError(f"compile left {node.id!r} unwired with nothing fatal to say why")


def _with_edges_out_of_embedded_graphs_resolved(node: GraphNode, walk: WalkSoFar) -> GraphNode:
    """``node`` with every edge out of an embedded graph pointed at the field
    inside it that the author's address names (``approve.check.amount`` at
    ``approve/check.amount``): the one place compile reads an address as the
    author wrote it. An edge naming an output it doesn't offer stays as written,
    and wiring reports it (``unknown_ref_output``)."""
    def resolved(ref: Ref) -> Ref:
        source = walk.compiled.get(ref.node_id)
        if source is not None and source.embedded_graph is not None and ref.field in source.embedded_graph._for_embedding.outputs:
            return Ref(f"{ref.node_id}{SEPARATOR}{ref.field}")
        return ref

    if not any(isinstance(b, From) and any(r.node_id in walk.graph_nodes for r in b.refs) for b in node.bindings.values()):
        return node
    return node.model_copy(update={"bindings": {
        name: From(*(resolved(ref) for ref in binding.refs)) if isinstance(binding, From) else binding
        for name, binding in node.bindings.items()
    }})


def _graph_as_this_node_fills_it(
    node: GraphNode, stored: CompiledGraph, title: str, typed: dict[str, Any], listed: frozenset[str], problems: list[Problem],
) -> tuple[CompiledGraph, bool]:
    """The compiled graph as this graph node fills it, and whether it can be embedded.

    A value typed on an embedded graph's input belongs to the node inside, and
    an outer edge into an input the graph's own author filled replaces that
    value. Mostly that just puts the value or edge on the inner field. But a
    value can decide something when compile runs: a list makes the node
    holding it run per item, and a hook reads the values typed on its node.
    Only then is the graph compiled again with the graph node's values.
    """
    embeddable = stored._for_embedding
    refill = {name: value for name, value in typed.items() if name in listed or (name in embeddable.inputs and embeddable.inputs[name].read_by_a_hook)}
    cleared = [
        name for name, binding in node.bindings.items()
        if isinstance(binding, From) and name in embeddable.inputs and embeddable.inputs[name].filled_inside
        and (embeddable.inputs[name].list_typed_inside or embeddable.inputs[name].read_by_a_hook)
    ]
    filled = stored._refilled(refill, cleared) if refill or cleared else stored
    faults = filled._for_embedding.faults
    if not faults:
        return filled, True
    # A fault this graph node's own values brought in is its author's to fix,
    # so it is reported where they typed it; every other fault is the graph's.
    had = {(found.type, found.loc) for found in embeddable.faults} if filled is not stored else None
    brought = [found for found in faults if had is not None and (found.type, found.loc) not in had]
    for found in brought:
        where = next((name for name in refill if as_drawn(Ref(name), embeddable.graph_ids)[0] == found.loc[0]), None)
        problems.append(found.model_copy(update={"loc": (node.id,) if where is None else (node.id, where)}))
    if len(brought) < len(faults):
        problems.append(problem("embedded_graph_broken", node.id, graph=title, problems=len(faults) - len(brought)))
    return filled, False


def _check_bindings(
    node: GraphNode, inputs: tuple[Input, ...], version: Any, embeddable: EmbeddableGraph | None, walk: WalkSoFar, problems: list[Problem],
) -> bool:
    """The stored bindings against the inputs the node actually has, and those
    inputs against the field rules. ``False`` when its edges can't be wired.

    A typed required input left empty is fatal but wiring goes on: its type is
    declared, so the node and everything after it are still worked out."""
    declared = {i.name: i for i in inputs}
    problems.extend(problem("unknown_locked_field", node.id, name) for name in node.locked if name not in declared)
    can_wire = True
    faults = field_problems(node.id, inputs, untyped_ok=True)
    faults += [problem("parameter_name_invalid", node.id, i.name) for i in inputs if i.name.startswith("_")]
    if faults:
        can_wire = False
        problems.extend(faults)
    for name, binding in node.bindings.items():
        if name not in declared:
            hook_decides_inputs = embeddable is None and version.compute_inputs is not None
            problems.append(problem("unused_binding" if hook_decides_inputs else "stale_binding", node.id, name, inputs=", ".join(sorted(declared)) or "none"))
            continue
        if not isinstance(binding, From):
            continue
        if not declared[name].connectable:
            can_wire = False
            problems.append(problem("edge_into_unconnectable_input", node.id, name))
        for ref in binding.refs:
            if ref.node_id in walk.refused:
                can_wire = False
            elif ref.node_id not in walk.compiled and as_drawn(ref, walk.graph_nodes)[0] not in walk.compiled:
                can_wire = False
                problems.append(problem("unknown_ref_node", node.id, name, source_node=ref.node_id))
    for inp in inputs:
        if is_untyped(inp.dtype):
            if not isinstance(node.bindings.get(inp.name), From):
                can_wire = False
                problems.append(problem("unbound_required", node.id, inp.name))
        elif inp.required and inp.name not in node.bindings and not (
            embeddable is not None and inp.name in embeddable.inputs and embeddable.inputs[inp.name].filled_inside
        ):
            problems.append(problem("unbound_required", node.id, inp.name))
    return can_wire


def _interface_without_wiring(
    node: GraphNode, version: Any, inputs: tuple[Input, ...], found: InputsRead, typed: dict[str, Any], problems: list[Problem],
) -> Interface:
    """A source couldn't be read: complete the node's fields from what did arrive, so an editor can draw it."""
    try:
        answered = _ask_outputs(version, found.arriving, typed)
    except CompileProblem:
        # The broken source carries the fault; a refusal about its missing
        # arrival would report the same fact twice.
        answered = ()
    typed_inputs = tuple(_with_type_from_its_edge(inp, found.bound) for inp in inputs)
    problems.extend(field_problems(node.id, answered, frozenset(i.name for i in typed_inputs), untyped_ok=True))
    return replace(version.interface, inputs=typed_inputs, outputs=answered)


def _ask_outputs(version: Any, arriving: dict[str, Any], typed: dict[str, Any]) -> tuple[Output, ...]:
    """What ``compute_outputs`` answers now that it can be told what arrives; a compiled graph has no hooks."""
    if not isinstance(version, NodeVersion) or version.compute_outputs is None:
        return version.interface.outputs
    values = {**{i.name: i.default for i in version.interface.inputs if not i.required}, **typed}
    answered = version.compute_outputs.answer(values, arriving, typed)
    return version.interface.outputs if answered is None else answered


def _with_type_from_its_edge(field: Input, bound: dict[str, Any]) -> Input:
    """``field`` with the type its edge gave it, if an edge gave one."""
    return field.model_copy(update={"dtype": bound[field.name]}) if field.name in bound else field


def _condition_of_source_above(walk: WalkSoFar) -> Any:
    def condition_of(ref: Ref) -> Condition:
        node = walk.compiled.get(ref.node_id)
        field = None if node is None or node.wiring is None else node.wiring.fields.get(ref.field)
        return field.condition if field is not None and field.condition is not None else ALWAYS

    return condition_of


def _wiring_of_function_node(
    node: GraphNode, interface: Interface, found: InputsRead, listed: frozenset[str], row: Index | None, walk: WalkSoFar, problems: list[Problem],
) -> Wiring:
    """Every field of a node that is a function, built once its row is known."""
    outputs = interface.outputs
    names = {out.name for out in outputs}
    taken_in = origins_taken_in(found.tables_taken_in)
    for out in outputs:
        if columns_of(out.dtype) or holds_tables(out.dtype):
            for column, _ in columns_of(out.dtype):
                name = f"{out.name}.{column}"
                if name in names:
                    problems.append(problem("view_name_taken", node.id, name, table=out.name, column=column))
    on_field: dict[str, list[Problem]] = {}
    for fault in problems:
        if len(fault.loc) > 1:
            on_field.setdefault(str(fault.loc[1]), []).append(fault)
    fields: dict[str, CompiledField] = {}
    node_id = node.id
    for inp in interface.inputs:
        arrived = found.arrivals[inp.name]
        fields[inp.name] = CompiledField(
            ref=Ref(node_id, inp.name), type=arrived.dtype, binding=node.bindings.get(inp.name), condition=None,
            problems=tuple(on_field.get(inp.name, ())), _index=arrived.rows, _receives=found.receives[inp.name],
            _listed=inp.name in listed, _makes_rows=None, _view_of=None, _takes_blank=False, _row_origins=None,
        )
    at_node = condition_at_node(node.bindings, _condition_of_source_above(walk))
    runs_once = row is None
    for out in outputs:
        ref = Ref(node_id, out.name)
        condition = output_condition(node_id, out, at_node, runs_once)
        takes_blank, field_problems_here = out.takes_blank, tuple(on_field.get(out.name, ()))
        dtype = out.dtype
        if columns_of(dtype) or holds_tables(dtype):
            rows = Index(str(ref), parent=row)
            known = walk.row_origins.note_table(rows, taken_in)
            fields[out.name] = CompiledField(
                ref=ref, type=Series[dtype] if row is not None else dtype, binding=None, condition=condition, problems=field_problems_here,
                _index=row, _receives=None, _listed=False, _makes_rows=rows, _view_of=None, _takes_blank=takes_blank, _row_origins=known,
            )
            for column, holds in columns_of(dtype):
                name = f"{out.name}.{column}"
                if name in names:
                    continue
                fields[name] = CompiledField(
                    ref=Ref(node_id, name), type=Series[holds], binding=None, condition=condition, problems=(),
                    _index=rows, _receives=None, _listed=False, _makes_rows=None, _view_of=(ref, column), _takes_blank=True, _row_origins=known,
                )
            continue
        if dtype.element is not None:
            carried, index, makes = dtype, Index(str(ref), parent=row), True
        elif row is not None:
            carried, index, makes = Series[dtype], row, False
        else:
            carried, index, makes = dtype, None, False
        fields[out.name] = CompiledField(
            ref=ref, type=carried, binding=None, condition=condition, problems=field_problems_here,
            _index=index, _receives=None, _listed=False, _makes_rows=index if makes else None, _view_of=None, _takes_blank=takes_blank,
            _row_origins=walk.row_origins.of_rows_on(index),
        )
    return Wiring(fields, row)


def _place_embedded_graph(
    node: GraphNode, definition: Any, graph: CompiledGraph, embeddable: EmbeddableGraph, interface: Interface, typed: dict[str, Any], row: Index | None,
    found: InputsRead, walk: WalkSoFar, problems: list[Problem],
) -> None:
    """The graph node's record, and every node inside it under ``row``."""
    offered = {str(inp.name) for inp in graph.interface.inputs}
    edges = {name: binding for name, binding in node.bindings.items() if isinstance(binding, From) and name in offered}
    crossings = {name: Crossing(type=found.arrivals[name].dtype, index=found.arrivals[name].rows, receives=found.receives[name]) for name in edges}
    lifted = lift_embedded_nodes(embeddable, node.id, edges, typed, row, crossings, _condition_of_source_above(walk))
    _add(walk, node, problems, definition=definition, version=graph, interface=interface, statics=typed, wiring=Wiring({}, row))
    walk.compiled.update(lifted)
    walk.graph_nodes.update(f"{node.id}{SEPARATOR}{inner}" for inner in embeddable.graph_ids)
    walk.row_origins.of_table_rows.update(dict.fromkeys(table_rows_inside(lifted)))
    walk.row_origins.must_survive.update(Ref(f"{node.id}{SEPARATOR}{ref.node_id}", ref.field) for ref in embeddable.row_ids_must_survive)
