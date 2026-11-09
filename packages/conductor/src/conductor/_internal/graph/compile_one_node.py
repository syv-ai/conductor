"""One node, compiled from the nodes compiled above it.

Find its definition and version; read what the author typed; ask
``compute_inputs``; check its bindings; wire each input from the nodes above;
ask ``compute_outputs``; build its fields. It returns at the first stage that
can't go on, with what it has, and the node says so by what it lacks
(``interface``, ``wiring``). A node a graph stands for gets its inner nodes
placed under the row it runs on, as if they had been drawn there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from conductor._internal.dtype import is_untyped
from conductor._internal.errors import CompileProblem, Problem, make_problem
from conductor._internal.field_records import Input, Output
from conductor._internal.graph.binding import From
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode, Wiring
from conductor._internal.graph.conditions import (
    ALWAYS,
    Condition,
    condition_at_node,
    output_condition,
)
from conductor._internal.graph.embedding import (
    SEPARATOR,
    Crossing,
    EmbeddableGraph,
    embedded_in,
    lift_embedded_nodes,
)
from conductor._internal.graph.graph_interface import field_problems
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.graph.row_origins import origins_taken_in
from conductor._internal.graph.typed_in_values import read_static_values
from conductor._internal.graph.wire_inputs import InputsRead, decide_rows, read_inputs
from conductor._internal.interface import Interface
from conductor._internal.node_decorator import NodeVersion
from conductor._internal.ref import Ref
from conductor._internal.registry import NodeRegistry
from conductor._internal.series import Index, Series
from conductor._internal.table import columns_of, holds_tables

if TYPE_CHECKING:
    from conductor._internal.graph.compile_graph import WalkSoFar
    from conductor._internal.graph.compiled import CompiledGraph


def _replaced(interface: Interface, inputs: tuple[Input, ...], outputs: tuple[Output, ...]) -> Interface:
    return Interface(inputs=inputs, outputs=outputs, needs=interface.needs)


def compile_one_node(node: GraphNode, walk: WalkSoFar, registry: NodeRegistry) -> None:
    """Compile ``node`` and add it to ``walk``, with its inner nodes if a graph stands for it."""
    problems = walk.found_before.pop(node.id, [])
    if node.type not in registry:
        problems.append(make_problem("unknown_node_type", node.id, node_type=node.type, choices=", ".join(sorted(registry)) or "none"))
        return _add(walk, node, problems)
    definition = registry[node.type]
    version: Any = definition.versions.get(node.version)
    if version is None:
        problems.append(make_problem("unknown_node_version", node.id, node_type=node.type, version=node.version))
        return _add(walk, node, problems)
    drawn = node
    if walk.graph_nodes:
        node = _with_edges_out_of_embedded_graphs_resolved(node, walk)
    is_graph = not isinstance(version, NodeVersion)
    if is_graph:
        walk.graph_nodes.add(node.id)

    declared = _interface_it_declares(version)
    statics, inputs_given_a_list, unreadable_static_problems = read_static_values(declared.inputs, node)
    inputs = declared.inputs
    if not is_graph and version.compute_inputs is not None:
        defaults = {i.name: i.default for i in inputs if not i.required}
        try:
            added = version.compute_inputs.answer({**defaults, **statics}, {}, statics) or ()
        except CompileProblem as hook_error:
            problems.append(hook_error.at(node.id))
            return _add(walk, node, problems, definition, version, None, {})
        inputs = (*inputs, *added)
        statics_added, lists_added, unreadable_added = read_static_values(added, node)
        statics, inputs_given_a_list, unreadable_static_problems = {**statics, **statics_added}, inputs_given_a_list | lists_added, [*unreadable_static_problems, *unreadable_added]
    problems.extend(unreadable_static_problems)

    embeddable = None
    can_be_wired = True
    if is_graph:
        version, can_be_wired = _graph_as_this_node_fills_it(node, version, definition.title, statics, inputs_given_a_list, problems)
        embeddable = version._for_embedding
        inputs_given_a_list = frozenset()
    elif version.open_input is not None:
        named = {i.name for i in inputs}
        inputs = (*inputs, *(
            version.open_input.model_copy(update={"name": name, "title": name})
            for name, binding in node.bindings.items()
            if name not in named and isinstance(binding, From)
        ))

    can_be_wired = _check_bindings(drawn, inputs, version, embeddable, walk, problems) and can_be_wired
    if not can_be_wired:
        return _add(walk, node, problems, definition, version, _replaced(declared, inputs, declared.outputs), statics)

    found = read_inputs(node, inputs, inputs_given_a_list, walk, problems)
    if found.broken:
        return _add(walk, node, problems, definition, version, _interface_without_wiring(node, version, inputs, found, statics, problems), statics)
    inputs_typed_by_their_edges = tuple(_with_type_from_its_edge(inp, found.bound) for inp in inputs)
    try:
        outputs = _ask_outputs(version, found.arriving_types, statics)
    except CompileProblem as hook_error:
        problems.append(hook_error.at(node.id))
        return _add(walk, node, problems, definition, version, _replaced(version.interface, inputs_typed_by_their_edges, ()), statics)
    interface = _replaced(version.interface, inputs_typed_by_their_edges, outputs)
    problems_found = field_problems(node.id, outputs, frozenset(i.name for i in inputs_typed_by_their_edges), untyped_ok=False)
    problems_found += [make_problem("series_of_tables", node.id, out.name) for out in outputs if out.dtype.element is not None and holds_tables(out.dtype.element)]
    if problems_found:
        problems.extend(problems_found)
        return _add(walk, node, problems, definition, version, interface, statics)
    if not outputs and (is_graph or version.returns is not None):
        problems.append(make_problem("no_outputs", node.id))
    rows_agree, runs_per_row_of = decide_rows(node, found, problems, walk.row_origins)
    if not rows_agree:
        return _add(walk, node, problems, definition, version, interface, statics)
    if is_graph:
        _place_embedded_graph(node, definition, version, embeddable, interface, statics, runs_per_row_of, found, walk, problems)
        return
    wiring = _wiring_of_function_node(node, interface, found, inputs_given_a_list, runs_per_row_of, walk, problems)
    _add(walk, node, problems, definition, version, interface, statics, wiring)


def _add(
    walk: WalkSoFar, node: GraphNode, problems: list[Problem], definition: Any = None, version: Any = None,
    interface: Interface | None = None, statics: dict[str, Any] | None = None, wiring: Wiring | None = None,
) -> None:
    """One finished record, with the cause of a node that isn't wired."""
    walk.compiled[node.id] = CompiledNode(
        id=node.id, graph_node=node, embedded_in=embedded_in(node.id), problems=tuple(problems),
        cause=None if wiring is not None else _cause_of(node, problems, walk),
        definition=definition, version=version, interface=interface, statics=statics or {}, wiring=wiring,
    )
    walk.problems.extend(problems)


def _interface_it_declares(version: Any) -> Interface:
    """The fields a version offers the graph it is placed in: a compiled graph offers its own, under its offered names."""
    return version.interface if isinstance(version, NodeVersion) else version._for_embedding.interface


def _cause_of(node: GraphNode, problems: list[Problem], walk: WalkSoFar) -> Problem:
    """Why a node isn't wired: its own first fatal problem, or else the cause of the first source that isn't."""
    own = next((p for p in problems if p.fatal), None)
    if own is not None:
        return own
    for binding in node.bindings.values():
        if not isinstance(binding, From):
            continue
        for ref in binding.refs:
            if ref.node_id in walk.invalid_node_ids:
                return walk.invalid_node_ids[ref.node_id]
            source = walk.compiled.get(ref.node_id)
            if source is not None and source.wiring is None:
                return source.cause
    raise AssertionError(f"compile left {node.id!r} unwired with nothing fatal to say why")


def _with_edges_out_of_embedded_graphs_resolved(node: GraphNode, walk: WalkSoFar) -> GraphNode:
    """``node`` with every edge out of an embedded graph pointed at the field
    inside it that the name offers (``approve.total`` at ``approve/check.total``):
    the one place compile reads an offered name. An edge naming an output it
    doesn't offer stays as written, and wiring reports it (``unknown_ref_output``)."""
    def resolved(ref: Ref) -> Ref:
        source = walk.compiled.get(ref.node_id)
        inside = None if source is None or source.wiring is None or source.embedded_graph is None else source.embedded_graph._for_embedding.outputs.get(ref.field)
        return ref if inside is None else Ref(f"{ref.node_id}{SEPARATOR}{inside.node_id}", inside.field)

    if not any(isinstance(b, From) and any(r.node_id in walk.graph_nodes for r in b.refs) for b in node.bindings.values()):
        return node
    return node.model_copy(update={"bindings": {
        name: From(*(resolved(ref) for ref in binding.refs)) if isinstance(binding, From) else binding
        for name, binding in node.bindings.items()
    }})


def _graph_as_this_node_fills_it(
    node: GraphNode, stored: CompiledGraph, title: str, statics: dict[str, Any], inputs_given_a_list: frozenset[str], problems: list[Problem],
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
    refill = {name: value for name, value in statics.items() if name in inputs_given_a_list or (name in embeddable.inputs and embeddable.inputs[name].read_by_a_hook)}
    cleared = [
        name for name, binding in node.bindings.items()
        if isinstance(binding, From) and name in embeddable.inputs and embeddable.inputs[name].has_static_inside
        and (embeddable.inputs[name].list_on_node_inside or embeddable.inputs[name].read_by_a_hook)
    ]
    filled = stored._refilled(refill, cleared) if refill or cleared else stored
    problems_found = filled._for_embedding.problems
    if not problems_found:
        return filled, True
    # A fault this graph node's own values brought in is its author's to fix,
    # so it is reported where they typed it; every other fault is the graph's.
    existing_problems = {(problem.type, problem.loc) for problem in embeddable.problems} if filled is not stored else None
    problems_from_this_node = [problem for problem in problems_found if existing_problems is not None and (problem.type, problem.loc) not in existing_problems]
    for problem in problems_from_this_node:
        where = next((name for name in refill if embeddable.inputs[name].drawn.node_id == problem.loc[0]), None)
        problems.append(problem.model_copy(update={"loc": (node.id,) if where is None else (node.id, where)}))
    if len(problems_from_this_node) < len(problems_found):
        problems.append(make_problem("embedded_graph_broken", node.id, graph=title, problems=len(problems_found) - len(problems_from_this_node)))
    return filled, False


def _check_bindings(
    node: GraphNode, inputs: tuple[Input, ...], version: Any, embeddable: EmbeddableGraph | None, walk: WalkSoFar, problems: list[Problem],
) -> bool:
    """The stored bindings against the inputs the node actually has, and those
    inputs against the field rules. ``False`` when its edges can't be wired.

    A typed required input left empty is fatal but wiring goes on: its type is
    declared, so the node and everything after it are still worked out."""
    declared = {i.name: i for i in inputs}
    if node.locked:
        problems.extend(make_problem("unknown_locked_field", node.id, name) for name in node.locked if name not in declared)
    can_wire = True
    #: A misspelled binding also leaves its input unbound: one problem, the typo, for each.
    unbound_explained_by_typos = 0
    problems_found = field_problems(node.id, inputs, untyped_ok=True)
    problems_found += [make_problem("parameter_name_invalid", node.id, i.name) for i in inputs if i.name.startswith("_")]
    if problems_found:
        can_wire = False
        problems.extend(problems_found)
    for name, binding in node.bindings.items():
        if name not in declared:
            hook_decides_inputs = embeddable is None and version.compute_inputs is not None
            problems.append(make_problem("unused_binding" if hook_decides_inputs else "stale_binding", node.id, name, inputs=", ".join(sorted(declared)) or "none"))
            unbound_explained_by_typos += not hook_decides_inputs
            continue
        if not isinstance(binding, From):
            continue
        if not declared[name].connectable:
            can_wire = False
            problems.append(make_problem("edge_into_unconnectable_input", node.id, name))
        for ref in binding.refs:
            source_id = ref.node_id
            if source_id in walk.invalid_node_ids:
                can_wire = False
            elif SEPARATOR in source_id or source_id not in walk.compiled:
                can_wire = False
                problems.append(make_problem("unknown_ref_node", node.id, name, source_node=source_id))
    for inp in inputs:
        if is_untyped(inp.dtype):
            if isinstance(node.bindings.get(inp.name), From):
                continue
            can_wire = False
        elif not inp.required or inp.name in node.bindings or (
            embeddable is not None and inp.name in embeddable.inputs and embeddable.inputs[inp.name].has_static_inside
        ):
            continue
        if unbound_explained_by_typos:
            unbound_explained_by_typos -= 1
            continue
        problems.append(make_problem("unbound_required", node.id, inp.name))
    return can_wire


def _interface_without_wiring(
    node: GraphNode, version: Any, inputs: tuple[Input, ...], found: InputsRead, statics: dict[str, Any], problems: list[Problem],
) -> Interface:
    """A source couldn't be read: complete the node's fields from what did arrive, so an editor can draw it."""
    try:
        answered = _ask_outputs(version, found.arriving_types, statics)
    except CompileProblem:
        # The broken source carries the fault; a refusal about its missing
        # arrival would report the same fact twice.
        answered = ()
    inputs_typed_by_their_edges = tuple(_with_type_from_its_edge(inp, found.bound) for inp in inputs)
    problems.extend(field_problems(node.id, answered, frozenset(i.name for i in inputs_typed_by_their_edges), untyped_ok=True))
    return _replaced(version.interface, inputs_typed_by_their_edges, answered)


def _ask_outputs(version: Any, arriving_types: dict[str, Any], statics: dict[str, Any]) -> tuple[Output, ...]:
    """What ``compute_outputs`` answers now that it can be told what arrives; a compiled graph has no hooks."""
    if not isinstance(version, NodeVersion) or version.compute_outputs is None:
        return _interface_it_declares(version).outputs
    values = {**{i.name: i.default for i in version.interface.inputs if not i.required}, **statics}
    answered = version.compute_outputs.answer(values, arriving_types, statics)
    return version.interface.outputs if answered is None else answered


def _with_type_from_its_edge(field: Input, bound: dict[str, Any]) -> Input:
    """``field`` with the type its edge gave it, if an edge gave one."""
    return field.model_copy(update={"dtype": bound[field.name]}) if field.name in bound else field


def _field_above(walk: WalkSoFar, ref: Ref) -> CompiledField | None:
    """The field ``ref`` names on a node compiled so far, if it was worked out."""
    node = walk.compiled.get(ref.node_id)
    return None if node is None or node.wiring is None else node.wiring.fields.get(ref.field)


def _condition_of_source_above(walk: WalkSoFar) -> Any:
    def condition_of(ref: Ref) -> Condition:
        field = _field_above(walk, ref)
        return field.condition if field is not None and field.condition is not None else ALWAYS

    return condition_of


def _wiring_of_function_node(
    node: GraphNode, interface: Interface, found: InputsRead, inputs_given_a_list: frozenset[str], runs_per_row_of: Index | None, walk: WalkSoFar, problems: list[Problem],
) -> Wiring:
    """Every field of a node that is a function, built once its row is known."""
    outputs = interface.outputs
    names = {out.name for out in outputs}
    taken_in = origins_taken_in(found.tables_taken_in)
    for out in outputs:
        if holds_tables(out.dtype):
            for column, _ in columns_of(out.dtype):
                name = f"{out.name}.{column}"
                if name in names:
                    problems.append(make_problem("view_name_taken", node.id, name, table=out.name, column=column))
    on_field: dict[str, list[Problem]] = {}
    for problem in problems:
        if len(problem.loc) > 1:
            on_field.setdefault(str(problem.loc[1]), []).append(problem)
    fields: dict[str, CompiledField] = {}
    node_id = node.id
    for inp in interface.inputs:
        carried = found.carries[inp.name]
        fields[inp.name] = CompiledField(
            ref=Ref._of_parts(node_id, inp.name), type=carried.dtype, binding=node.bindings.get(inp.name), condition=None,
            problems=tuple(on_field.get(inp.name, ())), _index=carried.index, _receive_mode=found.receive_modes[inp.name],
            _is_list_on_node=inp.name in inputs_given_a_list, _makes_rows=None, _view_of=None, _takes_blank=False, _title=inp.title,
            _row_origins=walk.row_origins.of_rows_on(carried.index) if inp.name in inputs_given_a_list else None,
        )
    at_node = condition_at_node(node.bindings, _condition_of_source_above(walk))
    runs_once = runs_per_row_of is None
    for out in outputs:
        ref = Ref._of_parts(node_id, out.name)
        condition = output_condition(node_id, out, at_node, runs_once)
        takes_blank, field_problems_here = out.takes_blank, tuple(on_field.get(out.name, ()))
        dtype = out.dtype
        if holds_tables(dtype):
            table_rows_index = Index(str(ref), parent=runs_per_row_of)
            known = walk.row_origins.note_table(table_rows_index, taken_in)
            fields[out.name] = CompiledField(
                ref=ref, type=Series[dtype] if runs_per_row_of is not None else dtype, binding=None, condition=condition, problems=field_problems_here,
                _index=runs_per_row_of, _receive_mode=None, _is_list_on_node=False, _makes_rows=table_rows_index, _view_of=None, _takes_blank=takes_blank, _title=out.title, _row_origins=known,
            )
            for column, holds in columns_of(dtype):
                name = f"{out.name}.{column}"
                if name in names:
                    continue
                fields[name] = CompiledField(
                    ref=Ref._of_parts(node_id, name), type=Series[holds], binding=None, condition=condition, problems=(),
                    _index=table_rows_index, _receive_mode=None, _is_list_on_node=False, _makes_rows=None, _view_of=(ref, column), _takes_blank=True, _title=column, _row_origins=known,
                )
            continue
        if dtype.element is not None:
            carried, index, makes = dtype, Index(str(ref), parent=runs_per_row_of), True
        elif runs_per_row_of is not None:
            carried, index, makes = Series[dtype], runs_per_row_of, False
        else:
            carried, index, makes = dtype, None, False
        fields[out.name] = CompiledField(
            ref=ref, type=carried, binding=None, condition=condition, problems=field_problems_here,
            _index=index, _receive_mode=None, _is_list_on_node=False, _makes_rows=index if makes else None, _view_of=None, _takes_blank=takes_blank, _title=out.title,
            _row_origins=walk.row_origins.of_rows_on(index),
        )
    return Wiring(fields, runs_per_row_of)


def _place_embedded_graph(
    node: GraphNode, definition: Any, graph: CompiledGraph, embeddable: EmbeddableGraph, interface: Interface, statics: dict[str, Any], runs_per_row_of: Index | None,
    found: InputsRead, walk: WalkSoFar, problems: list[Problem],
) -> None:
    """The graph node's record, and every node inside it under ``runs_per_row_of``."""
    offered = embeddable.inputs
    connected = [name for name, binding in node.bindings.items() if isinstance(binding, From) and name in offered]
    edges: dict[Ref, From] = {offered[name].runs_at: node.bindings[name] for name in connected}  # type: ignore[misc]
    crossings = {offered[name].runs_at: Crossing(type=found.carries[name].dtype, index=found.carries[name].index, receive_mode=found.receive_modes[name]) for name in connected}
    values = {offered[name].runs_at: value for name, value in statics.items()}
    lifted = lift_embedded_nodes(embeddable, node.id, edges, values, runs_per_row_of, crossings, lambda ref: _field_above(walk, ref), walk.row_origins)
    _add(walk, node, problems, definition, graph, interface, statics, Wiring({}, runs_per_row_of))
    walk.compiled.update(lifted)
