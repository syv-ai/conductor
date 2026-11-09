"""One node, compiled from the nodes compiled above it.

Find its definition and version; read the statics; ask
``compute_added_inputs``; check its bindings; wire each input from the nodes above;
ask ``compute_outputs``; build its fields. It returns at the first stage that
can't go on, with what it has, and the node says so by what it lacks
(``interface``, ``_wiring``). A node a graph stands for then compiles each node
of that graph the same way, under the row it runs on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from conductor._internal.dtype import is_untyped
from conductor._internal.errors import ComputeHookError, Problem, make_problem
from conductor._internal.execution.values_of_run import (
    CellsInAList,
    ColumnOfTable,
    ListOnNode,
    NotHeld,
    OneValue,
    TableAtRow,
)
from conductor._internal.field_records import Input, Output, was_titled_by_its_author
from conductor._internal.graph.binding import Binding, From
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode, Wiring
from conductor._internal.graph.decide_rows_and_receive_modes import (
    WhatArrivesOnInputs,
    decide_runs_per_row_of_and_receive_modes,
    read_what_arrives_on_inputs,
)
from conductor._internal.graph.embedding import (
    inner_node_placed_at,
    parent_graph_of_node_id,
)
from conductor._internal.graph.graph_interface import field_problems
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.graph.output_conditions import (
    ALWAYS,
    Condition,
    condition_node_runs_under,
    output_condition,
)
from conductor._internal.graph.read_static_values import read_static_values
from conductor._internal.graph.row_origins import origins_taken_in
from conductor._internal.index import Index
from conductor._internal.node_signature import Interface
from conductor._internal.ref import NODE_PATH_SEPARATOR, Ref
from conductor._internal.registry import NodeRegistry
from conductor._internal.table import columns_of, is_table_type

if TYPE_CHECKING:
    from conductor._internal.graph.compile_graph import CompileProgress
    from conductor._internal.node_definition import NodeVersion


def _interface_with_fields(interface: Interface, inputs: tuple[Input, ...], outputs: tuple[Output, ...]) -> Interface:
    return Interface(inputs=inputs, outputs=outputs, needs=interface.needs)


def compile_one_node(node: GraphNode, progress: CompileProgress, registry: NodeRegistry, runs_under: Index | None = None) -> None:
    """Compile ``node`` and add it to ``progress``, with its inner nodes if a graph stands for it.

    ``runs_under``, on a node inside an embedded graph, is the row the graph node runs on."""
    problems = progress.problems_found_early.pop(node.id, [])
    if node.type not in registry:
        problems.append(make_problem("unknown_node_type", node.id, node_type=node.type, choices=", ".join(sorted(registry)) or "none"))
        return _record_compiled_node(progress, node, problems)
    definition = registry[node.type]
    version: Any = definition.versions.get(node.version)
    if version is None:
        problems.append(make_problem("unknown_node_version", node.id, node_type=node.type, version=node.version))
        return _record_compiled_node(progress, node, problems)
    drawn = node
    #: An inner node's edges read as compile left them, and its fields passed the same checks when its graph was compiled.
    placed_inside_a_graph = NODE_PATH_SEPARATOR in node.id
    if progress.embedded_graph_node_ids and not placed_inside_a_graph:
        node = _with_edges_out_of_embedded_graphs_resolved(node, progress)
    is_graph = version.embedded_graph is not None
    if is_graph:
        progress.embedded_graph_node_ids.add(node.id)

    declared = version.interface
    statics, inputs_given_a_list, unreadable_static_problems = read_static_values(declared.inputs, node)
    inputs = declared.inputs
    if version.compute_added_inputs is not None:
        defaults = {input_.name: input_.default for input_ in inputs if not input_.required}
        try:
            added = version.compute_added_inputs.call_with({**defaults, **statics}, {}, statics) or ()
        except ComputeHookError as hook_error:
            problems.append(hook_error.as_problem_on_node(node.id))
            return _record_compiled_node(progress, node, problems, definition, version, None, {})
        inputs = (*inputs, *added)
        statics_added, lists_added, unreadable_added = read_static_values(added, node)
        statics, inputs_given_a_list, unreadable_static_problems = {**statics, **statics_added}, inputs_given_a_list | lists_added, [*unreadable_static_problems, *unreadable_added]
    problems.extend(unreadable_static_problems)

    can_be_wired = True
    if is_graph:
        inputs_given_a_list = frozenset()  # a list typed on a graph node lands on the node inside
        if version.embedded_graph._embeddable.problems:
            can_be_wired = False
            problems.append(make_problem("embedded_graph_broken", node.id, graph=definition.title, problems=len(version.embedded_graph._embeddable.problems)))
    elif version.open_input_template is not None:
        named = {input_.name for input_ in inputs}
        inputs = (*inputs, *(
            version.open_input_template.model_copy(update={"name": name, "title": name})
            for name, binding in node.bindings.items()
            if name not in named and isinstance(binding, From)
        ))

    hooks_decide_fields = version.compute_added_inputs is not None or version.compute_outputs is not None
    can_be_wired = (placed_inside_a_graph and not hooks_decide_fields) or (_bindings_can_be_wired(drawn, inputs, version, progress, problems) and can_be_wired)
    if not can_be_wired:
        return _record_compiled_node(progress, node, problems, definition, version, _interface_with_fields(declared, inputs, declared.outputs), statics)

    inputs_read = read_what_arrives_on_inputs(node, inputs, statics, inputs_given_a_list, progress, problems)
    if inputs_read.stopped_at_unreadable_source:
        return _record_compiled_node(progress, node, problems, definition, version, _interface_without_wiring(node, version, inputs, inputs_read, statics, problems), statics)
    inputs_typed_by_their_edges = tuple(_with_type_from_its_edge(input_, inputs_read.given_by_edge) for input_ in inputs)
    try:
        outputs = _outputs_after_compute_outputs(version, inputs_read.arriving_types, statics)
    except ComputeHookError as hook_error:
        problems.append(hook_error.as_problem_on_node(node.id))
        return _record_compiled_node(progress, node, problems, definition, version, _interface_with_fields(version.interface, inputs_typed_by_their_edges, ()), statics)
    interface = _interface_with_fields(version.interface, inputs_typed_by_their_edges, outputs)
    problems_found: list[Problem] = []
    if not placed_inside_a_graph or hooks_decide_fields:
        problems_found = field_problems(node.id, outputs, frozenset(input_.name for input_ in inputs_typed_by_their_edges), untyped_ok=False)
        problems_found += [make_problem("series_of_tables", node.id, out.name) for out in outputs if out.many and is_table_type(out.dtype)]
    if problems_found:
        problems.extend(problems_found)
        return _record_compiled_node(progress, node, problems, definition, version, interface, statics)
    if not outputs and (is_graph or version.return_shape is not None):
        problems.append(make_problem("no_outputs", node.id))
    rows_agree, runs_per_row_of = decide_runs_per_row_of_and_receive_modes(node, inputs_read, problems, progress.row_origins, runs_under)
    if not rows_agree:
        return _record_compiled_node(progress, node, problems, definition, version, interface, statics)
    if is_graph:
        return _place_embedded_graph(node, definition, version, interface, statics, runs_per_row_of, progress, problems)
    wiring = _wiring_of_function_node(node, interface, inputs_read, {name: statics[name] for name in inputs_given_a_list}, runs_per_row_of, progress, problems)
    _record_compiled_node(progress, node, problems, definition, version, interface, statics, wiring)


def _record_compiled_node(
    progress: CompileProgress, node: GraphNode, problems: list[Problem], definition: Any = None, version: Any = None,
    interface: Interface | None = None, statics: dict[str, Any] | None = None, wiring: Wiring | None = None,
) -> None:
    """One finished record, with the cause of a node that isn't wired."""
    progress.compiled[node.id] = CompiledNode(
        id=node.id, graph_node=node, embedded_in=parent_graph_of_node_id(node.id), problems=tuple(problems),
        cause=None if wiring is not None else _why_node_is_unwired(node, problems, progress),
        definition=definition, version=version, interface=interface, statics=statics or {}, wiring=wiring,
    )
    progress.problems.extend(problems)


def _why_node_is_unwired(node: GraphNode, problems: list[Problem], progress: CompileProgress) -> Problem:
    """Why a node isn't wired: its own first fatal problem, or else the cause of the first source that isn't."""
    own = next((problem for problem in problems if problem.fatal), None)
    if own is not None:
        return own
    for binding in node.bindings.values():
        if not isinstance(binding, From):
            continue
        for ref in binding.refs:
            if ref.node_id in progress.invalid_node_ids:
                return progress.invalid_node_ids[ref.node_id]
            source = progress.compiled.get(ref.node_id) or _refused_graph_around(ref.node_id, progress)
            if source is not None and source._wiring is None:
                return source.cause
    raise AssertionError(f"compile left {node.id!r} unwired with nothing fatal to say why")


def _refused_graph_around(node_id: str, progress: CompileProgress) -> CompiledNode | None:
    """The nearest graph node around ``node_id`` on record: a refused graph's inner nodes are taken back out, and it holds their cause."""
    graph_id = parent_graph_of_node_id(node_id)
    while graph_id is not None and graph_id not in progress.compiled:
        graph_id = parent_graph_of_node_id(graph_id)
    return None if graph_id is None else progress.compiled[graph_id]


def _with_edges_out_of_embedded_graphs_resolved(node: GraphNode, progress: CompileProgress) -> GraphNode:
    """``node`` with every edge out of an embedded graph pointed at the field
    inside it that the name offers (``approve.total`` at ``approve/check.total``):
    the one place compile reads an offered name. An edge naming an output it
    doesn't offer stays as written, and wiring reports it (``unknown_ref_output``)."""
    def resolved(ref: Ref) -> Ref:
        source = progress.compiled.get(ref.node_id)
        inner_address = None if source is None or source._wiring is None or source.embedded_graph is None else source.embedded_graph._embeddable.outputs.get(ref.field)
        return ref if inner_address is None else Ref(f"{ref.node_id}{NODE_PATH_SEPARATOR}{inner_address.node_id}", inner_address.field)

    if not any(isinstance(binding, From) and any(ref.node_id in progress.embedded_graph_node_ids for ref in binding.refs) for binding in node.bindings.values()):
        return node
    return node.model_copy(update={"bindings": {
        name: From(*(resolved(ref) for ref in binding.refs)) if isinstance(binding, From) else binding
        for name, binding in node.bindings.items()
    }})


def _bindings_can_be_wired(
    node: GraphNode, inputs: tuple[Input, ...], version: Any, progress: CompileProgress, problems: list[Problem],
) -> bool:
    """The stored bindings against the inputs the node actually has, and those
    inputs against the field rules. ``False`` when its edges can't be wired.

    A required input left empty is fatal but wiring goes on: its type is
    declared, so the node and everything after it are still worked out."""
    declared = {input_.name: input_ for input_ in inputs}
    if node.locked:
        problems.extend(make_problem("unknown_locked_field", node.id, name) for name in node.locked if name not in declared)
    can_wire = True
    #: A misspelled binding also leaves its input unbound: one problem, the typo, for each.
    unbound_explained_by_typos = 0
    problems_found = field_problems(node.id, inputs, untyped_ok=True)
    problems_found += [make_problem("parameter_name_invalid", node.id, input_.name) for input_ in inputs if input_.name.startswith("_")]
    if problems_found:
        can_wire = False
        problems.extend(problems_found)
    for name, binding in node.bindings.items():
        if name not in declared:
            hook_decides_inputs = version.compute_added_inputs is not None
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
            if source_id in progress.invalid_node_ids and NODE_PATH_SEPARATOR not in node.id:
                can_wire = False
            elif source_id not in progress.compiled or (NODE_PATH_SEPARATOR in source_id and NODE_PATH_SEPARATOR not in node.id):  # only a placed inner node reads a path
                can_wire = False
                problems.append(make_problem("unknown_ref_node", node.id, name, source_node=source_id))
    for input_ in inputs:
        if is_untyped(input_.dtype):
            if isinstance(node.bindings.get(input_.name), From):
                continue
            can_wire = False
        elif not input_.required or input_.name in node.bindings:
            continue
        if unbound_explained_by_typos:
            unbound_explained_by_typos -= 1
            continue
        problems.append(make_problem("unbound_required", node.id, input_.name))
    return can_wire


def _interface_without_wiring(
    node: GraphNode, version: Any, inputs: tuple[Input, ...], inputs_read: WhatArrivesOnInputs, statics: dict[str, Any], problems: list[Problem],
) -> Interface:
    """A source couldn't be read: complete the node's fields from what did arrive, so an editor can draw it."""
    try:
        answered = _outputs_after_compute_outputs(version, inputs_read.arriving_types, statics)
    except ComputeHookError:
        # The broken source carries the problem; a problem about its missing
        # arrival would report the same fact twice.
        answered = ()
    inputs_typed_by_their_edges = tuple(_with_type_from_its_edge(input_, inputs_read.given_by_edge) for input_ in inputs)
    problems.extend(field_problems(node.id, answered, frozenset(input_.name for input_ in inputs_typed_by_their_edges), untyped_ok=True))
    return _interface_with_fields(version.interface, inputs_typed_by_their_edges, answered)


def _outputs_after_compute_outputs(version: Any, arriving_types: dict[str, Any], statics: dict[str, Any]) -> tuple[Output, ...]:
    """What ``compute_outputs`` says once it can be told what arrives."""
    if version.compute_outputs is None:
        return version.interface.outputs
    values = {**{input_.name: input_.default for input_ in version.interface.inputs if not input_.required}, **statics}
    answered = version.compute_outputs.call_with(values, arriving_types, statics)
    return version.interface.outputs if answered is None else answered


def _with_type_from_its_edge(field: Input, given_by_edge: dict[str, dict[str, Any]]) -> Input:
    """``field`` with what its edge gave it, if an edge gave anything."""
    return field.model_copy(update=given_by_edge[field.name]) if field.name in given_by_edge else field


def _field_above(progress: CompileProgress, ref: Ref) -> CompiledField | None:
    """The field ``ref`` names on a node compiled so far, if it was worked out."""
    node = progress.compiled.get(ref.node_id)
    return None if node is None or node._wiring is None else node._wiring.fields.get(ref.field)


def _condition_lookup_for_sources(progress: CompileProgress) -> Any:
    def condition_of(ref: Ref) -> Condition:
        field = _field_above(progress, ref)
        return field.condition if field is not None and field.condition is not None else ALWAYS

    return condition_of


def _wiring_of_function_node(
    node: GraphNode, interface: Interface, inputs_read: WhatArrivesOnInputs, lists_set_on_inputs: Mapping[str, Sequence[Any]], runs_per_row_of: Index | None, progress: CompileProgress, problems: list[Problem],
) -> Wiring:
    """Every field of a node that is a function, built once its row is known."""
    outputs = interface.outputs
    names = {out.name for out in outputs}
    taken_in = origins_taken_in(inputs_read.tables_taken_in)
    for out in outputs:
        if is_table_type(out.dtype):
            for column, _ in columns_of(out.dtype):
                name = f"{out.name}.{column}"
                if name in names:
                    problems.append(make_problem("view_name_taken", node.id, name, table=out.name, column=column))
    problems_by_field: dict[str, list[Problem]] = {}
    for problem in problems:
        if len(problem.loc) > 1:
            problems_by_field.setdefault(str(problem.loc[1]), []).append(problem)
    fields: dict[str, CompiledField] = {}
    node_id = node.id
    for input_ in interface.inputs:
        field_type = inputs_read.carries[input_.name]
        is_list_set_on_it = input_.name in lists_set_on_inputs
        fields[input_.name] = CompiledField(
            ref=Ref._from_checked_parts(node_id, input_.name), type=field_type.dtype, binding=node.bindings.get(input_.name), condition=None,
            problems=tuple(problems_by_field.get(input_.name, ())), _index=field_type.index, _receive_mode=inputs_read.receive_modes[input_.name],
            _held_as=ListOnNode(field_type.index, list(lists_set_on_inputs[input_.name])) if is_list_set_on_it else NotHeld(),  # type: ignore[arg-type]  # a list on a node has its index
            _takes_blank=False, _title=input_.title if was_titled_by_its_author(input_) else input_.name,
            _row_origins=progress.row_origins.origins_of_rows_on(field_type.index) if is_list_set_on_it else None,
        )
    node_condition = condition_node_runs_under(node.bindings, _condition_lookup_for_sources(progress))
    runs_once = runs_per_row_of is None
    for out in outputs:
        ref = Ref._from_checked_parts(node_id, out.name)
        condition = output_condition(node_id, out, node_condition, runs_once)
        takes_blank, field_problems_here = out.takes_blank, tuple(problems_by_field.get(out.name, ()))
        dtype = out.dtype
        if is_table_type(dtype):
            table_rows_index = Index(str(ref), parent=runs_per_row_of)
            origins = progress.row_origins.record_table(table_rows_index, taken_in)
            fields[out.name] = CompiledField(
                ref=ref, type=dtype, binding=None, condition=condition, problems=field_problems_here,
                _index=runs_per_row_of, _receive_mode=None, _held_as=TableAtRow(table_rows_index), _takes_blank=takes_blank, _title=out.title if was_titled_by_its_author(out) else out.name, _row_origins=origins,
            )
            for column, column_dtype in columns_of(dtype):
                name = f"{out.name}.{column}"
                if name in names:
                    continue
                fields[name] = CompiledField(
                    ref=Ref._from_checked_parts(node_id, name), type=column_dtype, binding=None, condition=condition, problems=(),
                    _index=table_rows_index, _receive_mode=None, _held_as=ColumnOfTable(ref, column), _takes_blank=True, _title=column, _row_origins=origins,
                )
            continue
        index = Index(str(ref), parent=runs_per_row_of) if out.many else runs_per_row_of
        fields[out.name] = CompiledField(
            ref=ref, type=dtype, binding=None, condition=condition, problems=field_problems_here,
            _index=index, _receive_mode=None, _held_as=CellsInAList(index) if out.many else OneValue(), _takes_blank=takes_blank, _title=out.title if was_titled_by_its_author(out) else out.name,
            _row_origins=progress.row_origins.origins_of_rows_on(index),
        )
    return Wiring(fields, runs_per_row_of)


def _place_embedded_graph(
    node: GraphNode, definition: Any, version: NodeVersion, interface: Interface, statics: dict[str, Any], runs_per_row_of: Index | None, progress: CompileProgress, problems: list[Problem],
) -> None:
    """The graph node's record, and every node inside it compiled where it stands.

    A fatal problem the walk finds is one this graph node's edges or values
    brought in, so it is reported on the graph node, and the graph is not
    embedded. Anything else was its author's to see when it was compiled."""
    prefix = f"{node.id}{NODE_PATH_SEPARATOR}"
    graph = version.embedded_graph
    assert graph is not None
    offered = graph._embeddable.inputs
    filled_from_outside: dict[str, dict[str, Binding]] = {}
    for name, binding in node.bindings.items():
        if name in offered and (isinstance(binding, From) or name in statics):
            filled_from_outside.setdefault(offered[name].node_id, {})[offered[name].field] = binding
    problems_before = len(progress.problems)
    _record_compiled_node(progress, node, problems, definition, version, interface, statics, Wiring({}, runs_per_row_of))  # inner nodes read the row off this record
    problems_from_this_node = len(progress.problems)
    placed_after = len(progress.compiled)
    for inner in graph._nodes.values():
        if inner.embedded_in is None:
            compile_one_node(inner_node_placed_at(inner.graph_node, prefix, filled_from_outside.get(inner.id, {})), progress, graph._registry, runs_under=runs_per_row_of)
    found_inside = progress.problems[problems_from_this_node:]
    del progress.problems[problems_before:]
    offered_at = {prefix + address.node_id: name for name, address in offered.items() if name in node.bindings}
    brought_in = [
        problem.model_copy(update={"loc": (node.id, offered_at[problem.loc[0]]) if problem.loc[0] in offered_at else (node.id,)})
        for problem in found_inside if problem.fatal and not _is_an_offered_input_left_empty(problem, prefix, offered)
    ]
    problems.extend(brought_in)
    if brought_in:
        for inner_id in list(progress.compiled)[placed_after:]:
            del progress.compiled[inner_id]
    _record_compiled_node(progress, node, problems, definition, version, interface, statics, None if brought_in else Wiring({}, runs_per_row_of))


def _is_an_offered_input_left_empty(problem: Problem, prefix: str, offered: Mapping[str, Ref]) -> bool:
    """The graph node already said this one: the input it offers has nothing in it."""
    return problem.type == "unbound_required" and any(str(problem.loc[0]) == prefix + address.node_id and str(problem.loc[1]) == address.field for address in offered.values())
