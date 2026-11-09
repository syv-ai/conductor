"""A version chain written with ``replaces``, a version that is a graph, an input's bounds and a hook's error."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

import pytest
from conductor import (
    CompiledGraph,
    ComputeHookError,
    Deprecation,
    From,
    Graph,
    GraphNode,
    Input,
    NodeDeclarationError,
    NodeDefinition,
    NodeRegistry,
    Param,
    Policy,
    Series,
    Static,
    Upgrade,
    node,
    upgrade_node,
)
from conductor.widgets import Range, Widget


@node("split")
def split_v1(text: str, sep: str = ",") -> str:
    return text.split(sep)[0]


@node("split", policy=Policy(retries=2), replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
def split_v2(text: str, separator: str = ",") -> str:
    return text.split(separator)[-1]


def test_a_version_is_numbered_by_its_place_in_the_chain_and_keeps_its_own_settings():
    assert sorted(split_v2.versions) == [1, 2] and split_v2.current_version == 2
    assert split_v2.versions[1].policy == Policy() and split_v2.versions[2].policy == Policy(retries=2)
    assert split_v2("a,b") == "b" and split_v1("a,b") == "a"


def test_only_the_newest_goes_in_a_registry():
    with pytest.raises(NodeDeclarationError, match="declared twice"):
        NodeRegistry([split_v1, split_v2])


def test_the_upgrade_given_beside_replaces_moves_a_node_saved_at_the_older_version():
    graph = Graph(nodes=(GraphNode(id="s", type="split", version=1, bindings={"text": Static("a"), "sep": Static(";")}),))

    moved = upgrade_node(graph, "s", NodeRegistry([split_v2]))

    assert moved.nodes[0].version == 2 and set(moved.nodes[0].bindings) == {"text", "separator"}


@pytest.mark.parametrize(
    ("written", "code"),
    [
        (lambda: node("other", replaces=split_v1), "replaces_another_node"),
        (lambda: node("split", upgrade=Upgrade()), "upgrade_without_replaces"),
        (lambda: node("split", replaces=split_v1), "upgrade_missing"),
    ],
)
def test_a_chain_written_wrong_is_refused_where_it_is_written(written, code):
    def newer(text: str) -> str:
        return text

    with pytest.raises(NodeDeclarationError) as raised:
        written()(newer)
    assert raised.value.code == code


def _compiled_inner():
    @node("up")
    def up(text: str) -> str:
        return text.upper()

    return CompiledGraph.from_graph(Graph(nodes=(GraphNode(id="u", type="up", version=1),)), NodeRegistry([up]))


def test_a_version_that_is_a_graph_has_the_fields_it_offers_and_the_notice_of_any_version():
    inner = _compiled_inner()
    definition = NodeDefinition.from_compiled_graphs("wrapped", versions={1: inner, 3: inner}, deprecation=Deprecation(header="Gone"))

    assert definition.versions[3].embedded_graph is inner and definition.versions[3].policy is None
    assert [input_.name for input_ in definition.versions[3].interface.inputs] == ["text"]
    assert definition.versions[1].deprecation is None and definition.versions[3].deprecation == Deprecation(header="Gone")
    assert definition.describe().versions[3].deprecation == Deprecation(header="Gone")


@node("temper")
def temper(temperature: Annotated[float, Param(ge=0, le=2, widget=Range())] = 1.0, name: Annotated[str | None, Param(max_length=3)] = None) -> float:
    return temperature


def _temper_graph(**statics) -> Graph:
    return Graph(nodes=(GraphNode(id="t", type="temper", version=1, bindings={name: Static(value) for name, value in statics.items()}),))


def test_a_typed_in_value_outside_its_bounds_is_a_problem_at_compile():
    compiled = CompiledGraph.from_graph(_temper_graph(temperature=5.0), NodeRegistry([temper]))

    assert [(problem.type, problem.loc) for problem in compiled.problems] == [("invalid_static", ("t", "temperature"))]


def test_a_value_on_an_edge_outside_its_bounds_fails_the_node_for_that_row():
    @node("source")
    def source() -> float:
        return 5.0

    graph = Graph(nodes=(GraphNode(id="s", type="source", version=1), GraphNode(id="t", type="temper", version=1, bindings={"temperature": From("s.result")})))
    compiled = CompiledGraph.from_graph(graph, NodeRegistry([source, temper]))
    assert compiled.is_runnable

    state = compiled.run_sync()

    assert state.status == "failed" and state.failure.type == "invalid_input"


def test_a_blank_on_an_input_that_takes_one_is_held_to_no_bound():
    assert not CompiledGraph.from_graph(_temper_graph(temperature=1.0), NodeRegistry([temper])).problems
    assert temper.versions[1].interface.inputs[1].bounds == {"max_length": 3}


def test_bounds_are_on_the_input_an_editor_reads():
    assert temper.versions[1].interface.inputs[0].model_dump(mode="json")["bounds"] == {"ge": 0.0, "le": 2.0}


@pytest.mark.parametrize("written", [lambda: Param(gte=1), lambda: Input("a", str, max_length="x"), lambda: Input("a", Series[str], ge=0)])
def test_a_bound_written_wrong_is_refused_where_it_is_written(written):
    with pytest.raises(NodeDeclarationError) as raised:
        written()
    assert raised.value.code == "bound_invalid"


class _HostControl(Widget):
    kind: Literal["host-control"] = "host-control"


def test_compile_and_a_run_take_a_control_the_library_has_never_seen():
    @node("hosted")
    def hosted(text: Annotated[str, Param(widget=_HostControl())]) -> str:
        return text

    compiled = CompiledGraph.from_graph(Graph(nodes=(GraphNode(id="h", type="hosted", version=1, bindings={"text": Static("a")}),)), NodeRegistry([hosted]))

    assert compiled.run_sync().status == "complete"


def _graph_with_hook(error: ComputeHookError) -> tuple[Graph, NodeRegistry]:
    def refuse():
        raise error

    @node("hooked", compute_outputs=refuse)
    def hooked() -> Mapping[str, Any]: ...

    return Graph(nodes=(GraphNode(id="h", type="hooked", version=1),)), NodeRegistry([hooked])


def test_a_hook_writes_its_error_as_any_author_error_is_written():
    graph, registry = _graph_with_hook(ComputeHookError("The schema has no columns.", code="schema_empty", ctx={"schema": "s"}, field="columns"))

    (problem,) = CompiledGraph.from_graph(graph, registry).problems

    assert (problem.type, problem.msg, problem.loc, problem.ctx) == ("schema_empty", "The schema has no columns.", ("h", "columns"), {"schema": "s"})


def test_a_hook_error_with_no_code_still_has_one_a_host_can_translate():
    graph, registry = _graph_with_hook(ComputeHookError("No."))

    assert CompiledGraph.from_graph(graph, registry).problems[0].type == "compute_hook_failed"


@node("first_two")
def first_two(items: Annotated[Series[str], Param(min_length=1, max_length=2)]) -> str:
    assert type(items) is list
    return ",".join(items)


def test_a_list_typed_in_is_held_to_its_length_at_compile():
    def graph(items: list[str]) -> Graph:
        return Graph(nodes=(GraphNode(id="f", type="first_two", version=1, bindings={"items": Static(items)}),))

    assert [problem.type for problem in CompiledGraph.from_graph(graph(["a", "b", "c"]), NodeRegistry([first_two])).problems] == ["invalid_static"]
    assert [problem.type for problem in CompiledGraph.from_graph(graph([]), NodeRegistry([first_two])).problems] == ["invalid_static"]
    assert CompiledGraph.from_graph(graph(["a", "b"]), NodeRegistry([first_two])).run_sync().status == "complete"


@pytest.mark.parametrize(("rows", "status"), [(2, "complete"), (3, "failed")])
def test_many_values_on_an_edge_arrive_as_a_list_held_to_its_length(rows, status):
    @node("many")
    def many() -> Series[str]:
        return ["x"] * rows

    graph = Graph(nodes=(GraphNode(id="m", type="many", version=1), GraphNode(id="f", type="first_two", version=1, bindings={"items": From("m.result")})))

    state = CompiledGraph.from_graph(graph, NodeRegistry([many, first_two])).run_sync()

    assert state.status == status
    assert state.failure is None or state.failure.type == "invalid_input"
