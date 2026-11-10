"""A node whose version is a compiled graph: embedded as one node, its inner nodes compiled under the row it runs on.

A host compiles each graph version once and hands the ``CompiledGraph``
to conductor as the version. The outer compile treats the graph node as
a plain node with the compiled graph's interface, which decides the row it
runs on, then compiles the graph's own nodes under that row. What the engine
reads must come out as the graph drawn there: the oracle below checks every
scenario, field by field, against ``fixtures/embedded_expected.json``, and a
run of some of them against ``fixtures/embedded_expected_results.json``.

Which scenario pins which rule of a node compiled under a row:

1. an index renamed and hung under the row: ``rows_made_inside``, ``nested_iterating``;
2. a value that ran once becomes a series on the row: ``entering_scalar``, ``gather_two_inside``;
3. a ``Series[X]`` input over values of one run of the graph gathers them at its row: ``entering_scalar``, ``rows_made_inside``, ``gather_two_inside``;
4. an edge in from outside reads as it would anywhere: ``two_crossings``, ``series_field``, ``gathered_into_the_graph_node``;
5. an output read outside takes the inner field's index: ``rows_made_inside_exposed``, ``read_from_outside``, ``rows_made_under_a_row_inside``;
6. conditions put in place: every ``cond_`` scenario.

A value typed on the graph node's input lands on the inner node, read through
its type (``value_on_the_graph_node``, ``many_values_on_the_graph_node``).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    ArrivingType,
    AsItArrives,
    CompiledGraph,
    ComputeHookError,
    DType,
    GraphRecord,
    NodeDefinition,
    NodeRecord,
    NodeRegistry,
    Output,
    Param,
    Ref,
    Result,
    Series,
    node,
)
from conductor._internal.graph.binding import From, Static
from conductor.widgets import Switch, Textarea
from helpers import Placed, placed, record, registry_of, with_added

FIXTURE = Path(__file__).parent / "fixtures" / "embedded_expected.json"
RESULTS = Path(__file__).parent / "fixtures" / "embedded_expected_results.json"


class Txt(DType, str):
    id = "embedding-test-txt"
    display_title = "Text"


class Flag(DType, int):
    id = "embedding-test-flag"
    display_title = "Yes/No"


Out = Annotated[Txt, Result(title="Result")]


@node("holder", title="Text", description="d", category="test")
def holder(value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return value


@node("upper", title="Upper case", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    """A reduction."""
    return Txt("+".join(texts))


@node("docs", title="Documents", description="d", category="test")
def docs(folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
    return [Txt("a"), Txt("b")]


@node("lines", title="Lines", description="d", category="test")
def lines(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    return text.splitlines()


@node("pair", title="Pair", description="d", category="test")
def pair(
    a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""),
    b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(f"{a}:{b}")


@node("need", title="Need", description="d", category="test")
def need(text: Annotated[Txt, Param(title="Text", widget=Textarea())]) -> Out:
    """A required input with a type and no default."""
    return text


@node("both", title="Both", description="d", category="test")
def both(
    a: Annotated[Txt, Param(title="A", widget=Textarea())],
    b: Annotated[Txt, Param(title="B", widget=Textarea())],
) -> Out:
    """Two required inputs with a type and no default."""
    return Txt(f"{a}:{b}")


def picky_inputs(mode):
    if mode == "bad":
        raise ComputeHookError("Not that mode.", code="picky_refuses")
    return ()

@node("picky", title="Picky", description="d", category="test", compute_added_inputs=picky_inputs)
def picky(mode: Annotated[Txt, Param(title="Mode", widget=Textarea())] = Txt("ok"), **more: Txt) -> Out:
    """CompileProblem the mode ``bad`` in its hook."""
    return mode


@dataclass(frozen=True)
class Branches:
    if_true: Annotated[Any, Result(title="If true", exclusive_group="branches")]
    if_false: Annotated[Any, Result(title="If false", exclusive_group="branches")]


def gate_outputs(value: ArrivingType):
    dtype = value
    return (Output("if_true", dtype, title="If true", exclusive_group="branches"), Output("if_false", dtype, title="If false", exclusive_group="branches"))

@node("gate", title="If/else", description="d", category="test", compute_outputs=gate_outputs)
def gate(
    value: Annotated[Any, Param(title="Value")],
    when: Annotated[Flag, Param(title="When", widget=Switch())] = Flag(1),
) -> Branches:
    return Branches(if_true=value, if_false=SKIPPED) if when else Branches(if_true=SKIPPED, if_false=value)


def single_outputs(**arriving: ArrivingType):
    series = arriving.get("values")
    dtype = series if series is not None else Any
    return (Output("result", dtype, title="The value"),)

@node("single", title="Only one", description="d", category="test", compute_outputs=single_outputs)
def single(values: Annotated[Series[Any], Param(title="Values")]) -> Annotated[Any, Result(title="The value")]:
    """The merge for scalars: whichever branch fired is the one value it receives."""
    return values[0]


PLAIN = (holder, upper_node, join_node, docs, lines, pair, need, both, picky, gate, single)


# -- the scenarios ----------------------------------------------------------------------
#
# Each is its inner graphs, innermost first, and the outer graph. Each inner
# graph becomes a definition whose version is the graph compiled on its own,
# against the registry of the ones before it.


@dataclass(frozen=True)
class Scenario:
    graphs: tuple[tuple[str, tuple[Placed, ...]], ...]
    nodes: tuple[Placed, ...]


def _inner_graph():
    return (
        placed("holder", "holder", defaults_={"value": "inner"}),
        placed("up", "upper", text=From("holder.result")),
        placed("join", "join", texts=From("up.result")),
    )


def _graph_like_a_host_stores(width=8):
    """A graph like the ones a host stores: a holder for its input, a chain, a split into lines, a reduction."""
    nodes = [placed("holder", "holder", defaults_={"value": "x"})]
    before = "holder"
    for i in range(width):
        nodes.append(placed(f"s{i}", "upper", text=From(f"{before}.result")))
        before = f"s{i}"
    nodes.append(placed("lines", "lines", text=From(f"{before}.result")))
    nodes.append(placed("up", "upper", text=From("lines.result")))
    nodes.append(placed("join", "join", texts=From("up.result")))
    return tuple(nodes)


def _branching():
    """holder → gate → an upper on the true branch, and whichever branch fired."""
    return (
        placed("holder", "holder", defaults_={"value": "x"}),
        placed("gate", "gate", value=From("holder.result")),
        placed("up", "upper", text=From("gate.if_true")),
        placed("single", "single", values=From("gate.if_true", "gate.if_false")),
    )


def _middle_branching():
    return (
        placed("h", "holder", defaults_={"value": "m"}),
        placed("g", "gate", value=From("h.result")),
        placed("b", "branchy", value=From("g.if_true")),
        placed("s", "single", values=From("b.up.result", "g.if_false")),
    )


def _docs():
    return placed("docs", "docs")


def _embedded_many(k):
    return Scenario(
        graphs=(("stored-graph", _graph_like_a_host_stores()),),
        nodes=(_docs(), *(placed(f"e{i}", "stored-graph", value=From("docs.result")) for i in range(k))),
    )


def _nested(depth):
    graphs = [("level0", _graph_like_a_host_stores())]
    for level in range(1, depth):
        graphs.append((f"level{level}", (
            placed("holder", "holder", defaults_={"value": "x"}),
            placed("inner", f"level{level - 1}", value=From("holder.result")),
            placed("lines", "lines", text=From("inner.result")),
            placed("up", "upper", text=From("lines.result")),
            placed("join", "join", texts=From("up.result")),
        )))
    return Scenario(
        graphs=tuple(graphs),
        nodes=(_docs(), placed("top", f"level{depth - 1}", value=From("docs.result"))),
    )


SCENARIOS: dict[str, Scenario] = {
    "runs_once": Scenario((("inner-graph", _inner_graph()),), (placed("emb", "inner-graph"),)),
    "runs_once_fed": Scenario((("inner-graph", _inner_graph()),), (
        placed("h", "holder", defaults_={"value": "z"}),
        placed("emb", "inner-graph", value=From("h.result")),
    )),
    "entering_scalar": Scenario((("inner-graph", _inner_graph()),), (
        _docs(), placed("emb", "inner-graph", value=From("docs.result")),
    )),
    "read_from_outside": Scenario((("inner-graph", _inner_graph()),), (
        _docs(),
        placed("emb", "inner-graph", value=From("docs.result")),
        placed("after", "join", texts=From("emb.result")),
    )),
    "rows_made_inside": Scenario((("splitter", (
        placed("holder", "holder", defaults_={"value": "a\nb"}),
        placed("lines", "lines", text=From("holder.result")),
        placed("join", "join", texts=From("lines.result")),
    )),), (_docs(), placed("emb", "splitter", value=From("docs.result")))),
    "rows_made_inside_unfed": Scenario((("splitter", (
        placed("holder", "holder", defaults_={"value": "a\nb"}),
        placed("entered", "holder"),
        placed("lines", "lines", text=From("holder.result")),
        placed("join", "join", texts=From("lines.result")),
    )),), (_docs(), placed("emb", "splitter", **{"entered.value": From("docs.result")}))),
    "rows_made_inside_exposed": Scenario((("splitter", (
        placed("holder", "holder", defaults_={"value": "a\nb"}),
        placed("lines", "lines", text=From("holder.result")),
    )),), (
        _docs(),
        placed("emb", "splitter", value=From("docs.result")),
        placed("after", "join", texts=From("emb.result")),
    )),
    "two_crossings": Scenario((("pair-graph", (
        placed("a", "holder"), placed("b", "holder"),
        placed("ua", "upper", text=From("a.result")),
    )),), (
        _docs(),
        placed("lines", "lines", text=From("docs.result")),
        placed("emb", "pair-graph", **{"a.value": From("docs.result"), "b.value": From("lines.result")}),
    )),
    "series_field": Scenario((("joiner", (placed("join", "join"),)),), (
        _docs(), placed("emb", "joiner", texts=From("docs.result")),
    )),
    "gathered_into_the_graph_node": Scenario((("joiner", (placed("join", "join"),)),), (
        placed("h1", "holder", defaults_={"value": "a"}),
        placed("h2", "holder", defaults_={"value": "b"}),
        placed("emb", "joiner", texts=From("h1.result", "h2.result")),
        placed("after", "upper", text=From("emb.result")),
    )),
    "rows_made_under_a_row_inside": Scenario((("rows-inside", (
        _docs(),
        placed("lines", "lines", text=From("docs.result")),
    )),), (
        placed("emb", "rows-inside"),
        placed("after", "join", texts=From("emb.result")),
    )),
    "per_row_output_read_outside": Scenario((("splitter", (
        placed("holder", "holder", defaults_={"value": "a\nb"}),
        placed("lines", "lines", text=From("holder.result")),
        placed("up", "upper", text=From("lines.result")),
    )),), (
        _docs(),
        placed("emb", "splitter", value=From("docs.result")),
        placed("after", "join", texts=From("emb.result")),
    )),
    "value_and_edge_per_row": Scenario((("pair-graph", (
        placed("a", "holder"), placed("b", "holder"),
        placed("ua", "upper", text=From("a.result")),
        placed("lines", "lines", text=From("b.result")),
        placed("p", "pair", a=From("ua.result"), b=From("lines.result")),
    )),), (
        _docs(),
        placed("emb", "pair-graph", **{"a.value": From("docs.result"), "b.value": Static("x\ny")}),
        placed("after", "join", texts=From("emb.result")),
    )),
    "required_input_filled_inside": Scenario((("filled", (
        placed("n", "need", defaults_={"text": "inside"}),
        placed("u", "upper", text=From("n.result")),
    )),), (
        placed("emb", "filled"),
        placed("after", "upper", text=From("emb.result")),
    )),
    "edge_into_a_value_filled_inside": Scenario((("filled", (
        placed("n", "need", defaults_={"text": "inside"}),
        placed("u", "upper", text=From("n.result")),
    )),), (
        placed("h", "holder", defaults_={"value": "outside"}),
        placed("emb", "filled", text=From("h.result")),
        placed("after", "upper", text=From("emb.result")),
    )),
    "edge_into_a_list_filled_inside": Scenario((("listy", (
        placed("n", "upper", defaults_={"text": ["p", "q"]}),
        placed("j", "join", texts=From("n.result")),
    )),), (
        placed("h", "holder", defaults_={"value": "outside"}),
        placed("emb", "listy", text=From("h.result")),
        placed("after", "upper", text=From("emb.result")),
    )),
    "typed_list_inside": Scenario((("typed-inside", (
        placed("h", "upper"),
        placed("t", "pair", a=From("h.result"), b=Static(["p", "q"])),
        placed("j", "join", texts=From("t.result")),
    )),), (_docs(), placed("emb", "typed-inside", text=From("docs.result")))),
    "gather_two_inside": Scenario((("two", (
        placed("holder", "holder", defaults_={"value": "x"}),
        placed("up", "upper", text=From("holder.result")),
        placed("join", "join", texts=From("holder.result", "up.result")),
    )),), (_docs(), placed("emb", "two", value=From("docs.result")))),
    "embedded_3": _embedded_many(3),
    "nested_2": _nested(2),
    "nested_4": _nested(4),
    "nested_iterating": Scenario((
        ("leaf", _graph_like_a_host_stores(3)),
        ("middle", (
            placed("holder", "holder", defaults_={"value": "a\nb"}),
            placed("lines", "lines", text=From("holder.result")),
            placed("leafy", "leaf", value=From("lines.result")),
            placed("join", "join", texts=From("leafy.result")),
        )),
    ), (
        _docs(),
        placed("top", "middle", value=From("docs.result")),
        placed("after", "join", texts=From("top.result")),
    )),
    "required_input_left_empty": Scenario((("needy", (
        placed("n", "need"),
        placed("u", "upper", text=From("n.result")),
        placed("join", "join", texts=From("u.result")),
    )),), (_docs(), placed("emb", "needy", text=From("docs.result")))),
    "value_on_the_graph_node": Scenario((("inner-graph", _inner_graph()),), (
        placed("emb", "inner-graph", value=Static("set outside")),
        placed("after", "upper", text=From("emb.result")),
    )),
    "many_values_on_the_graph_node": Scenario((("inner-graph", _inner_graph()),), (
        placed("emb", "inner-graph", value=Static(["a", "b"])),
        placed("after", "join", texts=From("emb.result")),
    )),
    "cond_inner_choice_once": Scenario((("branchy", _branching()),), (
        placed("emb", "branchy"),
        placed("after", "upper", text=From("emb.up.result")),
    )),
    "cond_inner_choice_per_row": Scenario((("branchy", _branching()),), (
        _docs(),
        placed("emb", "branchy", value=From("docs.result")),
        placed("after", "single", values=From("emb.up.result")),
    )),
    "cond_outer_gate_into_graph": Scenario((("branchy", _branching()),), (
        placed("h", "holder", defaults_={"value": "y"}),
        placed("og", "gate", value=From("h.result")),
        placed("emb", "branchy", value=From("og.if_false")),
        placed("after", "upper", text=From("emb.single.result")),
    )),
    "cond_merge_inside": Scenario((("merge", (
        placed("a", "holder", defaults_={"value": "x"}),
        placed("b", "holder", defaults_={"value": "z"}),
        placed("g", "gate", value=From("b.result")),
        placed("single", "single", values=From("a.result", "g.if_true")),
    )),), (
        placed("h", "holder", defaults_={"value": "y"}),
        placed("og", "gate", value=From("h.result")),
        placed("emb", "merge", **{"a.value": From("og.if_true")}),
    )),
    "cond_nested_once": Scenario((("branchy", _branching()), ("mid", _middle_branching())), (
        placed("top", "mid"),
        placed("after", "upper", text=From("top.result")),
    )),
    "cond_nested_per_row": Scenario((("branchy", _branching()), ("mid", _middle_branching())), (
        _docs(),
        placed("top", "mid", value=From("docs.result")),
        placed("after", "single", values=From("top.result")),
    )),
}


# -- the builder -----------------------------------------------------------------------


def _plain_registry() -> NodeRegistry:
    return registry_of(*PLAIN)


def _definition(type_id: str, version: Any) -> NodeDefinition:
    return NodeDefinition.from_compiled_graphs(type_id, title=f"Graph {type_id}", description="d", category="test", versions={1: version})


def _registry(scenario: Scenario) -> NodeRegistry:
    """The plain nodes, then each inner graph compiled once and offered as a definition, innermost first."""
    registry = _plain_registry()
    for type_id, nodes in scenario.graphs:
        registry = with_added(registry, _definition(type_id, CompiledGraph.from_graph(record(*list(nodes)), registry)))
    return registry


def _compiled(name: str) -> CompiledGraph:
    scenario = SCENARIOS[name]
    return CompiledGraph.from_graph(record(*list(scenario.nodes)), _registry(scenario))


# -- the oracle --------------------------------------------------------------------------


def _condition(condition) -> list[list[str]]:
    """A condition in one order, so two equal ones write the same."""
    return sorted(sorted(repr(branch_taken) for branch_taken in alternative) for alternative in condition)


def _what_compile_says(compiled: CompiledGraph) -> dict[str, Any]:
    """Every node that runs, in order: its run plan, the values set on it, and
    each field's type, binding and condition, as reprs. What an editor draws is
    here too, so a value that goes missing can't hide behind a run that reads right."""
    assert compiled.is_runnable, compiled.problems
    nodes: dict[str, Any] = {}
    for node_id in compiled.execution_order:
        graph_node = compiled.node(node_id)
        inputs = {input_.name for input_ in graph_node.interface.inputs}
        fields: dict[str, Any] = {}
        for declared in (*graph_node.interface.inputs, *graph_node.interface.outputs):
            found = compiled.field(Ref(node_id, declared.name))
            fields[declared.name] = {
                "type": repr(found.dtype),
                "many": found.many,
                **({"binding": repr(graph_node.graph_node.bindings.get(declared.name))} if declared.name in inputs else {"condition": _condition(found.condition)}),
            }
        nodes[node_id] = {
            "plan": compiled.run_plan.nodes[node_id].model_dump(mode="json"),
            "statics": repr(dict(sorted(graph_node.statics.items()))),
            "fields": fields,
        }
    return {"order": list(compiled.execution_order), "nodes": nodes}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_compiled_graph_embedded_reads_as_the_pinned_graph_drawn_in_place(name):
    assert _what_compile_says(_compiled(name)) == json.loads(FIXTURE.read_text())[name]


def test_the_pinned_values_are_the_scenarios():
    assert set(json.loads(FIXTURE.read_text())) == set(SCENARIOS)



RUN = (
    "entering_scalar", "rows_made_inside_exposed", "gathered_into_the_graph_node", "many_values_on_the_graph_node",
    "nested_iterating", "cond_outer_gate_into_graph", "cond_nested_per_row",
    "per_row_output_read_outside", "value_and_edge_per_row", "required_input_filled_inside", "gather_two_inside",
    "edge_into_a_value_filled_inside", "edge_into_a_list_filled_inside",
)


def _results(compiled: CompiledGraph) -> dict[str, dict[str, str]]:
    """What a run produced, each value as its repr."""
    return {
        node_id: {name: repr(value) for name, value in outputs.items()}
        for node_id, outputs in compiled.results(compiled.run_sync()).items()
    }


@pytest.mark.parametrize("name", RUN)
def test_a_compiled_graph_placed_runs_to_the_same_results(name):
    assert _results(_compiled(name)) == json.loads(RESULTS.read_text())[name]


def test_the_pinned_results_are_the_run_scenarios():
    assert set(json.loads(RESULTS.read_text())) == set(RUN)

# -- the graph node ----------------------------------------------------------------------


def test_the_placed_node_is_a_graph_whose_version_is_the_compiled_graph():
    compiled = _compiled("read_from_outside")
    graph_node = compiled.node("emb")

    assert (graph_node.status, graph_node.embedded_graph is not None) == ("ready", True)
    assert isinstance(graph_node.embedded_graph, CompiledGraph)
    assert compiled.node("emb/join").embedded_in == "emb"
    assert compiled.field(Ref("emb/join", "result")) is compiled.field(Ref("emb/join", "result"))


def test_a_value_set_on_the_placed_node_is_read_by_the_inner_node():
    compiled = _compiled("value_on_the_graph_node")

    assert compiled.node("emb/holder").statics == {"value": Txt("set outside")}
    assert compiled.node("emb").statics == {"value": Txt("set outside")}
    assert compiled.node("emb/holder").graph_node.bindings.get("value") == Static("set outside")


def _broken_graph_registry():
    broken = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="a", node="holder"),
        NodeRecord(id="b", node="upper"),
    ], edges=[("a.nope", "b.text")]), _plain_registry())
    return broken, with_added(_plain_registry(), _definition("broken", broken))


def test_a_broken_graph_is_one_problem_on_the_graph_node():
    """The graph's own problems stay on the graph, in its own words; where
    it is embedded, the author sees one problem saying the graph needs fixing."""
    broken, registry = _broken_graph_registry()
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="emb", node="broken"),
        NodeRecord(id="after", node="upper"),
    ], edges=[("emb.result", "after.text")]), registry)

    (found,) = compiled.problems
    assert (found.type, found.fatal, found.loc) == ("embedded_graph_broken", True, ("emb",))
    assert found.ctx == {"graph": "Graph broken", "problems": 1}
    assert compiled.node("emb").status == "wiring_failed"
    assert compiled.node("after").status == "wiring_failed"
    assert compiled.node("emb").embedded_graph is broken
    assert [p.type for p in compiled.node("emb").embedded_graph.problems] == ["unknown_ref_output"]


def test_an_outer_edge_may_only_read_what_the_graph_offers():
    """``up`` feeds ``join`` inside, so the graph does not offer its output."""
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="emb", node="inner-graph"),
        NodeRecord(id="after", node="upper"),
    ], edges=[("emb.up.result", "after.text")]), _registry(SCENARIOS["runs_once"]))

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("unknown_ref_output", "after")]



def test_two_series_made_by_different_nodes_inside_do_not_line_up_outside():
    """Each series output of an embedded graph sits on the index of the node
    inside that makes it, so two unrelated ones are ``misaligned`` where they
    meet, exactly as if the graph were drawn in place."""
    scenario = Scenario((("two-series", (
        placed("a", "lines", text=Static("x\ny")),
        placed("b", "lines", text=Static("p\nq")),
    )),), (
        placed("emb", "two-series"),
        placed("after", "pair", a=From("emb.a.result"), b=From("emb.b.result")),
    ))
    problems = CompiledGraph.from_graph(record(*list(scenario.nodes)), _registry(scenario)).problems

    assert [(p.type, p.loc[0]) for p in problems] == [("misaligned", "after")]



def test_a_graph_that_cannot_run_is_refused_even_when_every_node_is_ready():
    """A value no type can read is fatal without breaking its node."""
    graph = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="h", node="holder"),
        NodeRecord(id="g", node="gate", values={"when": "not a number"}),
    ], edges=[("h.result", "g.value")]), _plain_registry())
    assert not graph.is_runnable
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="emb", node="bad"),
        ]), with_added(_plain_registry(), _definition("bad", graph)),
    )

    assert [(p.type, p.ctx) for p in compiled.problems] == [("embedded_graph_broken", {"graph": "Graph bad", "problems": 1})]


def test_a_graph_whose_required_input_is_left_empty_can_be_placed():
    """The value arrives from outside, so the one fatal problem inside is not the graph's fault."""
    compiled = _compiled("required_input_left_empty")

    assert compiled.is_runnable, compiled.problems


def _embedded_alone(inner: tuple[Placed, ...], **bindings: Any) -> CompiledGraph:
    graph = CompiledGraph.from_graph(record(*list(inner)), _plain_registry())
    registry = with_added(_plain_registry(), _definition("g", graph))
    return CompiledGraph.from_graph(record(placed("emb", "g", **bindings)), registry)


def test_a_graph_with_an_empty_input_it_does_not_offer_cannot_be_embedded():
    """Only an input the graph offers can be filled where it is embedded. One
    on a node that also has an edge in stays empty."""
    fed_beside = _embedded_alone((
        placed("h", "holder", defaults_={"value": "x"}),
        placed("p", "both", a=From("h.result")),
    ))

    assert [(p.type, p.ctx) for p in fed_beside.problems] == [("embedded_graph_broken", {"graph": "Graph g", "problems": 1})]


def test_a_value_typed_on_a_placed_graph_that_a_hook_inside_refuses_is_the_outer_authors_problem():
    """Inside the graph, the value is refused. The outer author
    typed it, so they see the hook's own words on the field they typed in."""
    compiled = _embedded_alone((placed("p", "picky"),), **{"mode": Static("bad")})

    assert [(p.type, p.msg, p.loc[0], p.loc[1]) for p in compiled.problems] == [
        ("picky_refuses", "Not that mode.", "emb", "mode"),
    ]
    assert compiled.node("emb").status == "wiring_failed"


def test_an_input_filled_inside_is_offered_as_not_required():
    """The graph node is whole without it: its author already put a default there."""
    graph = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="n", node="need", defaults={"text": "inside"}),
    ]), _plain_registry())
    empty = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="n", node="need"),
    ]), _plain_registry())

    assert [(input_.name, input_.required) for input_ in graph._embeddable.interface.inputs] == [("text", False)]
    assert [(input_.name, input_.required) for input_ in empty._embeddable.interface.inputs] == [("text", True)]


def _pass_on_what_arrives(**arriving: ArrivingType):
    return (Output("result", arriving["thing"], title="Thing"),)


@node("as_it_arrives", title="As it arrives", description="d", category="test", compute_outputs=_pass_on_what_arrives)
def as_it_arrives(**thing: AsItArrives) -> Annotated[Any, Result(title="Thing")]:
    return thing["thing"]


def test_a_value_inside_a_graph_run_per_row_arrives_as_itself_on_an_as_it_arrives_input():
    """One value per run of the graph: a hook is told its type, and the call gets it at its own row."""
    inner = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="h", node="holder"),
        NodeRecord(id="kw", node="as_it_arrives"),
    ], edges=[("h.result", "kw.thing")]), with_added(_plain_registry(), as_it_arrives))
    registry = with_added(_plain_registry(), as_it_arrives, _definition("g", inner))
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="docs", node="docs"),
        NodeRecord(id="emb", node="g"),
    ], edges=[("docs.result", "emb.value")]), registry)

    assert compiled.is_runnable, compiled.problems
    plan = compiled.run_plan.nodes["emb/kw"].inputs["thing"]
    assert (plan.receive_mode, plan.rows_of) == ("value_at_row", "docs.result")
    assert [str(value) for value in compiled.results(compiled.run_sync())["emb/kw"]["result"]["values"]] == ["a", "b"]


def test_a_graph_run_per_row_sees_an_input_fed_whole_as_a_whole():
    """``u`` is fed one doc per call, so the embedded graph runs once per doc.
    ``j`` is fed the docs too, into a series input: like any node that runs
    per row, each run sees the series whole."""
    scenario = Scenario((("mixed", (placed("u", "upper"), placed("j", "join"))),), (
        _docs(),
        placed("emb", "mixed", text=From("docs.result"), texts=From("docs.result")),
    ))
    compiled = CompiledGraph.from_graph(record(*list(scenario.nodes)), _registry(scenario))

    assert [str(value) for value in compiled.results(compiled.run_sync())["emb/j"]["result"]["values"]] == ["a+b", "a+b"]


def test_a_broken_graph_with_a_value_typed_on_it_is_one_problem():
    broken, registry = _broken_graph_registry()
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="emb", node="broken", values={"value": "v"}),
    ]), registry)

    assert [p.type for p in compiled.problems] == ["embedded_graph_broken"]


def test_a_graph_the_walk_leaves_out_is_one_node_with_nothing_inside():
    """Its edge in is broken, or it can't be embedded: either way it stays one
    ``graph`` node, saying why it is not ready, and nothing inside it is a
    node of this graph. What is inside is read through its version."""
    upstream_broken = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="bad", node="nope"),
        NodeRecord(id="emb", node="inner-graph"),
    ], edges=[("bad.result", "emb.value")]), _registry(SCENARIOS["runs_once"]))
    broken, registry = _broken_graph_registry()
    unembeddable = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="emb", node="broken"),
    ]), registry)

    for compiled, cause in ((upstream_broken, "unknown_node_type"), (unembeddable, "embedded_graph_broken")):
        graph_node = compiled.node("emb")
        assert (graph_node.status, graph_node.embedded_graph is not None, graph_node.cause.type) == ("wiring_failed", True, cause)
        with pytest.raises(KeyError):
            compiled.node("emb/holder" if compiled is upstream_broken else "emb/a")
    assert unembeddable.node("emb").embedded_graph.node("a").status == "ready"


def test_a_value_on_a_field_the_graph_does_not_offer_is_a_stale_binding():
    """``up.text`` is fed inside, so the graph does not offer it; its value
    inside is not what the graph node reports holding."""
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="emb", node="inner-graph", values={"text": "x"}),
    ]), _registry(SCENARIOS["runs_once"]))

    assert [(p.type, p.loc[0], p.loc[1]) for p in compiled.problems] == [("stale_binding", "emb", "text")]
    assert compiled.node("emb").statics == {}


def test_an_edge_into_a_field_the_graph_does_not_offer_is_a_stale_binding():
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="h", node="holder"),
        NodeRecord(id="emb", node="inner-graph"),
    ], edges=[("h.result", "emb.text")]), _registry(SCENARIOS["runs_once"]))

    assert [(p.type, p.loc[0], p.loc[1]) for p in compiled.problems] == [("stale_binding", "emb", "text")]

