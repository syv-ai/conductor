"""The condition under which an output appears, derived from choice and edges."""

from dataclasses import dataclass
from typing import Annotated, Any

from conductor import (
    ALWAYS,
    SKIPPED,
    Arriving,
    Atom,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Output,
    Param,
    Ref,
    Result,
    Series,
    Static,
    node,
)
from conductor.widgets import Switch, Textarea


class Txt(DType, str):
    id = "conditions-test-txt"
    display_title = "Text"


class Flag(DType, int):
    id = "conditions-test-flag"
    display_title = "Yes/No"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Branches:
    if_true: Annotated[Any, Result(title="If true", choice="branches")]
    if_false: Annotated[Any, Result(title="If false", choice="branches")]


def gate_outputs(value: Arriving):
    dtype = value
    return (Output("if_true", dtype, title="If true", choice="branches"), Output("if_false", dtype, title="If false", choice="branches"))

@node("gate", title="If/else", description="d", category="test", compute_outputs=gate_outputs)
def gate(
    value: Annotated[Any, Param(title="Value")],
    when: Annotated[Flag, Param(title="When", widget=Switch())] = Flag(1),
) -> Branches:
    return Branches(if_true=value, if_false=SKIPPED) if when else Branches(if_true=SKIPPED, if_false=value)


@node("holder", title="Text", description="d", category="test")
def holder(value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return value


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


def single_outputs(**arriving: Arriving):
    series = arriving.get("values")
    dtype = series.element if series is not None else Any
    return (Output("result", dtype, title="The value"),)

@node("single", title="Only one", description="d", category="test", compute_outputs=single_outputs)
def single(values: Annotated[Series[Any], Param(title="Values")]) -> Annotated[Any, Result(title="The value")]:
    """The merge for scalars: a `Series[X]` input drops the skipped edges, so
    whichever branch fired is the one value it receives."""
    return values[0]


@node("docs", title="Docs", description="d", category="test")
def docs(folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
    return [Txt("a")]


def _compiled(nodes):
    registry = NodeRegistry()
    for node_cls in (gate, holder, upper_node, docs, single):
        registry = registry.extended_with([node_cls])
    return CompiledGraph.from_graph(Graph(nodes=nodes), registry)


def _edge(*refs):
    return From(*(Ref(n, f) for n, f in refs))


def test_an_output_nothing_gates_appears_always():
    compiled = _compiled([GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")})])

    assert compiled.field(Ref("h", "result")).condition == ALWAYS


def test_a_branch_appears_when_its_decision_went_that_way():
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        GraphNode(id="g", type="gate", version=1, bindings={"value": _edge(("h", "result"))}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": _edge(("g", "if_true"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("g", "if_true")).condition == frozenset({frozenset({Atom("g", "branches", "if_true")})})
    assert compiled.field(Ref("yes", "result")).condition == frozenset({frozenset({Atom("g", "branches", "if_true")})})
    assert compiled.decisions == {"g": {"branches": ("if_true", "if_false")}}


def test_a_merge_adds_an_alternative():
    """A `Series[X]` input drops the skipped edges, so past the
    merge the output appears when either branch fired."""
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        GraphNode(id="g", type="gate", version=1, bindings={"value": _edge(("h", "result"))}),
        GraphNode(id="a", type="upper", version=1, bindings={"text": _edge(("g", "if_true"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": _edge(("g", "if_false"))}),
        GraphNode(id="m", type="single", version=1, bindings={"values": _edge(("a", "result"), ("b", "result"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("m", "result")).condition == frozenset({
        frozenset({Atom("g", "branches", "if_true")}), frozenset({Atom("g", "branches", "if_false")}),
    })


def test_two_decisions_in_series_conjoin_and_a_contradiction_is_dropped():
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        GraphNode(id="g1", type="gate", version=1, bindings={"value": _edge(("h", "result"))}),
        GraphNode(id="g2", type="gate", version=1, bindings={"value": _edge(("g1", "if_true"))}),
        GraphNode(id="both", type="single", version=1, bindings={"values": _edge(("g2", "if_true"), ("g1", "if_false"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("g2", "if_true")).condition == frozenset({
        frozenset({Atom("g1", "branches", "if_true"), Atom("g2", "branches", "if_true")}),
    })
    assert compiled.field(Ref("both", "result")).condition == frozenset({
        frozenset({Atom("g1", "branches", "if_true"), Atom("g2", "branches", "if_true")}),
        frozenset({Atom("g1", "branches", "if_false")}),
    })


def test_an_iterating_decision_masks_rows_and_is_not_a_condition():
    """Both branches produce, each sparse, so nothing below a
    iterating decision is conditional on it — and it is not a dimension."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="g", type="gate", version=1, bindings={"value": _edge(("docs", "result"))}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": _edge(("g", "if_true"))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("g").wiring._runs_per_row_of is not None
    assert compiled.field(Ref("yes", "result")).condition == ALWAYS
    assert compiled.decisions == {}


def test_a_decision_on_a_node_whose_wiring_failed_is_not_one_a_run_makes():
    """A gate whose edge names an output its source lacks has failed wiring: it
    is not counted as a decision, and asking whether it runs per row says why."""
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        GraphNode(id="g", type="gate", version=1, bindings={"value": From("h.nope")}),
    ])

    assert compiled.decisions == {}
    assert compiled.node("g").state == "wiring_failed"
    assert compiled.node("g").wiring is None
    raised = compiled.node("g").cause
    assert raised.type == "unknown_ref_output"
