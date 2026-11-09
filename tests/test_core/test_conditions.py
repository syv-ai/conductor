"""The condition under which an output appears, derived from choice and edges."""

from dataclasses import dataclass
from typing import Annotated, Any

from conductor import (
    ALWAYS,
    SKIPPED,
    ArrivingType,
    BranchTaken,
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


@node("holder", title="Text", description="d", category="test")
def holder(value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return value


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


def single_outputs(**arriving: ArrivingType):
    series = arriving.get("values")
    dtype = series if series is not None else Any
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
    registry = NodeRegistry([gate, holder, upper_node, docs, single])
    return CompiledGraph.from_graph(Graph(nodes=nodes), registry)


def test_an_output_nothing_gates_appears_always():
    compiled = _compiled([GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")})])

    assert compiled.field(Ref("h", "result")).condition == ALWAYS


def test_a_branch_appears_when_its_decision_went_that_way():
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        GraphNode(id="g", type="gate", version=1, bindings={"value": From("h.result")}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": From("g.if_true")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("g", "if_true")).condition == frozenset({frozenset({BranchTaken("g", "branches", "if_true")})})
    assert compiled.field(Ref("yes", "result")).condition == frozenset({frozenset({BranchTaken("g", "branches", "if_true")})})
    assert compiled.decisions == {"g": {"branches": ("if_true", "if_false")}}


def test_a_merge_adds_an_alternative():
    """A `Series[X]` input drops the skipped edges, so past the
    merge the output appears when either branch fired."""
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        GraphNode(id="g", type="gate", version=1, bindings={"value": From("h.result")}),
        GraphNode(id="a", type="upper", version=1, bindings={"text": From("g.if_true")}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("g.if_false")}),
        GraphNode(id="m", type="single", version=1, bindings={"values": From("a.result", "b.result")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("m", "result")).condition == frozenset({
        frozenset({BranchTaken("g", "branches", "if_true")}), frozenset({BranchTaken("g", "branches", "if_false")}),
    })


def test_two_decisions_in_series_conjoin_and_a_contradiction_is_dropped():
    compiled = _compiled([
        GraphNode(id="h", type="holder", version=1, bindings={"value": Static("x")}),
        GraphNode(id="g1", type="gate", version=1, bindings={"value": From("h.result")}),
        GraphNode(id="g2", type="gate", version=1, bindings={"value": From("g1.if_true")}),
        GraphNode(id="both", type="single", version=1, bindings={"values": From("g2.if_true", "g1.if_false")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("g2", "if_true")).condition == frozenset({
        frozenset({BranchTaken("g1", "branches", "if_true"), BranchTaken("g2", "branches", "if_true")}),
    })
    assert compiled.field(Ref("both", "result")).condition == frozenset({
        frozenset({BranchTaken("g1", "branches", "if_true"), BranchTaken("g2", "branches", "if_true")}),
        frozenset({BranchTaken("g1", "branches", "if_false")}),
    })


def test_an_iterating_decision_masks_rows_and_is_not_a_condition():
    """Both branches produce, each sparse, so nothing below a
    iterating decision is conditional on it — and it is not a dimension."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="g", type="gate", version=1, bindings={"value": From("docs.result")}),
        GraphNode(id="yes", type="upper", version=1, bindings={"text": From("g.if_true")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["g"].runs_per_row_of is not None
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
    assert compiled.node("g")._fields is None
    raised = compiled.node("g").cause
    assert raised.type == "unknown_ref_output"


def _per_row_branches():
    @dataclass(frozen=True)
    class Branched:
        yes: Annotated[str | None, Result(exclusive_group="g")]
        no: Annotated[str | None, Result(exclusive_group="g")]

    @node("branch_on_each_row", category="test")
    def branch_on_each_row(value: str) -> Branched:
        return Branched(yes=value, no=SKIPPED) if value == "a" else Branched(yes=SKIPPED, no=value)

    @node("rows", category="test")
    def rows() -> Series[str]:
        return ["a", "b"]

    @node("split_row", category="test")
    def split_row(value: str) -> Series[str]:
        return [f"{value}1", f"{value}2"]

    @node("pair_up", category="test")
    def pair_up(cell: str, up: str) -> str:
        return cell + up

    @node("both_series", category="test")
    def both_series(xs: Series[str], ys: Series[str]) -> str:
        return f"{xs}|{ys}"

    return [branch_on_each_row, rows, split_row, pair_up, both_series]


def _types_of_problems(*nodes):
    definitions = _per_row_branches()
    by_id = {definition.id: definition for definition in definitions}
    built = Graph(nodes=[by_id[kind].graph_node(node_id, **bindings) for kind, node_id, bindings in nodes])
    compiled = CompiledGraph.from_graph(built, NodeRegistry(definitions))
    return compiled, [problem.type for problem in compiled.problems]


def test_a_per_row_decision_read_from_a_row_above_still_keeps_two_branches_apart():
    _, problems = _types_of_problems(
        ("rows", "s", {}), ("branch_on_each_row", "d", {"value": From("s.result")}), ("split_row", "sp", {"value": From("s.result")}),
        ("pair_up", "ua", {"cell": From("sp.result"), "up": From("d.yes")}), ("pair_up", "ub", {"cell": From("sp.result"), "up": From("d.no")}),
        ("both_series", "z", {"xs": From("ua.result"), "ys": From("ub.result")}),
    )
    assert problems == ["series_inputs_never_share_a_row"]


def test_a_cell_of_several_sources_carries_the_per_row_decisions_it_reads():
    _, problems = _types_of_problems(
        ("rows", "s", {}), ("branch_on_each_row", "d", {"value": From("s.result")}),
        ("pair_up", "ua", {"cell": From("s.result"), "up": From("d.yes")}), ("pair_up", "ub", {"cell": From("s.result"), "up": From("d.yes", "d.no")}),
        ("both_series", "z", {"xs": From("ua.result"), "ys": From("ub.result")}),
    )
    assert problems == []


def test_a_field_s_public_condition_holds_only_decisions_on_nodes_that_run_once():
    compiled, problems = _types_of_problems(("rows", "s", {}), ("branch_on_each_row", "d", {"value": From("s.result")}))
    assert problems == [] and compiled.field("d.yes").condition == ALWAYS


def test_a_long_chain_of_decisions_merged_per_row_stays_small():
    definitions = _per_row_branches()
    by_id = {definition.id: definition for definition in definitions}
    nodes = [by_id["rows"].graph_node("s"), by_id["branch_on_each_row"].graph_node("d0", value=From("s.result"))]
    nodes += [by_id["branch_on_each_row"].graph_node(f"d{i}", value=From(f"d{i-1}.yes", f"d{i-1}.no")) for i in range(1, 40)]
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry(definitions))
    assert compiled.is_runnable and len(compiled.field("d39.yes")._condition_with_rows) <= 32
