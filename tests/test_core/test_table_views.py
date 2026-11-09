"""A table output offers each of its columns as a field of its own, a series on rows the table output owns."""

from dataclasses import dataclass
from typing import Annotated

from conductor import (
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    NodeDefinition,
    Output,
    Ref,
    Result,
    Series,
    Static,
    Table,
    node,
)

from test_core.test_table import Num, Txt
from test_core.test_table_runs import CSV, Contacts, Text, _registry


@dataclass(frozen=True)
class Clashing:
    rows: Annotated[Contacts, Result(title="Rows")]
    clash: Annotated[Txt, Result(title="Clash")]


def clash_node_outputs():
    return (Output("rows", Contacts, title="Rows"), Output("rows.email", Txt, title="Clash"))

@node("clash", title="Clash", description="d", category="test", compute_outputs=clash_node_outputs)
def clash_node(text: Text = Txt("")) -> Clashing:
    """Declares an output named like one of its table's columns would be offered."""
    raise NotImplementedError


@node("bare", title="Bare", description="d", category="test")
def bare(text: Text = Txt("")) -> Annotated[Table, Result(title="Anything")]:
    raise NotImplementedError


def _compiled(*nodes: GraphNode, extra: tuple[type[NodeDefinition], ...] = ()) -> CompiledGraph:
    registry = _registry().extended_with(extra)
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry)


def test_a_column_is_a_series_on_rows_the_table_output_owns():
    compiled = _compiled(CSV)
    email = compiled.field(Ref("csv", "result.email"))

    assert email.type is Txt and email.many
    assert compiled.run_plan.nodes["csv"].makes_rows == (Ref("csv", "result"),)


def test_a_column_wired_into_a_single_value_input_runs_the_node_per_row_of_the_table():
    compiled = _compiled(CSV, GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")}))

    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["send"].runs_per_row_of == "csv.result"


def test_a_table_per_document_offers_columns_on_rows_under_each_document():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("a:1|b:2")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="check", type="check", version=1, bindings={"vendor": From("read.result.vendor")}),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["read"].runs_per_row_of == "docs.result"
    assert compiled.run_plan.nodes["check"].runs_per_row_of == "read.result"
    assert compiled.field(Ref("read", "result.amount")).type is Num and compiled.field(Ref("read", "result.amount")).many


def test_a_column_appears_exactly_when_its_table_does():
    compiled = _compiled(CSV)

    assert compiled.field(Ref("csv", "result.name")).condition == compiled.field(Ref("csv", "result")).condition


def test_a_bare_table_offers_no_columns():
    compiled = _compiled(GraphNode(id="b", type="bare", version=1), extra=(bare,))

    assert compiled.is_runnable, compiled.problems
    assert set(compiled.node("b")._wiring.fields) == {"text", "result"}


def test_an_output_named_like_a_column_is_refused():
    compiled = _compiled(GraphNode(id="c", type="clash", version=1), extra=(clash_node,))

    assert [(p.type, p.loc[1]) for p in compiled.problems] == [("view_name_taken", "rows.email")]


@node("many-tables", title="Many tables", description="d", category="test")
def many_tables(text: Text = Txt("")) -> Annotated[Series[Contacts], Result(title="Tables")]:
    raise NotImplementedError


def test_a_series_of_tables_is_refused_since_a_table_is_already_many_rows():
    compiled = _compiled(GraphNode(id="m", type="many-tables", version=1), extra=(many_tables,))

    assert [(p.type, p.loc[1], p.fatal) for p in compiled.problems] == [("series_of_tables", "result", True)]
