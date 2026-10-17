"""A table output offers each of its columns as a field of its own, a series on rows the table output owns."""

from dataclasses import dataclass
from typing import Annotated

from conductor import CompiledGraph, GraphNode, Static, Table
from conductor.graph.binding import From
from conductor.graph.model import Graph
from conductor.metadata import Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series

from test_core.test_table import Num, Txt
from test_core.test_table_runs import CSV, Contacts, Text, _registry


@dataclass(frozen=True)
class Clashing:
    rows: Annotated[Contacts, Result(title="Rows")]
    clash: Annotated[Txt, Result(title="Clash")]


class Clash(NodeDefinition):
    """Declares an output named like one of its table's columns would be offered."""

    id = "clash"
    title = "Clash"
    description = "d"
    category = "test"

    def run(self, text: Text = Txt("")) -> Clashing:
        raise NotImplementedError

    @classmethod
    def compute_outputs(cls, declared, values, arriving):  # noqa: ARG003
        return (*declared[:1], declared[1].model_copy(update={"name": "rows.email"}))


class Bare(NodeDefinition):
    id = "bare"
    title = "Bare"
    description = "d"
    category = "test"

    def run(self, text: Text = Txt("")) -> Annotated[Table, Result(title="Anything")]:
        raise NotImplementedError


def _compiled(*nodes: GraphNode, extra: tuple[type[NodeDefinition], ...] = ()) -> CompiledGraph:
    registry = _registry()
    for node in extra:
        registry.register(node)
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry)


def test_a_column_is_a_series_on_rows_the_table_output_owns():
    compiled = _compiled(CSV)
    email = compiled.field(Ref("csv", "result.email"))

    assert email.type is Series[Txt] and email.index == Index("csv.result")
    assert email._view_of == (Ref("csv", "result"), "email")
    assert compiled.field(Ref("csv", "result"))._makes_rows == Index("csv.result")
    assert compiled.field(Ref("csv", "result")).index is None


def test_a_column_wired_into_a_single_value_input_runs_the_node_per_row_of_the_table():
    compiled = _compiled(CSV, GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")}))

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("send").runs_per_row_of == Index("csv.result")


def test_a_table_per_document_offers_columns_on_rows_under_each_document():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("a:1|b:2")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="check", type="check", version=1, bindings={"vendor": From("read.result.vendor")}),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("read", "result")).index == Index("docs.result")
    assert compiled.node("check").runs_per_row_of == Index("read.result", parent=Index("docs.result"))
    assert compiled.field(Ref("read", "result.amount")).type is Series[Num]


def test_a_column_appears_exactly_when_its_table_does():
    compiled = _compiled(CSV)

    assert compiled.field(Ref("csv", "result.name")).condition == compiled.field(Ref("csv", "result")).condition


def test_a_bare_table_offers_no_columns():
    compiled = _compiled(GraphNode(id="b", type="bare", version=1), extra=(Bare,))

    assert compiled.is_runnable, compiled.problems
    assert set(compiled.node("b")._fields) == {"text", "result"}


def test_an_output_named_like_a_column_is_refused():
    compiled = _compiled(GraphNode(id="c", type="clash", version=1), extra=(Clash,))

    assert [(p.code, p.field) for p in compiled.problems] == [("view_name_taken", "rows.email")]


class ManyTables(NodeDefinition):
    id = "many-tables"
    title = "Many tables"
    description = "d"
    category = "test"

    def run(self, text: Text = Txt("")) -> Annotated[Series[Contacts], Result(title="Tables")]:
        raise NotImplementedError


def test_a_series_of_tables_is_refused_since_a_table_is_already_many_rows():
    compiled = _compiled(GraphNode(id="m", type="many-tables", version=1), extra=(ManyTables,))

    assert [(p.code, p.field, p.fatal) for p in compiled.problems] == [("series_of_tables", "result", True)]
