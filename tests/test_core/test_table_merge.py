"""Several tables or columns into one ``Table`` input are merged on their rows' ids as the graph runs, never refused when it is checked."""

from typing import Annotated

import polars as pl
from conductor import CompiledGraph, GraphNode, NodeRegistry, Param, Static, Table, run_sync
from conductor.graph.binding import From
from conductor.graph.model import Graph
from conductor.graph.receive import Broadcast, Merge, PerRow, RowsUnder, Whole
from conductor.metadata import Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.table import ROW_ID, merge
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node
from test_core.test_table import Num, Txt
from test_core.test_table_runs import CSV, Contacts, _leg, _registry


class Export(NodeDefinition):
    """«Til Excel»: takes whatever is wired in as one table, and hands it back."""

    id = "export"
    title = "Export"
    description = "d"
    category = "test"

    def run(self, table: Annotated[Table, Param(title="Table")]) -> Annotated[Table, Result(title="Sheet")]:
        return table.frame


class OnlyWithEmail(NodeDefinition):
    """A filter: keeps the contacts with an email, rows and ids as they were."""

    id = "with-email"
    title = "With email"
    description = "d"
    category = "test"

    def run(self, contacts: Annotated[Contacts, Param(title="Contacts")]) -> Annotated[Contacts, Result(title="Contacts")]:
        return contacts.frame.filter(pl.col("email").is_not_null())


def _registry_with(*extra: type[NodeDefinition]) -> NodeRegistry:
    registry = _registry()
    for node in (Export, OnlyWithEmail, *extra):
        registry.register(node)
    return registry


def _compiled(*nodes: GraphNode, registry: NodeRegistry | None = None) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry or _registry_with())


def _sheet(compiled: CompiledGraph, node: str = "excel") -> pl.DataFrame:
    assert compiled.is_runnable, compiled.problems
    return run_sync(compiled).state.results(compiled)[node]["result"].frame.drop(ROW_ID)


SEND = GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")})


def test_send_mail_per_contact_then_excel_has_name_email_and_sent():
    compiled = _compiled(CSV, SEND, GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "send.result")}))

    assert compiled.field(Ref("excel", "table")).receives == Merge((
        (Ref("csv", "result"), Broadcast()), (Ref("send", "result"), Whole()),
    ))
    assert compiled.problems == ()
    assert _sheet(compiled).to_dicts() == [
        {"name": "Ida", "email": "ida@x", "result": "sent to ida@x"},
        {"name": "Bo", "email": None, "result": "no address"},
        {"name": "Nina", "email": "nina@x", "result": "sent to nina@x"},
    ]


def test_a_filtered_table_merged_with_its_source_leaves_the_filtered_rows_blank():
    compiled = _compiled(
        CSV,
        GraphNode(id="keep", type="with-email", version=1, bindings={"contacts": From("csv.result")}),
        GraphNode(id="mail", type="send", version=1, bindings={"email": From("keep.result.email")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "mail.result")}),
    )

    assert compiled.problems == ()
    assert _sheet(compiled)["result"].to_list() == ["sent to ida@x", None, "sent to nina@x"]


def test_an_embedded_graph_given_the_whole_table_lands_its_result_on_the_contacts():
    registry = _registry_with()
    inner = embedded_graph_node("mailer", (
        GraphNode(id="keep", type="with-email", version=1),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("keep.result.email")}),
    ), registry, "Mailer")
    registry = registry.extended_with({"mailer": inner})
    compiled = _compiled(
        CSV,
        GraphNode(id="emb", type="mailer", version=1, bindings={"keep.contacts": From("csv.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "emb.send.result")}),
        registry=registry,
    )

    assert _sheet(compiled).to_dicts() == [
        {"name": "Ida", "email": "ida@x", "result": "sent to ida@x"},
        {"name": "Bo", "email": None, "result": None},
        {"name": "Nina", "email": "nina@x", "result": "sent to nina@x"},
    ]


def test_two_unrelated_tables_merge_into_blanks_and_the_author_is_told_while_drawing():
    other = GraphNode(id="other", type="csv", version=1, bindings={"text": Static("Ole,ole@x")})
    compiled = _compiled(CSV, other, GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "other.result")}))

    (told,) = compiled.problems
    assert (told.code, told.fatal, told.field, told.details) == ("no_rows_in_common", False, "table", {"source": "csv.result"})
    sheet = _sheet(compiled)
    assert sheet.height == 4
    assert sheet["name"].to_list() == ["Ida", "Bo", "Nina", None]
    assert sheet["name_other"].to_list() == [None, None, None, "Ole"]


def test_one_column_is_a_table_of_one_column_named_after_it():
    compiled = _compiled(CSV, SEND, GraphNode(id="excel", type="export", version=1, bindings={"table": From("send.result")}))

    assert _sheet(compiled).to_dicts() == [{"result": "sent to ida@x"}, {"result": "no address"}, {"result": "sent to nina@x"}]


def test_a_parent_value_is_copied_down_onto_each_row_under_it():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5|gamma:7")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result.vendor", "docs.result")}),
    )

    assert compiled.problems == ()
    assert _sheet(compiled).select("vendor", "result").to_dicts() == [
        {"vendor": "acme", "result": "acme:10,beta:5"},
        {"vendor": "beta", "result": "acme:10,beta:5"},
        {"vendor": "gamma", "result": "gamma:7"},
    ]


def test_a_table_input_on_a_node_running_per_document_gets_that_documents_rows():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5|gamma:7")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="check", type="check", version=1, bindings={"vendor": From("read.result.vendor")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result", "check.result")}),
    )

    assert compiled.field(Ref("excel", "table")).receives == Merge((
        (Ref("read", "result"), PerRow(Index("docs.result"))),
        (Ref("check", "result"), RowsUnder(Index("read.result", parent=Index("docs.result")), 1)),
    ))
    sheets = run_sync(compiled).state.results(compiled)["excel"]["result"]
    assert [sheet.frame.drop(ROW_ID).to_dicts() for sheet in sheets] == [
        [{"vendor": "acme", "amount": 10.0, "result": "acme ok"}, {"vendor": "beta", "amount": 5.0, "result": "beta ok"}],
        [{"vendor": "gamma", "amount": 7.0, "result": "gamma ok"}],
    ]


def test_columns_of_two_unrelated_tables_into_a_node_running_per_row_are_misaligned():
    other = GraphNode(id="other", type="csv", version=1, bindings={"text": Static("Ole,ole@x")})
    compiled = _compiled(CSV, other, GraphNode(id="route", type="route", version=1, bindings={"name": From("csv.result.name"), "email": From("other.result.email")}))

    assert [problem.code for problem in compiled.problems] == ["misaligned"]


def test_merge_matches_rows_on_their_ids_and_keeps_a_clashing_name_apart():
    left = pl.DataFrame({ROW_ID: ["c:0", "c:1"], "name": ["Ida", "Bo"]})
    right = pl.DataFrame({ROW_ID: ["c:1", "x:0"], "name": ["Bob", "Ole"]})

    assert merge([("csv", left, ROW_ID), ("crm", right, ROW_ID)]).to_dicts() == [
        {ROW_ID: "c:0", "name": "Ida", "name_crm": None},
        {ROW_ID: "c:1", "name": "Bo", "name_crm": "Bob"},
        {ROW_ID: "x:0", "name": None, "name_crm": "Ole"},
    ]


def test_a_single_value_into_a_table_input_is_refused():
    compiled = _compiled(
        GraphNode(id="t", type="send", version=1, bindings={"email": Static("a")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("t.result")}),
    )

    assert [problem.code for problem in compiled.problems] == ["type_mismatch"]



class Words(NodeDefinition):
    id = "words"
    title = "Words"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Words")]:
        return [Txt(word) for word in text.split(",") if word]


NamesSent = Table[("name", Txt), ("sent", Txt)]
NamesAsNumbers = Table[("name", Num)]


class Sheet(NodeDefinition):
    """A sheet that needs a name and a sent column."""

    id = "sheet"
    title = "Sheet"
    description = "d"
    category = "test"

    def run(self, table: Annotated[NamesSent, Param(title="Table")]) -> Annotated[Txt, Result(title="Rows")]:
        return Txt(str(table.frame.height))


class Scores(NodeDefinition):
    """A sheet whose name column is a number, which no contact's name reads as."""

    id = "scores"
    title = "Scores"
    description = "d"
    category = "test"

    def run(self, table: Annotated[NamesAsNumbers, Param(title="Table")]) -> Annotated[Txt, Result(title="Rows")]:
        return Txt(str(table.frame.height))


DOCS = GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5|gamma:7")})
READ = GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")})


def test_a_table_input_inside_an_embedded_graph_merges_its_inner_sources():
    registry = _registry_with()
    inner = embedded_graph_node("sheet-maker", (
        GraphNode(id="csv", type="csv", version=1, bindings={"text": Static("Ida,ida@x;Bo,")}),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "send.result")}),
    ), registry, "Sheet maker")
    compiled = _compiled(GraphNode(id="emb", type="sheet-maker", version=1), registry=registry.extended_with({"sheet-maker": inner}))

    assert _sheet(compiled, "emb/excel")["result"].to_list() == ["sent to ida@x", "no address"]


def test_a_parent_is_copied_down_onto_the_rows_of_every_child_and_keeps_a_row_with_none():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10|")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="words", type="words", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("docs.result", "read.result.vendor", "words.result")}),
        registry=_registry_with(Words),
    )
    sheet = _sheet(compiled)

    assert sheet.columns == ["result", "vendor", "result_words"]
    assert sheet.to_dicts() == [
        {"result": "acme:10", "vendor": "acme", "result_words": None},
        {"result": "acme:10", "vendor": None, "result_words": "acme:10"},
        {"result": "", "vendor": None, "result_words": None},
    ]


def test_a_clashing_column_takes_its_nodes_name():
    compiled = _compiled(
        CSV, SEND,
        GraphNode(id="mail", type="send", version=1, bindings={"email": From("csv.result.name")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("send.result", "mail.result")}),
    )

    assert _sheet(compiled).columns == ["result", "result_mail"]


def test_an_embedded_graphs_table_rows_carry_ids_in_so_nothing_is_told():
    registry = _registry_with()
    inner = embedded_graph_node("filter", (GraphNode(id="keep", type="with-email", version=1),), registry, "Filter")
    compiled = _compiled(
        CSV,
        GraphNode(id="emb", type="filter", version=1, bindings={"keep.contacts": From("csv.result")}),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("emb.keep.result.email")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "send.result")}),
        registry=registry.extended_with({"filter": inner}),
    )

    assert compiled.problems == ()
    assert _sheet(compiled)["result"].to_list() == ["sent to ida@x", None, "sent to nina@x"]


def test_two_tables_made_per_document_share_no_rows_and_the_author_is_told():
    compiled = _compiled(
        DOCS, READ,
        GraphNode(id="again", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result.vendor", "again.result.amount")}),
    )

    assert [problem.code for problem in compiled.problems] == ["no_rows_in_common"]


def test_a_declared_table_input_missing_a_column_is_refused_when_checked():
    compiled = _compiled(CSV, SEND, GraphNode(id="s", type="sheet", version=1, bindings={"table": From("csv.result", "send.result")}), registry=_registry_with(Sheet))

    (missing,) = compiled.problems
    assert (missing.code, missing.fatal, missing.field, missing.details) == ("table_column_missing", True, "table", {"column": "sent"})


def test_a_merged_table_that_wont_read_as_the_input_fails_that_node():
    compiled = _compiled(CSV, GraphNode(id="s", type="scores", version=1, bindings={"table": From("csv.result")}), registry=_registry_with(Scores))
    assert compiled.is_runnable, compiled.problems

    events = list(_events(compiled))
    (failed,) = [event for event in events if event.type == "node_error"]
    assert (failed.node_id, failed.cause.code) == ("s", "invalid_input")


def test_a_column_two_levels_down_is_received_under_the_nodes_row():
    compiled = _compiled(
        DOCS, READ,
        GraphNode(id="words", type="words", version=1, bindings={"text": From("read.result.vendor")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result", "words.result")}),
        registry=_registry_with(Words),
    )

    parts = dict(compiled.field(Ref("excel", "table")).receives.parts)
    assert parts[Ref("words", "result")] == RowsUnder(Index("words.result", parent=Index("read.result", parent=Index("docs.result"))), 1)


def _events(compiled: CompiledGraph):
    return _leg(compiled)
