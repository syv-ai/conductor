"""Several tables or columns into one ``Table`` input are merged on their rows' ids as the graph runs, never refused when it is checked."""

from typing import Annotated

import polars as pl
from conductor import (
    ROW_ID_COLUMN,
    SKIPPED,
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    NodeDefinition,
    NodeRegistry,
    Param,
    Ref,
    Result,
    Series,
    Static,
    Table,
    node,
    table_with_columns,
)
from conductor._internal.index import Index
from conductor._internal.table import MergeSource, merge_tables_on_row_ids
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node
from test_core.test_table import Num, Txt
from test_core.test_table_runs import CSV, Contacts, Invoices, _leg, _registry


@node("export", title="Export", description="d", category="test")
def export(table: Annotated[Table, Param(title="Table")]) -> Annotated[Table, Result(title="Sheet")]:
    """«Til Excel»: takes whatever is wired in as one table, and hands it back."""
    return table


@node("with-email", title="With email", description="d", category="test")
def only_with_email(contacts: Annotated[Contacts, Param(title="Contacts")]) -> Annotated[Contacts, Result(title="Contacts")]:
    """A filter: keeps the contacts with an email, rows and ids as they were."""
    return contacts.filter(pl.col("email").is_not_null())


def _registry_with(*extra: type[NodeDefinition]) -> NodeRegistry:
    registry = _registry()
    for graph_node in (export, only_with_email, *extra):
        registry = registry.extended_with([graph_node])
    return registry


def _compiled(*nodes: GraphNode, registry: NodeRegistry | None = None) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry or _registry_with())


def _sheet(compiled: CompiledGraph, graph_node: str = "excel") -> pl.DataFrame:
    assert compiled.is_runnable, compiled.problems
    return compiled.results(compiled.run_sync())[graph_node]["result"].drop(ROW_ID_COLUMN)


SEND = GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")})


def test_send_mail_per_contact_then_excel_has_name_email_and_sent():
    compiled = _compiled(CSV, SEND, GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "send.result")}))

    assert [(part.sources[0], part.receive_mode) for part in compiled.run_plan.nodes["excel"].inputs["table"].source_plans] == [
        ("csv.result", "same_value_every_row"), ("send.result", "entire_value"),
    ]
    assert compiled.problems == ()
    assert _sheet(compiled).to_dicts() == [
        {"name": "Ida", "email": "ida@x", "Sent": "sent to ida@x"},
        {"name": "Bo", "email": None, "Sent": "no address"},
        {"name": "Nina", "email": "nina@x", "Sent": "sent to nina@x"},
    ]


def test_a_filtered_table_merged_with_its_source_leaves_the_filtered_rows_blank():
    compiled = _compiled(
        CSV,
        GraphNode(id="keep", type="with-email", version=1, bindings={"contacts": From("csv.result")}),
        GraphNode(id="mail", type="send", version=1, bindings={"email": From("keep.result.email")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "mail.result")}),
    )

    assert compiled.problems == ()
    assert _sheet(compiled)["Sent"].to_list() == ["sent to ida@x", None, "sent to nina@x"]


def test_an_embedded_graph_given_the_whole_table_lands_its_result_on_the_contacts():
    registry = _registry_with()
    inner = embedded_graph_node("mailer", (
        GraphNode(id="keep", type="with-email", version=1),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("keep.result.email")}),
    ), registry, "Mailer")
    registry = registry.extended_with([inner])
    compiled = _compiled(
        CSV,
        GraphNode(id="emb", type="mailer", version=1, bindings={"contacts": From("csv.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "emb.result")}),
        registry=registry,
    )

    assert _sheet(compiled).to_dicts() == [
        {"name": "Ida", "email": "ida@x", "Sent": "sent to ida@x"},
        {"name": "Bo", "email": None, "Sent": None},
        {"name": "Nina", "email": "nina@x", "Sent": "sent to nina@x"},
    ]


def test_two_unrelated_tables_merge_into_blanks_and_the_author_is_told_while_drawing():
    other = GraphNode(id="other", type="csv", version=1, bindings={"text": Static("Ole,ole@x")})
    compiled = _compiled(CSV, other, GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "other.result")}))

    (told,) = compiled.problems
    assert (told.type, told.fatal, told.loc[1], told.ctx) == ("no_rows_in_common", False, "table", {"source": "csv.result"})
    sheet = _sheet(compiled)
    assert sheet.height == 4
    assert sheet["name"].to_list() == ["Ida", "Bo", "Nina", None]
    assert sheet["name_other"].to_list() == [None, None, None, "Ole"]


def test_one_column_is_a_table_of_one_column_named_after_it():
    compiled = _compiled(CSV, SEND, GraphNode(id="excel", type="export", version=1, bindings={"table": From("send.result")}))

    assert _sheet(compiled).to_dicts() == [{"Sent": "sent to ida@x"}, {"Sent": "no address"}, {"Sent": "sent to nina@x"}]


def test_a_parent_value_is_copied_down_onto_each_row_under_it():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5|gamma:7")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result.vendor", "docs.result")}),
    )

    assert compiled.problems == ()
    assert _sheet(compiled).select("vendor", "Documents").to_dicts() == [
        {"vendor": "acme", "Documents": "acme:10,beta:5"},
        {"vendor": "beta", "Documents": "acme:10,beta:5"},
        {"vendor": "gamma", "Documents": "gamma:7"},
    ]


def test_a_table_input_on_a_node_running_per_document_gets_that_documents_rows():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5|gamma:7")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="check", type="check", version=1, bindings={"vendor": From("read.result.vendor")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result", "check.result")}),
    )

    read, check = compiled.run_plan.nodes["excel"].inputs["table"].source_plans
    assert (read.sources[0], read.receive_mode, read.rows_of) == ("read.result", "value_at_row", "docs.result")
    assert (check.sources[0], check.receive_mode, check.rows_of, check.grouped_under) == ("check.result", "all_rows_under_row", "read.result", "docs.result")
    sheets = compiled.results(compiled.run_sync())["excel"]["result"]["values"]
    assert [sheet.drop(ROW_ID_COLUMN).to_dicts() for sheet in sheets] == [
        [{"vendor": "acme", "amount": 10.0, "Approved": "acme ok"}, {"vendor": "beta", "amount": 5.0, "Approved": "beta ok"}],
        [{"vendor": "gamma", "amount": 7.0, "Approved": "gamma ok"}],
    ]


def test_columns_of_two_unrelated_tables_into_a_node_running_per_row_are_misaligned():
    other = GraphNode(id="other", type="csv", version=1, bindings={"text": Static("Ole,ole@x")})
    compiled = _compiled(CSV, other, GraphNode(id="route", type="route", version=1, bindings={"name": From("csv.result.name"), "email": From("other.result.email")}))

    assert [make_problem.type for make_problem in compiled.problems] == ["misaligned"]


def test_merge_matches_rows_on_their_ids_and_keeps_a_clashing_name_apart():
    left = pl.DataFrame({ROW_ID_COLUMN: ["c:0", "c:1"], "name": ["Ida", "Bo"]})
    right = pl.DataFrame({ROW_ID_COLUMN: ["c:1", "x:0"], "name": ["Bob", "Ole"]})

    assert merge_tables_on_row_ids([MergeSource("csv", left, Index("c"), [(0,), (1,)], False), MergeSource("crm", right, Index("c"), [(1,), (2,)], False)]).to_dicts() == [
        {ROW_ID_COLUMN: "c:0", "name": "Ida", "name_crm": None},
        {ROW_ID_COLUMN: "c:1", "name": "Bo", "name_crm": "Bob"},
        {ROW_ID_COLUMN: "x:0", "name": None, "name_crm": "Ole"},
    ]


def test_merge_copies_a_parent_down_by_each_rows_path():
    docs = pl.DataFrame({ROW_ID_COLUMN: ["d:0", "d:1"], "doc": ["a", "b"]})
    lines = pl.DataFrame({ROW_ID_COLUMN: ["l:0.0", "l:0.1"], "line": ["x", "y"]})
    parent = Index("d")

    assert merge_tables_on_row_ids([MergeSource("docs", docs, parent, [(0,), (1,)], True), MergeSource("split", lines, Index("l", parent=parent), [(0, 0), (0, 1)], False)]).to_dicts() == [
        {ROW_ID_COLUMN: "l:0.0", "doc": "a", "line": "x"},
        {ROW_ID_COLUMN: "l:0.1", "doc": "a", "line": "y"},
    ]


def test_a_single_value_into_a_table_input_is_refused():
    compiled = _compiled(
        GraphNode(id="t", type="send", version=1, bindings={"email": Static("a")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("t.result")}),
    )

    assert [make_problem.type for make_problem in compiled.problems] == ["type_mismatch"]



@node("words", title="Words", description="d", category="test")
def words(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Words")]:
    return [Txt(word) for word in text.split(",") if word]


NamesSent = table_with_columns({"name": Txt, "sent": Txt})
NamesAsNumbers = table_with_columns({"name": Num})


@node("sheet", title="Sheet", description="d", category="test")
def sheet_node(table: Annotated[NamesSent, Param(title="Table")]) -> Annotated[Txt, Result(title="Rows")]:
    """A sheet that needs a name and a sent column."""
    return Txt(str(table.height))


@node("scores", title="Scores", description="d", category="test")
def scores(table: Annotated[NamesAsNumbers, Param(title="Table")]) -> Annotated[Txt, Result(title="Rows")]:
    """A sheet whose name column is a number, which no contact's name reads as."""
    return Txt(str(table.height))


DOCS = GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5|gamma:7")})
READ = GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")})


def test_a_table_input_inside_an_embedded_graph_merges_its_inner_sources():
    registry = _registry_with()
    inner = embedded_graph_node("sheet-maker", (
        GraphNode(id="csv", type="csv", version=1, bindings={"text": Static("Ida,ida@x;Bo,")}),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("csv.result.email")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "send.result")}),
    ), registry, "Sheet maker")
    compiled = _compiled(GraphNode(id="emb", type="sheet-maker", version=1), registry=registry.extended_with([inner]))

    assert _sheet(compiled, "emb/excel")["Sent"].to_list() == ["sent to ida@x", "no address"]


def test_a_parent_is_copied_down_onto_the_rows_of_every_child_and_has_no_row_of_its_own():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10|")}),
        GraphNode(id="read", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="words", type="words", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("docs.result", "read.result.vendor", "words.result")}),
        registry=_registry_with(words),
    )
    sheet = _sheet(compiled)

    assert sheet.columns == ["Documents", "vendor", "Words"]
    assert sheet.to_dicts() == [
        {"Documents": "acme:10", "vendor": "acme", "Words": None},
        {"Documents": "acme:10", "vendor": None, "Words": "acme:10"},
    ]


def test_a_clashing_column_takes_its_nodes_name():
    compiled = _compiled(
        CSV, SEND,
        GraphNode(id="mail", type="send", version=1, bindings={"email": From("csv.result.name")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("send.result", "mail.result")}),
    )

    assert _sheet(compiled).columns == ["Sent", "Sent_mail"]


def test_an_embedded_graphs_table_rows_carry_ids_in_so_nothing_is_told():
    registry = _registry_with()
    inner = embedded_graph_node("filter", (GraphNode(id="keep", type="with-email", version=1),), registry, "Filter")
    compiled = _compiled(
        CSV,
        GraphNode(id="emb", type="filter", version=1, bindings={"contacts": From("csv.result")}),
        GraphNode(id="send", type="send", version=1, bindings={"email": From("emb.result.email")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "send.result")}),
        registry=registry.extended_with([inner]),
    )

    assert compiled.problems == ()
    assert _sheet(compiled)["Sent"].to_list() == ["sent to ida@x", None, "sent to nina@x"]


def test_two_tables_made_per_document_share_no_rows_and_the_author_is_told():
    compiled = _compiled(
        DOCS, READ,
        GraphNode(id="again", type="structured", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result.vendor", "again.result.amount")}),
    )

    assert [make_problem.type for make_problem in compiled.problems] == ["no_rows_in_common"]


def test_a_declared_table_input_missing_a_column_is_refused_when_checked():
    compiled = _compiled(CSV, SEND, GraphNode(id="s", type="sheet", version=1, bindings={"table": From("csv.result", "send.result")}), registry=_registry_with(sheet_node))

    (missing,) = compiled.problems
    assert (missing.type, missing.fatal, missing.loc[1], missing.ctx) == ("table_column_missing", True, "table", {"column": "sent"})


def test_a_merged_table_that_wont_read_as_the_input_fails_that_node():
    compiled = _compiled(CSV, GraphNode(id="s", type="scores", version=1, bindings={"table": From("csv.result")}), registry=_registry_with(scores))
    assert compiled.is_runnable, compiled.problems

    events = list(_events(compiled))
    (failed,) = [event for event in events if event.type == "node_failed"]
    assert (failed.node_id, failed.cause.type) == ("s", "invalid_input")


def test_a_column_two_levels_down_is_received_under_the_nodes_row():
    compiled = _compiled(
        DOCS, READ,
        GraphNode(id="words", type="words", version=1, bindings={"text": From("read.result.vendor")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result", "words.result")}),
        registry=_registry_with(words),
    )

    from_words = next(part for part in compiled.run_plan.nodes["excel"].inputs["table"].source_plans if part.sources == (Ref("words", "result"),))
    assert (from_words.receive_mode, from_words.rows_of, from_words.grouped_under) == ("all_rows_under_row", "words.result", "docs.result")


def _events(compiled: CompiledGraph):
    return _leg(compiled)


@node("paragraphs", title="Paragraphs", description="d", category="test")
def splits(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Paragraphs")]:
    """Splits on ``;``, leaving out empty parts."""
    return [Txt(part) for part in text.split(";") if part]


@node("labelled", title="Labelled", description="d", category="test")
def labelled(label: Annotated[Txt, Param(title="Label", widget=Textarea())] = Txt(""), texts: Annotated[Series[Txt], Param(title="Texts")] = (),
) -> Annotated[Txt, Result(title="Said")]:
    """A value and a column read together: what it says shows which rows it got."""
    return Txt(f"{label}={','.join(texts)}")


@node("labelled-table", title="Labelled table", description="d", category="test")
def labelled_table(label: Annotated[Txt, Param(title="Label", widget=Textarea())], table: Annotated[Table, Param(title="Table")],
) -> Annotated[Txt, Result(title="Said")]:
    return Txt(f"{label}={','.join(table[table.columns[-1]].to_list())}")


@node("two-columns", title="Two columns", description="d", category="test")
def two_columns(a: Annotated[Series[Txt], Param(title="A")], b: Annotated[Series[Txt], Param(title="B")]) -> Annotated[Txt, Result(title="Said")]:
    return Txt(f"{','.join(a)}+{','.join(b)}")


def _three_levels(text: str) -> tuple[GraphNode, GraphNode, GraphNode]:
    """Documents split on ``|``, paragraphs under them on ``;``, lines under those on ``,``."""
    return (
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(text)}),
        GraphNode(id="paras", type="paragraphs", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="lines", type="words", version=1, bindings={"text": From("paras.result")}),
    )


def _deep_registry() -> NodeRegistry:
    return _registry_with(words, splits, labelled, labelled_table, two_columns)


def _said(compiled: CompiledGraph, graph_node: str = "n") -> list[str]:
    assert compiled.is_runnable, compiled.problems
    return compiled.results(compiled.run_sync())[graph_node]["result"]["values"]


def test_wiring_the_parent_as_well_adds_no_row():
    docs = GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10,beta:5||gamma:7")})
    alone = _compiled(docs, READ, GraphNode(id="excel", type="export", version=1, bindings={"table": From("read.result.vendor")}))
    beside = _compiled(docs, READ, GraphNode(id="excel", type="export", version=1, bindings={"table": From("docs.result", "read.result.vendor")}))

    assert _sheet(alone)["vendor"].to_list() == _sheet(beside)["vendor"].to_list() == ["acme", "beta", "gamma"]


def test_a_merge_three_levels_deep_copies_every_level_down_and_a_paragraph_with_no_lines_has_no_row():
    compiled = _compiled(
        *_three_levels("a,b;;c|d"),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("docs.result", "paras.result", "lines.result")}),
        registry=_deep_registry(),
    )
    sheet = _sheet(compiled)

    assert sheet.columns == ["Documents", "Paragraphs", "Words"]
    assert sheet.rows() == [
        ("a,b;;c", "a,b", "a"),
        ("a,b;;c", "a,b", "b"),
        ("a,b;;c", "c", "c"),
        ("d", "d", "d"),
    ]


def test_parents_with_nothing_under_them_give_a_table_of_no_rows_with_every_column():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("|")}),
        READ,
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("docs.result", "read.result.vendor", "read.result.amount")}),
    )
    sheet = _sheet(compiled)

    assert (sheet.height, sheet.columns) == (0, ["Documents", "vendor", "amount"])


def test_a_parent_column_beside_a_table_made_per_row_of_it_is_copied_down():
    compiled = _compiled(DOCS, READ, GraphNode(id="excel", type="export", version=1, bindings={"table": From("docs.result", "read.result")}))
    sheets = compiled.results(compiled.run_sync())["excel"]["result"]["values"]

    assert [sheet.drop(ROW_ID_COLUMN).to_dicts() for sheet in sheets] == [
        [{"Documents": "acme:10,beta:5", "vendor": "acme", "amount": 10.0}, {"Documents": "acme:10,beta:5", "vendor": "beta", "amount": 5.0}],
        [{"Documents": "gamma:7", "vendor": "gamma", "amount": 7.0}],
    ]


def test_a_child_table_a_branch_skipped_gives_no_rows_and_the_sheet_keeps_its_columns():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1, bindings={"text": Static("acme:10|skip")}),
        GraphNode(id="read", type="skip-or-read", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("docs.result", "read.result")}),
        registry=_registry_with(skip_or_read),
    )
    sheets = compiled.results(compiled.run_sync())["excel"]["result"]["values"]

    assert [sheet.height for sheet in sheets] == [1, 0]
    assert sheets[1].columns == [ROW_ID_COLUMN, "Documents", "vendor", "amount"]


def test_a_series_two_levels_down_with_nothing_else_runs_per_row_of_the_level_above_it():
    compiled = _compiled(*_three_levels("a,b;c|d"), GraphNode(id="n", type="labelled", version=1, bindings={"texts": From("lines.result")}), registry=_deep_registry())

    assert compiled.run_plan.nodes["n"].runs_per_row_of == "paras.result"
    assert _said(compiled) == ["=a,b", "=c", "=d"]


def test_a_value_on_a_document_with_a_series_of_lines_gets_every_line_of_the_document_flat():
    compiled = _compiled(*_three_levels("a,b;c|d"), GraphNode(id="n", type="labelled", version=1, bindings={"label": From("docs.result"), "texts": From("lines.result")}), registry=_deep_registry())

    assert compiled.run_plan.nodes["n"].runs_per_row_of == "docs.result"
    assert compiled.run_plan.nodes["n"].inputs["texts"].grouped_under == "docs.result"
    assert _said(compiled) == ["a,b;c=a,b,c", "d=d"]


def test_a_table_of_lines_with_a_value_on_a_document_gets_every_line_of_the_document():
    compiled = _compiled(*_three_levels("a,b;c|d"), GraphNode(id="n", type="labelled-table", version=1, bindings={"label": From("docs.result"), "table": From("lines.result")}), registry=_deep_registry())

    assert _said(compiled) == ["a,b;c=a,b,c", "d=d"]


def test_a_value_on_a_paragraph_keeps_a_series_of_lines_to_that_paragraph():
    compiled = _compiled(*_three_levels("a,b;c|d"), GraphNode(id="n", type="labelled", version=1, bindings={"label": From("paras.result"), "texts": From("lines.result")}), registry=_deep_registry())

    assert _said(compiled) == ["a,b=a,b", "c=c", "d=d"]


def test_a_document_with_no_paragraphs_runs_with_an_empty_series():
    compiled = _compiled(*_three_levels("a|;|b"), GraphNode(id="n", type="labelled", version=1, bindings={"label": From("docs.result"), "texts": From("lines.result")}), registry=_deep_registry())

    assert _said(compiled) == ["a=a", ";=", "b=b"]


def test_two_series_at_different_depths_are_both_read_per_row_of_the_shallowest_level_above_them():
    compiled = _compiled(*_three_levels("a,b;c|d"), GraphNode(id="n", type="two-columns", version=1, bindings={"a": From("paras.result"), "b": From("lines.result")}), registry=_deep_registry())

    assert compiled.run_plan.nodes["n"].runs_per_row_of == "docs.result"
    assert _said(compiled) == ["a,b,c+a,b,c", "d+d"]


def test_two_series_under_unrelated_rows_are_misaligned():
    compiled = _compiled(
        *_three_levels("a|b"),
        GraphNode(id="again", type="paragraphs", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="lines2", type="words", version=1, bindings={"text": From("again.result")}),
        GraphNode(id="n", type="two-columns", version=1, bindings={"a": From("lines.result"), "b": From("lines2.result")}),
        registry=_deep_registry(),
    )

    assert [make_problem.type for make_problem in compiled.problems] == ["misaligned"]


@node("skip-or-read", title="Skip or read", description="d", category="test")
def skip_or_read(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Invoices, Result(title="Invoices")]:
    """A branch per document: the word ``skip`` takes it out."""
    if text == "skip":
        return SKIPPED
    return pl.DataFrame({"vendor": ["acme"], "amount": [10.0]})


@node("picks", title="Picks", description="d", category="test")
def picks(contacts: Annotated[Contacts, Param(title="Contacts")]) -> Annotated[Contacts, Result(title="Contacts")]:
    """A select that names columns, the natural way to lose ``__row``."""
    return contacts.select("name", "email")


@node("empties", title="Empties", description="d", category="test")
def empty(contacts: Annotated[Contacts, Param(title="Contacts")]) -> Annotated[Contacts, Result(title="Contacts")]:
    return contacts.select("name", "email").clear()


def _rebuilt(node_type: str = "picks") -> tuple[GraphNode, GraphNode]:
    return (
        GraphNode(id="keep", type=node_type, version=1, bindings={"contacts": From("csv.result")}),
        GraphNode(id="mail", type="send", version=1, bindings={"email": From("keep.result.email")}),
    )


def test_a_table_that_lost_every_rows_id_fails_when_merged_with_the_table_it_was_built_from():
    compiled = _compiled(CSV, *_rebuilt(), GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "mail.result")}), registry=_registry_with(picks))

    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert (state.status, state.failure.type, state.failure.loc[0]) == ("failed", "row_ids_lost", "keep")
    assert "mail" not in compiled.results(state)


def test_a_table_built_from_another_used_on_its_own_gets_fresh_ids():
    compiled = _compiled(CSV, *_rebuilt(), GraphNode(id="excel", type="export", version=1, bindings={"table": From("keep.result", "mail.result")}), registry=_registry_with(picks))

    assert _sheet(compiled)["Sent"].to_list() == ["sent to ida@x", "no address", "sent to nina@x"]


def test_a_table_with_no_ids_merged_with_an_unrelated_table_is_not_stopped():
    other = GraphNode(id="other", type="csv", version=1, bindings={"text": Static("Ole,ole@x")})
    compiled = _compiled(
        CSV, other, GraphNode(id="keep", type="picks", version=1, bindings={"contacts": From("other.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "keep.result")}), registry=_registry_with(picks),
    )

    assert _sheet(compiled).height == 4


def test_a_table_of_no_rows_has_lost_no_ids():
    compiled = _compiled(CSV, *_rebuilt("empties"), GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "mail.result")}), registry=_registry_with(empty))

    assert compiled.run_sync().status == "complete"


def test_a_table_merged_with_what_it_was_built_from_that_keeps_its_ids_passes():
    compiled = _compiled(
        CSV, GraphNode(id="keep", type="with-email", version=1, bindings={"contacts": From("csv.result")}),
        GraphNode(id="excel", type="export", version=1, bindings={"table": From("csv.result", "keep.result")}),
    )

    assert Ref("keep", "result") in compiled._row_ids_must_survive
    assert compiled.run_sync().status == "complete"
