"""Several tables or columns into one ``Table`` input are merged on their rows' ids as the graph runs, never refused when it is checked."""

from typing import Annotated

import polars as pl
from conductor import (
    ROW_ID_COLUMN,
    SKIPPED,
    CompiledGraph,
    NodeDefinition,
    NodeRegistry,
    Param,
    Ref,
    Result,
    Series,
    Table,
    node,
    table_with_columns,
)
from conductor._internal.graph.binding import From, Static
from conductor._internal.index import Index
from conductor._internal.table import MergeSource, merge_tables_on_row_ids
from conductor.widgets import Textarea
from helpers import Placed, events_of, placed, record, with_added

from test_core.embedded import embedded_graph_node
from test_core.test_table import Num, Txt
from test_core.test_table_runs import CSV, Contacts, Invoices, _registry


@node("export", title="Export", description="d", category="test")
def export(table: Annotated[Table, Param(title="Table")]) -> Annotated[Table, Result(title="Sheet")]:
    """«Til Excel»: takes whatever is wired in as one table, and hands it back."""
    return table


@node("with-email", title="With email", description="d", category="test")
def only_with_email(contacts: Annotated[Contacts, Param(title="Contacts")]) -> Annotated[Contacts, Result(title="Contacts")]:
    """A filter: keeps the contacts with an email, rows and ids as they were."""
    return contacts.filter(pl.col("email").is_not_null())


def _registry_with(*extra: type[NodeDefinition]) -> NodeRegistry:
    return with_added(_registry(), export, only_with_email, *extra)


def _compiled(*nodes: Placed, registry: NodeRegistry | None = None) -> CompiledGraph:
    return CompiledGraph.from_graph(record(*list(nodes)), registry or _registry_with())


def _sheet(compiled: CompiledGraph, graph_node: str = "excel") -> pl.DataFrame:
    assert compiled.is_runnable, compiled.problems
    return compiled.results(compiled.run_sync())[graph_node]["result"].drop(ROW_ID_COLUMN)


SEND = placed("send", "send", email=From("csv.result.email"))


def test_send_mail_per_contact_then_excel_has_name_email_and_sent():
    compiled = _compiled(CSV, SEND, placed("excel", "export", table=From("csv.result", "send.result")))

    assert [(part.sources[0], part.receive_mode) for part in compiled.run_plan.nodes["excel"].inputs["table"].source_reads] == [
        ("csv.result", "same_value_every_row"), ("send.result", "entire_value"),
    ]
    assert compiled.problems == ()
    assert _sheet(compiled).to_dicts() == [
        {"name": "Ida", "email": "ida@x", "result": "sent to ida@x"},
        {"name": "Bo", "email": None, "result": "no address"},
        {"name": "Nina", "email": "nina@x", "result": "sent to nina@x"},
    ]


def test_a_filtered_table_merged_with_its_source_leaves_the_filtered_rows_blank():
    compiled = _compiled(
        CSV,
        placed("keep", "with-email", contacts=From("csv.result")),
        placed("mail", "send", email=From("keep.result.email")),
        placed("excel", "export", table=From("csv.result", "mail.result")),
    )

    assert compiled.problems == ()
    assert _sheet(compiled)["result"].to_list() == ["sent to ida@x", None, "sent to nina@x"]


def test_an_embedded_graph_given_the_whole_table_lands_its_result_on_the_contacts():
    registry = _registry_with()
    inner = embedded_graph_node("mailer", (
        placed("keep", "with-email"),
        placed("send", "send", email=From("keep.result.email")),
    ), registry, "Mailer")
    registry = with_added(registry, inner)
    compiled = _compiled(
        CSV,
        placed("emb", "mailer", contacts=From("csv.result")),
        placed("excel", "export", table=From("csv.result", "emb.result")),
        registry=registry,
    )

    assert _sheet(compiled).to_dicts() == [
        {"name": "Ida", "email": "ida@x", "result": "sent to ida@x"},
        {"name": "Bo", "email": None, "result": None},
        {"name": "Nina", "email": "nina@x", "result": "sent to nina@x"},
    ]


def test_two_unrelated_tables_are_refused_when_checked_and_told_to_join_on_a_key():
    other = placed("other", "csv", text=Static("Ole,ole@x"))
    compiled = _compiled(CSV, other, placed("excel", "export", table=From("csv.result", "other.result")))

    (told,) = compiled.problems
    assert (told.type, told.fatal, told.loc[1], told.ctx) == ("no_rows_in_common", True, "table", {"source": "csv.result"})
    assert "joins them on a key column" in told.msg
    assert not compiled.is_runnable


def test_one_column_is_a_table_of_one_column_named_after_it():
    compiled = _compiled(CSV, SEND, placed("excel", "export", table=From("send.result")))

    assert _sheet(compiled).to_dicts() == [{"result": "sent to ida@x"}, {"result": "no address"}, {"result": "sent to nina@x"}]


def test_a_parent_value_is_copied_down_onto_each_row_under_it():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10,beta:5|gamma:7")),
        placed("read", "structured", text=From("docs.result")),
        placed("excel", "export", table=From("read.result.vendor", "docs.result")),
    )

    assert compiled.problems == ()
    assert _sheet(compiled).select("vendor", "result").to_dicts() == [
        {"vendor": "acme", "result": "acme:10,beta:5"},
        {"vendor": "beta", "result": "acme:10,beta:5"},
        {"vendor": "gamma", "result": "gamma:7"},
    ]


def test_a_table_input_on_a_node_running_per_document_gets_that_documents_rows():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10,beta:5|gamma:7")),
        placed("read", "structured", text=From("docs.result")),
        placed("check", "check", vendor=From("read.result.vendor")),
        placed("excel", "export", table=From("read.result", "check.result")),
    )

    read, check = compiled.run_plan.nodes["excel"].inputs["table"].source_reads
    assert (read.sources[0], read.receive_mode, read.rows_of) == ("read.result", "value_at_row", "docs.result")
    assert (check.sources[0], check.receive_mode, check.rows_of, check.grouped_under) == ("check.result", "all_rows_under_row", "read.result", "docs.result")
    sheets = compiled.results(compiled.run_sync())["excel"]["result"]["values"]
    assert [sheet.drop(ROW_ID_COLUMN).to_dicts() for sheet in sheets] == [
        [{"vendor": "acme", "amount": 10.0, "result": "acme ok"}, {"vendor": "beta", "amount": 5.0, "result": "beta ok"}],
        [{"vendor": "gamma", "amount": 7.0, "result": "gamma ok"}],
    ]


def test_columns_of_two_unrelated_tables_into_a_node_running_per_row_are_misaligned():
    other = placed("other", "csv", text=Static("Ole,ole@x"))
    compiled = _compiled(CSV, other, placed("route", "route", name=From("csv.result.name"), email=From("other.result.email")))

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
        placed("t", "send", email=Static("a")),
        placed("excel", "export", table=From("t.result")),
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


DOCS = placed("docs", "docs", text=Static("acme:10,beta:5|gamma:7"))
READ = placed("read", "structured", text=From("docs.result"))


def test_a_table_input_inside_an_embedded_graph_merges_its_inner_sources():
    registry = _registry_with()
    inner = embedded_graph_node("sheet-maker", (
        placed("csv", "csv", text=Static("Ida,ida@x;Bo,")),
        placed("send", "send", email=From("csv.result.email")),
        placed("excel", "export", table=From("csv.result", "send.result")),
    ), registry, "Sheet maker")
    compiled = _compiled(placed("emb", "sheet-maker"), registry=with_added(registry, inner))

    assert _sheet(compiled, "emb/excel")["result"].to_list() == ["sent to ida@x", "no address"]


def test_a_parent_is_copied_down_onto_the_rows_of_every_child_and_has_no_row_of_its_own():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10|")),
        placed("read", "structured", text=From("docs.result")),
        placed("words", "words", text=From("docs.result")),
        placed("excel", "export", table=From("docs.result", "read.result.vendor", "words.result")),
        registry=_registry_with(words),
    )
    sheet = _sheet(compiled)

    assert sheet.columns == ["result", "vendor", "result_2"]
    assert sheet.to_dicts() == [
        {"result": "acme:10", "vendor": "acme", "result_2": None},
        {"result": "acme:10", "vendor": None, "result_2": "acme:10"},
    ]


def test_a_clashing_column_takes_the_name_of_its_output():
    compiled = _compiled(
        CSV, SEND,
        placed("mail", "send", email=From("csv.result.name")),
        placed("excel", "export", table=From("send.result", "mail.result")),
    )

    assert _sheet(compiled).columns == ["result", "result_2"]


def test_an_embedded_graphs_table_rows_carry_ids_in_so_nothing_is_told():
    registry = _registry_with()
    inner = embedded_graph_node("filter", (placed("keep", "with-email"),), registry, "Filter")
    compiled = _compiled(
        CSV,
        placed("emb", "filter", contacts=From("csv.result")),
        placed("send", "send", email=From("emb.result.email")),
        placed("excel", "export", table=From("csv.result", "send.result")),
        registry=with_added(registry, inner),
    )

    assert compiled.problems == ()
    assert _sheet(compiled)["result"].to_list() == ["sent to ida@x", None, "sent to nina@x"]


def test_two_tables_made_per_document_share_no_rows_and_the_author_is_told():
    compiled = _compiled(
        DOCS, READ,
        placed("again", "structured", text=From("docs.result")),
        placed("excel", "export", table=From("read.result.vendor", "again.result.amount")),
    )

    assert [make_problem.type for make_problem in compiled.problems] == ["no_rows_in_common"]


def test_a_declared_table_input_missing_a_column_is_refused_when_checked():
    compiled = _compiled(CSV, SEND, placed("s", "sheet", table=From("csv.result", "send.result")), registry=_registry_with(sheet_node))

    (missing,) = compiled.problems
    assert (missing.type, missing.fatal, missing.loc[1], missing.ctx) == ("table_column_missing", True, "table", {"column": "sent"})


def test_a_merged_table_that_wont_read_as_the_input_fails_that_node():
    compiled = _compiled(CSV, placed("s", "scores", table=From("csv.result")), registry=_registry_with(scores))
    assert compiled.is_runnable, compiled.problems

    events = list(events_of(compiled))
    (failed,) = [event for event in events if event.type == "node_failed"]
    assert (failed.node_id, failed.cause.type) == ("s", "invalid_input")


def test_a_column_two_levels_down_is_received_under_the_nodes_row():
    compiled = _compiled(
        DOCS, READ,
        placed("words", "words", text=From("read.result.vendor")),
        placed("excel", "export", table=From("read.result", "words.result")),
        registry=_registry_with(words),
    )

    from_words = next(part for part in compiled.run_plan.nodes["excel"].inputs["table"].source_reads if part.sources == (Ref("words", "result"),))
    assert (from_words.receive_mode, from_words.rows_of, from_words.grouped_under) == ("all_rows_under_row", "words.result", "docs.result")


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


def _three_levels(text: str) -> tuple[Placed, Placed, Placed]:
    """Documents split on ``|``, paragraphs under them on ``;``, lines under those on ``,``."""
    return (
        placed("docs", "docs", text=Static(text)),
        placed("paras", "paragraphs", text=From("docs.result")),
        placed("lines", "words", text=From("paras.result")),
    )


def _deep_registry() -> NodeRegistry:
    return _registry_with(words, splits, labelled, labelled_table, two_columns)


def _said(compiled: CompiledGraph, graph_node: str = "n") -> list[str]:
    assert compiled.is_runnable, compiled.problems
    return compiled.results(compiled.run_sync())[graph_node]["result"]["values"]


def test_wiring_the_parent_as_well_adds_no_row():
    docs = placed("docs", "docs", text=Static("acme:10,beta:5||gamma:7"))
    alone = _compiled(docs, READ, placed("excel", "export", table=From("read.result.vendor")))
    beside = _compiled(docs, READ, placed("excel", "export", table=From("docs.result", "read.result.vendor")))

    assert _sheet(alone)["vendor"].to_list() == _sheet(beside)["vendor"].to_list() == ["acme", "beta", "gamma"]


def test_a_merge_three_levels_deep_copies_every_level_down_and_a_paragraph_with_no_lines_has_no_row():
    compiled = _compiled(
        *_three_levels("a,b;;c|d"),
        placed("excel", "export", table=From("docs.result", "paras.result", "lines.result")),
        registry=_deep_registry(),
    )
    sheet = _sheet(compiled)

    assert sheet.columns == ["result", "result_2", "result_3"]
    assert sheet.rows() == [
        ("a,b;;c", "a,b", "a"),
        ("a,b;;c", "a,b", "b"),
        ("a,b;;c", "c", "c"),
        ("d", "d", "d"),
    ]


def test_parents_with_nothing_under_them_give_a_table_of_no_rows_with_every_column():
    compiled = _compiled(
        placed("docs", "docs", text=Static("|")),
        READ,
        placed("excel", "export", table=From("docs.result", "read.result.vendor", "read.result.amount")),
    )
    sheet = _sheet(compiled)

    assert (sheet.height, sheet.columns) == (0, ["result", "vendor", "amount"])


def test_a_parent_column_beside_a_table_made_per_row_of_it_is_copied_down():
    compiled = _compiled(DOCS, READ, placed("excel", "export", table=From("docs.result", "read.result")))
    sheets = compiled.results(compiled.run_sync())["excel"]["result"]["values"]

    assert [sheet.drop(ROW_ID_COLUMN).to_dicts() for sheet in sheets] == [
        [{"result": "acme:10,beta:5", "vendor": "acme", "amount": 10.0}, {"result": "acme:10,beta:5", "vendor": "beta", "amount": 5.0}],
        [{"result": "gamma:7", "vendor": "gamma", "amount": 7.0}],
    ]


def test_a_child_table_a_branch_skipped_gives_no_rows_and_the_sheet_keeps_its_columns():
    compiled = _compiled(
        placed("docs", "docs", text=Static("acme:10|skip")),
        placed("read", "skip-or-read", text=From("docs.result")),
        placed("excel", "export", table=From("docs.result", "read.result")),
        registry=_registry_with(skip_or_read),
    )
    sheets = compiled.results(compiled.run_sync())["excel"]["result"]["values"]

    assert [sheet.height for sheet in sheets] == [1, 0]
    assert sheets[1].columns == [ROW_ID_COLUMN, "result", "vendor", "amount"]


def test_a_series_two_levels_down_with_nothing_else_runs_per_row_of_the_level_above_it():
    compiled = _compiled(*_three_levels("a,b;c|d"), placed("n", "labelled", texts=From("lines.result")), registry=_deep_registry())

    assert compiled.run_plan.nodes["n"].runs_per_row_of == "paras.result"
    assert _said(compiled) == ["=a,b", "=c", "=d"]


def test_a_value_on_a_document_with_a_series_of_lines_gets_every_line_of_the_document_flat():
    compiled = _compiled(*_three_levels("a,b;c|d"), placed("n", "labelled", label=From("docs.result"), texts=From("lines.result")), registry=_deep_registry())

    assert compiled.run_plan.nodes["n"].runs_per_row_of == "docs.result"
    assert compiled.run_plan.nodes["n"].inputs["texts"].grouped_under == "docs.result"
    assert _said(compiled) == ["a,b;c=a,b,c", "d=d"]


def test_a_table_of_lines_with_a_value_on_a_document_gets_every_line_of_the_document():
    compiled = _compiled(*_three_levels("a,b;c|d"), placed("n", "labelled-table", label=From("docs.result"), table=From("lines.result")), registry=_deep_registry())

    assert _said(compiled) == ["a,b;c=a,b,c", "d=d"]


def test_a_value_on_a_paragraph_keeps_a_series_of_lines_to_that_paragraph():
    compiled = _compiled(*_three_levels("a,b;c|d"), placed("n", "labelled", label=From("paras.result"), texts=From("lines.result")), registry=_deep_registry())

    assert _said(compiled) == ["a,b=a,b", "c=c", "d=d"]


def test_a_document_with_no_paragraphs_runs_with_an_empty_series():
    compiled = _compiled(*_three_levels("a|;|b"), placed("n", "labelled", label=From("docs.result"), texts=From("lines.result")), registry=_deep_registry())

    assert _said(compiled) == ["a=a", ";=", "b=b"]


def test_two_series_at_different_depths_are_both_read_per_row_of_the_shallowest_level_above_them():
    compiled = _compiled(*_three_levels("a,b;c|d"), placed("n", "two-columns", a=From("paras.result"), b=From("lines.result")), registry=_deep_registry())

    assert compiled.run_plan.nodes["n"].runs_per_row_of == "docs.result"
    assert _said(compiled) == ["a,b,c+a,b,c", "d+d"]


def test_two_series_under_unrelated_rows_are_misaligned():
    compiled = _compiled(
        *_three_levels("a|b"),
        placed("again", "paragraphs", text=From("docs.result")),
        placed("lines2", "words", text=From("again.result")),
        placed("n", "two-columns", a=From("lines.result"), b=From("lines2.result")),
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


def _rebuilt(node_type: str = "picks") -> tuple[Placed, Placed]:
    return (
        placed("keep", node_type, contacts=From("csv.result")),
        placed("mail", "send", email=From("keep.result.email")),
    )


def test_a_table_that_lost_every_rows_id_fails_when_merged_with_the_table_it_was_built_from():
    compiled = _compiled(CSV, *_rebuilt(), placed("excel", "export", table=From("csv.result", "mail.result")), registry=_registry_with(picks))

    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert (state.status, state.cause.type, state.cause.loc[0]) == ("failed", "row_ids_lost", "keep")
    assert "mail" not in compiled.results(state)


def test_a_table_built_from_another_used_on_its_own_gets_fresh_ids():
    compiled = _compiled(CSV, *_rebuilt(), placed("excel", "export", table=From("keep.result", "mail.result")), registry=_registry_with(picks))

    assert _sheet(compiled)["result"].to_list() == ["sent to ida@x", "no address", "sent to nina@x"]


def test_a_table_made_from_another_table_is_unrelated_to_a_third():
    other = placed("other", "csv", text=Static("Ole,ole@x"))
    compiled = _compiled(
        CSV, other, placed("keep", "picks", contacts=From("other.result")),
        placed("excel", "export", table=From("csv.result", "keep.result")), registry=_registry_with(picks),
    )

    assert [problem.type for problem in compiled.problems] == ["no_rows_in_common"]


def test_a_table_of_no_rows_has_lost_no_ids():
    compiled = _compiled(CSV, *_rebuilt("empties"), placed("excel", "export", table=From("csv.result", "mail.result")), registry=_registry_with(empty))

    assert compiled.run_sync().status == "complete"


def test_a_table_merged_with_what_it_was_built_from_that_keeps_its_ids_passes():
    compiled = _compiled(
        CSV, placed("keep", "with-email", contacts=From("csv.result")),
        placed("excel", "export", table=From("csv.result", "keep.result")),
    )

    assert Ref("keep", "result") in compiled._row_ids_must_survive
    assert compiled.run_sync().status == "complete"
