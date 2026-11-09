"""What a person's answer stays with, and what decides that work can be reused.

Whether a call can be reused goes by what it received, leaving out where the
rows sit: a row inserted above changes nothing a whole-table reader reads. An
answer goes further: it belongs to the question and to the row it was asked on,
that row read whole, every row above it, the row its ``__row`` came from, and
which of how many identical rows it was.
"""

from collections import Counter
from typing import Annotated, TypedDict

import polars as pl
import pytest
from conductor import (
    ROW_ID_COLUMN,
    AskPerson,
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeRegistry,
    Param,
    Result,
    RunState,
    Series,
    Static,
    Table,
    node,
)

CALLS: Counter = Counter()


class Invoice(TypedDict):
    who: str
    amount: str


class Amount(TypedDict):
    amount: str


class Line(TypedDict):
    text: str


Tag = Annotated[str, Param(title="T", connectable=False)]


@node("invoices", category="test")
def invoices(body: Tag) -> Annotated[Table[Invoice], Result(title="i")]:
    rows = [line.split("|") for line in body.splitlines() if line]
    return pl.DataFrame({"who": [r[0] for r in rows], "amount": [r[1] for r in rows]}, schema={"who": pl.String, "amount": pl.String})


@node("approve", category="test")
def approve(amount: str) -> Annotated[str, Result(title="ok")] | AskPerson:
    CALLS["approve"] += 1
    return AskPerson(prompt=amount)


@node("drop_who", category="test")
def drop_who(t: Table[Invoice]) -> Annotated[Table[Amount], Result(title="t")]:
    return t.select(ROW_ID_COLUMN, "amount")


@node("note", category="test")
def note(who: str) -> Annotated[str, Result(title="n")]:
    CALLS["note"] += 1
    return "noted " + who


@node("lines_of", category="test")
def lines_of(amount: str) -> Annotated[Table[Line], Result(title="l")]:
    CALLS["lines_of"] += 1
    return pl.DataFrame({"text": [amount + "-1", amount + "-2"]})


@node("keep_ends_2", category="test")
def keep_ends_2(lines: Table[Line]) -> Annotated[Table[Line], Result(title="k")]:
    CALLS["keep_ends_2"] += 1
    return lines.filter(pl.col("text").str.ends_with("2"))


@node("whole_table", category="test")
def whole_table(t: Table) -> Annotated[int, Result(title="n")]:
    CALLS["whole_table"] += 1
    return t.height


@node("only_big", category="test")
def only_big(t: Table[Invoice]) -> Annotated[Table[Invoice], Result(title="b")]:
    return t.filter(pl.col("amount").str.len_chars() > 1)


@node("words_of", category="test")
def words_of(amount: str) -> Annotated[Series[str], Result(title="w")]:
    return amount.split(" ")


@node("approve_word", category="test")
def approve_word(word: str) -> Annotated[str, Result(title="ok")] | AskPerson:
    CALLS["approve_word"] += 1
    return AskPerson(prompt=word)


@node("merge_all", category="test")
def merge_all(t: Table) -> Annotated[Table, Result(title="m")]:
    return t


@node("pass_table", category="test")
def pass_table(amount: str, t: Table[Invoice]) -> Annotated[Table[Invoice], Result(title="p")]:
    return t


@node("upper", category="test")
def upper(x: str) -> Annotated[str, Result(title="up")]:
    return x.upper()


@node("same_again", category="test")
def same_again(t: Table[Invoice]) -> Annotated[Table[Invoice], Result(title="s")]:
    return t


@node("left_over", category="test")
def left_over(tag: Tag) -> Annotated[Table[Invoice], Result(title="f")]:
    trailing = [None] if tag == "tail" else []
    frame = pl.DataFrame({"who": [None, "Ida", "Bo", *trailing], "amount": ["1", "2", "3", *["4"] * len(trailing)]}, schema={"who": pl.String, "amount": pl.String})
    return frame.filter(pl.col("amount").is_not_null()).head(3)


registry = NodeRegistry([
    invoices, approve, drop_who, note, lines_of, keep_ends_2, whole_table, only_big, words_of,
    approve_word, merge_all, pass_table, upper, same_again, left_over,
])


def n(node_id: str, node_type: str, /, **bindings) -> GraphNode:
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def compile_nodes(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry)
    assert compiled.is_runnable, compiled.problems
    return compiled


def through_json(state: RunState) -> RunState:
    return RunState.model_validate_json(state.model_dump_json())


def ask_graph(body: str) -> CompiledGraph:
    return compile_nodes(n("i", "invoices", body=Static(body)), n("a", "approve", amount=From("i.result.amount")))


def answered_all(compiled: CompiledGraph, state: RunState, *answers: str) -> RunState:
    """``state`` with each waiting unit answered in row order, run on; ``CALLS`` starts afresh."""
    units = sorted(state.waiting, key=lambda unit: unit.row)
    answered = compiled.run_sync(through_json(compiled.answer(state, {unit.id: {"result": text} for unit, text in zip(units, answers, strict=True)})))
    CALLS.clear()
    return answered


def answers_by_row(compiled: CompiledGraph, state: RunState, node_id: str = "a") -> dict[tuple[int, ...], str]:
    result = compiled.results(state).get(node_id, {}).get("result")
    return {} if result is None else dict(zip(result["rows"], result["values"], strict=True))


def waiting_rows(state: RunState) -> list[tuple[int, ...]]:
    return sorted(unit.row for unit in state.waiting)


def test_a_row_replaced_by_another_person_with_the_same_shown_amount_is_asked_again() -> None:
    compiled = ask_graph("Ida|100\n")
    state = answered_all(compiled, compiled.run_sync(), "pay Ida")

    replaced = ask_graph("Bo|100\n")
    ending = replaced.run_sync(through_json(state))

    assert answers_by_row(replaced, ending) == {} and waiting_rows(ending) == [(0,)] and CALLS["approve"] == 1


def test_a_row_replaced_is_asked_again_behind_a_select_that_dropped_the_name() -> None:
    def graph(body: str) -> CompiledGraph:
        return compile_nodes(n("i", "invoices", body=Static(body)), n("d", "drop_who", t=From("i.result")), n("a", "approve", amount=From("d.result.amount")))

    compiled = graph("Ida|100\n")
    state = answered_all(compiled, compiled.run_sync(), "pay Ida")

    ending = graph("Bo|100\n").run_sync(through_json(state))

    assert waiting_rows(ending) == [(0,)]


def test_a_form_opened_for_a_row_since_replaced_is_refused() -> None:
    compiled = ask_graph("Ida|100\n")
    state = compiled.run_sync()
    opened_for_ida = state.waiting[0].id

    replaced = ask_graph("Bo|100\n")
    ending = replaced.run_sync(through_json(state))

    assert waiting_rows(ending) == [(0,)] and ending.waiting[0].id != opened_for_ida
    with pytest.raises(InvalidCallError) as stale:
        replaced.answer(ending, {opened_for_ida: {"result": "pay"}})
    assert stale.value.errors()[0].type == "not_waiting"


def test_an_answer_follows_its_row_when_the_row_moves_and_a_stale_form_is_refused() -> None:
    compiled = ask_graph("Ida|100\nBo|200\nKim|300\n")
    first = compiled.run_sync()
    by_row = {unit.row: unit for unit in first.waiting}
    state = compiled.run_sync(through_json(compiled.answer(first, {by_row[(0,)].id: {"result": "pay Ida"}})))
    stale_form_of_kim = by_row[(2,)].id

    moved = ask_graph("New|9\nIda|100\nBo|200\nKim|300\n")
    ending = moved.run_sync(through_json(state))

    assert answers_by_row(moved, ending) == {(1,): "pay Ida"}
    assert waiting_rows(ending) == [(0,), (2,), (3,)]
    with pytest.raises(InvalidCallError) as stale:
        moved.answer(ending, {stale_form_of_kim: {"result": "stale"}})
    assert stale.value.errors()[0].type == "not_waiting"


def test_a_column_the_form_never_showed_edited_asks_again_and_reruns_nothing_else() -> None:
    def graph(body: str) -> CompiledGraph:
        return compile_nodes(
            n("i", "invoices", body=Static(body)),
            n("a", "approve", amount=From("i.result.amount")),
            n("l", "lines_of", amount=From("i.result.amount")),
        )

    compiled = graph("Ida|100\nBo|200\n")
    state = answered_all(compiled, compiled.run_sync(), "pay Ida", "pay Bo")

    edited = graph("Ida Jensen|100\nBo|200\n")
    ending = edited.run_sync(through_json(state))

    assert answers_by_row(edited, ending) == {(1,): "pay Bo"} and waiting_rows(ending) == [(0,)]
    assert CALLS["approve"] == 1 and CALLS["lines_of"] == 0


def test_identical_rows_keep_their_answers_kth_to_kth_when_their_count_is_unchanged() -> None:
    compiled = ask_graph("Ida|100\nIda|100\n")
    state = answered_all(compiled, compiled.run_sync(), "yes", "no")

    moved = ask_graph("New|9\nIda|100\nIda|100\n")
    ending = moved.run_sync(through_json(state))

    assert answers_by_row(moved, ending) == {(1,): "yes", (2,): "no"} and waiting_rows(ending) == [(0,)]


def test_identical_rows_are_all_asked_again_when_their_count_changes() -> None:
    compiled = ask_graph("Ida|100\nIda|100\n")
    state = answered_all(compiled, compiled.run_sync(), "yes", "no")

    fewer = ask_graph("Ida|100\n")
    assert answers_by_row(fewer, ending := fewer.run_sync(through_json(state))) == {} and waiting_rows(ending) == [(0,)]

    more = ask_graph("Ida|100\nIda|100\nIda|100\n")
    assert answers_by_row(more, ending := more.run_sync(through_json(state))) == {} and waiting_rows(ending) == [(0,), (1,), (2,)]


def test_identical_lines_answered_differently_under_documents_keep_their_documents() -> None:
    def graph(body: str) -> CompiledGraph:
        return compile_nodes(n("i", "invoices", body=Static(body)), n("w", "words_of", amount=From("i.result.amount")), n("a", "approve_word", word=From("w.result")))

    compiled = graph("A|same\nB|same\n")
    state = answered_all(compiled, compiled.run_sync(), "yes for A", "no for B")

    moved = graph("B|same\nA|same\n")
    ending = moved.run_sync(through_json(state))

    assert answers_by_row(moved, ending) == {(0, 0): "no for B", (1, 0): "yes for A"} and ending.status == "complete"


def per_document_graph(body: str) -> CompiledGraph:
    return compile_nodes(
        n("i", "invoices", body=Static(body)),
        n("l", "lines_of", amount=From("i.result.amount")),
        n("k", "keep_ends_2", lines=From("l.result")),
        n("w", "whole_table", t=From("k.result")),
    )


def test_a_document_inserted_at_the_top_does_not_call_a_per_document_table_reader_again() -> None:
    state = per_document_graph("A|a\nB|b\nC|c\n").run_sync()
    CALLS.clear()

    ending = per_document_graph("N|n\nA|a\nB|b\nC|c\n").run_sync(through_json(state))

    assert ending.status == "complete"
    assert CALLS == {"lines_of": 1, "keep_ends_2": 1, "whole_table": 1}


def test_a_row_a_filter_drops_inserted_above_does_not_send_the_whole_table_again() -> None:
    def graph(body: str) -> CompiledGraph:
        return compile_nodes(n("i", "invoices", body=Static(body)), n("f", "only_big", t=From("i.result")), n("w", "whole_table", t=From("f.result")))

    state = graph("Small|1\nBig|100\n").run_sync()
    CALLS.clear()

    ending = graph("Tiny|2\nSmall|1\nBig|100\n").run_sync(through_json(state))

    assert ending.status == "complete" and CALLS["whole_table"] == 0


def test_a_reused_table_that_took_a_table_carries_the_ids_a_fresh_run_would() -> None:
    state = per_document_graph("A|a\nB|b\n").run_sync()
    moved = per_document_graph("N|n\nA|a\nB|b\n")

    ending = moved.run_sync(through_json(state))
    fresh = moved.run_sync()

    def ids(run: RunState) -> list[list[str]]:
        return [frame[ROW_ID_COLUMN].to_list() for frame in moved.results(run)["k"]["result"]["values"]]

    assert ids(ending) == ids(fresh)
    assert ending.nodes["k"].reused_units == 2


def test_a_reused_table_that_moved_lines_up_in_a_merge_with_what_was_called_afresh() -> None:
    @node("upper_text", category="test")
    def upper_text(text: str) -> Annotated[str, Result(name="up")]:
        return text.upper()

    reg = NodeRegistry([invoices, lines_of, keep_ends_2, upper_text, merge_all])

    def graph(body: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[
            n("i", "invoices", body=Static(body)),
            n("l", "lines_of", amount=From("i.result.amount")),
            n("k", "keep_ends_2", lines=From("l.result")),
            n("u", "upper_text", text=From("k.result.text")),
            n("m", "merge_all", t=From("k.result", "u.up")),
        ]), reg)

    state = graph("A|a\nB|b\n").run_sync()
    moved = graph("N|n\nA|a\nB|b\n")

    ending = moved.run_sync(through_json(state))

    per_document = moved.results(ending)["m"]["result"]["values"]
    assert [(frame["text"].to_list(), frame["up"].to_list()) for frame in per_document] == [(["n-2"], ["N-2"]), (["a-2"], ["A-2"]), (["b-2"], ["B-2"])]


def test_equal_tables_are_the_same_work_whatever_was_filtered_away_to_make_them() -> None:
    def graph(tag: str) -> CompiledGraph:
        return compile_nodes(n("f", "left_over", tag=Static(tag)), n("w", "whole_table", t=From("f.result")))

    state = graph("head").run_sync()
    CALLS.clear()

    ending = graph("tail").run_sync(through_json(state))

    assert ending.status == "complete" and CALLS["whole_table"] == 0


def test_a_table_passed_through_per_row_merges_row_by_row_without_multiplying() -> None:
    @node("up_who", category="test")
    def up_who(who: str) -> Annotated[str, Result(name="up")]:
        return who.upper()

    @node("low_who", category="test")
    def low_who(who: str) -> Annotated[str, Result(name="low")]:
        return who.lower()

    reg = NodeRegistry([invoices, pass_table, same_again, up_who, low_who, merge_all])

    compiled = CompiledGraph.from_graph(Graph(nodes=[
        n("d", "invoices", body=Static("a|1\nb|2\n")),
        n("c", "invoices", body=Static("Ida|x\nBo|y\nNina|z\n")),
        n("p", "pass_table", amount=From("d.result.amount"), t=From("c.result")),
        n("s", "same_again", t=From("p.result")),
        n("u", "up_who", who=From("p.result.who")),
        n("l", "low_who", who=From("s.result.who")),
        n("m", "merge_all", t=From("u.up", "l.low")),
    ]), reg)

    merged = compiled.results(compiled.run_sync())["m"]["result"]

    assert merged.height == 6 and merged["up"].null_count() == 0 and merged["low"].null_count() == 0


def test_a_renamed_node_is_a_new_node_and_is_called_again() -> None:
    def graph(node_id: str) -> CompiledGraph:
        return compile_nodes(n("i", "invoices", body=Static("Ida|100\n")), n(node_id, "note", who=From("i.result.who")))

    state = graph("a").run_sync()
    CALLS.clear()

    graph("renamed").run_sync(through_json(state))

    assert CALLS["note"] == 1
