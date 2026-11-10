"""What a run does at its edges: a node's own frame, values JSON can't spell, a damaged state, a person's odd answer."""

import asyncio
import datetime
import decimal
import json
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    ALWAYS,
    NEVER,
    ROW_ID_COLUMN,
    SKIPPED,
    AskPerson,
    CompiledGraph,
    DType,
    InvalidCallError,
    NodeDefinition,
    NodeRegistry,
    Output,
    Param,
    Policy,
    Ref,
    Result,
    RunState,
    Series,
    Table,
    node,
)
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.graph.binding import From, Static
from conductor.widgets import Textarea
from helpers import Placed, answer, placed, read_table, record, registry_of, with_added


class Txt(DType, str):
    id = "edges-txt"
    display_title = "Text"


class Num(DType, float):
    id = "edges-num"
    display_title = "Number"


class Contact(TypedDict):
    name: Txt
    email: Txt


Contacts = Table[Contact]
In = Annotated[Txt, Param(title="In", widget=Textarea())]
Wire = Annotated[Txt, Param(title="In")]
Out = Annotated[Txt, Result(title="result")]
Parts = Annotated[Series[Txt], Result(title="Parts")]
TableOut = Annotated[Contacts, Result(title="Table")]
CALLS: list[Any] = []
CANNED: dict[str, Any] = {}


def n(id: str, type: str | None = None, **bindings: Any) -> Placed:
    return placed(id, type or id, **bindings)


@node("source", title="s", description="d", category="t")
def source(text: In = Txt("")) -> Out:
    return Txt(text)


@node("split", title="s", description="d", category="t")
def split(text: Wire) -> Parts:
    return [Txt(part) for part in text.split(",") if part]


@node("upper", title="s", description="d", category="t")
def upper(text: Wire) -> Out:
    CALLS.append(("upper", str(text)))
    return Txt(text.upper())


@node("join", title="s", description="d", category="t")
def join(texts: Annotated[Series[Txt], Param(title="T")]) -> Out:
    CALLS.append(("join", list(texts)))
    return Txt(",".join(texts))


@node("two", title="s", description="d", category="t")
def two(a: Wire, b: Wire) -> Out:
    CALLS.append(("two", str(a), str(b)))
    return Txt(f"{a}+{b}")


@node("lister", title="s", description="d", category="t")
def lister(text: Wire, tag: Wire) -> Out:
    return Txt(f"{text}:{tag}")


@dataclass(frozen=True)
class Branch:
    yes: Annotated[Txt, Result(title="Yes", exclusive_group="c")]
    no: Annotated[Txt, Result(title="No", exclusive_group="c")]


@node("route", title="s", description="d", category="t")
def route(text: Wire) -> Branch:
    if text == "slow-no":
        time.sleep(0.05)
    return Branch(yes=SKIPPED, no=text) if text.endswith("no") else Branch(yes=text, no=SKIPPED)


@node("num", title="s", description="d", category="t")
def num(k: In = Txt("")) -> Annotated[Num, Result(title="N")]:
    return CANNED[k]


@node("num_in", title="s", description="d", category="t")
def num_in(v: Annotated[Num, Param(title="N")]) -> Annotated[Num, Result(title="N")]:
    CALLS.append(("num_in", repr(v)))
    return v


@node("frame", title="s", description="d", category="t")
def frame_node(k: In = Txt("")) -> Annotated[Table, Result(title="T")]:
    return CANNED[k]


@node("frame_in", title="s", description="d", category="t")
def frame_in(table: Annotated[Table, Param(title="T")]) -> Out:
    CALLS.append(("frame_in", table.schema, table.rows()))
    return Txt(str(table.height))


def _any_outputs() -> list[Output]:
    return [Output("result", Series[Any], title="A")]


@node("anything", title="s", description="d", category="t", compute_outputs=_any_outputs)
def anything(k: In = Txt("")) -> Mapping[str, Any]:
    return {"result": [CANNED[k]]}


@node("pass_table", title="s", description="d", category="t")
def pass_table(x: Wire, t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    return t


@node("merged", title="s", description="d", category="t")
def merged(table: Annotated[Table, Param(title="T")]) -> Annotated[Table, Result(title="T")]:
    return table


MODULE_FRAME = pl.DataFrame({"name": ["a", "b"], "email": ["x", "y"]})


@node("module_frame", title="s", description="d", category="t")
def module_frame(x: Wire) -> TableOut:
    return MODULE_FRAME  # type: ignore[return-value]


@node("csv", title="s", description="d", category="t")
def csv(k: In = Txt("")) -> TableOut:
    names = [x for x in k.split(",") if x]
    return read_table(Contacts, pl.DataFrame({"name": names, "email": [f"{x}@m" for x in names]}, schema={"name": pl.String, "email": pl.String}))


@node("rebuild", title="s", description="d", category="t")
def rebuild(t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    names = t["name"].to_list() or ["placeholder"]
    return read_table(Contacts, [{"name": x, "email": "e"} for x in names])


@node("none_left", title="s", description="d", category="t")
def none_left(t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    return read_table(Contacts, [r for r in t.iter_rows(named=True) if r["name"] == "nobody"])


@node("ask", title="s", description="d", category="t")
def ask(text: Wire) -> Out | AskPerson:
    return AskPerson(prompt=text)


@node("ask_table", title="s", description="d", category="t")
def ask_table(t: Annotated[Contacts, Param(title="T")]) -> TableOut | AskPerson:
    return AskPerson(defaults={"result": t})


@node("ask_rows", title="s", description="d", category="t")
def ask_rows(k: In = Txt("")) -> TableOut | AskPerson:
    return AskPerson()


@node("ask_choice_default", title="s", description="d", category="t")
def ask_choice_default(k: In = Txt("")) -> Branch | AskPerson:
    return AskPerson(defaults={"yes": Txt("x")})


@node("ask_choice", title="s", description="d", category="t")
def ask_choice(k: In = Txt("")) -> Branch | AskPerson:
    return AskPerson()


@node("ask_default_skipped", title="s", description="d", category="t")
def ask_default_skipped(k: In = Txt("")) -> Parts | AskPerson:
    return AskPerson(defaults={"result": [Txt("a"), SKIPPED]})


@node("ask_bad_widget", title="s", description="d", category="t")
def ask_bad_widget(k: In = Txt("")) -> Out | AskPerson:
    return AskPerson(widgets={"result": "not a widget"})  # type: ignore[dict-item]


@node("ask_slow_first", title="s", description="d", category="t")
def ask_slow_first(text: Wire) -> Out | AskPerson:
    time.sleep(0.05 if text == "a" else 0)
    return AskPerson(prompt=text)


@node("reference", title="s", description="d", category="t")
def reference(k: In = Txt("")) -> TableOut:
    return read_table(Contacts, [{"name": "a", "email": "x"}, {"name": "b", "email": "y"}])


@node("lookup", title="s", description="d", category="t")
def lookup(x: Wire, t: Annotated[Contacts, Param(title="T")]) -> TableOut:
    return t.filter(pl.col("name") == "a")  # type: ignore[return-value]


REGISTRY = registry_of(source, split, upper, join, two, lister, route, num, num_in, frame_node, frame_in, anything, pass_table, merged, module_frame, csv, rebuild, none_left, reference, lookup, ask, ask_table, ask_rows, ask_choice_default, ask_choice, ask_default_skipped, ask_bad_widget, ask_slow_first)


def compiled_of(*nodes: Placed, registry: NodeRegistry = REGISTRY) -> CompiledGraph:
    return CompiledGraph.from_graph(record(*nodes), registry)


def through_json(state: RunState) -> RunState:
    return RunState.model_validate_json(state.model_dump_json())


@pytest.fixture(autouse=True)
def _clear_calls() -> None:
    CALLS.clear()


# -- the frame a node returned ------------------------------------------------------


def test_a_table_returned_on_two_rows_has_its_own_ids_on_each_and_the_merge_has_its_rows():
    typed_in = read_table(Contacts, [{"name": "a", "email": "x"}, {"name": "b", "email": "y"}])
    compiled = compiled_of(
        n("s", "source", text=Static(["p1", "p2"])),
        n("h", "pass_table", x=From("s.result"), t=Static(typed_in)),
        n("u", "upper", text=From("h.result.name")),
        n("m", "merged", table=From("h.result.name", "u.result")),
    )
    state = compiled.run_sync()
    assert state.status == "complete", state.cause
    assert [v[ROW_ID_COLUMN].to_list() for v in state.nodes["h"].outputs["result"].values] == [["h.result:0.0", "h.result:0.1"], ["h.result:1.0", "h.result:1.1"]]
    assert compiled.results(state)["m"]["result"].height == 4


def test_a_static_table_is_the_same_after_a_run_so_a_later_compile_finds_nothing_changed():
    def graph() -> CompiledGraph:
        return compiled_of(n("s", "source", text=Static(["p"])), n("h", "pass_table", x=From("s.result"), t=Static(read_table(Contacts, [{"name": "a", "email": "x"}]))))
    ran = graph()
    state = ran.run_sync()
    assert ROW_ID_COLUMN not in ran.node("h").statics["t"].columns
    assert graph()._nodes_changed_since(through_json(state)) == []


def test_a_frame_a_node_shares_with_its_module_is_left_as_it_was():
    compiled_of(n("s", "source", text=Static(["p1", "p2"])), n("h", "module_frame", x=From("s.result"))).run_sync()
    assert MODULE_FRAME.columns == ["name", "email"]


# -- values JSON has no spelling for -------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_number_that_is_not_finite_comes_back_a_number_and_its_reader_runs(value):
    CANNED["k"] = Num(value)
    compiled = compiled_of(n("v", "num", k=Static("k")), n("r", "num_in", v=From("v.result")))
    state = through_json(compiled.run_sync())
    back = compiled.results(state)["v"]["result"]
    assert isinstance(back, Num) and (math.isnan(back) if math.isnan(value) else back == value)
    assert compiled.results(compiled.run_sync(state.without_nodes("r")))["r"]["result"] is not None
    assert len(CALLS) == 2


def test_a_float_column_keeps_nan_and_infinity_through_json():
    CANNED["k"] = pl.DataFrame({"a": [1.5, float("nan"), float("inf"), float("-inf"), None]})
    compiled = compiled_of(n("v", "frame", k=Static("k")))
    state = compiled.run_sync()
    back = compiled.results(through_json(state))["v"]["result"]
    assert back.schema == state.nodes["v"].outputs["result"].values[0].schema
    assert back.equals(state.nodes["v"].outputs["result"].values[0])


def test_a_state_is_valid_json_when_a_number_is_not_finite():
    CANNED["k"] = pl.DataFrame({"a": [float("nan")]})
    CANNED["n"] = Num(float("nan"))
    compiled = compiled_of(n("v", "frame", k=Static("k")), n("w", "num", k=Static("n")))
    json.loads(compiled.run_sync().model_dump_json(), parse_constant=lambda bad: pytest.fail(f"{bad} is not JSON"))


@pytest.mark.parametrize(("value", "written"), [(float("nan"), "NaN"), (float("inf"), "Infinity"), ({"deep": [float("nan")]}, {"deep": ["NaN"]})])
def test_a_float_under_any_is_written_as_text_and_comes_back_as_the_text(value, written):
    """A value typed ``Any`` comes back as its JSON form, whatever it held, so this rule is that one."""
    CANNED["k"] = value
    compiled = compiled_of(n("v", "anything", k=Static("k")))
    state = compiled.run_sync()
    document = json.loads(state.model_dump_json(), parse_constant=lambda bad: pytest.fail(f"{bad} is not JSON"))
    assert document["nodes"]["v"]["outputs"]["result"]["values"] == [[written]]
    assert compiled.results(through_json(state))["v"]["result"]["values"] == [written]


# -- a table's JSON form --------------------------------------------------------------


UTC = datetime.timezone.utc


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"a": [decimal.Decimal("1.25"), None]}, schema={"a": pl.Decimal(38, 2)}),
        pl.DataFrame({"a": ["x", None]}, schema={"a": pl.Enum(["x", "y"])}),
        pl.DataFrame({"a": [datetime.timedelta(seconds=3, microseconds=7), None]}, schema={"a": pl.Duration("us")}),
        pl.DataFrame({"a": [datetime.timedelta(milliseconds=3)]}, schema={"a": pl.Duration("ms")}),
        pl.DataFrame({"a": [datetime.datetime(2024, 1, 1, 12, 0, 0, 123456, tzinfo=UTC), None]}, schema={"a": pl.Datetime("us", "UTC")}),
        pl.DataFrame({"a": [datetime.datetime(2024, 1, 1, 12, tzinfo=UTC)]}, schema={"a": pl.Datetime("ms", "Europe/Copenhagen")}),
        pl.DataFrame({"a": [datetime.date(2024, 1, 1), None]}),
        pl.DataFrame({"a": [datetime.time(1, 2, 3, 4000)]}),
        pl.DataFrame({"a": ["x", None]}, schema={"a": pl.Categorical}),
        pl.DataFrame({"a": [1]}, schema={"a": pl.UInt8}),
        pl.DataFrame({"a": [None, None]}),
        pl.DataFrame({"a": [["x"], None, []]}, schema={"a": pl.List(pl.String)}),
        pl.DataFrame({"a": [[1], []]}, schema={"a": pl.List(pl.Int32)}),
        pl.DataFrame({"a": [{"x": 1, "y": "q"}]}),
        pl.DataFrame({"a": [b"utf8 bytes"]}, schema={"a": pl.Binary}),
        pl.DataFrame({"rows": [1], "data": [2], "columns": [3]}),
    ],
    ids=lambda frame: str(frame.schema["a"]) if "a" in frame.columns else "wire-words",
)
def test_every_column_type_comes_back_exactly_or_the_dump_refuses(frame):
    CANNED["k"] = frame
    compiled = compiled_of(n("v", "frame", k=Static("k")), n("r", "frame_in", table=From("v.result")))
    state = through_json(compiled.run_sync())
    back = compiled.results(state)["v"]["result"].drop(ROW_ID_COLUMN)
    assert dict(back.schema) == dict(frame.schema)
    assert back.equals(frame)
    compiled.run_sync(state.without_nodes("r"))
    assert CALLS[0] == CALLS[1]


@pytest.mark.parametrize(
    "frame",
    [
        pl.DataFrame({"a": [b"\xff"]}, schema={"a": pl.Binary}),
        pl.DataFrame({"a": [datetime.datetime(2024, 1, 1)]}, schema={"a": pl.Datetime("ns")}).with_columns(pl.col("a").dt.offset_by("1ns")),
    ],
    ids=["bytes-not-text", "nanoseconds"],
)
def test_a_column_json_can_not_keep_is_refused_at_the_dump_naming_the_column(frame):
    CANNED["k"] = frame
    state = compiled_of(n("v", "frame", k=Static("k"))).run_sync()
    with pytest.raises(Exception, match=r"v\.result: a value has no JSON form .*column 'a'"):
        state.model_dump_json()


def test_integer_row_ids_read_as_text_the_same_fresh_and_after_json():
    CANNED["k"] = pl.DataFrame({ROW_ID_COLUMN: [1, 2], "a": ["x", "y"]})
    compiled = compiled_of(n("v", "frame", k=Static("k")))
    state = compiled.run_sync()
    assert compiled.results(state)["v"]["result"][ROW_ID_COLUMN].to_list() == ["1", "2"]
    assert compiled.results(through_json(state))["v"]["result"][ROW_ID_COLUMN].to_list() == ["1", "2"]


def test_a_json_form_cannot_name_anything_but_a_column_type():
    with pytest.raises(ValueError, match="does not line up"):
        read_table(Table, {"columns": [{"name": "a", "dtype": {"type": "read_csv", "source": "/etc/hosts"}}], "data": {"a": []}, "rows": None})


# -- zero rows ------------------------------------------------------------------------


def test_a_typed_table_of_no_records_is_a_table_with_its_columns_and_no_rows():
    assert read_table(Contacts, []).columns == [ROW_ID_COLUMN, "name", "email"] and read_table(Contacts, []).height == 0


def test_a_node_filtering_every_row_away_returns_zero_rows():
    compiled = compiled_of(n("c", "csv", k=Static("a,b")), n("f", "none_left", t=From("c.result")))
    state = compiled.run_sync()
    assert state.status == "complete", state.cause
    assert compiled.results(state)["f"]["result"].height == 0


def test_a_person_answering_an_empty_list_for_a_table_gives_a_table_of_no_rows():
    compiled = compiled_of(n("a", "ask_rows"))
    state = compiled.run_sync(answer(compiled, compiled.run_sync(), {"a": {"result": []}}))
    assert state.status == "complete"
    assert compiled.results(state)["a"]["result"].height == 0


# -- row ids --------------------------------------------------------------------------


def test_a_persons_records_in_the_published_shape_are_new_rows_not_a_lost_id():
    compiled = compiled_of(n("c", "csv", k=Static("a,b")), n("r", "ask_table", t=From("c.result")), n("m", "merged", table=From("c.result", "r.result")))
    state = compiled.run_sync()
    assert state.waiting[0].questions[0].json_schema["type"] == "array"
    ended = compiled.run_sync(answer(compiled, state, {"r": {"result": [{"name": "a", "email": "9"}]}}))
    assert ended.status == "complete", ended.cause
    assert compiled.results(ended)["m"]["result"].height == 3


def test_a_node_given_no_rows_has_no_id_to_lose_when_it_makes_a_row_of_its_own():
    compiled = compiled_of(n("c", "csv", k=Static("")), n("r", "rebuild", t=From("c.result")), n("m", "merged", table=From("c.result", "r.result")))
    state = compiled.run_sync()
    assert state.status == "complete", state.cause


def test_a_fresh_id_never_takes_one_a_later_row_holds():
    CANNED["k"] = pl.DataFrame({ROW_ID_COLUMN: [None, "v.result:0"], "a": ["x", "y"]}, schema={ROW_ID_COLUMN: pl.String, "a": pl.String})
    compiled = compiled_of(n("v", "frame", k=Static("k")))
    ids = compiled.results(compiled.run_sync())["v"]["result"][ROW_ID_COLUMN].to_list()
    assert len(set(ids)) == 2 and ids[1] == "v.result:0"


def test_a_table_and_columns_of_it_wired_beside_it_are_one_table():
    compiled = compiled_of(n("c", "csv", k=Static("a,b")), n("m", "merged", table=From("c.result.name", "c.result")))
    state = compiled.run_sync()
    assert state.status == "complete", state.cause
    assert compiled.results(state)["m"]["result"].columns == ["__row", "name", "email"]


# -- lists set on a node over no rows --------------------------------------------------


def test_an_empty_list_above_a_list_and_two_more_levels_ends_cleanly():
    compiled = compiled_of(
        n("s", "source", text=Static([])), n("l", "lister", text=From("s.result"), tag=Static(["p", "q"])),
        n("i", "split", text=From("l.result")), n("q", "upper", text=From("i.result")),
    )
    assert compiled.run_sync().status == "complete"


# -- reading a state -------------------------------------------------------------------


def test_outputs_of_a_complete_state_over_a_changed_graph_is_refused():
    done = compiled_of(n("x", "source", text=Static("1")), n("y", "upper", text=From("x.result"))).run_sync()
    edited = compiled_of(n("x", "source", text=Static("2")), n("y", "upper", text=From("x.result")))
    with pytest.raises(InvalidCallError) as raised:
        edited.outputs(done)
    assert [p.type for p in raised.value.errors()] == ["run_not_complete"]
    assert raised.value.errors()[0].ctx["status"] == "stale"


def test_outputs_of_a_complete_state_with_a_last_node_dropped_still_answers():
    done = compiled_of(n("x", "source", text=Static("1")), n("y", "upper", text=From("x.result")), n("z", "upper", text=From("x.result"))).run_sync()
    assert compiled_of(n("x", "source", text=Static("1")), n("y", "upper", text=From("x.result"))).outputs(done) == {"y.result": "1"}


def _waiting_run() -> tuple[CompiledGraph, RunState]:
    compiled = compiled_of(n("s", "source", text=Static("a,b")), n("sp", "split", text=From("s.result")), n("a", "ask", text=From("sp.result")), n("j", "join", texts=From("a.result")))
    return compiled, compiled.run_sync()


def _damage(change: Any) -> tuple[CompiledGraph, RunState]:
    compiled, state = _waiting_run()
    dumped = json.loads(state.model_dump_json())
    change(dumped)
    return compiled, RunState.model_validate(dumped)


def _drop(d: dict[str, Any], key: str) -> None:
    del d[key]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["nodes"]["a"].update(finished_units=[[[99], None]]),
        lambda d: d["nodes"]["a"].update(finished_units=[[[-1], None]]),
        lambda d: d["nodes"]["sp"]["outputs"]["result"]["rows"].__setitem__(0, [-1]),
        lambda d: d["nodes"]["s"]["outputs"]["result"].update(rows=[], values=[]),
        lambda d: d["nodes"]["a"].update(finished_units=[[[0], None]]),
        lambda d: d["nodes"]["a"]["waiting"][0].update(row=[5]),
        lambda d: _drop(d["nodes"]["s"]["outputs"], "result"),
    ],
    ids=["unit-at-a-row-never-made", "unit-at-a-negative-row", "value-at-a-negative-row", "a-finished-unit-with-no-value", "a-finished-unit-its-output-missing", "waiting-at-a-row-never-made", "a-column-gone"],
)
def test_a_damaged_state_is_refused_at_the_call(change):
    compiled, state = _damage(change)
    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync(state)
    assert {p.type for p in raised.value.errors()} == {"state_unreadable"}


def test_an_answer_to_a_tampered_row_is_refused_at_the_next_leg():
    compiled, state = _damage(lambda d: d["nodes"]["a"]["waiting"][0].update(row=[5]))
    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync(answer(compiled, state, {"a@0": {"result": "x"}, "a@1": {"result": "y"}}))
    assert {p.type for p in raised.value.errors()} == {"state_unreadable"}


def test_a_stored_table_with_no_row_ids_is_unreadable():
    compiled = compiled_of(n("c", "csv", k=Static("a")))
    dumped = json.loads(compiled.run_sync().model_dump_json())
    digest = dumped["nodes"]["c"]["outputs"]["result"]["values"][0]["table"]
    dumped["tables"][digest]["rows"] = None
    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync(RunState.model_validate(dumped))
    assert {p.type for p in raised.value.errors()} == {"state_unreadable"}


def test_results_reads_the_state_without_building_a_run():
    compiled, state = _waiting_run()
    from conductor._internal.execution import run_ledger

    def not_called(*_: Any, **__: Any) -> None:
        raise AssertionError("results built the run's live values")

    original = run_ledger.RunLedger.__init__
    run_ledger.RunLedger.__init__ = not_called  # type: ignore[method-assign]
    try:
        assert compiled.results(state)["sp"]["result"]["values"] == ["a", "b"]
    finally:
        run_ledger.RunLedger.__init__ = original  # type: ignore[method-assign]


def test_a_node_that_ran_on_no_rows_is_in_the_results_with_no_rows_before_and_after_json():
    compiled = compiled_of(n("s", "source", text=Static([])), n("u", "upper", text=From("s.result")))
    state = compiled.run_sync()
    assert compiled.results(state)["u"]["result"]["values"] == compiled.results(through_json(state))["u"]["result"]["values"] == []


def test_two_identical_runs_dump_identical_json():
    compiled = compiled_of(n("s", "source", text=Static([str(i) for i in range(50)])), n("u", "upper", text=From("s.result")), n("sp", "split", text=From("u.result")))
    assert compiled.run_sync().model_dump_json() == compiled.run_sync().model_dump_json()


def test_finished_rows_come_out_in_row_order():
    compiled = compiled_of(n("s", "source", text=Static([str(i) for i in range(50)])), n("u", "upper", text=From("s.result")))
    assert [row for row, _ in compiled.run_sync().nodes["u"].finished_units] == [(i,) for i in range(50)]


def test_waiting_units_are_in_row_order_whatever_order_they_asked_in():
    compiled = compiled_of(n("s", "source", text=Static(["a", "b", "c"])), n("q", "ask_slow_first", text=From("s.result")))
    assert [unit.row for unit in compiled.run_sync().waiting] == [(0,), (1,), (2,)]


def test_waiting_units_are_in_execution_order_across_nodes():
    compiled = compiled_of(n("s", "source", text=Static(["a"])), n("q", "ask_slow_first", text=From("s.result")), n("p", "ask", text=From("q.result")), n("r", "ask", text=From("s.result")))
    assert [unit.node_id for unit in compiled.run_sync().waiting] == ["q", "r"]


# -- what can be written and asked -----------------------------------------------------


def test_a_node_id_with_the_separator_of_a_unit_id_is_refused():
    assert "invalid_node_id" in {p.type for p in compiled_of(n("a@0", "source", text=Static("x"))).problems}


def test_a_choice_is_never_optional_and_leaving_it_out_is_refused():
    compiled = compiled_of(n("a", "ask_choice_default"))
    state = compiled.run_sync()
    assert [q.required for q in state.waiting[0].questions] == [True, True]
    assert state.waiting[0].questions[0].default == "x"
    with pytest.raises(InvalidCallError) as raised:
        answer(compiled, state, {"a": {}})
    assert [p.type for p in raised.value.errors()] == ["choice_unanswered"]


def test_an_answer_naming_none_of_a_choice_is_refused():
    compiled = compiled_of(n("a", "ask_choice"))
    with pytest.raises(InvalidCallError) as raised:
        answer(compiled, compiled.run_sync(), {"a": {}})
    assert [p.type for p in raised.value.errors()] == ["choice_unanswered"]


@pytest.mark.parametrize("node_type", ["ask_default_skipped", "ask_bad_widget"])
def test_a_node_asking_wrongly_is_its_own_invalid_output(node_type):
    state = compiled_of(n("a", node_type)).run_sync()
    assert (state.status, state.cause.type) == ("failed", "invalid_output")


def test_questions_of_one_node_share_one_record_and_a_default_gets_its_own():
    compiled = compiled_of(n("s", "source", text=Static(["a", "b"])), n("q", "ask", text=From("s.result")))
    waiting = compiled.run_sync().waiting
    assert waiting[0].questions[0] is waiting[1].questions[0]
    defaulted = compiled_of(n("a", "ask_choice_default")).run_sync().waiting[0].questions
    assert defaulted[0].default == "x" and defaulted[1].default is None


# -- skips, blanks, without ------------------------------------------------------------


def test_a_blank_row_leaves_every_series_on_its_index_so_a_zip_stays_paired():
    @node("blank_some", title="s", description="d", category="t")
    def blank_some(k: In = Txt("")) -> Annotated[Series[Txt | None], Result(title="L")]:
        return [None if x == "_" else Txt(x) for x in k.split(",")]

    @node("keep_all", title="s", description="d", category="t")
    def keep_all(k: Wire) -> Annotated[Txt | None, Result(title="B")]:
        return Txt(k + "?")

    @node("zipped", title="s", description="d", category="t")
    def zipped(a: Annotated[Series[Txt], Param(title="A")], b: Annotated[Series[Txt | None], Param(title="B")]) -> Out:
        return Txt(str(list(zip(a, b, strict=True))))

    registry = with_added(REGISTRY, blank_some, keep_all, zipped)
    compiled = compiled_of(n("a", "blank_some", k=Static("1,_,3")), n("b", "keep_all", k=From("a.result")), n("z", "zipped", a=From("a.result"), b=From("b.result")), registry=registry)
    state = compiled.run_sync()
    assert compiled.results(state)["z"]["result"] == str([("1", "1?"), ("3", "3?")])


def test_a_skip_above_a_row_reaches_the_reader_at_that_row():
    compiled = compiled_of(
        n("d", "source", text=Static(["a,b-no", "c-no,e"])), n("g", "route", text=From("d.result")), n("l", "split", text=From("d.result")),
        n("t", "two", a=From("l.result"), b=From("g.no")), n("u", "upper", text=From("t.result")),
    )
    state = compiled.run_sync()
    assert state.status == "complete", state.cause
    assert compiled.results(state)["u"]["result"]["rows"] == [(0, 0), (0, 1)]


def test_a_reduction_waits_for_the_unit_above_the_rows_for_a_skip_above_its_group():
    release = threading.Event()

    @node("held", title="s", description="d", category="t", policy=Policy(concurrency=1))
    def held(text: Wire) -> Out:
        release.wait(5)
        CALLS.append(("held", str(text)))
        return Txt(text)

    registry = with_added(REGISTRY, held)
    compiled = compiled_of(
        n("docs", "source", text=Static(["a", "slow-no"])), n("g", "route", text=From("docs.result")), n("l", "split", text=From("g.yes")),
        n("h", "held", text=From("l.result")), n("j", "join", texts=From("h.result")), registry=registry,
    )

    async def go() -> RunState:
        running = asyncio.create_task(compiled.run())
        await asyncio.sleep(0.2)
        assert ("join", []) not in CALLS
        release.set()
        return await running

    state = asyncio.run(go())
    assert state.status == "complete", state.cause
    assert ("join", []) not in CALLS
    assert compiled.results(state)["j"]["result"]["rows"] == [(0,)]


def test_a_decision_over_a_whole_column_gates_what_hangs_off_the_branch_not_taken():
    """An "if the list is empty" decision: the branch not taken is skipped at the top, not an empty
    series, so a node per row of it stands in once and the reduction below it never runs."""

    @dataclass(frozen=True)
    class Decided:
        full: Annotated[Series[Txt], Result(title="Full", exclusive_group="d")]
        empty: Annotated[Series[Txt], Result(title="Empty", exclusive_group="d")]

    @node("decide", title="s", description="d", category="t")
    def decide(texts: Annotated[Series[Txt], Param(title="T")]) -> Decided:
        return Decided(full=list(texts), empty=SKIPPED) if len(texts) else Decided(full=SKIPPED, empty=[])

    compiled = compiled_of(
        n("docs", "split", text=Static("a,b")), n("d", "decide", texts=From("docs.result")),
        n("full", "join", texts=From("d.full")), n("up", "upper", text=From("d.empty")), n("after", "join", texts=From("up.result")),
        registry=with_added(REGISTRY, decide),
    )
    state = compiled.run_sync()
    results = compiled.results(state)
    assert state.status == "complete", state.cause
    assert results["full"]["result"] == "a,b"
    assert "up" not in results and "after" not in results
    assert [call for call in CALLS if call[0] == "join"] == [("join", ["a", "b"])]
    assert "empty" not in results["d"]


def test_a_skip_above_an_empty_group_hides_what_the_reduction_returned():
    """Whether the reduction over the empty group ran before the skip landed is timing; the results are not."""
    compiled = compiled_of(
        n("d", "source", text=Static(["p-no"])), n("g", "route", text=From("d.result")), n("l", "split", text=From("d.result")),
        n("t", "two", a=From("l.result"), b=From("g.no")), n("m", "split", text=From("l.result")),
        n("x", "two", a=From("t.result"), b=From("m.result")), n("j", "join", texts=From("x.result")),
    )
    state = compiled.run_sync()
    assert state.status == "complete", state.cause
    assert compiled.results(state)["j"]["result"]["rows"] == [(0, 0)]


def test_without_covers_the_nodes_of_an_embedded_graph():
    compiled, state = _waiting_run()
    inner = NodeDefinition.from_compiled_graphs("inner", title="I", description="d", category="t", versions={1: compiled_of(n("check", "source", text=Static("x")))})
    outer = compiled_of(n("approve", "inner"), registry=with_added(REGISTRY, inner))
    done = outer.run_sync()
    assert done.nodes["approve/check"].finished_units is not None
    assert "approve/check" not in done.without_nodes("approve").nodes


def test_an_alternative_taking_both_ways_of_one_decision_can_never_hold():
    compiled = compiled_of(
        n("g", "route", text=Static("x")), n("a", "upper", text=From("g.yes")), n("b", "upper", text=From("g.no")), n("m", "two", a=From("a.result"), b=From("b.result")),
    )
    assert compiled.field(Ref("a.result")).condition != ALWAYS
    assert compiled.field(Ref("m.result")).condition == NEVER


def test_a_loaded_definition_cannot_take_the_id_of_one_already_registered():
    @node("source", title="other", description="d", category="t")
    def other(text: In = Txt("")) -> Out:
        return Txt("not the built-in")

    with pytest.raises(NodeDeclarationError, match="two nodes have the id 'source'"):
        with_added(REGISTRY, other)


# -- rows that repeat an id under different parents -------------------------------------


def test_a_table_passed_through_per_row_merges_row_by_row_under_each_parent():
    compiled = compiled_of(
        n("s", "source", text=Static(["p1", "p2"])), n("k", "reference"),
        n("h", "pass_table", x=From("s.result"), t=From("k.result")), n("u", "upper", text=From("h.result.name")),
        n("m", "merged", table=From("h.result.name", "u.result")),
    )
    state = compiled.run_sync()
    assert state.status == "complete", state.cause
    frame = compiled.results(state)["m"]["result"]
    assert frame.height == 4
    assert frame.select("name", "result").rows() == [("a", "A"), ("b", "B"), ("a", "A"), ("b", "B")]


def test_a_lookup_repeating_a_reference_row_under_two_parents_still_joins_the_reference_on_its_id():
    compiled = compiled_of(
        n("s", "source", text=Static(["p1", "p2"])), n("k", "reference"),
        n("h", "lookup", x=From("s.result"), t=From("k.result")), n("u", "upper", text=From("h.result.name")),
        n("m", "merged", table=From("k.result", "h.result.name", "u.result")),
    )
    state = compiled.run_sync()
    assert state.status == "complete", state.cause
    rows = compiled.results(state)["m"]["result"].select("name", "email", "result").rows()
    assert sorted(rows, key=str) == sorted([("a", "x", "A"), ("a", "x", "A"), ("b", "y", None)], key=str)


def test_which_parent_a_repeated_id_lands_on_does_not_depend_on_which_finished_first():
    @node("pass_slow_first", title="s", description="d", category="t")
    def pass_slow_first(x: Wire, t: Annotated[Contacts, Param(title="T")]) -> TableOut:
        time.sleep(0.05 if x == "p1" else 0)
        return t

    registry = with_added(REGISTRY, pass_slow_first)
    compiled = compiled_of(
        n("s", "source", text=Static(["p1", "p2"])), n("k", "reference"),
        n("h", "pass_slow_first", x=From("s.result"), t=From("k.result")), n("u", "upper", text=From("h.result.name")),
        n("m", "merged", table=From("h.result.name", "u.result")), registry=registry,
    )
    state = compiled.run_sync()
    assert compiled.results(state)["m"]["result"].select("name", "result").rows() == [("a", "A"), ("b", "B"), ("a", "A"), ("b", "B")]


# -- an answer that keeps a table's ids; results straight after an answer -----------------


def test_the_published_table_answer_may_carry_row_ids_and_need_not():
    compiled = compiled_of(n("c", "csv", k=Static("a,b")), n("r", "ask_table", t=From("c.result")))
    record = compiled.run_sync().waiting[0].questions[0].json_schema["items"]
    assert ROW_ID_COLUMN in record["properties"] and ROW_ID_COLUMN not in record["required"]


def test_records_in_the_published_shape_that_keep_row_ids_merge_on_their_source_without_doubling():
    compiled = compiled_of(n("c", "csv", k=Static("a,b")), n("r", "ask_table", t=From("c.result")), n("m", "merged", table=From("c.result", "r.result")))
    state = compiled.run_sync()
    default = state.waiting[0].questions[0].default
    records = [{ROW_ID_COLUMN: global_row_id, "name": name, "email": "edited"} for global_row_id, name in zip(default[ROW_ID_COLUMN], default["name"], strict=True)]
    ended = compiled.run_sync(answer(compiled, state, {"r": {"result": records}}))
    assert ended.status == "complete", ended.cause
    assert compiled.results(ended)["m"]["result"].height == 2


@node("ask_list", title="s", description="d", category="t")
def ask_list(k: In = Txt("")) -> Parts | AskPerson:
    return AskPerson()


def test_a_reader_left_with_no_rows_by_an_answer_is_in_the_results_before_the_next_leg():
    compiled = compiled_of(n("a", "ask_list"), n("u", "upper", text=From("a.result")), registry=with_added(REGISTRY, ask_list))
    answered = answer(compiled, compiled.run_sync(), {"a": {"result": []}})
    assert compiled.results(answered)["u"]["result"]["values"] == []
    assert answered.nodes["u"].finished_units == []
