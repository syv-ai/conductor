"""Saving a run and loading it back: ``run.to_json()`` and ``graph.load_run(...)``.

A saved run is strict JSON: each node's outputs are a column per output,
written by the output's own type, so a value reads back as itself, a float
that isn't finite included. A save that doesn't fit the graph, damaged or
written for another one, is refused at ``load_run`` as ``state_unreadable``,
and a value with no JSON form at ``to_json`` as ``state_unwritable``; each
names where, and quotes no stored value.
"""

import datetime as dt
import json
import math
import struct
from decimal import Decimal
from typing import Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    Ask,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Series,
    Table,
    make_graph,
    node,
)
from pydantic import BaseModel

from tests.test_core.nodes import registry_of


def _round_trip(run: Any) -> Any:
    return run.graph.load_run(run.to_json())


def _refused(graph: Any, saved: dict[str, Any]) -> Any:
    with pytest.raises(InvalidCallError) as refused:
        graph.load_run(json.dumps(saved))
    (problem,) = refused.value.errors()
    assert problem.type == "state_unreadable"
    return problem


def _strict(text: str) -> Any:
    """``text`` read as JSON that refuses ``NaN`` and ``Infinity``."""

    def refuse(constant: str) -> Any:
        raise ValueError(f"not strict JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


@node("split")
def split(text: str) -> Series[str]:
    return text.split(",")


# -- values read back as themselves -------------------------------------------------------------


def test_a_float_comes_back_to_the_last_bit():
    floats = [0.1 + 0.2, 5e-324, -0.0, 1e308, 1 / 3]

    @node("floats")
    def make_floats(text: str) -> Series[float]:
        return floats

    run = registry_of(make_floats).load(make_graph(make_floats)).run(text="x")

    loaded = _round_trip(run).results["floats"]["result"]

    assert [struct.pack("<d", f) for f in loaded] == [struct.pack("<d", f) for f in floats]


def test_a_text_that_spells_a_skip_marker_stays_text():
    @node("marker")
    def marker(text: str) -> Series[str]:
        return ["__skipped__", "SKIPPED", ""]

    run = registry_of(marker).load(make_graph(marker)).run(text="x")

    assert _round_trip(run).results["marker"]["result"] == ["__skipped__", "SKIPPED", ""]


def _paused_after(definition: Any, *, per_row: bool = False) -> Any:
    """``definition`` as node ``m``, its ``result`` read by a node that asks and by ``seen``, which takes the answer and what ``m`` made."""
    seen: list[Any] = []

    @node("check")
    def check(value: Any) -> str:
        raise Ask("Right?")

    @node("seen")
    def see(value: Any, checked: str) -> str:
        seen.append(value)
        return checked

    nodes = [NodeRecord(id="m", node=definition.id), NodeRecord(id="c", node="check"), NodeRecord(id="s", node="seen")]
    edges = [("m.result", "c.value"), ("m.result", "s.value"), ("c.result", "s.checked")]
    if per_row:
        nodes.insert(0, NodeRecord(id="split", node="split"))
        edges.append(("split.result", "m.text"))
    graph = registry_of(split, definition, check, see).load(GraphRecord(nodes=nodes, edges=edges))
    return graph, seen


def _is_nan_inf_minus_inf(values: list[float]) -> bool:
    return math.isnan(values[0]) and values[1:] == [math.inf, -math.inf]


def test_a_float_that_is_not_finite_comes_back_a_float_and_its_reader_receives_it():
    @node("ratio")
    def ratio(text: str) -> float:
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[text]

    for text in ("nan", "inf", "-inf"):
        graph, seen = _paused_after(ratio)
        paused = graph.run(text=text)
        saved = paused.to_json()
        _strict(saved)

        loaded = graph.load_run(saved)
        value = loaded.results["m"]["result"]
        done = loaded.answer("c", "yes").resume()

        assert isinstance(value, float) and (math.isnan(value) if text == "nan" else value == float(text))
        assert done.status == "complete" and len(seen) == 1 and (math.isnan(seen[0]) if text == "nan" else seen[0] == float(text))


def test_a_series_of_floats_keeps_the_ones_that_are_not_finite():
    @node("readings")
    def readings(text: str) -> Series[float]:
        return [math.nan, math.inf, -math.inf]

    run = registry_of(readings).load(make_graph(readings)).run(text="x")

    assert _is_nan_inf_minus_inf(_round_trip(run).results["readings"]["result"])


def test_a_float_per_row_keeps_the_ones_that_are_not_finite():
    @node("per-row")
    def per_row(text: str) -> float:
        return float(text)

    graph, _ = _paused_after(per_row, per_row=True)
    paused = graph.run(text="nan,inf,-inf")

    assert _is_nan_inf_minus_inf(graph.load_run(paused.to_json()).results["m"]["result"])


class Reading(TypedDict):
    value: float | None


def test_a_float_column_keeps_nan_infinity_and_a_blank():
    @node("table")
    def table(text: str) -> Table[Reading]:
        return pl.DataFrame({"value": [math.nan, math.inf, -math.inf, None]})

    run = registry_of(table).load(make_graph(table)).run(text="x")

    column = _round_trip(run).results["table"]["result"]["value"].to_list()

    assert math.isnan(column[0]) and column[1:] == [math.inf, -math.inf, None]


def test_a_saved_run_is_strict_json_when_a_number_is_not_finite():
    @node("odd")
    def odd(text: str) -> Series[float]:
        return [math.nan, math.inf]

    saved = registry_of(odd).load(make_graph(odd)).run(text="x").to_json()

    assert _strict(saved)["status"] == "complete"


def test_a_float_that_is_not_finite_under_any_is_written_and_read_back_as_its_text():
    class Loose(BaseModel):
        data: Any

    @node("loose")
    def loose(text: str) -> Loose:
        return Loose(data={"a": math.nan, "b": [math.inf, -math.inf]})

    run = registry_of(loose).load(make_graph(loose)).run(text="x")
    _strict(run.to_json())

    assert _round_trip(run).results["loose"]["result"].data == {"a": "NaN", "b": ["Infinity", "-Infinity"]}


class Point(TypedDict):
    x: int
    y: int


class Everything(TypedDict):
    flag: bool
    count: int
    amount: Decimal
    kind: Literal["a", "b"]
    day: dt.date
    at: dt.datetime
    clock: dt.time
    took: dt.timedelta
    tags: list[str]
    point: Point


_EVERYTHING = {
    "flag": [True, False],
    "count": [1, -2],
    "amount": [Decimal("1.25"), Decimal("-0.10")],
    "kind": ["a", "b"],
    "day": [dt.date(2026, 1, 2), dt.date(1999, 12, 31)],
    "at": [dt.datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=dt.UTC), dt.datetime(2000, 1, 1, tzinfo=dt.UTC)],
    "clock": [dt.time(12, 30), dt.time(0, 0, 1)],
    "took": [dt.timedelta(seconds=90), dt.timedelta(days=-1)],
    "tags": [["x", "y"], []],
    "point": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
}


def test_every_column_type_comes_back_exactly_and_a_reader_receives_the_same():
    received: list[pl.DataFrame] = []

    @node("everything")
    def everything(text: str) -> Table[Everything]:
        return pl.DataFrame(_EVERYTHING)

    @node("check-table")
    def check_table(rows: pl.DataFrame) -> str:
        raise Ask("Right?")

    @node("take")
    def take(rows: pl.DataFrame, checked: str) -> str:
        received.append(rows)
        return checked

    graph = registry_of(everything, check_table, take).load(GraphRecord(
        nodes=[NodeRecord(id="e", node="everything"), NodeRecord(id="c", node="check-table"), NodeRecord(id="t", node="take")],
        edges=[("e.result", "c.rows"), ("e.result", "t.rows"), ("c.result", "t.checked")],
    ))
    paused = graph.run(text="x")
    before = paused.results["e"]["result"]

    loaded = graph.load_run(paused.to_json())
    after = loaded.results["e"]["result"]
    loaded.answer("c", "yes").resume()

    assert after.to_dict(as_series=False) == before.to_dict(as_series=False) == _EVERYTHING
    assert after.schema == before.schema
    assert received[0].to_dict(as_series=False) == _EVERYTHING


def test_a_table_with_no_row_type_keeps_its_list_struct_and_time_columns():
    columns = {"tags": [["a"], ["b", "c"]], "point": [{"x": 1}, {"x": 2}], "clock": [dt.time(1, 2), dt.time(3, 4)]}

    @node("bare")
    def bare(text: str) -> pl.DataFrame:
        return pl.DataFrame(columns)

    run = registry_of(bare).load(make_graph(bare)).run(text="x")

    loaded = _round_trip(run).results["bare"]["result"]

    assert loaded.to_dict(as_series=False) == columns and loaded.schema == run.results["bare"]["result"].schema


def test_tables_of_two_nodes_read_back_each_as_itself():
    class Name(TypedDict):
        name: str

    class Age(TypedDict):
        age: int

    @node("names")
    def names(text: str) -> Table[Name]:
        return pl.DataFrame({"name": ["ida", "bo"]})

    @node("ages")
    def ages(text: str) -> Table[Age]:
        return pl.DataFrame({"age": [30, 40, 50]})

    run = registry_of(names, ages).load(GraphRecord(nodes=[NodeRecord(id="n", node="names"), NodeRecord(id="a", node="ages")])).run(
        {"n.text": "x", "a.text": "y"},
    )

    loaded = _round_trip(run)

    assert loaded.results["n"]["result"].to_dicts() == [{"name": "ida"}, {"name": "bo"}]
    assert loaded.results["a"]["result"].to_dicts() == [{"age": 30}, {"age": 40}, {"age": 50}]


def test_two_identical_runs_save_byte_identical_json():
    @node("mixed")
    def mixed(text: str) -> Series[float]:
        return [1.5, math.nan, 2.0]

    graph = registry_of(split, mixed).load(make_graph(mixed))

    assert graph.run(text="x").to_json() == graph.run(text="x").to_json()


# -- what a save can't hold ------------------------------------------------------------------


class _Opaque:
    """A value pydantic carries but has no JSON form for."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.is_instance_schema(cls)


def test_saving_a_run_that_holds_a_value_with_no_json_form_raises_naming_the_output():
    @node("opaque")
    def opaque(text: str) -> _Opaque:
        return _Opaque()

    run = registry_of(opaque).load(GraphRecord(nodes=[NodeRecord(id="o", node="opaque")])).run(text="x")

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert (problem.type, problem.ctx["at"]) == ("state_unwritable", "o.result")


def test_a_column_json_cannot_keep_makes_the_save_raise_naming_the_output_and_column():
    class Blob(TypedDict):
        data: bytes

    @node("blob")
    def blob(text: str) -> Table[Blob]:
        return pl.DataFrame({"data": [b"\xff\xfe"]})

    run = registry_of(blob).load(GraphRecord(nodes=[NodeRecord(id="b", node="blob")])).run(text="x")

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert problem.type == "state_unwritable" and problem.ctx["at"].startswith("b.result") and "data" in problem.ctx["at"]
    assert "\\xff" not in json.dumps(problem.model_dump(mode="json"))


# -- a save that doesn't fit is refused ------------------------------------------------------


def _saved_mails() -> tuple[Any, dict[str, Any]]:
    """``split`` making three rows read by ``echo``, run to the end, and its save."""

    @node("echo")
    def echo(text: str) -> str:
        return text

    graph = registry_of(split, echo).load(make_graph(split, echo))
    return graph, json.loads(graph.run(text="secret-a,secret-b,secret-c").to_json())


def test_a_saved_value_its_type_does_not_read_is_refused_naming_where_and_quoting_nothing():
    graph, saved = _saved_mails()
    saved["units"]["echo"]["outputs"]["result"]["values"][1] = {"secret": "value"}

    problem = _refused(graph, saved)

    assert "echo.result" in json.dumps(problem.model_dump(mode="json"))
    assert "secret" not in json.dumps(problem.model_dump(mode="json"))


def test_a_blank_saved_on_an_output_reads_back_as_none():
    graph, saved = _saved_mails()
    saved["units"]["echo"]["outputs"]["result"]["values"][1] = None

    assert graph.load_run(json.dumps(saved)).results["echo"]["result"] == ["secret-a", None, "secret-c"]


@pytest.mark.parametrize(
    "damage",
    ["a row never made", "a negative row", "a row listed twice", "a row on a node that runs once", "a missing value", "a missing output", "an output the node lacks"],
)
def test_a_damaged_save_is_refused(damage):
    graph, saved = _saved_mails()
    echo, splitting = saved["units"]["echo"], saved["units"]["split"]
    if damage == "a row never made":
        echo["keys"][2] = [7]
    elif damage == "a negative row":
        echo["keys"][2] = [-1]
    elif damage == "a row listed twice":
        echo["keys"][2] = [1]
    elif damage == "a row on a node that runs once":
        splitting["keys"][0] = [0]
    elif damage == "a missing value":
        echo["outputs"]["result"]["values"].pop()
    elif damage == "a missing output":
        del echo["outputs"]["result"]
    else:
        echo["outputs"]["other"] = {"values": ["x", "y", "z"], "skipped": []}

    _refused(graph, saved)


def _saved_table() -> tuple[Any, dict[str, Any]]:
    class Name(TypedDict):
        name: str

    @node("names")
    def names(text: str) -> Table[Name]:
        return pl.DataFrame({"name": text.split(",")})

    @node("greet")
    def greet(name: str) -> str:
        return f"hej {name}"

    graph = registry_of(names, greet).load(GraphRecord(
        nodes=[NodeRecord(id="n", node="names"), NodeRecord(id="g", node="greet")], edges=[("n.result.name", "g.name")],
    ))
    return graph, json.loads(graph.run(text="ida,bo").to_json())


def test_a_null_where_a_table_that_makes_rows_was_is_refused():
    graph, saved = _saved_table()
    saved["units"]["n"]["outputs"]["result"]["values"][0] = None

    _refused(graph, saved)


def test_a_malformed_table_is_refused():
    graph, saved = _saved_table()
    saved["units"]["n"]["outputs"]["result"]["values"][0] = {"rows": 2, "columns": {"name": ["ida"]}}

    _refused(graph, saved)
