"""Tests for saving a run and reading it back: ``saved = run.to_json()`` and ``Run.from_json(saved, graph)``.

A saved run is strict JSON. A node's outputs are saved as one list per output, with one entry per call. Each value
is written by its output's declared type, so it reads back as itself. That includes a float that isn't finite.

Some saved runs can't be read back. ``Run.from_json`` refuses a damaged one with ``saved_run_unreadable``, and one
saved from another graph with ``saved_run_of_another_graph``. ``to_json`` refuses a value with no JSON form with
``run_unwritable``. Each problem says where it is, in ``ctx["where"]``. That is a path into the JSON, such as
``outcomes.echo.outputs.result.1``, or an output, such as ``o.result``. A problem never quotes a saved value.
"""

import datetime as dt
import json
import math
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, Optional, TypedDict

import polars as pl
import pytest
from conductor import (
    JSON_CONFIG,
    SKIPPED,
    Ask,
    Graph,
    GraphInputRecord,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    NoneInputCall,
    Param,
    Run,
    Series,
    Source,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import AfterValidator, BaseModel, ConfigDict, Discriminator, TypeAdapter, with_config
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pytest import MonkeyPatch

from tests.test_core.nodes import input_entry, registry_of


@pytest.fixture(autouse=True)
def _every_saved_run_reads_back_and_saves_the_same(monkeypatch: MonkeyPatch) -> None:
    """Makes every ``run.to_json()`` in this file check itself.

    Each call also reads its text back into the graph, and checks that the run read back saves to the same bytes.
    """
    write = Run.to_json

    def written_and_read(run: Run) -> str:
        text = write(run)
        assert write(Run.from_json(text, run.graph)) == text
        return text

    monkeypatch.setattr(Run, "to_json", written_and_read)


def _round_trip(run: Any) -> Any:
    return Run.from_json(run.to_json(), run.graph)


def _refused(graph: Any, saved: dict[str, Any]) -> Any:
    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc) == ("saved_run_unreadable", ("saved_run",))
    return problem


def _strict(text: str) -> Any:
    """Reads ``text`` as strict JSON, which refuses ``NaN`` and ``Infinity``."""

    def refuse(constant: str) -> Any:
        raise ValueError(f"not strict JSON: {constant}")

    return json.loads(text, parse_constant=refuse)


@node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",")


# -- values read back as themselves -------------------------------------------------------------


def test_a_float_reads_back_bit_for_bit():
    floats = [0.1 + 0.2, 5e-324, -0.0, 1e308, 1 / 3]

    @node("floats", title="Make floats", description="For a test.")
    def make_floats(text: str) -> Series[float]:
        return floats

    run = Graph.compile(make_graph(make_floats), registry_of(make_floats)).run({"text": "x"})

    read_back = _round_trip(run).results["floats"]["result"]

    assert [struct.pack("<d", f) for f in read_back] == [struct.pack("<d", f) for f in floats]


def test_a_text_that_spells_the_skipped_sentinel_reads_back_as_text():
    @node("spelled", title="Spelled", description="For a test.")
    def spelled(text: str) -> Series[str]:
        return ["__skipped__", "SKIPPED", ""]

    run = Graph.compile(make_graph(spelled), registry_of(spelled)).run({"text": "x"})

    assert _round_trip(run).results["spelled"]["result"] == ["__skipped__", "SKIPPED", ""]


def _waiting_after(definition: Any, *, per_item: bool = False) -> Any:
    """Returns a graph that waits for a person right after ``definition``, and the values its last node gets.

    ``definition`` is the node ``m``. Node ``c`` asks a person about ``m.result``. Node ``s`` takes ``m.result`` and
    the answer, and appends ``m.result`` to the list. With ``per_item``, ``m`` is called once per item of ``split``.
    """
    seen: list[Any] = []

    @node("check", title="Check", description="For a test.", asks=True)
    def check(value: Any) -> str:
        raise Ask("Right?")

    @node("seen", title="See", description="For a test.")
    def see(value: Any, checked: str) -> str:
        seen.append(value)
        return checked

    def sources(node_id: str) -> dict[str, Any]:
        return {"sources": [{"node": node_id, "output": "result"}]}

    nodes: dict[str, Any] = {
        "m": {
            "node": definition.id,
            "inputs": {"text": sources("split") if per_item else {"graph_input": "text"}},
        },
        "c": {"node": "check", "inputs": {"value": sources("m")}},
        "s": {"node": "seen", "inputs": {"value": sources("m"), "checked": sources("c")}},
    }
    if per_item:
        nodes["split"] = {"node": "split", "inputs": {"text": {"graph_input": "text"}}}
    record = GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": nodes,
            "outputs": {"result": {"node": "s", "output": "result"}},
        }
    )
    graph = Graph.compile(record, registry_of(split, definition, check, see))
    return graph, seen


def _is_nan_inf_minus_inf(values: list[float]) -> bool:
    return math.isnan(values[0]) and values[1:] == [math.inf, -math.inf]


def test_a_float_that_is_not_finite_reads_back_as_a_float_and_the_downstream_node_gets_it():
    @node("ratio", title="Ratio", description="For a test.")
    def ratio(text: str) -> float:
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}[text]

    for text in ("nan", "inf", "-inf"):
        graph, seen = _waiting_after(ratio)
        paused = graph.run({"text": text})
        saved = paused.to_json()
        _strict(saved)

        read_back = Run.from_json(saved, graph)
        value = read_back.results["m"]["result"]
        done = read_back.with_answers("c", "yes").resume()

        assert isinstance(value, float) and (
            math.isnan(value) if text == "nan" else value == float(text)
        )
        assert (
            done.status == "complete"
            and len(seen) == 1
            and (math.isnan(seen[0]) if text == "nan" else seen[0] == float(text))
        )


def test_a_series_of_floats_keeps_the_ones_that_are_not_finite():
    @node("readings", title="Readings", description="For a test.")
    def readings(text: str) -> Series[float]:
        return [math.nan, math.inf, -math.inf]

    run = Graph.compile(make_graph(readings), registry_of(readings)).run({"text": "x"})

    assert _is_nan_inf_minus_inf(_round_trip(run).results["readings"]["result"])


def test_floats_returned_per_item_keep_the_ones_that_are_not_finite():
    @node("per-item", title="Per item", description="For a test.")
    def per_item(text: str) -> float:
        return float(text)

    graph, _ = _waiting_after(per_item, per_item=True)
    paused = graph.run({"text": "nan,inf,-inf"})

    assert _is_nan_inf_minus_inf(Run.from_json(paused.to_json(), graph).results["m"]["result"])


class Reading(TypedDict):
    value: float | None


def test_a_float_column_keeps_nan_infinity_and_null():
    @node("table", title="Table", description="For a test.")
    def table(text: str) -> Table[Reading]:
        return pl.DataFrame({"value": [math.nan, math.inf, -math.inf, None]})

    run = Graph.compile(make_graph(table), registry_of(table)).run({"text": "x"})

    column = _round_trip(run).results["table"]["result"]["value"].to_list()

    assert math.isnan(column[0]) and column[1:] == [math.inf, -math.inf, None]


def test_a_saved_run_is_strict_json_when_a_number_is_not_finite():
    @node("odd", title="Odd", description="For a test.")
    def odd(text: str) -> Series[float]:
        return [math.nan, math.inf]

    saved = Graph.compile(make_graph(odd), registry_of(odd)).run({"text": "x"}).to_json()

    assert _strict(saved)["ended_early"] is None


def test_none_returned_for_an_output_typed_series_or_none_reads_back_as_none():
    @node("maybe-readings", title="Maybe readings", description="For a test.")
    def maybe_readings(text: str) -> Series[int] | None:
        return None

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "m": {"node": "maybe-readings", "inputs": {"text": {"graph_input": "text"}}}
                },
                "outputs": {"result": {"node": "m", "output": "result"}},
            }
        ),
        registry_of(maybe_readings),
    ).run({"text": "x"})

    assert run.status == "complete"
    assert _round_trip(run).results["m"]["result"] is None


def test_a_float_that_is_not_finite_in_a_tuple_reads_back_by_the_type_declared_at_its_position():
    """A float that isn't finite, inside a tuple, reads back by the type declared at its position.

    A ``tuple`` inside a ``dict`` output or a model's field is a list in the saved run. A position declared ``float``
    reads back as the float. A position declared ``Any`` reads back as its text, ``"Infinity"``.
    """

    @host_type("Point", title="Point")
    class Point(BaseModel, frozen=True):
        xy: tuple[float, Any]

    @node("pair", title="Pair", description="For a test.")
    def pair(text: str) -> dict[str, tuple[float, ...]]:
        return {"a": (math.nan, 1.5)}

    @node("point", title="Point", description="For a test.")
    def point(text: str) -> Point:
        return Point(xy=(math.inf, math.inf))

    record = make_graph(
        pair,
        point,
        edges=[],
        inputs={"text": ["pair.text", "point.text"]},
        outputs={"pair": "pair.result", "point": "point.result"},
    )
    graph = Graph.compile(record, registry_of(pair, point))
    results = _round_trip(graph.run({"text": "x"})).results

    assert (
        math.isnan(results["pair"]["result"]["a"][0]) and results["pair"]["result"]["a"][1] == 1.5
    )
    assert results["point"]["result"].xy == (math.inf, "Infinity")


def test_a_float_that_is_not_finite_where_its_type_does_not_say_float_reads_back_as_its_text():
    """A float that isn't finite reads back as its text where its type doesn't say ``float``.

    A ``nan`` under ``dict``, ``Any`` or a model's ``Any`` field reads back as the text it was written as, ``"NaN"``.
    A date under ``Any`` reads back as text in the same way. An author who wants the float back declares ``float``.
    """

    @host_type("Loose", title="Loose")
    class Loose(BaseModel, frozen=True):
        data: Any

    @node("loose", title="Loose", description="For a test.")
    def loose(text: str) -> Loose:
        return Loose(data={"a": math.nan, "b": [math.inf, -math.inf]})

    @node("mapping", title="Mapping", description="For a test.")
    def mapping(text: str) -> dict:
        return {"v": math.nan, "w": [{"x": math.inf}], "s": "NaN", "d": dt.date(2026, 1, 1)}

    @node("anything", title="Anything", description="For a test.")
    def anything(text: str) -> Series[Any]:
        return [math.nan, "Infinity", {"z": -math.inf}]

    @node("maybe-mapping", title="Maybe mapping", description="For a test.")
    def maybe_mapping(text: str) -> dict[str, Any] | None:
        return {"v": math.nan}

    names = ("loose", "mapping", "anything", "maybe-mapping")
    record = make_graph(
        loose,
        mapping,
        anything,
        maybe_mapping,
        edges=[],
        inputs={"text": [f"{n}.text" for n in names]},
        outputs={n: f"{n}.result" for n in names},
    )
    graph = Graph.compile(record, registry_of(loose, mapping, anything, maybe_mapping))
    run = graph.run({"text": "x"})
    _strict(run.to_json())
    results = _round_trip(run).results

    assert results["loose"]["result"].data == {"a": "NaN", "b": ["Infinity", "-Infinity"]}
    assert results["mapping"]["result"] == {
        "v": "NaN",
        "w": [{"x": "Infinity"}],
        "s": "NaN",
        "d": "2026-01-01",
    }
    assert results["anything"]["result"] == ["NaN", "Infinity", {"z": "-Infinity"}]
    assert results["maybe-mapping"]["result"] == {"v": "NaN"}


def test_a_float_or_text_field_reads_a_float_that_is_not_finite_back_as_the_text_it_takes():
    @node("up", title="Up", description="For a test.")
    def up(k: int) -> Series[float | str]:
        return [math.nan, "NaN", 1.5]

    @node("down", title="Down", description="For a test.")
    def down(x: float | str) -> str:
        return type(x).__name__

    graph = Graph.compile(
        make_graph(up, down, edges=[("up.result", "down.x")]), registry_of(up, down)
    )
    read_back = _round_trip(graph.run({"k": 1}))

    assert read_back.results["up"]["result"] == ["NaN", "NaN", 1.5]
    assert read_back.with_forgotten("down").resume().outputs["result"] == ["str", "str", "float"]


@host_type("point", title="Point")
class Point(BaseModel, frozen=True):
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
    "at": [
        dt.datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=dt.UTC),
        dt.datetime(2000, 1, 1, tzinfo=dt.UTC),
    ],
    "clock": [dt.time(12, 30), dt.time(0, 0, 1)],
    "took": [dt.timedelta(seconds=90), dt.timedelta(days=-1)],
    "tags": [["x", "y"], []],
    "point": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
}


def test_every_column_type_reads_back_exactly_and_the_downstream_node_gets_the_same():
    received: list[pl.DataFrame] = []

    @node("everything", title="Everything", description="For a test.")
    def everything(text: str) -> Table[Everything]:
        return pl.DataFrame(_EVERYTHING)

    @node("check-table", title="Check table", description="For a test.", asks=True)
    def check_table(rows: pl.DataFrame) -> str:
        raise Ask("Right?")

    @node("take", title="Take", description="For a test.")
    def take(rows: pl.DataFrame, checked: str) -> str:
        received.append(rows)
        return checked

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "e": {"node": "everything", "inputs": {"text": {"graph_input": "text"}}},
                    "c": {
                        "node": "check-table",
                        "inputs": {"rows": {"sources": [{"node": "e", "output": "result"}]}},
                    },
                    "t": {
                        "node": "take",
                        "inputs": {
                            "rows": {"sources": [{"node": "e", "output": "result"}]},
                            "checked": {"sources": [{"node": "c", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "t", "output": "result"}},
            }
        ),
        registry_of(everything, check_table, take),
    )
    paused = graph.run({"text": "x"})
    before = paused.results["e"]["result"]

    read_back = Run.from_json(paused.to_json(), graph)
    after = read_back.results["e"]["result"]
    read_back.with_answers("c", "yes").resume()

    assert after.to_dict(as_series=False) == before.to_dict(as_series=False) == _EVERYTHING
    assert after.schema == before.schema
    assert received[0].to_dict(as_series=False) == _EVERYTHING


def test_a_table_with_no_row_type_keeps_its_list_struct_time_and_decimal_columns():
    columns = {
        "tags": [["a"], ["b", "c"]],
        "point": [{"x": 1}, {"x": 2}],
        "clock": [dt.time(1, 2), dt.time(3, 4)],
        "amount": [Decimal("1.10"), Decimal("2.25")],
    }

    @node("bare", title="Bare", description="For a test.")
    def bare(text: str) -> pl.DataFrame:
        return pl.DataFrame(columns)

    run = Graph.compile(make_graph(bare), registry_of(bare)).run({"text": "x"})

    read_back = _round_trip(run).results["bare"]["result"]

    assert (
        read_back.to_dict(as_series=False) == columns
        and read_back.schema == run.results["bare"]["result"].schema
    )


@pytest.mark.parametrize(
    ("column", "kept_as"),
    [
        (pl.Series([1, -2], dtype=pl.Int32), pl.Int64),
        (pl.Series([1, 2], dtype=pl.UInt32), pl.Int64),
        (pl.Series([2**63 - 1], dtype=pl.UInt64), pl.Int64),
        (pl.Series([1.5, None], dtype=pl.Float32), pl.Float64),
        (pl.Series(["x", "y"], dtype=pl.Categorical), pl.String),
        (pl.Series(["x"], dtype=pl.Enum(["x", "y"])), pl.String),
        (pl.Series([dt.datetime(2026, 1, 1, 12, 30)], dtype=pl.Datetime("ms")), pl.Datetime("us")),
        (
            pl.Series(
                [dt.datetime(2026, 1, 1, 12, 30)], dtype=pl.Datetime("ms")
            ).dt.replace_time_zone("UTC"),
            pl.Datetime("us", "UTC"),
        ),
        (pl.Series([dt.timedelta(seconds=90)], dtype=pl.Duration("ms")), pl.Duration("us")),
        (
            pl.Series([dt.datetime(2026, 1, 1, 12, 30, 0, 5)], dtype=pl.Datetime("ns")),
            pl.Datetime("us"),
        ),
        (
            pl.Series(
                [dt.datetime(2026, 1, 1, 12, 30, 0, 5)], dtype=pl.Datetime("ns")
            ).dt.replace_time_zone("UTC"),
            pl.Datetime("us", "UTC"),
        ),
        (pl.Series([dt.timedelta(microseconds=5)], dtype=pl.Duration("ns")), pl.Duration("us")),
        (pl.Series([["x"]], dtype=pl.List(pl.Enum(["x"]))), pl.List(pl.String)),
        (pl.Series([Decimal("1.50")], dtype=pl.Decimal(10, 2)), pl.Decimal(38, 2)),
        (pl.Series([[1, 2]], dtype=pl.List(pl.Int32)), pl.List(pl.Int64)),
        (pl.Series([[1, 2]], dtype=pl.Array(pl.Int64, 2)), pl.List(pl.Int64)),
        (
            pl.Series(
                [{"n": 1, "tag": "a"}], dtype=pl.Struct({"n": pl.Int16, "tag": pl.Categorical})
            ),
            pl.Struct({"n": pl.Int64, "tag": pl.String}),
        ),
    ],
)
def test_a_returned_table_column_is_kept_as_the_type_a_saved_run_reads_it_back_as_when_no_cell_changes(
    column: pl.Series, kept_as: Any
):
    received: list[pl.DataFrame] = []

    @node("narrow", title="Narrow", description="For a test.")
    def narrow(text: str) -> Table:
        return pl.DataFrame({"a": column})

    @node("take", title="Take", description="For a test.")
    def take(rows: Table) -> int:
        received.append(rows)
        return rows.height

    graph = Graph.compile(
        make_graph(narrow, take, edges=[("narrow.result", "take.rows")]), registry_of(narrow, take)
    )
    run = graph.run({"text": "x"})
    kept, read_back = run.results["narrow"]["result"], _round_trip(run).results["narrow"]["result"]

    assert kept.schema == read_back.schema == received[0].schema == {"a": kept_as}
    assert read_back.to_dicts() == kept.to_dicts() == pl.DataFrame({"a": column}).to_dicts()


@pytest.mark.parametrize(
    ("column", "saved_as"),
    [
        (
            pl.Series([dt.datetime(2026, 1, 1)]).dt.replace_time_zone("Europe/Copenhagen"),
            "Datetime(time_unit='us', time_zone='UTC')",
        ),
        (pl.Series([[dt.date(2026, 1, 1)]]), "List(String)"),
        (pl.Series([[Decimal("1.5")]], dtype=pl.List(pl.Decimal(10, 1))), "List(String)"),
        (
            pl.Series([{"amount": Decimal("1.5")}], dtype=pl.Struct({"amount": pl.Decimal(10, 1)})),
            "Struct({'amount': String})",
        ),
        (pl.Series([[b"ab"]]), "List(String)"),
    ],
)
def test_a_returned_table_column_a_saved_run_reads_back_changed_is_refused(
    column: pl.Series, saved_as: str
):
    @node("odd", title="Odd", description="For a test.")
    def odd(text: str) -> Table:
        return pl.DataFrame({"a": column})

    run = Graph.compile(make_graph(odd), registry_of(odd)).run({"text": "x"})

    assert (run.cause.type, run.cause.loc) == ("invalid_output", ("odd",))
    assert (
        f"column 'a' is {column.dtype}, and a saved run would read it back as {saved_as}"
        in run.cause.msg
    )


class _Count(TypedDict):
    count: int


class _Clock(TypedDict):
    count: dt.time


_NANOSECOND = pl.Series(
    [1], dtype=pl.Int64
)  # one nanosecond, finer than Python's time, datetime and timedelta, which stop at microseconds


class _Moment(TypedDict):
    count: dt.datetime


class _Span(TypedDict):
    count: dt.timedelta


class _Numbers(TypedDict):
    count: list[float]


@pytest.mark.parametrize(
    ("column", "row", "fault"),
    [
        (
            pl.Series([2**64 - 1], dtype=pl.UInt64),
            _Count,
            "has a whole number too big for an Int64",
        ),
        (_NANOSECOND.cast(pl.Time), _Clock, "has a time finer than a microsecond"),
        (_NANOSECOND.cast(pl.Datetime("ns")), _Moment, "has a time finer than a microsecond"),
        (_NANOSECOND.cast(pl.Duration("ns")), _Span, "has a time finer than a microsecond"),
        (
            pl.Series([2**62], dtype=pl.Int64).cast(pl.Datetime("ms")),
            _Moment,
            "has a time too far out to be held in microseconds",
        ),
        (
            pl.Series([[1.5, math.nan]]),
            _Numbers,
            "has a float that isn't finite (nan or inf) inside a list or struct",
        ),
        (
            pl.Series([[math.inf]]),
            _Numbers,
            "has a float that isn't finite (nan or inf) inside a list or struct",
        ),
    ],
)
def test_a_returned_table_cell_a_saved_run_reads_back_otherwise_is_refused(
    column: pl.Series, row: Any, fault: str
):
    @node("any-columns", title="Any columns", description="For a test.")
    def any_columns(text: str) -> Table:
        return pl.DataFrame({"count": column})

    @node("declared-columns", title="Declared columns", description="For a test.")
    def declared_columns(text: str) -> Table[row]:  # type: ignore[valid-type]  # the row type is a parameter of the test
        return pl.DataFrame({"count": column})

    for returns in (any_columns, declared_columns):
        run = Graph.compile(make_graph(returns), registry_of(returns)).run({"text": "x"})

        assert (run.cause.type, run.cause.loc) == ("invalid_output", (returns.id,))
        assert (
            f"column 'count' {fault}" in run.cause.msg
            or f"column 'count' is {column.dtype}, and {fault}" in run.cause.msg
        )


def test_a_struct_column_with_a_float_that_is_not_finite_is_refused_and_a_whole_number_past_int64_inside_a_list_too():
    @node("rows", title="Rows", description="For a test.")
    def rows(text: str) -> Table:
        return (
            pl.DataFrame({"point": [{"x": 1.5}, {"x": math.nan}]})
            if text == "nan"
            else pl.DataFrame({"big": pl.Series([[2**64 - 1]], dtype=pl.List(pl.UInt64))})
        )

    graph = Graph.compile(make_graph(rows), registry_of(rows))

    assert (
        "column 'point' has a float that isn't finite (nan or inf) inside a list or struct"
        in graph.run({"text": "nan"}).cause.msg
    )
    assert (
        "column 'big' is List(UInt64), and has a whole number too big for an Int64"
        in graph.run({"text": "big"}).cause.msg
    )


def test_a_frame_given_as_a_graph_input_is_kept_as_a_saved_run_reads_it_back():
    @node("rows", title="Rows", description="For a test.")
    def rows(table: Table) -> Table:
        return table

    graph = Graph.compile(make_graph(rows), registry_of(rows))

    run = graph.run({"table": pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)})})
    assert run.results["rows"]["result"].schema == {"a": pl.Int64}

    with pytest.raises(InvalidCallError) as refused:
        graph.run(
            {
                "table": pl.DataFrame(
                    {
                        "a": pl.Series([dt.datetime(2026, 1, 1)]).dt.replace_time_zone(
                            "Europe/Copenhagen"
                        )
                    }
                )
            }
        )
    (problem,) = refused.value.problems
    assert (
        problem.type == "invalid_input"
        and "column 'a' is Datetime(time_unit='us', time_zone='Europe/Copenhagen')" in problem.msg
    )


@pytest.mark.parametrize(
    ("records", "read"),
    [
        ([{"a": 1}, {"a": 2.5}], {"a": [1.0, 2.5]}),
        ([{"a": 1}, {"a": "x"}], {"a": ["1", "x"]}),
        ([{"a": [1]}, {"a": [1.5]}], {"a": [[1.0], [1.5]]}),
        ([{"a": {"x": 1}}, {"a": {"y": 2}}], {"a": [{"x": 1, "y": None}, {"x": None, "y": 2}]}),
    ],
)
def test_rows_given_for_a_table_are_read_as_polars_reads_rows_one_type_per_column(
    records: list[dict[str, Any]], read: dict[str, Any]
):
    @node("rows", title="Rows", description="For a test.")
    def rows(table: Table) -> Table:
        return table

    run = Graph.compile(make_graph(rows), registry_of(rows)).run({"table": records})

    assert (
        run.results["rows"]["result"].to_dict(as_series=False)
        == _round_trip(run).results["rows"]["result"].to_dict(as_series=False)
        == read
    )


@pytest.mark.parametrize(
    "column",
    [
        pl.Series([[{"y": None}], [{"y": 5}]], dtype=pl.List(pl.Struct({"y": pl.Int64}))),
        pl.Series([[[]], [[5]]], dtype=pl.List(pl.List(pl.Int64))),
    ],
)
def test_a_nested_column_whose_first_cells_have_less_in_them_is_loaded_back_unchanged(
    column: pl.Series,
):
    @node("nested", title="Nested", description="For a test.")
    def nested(text: str) -> Table:
        return pl.DataFrame({"a": column})

    run = Graph.compile(make_graph(nested), registry_of(nested)).run({"text": "x"})
    read_back = _round_trip(run).results["nested"]["result"]

    assert read_back.schema == {"a": column.dtype} and read_back["a"].to_list() == column.to_list()


@pytest.mark.parametrize(
    ("values", "read"),
    [
        ([[1], [1.5]], [[1.0], [1.5]]),
        ([{"x": 1}, {"y": 2}], [{"x": 1, "y": None}, {"x": None, "y": 2}]),
    ],
)
def test_a_json_column_of_a_table_given_as_json_takes_a_type_that_fits_every_cell(
    values: list[Any], read: list[Any]
):
    @node("rows", title="Rows", description="For a test.")
    def rows(table: Table) -> Table:
        return table

    run = Graph.compile(make_graph(rows), registry_of(rows)).run(
        {"table": {"columns": [{"name": "a", "type": "json", "values": values}]}}
    )

    assert run.results["rows"]["result"]["a"].to_list() == read


def test_rows_given_for_a_table_keep_every_key_of_a_mapping_column():
    class Counts(TypedDict):
        m: dict[str, int]

    @node("rows", title="Rows", description="For a test.")
    def rows(table: Table[Counts]) -> int:
        return sum(
            len({key for key, count in (cell or {}).items() if count is not None})
            for cell in table["m"].to_list()
        )

    run = Graph.compile(make_graph(rows), registry_of(rows)).run(
        {"table": [{"m": {"x": 1}}, {"m": {"y": 2}}]}
    )

    assert run.results["rows"]["result"] == 2


@pytest.mark.parametrize(
    ("answer", "fault"),
    [
        (
            pl.DataFrame({"a": pl.Series([1], dtype=pl.Int64).cast(pl.Datetime("ns"))}),
            "column 'a' has a time finer than a microsecond",
        ),
        (
            pl.DataFrame(
                {
                    "a": pl.Series([dt.datetime(2026, 1, 1)]).dt.replace_time_zone(
                        "Europe/Copenhagen"
                    )
                }
            ),
            "column 'a' is Datetime(time_unit='us', time_zone='Europe/Copenhagen')",
        ),
    ],
)
def test_a_table_answer_is_refused_where_a_returned_table_is(answer: pl.DataFrame, fault: str):
    @node("ask-table", title="Ask table", description="For a test.", asks=True)
    def ask_table(text: str) -> Table:
        raise Ask("Which rows?")

    run = Graph.compile(make_graph(ask_table), registry_of(ask_table)).run({"text": "x"})
    (waiting,) = run.waiting

    with pytest.raises(InvalidCallError) as refused:
        run.with_answers(waiting.id, answer)

    (problem,) = refused.value.problems
    assert problem.type == "invalid_answer" and fault in problem.msg


def test_a_table_answer_is_kept_as_a_returned_table_is():
    @node("ask-table", title="Ask table", description="For a test.", asks=True)
    def ask_table(text: str) -> Table:
        raise Ask("Which rows?")

    run = Graph.compile(make_graph(ask_table), registry_of(ask_table)).run({"text": "x"})
    (waiting,) = run.waiting
    answered = run.with_answers(
        waiting.id, pl.DataFrame({"a": pl.Series([1], dtype=pl.Int32)})
    ).resume()

    assert (
        answered.results["ask-table"]["result"].schema
        == _round_trip(answered).results["ask-table"]["result"].schema
        == {"a": pl.Int64}
    )


def test_a_series_of_records_returned_as_a_frame_is_kept_as_a_saved_run_reads_it_back():
    @node("records", title="Records", description="For a test.")
    def records(text: str) -> Series[_Count]:
        return pl.DataFrame({"count": pl.Series([1, 2], dtype=pl.Int32)})  # type: ignore[return-value]  # a frame is a Series of records

    run = Graph.compile(make_graph(records), registry_of(records)).run({"text": "x"})

    assert (
        run.results["records"]["result"]
        == _round_trip(run).results["records"]["result"]
        == [{"count": 1}, {"count": 2}]
    )


def test_a_returned_table_reads_back_with_the_same_column_types_and_cells():
    columns = {
        "flag": pl.Series([True, None]),
        "count": pl.Series([2**63 - 1, None]),
        "ratio": pl.Series([math.nan, -0.0]),
        "at": pl.Series([dt.datetime(2026, 1, 1), None]).dt.replace_time_zone("UTC"),
        "nothing": pl.Series([None, None]),
        "amount": pl.Series([Decimal("1.50"), Decimal("-2.25")]),
        "point": pl.Series([{"x": 1}, {"x": None}]),
    }

    @node("kept", title="Kept", description="For a test.")
    def kept(text: str) -> Table:
        return pl.DataFrame(columns)

    run = Graph.compile(make_graph(kept), registry_of(kept)).run({"text": "x"})
    before, after = run.results["kept"]["result"], _round_trip(run).results["kept"]["result"]

    assert after.schema == before.schema == pl.DataFrame(columns).schema
    assert str(after.to_dict(as_series=False)) == str(
        before.to_dict(as_series=False)
    )  # compared as text, so a NaN equals itself


@pytest.mark.slow
def test_reading_a_table_costs_less_than_writing_it():
    """Reading a table back from its JSON costs less than writing it.

    Whether a table's JSON reads back is decided by its column types. So reading writes nothing out, and only casts
    the columns it has to convert. Reading a 2,000-row table of ten columns, half of them ``Int32``, should cost less
    than writing it as JSON. A reader that wrote each table out and read it back would cost more than the write.
    """
    table = pl.DataFrame(
        {
            f"c{i}": pl.Series(range(2_000), dtype=pl.Int32)
            if i % 2
            else [str(n) for n in range(2_000)]
            for i in range(10)
        }
    )
    tables = TypeAdapter(Table)
    kept = tables.validate_python(table)

    def fastest(call: Any, times: int) -> float:
        took = []
        for _ in range(3):
            started = time.perf_counter()
            for _ in range(times):
                call()
            took.append(time.perf_counter() - started)
        return min(took)

    read = fastest(lambda: tables.validate_python(table), 100)
    written = fastest(lambda: tables.dump_json(kept), 100)

    assert read < written, (read, written)


def test_tables_of_two_nodes_read_back_each_as_itself():
    class Name(TypedDict):
        name: str

    class Age(TypedDict):
        age: int

    @node("names", title="Names", description="For a test.")
    def names(text: str) -> Table[Name]:
        return pl.DataFrame({"name": ["ida", "bo"]})

    @node("ages", title="Ages", description="For a test.")
    def ages(text: str) -> Table[Age]:
        return pl.DataFrame({"age": [30, 40, 50]})

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"n_text": {}, "a_text": {}},
                "nodes": {
                    "n": {"node": "names", "inputs": {"text": {"graph_input": "n_text"}}},
                    "a": {"node": "ages", "inputs": {"text": {"graph_input": "a_text"}}},
                },
                "outputs": {
                    "a_result": {"node": "a", "output": "result"},
                    "n_result": {"node": "n", "output": "result"},
                },
            }
        ),
        registry_of(names, ages),
    ).run({"n_text": "x", "a_text": "y"})

    read_back = _round_trip(run)

    assert read_back.results["n"]["result"].to_dicts() == [{"name": "ida"}, {"name": "bo"}]
    assert read_back.results["a"]["result"].to_dicts() == [{"age": 30}, {"age": 40}, {"age": 50}]


def test_two_identical_runs_save_byte_identical_json():
    @node("mixed", title="Mixed", description="For a test.")
    def mixed(text: str) -> Series[float]:
        return [1.5, math.nan, 2.0]

    graph = Graph.compile(make_graph(mixed), registry_of(split, mixed))

    assert graph.run({"text": "x"}).to_json() == graph.run({"text": "x"}).to_json()


# -- bytes are base64 in JSON ---------------------------------------------------------------

#: Bytes that aren't UTF-8 text, and their URL-safe base64 text with its padding.
_BINARY, _BINARY_TEXT = b"\xff\xfe\x00A\xfb", "__4AQfs="


class _Attachment(TypedDict):
    data: bytes


@node("checksum", title="Checksum", description="For a test.")
def checksum(data: bytes) -> bytes:
    return bytes(255 - byte for byte in data)


@node("attach", title="Attach", description="For a test.")
def attach(text: str) -> _Attachment:
    return {"data": _BINARY + text.encode()}


@node("attach-all", title="Attach all", description="For a test.")
def attach_all(text: str) -> list[_Attachment]:
    return [{"data": _BINARY + text.encode()}]


@node("approve-file", title="Approve file", description="For a test.", asks=True)
def approve_file(data: bytes) -> bytes:
    raise Ask("Which file?")


def test_a_binary_input_is_given_as_its_base64_text_or_as_bytes_and_other_text_is_refused():
    """A caller gives a ``bytes`` input as base64 text, standard or URL-safe, or as the bytes themselves.

    JSON has no bytes, so a binary value crosses JSON as base64 text, and one given from Python reads as the same
    value sent over HTTP.
    """
    graph = Graph.compile(make_graph(checksum), registry_of(checksum))
    flipped = bytes(255 - byte for byte in _BINARY)

    for given in (_BINARY, _BINARY_TEXT, "//4AQfs="):
        run = graph.run({"data": given})
        assert run.results["checksum"]["result"] == flipped, given
        assert _round_trip(run).results["checksum"]["result"] == flipped
    assert json.loads(run.outputs_to_json()) == {"result": "AAH_vgQ="}
    assert json.loads(run.to_json())["inputs"] == {"data": _BINARY_TEXT}

    with pytest.raises(InvalidCallError) as refused:
        graph.run({"data": "not base64!"})
    (problem,) = refused.value.problems
    assert (problem.type, problem.ctx["errors"][0]["type"]) == (
        "invalid_input",
        "bytes_invalid_encoding",
    )


def test_bytes_a_node_returns_are_saved_as_base64_and_read_back_in_a_record_too():
    """Bytes that aren't text are saved, in a record's field and in a list of records."""
    for returning, saved, returned in (
        (attach, {"data": "__4AQfth"}, {"data": _BINARY + b"a"}),
        (attach_all, [{"data": "__4AQfth"}], [{"data": _BINARY + b"a"}]),
    ):
        run = Graph.compile(make_graph(returning), registry_of(returning)).run({"text": "a"})

        outcomes = json.loads(run.to_json())["outcomes"]
        assert outcomes[returning.id]["outputs"]["result"] == [saved]
        assert _round_trip(run).results[returning.id]["result"] == returned


def test_a_record_given_with_a_bytes_field_reads_the_field_from_base64():
    @node("attached-size", title="Attached size", description="For a test.")
    def attached_size(file: _Attachment) -> int:
        return len(file["data"])

    graph = Graph.compile(make_graph(attached_size), registry_of(attached_size))

    for given in (_BINARY, _BINARY_TEXT):
        run = graph.run({"file": {"data": given}})
        assert run.results["attached-size"]["result"] == len(_BINARY), given


def test_a_person_answers_a_binary_question_with_base64_text_and_a_queued_answer_is_its_base64():
    graph = Graph.compile(make_graph(checksum, approve_file), registry_of(checksum, approve_file))
    run = graph.run({"data": _BINARY})
    (waiting,) = run.waiting

    assert waiting.inputs == {"data": "AAH_vgQ="}
    assert run.graph.answers_as_json({waiting.id: _BINARY}) == {waiting.id: _BINARY_TEXT}
    for answer in (_BINARY, _BINARY_TEXT):
        answered = run.with_answers(waiting, answer).resume()
        assert answered.results["approve-file"]["result"] == _BINARY, answer
        assert _round_trip(answered).results["approve-file"]["result"] == _BINARY


def test_a_call_on_bytes_is_matched_by_their_base64_text_given_as_bytes_or_as_text():
    """A call's arguments are hashed by their JSON text, so bytes and their base64 text make one waiting id."""
    graph = Graph.compile(make_graph(approve_file), registry_of(approve_file))

    (as_bytes,) = graph.run({"data": _BINARY}).waiting
    (as_text,) = graph.run({"data": _BINARY_TEXT}).waiting

    assert as_bytes.id == as_text.id


def test_a_binary_constant_and_default_are_written_as_base64_and_described_as_base64url():
    @node("padded-size", title="Padded size", description="For a test.")
    def padded_size(data: bytes = b"\xff\x00") -> int:
        return len(data)

    record = make_graph(padded_size, constants={"padded-size.data": _BINARY})

    assert record.nodes["padded-size"].inputs["data"].constant == _BINARY_TEXT
    assert Graph.compile(record, registry_of(padded_size)).run({}).results == {
        "padded-size": {"result": len(_BINARY)}
    }
    (described,) = padded_size.describe().inputs
    assert (described.json_schema, described.default) == (
        {"format": "base64url", "type": "string"},
        "_wA=",
    )


def test_binary_choices_are_written_and_compared_as_base64():
    @node("pick-bytes", title="Pick bytes", description="For a test.")
    def pick_bytes(data: Annotated[bytes, Param(choices=[b"\xff", b"a"])]) -> bytes:
        return data

    graph = Graph.compile(make_graph(pick_bytes), registry_of(pick_bytes))

    assert pick_bytes.describe().inputs[0].json_schema["enum"] == ["_w==", "YQ=="]
    assert graph.run({"data": "_w=="}).results["pick-bytes"]["result"] == b"\xff"
    with pytest.raises(InvalidCallError) as refused:
        graph.run({"data": b"b"})
    assert refused.value.problems[0].type == "invalid_input"


# -- a table's bytes column is base64 in JSON, as every bytes value is -------------------------


class _Blob(TypedDict):
    name: str
    data: bytes


#: A frame with a ``Binary`` column, as a node returns it, and the column as ``JSON_CONFIG`` writes each cell.
_BLOBS = {"name": ["a", "b"], "data": [_BINARY, b"hi"]}
_BLOBS_WRITTEN = [
    {"name": "name", "type": "str", "values": ["a", "b"]},
    {
        "name": "data",
        "type": "bytes",
        "values": [
            json.loads(TypeAdapter(bytes, config=JSON_CONFIG).dump_json(one))
            for one in _BLOBS["data"]
        ],
    },
]


@node("blobs", title="Blobs", description="For a test.")
def blobs(text: str) -> Table[_Blob]:
    return pl.DataFrame(_BLOBS)


@node("blob-rows", title="Blob rows", description="For a test.")
def blob_rows(text: str) -> Series[_Blob]:
    return pl.DataFrame(_BLOBS)  # type: ignore[return-value]  a table may stand for a Series of its rows


@node("blob-sizes", title="Blob sizes", description="For a test.")
def blob_sizes(table: Table[_Blob]) -> list[int]:
    assert table.schema["data"] == pl.Binary
    return [len(cell) for cell in table["data"]]


@node("blob-size", title="Blob size", description="For a test.")
def blob_size(blob: _Blob) -> int:
    return len(blob["data"])


def test_a_binary_column_a_node_returns_is_saved_as_base64_and_reads_back_as_bytes():
    """Each cell is the base64 text ``JSON_CONFIG`` writes for the same bytes alone, ``"__4AQfs="`` for ``_BINARY``."""
    graph = Graph.compile(make_graph(blobs, blob_sizes), registry_of(blobs, blob_sizes))
    run = graph.run({"text": "x"})

    stored = run.results["blobs"]["result"]
    assert (stored.schema["data"], stored.to_dict(as_series=False)) == (pl.Binary, _BLOBS)
    assert _BLOBS_WRITTEN[1]["values"] == [_BINARY_TEXT, "aGk="]
    saved = json.loads(run.to_json())["outcomes"]
    assert saved["blobs"]["outputs"]["result"] == [{"columns": _BLOBS_WRITTEN}]
    read_back = _round_trip(run).results
    assert read_back["blobs"]["result"].schema["data"] == pl.Binary
    assert read_back["blobs"]["result"].to_dict(as_series=False) == _BLOBS
    assert read_back["blob-sizes"]["result"] == run.results["blob-sizes"]["result"] == [5, 2]


def test_a_table_with_a_bytes_field_is_given_as_base64_by_a_caller_and_a_person():
    """Rows, the JSON of a table and a frame each read the field into a ``Binary`` column of the bytes."""

    @node("pick-blobs", title="Pick blobs", description="For a test.", asks=True)
    def pick_blobs(text: str) -> Table[_Blob]:
        raise Ask("Which files?")

    given = Graph.compile(make_graph(blob_sizes), registry_of(blob_sizes))
    rows = [{"name": "a", "data": _BINARY_TEXT}, {"name": "b", "data": "aGk="}]
    for table in (rows, {"columns": _BLOBS_WRITTEN}, pl.DataFrame(_BLOBS)):
        run = given.run({"table": table})
        assert run.results["blob-sizes"]["result"] == [5, 2], table
        assert run.inputs["table"].to_dict(as_series=False) == _BLOBS
        assert json.loads(run.to_json())["inputs"] == {"table": {"columns": _BLOBS_WRITTEN}}

    asked = Graph.compile(make_graph(pick_blobs), registry_of(pick_blobs)).run({"text": "x"})
    (waiting,) = asked.waiting
    assert asked.graph.answers_as_json({waiting.id: pl.DataFrame(_BLOBS)}) == {
        waiting.id: {"columns": _BLOBS_WRITTEN}
    }
    for answer in (rows, pl.DataFrame(_BLOBS)):
        answered = asked.with_answers(waiting, answer).resume()
        assert answered.results["pick-blobs"]["result"].to_dict(as_series=False) == _BLOBS
        assert (
            _round_trip(answered).results["pick-blobs"]["result"].to_dict(as_series=False) == _BLOBS
        )


@with_config(ConfigDict(**JSON_CONFIG, str_strip_whitespace=True))
class _TrimmedBlob(TypedDict):
    name: str
    data: bytes


def test_a_table_row_is_read_by_its_own_config_as_the_same_typed_dict_alone_is():
    """A row type's ``with_config`` holds for each row, so its text is trimmed and its bytes still read base64."""

    @node("trimmed", title="Trimmed", description="For a test.")
    def trimmed(table: Table[_TrimmedBlob], alone: _TrimmedBlob) -> list[_TrimmedBlob]:
        return [*table.to_dicts(), alone]

    graph = Graph.compile(make_graph(trimmed), registry_of(trimmed))
    row = {"name": "  a ", "data": _BINARY_TEXT}
    run = graph.run({"table": [row], "alone": row})

    assert run.results["trimmed"]["result"] == [{"name": "a", "data": _BINARY}] * 2


def test_a_blank_cell_of_an_optional_bytes_field_reads_back_blank():
    class MaybeBlob(TypedDict):
        data: bytes | None

    @node("maybe-blobs", title="Maybe blobs", description="For a test.")
    def maybe_blobs(text: str) -> Table[MaybeBlob]:
        return pl.DataFrame({"data": [_BINARY, None]})

    run = Graph.compile(make_graph(maybe_blobs), registry_of(maybe_blobs)).run({"text": "x"})

    (column,) = json.loads(run.to_json())["outcomes"]["maybe-blobs"]["outputs"]["result"][0][
        "columns"
    ]
    assert column == {"name": "data", "type": "bytes", "values": [_BINARY_TEXT, None]}
    read_back = _round_trip(run).results["maybe-blobs"]["result"]
    assert (read_back.schema["data"], read_back["data"].to_list()) == (pl.Binary, [_BINARY, None])


def test_a_table_with_a_bytes_field_is_described_as_base64url_cells():
    (described,) = blob_sizes.describe().inputs
    assert described.type == "Table[name: str, data: bytes]"
    assert described.json_schema["properties"]["columns"]["prefixItems"][1] == {
        "type": "object",
        "properties": {
            "name": {"const": "data"},
            "values": {"type": "array", "items": {"type": "string", "format": "base64url"}},
        },
    }


def test_a_series_of_rows_with_a_bytes_field_returned_as_a_frame_is_saved_as_base64():
    graph = Graph.compile(make_graph(blob_rows, blob_size), registry_of(blob_rows, blob_size))
    run = graph.run({"text": "x"})

    assert run.results["blob-size"]["result"] == [5, 2]
    saved = json.loads(run.to_json())["outcomes"]
    assert saved["blob-rows"]["outputs"]["result"] == [{"columns": _BLOBS_WRITTEN}]
    read_back = _round_trip(run).results
    assert read_back["blob-rows"]["result"] == [
        dict(zip(_BLOBS, row)) for row in zip(*_BLOBS.values())
    ]
    assert read_back["blob-size"]["result"] == [5, 2]


_BLOBS_IN_ANOTHER_PROCESS = """
import polars as pl
from typing import TypedDict
from conductor import Graph, NodeRegistry, Table, make_graph, node

class Blob(TypedDict):
    data: bytes

@node("blobs", title="Blobs", description="For a test.")
def blobs(text: str) -> Table[Blob]:
    return pl.DataFrame({"data": [b"\\xff\\xfe\\x00A\\xfb", b"hi"]})

@node("count", title="Count", description="For a test.")
def count(table: Table[Blob]) -> int:
    return table.height

registry = NodeRegistry()
registry.add(blobs, count)
print(Graph.compile(make_graph(blobs, count), registry).run({"text": "x"}).to_json())
"""


def test_a_call_given_a_binary_column_has_the_same_fingerprint_in_every_process():
    """The table's digest, which the fingerprint of ``count``'s call is made from, hashes the column's base64 text."""
    saved = {
        subprocess.run(
            [sys.executable, "-c", _BLOBS_IN_ANOTHER_PROCESS],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    (one,) = saved
    assert json.loads(one)["outcomes"]["count"]["fingerprints"]


@host_type("text-attachment", title="Text attachment")
class _TextAttachment(BaseModel, frozen=True):
    """A host's model with pydantic's default config, which reads and writes bytes as text."""

    data: bytes


@host_type("text-attachment-record", title="Text attachment record")
@pydantic_dataclass(frozen=True)
class _TextAttachmentRecord:
    data: bytes


@host_type("binary-attachment", title="Binary attachment")
class _BinaryAttachment(BaseModel, frozen=True):
    """A host's model whose config reads and writes bytes as base64, as conductor does."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")
    data: bytes


class _Holder(TypedDict):
    attachment: _TextAttachment


@host_type("binary-folder", title="Binary folder")
class _BinaryFolder(BaseModel, frozen=True):
    """A model that reads bytes as base64, holding one that reads them as text."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")
    attachment: _TextAttachment


@with_config(ConfigDict(extra="forbid"))
class _StrictAttachmentRow(TypedDict):
    """A ``TypedDict`` with a config of its own, which reads its bytes as text."""

    data: bytes


@host_type("strict-attachment-record", title="Strict attachment record")
@with_config(ConfigDict(extra="forbid"))
@dataclass(frozen=True)
class _StrictAttachmentRecord:
    """A plain dataclass with a config of its own, which reads its bytes as text."""

    data: bytes


@pytest.mark.parametrize(
    ("hint", "where"),
    [
        (_StrictAttachmentRow, "_StrictAttachmentRow"),
        (list[_StrictAttachmentRow], "_StrictAttachmentRow"),
        (Table[_StrictAttachmentRow], "_StrictAttachmentRow"),
        (_StrictAttachmentRecord, "_StrictAttachmentRecord"),
        (list[_StrictAttachmentRecord], "_StrictAttachmentRecord"),
        (_TextAttachment, "_TextAttachment"),
        (list[_TextAttachment], "_TextAttachment"),
        (_TextAttachment | None, "_TextAttachment"),
        (list[_TextAttachmentRecord], "_TextAttachmentRecord"),
        (_Holder, "_TextAttachment"),
        (_BinaryFolder, "_TextAttachment"),
    ],
)
@pytest.mark.parametrize("on", ["input", "output"])
def test_a_host_model_that_reads_its_bytes_as_text_is_refused_where_the_node_is_declared(
    hint: Any, where: str, on: str
) -> None:
    """A dict given for such a model, as a run's input or a record's constant, would read the base64 text as bytes."""
    with pytest.raises(NodeDeclarationError) as raised:
        if on == "input":

            @node("upload", title="Upload", description="For a test.")
            def upload(file: hint) -> int:  # type: ignore[valid-type]
                return 1

        else:

            @node("upload", title="Upload", description="For a test.")
            def upload_out(text: str) -> hint:  # type: ignore[valid-type]
                raise AssertionError

    (problem,) = raised.value.problems
    assert (problem.type, problem.ctx) == (
        "type_not_supported_bytes_as_text",
        {"cls": where, "field": "data"},
    )


def test_bytes_read_as_text_name_the_class_whose_config_is_at_fault():
    """A ``TypedDict`` with no config of its own reads by the config it sits in, so the class to fix is the model around
    it, and the field named is that model's."""

    class Plain(TypedDict):
        data: bytes

    class Cat(TypedDict):
        kind: Literal["cat"]
        data: bytes

    class Dog(TypedDict):
        kind: Literal["dog"]

    @host_type("text-holds", title="Text holds")
    class TextHolds(BaseModel, frozen=True):
        inner: Plain

    @host_type("text-union", title="Text union")
    class TextUnion(BaseModel, frozen=True):
        pet: Annotated[Cat | Dog, Discriminator("kind")]

    for hint, named in (
        (TextHolds, {"cls": "TextHolds", "field": "inner"}),
        (TextUnion, {"cls": "TextUnion", "field": "pet"}),
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            node("upload", title="Upload", description="For a test.")(_taking(hint))
        (problem,) = raised.value.problems
        assert (problem.type, problem.ctx) == ("type_not_supported_bytes_as_text", named), hint


def _taking(hint: Any) -> Any:
    def upload(file: Any) -> int:
        return 1

    upload.__annotations__ = {"file": hint, "return": int}
    return upload


def test_a_host_model_that_reads_its_bytes_as_base64_takes_them_at_every_door():
    @node("attachment-size", title="Attachment size", description="For a test.", asks=True)
    def attachment_size(files: list[_BinaryAttachment]) -> _BinaryAttachment:
        if files[0].data == b"ask":
            raise Ask("Which file?")
        return _BinaryAttachment(data=b"".join(file.data for file in files))

    registry = registry_of(attachment_size)
    graph = Graph.compile(make_graph(attachment_size), registry)
    returned = _BinaryAttachment(data=_BINARY + _BINARY)

    for given in ({"data": _BINARY}, {"data": _BINARY_TEXT}, _BinaryAttachment(data=_BINARY)):
        run = graph.run({"files": [given, given]})
        assert run.results["attachment-size"]["result"] == returned, given
        assert _round_trip(run).results["attachment-size"]["result"] == returned
    record = make_graph(attachment_size, constants={"attachment-size.files": [{"data": _BINARY}]})
    assert record.nodes["attachment-size"].inputs["files"].constant == [{"data": _BINARY_TEXT}]
    assert Graph.compile(record, registry).run({}).results["attachment-size"]["result"] == (
        _BinaryAttachment(data=_BINARY)
    )
    waiting_run = graph.run({"files": [{"data": b"ask"}]})
    (waiting,) = waiting_run.waiting
    answered = waiting_run.with_answers(waiting, {"data": _BINARY_TEXT}).resume()
    assert _round_trip(answered).results["attachment-size"]["result"] == _BinaryAttachment(
        data=_BINARY
    )


# -- what a run cannot be saved with ----------------------------------------------------------


@host_type("_Opaque", title="Opaque")
class _Opaque:
    """A value pydantic accepts but cannot write as JSON."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.is_instance_schema(cls)


def test_saving_a_run_with_an_output_that_has_no_json_form_raises_naming_the_output():
    @node("opaque", title="Opaque", description="For a test.")
    def opaque(text: str) -> _Opaque:
        return _Opaque()

    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {"o": {"node": "opaque", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "o", "output": "result"}},
            }
        ),
        registry_of(opaque),
    ).run({"text": "x"})

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx["where"]) == (
        "run_unwritable",
        ("saved_run", "o.result"),
        "o.result",
    )


def test_saving_a_run_whose_node_has_outcomes_and_reusable_outcomes_names_the_output_with_no_json_form():
    """Saving names the output with no JSON form, even when its node also has outcomes it may reuse.

    The value with no JSON form is in this run's outcomes of ``odd``. ``odd`` also has outcomes from before that it
    may reuse.
    """

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("odd", title="Odd", description="For a test.", asks=True)
    def odd(word: str) -> Any:
        if word == "q":
            raise Ask("q?")
        return "fine" if word in ("a", "c") else _Opaque()

    graph = Graph.compile(
        make_graph(split, odd, edges=[("split.result", "odd.word")]), registry_of(split, odd)
    )
    second = graph.run({"text": "a,c,q"}).with_inputs({"text": "b,q"}).resume()

    with pytest.raises(InvalidCallError) as refused:
        second.to_json()

    (problem,) = refused.value.problems
    assert (problem.type, problem.ctx["where"]) == ("run_unwritable", "odd.result")


def test_an_input_with_no_json_form_is_refused_where_it_is_given_so_every_run_can_be_saved():
    @node("take", title="Take", description="For a test.")
    def take(value: Any) -> str:
        return "ok"

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"value": {}},
                "nodes": {"t": {"node": "take", "inputs": {"value": {"graph_input": "value"}}}},
                "outputs": {"result": {"node": "t", "output": "result"}},
            }
        ),
        registry_of(take),
    )

    with pytest.raises(InvalidCallError) as refused:
        graph.run({"value": _Opaque()})

    (problem,) = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx["reason"]) == (
        "invalid_input",
        ("inputs", "value"),
        "the value can't be written as JSON",
    )


def test_a_returned_table_column_a_saved_run_reads_back_as_text_is_refused_naming_the_column():
    """Bytes inside a list are written as plain JSON, so they would read back as their base64 text."""

    class Blobs(TypedDict):
        data: list[bytes]

    @node("blob", title="Blob", description="For a test.")
    def blob(text: str) -> Table[Blobs]:
        return pl.DataFrame({"data": [[b"\xff\xfe"]]})

    run = Graph.compile(make_graph(blob), registry_of(blob)).run({"text": "x"})

    assert (run.cause.type, run.cause.loc) == ("invalid_output", ("blob",))
    assert "column 'data' is List(Binary)" in run.cause.msg and "\\xff" not in json.dumps(
        run.cause.model_dump(mode="json")
    )


# -- a saved run that does not fit its graph is refused ---------------------------------------


def _saved_echoes() -> tuple[Any, dict[str, Any]]:
    """Returns a graph and its complete saved run, as a dict.

    In the graph, ``split`` returns three items, and ``echo`` is called once per item.
    """

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    graph = Graph.compile(make_graph(split, echo), registry_of(split, echo))
    return graph, json.loads(graph.run({"text": "secret-a,secret-b,secret-c"}).to_json())


def test_a_saved_value_its_type_does_not_read_is_refused_naming_where_and_quoting_nothing():
    graph, saved = _saved_echoes()
    saved["outcomes"]["echo"]["outputs"]["result"][1] = {"secret": "value"}

    problem = _refused(graph, saved)

    assert problem.ctx["where"] == "outcomes.echo.outputs.result.1"
    assert "secret" not in json.dumps(problem.model_dump(mode="json"))


class Counted(TypedDict):
    n: int


@pytest.mark.parametrize(
    ("damage", "at"),
    [
        ("an int input", "inputs.x"),
        ("a float input", "inputs.f"),
        ("an int output", "outcomes.whole.outputs.result.0"),
        ("an item of a Series of ints", "outcomes.many.outputs.result.0"),
        ("an int field of a record", "outcomes.one.outputs.result.0"),
        ("a cell of an int column", "outcomes.table.outputs.result.0"),
    ],
)
def test_a_yes_or_no_saved_for_a_number_is_refused_with_the_reason_a_caller_gets(
    damage: str, at: str
) -> None:
    """A ``true`` saved where a number belongs is refused, with the same reason a caller would get.

    A store may hand back JSON that something else wrote. ``true`` is no more a number there than when a caller
    passes it.
    """

    @node("whole", title="Whole", description="For a test.")
    def whole(x: int, f: float) -> int:
        return x

    @node("many", title="Many", description="For a test.")
    def many(x: int) -> Series[int]:
        return [x, x]

    @node("one", title="One", description="For a test.")
    def one(x: int) -> Counted:
        return {"n": x}

    @node("table", title="Table", description="For a test.")
    def table(x: int) -> Table[Counted]:
        return pl.DataFrame({"n": [x]})

    from_x = {"x": {"graph_input": "x"}}
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"x": {}, "f": {}},
                "nodes": {
                    "whole": {"node": "whole", "inputs": {**from_x, "f": {"graph_input": "f"}}},
                    "many": {"node": "many", "inputs": from_x},
                    "one": {"node": "one", "inputs": from_x},
                    "table": {"node": "table", "inputs": from_x},
                },
                "outputs": {
                    name: {"node": name, "output": "result"}
                    for name in ("whole", "many", "one", "table")
                },
            }
        ),
        registry_of(whole, many, one, table),
    )
    saved = json.loads(graph.run({"x": 1, "f": 1.0}).to_json())
    outputs = {
        name: saved["outcomes"][name]["outputs"]["result"]
        for name in ("whole", "many", "one", "table")
    }
    if damage == "an int input":
        saved["inputs"]["x"] = True
    elif damage == "a float input":
        saved["inputs"]["f"] = True
    elif damage == "an int output":
        outputs["whole"][0] = True
    elif damage == "an item of a Series of ints":
        outputs["many"][0][1] = True
    elif damage == "an int field of a record":
        outputs["one"][0]["n"] = True
    else:
        outputs["table"][0]["columns"][0]["values"][0] = True

    problem = _refused(graph, saved)

    assert problem.ctx == {
        "where": at,
        "reason": "a yes or no is not a number",
        "errors": [
            {"type": "bool_is_not_a_number", "loc": [], "msg": "a yes or no is not a number"}
        ],
    }


def test_a_null_saved_for_a_done_call_whose_output_takes_no_none_is_refused():
    graph, saved = _saved_echoes()
    saved["outcomes"]["echo"]["outputs"]["result"][1] = None

    problem = _refused(graph, saved)

    reason = "call 1 is done and has null for 'result', whose type takes no None"
    assert problem.ctx == {
        "where": "outcomes.echo",
        "reason": reason,
        "errors": [{"type": "saved_lists_disagree", "loc": [], "msg": reason}],
    }


def test_an_output_list_shorter_than_states_is_refused_saying_which_lists_disagree():
    graph, saved = _saved_echoes()
    saved["outcomes"]["echo"]["outputs"]["result"].pop()

    problem = _refused(graph, saved)

    reason = "the list of an output is not as long as 'states'"
    assert problem.ctx == {
        "where": "outcomes.echo",
        "reason": reason,
        "errors": [{"type": "saved_lists_disagree", "loc": [], "msg": reason}],
    }
    assert problem.msg.endswith(f"{reason}. Start a new run.")


@pytest.mark.parametrize(
    ("damage", "at"),
    [
        ("one entry more than the items", "outcomes.echo"),
        ("one entry fewer than the items", "outcomes.echo"),
        ("two entries for a node inside no loop", "outcomes.split"),
        ("a missing value", "outcomes.echo"),
        ("a missing output", "outcomes.echo.outputs.result"),
        ("an output the node lacks", "outcomes.echo.outputs.other"),
        ("a node the graph lacks", "outcomes.nowhere"),
    ],
)
def test_a_damaged_saved_run_is_refused_where_it_is_damaged(damage: str, at: str) -> None:
    graph, saved = _saved_echoes()
    echo, splitting = saved["outcomes"]["echo"], saved["outcomes"]["split"]
    if damage == "one entry more than the items":
        echo["states"].append("done")
        echo["outputs"]["result"].append("x")
    elif damage == "one entry fewer than the items":
        echo["states"].pop()
        echo["outputs"]["result"].pop()
    elif damage == "two entries for a node inside no loop":
        splitting["states"].append("done")
        splitting["outputs"]["result"].append(["x"])
    elif damage == "a missing value":
        echo["outputs"]["result"].pop()
    elif damage == "a missing output":
        del echo["outputs"]["result"]
    elif damage == "an output the node lacks":
        echo["outputs"]["other"] = ["x", "y", "z"]
    else:
        saved["outcomes"]["nowhere"] = {"states": [], "outputs": {}}

    assert _refused(graph, saved).ctx["where"] == at


def test_a_series_saved_as_null_for_a_done_call_is_refused_at_the_node_that_returned_it():
    graph, saved = _saved_echoes()
    saved["outcomes"]["split"]["outputs"]["result"][0] = None

    assert _refused(graph, saved).ctx["where"] == "outcomes.split"


def test_a_null_item_in_a_saved_series_whose_items_take_no_none_is_refused_unless_it_is_listed_skipped():
    graph, saved = _saved_echoes()
    saved["outcomes"]["split"]["outputs"]["result"][0][1] = None

    problem = _refused(graph, saved)

    assert problem.ctx["where"] == "outcomes.split"
    saved["outcomes"]["split"]["skipped_items"] = {"result": {"0": [1]}}
    assert Run.from_json(json.dumps(saved), graph).results["split"]["result"] == [
        "secret-a",
        None,
        "secret-c",
    ]


def _not_none(value: Any) -> Any:
    if value is None:
        raise ValueError("no None here")
    return value


@pytest.mark.parametrize(
    "item",
    [
        Literal["a", None],
        Annotated[Optional[str], Param(title="t")],  # noqa: UP045
        Optional[Annotated[str, Param(title="t")]],  # noqa: UP045
        Any,
        object,
    ],
    ids=["literal-none", "annotated-optional", "optional-annotated", "any", "object"],
)
def test_a_null_item_a_node_may_return_reads_back(item: Any) -> None:
    @node("make", title="Make", description="For a test.")
    def make(text: str) -> Series[item]:  # type: ignore[valid-type]
        return ["a", None]

    graph = Graph.compile(make_graph(make), registry_of(make))
    run = graph.run({"text": "x"})

    assert run.status == "complete"
    assert Run.from_json(run.to_json(), graph).results["make"]["result"] == ["a", None]


@pytest.mark.parametrize("whole", [Literal["a", None], object], ids=["literal-none", "object"])
def test_a_null_output_a_node_may_return_reads_back(whole: Any) -> None:
    @node("one", title="One", description="For a test.")
    def one(text: str) -> whole:  # type: ignore[valid-type]
        return None

    graph = Graph.compile(make_graph(one), registry_of(one))
    run = graph.run({"text": "x"})

    assert run.status == "complete"
    assert Run.from_json(run.to_json(), graph).results["one"]["result"] is None


def test_a_null_item_a_node_may_not_return_is_refused_on_reading_as_its_own_check_refuses_it():
    @node("make", title="Make", description="For a test.")
    def make(text: str) -> Series[Annotated[str | None, AfterValidator(_not_none)]]:
        return ["a", "b"]

    graph = Graph.compile(make_graph(make), registry_of(make))
    saved = json.loads(graph.run({"text": "x"}).to_json())
    saved["outcomes"]["make"]["outputs"]["result"][0][1] = None

    assert _refused(graph, saved).ctx["where"] == "outcomes.make"


def _saved_table() -> tuple[Any, dict[str, Any]]:
    class Name(TypedDict):
        name: str

    @node("names", title="Names", description="For a test.")
    def names(text: str) -> Table[Name]:
        return pl.DataFrame({"name": text.split(",")})

    @node("count", title="Count", description="For a test.")
    def count(table: Table[Name]) -> int:
        return table.height

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "n": {"node": "names", "inputs": {"text": {"graph_input": "text"}}},
                    "c": {
                        "node": "count",
                        "inputs": {"table": {"sources": [{"node": "n", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "c", "output": "result"}},
            }
        ),
        registry_of(names, count),
    )
    return graph, json.loads(graph.run({"text": "ida,bo"}).to_json())


def test_a_malformed_table_is_refused():
    graph, saved = _saved_table()
    saved["outcomes"]["n"]["outputs"]["result"][0] = {"rows": 2, "columns": {"name": ["ida"]}}

    assert _refused(graph, saved).ctx["where"].startswith("outcomes.n.outputs.result.0")


def test_a_saved_run_is_refused_by_an_edited_graph_but_not_by_a_change_of_display_only():
    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    def with_default(default: str, **display: Any) -> GraphRecord:
        node_record = NodeRecord(
            node="shout",
            inputs={"text": input_entry({"graph_input": "text"})},
            display=display,
        )
        return GraphRecord(
            inputs={"text": GraphInputRecord(default=default)},
            nodes={"s": node_record},
            outputs={"result": Source("s", "result")},
        )

    record = with_default("a")
    saved = Graph.compile(record, registry_of(shout)).run().to_json()
    edited, moved = with_default("b"), with_default("a", x=1)

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(saved, Graph.compile(edited, registry_of(shout)))
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc) == ("saved_run_of_another_graph", ("saved_run",)) and set(
        problem.ctx
    ) == {"saved", "graph"}
    assert Run.from_json(saved, Graph.compile(moved, registry_of(shout))).results == {
        "s": {"result": "A"}
    }


@pytest.mark.parametrize(
    ("saved_graph", "quoted"),
    [
        ("0123456789abcdef", True),
        ("my secret value", False),
        ("0123456789ABCDEF", False),
        (42, False),
    ],
    ids=["a-fingerprint", "text", "upper-case-hex", "a-number"],
)
def test_another_graphs_saved_run_quotes_its_graph_only_when_it_is_a_fingerprint(
    saved_graph: Any, quoted: bool
):
    """``saved_run_of_another_graph`` keeps ``ctx.saved`` only for a value of a fingerprint's form.

    Every run conductor saves has one there, which tells a host which graph the run is of. A hand-edited save may
    hold any value, and a problem never quotes a value.
    """

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    graph = Graph.compile(make_graph(shout), registry_of(shout))
    saved = json.loads(graph.run({"text": "a"}).to_json())
    saved["graph"] = saved_graph

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)
    (problem,) = refused.value.problems
    expected = {"graph": graph.fingerprint, **({"saved": saved_graph} if quoted else {})}
    assert (problem.type, problem.ctx) == ("saved_run_of_another_graph", expected)
    assert quoted or str(saved_graph) not in problem.model_dump_json()


def test_a_saved_run_is_refused_by_a_graph_whose_subgraph_was_edited():
    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    @node("quiet", title="Quiet", description="For a test.")
    def quiet(text: str) -> str:
        return text.lower()

    def with_inner(definition: Any) -> Any:
        return registry_of(
            shout,
            quiet,
            subgraph("inner", title="Inner", description="For a test.")(make_graph(definition)),
        )

    outer = GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {"e": {"node": "inner", "inputs": {"text": {"graph_input": "text"}}}},
        }
    )
    saved = Graph.compile(outer, with_inner(shout)).run({"text": "a"}).to_json()

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(saved, Graph.compile(outer, with_inner(quiet)))
    assert [p.type for p in refused.value.problems] == ["saved_run_of_another_graph"]


def _jsonb(text: str) -> str:
    """Returns ``text`` the way Postgres ``jsonb`` gives it back, with every object's keys reordered.

    ``jsonb`` puts shorter keys first, then sorts keys of the same length bytewise.
    """

    def order(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: order(value[k])
                for k in sorted(value, key=lambda k: (len(k.encode()), k.encode()))
            }
        return [order(v) for v in value] if isinstance(value, list) else value

    return json.dumps(order(json.loads(text)))


def _shuffled(text: str, seed: int) -> str:
    """Returns ``text`` with the keys of every object in a random order."""
    from random import Random

    chance = Random(seed)

    def order(value: Any) -> Any:
        if isinstance(value, dict):
            keys = list(value)
            chance.shuffle(keys)
            return {k: order(value[k]) for k in keys}
        return [order(v) for v in value] if isinstance(value, list) else value

    return json.dumps(order(json.loads(text)))


_CALLS: list[str] = []


@node("sheet", title="Sheet", description="For a test.")
def sheet(text: str, options: dict[str, int]) -> Table:
    _CALLS.append("sheet")
    names = text.split(",")
    return pl.DataFrame(
        {"zeta": names, "alpha": [options[n] for n in names], "mid": [len(n) for n in names]}
    )


class SheetRow(TypedDict):
    zeta: str
    alpha: int
    mid: int


@node("sheet_rows", title="Sheet rows", description="For a test.")
def sheet_rows(sheet: Table) -> Series[SheetRow]:
    return sheet.to_dicts()


@node("ok_row", title="Ok row", description="For a test.", asks=True)
def ok_row(zeta: str, alpha: int) -> str:
    _CALLS.append(f"ok {zeta}")
    raise Ask(f"{zeta} for {alpha}?")


@node("column_order", title="Column order", description="For a test.")
def column_order(sheet: Table, answers: Series[str]) -> str:
    _CALLS.append("order")
    return ",".join(sheet.columns) + "|" + ",".join(answers)


@pytest.mark.parametrize(
    "reorder",
    [_jsonb, lambda text: _shuffled(text, 1), lambda text: _shuffled(text, 7)],
    ids=["jsonb", "shuffle-1", "shuffle-7"],
)
def test_a_graph_record_and_saved_run_whose_keys_a_store_reordered_compile_read_back_and_resume(
    reorder: Any,
) -> None:
    registry = registry_of(sheet, sheet_rows, ok_row, column_order)
    record = {
        "inputs": {"text": {}},
        "nodes": {
            "s": {
                "node": "sheet",
                "inputs": {
                    "options": {"constant": {"bb": 2, "a": 1, "ccc": 3}},
                    "text": {"graph_input": "text"},
                },
            },
            "r": {
                "node": "sheet_rows",
                "inputs": {"sheet": {"sources": [{"node": "s", "output": "result"}]}},
            },
            "q": {
                "node": "ok_row",
                "inputs": {
                    "zeta": {"sources": [{"node": "r", "output": "result", "field": "zeta"}]},
                    "alpha": {"sources": [{"node": "r", "output": "result", "field": "alpha"}]},
                },
            },
            "o": {
                "node": "column_order",
                "inputs": {
                    "sheet": {"sources": [{"node": "s", "output": "result"}]},
                    "answers": {"sources": [{"node": "q", "output": "result"}]},
                },
            },
        },
        "outputs": {"result": {"node": "o", "output": "result"}},
    }
    graph = Graph.compile(GraphRecord.model_validate(record), registry)
    first = graph.run({"text": "ccc,a,bb"})
    ids = [w.id for w in first.waiting]  # what a host stores when the run waits
    answers = {call_id: f"yes {i}" for i, call_id in enumerate(ids)}
    straight = first.with_answers(answers).resume()
    assert straight.results["o"]["result"] == "zeta,alpha,mid|yes 0,yes 1,yes 2"

    stored = Graph.compile(
        reorder(json.dumps(record)), registry
    )  # the record as the store gave it back
    assert stored.fingerprint == graph.fingerprint
    read_back = Run.from_json(reorder(first.to_json()), stored)
    assert [w.id for w in read_back.waiting] == ids
    resumed = read_back.with_answers(answers).resume()
    assert (
        resumed.results["o"] == straight.results["o"]
        and resumed.results["q"] == straight.results["q"]
    )
    assert resumed.results["s"]["result"].equals(straight.results["s"]["result"])
    assert resumed.results["s"]["result"].columns == ["zeta", "alpha", "mid"]

    _CALLS.clear()
    again = Run.from_json(
        reorder(resumed.to_json()), stored
    ).again()  # a dict constant with its keys in another order is the same value, so nothing is called again
    assert again.results["o"] == straight.results["o"] and _CALLS == []


@pytest.mark.parametrize("gone", ["first", "middle", "ask"])
def test_a_saved_run_read_into_a_graph_missing_a_node_definition_is_refused_with_the_graphs_problems(
    gone: str,
) -> None:
    @node("first", title="First", description="For a test.")
    def first(x: int) -> int:
        return x

    @node("middle", title="Middle", description="For a test.")
    def middle(x: int) -> int:
        return x

    @node("ask", title="Ask", description="For a test.", asks=True)
    def ask(x: int) -> bool:
        raise Ask("ok?")

    full = registry_of(first, middle, ask)
    record = make_graph(first, middle, ask)
    saved = Graph.compile(record, full).run({"x": 1}).to_json()
    narrow = Graph.compile(record, full.filter(lambda d: d.id != gone))

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(saved, narrow)

    assert refused.value.problems == narrow.problems and (
        narrow.problems[0].type,
        narrow.problems[0].loc[0],
    ) == ("unknown_node", gone)


def test_a_float_that_is_not_finite_in_a_model_reads_back_by_the_type_of_its_field():
    @host_type("Reading", title="Reading")
    class Reading(BaseModel, frozen=True):
        value: float
        note: Any
        d: dict

    @node("reading", title="Reading", description="For a test.")
    def reading(x: int) -> Reading:
        return Reading(value=math.nan, note=math.nan, d={"w": math.inf})

    @node("readings", title="Readings", description="For a test.")
    def readings(x: int) -> Series[Reading]:
        return [
            Reading(value=-math.inf, note=-math.inf, d={}),
            Reading(value=1, note=1, d={"s": "NaN"}),
        ]

    record = make_graph(
        reading,
        readings,
        edges=[],
        inputs={"x": ["reading.x", "readings.x"]},
        outputs={"reading": "reading.result", "readings": "readings.result"},
    )
    graph = Graph.compile(record, registry_of(reading, readings))
    run = graph.run({"x": 1})
    _strict(run.to_json())
    back = _round_trip(run)

    one = back.results["reading"]["result"]
    assert math.isnan(one.value) and one.note == "NaN" and one.d == {"w": "Infinity"}
    first, second = back.results["readings"]["result"]
    assert (first.value, first.note) == (-math.inf, "-Infinity") and (
        second.value,
        second.note,
        second.d,
    ) == (1.0, 1, {"s": "NaN"})


def test_a_saved_run_missing_a_key_is_refused_naming_the_key_not_with_a_python_error():
    @node("same", title="Same", description="For a test.")
    def same(text: str) -> str:
        return text

    graph = Graph.compile(make_graph(same), registry_of(same))
    problem = _refused(graph, {})
    # the graph's fingerprint is read first, to check that the saved run is of this graph
    assert problem.ctx["where"] == "graph" and "KeyError" not in problem.msg
    saved = json.loads(graph.run({"text": "x"}).to_json())
    del saved["inputs"]
    assert _refused(graph, saved).ctx["where"] == "inputs"


@pytest.mark.parametrize(
    ("where", "damaged"),
    [
        ("units", lambda s: s.__setitem__("units", [])),  # a key a saved run does not have
        ("inputs", lambda s: s.__setitem__("inputs", [])),
        ("ended_early", lambda s: s.__setitem__("ended_early", "banana")),
        ("max_items", lambda s: s.__setitem__("max_items", -1)),
        ("outcomes.split.states", lambda s: s["outcomes"]["split"].__setitem__("states", {})),
        (
            "outcomes.split.states.0",
            lambda s: s["outcomes"]["split"].__setitem__("states", ["waiting"]),
        ),  # a waiting call is saved as its question, never as a state word
        (
            "outcomes.split.keys",
            lambda s: s["outcomes"]["split"].__setitem__("keys", [[]]),
        ),  # item indexes are never saved: an entry's place in the list gives them
        (
            "outcomes.split",
            lambda s: s["outcomes"]["split"].__setitem__("skipped", {"result": [7]}),
        ),
        (
            "outcomes.split.skipped_items.result.0.0",
            lambda s: s["outcomes"]["split"].__setitem__("skipped_items", {"result": {"0": [-1]}}),
        ),
        (
            "outcomes.split",
            lambda s: s["outcomes"]["split"].__setitem__("skipped_items", {"result": {"0": [7]}}),
        ),
    ],
)
def test_a_saved_run_damaged_in_shape_is_refused_where_it_is_damaged(
    where: str, damaged: Any
) -> None:
    graph = Graph.compile(make_graph(split), registry_of(split))
    saved = json.loads(graph.run({"text": "a,b"}).to_json())
    damaged(saved)
    assert _refused(graph, saved).ctx["where"] == where


@node("upper_word", title="Upper word", description="For a test.")
def upper_word(word: str) -> str:
    return word.upper()


@node("ok_word", title="Ok word", description="For a test.", asks=True)
def ok_word(word: str) -> bool:
    raise Ask("ok?")


@pytest.mark.parametrize(
    ("where", "damaged"),
    [
        ("units", lambda s: s.__setitem__("units", [])),  # a key a saved run does not have
        ("reusable", lambda s: s.__setitem__("reusable", [])),
        ("inputs", lambda s: s.__setitem__("inputs", [])),
        ("complete", lambda s: s.__setitem__("complete", 3)),
        ("ended_early", lambda s: s.__setitem__("ended_early", "banana")),
        ("max_items", lambda s: s.__setitem__("max_items", "ten")),
        ("max_items", lambda s: s.__setitem__("max_items", -1)),
        ("outcomes.upper.states", lambda s: s["outcomes"]["upper"].__setitem__("states", {})),
        ("outcomes.ok.questions", lambda s: s["outcomes"]["ok"].__setitem__("questions", [])),
        (
            "outcomes.ok.questions.0.prompt",
            lambda s: s["outcomes"]["ok"]["questions"].__setitem__("0", {}),
        ),
        (
            "outcomes.upper.none_inputs.0",
            lambda s: s["outcomes"]["upper"].__setitem__("none_inputs", {"0": 1}),
        ),
        (
            "outcomes.upper.undeclared",
            lambda s: s["outcomes"]["upper"].__setitem__("undeclared", {"0": -1}),
        ),  # a key a node's saved outcomes do not have
        (
            "outcomes.upper",
            lambda s: s["outcomes"]["upper"].__setitem__("skipped", {"result": [7]}),
        ),
        (
            "outcomes.upper.skipped.other",
            lambda s: s["outcomes"]["upper"].__setitem__("skipped", {"other": [0]}),
        ),  # no such output
        (
            "outcomes.upper.skipped_items.result",
            lambda s: s["outcomes"]["upper"].__setitem__("skipped_items", {"result": {"0": [0]}}),
        ),  # only a Series has items
        (
            "outcomes.ok",
            lambda s: s["outcomes"]["ok"]["questions"].__setitem__("7", {"prompt": "?"}),
        ),
        (
            "outcomes.upper",
            lambda s: s["outcomes"]["upper"].__setitem__("none_inputs", {"0": "word"}),
        ),  # a call that is done can't also be not called for a None
        (
            "outcomes.ok",
            lambda s: s["outcomes"]["ok"].__setitem__("none_inputs", {"0": "word"}),
        ),  # a call that waits can't also be not called for a None
        (
            "graph",
            lambda s: s.clear(),
        ),  # an empty saved run: the graph's fingerprint is read first, since it says which graph the run is of
        ("inputs.split.text", lambda s: s["inputs"].__setitem__("split.text", ["a"])),
        ("inputs.nowhere.text", lambda s: s["inputs"].__setitem__("nowhere.text", "a")),
        (
            "outcomes.upper.fingerprints",
            lambda s: s["outcomes"]["upper"].__setitem__("fingerprints", {}),
        ),
        (
            "outcomes.upper",
            lambda s: s["outcomes"]["upper"].__setitem__("fingerprints", []),
        ),  # not one fingerprint per call
        (
            "outcomes.upper",
            lambda s: s["outcomes"]["upper"]["states"].__setitem__(0, "skipped"),
        ),  # a skipped call keeps the fingerprint of a call that was made
        ("complete.0", lambda s: s.__setitem__("complete", ["nowhere"])),
        (
            "reusable.ok",
            lambda s: s["reusable"].__setitem__(
                "ok",
                {
                    "states": [None],
                    "outputs": {"result": [None]},
                    "fingerprints": ["f"],
                    "item_indexes": [[0]],
                },
            ),
        ),  # a call with no outcome keeps a fingerprint
        (
            "reusable.ok",
            lambda s: s["reusable"].__setitem__(
                "ok",
                {
                    "states": ["done"],
                    "outputs": {"result": [True]},
                    "fingerprints": [],
                    "item_indexes": [[0]],
                },
            ),
        ),
        (
            "reusable.ok",
            lambda s: s["reusable"].__setitem__(
                "ok",
                {
                    "states": ["done"],
                    "outputs": {"result": [True]},
                    "fingerprints": ["f"],
                    "item_indexes": [],
                },
            ),
        ),  # not one item indexes per outcome
        (
            "reusable.ok.fingerprints.0",
            lambda s: s["reusable"].__setitem__(
                "ok",
                {
                    "states": ["done"],
                    "outputs": {"result": [True]},
                    "fingerprints": [None],
                    "item_indexes": [[0]],
                },
            ),
        ),  # a reusable outcome with no fingerprint
        (
            "reusable.ok.fingerprints.0",
            lambda s: s["reusable"].__setitem__(
                "ok",
                {
                    "states": ["skipped"],
                    "outputs": {"result": [None]},
                    "fingerprints": [None],
                    "item_indexes": [[0]],
                },
            ),
        ),  # a skipped call, which has no outcome to reuse
        (
            "reusable.ok.item_indexes",
            lambda s: s["reusable"].__setitem__(
                "ok",
                {"states": ["done"], "outputs": {"result": [True]}, "fingerprints": ["f"]},
            ),
        ),
    ],
)
def test_a_saved_waiting_run_damaged_in_shape_is_refused_where_it_is_damaged(
    where: str, damaged: Any
) -> None:
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "split": {"node": "split", "inputs": {"text": {"graph_input": "text"}}},
                    "upper": {
                        "node": "upper_word",
                        "inputs": {"word": {"sources": [{"node": "split", "output": "result"}]}},
                    },
                    "ok": {
                        "node": "ok_word",
                        "inputs": {"word": {"sources": [{"node": "upper", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "ok", "output": "result"}},
            }
        ),
        registry_of(split, upper_word, ok_word),
    )
    run = graph.run({"text": "a,b"})
    assert run.status == "waiting"
    saved = json.loads(run.to_json())
    damaged(saved)
    assert _refused(graph, saved).ctx["where"] == where


@node("addresses", title="Addresses", description="For a test.")
def addresses(text: str) -> Series[str | None]:
    return [part or None for part in text.split(",")]


@node("short_or_skipped", title="Short or skipped", description="For a test.")
def short_or_skipped(email: str) -> str:
    return email if len(email) < 3 else SKIPPED


@node("confirm", title="Confirm", description="For a test.", asks=True)
def confirm(email: str) -> bool:
    raise Ask(f"Send to {email}?")


def test_a_none_input_and_a_skipped_call_read_back_and_the_resumed_run_lists_the_same_none_inputs():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "a": {"node": "addresses", "inputs": {"text": {"graph_input": "text"}}},
                    "s": {
                        "node": "short_or_skipped",
                        "inputs": {"email": {"sources": [{"node": "a", "output": "result"}]}},
                    },
                    "c": {
                        "node": "confirm",
                        "inputs": {"email": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "c", "output": "result"}},
            }
        ),
        registry_of(addresses, short_or_skipped, confirm),
    )
    run = graph.run({"text": "ab,,long,cd"})
    assert run.status == "waiting" and [w.loc for w in run.waiting] == [("c", 0), ("c", 3)]
    saved = json.loads(run.to_json())["outcomes"]
    # s is not called for the empty address and returns SKIPPED for the long one
    assert (saved["s"]["states"], saved["s"]["none_inputs"], saved["s"]["skipped"]) == (
        ["done", None, "done", "done"],
        {"1": "email"},
        {"result": [2]},
    )
    assert saved["s"]["outputs"] == {"result": ["ab", None, None, "cd"]}
    # c asks about ab and cd. It isn't called after s's None, and is skipped after s's SKIPPED
    assert (
        saved["c"]["states"],
        saved["c"]["none_inputs"],
        list(saved["c"]["questions"]),
    ) == (
        [None, None, "skipped", None],
        {"1": "email"},
        ["0", "3"],
    )
    read_back = Run.from_json(run.to_json(), graph)
    assert read_back.to_json() == run.to_json() and read_back.waiting == run.waiting
    assert (
        read_back.resume().none_input_calls
        == run.none_input_calls
        == [
            NoneInputCall(loc=("s", 1), input="email"),
            NoneInputCall(loc=("c", 1), input="email"),
        ]
    )


_SAVED_IN_ANOTHER_PROCESS = """
from conductor import Ask, Graph, NodeRegistry, make_graph, node

@node("check", title="Check", description="For a test.", asks=True)
def check(kind: str) -> str:
    if kind in {"invoice", "receipt", "order", "bill"}:
        raise Ask("ok?")
    return kind

registry = NodeRegistry()
registry.add(check)
print(Graph.compile(make_graph(check), registry).run({"kind": "invoice"}).waiting[0].id)
"""


@pytest.mark.parametrize("seeds", [("1", "2"), ("3", "4")])
def test_a_waiting_call_has_the_same_id_in_every_process_though_its_node_tests_a_set(seeds):
    ids = {
        subprocess.run(
            [sys.executable, "-c", _SAVED_IN_ANOTHER_PROCESS],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in seeds
    }
    assert len(ids) == 1


_WRITTEN_IN_ANOTHER_PROCESS = """
from conductor import Graph, NodeRegistry, make_graph, node

@node("a", title="A", description="For a test.")
def a(text: str) -> str:
    return text

@node("b", title="B", description="For a test.")
def b(text: str) -> str:
    return text + "!"

@node("c", title="C", description="For a test.")
def c(text: str) -> str:
    return text + "?"

@node("d", title="D", description="For a test.")
def d(text: str) -> str:
    return text + "."

registry = NodeRegistry()
registry.add(a, b, c, d)
print(Graph.compile(make_graph(a, b, c, d), registry).run({"text": "x"}).to_json())
"""


def test_a_run_is_written_to_the_same_bytes_in_every_process():
    saved = {
        subprocess.run(
            [sys.executable, "-c", _WRITTEN_IN_ANOTHER_PROCESS],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3", "4")
    }
    assert len(saved) == 1


def test_a_saved_run_that_ended_early_without_a_cause_or_as_no_early_end_is_refused_where_it_says_so():
    """A saved run whose ``ended_early`` doesn't add up is refused, and the problem says where.

    A failed run is saved with how its last resume ended early: its status and its cause. A complete run has nothing
    there. So a failed status with no cause is refused, and so is an early end whose status is ``complete``.
    """

    @node("boom", title="Boom", description="Fails on 'x'.")
    def boom(text: str) -> str:
        if text == "x":
            raise ValueError("no")
        return text

    graph = Graph.compile(make_graph(boom), registry_of(boom))
    failed, complete = (
        json.loads(graph.run({"text": "x"}).to_json()),
        json.loads(graph.run({"text": "y"}).to_json()),
    )
    assert failed["ended_early"]["status"] == "failed" and complete["ended_early"] is None
    assert (
        _refused(graph, {**failed, "ended_early": {"status": "failed"}}).ctx["where"]
        == "ended_early.cause"
    )
    assert (
        _refused(
            graph,
            {
                **complete,
                "ended_early": {"status": "complete", "cause": failed["ended_early"]["cause"]},
            },
        ).ctx["where"]
        == "ended_early.status"
    )
