"""Saving a run and reading it back: ``saved = run.to_json()`` and ``Run.from_json(saved, graph)``.

A saved run is strict JSON. A node's outputs are saved as one list per output, with one entry per call, each value
written by the output's declared type, so a value reads back as itself, a float that is not finite included.
``Run.from_json`` refuses a saved run that does not fit the graph with the problem ``saved_run_unreadable`` (damaged)
or ``saved_run_of_another_graph``, and ``to_json`` refuses a value with no JSON form with ``run_unwritable``. Each
problem says where, as ``ctx["where"]`` (a path into the JSON such as ``outcomes.echo.outputs.result.1``, or an
output such as ``o.result``), and never quotes a saved value.
"""

import datetime as dt
import json
import math
import os
import struct
import subprocess
import sys
from decimal import Decimal
from typing import Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Address,
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    NotCalled,
    RecordGraphInput,
    Series,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from conductor._internal.run import Run
from conductor._internal.run.outcome import Done, NoneInput, Skipped, Unanswered
from pydantic import BaseModel
from pytest import MonkeyPatch

from tests.test_core.nodes import registry_of


@pytest.fixture(autouse=True)
def _every_saved_run_reads_back_and_saves_the_same(monkeypatch: MonkeyPatch) -> None:
    """Every ``run.to_json()`` in this file also reads its text back into the graph and checks that the run read back
    saves to the same bytes."""
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
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc) == ("saved_run_unreadable", ("saved_run",))
    return problem


def _strict(text: str) -> Any:
    """``text`` read as JSON that refuses ``NaN`` and ``Infinity``."""

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

    run = Graph.compile(make_graph(make_floats), registry_of(make_floats)).run(text="x")

    read_back = _round_trip(run).results["floats"]["result"]

    assert [struct.pack("<d", f) for f in read_back] == [struct.pack("<d", f) for f in floats]


def test_a_text_that_spells_the_skipped_sentinel_reads_back_as_text():
    @node("spelled", title="Spelled", description="For a test.")
    def spelled(text: str) -> Series[str]:
        return ["__skipped__", "SKIPPED", ""]

    run = Graph.compile(make_graph(spelled), registry_of(spelled)).run(text="x")

    assert _round_trip(run).results["spelled"]["result"] == ["__skipped__", "SKIPPED", ""]


def _waiting_after(definition: Any, *, per_item: bool = False) -> Any:
    """A graph that waits for a person right after ``definition``, and the list of values its last node is called with.

    ``definition`` is node ``m``. Node ``c`` asks a person about ``m.result``; node ``s`` takes ``m.result`` and the
    answer, and appends the first to the list. With ``per_item``, ``m`` is called once per item of ``split``.
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
        "m": {"node": definition.id, "inputs": {"text": sources("split") if per_item else {"graph_input": "text"}}},
        "c": {"node": "check", "inputs": {"value": sources("m")}},
        "s": {"node": "seen", "inputs": {"value": sources("m"), "checked": sources("c")}},
    }
    if per_item:
        nodes["split"] = {"node": "split", "inputs": {"text": {"graph_input": "text"}}}
    record = GraphRecord.model_validate({"nodes": nodes, "outputs": {"result": {"node": "s", "output": "result"}}})
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
        paused = graph.run(text=text)
        saved = paused.to_json()
        _strict(saved)

        read_back = Run.from_json(saved, graph)
        value = read_back.results["m"]["result"]
        done = read_back.answer("c", "yes").resume()

        assert isinstance(value, float) and (math.isnan(value) if text == "nan" else value == float(text))
        assert done.status == "complete" and len(seen) == 1 and (math.isnan(seen[0]) if text == "nan" else seen[0] == float(text))


def test_a_series_of_floats_keeps_the_ones_that_are_not_finite():
    @node("readings", title="Readings", description="For a test.")
    def readings(text: str) -> Series[float]:
        return [math.nan, math.inf, -math.inf]

    run = Graph.compile(make_graph(readings), registry_of(readings)).run(text="x")

    assert _is_nan_inf_minus_inf(_round_trip(run).results["readings"]["result"])


def test_floats_returned_per_item_keep_the_ones_that_are_not_finite():
    @node("per-item", title="Per item", description="For a test.")
    def per_item(text: str) -> float:
        return float(text)

    graph, _ = _waiting_after(per_item, per_item=True)
    paused = graph.run(text="nan,inf,-inf")

    assert _is_nan_inf_minus_inf(Run.from_json(paused.to_json(), graph).results["m"]["result"])


class Reading(TypedDict):
    value: float | None


def test_a_float_column_keeps_nan_infinity_and_null():
    @node("table", title="Table", description="For a test.")
    def table(text: str) -> Table[Reading]:
        return pl.DataFrame({"value": [math.nan, math.inf, -math.inf, None]})

    run = Graph.compile(make_graph(table), registry_of(table)).run(text="x")

    column = _round_trip(run).results["table"]["result"]["value"].to_list()

    assert math.isnan(column[0]) and column[1:] == [math.inf, -math.inf, None]


def test_a_saved_run_is_strict_json_when_a_number_is_not_finite():
    @node("odd", title="Odd", description="For a test.")
    def odd(text: str) -> Series[float]:
        return [math.nan, math.inf]

    saved = Graph.compile(make_graph(odd), registry_of(odd)).run(text="x").to_json()

    assert _strict(saved)["state"] == {"status": "complete"}


def test_none_returned_for_an_output_typed_series_or_none_reads_back_as_none():
    @node("maybe-readings", title="Maybe readings", description="For a test.")
    def maybe_readings(text: str) -> Series[int] | None:
        return None

    run = Graph.compile(GraphRecord.model_validate({"nodes": {"m": {"node": "maybe-readings", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "m", "output": "result"}}}), registry_of(maybe_readings)).run(text="x")

    assert run.status == "complete"
    assert _round_trip(run).results["m"]["result"] is None


def test_a_float_that_is_not_finite_in_a_tuple_reads_back_by_the_type_declared_at_its_position():
    """A ``tuple`` inside a ``dict`` output or a model's field is a list in the saved run; a position declared
    ``float`` reads back as the float, and one declared ``Any`` as its text, ``"Infinity"``."""

    @host_type("Point", title="Point")
    class Point(BaseModel):
        xy: tuple[float, Any]

    @node("pair", title="Pair", description="For a test.")
    def pair(text: str) -> dict[str, tuple[float, ...]]:
        return {"a": (math.nan, 1.5)}

    @node("point", title="Point", description="For a test.")
    def point(text: str) -> Point:
        return Point(xy=(math.inf, math.inf))

    record = make_graph(pair, point, edges=[], inputs={"text": ["pair.text", "point.text"]}, outputs={"pair": "pair.result", "point": "point.result"})
    graph = Graph.compile(record, registry_of(pair, point))
    results = _round_trip(graph.run(text="x")).results

    assert math.isnan(results["pair"]["result"]["a"][0]) and results["pair"]["result"]["a"][1] == 1.5
    assert results["point"]["result"].xy == (math.inf, "Infinity")


def test_a_float_that_is_not_finite_where_its_type_does_not_say_float_reads_back_as_its_text():
    """A ``nan`` under ``dict``, ``Any`` or a model's ``Any`` field reads back from a saved run as the text it was
    written as, ``"NaN"``, as a date under ``Any`` reads back as text: an author who wants the float declares ``float``."""

    @host_type("Loose", title="Loose")
    class Loose(BaseModel):
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
    record = make_graph(loose, mapping, anything, maybe_mapping, edges=[], inputs={"text": [f"{n}.text" for n in names]}, outputs={n: f"{n}.result" for n in names})
    graph = Graph.compile(record, registry_of(loose, mapping, anything, maybe_mapping))
    run = graph.run(text="x")
    _strict(run.to_json())
    results = _round_trip(run).results

    assert results["loose"]["result"].data == {"a": "NaN", "b": ["Infinity", "-Infinity"]}
    assert results["mapping"]["result"] == {"v": "NaN", "w": [{"x": "Infinity"}], "s": "NaN", "d": "2026-01-01"}
    assert results["anything"]["result"] == ["NaN", "Infinity", {"z": "-Infinity"}]
    assert results["maybe-mapping"]["result"] == {"v": "NaN"}


def test_a_float_or_text_field_reads_a_float_that_is_not_finite_back_as_the_text_it_takes():
    @node("up", title="Up", description="For a test.")
    def up(k: int) -> Series[float | str]:
        return [math.nan, "NaN", 1.5]

    @node("down", title="Down", description="For a test.")
    def down(x: float | str) -> str:
        return type(x).__name__

    graph = Graph.compile(make_graph(up, down, edges=[("up.result", "down.x")]), registry_of(up, down))
    read_back = _round_trip(graph.run(k=1))

    assert read_back.results["up"]["result"] == ["NaN", "NaN", 1.5]
    assert read_back.forget("down").resume().outputs["result"] == ["str", "str", "float"]


@host_type("point", title="Point")
class Point(BaseModel):
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

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "everything", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "check-table", "inputs": {"rows": {"sources": [{"node": "e", "output": "result"}]}}}, "t": {"node": "take", "inputs": {"rows": {"sources": [{"node": "e", "output": "result"}]}, "checked": {"sources": [{"node": "c", "output": "result"}]}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}), registry_of(everything, check_table, take))
    paused = graph.run(text="x")
    before = paused.results["e"]["result"]

    read_back = Run.from_json(paused.to_json(), graph)
    after = read_back.results["e"]["result"]
    read_back.answer("c", "yes").resume()

    assert after.to_dict(as_series=False) == before.to_dict(as_series=False) == _EVERYTHING
    assert after.schema == before.schema
    assert received[0].to_dict(as_series=False) == _EVERYTHING


def test_a_table_with_no_row_type_keeps_its_list_struct_time_and_decimal_columns():
    columns = {
        "tags": [["a"], ["b", "c"]], "point": [{"x": 1}, {"x": 2}], "clock": [dt.time(1, 2), dt.time(3, 4)],
        "amount": [Decimal("1.10"), Decimal("2.25")],
    }

    @node("bare", title="Bare", description="For a test.")
    def bare(text: str) -> pl.DataFrame:
        return pl.DataFrame(columns)

    run = Graph.compile(make_graph(bare), registry_of(bare)).run(text="x")

    read_back = _round_trip(run).results["bare"]["result"]

    assert read_back.to_dict(as_series=False) == columns and read_back.schema == run.results["bare"]["result"].schema


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

    run = Graph.compile(GraphRecord.model_validate({"nodes": {"n": {"node": "names", "inputs": {"text": {"graph_input": "n_text"}}}, "a": {"node": "ages", "inputs": {"text": {"graph_input": "a_text"}}}}, "outputs": {"a_result": {"node": "a", "output": "result"}, "n_result": {"node": "n", "output": "result"}}}), registry_of(names, ages)).run(n_text="x", a_text="y")

    read_back = _round_trip(run)

    assert read_back.results["n"]["result"].to_dicts() == [{"name": "ida"}, {"name": "bo"}]
    assert read_back.results["a"]["result"].to_dicts() == [{"age": 30}, {"age": 40}, {"age": 50}]


def test_two_identical_runs_save_byte_identical_json():
    @node("mixed", title="Mixed", description="For a test.")
    def mixed(text: str) -> Series[float]:
        return [1.5, math.nan, 2.0]

    graph = Graph.compile(make_graph(mixed), registry_of(split, mixed))

    assert graph.run(text="x").to_json() == graph.run(text="x").to_json()


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

    run = Graph.compile(GraphRecord.model_validate({"nodes": {"o": {"node": "opaque", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "o", "output": "result"}}}), registry_of(opaque)).run(text="x")

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx["where"]) == ("run_unwritable", ("saved_run", "o.result"), "o.result")


def test_saving_a_run_given_an_input_with_no_json_form_raises_naming_the_input():
    @node("take", title="Take", description="For a test.")
    def take(value: Any) -> str:
        return "ok"

    run = Graph.compile(GraphRecord.model_validate({"nodes": {"t": {"node": "take", "inputs": {"value": {"graph_input": "value"}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}), registry_of(take)).run(value=_Opaque())

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert (problem.type, problem.ctx["where"]) == ("run_unwritable", "value")


def test_saving_a_table_column_json_cannot_write_raises_naming_the_output():
    class Blob(TypedDict):
        data: bytes

    @node("blob", title="Blob", description="For a test.")
    def blob(text: str) -> Table[Blob]:
        return pl.DataFrame({"data": [b"\xff\xfe"]})

    run = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "blob", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "b", "output": "result"}}}), registry_of(blob)).run(text="x")

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert problem.type == "run_unwritable" and problem.ctx["where"].startswith("b.result")
    assert "\\xff" not in json.dumps(problem.model_dump(mode="json"))


# -- a saved run that does not fit its graph is refused ---------------------------------------


def _saved_echoes() -> tuple[Any, dict[str, Any]]:
    """A graph in which ``split`` returns three items and ``echo`` is called once per item, and its complete saved run
    as a dict."""

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    graph = Graph.compile(make_graph(split, echo), registry_of(split, echo))
    return graph, json.loads(graph.run(text="secret-a,secret-b,secret-c").to_json())


def test_a_saved_value_its_type_does_not_read_is_refused_naming_where_and_quoting_nothing():
    graph, saved = _saved_echoes()
    saved["outcomes"]["echo"]["outputs"]["result"][1] = {"secret": "value"}

    problem = _refused(graph, saved)

    assert problem.ctx["where"] == "outcomes.echo.outputs.result.1"
    assert "secret" not in json.dumps(problem.model_dump(mode="json"))


def test_a_null_saved_for_an_output_reads_back_as_none():
    graph, saved = _saved_echoes()
    saved["outcomes"]["echo"]["outputs"]["result"][1] = None

    assert Run.from_json(json.dumps(saved), graph).results["echo"]["result"] == ["secret-a", None, "secret-c"]


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


def test_a_series_saved_as_null_is_refused_at_the_node_called_once_per_item_of_it():
    graph, saved = _saved_echoes()
    saved["outcomes"]["split"]["outputs"]["result"][0] = None

    assert _refused(graph, saved).ctx["where"] == "outcomes.echo"


def _saved_table() -> tuple[Any, dict[str, Any]]:
    class Name(TypedDict):
        name: str

    @node("names", title="Names", description="For a test.")
    def names(text: str) -> Table[Name]:
        return pl.DataFrame({"name": text.split(",")})

    @node("count", title="Count", description="For a test.")
    def count(table: Table[Name]) -> int:
        return table.height

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"n": {"node": "names", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "count", "inputs": {"table": {"sources": [{"node": "n", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), registry_of(names, count))
    return graph, json.loads(graph.run(text="ida,bo").to_json())


def test_a_malformed_table_is_refused():
    graph, saved = _saved_table()
    saved["outcomes"]["n"]["outputs"]["result"][0] = {"rows": 2, "columns": {"name": ["ida"]}}

    assert _refused(graph, saved).ctx["where"].startswith("outcomes.n.outputs.result.0")


def test_a_saved_run_is_refused_by_an_edited_graph_but_not_by_a_change_of_display_only():
    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    def with_default(default: str, **display: Any) -> GraphRecord:
        node_record = NodeRecord(node="shout", inputs={"text": RecordGraphInput(graph_input="text", default=default)}, display=display)
        return GraphRecord(nodes={"s": node_record}, outputs={"result": Address("s", "result")})

    record = with_default("a")
    saved = Graph.compile(record, registry_of(shout)).run().to_json()
    edited, moved = with_default("b"), with_default("a", x=1)

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(saved, Graph.compile(edited, registry_of(shout)))
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc) == ("saved_run_of_another_graph", ("saved_run",)) and set(problem.ctx) == {"saved", "graph"}
    assert Run.from_json(saved, Graph.compile(moved, registry_of(shout))).results == {"s": {"result": "A"}}


def test_a_saved_run_is_refused_by_a_graph_whose_subgraph_was_edited():
    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    @node("quiet", title="Quiet", description="For a test.")
    def quiet(text: str) -> str:
        return text.lower()

    def with_inner(definition: Any) -> Any:
        return registry_of(shout, quiet, subgraph("inner", title="Inner", description="For a test.")(make_graph(definition)))

    outer = GraphRecord.model_validate({"nodes": {"e": {"node": "inner", "inputs": {"text": {"graph_input": "text"}}}}})
    saved = Graph.compile(outer, with_inner(shout)).run(text="a").to_json()

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(saved, Graph.compile(outer, with_inner(quiet)))
    assert [p.type for p in refused.value.errors()] == ["saved_run_of_another_graph"]


def _jsonb(text: str) -> str:
    """``text`` as Postgres ``jsonb`` gives it back: every object's keys shorter first, then bytewise."""

    def order(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: order(value[k]) for k in sorted(value, key=lambda k: (len(k.encode()), k.encode()))}
        return [order(v) for v in value] if isinstance(value, list) else value

    return json.dumps(order(json.loads(text)))


def _shuffled(text: str, seed: int) -> str:
    """``text`` with every object's keys in a random order."""
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
    return pl.DataFrame({"zeta": names, "alpha": [options[n] for n in names], "mid": [len(n) for n in names]})


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


@pytest.mark.parametrize("reorder", [_jsonb, lambda text: _shuffled(text, 1), lambda text: _shuffled(text, 7)], ids=["jsonb", "shuffle-1", "shuffle-7"])
def test_a_graph_record_and_saved_run_whose_keys_a_store_reordered_compile_read_back_and_resume(reorder: Any) -> None:
    registry = registry_of(sheet, sheet_rows, ok_row, column_order)
    record = {"nodes": {"s": {"node": "sheet", "inputs": {"options": {"constant": {"bb": 2, "a": 1, "ccc": 3}}, "text": {"graph_input": "text"}}}, "r": {"node": "sheet_rows", "inputs": {"sheet": {"sources": [{"node": "s", "output": "result"}]}}}, "q": {"node": "ok_row", "inputs": {"zeta": {"sources": [{"node": "r", "output": "result", "field": "zeta"}]}, "alpha": {"sources": [{"node": "r", "output": "result", "field": "alpha"}]}}}, "o": {"node": "column_order", "inputs": {"sheet": {"sources": [{"node": "s", "output": "result"}]}, "answers": {"sources": [{"node": "q", "output": "result"}]}}}}, "outputs": {"result": {"node": "o", "output": "result"}}}
    graph = Graph.compile(GraphRecord.model_validate(record), registry)
    first = graph.run(text="ccc,a,bb")
    ids = [w.id for w in first.waiting]  # what a host stores when the run waits
    answers = {call_id: f"yes {i}" for i, call_id in enumerate(ids)}
    straight = first.answer(answers).resume()
    assert straight.results["o"]["result"] == "zeta,alpha,mid|yes 0,yes 1,yes 2"

    stored = Graph.compile(reorder(json.dumps(record)), registry)  # the record as the store gave it back
    assert stored.fingerprint == graph.fingerprint
    read_back = Run.from_json(reorder(first.to_json()), stored)
    assert [w.id for w in read_back.waiting] == ids
    resumed = read_back.answer(answers).resume()
    assert resumed.results["o"] == straight.results["o"] and resumed.results["q"] == straight.results["q"]
    assert resumed.results["s"]["result"].equals(straight.results["s"]["result"])
    assert resumed.results["s"]["result"].columns == ["zeta", "alpha", "mid"]

    _CALLS.clear()
    again = Run.from_json(reorder(resumed.to_json()), stored).again()  # a dict constant read back with its keys in another order is the same value, so nothing is called
    assert again.results["o"] == straight.results["o"] and _CALLS == []


@pytest.mark.parametrize("gone", ["first", "middle", "ask"])
def test_a_saved_run_read_into_a_graph_missing_a_node_definition_is_refused_with_the_graphs_problems(gone: str) -> None:
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
    saved = Graph.compile(record, full).run(x=1).to_json()
    narrow = Graph.compile(record, full.filter(lambda d: d.id != gone))

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(saved, narrow)

    assert refused.value.errors() == narrow.problems and (narrow.problems[0].type, narrow.problems[0].loc[0]) == ("unknown_node", gone)


def test_a_float_that_is_not_finite_in_a_model_reads_back_by_the_type_of_its_field():
    @host_type("Reading", title="Reading")
    class Reading(BaseModel):
        value: float
        note: Any
        d: dict

    @node("reading", title="Reading", description="For a test.")
    def reading(x: int) -> Reading:
        return Reading(value=math.nan, note=math.nan, d={"w": math.inf})

    @node("readings", title="Readings", description="For a test.")
    def readings(x: int) -> Series[Reading]:
        return [Reading(value=-math.inf, note=-math.inf, d={}), Reading(value=1, note=1, d={"s": "NaN"})]

    record = make_graph(reading, readings, edges=[], inputs={"x": ["reading.x", "readings.x"]}, outputs={"reading": "reading.result", "readings": "readings.result"})
    graph = Graph.compile(record, registry_of(reading, readings))
    run = graph.run(x=1)
    _strict(run.to_json())
    back = _round_trip(run)

    one = back.results["reading"]["result"]
    assert math.isnan(one.value) and one.note == "NaN" and one.d == {"w": "Infinity"}
    first, second = back.results["readings"]["result"]
    assert (first.value, first.note) == (-math.inf, "-Infinity") and (second.value, second.note, second.d) == (1.0, 1, {"s": "NaN"})


def test_a_float_that_is_not_finite_given_by_a_caller_reads_back_by_the_type_of_its_input():
    @node("given", title="Given", description="For a test.")
    def given(options: dict, level: Any, ratio: float) -> str:
        return repr((options, level, ratio))

    graph = Graph.compile(make_graph(given), registry_of(given))
    run = graph.run(options={"a": math.nan, "b": [math.inf]}, level=-math.inf, ratio=math.inf)
    _strict(run.to_json())
    back = _round_trip(run)

    assert back.inputs == {"options": {"a": "NaN", "b": ["Infinity"]}, "level": "-Infinity", "ratio": math.inf}


def test_a_saved_run_missing_a_key_is_refused_naming_the_key_not_with_a_python_error():
    @node("same", title="Same", description="For a test.")
    def same(text: str) -> str:
        return text

    graph = Graph.compile(make_graph(same), registry_of(same))
    problem = _refused(graph, {})
    assert problem.ctx["where"] == "the run" and "'graph'" in problem.msg and "KeyError" not in problem.msg  # the graph's fingerprint is read first, to know the saved run is of this graph
    saved = json.loads(graph.run(text="x").to_json())
    del saved["inputs"]
    assert _refused(graph, saved).ctx["where"] == "inputs"


@pytest.mark.parametrize(
    ("where", "damaged"),
    [
        ("units", lambda s: s.__setitem__("units", [])),  # a key a saved run does not have
        ("inputs", lambda s: s.__setitem__("inputs", [])),
        ("state", lambda s: s["state"].__setitem__("status", "banana")),
        ("max_items", lambda s: s.__setitem__("max_items", -1)),
        ("outcomes.split.states", lambda s: s["outcomes"]["split"].__setitem__("states", {})),
        ("outcomes.split.states.0", lambda s: s["outcomes"]["split"].__setitem__("states", ["waiting"])),  # a waiting call is saved as its question, never as a state word
        ("outcomes.split.keys", lambda s: s["outcomes"]["split"].__setitem__("keys", [[]])),  # item indexes are never written: an entry's place in the list says them
        ("outcomes.split", lambda s: s["outcomes"]["split"].__setitem__("skipped", {"result": [7]})),
        ("outcomes.split.skipped_items.result.0.0", lambda s: s["outcomes"]["split"].__setitem__("skipped_items", {"result": {"0": [-1]}})),
        ("outcomes.split", lambda s: s["outcomes"]["split"].__setitem__("skipped_items", {"result": {"0": [7]}})),
    ],
)
def test_a_saved_run_damaged_in_shape_is_refused_where_it_is_damaged(where: str, damaged: Any) -> None:
    graph = Graph.compile(make_graph(split), registry_of(split))
    saved = json.loads(graph.run(text="a,b").to_json())
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
        ("code", lambda s: s.__setitem__("code", [])),
        ("complete", lambda s: s.__setitem__("complete", 3)),
        ("state", lambda s: s["state"].__setitem__("status", "banana")),
        ("max_items", lambda s: s.__setitem__("max_items", "ten")),
        ("max_items", lambda s: s.__setitem__("max_items", -1)),
        ("outcomes.upper.states", lambda s: s["outcomes"]["upper"].__setitem__("states", {})),
        ("outcomes.ok.questions", lambda s: s["outcomes"]["ok"].__setitem__("questions", [])),
        ("outcomes.ok.questions.0.prompt", lambda s: s["outcomes"]["ok"]["questions"].__setitem__("0", {"inputs": {}})),
        ("outcomes.upper.not_called_because.0", lambda s: s["outcomes"]["upper"].__setitem__("not_called_because", {"0": 1})),
        ("outcomes.upper.undeclared", lambda s: s["outcomes"]["upper"].__setitem__("undeclared", {"0": -1})),  # a key a node's saved outcomes do not have
        ("outcomes.upper", lambda s: s["outcomes"]["upper"].__setitem__("skipped", {"result": [7]})),
        ("outcomes.upper.skipped.other", lambda s: s["outcomes"]["upper"].__setitem__("skipped", {"other": [0]})),  # no such output
        ("outcomes.upper.skipped_items.result", lambda s: s["outcomes"]["upper"].__setitem__("skipped_items", {"result": {"0": [0]}})),  # only a Series has items
        ("outcomes.ok", lambda s: s["outcomes"]["ok"]["questions"].__setitem__("7", {"prompt": "?", "inputs": {}})),
        ("outcomes.upper", lambda s: s["outcomes"]["upper"].__setitem__("not_called_because", {"0": "word"})),  # not called for a None, on a call that is done
        ("outcomes.ok", lambda s: s["outcomes"]["ok"].__setitem__("not_called_because", {"0": "word"})),  # not called for a None, on a call that waits
        ("the run", lambda s: s.clear()),  # an empty saved run: the graph's fingerprint, which says which graph it is of, is read first
        ("inputs.split.text", lambda s: s["inputs"].__setitem__("split.text", ["a"])),
        ("inputs.nowhere.text", lambda s: s["inputs"].__setitem__("nowhere.text", "a")),
        ("code.nowhere", lambda s: s["code"].__setitem__("nowhere", "0")),
        ("complete.0", lambda s: s.__setitem__("complete", ["nowhere"])),
        ("reusable.ok", lambda s: s["reusable"].__setitem__("ok", {"outcomes": {"states": [None], "outputs": {"result": [None]}}, "fingerprints": ["f"]})),  # a reusable entry with no outcome
        ("reusable.ok", lambda s: s["reusable"].__setitem__("ok", {"outcomes": {"states": ["done"], "outputs": {"result": [True]}}, "fingerprints": []})),
    ],
)
def test_a_saved_waiting_run_damaged_in_shape_is_refused_where_it_is_damaged(where: str, damaged: Any) -> None:
    graph = Graph.compile(
        GraphRecord.model_validate({"nodes": {"split": {"node": "split", "inputs": {"text": {"graph_input": "text"}}}, "upper": {"node": "upper_word", "inputs": {"word": {"sources": [{"node": "split", "output": "result"}]}}}, "ok": {"node": "ok_word", "inputs": {"word": {"sources": [{"node": "upper", "output": "result"}]}}}}, "outputs": {"result": {"node": "ok", "output": "result"}}}), registry_of(split, upper_word, ok_word)
    )
    run = graph.run(text="a,b")
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


def test_a_not_called_and_a_skipped_call_read_back_and_the_resumed_run_lists_the_same_not_called():
    graph = Graph.compile(
        GraphRecord.model_validate({"nodes": {"a": {"node": "addresses", "inputs": {"text": {"graph_input": "text"}}}, "s": {"node": "short_or_skipped", "inputs": {"email": {"sources": [{"node": "a", "output": "result"}]}}}, "c": {"node": "confirm", "inputs": {"email": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), registry_of(addresses, short_or_skipped, confirm)
    )
    run = graph.run(text="ab,,long,cd")
    assert run.status == "waiting"
    assert list(run._outcomes["s"].values()) == [Done({"result": "ab"}), NoneInput({"result": None}, "email"), Done({"result": SKIPPED}), Done({"result": "cd"})]
    assert [type(u) for u in run._outcomes["c"].values()] == [Unanswered, NoneInput, Skipped, Unanswered]
    assert run._outcomes["c"][(1,)] == NoneInput({"result": None}, "email")
    read_back = Run.from_json(run.to_json(), graph)
    assert read_back._outcomes == run._outcomes
    assert read_back.resume().not_called == run.not_called == [
        NotCalled(loc=("s", 1), input="email"),
        NotCalled(loc=("c", 1), input="email"),
    ]


_SAVED_IN_ANOTHER_PROCESS = """
from conductor import Ask, Graph, NodeRegistry, make_graph, node

@node("check", title="Check", description="For a test.", asks=True)
def check(kind: str) -> str:
    if kind in {"invoice", "receipt", "order", "bill"}:
        raise Ask("ok?")
    return kind

registry = NodeRegistry()
registry.add(check)
print(Graph.compile(make_graph(check), registry).run(kind="invoice").waiting[0].id)
"""


@pytest.mark.parametrize("seeds", [("1", "2"), ("3", "4")])
def test_a_waiting_call_has_the_same_id_in_every_process_though_its_node_tests_a_set(seeds):
    ids = {
        subprocess.run([sys.executable, "-c", _SAVED_IN_ANOTHER_PROCESS], env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True).stdout
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
print(Graph.compile(make_graph(a, b, c, d), registry).run(text="x").to_json())
"""


def test_a_run_is_written_to_the_same_bytes_in_every_process():
    saved = {
        subprocess.run([sys.executable, "-c", _WRITTEN_IN_ANOTHER_PROCESS], env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True).stdout
        for seed in ("1", "2", "3", "4")
    }
    assert len(saved) == 1


def test_a_saved_run_whose_status_and_cause_disagree_is_refused_at_the_cause():
    """A failed run is saved with its cause and a complete one without; a saved run that says otherwise is refused."""

    @node("boom", title="Boom", description="Fails on 'x'.")
    def boom(text: str) -> str:
        if text == "x":
            raise ValueError("no")
        return text

    graph = Graph.compile(make_graph(boom), registry_of(boom))
    failed, complete = json.loads(graph.run(text="x").to_json()), json.loads(graph.run(text="y").to_json())
    assert failed["state"]["status"] == "failed" and complete["state"] == {"status": "complete"}
    assert _refused(graph, {**failed, "state": {"status": "failed"}}).ctx["where"] == "state.failed.cause"
    assert _refused(graph, {**complete, "state": {"status": "complete", "cause": failed["state"]["cause"]}}).ctx["where"] == "state.complete.cause"
