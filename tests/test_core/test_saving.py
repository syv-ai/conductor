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
import os
import struct
import subprocess
import sys
from decimal import Decimal
from typing import Any, ClassVar, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    NotCalled,
    Series,
    Table,
    make_graph,
    node,
)
from conductor._internal.run import _SAVED, Asking, Blank, Done, Run, Skipped
from pydantic import BaseModel

from tests.test_core.nodes import registry_of


@pytest.fixture(autouse=True)
def _every_save_reads_as_the_declared_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """What ``run.to_json()`` writes in these tests is read back by the one model ``load_run`` reads a save with."""
    write = Run.to_json

    def written_and_read(run: Run) -> str:
        text = write(run)
        _SAVED.validate_json(text)
        return text

    monkeypatch.setattr(Run, "to_json", written_and_read)


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


@node("split", title="Split", description="")
def split(text: str) -> Series[str]:
    return text.split(",")


# -- values read back as themselves -------------------------------------------------------------


def test_a_float_comes_back_to_the_last_bit():
    floats = [0.1 + 0.2, 5e-324, -0.0, 1e308, 1 / 3]

    @node("floats", title="Make floats", description="")
    def make_floats(text: str) -> Series[float]:
        return floats

    run = registry_of(make_floats).load(make_graph(make_floats)).run(text="x")

    loaded = _round_trip(run).results["floats"]["result"]

    assert [struct.pack("<d", f) for f in loaded] == [struct.pack("<d", f) for f in floats]


def test_a_text_that_spells_a_skip_marker_stays_text():
    @node("marker", title="Marker", description="")
    def marker(text: str) -> Series[str]:
        return ["__skipped__", "SKIPPED", ""]

    run = registry_of(marker).load(make_graph(marker)).run(text="x")

    assert _round_trip(run).results["marker"]["result"] == ["__skipped__", "SKIPPED", ""]


def _paused_after(definition: Any, *, per_item: bool = False) -> Any:
    """``definition`` as node ``m``, its ``result`` read by a node that asks and by ``seen``, which takes the answer and what ``m`` made."""
    seen: list[Any] = []

    @node("check", title="Check", description="")
    def check(value: Any) -> str:
        raise Ask("Right?")

    @node("seen", title="See", description="")
    def see(value: Any, checked: str) -> str:
        seen.append(value)
        return checked

    nodes = [NodeRecord(id="m", node=definition.id), NodeRecord(id="c", node="check"), NodeRecord(id="s", node="seen")]
    edges = [("m.result", "c.value"), ("m.result", "s.value"), ("c.result", "s.checked")]
    if per_item:
        nodes.insert(0, NodeRecord(id="split", node="split"))
        edges.append(("split.result", "m.text"))
    graph = registry_of(split, definition, check, see).load(GraphRecord(nodes=nodes, edges=edges))
    return graph, seen


def _is_nan_inf_minus_inf(values: list[float]) -> bool:
    return math.isnan(values[0]) and values[1:] == [math.inf, -math.inf]


def test_a_float_that_is_not_finite_comes_back_a_float_and_its_reader_receives_it():
    @node("ratio", title="Ratio", description="")
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
    @node("readings", title="Readings", description="")
    def readings(text: str) -> Series[float]:
        return [math.nan, math.inf, -math.inf]

    run = registry_of(readings).load(make_graph(readings)).run(text="x")

    assert _is_nan_inf_minus_inf(_round_trip(run).results["readings"]["result"])


def test_a_float_per_item_keeps_the_ones_that_are_not_finite():
    @node("per-item", title="Per item", description="")
    def per_item(text: str) -> float:
        return float(text)

    graph, _ = _paused_after(per_item, per_item=True)
    paused = graph.run(text="nan,inf,-inf")

    assert _is_nan_inf_minus_inf(graph.load_run(paused.to_json()).results["m"]["result"])


class Reading(TypedDict):
    value: float | None


def test_a_float_column_keeps_nan_infinity_and_a_blank():
    @node("table", title="Table", description="")
    def table(text: str) -> Table[Reading]:
        return pl.DataFrame({"value": [math.nan, math.inf, -math.inf, None]})

    run = registry_of(table).load(make_graph(table)).run(text="x")

    column = _round_trip(run).results["table"]["result"]["value"].to_list()

    assert math.isnan(column[0]) and column[1:] == [math.inf, -math.inf, None]


def test_a_saved_run_is_strict_json_when_a_number_is_not_finite():
    @node("odd", title="Odd", description="")
    def odd(text: str) -> Series[float]:
        return [math.nan, math.inf]

    saved = registry_of(odd).load(make_graph(odd)).run(text="x").to_json()

    assert _strict(saved)["status"] == "complete"


def test_a_blank_returned_for_a_blank_taking_series_is_saved_and_read_back_as_a_blank():
    @node("maybe-readings", title="Maybe readings", description="")
    def maybe_readings(text: str) -> Series[int] | None:
        return None

    run = registry_of(maybe_readings).load(GraphRecord(nodes=[NodeRecord(id="m", node="maybe-readings")])).run(text="x")

    assert run.status == "complete"
    assert _round_trip(run).results["m"]["result"] is None


def test_a_float_that_is_not_finite_in_a_tuple_its_type_reads_back_comes_back_the_float():
    """A ``tuple`` inside a ``dict`` output or a model's field is a list in the save; its ``nan`` reads back all the same."""

    class Point(BaseModel):
        type_id: ClassVar[str] = "Point"
        type_title: ClassVar[str] = "Point"
        xy: tuple[float, Any]

    @node("pair", title="Pair", description="")
    def pair(text: str) -> dict[str, tuple[Any, ...]]:
        return {"a": (math.nan, text)}

    @node("point", title="Point", description="")
    def point(text: str) -> Point:
        return Point(xy=(1.0, math.inf))

    record = GraphRecord(nodes=[NodeRecord(id=n, node=n) for n in ("pair", "point")])
    graph = registry_of(pair, point).load(record)
    results = _round_trip(graph.run({"pair.text": "x", "point.text": "x"})).results

    assert math.isnan(results["pair"]["result"]["a"][0]) and results["pair"]["result"]["a"][1] == "x"
    assert results["point"]["result"].xy == (1.0, math.inf)


def test_a_float_that_is_not_finite_where_its_type_does_not_say_float_reads_back_as_the_float():
    """A ``dict``, ``Any`` or a model's ``Any`` field holding ``nan`` gives the same value after a save as on a straight run."""

    class Loose(BaseModel):
        type_id: ClassVar[str] = "Loose"
        type_title: ClassVar[str] = "Loose"
        data: Any

    @node("loose", title="Loose", description="")
    def loose(text: str) -> Loose:
        return Loose(data={"a": math.nan, "b": [math.inf, -math.inf]})

    @node("mapping", title="Mapping", description="")
    def mapping(text: str) -> dict:
        return {"v": math.nan, "w": [{"x": math.inf}], "s": "NaN"}

    @node("anything", title="Anything", description="")
    def anything(text: str) -> Series[Any]:
        return [math.nan, "Infinity", {"z": -math.inf}]

    @node("maybe-mapping", title="Maybe mapping", description="")
    def maybe_mapping(text: str) -> dict[str, Any] | None:
        return {"v": math.nan}

    names = ("loose", "mapping", "anything", "maybe-mapping")
    record = GraphRecord(nodes=[NodeRecord(id=n, node=n) for n in names])
    graph = registry_of(loose, mapping, anything, maybe_mapping).load(record)
    run = graph.run({f"{n}.text": "x" for n in names})
    _strict(run.to_json())
    results = _round_trip(run).results

    data = results["loose"]["result"].data
    assert math.isnan(data["a"]) and data["b"] == [math.inf, -math.inf]
    made = results["mapping"]["result"]
    assert math.isnan(made["v"]) and made["w"] == [{"x": math.inf}] and made["s"] == "NaN"  # text that was text stays text
    first, text, inner = results["anything"]["result"]
    assert math.isnan(first) and text == "Infinity" and inner == {"z": -math.inf}
    assert math.isnan(results["maybe-mapping"]["result"]["v"])


def test_a_float_or_text_field_keeps_a_nan_a_float_and_the_text_nan_text_after_a_save():
    seen: list[Any] = []

    @node("up", title="Up", description="")
    def up(k: int) -> Series[float | str]:
        return [math.nan, "NaN"]

    @node("down", title="Down", description="")
    def down(x: float | str) -> str:
        seen.append(x)
        return type(x).__name__

    graph = registry_of(up, down).load(make_graph(up, down, edges=[("up.result", "down.x")]))
    run = graph.run(k=1)
    loaded = _round_trip(run)
    first, text = loaded.results["up"]["result"]
    assert math.isnan(first) and text == "NaN"

    seen.clear()
    assert loaded.forget("down").resume().outputs["down.result"] == ["float", "str"]
    seen.clear()
    loaded.forget("up").resume()
    assert seen == []  # down receives what it received before the save, so it is not called again


class Point(BaseModel):
    type_id: ClassVar[str] = "point"
    type_title: ClassVar[str] = "Point"
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

    @node("everything", title="Everything", description="")
    def everything(text: str) -> Table[Everything]:
        return pl.DataFrame(_EVERYTHING)

    @node("check-table", title="Check table", description="")
    def check_table(rows: pl.DataFrame) -> str:
        raise Ask("Right?")

    @node("take", title="Take", description="")
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


def test_a_table_with_no_row_type_keeps_its_list_struct_time_and_decimal_columns():
    columns = {
        "tags": [["a"], ["b", "c"]], "point": [{"x": 1}, {"x": 2}], "clock": [dt.time(1, 2), dt.time(3, 4)],
        "amount": [Decimal("1.10"), Decimal("2.25")],
    }

    @node("bare", title="Bare", description="")
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

    @node("names", title="Names", description="")
    def names(text: str) -> Table[Name]:
        return pl.DataFrame({"name": ["ida", "bo"]})

    @node("ages", title="Ages", description="")
    def ages(text: str) -> Table[Age]:
        return pl.DataFrame({"age": [30, 40, 50]})

    run = registry_of(names, ages).load(GraphRecord(nodes=[NodeRecord(id="n", node="names"), NodeRecord(id="a", node="ages")])).run(
        {"n.text": "x", "a.text": "y"},
    )

    loaded = _round_trip(run)

    assert loaded.results["n"]["result"].to_dicts() == [{"name": "ida"}, {"name": "bo"}]
    assert loaded.results["a"]["result"].to_dicts() == [{"age": 30}, {"age": 40}, {"age": 50}]


def test_two_identical_runs_save_byte_identical_json():
    @node("mixed", title="Mixed", description="")
    def mixed(text: str) -> Series[float]:
        return [1.5, math.nan, 2.0]

    graph = registry_of(split, mixed).load(make_graph(mixed))

    assert graph.run(text="x").to_json() == graph.run(text="x").to_json()


# -- what a save can't hold ------------------------------------------------------------------


class _Opaque:
    """A value pydantic carries but has no JSON form for."""
    type_id: ClassVar[str] = "_Opaque"
    type_title: ClassVar[str] = "Opaque"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.is_instance_schema(cls)


def test_saving_a_run_that_holds_a_value_with_no_json_form_raises_naming_the_output():
    @node("opaque", title="Opaque", description="")
    def opaque(text: str) -> _Opaque:
        return _Opaque()

    run = registry_of(opaque).load(GraphRecord(nodes=[NodeRecord(id="o", node="opaque")])).run(text="x")

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert (problem.type, problem.ctx["at"]) == ("state_unwritable", "o.result")


def test_saving_a_run_given_an_input_with_no_json_form_raises_naming_the_input():
    @node("take", title="Take", description="")
    def take(value: Any) -> str:
        return "ok"

    run = registry_of(take).load(GraphRecord(nodes=[NodeRecord(id="t", node="take")])).run(value=_Opaque())

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert (problem.type, problem.ctx["at"]) == ("state_unwritable", "t.value")


def test_a_column_json_cannot_keep_makes_the_save_raise_naming_the_output_and_column():
    class Blob(TypedDict):
        data: bytes

    @node("blob", title="Blob", description="")
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
    """``split`` making three items read by ``echo``, run to the end, and its save."""

    @node("echo", title="Echo", description="")
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
    ["an item never made", "a negative place", "an item listed twice", "an item on a node that runs once", "a missing value", "a missing output", "an output the node lacks"],
)
def test_a_damaged_save_is_refused(damage):
    graph, saved = _saved_mails()
    echo, splitting = saved["units"]["echo"], saved["units"]["split"]
    if damage == "an item never made":
        echo["keys"][2] = [7]
    elif damage == "a negative place":
        echo["keys"][2] = [-1]
    elif damage == "an item listed twice":
        echo["keys"][2] = [1]
    elif damage == "an item on a node that runs once":
        splitting["keys"][0] = [0]
    elif damage == "a missing value":
        echo["outputs"]["result"]["values"].pop()
    elif damage == "a missing output":
        del echo["outputs"]["result"]
    else:
        echo["outputs"]["other"] = {"values": ["x", "y", "z"], "skipped": []}

    _refused(graph, saved)


def test_a_null_where_a_series_that_makes_items_was_is_refused():
    graph, saved = _saved_mails()
    saved["units"]["split"]["outputs"]["result"]["values"][0] = None

    assert _refused(graph, saved).ctx["at"] == "split.result"


def _saved_table() -> tuple[Any, dict[str, Any]]:
    class Name(TypedDict):
        name: str

    @node("names", title="Names", description="")
    def names(text: str) -> Table[Name]:
        return pl.DataFrame({"name": text.split(",")})

    @node("count", title="Count", description="")
    def count(table: Table[Name]) -> int:
        return table.height

    graph = registry_of(names, count).load(GraphRecord(
        nodes=[NodeRecord(id="n", node="names"), NodeRecord(id="c", node="count")], edges=[("n.result", "c.table")],
    ))
    return graph, json.loads(graph.run(text="ida,bo").to_json())


def test_a_malformed_table_is_refused():
    graph, saved = _saved_table()
    saved["units"]["n"]["outputs"]["result"]["values"][0] = {"rows": 2, "columns": {"name": ["ida"]}}

    _refused(graph, saved)


def test_a_save_names_its_graph_and_an_edited_graph_refuses_it_while_display_alone_changes_nothing():
    from conductor import NodeRecord as _Node

    @node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    record = GraphRecord(nodes=[_Node(id="s", node="shout", defaults={"text": "a"})])
    saved = registry_of(shout).load(record).run().to_json()
    edited = record.model_copy(update={"nodes": (_Node(id="s", node="shout", defaults={"text": "b"}),)})
    drawn = record.model_copy(update={"nodes": (_Node(id="s", node="shout", defaults={"text": "a"}, display={"x": 1}),)})

    with pytest.raises(InvalidCallError) as refused:
        registry_of(shout).load(edited).load_run(saved)
    (problem,) = refused.value.errors()
    assert problem.type == "state_of_another_graph" and set(problem.ctx) == {"saved", "graph"}
    assert registry_of(shout).load(drawn).load_run(saved).results == {"s": {"result": "A"}}


@pytest.mark.skip(reason="embedding: nested run not built")
def test_a_save_names_the_graphs_embedded_in_its_graph_too():
    @node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    @node("quiet", title="Quiet", description="")
    def quiet(text: str) -> str:
        return text.lower()

    def with_inner(*nodes: NodeRecord) -> Any:
        return registry_of(shout, quiet, node("inner", title="Inner", description="")(GraphRecord(nodes=list(nodes))))

    outer = GraphRecord(nodes=[NodeRecord(id="e", node="inner")])
    saved = with_inner(NodeRecord(id="x", node="shout")).load(outer).run(text="a").to_json()

    with pytest.raises(InvalidCallError) as refused:
        with_inner(NodeRecord(id="x", node="quiet")).load(outer).load_run(saved)
    assert [p.type for p in refused.value.errors()] == ["state_of_another_graph"]


def _jsonb(text: str) -> str:
    """``text`` as Postgres ``jsonb`` gives it back: every object's keys shorter first, then bytewise."""

    def order(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: order(value[k]) for k in sorted(value, key=lambda k: (len(k.encode()), k.encode()))}
        return [order(v) for v in value] if isinstance(value, list) else value

    return json.dumps(order(json.loads(text)))


def _shuffled(text: str, seed: int) -> str:
    """``text`` with every object's keys in a random order."""
    import random

    chance = random.Random(seed)

    def order(value: Any) -> Any:
        if isinstance(value, dict):
            keys = list(value)
            chance.shuffle(keys)
            return {k: order(value[k]) for k in keys}
        return [order(v) for v in value] if isinstance(value, list) else value

    return json.dumps(order(json.loads(text)))


_CALLS: list[str] = []


@node("sheet", title="Sheet", description="")
def sheet(text: str, options: dict[str, int]) -> Table:
    _CALLS.append("sheet")
    names = text.split(",")
    return pl.DataFrame({"zeta": names, "alpha": [options[n] for n in names], "mid": [len(n) for n in names]})


class SheetRow(TypedDict):
    zeta: str
    alpha: int
    mid: int


@node("sheet_rows", title="Sheet rows", description="")
def sheet_rows(sheet: Table) -> Series[SheetRow]:
    return sheet.to_dicts()


@node("ok_row", title="Ok row", description="")
def ok_row(zeta: str, alpha: int) -> str:
    _CALLS.append(f"ok {zeta}")
    raise Ask(f"{zeta} for {alpha}?")


@node("column_order", title="Column order", description="")
def column_order(sheet: Table, answers: Series[str]) -> str:
    _CALLS.append("order")
    return ",".join(sheet.columns) + "|" + ",".join(answers)


@pytest.mark.parametrize("reorder", [_jsonb, lambda text: _shuffled(text, 1), lambda text: _shuffled(text, 7)], ids=["jsonb", "shuffle-1", "shuffle-7"])
def test_a_record_and_a_save_whose_keys_a_store_reordered_still_load_answer_and_resume(reorder: Any) -> None:
    registry = registry_of(sheet, sheet_rows, ok_row, column_order)
    record = {
        "nodes": [
            {"id": "s", "node": "sheet", "values": {"options": {"bb": 2, "a": 1, "ccc": 3}}},
            {"id": "r", "node": "sheet_rows"},
            {"id": "q", "node": "ok_row"},
            {"id": "o", "node": "column_order"},
        ],
        "edges": [["s.result", "r.sheet"], ["r.result.zeta", "q.zeta"], ["r.result.alpha", "q.alpha"], ["s.result", "o.sheet"], ["q.result", "o.answers"]],
    }
    graph = registry.load(record)
    first = graph.run(text="ccc,a,bb")
    ids = [w.id for w in first.waiting]  # what a host stored when the run paused
    answers = {unit_id: f"yes {i}" for i, unit_id in enumerate(ids)}
    straight = first.answer(answers).resume()
    assert straight.results["o"]["result"] == "zeta,alpha,mid|yes 0,yes 1,yes 2"

    stored = registry.load(reorder(json.dumps(record)))  # the record as the store gave it back
    assert stored.fingerprint == graph.fingerprint
    loaded = stored.load_run(reorder(first.to_json()))
    assert [w.id for w in loaded.waiting] == ids
    resumed = loaded.answer(answers).resume()
    assert resumed.results["o"] == straight.results["o"] and resumed.results["q"] == straight.results["q"]
    assert resumed.results["s"]["result"].equals(straight.results["s"]["result"])
    assert resumed.results["s"]["result"].columns == ["zeta", "alpha", "mid"]

    _CALLS.clear()
    again = stored.load_run(reorder(resumed.to_json())).again()  # the dict value read back in another order is the same value
    assert again.results["o"] == straight.results["o"] and _CALLS == []


@pytest.mark.parametrize("gone", ["first", "middle", "ask"])
def test_a_save_loaded_into_a_graph_that_lost_a_node_is_refused_with_the_graphs_own_problems(gone: str) -> None:
    @node("first", title="First", description="")
    def first(x: int) -> int:
        return x

    @node("middle", title="Middle", description="")
    def middle(x: int) -> int:
        return x

    @node("ask", title="Ask", description="")
    def ask(x: int) -> bool:
        raise Ask("ok?")

    full = registry_of(first, middle, ask)
    record = make_graph(first, middle, ask)
    saved = full.load(record).run(x=1).to_json()
    narrow = full.filter(lambda d: d.id != gone).load(record)

    with pytest.raises(InvalidCallError) as refused:
        narrow.load_run(saved)

    assert refused.value.errors() == narrow.problems and (narrow.problems[0].type, narrow.problems[0].loc[0]) == ("unknown_node", gone)


def test_a_float_that_is_not_finite_in_a_models_any_field_reads_back_as_the_float():
    class Reading(BaseModel):
        type_id: ClassVar[str] = "Reading"
        type_title: ClassVar[str] = "Reading"
        v: Any
        d: dict

    @node("reading", title="Reading", description="")
    def reading(x: int) -> Reading:
        return Reading(v=math.nan, d={"w": math.inf})

    @node("readings", title="Readings", description="")
    def readings(x: int) -> Series[Reading]:
        return [Reading(v=-math.inf, d={}), Reading(v=1, d={"s": "NaN"})]

    record = GraphRecord(nodes=[NodeRecord(id=n, node=n) for n in ("reading", "readings")])
    graph = registry_of(reading, readings).load(record)
    run = graph.run({"reading.x": 1, "readings.x": 1})
    _strict(run.to_json())
    back = _round_trip(run)

    one = back.results["reading"]["result"]
    assert math.isnan(one.v) and one.d == {"w": math.inf}
    first, second = back.results["readings"]["result"]
    assert first.v == -math.inf and second.d == {"s": "NaN"}


def test_a_float_that_is_not_finite_in_a_callers_dict_reads_back_as_the_float():
    @node("given", title="Given", description="")
    def given(options: dict, level: Any) -> str:
        return repr((options, level))

    graph = registry_of(given).load(make_graph(given))
    run = graph.run(options={"a": math.nan, "b": [math.inf]}, level=-math.inf)
    _strict(run.to_json())
    back = _round_trip(run)

    options = back.inputs["given.options"]
    assert math.isnan(options["a"]) and options["b"] == [math.inf] and back.inputs["given.level"] == -math.inf


def test_a_save_missing_a_key_says_which_key_not_a_python_error():
    @node("same", title="Same", description="")
    def same(text: str) -> str:
        return text

    graph = registry_of(same).load(make_graph(same))
    problem = _refused(graph, {})
    assert problem.ctx["at"] == "status" and "KeyError" not in problem.msg  # a save is read as the kind its status names, so the status is read first


@pytest.mark.parametrize(
    ("where", "damaged"),
    [
        ("units", lambda s: s.__setitem__("units", [])),
        ("inputs", lambda s: s.__setitem__("inputs", [])),
        ("status", lambda s: s.__setitem__("status", "banana")),
        ("max_items", lambda s: s.__setitem__("max_items", -1)),
        ("units.split.states", lambda s: s["units"]["split"].__setitem__("states", {})),
        ("units.split.keys.0.0", lambda s: s["units"]["split"].__setitem__("keys", [["x"]])),
        ("split.result", lambda s: s["units"]["split"]["outputs"]["result"].__setitem__("skipped", [7])),
    ],
)
def test_a_save_damaged_in_shape_is_refused_by_where(where: str, damaged: Any) -> None:
    graph = registry_of(split).load(make_graph(split))
    saved = json.loads(graph.run(text="a,b").to_json())
    damaged(saved)
    assert _refused(graph, saved).ctx["at"] == where


@node("upper_word", title="Upper word", description="")
def upper_word(word: str) -> str:
    return word.upper()


@node("ok_word", title="Ok word", description="")
def ok_word(word: str) -> bool:
    raise Ask("ok?")


@pytest.mark.parametrize(
    ("where", "damaged"),
    [
        ("units", lambda s: s.__setitem__("units", [])),
        ("earlier", lambda s: s.__setitem__("earlier", [])),
        ("inputs", lambda s: s.__setitem__("inputs", [])),
        ("code", lambda s: s.__setitem__("code", [])),
        ("complete", lambda s: s.__setitem__("complete", 3)),
        ("status", lambda s: s.__setitem__("status", "banana")),
        ("max_items", lambda s: s.__setitem__("max_items", "ten")),
        ("max_items", lambda s: s.__setitem__("max_items", -1)),
        ("units.upper.states", lambda s: s["units"]["upper"].__setitem__("states", {})),
        ("units.upper.keys.0.0", lambda s: s["units"]["upper"].__setitem__("keys", [["x"], ["y"]])),
        ("units.ok.questions", lambda s: s["units"]["ok"].__setitem__("questions", [])),
        ("units.ok.questions.0.prompt", lambda s: s["units"]["ok"]["questions"].__setitem__("0", {"inputs": {}})),
        ("units.upper.blank_inputs.0", lambda s: s["units"]["upper"].__setitem__("blank_inputs", {"0": 1})),
        ("units.upper.undeclared", lambda s: s["units"]["upper"].__setitem__("undeclared", {"0": -1})),  # a save holds only what it declares
        ("upper.result", lambda s: s["units"]["upper"]["outputs"]["result"].__setitem__("skipped", [7])),
        ("status", lambda s: s.clear()),  # an empty save: the status, which says what kind of save it is, is read first
        ("text", lambda s: s["inputs"].__setitem__("text", ["a"])),
    ],
)
def test_a_save_of_a_waiting_run_damaged_in_shape_is_refused_by_where(where: str, damaged: Any) -> None:
    graph = registry_of(split, upper_word, ok_word).load(
        {
            "nodes": [{"id": "split", "node": "split"}, {"id": "upper", "node": "upper_word"}, {"id": "ok", "node": "ok_word"}],
            "edges": [["split.result", "upper.word"], ["upper.result", "ok.word"]],
        }
    )
    run = graph.run(text="a,b")
    assert run.status == "waiting"
    saved = json.loads(run.to_json())
    damaged(saved)
    assert _refused(graph, saved).ctx["at"] == where


@node("addresses", title="Addresses", description="")
def addresses(text: str) -> Series[str | None]:
    return [part or None for part in text.split(",")]


@node("short_or_skipped", title="Short or skipped", description="")
def short_or_skipped(email: str) -> str:
    return email if len(email) < 3 else SKIPPED


@node("confirm", title="Confirm", description="")
def confirm(email: str) -> bool:
    raise Ask(f"Send to {email}?")


def test_a_blank_and_a_skipped_unit_come_back_from_a_save_and_a_resumed_run_names_the_same_blanks():
    graph = registry_of(addresses, short_or_skipped, confirm).load(
        {
            "nodes": [{"id": "a", "node": "addresses"}, {"id": "s", "node": "short_or_skipped"}, {"id": "c", "node": "confirm"}],
            "edges": [["a.result", "s.email"], ["s.result", "c.email"]],
        }
    )
    run = graph.run(text="ab,,long,cd")
    assert run.status == "waiting"
    assert list(run._units["s"].values()) == [Done({"result": "ab"}), Blank("email"), Skipped(), Done({"result": "cd"})]
    assert [type(u) for u in run._units["c"].values()] == [Asking, Blank, Skipped, Asking]
    assert run._units["c"][(1,)] == Blank("email")
    loaded = graph.load_run(run.to_json())
    assert loaded._units == run._units
    assert loaded.resume().not_called == run.not_called == [
        NotCalled(node_id="s", item=(1,), input="email"),
        NotCalled(node_id="c", item=(1,), input="email"),
    ]


_SAVED_IN_ANOTHER_PROCESS = """
from conductor import Ask, NodeRegistry, make_graph, node

@node("check", title="Check", description="")
def check(kind: str) -> str:
    if kind in {"invoice", "receipt", "order", "bill"}:
        raise Ask("ok?")
    return kind

registry = NodeRegistry()
registry.add(check)
print(registry.load(make_graph(check)).run(kind="invoice").waiting[0].id)
"""


@pytest.mark.parametrize("seeds", [("1", "2"), ("3", "4")])
def test_a_waiting_unit_has_one_id_in_every_process_though_its_node_tests_a_set(seeds):
    ids = {
        subprocess.run([sys.executable, "-c", _SAVED_IN_ANOTHER_PROCESS], env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True).stdout
        for seed in seeds
    }
    assert len(ids) == 1


_WRITTEN_IN_ANOTHER_PROCESS = """
from conductor import NodeRegistry, make_graph, node

@node("a", title="A", description="")
def a(text: str) -> str:
    return text

@node("b", title="B", description="")
def b(text: str) -> str:
    return text + "!"

@node("c", title="C", description="")
def c(text: str) -> str:
    return text + "?"

@node("d", title="D", description="")
def d(text: str) -> str:
    return text + "."

registry = NodeRegistry()
registry.add(a, b, c, d)
print(registry.load(make_graph(a, b, c, d)).run(text="x").to_json())
"""


def test_a_run_is_written_to_the_same_bytes_in_every_process():
    saved = {
        subprocess.run([sys.executable, "-c", _WRITTEN_IN_ANOTHER_PROCESS], env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, check=True).stdout
        for seed in ("1", "2", "3", "4")
    }
    assert len(saved) == 1


def test_a_save_whose_status_and_cause_disagree_is_refused_at_the_cause():
    """A run that ended early holds its cause, and one going on holds none: a save that says otherwise is not read."""

    @node("boom", title="Boom", description="Fails on 'x'.")
    def boom(text: str) -> str:
        if text == "x":
            raise ValueError("no")
        return text

    graph = registry_of(boom).load(make_graph(boom))
    failed, complete = json.loads(graph.run(text="x").to_json()), json.loads(graph.run(text="y").to_json())
    assert failed["status"] == "failed" and "cause" not in complete
    assert _refused(graph, {**failed, "cause": None}).ctx["at"] == "cause"
    assert _refused(graph, {**complete, "cause": failed["cause"]}).ctx["at"] == "cause"
    assert graph.load_run({**complete, "cause": None}).status == "complete"  # a save that wrote null for no cause still reads
