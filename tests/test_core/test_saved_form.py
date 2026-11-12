"""The saved form: a run reads back as itself and goes on to the same result, a float that isn't finite follows
its declared type, and a damaged save is refused at the place pydantic names."""

import dataclasses
import datetime
import json
import math
import random
from dataclasses import dataclass
from typing import Any, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    NodeRegistry,
    Run,
    Series,
    Table,
    host_type,
    node,
    subgraph,
)
from conductor._internal.run.json.write import saved_run
from pydantic import BaseModel


class Line(TypedDict):
    text: str
    size: float


@dataclass(frozen=True)
class Length:
    long: str
    short: str


@host_type("Reading", title="Reading")
class Reading(BaseModel):
    value: float
    note: Any = None


@node("lines", title="Lines", description="For a test.")
def lines(text: str) -> Series[Line]:
    return [{"text": part, "size": float(len(part))} for part in text.split(",") if part]


@node("gate", title="Gate", description="For a test.")
def gate(text: str) -> Length:
    return Length(text, SKIPPED) if len(text) > 2 else Length(SKIPPED, text)


@node("maybe", title="Maybe", description="For a test.")
def maybe(text: str) -> str | None:
    return None if text.startswith("b") else text.upper()


@node("letters", title="Letters", description="For a test.")
def letters(text: str) -> Series[str]:
    return [SKIPPED if letter == "x" else letter for letter in text]


@node("ask", title="Ask", description="For a test.")
def ask(text: str) -> str:
    if text.startswith("a"):
        raise Ask(f"What of {text}?")
    return text


@node("frame", title="Frame", description="For a test.")
def frame(text: str) -> Table:
    parts = text.split(",")
    return pl.DataFrame({
        "part": parts, "when": [datetime.date(2026, 1, 1 + i) for i in range(len(parts))], "tags": [[part, "t"] for part in parts],
        "ratio": [math.nan] * len(parts), "nothing": [None] * len(parts),
    })


@node("typed-frame", title="Typed frame", description="For a test.")
def typed_frame(text: str) -> Table[Line]:
    return pl.DataFrame({"size": [1.5, math.inf], "text": ["a", "b"], "extra": [datetime.time(1, 2), None]})


@node("reading", title="Reading", description="For a test.")
def reading(text: str) -> Reading:
    return Reading(value=math.nan, note=math.inf if text == "inf" else text)


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("loose", title="Loose", description="For a test.")
def loose(text: str) -> dict[str, Any]:
    return {"nan": math.nan, "when": datetime.date(2026, 1, 1), "text": text}


@node("float-or-text", title="Float or text", description="For a test.")
def float_or_text(text: str) -> float | str:
    return math.inf


def _registry() -> NodeRegistry:
    registry = NodeRegistry()
    registry.add(lines, gate, maybe, letters, ask, frame, typed_frame, reading, join, loose, float_or_text)
    inner = GraphRecord(nodes=[NodeRecord(id="q", node="ask"), NodeRecord(id="m", node="maybe")], edges=[("q.result", "m.text")])
    registry.add(subgraph("inner", title="Inner", description="For a test.")(inner))
    return registry


def _everything() -> Any:
    """A graph with a skipped output, a blank output, a skipped item, an empty Series, a waiting unit, an embedded
    graph and tables."""
    return Graph.compile(GraphRecord(
        nodes=[NodeRecord(id=n, node=kind) for n, kind in (
            ("lines", "lines"), ("gate", "gate"), ("maybe", "maybe"), ("letters", "letters"), ("ask", "ask"), ("frame", "frame"),
            ("typed", "typed-frame"), ("reading", "reading"), ("long", "join"), ("e", "inner"),
        )],
        edges=[
            ("lines.result.text", "gate.text"), ("lines.result.text", "maybe.text"), ("lines.result.text", "letters.text"),
            ("lines.result.text", "ask.text"), ("gate.long", "long.texts"), ("lines.result.text", "e.text"),
        ],
    ), _registry())


def _inputs(text: str) -> dict[str, str]:
    return {"lines.text": text, "frame.text": "p,q", "typed.text": "x", "reading.text": "x"}


def _same(a: Any, b: Any) -> bool:
    """Whether two results are the same, a table by its columns and a ``nan`` as itself."""
    if isinstance(a, pl.DataFrame):
        return isinstance(b, pl.DataFrame) and a.equals(b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, BaseModel):
        return type(a) is type(b) and _same(a.model_dump(), b.model_dump())
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return type(a) is type(b) and a == b


def _jsonb(text: str) -> str:
    """``text`` as Postgres ``jsonb`` gives it back: every object's keys shorter first, then bytewise."""

    def order(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: order(value[k]) for k in sorted(value, key=lambda k: (len(k.encode()), k.encode()))}
        return [order(v) for v in value] if isinstance(value, list) else value

    return json.dumps(order(json.loads(text)))


def _finished(run: Any) -> Any:
    while run.status == "waiting":
        for unit in run.waiting:
            run = run.answer(unit, "bee")
        run = run.resume()
    return run


@pytest.mark.parametrize("text", ["a,bb,ccc,xax,b", "", "abc", "x,xx,,b,aaaa", "zzz,yy"])
def test_a_run_reads_back_as_itself_and_goes_on_to_the_same_result(text: str) -> None:
    graph = _everything()
    run = graph.run(_inputs(text))
    saved = run.to_json()

    for written in (saved, saved.encode(), _jsonb(saved)):
        loaded = Run.from_json(written, graph)
        assert loaded.status == run.status and [w.id for w in loaded.waiting] == [w.id for w in run.waiting]
        for node_id, results in run.results.items():
            assert _same(loaded.results[node_id], results), node_id
        assert loaded.to_json() == saved
        assert _same(_finished(loaded).outputs, _finished(run).outputs)
        assert _same(_finished(loaded).results, _finished(run).results)


def test_a_save_holds_a_slot_for_each_instance_and_no_instance_is_written() -> None:
    saved = json.loads(_everything().run(_inputs("a,bb,ccc,xax,b")).to_json())

    assert saved["state"] == {"status": "waiting"}
    assert saved["outcomes"]["maybe"] == {"states": ["done"] * 5, "outputs": {"result": ["A", None, "CCC", "XAX", None]}}
    assert saved["outcomes"]["gate"] == {
        "states": ["done"] * 5, "outputs": {"long": [None, None, "ccc", "xax", None], "short": ["a", "bb", None, None, "b"]},
        "skipped": {"long": [0, 1, 4], "short": [2, 3]},
    }
    assert saved["outcomes"]["letters"]["skipped_items"] == {"result": {"3": [0, 2]}}
    assert saved["outcomes"]["ask"] == {
        "states": [None, "done", "done", "done", "done"], "outputs": {"result": [None, "bb", "ccc", "xax", "b"]},
        "questions": {"0": {"prompt": "What of a?", "inputs": {"text": "a"}}},
    }
    assert saved["outcomes"]["e/q"]["states"] == [None, "done", "done", "done", "done"]  # a node inside a subgraph, as any node
    assert saved["outcomes"]["e/m"]["states"] == [None, "done", "done", "done", "done"]  # the one that reads the waiting unit hasn't run
    assert "e" not in saved["outcomes"] and "keys" not in json.dumps(saved)


def test_a_run_stopped_part_way_holds_a_slot_for_each_instance_and_goes_on() -> None:
    graph = _everything()
    stopped = graph.run(_inputs("bb,ccc,dd,e"), max_items=2)
    assert stopped.status == "stopped"

    loaded = Run.from_json(stopped.to_json(), graph)

    assert _same(loaded.results, stopped.results)
    assert _same(loaded.with_max_items(100).resume().outputs, graph.run(_inputs("bb,ccc,dd,e")).outputs)


def test_a_run_again_after_a_save_takes_the_work_it_carried() -> None:
    graph = _everything()
    first = _finished(graph.run(_inputs("a,bb,ccc")))
    again = Run.from_json(first.to_json(), graph).with_inputs({"lines.text": "a,ccc,dd"})
    carried = Run.from_json(again.to_json(), graph)  # saved before a leg: its earlier work is in the save

    assert carried.to_json() == again.to_json()
    assert _same(_finished(carried.resume()).outputs, _finished(again.resume()).outputs)
    assert [w.id for w in carried.resume().waiting] == []  # the answer to "a" came with the carried work


def test_random_edits_of_a_save_load_or_are_refused_as_unreadable() -> None:
    """Whatever a save is damaged into, loading it gives a run or ``saved_run_unreadable``: never another error."""
    graph = _everything()
    first = _finished(graph.run(_inputs("a,bb,ccc,xax,b")))
    waiting = first.with_inputs({"lines.text": "a,ccc,ab,xax"}).resume()  # it waits on "ab", holding the work of "bb" and "b" it didn't take
    saved = json.loads(waiting.to_json())
    assert waiting.status == "waiting" and saved["reusable"] and saved["outcomes"]
    random.seed(7)

    def places(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        found = [path] if path else []
        for key, part in value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ():
            found += places(part, (*path, key))
        return found

    every = places(saved)
    for _ in range(3000):
        damaged = json.loads(json.dumps(saved))
        *above, last = random.choice(every)
        holder = damaged
        for step in above:
            holder = holder[step]
        change = random.choice(["drop", "null", "text", "number", "list", "object", "twice"])
        if change == "drop":
            del holder[last]
        elif change == "twice" and isinstance(holder, list):
            holder.insert(last, holder[last])
        else:
            holder[last] = {"null": None, "text": "x", "number": 99, "list": [1, [2]], "object": {"a": 1}, "twice": 7}[change]
        try:
            Run.from_json(json.dumps(damaged), graph).to_json()
        except InvalidCallError as refused:
            assert [p.type for p in refused.errors()] in (["saved_run_unreadable"], ["saved_run_of_another_graph"]), (above, last, change)


def test_a_float_that_is_not_finite_reads_back_by_its_declared_type() -> None:
    graph = Graph.compile(GraphRecord(nodes=[NodeRecord(id=n, node=n) for n in ("lines", "reading", "loose", "float-or-text", "typed-frame", "frame")]), _registry())
    run = graph.run({"lines.text": "ab", "reading.text": "inf", "loose.text": "x", "float-or-text.text": "x", "typed-frame.text": "x", "frame.text": "p"})
    json.loads(run.to_json(), parse_constant=lambda constant: pytest.fail(f"not strict JSON: {constant}"))

    results = Run.from_json(run.to_json(), graph).results

    assert math.isnan(results["reading"]["result"].value)  # a float field of a model: the float
    assert results["reading"]["result"].note == "Infinity"  # under Any: its JSON form, the text
    assert results["loose"]["result"] == {"nan": "NaN", "when": "2026-01-01", "text": "x"}  # under Any: its JSON form, as a date is
    assert results["float-or-text"]["result"] == "Infinity"  # a type that takes the text takes it
    assert results["typed-frame"]["result"]["size"].to_list() == [1.5, math.inf]  # a float column: the float
    assert results["frame"]["result"]["ratio"].is_nan().all()


@pytest.mark.parametrize(("damage", "at"), [
    (lambda s: s["outcomes"]["maybe"]["outputs"]["result"].__setitem__(1, {"secret": 1}), "outcomes.maybe.outputs.result.1"),
    (lambda s: s["outcomes"]["maybe"]["outputs"]["result"].append("X"), "outcomes.maybe"),  # a column longer than its states
    (lambda s: s["outcomes"]["maybe"]["states"].append("done"), "outcomes.maybe"),
    (lambda s: s["outcomes"]["maybe"]["states"].__setitem__(0, "gone"), "outcomes.maybe.states.0"),
    (lambda s: s["outcomes"]["maybe"].pop("outputs"), "outcomes.maybe.outputs"),
    (lambda s: s["outcomes"]["gate"]["outputs"].pop("long"), "outcomes.gate.outputs.long"),
    (lambda s: s["outcomes"]["gate"]["outputs"].__setitem__("other", []), "outcomes.gate.outputs.other"),
    (lambda s: s["outcomes"]["gate"]["skipped"]["long"].append(99), "outcomes.gate"),
    (lambda s: s["outcomes"]["letters"]["skipped_items"]["result"].__setitem__("0", [99]), "outcomes.letters"),
    (lambda s: s["outcomes"]["ask"]["questions"].__setitem__("77", {"prompt": "?", "inputs": {}}), "outcomes.ask"),
    (lambda s: s["outcomes"].__setitem__("nowhere", {"states": [], "outputs": {}}), "outcomes.nowhere"),
    (lambda s: s["inputs"].__setitem__("nowhere.text", "x"), "inputs.nowhere.text"),
    (lambda s: s["inputs"].__setitem__("lines.text", 7), "inputs.lines.text"),
    (lambda s: s["outcomes"]["frame"]["outputs"]["result"][0]["columns"][0]["values"].append("more"), "outcomes.frame.outputs.result.0.written"),
    (lambda s: s["outcomes"]["frame"]["outputs"]["result"][0]["columns"][1].__setitem__("name", "part"), "outcomes.frame.outputs.result.0.written"),
    (lambda s: s["outcomes"]["frame"]["outputs"]["result"][0]["columns"][1]["values"].__setitem__(0, "soon"), "outcomes.frame.outputs.result.0.written.columns.1.date.values.0"),
    (lambda s: s["outcomes"]["typed"]["outputs"]["result"][0]["columns"].pop(1), "outcomes.typed.outputs.result.0.written"),  # the column its row type names
    (lambda s: s["outcomes"]["e/q"]["states"].__setitem__(0, 5), "outcomes.e/q.states.0"),
    (lambda s: s.pop("state"), "state"),
    (lambda s: s.__setitem__("keys", []), "keys"),
])
def test_a_damaged_save_is_refused_at_the_place_pydantic_names(damage: Any, at: str) -> None:
    graph = _everything()
    saved = json.loads(graph.run(_inputs("a,bb,ccc,xax,b")).to_json())
    damage(saved)

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)

    (problem,) = refused.value.errors()
    assert (problem.type, problem.ctx["where"]) == ("saved_run_unreadable", at)
    assert "secret" not in problem.msg


def test_a_value_with_no_json_form_is_refused_at_the_save_naming_its_output() -> None:
    @node("odd", title="Odd", description="For a test.")
    def odd(text: str) -> Any:
        return object()

    registry = NodeRegistry()
    registry.add(odd)
    run = Graph.compile(GraphRecord(nodes=[NodeRecord(id="o", node="odd")]), registry).run(text="x")

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.errors()
    assert (problem.type, problem.ctx["where"]) == ("run_unwritable", "o.result")


def test_a_run_writes_its_outputs_as_json_by_their_types() -> None:
    @node("rows", title="Rows", description="For a test.")
    def rows(text: str) -> Series[Line]:
        return pl.DataFrame({"text": ["a"], "size": [math.inf]})  # type: ignore[return-value]

    registry = _registry()
    registry.add(rows)
    graph = Graph.compile(GraphRecord(
        nodes=[NodeRecord(id=n, node=n) for n in ("lines", "letters", "reading", "loose", "typed-frame", "rows")], edges=[("lines.result.text", "letters.text")],
    ), registry)
    run = graph.run({"lines.text": "ab,xa", "reading.text": "inf", "loose.text": "x", "typed-frame.text": "x", "rows.text": "x"})

    written = run.outputs_to_json()

    assert json.loads(written, parse_constant=lambda constant: pytest.fail(f"not strict JSON: {constant}")) == {
        "letters.result": [["a", "b"], [None, "a"]],  # a branch not taken for an item is a blank
        "reading.result": {"value": "NaN", "note": "Infinity"},  # a model writes a float that isn't finite as the run does
        "loose.result": {"nan": "NaN", "when": "2026-01-01", "text": "x"},
        "typed-frame.result": {"columns": [{"name": "text", "type": "str", "values": ["a", "b"]}, {"name": "size", "type": "float", "values": [1.5, "Infinity"]}]},
        "rows.result": [{"text": "a", "size": "Infinity"}],  # a Series of records is its records, however the run holds it
    }


def test_outputs_as_json_are_of_a_complete_run_and_a_value_with_no_json_form_names_its_output() -> None:
    graph = _everything()
    with pytest.raises(InvalidCallError) as refused:
        graph.run(_inputs("a")).outputs_to_json()
    assert [problem.type for problem in refused.value.errors()] == ["run_not_complete"]

    @node("odd", title="Odd", description="For a test.")
    def odd(text: str) -> Any:
        return object()

    registry = NodeRegistry()
    registry.add(odd)
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(GraphRecord(nodes=[NodeRecord(id="o", node="odd")]), registry).run(text="x").outputs_to_json()
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx["where"]) == ("run_unwritable", ("outputs", "o.result"), "o.result")


def test_a_field_added_to_a_saved_record_is_written_and_read_with_no_other_edit() -> None:
    """A save's keys are the fields of its records: one more field on a record is one more key, written and read."""
    from conductor._internal.run.json.codec import _inline_definitions, _with_field_schemas
    from conductor._internal.run.json.type import SavedOutcomes
    from pydantic import TypeAdapter
    from pydantic_core import SchemaSerializer, SchemaValidator, core_schema

    @dataclasses.dataclass(slots=True, kw_only=True)
    class Noted(SavedOutcomes):
        note: str = ""

    column = core_schema.list_schema(core_schema.nullable_schema(core_schema.str_schema()))
    shared: dict[str, Any] = {}
    record = _inline_definitions(TypeAdapter(Noted).core_schema, shared)  # the record's schema, as pydantic makes it from the record
    composed = _with_field_schemas(record, outputs=core_schema.typed_dict_schema({"result": core_schema.typed_dict_field(column)}))
    schema = core_schema.definitions_schema(composed, list(shared.values()))
    noted = Noted(states=["done", None], outputs={"result": ["a", None]}, not_called_because={1: "text"}, note="kept")

    written = SchemaSerializer(schema).to_json(noted, exclude_defaults=True)

    assert json.loads(written) == {"states": ["done", None], "outputs": {"result": ["a", None]}, "not_called_because": {"1": "text"}, "note": "kept"}
    assert SchemaValidator(schema).validate_json(written) == noted
    with pytest.raises(KeyError):
        _with_field_schemas(record, undeclared=column)  # composing for a field the record doesn't declare


def test_a_key_a_saved_record_does_not_declare_cannot_be_written() -> None:
    """Nothing reaches a save beside its records' fields, so a key can't be written that the reader doesn't know."""
    saved = saved_run(_everything().run(_inputs("a,bb")))
    for record in (saved, saved.outcomes["ask"]):
        with pytest.raises(AttributeError):
            record.undeclared = {0: "x"}
        with pytest.raises(TypeError):
            type(record)(**{declared.name: getattr(record, declared.name) for declared in dataclasses.fields(record)}, undeclared={0: "x"})


def test_every_field_a_run_sets_in_its_save_is_a_key_of_the_json_and_reads_back() -> None:
    graph = _everything()
    first = _finished(graph.run(_inputs("a,bb,ccc,xax,b")))
    waiting = first.with_inputs({"lines.text": "a,ccc,ab,xax"}).resume()
    saved = saved_run(waiting)
    written = json.loads(waiting.to_json())

    def set_fields(record: Any) -> set[str]:
        return {declared.name for declared in dataclasses.fields(record) if declared.default is dataclasses.MISSING or getattr(record, declared.name) is not None}

    assert set(written) == set_fields(saved)
    for name, units in saved.outcomes.items():
        assert set(written["outcomes"][name]) == set_fields(units), name
    for name, carried in saved.reusable.items():
        assert set(written["reusable"][name]) == set_fields(carried) and set(written["reusable"][name]["outcomes"]) == set_fields(carried.outcomes), name
    assert Run.from_json(waiting.to_json(), graph).to_json() == waiting.to_json()


def test_only_a_series_output_can_be_saved_with_items_on_a_branch_not_taken() -> None:
    """A one-value output's saved form has no ``skipped_items``, even when its value is a list; a Series' has."""

    @node("listed", title="Listed", description="For a test.")
    def listed(text: str) -> list[str]:
        return list(text)

    registry = NodeRegistry()
    registry.add(listed, letters)
    graph = Graph.compile(GraphRecord(nodes=[NodeRecord(id="listed", node="listed"), NodeRecord(id="letters", node="letters")]), registry)
    run = graph.run({"listed.text": "ax", "letters.text": "ax"})
    saved = json.loads(run.to_json())
    assert saved["outcomes"]["letters"]["skipped_items"] == {"result": {"0": [1]}} and "skipped_items" not in saved["outcomes"]["listed"]
    assert Run.from_json(json.dumps(saved), graph).to_json() == run.to_json()

    saved["outcomes"]["listed"]["skipped_items"] = {"result": {"0": [1]}}
    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)

    (problem,) = refused.value.errors()
    assert (problem.type, problem.ctx["where"]) == ("saved_run_unreadable", "outcomes.listed.skipped_items.result")
