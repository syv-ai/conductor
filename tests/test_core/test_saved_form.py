"""Tests for a run saved as JSON and read back.

A saved run reads back as the same run, and resumes to the same result. A float that isn't finite, such as ``inf``
or ``nan``, is read back by the type its output declares. And a damaged saved run is refused, with a problem that
names the place pydantic found wrong.
"""

import asyncio
import datetime
import json
import math
import random
from collections.abc import Mapping, Sequence
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
    NodeRegistry,
    Run,
    Series,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import BaseModel


class Line(TypedDict):
    text: str
    size: float


@dataclass(frozen=True)
class Length:
    long: str = SKIPPED
    short: str = SKIPPED


@host_type("Reading", title="Reading")
class Reading(BaseModel, frozen=True):
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


@node("ask", title="Ask", description="For a test.", asks=True)
def ask(text: str) -> str:
    if text.startswith("a"):
        raise Ask(f"What of {text}?")
    return text


@node("frame", title="Frame", description="For a test.")
def frame(text: str) -> Table:
    parts = text.split(",")
    return pl.DataFrame(
        {
            "part": parts,
            "when": [datetime.date(2026, 1, 1 + i) for i in range(len(parts))],
            "tags": [[part, "t"] for part in parts],
            "ratio": [math.nan] * len(parts),
            "nothing": [None] * len(parts),
        }
    )


@node("typed-frame", title="Typed frame", description="For a test.")
def typed_frame(text: str) -> Table[Line]:
    return pl.DataFrame(
        {"size": [1.5, math.inf], "text": ["a", "b"], "extra": [datetime.time(1, 2), None]}
    )


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
    registry.add(
        lines, gate, maybe, letters, ask, frame, typed_frame, reading, join, loose, float_or_text
    )
    inner = GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {
                "q": {"node": "ask", "inputs": {"text": {"graph_input": "text"}}},
                "m": {
                    "node": "maybe",
                    "inputs": {"text": {"sources": [{"node": "q", "output": "result"}]}},
                },
            },
            "outputs": {"result": {"node": "m", "output": "result"}},
        }
    )
    registry.add(subgraph("inner", title="Inner", description="For a test.")(inner))
    return registry


def _record(
    registry: NodeRegistry,
    nodes: Mapping[str, str],
    edges: Sequence[tuple[str, str]],
    *,
    inputs: Mapping[str, str],
    outputs: Mapping[str, str],
) -> GraphRecord:
    """Returns ``make_graph`` of ``nodes``, joined by exactly ``edges``, with these graph ``inputs`` and ``outputs``.

    ``nodes`` maps each node id to the id of its definition in ``registry``.
    """
    return make_graph(
        *((node_id, registry[definition_id]) for node_id, definition_id in nodes.items()),
        edges=list(edges),
        inputs=inputs,
        outputs=outputs,
    )


def _everything() -> Any:
    """Returns a graph with a bit of everything a saved run has to keep.

    It has a skipped output, a ``None`` output, a skipped item, an empty series, a waiting call, a subgraph and
    tables.
    """
    registry = _registry()
    return Graph.compile(
        _record(
            registry,
            {
                "lines": "lines",
                "gate": "gate",
                "maybe": "maybe",
                "letters": "letters",
                "ask": "ask",
                "frame": "frame",
                "typed": "typed-frame",
                "reading": "reading",
                "long": "join",
                "e": "inner",
            },
            [
                ("lines.result.text", "gate.text"),
                ("lines.result.text", "maybe.text"),
                ("lines.result.text", "letters.text"),
                ("lines.result.text", "ask.text"),
                ("gate.long", "long.texts"),
                ("lines.result.text", "e.text"),
            ],
            inputs={name: f"{name}.text" for name in ("lines", "frame", "typed", "reading")},
            outputs={
                name: f"{name}.result"
                for name in ("maybe", "letters", "ask", "frame", "typed", "reading", "long", "e")
            }
            | {"short": "gate.short"},
        ),
        registry,
    )


def _inputs(text: str) -> dict[str, str]:
    return {"lines": text, "frame": "p,q", "typed": "x", "reading": "x"}


def _same(a: Any, b: Any) -> bool:
    """Whether two results are the same. Tables are compared by their columns, and a ``nan`` equals itself."""
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


def _finished(run: Any) -> Any:
    while run.status == "waiting":
        for call in run.waiting:
            run = run.with_answers(call, "bee")
        run = run.resume()
    return run


@pytest.mark.parametrize("text", ["a,bb,ccc,xax,b", "", "abc", "x,xx,,b,aaaa", "zzz,yy"])
def test_a_run_reads_back_as_itself_and_resumes_to_the_same_result(text: str) -> None:
    graph = _everything()
    run = graph.run(_inputs(text))
    saved = run.to_json()

    for written in (saved, saved.encode(), _jsonb(saved)):
        read_back = Run.from_json(written, graph)
        assert read_back.status == run.status and [w.id for w in read_back.waiting] == [
            w.id for w in run.waiting
        ]
        for node_id, results in run.results.items():
            assert _same(read_back.results[node_id], results), node_id
        assert read_back.to_json() == saved
        assert _same(_finished(read_back).outputs, _finished(run).outputs)
        assert _same(_finished(read_back).results, _finished(run).results)


def test_a_saved_run_has_one_list_entry_per_call_and_writes_no_item_indexes() -> None:
    saved = json.loads(_everything().run(_inputs("a,bb,ccc,xax,b")).to_json())
    fingerprints = {
        name: of_node.pop("fingerprints") for name, of_node in saved["outcomes"].items()
    }

    assert saved["ended_early"] is None
    # one fingerprint per call, for a waiting call too
    assert len(fingerprints["ask"]) == 5 and all(isinstance(f, str) for f in fingerprints["ask"])
    assert saved["outcomes"]["maybe"] == {
        "states": ["done"] * 5,
        "outputs": {"result": ["A", None, "CCC", "XAX", None]},
    }
    assert saved["outcomes"]["gate"] == {
        "states": ["done"] * 5,
        "outputs": {
            "long": [None, None, "ccc", "xax", None],
            "short": ["a", "bb", None, None, "b"],
        },
        "skipped": {"long": [0, 1, 4], "short": [2, 3]},
    }
    assert saved["outcomes"]["letters"]["skipped_items"] == {"result": {"3": [0, 2]}}
    assert saved["outcomes"]["ask"] == {
        "states": [None, "done", "done", "done", "done"],
        "outputs": {"result": [None, "bb", "ccc", "xax", "b"]},
        "questions": {"0": {"prompt": "What of a?"}},
    }
    assert saved["outcomes"]["e/q"]["states"] == [
        None,
        "done",
        "done",
        "done",
        "done",
    ]  # a node inside a subgraph is saved like any other node
    assert saved["outcomes"]["e/m"]["states"] == [
        None,
        "done",
        "done",
        "done",
        "done",
    ]  # m takes the waiting call's output, so it has no outcome either
    assert "e" not in saved["outcomes"] and "keys" not in json.dumps(saved)


def test_a_run_stopped_at_the_item_limit_reads_back_and_resumes_to_the_full_result() -> None:
    graph = _everything()
    stopped = graph.run(_inputs("bb,ccc,dd,e"), max_items=2)
    assert stopped.status == "stopped"

    read_back = Run.from_json(stopped.to_json(), graph)

    assert _same(read_back.results, stopped.results)
    assert _same(
        read_back.with_max_items(100).resume().outputs, graph.run(_inputs("bb,ccc,dd,e")).outputs
    )


def test_a_run_given_new_inputs_and_saved_before_resuming_keeps_its_reusable_outcomes() -> None:
    graph = _everything()
    first = _finished(graph.run(_inputs("a,bb,ccc")))
    again = Run.from_json(first.to_json(), graph).with_inputs({"lines": "a,ccc,dd"})
    restored = Run.from_json(
        again.to_json(), graph
    )  # saved before resume(), so the previous run's outcomes are still in it

    assert restored.to_json() == again.to_json()
    assert _same(_finished(restored.resume()).outputs, _finished(again.resume()).outputs)
    assert [
        w.id for w in restored.resume().waiting
    ] == []  # the answer to "a" is reused with the other outcomes


def _shout_graph(returned: Any) -> Any:
    """Returns ``each -> tag``, where ``tag`` labels each item with the next of ``returned``, so a reused outcome
    shows by its label."""

    @node("each", title="Each", description="For a test.")
    def each(items: list[str]) -> Series[str]:
        return items

    @node("tag", title="Tag", description="For a test.")
    def tag(text: str) -> str:
        if text == "boom":
            raise RuntimeError("boom")
        return f"{text} {next(returned)}"

    registry = NodeRegistry()
    registry.add(each, tag)
    return Graph.compile(make_graph(each, tag), registry)


def test_a_run_given_new_inputs_saves_each_outcome_it_may_reuse_once() -> None:
    graph = _shout_graph(iter(range(100)))
    first = graph.run({"items": ["a", "b", "c"]})

    saved = json.loads(first.with_inputs({"items": ["a", "b", "d"]}).to_json())["reusable"]["tag"]

    assert saved["outputs"]["result"] == ["a 0", "b 1", "c 2"]
    assert len(saved["fingerprints"]) == 3
    assert saved["item_indexes"] == {"0": [0], "1": [1], "2": [2]}


def test_an_outcome_whose_item_has_a_newer_outcome_is_still_reused_by_its_arguments() -> None:
    """``a 0`` loses its item, ``(0,)``, to ``b 1``, but a later call that gets ``"a"`` still reuses it.

    The run on ``"b"`` fails on ``"boom"``, so ``tag`` isn't complete and keeps the outcomes it may reuse.
    """
    graph = _shout_graph(iter(range(100)))
    failed = graph.run({"items": ["a"]}).again({"items": ["b", "boom"]})
    assert failed.status == "failed" and failed.results["tag"]["result"] == ["b 1", None]

    in_memory = failed.again({"items": ["b", "a"]}).results["tag"]["result"]
    read_back = Run.from_json(failed.with_inputs({"items": ["b", "a"]}).to_json(), graph).resume()

    assert in_memory == read_back.results["tag"]["result"] == ["b 1", "a 0"]


def test_a_saved_run_reuses_each_call_s_own_outcome_whatever_order_its_fingerprints_are_saved_in() -> (
    None
):
    """The two ``"a"`` calls keep their own outcomes, though their items, ``(0,)`` and ``(1,)``, also have the
    outcomes for ``"q"`` and ``"z"``. Read back with the fingerprints in the other order, they still do."""
    graph = _shout_graph(iter(range(100)))
    failed = graph.run({"items": ["q", "z"]}).again({"items": ["a", "a", "boom"]})
    assert failed.status == "failed" and failed.results["tag"]["result"] == ["a 2", "a 3", None]
    edited = failed.with_inputs({"items": ["a", "a"]})
    saved = json.loads(edited.to_json())
    tag = saved["reusable"]["tag"]
    reversed_places = list(range(len(tag["fingerprints"])))[::-1]
    reversed_saved = {
        **tag,
        "states": [tag["states"][p] for p in reversed_places],
        "outputs": {"result": [tag["outputs"]["result"][p] for p in reversed_places]},
        "fingerprints": [tag["fingerprints"][p] for p in reversed_places],
        "item_indexes": {
            str(place): tag["item_indexes"][str(p)] for place, p in enumerate(reversed_places)
        },
    }

    in_memory = edited.resume().results["tag"]["result"]
    read_back = Run.from_json(edited.to_json(), graph).resume().results["tag"]["result"]
    saved["reusable"]["tag"] = reversed_saved
    reordered = Run.from_json(json.dumps(saved), graph).resume().results["tag"]["result"]

    assert in_memory == read_back == reordered == ["a 2", "a 3"]


def test_a_saved_outcome_of_a_node_that_never_asks_without_its_item_indexes_is_refused() -> None:
    graph = _shout_graph(iter(range(100)))
    saved = json.loads(graph.run({"items": ["a", "a"]}).with_inputs({"items": ["a"]}).to_json())
    del saved["reusable"]["tag"]["item_indexes"]["1"]

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)

    assert [(p.type, p.ctx["where"]) for p in refused.value.problems] == [
        ("saved_run_unreadable", "reusable.tag")
    ]


def test_a_randomly_damaged_saved_run_reads_back_or_is_refused_with_a_problem() -> None:
    """However a saved run is damaged, ``Run.from_json`` either reads it back or refuses it with a problem.

    It never raises any other error.
    """
    graph = _everything()
    first = _finished(graph.run(_inputs("a,bb,ccc,xax,b")))
    waiting = first.with_inputs(
        {"lines": "a,ccc,ab,xax"}
    ).resume()  # it waits on "ab", and keeps its outcomes for "bb" and "b" to reuse
    saved = json.loads(waiting.to_json())
    assert waiting.status == "waiting" and saved["reusable"] and saved["outcomes"]
    random.seed(7)

    def places(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        found = [path] if path else []
        if isinstance(value, dict):
            parts = value.items()
        elif isinstance(value, list):
            parts = enumerate(value)
        else:
            parts = ()
        for key, part in parts:
            found += places(part, (*path, key))
        return found

    every = places(saved)
    for _ in range(3000):
        damaged = json.loads(json.dumps(saved))
        *above, last = random.choice(every)
        parent = damaged
        for step in above:
            parent = parent[step]
        change = random.choice(["drop", "null", "text", "number", "list", "object", "twice"])
        if change == "drop":
            del parent[last]
        elif change == "twice" and isinstance(parent, list):
            parent.insert(last, parent[last])
        else:
            parent[last] = {
                "null": None,
                "text": "x",
                "number": 99,
                "list": [1, [2]],
                "object": {"a": 1},
                "twice": 7,
            }[change]
        try:
            Run.from_json(json.dumps(damaged), graph).to_json()
        except InvalidCallError as refused:
            assert [p.type for p in refused.problems] in (
                ["saved_run_unreadable"],
                ["saved_run_of_another_graph"],
            ), (above, last, change)


def test_a_float_that_is_not_finite_reads_back_by_its_declared_type() -> None:
    names = ("lines", "reading", "loose", "float-or-text", "typed-frame", "frame")
    registry = _registry()
    graph = Graph.compile(
        _record(
            registry,
            {n: n for n in names},
            [],
            inputs={n: f"{n}.text" for n in names},
            outputs={n: f"{n}.result" for n in names},
        ),
        registry,
    )
    run = graph.run(
        {
            "lines": "ab",
            "reading": "inf",
            "loose": "x",
            "float-or-text": "x",
            "typed-frame": "x",
            "frame": "p",
        }
    )
    json.loads(
        run.to_json(), parse_constant=lambda constant: pytest.fail(f"not strict JSON: {constant}")
    )

    results = Run.from_json(run.to_json(), graph).results

    assert math.isnan(results["reading"]["result"].value)  # a float field of a model gets the float
    assert results["reading"]["result"].note == "Infinity"  # under Any: its JSON text
    assert results["loose"]["result"] == {
        "nan": "NaN",
        "when": "2026-01-01",
        "text": "x",
    }  # under Any, each value gets its JSON form, as a date does
    assert results["float-or-text"]["result"] == "Infinity"  # a type that takes text gets it
    assert results["typed-frame"]["result"]["size"].to_list() == [
        1.5,
        math.inf,
    ]  # a float column gets the float
    assert results["frame"]["result"]["ratio"].is_nan().all()


@pytest.mark.parametrize(
    ("damage", "at"),
    [
        (
            lambda s: s["outcomes"]["maybe"]["outputs"]["result"].__setitem__(1, {"secret": 1}),
            "outcomes.maybe.outputs.result.1",
        ),
        (
            lambda s: s["outcomes"]["maybe"]["outputs"]["result"].append("X"),
            "outcomes.maybe",
        ),  # an output's list longer than its states
        (lambda s: s["outcomes"]["maybe"]["states"].append("done"), "outcomes.maybe"),
        (
            lambda s: s["outcomes"]["maybe"]["states"].__setitem__(0, "gone"),
            "outcomes.maybe.states.0",
        ),
        (lambda s: s["outcomes"]["maybe"].pop("outputs"), "outcomes.maybe.outputs"),
        (lambda s: s["outcomes"]["gate"]["outputs"].pop("long"), "outcomes.gate.outputs.long"),
        (
            lambda s: s["outcomes"]["gate"]["outputs"].__setitem__("other", []),
            "outcomes.gate.outputs.other",
        ),
        (lambda s: s["outcomes"]["gate"]["skipped"]["long"].append(99), "outcomes.gate"),
        (
            lambda s: s["outcomes"]["letters"]["skipped_items"]["result"].__setitem__("0", [99]),
            "outcomes.letters",
        ),
        (
            lambda s: s["outcomes"]["ask"]["questions"].__setitem__("77", {"prompt": "?"}),
            "outcomes.ask",
        ),
        (
            lambda s: s["outcomes"].__setitem__("nowhere", {"states": [], "outputs": {}}),
            "outcomes.nowhere",
        ),
        (lambda s: s["inputs"].__setitem__("nowhere", "x"), "inputs.nowhere"),
        (lambda s: s["inputs"].__setitem__("lines", 7), "inputs.lines"),
        (
            lambda s: s["outcomes"]["frame"]["outputs"]["result"][0]["columns"][0]["values"].append(
                "more"
            ),
            "outcomes.frame.outputs.result.0",
        ),
        (
            lambda s: s["outcomes"]["frame"]["outputs"]["result"][0]["columns"][1].__setitem__(
                "name", "part"
            ),
            "outcomes.frame.outputs.result.0",
        ),
        (
            lambda s: s["outcomes"]["frame"]["outputs"]["result"][0]["columns"][1][
                "values"
            ].__setitem__(0, "soon"),
            "outcomes.frame.outputs.result.0",
        ),
        (
            lambda s: s["outcomes"]["typed"]["outputs"]["result"][0]["columns"].pop(1),
            "outcomes.typed.outputs.result.0",
        ),  # a column its row type names is gone
        (lambda s: s["outcomes"]["e/q"]["states"].__setitem__(0, 5), "outcomes.e/q.states.0"),
        (lambda s: s.pop("ended_early"), "ended_early"),
        (lambda s: s.__setitem__("keys", []), "keys"),
    ],
)
def test_a_damaged_saved_run_is_refused_at_the_place_pydantic_names(damage: Any, at: str) -> None:
    graph = _everything()
    saved = json.loads(graph.run(_inputs("a,bb,ccc,xax,b")).to_json())
    damage(saved)

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)

    (problem,) = refused.value.problems
    assert (problem.type, problem.ctx["where"]) == ("saved_run_unreadable", at)
    assert "secret" not in problem.msg


def test_a_value_with_no_json_form_is_refused_when_the_run_is_saved_naming_its_output() -> None:
    @node("odd", title="Odd", description="For a test.")
    def odd(text: str) -> Any:
        return object()

    registry = NodeRegistry()
    registry.add(odd)
    run = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {"o": {"node": "odd", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "o", "output": "result"}},
            }
        ),
        registry,
    ).run({"text": "x"})

    with pytest.raises(InvalidCallError) as refused:
        run.to_json()

    (problem,) = refused.value.problems
    assert (problem.type, problem.ctx["where"]) == ("run_unwritable", "o.result")


def test_a_run_writes_its_outputs_as_json_by_their_types() -> None:
    @node("rows", title="Rows", description="For a test.")
    def rows(text: str) -> Series[Line]:
        return pl.DataFrame({"text": ["a"], "size": [math.inf]})  # type: ignore[return-value]

    registry = _registry()
    registry.add(rows)
    graph = Graph.compile(
        _record(
            registry,
            {n: n for n in ("lines", "letters", "reading", "loose", "typed-frame", "rows")},
            [("lines.result.text", "letters.text")],
            inputs={n: f"{n}.text" for n in ("lines", "reading", "loose", "typed-frame", "rows")},
            outputs={
                n: f"{n}.result" for n in ("letters", "reading", "loose", "typed-frame", "rows")
            },
        ),
        registry,
    )
    run = graph.run(
        {"lines": "ab,xa", "reading": "inf", "loose": "x", "typed-frame": "x", "rows": "x"}
    )

    written = run.outputs_to_json()

    assert json.loads(
        written, parse_constant=lambda constant: pytest.fail(f"not strict JSON: {constant}")
    ) == {
        "letters": [["a", "b"], [None, "a"]],  # an item on a skipped branch is None
        "reading": {
            "value": "NaN",
            "note": "Infinity",
        },  # a model writes a float that isn't finite as the run does
        "loose": {"nan": "NaN", "when": "2026-01-01", "text": "x"},
        "typed-frame": {
            "columns": [
                {"name": "text", "type": "str", "values": ["a", "b"]},
                {"name": "size", "type": "float", "values": [1.5, "Infinity"]},
            ]
        },
        "rows": [
            {"text": "a", "size": "Infinity"}
        ],  # a series of records is written as its records, however the run stores it
    }


def test_outputs_as_json_are_of_a_complete_run_and_a_value_with_no_json_form_names_its_output() -> (
    None
):
    graph = _everything()
    with pytest.raises(InvalidCallError) as refused:
        graph.run(_inputs("a")).outputs_to_json()
    assert [problem.type for problem in refused.value.problems] == ["run_not_complete"]

    @node("odd", title="Odd", description="For a test.")
    def odd(text: str) -> Any:
        return object()

    registry = NodeRegistry()
    registry.add(odd)
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(
            GraphRecord.model_validate(
                {
                    "inputs": {"text": {}},
                    "nodes": {"o": {"node": "odd", "inputs": {"text": {"graph_input": "text"}}}},
                    "outputs": {"result": {"node": "o", "output": "result"}},
                }
            ),
            registry,
        ).run({"text": "x"}).outputs_to_json()
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx["where"]) == (
        "run_unwritable",
        ("outputs", "result"),
        "result",
    )


def test_a_key_a_saved_run_does_not_declare_is_refused_where_it_is() -> None:
    """A key that a saved run doesn't declare is refused, not dropped, and the problem says where it is.

    A saved run reads only the keys its records declare.
    """
    graph = _everything()
    written = json.loads(graph.run(_inputs("a,bb")).to_json())
    for record, where in (
        (written, "undeclared"),
        (written["outcomes"]["ask"], "outcomes.ask.undeclared"),
    ):
        record["undeclared"] = {"0": "x"}
        with pytest.raises(InvalidCallError) as refused:
            Run.from_json(json.dumps(written), graph)
        (problem,) = refused.value.problems
        assert (problem.type, problem.ctx["where"]) == ("saved_run_unreadable", where)
        del record["undeclared"]


def test_a_run_read_back_goes_on_as_the_run_it_was_saved_from() -> None:
    """A run read back from JSON goes on just as the run it was saved from would.

    The run reuses calls of the run before it, and is saved with what it may still reuse. Read back, it waits for the
    same calls and has the same results and progress. When it goes on, it reuses the same calls.
    """
    graph = _everything()
    first = _finished(graph.run(_inputs("a,bb,ccc,xax,b")))
    waiting = first.with_inputs({"lines": "a,ccc,ab,xax"}).resume()
    read_back = Run.from_json(waiting.to_json(), graph)

    assert waiting.status == "waiting"
    assert (read_back.status, read_back.waiting, read_back.progress) == (
        waiting.status,
        waiting.waiting,
        waiting.progress,
    )
    assert _same(read_back.results, waiting.results)
    assert _went_on(read_back) == _went_on(waiting)


def _went_on(run: Any) -> list[tuple[Any, ...]]:
    """Answers the waiting calls of ``run``, lets it go on, and returns what each node did.

    Each item is a node's place, how many outcomes it has and how many of them were reused.
    """
    for call in run.waiting:
        run = run.with_answers(call, "bee")

    async def streamed() -> list[Any]:
        return [event async for event in run.stream()]

    return [
        (event.loc, event.count.finished, event.reused)
        for event in asyncio.run(streamed())
        if event.type == "node_complete"
    ]


def test_only_a_series_output_can_be_saved_with_items_on_a_skipped_branch() -> None:
    """Only a ``Series`` output can be saved with items on a skipped branch.

    An output that isn't a series is saved with no ``skipped_items``, even when its value is a list.
    """

    @node("listed", title="Listed", description="For a test.")
    def listed(text: str) -> list[str]:
        return list(text)

    registry = NodeRegistry()
    registry.add(listed, letters)
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"listed_text": {}, "letters_text": {}},
                "nodes": {
                    "listed": {
                        "node": "listed",
                        "inputs": {"text": {"graph_input": "listed_text"}},
                    },
                    "letters": {
                        "node": "letters",
                        "inputs": {"text": {"graph_input": "letters_text"}},
                    },
                },
                "outputs": {
                    "letters_result": {"node": "letters", "output": "result"},
                    "listed_result": {"node": "listed", "output": "result"},
                },
            }
        ),
        registry,
    )
    run = graph.run({"listed_text": "ax", "letters_text": "ax"})
    saved = json.loads(run.to_json())
    assert (
        saved["outcomes"]["letters"]["skipped_items"] == {"result": {"0": [1]}}
        and "skipped_items" not in saved["outcomes"]["listed"]
    )
    assert Run.from_json(json.dumps(saved), graph).to_json() == run.to_json()

    saved["outcomes"]["listed"]["skipped_items"] = {"result": {"0": [1]}}
    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(json.dumps(saved), graph)

    (problem,) = refused.value.problems
    assert (problem.type, problem.ctx["where"]) == (
        "saved_run_unreadable",
        "outcomes.listed.skipped_items.result",
    )
