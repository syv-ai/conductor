"""A blank reaching a node: the hint says whether it is called."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Optional

import polars as pl
import pytest
from conductor import (
    AsItArrives,
    AskPerson,
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    Input,
    NodeDeclarationError,
    NodeRegistry,
    Output,
    Param,
    Result,
    Static,
    Table,
    node,
)
from conductor.dtype import DType
from conductor.events import NodeComplete, NodeSkipped
from conductor.interface import read_signature
from conductor.series import Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "blanks-txt"


class Int(DType, int):
    id = "blanks-int"


CALLED: list[Any] = []


@node("rows")
def rows() -> Table:
    return Table(pl.DataFrame({"name": ["Ida", "Bo", "Nina"], "email": ["ida@x", None, "nina@x"]}))


@node("rows2", compute_outputs=lambda: [Output("result", Table.with_columns({"name": Txt, "email": Txt}))])
def rows2() -> Mapping[str, Any]:
    return {"result": rows.versions[1].run()}


@node("plain")
def plain(email: Txt) -> Txt:
    CALLED.append(email)
    return Txt(email.upper())


@node("takes")
def takes(email: Txt | None) -> Txt:
    return Txt("none" if email is None else email)


@dataclass(frozen=True)
class Branched:
    known: Annotated[Txt, Result(choice="b")]
    unknown: Annotated[Txt, Result(choice="b")]


@node("branched")
def branched(email: Txt) -> Branched:
    return Branched(email, Txt(""))


@node("after-branch")
def after_branch(value: Txt) -> Txt:
    return value


@node("splits")
def splits(email: Txt) -> Series[Txt]:
    return [Txt(c) for c in email]


@node("count")
def count(values: Series[Any]) -> Int:
    return Int(len(values))


@node("join")
def join(values: Series[Txt]) -> Txt:
    return Txt(",".join(values))


@node("keep")
def keep(values: Series[Txt | None]) -> Int:
    return Int(sum(1 for value in values if value is None))


@node("zipped")
def zipped(a: Series[Txt], b: Series[Txt]) -> Txt:
    return Txt("|".join(f"{x}{y}" for x, y in zip(a, b, strict=True)))


@node("make-table")
def make_table(email: Txt) -> Table:
    return Table(pl.DataFrame({"x": [email]}))


def _registry(*more):
    return NodeRegistry([rows, rows2, plain, takes, branched, after_branch, splits, count, join, keep, zipped, make_table, *more])


def _results(*nodes, registry=None):
    compiled = CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry or _registry())
    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert state.status == "complete", state.failure
    return compiled, state, compiled.results(state)


CSV = GraphNode(id="csv", type="rows2", version=1)


def _per_row(type_id, **bindings):
    return GraphNode(id="n", type=type_id, version=1, bindings=bindings or {"email": From("csv.result.email")})


# --- what can be written --------------------------------------------------------


@pytest.mark.parametrize("annotation, takes_blank", [
    (Txt, False),
    (Txt | None, True),
    (Optional[Txt], True),
    (Annotated[Txt | None, Param(title="X")], True),
    (Annotated[Txt, Param(title="X")] | None, True),
    (Any, True),
    (Series[Txt], False),
    (Series[Txt | None], True),
    (Series[Any], True),
])
def test_an_input_takes_a_blank_when_its_hint_says_so(annotation, takes_blank):
    def run(x): ...

    run.__annotations__ = {"x": annotation, "return": Txt}
    (inp,) = read_signature(run).interface.inputs
    assert inp.takes_blank is takes_blank


def test_the_bare_type_is_what_travels_on_an_edge():
    def run(x: Series[Txt | None], y: Txt | None) -> Txt | None: ...

    x, y = read_signature(run).interface.inputs
    assert x.dtype is Series[Txt] and y.dtype is Txt


def test_an_output_takes_a_blank_when_its_hint_says_so():
    def run(x: Txt) -> Annotated[Txt | None, Result(title="R")]: ...

    (out,) = read_signature(run).interface.outputs
    assert out.dtype is Txt and out.takes_blank


def test_a_record_field_takes_a_blank_when_its_hint_says_so():
    @dataclass(frozen=True)
    class Parts:
        a: Txt | None
        b: Txt

    def run(x: Txt) -> Parts: ...

    a, b = read_signature(run).interface.outputs
    assert a.takes_blank and not b.takes_blank


def test_a_field_built_by_a_hook_says_it_the_same_way():
    assert Input("x", Txt | None).takes_blank and Output("x", Txt | None).takes_blank
    assert not Input("x", Txt).takes_blank


def test_as_it_arrives_takes_a_blank():
    def run(**inputs: AsItArrives) -> Mapping[str, Any]: ...

    declared = node("n", compute_outputs=lambda: [])(run)
    assert declared.versions[1].open_input.takes_blank


def test_a_none_default_on_an_input_that_takes_no_blank_is_refused():
    def run(x: Txt = None) -> Txt: ...

    with pytest.raises(NodeDeclarationError) as raised:
        node("n")(run)
    assert raised.value.code == "none_default_needs_blank"


def test_a_none_default_with_a_blank_hint_is_fine():
    def run(x: Txt | None = None) -> Txt: ...

    (x,) = node("n")(run).versions[1].interface.inputs
    assert x.takes_blank and not x.required and x.default is None


# --- a node is not called on a blank unless it says so ---------------------------


def test_a_blank_on_a_plain_input_is_not_called_and_its_outputs_are_blank():
    CALLED.clear()
    compiled, _, results = _results(CSV, _per_row("plain"))

    assert CALLED == ["ida@x", "nina@x"]
    assert results["n"]["result"].values == ("IDA@X", None, "NINA@X")


def test_an_input_saying_none_is_called_with_the_blank():
    _, _, results = _results(CSV, _per_row("takes"))

    assert results["n"]["result"].values == ("ida@x", "none", "nina@x")


def test_a_default_is_for_an_input_nothing_is_wired_to_not_for_a_blank():
    @node("defaulted")
    def defaulted(x: Txt = Txt("fallback")) -> Txt:
        return x

    _, _, results = _results(CSV, GraphNode(id="n", type="defaulted", version=1, bindings={"x": From("csv.result.email")}), registry=_registry(defaulted))
    assert results["n"]["result"].values == ("ida@x", None, "nina@x")


def test_the_event_counts_the_rows_not_called_and_the_state_keeps_them():
    compiled = CompiledGraph.from_graph(Graph(nodes=[CSV, _per_row("plain")]), _registry())

    async def collect():
        async with compiled.stream() as events:
            return [event async for event in events]

    import asyncio
    events = asyncio.run(collect())
    complete = next(e for e in events if isinstance(e, NodeComplete) and e.node_id == "n")
    assert complete.rows_not_called == {"email": 1}
    state = events[-1].state
    assert state.rows_not_called == {"n": {"email": 1}}
    assert compiled.results(compiled.run_sync(state))["n"]["result"].values[1] is None


def test_a_node_all_blank_completes_it_is_never_skipped():
    @node("nothing")
    def nothing() -> Txt | None:
        return None

    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="s", type="nothing", version=1),
        GraphNode(id="n", type="plain", version=1, bindings={"email": From("s.result")}),
    ]), _registry(nothing))

    async def collect():
        async with compiled.stream() as events:
            return [event async for event in events]

    import asyncio
    events = asyncio.run(collect())
    assert not [e for e in events if isinstance(e, NodeSkipped) and e.node_id == "n"]
    assert next(e for e in events if isinstance(e, NodeComplete) and e.node_id == "n").rows_not_called == {"email": 1}


def test_choice_outputs_are_skipped_when_not_called():
    _, _, results = _results(
        CSV, _per_row("branched"),
        GraphNode(id="k", type="after-branch", version=1, bindings={"value": From("n.known")}),
    )
    assert results["n"]["known"].rows == ((0,), (2,))
    assert results["k"]["result"].rows == ((0,), (2,))


def test_a_row_making_output_has_no_rows_when_not_called_so_a_count_gives_zero():
    _, _, results = _results(
        CSV, _per_row("splits"),
        GraphNode(id="c", type="count", version=1, bindings={"values": From("n.result")}),
    )
    assert results["c"]["result"].values == (5, 0, 6)


def test_a_table_output_has_no_rows_when_not_called():
    _, state, results = _results(CSV, _per_row("make-table"))
    assert results["n"]["result"].values[1].frame.height == 0
    assert state.status == "complete"


def test_a_series_leaves_blanks_out():
    _, _, results = _results(CSV, GraphNode(id="n", type="join", version=1, bindings={"values": From("csv.result.email")}))
    assert results["n"]["result"] == "ida@x,nina@x"


def test_a_series_saying_none_keeps_them():
    _, _, results = _results(CSV, GraphNode(id="n", type="keep", version=1, bindings={"values": From("csv.result.email")}))
    assert results["n"]["result"] == 1


def test_series_on_one_index_lose_a_blank_row_together():
    _, _, results = _results(CSV, GraphNode(id="n", type="zipped", version=1, bindings={
        "a": From("csv.result.name"), "b": From("csv.result.email"),
    }))
    assert results["n"]["result"] == "Idaida@x|Ninanina@x"


def test_a_static_none_needs_the_hint():
    plain_static = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="plain", version=1, bindings={"email": Static(None)})]), _registry())
    takes_static = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="takes", version=1, bindings={"email": Static(None)})]), _registry())

    assert [p.type for p in plain_static.problems] == ["invalid_static"]
    assert takes_static.is_runnable


def test_returning_none_needs_the_hint():
    @node("bad")
    def bad(x: Txt) -> Txt:
        return None

    @node("good")
    def good(x: Txt) -> Txt | None:
        return None

    registry = _registry(bad, good)
    failed = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="bad", version=1, bindings={"x": Static("a")})]), registry).run_sync()
    fine = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="good", version=1, bindings={"x": Static("a")})]), registry).run_sync()

    assert failed.status == "failed" and failed.failure.type == "invalid_output"
    assert fine.status == "complete"


def test_a_question_with_a_blank_hint_may_be_left_out():
    @node("ask")
    def ask(x: Txt) -> Annotated[Txt | None, Result(title="A")] | AskPerson:
        return AskPerson()

    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="ask", version=1, bindings={"x": Static("a")})]), _registry(ask))
    waiting = compiled.run_sync()
    unit = waiting.waiting[0]
    assert unit.questions[0].optional
    answered = compiled.answer(waiting, {unit.id: {}})
    assert compiled.results(compiled.run_sync(answered))["n"]["result"] is None


def test_a_reader_of_a_blank_chain_is_not_called_either():
    CALLED.clear()
    _, _, results = _results(CSV, _per_row("plain"), GraphNode(id="m", type="plain", version=1, bindings={"email": From("n.result")}))
    assert results["m"]["result"].values == ("IDA@X", None, "NINA@X")
    assert len(CALLED) == 4


def test_a_blank_in_a_widget_input_is_read_through_the_annotated_form():
    @node("wid")
    def wid(x: Annotated[Txt | None, Param(widget=Textarea())] = None) -> Txt:
        return Txt("empty" if x is None else x)

    _, _, results = _results(GraphNode(id="n", type="wid", version=1), registry=_registry(wid))
    assert results["n"]["result"] == "empty"


def test_a_blank_on_a_state_round_trips_through_its_json():
    compiled = CompiledGraph.from_graph(Graph(nodes=[CSV, _per_row("plain")]), _registry())
    state = compiled.run_sync()

    restored = type(state).model_validate_json(state.model_dump_json())

    assert restored.rows_not_called == {"n": {"email": 1}}
    assert compiled.results(compiled.run_sync(restored))["n"]["result"].values == ("IDA@X", None, "NINA@X")
