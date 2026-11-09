"""One codec: a value goes to JSON and back by its declared type.

``to_wire`` and pydantic's reading back are the one crossing between a value in
memory and its JSON form — the ledger's values, a cached answer, a run's
return all go through them. A series carries its index and rows and comes
back on that index; a skip is marked beside its address, never as a string in
the value's place.
"""

import json
from typing import Annotated, Any

import pytest
from conductor import (
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeDefinition,
    NodeRegistry,
    Param,
    Result,
    RunState,
    Series,
    Static,
    node,
    to_wire,
)
from conductor._internal.execution.live_run_values import LiveRunValues
from conductor._internal.series import Index
from conductor.widgets import Textarea
from conductor_nodes.types import Json
from pydantic import TypeAdapter


class Txt(DType, str):
    id = "codec-test-text"
    display_title = "Text"


class Text(DType, str):
    id = "text"


class Number(DType, float):
    id = "number"


class Flag(DType, int):
    id = "flag"


# -- types declared with DType -------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "value", "wire"),
    [
        (Text, Text("hej"), "hej"),
        (Number, Number(2.5), 2.5),
        (Flag, Flag(True), 1),
        (Json, Json({"a": [1, "b", None]}), {"a": [1, "b", None]}),
        (Json, Json(3), 3),
    ],
)
def test_each_shipped_type_round_trips(dtype, value, wire):
    on_the_wire = to_wire(value, dtype)

    assert on_the_wire == wire
    assert json.loads(json.dumps(on_the_wire)) == wire
    back = TypeAdapter(dtype).validate_python(on_the_wire)
    assert back == value and type(back) is dtype


def test_any_passes_through_as_json_ready_data():
    assert to_wire({"k": (1, 2)}, Any) == {"k": [1, 2]}
    assert TypeAdapter(Any).validate_python({"k": [1, 2]}) == {"k": [1, 2]}


# -- a series -------------------------------------------------------------------------


def test_a_series_of_json_round_trips_in_both_modes():
    docs = Index("docs")
    lines = Index("lines", parent=docs)
    series = Series[Json](lines, [Json({"n": 1}), Json([2])], rows=((0, 0), (1, 0)))

    wire = to_wire(series, Series[Json])

    assert wire == {
        "index": {"id": "lines", "parent": {"id": "docs", "parent": None}},
        "rows": [[0, 0], [1, 0]],
        "values": [{"n": 1}, [2]],
    }
    for back in (
        TypeAdapter(Series[Json]).validate_python(wire),
        TypeAdapter(Series[Json]).validate_python(wire),
        TypeAdapter(Series[Json]).validate_json(json.dumps(wire)),
    ):
        assert isinstance(back, Series)
        assert (back.index, back.rows, back.values) == (lines, series.rows, series.values)
        assert back.index.parent == docs
        assert all(type(v) is Json for v in back.values)


def test_a_series_still_reads_a_plain_list_as_dense_rows_on_a_fresh_index():
    back = TypeAdapter(Series[Text]).validate_python(["a", "b"])

    assert back.rows == ((0,), (1,)) and list(back) == ["a", "b"]


# -- the ledger's values --------------------------------------------------------------


@node("split", title="Split", description="d", category="test")
def split_node(text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
    return [Txt(part) for part in text.split(",")]


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Out")]:
    return Txt(text.upper())


def _compiled(*classes: type[NodeDefinition], nodes: list[GraphNode]) -> CompiledGraph:
    reg = NodeRegistry()
    for cls in classes:
        reg = reg.extended_with([cls])
    return CompiledGraph.from_graph(Graph(nodes=nodes), reg)


def test_a_text_that_spells_the_old_skip_marker_is_a_text_after_a_round_trip():
    compiled = _compiled(split_node, upper_node, nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static("__skipped__,x")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("split.result")}),
    ])
    ledger = LiveRunValues(compiled)
    ledger.record(("split", ()), {"result": [Txt("__skipped__"), Txt("x")]})
    ledger.record(("up", (0,)), {"result": Txt("__SKIPPED__")})
    ledger.record(("up", (1,)), {"result": Txt("X")})

    state = ledger.state("complete")
    restored = LiveRunValues.restore(compiled, RunState.model_validate(json.loads(state.model_dump_json())))

    values = restored.result_of("split")["result"]
    assert list(values) == ["__skipped__", "x"] and all(type(v) is Txt for v in values)
    assert list(restored.result_of("up")["result"]) == ["__SKIPPED__", "X"]
    assert state.output_columns["split.result"].skipped == []


def test_a_skip_is_listed_beside_the_values():
    from conductor import SKIPPED

    compiled = _compiled(split_node, upper_node, nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static("a,b")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("split.result")}),
    ])
    ledger = LiveRunValues(compiled)
    ledger.record(("split", ()), {"result": [Txt("a"), Txt("b")]})
    ledger.record(("up", (0,)), {"result": SKIPPED})
    ledger.record(("up", (1,)), {"result": Txt("B")})

    state = ledger.state("complete")
    wire = json.loads(state.model_dump_json())["output_columns"]

    assert wire["up.result"] == {"rows": [[1]], "values": ["B"], "skipped": [[0]]}
    assert wire["split.result"] == {"rows": [[0], [1]], "values": ["a", "b"], "skipped": []}


def test_a_value_with_no_json_form_raises_naming_the_field():
    class Opaque(DType):
        id = "codec-test-opaque"
        display_title = "Opaque"

        def __init__(self, handle: object) -> None:
            self.handle = handle

    @node("opens", title="Opens", description="d", category="test")
    def opens(text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Opaque, Result(title="Handle")]:
        return Opaque(object())

    compiled = _compiled(opens, nodes=[GraphNode(id="o", type="opens", version=1, bindings={"text": Static("x")})])
    ledger = LiveRunValues(compiled)
    ledger.record(("o", ()), {"result": Opaque(object())})

    with pytest.raises(Exception, match=r"o\.result"):
        ledger.state("complete").model_dump_json()
