"""One codec: a value goes to JSON and back by its declared type.

``to_jsonable`` and pydantic's reading back are the one crossing between a value in
memory and its JSON form — the ledger's values, a cached answer, a run's
return all go through them. A skip is marked beside its address, never as a
string in the value's place.
"""

import json
from typing import Annotated

import pytest
from conductor import (
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeDefinition,
    Param,
    Result,
    RunState,
    Series,
    Static,
    node,
    to_jsonable,
)
from conductor._internal.execution.run_ledger import RunLedger
from conductor.widgets import Textarea
from helpers import Json, registry_of
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
    on_the_wire = to_jsonable(value)

    assert on_the_wire == wire
    assert json.loads(json.dumps(on_the_wire)) == wire
    back = TypeAdapter(dtype).validate_python(on_the_wire)
    assert back == value and type(back) is dtype


def test_plain_data_comes_out_json_ready_with_the_typed_values_inside_written_by_their_own_type():
    assert to_jsonable({"k": (1, Json({"a": 2}), {"rows": [(0,)], "values": [Text("b")]})}) == {"k": [1, {"a": 2}, {"rows": [[0]], "values": ["b"]}]}


def test_a_float_survives_the_writer_to_the_last_bit():
    values = [0.1 + 0.2, 1e308, 5e-324, -0.0, 2**70]
    assert to_jsonable(values) == values


@pytest.mark.parametrize(("value", "written"), [(float("nan"), "NaN"), (float("inf"), "Infinity"), (float("-inf"), "-Infinity"), (Number(float("nan")), "NaN"), ({"a": [float("nan")]}, {"a": ["NaN"]})])
def test_a_float_that_is_not_finite_is_text_at_any_depth(value, written):
    assert to_jsonable(value) == written
    assert to_jsonable(value) == json.loads(json.dumps(to_jsonable(value), allow_nan=False))


# -- the ledger's values --------------------------------------------------------------


@node("split", title="Split", description="d", category="test")
def split_node(text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
    return [Txt(part) for part in text.split(",")]


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Out")]:
    return Txt(text.upper())


def _compiled(*classes: type[NodeDefinition], nodes: list[GraphNode]) -> CompiledGraph:
    reg = registry_of(*classes)
    return CompiledGraph.from_graph(Graph(nodes=nodes), reg)


def test_a_text_that_spells_the_old_skip_marker_is_a_text_after_a_round_trip():
    compiled = _compiled(split_node, upper_node, nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static("__skipped__,x")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("split.result")}),
    ])
    ledger = RunLedger(compiled)
    ledger.record(("split", ()), {"result": [Txt("__skipped__"), Txt("x")]})
    ledger.record(("up", (0,)), {"result": Txt("__SKIPPED__")})
    ledger.record(("up", (1,)), {"result": Txt("X")})

    state = ledger.state("complete")
    restored = RunLedger.restore(compiled, RunState.model_validate(json.loads(state.model_dump_json())))

    values = restored._values.result_of("split")["result"]["values"]
    assert values == ["__skipped__", "x"] and all(type(v) is Txt for v in values)
    assert restored._values.result_of("up")["result"]["values"] == ["__SKIPPED__", "X"]
    assert state.nodes["split"].outputs["result"].skipped == []


def test_a_skip_is_listed_beside_the_values():
    from conductor import SKIPPED

    compiled = _compiled(split_node, upper_node, nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static("a,b")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("split.result")}),
    ])
    ledger = RunLedger(compiled)
    ledger.record(("split", ()), {"result": [Txt("a"), Txt("b")]})
    ledger.record(("up", (0,)), {"result": SKIPPED})
    ledger.record(("up", (1,)), {"result": Txt("B")})

    state = ledger.state("complete")
    wire = json.loads(state.model_dump_json())["nodes"]

    assert wire["up"]["outputs"]["result"] == {"rows": [[1]], "values": ["B"], "skipped": [[0]]}
    assert wire["split"]["outputs"]["result"] == {"rows": [[]], "values": [["a", "b"]], "skipped": []}


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
    ledger = RunLedger(compiled)
    ledger.record(("o", ()), {"result": Opaque(object())})

    with pytest.raises(Exception, match=r"o\.result"):
        ledger.state("complete").model_dump_json()
