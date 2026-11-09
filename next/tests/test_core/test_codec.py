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
    Graph,
    GraphNode,
    NodeDefinition,
    NodeRegistry,
    Param,
    Result,
    Series,
    node,
    to_jsonable,
)
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
    reg = NodeRegistry()
    for cls in classes:
        reg = reg.extended_with([cls])
    return CompiledGraph.from_graph(Graph(nodes=nodes), reg)


