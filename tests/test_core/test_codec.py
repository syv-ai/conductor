"""One codec: a value goes to JSON and back by its declared type.

``to_wire`` and ``from_wire`` are the one crossing between a value in
memory and its JSON form — the ledger's cells, a cached answer, a run's
return all go through them. A series carries its index and rows and comes
back on that index; a skip is marked beside a cell, never as a string in
the value's place.
"""

import json
from typing import Annotated, Any

import pytest
from conductor import CompiledGraph, GraphNode, NodeRegistry
from conductor.codec import from_wire, to_wire
from conductor.dtype import DType
from conductor.execution.ledger import Ledger
from conductor.execution.record import RunRecord
from conductor.graph.binding import Edges, Static
from conductor.graph.model import Graph
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Index, Series
from conductor.widgets import Textarea
from conductor_nodes.types import Flag, Json, Number, Text
from pydantic import TypeAdapter


class Txt(DType, str):
    id = "codec-test-text"
    title = "Text"


# -- the shipped types -------------------------------------------------------------


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
    back = from_wire(on_the_wire, dtype)
    assert back == value and type(back) is dtype


def test_any_passes_through_as_json_ready_data():
    assert to_wire({"k": (1, 2)}, Any) == {"k": [1, 2]}
    assert from_wire({"k": [1, 2]}, Any) == {"k": [1, 2]}


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
        from_wire(wire, Series[Json]),
        TypeAdapter(Series[Json]).validate_python(wire),
        TypeAdapter(Series[Json]).validate_json(json.dumps(wire)),
    ):
        assert isinstance(back, Series)
        assert (back.index, back.rows, back.values) == (lines, series.rows, series.values)
        assert back.index.parent == docs
        assert all(type(v) is Json for v in back.values)


def test_a_series_still_reads_a_plain_list_as_dense_rows_on_a_fresh_index():
    back = from_wire(["a", "b"], Series[Text])

    assert back.rows == ((0,), (1,)) and list(back) == ["a", "b"]


def test_a_series_has_no_wire_form_of_its_own():
    assert not hasattr(Series, "_as_wire")


# -- the ledger's cells ---------------------------------------------------------------


class Split(NodeDefinition):
    id = "split"
    title = "Split"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="In")] = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
        return [Txt(part) for part in text.split(",")]


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="In")] = Txt("")) -> Annotated[Txt, Result(title="Out")]:
        return Txt(text.upper())


def _compiled(*classes: type[NodeDefinition], nodes: list[GraphNode]) -> CompiledGraph:
    reg = NodeRegistry()
    for cls in classes:
        reg.register(cls)
    return CompiledGraph.from_graph(Graph(nodes=nodes), reg)


def test_a_text_that_spells_the_old_skip_marker_is_a_text_after_a_round_trip():
    compiled = _compiled(Split, Upper, nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="__skipped__,x")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": Edges(refs=(Ref("split", "result"),))}),
    ])
    ledger = Ledger(compiled)
    ledger.record(("split", None), {"result": [Txt("__skipped__"), Txt("x")]})
    ledger.record(("up", (0,)), {"result": Txt("__SKIPPED__")})
    ledger.record(("up", (1,)), {"result": Txt("X")})

    record = ledger.cells()
    restored = Ledger.restore(compiled, RunRecord.model_validate(json.loads(json.dumps(record.model_dump()))))

    values = restored.result_of("split")["result"]
    assert list(values) == ["__skipped__", "x"] and all(type(v) is Txt for v in values)
    assert list(restored.result_of("up")["result"]) == ["__SKIPPED__", "X"]
    assert all("skipped" not in cell for cell in record.cells)


def test_a_skip_is_marked_beside_the_cell_with_its_depth():
    from conductor import SKIPPED

    compiled = _compiled(Split, Upper, nodes=[
        GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b")}),
        GraphNode(id="up", type="upper", version=1, bindings={"text": Edges(refs=(Ref("split", "result"),))}),
    ])
    ledger = Ledger(compiled)
    ledger.record(("split", None), {"result": [Txt("a"), Txt("b")]})
    ledger.record(("up", (0,)), {"result": SKIPPED})
    ledger.record(("up", (1,)), {"result": Txt("B")})

    record = ledger.cells()
    skipped = [cell for cell in record.cells if "skipped" in cell]

    assert skipped == [{"ref": ["up", "result"], "row": [0], "skipped": 1}]
    assert all("value" not in cell for cell in skipped)
    restored = Ledger.restore(compiled, RunRecord.model_validate(json.loads(json.dumps(record.model_dump()))))
    assert list(restored.result_of("up")["result"]) == ["B"]
    assert restored.result_of("up")["result"].rows == ((1,),)


def test_a_value_with_no_json_form_raises_naming_the_field():
    class Opaque(DType):
        id = "codec-test-opaque"
        title = "Opaque"

        def __init__(self, handle: object) -> None:
            self.handle = handle

    class Opens(NodeDefinition):
        id = "opens"
        title = "Opens"
        description = "d"
        category = "test"

        def run(self, text: Annotated[Txt, Textarea(title="In")] = Txt("")) -> Annotated[Opaque, Result(title="Handle")]:
            return Opaque(object())

    compiled = _compiled(Opens, nodes=[GraphNode(id="o", type="opens", version=1, bindings={"text": Static(value="x")})])
    ledger = Ledger(compiled)
    ledger.record(("o", None), {"result": Opaque(object())})

    with pytest.raises(TypeError, match=r"o\.result"):
        ledger.cells()
