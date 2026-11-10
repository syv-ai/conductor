"""What a host saves is a pydantic model, and a graph saves itself."""

import dataclasses

import pytest
from conductor import (
    CompiledGraph,
    Deprecation,
    DType,
    GraphRecord,
    Input,
    Interface,
    NodeDescription,
    NodeRecord,
    Output,
    Param,
    Policy,
    Problem,
    Result,
)
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.execution.run_ledger import SkippedAtRow
from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
from conductor._internal.index import Index
from conductor._internal.node_definition import NodeVersion, VersionDescription
from conductor.widgets import Choice, Dropdown, Textarea, Widget
from pydantic import ValidationError


class Txt(DType, str):
    id = "model-test-txt"
    display_title = "Text"


def _graph() -> GraphRecord:
    return GraphRecord(nodes=[
        NodeRecord(id="a", node="echo", values={"x": "hi"}),
        NodeRecord(id="b", node="echo", display={"x": 1}),
    ], edges=[("a.result", "b.x")])


def test_a_graph_round_trips_through_json():
    assert GraphRecord.model_validate_json(_graph().model_dump_json()) == _graph()


def test_the_dump_is_the_record():
    """An edge stores as two addresses, the one form a field has anywhere: the
    same string a graph-level ``Input`` is named by, so the stored graph and the
    derived interface cannot spell one field two ways."""
    data = _graph().model_dump(mode="json")

    assert data == {
        "nodes": [{"id": "a", "node": "echo", "values": {"x": "hi"}}, {"id": "b", "node": "echo", "display": {"x": 1}}],
        "edges": [["a.result", "b.x"]],
    }


def test_a_node_without_its_node_is_refused():
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": [{"id": "a", "version": 1}]})


def test_a_node_id_with_a_dot_reads_back_and_is_for_compile_to_refuse():
    assert GraphRecord.model_validate({"nodes": [{"id": "a.b", "node": "echo"}]}).nodes[0].id == "a.b"


def test_an_index_is_its_id_whatever_its_parent():
    """Written out on the model: pydantic's own equality would compare the parent chain."""
    docs = Index("docs")

    assert Index("lines", parent=docs) == Index("lines")
    assert len({Index("lines", parent=docs), Index("lines")}) == 1
    assert Index.model_validate({"id": "lines", "parent": {"id": "docs"}}).parent == docs
    assert Index("lines", parent=docs).model_dump() == {"id": "lines", "parent": {"id": "docs", "parent": None}}


READ_BACK = (
    NodeRecord(id="a", node="echo", version=2, values={"x": {"rows": [1, 2]}}, defaults={"y": 1}, display={"x": 1}),
    _graph(),
    Problem(type="cycle", msg="m", loc=("a", "x"), ctx={"path": ["a", "b"]}),
    Deprecation(header="Gone", alternative="echo"),
    Policy(retries=2, retry_delay=0.5, attempt_timeout=3.0, concurrency=1),
    Choice(id="en", title="English"),
    Index("lines", parent=Index("docs")),
    Textarea(rows=2),
)


@pytest.mark.parametrize("record", READ_BACK, ids=lambda r: type(r).__name__)
def test_a_record_a_host_saves_reads_back_what_it_wrote(record):
    assert type(record).model_validate_json(record.model_dump_json()) == record


def test_a_description_does_not_read_back_as_a_type():
    """An ``Input`` is written from a ``run`` signature: its type dumps as a
    description, and the class the signature holds is not rebuilt from it —
    a description in the type's place is refused, never carried as a dict.
    The widget reads back as written."""
    text = Input("text", Txt, title="Text", widget=Textarea())
    dumped = text.model_dump(mode="json")

    assert dumped["dtype"] == Txt.describe()
    assert "title" not in dumped["widget"]
    assert Textarea.model_validate(dumped["widget"]) == Textarea()
    with pytest.raises(TypeError):
        Input.model_validate(dumped)


SAVED = (
    GraphRecord, NodeRecord, Problem,
    Deprecation, Policy, VersionDescription, NodeDescription,
    Param, Result, Input, Output, Widget, Textarea, Dropdown, Choice, Index,
)
BUILT_PER_CALL = (CompiledGraph, CompiledNode, CompiledField, NodeVersion, Interface, SkippedAtRow)


@pytest.mark.parametrize("record", SAVED, ids=lambda r: r.__name__)
def test_what_a_host_saves_or_sends_is_a_model(record):
    assert issubclass(record, ConductorModel)


@pytest.mark.parametrize("record", BUILT_PER_CALL, ids=lambda r: r.__name__)
def test_what_compile_and_the_engine_build_per_call_stays_a_dataclass(record):
    """Built per call and never parsed, so they stay plain dataclasses."""
    assert dataclasses.is_dataclass(record) and not issubclass(record, ConductorModel)
