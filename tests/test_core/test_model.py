"""What a host saves is a pydantic model, and a graph saves itself."""

import dataclasses
import sys

import pytest
from conductor.errors import ErrorCause
from conductor.execution.ledger import Skip
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledField, CompiledGraph, CompiledNode
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.graph.problem import Problem
from conductor.interface import Interface
from conductor.metadata import Field, Input, Output
from conductor.model import ConductorModel
from conductor.node import (
    Deprecation,
    GraphVersion,
    NodeDescription,
    NodeVersion,
    Policy,
    VersionDescription,
)
from conductor.ref import Ref
from conductor.series import Index
from conductor.widgets import Choice, Dropdown, OperatorChoice, SchemaBuilder, Textarea, Widget
from pydantic import ValidationError


def _graph() -> Graph:
    return Graph(
        nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": Static(value="hi")}, locked=("x",), title="A", fields={"x": FieldContent(title="X")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),))}, display={"x": 1}),
        ],
    )


def test_a_graph_round_trips_through_yaml():
    assert Graph.from_yaml(_graph().to_yaml()) == _graph()


def test_a_graph_round_trips_through_json():
    assert Graph.model_validate_json(_graph().model_dump_json()) == _graph()


@pytest.mark.parametrize("name", ["approval.yaml", "approval.yml", "approval.json"])
def test_a_graph_reads_back_what_it_wrote_to_a_path(tmp_path, name):
    path = tmp_path / name
    _graph().to_path(path)

    assert Graph.from_path(path) == _graph()


def test_the_suffix_decides_the_format(tmp_path):
    _graph().to_path(tmp_path / "approval.json")
    assert (tmp_path / "approval.json").read_text().lstrip().startswith("{")


@pytest.mark.parametrize("name", ["approval.txt", "approval.JSON", "approval.Yaml", "approval.yaml.bak", "approval"])
def test_a_path_whose_suffix_is_not_exactly_json_yaml_or_yml_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match=name):
        _graph().to_path(tmp_path / name)
    with pytest.raises(ValueError, match=name):
        Graph.from_path(tmp_path / name)
    assert not (tmp_path / name).exists()


def test_the_dump_is_the_record():
    """A ref stores as its address, the one form it has anywhere: the same
    string a graph-level ``Input`` is named by, so the stored graph and the
    derived interface cannot spell one edge two ways."""
    data = _graph().model_dump(mode="json")

    assert data["nodes"][1]["bindings"]["x"] == {"refs": ["a.result"]}
    assert data["nodes"][0]["locked"] == ["x"]
    assert "inputs" not in data and "outputs" not in data and "edges" not in data


def test_a_node_without_a_type_is_refused():
    with pytest.raises(ValidationError):
        Graph.model_validate({"nodes": [{"id": "a", "version": 1}]})


def test_a_node_id_with_a_dot_is_refused_however_the_node_is_built():
    with pytest.raises(ValueError, match="contains '.'"):
        GraphNode(id="a.b", type="echo", version=1)
    with pytest.raises(ValueError, match="contains '.'"):
        Graph.from_yaml("nodes:\n- {id: a.b, type: echo, version: 1}\n")


def test_yaml_without_pyyaml_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(ImportError, match=r"syv-conductor\[yaml\]"):
        _graph().to_yaml()


def test_an_index_is_its_id_whatever_its_parent():
    """Written out on the model: pydantic's own equality would compare the parent chain."""
    docs = Index("docs")

    assert Index("lines", parent=docs) == Index("lines")
    assert len({Index("lines", parent=docs), Index("lines")}) == 1
    assert Index.model_validate({"id": "lines", "parent": {"id": "docs"}}).parent == docs
    assert Index("lines", parent=docs).model_dump() == {"id": "lines", "parent": {"id": "docs", "parent": None}}


def test_a_schema_builder_keeps_its_schema_key_on_the_wire():
    """``schema`` is also a pydantic method name, so the attribute is ``schema_`` and the wire and keyword say ``schema``."""
    builder = SchemaBuilder(title="Felter", schema={"type": "object"})

    assert builder.schema_ == {"type": "object"}
    assert builder.model_dump()["schema"] == {"type": "object"}


SAVED = (
    Graph, GraphNode, FieldContent, Edges, Static, Problem, ErrorCause,
    Deprecation, Policy, VersionDescription, NodeDescription,
    Field, Input, Output, Widget, Textarea, Dropdown, Choice, OperatorChoice, Index,
)
BUILT_PER_CALL = (CompiledGraph, CompiledNode, CompiledField, NodeVersion, GraphVersion, Interface, Skip)


@pytest.mark.parametrize("record", SAVED, ids=lambda r: r.__name__)
def test_what_a_host_saves_or_sends_is_a_model(record):
    assert issubclass(record, ConductorModel)


@pytest.mark.parametrize("record", BUILT_PER_CALL, ids=lambda r: r.__name__)
def test_what_compile_and_the_engine_build_per_call_stays_a_dataclass(record):
    """Built per call and never parsed, so they stay plain dataclasses."""
    assert dataclasses.is_dataclass(record) and not issubclass(record, ConductorModel)
