"""``conductor_providers.react`` — a conductor graph to ReactFlow JSON and back.

``graph_to_react`` emits the ``{nodes, edges}`` dict a ReactFlow canvas
renders (camelCase handles, a position per node, the placement's data
under ``data.data``); ``react_to_graph`` reads it back into ``GraphNode``
and ``GraphEdge``. The round trip must survive ``json.dumps`` and compile.
"""

from __future__ import annotations

import json
from typing import Annotated

import conductor_nodes
import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.execution.engine import execute_sync
from conductor.node import NodeDefinition
from conductor.returns import Result
from conductor_nodes.types import Text
from conductor_providers import react


class BuildPair(NodeDefinition):
    id = "build-pair"
    title = "Build Pair"
    description = "Emits a value"
    category = "test"

    def run(self) -> Annotated[Text, Result(title="Value")]:
        return Text("val")


@pytest.fixture
def registry() -> NodeRegistry:
    reg = NodeRegistry()
    conductor_nodes.register_all(reg)
    reg.register(BuildPair)
    return reg


@pytest.fixture
def sample_graph():
    nodes = [
        GraphNode("n1", "build-pair", 1, {"seed": "x"}),
        GraphNode("n2", "text-uppercase", 1, {"text": "hi"}),
        GraphNode("n3", "text-concat", 1, {"separator": "+"}),
    ]
    edges = [
        GraphEdge("e0", "n1", "n3", "result", "a"),
        GraphEdge("e1", "n2", "n3", "result", "b"),
    ]
    return nodes, edges


class TestGraphToReact:
    def test_emits_nodes_and_edges_keys(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        assert set(out.keys()) == {"nodes", "edges"}

    def test_node_structure_has_id_type_position_data(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        for n in out["nodes"]:
            assert set(n.keys()) >= {"id", "type", "position", "data"}
            assert set(n["position"].keys()) == {"x", "y"}
            assert isinstance(n["position"]["x"], int)
            assert isinstance(n["position"]["y"], int)

    def test_static_data_passed_through(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        n2 = next(n for n in out["nodes"] if n["id"] == "n2")
        assert n2["data"]["data"] == {"text": "hi"}

    def test_optional_fields_omitted_when_empty(self):
        nodes = [GraphNode("n", "text-uppercase", 1, {"text": "x"})]
        out = react.graph_to_react(nodes, [])
        n = out["nodes"][0]
        assert "produces" not in n["data"]
        assert "consumes" not in n["data"]

    def test_edges_use_camel_case_handles(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        e = out["edges"][1]
        assert e["sourceHandle"] == "result"
        assert e["targetHandle"] == "b"
        assert e["source"] == "n2"
        assert e["target"] == "n3"

    def test_caller_positions_respected(self, sample_graph):
        nodes, edges = sample_graph
        given = {"n1": {"x": 999, "y": 111}}
        out = react.graph_to_react(nodes, edges, positions=given)
        n1 = next(n for n in out["nodes"] if n["id"] == "n1")
        assert n1["position"] == {"x": 999, "y": 111}
        # Unpositioned nodes still get auto-layout
        n2 = next(n for n in out["nodes"] if n["id"] == "n2")
        assert n2["position"] != {"x": 0, "y": 0} or n2["id"] == "n0"

    def test_auto_layout_when_no_positions(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        positions = {n["id"]: n["position"] for n in out["nodes"]}
        # n1 and n2 are roots (x=0); n3 is fed by both, so it sits one
        # column to the right.
        assert positions["n3"]["x"] > positions["n1"]["x"]
        assert positions["n3"]["x"] > positions["n2"]["x"]

    def test_whole_output_is_json_serializable(self, sample_graph):
        """No tuples, no sets, nothing exotic — must survive json.dumps."""
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        assert json.loads(json.dumps(out)) == out


class TestReactToGraph:
    def test_roundtrip_preserves_everything_non_position(self, sample_graph):
        nodes, edges = sample_graph
        wire = react.graph_to_react(nodes, edges)
        nodes2, edges2 = react.react_to_graph(wire)

        by_id = {n.id: n for n in nodes}
        by_id2 = {n.id: n for n in nodes2}
        assert set(by_id) == set(by_id2)
        for nid in by_id:
            a, b = by_id[nid], by_id2[nid]
            assert a.type == b.type
            assert a.version == b.version
            assert a.data == b.data

        assert len(edges) == len(edges2)
        for e1, e2 in zip(edges, edges2, strict=True):
            assert e1.id == e2.id
            assert e1.source == e2.source
            assert e1.target == e2.target
            assert e1.source_handle == e2.source_handle
            assert e1.target_handle == e2.target_handle

    def test_handles_missing_data_payload(self):
        wire = {
            "nodes": [
                {"id": "x", "type": "text-uppercase", "position": {"x": 0, "y": 0}},
            ],
            "edges": [],
        }
        nodes, _ = react.react_to_graph(wire)
        assert nodes[0].data is None
        assert nodes[0].produces is None
        assert nodes[0].consumes is None

    def test_unknown_keys_are_ignored(self):
        """Hosts that attach extra metadata to the wire must not break us."""
        wire = {
            "nodes": [
                {
                    "id": "x", "type": "text-uppercase",
                    "position": {"x": 0, "y": 0},
                    "data": {"data": {"text": "hi"}, "__host_flag__": True},
                    "__draft__": True,
                },
            ],
            "edges": [],
            "viewport": {"x": 0, "y": 0, "zoom": 1},
        }
        nodes, edges = react.react_to_graph(wire)
        assert nodes[0].data == {"text": "hi"}
        assert edges == []


class TestEndToEnd:
    def test_wire_format_can_be_compiled_and_executed(self, registry):
        nodes_in = [
            GraphNode("src", "text-uppercase", 1, {"text": "hello"}),
            GraphNode("down", "text-reverse", 1, None),
        ]
        edges_in = [GraphEdge("e1", "src", "down", "result", "text")]

        wire = react.graph_to_react(nodes_in, edges_in)
        as_json = json.dumps(wire)                  # simulate network
        back = json.loads(as_json)
        nodes_out, edges_out = react.react_to_graph(back)

        compiled = compile(nodes=nodes_out, edges=edges_out, registry=registry)
        results = execute_sync(compiled)
        assert results["down"]["result"] == "OLLEH"


class TestProcessStandardFields:
    """``when`` / ``priority`` / ``compensation`` / ``on_error`` round-trip."""

    def test_edge_when_and_priority_preserved(self):
        edge = GraphEdge(
            "e1", "a", "b", "result", "text",
            when="amount > 100", priority=5,
        )
        nodes = [GraphNode("a", "build-pair", 1, None),
                 GraphNode("b", "text-uppercase", 1, {})]
        wire = react.graph_to_react(nodes, [edge])
        back_nodes, back_edges = react.react_to_graph(
            json.loads(json.dumps(wire))
        )
        assert back_edges[0].when == "amount > 100"
        assert back_edges[0].priority == 5

    def test_compensation_and_on_error_preserved(self):
        nodes = [
            GraphNode("n1", "text-uppercase", 1, {"text": "x"},
                      compensation="undo",
                      on_error="compensate"),
            GraphNode("undo", "text-uppercase", 1, {"text": "u"}),
        ]
        wire = react.graph_to_react(nodes, [])
        back_nodes, _ = react.react_to_graph(json.loads(json.dumps(wire)))
        n1 = next(n for n in back_nodes if n.id == "n1")
        assert n1.compensation == "undo"
        assert n1.on_error == "compensate"
