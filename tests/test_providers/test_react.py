"""Tests for conductor_providers.react — ReactFlow JSON round-trips."""

from __future__ import annotations

import json
from typing import Annotated

import conductor_nodes
import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.execution.engine import execute_sync
from conductor.widgets import Output
from conductor_providers import react

# ---------------------------------------------------------------------------
# Fixture: a non-trivial graph with every surface we want to preserve
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> NodeRegistry:
    reg = NodeRegistry()
    conductor_nodes.register_all(reg)

    @reg.node("build-pair", version=1, name="Build Pair",
              description="Emits two strings")
    def build_pair() -> Annotated[str, Output(label="Value")]:
        return "val"

    return reg


@pytest.fixture
def sample_graph():
    nodes = [
        GraphNode("n1", "build-pair@1", {"seed": "x"},
                  produces={"result": "shared-value"}),
        GraphNode("n2", "text-uppercase@1", {"text": "hi"}),
        GraphNode("n3", "text-concat@1", {"separator": "+"},
                  consumes={"a": ("n1", "result")}),
    ]
    edges = [
        GraphEdge("e1", "n2", "n3", "result", "b"),
    ]
    return nodes, edges


# ---------------------------------------------------------------------------
# palette_from_registry
# ---------------------------------------------------------------------------


class TestPalette:
    def test_palette_has_entry_per_registered_node(self, registry):
        palette = react.palette_from_registry(registry)
        ids = {e["id"] for e in palette}
        assert "text-uppercase@1" in ids
        assert "math-add@1" in ids
        assert "build-pair@1" in ids

    def test_palette_entry_has_expected_shape(self, registry):
        palette = react.palette_from_registry(registry)
        entry = next(e for e in palette if e["id"] == "text-uppercase@1")
        assert entry["name"] == "Uppercase"
        assert isinstance(entry["inputs"], list)
        assert isinstance(entry["outputs"], list)
        assert "deprecated" in entry


# ---------------------------------------------------------------------------
# graph_to_react
# ---------------------------------------------------------------------------


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

    def test_produces_preserved(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        n1 = next(n for n in out["nodes"] if n["id"] == "n1")
        assert n1["data"]["produces"] == {"result": "shared-value"}

    def test_consumes_serialized_as_lists_for_json(self, sample_graph):
        """Tuples can't survive JSON; the wire format uses lists."""
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        n3 = next(n for n in out["nodes"] if n["id"] == "n3")
        assert n3["data"]["consumes"] == {"a": ["n1", "result"]}

    def test_static_data_passed_through(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        n2 = next(n for n in out["nodes"] if n["id"] == "n2")
        assert n2["data"]["data"] == {"text": "hi"}

    def test_optional_fields_omitted_when_empty(self, registry):
        nodes = [GraphNode("n", "text-uppercase@1", {"text": "x"})]
        out = react.graph_to_react(nodes, [])
        n = out["nodes"][0]
        assert "produces" not in n["data"]
        assert "consumes" not in n["data"]

    def test_edges_use_camel_case_handles(self, sample_graph):
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        e = out["edges"][0]
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
        # n1 and n2 are roots (x=0). n3 consumes from n1 and has edge from
        # n2, so its depth is 1 (x>0).
        assert positions["n3"]["x"] > positions["n1"]["x"]
        assert positions["n3"]["x"] > positions["n2"]["x"]

    def test_whole_output_is_json_serializable(self, sample_graph):
        """No tuples, no sets, nothing exotic — must survive json.dumps."""
        nodes, edges = sample_graph
        out = react.graph_to_react(nodes, edges)
        # Should not raise
        text = json.dumps(out)
        assert "\"consumes\"" in text


# ---------------------------------------------------------------------------
# react_to_graph
# ---------------------------------------------------------------------------


class TestReactToGraph:
    def test_roundtrip_preserves_everything_non_position(self, sample_graph):
        nodes, edges = sample_graph
        wire = react.graph_to_react(nodes, edges)
        nodes2, edges2 = react.react_to_graph(wire)

        # Compare as dicts keyed by id for stable diffs
        by_id = {n.id: n for n in nodes}
        by_id2 = {n.id: n for n in nodes2}
        assert set(by_id) == set(by_id2)
        for nid in by_id:
            a, b = by_id[nid], by_id2[nid]
            assert a.type == b.type
            assert a.data == b.data
            assert a.produces == b.produces
            assert a.consumes == b.consumes   # tuples restored

        assert len(edges) == len(edges2)
        for e1, e2 in zip(edges, edges2, strict=True):
            assert e1.id == e2.id
            assert e1.source == e2.source
            assert e1.target == e2.target
            assert e1.source_handle == e2.source_handle
            assert e1.target_handle == e2.target_handle

    def test_consumes_list_form_parsed_back_as_tuple(self):
        """A frontend sending raw JSON will have lists, not tuples."""
        wire = {
            "nodes": [
                {"id": "c", "type": "text-concat@1", "position": {"x": 0, "y": 0},
                 "data": {"consumes": {"a": ["p", "result"]}}},
            ],
            "edges": [],
        }
        nodes, _ = react.react_to_graph(wire)
        assert nodes[0].consumes == {"a": ("p", "result")}
        assert isinstance(nodes[0].consumes["a"], tuple)

    def test_handles_missing_data_payload(self):
        wire = {
            "nodes": [
                {"id": "x", "type": "text-uppercase@1", "position": {"x": 0, "y": 0}},
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
                    "id": "x", "type": "text-uppercase@1",
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


# ---------------------------------------------------------------------------
# End-to-end: wire → conductor → execute
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_wire_format_can_be_compiled_and_executed(self, registry):
        # Build a graph, round-trip through the wire format, then execute.
        nodes_in = [
            GraphNode("src", "text-uppercase@1", {"text": "hello"}),
            GraphNode("down", "text-reverse@1", None),
        ]
        edges_in = [GraphEdge("e1", "src", "down", "result", "text")]

        wire = react.graph_to_react(nodes_in, edges_in)
        as_json = json.dumps(wire)                  # simulate network
        back = json.loads(as_json)
        nodes_out, edges_out = react.react_to_graph(back)

        compiled = compile(nodes=nodes_out, edges=edges_out, registry=registry)
        results = execute_sync(compiled)
        assert results["down"]["result"] == "OLLEH"

    def test_shared_references_preserved_through_wire(self, registry):
        nodes_in = [
            GraphNode("mapper", "build-pair@1", None,
                      produces={"result": "pair"}),
            GraphNode("cons", "text-concat@1", {"b": "!"},
                      consumes={"a": ("mapper", "result")}),
        ]
        wire = react.graph_to_react(nodes_in, [])
        back_nodes, back_edges = react.react_to_graph(json.loads(json.dumps(wire)))

        compiled = compile(nodes=back_nodes, edges=back_edges, registry=registry)
        results = execute_sync(compiled)
        assert results["cons"]["result"] == "val!"
