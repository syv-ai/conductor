"""``conductor_providers.react`` — a ``Flow`` to ReactFlow JSON and back.

``graph_to_react`` emits the ``{nodes, edges}`` dict a ReactFlow canvas
renders: one node per placement carrying the placement record under
``data`` and a position, and one cable per ref, derived from the
bindings. ``react_to_graph`` reads the nodes back into a ``Flow``; the
cables are the canvas's and are not read. The round trip must survive
``json.dumps`` and compile.
"""

from __future__ import annotations

import json
from typing import Annotated

import conductor_nodes
import pytest
from conductor import Flow, GraphNode, NodeRegistry, compile
from conductor.execution.engine import execute_sync
from conductor.graph.binding import Sources, Static
from conductor.node import NodeDefinition
from conductor.ref import Ref
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
def sample_flow() -> Flow:
    return Flow(nodes=[
        GraphNode("n1", "build-pair", 1),
        GraphNode("n2", "text-uppercase", 2, bindings={"text": Static(value="hi")}),
        GraphNode(
            "n3", "text-concat", 1,
            bindings={
                "separator": Static(value="+"),
                "a": Sources(refs=(Ref("n1", "result"),)),
                "b": Sources(refs=(Ref("n2", "result"),)),
            },
        ),
    ])


class TestGraphToReact:
    def test_emits_nodes_and_edges_keys(self, sample_flow):
        assert set(react.graph_to_react(sample_flow).keys()) == {"nodes", "edges"}

    def test_node_structure_has_id_type_position_data(self, sample_flow):
        for n in react.graph_to_react(sample_flow)["nodes"]:
            assert set(n.keys()) == {"id", "type", "position", "data"}
            assert set(n["position"].keys()) == {"x", "y"}
            assert isinstance(n["position"]["x"], int)
            assert isinstance(n["position"]["y"], int)

    def test_data_is_the_placement_record(self, sample_flow):
        """The record whole — ``version`` and the bindings included."""
        n2 = next(n for n in react.graph_to_react(sample_flow)["nodes"] if n["id"] == "n2")
        assert n2["data"]["version"] == 2
        assert n2["data"]["bindings"] == {"text": {"value": "hi"}}

    def test_edges_are_derived_one_per_ref(self, sample_flow):
        edges = react.graph_to_react(sample_flow)["edges"]
        assert [(e["source"], e["sourceHandle"], e["target"], e["targetHandle"]) for e in edges] == [
            ("n1", "result", "n3", "a"),
            ("n2", "result", "n3", "b"),
        ]
        assert len({e["id"] for e in edges}) == 2

    def test_a_position_on_the_placement_is_kept(self, sample_flow):
        flow = Flow(nodes=[
            GraphNode("n1", "build-pair", 1, display={"position": {"x": 999, "y": 111}}),
            *sample_flow.nodes[1:],
        ])
        out = react.graph_to_react(flow)
        assert next(n for n in out["nodes"] if n["id"] == "n1")["position"] == {"x": 999, "y": 111}
        # Unpositioned nodes still get the layout.
        assert next(n for n in out["nodes"] if n["id"] == "n3")["position"] != {"x": 0, "y": 0}

    def test_auto_layout_when_no_positions(self, sample_flow):
        positions = {n["id"]: n["position"] for n in react.graph_to_react(sample_flow)["nodes"]}
        # n1 and n2 are roots (x=0); n3 is fed by both, so it sits one
        # column to the right.
        assert positions["n3"]["x"] > positions["n1"]["x"]
        assert positions["n3"]["x"] > positions["n2"]["x"]

    def test_whole_output_is_json_serializable(self, sample_flow):
        """No tuples, no sets, nothing exotic — must survive json.dumps."""
        out = react.graph_to_react(sample_flow)
        assert json.loads(json.dumps(out)) == out


class TestReactToGraph:
    def test_roundtrip_keeps_every_placement(self, sample_flow):
        back = react.react_to_graph(json.loads(json.dumps(react.graph_to_react(sample_flow))))

        assert [(n.id, n.type, n.version, dict(n.bindings)) for n in back.nodes] == [
            (n.id, n.type, n.version, dict(n.bindings)) for n in sample_flow.nodes
        ]

    def test_the_canvas_position_lands_in_display(self):
        wire = react.graph_to_react(Flow(nodes=[GraphNode("x", "text-uppercase", 1)]))
        wire["nodes"][0]["position"] = {"x": 5, "y": 6}

        assert react.react_to_graph(wire).nodes[0].display == {"position": {"x": 5, "y": 6}}

    def test_a_node_without_data_is_refused(self):
        with pytest.raises(KeyError):
            react.react_to_graph({"nodes": [{"id": "x", "type": "text-uppercase", "position": {"x": 0, "y": 0}}], "edges": []})

    def test_unknown_keys_are_ignored(self):
        """Hosts that attach extra metadata to the wire must not break us."""
        wire = {
            "nodes": [
                {
                    "id": "x", "type": "text-uppercase",
                    "position": {"x": 0, "y": 0},
                    "data": {"id": "x", "type": "text-uppercase", "version": 1, "bindings": {"text": {"value": "hi"}}},
                    "__draft__": True,
                },
            ],
            "edges": [],
            "viewport": {"x": 0, "y": 0, "zoom": 1},
        }
        flow = react.react_to_graph(wire)
        assert flow.nodes[0].data == {"text": "hi"}


class TestEndToEnd:
    def test_wire_format_can_be_compiled_and_executed(self, registry):
        flow_in = Flow(nodes=[
            GraphNode("src", "text-uppercase", 1, bindings={"text": Static(value="hello")}),
            GraphNode("down", "text-reverse", 1, bindings={"text": Sources(refs=(Ref("src", "result"),))}),
        ])

        wire = react.graph_to_react(flow_in)
        back = json.loads(json.dumps(wire))  # simulate network
        flow_out = react.react_to_graph(back)

        results = execute_sync(compile(flow_out, registry))
        assert results["down"]["result"] == "OLLEH"
