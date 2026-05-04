"""Tests for the Mermaid viz helper."""

from typing import Annotated

import pytest
from conductor.compound.for_each import FOR_EACH
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.viz import to_mermaid
from conductor.widgets import ConnectionList, Output, Text


@pytest.fixture
def simple_compiled(registry):
    @registry.node("upper", version=1, name="Upper", description="Uppercase")
    def upper(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Out")]:
        return text.upper()

    @registry.node("greet", version=1, name="Greet", description="Greet")
    def greet(name: Annotated[str, Text(label="Name")]) -> Annotated[str, Output(label="Out")]:
        return f"Hi {name}"

    nodes = [
        GraphNode("a", "upper@1", {"text": "hello"}),
        GraphNode("b", "greet@1", None),
    ]
    edges = [GraphEdge("e1", "a", "b", "result", "name")]
    return compile(nodes=nodes, edges=edges, registry=registry)


def test_renders_flowchart_header(simple_compiled):
    out = to_mermaid(simple_compiled)
    assert out.startswith("flowchart LR")


def test_renders_each_node(simple_compiled):
    out = to_mermaid(simple_compiled)
    # Both nodes show up with their id and a node-type label.
    assert "<b>a</b>" in out
    assert "<b>b</b>" in out
    assert "upper" in out
    assert "greet" in out


def test_edges_show_target_handle(simple_compiled):
    out = to_mermaid(simple_compiled)
    # source handle is "result" (suppressed), target handle is "name".
    assert "a -->|name| b" in out


def test_direction_is_configurable(simple_compiled):
    assert to_mermaid(simple_compiled, direction="TB").startswith("flowchart TB")


def test_decision_node_uses_diamond_shape(registry):
    @registry.node("decide", version=1, name="Decide", description="Decide", is_decision=True)
    def decide(x: Annotated[str, Text(label="X")]) -> Annotated[str, Output(label="Out")]:
        return x

    @registry.node("a", version=1, name="A", description="A")
    def a(x: Annotated[str, Text(label="X")]) -> Annotated[str, Output(label="O")]:
        return "a"

    @registry.node("b", version=1, name="B", description="B")
    def b(x: Annotated[str, Text(label="X")]) -> Annotated[str, Output(label="O")]:
        return "b"

    nodes = [
        GraphNode("d", "decide@1", {"x": "hi"}),
        GraphNode("a", "a@1", None),
        GraphNode("b", "b@1", None),
    ]
    edges = [
        GraphEdge("e1", "d", "a", "result", "x", when="result == 'hi'", priority=10),
        GraphEdge("e2", "d", "b", "result", "x"),
    ]
    compiled = compile(nodes=nodes, edges=edges, registry=registry)
    out = to_mermaid(compiled)
    assert '{"<b>d</b>' in out  # diamond for decision node
    assert "⟦result == 'hi'⟧" in out
    assert "⟦else⟧" in out


def test_for_each_uses_subroutine_and_managed_shapes(registry):
    @registry.node("upper", version=1, name="Upper", description="Up")
    def upper(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
        return text.upper()

    @registry.node(
        "for-each-start", version=1, name="Start", description="Start",
    )
    def for_each_start(
        items: Annotated[list[str], ConnectionList(label="Items")],
    ) -> tuple[
        Annotated[str, Output(label="Item")],
        Annotated[int, Output(label="Index")],
    ]:
        raise NotImplementedError

    @registry.node("for-each-end", version=1, name="End", description="End")
    def for_each_end(
        item: Annotated[str, Text(label="Item")],
    ) -> Annotated[list[str], Output(label="Out")]:
        raise NotImplementedError

    nodes = [
        GraphNode("start", "for-each-start@1", {"items": ["a"]}),
        GraphNode("body", "upper@1", None),
        GraphNode("end", "for-each-end@1", None),
    ]
    edges = [
        GraphEdge("e1", "start", "body", "output_1", "text"),
        GraphEdge("e2", "body", "end", "result", "item"),
    ]
    compiled = compile(nodes=nodes, edges=edges, registry=registry, compound_types=[FOR_EACH])
    out = to_mermaid(compiled)
    assert "[[" in out and "]]" in out  # subroutine shape for the start (compound)
    assert "[/" in out and "/]" in out  # parallelogram shape for managed body/end
    # Mermaid reserved word — the `end` id must be aliased.
    assert "n_end" in out


def test_shared_references_render_as_dashed_arrows(registry):
    @registry.node("producer", version=1, name="P", description="P")
    def producer() -> Annotated[str, Output(label="V")]:
        return "x"

    @registry.node("consumer", version=1, name="C", description="C")
    def consumer(v: Annotated[str, Text(label="V")]) -> Annotated[str, Output(label="O")]:
        return v

    nodes = [
        GraphNode("p", "producer@1", None, produces={"result": "shared"}),
        GraphNode("c", "consumer@1", None, consumes={"v": ("p", "result")}),
    ]
    compiled = compile(nodes=nodes, edges=[], registry=registry)
    out = to_mermaid(compiled)
    assert "-.->|v|" in out  # dashed arrow with consume target handle
