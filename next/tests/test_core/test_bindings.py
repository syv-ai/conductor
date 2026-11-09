from typing import Annotated

import pytest
from conductor import (
    DType,
    FieldContent,
    From,
    FromRun,
    Graph,
    GraphNode,
    Interface,
    NodeRegistry,
    Param,
    Ref,
    Result,
    Static,
    node,
)
from conductor._internal.graph.binding import static_values
from conductor.widgets import Textarea
from pydantic import TypeAdapter


def test_an_edge_carries_ordered_refs():
    edge = From("a.result", "b.result")

    assert [r.node_id for r in edge.refs] == ["a", "b"]


def test_edge_order_is_operand_order():
    first = From("a.r", "b.r")
    second = From("b.r", "a.r")
    assert first != second


def test_absence_means_the_declared_default_applies():
    """One state, not two — there is nothing to dismiss."""
    bindings = {"b": Static(3)}

    assert "a" not in bindings


def test_static_values_extracts_only_static_bindings():
    bindings = {
        "text": Static("hi"),
        "number": Static(3),
        "source": From("a.result"),
    }
    assert static_values(bindings) == {"text": "hi", "number": 3}


def test_bindings_are_frozen():
    with pytest.raises(Exception):
        Static(1).value = 2


def test_a_slash_is_the_namespace_separator():
    """Compile puts an embedded graph's nodes under the graph node's name
    — `approve/check` — so `/` is legal in an id and a Ref still reads one way:
    the split is at the first dot, and there is none in the namespace."""
    inner = GraphNode(id="approve/check", type="upper", version=1)

    assert inner.id == "approve/check"
    assert Ref("approve/check", "amount") == "approve/check.amount"
    assert Ref("approve/check.amount").node_id == "approve/check"


def test_a_node_stores_bindings_and_derives_data():
    graph_node = GraphNode(
        id="n1",
        type="translate",
        version=1,
        bindings={
            "text": Static("hi"),
            "source": From("n0.result"),
        },
    )
    assert static_values(graph_node.bindings) == {"text": "hi"}


def test_a_node_is_behaviour_content_and_chrome():
    """The three categories, and nothing from the deleted models."""
    names = list(GraphNode.model_fields)

    assert names == ["id", "type", "version", "bindings", "locked", "title", "description", "fields", "display"]


def test_a_graph_is_nodes_and_display():
    names = list(Graph.model_fields)

    assert names == ["nodes", "display"]


def test_chrome_is_opaque():
    graph_node = GraphNode(id="n", type="t", version=1, display={"x": 10, "y": 20, "anything": [1]})

    assert graph_node.display == {"x": 10, "y": 20, "anything": [1]}


def test_the_record_is_the_schema():
    """A Graph dumps and loads through pydantic, bindings included. A
    Static whose value happens to look like a From comes back a Static,
    because Static nests its payload under `value`."""
    graph = Graph(
        nodes=[
            GraphNode(
                id="a",
                type="t",
                version=1,
                bindings={"x": Static({"refs": [{"node_id": "q", "field": "r"}]})},
                locked=("x",),
                title="A",
                fields={"x": FieldContent(title="X")},
            ),
            GraphNode(id="b", type="t", version=2, bindings={"y": From("a.result")}),
        ],
        display={"zoom": 1},
    )
    adapter = TypeAdapter(Graph)

    assert adapter.validate_python(adapter.dump_python(graph, mode="json")) == graph


class Txt(DType, str):
    id = "bindings-test-txt"
    display_title = "Text"


class Clock:
    """Something the run supplies, not the graph (`FromRun`)."""


@node("text-input", title="Text", description="A value the caller supplies", category="test")
def text_input(value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")
) -> Annotated[Txt, Result(title="Text")]:
    """An ordinary node. Nothing marks it as one the graph exposes."""
    return value


@node("summarise", title="Opsummering", description="An ordinary node — neither end of what the graph exposes", category="test")
def summarise(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")
) -> Annotated[Txt, Result(title="Result")]:
    return text


@node("stamped", title="Stemplet", description="d", category="test")
def stamped(
    clock: FromRun[Clock],
    text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
) -> Annotated[Txt, Result(title="Result")]:
    """A node with a need: the run must provide a `Clock`."""
    return text


def _registry():
    reg = NodeRegistry([text_input, summarise, stamped])
    return reg


def _resolved(graph, registry=None):
    """What the compiler hands `derive_interface`: each node's
    computed interface, and the version its pin resolved to. Here it
    is the declaration read through the pin — none of these nodes shapes
    itself."""
    registry = registry or _registry()
    interfaces = {}
    versions = {}
    for graph_node in graph.nodes:
        version = registry[graph_node.type].versions[graph_node.version]
        versions[graph_node.id] = version
        interfaces[graph_node.id] = version.interface
    return interfaces, versions


def _graph(application_locked=(), language_bindings=None):
    """Two value holders and one summariser connected from the first:
    `application` and `language` are input nodes; `language` and `summary`
    are output nodes (nothing consumes them); `application` is consumed."""
    return Graph(
        nodes=[
            GraphNode(
                id="application", type="text-input", version=1, title="Application",
                fields={"value": FieldContent(title="Application")},
                bindings={"value": Static("")},
                locked=application_locked,
            ),
            GraphNode(
                id="language", type="text-input", version=1, title="Language",
                fields={"value": FieldContent(title="Language"), "result": FieldContent(title="Text")},
                bindings=language_bindings or {},
            ),
            GraphNode(
                id="summary", type="summarise", version=1, title="Opsummering",
                fields={"text": FieldContent(title="Text"), "result": FieldContent(title="Result")},
                bindings={"text": From("application.result")},
            ),
        ],
    )


def test_a_nodes_computed_interface_is_the_same_record_its_version_declares():
    """One record at every scale: a version declares an ``Interface``, compile
    computes one per node in the graph, and derives one for the graph."""

    interfaces, _ = _resolved(_graph())
    assert all(isinstance(i, Interface) for i in interfaces.values())


def _echo_registry():
    @node("echo", title="Echo", description="d", category="test")
    def echo(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Result")]:
        return Txt(x.upper())

    registry = NodeRegistry([echo])
    return registry


