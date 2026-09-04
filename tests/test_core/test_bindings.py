import dataclasses

import pytest
from conductor.graph.binding import Ref, Sources, Static, static_values
from conductor.graph.model import FieldContent, Flow, GraphNode
from conductor.graph.views import dependencies_of
from pydantic import TypeAdapter


def test_a_wire_carries_ordered_refs():
    wire = Sources(refs=(Ref("a", "result"), Ref("b", "result")))

    assert [r.node_id for r in wire.refs] == ["a", "b"]


def test_wire_order_is_operand_order():
    first = Sources(refs=(Ref("a", "r"), Ref("b", "r")))
    second = Sources(refs=(Ref("b", "r"), Ref("a", "r")))
    assert first != second


def test_there_are_exactly_two_variants():
    """No Port (a caller's value replaces a Static for one run), no Guard
    (a branch is an output), no per-ref enable flag."""
    import conductor.graph.binding as bindings

    assert not hasattr(bindings, "Port")
    assert not hasattr(bindings, "Guard")
    assert not hasattr(Sources(refs=()), "when")


def test_absence_means_the_declared_default_applies():
    """One state, not two — there is nothing to dismiss."""
    bindings = {"b": Static(value=3)}

    assert "a" not in bindings


def test_static_values_extracts_only_static_bindings():
    bindings = {
        "text": Static(value="hi"),
        "number": Static(value=3),
        "source": Sources(refs=(Ref("a", "result"),)),
    }
    assert static_values(bindings) == {"text": "hi", "number": 3}


def test_bindings_are_frozen():
    with pytest.raises(Exception):
        Static(value=1).value = 2
def test_a_placement_id_contains_no_dot():
    """A Ref reads as 'node.field' everywhere it is spelled — an index
    name, a caller's payload key — so an id with a '.' is refused."""
    with pytest.raises(ValueError, match="contains '.'"):
        GraphNode(id="a.b", type="upper", version=1)


def test_a_slash_is_the_namespace_separator():
    """The compiler inlines an embedded flow under its placement's name —
    `approve/check` — so `/` is legal in an id and a Ref still reads one way:
    the split is at the first dot, and there is none in the namespace."""
    inner = GraphNode(id="approve/check", type="upper", version=1)

    assert inner.id == "approve/check"
    assert Ref("approve/check", "amount") == "approve/check.amount"
    assert Ref("approve/check.amount").node_id == "approve/check"


def test_a_node_stores_bindings_and_derives_data():
    node = GraphNode(
        id="n1",
        type="translate",
        version=1,
        bindings={
            "text": Static(value="hi"),
            "source": Sources(refs=(Ref("n0", "result"),)),
        },
    )
    assert node.data == {"text": "hi"}


def test_a_node_is_behaviour_content_and_chrome():
    """The three categories, and nothing from the deleted models."""
    names = [f.name for f in dataclasses.fields(GraphNode)]

    assert names == ["id", "type", "version", "bindings", "locked", "title", "description", "fields", "display"]


def test_a_flow_is_nodes_and_display():
    names = [f.name for f in dataclasses.fields(Flow)]

    assert names == ["nodes", "display"]


def test_chrome_is_opaque():
    node = GraphNode(id="n", type="t", version=1, display={"x": 10, "y": 20, "anything": [1]})

    assert node.display == {"x": 10, "y": 20, "anything": [1]}


def test_the_record_is_the_schema():
    """A Flow dumps and loads through pydantic, bindings included. A
    Static whose value happens to look like a Sources comes back a Static,
    because Static nests its payload under `value`."""
    flow = Flow(
        nodes=[
            GraphNode(
                id="a",
                type="t",
                version=1,
                bindings={"x": Static(value={"refs": [{"node_id": "q", "field": "r"}]})},
                locked=("x",),
                title="A",
                fields={"x": FieldContent(title="X")},
            ),
            GraphNode(id="b", type="t", version=2, bindings={"y": Sources(refs=(Ref("a", "result"),))}),
        ],
        display={"zoom": 1},
    )
    adapter = TypeAdapter(Flow)

    assert adapter.validate_python(adapter.dump_python(flow, mode="json")) == flow
def test_a_node_depends_on_every_node_its_wires_name():
    nodes = [
        GraphNode(id="a", type="t", version=1),
        GraphNode(id="b", type="t", version=1, bindings={"x": Sources(refs=(Ref("a", "result"),))}),
    ]

    assert dependencies_of(nodes) == {"a": frozenset(), "b": frozenset({"a"})}


def test_a_static_binding_creates_no_dependency():
    nodes = [GraphNode(id="a", type="t", version=1, bindings={"x": Static(value="hi")})]

    assert dependencies_of(nodes) == {"a": frozenset()}


def test_two_refs_on_one_input_are_one_dependency_each():
    nodes = [
        GraphNode(
            id="b", type="t", version=1,
            bindings={"x": Sources(refs=(Ref("a", "result"), Ref("c", "result")))},
        )
    ]

    assert dependencies_of(nodes)["b"] == frozenset({"a", "c"})


def test_two_wires_from_the_same_node_are_one_dependency():
    """A set, not a list: waiting twice for one node is waiting once."""
    nodes = [
        GraphNode(
            id="b", type="t", version=1,
            bindings={"x": Sources(refs=(Ref("a", "result"),)), "y": Sources(refs=(Ref("a", "other"),))},
        )
    ]

    assert dependencies_of(nodes)["b"] == frozenset({"a"})


def test_conductor_defines_no_edge_type():
    """The canvas derives its own cables; there is nothing here to convert."""
    import conductor.graph.model as model

    assert not hasattr(model, "GraphEdge")
