"""A graph's inputs are filled by name, and its outputs read by address.

``compiled.with_inputs(...)`` is a copy of the graph with those inputs
filled, compiled again; ``run`` takes the copy like any other, and
``compiled.outputs(ending)`` reads what the graph returns, keyed
by address. The two are the graph's interface, used from each side.
"""

from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeRegistry,
    Param,
    Result,
    Static,
    node,
)
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "inputs-test-text"
    display_title = "Text"


Text = Annotated[Txt, Param(title="Text", widget=Textarea())]
Out = Annotated[Txt, Result(title="Result")]


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Text = Txt("")) -> Out:
    return Txt(text.upper())


@node("needs", title="Needs", description="d", category="test")
def needs(text: Annotated[Txt, Param(title="Text", widget=Textarea())]) -> Out:
    """An input with no default: the graph cannot run until it is filled."""
    return text


@node("fails", title="Fails", description="d", category="test")
def fails(text: Text = Txt("")) -> Out:
    raise ValueError("no")


@node("skips", title="Skips", description="d", category="test")
def skips(text: Text = Txt("")) -> Out:
    return SKIPPED


def _compiled(*nodes: GraphNode) -> CompiledGraph:
    registry = NodeRegistry()
    for node_cls in (upper_node, needs, fails, skips):
        registry = registry.extended_with([node_cls])
    registry = registry.extended_with([embedded_graph_node("shout", (GraphNode(id="up", type="upper", version=1),), registry, "Shout")])
    return CompiledGraph.from_graph(Graph(nodes=nodes), registry)


def _outputs(compiled: CompiledGraph) -> dict:
    ending = compiled.run_sync()
    assert ending.status == "complete", ending
    return compiled.outputs(ending)


def test_a_bare_name_fills_the_one_input_that_has_it():
    compiled = _compiled(GraphNode(id="a", type="upper", version=1))

    ready = compiled.with_inputs(text="hi")

    assert _outputs(ready) == {"a.result": "HI"}


def test_the_copy_carries_the_authored_graph_with_the_value_on_it():
    """A host stores what ran; the graph it was compiled from is on the copy, the original untouched."""
    compiled = _compiled(GraphNode(id="a", type="upper", version=1))

    ready = compiled.with_inputs(text="hi")

    assert ready.graph.nodes[0].bindings == {"text": Static("hi")}
    assert compiled.graph.nodes[0].bindings == {}


def test_a_filled_input_stays_offered_and_filling_it_again_overrides_it():
    """A static is a value a caller may answer over, so the copy still offers the input."""
    compiled = _compiled(GraphNode(id="a", type="upper", version=1))

    once = compiled.with_inputs(text="hi")

    assert [str(i.name) for i in once.interface.inputs] == ["a.text"]
    assert _outputs(once.with_inputs(text="yo")) == {"a.result": "YO"}
    assert _outputs(compiled) == {"a.result": ""}


def test_a_bare_name_two_inputs_share_is_refused_naming_both_and_the_address_fills_one():
    compiled = _compiled(GraphNode(id="a", type="upper", version=1), GraphNode(id="b", type="upper", version=1))

    with pytest.raises(InvalidCallError, match=r"a\.text.*b\.text"):
        compiled.with_inputs(text="hi")
    assert _outputs(compiled.with_inputs(**{"a.text": "x", "b.text": "y"})) == {"a.result": "X", "b.result": "Y"}


def test_an_unknown_name_is_refused_listing_the_names_offered():
    compiled = _compiled(GraphNode(id="a", type="upper", version=1))

    with pytest.raises(InvalidCallError, match=r"'txt'.*a\.text"):
        compiled.with_inputs(txt="hi")
    with pytest.raises(InvalidCallError, match=r"'b\.text'.*a\.text"):
        compiled.with_inputs(**{"b.text": "hi"})


def test_a_locked_input_is_refused_by_name():
    compiled = _compiled(GraphNode(id="a", type="upper", version=1, locked=("text",)))

    with pytest.raises(InvalidCallError, match="takes no inputs"):
        compiled.with_inputs(text="hi")


def test_a_list_on_an_input_for_one_value_runs_the_graph_once_per_item():
    compiled = _compiled(GraphNode(id="a", type="upper", version=1))

    out = _outputs(compiled.with_inputs(text=["x", "y"]))

    assert out["a.result"]["values"] == ["X", "Y"]


def test_a_required_input_left_unfilled_is_compiles_own_problem():
    compiled = _compiled(GraphNode(id="n", type="needs", version=1))

    assert "unbound_required" in [p.type for p in compiled.problems]
    assert compiled.with_inputs(text="x").is_runnable


def test_an_input_inside_an_embedded_graph_is_filled_by_its_address():
    compiled = _compiled(GraphNode(id="emb", type="shout", version=1))

    assert _outputs(compiled.with_inputs(**{"emb.text": "hi"})) == {"emb.result": "HI"}


def test_an_input_inside_an_embedded_graph_is_filled_by_its_bare_name_when_no_other_has_it():
    compiled = _compiled(GraphNode(id="emb", type="shout", version=1))

    assert _outputs(compiled.with_inputs(text="hi")) == {"emb.result": "HI"}


def test_outputs_leave_out_a_skipped_output_and_read_what_a_failed_leg_produced():
    skipped = _compiled(GraphNode(id="s", type="skips", version=1))
    assert _outputs(skipped) == {}

    failed = _compiled(
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static("ok")}),
        GraphNode(id="f", type="fails", version=1),
    )
    ending = failed.run_sync()
    assert ending.status == "failed"
    assert failed.results(ending)["a"]["result"] == "OK"
    with pytest.raises(InvalidCallError, match="run_not_complete"):
        failed.outputs(ending)


def test_two_fillings_are_two_graphs_with_two_fingerprints():
    compiled = _compiled(GraphNode(id="a", type="upper", version=1))

    one, other = compiled.with_inputs(text="x"), compiled.with_inputs(text="y")

    assert one.node("a").fingerprint != other.node("a").fingerprint
    assert compiled.node("a").fingerprint not in (one.node("a").fingerprint, other.node("a").fingerprint)


def test_an_edge_input_is_not_offered():
    compiled = _compiled(GraphNode(id="a", type="upper", version=1), GraphNode(id="b", type="upper", version=1, bindings={"text": From("a.result")}))

    assert _outputs(compiled.with_inputs(text="hi")) == {"b.result": "HI"}
