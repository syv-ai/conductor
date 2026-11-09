"""The read plan compile stores for a run: who reads each output, and per index who runs on it, what sits on it and which lists set on nodes are made under it."""

from dataclasses import dataclass
from typing import Annotated

from conductor import NodeRegistry, node
from conductor.dtype import DType
from conductor.field_records import Param, Result
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.graph_records import Graph, GraphNode
from conductor.graph.receive import EntireValue, PerRow
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "plan-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Found:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    names: Annotated[Series[Txt], Result(title="Names")]


@node("docs", title="Documents", description="d", category="test")
def docs(folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Found:
    return Found(texts=[Txt("a")], names=[Txt("x")])


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("+".join(texts))


def _pair(registry: NodeRegistry) -> NodeRegistry:
    """A stored graph: ``a`` takes the graph node's text, ``b`` holds a list set on it."""
    return registry.extended_with([embedded_graph_node("pair", (
        GraphNode(id="a", type="upper", version=1),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static([Txt("x"), Txt("y")])}),
    ), registry, "Pair")])


def _compiled(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _pair(NodeRegistry((docs, upper_node, join_node))))
    assert compiled.is_runnable, compiled.problems
    return compiled


def test_a_node_with_series_outputs_owns_the_index_they_make_rows_of():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="join", type="join", version=1, bindings={"texts": From("up.result")}),
    )
    texts, names, up = compiled.field(Ref("docs", "texts")), compiled.field(Ref("docs", "names")), compiled.field(Ref("up", "result"))

    assert texts._makes_rows == Index("docs.texts") and names._makes_rows == Index("docs.names")
    plan = compiled._read_plan
    assert plan.run_per_row_by["docs.texts"] == ("up",) and plan.run_per_row_by["docs.names"] == ()
    assert plan.carried_by["docs.texts"] == (Ref("docs", "texts"), Ref("up", "result"))
    assert plan.carried_by["docs.names"] == (Ref("docs", "names"),)
    assert plan.read_by[texts.ref] == ((Ref("up", "text"), PerRow(Index("docs.texts"))),)
    assert plan.read_by[up.ref] == ((Ref("join", "texts"), EntireValue()),)
    assert plan.reads["join"] == ((Ref("up", "result"), EntireValue()),)
    assert up._makes_rows is None
    assert plan.owners["docs.texts"] is texts


def test_an_input_holding_a_typed_in_list_owns_its_index():
    compiled = _compiled(
        GraphNode(id="u", type="upper", version=1, bindings={"text": Static([Txt("a"), Txt("b")])}),
        GraphNode(id="v", type="upper", version=1, bindings={"text": From("u.result")}),
    )
    typed = compiled.field(Ref("u", "text"))

    assert typed._listed and not compiled.field(Ref("v", "text"))._listed
    plan = compiled._read_plan
    assert plan.run_per_row_by["u.text"] == ("u", "v")
    assert plan.carried_by["u.text"] == (Ref("u", "result"), Ref("v", "result"))


def test_a_list_set_on_a_node_inside_an_embedded_graph_running_per_row_is_made_under_each_row():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="pair", version=1, bindings={"a.text": From("docs.texts")}),
    )

    assert compiled._read_plan.listed_inputs["docs.texts"] == (Ref("emb/b", "text"),)


def test_downstream_is_every_node_a_value_reaches():
    compiled = _compiled(
        GraphNode(id="a", type="upper", version=1),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("a.result")}),
        GraphNode(id="c", type="upper", version=1, bindings={"text": From("b.result")}),
        GraphNode(id="d", type="upper", version=1),
    )

    assert compiled._readers_of(["a"]) == {"b", "c"}
    assert compiled._readers_of(["d"]) == set()
    assert compiled._readers_of(["b", "d"]) == {"c"}
