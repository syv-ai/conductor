"""The read plan compile stores for a run: who reads each output, and per index who runs on it, what sits on it and which typed-in lists are born under it."""

from dataclasses import dataclass
from typing import Annotated

from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.execution.ledger import Ledger
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.graph.receive import Iterate, Whole
from conductor.metadata import Param, Result
from conductor.node import NodeDefinition
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


class Docs(NodeDefinition):
    id = "docs"
    title = "Documents"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Found:
        return Found(texts=[Txt("a")], names=[Txt("x")])


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(text.upper())


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


def _pair(registry: NodeRegistry) -> NodeRegistry:
    """A stored graph: ``a`` takes the placement's text, ``b`` holds a typed-in list."""
    return registry.extended_with({"pair": embedded_graph_node("pair", (
        GraphNode(id="a", type="upper", version=1),
        GraphNode(id="b", type="upper", version=1, bindings={"text": Static([Txt("x"), Txt("y")])}),
    ), registry, "Pair")})


def _compiled(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _pair(NodeRegistry(nodes=(Docs, Upper, Join))))
    assert compiled.is_runnable, compiled.problems
    return compiled


def test_a_node_with_series_outputs_owns_the_index_they_birth():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.texts")}),
        GraphNode(id="join", type="join", version=1, bindings={"texts": From("up.result")}),
    )
    docs, up = compiled.node("docs"), compiled.node("up")

    assert docs._births == Index("docs")
    assert docs._iterated_by == ("up",)
    assert set(docs._carried_by) == {Ref("docs", "texts"), Ref("docs", "names"), Ref("up", "result")}
    assert compiled.field(Ref("docs", "texts"))._read_by == ((Ref("up", "text"), Iterate(Index("docs"))),)
    assert compiled.field(Ref("up", "result"))._read_by == ((Ref("join", "texts"), Whole()),)
    assert compiled.node("join")._reads == ((Ref("up", "result"), Whole()),)
    assert (up._births, up._iterated_by, up._carried_by) == (None, (), ())


def test_an_input_holding_a_typed_in_list_owns_its_index():
    compiled = _compiled(
        GraphNode(id="u", type="upper", version=1, bindings={"text": Static([Txt("a"), Txt("b")])}),
        GraphNode(id="v", type="upper", version=1, bindings={"text": From("u.result")}),
    )
    typed = compiled.field(Ref("u", "text"))

    assert typed.listed and not compiled.field(Ref("v", "text")).listed
    assert typed._iterated_by == ("u", "v")
    assert typed._carried_by == (Ref("u", "result"), Ref("v", "result"))


def test_a_typed_in_list_inside_an_iterating_embedded_graph_is_born_under_each_row():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="pair", version=1, bindings={"a.text": From("docs.texts")}),
    )

    assert compiled.node("docs")._typed_lists == (Ref("emb/b", "text"),)


def test_downstream_is_every_node_a_value_reaches():
    compiled = _compiled(
        GraphNode(id="a", type="upper", version=1),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("a.result")}),
        GraphNode(id="c", type="upper", version=1, bindings={"text": From("b.result")}),
        GraphNode(id="d", type="upper", version=1),
    )

    ledger = Ledger(compiled)

    assert ledger._downstream(["a"]) == {"b", "c"}
    assert ledger._downstream(["d"]) == frozenset()
    assert ledger._downstream(["b", "d"]) == {"c"}
