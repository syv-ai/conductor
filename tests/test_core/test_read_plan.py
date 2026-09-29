"""The read plan compile stores for a run: who reads each output, and per index who runs on it, what sits on it and which typed-in lists are born under it."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, ClassVar

from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.graph.receive import Iterate, Whole
from conductor.interface import Interface
from conductor.metadata import Input, Output, Param, Result
from conductor.node import GraphVersion, NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea


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


class Pair(NodeDefinition):
    """A stored graph: ``a`` takes the placement's text, ``b`` holds a typed-in list."""

    id = "pair"
    title = "Pair"
    description = "d"
    category = "test"
    versions: ClassVar[dict[int, GraphVersion]] = {
        1: GraphVersion(
            graph=(
                GraphNode(id="a", type="upper", version=1),
                GraphNode(id="b", type="upper", version=1, bindings={"text": Static([Txt("x"), Txt("y")])}),
            ),
            interface=Interface(
                inputs=(Input(name="a.text", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),),
                outputs=(Output(name="a.result", dtype=Txt, title="A"), Output(name="b.result", dtype=Txt, title="B")),
                returns=Mapping,
            ),
        )
    }


def _compiled(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), NodeRegistry(nodes=(Docs, Upper, Join, Pair)))
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

    assert compiled._downstream(["a"]) == {"b", "c"}
    assert compiled._downstream(["d"]) == frozenset()
    assert compiled._downstream(["b", "d"]) == {"c"}
