"""``conductor_providers.mermaid.flowchart`` draws a compiled graph as a Mermaid flowchart.

One box per node the author placed, titled by its id and its type; an
embedded graph is a subgraph holding its inner nodes under their expanded
ids. One arrow per ref on every connected input, labelled with how the
reading input receives it. A node with a fatal problem carries the
``fault`` class; the problems themselves are not drawn.
"""

from collections.abc import Mapping
from textwrap import dedent
from typing import Annotated, ClassVar

from conductor import GraphNode, NodeRegistry, Param
from conductor.dtype import DType
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph
from conductor.interface import Interface
from conductor.metadata import Input, Output, Result
from conductor.node import GraphVersion, NodeDefinition
from conductor.series import Series
from conductor.widgets import Textarea
from conductor_providers.mermaid import flowchart


class Txt(DType, str):
    id = "render-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


class Docs(NodeDefinition):
    id = "docs"
    title = "Documents"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
        return [Txt("a"), Txt("b")]


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


class Shout(NodeDefinition):
    """A stored graph placed as a node: upper-case each text, then join them."""

    id = "shout"
    title = "Shout"
    description = "d"
    category = "test"
    versions: ClassVar[dict[int, GraphVersion]] = {
        1: GraphVersion(
            graph=(
                GraphNode(id="up", type="upper", version=1),
                GraphNode(id="all", type="join", version=1, bindings={"texts": From("up.result")}),
            ),
            interface=Interface(
                inputs=(Input(name="up.text", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),),
                outputs=(Output(name="all.result", dtype=Txt, title="Result"),),
                returns=Mapping,
            ),
        )
    }


def _compiled(*nodes: GraphNode) -> CompiledGraph:
    registry = NodeRegistry()
    for node_cls in (Docs, Upper, Join, Shout):
        registry.register(node_cls)
    return CompiledGraph.from_graph(Graph(nodes=nodes), registry)


def test_every_arrow_says_how_its_input_receives_it():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="all", type="join", version=1, bindings={"texts": From("up.result")}),
        GraphNode(id="note", type="upper", version=1, bindings={"text": From("all.result")}),
        GraphNode(id="both", type="join", version=1, bindings={"texts": From("note.result", "all.result")}),
    )

    assert flowchart(compiled) == dedent("""\
        flowchart LR
            n0["docs · docs"]
            n1["up · upper"]
            n2["all · join"]
            n3["note · upper"]
            n4["both · join"]
            n0 -->|per row of docs| n1
            n1 -->|whole| n2
            n2 --> n3
            n3 -->|gathered| n4
            n2 -->|gathered| n4
        """)


def test_an_embedded_graph_is_a_subgraph_of_its_inner_nodes():
    compiled = _compiled(
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="shout", version=1, bindings={"up.text": From("docs.result")}),
    )

    assert flowchart(compiled) == dedent("""\
        flowchart LR
            n0["docs · docs"]
            subgraph n1 ["emb · shout"]
                n2["emb/up · upper"]
                n3["emb/all · join"]
            end
            n0 -->|per row of docs| n2
            n2 -->|grouped by docs| n3
        """)


def test_a_node_with_a_fatal_problem_carries_the_fault_class_and_its_broken_edge_is_not_drawn():
    compiled = _compiled(
        GraphNode(id="up", type="upper", version=1, bindings={"text": From("nowhere.result")}),
        GraphNode(id="odd", type="nonesuch", version=1),
    )

    assert flowchart(compiled) == dedent("""\
        flowchart LR
            n0["up · upper"]:::fault
            n1["odd · nonesuch"]:::fault
            classDef fault stroke:#c62828,stroke-width:2px
        """)


def test_a_graph_with_a_broken_edge_and_something_downstream_draws():
    """An arrow into a node whose wiring failed has no label: how it receives is not known."""
    compiled = _compiled(
        GraphNode(id="a", type="upper", version=1, bindings={"text": Static(Txt("hi"))}),
        GraphNode(id="b", type="upper", version=1, bindings={"text": From("ghost.result")}),
        GraphNode(id="c", type="upper", version=1, bindings={"text": From("b.result")}),
    )

    assert flowchart(compiled) == dedent("""\
        flowchart LR
            n0["a · upper"]
            n1["b · upper"]:::fault
            n2["c · upper"]
            n1 --> n2
            classDef fault stroke:#c62828,stroke-width:2px
        """)


class Lost(NodeDefinition):
    """A stored graph that names a type the catalog has lost."""

    id = "lost"
    title = "Lost"
    description = "d"
    category = "test"
    versions: ClassVar[dict[int, GraphVersion]] = {
        1: GraphVersion(
            graph=(GraphNode(id="gone", type="nonesuch", version=1),),
            interface=Interface(inputs=(), outputs=(), returns=Mapping),
        )
    }


def test_a_placement_whose_inner_node_is_lost_still_draws_its_subgraph():
    registry = NodeRegistry()
    registry.register(Lost)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="emb", type="lost", version=1)]), registry)

    assert flowchart(compiled) == dedent("""\
        flowchart LR
            subgraph n0 ["emb · lost"]
                n1["emb/gone · nonesuch"]
            end
            class n0 fault
            classDef fault stroke:#c62828,stroke-width:2px
        """)
