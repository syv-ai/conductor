"""An embedded graph expands under its placement's name, and its boundary is an index scope."""

from collections.abc import Mapping
from typing import Annotated, ClassVar

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.interface import Interface
from conductor.metadata import Input, Output, Param, Result
from conductor.node import GraphVersion, NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "expansion-test-txt"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


class Holder(NodeDefinition):
    id = "holder"
    title = "Text"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return value


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper case"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return Txt(text.upper())


class Join(NodeDefinition):
    """A reduction."""

    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class Docs(NodeDefinition):
    id = "docs"
    title = "Documents"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
        return [Txt("a"), Txt("b")]


class Lines(NodeDefinition):
    id = "lines"
    title = "Lines"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
        return text.splitlines()


def _inner_graph():
    """The embedded graph: a holder, uppercased, then joined. Standalone it
    takes one text at `holder.value` and returns `join.result`."""
    return (
        GraphNode(id="holder", type="holder", version=1, title="Text", bindings={"value": Static(value="inner")}),
        GraphNode(id="up", type="upper", version=1, title="Upper", bindings={"text": Edges(refs=(Ref("holder", "result"),))}),
        GraphNode(id="join", type="join", version=1, title="Join", bindings={"texts": Edges(refs=(Ref("up", "result"),))}),
    )


def _embedded_definition(node_id, graph, inputs, outputs):
    """What a host builds from a graph it stores: a definition whose one
    version is a `GraphVersion` — the graph's interface, and its nodes."""

    class Embedded(NodeDefinition):
        id = node_id
        title = "Embedded"
        description = "d"
        category = "test"
        versions: ClassVar[dict[int, GraphVersion]] = {
            1: GraphVersion(graph=graph, interface=Interface(inputs=inputs, outputs=outputs, returns=Mapping))
        }

    return Embedded


def _registry(*extra):
    registry = NodeRegistry()
    for node_cls in (Holder, Upper, Join, Docs, *extra):
        registry.register(node_cls)
    return registry


def _inner_definition():
    return _embedded_definition(
        "inner-graph",
        _inner_graph(),
        inputs=(Input(name="holder.value", dtype=Txt, title="Text", widget=Textarea(), default=Txt("indre"), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )


def _compiled(nodes, *extra):
    return CompiledGraph.from_graph(Graph(nodes=nodes), _registry(_inner_definition(), *extra))


# --- expansion ---------------------------------------------------------------


def test_the_inner_nodes_are_nodes_of_the_one_run_under_the_placements_name():
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static(value="outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": Edges(refs=(Ref("src", "result"),))}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": Edges(refs=(Ref("emb", "join.result"),))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.execution_order() == ("src", "emb/holder", "emb/up", "emb/join", "after")
    assert compiled.node("emb/up").embedded_in == "emb"
    assert compiled.node("src").embedded_in is None
    with pytest.raises(KeyError):
        compiled.node("emb").graph_node  # the placement is not a node of the run; its inner nodes are


def test_the_placements_bindings_move_onto_the_inner_fields_they_name():
    """An edge into `emb.holder.value` replaces the inner author's static;
    an outer edge from `emb.join.result` reads the inner node's output."""
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static(value="outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": Edges(refs=(Ref("src", "result"),))}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": Edges(refs=(Ref("emb", "join.result"),))}),
    ])

    assert compiled.field(Ref("emb/holder", "value")).binding == Edges(refs=(Ref("src", "result"),))
    assert compiled.field(Ref("after", "text")).binding == Edges(refs=(Ref("emb/join", "result"),))


def test_an_unconnected_placement_keeps_the_inner_statics_and_expands_flat():
    compiled = _compiled([GraphNode(id="emb", type="inner-graph", version=1)])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("emb/holder", "value")).binding == Static(value="inner")
    assert compiled.node("emb").iterates_on is None
    assert all(compiled.node(node_id).iterates_on is None for node_id in compiled.execution_order())


def test_a_placement_is_a_node_in_the_interface_named_by_inner_address():
    """At the interface the embedded graph is one node
    whose fields are its graph's, under the placement's name."""
    compiled = _compiled([
        GraphNode(
            id="emb", type="inner-graph", version=1, title="Approve",
            fields={"holder.value": FieldContent(title="Application"), "join.result": FieldContent(title="Answer")},
        ),
    ])

    assert [(i.name, i.title) for i in compiled.interface.inputs] == [("emb.holder.value", "Application")]
    assert [(o.name, o.title) for o in compiled.interface.outputs] == [("emb.join.result", "Answer")]
    assert [i.name for i in compiled.node("emb").interface.inputs] == ["holder.value"]


def test_a_question_about_a_placements_field_reads_through_to_the_inner_field():
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static(value="outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": Edges(refs=(Ref("src", "result"),))}),
    ])

    assert compiled.field(Ref("emb", "join.result")).type is compiled.field(Ref("emb/join", "result")).type
    assert compiled.field(Ref("emb", "join.result")).index == compiled.field(Ref("emb/join", "result")).index
    assert compiled.field(Ref("emb", "holder.value")).binding == compiled.field(Ref("emb/holder", "value")).binding


def test_a_problem_found_inside_surfaces_on_the_placement():
    """The author sees `emb`, under the inner address, with the inner node named."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="upper", version=1, bindings={"text": Edges(refs=(Ref("docs", "result"),))}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": Edges(refs=(Ref("ghost", "result"),))}),
    ])

    problems = [p for p in compiled.problems if p.node_id == "emb"]
    assert [(p.code, p.field) for p in problems] == [("unknown_ref_node", "holder.value")]
    assert [p for p in compiled.problems if p.node_id == "emb/holder"] == []  # anchored on the node the author placed
    assert problems[0].message.startswith("In 'Text':")
    assert problems[0].details == {
        "source_node": "ghost",
        "placement": "Text",
        "inner_message": "Field 'value' is connected to 'ghost', which is not in the graph.",
    }
    assert not any("/" in (p.node_id or "") for p in compiled.problems)


def test_a_stale_key_on_the_placement_is_reported_on_the_placement():
    compiled = _compiled([GraphNode(id="emb", type="inner-graph", version=1, bindings={"nope.value": Static(value=1)})])

    (problem,) = compiled.problems
    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("stale_binding", False, "emb", "nope.value")


# --- the boundary scope ---------------------------------------------------------


def test_a_series_entering_through_a_scalar_field_makes_the_placement_iterate():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": Edges(refs=(Ref("docs", "result"),))}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("emb").iterates_on == Index("docs")
    assert compiled.node("emb/holder").iterates_on == Index("docs")
    assert compiled.node("emb/up").iterates_on == Index("docs")


def test_an_inner_reduction_over_the_entering_series_is_a_fold_of_one():
    """The join test: standalone the graph joins one text; embedded and fed
    three, it joins one text three times — never all three once."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"holder.value": Edges(refs=(Ref("docs", "result"),))}),
    ])

    assert compiled.node("emb/join").iterates_on == Index("docs")
    assert (compiled.field(Ref("emb", "join.result")).type, compiled.field(Ref("emb", "join.result")).index) == (Series[Txt], Index("docs"))


def test_a_series_born_inside_reduces_to_the_outer_row_by_lineage_alone():
    """An iterating inner unfold births a child of the outer index; an inner
    reduction over it collapses to the outer row without any scope rule.
    ``Index.__eq__`` reads the id alone, so the parent is asserted by name."""
    inner = _embedded_definition(
        "splitter",
        (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static(value="a\nb")}),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": Edges(refs=(Ref("holder", "result"),))}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": Edges(refs=(Ref("lines", "result"),))}),
        ),
        inputs=(Input(name="holder.value", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter", version=1, bindings={"holder.value": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(inner, Lines),
    )

    assert compiled.is_runnable, compiled.problems
    born = compiled.field(Ref("emb/lines", "result")).index
    assert (born, born.parent) == (Index("emb/lines"), Index("docs"))
    assert compiled.node("emb/join").iterates_on == Index("docs")

    # The same, when the series is born off an inner *static* the entering
    # series never touches: the block iterates, so the birth is a child of
    # the entering index all the same — never a root beside the outer rows.
    unfed = _embedded_definition(
        "splitter-unfed",
        (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static(value="a\nb")}),
            GraphNode(id="entered", type="holder", version=1),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": Edges(refs=(Ref("holder", "result"),))}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": Edges(refs=(Ref("lines", "result"),))}),
        ),
        inputs=(Input(name="entered.value", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter-unfed", version=1, bindings={"entered.value": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(unfed, Lines),
    )
    assert compiled.is_runnable, compiled.problems
    born = compiled.field(Ref("emb/lines", "result")).index
    assert (born, born.parent) == (Index("emb/lines"), Index("docs"))
    # The nodes the entering series never reaches still run once per outer row.
    assert compiled.node("emb/holder").iterates_on == Index("docs")
    assert compiled.node("emb/lines").iterates_on == Index("docs")
    assert compiled.node("emb/join").iterates_on == Index("docs")


def test_two_crossings_on_one_lineage_make_the_whole_block_iterate_on_the_deeper():
    """`docs` → `lines` (a child of `docs`); the block takes one field from
    each. The block folds over the deeper index, and so does every node
    inside it — the one fed from the shallower index broadcasts down;
    it does not run per `docs` row while its neighbours run per line."""
    inner = _embedded_definition(
        "pair",
        (
            GraphNode(id="a", type="holder", version=1),
            GraphNode(id="b", type="holder", version=1),
            GraphNode(id="ua", type="upper", version=1, bindings={"text": Edges(refs=(Ref("a", "result"),))}),
        ),
        inputs=(
            Input(name="a.value", dtype=Txt, title="A", widget=Textarea(), default=Txt(""), optional=True),
            Input(name="b.value", dtype=Txt, title="B", widget=Textarea(), default=Txt(""), optional=True),
        ),
        outputs=(Output(name="ua.result", dtype=Txt, title="A upper"), Output(name="b.result", dtype=Txt, title="B")),
    )
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": Edges(refs=(Ref("docs", "result"),))}),
            GraphNode(id="emb", type="pair", version=1, bindings={
                "a.value": Edges(refs=(Ref("docs", "result"),)),
                "b.value": Edges(refs=(Ref("lines", "result"),)),
            }),
        ]),
        _registry(inner, Lines),
    )

    assert compiled.is_runnable, compiled.problems
    per_line = Index("lines", parent=Index("docs"))
    assert compiled.node("emb").iterates_on == per_line
    assert compiled.node("emb/a").iterates_on == per_line
    assert compiled.node("emb/ua").iterates_on == per_line
    assert compiled.node("emb/b").iterates_on == per_line
    assert compiled.field(Ref("emb", "ua.result")).index == per_line


def test_a_series_entering_a_series_field_is_read_whole_and_the_block_expands_flat():
    """The graph declares it takes a series, so a series is one value to it:
    no scalar crossing, no scope, one reduction over the whole pile."""
    inner = _embedded_definition(
        "joiner",
        (GraphNode(id="join", type="join", version=1),),
        inputs=(Input(name="join.texts", dtype=Series[Txt], title="Texts", default=(), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="joiner", version=1, bindings={"join.texts": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(inner),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("emb").iterates_on is None
    assert compiled.node("emb/join").iterates_on is None
    assert compiled.field(Ref("emb", "join.result")).type is Txt


def test_two_unrelated_series_entering_one_placement_are_its_misaligned():
    inner = _embedded_definition(
        "pair",
        (
            GraphNode(id="a", type="holder", version=1),
            GraphNode(id="b", type="holder", version=1),
        ),
        inputs=(
            Input(name="a.value", dtype=Txt, title="A", widget=Textarea(), default=Txt(""), optional=True),
            Input(name="b.value", dtype=Txt, title="B", widget=Textarea(), default=Txt(""), optional=True),
        ),
        outputs=(Output(name="a.result", dtype=Txt, title="A"), Output(name="b.result", dtype=Txt, title="B")),
    )
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="d1", type="docs", version=1),
            GraphNode(id="d2", type="docs", version=1),
            GraphNode(id="emb", type="pair", version=1, bindings={
                "a.value": Edges(refs=(Ref("d1", "result"),)), "b.value": Edges(refs=(Ref("d2", "result"),)),
            }),
        ]),
        _registry(inner),
    )

    assert [(p.code, p.node_id) for p in compiled.problems] == [("misaligned", "emb")]


def test_a_nested_placement_expands_under_both_names():
    outer = _embedded_definition(
        "outer-graph",
        (
            GraphNode(id="pre", type="holder", version=1, bindings={"value": Static(value="x")}),
            GraphNode(id="inner", type="inner-graph", version=1, bindings={"holder.value": Edges(refs=(Ref("pre", "result"),))}),
        ),
        inputs=(Input(name="pre.value", dtype=Txt, title="Text", widget=Textarea(), default=Txt("x"), optional=True),),
        outputs=(Output(name="inner.join.result", dtype=Txt, title="Result"),),
    )
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="top", type="outer-graph", version=1),
            GraphNode(id="after", type="upper", version=1, bindings={"text": Edges(refs=(Ref("top", "inner.join.result"),))}),
        ]),
        _registry(_inner_definition(), outer),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.execution_order() == ("top/pre", "top/inner/holder", "top/inner/up", "top/inner/join", "after")
    assert compiled.node("top/inner/up").embedded_in == "top/inner"
    assert compiled.node("top/pre").embedded_in == "top"
    assert compiled.field(Ref("after", "text")).binding == Edges(refs=(Ref("top/inner/join", "result"),))
    assert compiled.field(Ref("top", "inner.join.result")).type is Txt


# --- what compile refuses about an embedded graph ---------------------------------------


def test_an_authored_id_with_a_slash_is_refused_and_never_collides_with_an_inner_node():
    """C5: ``/`` names the nodes of an embedded graph. An authored ``e/holder``
    beside a placement ``e`` used to appear twice in the order and be read by
    the inner nodes; now it is refused on its own."""
    compiled = _compiled([
        GraphNode(id="e/holder", type="holder", version=1, bindings={"value": Static(value="impostor")}),
        GraphNode(id="e", type="inner-graph", version=1),
    ])

    assert [(p.code, p.fatal, p.node_id) for p in compiled.problems] == [("invalid_node_id", True, "e/holder")]
    assert compiled.execution_order().count("e/holder") == 1
    assert compiled.field(Ref("e/holder", "value")).binding == Static(value="inner")


def test_a_graph_that_embeds_itself_is_a_cycle_not_a_recursion_error():
    """C6: a definition whose graph holds a node of its own type, directly or
    through another, is reported as a cycle on the node that closes it."""
    selfish = _embedded_definition(
        "selfish",
        (GraphNode(id="again", type="selfish", version=1),),
        inputs=(),
        outputs=(),
    )
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="s", type="selfish", version=1)]), _registry(selfish))

    assert [(p.code, p.node_id, p.field) for p in compiled.problems] == [("cycle", "s", "again")]
    assert not compiled.is_runnable


def test_two_graphs_that_embed_each_other_are_a_cycle():
    """C6, through another: A holds B, B holds A."""
    a = _embedded_definition("ring-a", (GraphNode(id="b", type="ring-b", version=1),), inputs=(), outputs=())
    b = _embedded_definition("ring-b", (GraphNode(id="a", type="ring-a", version=1),), inputs=(), outputs=())
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="top", type="ring-a", version=1)]), _registry(a, b))

    assert [(p.code, p.node_id, p.field) for p in compiled.problems] == [("cycle", "top", "b.a")]


def test_a_placements_interface_is_derived_from_its_graph_and_a_declaration_it_lacks_is_reported():
    """C9: the host declares what its embedded graph takes and returns; compile
    reads it off the inner graph and says where the declaration disagrees. A
    field the graph does not have is not advertised and cannot be asked for."""
    inner = _embedded_definition(
        "declared-wrong",
        _inner_graph(),
        inputs=(
            Input(name="holder.value", dtype=Txt, title="Text", widget=Textarea(), default=Txt(""), optional=True),
            Input(name="phantom.value", dtype=Txt, title="Phantom", widget=Textarea(), default=Txt(""), optional=True),
        ),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"), Output(name="ghost.out", dtype=Txt, title="Ghost")),
    )
    compiled = _compiled([GraphNode(id="emb", type="declared-wrong", version=1)], inner)

    assert [(p.code, p.fatal, p.node_id, p.field) for p in compiled.problems] == [
        ("graph_interface_mismatch", False, "emb", "phantom.value"),
        ("graph_interface_mismatch", False, "emb", "ghost.out"),
    ]
    assert compiled.is_runnable
    assert [i.name for i in compiled.interface.inputs] == ["emb.holder.value"]
    assert [o.name for o in compiled.interface.outputs] == ["emb.join.result"]
    assert [o.name for o in compiled.node("emb").interface.outputs] == ["join.result"]
    with pytest.raises(KeyError):
        compiled.field(Ref("emb", "ghost.out"))


def test_a_declared_type_that_differs_from_the_inner_graphs_is_reported_and_the_graphs_wins():
    """C9: declared and inner types can differ; the inner graph's is the one that runs."""

    class Num(DType, float):
        id = "expansion-test-num"
        title = "Number"

    inner = _embedded_definition(
        "declared-wrong-type",
        _inner_graph(),
        inputs=(Input(name="holder.value", dtype=Num, title="Text", widget=Textarea(), default=Num(0), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = _compiled([GraphNode(id="emb", type="declared-wrong-type", version=1)], inner)

    (problem,) = compiled.problems
    assert (problem.code, problem.fatal, problem.field) == ("graph_interface_mismatch", False, "holder.value")
    assert problem.details == {"declared": "expansion-test-num", "actual": "expansion-test-txt"}
    assert compiled.interface.inputs[0].dtype is Txt


def test_misaligned_inside_an_embedded_graph_is_reported_once_with_the_authors_addresses():
    """C12: one inner node fed two unrelated outer series was reported twice —
    once on the placement with expanded addresses leaking, once surfaced from
    the inner node with a different details shape. Now once, on the
    placement, with the addresses the author sees, in the shape every
    ``misaligned`` has."""
    inner = _embedded_definition(
        "pairs",
        (GraphNode(id="p", type="pair", version=1),),
        inputs=(
            Input(name="p.a", dtype=Txt, title="A", widget=Textarea(), default=Txt(""), optional=True),
            Input(name="p.b", dtype=Txt, title="B", widget=Textarea(), default=Txt(""), optional=True),
        ),
        outputs=(Output(name="p.result", dtype=Txt, title="Result"),),
    )

    class Pair(NodeDefinition):
        id = "pair"
        title = "Pair"
        description = "d"
        category = "test"

        def run(
            self,
            a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""),
            b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt(""),
        ) -> Out:
            return Txt(a + b)

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="d1", type="docs", version=1),
            GraphNode(id="d2", type="docs", version=1),
            GraphNode(id="emb", type="pairs", version=1, bindings={"p.a": Edges(refs=(Ref("d1", "result"),)), "p.b": Edges(refs=(Ref("d2", "result"),))}),
        ]),
        _registry(inner, Pair),
    )

    (problem,) = compiled.problems
    assert (problem.code, problem.node_id) == ("misaligned", "emb")
    assert problem.details == {"a": "emb.p.a", "b": "emb.p.b"}
    assert "emb/" not in problem.message and "emb/" not in str(problem.details)


def test_a_problem_surfaced_from_inside_keeps_its_details_and_rewrites_the_addresses():
    """C12: surfacing adds ``placement`` and ``inner_message`` beside the
    inner problem's own details, with every address in them rewritten to the
    author's — never a nested ``inner_details``."""
    compiled = _compiled([
        GraphNode(id="d1", type="docs", version=1),
        GraphNode(id="d2", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"up.text": Edges(refs=(Ref("d1", "result"), Ref("d2", "result")))}),
    ])

    (problem,) = compiled.problems
    assert (problem.code, problem.node_id, problem.field) == ("union_needs_one_index", "emb", "up.text")
    assert problem.details == {"placement": "Upper", "inner_message": "Field 'text' has several edges; that only works when they are all rows of one table."}
