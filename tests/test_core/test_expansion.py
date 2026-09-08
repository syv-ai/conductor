"""An embedded flow expands under its placement's name, and its boundary is an index scope."""

from collections.abc import Mapping
from typing import Annotated, ClassVar

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.graph.binding import Edges, Static
from conductor.graph.compiler import compile_graph
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.interface import Interface
from conductor.metadata import Input, Output
from conductor.node import GraphVersion, NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Index, Series
from conductor.widgets import ConnectionList, Textarea


class Txt(DType, str):
    id = "expansion-test-txt"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


class Holder(NodeDefinition):
    id = "holder"
    title = "Text"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return value


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper case"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return Txt(text.upper())


class Join(NodeDefinition):
    """A reduction."""

    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], ConnectionList(title="Texts")] = ()) -> Out:
        return Txt("+".join(texts))


class Docs(NodeDefinition):
    id = "docs"
    title = "Documents"
    description = "d"
    category = "test"

    def run(self, folder: Annotated[Txt, Textarea(title="Folder")] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
        return [Txt("a"), Txt("b")]


class Lines(NodeDefinition):
    id = "lines"
    title = "Lines"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
        return text.splitlines()


def _inner_graph():
    """The embedded flow: a holder, uppercased, then joined. Standalone it
    takes one text at `holder.value` and returns `join.result`."""
    return (
        GraphNode(id="holder", type="holder", version=1, title="Text", bindings={"value": Static(value="inner")}),
        GraphNode(id="up", type="upper", version=1, title="Upper", bindings={"text": Edges(refs=(Ref("holder", "result"),))}),
        GraphNode(id="join", type="join", version=1, title="Join", bindings={"texts": Edges(refs=(Ref("up", "result"),))}),
    )


def _embedded_definition(node_id, graph, inputs, outputs):
    """What a host builds from a FlowVersion: a definition whose one
    version is a `GraphVersion` — the flow's interface, and the graph."""

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
        "inner-flow",
        _inner_graph(),
        inputs=(Input(name="holder.value", dtype=Txt, title="Text", widget=Textarea(title="Text"), default=Txt("indre"), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )


def _compiled(nodes, *extra):
    return compile_graph(Graph(nodes=nodes), _registry(_inner_definition(), *extra))


# --- expansion ---------------------------------------------------------------


def test_the_inner_nodes_are_nodes_of_the_one_run_under_the_placements_name():
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static(value="outer")}),
        GraphNode(id="emb", type="inner-flow", version=1, bindings={"holder.value": Edges(refs=(Ref("src", "result"),))}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": Edges(refs=(Ref("emb", "join.result"),))}),
    ])

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.execution_order() == ("src", "emb/holder", "emb/up", "emb/join", "after")
    assert compiled.placement_of("emb/up") == "emb"
    assert compiled.placement_of("src") is None
    with pytest.raises(KeyError):
        compiled.node("emb")


def test_the_placements_bindings_move_onto_the_inner_fields_they_name():
    """An edge into `emb.holder.value` replaces the inner author's static;
    an outer edge from `emb.join.result` reads the inner node's output."""
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static(value="outer")}),
        GraphNode(id="emb", type="inner-flow", version=1, bindings={"holder.value": Edges(refs=(Ref("src", "result"),))}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": Edges(refs=(Ref("emb", "join.result"),))}),
    ])

    assert compiled.value_source("emb/holder", "value") == Edges(refs=(Ref("src", "result"),))
    assert compiled.value_source("after", "text") == Edges(refs=(Ref("emb/join", "result"),))
    assert compiled.dependencies("emb/holder") == frozenset({"src"})
    assert compiled.dependencies("after") == frozenset({"emb/join"})


def test_an_unconnected_placement_keeps_the_inner_statics_and_expands_flat():
    compiled = _compiled([GraphNode(id="emb", type="inner-flow", version=1)])

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.value_source("emb/holder", "value") == Static(value="inner")
    assert compiled.lifted_on("emb") is None
    assert all(compiled.lifted_on(node_id) is None for node_id in compiled.execution_order())


def test_a_placement_is_a_node_in_the_interface_named_by_inner_address():
    """At the interface the embedded flow is one node
    whose fields are its flow's, under the placement's name."""
    compiled = _compiled([
        GraphNode(
            id="emb", type="inner-flow", version=1, title="Approve",
            fields={"holder.value": FieldContent(title="Application"), "join.result": FieldContent(title="Answer")},
        ),
    ])

    assert [(i.name, i.title) for i in compiled.interface.inputs] == [("emb.holder.value", "Application")]
    assert [(o.name, o.title) for o in compiled.interface.outputs] == [("emb.join.result", "Answer")]
    assert [i.name for i in compiled.roster("emb").inputs] == ["holder.value"]


def test_a_question_about_a_placements_field_reads_through_to_the_inner_field():
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static(value="outer")}),
        GraphNode(id="emb", type="inner-flow", version=1, bindings={"holder.value": Edges(refs=(Ref("src", "result"),))}),
    ])

    assert compiled.carried(Ref("emb", "join.result")) == compiled.carried(Ref("emb/join", "result"))
    assert compiled.value_source("emb", "holder.value") == compiled.value_source("emb/holder", "value")


def test_a_problem_found_inside_surfaces_on_the_placement():
    """The author sees `emb`, under the inner address, with the inner node named."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="upper", version=1, bindings={"text": Edges(refs=(Ref("docs", "result"),))}),
        GraphNode(id="emb", type="inner-flow", version=1, bindings={"holder.value": Edges(refs=(Ref("ghost", "result"),))}),
    ])

    problems = compiled.problems_for(node_id="emb")
    assert [(p.code, p.field) for p in problems] == [("unknown_ref_node", "holder.value")]
    assert problems[0].message.startswith("In 'Text':")
    assert problems[0].details == {
        "placement": "Text",
        "inner_message": "Field 'value' is connected to 'ghost', which is not in the flow.",
        "inner_details": {"source_node": "ghost"},
    }
    assert not any("/" in (p.node_id or "") for p in compiled.problems_for())


def test_a_stale_key_on_the_placement_is_reported_on_the_placement():
    compiled = _compiled([GraphNode(id="emb", type="inner-flow", version=1, bindings={"nope.value": Static(value=1)})])

    (problem,) = compiled.problems_for()
    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("stale_binding", False, "emb", "nope.value")


# --- the boundary scope ---------------------------------------------------------


def test_a_series_entering_through_a_scalar_field_lifts_the_placement():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-flow", version=1, bindings={"holder.value": Edges(refs=(Ref("docs", "result"),))}),
    ])

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.lifted_on("emb") == Index("docs")
    assert compiled.lifted_on("emb/holder") == Index("docs")
    assert compiled.lifted_on("emb/up") == Index("docs")


def test_an_inner_reduction_over_the_entering_series_is_a_fold_of_one():
    """The join test: standalone the flow joins one text; embedded and fed
    three, it joins one text three times — never all three once."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-flow", version=1, bindings={"holder.value": Edges(refs=(Ref("docs", "result"),))}),
    ])

    assert compiled.lifted_on("emb/join") == Index("docs")
    joined = compiled.carried(Ref("emb", "join.result"))
    assert (joined.dtype, joined.index) == (Series[Txt], Index("docs"))


def test_a_series_born_inside_reduces_to_the_outer_row_by_lineage_alone():
    """A lifted inner unfold births a child of the outer index; an inner
    reduction over it collapses to the outer row without any scope rule.
    ``Index.__eq__`` reads the id alone, so the parent is asserted by name."""
    inner = _embedded_definition(
        "splitter",
        (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static(value="a\nb")}),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": Edges(refs=(Ref("holder", "result"),))}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": Edges(refs=(Ref("lines", "result"),))}),
        ),
        inputs=(Input(name="holder.value", dtype=Txt, title="Text", widget=Textarea(title="Text"), default=Txt(""), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = compile_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter", version=1, bindings={"holder.value": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(inner, Lines),
    )

    assert compiled.is_runnable, compiled.problems_for()
    born = compiled.carried(Ref("emb/lines", "result")).index
    assert (born, born.parent) == (Index("emb/lines"), Index("docs"))
    assert compiled.lifted_on("emb/join") == Index("docs")

    # The same, when the series is born off an inner *static* the entering
    # series never touches: the block is lifted, so the birth is a child of
    # the entering index all the same — never a root beside the outer rows.
    unfed = _embedded_definition(
        "splitter-unfed",
        (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static(value="a\nb")}),
            GraphNode(id="entered", type="holder", version=1),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": Edges(refs=(Ref("holder", "result"),))}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": Edges(refs=(Ref("lines", "result"),))}),
        ),
        inputs=(Input(name="entered.value", dtype=Txt, title="Text", widget=Textarea(title="Text"), default=Txt(""), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = compile_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter-unfed", version=1, bindings={"entered.value": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(unfed, Lines),
    )
    assert compiled.is_runnable, compiled.problems_for()
    born = compiled.carried(Ref("emb/lines", "result")).index
    assert (born, born.parent) == (Index("emb/lines"), Index("docs"))
    # The nodes the entering series never reaches still run once per outer row.
    assert compiled.lifted_on("emb/holder") == Index("docs")
    assert compiled.lifted_on("emb/lines") == Index("docs")
    assert compiled.lifted_on("emb/join") == Index("docs")


def test_two_crossings_on_one_lineage_lift_the_whole_block_on_the_deeper():
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
            Input(name="a.value", dtype=Txt, title="A", widget=Textarea(title="A"), default=Txt(""), optional=True),
            Input(name="b.value", dtype=Txt, title="B", widget=Textarea(title="B"), default=Txt(""), optional=True),
        ),
        outputs=(Output(name="ua.result", dtype=Txt, title="A upper"), Output(name="b.result", dtype=Txt, title="B")),
    )
    compiled = compile_graph(
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

    assert compiled.is_runnable, compiled.problems_for()
    per_line = Index("lines", parent=Index("docs"))
    assert compiled.lifted_on("emb") == per_line
    assert compiled.lifted_on("emb/a") == per_line
    assert compiled.lifted_on("emb/ua") == per_line
    assert compiled.lifted_on("emb/b") == per_line
    assert compiled.carried(Ref("emb", "ua.result")).index == per_line


def test_a_series_entering_a_series_field_is_read_whole_and_the_block_expands_flat():
    """The flow declares it takes a series, so a series is one value to it:
    no scalar crossing, no scope, one reduction over the whole pile."""
    inner = _embedded_definition(
        "joiner",
        (GraphNode(id="join", type="join", version=1),),
        inputs=(Input(name="join.texts", dtype=Series[Txt], title="Texts", widget=ConnectionList(title="Texts"), default=(), optional=True),),
        outputs=(Output(name="join.result", dtype=Txt, title="Result"),),
    )
    compiled = compile_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="joiner", version=1, bindings={"join.texts": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(inner),
    )

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.lifted_on("emb") is None
    assert compiled.lifted_on("emb/join") is None
    assert compiled.carried(Ref("emb", "join.result")).dtype is Txt


def test_two_unrelated_series_entering_one_placement_are_its_misaligned():
    inner = _embedded_definition(
        "pair",
        (
            GraphNode(id="a", type="holder", version=1),
            GraphNode(id="b", type="holder", version=1),
        ),
        inputs=(
            Input(name="a.value", dtype=Txt, title="A", widget=Textarea(title="A"), default=Txt(""), optional=True),
            Input(name="b.value", dtype=Txt, title="B", widget=Textarea(title="B"), default=Txt(""), optional=True),
        ),
        outputs=(Output(name="a.result", dtype=Txt, title="A"), Output(name="b.result", dtype=Txt, title="B")),
    )
    compiled = compile_graph(
        Graph(nodes=[
            GraphNode(id="d1", type="docs", version=1),
            GraphNode(id="d2", type="docs", version=1),
            GraphNode(id="emb", type="pair", version=1, bindings={
                "a.value": Edges(refs=(Ref("d1", "result"),)), "b.value": Edges(refs=(Ref("d2", "result"),)),
            }),
        ]),
        _registry(inner),
    )

    assert [(p.code, p.node_id) for p in compiled.problems_for()] == [("misaligned", "emb")]


def test_a_nested_placement_expands_under_both_names():
    outer = _embedded_definition(
        "outer-flow",
        (
            GraphNode(id="pre", type="holder", version=1, bindings={"value": Static(value="x")}),
            GraphNode(id="inner", type="inner-flow", version=1, bindings={"holder.value": Edges(refs=(Ref("pre", "result"),))}),
        ),
        inputs=(Input(name="pre.value", dtype=Txt, title="Text", widget=Textarea(title="Text"), default=Txt("x"), optional=True),),
        outputs=(Output(name="inner.join.result", dtype=Txt, title="Result"),),
    )
    compiled = compile_graph(
        Graph(nodes=[
            GraphNode(id="top", type="outer-flow", version=1),
            GraphNode(id="after", type="upper", version=1, bindings={"text": Edges(refs=(Ref("top", "inner.join.result"),))}),
        ]),
        _registry(_inner_definition(), outer),
    )

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.execution_order() == ("top/pre", "top/inner/holder", "top/inner/up", "top/inner/join", "after")
    assert compiled.placement_of("top/inner/up") == "top/inner"
    assert compiled.placement_of("top/pre") == "top"
    assert compiled.value_source("after", "text") == Edges(refs=(Ref("top/inner/join", "result"),))
    assert compiled.carried(Ref("top", "inner.join.result")).dtype is Txt
