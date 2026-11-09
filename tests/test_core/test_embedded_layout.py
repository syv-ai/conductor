"""A graph compiled on its own and embedded as a node: its nodes run under the graph node's name, under the row the graph node runs on."""

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from conductor import (
    CompiledGraph,
    DType,
    FieldContent,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Output,
    Param,
    Ref,
    Result,
    Series,
    Static,
    node,
)
from conductor._internal.series import Index
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "embedded-layout-test-txt"
    display_title = "Text"


Out = Annotated[Txt, Result(title="Result")]


@node("holder", title="Text", description="d", category="test")
def holder(value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return value


@node("upper", title="Upper case", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    """A reduction."""
    return Txt("+".join(texts))


@node("docs", title="Documents", description="d", category="test")
def docs(folder: Annotated[Txt, Param(title="Folder", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Texts")]:
    return [Txt("a"), Txt("b")]


@node("lines", title="Lines", description="d", category="test")
def lines(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    return text.splitlines()


def _inner_graph():
    """The embedded graph: a holder, uppercased, then joined. Standalone it
    takes one text at `holder.value` and returns `join.result`."""
    return (
        GraphNode(id="holder", type="holder", version=1, title="Text", bindings={"value": Static("inner")}),
        GraphNode(id="up", type="upper", version=1, title="Upper", bindings={"text": From("holder.result")}),
        GraphNode(id="join", type="join", version=1, title="Join", bindings={"texts": From("up.result")}),
    )


def _registry(*extra):
    registry = NodeRegistry()
    for node_cls in (holder, upper_node, join_node, docs, *extra):
        registry = registry.extended_with([node_cls])
    return registry


def _inner_definition():
    return embedded_graph_node("inner-graph", _inner_graph(), _registry())


def _compiled(nodes, *extra):
    return CompiledGraph.from_graph(Graph(nodes=nodes), _registry(_inner_definition(), *extra))


# --- the nodes of a graph node -----------------------------------------------


def test_the_inner_nodes_are_nodes_of_the_one_run_under_the_graph_nodes_name():
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static("outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": From("src.result")}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": From("emb.result")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.execution_order == ("src", "emb/holder", "emb/up", "emb/join", "after")
    assert compiled.node("emb/up").embedded_in == "emb"
    assert compiled.node("src").embedded_in is None
    # The node that embeds the graph is not a unit of the run; its inner nodes are.
    emb = compiled.node("emb")
    assert emb.embedded_graph is not None and compiled.node("src").embedded_graph is None
    assert emb.fingerprint is None


def test_a_graph_holds_the_values_typed_on_it_by_the_name_it_offers():
    """Typed on the embedding node, read where it landed: the inner field's type reads it."""
    compiled = _compiled([
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": Static("outer"), "x": Static("y")}),
    ])

    assert compiled.node("emb").statics == {"value": "outer"}
    assert compiled.node("emb").statics["value"] == compiled.node("emb/holder").statics["value"]


def test_the_graph_nodes_bindings_move_onto_the_inner_fields_they_name():
    """An edge into `emb.holder.value` replaces the inner author's static;
    an outer edge from `emb.join.result` reads the inner node's output."""
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static("outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": From("src.result")}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": From("emb.result")}),
    ])

    assert compiled.field(Ref("emb/holder", "value")).binding == From("src.result")
    assert compiled.field(Ref("after", "text")).binding == From("emb/join.result")


def test_an_unconnected_placement_keeps_the_inner_statics_and_lays_out_flat():
    compiled = _compiled([GraphNode(id="emb", type="inner-graph", version=1)])

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("emb/holder", "value")).binding == Static("inner")
    assert compiled.node("emb").wiring._runs_per_row_of is None
    assert all(compiled.node(node_id).wiring._runs_per_row_of is None for node_id in compiled.execution_order)


def test_a_graph_node_is_a_node_in_the_interface_named_by_what_it_offers():
    """At the interface the embedded graph is one node
    whose fields are its graph's, under the graph node's name."""
    compiled = _compiled([
        GraphNode(
            id="emb", type="inner-graph", version=1, title="Approve",
            fields={"value": FieldContent(title="Application"), "result": FieldContent(title="Answer")},
        ),
    ])

    assert [(i.name, i.title) for i in compiled.interface.inputs] == [("emb.value", "Application")]
    assert [(o.name, o.title) for o in compiled.interface.outputs] == [("emb.result", "Answer")]
    assert [i.name for i in compiled.node("emb").interface.inputs] == ["value"]


def test_a_field_inside_a_placement_has_one_address():
    """``emb/join.result`` is the field; the author's spelling of it,
    ``emb.join.result``, names a field of the graph node, which has none."""
    compiled = _compiled([
        GraphNode(id="src", type="holder", version=1, bindings={"value": Static("outer")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": From("src.result")}),
    ])

    assert compiled.field(Ref("emb/join", "result")).type is Txt
    assert compiled.field(Ref("emb/holder", "value")).binding == From("src.result")
    with pytest.raises(KeyError, match="'emb' is a graph"):
        compiled.field(Ref("emb", "join.result"))


def sheet_outputs():
    return (Output("excl. VAT", Txt),)

@node("sheet", title="Sheet", description="d", category="t", compute_outputs=sheet_outputs)
def sheet() -> Mapping[str, Any]:
    """One output named after a column, as a table's fold names them: free text, dots and all."""
    return {"excl. VAT": Txt("net")}


def test_a_field_name_with_a_dot_beside_an_embedded_graph_is_a_plain_field():
    """The first dot splits an address, and the field is free text: ``sheet.excl. VAT``
    is the field ``excl. VAT`` on ``sheet``, wherever an embedded graph sits."""
    compiled = _compiled([
        GraphNode(id="sheet", type="sheet", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": From(Ref("sheet", "excl. VAT"))}),
        GraphNode(id="after", type="upper", version=1, bindings={"text": From("emb.result")}),
    ], sheet)

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("sheet", "excl. VAT")).type is Txt
    assert compiled.field(Ref("emb/holder", "value")).binding == From(Ref("sheet", "excl. VAT"))
    assert compiled.results(compiled.run_sync())["after"]["result"] == "NET"


def test_a_broken_edge_into_a_graph_node_is_the_graph_nodes_own_problem():
    """The edge sits on the node the author drew, on the field it names,
    in the plain words any node gets; the graph node is not worked out, and
    says why."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="n", type="upper", version=1, bindings={"text": From("docs.result")}),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": From("ghost.result")}),
    ])

    (problem,) = compiled.problems
    assert (problem.type, problem.loc[0], problem.loc[1]) == ("unknown_ref_node", "emb", "value")
    assert problem.ctx == {"source_node": "ghost"}
    assert compiled.node("emb").state == "wiring_failed"
    assert compiled.node("emb").cause is problem


def test_a_stale_key_on_the_placement_is_reported_on_the_placement():
    compiled = _compiled([GraphNode(id="emb", type="inner-graph", version=1, bindings={"nope.value": Static(1)})])

    (problem,) = compiled.problems
    assert (problem.type, problem.fatal, problem.loc[0], problem.loc[1]) == ("stale_binding", True, "emb", "nope.value")


# --- the row a graph node runs on -----------------------------------------------


def test_a_series_entering_through_a_scalar_field_makes_the_placement_iterate():
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": From("docs.result")}),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("emb").wiring._runs_per_row_of == Index("docs.result")
    assert compiled.node("emb/holder").wiring._runs_per_row_of == Index("docs.result")
    assert compiled.node("emb/up").wiring._runs_per_row_of == Index("docs.result")


def test_an_inner_reduction_over_the_entering_series_is_a_fold_of_one():
    """The join test: standalone the graph joins one text; embedded and fed
    three, it joins one text three times — never all three once."""
    compiled = _compiled([
        GraphNode(id="docs", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"value": From("docs.result")}),
    ])

    assert compiled.node("emb/join").wiring._runs_per_row_of == Index("docs.result")
    assert (compiled.field(Ref("emb/join", "result")).type, compiled.field(Ref("emb/join", "result"))._index) == (Series[Txt], Index("docs.result"))


def test_a_series_made_inside_reduces_to_the_outer_row_by_lineage_alone():
    """An inner unfold running per row makes rows of a child of the outer index; an inner
    reduction over it collapses to the outer row without any scope rule.
    ``Index.__eq__`` reads the id alone, so the parent is asserted by name."""
    inner = embedded_graph_node("splitter", (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": From("lines.result")}),
        ), _registry(lines))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter", version=1, bindings={"value": From("docs.result")}),
        ]),
        _registry(inner, lines),
    )

    assert compiled.is_runnable, compiled.problems
    made = compiled.field(Ref("emb/lines", "result"))._index
    assert (made, made.parent) == (Index("emb/lines.result"), Index("docs.result"))
    assert compiled.node("emb/join").wiring._runs_per_row_of == Index("docs.result")

    # The same, when the series is made off an inner *static* the entering
    # series never touches: the block runs per row, so the new index is a child of
    # the entering index all the same — never a root beside the outer rows.
    unfed = embedded_graph_node("splitter-unfed", (
            GraphNode(id="holder", type="holder", version=1, bindings={"value": Static("a\nb")}),
            GraphNode(id="entered", type="holder", version=1),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": From("holder.result")}),
            GraphNode(id="join", type="join", version=1, bindings={"texts": From("lines.result")}),
        ), _registry(lines))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="splitter-unfed", version=1, bindings={"holder.value": From("docs.result")}),
        ]),
        _registry(unfed, lines),
    )
    assert compiled.is_runnable, compiled.problems
    made = compiled.field(Ref("emb/lines", "result"))._index
    assert (made, made.parent) == (Index("emb/lines.result"), Index("docs.result"))
    # The nodes the entering series never reaches still run once per outer row.
    assert compiled.node("emb/holder").wiring._runs_per_row_of == Index("docs.result")
    assert compiled.node("emb/lines").wiring._runs_per_row_of == Index("docs.result")
    assert compiled.node("emb/join").wiring._runs_per_row_of == Index("docs.result")


def test_two_crossings_on_one_lineage_make_the_whole_block_iterate_on_the_deeper():
    """`docs` → `lines` (a child of `docs`); the block takes one field from
    each. The block folds over the deeper index, and so does every node
    inside it — the one fed from the shallower index broadcasts down;
    it does not run per `docs` row while its neighbours run per line."""
    inner = embedded_graph_node("pair", (
            GraphNode(id="a", type="holder", version=1),
            GraphNode(id="b", type="holder", version=1),
            GraphNode(id="ua", type="upper", version=1, bindings={"text": From("a.result")}),
        ), _registry())
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="lines", type="lines", version=1, bindings={"text": From("docs.result")}),
            GraphNode(id="emb", type="pair", version=1, bindings={
                "a.value": From("docs.result"),
                "b.value": From("lines.result"),
            }),
        ]),
        _registry(inner, lines),
    )

    assert compiled.is_runnable, compiled.problems
    per_line = Index("lines.result", parent=Index("docs.result"))
    assert compiled.node("emb").wiring._runs_per_row_of == per_line
    assert compiled.node("emb/a").wiring._runs_per_row_of == per_line
    assert compiled.node("emb/ua").wiring._runs_per_row_of == per_line
    assert compiled.node("emb/b").wiring._runs_per_row_of == per_line
    assert compiled.field(Ref("emb/ua", "result"))._index == per_line


def test_a_series_entering_a_series_field_is_read_whole_and_the_block_lays_out_flat():
    """The graph declares it takes a series, so a series is one value to it:
    no scalar crossing, no scope, one reduction over the whole pile."""
    inner = embedded_graph_node("joiner", (GraphNode(id="join", type="join", version=1),), _registry())
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1),
            GraphNode(id="emb", type="joiner", version=1, bindings={"texts": From("docs.result")}),
        ]),
        _registry(inner),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("emb").wiring._runs_per_row_of is None
    assert compiled.node("emb/join").wiring._runs_per_row_of is None
    assert compiled.field(Ref("emb/join", "result")).type is Txt


def test_two_unrelated_series_entering_one_placement_are_its_misaligned():
    inner = embedded_graph_node("pair", (
            GraphNode(id="a", type="holder", version=1),
            GraphNode(id="b", type="holder", version=1),
        ), _registry())
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="d1", type="docs", version=1),
            GraphNode(id="d2", type="docs", version=1),
            GraphNode(id="emb", type="pair", version=1, bindings={
                "a.value": From("d1.result"), "b.value": From("d2.result"),
            }),
        ]),
        _registry(inner),
    )

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("misaligned", "emb")]
    (misaligned,) = compiled.problems
    graph_node = compiled.node("emb")
    assert graph_node.state == "wiring_failed"
    assert graph_node.problems == compiled.problems
    assert graph_node.wiring is None
    raised = graph_node.cause
    assert raised == misaligned
    assert "emb" not in compiled.decisions


def test_a_nested_embedded_graph_is_addressed_by_its_path():
    outer = embedded_graph_node("outer-graph", (
            GraphNode(id="pre", type="holder", version=1, bindings={"value": Static("x")}),
            GraphNode(id="inner", type="inner-graph", version=1, bindings={"value": From("pre.result")}),
        ), _registry(_inner_definition()))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="top", type="outer-graph", version=1),
            GraphNode(id="after", type="upper", version=1, bindings={"text": From("top.result")}),
        ]),
        _registry(_inner_definition(), outer),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.execution_order == ("top/pre", "top/inner/holder", "top/inner/up", "top/inner/join", "after")
    assert compiled.node("top/inner/up").embedded_in == "top/inner"
    assert compiled.node("top/pre").embedded_in == "top"
    assert compiled.field(Ref("after", "text")).binding == From("top/inner/join.result")
    assert compiled.field(Ref("top/inner/join", "result")).type is Txt



def test_a_broken_graph_inside_a_graph_breaks_the_graph_around_it():
    """An unknown inner node leaves its graph unembeddable, and so the graph
    that embeds it too. Each level shows one ``embedded_graph_broken`` where
    it is embedded; the unknown node is the innermost graph's own problem.
    Each is one node where it is embedded, read inside through its version."""
    broken = embedded_graph_node("broken-graph", (GraphNode(id="x", type="nothing", version=1), GraphNode(id="up", type="upper", version=1)), _registry())
    wrapper = embedded_graph_node("wrapper-graph", (GraphNode(id="mid", type="broken-graph", version=1),), _registry(broken))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="top", type="wrapper-graph", version=1)]), _registry(broken, wrapper),
    )

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("embedded_graph_broken", "top")]
    top = compiled.node("top")
    assert [(p.type, p.loc[0]) for p in top.version.problems] == [("embedded_graph_broken", "mid")]
    assert [(p.type, p.loc[0]) for p in top.version.node("mid").version.problems] == [("unknown_node_type", "x")]
    for graph_node in (top, top.version.node("mid")):
        assert (graph_node.state, graph_node.embedded_graph is not None) == ("wiring_failed", True)
        assert graph_node.interface is not None  # the box and its handles still draw
        assert graph_node.wiring is None
        raised = graph_node.cause
        assert raised.type == "embedded_graph_broken"


# --- what compile refuses about an embedded graph ---------------------------------------


def test_an_authored_id_with_a_slash_is_refused_and_never_collides_with_an_inner_node():
    """``/`` names the nodes of an embedded graph, so an authored ``e/holder``
    beside a graph node ``e`` is refused on its own: it neither appears twice
    in the order nor is read by the inner nodes."""
    compiled = _compiled([
        GraphNode(id="e/holder", type="holder", version=1, bindings={"value": Static("impostor")}),
        GraphNode(id="e", type="inner-graph", version=1),
    ])

    assert [(p.type, p.fatal, p.loc[0]) for p in compiled.problems] == [("invalid_node_id", True, "e/holder")]
    assert compiled.execution_order.count("e/holder") == 1
    assert compiled.field(Ref("e/holder", "value")).binding == Static("inner")


def test_a_refused_authored_id_is_not_the_problem_of_the_inner_node_sharing_it():
    """`e` can be embedded, so its `holder` runs as `e/holder`; the author also
    wrote `e/holder`, which is refused. The refusal is the author's: it stays
    in the graph's problems and never shows on the inner node."""
    compiled = _compiled([
        GraphNode(id="e", type="inner-graph", version=1),
        GraphNode(id="e/holder", type="upper", version=1),
    ])

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("invalid_node_id", "e/holder")]
    inner = compiled.node("e/holder")
    assert (inner.state, inner.embedded_in, inner.problems) == ("ready", "e", ())


def test_an_authored_id_holding_a_slash_is_refused_and_stays_the_authors():
    """The author wrote `e/holder` beside an embedded graph `e` whose own
    `holder` has a type the registry lacks. The authored id is refused; the
    graph cannot be embedded, which is one problem on `e`, and
    `node("e/holder")` is the author's node, refused."""
    lost = embedded_graph_node("lost", (GraphNode(id="holder", type="nope", version=1),), _registry())
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="e", type="lost", version=1),
        GraphNode(id="e/holder", type="upper", version=1),
    ]), _registry(lost))

    assert [(p.type, p.loc) for p in compiled.problems] == [
        ("invalid_node_id", ("e/holder",)), ("embedded_graph_broken", ("e",)),
    ]
    assert [(p.type, p.loc[0]) for p in compiled.node("e").version.problems] == [("unknown_node_type", "holder")]
    assert compiled.node("e").state == "wiring_failed"
    refused = compiled.node("e/holder")
    assert (refused.state, refused.embedded_in, refused.cause.type) == ("resolution_failed", None, "invalid_node_id")


def test_misaligned_inside_an_embedded_graph_is_reported_once_with_the_authors_addresses():
    """One inner node fed two unrelated outer series is reported once, on the
    graph node, with the addresses the author sees, in the shape every
    ``misaligned`` has."""

    @node("pair", title="Pair", description="d", category="test")
    def pair(
        a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""),
        b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt(""),
    ) -> Out:
        return Txt(a + b)

    inner = embedded_graph_node("pairs", (GraphNode(id="p", type="pair", version=1),), _registry(pair))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="d1", type="docs", version=1),
            GraphNode(id="d2", type="docs", version=1),
            GraphNode(id="emb", type="pairs", version=1, bindings={"a": From("d1.result"), "b": From("d2.result")}),
        ]),
        _registry(inner, pair),
    )

    (problem,) = compiled.problems
    assert (problem.type, problem.loc[0]) == ("misaligned", "emb")
    assert problem.ctx == {"a": "emb.a", "b": "emb.b"}
    assert "emb.p" not in problem.msg and "emb.p" not in str(problem.ctx)


def test_an_edge_into_a_field_fed_inside_the_graph_is_a_stale_binding():
    """``up.text`` is fed by ``holder`` inside, so the graph does not offer
    it: an edge from outside reaches nothing, as a field any node lacks."""
    compiled = _compiled([
        GraphNode(id="d1", type="docs", version=1),
        GraphNode(id="d2", type="docs", version=1),
        GraphNode(id="emb", type="inner-graph", version=1, bindings={"text": From("d1.result", "d2.result")}),
    ])

    (problem,) = compiled.problems
    assert (problem.type, problem.loc[0], problem.loc[1]) == ("stale_binding", "emb", "text")


