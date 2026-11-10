"""A graph compiled on its own and embedded as a node: its nodes run under the graph node's name, under the row the graph node runs on."""

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
from conductor import (
    CompiledGraph,
    DType,
    GraphRecord,
    NodeRecord,
    Output,
    Param,
    Ref,
    Result,
    Series,
    node,
)
from conductor._internal.graph.binding import From, Static
from conductor.widgets import Textarea
from helpers import placed, record, registry_of

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
        placed("holder", "holder", defaults_={"value": "inner"}),
        placed("up", "upper", text=From("holder.result")),
        placed("join", "join", texts=From("up.result")),
    )


def _registry(*extra):
    return registry_of(holder, upper_node, join_node, docs, *extra)


def _inner_definition():
    return embedded_graph_node("inner-graph", _inner_graph(), _registry())


def _compiled(nodes, *extra):
    return CompiledGraph.from_graph(record(*nodes), _registry(_inner_definition(), *extra))


# --- the nodes of a graph node -----------------------------------------------


def test_the_inner_nodes_are_nodes_of_the_one_run_under_the_graph_nodes_name():
    compiled = _compiled([
        placed("src", "holder", defaults_={"value": "outer"}),
        placed("emb", "inner-graph", value=From("src.result")),
        placed("after", "upper", text=From("emb.result")),
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
        placed("emb", "inner-graph", value=Static("outer"), x=Static("y")),
    ])

    assert compiled.node("emb").statics == {"value": "outer"}
    assert compiled.node("emb").statics["value"] == compiled.node("emb/holder").statics["value"]


def test_the_graph_nodes_bindings_move_onto_the_inner_fields_they_name():
    """An edge into `emb.holder.value` replaces the inner author's static;
    an outer edge from `emb.join.result` reads the inner node's output."""
    compiled = _compiled([
        placed("src", "holder", defaults_={"value": "outer"}),
        placed("emb", "inner-graph", value=From("src.result")),
        placed("after", "upper", text=From("emb.result")),
    ])

    assert compiled.node("emb/holder").graph_node.bindings.get("value") == From("src.result")
    assert compiled.node("after").graph_node.bindings.get("text") == From("emb/join.result")


def test_an_unconnected_placement_keeps_the_inner_statics_and_lays_out_flat():
    compiled = _compiled([placed("emb", "inner-graph")])

    assert compiled.is_runnable, compiled.problems
    assert compiled.node("emb/holder").graph_node.bindings.get("value") == Static("inner")
    assert all(plan.runs_per_row_of is None for plan in compiled.run_plan.nodes.values())


def test_a_graph_node_is_a_node_in_the_interface_named_by_what_it_offers():
    """At the interface the embedded graph is one node
    whose fields are its graph's, under the graph node's name."""
    compiled = _compiled([
        placed("emb", "inner-graph"),
    ])

    assert [(i.name, i.title) for i in compiled.interface.inputs] == [("emb.value", "Text")]
    assert [(o.name, o.title) for o in compiled.interface.outputs] == [("emb.result", "Result")]
    assert [i.name for i in compiled.node("emb").interface.inputs] == ["value"]


def test_a_field_inside_a_placement_is_reached_by_its_place_or_the_name_it_is_offered_under():
    """``emb/join.result`` is the field and ``emb.result`` the name the graph offers it by;
    ``emb.join.result`` names a field the graph node doesn't offer."""
    compiled = _compiled([
        placed("src", "holder", defaults_={"value": "outer"}),
        placed("emb", "inner-graph", value=From("src.result")),
    ])

    assert compiled.field(Ref("emb/join", "result")).dtype is Txt
    assert compiled.node("emb/holder").graph_node.bindings.get("value") == From("src.result")
    assert compiled.field(Ref("emb", "result")) is compiled.field(Ref("emb/join", "result"))
    with pytest.raises(KeyError, match="'emb' offers no field"):
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
        placed("sheet", "sheet"),
        placed("emb", "inner-graph", value=From(Ref("sheet", "excl. VAT"))),
        placed("after", "upper", text=From("emb.result")),
    ], sheet)

    assert compiled.is_runnable, compiled.problems
    assert compiled.field(Ref("sheet", "excl. VAT")).dtype is Txt
    assert compiled.node("emb/holder").graph_node.bindings.get("value") == From(Ref("sheet", "excl. VAT"))
    assert compiled.results(compiled.run_sync())["after"]["result"] == "NET"


def test_a_broken_edge_into_a_graph_node_is_the_graph_nodes_own_problem():
    """The edge sits on the node the author drew, on the field it names,
    in the plain words any node gets; the graph node is not worked out, and
    says why."""
    compiled = _compiled([
        placed("docs", "docs"),
        placed("n", "upper", text=From("docs.result")),
        placed("emb", "inner-graph", value=From("ghost.result")),
    ])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.loc[0], make_problem.loc[1]) == ("unknown_ref_node", "emb", "value")
    assert make_problem.ctx == {"source_node": "ghost"}
    assert compiled.node("emb").status == "wiring_failed"
    assert compiled.node("emb").cause is make_problem


def test_a_stale_key_on_the_placement_is_reported_on_the_placement():
    compiled = _compiled([placed("emb", "inner-graph", **{"nope.value": Static(1)})])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc[0], make_problem.loc[1]) == ("stale_binding", True, "emb", "nope.value")


# --- the row a graph node runs on -----------------------------------------------


def test_a_series_entering_through_a_scalar_field_makes_the_placement_iterate():
    compiled = _compiled([
        placed("docs", "docs"),
        placed("emb", "inner-graph", value=From("docs.result")),
    ])

    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["emb/holder"].runs_per_row_of == "docs.result"
    assert compiled.run_plan.nodes["emb/up"].runs_per_row_of == "docs.result"


def test_an_inner_reduction_over_the_entering_series_is_a_fold_of_one():
    """The join test: standalone the graph joins one text; embedded and fed
    three, it joins one text three times — never all three once."""
    compiled = _compiled([
        placed("docs", "docs"),
        placed("emb", "inner-graph", value=From("docs.result")),
    ])

    assert compiled.run_plan.nodes["emb/join"].runs_per_row_of == "docs.result"
    assert compiled.field(Ref("emb/join", "result")).dtype is Txt and compiled.field(Ref("emb/join", "result")).many


def test_a_series_made_inside_reduces_to_the_outer_row_by_lineage_alone():
    """An inner unfold running per row makes rows of a child of the outer index; an inner
    reduction over it collapses to the outer row without any scope rule.
    ``Index.__eq__`` reads the id alone, so the parent is asserted by name."""
    inner = embedded_graph_node("splitter", (
            placed("holder", "holder", defaults_={"value": "a\nb"}),
            placed("lines", "lines", text=From("holder.result")),
            placed("join", "join", texts=From("lines.result")),
        ), _registry(lines))
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="docs", node="docs"),
            NodeRecord(id="emb", node="splitter"),
        ], edges=[("docs.result", "emb.value")]),
        _registry(inner, lines),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["emb/lines"].makes_rows == (Ref("emb/lines", "result"),)
    assert compiled.run_plan.nodes["emb/lines"].runs_per_row_of == "docs.result"
    assert compiled.run_plan.nodes["emb/join"].runs_per_row_of == "docs.result"

    # The same, when the series is made off an inner *static* the entering
    # series never touches: the block runs per row, so the new index is a child of
    # the entering index all the same — never a root beside the outer rows.
    unfed = embedded_graph_node("splitter-unfed", (
            placed("holder", "holder", defaults_={"value": "a\nb"}),
            placed("entered", "holder"),
            placed("lines", "lines", text=From("holder.result")),
            placed("join", "join", texts=From("lines.result")),
        ), _registry(lines))
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="docs", node="docs"),
            NodeRecord(id="emb", node="splitter-unfed"),
        ], edges=[("docs.result", "emb.holder.value")]),
        _registry(unfed, lines),
    )
    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["emb/lines"].makes_rows == (Ref("emb/lines", "result"),)
    # The nodes the entering series never reaches still run once per outer row.
    assert compiled.run_plan.nodes["emb/holder"].runs_per_row_of == "docs.result"
    assert compiled.run_plan.nodes["emb/lines"].runs_per_row_of == "docs.result"
    assert compiled.run_plan.nodes["emb/join"].runs_per_row_of == "docs.result"


def test_two_crossings_on_one_lineage_make_the_whole_block_iterate_on_the_deeper():
    """`docs` → `lines` (a child of `docs`); the block takes one field from
    each. The block folds over the deeper index, and so does every node
    inside it — the one fed from the shallower index broadcasts down;
    it does not run per `docs` row while its neighbours run per line."""
    inner = embedded_graph_node("pair", (
            placed("a", "holder"),
            placed("b", "holder"),
            placed("ua", "upper", text=From("a.result")),
        ), _registry())
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="docs", node="docs"),
            NodeRecord(id="lines", node="lines"),
            NodeRecord(id="emb", node="pair"),
        ], edges=[("docs.result", "lines.text"), ("docs.result", "emb.a.value"), ("lines.result", "emb.b.value")]),
        _registry(inner, lines),
    )

    assert compiled.is_runnable, compiled.problems
    per_line = "lines.result"
    assert compiled.run_plan.nodes["emb/a"].runs_per_row_of == per_line
    assert compiled.run_plan.nodes["emb/ua"].runs_per_row_of == per_line
    assert compiled.run_plan.nodes["emb/b"].runs_per_row_of == per_line


def test_a_series_entering_a_series_field_is_read_whole_and_the_block_lays_out_flat():
    """The graph declares it takes a series, so a series is one value to it:
    no scalar crossing, no scope, one reduction over the whole pile."""
    inner = embedded_graph_node("joiner", (placed("join", "join"),), _registry())
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="docs", node="docs"),
            NodeRecord(id="emb", node="joiner"),
        ], edges=[("docs.result", "emb.texts")]),
        _registry(inner),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.run_plan.nodes["emb/join"].runs_per_row_of is None
    assert compiled.field(Ref("emb/join", "result")).dtype is Txt


def test_two_unrelated_series_entering_one_placement_are_its_misaligned():
    inner = embedded_graph_node("pair", (
            placed("a", "holder"),
            placed("b", "holder"),
        ), _registry())
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="d1", node="docs"),
            NodeRecord(id="d2", node="docs"),
            NodeRecord(id="emb", node="pair"),
        ], edges=[("d1.result", "emb.a.value"), ("d2.result", "emb.b.value")]),
        _registry(inner),
    )

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("misaligned", "emb")]
    (misaligned,) = compiled.problems
    graph_node = compiled.node("emb")
    assert graph_node.status == "wiring_failed"
    assert graph_node.problems == compiled.problems
    assert graph_node._fields is None
    raised = graph_node.cause
    assert raised == misaligned
    assert "emb" not in compiled.decisions


def test_a_nested_embedded_graph_is_addressed_by_its_path():
    outer = embedded_graph_node("outer-graph", (
            placed("pre", "holder", defaults_={"value": "x"}),
            placed("inner", "inner-graph", value=From("pre.result")),
        ), _registry(_inner_definition()))
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="top", node="outer-graph"),
            NodeRecord(id="after", node="upper"),
        ], edges=[("top.result", "after.text")]),
        _registry(_inner_definition(), outer),
    )

    assert compiled.is_runnable, compiled.problems
    assert compiled.execution_order == ("top/pre", "top/inner/holder", "top/inner/up", "top/inner/join", "after")
    assert compiled.node("top/inner/up").embedded_in == "top/inner"
    assert compiled.node("top/pre").embedded_in == "top"
    assert compiled.node("after").graph_node.bindings.get("text") == From("top/inner/join.result")
    assert compiled.field(Ref("top/inner/join", "result")).dtype is Txt



def test_a_broken_graph_inside_a_graph_breaks_the_graph_around_it():
    """An unknown inner node leaves its graph unembeddable, and so the graph
    that embeds it too. Each level shows one ``embedded_graph_broken`` where
    it is embedded; the unknown node is the innermost graph's own problem.
    Each is one node where it is embedded, read inside through its version."""
    broken = embedded_graph_node("broken-graph", (placed("x", "nothing"), placed("up", "upper")), _registry())
    wrapper = embedded_graph_node("wrapper-graph", (placed("mid", "broken-graph"),), _registry(broken))
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="top", node="wrapper-graph"),
        ]), _registry(broken, wrapper),
    )

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("embedded_graph_broken", "top")]
    top = compiled.node("top")
    assert [(p.type, p.loc[0]) for p in top.embedded_graph.problems] == [("embedded_graph_broken", "mid")]
    assert [(p.type, p.loc[0]) for p in top.embedded_graph.node("mid").embedded_graph.problems] == [("unknown_node_type", "x")]
    for graph_node in (top, top.embedded_graph.node("mid")):
        assert (graph_node.status, graph_node.embedded_graph is not None) == ("wiring_failed", True)
        assert graph_node.interface is not None  # the box and its handles still draw
        assert graph_node._fields is None
        raised = graph_node.cause
        assert raised.type == "embedded_graph_broken"


# --- what compile refuses about an embedded graph ---------------------------------------


def test_an_authored_id_with_a_slash_is_refused_and_never_collides_with_an_inner_node():
    """``/`` names the nodes of an embedded graph, so an authored ``e/holder``
    beside a graph node ``e`` is refused on its own: it neither appears twice
    in the order nor is read by the inner nodes."""
    compiled = _compiled([
        placed("e/holder", "holder", value=Static("impostor")),
        placed("e", "inner-graph"),
    ])

    assert [(p.type, p.fatal, p.loc[0]) for p in compiled.problems] == [("invalid_node_id", True, "e/holder")]
    assert compiled.execution_order.count("e/holder") == 1
    assert compiled.node("e/holder").graph_node.bindings.get("value") == Static("inner")


def test_a_refused_authored_id_is_not_the_problem_of_the_inner_node_sharing_it():
    """`e` can be embedded, so its `holder` runs as `e/holder`; the author also
    wrote `e/holder`, which is refused. The refusal is the author's: it stays
    in the graph's problems and never shows on the inner node."""
    compiled = _compiled([
        placed("e", "inner-graph"),
        placed("e/holder", "upper"),
    ])

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("invalid_node_id", "e/holder")]
    inner = compiled.node("e/holder")
    assert (inner.status, inner.embedded_in, inner.problems) == ("ready", "e", ())


def test_an_authored_id_holding_a_slash_is_refused_and_stays_the_authors():
    """The author wrote `e/holder` beside an embedded graph `e` whose own
    `holder` has a type the registry lacks. The authored id is refused; the
    graph cannot be embedded, which is one problem on `e`, and
    `node("e/holder")` is the author's node, refused."""
    lost = embedded_graph_node("lost", (placed("holder", "nope"),), _registry())
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="e", node="lost"),
        NodeRecord(id="e/holder", node="upper"),
    ]), _registry(lost))

    assert [(p.type, p.loc) for p in compiled.problems] == [
        ("invalid_node_id", ("e/holder",)), ("embedded_graph_broken", ("e",)),
    ]
    assert [(p.type, p.loc[0]) for p in compiled.node("e").embedded_graph.problems] == [("unknown_node_type", "holder")]
    assert compiled.node("e").status == "wiring_failed"
    raised = compiled.node("e/holder")
    assert (raised.status, raised.embedded_in, raised.cause.type) == ("resolution_failed", None, "invalid_node_id")


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

    inner = embedded_graph_node("pairs", (placed("p", "pair"),), _registry(pair))
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="d1", node="docs"),
            NodeRecord(id="d2", node="docs"),
            NodeRecord(id="emb", node="pairs"),
        ], edges=[("d1.result", "emb.a"), ("d2.result", "emb.b")]),
        _registry(inner, pair),
    )

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.loc[0]) == ("misaligned", "emb")
    assert make_problem.ctx == {"a": "emb.a", "b": "emb.b"}
    assert "emb.p" not in make_problem.msg and "emb.p" not in str(make_problem.ctx)


def test_an_edge_into_a_field_fed_inside_the_graph_is_a_stale_binding():
    """``up.text`` is fed by ``holder`` inside, so the graph does not offer
    it: an edge from outside reaches nothing, as a field any node lacks."""
    compiled = _compiled([
        placed("d1", "docs"),
        placed("d2", "docs"),
        placed("emb", "inner-graph", text=From("d1.result", "d2.result")),
    ])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.loc[0], make_problem.loc[1]) == ("stale_binding", "emb", "text")


