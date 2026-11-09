"""What is wrong with a graph, as data."""

from typing import Annotated

import pytest
from conductor import (
    ArrivingType,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Output,
    Param,
    Problem,
    Ref,
    Result,
    Series,
    Static,
    node,
)
from conductor.widgets import Textarea
from pydantic import ValidationError


def test_a_problem_names_a_code_a_message_and_a_place():
    p = Problem(type="unknown_input", msg="The node has no such field.", loc=("letter", "template"))

    assert p.type == "unknown_input"
    assert p.loc[0] == "letter"
    assert p.loc[1] == "template"


def test_a_problem_about_a_whole_node_names_no_field():
    p = Problem(type="cycle", msg="The graph loops back on itself.", loc=("letter",))

    assert p.loc[0] == "letter"
    assert p.loc[1:] == ()


def test_a_problem_is_frozen():
    p = Problem(type="c", msg="m", loc=("n",))

    with pytest.raises(Exception):
        p.type = "other"


def test_problems_compare_by_value():
    a = Problem(type="c", msg="m", loc=("n",))
    b = Problem(type="c", msg="m", loc=("n",))

    assert a == b


def test_details_carry_what_the_message_names():
    """A host that translates by code needs the values, not the sentence."""
    p = Problem(type="unknown_ref_node", msg="Field 'text' is connected to 'a', which is not in the graph.", loc=("b", "text"), ctx={"source_node": "a"})

    assert p.ctx == {"source_node": "a"}
    assert Problem(type="c", msg="m", loc=("n",)).ctx == {}


class Txt(DType, str):
    id = "problems-test-txt"
    display_title = "Text"


class Nonempty(DType, str):
    """A type with a constructor rule: a value is one only when it has text."""

    id = "problems-test-nonempty"
    display_title = "Non-empty text"

    def __new__(cls, value=""):
        if not value:
            raise ValueError("The field must be filled in.")
        return super().__new__(cls, value)


Out = Annotated[Txt, Result(title="Result")]


@node("echo", title="Echo", description="d", category="test")
def echo(
    x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt(""),
    y: Annotated[Txt, Param(title="Y", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(x + y)


@node("script", title="Python", description="d", category="test")
def script(code: Annotated[Txt, Param(title="Code", connectable=False, widget=Textarea())] = Txt("")) -> Out:
    return code


@node("needs", title="Needs", description="d", category="test")
def needs(x: Annotated[Txt, Param(title="X", widget=Textarea())]) -> Out:
    return x


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("\n".join(texts))


@node("picky", title="Picky", description="d", category="test")
def picky(x: Annotated[Nonempty, Param(title="X", widget=Textarea())] = Nonempty("x")) -> Out:
    """Its constraint is its type's: `x` must be a `Nonempty`."""
    return Txt(x)


def twice_outputs():
    return (
        Output("x", Txt, title="X"),
        Output("x", Txt, title="X again"),
        Output("value", Txt, title="Value"),
    )

@node("twice", title="Twice", description="d", category="test", compute_outputs=twice_outputs)
def twice(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
    """An interface naming one field twice: one name on two outputs, and one on both sides."""
    return value


def _registry():
    return NodeRegistry([echo, script, needs, join_node, picky, twice])


def _problems(nodes, **kw):
    return CompiledGraph.from_graph(Graph(nodes=nodes, **kw), _registry()).problems


def _codes(nodes, **kw):
    return [p.type for p in _problems(nodes, **kw)]


# --- a value's own rule is its type's ---------------------------------------------


def test_a_values_own_rule_is_its_types_constructor_and_reaches_compile_anchored():
    """What a value must satisfy to *be* one is the type's rule, and a
    static that breaks it is the statics pass's `invalid_static` — on the
    field, on the node. No hook is asked, and no node is told where it
    sits in the graph."""
    (make_problem,) = _problems([GraphNode(id="p", type="picky", version=1, bindings={"x": Static("")})])

    assert (make_problem.type, make_problem.fatal, make_problem.loc[0], make_problem.loc[1]) == ("invalid_static", True, "p", "x")


def test_the_problem_carries_what_the_type_itself_said():
    """The rule is only worth having if the constructor's sentence *arrives*: the
    type is where a value's rule is written, so the type's own words are the
    only ones that say what to change. The generic sentence leads, because it
    names the field; the type's follows it."""
    (make_problem,) = _problems([GraphNode(id="p", type="picky", version=1, bindings={"x": Static("")})])

    assert make_problem.msg.endswith("The field must be filled in.") and "'x'" in make_problem.msg
    assert make_problem.ctx["reason"] == "The field must be filled in."


def test_a_type_that_said_nothing_of_its_own_leaves_the_generic_sentence_alone():
    """Pydantic's own report is English and about a JSON shape. A `Series`
    input handed a scalar is the same: nothing a person should read."""
    (make_problem,) = _problems([GraphNode(id="j", type="join", version=1, bindings={"texts": Static(3)})])

    assert "'texts'" in make_problem.msg
    assert "reason" not in make_problem.ctx


def test_a_value_that_satisfies_its_type_contributes_nothing():
    assert _codes([GraphNode(id="p", type="picky", version=1, bindings={"x": Static("hi")})]) == []


# --- a field name is unique within a node -------------------------------------------


def test_a_computed_inputs_naming_one_field_twice_is_fatal():
    """`Interface.of` refuses the declaration; an interface a hook computed is
    compile's to refuse, once, on the finished interface — within a side and
    across sides alike, because a `Ref` is one address on either side."""
    problems = _problems([GraphNode(id="t", type="twice", version=1)])

    assert [(p.type, p.fatal, p.loc[0], p.loc[1]) for p in problems] == [
        ("duplicate_field_name", True, "t", "x"),
        ("duplicate_field_name", True, "t", "value"),
    ]


def test_a_hook_writing_an_output_of_no_dtype_is_a_bug_in_the_hook():
    """A record's ``dtype`` is a ``DType``, so the record refuses anything else where the hook writes it."""

    def odd_outputs():
        return (Output("raw", dict, title="Raw"))

    @node("odd", title="Odd", description="d", category="test", compute_outputs=odd_outputs)
    def odd(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
        return value

    registry = _registry().with_nodes([odd])
    with pytest.raises(ValidationError, match="dtype"):
        CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="o", type="odd", version=1)]), registry)


# --- the stored bindings, validated -----------------------------------------------


def test_a_clean_graph_reports_no_problems():
    assert _codes([
        GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")}),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
    ]) == []


def test_a_binding_on_an_input_the_placement_does_not_have_is_stale():
    (make_problem,) = _problems([GraphNode(id="a", type="echo", version=1, bindings={"z": Static(1)})])

    assert (make_problem.type, make_problem.fatal, make_problem.loc[0], make_problem.loc[1]) == ("stale_binding", True, "a", "z")


def test_an_edge_into_a_closed_handle_is_fatal():
    """`connectable=False` is the one wireability question, and the
    integrity it protects is structural — nothing upstream reaches `code`."""
    (make_problem,) = _problems([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="s", type="script", version=1, bindings={"code": From("a.result")}),
    ])

    assert (make_problem.type, make_problem.fatal, make_problem.loc[1]) == ("edge_into_unconnectable_input", True, "code")


def test_two_scalar_refs_into_a_scalar_input_is_fatal():
    """Several refs into a scalar input are a union, and a union is on an index:
    two scalars have none to share. Reported by the edges pass, where shapes are known."""
    codes = _codes([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1),
        GraphNode(id="c", type="echo", version=1, bindings={"x": From("a.result", "b.result")}),
    ])

    assert codes == ["union_needs_one_index"]


def test_two_refs_into_a_series_input_is_a_gather_not_a_problem():
    assert _codes([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From("a.result", "b.result")}),
    ]) == []


def test_a_ref_to_a_node_not_in_the_graph_is_fatal():
    (make_problem,) = _problems([GraphNode(id="b", type="echo", version=1, bindings={"x": From("ghost.result")})])

    assert (make_problem.type, make_problem.loc[0], make_problem.loc[1]) == ("unknown_ref_node", "b", "x")
    assert make_problem.ctx == {"source_node": "ghost"}


def test_a_ref_to_an_output_the_node_does_not_have_is_fatal():
    (make_problem,) = _problems([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.nope")}),
    ])

    assert (make_problem.type, make_problem.loc[0], make_problem.loc[1]) == ("unknown_ref_output", "b", "x")
    assert make_problem.ctx == {"source": "a.nope", "outputs": "result"}


def test_a_required_input_with_nothing_bound_is_fatal():
    """A graph is closed. The caller's answer replaces a value the
    author set; it does not fill a hole."""
    (make_problem,) = _problems([GraphNode(id="n", type="needs", version=1)])

    assert (make_problem.type, make_problem.fatal, make_problem.loc[1]) == ("unbound_required", True, "x")
    assert make_problem.msg == "Nothing is connected to the field."


def test_an_any_input_with_no_edge_is_unbound_required():
    """The field's type comes from the edge, so with no edge the field is
    simply unbound — one problem, one code; there is no second code for
    an untyped field."""
    from typing import Any

    def route_outputs(value: ArrivingType):
        dtype = value
        return (Output("result", dtype, title="R"),)

    @node("route-p", title="Pass on", description="d", category="test", compute_outputs=route_outputs)
    def route(value: Annotated[Any, Param(title="Value")]) -> Annotated[Any, Result(title="R")]:
        return value

    registry = _registry().with_nodes([route])
    problems = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="r", type="route-p", version=1)]), registry).problems

    assert [(p.type, p.fatal, p.loc[1]) for p in problems] == [("unbound_required", True, "value")]
    assert problems[0].msg == "Nothing is connected to the field."


def test_a_required_input_with_a_static_is_fine():
    assert _codes([GraphNode(id="n", type="needs", version=1, bindings={"x": Static("hi")})]) == []


def test_a_cycle_is_a_fatal_problem_on_each_node_in_it():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": From("b.result")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
            GraphNode(id="c", type="echo", version=1),
        ]),
        _registry(),
    )

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("cycle", "a"), ("cycle", "b")]
    assert compiled.execution_order == ("c",)
    assert not compiled.is_runnable


def test_problems_can_be_read_whole_or_by_node():
    """Every problem here is about a field the node does not have — a lock
    and two bindings naming nothing — so ``compiled.problems`` is where they
    are read, filtered by node, and no field view exists to ask."""
    compiled = CompiledGraph.from_graph(
        Graph(
            nodes=[GraphNode(id="a", type="echo", version=1, locked=("ghost",), bindings={"z": Static(1), "w": Static(2)})],
        ),
        _registry(),
    )

    assert len(compiled.problems) == 3
    assert [p.type for p in compiled.problems if p.loc[0] == "a" and p.loc[1] == "ghost"] == ["unknown_locked_field"]
    assert [p.type for p in compiled.problems if p.loc[0] == "a" and p.loc[1] == "z"] == ["stale_binding"]
    with pytest.raises(KeyError):
        compiled.field(Ref("a", "z"))


def test_a_non_fatal_problem_leaves_the_graph_runnable():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="a", type="echo", version=1, locked=("ghost",))]), _registry()
    )

    assert compiled.is_runnable


# --- the codes compile emits are declared once ------------------------------------


def test_every_code_is_declared_once_in_the_catalogue():
    """``PROBLEM_CODES`` is what a host translating by code has to cover: the keys of
    the one table every emitting site reads. A site naming a code the table
    lacks raises where it is written, so nothing can be emitted undeclared."""
    from conductor import PROBLEM_CODES
    from conductor._internal.errors import PROBLEM_CATALOGUE, make_problem

    assert PROBLEM_CODES == frozenset(PROBLEM_CATALOGUE)
    with pytest.raises(KeyError):
        make_problem("no_such_code", "n")


def test_a_slot_the_details_do_not_fill_raises_unless_it_is_optional():
    """A template typo or a forgotten keyword is a programming error, caught
    where the problem is built; only the type's own sentence may be absent."""
    from conductor._internal.errors import make_problem

    with pytest.raises(KeyError):
        make_problem("unknown_ref_node", "b", "text")  # no source_node
    assert make_problem("invalid_static", "b", "text", input="1", expected="text").msg == "The value 1 in 'text' cannot be read as text."


def test_a_problem_is_formatted_from_its_details_and_keeps_them():
    from conductor._internal.errors import make_problem

    p = make_problem("unknown_ref_node", "b", "text", source_node="a")
    assert p.msg == "Field 'text' is connected to 'a', which is not in the graph."
    assert p.ctx == {"source_node": "a"}
    assert p.fatal is True
    assert make_problem("unknown_locked_field", "b", "old").fatal is False


# --- what compile refuses ----------------------------------------------------------------


def test_a_slash_in_an_authored_id_is_fatal():
    """``/`` is how compile names the nodes of an embedded graph."""
    (make_problem,) = _problems([GraphNode(id="a/b", type="echo", version=1)])

    assert (make_problem.type, make_problem.fatal, make_problem.loc[0]) == ("invalid_node_id", True, "a/b")


def test_an_edge_with_no_refs_is_refused_where_it_is_written():
    """``From()`` is not a binding: the model refuses it before compile sees it."""

    with pytest.raises(ValidationError):
        From()


def test_a_cycle_is_reported_on_its_members_only():
    """A node downstream of a cycle is not part of it. It keeps its
    interface, so an editor draws it, and the cycle alone stops the run."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": From("b.result")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
            GraphNode(id="u", type="echo", version=1, bindings={"x": From("a.result")}),
            GraphNode(id="v", type="echo", version=1, bindings={"x": From("u.result")}),
        ]),
        _registry(),
    )

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("cycle", "a"), ("cycle", "b")]
    assert compiled.execution_order == ("u", "v")
    assert [i.name for i in compiled.node("u").interface.inputs] == ["x", "y"]
    assert not compiled.is_runnable


def test_an_edge_from_a_node_that_failed_to_resolve_is_not_reported_again():
    """The source carries ``unknown_node_type``; the reader is silent
    rather than told the node 'is not in the graph', which it is."""
    problems = _problems([
        GraphNode(id="x", type="no-such", version=1),
        GraphNode(id="u", type="echo", version=1, bindings={"x": From("x.result")}),
    ])

    assert [(p.type, p.loc[0]) for p in problems] == [("unknown_node_type", "x")]


def test_an_edge_into_a_field_an_embedded_graph_lacks_names_the_authors_address():
    """``e.nope`` is not an output of ``e``, and ``e.ghost.out`` is the address
    ``e/ghost.out``, which nothing inside ``e`` has."""
    from test_core.embedded import embedded_graph_node

    registry = _registry()
    registry = registry.with_nodes([embedded_graph_node(
        "emb-graph", (GraphNode(id="inner", type="echo", version=1),), registry,
    )])
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="e", type="emb-graph", version=1),
            GraphNode(id="u", type="echo", version=1, bindings={"x": From("e.nope")}),
            GraphNode(id="w", type="echo", version=1, bindings={"x": From("e.ghost.out")}),
        ]),
        registry,
    )

    assert [(p.type, p.loc[0], p.loc[1], p.ctx) for p in compiled.problems] == [
        ("unknown_ref_output", "u", "x", {"source": "e.nope", "outputs": "result"}),
        ("unknown_ref_output", "w", "x", {"source": "e.ghost.out", "outputs": "result"}),
    ]


def test_an_edge_from_a_slashed_name_under_a_plain_node_is_not_in_the_graph():
    """The other way: ``a/b`` collapses to ``a`` as an author's address,
    but ``a`` is an ordinary node, not an embedded graph, so the source is
    simply not in the graph — and the graph is not runnable with a node the
    walk never derived."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="echo", version=1),
            GraphNode(id="u", type="echo", version=1, bindings={"x": From("a/b.result")}),
        ]),
        _registry(),
    )

    assert [(p.type, p.loc[0], p.ctx) for p in compiled.problems] == [("unknown_ref_node", "u", {"source_node": "a/b"})]
    assert not compiled.is_runnable


def test_an_edge_from_a_refused_id_is_not_reported_again():
    """An id with a ``/`` carries ``invalid_node_id``; its reader is silent."""
    problems = _problems([
        GraphNode(id="x/y", type="echo", version=1),
        GraphNode(id="u", type="echo", version=1, bindings={"x": From("x/y.result")}),
    ])

    assert [(p.type, p.loc[0]) for p in problems] == [("invalid_node_id", "x/y")]


def test_an_edge_from_a_refused_id_beside_an_embedded_graph_does_not_read_inside_it():
    """``e/inner`` is a refused id, not the node ``inner`` of the graph ``e``:
    its reader stays broken and silent, and never reads the embedded graph."""
    from test_core.embedded import embedded_graph_node

    registry = _registry()
    registry = registry.with_nodes([embedded_graph_node(
        "emb-graph", (GraphNode(id="inner", type="echo", version=1),), registry,
    )])
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="e", type="emb-graph", version=1),
            GraphNode(id="e/inner", type="echo", version=1),
            GraphNode(id="u", type="echo", version=1, bindings={"x": From("e/inner.result")}),
        ]),
        registry,
    )

    assert [(p.type, p.loc[0]) for p in compiled.problems] == [("invalid_node_id", "e/inner")]
    assert compiled.node("u").state == "wiring_failed"
