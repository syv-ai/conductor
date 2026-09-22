"""What is wrong with a graph, as data."""

from typing import Annotated

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.graph.problem import Problem
from conductor.metadata import Output, Param, Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Series
from conductor.widgets import Textarea
from pydantic import ValidationError


def test_a_problem_names_a_code_a_message_and_a_place():
    p = Problem(
        code="unknown_input",
        message="The node has no such field.",
        fatal=True,
        node_id="letter",
        field="template",
    )

    assert p.code == "unknown_input"
    assert p.node_id == "letter"
    assert p.field == "template"


def test_a_problem_about_a_whole_node_names_no_field():
    p = Problem(code="cycle", message="The graph loops back on itself.", fatal=True, node_id="letter")

    assert p.node_id == "letter"
    assert p.field is None


def test_every_problem_is_about_a_node():
    """There is no graph-level problem: a graph is only ever wrong
    somewhere, so the anchor has two states and not three."""
    with pytest.raises(ValidationError, match="node_id"):
        Problem(code="empty", message="The graph is empty.", fatal=True)

def test_fatal_is_a_boolean_not_a_two_valued_enum():
    """One fact, once. There are exactly two audiences: the editor
    shows everything, a run stops at the first fatal one."""
    assert Problem(code="c", message="m", fatal=True, node_id="n").fatal is True
    assert Problem(code="c", message="m", fatal=False, node_id="n").fatal is False


def test_a_problem_is_frozen():
    p = Problem(code="c", message="m", fatal=True, node_id="n")

    with pytest.raises(Exception):
        p.code = "other"


def test_problems_compare_by_value():
    a = Problem(code="c", message="m", fatal=True, node_id="n")
    b = Problem(code="c", message="m", fatal=True, node_id="n")

    assert a == b


def test_details_carry_what_the_message_names():
    """A host that translates by code needs the values, not the sentence."""
    p = Problem(code="unknown_ref_node", message="Field 'text' is connected to 'a', which is not in the graph.",
                fatal=True, node_id="b", field="text", details={"source_node": "a"})

    assert p.details == {"source_node": "a"}
    assert Problem(code="c", message="m", fatal=True, node_id="n").details == {}


class Txt(DType, str):
    id = "problems-test-txt"
    title = "Text"


class Nonempty(DType, str):
    """A type with a constructor rule: a value is one only when it has text."""

    id = "problems-test-nonempty"
    title = "Non-empty text"

    def __new__(cls, value=""):
        if not value:
            raise ValueError("The field must be filled in.")
        return super().__new__(cls, value)


Out = Annotated[Txt, Result(title="Result")]


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "d"
    category = "test"

    def run(
        self,
        x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt(""),
        y: Annotated[Txt, Param(title="Y", widget=Textarea())] = Txt(""),
    ) -> Out:
        return Txt(x + y)


class Script(NodeDefinition):
    id = "script"
    title = "Python"
    description = "d"
    category = "test"

    def run(self, code: Annotated[Txt, Param(title="Code", show_handle=False, widget=Textarea())] = Txt("")) -> Out:
        return code


class Needs(NodeDefinition):
    id = "needs"
    title = "Needs"
    description = "d"
    category = "test"

    def run(self, x: Annotated[Txt, Param(title="X", widget=Textarea())]) -> Out:
        return x


class Join(NodeDefinition):
    id = "join"
    title = "Join"
    description = "d"
    category = "test"

    def run(self, texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
        return Txt("\n".join(texts))


class Picky(NodeDefinition):
    """Its constraint is its type's: `x` must be a `Nonempty`."""

    id = "picky"
    title = "Picky"
    description = "d"
    category = "test"

    def run(self, x: Annotated[Nonempty, Param(title="X", widget=Textarea())] = Nonempty("x")) -> Out:
        return Txt(x)


class Twice(NodeDefinition):
    """An interface naming one field twice: one name on two outputs, and one on both sides."""

    id = "twice"
    title = "Twice"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
        return value

    def compute_outputs(self, declared, values, arriving):
        return (
            Output(name="x", dtype=Txt, title="X"),
            Output(name="x", dtype=Txt, title="X again"),
            Output(name="value", dtype=Txt, title="Value"),
        )


def _registry():
    registry = NodeRegistry()
    for node_cls in (Echo, Script, Needs, Join, Picky, Twice):
        registry.register(node_cls)
    return registry


def _problems(nodes, **kw):
    return CompiledGraph.from_graph(Graph(nodes=nodes, **kw), _registry()).problems


def _codes(nodes, **kw):
    return [p.code for p in _problems(nodes, **kw)]


# --- a value's own rule is its type's ---------------------------------------------


def test_a_values_own_rule_is_its_types_constructor_and_reaches_compile_anchored():
    """What a value must satisfy to *be* one is the type's rule, and a
    static that breaks it is the statics pass's `invalid_static` — on the
    field, on the node. No hook is asked, and no node is told where it was
    placed."""
    (problem,) = _problems([GraphNode(id="p", type="picky", version=1, bindings={"x": Static("")})])

    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("invalid_static", True, "p", "x")


def test_the_problem_carries_what_the_type_itself_said():
    """The rule is only worth having if the constructor's sentence *arrives*: the
    type is where a value's rule is written, so the type's own words are the
    only ones that say what to change. The generic sentence leads, because it
    names the field; the type's follows it."""
    (problem,) = _problems([GraphNode(id="p", type="picky", version=1, bindings={"x": Static("")})])

    assert problem.message == "The value in 'x' cannot be read as the field's type. The field must be filled in."
    assert problem.details == {"reason": "The field must be filled in."}


def test_a_type_that_said_nothing_of_its_own_leaves_the_generic_sentence_alone():
    """Pydantic's own report is English and about a JSON shape. A `Series`
    input handed a scalar is the same: nothing a person should read."""
    (problem,) = _problems([GraphNode(id="j", type="join", version=1, bindings={"texts": Static(3)})])

    assert problem.message == "The value in 'texts' cannot be read as the field's type."
    assert problem.details == {}


def test_a_value_that_satisfies_its_type_contributes_nothing():
    assert _codes([GraphNode(id="p", type="picky", version=1, bindings={"x": Static("hi")})]) == []


# --- a field name is unique within a node -------------------------------------------


def test_a_computed_roster_naming_one_field_twice_is_fatal():
    """`Interface.of` refuses the declaration; an interface a hook computed is
    compile's to refuse, once, on the finished interface — within a side and
    across sides alike, because a `Ref` is one address on either side."""
    problems = _problems([GraphNode(id="t", type="twice", version=1)])

    assert [(p.code, p.fatal, p.node_id, p.field) for p in problems] == [
        ("duplicate_field_name", True, "t", "x"),
        ("duplicate_field_name", True, "t", "value"),
    ]


def test_a_computed_field_with_a_handle_needs_an_edge_type():
    """A DType is required exactly where a handle is. `Interface.of`
    refuses the declaration; a hook can compute an output nothing could
    edge, and compile refuses that once, on the completed interface."""

    class Odd(NodeDefinition):
        id = "odd"
        title = "Odd"
        description = "d"
        category = "test"

        def run(self, value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
            return value

        def compute_outputs(self, declared, values, arriving):
            return (*declared, Output(name="raw", dtype=dict, title="Raw"))

    registry = _registry()
    registry.register(Odd)
    problems = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="o", type="odd", version=1)]), registry).problems

    assert [(p.code, p.fatal, p.field) for p in problems] == [("handle_needs_dtype", True, "raw")]


# --- the stored bindings, validated -----------------------------------------------


def test_a_clean_graph_reports_no_problems():
    assert _codes([
        GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")}),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From(Ref("a", "result"))}),
    ]) == []


def test_a_binding_on_an_input_the_placement_does_not_have_is_stale():
    (problem,) = _problems([GraphNode(id="a", type="echo", version=1, bindings={"z": Static(1)})])

    assert (problem.code, problem.fatal, problem.node_id, problem.field) == ("stale_binding", True, "a", "z")


def test_an_edge_into_a_closed_handle_is_fatal():
    """`show_handle=False` is the one wireability question, and the
    integrity it protects is structural — nothing upstream reaches `code`."""
    (problem,) = _problems([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="s", type="script", version=1, bindings={"code": From(Ref("a", "result"))}),
    ])

    assert (problem.code, problem.fatal, problem.field) == ("edge_into_closed_handle", True, "code")


def test_two_scalar_refs_into_a_scalar_input_is_fatal():
    """Several refs into a scalar input are a union, and a union is on an index:
    two scalars have none to share. Reported by the edges pass, where shapes are known."""
    codes = _codes([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1),
        GraphNode(id="c", type="echo", version=1, bindings={"x": From(Ref("a", "result"), Ref("b", "result"))}),
    ])

    assert codes == ["union_needs_one_index"]


def test_two_refs_into_a_series_input_is_a_gather_not_a_problem():
    assert _codes([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1),
        GraphNode(id="j", type="join", version=1, bindings={"texts": From(Ref("a", "result"), Ref("b", "result"))}),
    ]) == []


def test_a_ref_to_a_node_not_in_the_graph_is_fatal():
    (problem,) = _problems([GraphNode(id="b", type="echo", version=1, bindings={"x": From(Ref("ghost", "result"))})])

    assert (problem.code, problem.node_id, problem.field) == ("unknown_ref_node", "b", "x")
    assert problem.details == {"source_node": "ghost"}


def test_a_ref_to_an_output_the_node_does_not_have_is_fatal():
    (problem,) = _problems([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From(Ref("a", "nope"))}),
    ])

    assert (problem.code, problem.node_id, problem.field) == ("unknown_ref_output", "b", "x")
    assert problem.details == {"source": "a.nope"}


def test_a_required_input_with_nothing_bound_is_fatal():
    """A graph is closed. The caller's answer replaces a value the
    author set; it does not fill a hole."""
    (problem,) = _problems([GraphNode(id="n", type="needs", version=1)])

    assert (problem.code, problem.fatal, problem.field) == ("unbound_required", True, "x")
    assert problem.message == "Nothing is connected to the field."


def test_an_any_input_with_no_edge_is_unbound_required():
    """The field's type comes from the edge, so with no edge the field is
    simply unbound — one problem, one code; there is no second code for
    an untyped field."""
    from typing import Any

    class Route(NodeDefinition):
        id = "route-p"
        title = "Pass on"
        description = "d"
        category = "test"

        def run(self, value: Annotated[Any, Param(title="Value")]) -> Annotated[Any, Result(title="R")]:
            return value

        def compute_outputs(self, declared, values, arriving):
            dtype = arriving.get("value", Any)
            return tuple(out.model_copy(update={"dtype": dtype}) for out in declared)

    registry = _registry()
    registry.register(Route)
    problems = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="r", type="route-p", version=1)]), registry).problems

    assert [(p.code, p.fatal, p.field) for p in problems] == [("unbound_required", True, "value")]
    assert problems[0].message == "Nothing is connected to the field."


def test_a_required_input_with_a_static_is_fine():
    assert _codes([GraphNode(id="n", type="needs", version=1, bindings={"x": Static("hi")})]) == []


def test_a_cycle_is_a_fatal_problem_on_each_node_in_it():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": From(Ref("b", "result"))}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": From(Ref("a", "result"))}),
            GraphNode(id="c", type="echo", version=1),
        ]),
        _registry(),
    )

    assert [(p.code, p.node_id) for p in compiled.problems] == [("cycle", "a"), ("cycle", "b")]
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
    assert [p.code for p in compiled.problems if p.node_id == "a" and p.field == "ghost"] == ["unknown_locked_field"]
    assert [p.code for p in compiled.problems if p.node_id == "a" and p.field == "z"] == ["stale_binding"]
    with pytest.raises(KeyError):
        compiled.field(Ref("a", "z"))


def test_a_non_fatal_problem_leaves_the_graph_runnable():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="a", type="echo", version=1, locked=("ghost",))]), _registry()
    )

    assert compiled.is_runnable


# --- the codes compile emits are declared once ------------------------------------


def test_every_code_is_declared_once_in_the_catalogue():
    """``CODES`` is what a host translating by code has to cover: the keys of
    the one table every emitting site reads. A site naming a code the table
    lacks raises where it is written, so nothing can be emitted undeclared."""
    from conductor.graph.problem import CATALOGUE, CODES, problem

    assert CODES == frozenset(CATALOGUE)
    with pytest.raises(KeyError):
        problem("no_such_code", "n")


def test_a_slot_the_details_do_not_fill_raises_unless_it_is_optional():
    """A template typo or a forgotten keyword is a programming error, caught
    where the problem is built; only the type's own sentence may be absent."""
    from conductor.graph.problem import problem

    with pytest.raises(KeyError):
        problem("unknown_ref_node", "b", "text")  # no source_node
    assert problem("invalid_static", "b", "text").message == "The value in 'text' cannot be read as the field's type."


def test_a_problem_is_formatted_from_its_details_and_keeps_them():
    from conductor.graph.problem import problem

    p = problem("unknown_ref_node", "b", "text", source_node="a")
    assert p.message == "Field 'text' is connected to 'a', which is not in the graph."
    assert p.details == {"source_node": "a"}
    assert p.fatal is True
    assert problem("unknown_locked_field", "b", "old").fatal is False


# --- what compile refuses that it used to let through ----------------------------------


def test_a_slash_in_an_authored_id_is_fatal():
    """C5: ``/`` is how compile names the nodes of an embedded graph."""
    (problem,) = _problems([GraphNode(id="a/b", type="echo", version=1)])

    assert (problem.code, problem.fatal, problem.node_id) == ("invalid_node_id", True, "a/b")


def test_an_edge_with_no_refs_is_refused_where_it_is_written():
    """C7: ``From()`` is not a binding; it used to crash compile."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        From()


def test_a_cycle_is_reported_on_its_members_only():
    """C10: a node downstream of a cycle is not part of it. It keeps its
    interface, so an editor draws it, and the cycle alone stops the run."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": From(Ref("b", "result"))}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": From(Ref("a", "result"))}),
            GraphNode(id="u", type="echo", version=1, bindings={"x": From(Ref("a", "result"))}),
            GraphNode(id="v", type="echo", version=1, bindings={"x": From(Ref("u", "result"))}),
        ]),
        _registry(),
    )

    assert [(p.code, p.node_id) for p in compiled.problems] == [("cycle", "a"), ("cycle", "b")]
    assert compiled.execution_order == ("u", "v")
    assert [i.name for i in compiled.node("u").interface.inputs] == ["x", "y"]
    assert not compiled.is_runnable


def test_an_edge_from_a_node_that_failed_to_resolve_is_not_reported_again():
    """C11: the source carries ``unknown_node_type``; the reader is silent
    rather than told the node 'is not in the graph', which it is."""
    problems = _problems([
        GraphNode(id="x", type="no-such", version=1),
        GraphNode(id="u", type="echo", version=1, bindings={"x": From(Ref("x", "result"))}),
    ])

    assert [(p.code, p.node_id) for p in problems] == [("unknown_node_type", "x")]


def test_an_edge_into_a_field_an_embedded_graph_lacks_names_the_authors_address():
    """C11: ``e.nope`` is not an output of ``e``; the message says so with
    the address the author wrote, and no expanded id leaks."""
    from collections.abc import Mapping
    from typing import ClassVar

    from conductor.interface import Interface
    from conductor.metadata import Input, Output
    from conductor.node import GraphVersion

    class Emb(NodeDefinition):
        id = "emb-graph"
        title = "Embedded"
        description = "d"
        category = "test"
        versions: ClassVar[dict[int, GraphVersion]] = {
            1: GraphVersion(
                graph=(GraphNode(id="inner", type="echo", version=1),),
                interface=Interface(
                    inputs=(Input(name="inner.x", dtype=Txt, title="X", widget=Textarea(), default=Txt(""), optional=True),),
                    outputs=(Output(name="inner.result", dtype=Txt, title="R"),),
                    returns=Mapping,
                ),
            )
        }

    registry = _registry()
    registry.register(Emb)
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="e", type="emb-graph", version=1),
            GraphNode(id="u", type="echo", version=1, bindings={"x": From(Ref("e", "nope"))}),
            GraphNode(id="w", type="echo", version=1, bindings={"x": From(Ref("e", "ghost.out"))}),
        ]),
        registry,
    )

    assert [(p.code, p.node_id, p.field, p.details) for p in compiled.problems] == [
        ("unknown_ref_output", "u", "x", {"source": "e.nope"}),
        ("unknown_ref_output", "w", "x", {"source": "e.ghost.out"}),
    ]
    assert not any("/" in p.message for p in compiled.problems)


def test_an_edge_from_a_slashed_name_under_a_plain_node_is_not_in_the_graph():
    """C11, the other way: ``a/b`` collapses to ``a`` as an author's address,
    but ``a`` is an ordinary node, not an embedded graph, so the source is
    simply not in the graph — and the graph is not runnable with a node the
    walk never derived."""
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="a", type="echo", version=1),
            GraphNode(id="u", type="echo", version=1, bindings={"x": From(Ref("a/b", "result"))}),
        ]),
        _registry(),
    )

    assert [(p.code, p.node_id, p.details) for p in compiled.problems] == [("unknown_ref_node", "u", {"source_node": "a/b"})]
    assert not compiled.is_runnable


def test_an_edge_from_a_refused_id_is_not_reported_again():
    """C11 with C5: the id carries ``invalid_node_id``; its reader is silent."""
    problems = _problems([
        GraphNode(id="x/y", type="echo", version=1),
        GraphNode(id="u", type="echo", version=1, bindings={"x": From(Ref("x/y", "result"))}),
    ])

    assert [(p.code, p.node_id) for p in problems] == [("invalid_node_id", "x/y")]
