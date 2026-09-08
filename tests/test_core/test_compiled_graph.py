"""A compiled flow is asked, not traversed."""

from collections.abc import Mapping
from typing import Annotated

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.compiler import compile_graph
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.graph.problem import Problem
from conductor.interface import Interface, Provided, model_of
from conductor.metadata import Output, Roster
from conductor.node import NodeDefinition, Policy, version
from conductor.ref import Ref
from conductor.returns import Result
from conductor.widgets import Choice, Dropdown, Textarea


class Txt(DType, str):
    id = "compiled-test-txt"
    title = "Text"


class Num(DType, float):
    id = "compiled-test-num"
    title = "Number"


Out = Annotated[Txt, Result(title="Result")]


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "d"
    category = "test"

    def run(
        self,
        x: Annotated[Txt, Textarea(title="X")] = Txt(""),
        y: Annotated[Txt, Textarea(title="Y")] = Txt(""),
    ) -> Out:
        return Txt(x + y)


class TextInput(NodeDefinition):
    id = "text-input"
    title = "Text"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return value


class Modes(NodeDefinition):
    """A roster that depends on what the author typed."""

    id = "modes"
    title = "Modes"
    description = "d"
    category = "test"

    def run(
        self,
        mode: Annotated[Txt, Dropdown(title="State", choices=(Choice(id="a", title="A"), Choice(id="b", title="B")))] = Txt("a"),
        extra: Annotated[Txt, Textarea(title="Extra")] = Txt(""),
    ) -> Out:
        return mode

    def compute_inputs(self, declared, values):
        if values.get("mode") == "b":
            return declared
        return tuple(i for i in declared if i.name != "extra")


class OpenSheet(NodeDefinition):
    """Columns come from the value."""

    id = "open-sheet"
    title = "Open sheet"
    description = "d"
    category = "test"

    def run(self, header: Annotated[Txt, Textarea(title="Header")] = Txt("")) -> Out:
        return header

    def compute_outputs(self, declared, values, arriving):
        return tuple(
            Output(name=col, dtype=Txt, title=col)
            for col in str(values.get("header", "")).split(",")
            if col
        )


class Renamed(NodeDefinition):
    id = "renamed"
    title = "Renamed"
    description = "d"
    category = "test"

    @version(1)
    def run_v1(self, old: Annotated[Txt, Textarea(title="Old")] = Txt("")) -> Out:
        return old

    @version(2, policy=Policy(retries=2))
    def run(self, new: Annotated[Txt, Textarea(title="New")] = Txt("")) -> Out:
        return new


class Clock:
    """Something the run supplies, not the flow (`Provided`)."""


class Stamped(NodeDefinition):
    """A node with a need: the run must provide a `Clock`."""

    id = "stamped"
    title = "Stamped"
    description = "d"
    category = "test"

    def run(
        self,
        clock: Annotated[Clock, Provided()],
        x: Annotated[Txt, Textarea(title="X")] = Txt(""),
    ) -> Out:
        return x


def _registry(*extra):
    registry = NodeRegistry()
    for node_cls in (Echo, TextInput, Modes, OpenSheet, Renamed, *extra):
        registry.register(node_cls)
    return registry


def _compiled(nodes, registry=None, **kw):
    return compile_graph(Graph(nodes=nodes, **kw), registry or _registry())


def _exposed(node_id="besked", value="hej"):
    """One node whose `value` field a flow can name."""
    return GraphNode(
        id=node_id,
        type="text-input",
        version=1,
        title="Message",
        fields={"value": FieldContent(title="Message"), "result": FieldContent(title="Text")},
        bindings={"value": Static(value=value)},
    )


# --- the artifact ---------------------------------------------------------


def test_compile_flow_returns_an_asked_artifact():
    compiled = _compiled([GraphNode(id="a", type="echo", version=1, bindings={"x": Static(value="hi")})])

    assert isinstance(compiled, CompiledGraph)
    assert compiled.execution_order() == ("a",)
    assert compiled.is_runnable
    assert compiled.problems_for() == ()


def test_callers_never_touch_a_binding_table():
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    for traversed in ("bindings", "node_map", "edge_map", "incoming_map", "for_engine"):
        assert not hasattr(compiled, traversed), traversed


def test_execution_order_follows_the_edges():
    compiled = _compiled([
        GraphNode(id="b", type="echo", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),))}),
        GraphNode(id="a", type="echo", version=1),
    ])

    assert compiled.execution_order() == ("a", "b")


# --- value_source ---------------------------------------------------------


def test_value_source_answers_where_an_input_comes_from():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"x": Static(value="hi")}),
        GraphNode(id="b", type="echo", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),))}),
    ])

    assert isinstance(compiled.value_source("a", "x"), Static)
    assert isinstance(compiled.value_source("b", "x"), Edges)


def test_an_unbound_input_reads_as_None_once_compiled():
    """None means "nothing binds it, so the declared default applies" — the
    only "nothing binds this" state there is."""
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    assert compiled.value_source("a", "y") is None


def test_value_source_for_an_input_the_node_does_not_declare_raises():
    """A programming error, not a state of the graph."""
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    with pytest.raises(KeyError, match="nonexistent"):
        compiled.value_source("a", "nonexistent")


# --- the pin is resolved once, here ------------------------------------------


def test_a_placement_reads_the_version_it_pins():
    """The version freezes which behaviour a placement points at.
    Reading the class's current version instead would silently re-point
    every old placement at the newest signature."""
    compiled = _compiled([GraphNode(id="old", type="renamed", version=1)])

    assert [i.name for i in compiled.roster("old").inputs] == ["old"]
    assert compiled.version("old").policy == Policy()
    assert compiled.version("old").run.__name__ == "run_v1"


def test_a_node_type_the_registry_lacks_is_a_fatal_problem():
    """A catalog can drift under a stored graph. That is a graph state, and
    the editor needs to show where."""
    compiled = _compiled([GraphNode(id="a", type="gone", version=1)])

    (problem,) = compiled.problems_for()
    assert (problem.code, problem.fatal, problem.node_id) == ("unknown_node_type", True, "a")
    assert not compiled.is_runnable
    with pytest.raises(KeyError):
        compiled.roster("a")


def test_a_version_the_class_no_longer_declares_is_a_fatal_problem():
    compiled = _compiled([GraphNode(id="a", type="renamed", version=7)])

    (problem,) = compiled.problems_for()
    assert (problem.code, problem.node_id) == ("unknown_node_version", "a")


def test_two_placements_with_one_id_is_a_fatal_problem():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="a", type="echo", version=1),
    ])

    assert [p.code for p in compiled.problems_for()] == ["duplicate_node_id"]


# --- rosters: the hooks are asked, once, on a fresh instance -----------------


def test_a_roster_is_what_the_hooks_answered():
    compiled = _compiled([
        GraphNode(id="m", type="modes", version=1, bindings={"mode": Static(value="a")}),
        GraphNode(id="s", type="open-sheet", version=1, bindings={"header": Static(value="name,email")}),
    ])

    assert isinstance(compiled.roster("m"), Roster)
    assert [i.name for i in compiled.roster("m").inputs] == ["mode"]
    assert [o.name for o in compiled.roster("s").outputs] == ["name", "email"]


def test_a_node_with_no_opinion_has_its_declaration_as_roster():
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    declared = Echo.versions[1].interface
    assert compiled.roster("a") == Roster(inputs=declared.inputs, outputs=declared.outputs)


def test_a_roster_depends_only_on_what_the_author_typed():
    """A connected input has no value until the flow runs, so the roster falls
    back to the declaration for it."""
    compiled = _compiled([
        GraphNode(id="src", type="text-input", version=1, bindings={"value": Static(value="b")}),
        GraphNode(id="m", type="modes", version=1, bindings={"mode": Edges(refs=(Ref("src", "result"),))}),
    ])

    assert [i.name for i in compiled.roster("m").inputs] == ["mode"]


def test_a_column_a_node_computed_can_be_a_flow_output():
    """End to end: the flow names a column that is on no declaration."""
    sheet = GraphNode(
        id="s", type="open-sheet", version=1, title="Sheet",
        fields={"header": FieldContent(title="Header"), "name": FieldContent(title="Name"), "email": FieldContent(title="E-mail")},
        bindings={"header": Static(value="name,email")},
    )
    compiled = _compiled([sheet])

    assert compiled.is_runnable
    assert [o.name for o in compiled.interface.outputs] == ["s.name", "s.email"]
    assert [o.title for o in compiled.interface.outputs] == ["Name", "E-mail"]


# --- what the flow takes and returns ----------------------------------------


def test_the_derived_interface_comes_through_compiled():
    """An unconnected, unconsumed node is both ends of the
    surface, and each field is named by its address."""
    compiled = _compiled([_exposed()])

    assert [i.name for i in compiled.interface.inputs] == ["besked.value"]
    assert [o.name for o in compiled.interface.outputs] == ["besked.result"]


def test_the_interface_is_an_interface_and_types_a_call_by_address():
    """The flow's surface is the record a node version declares, one
    scale out; a flow returns a computed roster by address, and `model_of`
    types a caller's answers under the addresses — pydantic takes a dotted
    field name outright."""
    compiled = _compiled([_exposed()])

    assert isinstance(compiled.interface, Interface)
    assert compiled.interface.returns is Mapping
    answers = model_of(compiled.interface.inputs).model_validate({"besked.value": "hej"})
    assert dict(answers)["besked.value"] == "hej"


def test_needs_is_the_union_of_the_placements_needs():
    """What the run must provide to this flow is what its nodes need,
    by parameter name — and `execute` refuses a flow whose needs it was not
    given."""
    compiled = _compiled([
        GraphNode(id="a", type="stamped", version=1),
        GraphNode(id="b", type="stamped", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),))}),
        GraphNode(id="c", type="echo", version=1),
    ], _registry(Stamped))

    assert compiled.is_runnable, compiled.problems_for()
    assert compiled.interface.needs == {"clock": Clock}


def test_an_input_arrives_with_its_whole_declaration():
    """Not a projection: a host renders a form from the widget and default."""
    compiled = _compiled([_exposed()])
    (declared,) = compiled.interface.inputs

    assert declared.name == "besked.value"
    assert declared.widget is not None
    assert declared.dtype is Txt
    # The placement's title, not the declaration's.
    assert declared.title == "Message"


def test_a_placement_that_did_not_resolve_contributes_no_fields():
    """The node's own fatal problem says it all; the derivation skips a
    placement compile could not resolve rather than anchoring it again."""
    compiled = _compiled([GraphNode(id="a", type="gone", version=1)])

    assert [p.code for p in compiled.problems_for()] == ["unknown_node_type"]
    assert compiled.interface.inputs == () and compiled.interface.outputs == ()


# --- for the engine ----------------------------------------------------------


def test_the_engine_asks_for_a_placement_its_version_and_its_runner():
    compiled = _compiled([GraphNode(id="old", type="renamed", version=1, bindings={"old": Static(value="hi")})])

    assert compiled.node("old").type == "renamed"
    assert compiled.version("old").interface.inputs[0].name == "old"
    assert compiled.runner("old")(old=Txt("hej")) == "hej"
    assert compiled.dependencies("old") == frozenset()


def test_dependencies_are_read_off_the_wires():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),)), "y": Edges(refs=(Ref("a", "result"),))}),
    ])

    assert compiled.dependencies("b") == frozenset({"a"})


# --- the artifact is a value ------------------------------------------------


def test_compiling_the_same_flow_twice_gives_the_same_answers():
    def build():
        return Graph(nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": Static(value="hi")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),))}),
        ])

    first = compile_graph(build(), _registry())
    second = compile_graph(build(), _registry())

    assert first.execution_order() == second.execution_order()
    assert first.problems_for() == second.problems_for()
    assert first.value_source("b", "x") == second.value_source("b", "x")
    assert first.interface == second.interface
    assert first.roster("b") == second.roster("b")
    assert first.carried(Ref("b", "result")) == second.carried(Ref("b", "result"))


def test_compiling_does_not_mutate_the_flow():
    import copy

    flow = Graph(nodes=[GraphNode(id="a", type="echo", version=1, bindings={"x": Static(value="hi")})])
    before = copy.deepcopy(flow)
    compile_graph(flow, _registry())

    assert flow == before


def test_the_artifact_and_its_diagnostics_are_importable_from_the_root():
    import conductor

    assert conductor.CompiledGraph is CompiledGraph
    assert conductor.Carried is not None
    assert conductor.Problem is Problem
    assert conductor.Condition is not None and conductor.Atom is not None
    assert callable(conductor.compile_graph)
    for gone in ("compile", "resolve_graph_outputs", "FOR_EACH"):
        assert not hasattr(conductor, gone), gone
