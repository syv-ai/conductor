"""A compiled graph is asked, not traversed."""

from collections import Counter
from collections.abc import Mapping
from typing import Annotated, ClassVar

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType, Single
from conductor.errors import (
    CompilationError,
    NodeKindError,
    NodeResolutionError,
    NodeWiringError,
)
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.graph.problem import Problem
from conductor.interface import FromRun, Interface, model_of
from conductor.metadata import Input, Output, Param, Result
from conductor.node import GraphVersion, NodeDefinition, Policy, upgrade, version
from conductor.ref import Ref
from conductor.widgets import Choice, Dropdown, Textarea
from pydantic import ValidationError


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
        x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt(""),
        y: Annotated[Txt, Param(title="Y", widget=Textarea())] = Txt(""),
    ) -> Out:
        return Txt(x + y)


class TextInput(NodeDefinition):
    id = "text-input"
    title = "Text"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return value


class Modes(NodeDefinition):
    """An interface that depends on what the author typed."""

    id = "modes"
    title = "Modes"
    description = "d"
    category = "test"

    def run(
        self,
        mode: Annotated[Txt, Param(title="State", widget=Dropdown(choices=(Choice(id="a", title="A"), Choice(id="b", title="B"))))] = Txt("a"),
        extra: Annotated[Txt, Param(title="Extra", widget=Textarea())] = Txt(""),
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

    def run(self, header: Annotated[Txt, Param(title="Header", widget=Textarea())] = Txt("")) -> Out:
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

    @upgrade(1, 2)
    def _v1_to_v2(values):
        return values

    @version(1)
    def run_v1(self, old: Annotated[Txt, Param(title="Old", widget=Textarea())] = Txt("")) -> Out:
        return old

    @version(2, policy=Policy(retries=2))
    def run(self, new: Annotated[Txt, Param(title="New", widget=Textarea())] = Txt("")) -> Out:
        return new


class Clock:
    """Something the run supplies, not the graph (`FromRun`)."""


class Stamped(NodeDefinition):
    """A node with a need: the run must provide a `Clock`."""

    id = "stamped"
    title = "Stamped"
    description = "d"
    category = "test"

    def run(
        self,
        clock: Annotated[Clock, FromRun()],
        x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt(""),
    ) -> Out:
        return x


class OpenInputs(NodeDefinition):
    """Takes any edge as an input of its own."""

    id = "open-inputs"
    title = "Open inputs"
    description = "d"
    category = "test"

    def run(self, **inputs: Single) -> Out:
        return Txt("")


class AddsHidden(NodeDefinition):
    """Its hook adds a closed input the author cannot see."""

    id = "hidden"
    title = "Hidden"
    description = "d"
    category = "test"

    def run(self, value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
        return value

    def compute_inputs(self, declared, values):
        return (*declared, Input(name="_x", dtype=Txt, title="X", show_handle=False, default=Txt(""), optional=True))


def _registry(*extra):
    registry = NodeRegistry()
    for node_cls in (Echo, TextInput, Modes, OpenSheet, Renamed, *extra):
        registry.register(node_cls)
    return registry


def _compiled(nodes, registry=None, **kw):
    return CompiledGraph.from_graph(Graph(nodes=nodes, **kw), registry or _registry())


def _exposed(node_id="besked", value="hej"):
    """One node whose `value` field a graph can name."""
    return GraphNode(
        id=node_id,
        type="text-input",
        version=1,
        title="Message",
        fields={"value": FieldContent(title="Message"), "result": FieldContent(title="Text")},
        bindings={"value": Static(value)},
    )


# --- the artifact ---------------------------------------------------------


def test_compiling_returns_an_asked_artifact():
    compiled = _compiled([GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")})])

    assert isinstance(compiled, CompiledGraph)
    assert compiled.execution_order == ("a",)
    assert compiled.is_runnable
    assert compiled.problems == ()


def test_callers_never_touch_a_binding_table():
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    for traversed in ("bindings", "node_map", "edge_map", "incoming_map", "for_engine"):
        assert not hasattr(compiled, traversed), traversed


def test_compile_builds_each_node_and_field_once():
    """``node`` and ``field`` hand back what compile stored: the same value on every ask."""
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])
    name = compiled.node("a").interface.inputs[0].name

    assert compiled.node("a") is compiled.node("a")
    assert compiled.field(Ref("a", name)) is compiled.field(Ref("a", name))


def test_a_compiled_value_is_equal_only_to_itself():
    """The same graph compiled twice gives two nodes and two fields, never equal across the two."""
    graph = [GraphNode(id="a", type="echo", version=1)]
    first, second = _compiled(graph), _compiled(graph)
    name = first.node("a").interface.inputs[0].name

    assert first.node("a") != second.node("a")
    assert first.field(Ref("a", name)) != second.field(Ref("a", name))
    assert len({first.node("a"), first.node("a"), second.node("a")}) == 2


def _half_finished():
    """``a`` is fine, ``b`` reads a node that is not there, ``c`` reads ``b``, ``x`` names a lost type."""
    return _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")}),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From("zzz.result")}),
        GraphNode(id="c", type="echo", version=1, bindings={"x": From("b.result")}),
        GraphNode(id="x", type="nope", version=1),
    ])


def test_every_node_the_author_wrote_has_a_state():
    """``node`` answers for every node compile met, however far it got; ``KeyError`` is only for an id the graph lacks."""
    compiled = _half_finished()

    assert {node_id: compiled.node(node_id).state for node_id in "abcx"} == {
        "a": "ready", "b": "wiring_failed", "c": "wiring_failed", "x": "resolution_failed",
    }
    with pytest.raises(KeyError):
        compiled.node("zzz")


def test_reading_what_a_state_lacks_names_the_problem():
    """A gated read raises a named error carrying the problem that explains it: the node's own, or the one upstream."""
    compiled = _half_finished()

    with pytest.raises(NodeWiringError) as own:
        compiled.node("b").iterates_on
    assert own.value.problems[0].code == "unknown_ref_node" and own.value.problems[0].node_id == "b"
    with pytest.raises(NodeWiringError) as upstream:
        compiled.node("c").iterates_on
    assert upstream.value.problems[0] is own.value.problems[0]
    with pytest.raises(NodeWiringError):
        compiled.field(Ref("c", "x")).type
    with pytest.raises(NodeResolutionError) as lost:
        compiled.node("x").interface
    assert lost.value.problems[0].code == "unknown_node_type"
    with pytest.raises(NodeResolutionError):
        compiled.field(Ref("x", "x"))
    # What each state has still answers.
    assert compiled.node("c").interface.inputs[0].name == "x"
    assert compiled.field(Ref("c", "x")).binding == From("b.result")
    assert compiled.node("x").graph_node.type == "nope"


def test_the_named_errors_are_not_key_errors():
    """``except KeyError`` catches a caller's wrong id and nothing else; the message names the node, the read and the cause."""
    compiled = _half_finished()
    with pytest.raises(NodeWiringError) as raised:
        compiled.node("c").iterates_on

    assert issubclass(NodeWiringError, CompilationError) and not issubclass(NodeWiringError, KeyError)
    assert issubclass(NodeResolutionError, CompilationError) and not issubclass(NodeResolutionError, KeyError)
    assert not issubclass(NodeKindError, CompilationError)
    assert len(raised.value.problems) == 1
    assert "'c'" in str(raised.value) and "iterates_on" in str(raised.value)
    assert "b.x — unknown_ref_node" in str(raised.value)


def test_a_node_downstream_of_a_cycle_carries_the_cycle():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"x": From("b.result")}),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
        GraphNode(id="c", type="echo", version=1, bindings={"x": From("b.result")}),
    ])

    assert (compiled.node("a").state, compiled.node("b").state) == ("resolution_failed", "resolution_failed")
    assert compiled.node("c").state == "wiring_failed"
    with pytest.raises(NodeWiringError) as raised:
        compiled.node("c").iterates_on
    assert raised.value.problems[0].code == "cycle"


def test_a_painter_reads_every_node_of_a_broken_graph():
    """What an editor's compile view does: ask each node its state and read what that state has. Nothing raises."""
    compiled = _half_finished()

    for placed in compiled.graph.nodes:
        node = compiled.node(placed.id)
        if node.state == "resolution_failed":
            continue
        assert node.interface.inputs
        if node.state == "ready":
            assert node.iterates_on is None
            assert all(compiled.field(Ref(placed.id, i.name)).type is Txt for i in node.interface.inputs)


def test_node_problems_are_the_problems_anchored_on_it():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1, locked=("ghost",), bindings={"x": Static("hi"), "z": Static(1)}),
        GraphNode(id="b", type="echo", version=1),
    ])

    assert compiled.node("a").problems == compiled.problems
    assert compiled.node("b").problems == ()
    assert compiled.field(Ref("a", "x")).problems == ()


def test_an_input_named_with_a_leading_underscore_is_refused_at_compile():
    """A call cannot carry such a name, so compile says so instead of the first call failing."""
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="s", type="open-inputs", version=1, bindings={"_x": From("a.result")}),
    ], _registry(OpenInputs))

    assert [(p.code, p.node_id, p.field) for p in compiled.problems] == [("parameter_name_invalid", "s", "_x")]
    assert compiled.node("s").state == "wiring_failed"


def test_a_hook_that_adds_an_input_named_with_a_leading_underscore_is_refused_at_compile():
    compiled = _compiled([GraphNode(id="h", type="hidden", version=1)], _registry(AddsHidden))

    assert [(p.code, p.field) for p in compiled.problems] == [("parameter_name_invalid", "_x")]
    assert compiled.node("h").state == "wiring_failed"


class Wrapped(NodeDefinition):
    """A stored graph placed as a node: one echo inside."""

    id = "wrapped"
    title = "Wrapped"
    description = "d"
    category = "test"
    versions: ClassVar[dict[int, GraphVersion]] = {
        1: GraphVersion(
            graph=(GraphNode(id="inner", type="echo", version=1),),
            interface=Interface(
                inputs=(Input(name="inner.x", dtype=Txt, title="X", widget=Textarea(), default=Txt(""), optional=True),),
                outputs=(Output(name="inner.result", dtype=Txt, title="Result"),),
                returns=Mapping,
            ),
        )
    }


class CountingRegistry(NodeRegistry):
    """Counts every lookup of a class by id."""

    def __getitem__(self, node_id: str):
        self.counts[node_id] += 1
        return super().__getitem__(node_id)


def test_compile_looks_each_placement_up_once():
    """One lookup per node the author placed and per inner node of an embedded graph; everything after reads the class compile resolved."""
    registry = CountingRegistry(nodes=(Echo, Wrapped))
    registry.counts = Counter()
    CompiledGraph.from_graph(Graph(nodes=[
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
        GraphNode(id="emb", type="wrapped", version=1, bindings={"inner.x": From("b.result")}),
    ]), registry)

    assert registry.counts == {"echo": 3, "wrapped": 1}


def test_a_compiled_node_holds_the_class_it_resolved_to():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="emb", type="wrapped", version=1),
    ], _registry(Wrapped))

    assert compiled.node("a").definition is Echo
    assert compiled.node("emb").definition is Wrapped
    assert compiled.node("emb/inner").definition is Echo


def _imports_of(module) -> set[str]:
    """Every module a source file names in an import, and every name it
    imports from one as ``module.name``, so ``from conductor.graph import
    compiler`` counts as reaching ``conductor.graph.compiler``."""
    import ast
    from pathlib import Path

    reached: set[str] = set()
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached |= {node.module, *(f"{node.module}.{alias.name}" for alias in node.names)}
        if isinstance(node, ast.Import):
            reached |= {alias.name for alias in node.names}
    return reached


def test_the_compiler_does_not_know_the_compiled_graph():
    """``compiled`` imports ``compiler`` and never the other way: the passes know nothing of their result."""
    import conductor.graph.compiler as compiler_module

    reached = _imports_of(compiler_module)
    compiled_names = {"CompiledGraph", "CompiledNode", "CompiledField"}

    assert not {name for name in reached if name.startswith("conductor.graph.compiled")}
    assert not {name for name in reached if name.rsplit(".", 1)[-1] in compiled_names}


def test_the_compiled_values_know_neither_the_compiler_nor_the_graph():
    """``compiled_node`` holds the answers for one node and one field; the
    compiler fills them in through ``compiled``. It imports neither, so the
    values stand on their own."""
    import conductor.graph.compiled_node as compiled_node_module

    assert not _imports_of(compiled_node_module) & {"conductor.graph.compiler", "conductor.graph.compiled"}


def test_execution_order_follows_the_edges():
    compiled = _compiled([
        GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
        GraphNode(id="a", type="echo", version=1),
    ])

    assert compiled.execution_order == ("a", "b")


# --- a field's binding ----------------------------------------------------


def test_binding_answers_where_an_input_comes_from():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")}),
        GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
    ])

    assert isinstance(compiled.field(Ref("a", "x")).binding, Static)
    assert isinstance(compiled.field(Ref("b", "x")).binding, From)


def test_an_unbound_input_reads_as_None_once_compiled():
    """None means "nothing binds it, so the declared default applies" — the
    only "nothing binds this" state there is."""
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    assert compiled.field(Ref("a", "y")).binding is None


def test_a_field_the_node_does_not_have_has_no_view():
    """A programming error, not a state of the graph."""
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    with pytest.raises(KeyError, match="nonexistent"):
        compiled.field(Ref("a", "nonexistent"))


# --- the pin is resolved once, here ------------------------------------------


def test_a_placement_reads_the_version_it_pins():
    """The version freezes which behaviour a placement points at.
    Reading the class's current version instead would silently re-point
    every old placement at the newest signature."""
    compiled = _compiled([GraphNode(id="old", type="renamed", version=1)])

    assert [i.name for i in compiled.node("old").interface.inputs] == ["old"]
    assert compiled.node("old").version.policy == Policy()
    assert compiled.node("old").version.run.__name__ == "run_v1"


def test_a_node_type_the_registry_lacks_is_a_fatal_problem():
    """A catalog can drift under a stored graph. That is a graph state, and
    the editor needs to show where."""
    compiled = _compiled([GraphNode(id="a", type="gone", version=1)])

    (problem,) = compiled.problems
    assert (problem.code, problem.fatal, problem.node_id) == ("unknown_node_type", True, "a")
    assert not compiled.is_runnable
    assert compiled.node("a").state == "resolution_failed"
    with pytest.raises(NodeResolutionError) as raised:
        compiled.node("a").interface
    assert raised.value.problems[0] == problem


def test_a_version_the_class_no_longer_declares_is_a_fatal_problem():
    compiled = _compiled([GraphNode(id="a", type="renamed", version=7)])

    (problem,) = compiled.problems
    assert (problem.code, problem.node_id) == ("unknown_node_version", "a")


def test_two_placements_with_one_id_is_a_fatal_problem():
    compiled = _compiled([
        GraphNode(id="a", type="echo", version=1),
        GraphNode(id="a", type="echo", version=1),
    ])

    assert [p.code for p in compiled.problems] == ["duplicate_node_id"]


# --- interfaces: the hooks are asked, once, on a fresh instance -----------------


def test_a_nodes_interface_is_what_the_hooks_answered():
    compiled = _compiled([
        GraphNode(id="m", type="modes", version=1, bindings={"mode": Static("a")}),
        GraphNode(id="s", type="open-sheet", version=1, bindings={"header": Static("name,email")}),
    ])

    assert isinstance(compiled.node("m").interface, Interface)
    assert [i.name for i in compiled.node("m").interface.inputs] == ["mode"]
    assert [o.name for o in compiled.node("s").interface.outputs] == ["name", "email"]


def test_a_node_with_no_opinion_has_its_declaration_as_its_interface():
    compiled = _compiled([GraphNode(id="a", type="echo", version=1)])

    declared = Echo.versions[1].interface
    assert compiled.node("a").interface == declared


def test_a_roster_depends_only_on_what_the_author_typed():
    """A connected input has no value until the graph runs, so the node's interface falls
    back to the declaration for it."""
    compiled = _compiled([
        GraphNode(id="src", type="text-input", version=1, bindings={"value": Static("b")}),
        GraphNode(id="m", type="modes", version=1, bindings={"mode": From("src.result")}),
    ])

    assert [i.name for i in compiled.node("m").interface.inputs] == ["mode"]


def test_a_column_a_node_computed_can_be_a_graph_output():
    """End to end: the graph names a column that is on no declaration."""
    sheet = GraphNode(
        id="s", type="open-sheet", version=1, title="Sheet",
        fields={"header": FieldContent(title="Header"), "name": FieldContent(title="Name"), "email": FieldContent(title="E-mail")},
        bindings={"header": Static("name,email")},
    )
    compiled = _compiled([sheet])

    assert compiled.is_runnable
    assert [o.name for o in compiled.interface.outputs] == ["s.name", "s.email"]
    assert [o.title for o in compiled.interface.outputs] == ["Name", "E-mail"]


# --- what the graph takes and returns ----------------------------------------


def test_the_derived_interface_comes_through_compiled():
    """An unconnected, unconsumed node is both ends of the
    surface, and each field is named by its address."""
    compiled = _compiled([_exposed()])

    assert [i.name for i in compiled.interface.inputs] == ["besked.value"]
    assert [o.name for o in compiled.interface.outputs] == ["besked.result"]


def test_the_interface_is_an_interface_and_types_a_call_by_address():
    """The graph's surface is the record a node version declares, one
    scale out; a graph returns a computed interface by address, and `model_of`
    types a caller's answers under the addresses — pydantic takes a dotted
    field name outright."""
    compiled = _compiled([_exposed()])

    assert isinstance(compiled.interface, Interface)
    assert compiled.interface.returns is Mapping
    answers = model_of(compiled.interface.inputs).model_validate({"besked.value": "hej"})
    assert dict(answers)["besked.value"] == "hej"


def test_needs_is_the_union_of_the_placements_needs():
    """What the run must provide to this graph is what its nodes need,
    by parameter name — and `execute` refuses a graph whose needs it was not
    given."""
    compiled = _compiled([
        GraphNode(id="a", type="stamped", version=1),
        GraphNode(id="b", type="stamped", version=1, bindings={"x": From("a.result")}),
        GraphNode(id="c", type="echo", version=1),
    ], _registry(Stamped))

    assert compiled.is_runnable, compiled.problems
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

    assert [p.code for p in compiled.problems] == ["unknown_node_type"]
    assert compiled.interface.inputs == () and compiled.interface.outputs == ()


# --- for the engine ----------------------------------------------------------


def test_the_engine_asks_for_a_placement_its_version_and_its_runner():
    compiled = _compiled([GraphNode(id="old", type="renamed", version=1, bindings={"old": Static("hi")})])

    assert compiled.node("old").graph_node.type == "renamed"
    assert compiled.node("old").version.interface.inputs[0].name == "old"
    assert compiled.node("old").runner(old=Txt("hej")) == "hej"


# --- the artifact is a value ------------------------------------------------


def test_compiling_the_same_graph_twice_gives_the_same_answers():
    def build():
        return Graph(nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
        ])

    first = CompiledGraph.from_graph(build(), _registry())
    second = CompiledGraph.from_graph(build(), _registry())

    assert first.execution_order == second.execution_order
    assert first.problems == second.problems
    assert first.field(Ref("b", "x")).binding == second.field(Ref("b", "x")).binding
    assert first.interface == second.interface
    assert first.node("b").interface == second.node("b").interface
    assert first.field(Ref("b", "result")).type is second.field(Ref("b", "result")).type
    assert first.field(Ref("b", "result")).index == second.field(Ref("b", "result")).index


def test_compiling_does_not_mutate_the_graph():
    import copy

    graph = Graph(nodes=[GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")})])
    before = copy.deepcopy(graph)
    CompiledGraph.from_graph(graph, _registry())

    assert graph == before


def test_the_artifact_and_its_diagnostics_are_importable_from_the_root():
    import conductor

    assert conductor.CompiledGraph is CompiledGraph
    assert conductor.Problem is Problem
    assert conductor.Condition is not None and conductor.Atom is not None
    assert callable(conductor.CompiledGraph.from_graph)


# --- the call model, and what the record does not carry ------------------------------


def test_a_compiled_node_validates_a_call_and_hands_back_its_keyword_arguments():
    """Validating a call is compile's, through a model built once per
    node; the engine asks ``validate`` and gets the keyword arguments the
    call runs with, and never meets the model."""
    compiled = _compiled([GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")})])
    node = compiled.node("a")

    kwargs = node.validate({"x": Txt("a")})
    assert set(kwargs) == {"x", "y"} and isinstance(kwargs["x"], Txt) and kwargs["y"] == Txt("")
    with pytest.raises(ValidationError):
        node.validate({"x": ["not", "text"]})
    with pytest.raises(NodeWiringError):
        _compiled([GraphNode(id="b", type="echo", version=1, bindings={"x": From("ghost.result")})]).node("b").validate({})


def test_the_record_keeps_the_authored_graph_and_the_registry_and_drops_what_nothing_calls():
    """A compiled graph carries no ``dependencies``; each node and field holds
    the problems about it, a slice of ``compiled.problems`` — while the
    authored graph and the registry stay, so a compiled graph can produce
    another."""
    graph = Graph(nodes=[GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi"), "z": Static(1)})])
    registry = _registry()
    compiled = CompiledGraph.from_graph(graph, registry)

    assert compiled.graph is graph and compiled._registry is registry
    assert not hasattr(compiled.node("a"), "dependencies")
    assert [p.code for p in compiled.node("a").problems] == ["stale_binding"]
    assert compiled.field(Ref("a", "x")).problems == ()
