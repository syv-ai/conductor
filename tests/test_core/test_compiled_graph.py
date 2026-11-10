"""A compiled graph is asked, not traversed."""

from collections import Counter
from typing import Annotated, Any

import pytest
from conductor import (
    AsItArrives,
    CompiledGraph,
    ComputeHookError,
    DType,
    FromRun,
    GraphRecord,
    Input,
    Interface,
    NodeRecord,
    NodeRegistry,
    Output,
    Param,
    Policy,
    Problem,
    Ref,
    Result,
    Series,
    Upgrade,
    node,
)
from conductor._internal.graph.binding import From, Static
from conductor._internal.node_signature import build_call_validator
from conductor.widgets import Choice, Dropdown, Textarea
from helpers import placed, record, registry_of
from pydantic import ValidationError

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "compiled-test-txt"
    display_title = "Text"


class Num(DType, float):
    id = "compiled-test-num"
    display_title = "Number"


Out = Annotated[Txt, Result(title="Result")]


@node("echo", title="Echo", description="d", category="test")
def echo(
    x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt(""),
    y: Annotated[Txt, Param(title="Y", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(x + y)


@node("text-input", title="Text", description="d", category="test")
def text_input(value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return value


def modes_inputs(mode):
    if mode == "a":
        return ()
    return (Input("extra", Txt, title="Extra", widget=Textarea(), default=Txt("")),)

@node("modes", title="Modes", description="d", category="test", compute_added_inputs=modes_inputs)
def modes(
    mode: Annotated[Txt, Param(title="State", widget=Dropdown(choices=(Choice(id="a", title="A"), Choice(id="b", title="B"))))] = Txt("a"),
    **more: Txt,
) -> Out:
    """An interface that depends on what the author typed."""
    return mode


def open_sheet_outputs(header=""):
    return tuple(
        Output(col, Txt)
        for col in str(header).split(",")
        if col
    )

@node("open-sheet", title="Open sheet", description="d", category="test", compute_outputs=open_sheet_outputs)
def open_sheet(header: Annotated[Txt, Param(title="Header", widget=Textarea())] = Txt("")) -> Out:
    """Columns come from the value."""
    return header


def renamed_v1(old: Annotated[Txt, Param(title="Old", widget=Textarea())] = Txt("")) -> Out:
    return old

@node("renamed", title="Renamed", description="d", category="test", policy=Policy(retries=2), replaces=node("renamed")(renamed_v1), upgrade=Upgrade(renamed_inputs={"old": "new"}))
def renamed(new: Annotated[Txt, Param(title="New", widget=Textarea())] = Txt("")) -> Out:
    return new


class Clock:
    """Something the run supplies, not the graph (`FromRun`)."""


@node("stamped", title="Stamped", description="d", category="test")
def stamped(
    clock: FromRun[Clock],
    x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt(""),
) -> Out:
    """A node with a need: the run must provide a `Clock`."""
    return x


@node("open-inputs", title="Open inputs", description="d", category="test")
def open_inputs(**inputs: AsItArrives) -> Out:
    """Takes any edge as an input of its own."""
    return Txt("")


def adds_hidden_inputs():
    return (Input("_x", Txt, title="X", connectable=False, default=Txt("")),)

@node("hidden", title="Hidden", description="d", category="test", compute_added_inputs=adds_hidden_inputs)
def adds_hidden(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt(""), **more: Any) -> Out:
    """Its hook adds a closed input the author cannot see."""
    return value


def _registry(*extra):
    return registry_of(echo, text_input, modes, open_sheet, renamed, *extra)


def _compiled(nodes, registry=None, **kw):
    return CompiledGraph.from_graph(record(*nodes), registry or _registry())


def _exposed(node_id="besked", value="hej"):
    """One node whose `value` field a graph can name."""
    return placed(node_id, "text-input", defaults_={"value": value})


# --- the artifact ---------------------------------------------------------


def test_compiling_returns_an_asked_artifact():
    compiled = _compiled([placed("a", "echo", x=Static("hi"))])

    assert isinstance(compiled, CompiledGraph)
    assert compiled.execution_order == ("a",)
    assert compiled.is_runnable
    assert compiled.problems == ()


def test_callers_never_touch_a_binding_table():
    compiled = _compiled([placed("a", "echo")])

    for traversed in ("bindings", "node_map", "edge_map", "incoming_map", "for_engine"):
        assert not hasattr(compiled, traversed), traversed


def test_compile_builds_each_node_and_field_once():
    """``node`` and ``field`` hand back what compile stored: the same value on every ask."""
    compiled = _compiled([placed("a", "echo")])
    name = compiled.node("a").interface.inputs[0].name

    assert compiled.node("a") is compiled.node("a")
    assert compiled.field(Ref("a", name)) is compiled.field(Ref("a", name))


def test_a_compiled_value_is_equal_only_to_itself():
    """The same graph compiled twice gives two nodes and two fields, never equal across the two."""
    graph = [placed("a", "echo")]
    first, second = _compiled(graph), _compiled(graph)
    name = first.node("a").interface.inputs[0].name

    assert first.node("a") != second.node("a")
    assert first.field(Ref("a", name)) != second.field(Ref("a", name))
    assert len({first.node("a"), first.node("a"), second.node("a")}) == 2


def _half_finished():
    """``a`` is fine, ``b`` reads a node that is not there, ``c`` reads ``b``, ``x`` names a lost type."""
    return _compiled([
        placed("a", "echo", x=Static("hi")),
        placed("b", "echo", x=From("zzz.result")),
        placed("c", "echo", x=From("b.result")),
        placed("x", "nope"),
    ])


def test_every_node_the_author_wrote_has_a_state():
    """``node`` answers for every node compile met, however far it got; ``KeyError`` is only for an id the graph lacks."""
    compiled = _half_finished()

    assert {node_id: compiled.node(node_id).status for node_id in "abcx"} == {
        "a": "ready", "b": "wiring_failed", "c": "wiring_failed", "x": "resolution_failed",
    }
    with pytest.raises(KeyError):
        compiled.node("zzz")


def test_reading_what_a_state_lacks_names_the_problem():
    """A gated read raises a named error carrying the problem that explains it: the node's own, or the one upstream."""
    compiled = _half_finished()

    assert compiled.node("b")._fields is None
    own = compiled.node("b").cause
    assert own.type == "unknown_ref_node" and own.loc[0] == "b"
    assert compiled.node("c")._fields is None
    upstream = compiled.node("c").cause
    assert upstream is own
    with pytest.raises(KeyError):
        compiled.field(Ref("c", "x")).dtype
    assert compiled.node("x").interface is None
    lost = compiled.node("x").cause
    assert lost.type == "unknown_node_type"
    with pytest.raises(KeyError):
        compiled.field(Ref("x", "x"))
    # What each state has still answers.
    assert compiled.node("c").interface.inputs[0].name == "x"
    assert compiled.node("c").graph_node.bindings["x"] == From("b.result")
    assert compiled.node("x").graph_node.node == "nope"


def test_a_node_downstream_of_a_cycle_carries_the_cycle():
    compiled = _compiled([
        placed("a", "echo", x=From("b.result")),
        placed("b", "echo", x=From("a.result")),
        placed("c", "echo", x=From("b.result")),
    ])

    assert (compiled.node("a").status, compiled.node("b").status) == ("resolution_failed", "resolution_failed")
    assert compiled.node("c").status == "wiring_failed"
    assert compiled.node("c")._fields is None
    raised = compiled.node("c").cause
    assert raised.type == "cycle"


def test_a_painter_reads_every_node_of_a_broken_graph():
    """What an editor's compile view does: ask each node its state and read what that state has. Nothing raises."""
    compiled = _half_finished()

    for drawn in compiled.graph.nodes:
        graph_node = compiled.node(drawn.id)
        if graph_node.status == "resolution_failed":
            continue
        assert graph_node.interface.inputs
        if graph_node.status == "ready":
            assert compiled.run_plan.nodes[drawn.id].runs_per_row_of is None
            assert all(compiled.field(Ref(drawn.id, i.name)).dtype is Txt for i in graph_node.interface.inputs)


def test_node_problems_are_the_problems_anchored_on_it():
    compiled = _compiled([
        placed("a", "echo", x=Static("hi"), z=Static(1)),
        placed("b", "echo"),
    ])

    assert compiled.node("a").problems == compiled.problems
    assert compiled.node("b").problems == ()
    assert compiled.field(Ref("a", "x")).problems == ()


def test_an_input_named_with_a_leading_underscore_is_refused_at_compile():
    """A call cannot carry such a name, so compile says so instead of the first call failing."""
    compiled = _compiled([
        placed("a", "echo"),
        placed("s", "open-inputs", _x=From("a.result")),
    ], _registry(open_inputs))

    assert [(p.type, p.loc[0], p.loc[1]) for p in compiled.problems] == [("parameter_name_invalid", "s", "_x")]
    assert compiled.node("s").status == "wiring_failed"


def test_a_hook_that_adds_an_input_named_with_a_leading_underscore_is_refused_at_compile():
    compiled = _compiled([placed("h", "hidden")], _registry(adds_hidden))

    assert [(p.type, p.loc[1]) for p in compiled.problems] == [("parameter_name_invalid", "_x")]
    assert compiled.node("h").status == "wiring_failed"


@node("needs", title="Needs", description="d", category="test")
def needs_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())]) -> Out:
    """One required input with a type and no default."""
    return text


def test_an_empty_typed_input_is_a_value_that_arrives_later():
    """A required input left empty: the graph cannot run, but compile still
    works out the node and everything after it, so an editor draws their
    types and a host can offer the field as an input of the graph."""
    compiled = _compiled([
        placed("n", "needs"),
        placed("e", "echo", x=From("n.result")),
    ], _registry(needs_node))

    assert not compiled.is_runnable
    assert [p.type for p in compiled.problems] == ["unbound_required"]
    assert (compiled.node("n").status, compiled.node("e").status) == ("ready", "ready")
    assert compiled.run_plan.nodes["n"].inputs["text"].receive_mode == "same_value_every_row"
    assert compiled.field(Ref("e", "result")).dtype is Txt



def headed_outputs(header=None):
    if header is None:
        raise ComputeHookError("Type a header first.", code="header_missing")
    return (Output("result", Txt, title="R"),)

@node("headed", title="Headed", description="d", category="test", compute_outputs=headed_outputs)
def headed(header: Annotated[Txt, Param(title="Header", widget=Textarea())]) -> Out:
    """Outputs named by a required header the author types; no header yet, no outputs to name."""
    return header


def test_a_hook_is_not_handed_a_required_input_left_empty():
    """``values`` has no entry for a required input nobody filled, just as
    for a connected one; a hook that needs it refuses, and compile reports."""
    compiled = _compiled([placed("n", "headed")], _registry(headed))

    assert [(p.type, p.fatal) for p in compiled.problems] == [("unbound_required", True), ("header_missing", True)]


@node("gathers", title="Gathers", description="d", category="test")
def gathers(items: Annotated[Series[Any], Param(title="Items", widget=Textarea())]) -> Out:
    """A required list of anything: only an edge can say what it holds."""
    return Txt("")


def test_an_empty_list_of_anything_breaks_its_node_like_anything():
    compiled = _compiled([placed("n", "gathers")], _registry(gathers))

    assert [p.type for p in compiled.problems] == ["unbound_required"]
    assert compiled.node("n").status == "wiring_failed"


#: A stored graph embedded as a node: one echo inside.
Wrapped = embedded_graph_node("wrapped", (placed("inner", "echo"),), _registry(), "Wrapped")


class CountingRegistry(NodeRegistry):
    """Counts every lookup of a class by id."""

    def __getitem__(self, node_id: str):
        self.counts[node_id] += 1
        return super().__getitem__(node_id)


def test_compile_looks_each_node_up_once():
    """One lookup per node the author drew; everything after reads the class compile resolved. An
    embedded graph's own nodes were looked up once, when the host compiled it."""
    registry = CountingRegistry()
    registry.add(echo, Wrapped)
    registry.counts = Counter()
    CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="a", node="echo"),
        NodeRecord(id="b", node="echo"),
        NodeRecord(id="emb", node="wrapped"),
    ], edges=[("a.result", "b.x"), ("b.result", "emb.x")]), registry)

    assert registry.counts == {"echo": 2, "wrapped": 1}


def test_a_compiled_node_holds_the_class_it_resolved_to():
    compiled = _compiled([
        placed("a", "echo"),
        placed("emb", "wrapped"),
    ], _registry(Wrapped))

    assert compiled.node("a").definition is echo
    assert compiled.node("emb").definition is Wrapped
    assert compiled.node("emb/inner").definition is echo


def _imports_of(module) -> set[str]:
    """Every module a source file imports when it runs, and every name it
    imports from one as ``module.name``, so ``from conductor._internal.graph import
    compiler`` counts as reaching ``conductor._internal.graph.compiler``. A type named
    only for the type checker (``if TYPE_CHECKING:``) is not an import."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(module.__file__).read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.If) and getattr(n.test, "id", None) == "TYPE_CHECKING")]
    reached: set[str] = set()
    for graph_node in ast.walk(tree):
        if isinstance(graph_node, ast.ImportFrom) and graph_node.module:
            reached |= {graph_node.module, *(f"{graph_node.module}.{alias.name}" for alias in graph_node.names)}
        if isinstance(graph_node, ast.Import):
            reached |= {alias.name for alias in graph_node.names}
    return reached


def test_compile_does_not_know_the_compiled_graph():
    """``compiled`` imports the walk and never the other way."""
    import conductor._internal.graph.compile_graph as walk_module

    reached = _imports_of(walk_module)

    assert not {name for name in reached if name == "conductor._internal.graph.compiled_graph" or name.startswith("conductor._internal.graph.compiled_graph.")}
    assert not {name for name in reached if name.rsplit(".", 1)[-1] == "CompiledGraph"}


def test_the_compiled_values_know_neither_the_compiler_nor_the_graph():
    """``compiled_node`` holds the answers for one node and one field;
    compile's last step fills them in. It imports neither the compiler nor
    ``compiled``, so the values stand on their own."""
    import conductor._internal.graph.compiled_node as compiled_node_module

    assert not _imports_of(compiled_node_module) & {"conductor._internal.graph.compiler", "conductor._internal.graph.compiled_graph"}


def test_execution_order_follows_the_edges():
    compiled = _compiled([
        placed("b", "echo", x=From("a.result")),
        placed("a", "echo"),
    ])

    assert compiled.execution_order == ("a", "b")


# --- a field's binding ----------------------------------------------------


def test_a_field_the_node_does_not_have_has_no_view():
    """A programming error, not a state of the graph."""
    compiled = _compiled([placed("a", "echo")])

    with pytest.raises(KeyError, match="nonexistent"):
        compiled.field(Ref("a", "nonexistent"))


# --- the pin is resolved once, here ------------------------------------------


def test_a_node_reads_the_version_it_pins():
    """The version freezes which behaviour a node in a graph points at.
    Reading the class's current version instead would silently re-point
    every old node at the newest signature."""
    compiled = _compiled([placed("old", "renamed")])

    assert [i.name for i in compiled.node("old").interface.inputs] == ["old"]
    assert compiled.node("old").version.policy == Policy()
    assert compiled.node("old").version.body.__name__ == "renamed_v1"


def test_a_node_type_the_registry_lacks_is_a_fatal_problem():
    """A catalog can drift under a stored graph. That is a graph state, and
    the editor needs to show where."""
    compiled = _compiled([placed("a", "gone")])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc[0]) == ("unknown_node_type", True, "a")
    assert not compiled.is_runnable
    assert compiled.node("a").status == "resolution_failed"
    assert compiled.node("a").interface is None
    raised = compiled.node("a").cause
    assert raised == make_problem


def test_a_version_the_class_no_longer_declares_is_a_fatal_problem():
    compiled = _compiled([placed("a", "renamed", version=7)])

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.loc[0]) == ("unknown_node_version", "a")


def test_two_nodes_with_one_id_is_a_fatal_problem():
    compiled = _compiled([
        placed("a", "echo"),
        placed("a", "echo"),
    ])

    assert [p.type for p in compiled.problems] == ["duplicate_node_id"]


# --- interfaces: the hooks are asked, once, on a fresh instance -----------------


def test_a_nodes_interface_is_what_the_hooks_answered():
    compiled = _compiled([
        placed("m", "modes", mode=Static("a")),
        placed("s", "open-sheet", header=Static("name,email")),
    ])

    assert isinstance(compiled.node("m").interface, Interface)
    assert [i.name for i in compiled.node("m").interface.inputs] == ["mode"]
    assert [o.name for o in compiled.node("s").interface.outputs] == ["name", "email"]


def test_a_node_with_no_opinion_has_its_declaration_as_its_interface():
    compiled = _compiled([placed("a", "echo")])

    declared = echo.versions[1].interface
    assert compiled.node("a").interface == declared


def test_a_inputs_depends_only_on_what_the_author_typed():
    """A connected input has no value until the graph runs, so the node's interface falls
    back to the declaration for it."""
    compiled = _compiled([
        placed("src", "text-input", value=Static("b")),
        placed("m", "modes", mode=From("src.result")),
    ])

    assert [i.name for i in compiled.node("m").interface.inputs] == ["mode"]


def test_a_column_a_node_computed_can_be_a_graph_output():
    """End to end: the graph names a column that is on no declaration."""
    sheet = placed("s", "open-sheet", header=Static("name,email"))
    compiled = _compiled([sheet])

    assert compiled.is_runnable
    assert [o.name for o in compiled.interface.outputs] == ["s.name", "s.email"]
    assert [o.title for o in compiled.interface.outputs] == ["Name", "Email"]


# --- what the graph takes and returns ----------------------------------------


def test_the_derived_interface_comes_through_compiled():
    """An unconnected, unconsumed node is both ends of the
    surface, and each field is named by its address."""
    compiled = _compiled([_exposed()])

    assert [i.name for i in compiled.interface.inputs] == ["besked.value"]
    assert [o.name for o in compiled.interface.outputs] == ["besked.result"]


def test_the_interface_is_an_interface_and_types_a_call_by_address():
    """The graph's surface is the record a node version declares, one
    scale out; a graph returns a computed interface by address, and `build_call_validator`
    types a caller's answers under the addresses, a dotted name as any other key."""
    compiled = _compiled([_exposed()])

    assert isinstance(compiled.interface, Interface)
    answers = build_call_validator(compiled.interface.inputs)({"besked.value": "hej"})
    assert answers["besked.value"] == "hej"


def test_needs_is_the_union_of_the_nodes_needs():
    """What the run must provide to this graph is what its nodes need,
    by parameter name — and `execute` refuses a graph whose needs it was not
    given."""
    compiled = _compiled([
        placed("a", "stamped"),
        placed("b", "stamped", x=From("a.result")),
        placed("c", "echo"),
    ], _registry(stamped))

    assert compiled.is_runnable, compiled.problems
    assert compiled.interface.needs == {Clock}


def test_an_input_arrives_with_its_whole_declaration():
    """Not a projection: a host renders a form from the widget and default."""
    compiled = _compiled([_exposed()])
    (declared,) = compiled.interface.inputs

    assert declared.name == "besked.value"
    assert declared.widget is not None
    assert declared.dtype is Txt
    assert declared.title == "Text"


def test_a_node_that_did_not_resolve_contributes_no_fields():
    """The node's own fatal problem says it all; the derivation skips a
    node compile could not resolve rather than anchoring it again."""
    compiled = _compiled([placed("a", "gone")])

    assert [p.type for p in compiled.problems] == ["unknown_node_type"]
    assert compiled.interface.inputs == () and compiled.interface.outputs == ()


# --- for the engine ----------------------------------------------------------


def test_the_engine_asks_for_a_placement_its_version_and_its_runner():
    compiled = _compiled([placed("old", "renamed", old=Static("hi"))])

    assert compiled.node("old").graph_node.node == "renamed"
    assert compiled.node("old").version.interface.inputs[0].name == "old"
    assert compiled.node("old").version.body(old=Txt("hej")) == "hej"


# --- the artifact is a value ------------------------------------------------


def test_compiling_the_same_graph_twice_gives_the_same_answers():
    def build():
        return GraphRecord(nodes=[
            NodeRecord(id="a", node="echo", values={"x": "hi"}),
            NodeRecord(id="b", node="echo"),
        ], edges=[("a.result", "b.x")])

    first = CompiledGraph.from_graph(build(), _registry())
    second = CompiledGraph.from_graph(build(), _registry())

    assert first.execution_order == second.execution_order
    assert first.problems == second.problems
    assert first.node("b").graph_node.bindings.get("x") == second.node("b").graph_node.bindings.get("x")
    assert first.interface == second.interface
    assert first.node("b").interface == second.node("b").interface
    assert first.field(Ref("b", "result")).dtype is second.field(Ref("b", "result")).dtype
    assert first.run_plan == second.run_plan


def test_compiling_does_not_mutate_the_graph():
    import copy

    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node="echo", values={"x": "hi"}),
    ])
    before = copy.deepcopy(graph)
    CompiledGraph.from_graph(graph, _registry())

    assert graph == before


def test_the_artifact_and_its_diagnostics_are_importable_from_the_root():
    import conductor

    assert conductor.CompiledGraph is CompiledGraph
    assert conductor.Problem is Problem
    assert conductor.BranchTaken is not None
    assert callable(conductor.CompiledGraph.from_graph)


# --- the call model, and what the record does not carry ------------------------------


def test_a_compiled_node_validates_a_call_and_hands_back_its_keyword_arguments():
    """Validating a call is compile's, through a model built once per
    node; the engine asks ``validate`` and gets the keyword arguments the
    call runs with, and never meets the model."""
    compiled = _compiled([placed("a", "echo", x=Static("hi"))])
    graph_node = compiled.node("a")

    kwargs = graph_node._validate_call_inputs({"x": Txt("a"), "y": Txt("")})
    assert set(kwargs) == {"x", "y"} and isinstance(kwargs["x"], Txt) and kwargs["y"] == Txt("")
    wired = _compiled([_exposed(), placed("b", "echo", x=From("besked.result"))]).node("b")
    with pytest.raises(ValidationError):
        wired._validate_call_inputs({"x": ["not", "text"], "y": Txt("")})


def test_a_default_reaches_the_node_as_its_input_s_type():
    """A default is a value of its input like any other: ``n: Num = 3``
    with nothing bound reaches ``run`` as a ``Num``, not a bare ``int``."""

    seen = []

    @node("counted", title="Counted", description="d", category="test")
    def counted(n: Annotated[Num, Param(title="N", widget=Textarea())] = 3) -> Out:  # type: ignore[assignment]
        seen.append(n)
        return Txt("")

    compiled = _compiled([placed("c", "counted")], registry_of(counted))

    assert compiled.run_sync().status == "complete"
    assert [type(n) for n in seen] == [Num] and seen == [3.0]


def test_the_record_keeps_the_authored_graph_and_the_registry_and_drops_what_nothing_calls():
    """A compiled graph carries no ``dependencies``; each node and field holds
    the problems about it, a slice of ``compiled.problems`` — while the
    authored graph and the registry stay, so a compiled graph can produce
    another."""
    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node="echo", values={"x": "hi", "z": 1}),
    ])
    registry = _registry()
    compiled = CompiledGraph.from_graph(graph, registry)

    assert compiled.graph is graph and compiled._registry is registry
    assert not hasattr(compiled.node("a"), "dependencies")
    assert [p.type for p in compiled.node("a").problems] == ["stale_binding"]
    assert compiled.field(Ref("a", "x")).problems == ()
