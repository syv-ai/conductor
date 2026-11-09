import dataclasses
import inspect
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    CompiledGraph,
    DType,
    FieldContent,
    From,
    FromRun,
    Graph,
    GraphNode,
    Interface,
    NodeRegistry,
    Output,
    Param,
    Ref,
    Result,
    Static,
    dependencies_of,
    node,
)
from conductor._internal.graph.binding import static_values
from conductor._internal.graph.graph_interface import derive_interface
from conductor.widgets import Textarea
from pydantic import TypeAdapter


def test_an_edge_carries_ordered_refs():
    edge = From("a.result", "b.result")

    assert [r.node_id for r in edge.refs] == ["a", "b"]


def test_edge_order_is_operand_order():
    first = From("a.r", "b.r")
    second = From("b.r", "a.r")
    assert first != second


def test_absence_means_the_declared_default_applies():
    """One state, not two — there is nothing to dismiss."""
    bindings = {"b": Static(3)}

    assert "a" not in bindings


def test_static_values_extracts_only_static_bindings():
    bindings = {
        "text": Static("hi"),
        "number": Static(3),
        "source": From("a.result"),
    }
    assert static_values(bindings) == {"text": "hi", "number": 3}


def test_bindings_are_frozen():
    with pytest.raises(Exception):
        Static(1).value = 2
def test_a_node_id_with_a_dot_or_a_slash_is_a_problem_on_the_graph():
    """A Ref reads as 'node.field', so '.' can't be in an id; '/' is an inner node's path. Both are the one refusal, on the stored graph."""
    for bad in ("a.b", "a/b"):
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id=bad, type="upper", version=1)]), _registry())
        assert [(p.type, p.loc) for p in compiled.problems] == [("invalid_node_id", (bad,))]
        assert compiled.node(bad).interface is None


def test_a_slash_is_the_namespace_separator():
    """Compile puts an embedded graph's nodes under the graph node's name
    — `approve/check` — so `/` is legal in an id and a Ref still reads one way:
    the split is at the first dot, and there is none in the namespace."""
    inner = GraphNode(id="approve/check", type="upper", version=1)

    assert inner.id == "approve/check"
    assert Ref("approve/check", "amount") == "approve/check.amount"
    assert Ref("approve/check.amount").node_id == "approve/check"


def test_a_node_stores_bindings_and_derives_data():
    graph_node = GraphNode(
        id="n1",
        type="translate",
        version=1,
        bindings={
            "text": Static("hi"),
            "source": From("n0.result"),
        },
    )
    assert static_values(graph_node.bindings) == {"text": "hi"}


def test_a_node_is_behaviour_content_and_chrome():
    """The three categories, and nothing from the deleted models."""
    names = list(GraphNode.model_fields)

    assert names == ["id", "type", "version", "bindings", "locked", "title", "description", "fields", "display"]


def test_a_graph_is_nodes_and_display():
    names = list(Graph.model_fields)

    assert names == ["nodes", "display"]


def test_chrome_is_opaque():
    graph_node = GraphNode(id="n", type="t", version=1, display={"x": 10, "y": 20, "anything": [1]})

    assert graph_node.display == {"x": 10, "y": 20, "anything": [1]}


def test_the_record_is_the_schema():
    """A Graph dumps and loads through pydantic, bindings included. A
    Static whose value happens to look like a From comes back a Static,
    because Static nests its payload under `value`."""
    graph = Graph(
        nodes=[
            GraphNode(
                id="a",
                type="t",
                version=1,
                bindings={"x": Static({"refs": [{"node_id": "q", "field": "r"}]})},
                locked=("x",),
                title="A",
                fields={"x": FieldContent(title="X")},
            ),
            GraphNode(id="b", type="t", version=2, bindings={"y": From("a.result")}),
        ],
        display={"zoom": 1},
    )
    adapter = TypeAdapter(Graph)

    assert adapter.validate_python(adapter.dump_python(graph, mode="json")) == graph
def test_a_node_depends_on_every_node_its_edges_name():
    nodes = [
        GraphNode(id="a", type="t", version=1),
        GraphNode(id="b", type="t", version=1, bindings={"x": From("a.result")}),
    ]

    assert dependencies_of(nodes) == {"a": frozenset(), "b": frozenset({"a"})}


def test_a_static_binding_creates_no_dependency():
    nodes = [GraphNode(id="a", type="t", version=1, bindings={"x": Static("hi")})]

    assert dependencies_of(nodes) == {"a": frozenset()}


def test_two_refs_on_one_input_are_one_dependency_each():
    nodes = [
        GraphNode(
            id="b", type="t", version=1,
            bindings={"x": From("a.result", "c.result")},
        )
    ]

    assert dependencies_of(nodes)["b"] == frozenset({"a", "c"})


def test_two_edges_from_the_same_node_are_one_dependency():
    """A set, not a list: waiting twice for one node is waiting once."""
    nodes = [
        GraphNode(
            id="b", type="t", version=1,
            bindings={"x": From("a.result"), "y": From("a.other")},
        )
    ]

    assert dependencies_of(nodes)["b"] == frozenset({"a"})


class Txt(DType, str):
    id = "bindings-test-txt"
    display_title = "Text"


class Clock:
    """Something the run supplies, not the graph (`FromRun`)."""


@node("text-input", title="Text", description="A value the caller supplies", category="test")
def text_input(value: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")
) -> Annotated[Txt, Result(title="Text")]:
    """An ordinary node. Nothing marks it as one the graph exposes."""
    return value


@node("summarise", title="Opsummering", description="An ordinary node — neither end of what the graph exposes", category="test")
def summarise(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")
) -> Annotated[Txt, Result(title="Result")]:
    return text


@node("stamped", title="Stemplet", description="d", category="test")
def stamped(
    clock: FromRun[Clock],
    text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
) -> Annotated[Txt, Result(title="Result")]:
    """A node with a need: the run must provide a `Clock`."""
    return text


def _registry():
    reg = NodeRegistry([text_input, summarise, stamped])
    return reg


def _resolved(graph, registry=None):
    """What the compiler hands `derive_interface`: each node's
    computed interface, and the version its pin resolved to. Here it
    is the declaration read through the pin — none of these nodes shapes
    itself."""
    registry = registry or _registry()
    interfaces = {}
    versions = {}
    for graph_node in graph.nodes:
        version = registry[graph_node.type].versions[graph_node.version]
        versions[graph_node.id] = version
        interfaces[graph_node.id] = version.interface
    return interfaces, versions


def _graph(application_locked=(), language_bindings=None):
    """Two value holders and one summariser connected from the first:
    `application` and `language` are input nodes; `language` and `summary`
    are output nodes (nothing consumes them); `application` is consumed."""
    return Graph(
        nodes=[
            GraphNode(
                id="application", type="text-input", version=1, title="Application",
                fields={"value": FieldContent(title="Application")},
                bindings={"value": Static("")},
                locked=application_locked,
            ),
            GraphNode(
                id="language", type="text-input", version=1, title="Language",
                fields={"value": FieldContent(title="Language"), "result": FieldContent(title="Text")},
                bindings=language_bindings or {},
            ),
            GraphNode(
                id="summary", type="summarise", version=1, title="Opsummering",
                fields={"text": FieldContent(title="Text"), "result": FieldContent(title="Result")},
                bindings={"text": From("application.result")},
            ),
        ],
    )


def _interface(graph, registry=None):
    return derive_interface(graph, _resolved(graph, registry)[0], dependencies_of(graph.nodes))


def _locks(graph, registry=None):
    return tuple(p for p in CompiledGraph.from_graph(graph, registry or _registry()).problems if p.type == "unknown_locked_field")


def test_the_interface_is_derived_node_level():
    """Input nodes' unlocked handle-bearing fields in; output
    nodes' interfaces out, each under its address. Nothing is stored."""
    interface = _interface(_graph())

    assert _locks(_graph()) == ()
    assert [i.name for i in interface.inputs] == ["application.value", "language.value"]
    assert [o.name for o in interface.outputs] == ["language.result", "summary.result"]


def test_the_interface_is_the_record_a_node_version_declares():
    """One type at both scales. A graph returns a computed interface by
    address, so `returns` is `Mapping`; these nodes need nothing provided."""
    interface = _interface(_graph())

    assert isinstance(interface, Interface)
    assert interface.needs == frozenset()


def test_a_graph_level_name_is_the_address_a_ref_spells():
    """The name *is* the `Ref`, not a rendering of it: one value, one
    writer, and the key on every edge and in every caller's payload."""
    interface = _interface(_graph())

    assert interface.inputs[0].name == Ref("application", "value")
    assert interface.inputs[0].name == "application.value"
    assert interface.outputs[1].name == Ref("summary", "result")


def test_needs_is_the_union_of_the_nodes_needs():
    """What a run must provide to the graph is what its nodes need, by name."""
    graph = Graph(nodes=[
        GraphNode(id="a", type="stamped", version=1),
        GraphNode(id="b", type="stamped", version=1, bindings={"text": From("a.result")}),
        GraphNode(id="c", type="text-input", version=1),
    ])

    assert _interface(graph).needs == {Clock}


def test_a_field_reports_the_title_its_node_carries():
    """The node's title is the title. Nothing else changes: the
    declaration comes through whole, under its address."""
    interface = _interface(_graph())

    declared = interface.inputs[0]
    assert declared.name == "application.value"
    assert declared.title == "Application"
    assert declared.dtype is Txt
    # The widget comes through too, which is the whole reason this is the
    # node's own `Input` rather than a projection of it.
    assert type(declared.widget).__name__ == "Textarea"


def test_an_output_reports_the_title_its_placement_carries():
    assert _interface(_graph()).outputs[1].title == "Result"


def test_a_nodes_computed_interface_is_the_same_record_its_version_declares():
    """One record at every scale: a version declares an ``Interface``, compile
    computes one per node in the graph, and derives one for the graph."""
    from conductor import Interface

    interfaces, _ = _resolved(_graph())
    assert all(isinstance(i, Interface) for i in interfaces.values())


def test_nodes_order_decides_the_order():
    graph = _graph()
    reordered = Graph(nodes=[graph.nodes[1], graph.nodes[0], graph.nodes[2]])

    assert [i.name for i in _interface(reordered).inputs] == ["language.value", "application.value"]


def test_an_edge_into_any_field_makes_the_whole_node_static():
    """The rule is node-level: one edge in, and every other field of the
    node is author config — not offered, not fillable."""
    graph = _graph(language_bindings={"value": From("application.result")})

    assert [i.name for i in _interface(graph).inputs] == ["application.value"]


def test_a_locked_field_is_not_an_input():
    """The lock is the author's narrowing, and the derivation reads it."""
    graph = _graph(application_locked=("value",))

    assert _locks(graph) == ()
    assert [i.name for i in _interface(graph).inputs] == ["language.value"]


def test_a_partly_consumed_node_offers_no_outputs():
    """`application` feeds the summariser, so it is an intermediate step:
    its leftover handle is a byproduct, not a result."""
    assert not any(o.name == "application.result" for o in _interface(_graph()).outputs)


def test_a_field_with_no_handle_is_never_an_input():
    """A field with no handle has no way in, for an edge or a caller.
    No lock is needed to say so; the field is simply not on the surface."""

    @node("script", title="Python", description="d", category="test")
    def script(code: Annotated[Txt, Param(title="Kode", connectable=False, widget=Textarea())] = Txt("")
    ) -> Annotated[Txt, Result(title="Result")]:
        return code

    reg = _registry().with_nodes([script])
    graph = Graph(
        nodes=[GraphNode(id="s", type="script", version=1, title="Python",
                         fields={"code": FieldContent(title="Kode"), "result": FieldContent(title="Result")})],
    )

    interface = _interface(graph, reg)

    assert interface.inputs == ()
    assert [o.name for o in interface.outputs] == ["s.result"]


def test_a_locked_name_the_node_does_not_declare_is_a_problem():
    """A stale lock — the node's interface moved under it — is a state an editor
    mid-edit can be in, so it reports rather than raises. Non-fatal:
    it narrows nothing and blocks nothing, and the interface is derived
    without it."""
    graph = _graph(application_locked=("ghost",))

    assert [(p.type, p.loc[0], p.fatal) for p in _locks(graph)] == [("unknown_locked_field", "application", False)]
    assert [i.name for i in _interface(graph).inputs] == ["application.value", "language.value"]


def test_a_stale_lock_reports_on_a_connected_node_too():
    """A dormant lock stays out of the derivation — a connected node
    contributes no inputs — but a stale one is repairable wherever it
    sits, so it reports there as it would anywhere."""
    graph = _graph(language_bindings={"value": From("application.result")})
    graph = Graph(nodes=[
        graph_node.model_copy(update={"locked": ("ghost",)}) if graph_node.id == "language" else graph_node
        for graph_node in graph.nodes
    ])

    assert [(p.type, p.loc[0], p.fatal) for p in _locks(graph)] == [("unknown_locked_field", "language", False)]
    assert [i.name for i in _interface(graph).inputs] == ["application.value"]


def test_a_column_a_node_computed_is_derivable():
    """The interface is what the node answers, and a column on no
    declaration is as real a field of the surface as one it declared."""
    # `GraphNode.fields` is a `Mapping`, so the node is *built* with the
    # authored entry.
    graph = _graph()
    graph = Graph(nodes=[
        graph_node.model_copy(update={"fields": {**graph_node.fields, "name": FieldContent(title="Name")}})
        if graph_node.id == "summary" else graph_node
        for graph_node in graph.nodes
    ])
    interfaces, versions = _resolved(graph)
    declared = interfaces["summary"]
    interfaces["summary"] = dataclasses.replace(
        declared, outputs=(*declared.outputs, Output("name", Txt, title="Name")),
    )

    interface = derive_interface(graph, interfaces, dependencies_of(graph.nodes))

    assert [o.name for o in interface.outputs] == ["language.result", "summary.result", "summary.name"]


def test_a_field_on_a_node_compile_could_not_resolve_contributes_nothing():
    """The node carries its own fatal problem; a second one
    on the field would anchor the same fact twice. A pin that resolved to
    nothing is in neither map."""
    graph = _graph()
    interfaces, versions = _resolved(graph)
    del interfaces["application"], versions["application"]

    interface = derive_interface(graph, interfaces, dependencies_of(graph.nodes))

    assert [i.name for i in interface.inputs] == ["language.value"]


def test_an_unauthored_field_reads_the_declarations_title():
    """A hand-built graph (no editor copying titles) carries no
    `fields` entries; with nothing authored, the declaration's own title
    is the one value there is — no chain, nothing resolved."""
    graph = Graph(nodes=[GraphNode(id="x", type="text-input", version=1, title="X")])

    interface = _interface(graph)

    assert [(i.name, i.title) for i in interface.inputs] == [("x.value", "Text")]
    assert [(o.name, o.title) for o in interface.outputs] == [("x.result", "Text")]


def _echo_registry():
    @node("echo", title="Echo", description="d", category="test")
    def echo(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Result")]:
        return Txt(x.upper())

    registry = NodeRegistry([echo])
    return registry


def test_a_graph_of_bindings_compiles_and_runs():
    graph = Graph(
        nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": Static("hi")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": From("a.result")}),
        ],
    )
    compiled = CompiledGraph.from_graph(graph=graph, registry=_echo_registry())
    results = compiled.results(compiled.run_sync())

    assert results["a"]["result"] == "HI"
    assert results["b"]["result"] == "HI"


def test_an_unbound_input_falls_back_to_its_declared_default():
    """Absence is the only "nothing binds this" state there is."""
    graph = Graph(nodes=[GraphNode(id="a", type="echo", version=1)])

    compiled = CompiledGraph.from_graph(graph=graph, registry=_echo_registry())

    assert compiled.results(compiled.run_sync())["a"]["result"] == ""


def test_a_branch_not_taken_is_skipped_downstream():
    """A branch is an output; SKIPPED on it skips what hangs off it."""

    @dataclass(frozen=True)
    class Answer:
        yes: Annotated[Txt, Result(title="Yes", exclusive_group="answer")]
        no: Annotated[Txt, Result(title="No", exclusive_group="answer")]

    @node("gate", title="Gate", description="d", category="test")
    def gate(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")
    ) -> Answer:
        return Answer(yes=x, no=SKIPPED) if x else Answer(yes=SKIPPED, no=x)

    registry = _echo_registry().with_nodes([gate])
    graph = Graph(
        nodes=[
            GraphNode(id="g", type="gate", version=1, bindings={"x": Static("hi")}),
            GraphNode(id="yes", type="echo", version=1, bindings={"x": From("g.yes")}),
            GraphNode(id="no", type="echo", version=1, bindings={"x": From("g.no")}),
        ],
    )
    compiled = CompiledGraph.from_graph(graph=graph, registry=registry)
    results = compiled.results(compiled.run_sync())

    assert results["yes"]["result"] == "HI"
    # The results a leg ends with omit a skipped node.
    assert "no" not in results


def test_compile_takes_a_graph_and_a_registry_and_nothing_else_positional():
    params = inspect.signature(CompiledGraph.from_graph).parameters

    assert list(params)[:2] == ["graph", "registry"]
    for gone in ("nodes", "edges", "extension_resolver", "subprocess_registry"):
        assert gone not in params, gone
