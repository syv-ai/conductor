import dataclasses
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import NodeRegistry
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.execution.engine import execute_sync
from conductor.graph.binding import Edges, Static, static_values
from conductor.graph.compiler import compile as compile_graph
from conductor.graph.model import FieldContent, Graph, GraphNode
from conductor.graph.topology import dependencies_of
from conductor.graph.views import derive_interface, is_input_node, lock_problems
from conductor.interface import Interface, Provided
from conductor.metadata import Output, Roster
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.returns import Result
from conductor.widgets import Textarea
from pydantic import TypeAdapter


def test_an_edge_carries_ordered_refs():
    edge = Edges(refs=(Ref("a", "result"), Ref("b", "result")))

    assert [r.node_id for r in edge.refs] == ["a", "b"]


def test_edge_order_is_operand_order():
    first = Edges(refs=(Ref("a", "r"), Ref("b", "r")))
    second = Edges(refs=(Ref("b", "r"), Ref("a", "r")))
    assert first != second


def test_there_are_exactly_two_variants():
    """No Port (a caller's value replaces a Static for one run), no Guard
    (a branch is an output), no per-ref enable flag."""
    import conductor.graph.binding as bindings

    assert not hasattr(bindings, "Port")
    assert not hasattr(bindings, "Guard")
    assert not hasattr(Edges(refs=()), "when")


def test_absence_means_the_declared_default_applies():
    """One state, not two — there is nothing to dismiss."""
    bindings = {"b": Static(value=3)}

    assert "a" not in bindings


def test_static_values_extracts_only_static_bindings():
    bindings = {
        "text": Static(value="hi"),
        "number": Static(value=3),
        "source": Edges(refs=(Ref("a", "result"),)),
    }
    assert static_values(bindings) == {"text": "hi", "number": 3}


def test_bindings_are_frozen():
    with pytest.raises(Exception):
        Static(value=1).value = 2
def test_a_placement_id_contains_no_dot():
    """A Ref reads as 'node.field' everywhere it is spelled — an index
    name, a caller's payload key — so an id with a '.' is refused."""
    with pytest.raises(ValueError, match="contains '.'"):
        GraphNode(id="a.b", type="upper", version=1)


def test_a_slash_is_the_namespace_separator():
    """The compiler inlines an embedded flow under its placement's name —
    `approve/check` — so `/` is legal in an id and a Ref still reads one way:
    the split is at the first dot, and there is none in the namespace."""
    inner = GraphNode(id="approve/check", type="upper", version=1)

    assert inner.id == "approve/check"
    assert Ref("approve/check", "amount") == "approve/check.amount"
    assert Ref("approve/check.amount").node_id == "approve/check"


def test_a_node_stores_bindings_and_derives_data():
    node = GraphNode(
        id="n1",
        type="translate",
        version=1,
        bindings={
            "text": Static(value="hi"),
            "source": Edges(refs=(Ref("n0", "result"),)),
        },
    )
    assert node.data == {"text": "hi"}


def test_a_node_is_behaviour_content_and_chrome():
    """The three categories, and nothing from the deleted models."""
    names = [f.name for f in dataclasses.fields(GraphNode)]

    assert names == ["id", "type", "version", "bindings", "locked", "title", "description", "fields", "display"]


def test_a_flow_is_nodes_and_display():
    names = [f.name for f in dataclasses.fields(Graph)]

    assert names == ["nodes", "display"]


def test_chrome_is_opaque():
    node = GraphNode(id="n", type="t", version=1, display={"x": 10, "y": 20, "anything": [1]})

    assert node.display == {"x": 10, "y": 20, "anything": [1]}


def test_the_record_is_the_schema():
    """A Graph dumps and loads through pydantic, bindings included. A
    Static whose value happens to look like a Edges comes back a Static,
    because Static nests its payload under `value`."""
    graph = Graph(
        nodes=[
            GraphNode(
                id="a",
                type="t",
                version=1,
                bindings={"x": Static(value={"refs": [{"node_id": "q", "field": "r"}]})},
                locked=("x",),
                title="A",
                fields={"x": FieldContent(title="X")},
            ),
            GraphNode(id="b", type="t", version=2, bindings={"y": Edges(refs=(Ref("a", "result"),))}),
        ],
        display={"zoom": 1},
    )
    adapter = TypeAdapter(Graph)

    assert adapter.validate_python(adapter.dump_python(graph, mode="json")) == graph
def test_a_node_depends_on_every_node_its_edges_name():
    nodes = [
        GraphNode(id="a", type="t", version=1),
        GraphNode(id="b", type="t", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),))}),
    ]

    assert dependencies_of(nodes) == {"a": frozenset(), "b": frozenset({"a"})}


def test_a_static_binding_creates_no_dependency():
    nodes = [GraphNode(id="a", type="t", version=1, bindings={"x": Static(value="hi")})]

    assert dependencies_of(nodes) == {"a": frozenset()}


def test_two_refs_on_one_input_are_one_dependency_each():
    nodes = [
        GraphNode(
            id="b", type="t", version=1,
            bindings={"x": Edges(refs=(Ref("a", "result"), Ref("c", "result")))},
        )
    ]

    assert dependencies_of(nodes)["b"] == frozenset({"a", "c"})


def test_two_edges_from_the_same_node_are_one_dependency():
    """A set, not a list: waiting twice for one node is waiting once."""
    nodes = [
        GraphNode(
            id="b", type="t", version=1,
            bindings={"x": Edges(refs=(Ref("a", "result"),)), "y": Edges(refs=(Ref("a", "other"),))},
        )
    ]

    assert dependencies_of(nodes)["b"] == frozenset({"a"})


def test_conductor_defines_no_edge_type():
    """The canvas derives its own edges; there is nothing here to convert."""
    import conductor.graph.model as model

    assert not hasattr(model, "GraphEdge")
class Txt(DType, str):
    id = "bindings-test-txt"
    title = "Text"


class Clock:
    """Something the run supplies, not the flow (`Provided`)."""


class TextInput(NodeDefinition):
    """An ordinary node. Nothing marks it as one the flow exposes."""

    id = "text-input"
    title = "Text"
    description = "A value the caller supplies"
    category = "test"

    def run(
        self, value: Annotated[Txt, Textarea(title="Text")] = Txt("")
    ) -> Annotated[Txt, Result(title="Text")]:
        return value


class Summarise(NodeDefinition):
    id = "summarise"
    title = "Opsummering"
    description = "An ordinary node — neither end of what the flow exposes"
    category = "test"

    def run(
        self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")
    ) -> Annotated[Txt, Result(title="Result")]:
        return text


class Stamped(NodeDefinition):
    """A node with a need: the run must provide a `Clock`."""

    id = "stamped"
    title = "Stemplet"
    description = "d"
    category = "test"

    def run(
        self,
        clock: Annotated[Clock, Provided()],
        text: Annotated[Txt, Textarea(title="Text")] = Txt(""),
    ) -> Annotated[Txt, Result(title="Result")]:
        return text


def _registry():
    reg = NodeRegistry()
    reg.register(TextInput)
    reg.register(Summarise)
    reg.register(Stamped)
    return reg


def _resolved(graph, registry=None):
    """What the compiler hands `derive_interface`: each placement's
    effective roster, and the version its pin resolved to. Here the roster
    is the declaration read through the pin — none of these nodes shapes
    itself."""
    registry = registry or _registry()
    rosters = {}
    versions = {}
    for node in graph.nodes:
        version = registry.get(node.type).versions[node.version]
        versions[node.id] = version
        rosters[node.id] = Roster(inputs=version.interface.inputs, outputs=version.interface.outputs)
    return rosters, versions


def _graph(application_locked=(), language_bindings=None):
    """Two value holders and one summariser connected from the first:
    `application` and `language` are input nodes; `language` and `summary`
    are output nodes (nothing consumes them); `application` is consumed."""
    return Graph(
        nodes=[
            GraphNode(
                id="application", type="text-input", version=1, title="Application",
                fields={"value": FieldContent(title="Application")},
                bindings={"value": Static(value="")},
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
                bindings={"text": Edges(refs=(Ref("application", "result"),))},
            ),
        ],
    )


def _interface(graph, registry=None):
    return derive_interface(graph, *_resolved(graph, registry), dependencies_of(graph.nodes))


def _locks(graph, registry=None):
    rosters, _ = _resolved(graph, registry)
    return lock_problems({n.id: n for n in graph.nodes}, rosters)


def test_the_interface_is_derived_node_level():
    """Input nodes' unlocked handle-bearing fields in; output
    nodes' rosters out, each under its address. Nothing is stored."""
    interface = _interface(_graph())

    assert _locks(_graph()) == ()
    assert [i.name for i in interface.inputs] == ["application.value", "language.value"]
    assert [o.name for o in interface.outputs] == ["language.result", "summary.result"]


def test_the_interface_is_the_record_a_node_version_declares():
    """One type at both scales. A flow returns a computed roster by
    address, so `returns` is `Mapping`; these nodes need nothing provided."""
    interface = _interface(_graph())

    assert isinstance(interface, Interface)
    assert interface.returns is Mapping
    assert interface.needs == {}


def test_a_flow_level_name_is_the_address_a_ref_spells():
    """The name *is* the `Ref`, not a rendering of it: one value, one
    writer, and the key on every edge and in every caller's payload."""
    interface = _interface(_graph())

    assert interface.inputs[0].name == Ref("application", "value")
    assert interface.inputs[0].name == "application.value"
    assert interface.outputs[1].name == Ref("summary", "result")


def test_needs_is_the_union_of_the_placements_needs():
    """What a run must provide to the flow is what its nodes need, by name."""
    graph = Graph(nodes=[
        GraphNode(id="a", type="stamped", version=1),
        GraphNode(id="b", type="stamped", version=1, bindings={"text": Edges(refs=(Ref("a", "result"),))}),
        GraphNode(id="c", type="text-input", version=1),
    ])

    assert _interface(graph).needs == {"clock": Clock}


def test_a_field_reports_the_title_its_placement_carries():
    """The placement's title is the title. Nothing else changes: the
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


def test_a_roster_is_the_two_tuples_a_nodes_hooks_answer():
    import dataclasses

    assert [f.name for f in dataclasses.fields(Roster)] == ["inputs", "outputs"]


def test_nodes_order_decides_the_order():
    graph = _graph()
    reordered = Graph(nodes=[graph.nodes[1], graph.nodes[0], graph.nodes[2]])

    assert [i.name for i in _interface(reordered).inputs] == ["language.value", "application.value"]


def test_is_input_node_is_the_one_home_of_the_predicate():
    """A Static does not disqualify a placement; any Edges does."""
    graph = _graph()
    assert is_input_node(graph.nodes[0]) and is_input_node(graph.nodes[1])
    assert not is_input_node(graph.nodes[2])


def test_an_edge_into_any_field_makes_the_whole_node_static():
    """The rule is node-level: one edge in, and every other field of the
    placement is author config — not offered, not fillable."""
    graph = _graph(language_bindings={"value": Edges(refs=(Ref("application", "result"),))})

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

    class Script(NodeDefinition):
        id = "script"
        title = "Python"
        description = "d"
        category = "test"

        def run(
            self, code: Annotated[Txt, Textarea(title="Kode", show_handle=False)] = Txt("")
        ) -> Annotated[Txt, Result(title="Result")]:
            return code

    reg = _registry()
    reg.register(Script)
    graph = Graph(
        nodes=[GraphNode(id="s", type="script", version=1, title="Python",
                         fields={"code": FieldContent(title="Kode"), "result": FieldContent(title="Result")})],
    )

    interface = _interface(graph, reg)

    assert interface.inputs == ()
    assert [o.name for o in interface.outputs] == ["s.result"]


def test_a_locked_name_the_node_does_not_declare_is_a_problem():
    """A stale lock — the roster moved under it — is a state an editor
    mid-edit can be in, so it reports rather than raises. Non-fatal:
    it narrows nothing and blocks nothing, and the interface is derived
    without it."""
    graph = _graph(application_locked=("ghost",))

    assert [(p.code, p.node_id, p.fatal) for p in _locks(graph)] == [("unknown_locked_field", "application", False)]
    assert [i.name for i in _interface(graph).inputs] == ["application.value", "language.value"]


def test_a_stale_lock_reports_on_a_connected_placement_too():
    """A dormant lock stays out of the derivation — a connected placement
    contributes no inputs — but a stale one is repairable wherever it
    sits, so it reports there as it would anywhere."""
    import dataclasses

    graph = _graph(language_bindings={"value": Edges(refs=(Ref("application", "result"),))})
    graph = Graph(nodes=[
        dataclasses.replace(node, locked=("ghost",)) if node.id == "language" else node
        for node in graph.nodes
    ])

    assert [(p.code, p.node_id, p.fatal) for p in _locks(graph)] == [("unknown_locked_field", "language", False)]
    assert [i.name for i in _interface(graph).inputs] == ["application.value"]


def test_a_column_a_node_computed_is_derivable():
    """The roster is what the node answers, and a column on no
    declaration is as real a field of the surface as one it declared."""
    import dataclasses

    # `GraphNode.fields` is a `Mapping`, so the placement is *built* with the
    # authored entry.
    graph = _graph()
    graph = Graph(nodes=[
        dataclasses.replace(node, fields={**node.fields, "name": FieldContent(title="Name")})
        if node.id == "summary" else node
        for node in graph.nodes
    ])
    rosters, versions = _resolved(graph)
    declared = rosters["summary"]
    rosters["summary"] = Roster(
        inputs=declared.inputs,
        outputs=(*declared.outputs, Output(name="name", dtype=Txt, title="Name")),
    )

    interface = derive_interface(graph, rosters, versions, dependencies_of(graph.nodes))

    assert [o.name for o in interface.outputs] == ["language.result", "summary.result", "summary.name"]


def test_a_field_on_a_placement_compile_could_not_resolve_contributes_nothing():
    """The placement carries its own fatal problem; a second one
    on the field would anchor the same fact twice. A pin that resolved to
    nothing is in neither map."""
    graph = _graph()
    rosters, versions = _resolved(graph)
    del rosters["application"], versions["application"]

    interface = derive_interface(graph, rosters, versions, dependencies_of(graph.nodes))

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
    class Echo(NodeDefinition):
        id = "echo"
        title = "Echo"
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Annotated[Txt, Result(title="Result")]:
            return Txt(x.upper())

    registry = NodeRegistry()
    registry.register(Echo)
    return registry


def test_a_flow_of_bindings_compiles_and_runs():
    graph = Graph(
        nodes=[
            GraphNode(id="a", type="echo", version=1, bindings={"x": Static(value="hi")}),
            GraphNode(id="b", type="echo", version=1, bindings={"x": Edges(refs=(Ref("a", "result"),))}),
        ],
    )
    results = execute_sync(compile_graph(graph=graph, registry=_echo_registry()))

    assert results["a"]["result"] == "HI"
    assert results["b"]["result"] == "HI"


def test_an_unbound_input_falls_back_to_its_declared_default():
    """Absence is the only "nothing binds this" state there is."""
    graph = Graph(nodes=[GraphNode(id="a", type="echo", version=1)])

    assert execute_sync(compile_graph(graph=graph, registry=_echo_registry()))["a"]["result"] == ""


def test_a_branch_not_taken_is_skipped_downstream():
    """A branch is an output; SKIPPED on it skips what hangs off it."""

    @dataclass(frozen=True)
    class Answer:
        yes: Annotated[Txt, Result(title="Yes", choice="answer")]
        no: Annotated[Txt, Result(title="No", choice="answer")]

    class Gate(NodeDefinition):
        id = "gate"
        title = "Gate"
        description = "d"
        category = "test"

        def run(
            self, x: Annotated[Txt, Textarea(title="X")] = Txt("")
        ) -> Answer:
            return Answer(yes=x, no=SKIPPED) if x else Answer(yes=SKIPPED, no=x)

    registry = _echo_registry()
    registry.register(Gate)
    graph = Graph(
        nodes=[
            GraphNode(id="g", type="gate", version=1, bindings={"x": Static(value="hi")}),
            GraphNode(id="yes", type="echo", version=1, bindings={"x": Edges(refs=(Ref("g", "yes"),))}),
            GraphNode(id="no", type="echo", version=1, bindings={"x": Edges(refs=(Ref("g", "no"),))}),
        ],
    )
    results = execute_sync(compile_graph(graph=graph, registry=registry))

    assert results["yes"]["result"] == "HI"
    # The aggregated results of ``execute_sync`` omit a skipped node.
    assert "no" not in results


def test_compile_takes_a_graph_and_a_registry_and_nothing_else_positional():
    params = inspect.signature(compile_graph).parameters

    assert list(params)[:2] == ["graph", "registry"]
    for gone in ("nodes", "edges", "extension_resolver", "subprocess_registry"):
        assert gone not in params, gone
