"""The graph record: its shape, how it is saved, and what loading it reports.

A record is ``{"nodes": [...], "edges": [[from, to], ...]}``. Compile reads it
through ``load_record``, which gathers each node's edges, values and defaults
onto its inputs.
"""

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    CompiledGraph,
    DType,
    FromRun,
    GraphRecord,
    Interface,
    NodeRecord,
    Output,
    Param,
    Ref,
    Result,
    node,
)
from conductor._internal.graph.binding import From, Static
from conductor._internal.graph.graph_interface import derive_interface
from conductor._internal.graph.graph_records import load_record
from conductor._internal.graph.topology import dependencies_of
from conductor.widgets import Textarea
from helpers import registry_of, with_added
from pydantic import ValidationError

# -- the record's shape ---------------------------------------------------------


def test_a_node_is_its_id_its_node_and_what_is_typed_in():
    assert list(NodeRecord.model_fields) == ["id", "node", "version", "values", "defaults", "display"]
    assert list(GraphRecord.model_fields) == ["nodes", "edges"]


def test_a_bare_node_saves_as_its_id_and_node():
    """Empty fields and the version 1 are left out when saved."""
    assert NodeRecord(id="a", node="upper").model_dump() == {"id": "a", "node": "upper"}
    assert NodeRecord(id="a", node="upper", version=2, values={"text": "hi"}).model_dump() == {"id": "a", "node": "upper", "version": 2, "values": {"text": "hi"}}


def test_the_record_reads_back_what_it_saved():
    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node="t", values={"x": {"refs": ["q.r"]}}, display={"x": 10}),
        NodeRecord(id="b", node="t", version=2, defaults={"y": 1}),
    ], edges=[("a.result", "b.y")])

    assert GraphRecord.model_validate_json(graph.model_dump_json()) == graph
    assert graph.model_dump(mode="json")["edges"] == [["a.result", "b.y"]]


def test_display_is_the_hosts_own_and_takes_any_key():
    assert NodeRecord(id="n", node="t", display={"x": 10, "y": 20, "anything": [1]}).display == {"x": 10, "y": 20, "anything": [1]}


def test_an_edge_is_two_addresses():
    with pytest.raises(ValidationError):
        GraphRecord(edges=[("a", "b.x")])
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": [{"id": "a", "node": "t", "bindings": {}}]})


# -- loading: edges, values and defaults onto the inputs --------------------------


def _loaded(graph: GraphRecord):
    nodes, problems = load_record(graph)
    return {loaded.id: loaded for loaded in nodes}, [(p.type, p.loc) for p in problems]


def test_loading_puts_each_edge_value_and_default_on_its_input():
    nodes, problems = _loaded(GraphRecord(nodes=[
        NodeRecord(id="a", node="t", values={"x": 1}, defaults={"y": 2}),
        NodeRecord(id="b", node="t"),
    ], edges=[("a.result", "b.x")]))

    assert problems == []
    assert nodes["a"].bindings == {"x": Static(1), "y": Static(2)} and nodes["a"].fixed == {"x"}
    assert nodes["b"].bindings == {"x": From("a.result")} and nodes["b"].fixed == frozenset()


def test_several_edges_into_one_input_gather_in_node_order_whatever_the_list_order():
    nodes = [NodeRecord(id="first", node="t"), NodeRecord(id="second", node="t"), NodeRecord(id="j", node="t")]
    one_way = _loaded(GraphRecord(nodes=nodes, edges=[("second.result", "j.texts"), ("first.result", "j.texts")]))[0]
    other_way = _loaded(GraphRecord(nodes=nodes, edges=[("first.result", "j.texts"), ("second.result", "j.texts")]))[0]

    assert one_way["j"].bindings["texts"] == other_way["j"].bindings["texts"] == From("first.result", "second.result")


def test_two_outputs_of_one_node_into_one_input_gather_in_the_order_of_their_addresses():
    nodes = [NodeRecord(id="g", node="t"), NodeRecord(id="j", node="t")]

    assert _loaded(GraphRecord(nodes=nodes, edges=[("g.yes", "j.texts"), ("g.no", "j.texts")]))[0]["j"].bindings["texts"] == From("g.no", "g.yes")


def test_edges_are_a_set_so_their_order_never_tells_two_records_apart():
    nodes = [NodeRecord(id="a", node="t"), NodeRecord(id="b", node="t")]
    one_way = GraphRecord(nodes=nodes, edges=[("a.x", "b.x"), ("a.y", "b.y")])
    other_way = GraphRecord(nodes=nodes, edges=[("a.y", "b.y"), ("a.x", "b.x")])

    assert one_way == other_way and one_way.model_dump_json() == other_way.model_dump_json()


def test_a_repeated_edge_is_a_problem_and_read_once():
    nodes, problems = _loaded(GraphRecord(nodes=[NodeRecord(id="a", node="t"), NodeRecord(id="b", node="t")], edges=[("a.result", "b.x"), ("a.result", "b.x")]))

    assert problems == [("repeated_edge", ("b", "x"))]
    assert nodes["b"].bindings == {"x": From("a.result")}


def test_an_edge_into_a_node_the_graph_does_not_have_is_a_problem():
    assert _loaded(GraphRecord(nodes=[NodeRecord(id="a", node="t")], edges=[("a.result", "gone.x")]))[1] == [("edge_into_unknown_node", ("gone", "x"))]


def test_an_input_set_two_ways_is_a_problem_and_the_edge_wins():
    nodes, problems = _loaded(GraphRecord(nodes=[
        NodeRecord(id="a", node="t"),
        NodeRecord(id="b", node="t", values={"x": 1, "y": 2}, defaults={"y": 3}),
    ], edges=[("a.result", "b.x")]))

    assert sorted(problems) == [("value_and_default", ("b", "y")), ("value_and_edge", ("b", "x"))]
    assert nodes["b"].bindings["x"] == From("a.result")


def test_what_loading_reports_is_on_the_compiled_graph():
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="a", node="text-input"),
        NodeRecord(id="b", node="summarise", values={"text": "x"}),
    ], edges=[("a.result", "b.text"), ("a.result", "b.text"), ("a.result", "c.text")]), _registry())

    assert [(p.type, p.loc, p.fatal) for p in compiled.problems] == [
        ("edge_into_unknown_node", ("c", "text"), True), ("repeated_edge", ("b", "text"), True), ("value_and_edge", ("b", "text"), True),
    ]


def test_a_node_id_with_a_dot_or_a_slash_is_a_problem_on_the_graph():
    """A Ref reads as 'node.field', so '.' can't be in an id; '/' is an inner node's path. Both are the one refusal, on the stored graph."""
    for bad in ("a.b", "a/b"):
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id=bad, node="upper"),
        ]), _registry())
        assert [(p.type, p.loc) for p in compiled.problems] == [("invalid_node_id", (bad,))]
        assert compiled.node(bad).interface is None


def test_a_slash_is_the_namespace_separator():
    """Compile puts an embedded graph's nodes under the graph node's name
    — `approve/check` — so a Ref still reads one way: the split is at the
    first dot, and there is none in the namespace."""
    assert Ref("approve/check", "amount") == "approve/check.amount"
    assert Ref("approve/check.amount").node_id == "approve/check"


# -- which nodes wait for which ----------------------------------------------------


def _dependencies(graph: GraphRecord):
    return dependencies_of(load_record(graph)[0])


def test_a_node_depends_on_every_node_its_edges_name():
    graph = GraphRecord(nodes=[NodeRecord(id="a", node="t"), NodeRecord(id="b", node="t", values={"y": 1})], edges=[("a.result", "b.x")])

    assert _dependencies(graph) == {"a": frozenset(), "b": frozenset({"a"})}


def test_two_edges_from_one_node_are_one_dependency():
    """A set, not a list: waiting twice for one node is waiting once."""
    graph = GraphRecord(nodes=[NodeRecord(id="a", node="t"), NodeRecord(id="b", node="t")], edges=[("a.result", "b.x"), ("a.other", "b.y")])

    assert _dependencies(graph)["b"] == frozenset({"a"})


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
    reg = registry_of(text_input, summarise, stamped)
    return reg


def _graph(language=None, application_values=None):
    """Two value holders and one summariser connected from the first:
    `application` and `language` are input nodes; `language` and `summary`
    are output nodes (nothing consumes them); `application` is consumed."""
    return GraphRecord(nodes=[
        NodeRecord(id="application", node="text-input", defaults={"value": ""}, values=application_values or {}),
        NodeRecord(id="language", node="text-input"),
        NodeRecord(id="summary", node="summarise"),
    ], edges=[("application.result", "summary.text"), *([("application.result", "language.value")] if language == "fed" else [])])


def _interface(graph, registry=None):
    return CompiledGraph.from_graph(graph, registry or _registry()).interface


def _resolved(graph, registry=None):
    """Each loaded node and the interface its version declares: what ``derive_interface`` is handed."""
    registry = registry or _registry()
    nodes = load_record(graph)[0]
    return nodes, {loaded.id: registry[loaded.node].versions[loaded.version].interface for loaded in nodes}


def test_the_interface_is_derived_node_level():
    """Input nodes' handle-bearing fields no value fixes in; output
    nodes' interfaces out, each under its address. Nothing is stored."""
    interface = _interface(_graph())

    assert [i.name for i in interface.inputs] == ["application.value", "language.value"]
    assert [o.name for o in interface.outputs] == ["language.result", "summary.result"]


def test_the_interface_is_the_record_a_node_version_declares():
    """One type at both scales; these nodes need nothing provided."""
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
    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node="stamped"),
        NodeRecord(id="b", node="stamped"),
        NodeRecord(id="c", node="text-input"),
    ], edges=[("a.result", "b.text")])

    assert _interface(graph).needs == {Clock}


def test_a_field_is_the_nodes_declaration_under_its_address():
    """The declaration comes through whole, its title and widget included."""
    declared = _interface(_graph()).inputs[0]

    assert declared.name == "application.value"
    assert declared.title == "Text"
    assert declared.dtype is Txt
    assert type(declared.widget).__name__ == "Textarea"


def test_nodes_order_decides_the_order():
    graph = _graph()
    reordered = graph.model_copy(update={"nodes": (graph.nodes[1], graph.nodes[0], graph.nodes[2])})

    assert [i.name for i in _interface(reordered).inputs] == ["language.value", "application.value"]


def test_an_edge_into_any_field_makes_the_whole_node_closed():
    """The rule is node-level: one edge in, and every other field of the
    node is the author's: not offered, not fillable."""
    assert [i.name for i in _interface(_graph(language="fed")).inputs] == ["application.value"]


def test_a_value_closes_its_input_and_a_default_leaves_it_open():
    graph = GraphRecord(nodes=[
        NodeRecord(id="fixed", node="text-input", values={"value": "a"}),
        NodeRecord(id="prefilled", node="text-input", defaults={"value": "b"}),
    ])

    assert [i.name for i in _interface(graph).inputs] == ["prefilled.value"]


def test_a_partly_consumed_node_offers_no_outputs():
    """`application` feeds the summariser, so it is an intermediate step:
    its leftover handle is a byproduct, not a result."""
    assert not any(o.name == "application.result" for o in _interface(_graph()).outputs)


def test_a_field_with_no_handle_is_never_an_input():
    """A field with no handle has no way in, for an edge or a caller."""

    @node("script", title="Python", description="d", category="test")
    def script(code: Annotated[Txt, Param(title="Kode", connectable=False, widget=Textarea())] = Txt("")
    ) -> Annotated[Txt, Result(title="Result")]:
        return code

    interface = _interface(GraphRecord(nodes=[NodeRecord(id="s", node="script")]), with_added(_registry(), script))

    assert interface.inputs == ()
    assert [o.name for o in interface.outputs] == ["s.result"]


def test_a_column_a_node_computed_is_derivable():
    """The interface is what the node answers, and a column on no
    declaration is as real a field of the surface as one it declared."""
    nodes, interfaces = _resolved(_graph())
    declared = interfaces["summary"]
    interfaces["summary"] = dataclasses.replace(declared, outputs=(*declared.outputs, Output("name", Txt, title="Name")))

    interface = derive_interface(nodes, interfaces, dependencies_of(nodes))

    assert [o.name for o in interface.outputs] == ["language.result", "summary.result", "summary.name"]


def test_a_field_on_a_node_compile_could_not_resolve_contributes_nothing():
    """The node carries its own fatal problem; a second one on the field
    would anchor the same fact twice."""
    nodes, interfaces = _resolved(_graph())
    del interfaces["application"]

    interface = derive_interface(nodes, interfaces, dependencies_of(nodes))

    assert [i.name for i in interface.inputs] == ["language.value"]


def _echo_registry():
    @node("echo", title="Echo", description="d", category="test")
    def echo(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Result")]:
        return Txt(x.upper())

    registry = registry_of(echo)
    return registry


def test_a_record_compiles_and_runs():
    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node="echo", values={"x": "hi"}),
        NodeRecord(id="b", node="echo"),
    ], edges=[("a.result", "b.x")])
    compiled = CompiledGraph.from_graph(graph=graph, registry=_echo_registry())
    results = compiled.results(compiled.run_sync())

    assert results["a"]["result"] == "HI"
    assert results["b"]["result"] == "HI"


def test_an_unbound_input_falls_back_to_its_declared_default():
    """Absence is the only "nothing binds this" state there is."""
    graph = GraphRecord(nodes=[
        NodeRecord(id="a", node="echo"),
    ])

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

    registry = with_added(_echo_registry(), gate)
    graph = GraphRecord(nodes=[
        NodeRecord(id="g", node="gate", values={"x": "hi"}),
        NodeRecord(id="yes", node="echo"),
        NodeRecord(id="no", node="echo"),
    ], edges=[("g.yes", "yes.x"), ("g.no", "no.x")])
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
