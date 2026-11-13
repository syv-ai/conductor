"""The graph record: its shape, what the record door refuses, the problems compiling it reports, and ``make_graph``.

A graph record is ``{"nodes": {id: {"node": ..., "inputs": {...}}}, "outputs": {name: {"node": ..., "output":
...}}}``. ``Graph.compile`` reads it into a ``Graph``. A record that is not this shape is refused at the door with
``invalid_record``; what a record of this shape says that cannot be meant is a problem of that graph, never an
exception.
"""

import json
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    Address,
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeRecord,
    Param,
    Problem,
    Run,
    Series,
    Table,
    Upgrade,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import ValidationError

from tests.test_core.nodes import registry_of


@node("text", title="Text in", description="For a test.")
def text_in(value: str = "") -> str:
    return value


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class YesNo:
    yes: Annotated[str, Param(exclusive_group="answer")]
    no: Annotated[str, Param(exclusive_group="answer")]


@node("decide", title="Decide", description="For a test.")
def decide(text: str) -> YesNo:
    return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)


@node("both", title="Both", description="For a test.")
def both(yes: str, no: str) -> str:
    return yes + no


@dataclass(frozen=True)
class Two:
    zed: str
    alpha: str


@node("two", title="Two", description="For a test.")
def two(text: str) -> Two:
    return Two(f"zed {text}", f"alpha {text}")


REGISTRY = registry_of(text_in, upper, join, decide, both, two)


def _problems(record: GraphRecord) -> list[tuple[str, tuple]]:
    return [(p.type, p.loc) for p in Graph.compile(record, REGISTRY).problems]


def _refused(record: Any) -> list[tuple[str, tuple, str]]:
    """The problems the record door raises for ``record``, JSON text or plain data, as ``(type, loc, ctx.error)``."""
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(record if isinstance(record, str) else json.dumps(record), REGISTRY)
    return [(p.type, p.loc, p.ctx["error"]) for p in refused.value.errors()]


# -- the record's shape ---------------------------------------------------------


def test_a_node_record_and_a_graph_record_have_these_fields_in_this_order():
    assert list(NodeRecord.model_fields) == ["node", "version", "inputs", "display"]
    assert list(GraphRecord.model_fields) == ["nodes", "outputs"]


def test_a_bare_node_saves_as_its_node():
    """Empty fields and the version 1 are left out when saved."""
    assert NodeRecord(node="upper").model_dump() == {"node": "upper"}
    assert NodeRecord(node="upper", version=2, inputs={"text": InputEntry(constant="hi")}).model_dump(mode="json") == {
        "node": "upper", "version": 2, "inputs": {"text": {"constant": "hi"}},
    }
    graph = GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "version": 1, "display": {}, "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "a", "output": "result"}}})
    assert graph.model_dump(mode="json") == {"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "a", "output": "result"}}}


def test_each_kind_of_input_entry_is_its_own_class_and_a_dict_reads_as_the_same():
    written = NodeRecord(node="both", inputs={
        "yes": InputEntry(constant="y"),
        "no": InputEntry(graph_input="no", default="n"),
    })
    edged = NodeRecord(node="upper", inputs={"text": InputEntry(sources=(Address("a", "result", "email"),))})

    assert NodeRecord.model_validate({"node": "both", "inputs": {"yes": {"constant": "y"}, "no": {"graph_input": "no", "default": "n"}}}) == written
    assert NodeRecord.model_validate({"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result", "field": "email"}]}}}) == edged
    assert edged.model_dump(mode="json")["inputs"] == {"text": {"sources": [{"node": "a", "output": "result", "field": "email"}]}}
    assert InputEntry(graph_input="no").model_dump(mode="json") == {"graph_input": "no"}


def test_the_record_reads_back_what_it_saved():
    graph = GraphRecord.model_validate({"nodes": {"a": {"node": "t", "inputs": {"x": {"constant": {"refs": ["q.r"]}}}, "display": {"x": 10}}, "b": {"node": "t", "version": 2, "inputs": {"y": {"sources": [{"node": "a", "output": "result"}]}}}}})

    assert GraphRecord.model_validate_json(graph.model_dump_json()) == graph
    assert graph.nodes["b"].inputs["y"] == InputEntry(sources=(Address("a", "result"),))


def test_display_is_the_hosts_own_and_takes_any_key():
    assert NodeRecord(node="t", display={"x": 10, "y": 20, "anything": [1]}).display == {"x": 10, "y": 20, "anything": [1]}


@pytest.mark.parametrize("not_a_name", ["a.b", "a/x", "a@1", "a~c", ""])
def test_a_source_names_a_node_and_an_output_each_a_name(not_a_name: str):
    """Every part of a source but its field is a name; a field of a record may have dots of its own."""
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": {"b": {"node": "b", "inputs": {"x": {"sources": [{"node": not_a_name, "output": "result"}]}}}}})
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": {"b": {"node": "b", "inputs": {"x": {"sources": [{"node": "a", "output": not_a_name}]}}}}})


def test_a_field_of_a_source_may_have_dots_but_never_starts_with_one():
    record = GraphRecord.model_validate({"nodes": {"b": {"node": "b", "inputs": {"x": {"sources": [{"node": "a", "output": "result", "field": "Amount in USD."}]}}}}})

    assert record.nodes["b"].inputs["x"] == InputEntry(sources=(Address("a", "result", "Amount in USD."),))
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": {"b": {"node": "b", "inputs": {"x": {"sources": [{"node": "a", "output": "result", "field": ".x"}]}}}}})
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": {"a": {"node": "t", "bindings": {}}}})


def test_the_json_schema_gives_an_input_one_entry_of_a_constant_a_graph_input_a_default_and_sources():
    schema = GraphRecord.model_json_schema()
    entry = schema["$defs"]["InputEntry"]

    assert schema["$defs"]["NodeRecord"]["properties"]["inputs"]["additionalProperties"] == {"$ref": "#/$defs/InputEntry"}
    assert list(entry["properties"]) == ["constant", "graph_input", "default", "sources"] and "required" not in entry
    assert entry["additionalProperties"] is False
    assert schema["$defs"]["Address"]["required"] == ["node", "output"]


def test_the_order_of_an_inputs_sources_never_tells_two_records_apart():
    """Several sources of one input are alternatives from opposite branches, so nothing depends on their order."""
    def record(*sources: str) -> GraphRecord:
        return GraphRecord.model_validate({"nodes": {
            "d": {"node": "decide", "inputs": {"text": {"graph_input": "text"}}},
            "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "d", "output": source} for source in sources]}}},
        }, "outputs": {"result": {"node": "u", "output": "result"}}})

    one_way, other_way = record("yes", "no"), record("no", "yes")

    assert one_way == other_way and one_way.model_dump_json() == other_way.model_dump_json()
    assert Graph.compile(one_way, REGISTRY).fingerprint == Graph.compile(other_way, REGISTRY).fingerprint


# -- the record door: what is not a record's shape is refused ----------------------


def test_two_nodes_under_one_id_are_refused_as_a_key_written_twice():
    text = '{"nodes": {"a": {"node": "upper"}, "a": {"node": "join"}}}'

    assert _refused(text) == [("invalid_record", ("record",), "repeated_key")]


def test_an_input_written_twice_in_one_node_is_refused_as_a_key_written_twice():
    text = '{"nodes": {"a": {"node": "upper", "inputs": {"text": {"constant": "x"}, "text": {"constant": "y"}}}}}'

    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(text, REGISTRY)
    [problem] = refused.value.errors()
    assert (problem.type, problem.ctx["error"]) == ("invalid_record", "repeated_key") and "'text'" in problem.msg


@pytest.mark.parametrize("bad", ["a.b", "a/b", "a@b", "a~b", ""])
def test_a_node_id_that_is_not_a_name_is_refused(bad):
    assert _refused({"nodes": {bad: {"node": "upper"}}}) == [("invalid_record", ("record", "nodes", bad, "[key]"), "not_a_name")]


def test_a_source_listed_twice_is_refused_naming_the_source():
    record = {"nodes": {"a": {"node": "text"}, "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}, {"node": "a", "output": "result"}]}}}}}

    with pytest.raises(ValidationError, match="a.result is listed twice"):
        GraphRecord.model_validate(record)


@pytest.mark.parametrize("entry", [
    {"constant": "a", "graph_input": "text"},
    {"constant": "a", "graph_input": "text", "default": "b"},
    {"constant": "a", "sources": [{"node": "a", "output": "result"}]},
    {"graph_input": "text", "sources": [{"node": "a", "output": "result"}]},
    {"default": "b"},
    {},
], ids=["constant-and-graph-input", "constant-and-default", "constant-and-sources", "graph-input-and-sources", "default-alone", "empty"])
def test_an_input_given_other_than_exactly_one_of_a_constant_a_graph_input_or_sources_is_refused(entry):
    record = {"nodes": {"a": {"node": "text"}, "b": {"node": "upper", "inputs": {"text": entry}}}}

    assert _refused(record) == [("invalid_record", ("record", "nodes", "b", "inputs", "text"), "one_of_constant_graph_input_sources")]


@pytest.mark.parametrize("bad", ["a.b", "a/b", ""])
def test_an_input_a_graph_input_or_a_graph_output_that_is_not_a_name_is_refused(bad):
    assert _refused({"nodes": {"u": {"node": "upper", "inputs": {bad: {"constant": "x"}}}}})[0][1][:4] == ("record", "nodes", "u", "inputs")
    assert _refused({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": bad}}}}})[0][1][:5] == ("record", "nodes", "u", "inputs", "text")
    assert _refused({"nodes": {"u": {"node": "upper"}}, "outputs": {bad: {"node": "u", "output": "result"}}})[0][1][:2] == ("record", "outputs")


_SOURCE = {"node": "a", "output": "result"}


@pytest.mark.parametrize(("text", "loc", "error"), [
    ('{"nodes": {"a": {"node": "text"}, "a": {"node": "text"}}}', ("record",), "repeated_key"),
    ('{"nodes": {"b": {"node": "upper", "inputs": {"text": {"constant": "x", "constant": "y"}}}}}', ("record",), "repeated_key"),
    (json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"a.b": {"constant": "x"}}}}}), ("record", "nodes", "b", "inputs", "a.b", "[key]"), "not_a_name"),
    (json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {"graph_input": "a.b"}}}}}), ("record", "nodes", "b", "inputs", "text", "graph_input"), "not_a_name"),
    (json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result", "field": ".x"}]}}}}}), ("record", "nodes", "b", "inputs", "text", "sources", 0, "field"), "not_a_field"),
    (json.dumps({"nodes": {"a": {"node": "text"}, "b": {"node": "upper", "inputs": {"text": {"sources": [_SOURCE, _SOURCE]}}}}}), ("record", "nodes", "b", "inputs", "text", "sources"), "repeated_source"),
    (json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {"sources": []}}}}}), ("record", "nodes", "b", "inputs", "text", "sources"), "no_sources"),
    (json.dumps({"nodes": {"b": {"node": "two", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"zed": {"node": "b", "output": "result", "field": "zed"}}}), ("record", "outputs", "zed"), "output_has_field"),
    (json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {"constant": "x", "graph_input": "text"}}}}}), ("record", "nodes", "b", "inputs", "text"), "one_of_constant_graph_input_sources"),
    (json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {"constant": "x", "default": "y"}}}}}), ("record", "nodes", "b", "inputs", "text"), "default_without_graph_input"),
], ids=["repeated_key-node", "repeated_key-entry", "not_a_name-input", "not_a_name-graph_input", "not_a_field", "repeated_source", "no_sources", "output_has_field", "one_of_constant_graph_input_sources", "default_without_graph_input"])
def test_each_refusal_of_the_record_door_is_one_problem_at_its_place_saying_why(text: str, loc: tuple, error: str):
    """A fault of an input's entry is at the input; a name or a field that is not one is where it is written; a key
    written twice is found while the JSON is read, before any place in the record is known. Each fault is one problem."""
    assert _refused(text) == [("invalid_record", loc, error)]


def test_a_refusal_of_a_name_says_which_text_is_not_a_name():
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {"graph_input": "a.b"}}}}}), REGISTRY)

    [problem] = refused.value.errors()
    assert problem.ctx["reason"].startswith("'a.b' is not a name") and problem.msg.endswith("Fix the record's JSON.")


# -- compiling: what a record says that cannot be meant ---------------------------


def test_several_edges_into_one_input_that_could_each_give_it_a_value_are_a_problem_naming_each():
    nodes = {
        "first": {"node": "text", "inputs": {"value": {"constant": "a"}}},
        "second": {"node": "text", "inputs": {"value": {"constant": "b"}}},
        "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "second", "output": "result"}, {"node": "first", "output": "result"}]}}},
    }
    graph = Graph.compile(GraphRecord.model_validate({"nodes": nodes}), REGISTRY)

    assert [(p.type, p.loc, p.ctx["sources"]) for p in graph.problems] == [("several_edges_into_one_input", ("j", "texts"), ["first.result", "second.result"])]


def test_an_edge_from_a_node_the_graph_does_not_have_is_a_problem_on_the_input_it_leads_into():
    out_of = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "gone", "output": "result"}]}}}}, "outputs": {"result": {"node": "b", "output": "result"}}}), REGISTRY)

    assert [(p.type, p.loc, p.ctx["missing"]) for p in out_of.problems] == [("edge_names_missing_node", ("b", "text"), "gone")]


def test_a_constant_for_an_input_the_node_does_not_have_is_unknown_input():
    assert _problems(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"constant": "a"}, "colour": {"constant": "red"}}}}, "outputs": {"result": {"node": "u", "output": "result"}}})) == [("unknown_input", ("u", "colour"))]


def test_a_required_input_with_no_entry_gets_no_value_on_any_node():
    first = GraphRecord.model_validate({"nodes": {"u": {"node": "upper"}}})
    later = GraphRecord.model_validate({"nodes": {"t": {"node": "text"}, "b": {"node": "both", "inputs": {"yes": {"sources": [{"node": "t", "output": "result"}]}}}}})

    assert _problems(first) == [("input_has_no_value", ("u", "text"))]
    assert _problems(later) == [("input_has_no_value", ("b", "no"))]


# -- what a graph takes and gives -------------------------------------------------


def test_graph_inputs_and_graph_outputs_are_the_ones_the_record_declares():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "text", "inputs": {"value": {"graph_input": "value"}}}, "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}}}, "c": {"node": "both", "inputs": {"no": {"constant": "!"}, "yes": {"graph_input": "yes"}}}}, "outputs": {"b_result": {"node": "b", "output": "result"}, "c_result": {"node": "c", "output": "result"}}}), REGISTRY)

    assert graph.problems == []
    assert {name: i.required for name, i in graph.inputs.items()} == {"value": True, "yes": True}
    assert graph.outputs == ["b_result", "c_result"]
    assert graph.run(value="v", yes="y").outputs == {"b_result": "V", "c_result": "y!"}


def test_an_optional_input_with_no_entry_gets_the_nodes_default_and_is_no_graph_input():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"t": {"node": "text"}}, "outputs": {"result": {"node": "t", "output": "result"}}}), REGISTRY)

    assert graph.problems == [] and graph.inputs == {}
    assert graph.run().outputs == {"result": ""}


def test_a_constant_is_not_a_graph_input_and_a_default_is_a_graph_input_with_that_default():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "both", "inputs": {"yes": {"constant": "y"}, "no": {"graph_input": "no", "default": "n"}}}}, "outputs": {"result": {"node": "b", "output": "result"}}}), REGISTRY)

    assert list(graph.inputs) == ["no"] and not graph.inputs["no"].required and graph.inputs["no"].default == "n"
    assert graph.run().outputs == {"result": "yn"}
    assert graph.run(no="!").outputs == {"result": "y!"}


def test_a_graph_input_on_a_node_an_edge_leads_into_is_given_by_the_caller_like_any_other():
    record = GraphRecord.model_validate({"nodes": {"a": {"node": "text", "inputs": {"value": {"graph_input": "value"}}}, "b": {"node": "both", "inputs": {"no": {"graph_input": "no", "default": "n"}, "yes": {"sources": [{"node": "a", "output": "result"}]}}}}, "outputs": {"result": {"node": "b", "output": "result"}}})
    graph = Graph.compile(record, REGISTRY)

    assert graph.problems == [] and list(graph.inputs) == ["value", "no"]
    assert graph.run(value="y").outputs == {"result": "yn"} and graph.run(value="y", no="!").outputs == {"result": "y!"}


def test_inputs_naming_one_graph_input_each_get_the_callers_one_value():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "b": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"a_result": {"node": "a", "output": "result"}, "b_result": {"node": "b", "output": "result"}}}), REGISTRY)

    assert list(graph.inputs) == ["text"]
    assert graph.run(text="x").outputs == {"a_result": "X", "b_result": "X"}


def test_one_graph_input_on_two_inputs_with_no_default_in_the_record_is_required_whatever_each_inputs_own_default():
    """``text``'s ``value`` has a default of its own and ``upper``'s ``text`` has none: neither plays a part, so both
    read the caller's one value and the caller must give it."""
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {
        "a": {"node": "text", "inputs": {"value": {"graph_input": "x"}}},
        "b": {"node": "upper", "inputs": {"text": {"graph_input": "x"}}},
    }, "outputs": {"a_result": {"node": "a", "output": "result"}, "b_result": {"node": "b", "output": "result"}}}), REGISTRY)

    assert graph.problems == []
    assert {name: (i.required, i.default) for name, i in graph.inputs.items()} == {"x": (True, None)}
    assert graph.run(x="q").outputs == {"a_result": "q", "b_result": "Q"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run()
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("missing_input", ("inputs", "x"))]


def test_one_graph_input_given_two_different_defaults_in_the_record_is_used_two_ways():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {
        "a": {"node": "text", "inputs": {"value": {"graph_input": "x", "default": "p"}}},
        "b": {"node": "upper", "inputs": {"text": {"graph_input": "x", "default": "q"}}},
    }, "outputs": {"a_result": {"node": "a", "output": "result"}, "b_result": {"node": "b", "output": "result"}}}), REGISTRY)

    assert [(p.type, p.loc, p.ctx) for p in graph.problems] == [("graph_input_used_two_ways", ("b", "text"), {"node_id": "b", "field": "text", "graph_input": "x"})]


def test_a_graph_output_naming_a_node_or_an_output_the_record_lacks_is_the_graphs_problem_at_no_node():
    """The node is called ``outputs`` so that a problem at the node's id would show as one of that node's."""
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"outputs": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {
        "gone": {"node": "nobody", "output": "result"},
        "missing": {"node": "outputs", "output": "nope"},
        "found": {"node": "outputs", "output": "result"},
    }}), REGISTRY)

    assert [(p.type, p.loc, p.ctx) for p in graph.problems] == [
        ("graph_output_not_found", (), {"output": "gone", "source": "nobody.result", "outputs": []}),
        ("graph_output_not_found", (), {"output": "missing", "source": "outputs.nope", "outputs": ["outputs.result"]}),
    ]
    assert graph.describe().nodes["outputs"].problems == []


# -- a graph written from Python: make_graph -----------------------------------------------


@node("hello", title="Hello", description="For a test.")
def _hello(name: str) -> str:
    return f"hello {name}"


@node("shout", title="Shout", description="For a test.")
def _shout(text: str, times: int = 1) -> str:
    return text.upper() + "!" * times


@node("email", title="Email", description="For a test.")
def _email() -> str:
    return "ida@x"


@node("count", title="Count", description="For a test.")
def _count() -> int:
    return 3


PLAIN = registry_of(_hello, _shout, _email, _count)


def _made(**arguments: Any) -> list[tuple[str, tuple]]:
    """The problems ``make_graph`` raises for ``arguments`` as ``(type, loc)``; its nodes are ``_hello`` and ``_shout``."""
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, _shout, **arguments)
    return [(p.type, p.loc) for p in refused.value.errors()]


def test_make_graph_with_no_edges_chains_each_node_to_the_one_before():
    graph = make_graph(_hello, _shout, constants={"hello.name": "ida"})

    assert graph == GraphRecord.model_validate({"nodes": {
        "hello": {"node": "hello", "inputs": {"name": {"constant": "ida"}}},
        "shout": {"node": "shout", "inputs": {"text": {"sources": [{"node": "hello", "output": "result"}]}}},
    }, "outputs": {"result": {"node": "shout", "output": "result"}}})
    assert Graph.compile(graph, PLAIN).run().outputs == {"result": "HELLO IDA!"}


def test_make_graph_numbers_a_node_used_twice():
    graph = make_graph(_hello, _shout, _shout, constants={"hello.name": "ida", "shout-2.times": 2})

    assert list(graph.nodes) == ["hello", "shout-1", "shout-2"]
    assert [(node_id, n.inputs["text"]) for node_id, n in graph.nodes.items() if node_id != "hello"] == [
        ("shout-1", InputEntry(sources=(Address("hello", "result"),))),
        ("shout-2", InputEntry(sources=(Address("shout-1", "result"),))),
    ]
    assert Graph.compile(graph, PLAIN).run().outputs == {"result": "HELLO IDA!!!"}


def test_make_graph_skips_an_id_another_node_already_has():
    @node("hello-1", title="Other", description="For a test.")
    def _other(name: str) -> str:
        return name

    record = make_graph(_hello, _hello, _other, edges=[], constants={"hello-2.name": "a", "hello-3.name": "b", "hello-1.name": "c"}, outputs={"a": "hello-2.result", "b": "hello-3.result", "c": "hello-1.result"})
    assert list(record.nodes) == ["hello-2", "hello-3", "hello-1"]


def test_make_graph_with_edges_takes_those_as_every_edge_and_matches_nothing_by_name():
    graph = make_graph(_hello, _shout, edges=[], outputs={"hello": "hello.result", "shout": "shout.result"})

    assert not any("sources" in n.model_dump(mode="json").get("inputs", {}).get(name, {}) for n in graph.nodes.values() for name in n.inputs)
    assert list(Graph.compile(graph, PLAIN).inputs) == ["name", "text"]


def test_make_graph_without_edges_refuses_nodes_it_cannot_chain_and_says_to_pass_edges():
    with pytest.raises(InvalidCallError, match="pass edges=") as refused:
        make_graph(_email, _count, outputs={"email": "email.result", "result": "count.result"})
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("not_a_chain", ("count",))]
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, _shout, constants={"hello.name": "x"}, outputs={"count": "count.result", "result": "shout.result"})
    assert [(p.type, p.loc, p.ctx["inputs"]) for p in refused.value.errors()] == [("not_a_chain", ("hello",), [])]
    assert refused.value.errors()[0].ctx == {"node_id": "hello", "before": "count", "inputs": [], "outputs": ["result"]}


def test_make_graph_refuses_a_place_on_a_node_it_was_not_given():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, constants={"nobody.name": "x"})
    assert [(p.type, p.ctx["nodes"]) for p in refused.value.errors()] == [("node_not_in_graph", ["hello"])]
    assert _made(edges=[("nobody.result", "shout.text")], constants={"hello.name": "x"}, outputs={"result": "shout.result", "hello": "hello.result"}) == [("node_not_in_graph", ("nobody",))]
    assert _made(edges=[], inputs={"who": "nobody.name"}, outputs={"result": "shout.result", "hello": "hello.result"}) == [("node_not_in_graph", ("nobody",))]
    assert _made(outputs={"result": "nobody.result"}) == [("node_not_in_graph", ("nobody",))]


def test_make_graph_writes_wrong_inputs_outputs_and_constants_as_given_and_compile_reports_each():
    record = make_graph(
        _hello, _shout,
        edges=[("hello.reslt", "shout.text"), ("hello.result", "shout.txt")],
        constants={"hello.nmae": "x"}, defaults={"hello.name": 3},
    )
    assert sorted((p.type, p.loc) for p in Graph.compile(record, PLAIN).problems) == [
        ("invalid_value", ("hello", "name")),
        ("unknown_input", ("hello", "nmae")),
        ("unknown_input", ("shout", "txt")),
        ("unknown_output", ("shout", "text")),
    ]


@pytest.mark.parametrize("value", [{"ida", "bo"}, frozenset({"ida"}), [{"ida"}], {"names": {"ida"}}], ids=["set", "frozenset", "in-a-list", "in-a-dict"])
def test_make_graph_refuses_a_set_as_a_constant_as_set_has_no_order(value: object):
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, constants={"hello.name": value})
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, [e["type"] for e in problem.ctx["errors"]]) == ("invalid_value", ("hello", "name"), ["set_has_no_order"])


def test_make_graph_writes_a_type_mismatch_and_an_output_a_hook_does_not_give_and_compile_reports_them():
    @node("fan", title="Fan", description="For a test.", compute_outputs=lambda n: {f"o{i}": int for i in range(n)})
    def fan(n: int = 2) -> dict:
        return {f"o{i}": i for i in range(n)}

    mismatched = make_graph(_count, _hello, edges=[("count.result", "hello.name")])
    assert [(p.type, p.loc) for p in Graph.compile(mismatched, PLAIN).problems] == [("type_mismatch", ("hello", "name"))]
    not_given = make_graph(fan, _shout, edges=[("fan.o5", "shout.text")])
    assert [p.type for p in Graph.compile(not_given, registry_of(fan, _shout)).problems] == ["unknown_output"]


def test_make_graph_writes_defaults_and_compile_reports_a_list_for_an_input_of_one_value():
    record = make_graph(_hello, defaults={"hello.name": "you"})
    assert record.nodes["hello"].inputs == {"name": InputEntry(graph_input="name", default="you")}
    assert Graph.compile(record, PLAIN).run().outputs == {"result": "hello you"}
    assert Graph.compile(record, PLAIN).run(name="ida").outputs == {"result": "hello ida"}
    listed = make_graph(_hello, defaults={"hello.name": ["you"]})
    assert [(p.type, p.loc) for p in Graph.compile(listed, PLAIN).problems] == [("invalid_value", ("hello", "name"))]


def test_make_graph_makes_a_default_a_graph_input_on_a_node_an_edge_leads_into():
    record = make_graph(_hello, _shout, constants={"hello.name": "ida"}, defaults={"shout.times": 2})
    graph = Graph.compile(record, PLAIN)

    assert list(graph.inputs) == ["times"]
    assert graph.run().outputs == {"result": "HELLO IDA!!"} and graph.run(times=3).outputs == {"result": "HELLO IDA!!!"}


def test_make_graph_writes_each_node_at_its_version():
    def older(name: str) -> str:
        return name

    @node("v", title="Newer", description="For a test.", replaces=node("v", title="Older", description="For a test.")(older), upgrade=Upgrade())
    def _newer(name: str) -> str:
        return name

    assert make_graph(_newer).nodes["v"].version == 2


# -- make_graph declares the graph's inputs and outputs --------------------------


def test_make_graph_declares_each_open_input_of_a_node_no_edge_leads_into_and_each_output_no_edge_leaves():
    record = make_graph(_hello, _shout)

    assert record.nodes["hello"].inputs == {"name": InputEntry(graph_input="name")}
    assert "times" not in record.nodes["shout"].inputs  # shout is led into: its optional input keeps the node's default
    assert record.outputs == {"result": Address("shout", "result")}


def test_make_graph_refuses_two_graph_inputs_or_two_graph_outputs_that_would_share_a_name():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(upper, upper, edges=[])

    assert [(p.type, p.loc, p.ctx) for p in refused.value.errors()] == [
        ("graph_inputs_share_a_name", ("inputs", "text"), {"name": "text", "inputs": ["upper-1.text", "upper-2.text"]}),
        ("graph_outputs_share_a_name", ("outputs", "result"), {"name": "result", "outputs": ["upper-1.result", "upper-2.result"]}),
    ]
    assert ["make_graph's inputs=" in refused.value.errors()[0].msg, "make_graph's outputs=" in refused.value.errors()[1].msg] == [True, True]


def test_make_graph_names_the_graph_inputs_and_outputs_it_is_given():
    record = make_graph(upper, upper, edges=[], inputs={"a": "upper-1.text", "b": "upper-2.text"}, outputs={"x": "upper-1.result", "y": "upper-2.result"})
    graph = Graph.compile(record, REGISTRY)

    assert list(graph.inputs) == ["a", "b"] and graph.outputs == ["x", "y"]
    assert graph.run(a="p", b="q").outputs == {"x": "P", "y": "Q"}


def test_make_graph_feeds_several_inputs_from_one_graph_input_given_a_list():
    record = make_graph(upper, upper, edges=[], inputs={"text": ["upper-1.text", "upper-2.text"]}, outputs={"x": "upper-1.result", "y": "upper-2.result"})
    graph = Graph.compile(record, REGISTRY)

    assert [n.inputs["text"] for n in record.nodes.values()] == [InputEntry(graph_input="text")] * 2
    assert list(graph.inputs) == ["text"] and graph.run(text="p").outputs == {"x": "P", "y": "P"}


def test_make_graph_refuses_a_constant_together_with_a_graph_input_or_an_edge():
    one_of = "one_of_constant_graph_input_sources"

    with pytest.raises(InvalidCallError) as with_input:
        make_graph(_hello, constants={"hello.name": "a"}, inputs={"name": "hello.name"})
    with pytest.raises(InvalidCallError) as with_edge:
        make_graph(_hello, _shout, edges=[("hello.result", "shout.text")], constants={"hello.name": "a", "shout.text": "x"})
    with pytest.raises(InvalidCallError) as edge_and_input:
        make_graph(_hello, _shout, edges=[("hello.result", "shout.text")], inputs={"name": "hello.name", "text": "shout.text"})

    assert [(p.loc, p.ctx["error"]) for p in with_input.value.errors()] == [(("record", "nodes", "hello", "inputs", "name"), one_of)]
    assert [(p.loc, p.ctx["error"]) for p in with_edge.value.errors()] == [(("record", "nodes", "shout", "inputs", "text"), one_of)]
    assert [(p.loc, p.ctx["error"]) for p in edge_and_input.value.errors()] == [(("record", "nodes", "shout", "inputs", "text"), one_of)]


def test_make_graph_refuses_an_edge_written_twice():
    [(kind, loc)] = _made(edges=[("hello.result", "shout.text"), ("hello.result", "shout.text")], constants={"hello.name": "a"})

    assert kind == "invalid_record" and loc[:5] == ("record", "nodes", "shout", "inputs", "text")


# -- the graph a record compiles to -------------------------------------------------


def test_a_compiled_graph_keeps_its_own_definitions_whatever_the_registry_does_later():
    registry = registry_of(upper)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), registry)

    registry.remove("upper")

    assert graph.run(text="a").outputs == {"result": "A"}
    assert [p.type for p in Graph.compile(graph.record, registry).problems] == ["unknown_node"]


def test_the_type_decides_one_or_many_so_a_list_given_for_an_input_of_one_value_is_refused():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}), REGISTRY)

    assert graph.runs_on["u"].mode == "once" and graph.run(text="a").outputs == {"result": "A"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a"])
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_input", ("inputs", "text"))]


def test_a_list_as_a_constant_or_a_default_for_an_input_of_one_value_is_invalid_value():
    assert _problems(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"constant": ["x", "y"]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}})) == [("invalid_value", ("u", "text"))]
    assert _problems(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text", "default": ["a", "b"]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}})) == [("invalid_value", ("u", "text"))]


def test_every_problem_in_a_callers_inputs_is_refused_at_once_unknown_names_first_then_the_inputs_in_order():
    """The graph inputs are in the order of their nodes' ids: ``b``'s before ``u``'s."""
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "b": {"node": "both", "inputs": {"yes": {"graph_input": "yes"}, "no": {"graph_input": "no"}}}}, "outputs": {"b_result": {"node": "b", "output": "result"}, "u_result": {"node": "u", "output": "result"}}}), REGISTRY)
    offered = ["yes", "no", "text"]

    def refused(given: dict[Any, Any]) -> list[tuple[str, tuple[str | int, ...], dict[str, Any]]]:
        with pytest.raises(InvalidCallError) as raised:
            graph.with_inputs(given)
        return [(p.type, p.loc, p.ctx) for p in raised.value.errors()]

    assert refused({"nope": 1, "text": "a", "zz": 2}) == [
        ("not_offered", ("inputs", "nope"), {"name": "nope", "inputs": offered}),
        ("not_offered", ("inputs", "zz"), {"name": "zz", "inputs": offered}),
        ("missing_input", ("inputs", "yes"), {"name": "yes"}),
        ("missing_input", ("inputs", "no"), {"name": "no"}),
    ]
    not_a_text = {"name": "no", "reason": "Input should be a valid string", "errors": [{"type": "string_type", "loc": [], "msg": "Input should be a valid string"}]}
    assert refused({"no": 1, "text": ["a", "b"], 1: "x"}) == [
        ("not_offered", ("inputs", 1), {"name": "1", "inputs": offered}),
        ("missing_input", ("inputs", "yes"), {"name": "yes"}),
        ("invalid_input", ("inputs", "no"), not_a_text),
        ("invalid_input", ("inputs", "text"), {"name": "text", "expected": "str", "length": 2, "variant": "make_a_series"}),
    ]
    run = graph.with_inputs(text="a", yes="y", no="n")
    assert run.with_inputs({"text": "b"}).inputs == {"yes": "y", "no": "n", "text": "b"}


def test_a_graph_input_is_given_by_its_name_never_by_an_address_and_an_item_limit_below_one_is_refused():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), REGISTRY)

    with pytest.raises(InvalidCallError) as by_address:
        graph.run({"text": "a", "u.text": "b"})
    with pytest.raises(InvalidCallError) as item_limit:
        graph.run(text="a", max_items=0)

    assert [(p.type, p.loc) for p in by_address.value.errors()] == [("not_offered", ("inputs", "u.text"))]
    assert [p.type for p in item_limit.value.errors()] == ["invalid_max_items"]
    with pytest.raises(InvalidCallError):
        graph.with_inputs(text="a").with_max_items(-1)


def test_a_subgraph_with_an_unknown_inner_node_is_one_problem_at_its_node_path():
    broken = subgraph("broken", title="Broken", description="For a test.")(GraphRecord.model_validate({"nodes": {"x": {"node": "gone"}}, "outputs": {"result": {"node": "x", "output": "result"}}}))
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "broken"}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), registry_of(upper, broken))

    assert [(problem.type, problem.loc) for problem in graph.problems] == [("unknown_node", ("b", "x"))]


def _a_table_read_by_one_text(source: str) -> Problem:
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    class Row(TypedDict):
        name: str

    @node("rows", title="Rows", description="For a test.")
    def rows() -> Table[Row]:
        return pl.DataFrame({"name": ["a"]})

    node_id, output, *field = source.split(".", 2)
    edge = InputEntry(sources=(Address(node_id, output, *field),))
    graph = Graph.compile(GraphRecord(nodes={"r": NodeRecord(node="rows"), "u": NodeRecord(node="upper", inputs={"text": edge})}), registry_of(rows, upper))
    (problem,) = graph.problems
    return problem


def test_an_edge_naming_a_column_of_a_whole_table_is_unknown_output_as_a_table_is_one_value():
    problem = _a_table_read_by_one_text("r.result.name")

    assert (problem.type, problem.loc, problem.ctx["record_field"]) == ("unknown_output", ("u", "text"), "name")


@pytest.mark.parametrize("source", ["r.result.name", "r.result"])
def test_a_table_wired_where_one_value_is_read_says_to_split_it_first(source):
    assert _a_table_read_by_one_text(source).ctx["variant"] == "make_a_series"


def test_a_series_of_tables_is_accepted_where_it_is_declared():
    import polars as pl
    from conductor import Table

    @node("tables", title="Tables", description="For a test.")
    def tables(text: str) -> Series[Table]:
        return [pl.DataFrame()]

    assert tables.outputs["result"].is_series is True
    assert tables.outputs["result"].type.id.startswith("Table")


# -- a subgraph is one node from outside -------------------------------------


@node("tail", title="Tail", description="For a test.")
def tail(text: str) -> str:
    return text + "."


@node("split", title="Split", description="For a test.")
def split(text: str) -> Series[str]:
    return text.split(",")


ENRICH = subgraph("enrich", title="Enrich", description="For a test.")(make_graph(upper, tail))
PAIR = subgraph("pair", title="Pair", description="For a test.")(GraphRecord.model_validate({"nodes": {"x": {"node": "upper", "inputs": {"text": {"graph_input": "x_text"}}}, "y": {"node": "upper", "inputs": {"text": {"graph_input": "y_text"}}}, "b": {"node": "both", "inputs": {"yes": {"sources": [{"node": "x", "output": "result"}]}, "no": {"sources": [{"node": "y", "output": "result"}]}}}}, "outputs": {"result": {"node": "b", "output": "result"}}}))
COLLECT = subgraph("collect", title="Collect", description="For a test.")(GraphRecord.model_validate({"nodes": {"shouted": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "all": {"node": "join", "inputs": {"texts": {"sources": [{"node": "shouted", "output": "result"}]}}}}, "outputs": {"result": {"node": "all", "output": "result"}}}))
SUBGRAPHS = registry_of(upper, tail, split, join, both, ENRICH, PAIR, COLLECT)


def test_a_subgraph_node_is_given_wired_and_read_by_its_own_id_and_the_names_its_record_declares():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"graph_input": "text"}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "e", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), SUBGRAPHS)

    assert graph.problems == [] and list(graph.inputs) == ["text"]
    run = graph.run(text="hi")
    assert run.outputs == {"result": "HI."} and set(run.results) == {"e", "u"} and run.results["e"] == {"result": "HI."}
    assert set(graph.runs_on) == {"e", "u"} and set(graph.describe().nodes) == {"e", "u"}
    assert [f.field.name for f in graph.describe().nodes["e"].inputs] == ["text"]


def test_make_graph_makes_a_graph_input_only_of_each_graph_input_a_subgraphs_record_gives_no_default():
    """``b`` has a default; ``c`` and ``d`` are each named by two entries with a default on one of them, so each has one."""
    halves = subgraph("halves", title="Halves", description="For a test.")(GraphRecord.model_validate({
        "nodes": {
            "x": {"node": "upper", "inputs": {"text": {"graph_input": "a"}}},
            "y": {"node": "upper", "inputs": {"text": {"graph_input": "b", "default": "b"}}},
            "z": {"node": "tail", "inputs": {"text": {"graph_input": "c"}}},
            "w": {"node": "tail", "inputs": {"text": {"graph_input": "c", "default": "c"}}},
            "u": {"node": "tail", "inputs": {"text": {"graph_input": "d", "default": "d"}}},
            "v": {"node": "tail", "inputs": {"text": {"graph_input": "d"}}},
        },
        "outputs": {"x": {"node": "x", "output": "result"}, "y": {"node": "y", "output": "result"}},
    }))

    record = make_graph(halves)

    assert record.nodes["halves"].inputs == {"a": InputEntry(graph_input="a")}
    assert record.outputs == {"x": Address("halves", "x"), "y": Address("halves", "y")}
    assert Graph.compile(record, registry_of(upper, tail, halves)).run(a="a").outputs == {"x": "A", "y": "B"}


def test_a_wrong_address_on_a_subgraph_node_is_a_problem_on_it_and_never_names_an_inner_node():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"graph_input": "text"}, "tail": {"sources": [{"node": "v", "output": "result"}]}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "e", "output": "nope"}]}}}, "v": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), SUBGRAPHS)

    found = sorted((p.type, p.loc) for p in graph.problems)
    assert found == [("unknown_input", ("e", "tail")), ("unknown_output", ("u", "text"))]
    assert not any("/" in str(p.loc) for p in graph.problems)


def test_a_subgraph_nodes_inputs_are_the_graph_inputs_its_record_declares():
    first = Graph.compile(GraphRecord.model_validate({"nodes": {"c": {"node": "pair", "inputs": {"x_text": {"graph_input": "x_text"}, "y_text": {"graph_input": "y_text"}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), SUBGRAPHS)
    later = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "pair", "inputs": {"x_text": {"sources": [{"node": "s", "output": "result"}]}}}}}), SUBGRAPHS)

    assert first.problems == [] and list(first.inputs) == ["x_text", "y_text"]
    assert [(p.type, p.loc) for p in later.problems] == [("input_has_no_value", ("c", "y_text"))]  # a subgraph node gets its inputs as any other node does


def test_a_subgraph_nodes_inputs_are_given_by_the_names_it_offers():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "pair", "inputs": {"y_text": {"constant": "b"}, "x_text": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), SUBGRAPHS)

    assert graph.problems == [] and list(graph.inputs) == ["text"]
    assert graph.run(text="a").outputs == {"result": "AB"}


def test_an_outer_record_never_names_an_inner_node_of_a_subgraph():
    """No name has a dot, so ``x.text`` is no input of ``pair``: the record door refuses it."""
    record = {"nodes": {"c": {"node": "pair", "inputs": {"x.text": {"constant": "a"}, "y_text": {"constant": "b"}}}}}

    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps(record), SUBGRAPHS)
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_record", ("record", "nodes", "c", "inputs", "x.text", "[key]"))]


def test_a_subgraph_lists_its_inputs_and_outputs_in_the_registrys_describe():
    described = next(n for n in SUBGRAPHS.describe().nodes if n.id == "pair")

    assert [f.name for f in described.inputs] == ["x_text", "y_text"] and [f.name for f in described.outputs] == ["result"]


def test_collecting_a_series_from_a_branch_not_chosen_is_skipped_not_called_with_an_empty_list():
    calls: list[list[str]] = []

    @node("gate", title="Gate", description="For a test.")
    def gate(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)

    @node("count-all", title="Count all", description="For a test.")
    def count_all(texts: Series[str]) -> str:
        calls.append(list(texts))
        return str(len(texts))

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"g": {"node": "gate", "inputs": {"text": {"graph_input": "text"}}}, "s": {"node": "split", "inputs": {"text": {"sources": [{"node": "g", "output": "no"}]}}}, "c": {"node": "count-all", "inputs": {"texts": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), registry_of(gate, split, count_all))

    assert graph.run(text="yes").results["c"]["result"] is None and calls == []
    assert graph.run(text="a,b").outputs == {"result": "2"}


def test_a_skipped_item_of_a_series_output_is_a_gap_that_reads_back_from_json():
    @node("some", title="Some", description="For a test.")
    def some(text: str) -> Series[str]:
        return ["a", SKIPPED, "c"]

    graph = Graph.compile(make_graph(some, upper), registry_of(some, upper))
    run = graph.run(text="x")

    assert run.results["some"]["result"] == ["a", None, "c"] and run.results["upper"]["result"] == ["A", None, "C"]
    assert Run.from_json(run.to_json(), graph).results == run.results


def test_a_table_input_gets_its_declared_columns():
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    class Names(TypedDict):
        name: str

    class Contacts(TypedDict):
        name: str
        email: str

    @node("contacts", title="Contacts", description="For a test.")
    def contacts() -> Table[Contacts]:
        return pl.DataFrame({"name": ["a"], "email": ["a@x"]})

    @node("columns", title="Columns", description="For a test.")
    def columns(table: Table[Names]) -> str:
        return ",".join(table.columns)

    registry = registry_of(contacts, columns)
    narrowed = Graph.compile(GraphRecord.model_validate({"nodes": {"c": {"node": "contacts"}, "n": {"node": "columns", "inputs": {"table": {"sources": [{"node": "c", "output": "result"}]}}}}, "outputs": {"result": {"node": "n", "output": "result"}}}), registry)

    assert Graph.compile(GraphRecord.model_validate({"nodes": {"n": {"node": "columns", "inputs": {"table": {"graph_input": "table"}}}}, "outputs": {"result": {"node": "n", "output": "result"}}}), registry).run(table=pl.DataFrame({"name": ["a"], "email": ["a@x"]})).outputs == {"result": "name"}
    assert narrowed.run().outputs == {"result": "name"}


def test_a_node_collecting_two_series_of_different_depths_is_called_per_item_of_the_outer_one():
    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Series[str]:
        return text.split("|")

    @node("pair-lists", title="Pair lists", description="For a test.")
    def pair_lists(a: Series[str], b: Series[str]) -> str:
        return f"{len(a)}/{len(b)}"

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"p": {"node": "parts", "inputs": {"text": {"graph_input": "text"}}}, "s": {"node": "split", "inputs": {"text": {"sources": [{"node": "p", "output": "result"}]}}}, "x": {"node": "pair-lists", "inputs": {"a": {"sources": [{"node": "p", "output": "result"}]}, "b": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "x", "output": "result"}}}), registry_of(parts, split, pair_lists))

    assert (graph.runs_on["x"].mode, graph.runs_on["x"].series) == ("per_item", "p.result")  # collecting the inner series ends the inner loop, not the outer one
    assert graph.run(text="a,b|c").outputs == {"result": ["2/2", "2/1"]}


def test_a_node_id_with_a_slash_beside_a_subgraph_node_is_refused_and_an_edge_into_it_names_the_outer_address():
    with pytest.raises(InvalidCallError) as beside:
        Graph.compile(json.dumps({"nodes": {"e": {"node": "enrich"}, "e/upper": {"node": "upper"}}}), SUBGRAPHS)
    ghost = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"sources": [{"node": "ghost", "output": "result"}]}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), SUBGRAPHS)

    assert [(p.type, p.loc) for p in beside.value.errors()] == [("invalid_record", ("record", "nodes", "e/upper", "[key]"))]
    assert [(p.type, p.loc, p.ctx["target"]) for p in ghost.problems] == [("edge_names_missing_node", ("e", "text"), "e.text")]


def test_collecting_items_that_a_branch_not_chosen_skipped_all_of_is_skipped():
    """A branch chosen per document, read by every line of the document, skips each line where it is not chosen, so
    that document's join is skipped too."""

    @node("pair-two", title="Pair two", description="For a test.")
    def pair_two(a: str, b: str) -> str:
        return f"{a}:{b}"

    @node("decide-doc", title="Decide doc", description="For a test.")
    def decide_doc(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("x") else YesNo(SKIPPED, text)

    @node("docs", title="Docs", description="For a test.")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "docs", "inputs": {"text": {"graph_input": "text"}}}, "g": {"node": "decide-doc", "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}}}, "l": {"node": "split", "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}}}, "p": {"node": "pair-two", "inputs": {"a": {"sources": [{"node": "l", "output": "result"}]}, "b": {"sources": [{"node": "g", "output": "yes"}]}}}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "p", "output": "result"}]}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}), registry_of(docs, decide_doc, split, pair_two, join))

    assert graph.run(text="x|y,no").results["j"]["result"] == ["x:x", None]


def test_collecting_downstream_of_a_series_output_not_chosen_is_skipped():
    @dataclass(frozen=True)
    class FullOrEmpty:
        full: Annotated[Series[str], Param(exclusive_group="size")]
        empty: Annotated[Series[str], Param(exclusive_group="size")]

    @node("decide-list", title="Decide list", description="For a test.")
    def decide_list(text: str) -> FullOrEmpty:
        return FullOrEmpty(text.split(","), SKIPPED) if text else FullOrEmpty(SKIPPED, [])

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "decide-list", "inputs": {"text": {"graph_input": "text"}}}, "up": {"node": "upper", "inputs": {"text": {"sources": [{"node": "d", "output": "empty"}]}}}, "after": {"node": "join", "inputs": {"texts": {"sources": [{"node": "up", "output": "result"}]}}}}, "outputs": {"result": {"node": "after", "output": "result"}}}), registry_of(decide_list, upper, join))

    assert graph.run(text="a,b").results["after"]["result"] is None


def test_a_records_field_whose_name_has_a_dot_is_wired_by_its_address():
    from typing import TypedDict

    import polars as pl
    from conductor import Table

    Amount = TypedDict("Amount", {"Amount in USD.": int})  # noqa: UP013

    @node("sheet", title="Sheet", description="For a test.")
    def sheet() -> Table[Amount]:  # type: ignore[valid-type]
        return pl.DataFrame({"Amount in USD.": [3, 4]})

    @node("each", title="Each", description="For a test.")
    def each(table: Table[Amount]) -> Series[Amount]:  # type: ignore[valid-type]
        return table.to_dicts()

    @node("double", title="Double", description="For a test.")
    def double(amount: int) -> int:
        return amount * 2

    record = make_graph(sheet, each, double, edges=[("sheet.result", "each.table"), ("each.result.Amount in USD.", "double.amount")])
    assert Graph.compile(record, registry_of(sheet, each, double)).run().outputs == {"result": [6, 8]}


def test_a_node_display_is_the_hosts_own_kept_and_outside_the_fingerprint():
    plain = GraphRecord.model_validate({"nodes": {"h": {"node": "hello", "inputs": {"name": {"graph_input": "name"}}}}, "outputs": {"result": {"node": "h", "output": "result"}}})
    shown = GraphRecord.model_validate({"nodes": {"h": {"node": "hello", "display": {"x": 3}, "inputs": {"name": {"graph_input": "name"}}}}, "outputs": {"result": {"node": "h", "output": "result"}}})
    assert GraphRecord.model_validate_json(shown.model_dump_json()) == shown
    assert Graph.compile(plain, PLAIN).fingerprint == Graph.compile(shown, PLAIN).fingerprint


@pytest.mark.parametrize("name", [name for name in NodeRecord.model_fields if name != "display"])
def test_every_field_of_a_node_but_its_display_is_in_the_fingerprint(name: str):
    node_record = NodeRecord(node="hello", inputs={"name": InputEntry(constant="Ida")})
    value = getattr(node_record, name)
    changed = value + 1 if isinstance(value, int) else f"{value}x" if isinstance(value, str) else {"name": InputEntry(constant="Bo")}
    other = node_record.model_copy(update={name: changed})
    assert Graph.compile(GraphRecord(nodes={"h": node_record}), PLAIN).fingerprint != Graph.compile(GraphRecord(nodes={"h": other}), PLAIN).fingerprint


def test_a_graphs_fingerprint_is_the_one_its_saved_runs_name():
    record = GraphRecord.model_validate({"nodes": {"h": {"node": "hello", "inputs": {"name": {"constant": "Ida"}}, "display": {"x": 3}}, "s": {"node": "shout", "inputs": {"text": {"sources": [{"node": "h", "output": "result"}]}}}}, "outputs": {"result": {"node": "s", "output": "result"}}})
    assert Graph.compile(record, PLAIN).fingerprint == "b480b8944238a28c"


def test_make_graph_refuses_a_constant_and_a_default_for_one_input():
    """A default makes the input a graph input, and an input gets exactly one of a constant or a graph input."""
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, constants={"hello.name": "a"}, defaults={"hello.name": "b"})

    assert [(p.type, p.loc, p.ctx["error"]) for p in refused.value.errors()] == [
        ("invalid_record", ("record", "nodes", "hello", "inputs", "name"), "one_of_constant_graph_input_sources"),
    ]


def test_a_constant_given_from_python_is_written_as_json_so_the_record_equals_itself_read_back():
    import datetime
    from enum import Enum

    @host_type("color", title="Color")
    class Color(Enum):
        RED = "red"

    @node("typed", title="Typed", description="For a test.")
    def typed(when: datetime.date, color: Color, pair: tuple[int, int]) -> str:
        return f"{when} {color.value} {pair[0]}"

    record = make_graph(typed, constants={"typed.when": datetime.date(2026, 1, 2), "typed.color": Color.RED, "typed.pair": (1, 2)})
    assert record.nodes["typed"].inputs == {"when": InputEntry(constant="2026-01-02"), "color": InputEntry(constant="red"), "pair": InputEntry(constant=[1, 2])}
    read_back = GraphRecord.model_validate_json(record.model_dump_json())
    assert read_back == record
    compiled, from_json = Graph.compile(record, registry_of(typed)), Graph.compile(read_back, registry_of(typed))
    assert compiled.fingerprint == from_json.fingerprint
    assert compiled.run().outputs == from_json.run().outputs == {"result": "2026-01-02 red 1"}


def test_a_table_as_a_constant_is_refused_at_the_call_and_said_to_be_given_as_records():
    import polars as pl

    @node("count", title="Count", description="For a test.")
    def count(rows: Table) -> int:
        return rows.height

    with pytest.raises(InvalidCallError) as refused:
        make_graph(count, constants={"count.rows": pl.DataFrame({"a": [1]})})
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc) == ("invalid_value", ("count", "rows"))
    assert problem.ctx["reason"] == "a value written in a graph record is saved with it as JSON; give a table as a list of records"


def test_a_node_record_accepts_only_json_in_its_constants_defaults_and_display():
    with pytest.raises(ValidationError):
        NodeRecord(node="a", display={"x": object()})
    with pytest.raises(ValidationError):
        InputEntry(constant=object())
    with pytest.raises(ValidationError):
        InputEntry(graph_input="x", default=object())


def test_a_float_that_is_not_finite_as_a_constant_is_written_as_text_so_a_record_equals_its_json():
    from conductor import Ask

    @node("half", title="Half", description="For a test.")
    def half(n: float) -> float:
        return n / 2

    @node("ask", title="Ask", description="For a test.", asks=True)
    def ask(x: float) -> str:
        raise Ask("ok?")

    registry = registry_of(half, ask)
    record = make_graph(half, constants={"half.n": float("nan")})
    read_back = GraphRecord.model_validate_json(record.model_dump_json())
    waiting = Graph.compile(make_graph(half, ask, constants={"half.n": float("inf")}), registry)

    assert record.nodes["half"].inputs == {"n": InputEntry(constant="NaN")}
    assert read_back == record and Graph.compile(read_back, registry).fingerprint == Graph.compile(record, registry).fingerprint
    compiled_from_json = Graph.compile(waiting.record.model_dump_json(), registry)
    assert Run.from_json(waiting.run().to_json(), compiled_from_json).status == "waiting"
    with pytest.raises(ValidationError):
        NodeRecord(node="half", display={"z": float("inf")})
    with pytest.raises(ValidationError):
        InputEntry(constant=float("nan"))


def test_make_graph_raises_every_problem_of_its_own_and_none_that_compile_finds():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, edges=[("count.result", "hello.name"), ("count", "hello.name")], constants={"nobody.x": 1})
    assert sorted(p.type for p in refused.value.errors()) == ["node_not_in_graph", "not_a_place"]


def test_make_graph_writes_each_node_at_the_id_and_version_of_the_definition_it_was_given():
    @node("v", title="V1", description="For a test.")
    def v1(text: str) -> str:
        return text

    @node("v", title="V2", description="For a test.", replaces=v1)
    def v2(text: str) -> str:
        return text

    record = make_graph(v2, v1, edges=[], constants={"v-1.nope": 1}, inputs={"a": "v-1.text", "b": "v-2.text"}, outputs={"a": "v-1.result", "b": "v-2.result"})
    assert [(node_id, n.node, n.version) for node_id, n in record.nodes.items()] == [("v-1", "v", 2), ("v-2", "v", 1)]
    assert [(p.type, p.loc) for p in Graph.compile(record, registry_of(v2)).problems] == [("unknown_input", ("v-1", "nope"))]

    @node("v", title="Other", description="For a test.")
    def other(n: int) -> int:
        return n

    unrelated = make_graph(v1, other, edges=[], outputs={"a": "v-1.result", "b": "v-2.result"})
    assert [(node_id, n.node, n.version) for node_id, n in unrelated.nodes.items()] == [("v-1", "v", 1), ("v-2", "v", 1)]


def test_make_graph_refuses_an_edge_that_is_not_node_and_name_and_says_how_to_write_it():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_count, _hello, edges=[("count.result", "hello.name"), ("count", "hello.name")])

    [problem] = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx) == ("not_a_place", ("edges", 1, 0), {"text": "count"})
    assert "write 'node.name'" in problem.msg and "column" not in problem.msg


def test_make_graph_refuses_a_constant_or_default_key_that_is_not_node_and_name_as_it_refuses_such_an_edge():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, constants={"hello": "Ida", "hello.name..x": "Ida"}, defaults={".name": "Bo"})

    assert [(p.type, p.loc) for p in refused.value.errors()] == [
        ("not_a_place", ("constants", "hello")),
        ("not_a_place", ("defaults", ".name")),
        ("invalid_record", ("record", "nodes", "hello", "inputs", "name..x", "[key]")),  # an input's name has no dot
    ]
    assert all("'node.name'" in p.msg for p in refused.value.errors()[:2])


def test_make_graph_takes_a_node_with_an_id_of_its_own_and_refuses_one_id_for_two_nodes():
    record = make_graph(("first", _hello), ("second", _hello), edges=[("first.result", "second.name")])
    assert list(record.nodes) == ["first", "second"] and record.nodes["second"].node == "hello"
    with pytest.raises(InvalidCallError) as refused:
        make_graph(("a", _hello), ("a", _hello), edges=[])
    assert [(p.type, p.loc, p.ctx["error"]) for p in refused.value.errors()] == [("invalid_record", ("record", "nodes", "a"), "repeated_key")]


def test_make_graph_refuses_a_text_that_is_not_node_and_name_at_the_argument_and_place_it_is_given():
    def refused(**arguments: Any) -> list[Problem]:
        with pytest.raises(InvalidCallError) as raised:
            make_graph(_hello, _shout, **arguments)
        return raised.value.errors()

    edge = refused(edges=[("hello.result", "shout")], outputs={"hello": "hello.result"})
    constant = refused(constants={"name": "x"})
    output = refused(outputs={"r": "shout"})

    assert [(p.type, p.loc, p.ctx) for p in [*edge, *constant, *output]] == [
        ("not_a_place", ("edges", 0, 1), {"text": "shout"}),
        ("not_a_place", ("constants", "name"), {"text": "name"}),
        ("not_a_place", ("outputs", "r"), {"text": "shout"}),
    ]
    assert all(p.msg.endswith("write 'node.name' in make_graph's arguments.") and "JSON" not in p.msg for p in [*edge, *constant, *output])


def test_a_record_door_refusal_of_what_make_graph_wrote_says_to_fix_make_graphs_arguments():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, _shout, edges=[("hello.result", "shout.text")], constants={"hello.name": "a", "shout.text": "x"})

    [problem] = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx["error"], problem.ctx["written_by"]) == (
        "invalid_record", ("record", "nodes", "shout", "inputs", "text"), "one_of_constant_graph_input_sources", "make_graph's arguments",
    )
    assert problem.msg.endswith("Fix make_graph's arguments.")


def test_make_graph_inputs_naming_one_input_leaves_the_other_automatic_graph_inputs_declared():
    record = make_graph(("a", upper), ("b", upper), edges=[], inputs={"x": "a.text"}, outputs={"a_result": "a.result", "b_result": "b.result"})
    graph = Graph.compile(record, REGISTRY)

    assert {node_id: n.inputs for node_id, n in record.nodes.items()} == {"a": {"text": InputEntry(graph_input="x")}, "b": {"text": InputEntry(graph_input="text")}}
    assert list(graph.inputs) == ["x", "text"] and graph.run(x="p", text="q").outputs == {"a_result": "P", "b_result": "Q"}


def test_make_graph_refuses_a_defaults_key_whose_name_is_also_given_in_inputs():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, _shout, edges=[], defaults={"shout.times": 2}, inputs={"times": "hello.name"}, outputs={"hello": "hello.result"})

    assert [(p.type, p.loc, p.ctx) for p in refused.value.errors()] == [
        ("graph_inputs_share_a_name", ("inputs", "times"), {"name": "times", "inputs": ["shout.times", "hello.name"]}),
    ]


def test_make_graph_refuses_an_output_with_a_field_as_the_record_does():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, outputs={"r": "hello.result.x"})

    assert [(p.type, p.loc, p.ctx["error"], p.ctx["written_by"]) for p in refused.value.errors()] == [
        ("invalid_record", ("record", "outputs", "r"), "output_has_field", "make_graph's arguments"),
    ]


def test_make_graph_leaves_an_optional_input_with_no_entry_out_and_it_is_no_graph_input():
    record = make_graph(_shout)

    assert record.nodes["shout"].inputs == {"text": InputEntry(graph_input="text")}
    assert list(Graph.compile(record, PLAIN).inputs) == ["text"]


def test_make_graph_makes_a_required_input_on_a_node_an_edge_leads_into_a_graph_input():
    record = make_graph(_hello, both, edges=[("hello.result", "both.yes")])
    graph = Graph.compile(record, registry_of(_hello, both))

    assert record.nodes["both"].inputs == {"yes": InputEntry(sources=(Address("hello", "result"),)), "no": InputEntry(graph_input="no")}
    assert list(graph.inputs) == ["name", "no"] and graph.run(name="ida", no="!").outputs == {"result": "hello ida!"}


def test_an_address_is_an_output_of_a_node_by_its_id_and_a_path_cannot_be_written_in_one():
    from pydantic import ValidationError

    assert str(Address("read", "result", "email")) == "read.result.email"
    with pytest.raises(ValidationError) as refused:
        Address("loud/shout", "result")
    assert [error["type"] for error in refused.value.errors()] == ["not_a_name"]
