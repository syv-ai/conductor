"""Tests for the graph record, the JSON a host saves for a graph, and for ``make_graph``, which writes one from Python.

A graph record looks like ``{"nodes": {id: {"node": ..., "inputs": {...}}}, "outputs": {name: {"node": ...,
"output": ...}}}``. ``Graph.compile`` turns it into a ``Graph``.

A record can be wrong in two ways. If it doesn't have this shape, it is refused as it is read, with
``invalid_record``. If it has the shape but says something that can't work, such as a graph output that names a node
the record doesn't have, that is a problem of the graph, in ``graph.problems``. It is never an exception.
"""

import json
import os
import subprocess
import sys
from collections import UserList, deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any

import pytest
from conductor import (
    SKIPPED,
    Graph,
    GraphInputRecord,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Run,
    Series,
    Source,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import MISSING, ValidationError

from tests.test_core.nodes import input_entry, registry_of


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
    yes: str = SKIPPED
    no: str = SKIPPED


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
    """Returns the problems the record validator raises for ``record``, as ``(type, loc, ctx["errors"][0]["type"])``.

    ``record`` may be JSON text or plain data.
    """
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(record if isinstance(record, str) else json.dumps(record), REGISTRY)
    return [(p.type, p.loc, p.ctx["errors"][0]["type"]) for p in refused.value.problems]


# -- the record's shape ---------------------------------------------------------


def test_a_node_record_and_a_graph_record_have_these_fields_in_this_order():
    assert list(NodeRecord.model_fields) == ["node", "version", "inputs", "off", "display"]
    assert list(GraphRecord.model_fields) == ["inputs", "nodes", "outputs"]


def test_a_bare_node_saves_as_its_node():
    """Empty fields, and a version of 1, are left out when a node record is saved."""
    assert NodeRecord(node="upper").model_dump() == {"node": "upper"}
    assert NodeRecord(
        node="upper", version=2, inputs={"text": input_entry({"constant": "hi"})}
    ).model_dump(mode="json") == {
        "node": "upper",
        "version": 2,
        "inputs": {"text": {"constant": "hi"}},
    }
    graph = GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {
                "a": {
                    "node": "upper",
                    "version": 1,
                    "display": {},
                    "inputs": {"text": {"graph_input": "text"}},
                }
            },
            "outputs": {"result": {"node": "a", "output": "result"}},
        }
    )
    assert graph.model_dump(mode="json") == {
        "inputs": {"text": {}},
        "nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}},
        "outputs": {"result": {"node": "a", "output": "result"}},
    }


def test_each_kind_of_input_entry_is_its_own_class_and_a_dict_validates_to_the_same():
    written = NodeRecord(
        node="both",
        inputs={
            "yes": input_entry({"constant": "y"}),
            "no": input_entry({"graph_input": "no"}),
        },
    )
    edged = NodeRecord(
        node="upper", inputs={"text": input_entry({"sources": (Source("a", "result", "email"),)})}
    )

    assert (
        NodeRecord.model_validate(
            {
                "node": "both",
                "inputs": {"yes": {"constant": "y"}, "no": {"graph_input": "no"}},
            }
        )
        == written
    )
    assert (
        NodeRecord.model_validate(
            {
                "node": "upper",
                "inputs": {
                    "text": {"sources": [{"node": "a", "output": "result", "field": "email"}]}
                },
            }
        )
        == edged
    )
    assert edged.model_dump(mode="json")["inputs"] == {
        "text": {"sources": [{"node": "a", "output": "result", "field": "email"}]}
    }
    assert input_entry({"graph_input": "no"}).model_dump(mode="json") == {"graph_input": "no"}


def test_the_record_reads_back_what_it_saved():
    graph = GraphRecord.model_validate(
        {
            "nodes": {
                "a": {
                    "node": "t",
                    "inputs": {"x": {"constant": {"refs": ["q.r"]}}},
                    "display": {"x": 10},
                },
                "b": {
                    "node": "t",
                    "version": 2,
                    "inputs": {"y": {"sources": [{"node": "a", "output": "result"}]}},
                },
            }
        }
    )

    assert GraphRecord.model_validate_json(graph.model_dump_json()) == graph
    assert graph.nodes["b"].inputs["y"] == input_entry({"sources": (Source("a", "result"),)})


def test_display_is_the_hosts_own_and_takes_any_key():
    assert NodeRecord(node="t", display={"x": 10, "y": 20, "anything": [1]}).display == {
        "x": 10,
        "y": 20,
        "anything": [1],
    }


@pytest.mark.parametrize("not_a_name", ["a.b", "a/x", "a@1", "a~c", ""])
def test_a_source_names_a_node_and_an_output_each_a_name(not_a_name: str):
    """Each part of a source is a valid name, except its field.

    A field of a record may have dots of its own, but the node and the output may not.
    """
    with pytest.raises(ValidationError):
        GraphRecord.model_validate(
            {
                "nodes": {
                    "b": {
                        "node": "b",
                        "inputs": {"x": {"sources": [{"node": not_a_name, "output": "result"}]}},
                    }
                }
            }
        )
    with pytest.raises(ValidationError):
        GraphRecord.model_validate(
            {
                "nodes": {
                    "b": {
                        "node": "b",
                        "inputs": {"x": {"sources": [{"node": "a", "output": not_a_name}]}},
                    }
                }
            }
        )


def test_a_field_of_a_source_may_have_dots_but_never_starts_with_one():
    record = GraphRecord.model_validate(
        {
            "nodes": {
                "b": {
                    "node": "b",
                    "inputs": {
                        "x": {
                            "sources": [
                                {"node": "a", "output": "result", "field": "Amount in USD."}
                            ]
                        }
                    },
                }
            }
        }
    )

    assert record.nodes["b"].inputs["x"] == input_entry(
        {"sources": (Source("a", "result", "Amount in USD."),)}
    )
    with pytest.raises(ValidationError):
        GraphRecord.model_validate(
            {
                "nodes": {
                    "b": {
                        "node": "b",
                        "inputs": {
                            "x": {"sources": [{"node": "a", "output": "result", "field": ".x"}]}
                        },
                    }
                }
            }
        )
    with pytest.raises(ValidationError):
        GraphRecord.model_validate({"nodes": {"a": {"node": "t", "bindings": {}}}})


def test_the_json_schema_gives_an_input_one_entry_of_a_constant_a_graph_input_or_sources():
    schema = GraphRecord.model_json_schema()
    kinds = schema["$defs"]["InputEntry"]["anyOf"]

    assert schema["$defs"]["NodeRecord"]["properties"]["inputs"]["additionalProperties"] == {
        "$ref": "#/$defs/InputEntry"
    }
    assert kinds == [
        {"$ref": "#/$defs/ConstantEntry"},
        {"$ref": "#/$defs/GraphInputEntry"},
        {"$ref": "#/$defs/SourcesEntry"},
    ]
    for kind, key in [
        ("ConstantEntry", "constant"),
        ("GraphInputEntry", "graph_input"),
        ("SourcesEntry", "sources"),
    ]:
        entry = schema["$defs"][kind]
        assert list(entry["properties"]) == entry["required"] == [key]
        assert entry["additionalProperties"] is False
    assert schema["properties"]["inputs"]["additionalProperties"] == {
        "$ref": "#/$defs/GraphInputRecord"
    }
    assert list(schema["$defs"]["GraphInputRecord"]["properties"]) == ["default"]
    assert "required" not in schema["$defs"]["GraphInputRecord"]
    assert schema["$defs"]["Source"]["required"] == ["node", "output"]


def test_the_order_of_an_inputs_sources_never_tells_two_records_apart():
    """The order of an input's sources never makes two records different.

    Several sources on one input come from different branches of one node. So nothing depends on their order.
    """

    def record(*sources: str) -> GraphRecord:
        return GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "d": {"node": "decide", "inputs": {"text": {"graph_input": "text"}}},
                    "u": {
                        "node": "upper",
                        "inputs": {
                            "text": {
                                "sources": [{"node": "d", "output": source} for source in sources]
                            }
                        },
                    },
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        )

    one_way, other_way = record("yes", "no"), record("no", "yes")

    assert one_way == other_way and one_way.model_dump_json() == other_way.model_dump_json()
    assert (
        Graph.compile(one_way, REGISTRY).fingerprint
        == Graph.compile(other_way, REGISTRY).fingerprint
    )


# -- the record validator: what is not a record's shape is refused ----------------------


@pytest.mark.parametrize("bad", ["a.b", "a/b", "a@b", "a~b", ""])
def test_a_node_id_that_is_not_a_name_is_refused(bad):
    assert _refused({"nodes": {bad: {"node": "upper"}}}) == [
        ("invalid_record", ("record", "nodes", bad, "[key]"), "not_a_name")
    ]


def test_a_source_listed_twice_is_refused_naming_the_source():
    record = {
        "nodes": {
            "a": {"node": "text"},
            "b": {
                "node": "upper",
                "inputs": {
                    "text": {
                        "sources": [
                            {"node": "a", "output": "result"},
                            {"node": "a", "output": "result"},
                        ]
                    }
                },
            },
        }
    }

    with pytest.raises(ValidationError, match="a.result is listed twice"):
        GraphRecord.model_validate(record)


@pytest.mark.parametrize(
    "entry",
    [
        {"constant": "a", "graph_input": "text", "default": "b"},
        {"constant": "a", "sources": [{"node": "a", "output": "result"}]},
        {"graph_input": "text", "sources": [{"node": "a", "output": "result"}]},
        {"default": "b"},
    ],
    ids=[
        "constant-and-default",
        "constant-and-sources",
        "graph-input-and-sources",
        "default-alone",
    ],
)
def test_an_input_given_other_than_exactly_one_of_a_constant_a_graph_input_or_sources_is_refused(
    entry,
):
    record = {"nodes": {"a": {"node": "text"}, "b": {"node": "upper", "inputs": {"text": entry}}}}

    assert _refused(record) == [
        (
            "invalid_record",
            ("record", "nodes", "b", "inputs", "text"),
            "one_of_constant_graph_input_sources",
        )
    ]


@pytest.mark.parametrize(
    ("version", "error"),
    [
        (True, "bool_is_not_a_number"),
        (False, "bool_is_not_a_number"),
        (1.0, "int_type"),
        ("2", "int_type"),
    ],
)
@pytest.mark.parametrize("given_as", ["json", "data"])
def test_a_version_that_is_not_a_whole_number_is_refused(version, error, given_as):
    """A version is read as JSON's own whole number, from text and from plain data alike: ``true`` is not 1."""
    data = {
        "nodes": {"u": {"node": "upper", "version": version, "inputs": {"text": {"constant": "x"}}}}
    }
    record = json.dumps(data) if given_as == "json" else data

    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(record, REGISTRY)

    assert [(p.type, p.loc, p.ctx["errors"][0]["type"]) for p in refused.value.problems] == [
        ("invalid_record", ("record", "nodes", "u", "version"), error)
    ]


_SOURCE = {"node": "a", "output": "result"}


@pytest.mark.parametrize(
    ("text", "loc", "error"),
    [
        ('{"nodes": {"a": {"node": "text"}, "a": {"node": "text"}}}', ("record",), "repeated_key"),
        (
            '{"nodes": {"b": {"node": "upper", "inputs": {"text": {"constant": "x", "constant": "y"}}}}}',
            ("record",),
            "repeated_key",
        ),
        (
            json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"a.b": {"constant": "x"}}}}}),
            ("record", "nodes", "b", "inputs", "a.b", "[key]"),
            "not_a_name",
        ),
        (
            json.dumps(
                {"nodes": {"b": {"node": "upper", "inputs": {"text": {"graph_input": "a.b"}}}}}
            ),
            ("record", "nodes", "b", "inputs", "text", "graph_input"),
            "not_a_name",
        ),
        (
            json.dumps({"inputs": {"a.b": {}}}),
            ("record", "inputs", "a.b", "[key]"),
            "not_a_name",
        ),
        (
            json.dumps(
                {
                    "nodes": {"u": {"node": "upper"}},
                    "outputs": {"a.b": {"node": "u", "output": "result"}},
                }
            ),
            ("record", "outputs", "a.b", "[key]"),
            "not_a_name",
        ),
        (
            json.dumps(
                {
                    "nodes": {
                        "b": {
                            "node": "upper",
                            "inputs": {
                                "text": {
                                    "sources": [{"node": "a", "output": "result", "field": ".x"}]
                                }
                            },
                        }
                    }
                }
            ),
            ("record", "nodes", "b", "inputs", "text", "sources", 0, "field"),
            "not_a_field",
        ),
        (
            json.dumps(
                {
                    "nodes": {
                        "a": {"node": "text"},
                        "b": {"node": "upper", "inputs": {"text": {"sources": [_SOURCE, _SOURCE]}}},
                    }
                }
            ),
            ("record", "nodes", "b", "inputs", "text", "sources"),
            "repeated_source",
        ),
        (
            json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {"sources": []}}}}}),
            ("record", "nodes", "b", "inputs", "text", "sources"),
            "no_sources",
        ),
        (
            json.dumps(
                {
                    "inputs": {"text": {}},
                    "nodes": {"b": {"node": "two", "inputs": {"text": {"graph_input": "text"}}}},
                    "outputs": {"zed": {"node": "b", "output": "result", "field": "zed"}},
                }
            ),
            ("record", "outputs", "zed"),
            "output_has_field",
        ),
        (
            json.dumps(
                {
                    "inputs": {"text": {}},
                    "nodes": {
                        "b": {
                            "node": "upper",
                            "inputs": {"text": {"constant": "x", "graph_input": "text"}},
                        }
                    },
                }
            ),
            ("record", "nodes", "b", "inputs", "text"),
            "one_of_constant_graph_input_sources",
        ),
        (
            json.dumps(
                {
                    "nodes": {
                        "b": {
                            "node": "upper",
                            "inputs": {"text": {"graph_input": "text", "default": "y"}},
                        }
                    },
                }
            ),
            ("record", "nodes", "b", "inputs", "text", "default"),
            "extra_forbidden",
        ),
        (
            json.dumps({"nodes": {"b": {"node": "upper", "inputs": {"text": {}}}}}),
            ("record", "nodes", "b", "inputs", "text"),
            "one_of_constant_graph_input_sources",
        ),
    ],
    ids=[
        "repeated_key-node",
        "repeated_key-entry",
        "not_a_name-input",
        "not_a_name-graph_input",
        "not_a_name-declared_graph_input",
        "not_a_name-graph_output",
        "not_a_field",
        "repeated_source",
        "no_sources",
        "output_has_field",
        "one_of_constant_graph_input_sources",
        "default_on_an_entry",
        "an_entry_of_no_kind",
    ],
)
def test_each_refusal_of_the_record_validator_is_one_problem_at_its_place_saying_why(
    text: str, loc: tuple, error: str
):
    """Each fault the record validator finds is one problem, at the place of the fault, saying why.

    A fault in an input's entry is placed at the input. A name or a field that isn't valid is placed where it is
    written. A key written twice is found while the JSON is parsed, before any place in the record is known, so it is
    placed at the record itself.
    """
    assert _refused(text) == [("invalid_record", loc, error)]


def test_a_refusal_of_a_name_says_which_text_is_not_a_name():
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(
            json.dumps(
                {"nodes": {"b": {"node": "upper", "inputs": {"text": {"graph_input": "a.b"}}}}}
            ),
            REGISTRY,
        )

    [problem] = refused.value.problems
    assert problem.ctx["reason"].startswith("'a.b' isn't a valid name") and problem.msg.endswith(
        "Fix the record's JSON."
    )


# -- compiling: what a record says that cannot be meant ---------------------------


def test_several_edges_into_one_input_that_could_each_give_it_a_value_are_a_problem_naming_each():
    nodes = {
        "first": {"node": "text", "inputs": {"value": {"constant": "a"}}},
        "second": {"node": "text", "inputs": {"value": {"constant": "b"}}},
        "j": {
            "node": "join",
            "inputs": {
                "texts": {
                    "sources": [
                        {"node": "second", "output": "result"},
                        {"node": "first", "output": "result"},
                    ]
                }
            },
        },
    }
    graph = Graph.compile(GraphRecord.model_validate({"nodes": nodes}), REGISTRY)

    assert [(p.type, p.loc, p.ctx["sources"]) for p in graph.problems] == [
        ("several_edges_into_one_input", ("j", "texts"), ["first.result", "second.result"])
    ]
    # One fix, the one that always works. Making outputs branches helps only for outputs of one node, so it is on
    # the problems page, not in the message.
    assert graph.problems[0].msg.endswith(
        "more than one of them can give it a value on the same call. Add a node with an input for each edge."
    )


def test_an_edge_from_a_node_the_graph_does_not_have_is_a_problem_on_the_input_it_leads_into():
    out_of = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "b": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "gone", "output": "result"}]}},
                    }
                },
                "outputs": {"result": {"node": "b", "output": "result"}},
            }
        ),
        REGISTRY,
    )

    assert [(p.type, p.loc, p.ctx["missing"]) for p in out_of.problems] == [
        ("edge_names_missing_node", ("b", "text"), "gone")
    ]


def test_a_constant_for_an_input_the_node_does_not_have_is_unknown_input():
    assert _problems(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"constant": "a"}, "colour": {"constant": "red"}},
                    }
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        )
    ) == [("unknown_input", ("u", "colour"))]


def test_a_required_input_with_no_entry_gets_no_value_on_any_node():
    first = GraphRecord.model_validate({"nodes": {"u": {"node": "upper"}}})
    later = GraphRecord.model_validate(
        {
            "nodes": {
                "t": {"node": "text"},
                "b": {
                    "node": "both",
                    "inputs": {"yes": {"sources": [{"node": "t", "output": "result"}]}},
                },
            }
        }
    )

    assert _problems(first) == [("input_has_no_value", ("u", "text"))]
    assert _problems(later) == [("input_has_no_value", ("b", "no"))]


# -- what a graph takes and gives -------------------------------------------------


def test_graph_inputs_and_graph_outputs_are_the_ones_the_record_declares():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"value": {}, "yes": {}},
                "nodes": {
                    "a": {"node": "text", "inputs": {"value": {"graph_input": "value"}}},
                    "b": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}},
                    },
                    "c": {
                        "node": "both",
                        "inputs": {"no": {"constant": "!"}, "yes": {"graph_input": "yes"}},
                    },
                },
                "outputs": {
                    "b_result": {"node": "b", "output": "result"},
                    "c_result": {"node": "c", "output": "result"},
                },
            }
        ),
        REGISTRY,
    )

    assert graph.problems == []
    assert {name: i.required for name, i in graph.inputs.items()} == {"value": True, "yes": True}
    assert list(graph.outputs) == ["b_result", "c_result"]
    assert graph.run({"value": "v", "yes": "y"}).outputs == {"b_result": "V", "c_result": "y!"}


def test_an_optional_input_with_no_entry_gets_the_nodes_default_and_is_no_graph_input():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"t": {"node": "text"}},
                "outputs": {"result": {"node": "t", "output": "result"}},
            }
        ),
        REGISTRY,
    )

    assert graph.problems == [] and graph.inputs == {}
    assert graph.run().outputs == {"result": ""}


def test_a_constant_is_not_a_graph_input_and_a_default_is_a_graph_input_with_that_default():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"no": {"default": "n"}},
                "nodes": {
                    "b": {
                        "node": "both",
                        "inputs": {
                            "yes": {"constant": "y"},
                            "no": {"graph_input": "no"},
                        },
                    }
                },
                "outputs": {"result": {"node": "b", "output": "result"}},
            }
        ),
        REGISTRY,
    )

    assert (
        list(graph.inputs) == ["no"]
        and not graph.inputs["no"].required
        and graph.inputs["no"].default == "n"
    )
    assert graph.run().outputs == {"result": "yn"}
    assert graph.run({"no": "!"}).outputs == {"result": "y!"}


def test_a_graph_input_on_a_node_an_edge_leads_into_is_given_by_the_caller_like_any_other():
    record = GraphRecord.model_validate(
        {
            "inputs": {"value": {}, "no": {"default": "n"}},
            "nodes": {
                "a": {"node": "text", "inputs": {"value": {"graph_input": "value"}}},
                "b": {
                    "node": "both",
                    "inputs": {
                        "no": {"graph_input": "no"},
                        "yes": {"sources": [{"node": "a", "output": "result"}]},
                    },
                },
            },
            "outputs": {"result": {"node": "b", "output": "result"}},
        }
    )
    graph = Graph.compile(record, REGISTRY)

    assert graph.problems == [] and list(graph.inputs) == ["value", "no"]
    assert graph.run({"value": "y"}).outputs == {"result": "yn"} and graph.run(
        {"value": "y", "no": "!"}
    ).outputs == {"result": "y!"}


def test_one_graph_input_on_two_inputs_with_no_default_in_the_record_is_required_whatever_each_inputs_own_default():
    """One graph input on two node inputs, with no default in the record, is required.

    ``text``'s ``value`` has a default of its own, and ``upper``'s ``text`` has none. Neither counts. Both inputs get
    the caller's one value, so the caller must give it.
    """
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"x": {}},
                "nodes": {
                    "a": {"node": "text", "inputs": {"value": {"graph_input": "x"}}},
                    "b": {"node": "upper", "inputs": {"text": {"graph_input": "x"}}},
                },
                "outputs": {
                    "a_result": {"node": "a", "output": "result"},
                    "b_result": {"node": "b", "output": "result"},
                },
            }
        ),
        REGISTRY,
    )

    assert graph.problems == []
    assert {name: (i.required, i.default) for name, i in graph.inputs.items()} == {
        "x": (True, MISSING)
    }
    assert graph.run({"x": "q"}).outputs == {"a_result": "q", "b_result": "Q"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run()
    assert [(p.type, p.loc) for p in refused.value.problems] == [("missing_input", ("inputs", "x"))]


def test_a_graph_input_is_declared_once_and_its_one_default_feeds_every_input_that_takes_it():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"x": {"default": "p"}},
                "nodes": {
                    "a": {"node": "text", "inputs": {"value": {"graph_input": "x"}}},
                    "b": {"node": "upper", "inputs": {"text": {"graph_input": "x"}}},
                },
                "outputs": {
                    "a_result": {"node": "a", "output": "result"},
                    "b_result": {"node": "b", "output": "result"},
                },
            }
        ),
        REGISTRY,
    )

    assert graph.problems == []
    assert graph.inputs["x"].default == "p" and not graph.inputs["x"].required
    assert graph.run({}).outputs == {"a_result": "p", "b_result": "P"}


def test_a_graph_input_the_record_does_not_declare_is_a_problem_at_the_input_that_takes_it():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"y": {}},
                "nodes": {
                    "a": {"node": "text", "inputs": {"value": {"graph_input": "y"}}},
                    "b": {"node": "upper", "inputs": {"text": {"graph_input": "x"}}},
                },
            }
        ),
        REGISTRY,
    )

    assert [(p.type, p.loc, p.ctx) for p in graph.problems] == [
        (
            "graph_input_not_declared",
            ("b", "text"),
            {"node_id": "b", "input": "text", "graph_input": "x", "declared": ["y"]},
        )
    ]
    assert list(graph.inputs) == ["y"]


def test_a_declared_graph_input_no_input_takes_is_the_graphs_problem_at_no_node():
    """A declared graph input that no entry names has no type, so it is a problem of the graph.

    An entry on a node whose definition isn't found still names its graph input, so only ``spare`` is a problem.
    """
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"x": {}, "spare": {"default": 1}, "lost": {}},
                "nodes": {
                    "a": {"node": "text", "inputs": {"value": {"graph_input": "x"}}},
                    "g": {"node": "ghost", "inputs": {"value": {"graph_input": "lost"}}},
                },
            }
        ),
        REGISTRY,
    )

    assert [(p.type, p.loc, p.ctx) for p in graph.problems if p.type != "unknown_node"] == [
        ("graph_input_not_used", (), {"graph_input": "spare"})
    ]


def test_a_graph_inputs_default_its_type_refuses_is_an_invalid_value_at_the_first_input_that_takes_it():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"x": {"default": 3}},
                "nodes": {
                    "a": {"node": "text", "inputs": {"value": {"graph_input": "x"}}},
                    "b": {"node": "upper", "inputs": {"text": {"graph_input": "x"}}},
                },
            }
        ),
        REGISTRY,
    )

    assert [(p.type, p.loc) for p in graph.problems] == [("invalid_value", ("a", "value"))]
    assert graph.inputs["x"].required


@node("as-int", title="As int", description="For a test.")
def _as_int(n: int) -> int:
    return n


@node("as-str", title="As str", description="For a test.")
def _as_str(n: str) -> str:
    return n


@pytest.mark.parametrize(("a", "b"), [("as-int", "as-str"), ("as-str", "as-int")])
def test_two_types_for_one_graph_input_are_used_two_ways_whichever_node_comes_first_and_its_default_is_not_read(
    a: str, b: str
):
    """A type clash shows whatever the order of the nodes, and the default isn't read by either type.

    ``"5"`` is no ``int``. If the default were read by the first type before the clash was found, the graph would
    show ``invalid_value`` alone when the ``int`` node came first.
    """
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"x": {"default": "5"}},
                "nodes": {
                    "a": {"node": a, "inputs": {"n": {"graph_input": "x"}}},
                    "b": {"node": b, "inputs": {"n": {"graph_input": "x"}}},
                },
            }
        ),
        registry_of(_as_int, _as_str),
    )

    assert [(p.type, p.loc) for p in graph.problems] == [("graph_input_used_two_ways", ("b", "n"))]


def test_a_graph_output_naming_a_node_or_an_output_the_record_lacks_is_the_graphs_problem_at_no_node():
    """A graph output that names a missing node or output is a problem of the graph, at no node.

    The node here is called ``outputs`` on purpose. A problem placed at that node's id would then look like one of
    the node's own, and the test would catch it.
    """
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "outputs": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}
                },
                "outputs": {
                    "gone": {"node": "nobody", "output": "result"},
                    "missing": {"node": "outputs", "output": "nope"},
                    "found": {"node": "outputs", "output": "result"},
                },
            }
        ),
        REGISTRY,
    )

    assert [(p.type, p.loc, p.ctx) for p in graph.problems] == [
        (
            "graph_output_names_missing_node",
            (),
            {"output": "gone", "source": "nobody.result", "missing": "nobody"},
        ),
        (
            "graph_output_not_found",
            (),
            {"output": "missing", "source": "outputs.nope", "outputs": ["outputs.result"]},
        ),
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
    """Returns the problems ``make_graph`` raises for ``arguments``, as ``(type, loc)``.

    The graph's nodes are ``_hello`` and ``_shout``.
    """
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, _shout, **arguments)
    return [(p.type, p.loc) for p in refused.value.problems]


def test_make_graph_with_no_edges_chains_each_node_to_the_one_before():
    graph = make_graph(_hello, _shout, constants={"hello.name": "ida"})

    assert graph == GraphRecord.model_validate(
        {
            "nodes": {
                "hello": {"node": "hello", "inputs": {"name": {"constant": "ida"}}},
                "shout": {
                    "node": "shout",
                    "inputs": {"text": {"sources": [{"node": "hello", "output": "result"}]}},
                },
            },
            "outputs": {"result": {"node": "shout", "output": "result"}},
        }
    )
    assert Graph.compile(graph, PLAIN).run().outputs == {"result": "HELLO IDA!"}


def test_make_graph_numbers_a_node_used_twice():
    graph = make_graph(_hello, _shout, _shout, constants={"hello.name": "ida", "shout-2.times": 2})

    assert list(graph.nodes) == ["hello", "shout-1", "shout-2"]
    assert [
        (node_id, n.inputs["text"]) for node_id, n in graph.nodes.items() if node_id != "hello"
    ] == [
        ("shout-1", input_entry({"sources": (Source("hello", "result"),)})),
        ("shout-2", input_entry({"sources": (Source("shout-1", "result"),)})),
    ]
    assert Graph.compile(graph, PLAIN).run().outputs == {"result": "HELLO IDA!!!"}


def test_make_graph_skips_an_id_another_node_already_has():
    @node("hello-1", title="Other", description="For a test.")
    def _other(name: str) -> str:
        return name

    record = make_graph(
        _hello,
        _hello,
        _other,
        edges=[],
        constants={"hello-2.name": "a", "hello-3.name": "b", "hello-1.name": "c"},
        outputs={"a": "hello-2.result", "b": "hello-3.result", "c": "hello-1.result"},
    )
    assert list(record.nodes) == ["hello-2", "hello-3", "hello-1"]


def test_make_graph_with_edges_takes_those_as_every_edge_and_matches_nothing_by_name():
    graph = make_graph(
        _hello, _shout, edges=[], outputs={"hello": "hello.result", "shout": "shout.result"}
    )

    assert not any(
        "sources" in n.model_dump(mode="json").get("inputs", {}).get(name, {})
        for n in graph.nodes.values()
        for name in n.inputs
    )
    assert list(Graph.compile(graph, PLAIN).inputs) == ["name", "text"]


def test_make_graph_without_edges_refuses_nodes_it_cannot_chain_and_says_to_pass_edges():
    with pytest.raises(InvalidCallError, match="Pass edges=") as refused:
        make_graph(_email, _count, outputs={"email": "email.result", "result": "count.result"})
    assert [(p.type, p.loc) for p in refused.value.problems] == [("not_a_chain", ("count",))]
    with pytest.raises(InvalidCallError) as refused:
        make_graph(
            _count,
            _hello,
            _shout,
            constants={"hello.name": "x"},
            outputs={"count": "count.result", "result": "shout.result"},
        )
    assert [(p.type, p.loc, p.ctx["inputs"]) for p in refused.value.problems] == [
        ("not_a_chain", ("hello",), [])
    ]
    assert refused.value.problems[0].ctx == {
        "node_id": "hello",
        "before": "count",
        "inputs": [],
        "outputs": ["result"],
    }
    assert refused.value.problems[0].msg == (
        "make_graph can't wire 'hello' after 'count' by itself. That needs exactly one required input on 'hello' "
        "that constants=, defaults= and inputs= don't name (it has none) "
        "and one output on 'count' (it has result). Pass edges= to make_graph."
    )


def test_make_graph_reports_only_the_gap_in_a_chain_not_the_graph_outputs_that_would_follow_from_it():
    """Unchained, both nodes' ``result`` would become graph outputs of one name. That clash is the gap's doing, so
    the author is shown one problem for one mistake."""
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_email, _count)
    assert [(p.type, p.loc) for p in refused.value.problems] == [("not_a_chain", ("count",))]


def test_make_graph_reports_an_edge_to_a_node_it_was_not_given_alone_not_the_ports_that_would_follow():
    """With the edge left out, both ``text`` inputs would become graph inputs, and both ``result`` outputs graph
    outputs, and each pair would clash. Those clashes are the typo's doing, so the author is shown the typo."""
    for edge in [("shuot.result", "loud.text"), ("shout.result", "luod.text")]:
        with pytest.raises(InvalidCallError) as refused:
            make_graph(_shout, ("loud", _shout), edges=[edge])
        assert [p.type for p in refused.value.problems] == ["node_not_in_graph"]


def test_make_graph_with_a_gap_in_its_chain_still_reports_the_places_inputs_and_outputs_name():
    """Only what follows from the gap is left out. A place the author wrote in ``outputs=`` or ``inputs=`` that
    names no node is their own fault, and is listed beside the gap."""
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_email, _count, outputs={"x": "gret.result"}, inputs={"y": "nope.text"})
    assert sorted((p.type, p.loc) for p in refused.value.problems) == [
        ("node_not_in_graph", ("gret",)),
        ("node_not_in_graph", ("nope",)),
        ("not_a_chain", ("count",)),
    ]


def test_make_graph_never_chains_or_opens_an_input_a_hook_reads_and_compile_says_it_needs_a_constant():
    def width_outputs(width: int) -> dict[str, Any]:
        return {"result": str}

    @node("pad", title="Pad", description="For a test.", compute_outputs=width_outputs)
    def pad(text: str, width: int) -> dict[str, Any]:
        return {"result": text.ljust(width)}

    @node("blank", title="Blank", description="For a test.", compute_outputs=width_outputs)
    def blank(width: int) -> dict[str, Any]:
        return {"result": " " * width}

    record = make_graph(_hello, pad, outputs={"padded": "pad.result"})

    assert record.nodes["pad"].inputs == {
        "text": input_entry({"sources": (Source("hello", "result"),)})
    }
    graph = Graph.compile(record, registry_of(_hello, pad))
    assert list(graph.inputs) == ["name"] and [(p.type, p.loc) for p in graph.problems] == [
        ("input_has_no_value_constant_only", ("pad", "width"))
    ]
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, blank, outputs={"blank": "blank.result"})
    assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [
        (
            "not_a_chain_constant_only",
            ("blank",),
            {
                "node_id": "blank",
                "before": "hello",
                "inputs": ["width"],
            },
        ),
    ]
    assert refused.value.problems[0].msg == (
        "make_graph can't wire 'blank' after 'hello' by itself, because only a constant can give width a value. "
        "Give width a constant, and pass edges= to make_graph."
    )


def test_make_graph_refuses_a_node_whose_hook_gives_its_outputs_when_no_edge_or_outputs_reads_them():
    """A hook's outputs are known only when the graph is compiled, so ``make_graph`` can't make them graph outputs.

    Without ``outputs=``, the graph would return nothing at all, so ``make_graph`` says so at the node. Named in
    ``outputs=``, or read by an edge, the hook's outputs are used as given.
    """

    def by_names(names: str) -> dict[str, Any]:
        return {name: str for name in names.split(",")}

    @node("pick", title="Pick", description="For a test.", compute_outputs=by_names)
    def pick(names: str) -> dict[str, Any]:
        return {name: name.upper() for name in names.split(",")}

    with pytest.raises(InvalidCallError, match="outputs=") as refused:
        make_graph(pick, constants={"pick.names": "a,b"})
    assert [(p.type, p.loc) for p in refused.value.problems] == [
        ("hook_outputs_not_named", ("pick",))
    ]
    named = make_graph(pick, constants={"pick.names": "a,b"}, outputs={"a": "pick.a"})
    assert Graph.compile(named, registry_of(pick)).run().outputs == {"a": "A"}
    make_graph(pick, upper, constants={"pick.names": "a"}, edges=[("pick.a", "upper.text")])


def test_make_graph_refuses_a_place_on_a_node_it_was_not_given():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, constants={"nobody.name": "x"})
    assert [(p.type, p.ctx["nodes"]) for p in refused.value.problems] == [
        ("node_not_in_graph", ["hello"])
    ]
    assert _made(
        edges=[("nobody.result", "shout.text")],
        constants={"hello.name": "x"},
        outputs={"result": "shout.result", "hello": "hello.result"},
    ) == [("node_not_in_graph", ("nobody",))]
    assert _made(
        edges=[],
        inputs={"who": "nobody.name"},
        outputs={"result": "shout.result", "hello": "hello.result"},
    ) == [("node_not_in_graph", ("nobody",))]
    assert _made(outputs={"result": "nobody.result"}) == [("node_not_in_graph", ("nobody",))]


def test_make_graph_writes_wrong_inputs_outputs_and_constants_as_given_and_compile_reports_each():
    record = make_graph(
        _hello,
        _shout,
        edges=[("hello.reslt", "shout.text"), ("hello.result", "shout.txt")],
        constants={"hello.nmae": "x"},
        defaults={"hello.name": 3},
    )
    assert sorted((p.type, p.loc) for p in Graph.compile(record, PLAIN).problems) == [
        ("invalid_value", ("hello", "name")),
        ("unknown_input", ("hello", "nmae")),
        ("unknown_input", ("shout", "txt")),
        ("unknown_output", ("shout", "text")),
    ]


@pytest.mark.parametrize(
    "value",
    [{"ida", "bo"}, frozenset({"ida"}), [{"ida"}], {"names": {"ida"}}],
    ids=["set", "frozenset", "in-a-list", "in-a-dict"],
)
def test_make_graph_refuses_a_set_as_a_constant_as_set_has_no_order(value: object):
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, constants={"hello.name": value})
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc, [e["type"] for e in problem.ctx["errors"]]) == (
        "invalid_value",
        ("hello", "name"),
        ["set_has_no_order"],
    )


def test_make_graph_writes_a_type_mismatch_and_an_output_a_hook_does_not_give_and_compile_reports_them():
    @node(
        "fan",
        title="Fan",
        description="For a test.",
        compute_outputs=lambda n: {f"o{i}": int for i in range(n)},
    )
    def fan(n: int = 2) -> dict:
        return {f"o{i}": i for i in range(n)}

    mismatched = make_graph(_count, _hello, edges=[("count.result", "hello.name")])
    assert [(p.type, p.loc) for p in Graph.compile(mismatched, PLAIN).problems] == [
        ("type_mismatch", ("hello", "name"))
    ]
    not_given = make_graph(fan, _shout, edges=[("fan.o5", "shout.text")])
    assert [p.type for p in Graph.compile(not_given, registry_of(fan, _shout)).problems] == [
        "unknown_output"
    ]


def test_make_graph_writes_defaults_and_compile_reports_a_list_for_an_input_of_one_value():
    record = make_graph(_hello, defaults={"hello.name": "you"})
    assert record.inputs == {"name": GraphInputRecord(default="you")}
    assert record.nodes["hello"].inputs == {"name": input_entry({"graph_input": "name"})}
    assert Graph.compile(record, PLAIN).run().outputs == {"result": "hello you"}
    assert Graph.compile(record, PLAIN).run({"name": "ida"}).outputs == {"result": "hello ida"}
    listed = make_graph(_hello, defaults={"hello.name": ["you"]})
    assert [(p.type, p.loc) for p in Graph.compile(listed, PLAIN).problems] == [
        ("invalid_value", ("hello", "name"))
    ]


def test_make_graph_makes_a_default_a_graph_input_on_a_node_an_edge_leads_into():
    record = make_graph(
        _hello, _shout, constants={"hello.name": "ida"}, defaults={"shout.times": 2}
    )
    graph = Graph.compile(record, PLAIN)

    assert list(graph.inputs) == ["times"]
    assert graph.run().outputs == {"result": "HELLO IDA!!"} and graph.run({"times": 3}).outputs == {
        "result": "HELLO IDA!!!"
    }


# -- make_graph declares the graph's inputs and outputs --------------------------


def test_make_graph_declares_each_open_input_of_a_node_no_edge_leads_into_and_each_output_no_edge_leaves():
    record = make_graph(_hello, _shout)

    assert record.nodes["hello"].inputs == {"name": input_entry({"graph_input": "name"})}
    # an edge leads into shout, so its optional input keeps the node's default
    assert "times" not in record.nodes["shout"].inputs
    assert record.outputs == {"result": Source("shout", "result")}


def test_make_graph_refuses_two_graph_inputs_or_two_graph_outputs_that_would_share_a_name():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(upper, upper, edges=[])

    assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [
        (
            "graph_inputs_share_a_name",
            ("inputs", "text"),
            {"name": "text", "inputs": ["upper-1.text", "upper-2.text"]},
        ),
        (
            "graph_outputs_share_a_name",
            ("outputs", "result"),
            {"name": "result", "outputs": ["upper-1.result", "upper-2.result"]},
        ),
    ]
    assert [
        "make_graph's inputs=" in refused.value.problems[0].msg,
        "make_graph's outputs=" in refused.value.problems[1].msg,
    ] == [True, True]


def test_graph_inputs_come_in_the_order_the_nodes_run_whatever_order_the_record_lists_them_in():
    # a reads from z, so z's graph input comes first, though a comes first by id
    nodes = {
        "z": {"node": "upper", "inputs": {"text": {"graph_input": "first"}}},
        "a": {
            "node": "both",
            "inputs": {
                "yes": {"sources": [{"node": "z", "output": "result"}]},
                "no": {"graph_input": "second"},
            },
        },
    }
    listed = GraphRecord.model_validate({"inputs": {"second": {}, "first": {}}, "nodes": nodes})
    listed_the_other_way = GraphRecord.model_validate(
        {"inputs": {"first": {}, "second": {}}, "nodes": dict(reversed(nodes.items()))}
    )

    assert [
        list(Graph.compile(record, REGISTRY).inputs) for record in (listed, listed_the_other_way)
    ] == [["first", "second"]] * 2


@node("words", title="Words", description="For a test.")
def _words(text: str) -> Series[str]:
    return text.split()


def test_graph_outputs_are_fields_described_as_a_complete_run_gives_them():
    record = GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {
                "w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}},
                "u": {
                    "node": "upper",
                    "inputs": {"text": {"sources": [{"node": "w", "output": "result"}]}},
                },
                "d": {"node": "decide", "inputs": {"text": {"graph_input": "text"}}},
            },
            "outputs": {
                "words": {"node": "w", "output": "result"},
                "loud": {"node": "u", "output": "result"},
                "kept": {"node": "d", "output": "yes"},
            },
        }
    )
    graph = Graph.compile(record, registry_of(_words, upper, decide))

    assert {
        name: (output.type, output.is_series, list(output.loops), output.required)
        for name, output in graph.outputs.items()
    } == {
        "kept": ("str", False, [], False),  # a branch, which may be skipped
        "loud": ("str", True, ["w.result"], True),  # one per word
        "words": ("str", True, [], True),
    }
    assert graph.describe().outputs == graph.outputs
    assert graph.run({"text": "yes no"}).outputs == {
        "kept": "yes no",
        "loud": ["YES", "NO"],
        "words": ["yes", "no"],
    }


def test_make_graph_names_the_graph_inputs_and_outputs_it_is_given():
    record = make_graph(
        upper,
        upper,
        edges=[],
        inputs={"a": "upper-1.text", "b": "upper-2.text"},
        outputs={"x": "upper-1.result", "y": "upper-2.result"},
    )
    graph = Graph.compile(record, REGISTRY)

    assert list(graph.inputs) == ["a", "b"] and list(graph.outputs) == ["x", "y"]
    assert graph.run({"a": "p", "b": "q"}).outputs == {"x": "P", "y": "Q"}


def test_make_graph_feeds_several_inputs_from_one_graph_input_given_a_list():
    record = make_graph(
        upper,
        upper,
        edges=[],
        inputs={"text": ["upper-1.text", "upper-2.text"]},
        outputs={"x": "upper-1.result", "y": "upper-2.result"},
    )
    graph = Graph.compile(record, REGISTRY)

    assert [n.inputs["text"] for n in record.nodes.values()] == [
        input_entry({"graph_input": "text"})
    ] * 2
    assert list(graph.inputs) == ["text"] and graph.run({"text": "p"}).outputs == {
        "x": "P",
        "y": "P",
    }


_EDGE = [("hello.result", "shout.text")]


@pytest.mark.parametrize(
    ("arguments", "loc", "ways"),
    [
        (
            {"constants": {"hello.name": "a"}, "inputs": {"name": "hello.name"}},
            ("hello", "name"),
            ["constants", "inputs"],
        ),
        (
            {"constants": {"hello.name": "a"}, "defaults": {"hello.name": "b"}},
            ("hello", "name"),
            ["constants", "defaults"],
        ),
        (
            {"edges": _EDGE, "constants": {"hello.name": "a", "shout.text": "x"}},
            ("shout", "text"),
            ["constants", "edges"],
        ),
        (
            {"edges": _EDGE, "inputs": {"name": "hello.name", "text": "shout.text"}},
            ("shout", "text"),
            ["inputs", "edges"],
        ),
        (
            {"edges": _EDGE, "defaults": {"hello.name": "a", "shout.text": "x"}},
            ("shout", "text"),
            ["defaults", "edges"],
        ),
        (
            {
                "edges": _EDGE,
                "constants": {"hello.name": "a", "shout.text": "x"},
                "inputs": {"text": "shout.text"},
            },
            ("shout", "text"),
            ["constants", "inputs", "edges"],
        ),
    ],
    ids=[
        "constant-and-inputs",
        "constant-and-default",
        "constant-and-edge",
        "edge-and-inputs",
        "edge-and-default",
        "constant-inputs-and-edge",
    ],
)
def test_make_graph_refuses_one_input_given_two_ways_and_names_the_arguments(arguments, loc, ways):
    """An input gets one of a constant, a graph input or edges. Given two, ``make_graph`` names the arguments that
    gave them, so the author knows which one to take the input out of."""
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, _shout, **arguments)

    [problem] = refused.value.problems
    place = ".".join(loc)
    assert (problem.type, problem.loc, problem.ctx) == (
        "input_given_two_ways",
        loc,
        {"input": place, "ways": ways},
    )
    assert problem.msg == (
        f"More than one of make_graph's arguments gives the input '{place}' a value: {', '.join(ways)}. "
        "An input takes one of a constant, a graph input or edges, so name it in only one of them."
    )


def test_make_graph_takes_inputs_and_defaults_on_one_input_as_one_graph_input():
    record = make_graph(_hello, defaults={"hello.name": "ida"}, inputs={"who": "hello.name"})

    assert Graph.compile(record, registry_of(_hello)).run().outputs == {"result": "hello ida"}


def test_make_graph_does_not_chain_into_an_input_that_inputs_names():
    """A chain skips an input ``inputs=`` names, as it skips one with a constant or a default. So naming one of two
    required inputs leaves the other for the chain."""

    @node("pair", title="Pair", description="For a test.")
    def pair(text: str, mark: str) -> str:
        return text + mark

    record = make_graph(_hello, pair, inputs={"m": "pair.mark"})

    assert record.nodes["pair"].inputs == {
        "text": input_entry({"sources": [{"node": "hello", "output": "result"}]}),
        "mark": input_entry({"graph_input": "m"}),
    }
    assert Graph.compile(record, registry_of(_hello, pair)).run(
        {"name": "ida", "m": "!"}
    ).outputs == {"result": "hello ida!"}


def test_make_graph_with_no_edges_and_the_one_input_named_in_inputs_is_not_a_chain():
    """The chain skips the input ``inputs=`` names. With no other required input left, the nodes can't be chained,
    and the message says why."""
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, _shout, inputs={"t": "shout.text"})

    [problem] = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx["inputs"]) == ("not_a_chain", ("shout",), [])
    assert "that constants=, defaults= and inputs= don't name (it has none)" in problem.msg


# -- the graph a record compiles to -------------------------------------------------


def test_a_compiled_graph_keeps_its_own_definitions_whatever_the_registry_does_later():
    registry = registry_of(upper)
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        registry,
    )

    registry.remove("upper")

    assert graph.run({"text": "a"}).outputs == {"result": "A"}
    assert [p.type for p in Graph.compile(graph.record, registry).problems] == ["unknown_node"]


def test_the_type_decides_one_or_many_so_a_list_given_for_an_input_of_one_value_is_refused():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        REGISTRY,
    )

    assert graph.describe().nodes["u"].per_item_of is None and graph.run({"text": "a"}).outputs == {
        "result": "A"
    }
    with pytest.raises(InvalidCallError) as refused:
        graph.run({"text": ["a"]})
    assert [(p.type, p.loc) for p in refused.value.problems] == [
        ("invalid_input_list_for_one_value", ("inputs", "text"))
    ]


def test_a_list_as_a_constant_or_a_default_for_an_input_of_one_value_is_invalid_value():
    assert _problems(
        GraphRecord.model_validate(
            {
                "nodes": {"u": {"node": "upper", "inputs": {"text": {"constant": ["x", "y"]}}}},
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        )
    ) == [("invalid_value", ("u", "text"))]
    assert _problems(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {"default": ["a", "b"]}},
                "nodes": {
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"graph_input": "text"}},
                    }
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        )
    ) == [("invalid_value", ("u", "text"))]


def test_every_problem_in_a_callers_inputs_is_refused_at_once_unknown_names_first_then_the_inputs_in_order():
    """Every fault in a caller's inputs is reported at once: unknown names first, then the inputs in order.

    The graph inputs are in the order of their nodes' ids, so ``b``'s come before ``u``'s.
    """
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}, "yes": {}, "no": {}},
                "nodes": {
                    "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
                    "b": {
                        "node": "both",
                        "inputs": {"yes": {"graph_input": "yes"}, "no": {"graph_input": "no"}},
                    },
                },
                "outputs": {
                    "b_result": {"node": "b", "output": "result"},
                    "u_result": {"node": "u", "output": "result"},
                },
            }
        ),
        REGISTRY,
    )
    offered = ["yes", "no", "text"]

    def refused(given: dict[Any, Any]) -> list[tuple[str, tuple[str | int, ...], dict[str, Any]]]:
        with pytest.raises(InvalidCallError) as raised:
            graph.with_inputs(given)
        return [(p.type, p.loc, p.ctx) for p in raised.value.problems]

    assert refused({"nope": 1, "text": "a", "zz": 2}) == [
        ("unknown_graph_input", ("inputs", "nope"), {"name": "nope", "inputs": offered}),
        ("unknown_graph_input", ("inputs", "zz"), {"name": "zz", "inputs": offered}),
        ("missing_input", ("inputs", "yes"), {"name": "yes"}),
        ("missing_input", ("inputs", "no"), {"name": "no"}),
    ]
    not_a_text = {
        "name": "no",
        "reason": "Input should be a valid string",
        "errors": [{"type": "string_type", "loc": [], "msg": "Input should be a valid string"}],
    }
    assert refused({"no": 1, "text": ["a", "b"], 1: "x"}) == [
        ("unknown_graph_input", ("inputs", 1), {"name": "1", "inputs": offered}),
        ("missing_input", ("inputs", "yes"), {"name": "yes"}),
        ("invalid_input", ("inputs", "no"), not_a_text),
        (
            "invalid_input_list_for_one_value",
            ("inputs", "text"),
            {"name": "text", "expected": "str", "length": 2},
        ),
    ]
    run = graph.with_inputs({"text": "a", "yes": "y", "no": "n"})
    assert run.with_inputs({"text": "b"}).inputs == {"yes": "y", "no": "n", "text": "b"}


@pytest.mark.parametrize(
    ("hint", "given", "reason"),
    [
        (Annotated[list[str], Param(min_length=1)], [], "at least 1 item"),
        (int | list[int], ["x"], "valid integer"),
    ],
    ids=["a list type's limit", "a union that holds a list"],
)
def test_a_list_a_graph_input_whose_type_takes_a_list_refuses_gives_its_reason_not_to_make_a_series(
    hint: Any, given: list[Any], reason: str
) -> None:
    """When an input whose type takes a list refuses a list, the problem gives the type's reason.

    It doesn't suggest making the input a ``Series``. That advice is only for a list given to an input whose type
    takes no list. Any other refusal gives the type's own reason, as it does for a constant. Here that is a list
    type's own limit, and a union's own refusal, which names the fault with its first option.
    """

    def take(words: hint) -> str:  # type: ignore[valid-type]
        return "taken"

    taking = node("take", title="Take", description="For a test.")(take)
    graph = Graph.compile(make_graph(taking), registry_of(taking))

    with pytest.raises(InvalidCallError) as raised:
        graph.run({"words": given})

    (problem,) = raised.value.problems
    assert problem.type == "invalid_input"
    assert reason in problem.ctx["reason"]


def test_a_graph_input_is_given_by_its_name_never_by_a_source_and_an_item_limit_below_one_is_refused():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}},
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        REGISTRY,
    )

    with pytest.raises(InvalidCallError) as by_source:
        graph.run({"text": "a", "u.text": "b"})
    with pytest.raises(InvalidCallError) as item_limit:
        graph.run({"text": "a"}, max_items=0)

    assert [(p.type, p.loc) for p in by_source.value.problems] == [
        ("unknown_graph_input", ("inputs", "u.text"))
    ]
    assert [p.type for p in item_limit.value.problems] == ["invalid_limit"]
    with pytest.raises(InvalidCallError):
        graph.with_inputs({"text": "a"}).with_max_items(-1)


def test_a_subgraph_with_an_unknown_inner_node_is_one_problem_at_its_node_path():
    broken = subgraph("broken", title="Broken", description="For a test.")(
        GraphRecord.model_validate(
            {
                "nodes": {"x": {"node": "gone"}},
                "outputs": {"result": {"node": "x", "output": "result"}},
            }
        )
    )
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "b": {"node": "broken"},
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        registry_of(upper, broken),
    )

    assert [(problem.type, problem.loc) for problem in graph.problems] == [
        ("unknown_node", ("b", "x"))
    ]


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
PAIR = subgraph("pair", title="Pair", description="For a test.")(
    GraphRecord.model_validate(
        {
            "inputs": {"x_text": {}, "y_text": {}},
            "nodes": {
                "x": {"node": "upper", "inputs": {"text": {"graph_input": "x_text"}}},
                "y": {"node": "upper", "inputs": {"text": {"graph_input": "y_text"}}},
                "b": {
                    "node": "both",
                    "inputs": {
                        "yes": {"sources": [{"node": "x", "output": "result"}]},
                        "no": {"sources": [{"node": "y", "output": "result"}]},
                    },
                },
            },
            "outputs": {"result": {"node": "b", "output": "result"}},
        }
    )
)
COLLECT = subgraph("collect", title="Collect", description="For a test.")(
    GraphRecord.model_validate(
        {
            "inputs": {"text": {}},
            "nodes": {
                "shouted": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
                "all": {
                    "node": "join",
                    "inputs": {"texts": {"sources": [{"node": "shouted", "output": "result"}]}},
                },
            },
            "outputs": {"result": {"node": "all", "output": "result"}},
        }
    )
)
SUBGRAPHS = registry_of(upper, tail, split, join, both, ENRICH, PAIR, COLLECT)


def test_make_graph_chains_a_subgraph_by_the_inputs_and_outputs_its_record_declares():
    """A subgraph is chained as a function node is: its one required graph input, and its one graph output.

    When it can't be, ``not_a_chain`` lists the subgraph's own inputs and outputs, never "none".
    """
    record = make_graph(split, ENRICH, join)
    assert record.nodes["enrich"].inputs == {
        "text": input_entry({"sources": (Source("split", "result"),)})
    }
    assert Graph.compile(record, SUBGRAPHS).run({"text": "a,b"}).outputs == {"result": "A.+B."}
    with pytest.raises(InvalidCallError) as refused:
        make_graph(upper, PAIR, outputs={"upper": "upper.result"})
    assert [(p.type, p.ctx["inputs"], p.ctx["outputs"]) for p in refused.value.problems] == [
        ("not_a_chain", ["x_text", "y_text"], ["result"])
    ]


def test_a_subgraph_node_is_given_wired_and_read_by_its_own_id_and_the_names_its_record_declares():
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "e": {"node": "enrich", "inputs": {"text": {"graph_input": "text"}}},
                    "u": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "e", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "u", "output": "result"}},
            }
        ),
        SUBGRAPHS,
    )

    assert graph.problems == [] and list(graph.inputs) == ["text"]
    run = graph.run({"text": "hi"})
    assert (
        run.outputs == {"result": "HI."}
        and set(run.results) == {"e", "u"}
        and run.results["e"] == {"result": "HI."}
    )
    assert set(graph.describe().nodes) == {"e", "u"}
    assert [f.port.name for f in graph.describe().nodes["e"].inputs] == ["text"]


def test_make_graph_makes_a_graph_input_only_of_each_graph_input_a_subgraphs_record_gives_no_default():
    """``make_graph`` makes an outer graph input only for a subgraph's graph input that has no default.

    ``b`` has a default. ``c`` and ``d`` are each named by two entries, with a default on one of them, so each has a
    default too.
    """
    halves = subgraph("halves", title="Halves", description="For a test.")(
        GraphRecord.model_validate(
            {
                "inputs": {
                    "a": {},
                    "b": {"default": "b"},
                    "c": {"default": "c"},
                    "d": {"default": "d"},
                },
                "nodes": {
                    "x": {"node": "upper", "inputs": {"text": {"graph_input": "a"}}},
                    "y": {
                        "node": "upper",
                        "inputs": {"text": {"graph_input": "b"}},
                    },
                    "z": {"node": "tail", "inputs": {"text": {"graph_input": "c"}}},
                    "w": {"node": "tail", "inputs": {"text": {"graph_input": "c"}}},
                    "u": {"node": "tail", "inputs": {"text": {"graph_input": "d"}}},
                    "v": {"node": "tail", "inputs": {"text": {"graph_input": "d"}}},
                },
                "outputs": {
                    "x": {"node": "x", "output": "result"},
                    "y": {"node": "y", "output": "result"},
                },
            }
        )
    )

    record = make_graph(halves)

    assert record.nodes["halves"].inputs == {"a": input_entry({"graph_input": "a"})}
    assert record.outputs == {"x": Source("halves", "x"), "y": Source("halves", "y")}
    assert Graph.compile(record, registry_of(upper, tail, halves)).run({"a": "a"}).outputs == {
        "x": "A",
        "y": "B",
    }


def test_a_subgraph_nodes_inputs_are_the_graph_inputs_its_record_declares():
    first = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"x_text": {}, "y_text": {}},
                "nodes": {
                    "c": {
                        "node": "pair",
                        "inputs": {
                            "x_text": {"graph_input": "x_text"},
                            "y_text": {"graph_input": "y_text"},
                        },
                    }
                },
                "outputs": {"result": {"node": "c", "output": "result"}},
            }
        ),
        SUBGRAPHS,
    )
    later = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "s": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
                    "c": {
                        "node": "pair",
                        "inputs": {"x_text": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                },
            }
        ),
        SUBGRAPHS,
    )

    assert first.problems == [] and list(first.inputs) == ["x_text", "y_text"]
    assert [(p.type, p.loc) for p in later.problems] == [
        ("input_has_no_value", ("c", "y_text"))
    ]  # a subgraph node gets its inputs as any other node does


def test_a_subgraph_lists_its_inputs_and_outputs_in_the_registrys_describe():
    described = next(n for n in SUBGRAPHS.describe().nodes if n.id == "pair")

    assert [f.name for f in described.inputs] == ["x_text", "y_text"] and [
        f.name for f in described.outputs
    ] == ["result"]


def test_collecting_a_series_from_a_skipped_branch_is_skipped_not_called_with_an_empty_list():
    calls: list[list[str]] = []

    @node("gate", title="Gate", description="For a test.")
    def gate(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("y") else YesNo(SKIPPED, text)

    @node("count-all", title="Count all", description="For a test.")
    def count_all(texts: Series[str]) -> str:
        calls.append(list(texts))
        return str(len(texts))

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "g": {"node": "gate", "inputs": {"text": {"graph_input": "text"}}},
                    "s": {
                        "node": "split",
                        "inputs": {"text": {"sources": [{"node": "g", "output": "no"}]}},
                    },
                    "c": {
                        "node": "count-all",
                        "inputs": {"texts": {"sources": [{"node": "s", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "c", "output": "result"}},
            }
        ),
        registry_of(gate, split, count_all),
    )

    assert graph.run({"text": "yes"}).results["c"]["result"] is None and calls == []
    assert graph.run({"text": "a,b"}).outputs == {"result": "2"}


def test_a_skipped_item_of_a_series_output_is_a_gap_that_reads_back_from_json():
    @node("some", title="Some", description="For a test.")
    def some(text: str) -> Series[str]:
        return ["a", SKIPPED, "c"]

    graph = Graph.compile(make_graph(some, upper), registry_of(some, upper))
    run = graph.run({"text": "x"})

    assert run.results["some"]["result"] == ["a", None, "c"] and run.results["upper"]["result"] == [
        "A",
        None,
        "C",
    ]
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
    narrowed = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {
                    "c": {"node": "contacts"},
                    "n": {
                        "node": "columns",
                        "inputs": {"table": {"sources": [{"node": "c", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "n", "output": "result"}},
            }
        ),
        registry,
    )

    assert Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"table": {}},
                "nodes": {"n": {"node": "columns", "inputs": {"table": {"graph_input": "table"}}}},
                "outputs": {"result": {"node": "n", "output": "result"}},
            }
        ),
        registry,
    ).run({"table": pl.DataFrame({"name": ["a"], "email": ["a@x"]})}).outputs == {"result": "name"}
    assert narrowed.run().outputs == {"result": "name"}


def test_a_node_collecting_two_series_of_different_depths_is_called_per_item_of_the_outer_one():
    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Series[str]:
        return text.split("|")

    @node("pair-lists", title="Pair lists", description="For a test.")
    def pair_lists(a: Series[str], b: Series[str]) -> str:
        return f"{len(a)}/{len(b)}"

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "p": {"node": "parts", "inputs": {"text": {"graph_input": "text"}}},
                    "s": {
                        "node": "split",
                        "inputs": {"text": {"sources": [{"node": "p", "output": "result"}]}},
                    },
                    "x": {
                        "node": "pair-lists",
                        "inputs": {
                            "a": {"sources": [{"node": "p", "output": "result"}]},
                            "b": {"sources": [{"node": "s", "output": "result"}]},
                        },
                    },
                },
                "outputs": {"result": {"node": "x", "output": "result"}},
            }
        ),
        registry_of(parts, split, pair_lists),
    )

    assert (
        graph.describe().nodes["x"].per_item_of == "p.result"
    )  # collecting the inner series ends the inner loop, not the outer one
    assert graph.run({"text": "a,b|c"}).outputs == {"result": ["2/2", "2/1"]}


def test_collecting_items_that_a_skipped_branch_skipped_all_of_is_skipped():
    """When a skipped branch skipped every item being collected, the collecting call is skipped too.

    A branch is taken per document, and every line of the document reads it. Where the branch is skipped, each
    line is skipped, so that document's join is skipped as well.
    """

    @node("pair-two", title="Pair two", description="For a test.")
    def pair_two(a: str, b: str) -> str:
        return f"{a}:{b}"

    @node("decide-doc", title="Decide doc", description="For a test.")
    def decide_doc(text: str) -> YesNo:
        return YesNo(text, SKIPPED) if text.startswith("x") else YesNo(SKIPPED, text)

    @node("docs", title="Docs", description="For a test.")
    def docs(text: str) -> Series[str]:
        return text.split("|")

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "d": {"node": "docs", "inputs": {"text": {"graph_input": "text"}}},
                    "g": {
                        "node": "decide-doc",
                        "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}},
                    },
                    "l": {
                        "node": "split",
                        "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}},
                    },
                    "p": {
                        "node": "pair-two",
                        "inputs": {
                            "a": {"sources": [{"node": "l", "output": "result"}]},
                            "b": {"sources": [{"node": "g", "output": "yes"}]},
                        },
                    },
                    "j": {
                        "node": "join",
                        "inputs": {"texts": {"sources": [{"node": "p", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "j", "output": "result"}},
            }
        ),
        registry_of(docs, decide_doc, split, pair_two, join),
    )

    assert graph.run({"text": "x|y,no"}).results["j"]["result"] == ["x:x", None]


def test_collecting_downstream_of_a_series_output_skipped_is_skipped():
    @dataclass(frozen=True)
    class FullOrEmpty:
        full: Series[str] = SKIPPED
        empty: Series[str] = SKIPPED

    @node("decide-list", title="Decide list", description="For a test.")
    def decide_list(text: str) -> FullOrEmpty:
        return FullOrEmpty(text.split(","), SKIPPED) if text else FullOrEmpty(SKIPPED, [])

    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "inputs": {"text": {}},
                "nodes": {
                    "d": {"node": "decide-list", "inputs": {"text": {"graph_input": "text"}}},
                    "up": {
                        "node": "upper",
                        "inputs": {"text": {"sources": [{"node": "d", "output": "empty"}]}},
                    },
                    "after": {
                        "node": "join",
                        "inputs": {"texts": {"sources": [{"node": "up", "output": "result"}]}},
                    },
                },
                "outputs": {"result": {"node": "after", "output": "result"}},
            }
        ),
        registry_of(decide_list, upper, join),
    )

    assert graph.run({"text": "a,b"}).results["after"]["result"] is None


def test_a_records_field_whose_name_has_a_dot_is_wired_by_its_source():
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

    record = make_graph(
        sheet,
        each,
        double,
        edges=[("sheet.result", "each.table"), ("each.result.Amount in USD.", "double.amount")],
    )
    assert Graph.compile(record, registry_of(sheet, each, double)).run().outputs == {
        "result": [6, 8]
    }


def test_a_node_display_is_the_hosts_own_kept_and_outside_the_fingerprint():
    plain = GraphRecord.model_validate(
        {
            "inputs": {"name": {}},
            "nodes": {"h": {"node": "hello", "inputs": {"name": {"graph_input": "name"}}}},
            "outputs": {"result": {"node": "h", "output": "result"}},
        }
    )
    shown = GraphRecord.model_validate(
        {
            "inputs": {"name": {}},
            "nodes": {
                "h": {
                    "node": "hello",
                    "display": {"x": 3},
                    "inputs": {"name": {"graph_input": "name"}},
                }
            },
            "outputs": {"result": {"node": "h", "output": "result"}},
        }
    )
    assert GraphRecord.model_validate_json(shown.model_dump_json()) == shown
    assert Graph.compile(plain, PLAIN).fingerprint == Graph.compile(shown, PLAIN).fingerprint


@pytest.mark.parametrize("name", [name for name in NodeRecord.model_fields if name != "display"])
def test_every_field_of_a_node_but_its_display_is_in_the_fingerprint(name: str):
    node_record = NodeRecord(node="hello", inputs={"name": input_entry({"constant": "Ida"})})
    value = getattr(node_record, name)
    if isinstance(value, int):
        changed = value + 1
    elif isinstance(value, str):
        changed = f"{value}x"
    else:
        changed = {"name": input_entry({"constant": "Bo"})}
    other = node_record.model_copy(update={name: changed})
    assert (
        Graph.compile(GraphRecord(nodes={"h": node_record}), PLAIN).fingerprint
        != Graph.compile(GraphRecord(nodes={"h": other}), PLAIN).fingerprint
    )


def test_a_graphs_fingerprint_is_the_one_its_saved_runs_name():
    record = GraphRecord.model_validate(
        {
            "nodes": {
                "h": {
                    "node": "hello",
                    "inputs": {"name": {"constant": "Ida"}},
                    "display": {"x": 3},
                },
                "s": {
                    "node": "shout",
                    "inputs": {"text": {"sources": [{"node": "h", "output": "result"}]}},
                },
            },
            "outputs": {"result": {"node": "s", "output": "result"}},
        }
    )
    assert Graph.compile(record, PLAIN).fingerprint == "60e731ed46c14905"


def test_a_constant_given_from_python_is_written_as_json_so_the_record_equals_itself_read_back():
    import datetime
    from enum import Enum

    @host_type("color", title="Color")
    class Color(Enum):
        RED = "red"

    @node("typed", title="Typed", description="For a test.")
    def typed(when: datetime.date, color: Color, pair: tuple[int, int]) -> str:
        return f"{when} {color.value} {pair[0]}"

    record = make_graph(
        typed,
        constants={
            "typed.when": datetime.date(2026, 1, 2),
            "typed.color": Color.RED,
            "typed.pair": (1, 2),
        },
    )
    assert record.nodes["typed"].inputs == {
        "when": input_entry({"constant": "2026-01-02"}),
        "color": input_entry({"constant": "red"}),
        "pair": input_entry({"constant": [1, 2]}),
    }
    read_back = GraphRecord.model_validate_json(record.model_dump_json())
    assert read_back == record
    compiled, from_json = (
        Graph.compile(record, registry_of(typed)),
        Graph.compile(read_back, registry_of(typed)),
    )
    assert compiled.fingerprint == from_json.fingerprint
    assert compiled.run().outputs == from_json.run().outputs == {"result": "2026-01-02 red 1"}


def test_a_table_as_a_constant_is_refused_at_the_call_and_the_reason_says_to_write_it_as_records():
    import polars as pl

    @node("count", title="Count", description="For a test.")
    def count(rows: Table) -> int:
        return rows.height

    with pytest.raises(InvalidCallError) as refused:
        make_graph(count, constants={"count.rows": pl.DataFrame({"a": [1]})})
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc) == ("invalid_value", ("count", "rows"))
    assert problem.ctx["reason"] == (
        "a frame is read as it is only where a run is given it for a table, or a union with one; anywhere else, such"
        " as a constant or a choice, give its rows as a list with one object per row"
    )
    assert [error["type"] for error in problem.ctx["errors"]] == ["no_json_form"]


def test_a_node_record_accepts_only_json_in_its_constants_defaults_and_display():
    with pytest.raises(ValidationError):
        NodeRecord(node="a", display={"x": object()})
    with pytest.raises(ValidationError):
        input_entry({"constant": object()})
    with pytest.raises(ValidationError):
        GraphInputRecord(default=object())


def test_a_float_that_is_not_finite_as_a_constant_is_refused_where_it_is_written_and_its_text_where_it_is_read():
    @node("half", title="Half", description="For a test.")
    def half(n: float) -> float:
        return n / 2

    with pytest.raises(InvalidCallError) as refused:
        make_graph(half, constants={"half.n": float("nan")})
    (written,) = refused.value.problems
    (read,) = Graph.compile(
        {
            "nodes": {"half": {"node": "half", "inputs": {"n": {"constant": "Infinity"}}}},
            "outputs": {"result": {"node": "half", "output": "result"}},
        },
        registry_of(half),
    ).problems

    assert (written.type, written.ctx["errors"][0]["type"]) == ("invalid_value", "finite_number")
    assert (read.type, read.ctx["errors"][0]["type"]) == ("invalid_value", "float_type")
    with pytest.raises(ValidationError):
        NodeRecord(node="half", display={"z": float("inf")})
    with pytest.raises(ValidationError):
        input_entry({"constant": float("nan")})


def test_make_graph_raises_every_problem_of_its_own_and_none_that_compile_finds():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(
            _count,
            _hello,
            edges=[("count.result", "hello.name"), ("count", "hello.name")],
            constants={"nobody.x": 1},
        )
    assert sorted(p.type for p in refused.value.problems) == ["invalid_place", "node_not_in_graph"]


def test_make_graph_writes_each_node_at_the_id_and_version_of_the_definition_it_was_given():
    @node("v", title="V1", description="For a test.")
    def v1(text: str) -> str:
        return text

    @node("v", title="V2", description="For a test.", replaces=v1)
    def v2(text: str) -> str:
        return text

    record = make_graph(
        v2,
        v1,
        edges=[],
        constants={"v-1.nope": 1},
        inputs={"a": "v-1.text", "b": "v-2.text"},
        outputs={"a": "v-1.result", "b": "v-2.result"},
    )
    assert [(node_id, n.node, n.version) for node_id, n in record.nodes.items()] == [
        ("v-1", "v", 2),
        ("v-2", "v", 1),
    ]
    assert [(p.type, p.loc) for p in Graph.compile(record, registry_of(v2)).problems] == [
        ("unknown_input", ("v-1", "nope"))
    ]

    @node("v", title="Other", description="For a test.")
    def other(n: int) -> int:
        return n

    unrelated = make_graph(v1, other, edges=[], outputs={"a": "v-1.result", "b": "v-2.result"})
    assert [(node_id, n.node, n.version) for node_id, n in unrelated.nodes.items()] == [
        ("v-1", "v", 1),
        ("v-2", "v", 1),
    ]


@pytest.mark.parametrize(
    ("arguments", "errors"),
    [
        ({"outputs": None}, [("outputs", "dict_type", [])]),
        ({"inputs": None}, [("inputs", "dict_type", [])]),
        ({"constants": None}, [("constants", "dict_type", [])]),
        ({"defaults": [("hello.name", "Ida")]}, [("defaults", "dict_type", [])]),
        ({"inputs": "name"}, [("inputs", "dict_type", [])]),
        ({"outputs": {"greeting": 1}}, [("outputs", "string_type", ["greeting"])]),
        ({"inputs": {"who": [1]}}, [("inputs", "string_type", ["who", 0])]),
        ({"inputs": {"who": {"hello.name"}}}, [("inputs", "is_instance_of", ["who"])]),
        ({"inputs": {"who": b"hello.name"}}, [("inputs", "sequence_str", ["who"])]),
        # a key that isn't text is part of what was given, so the loc stops before it
        ({"constants": {1: "Ida"}}, [("constants", "string_type", ["[key]"])]),
        ({"edges": "hello.result"}, [("edges", "sequence_str", [])]),
        ({"edges": {("count.result", "hello.name")}}, [("edges", "is_instance_of", [])]),
        # a mapping given for edges raised a raw IndexError
        ({"edges": {"a": "b"}}, [("edges", "is_instance_of", [])]),
        ({"edges": [b"hello.result"]}, [("edges", "sequence_str", [0])]),
        ({"edges": [("hello.result",)]}, [("edges", "too_short", [0])]),
        ({"edges": [(1, "hello.name")]}, [("edges", "string_type", [0, 0])]),
        (
            {"edges": [(b"count.result", "hello.name")], "outputs": None},
            [("edges", "string_type", [0, 0]), ("outputs", "dict_type", [])],
        ),
    ],
)
def test_make_graph_refuses_an_argument_of_the_wrong_type_with_invalid_argument_and_says_where_in_it(
    arguments: dict[str, Any], errors: list[tuple[str, str, list[str | int]]]
):
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, **arguments)

    found = [
        (p.loc, p.ctx["argument"], error["type"], error["loc"])
        for p in refused.value.problems
        for error in p.ctx["errors"]
    ]
    assert found == [((argument,), argument, kind, loc) for argument, kind, loc in errors]
    assert {p.type for p in refused.value.problems} == {"invalid_argument"}


@pytest.mark.parametrize(
    ("nodes", "index"),
    [
        ((None,), 0),
        (("hello",), 0),
        ((_hello, ("first",)), 1),
        (((1, _hello),), 0),
        ((["a", _hello],), 0),
    ],
)
def test_make_graph_refuses_a_node_that_is_neither_a_definition_nor_a_tuple_of_an_id_and_a_definition(
    nodes: tuple[Any, ...], index: int
):
    with pytest.raises(InvalidCallError) as refused:
        make_graph(*nodes)

    [problem] = refused.value.problems
    [error] = problem.ctx["errors"]
    assert (problem.type, problem.loc, error["type"], error["loc"]) == (
        "invalid_argument",
        ("nodes",),
        "not_a_node",
        [index],
    )
    assert "_Version" not in str(refused.value)


def test_make_graph_takes_every_form_of_its_arguments_a_type_checker_takes():
    class Place(str):
        pass

    record = make_graph(
        ("a", _hello),
        ("b", _shout),
        edges=[["a.result", "b.text"]],
        constants=MappingProxyType({Place("b.times"): 2}),
        inputs={"who": "a.name"},
        outputs=MappingProxyType({"loud": "b.result"}),
    )

    assert record.nodes["a"].inputs["name"].graph_input == "who"
    assert record.nodes["b"].inputs["times"].constant == 2
    assert record.nodes["b"].inputs["text"].sources == (Source("a", "result"),)
    assert record.outputs == {"loud": Source("b", "result")}
    assert make_graph(_hello, inputs={"who": ("hello.name",)}) == make_graph(
        _hello, inputs={"who": "hello.name"}
    )


class _Secret:
    """A key whose text must never reach a problem."""

    def __repr__(self) -> str:
        return "SECRET-REPR"

    __str__ = __repr__


@pytest.mark.parametrize("argument", ["inputs", "outputs"])
@pytest.mark.parametrize("key", [SKIPPED, _Secret(), 1], ids=["sentinel", "secret", "int"])
@pytest.mark.parametrize("value", [SKIPPED, 5], ids=["sentinel-value", "int-value"])
def test_make_graph_quotes_no_key_that_is_not_text_even_when_the_value_under_it_is_refused(
    argument: str, key: object, value: object
):
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, **{argument: {key: value}})

    [problem] = refused.value.problems
    assert [(e["type"], e["loc"]) for e in problem.ctx["errors"]] == [("string_type", ["[key]"])]
    written = str(refused.value) + repr(refused.value.errors())
    assert "SKIPPED" not in written and "SECRET-REPR" not in written


def test_make_graph_takes_any_sequence_that_is_not_text_bytes_or_a_set():
    edge = ("hello.result", "shout.text")
    expected = make_graph(_hello, _shout, edges=[edge], inputs={"t": ["shout.times"]})

    assert (
        make_graph(_hello, _shout, edges=deque([edge]), inputs={"t": ["shout.times"]}) == expected
    )
    assert (
        make_graph(_hello, _shout, edges=UserList([edge]), inputs={"t": ["shout.times"]})
        == expected
    )
    assert (
        make_graph(_hello, _shout, edges=[deque(edge)], inputs={"t": ["shout.times"]}) == expected
    )  # type: ignore[list-item]
    assert (
        make_graph(_hello, _shout, edges=[edge], inputs={"t": deque(["shout.times"])}) == expected
    )


def test_make_graph_quotes_no_argument_it_refuses():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(SKIPPED, edges=[(SKIPPED, "a.b")], constants={SKIPPED: 1}, outputs=SKIPPED)  # type: ignore[arg-type, list-item, dict-item]

    assert [p.loc for p in refused.value.problems] == [
        ("nodes",),
        ("edges",),
        ("constants",),
        ("outputs",),
    ]
    assert "SKIPPED" not in str(refused.value) + repr(refused.value.errors())


def test_make_graph_refuses_a_constant_or_default_key_that_is_not_node_and_name_as_it_refuses_such_an_edge():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(
            _hello, constants={"hello": "Ida", "hello.name..x": "Ida"}, defaults={".name": "Bo"}
        )

    assert [(p.type, p.loc) for p in refused.value.problems] == [
        ("invalid_place", ("constants", "hello")),
        # an input's name has no dot
        ("invalid_place", ("constants", "hello.name..x")),
        ("invalid_place", ("defaults", ".name")),
    ]
    assert all("'node.name'" in p.msg for p in refused.value.problems)


_THE_NAME_RULE = "where neither part is empty or contains any of '.', '/', '@', '~'."


@pytest.mark.parametrize(
    ("arguments", "loc", "text"),
    [
        ({"edges": [("hello.result~", "shout.text")]}, ("edges", 0, 0), "hello.result~"),
        ({"edges": [("hello.result..", "shout.text")]}, ("edges", 0, 0), "hello.result.."),
        ({"edges": [("hel@lo.result", "shout.text")]}, ("edges", 0, 0), "hel@lo.result"),
        ({"edges": [("hello.result", "shout.te/xt")]}, ("edges", 0, 1), "shout.te/xt"),
        (
            {"edges": [("hello.result", "shout.text")], "outputs": {"x": "shout.result~"}},
            ("outputs", "x"),
            "shout.result~",
        ),
        (
            {"edges": [("hello.result", "shout.text")], "inputs": {"who": "hello.na~me"}},
            ("inputs", "who"),
            "hello.na~me",
        ),
        # a text that is not node and name at all, at the argument and place it is given
        (
            {"edges": [("hello.result", "shout.text"), ("hello", "shout.text")]},
            ("edges", 1, 0),
            "hello",
        ),
        (
            {"edges": [("hello.result", "shout")], "outputs": {"hello": "hello.result"}},
            ("edges", 0, 1),
            "shout",
        ),
        ({"constants": {"name": "x"}}, ("constants", "name"), "name"),
        ({"outputs": {"r": "shout"}}, ("outputs", "r"), "shout"),
    ],
    ids=[
        "a-tilde-in-an-output",
        "a-field-starting-with-a-dot",
        "an-at-in-a-node",
        "a-slash-in-an-input",
        "a-tilde-in-a-graph-output",
        "a-tilde-in-a-graph-input",
        "an-edge-source-of-one-part",
        "an-edge-target-of-one-part",
        "a-constant-of-one-part",
        "a-graph-output-of-one-part",
    ],
)
def test_make_graph_refuses_a_place_that_is_not_node_and_name_as_invalid_place(
    arguments: dict[str, Any], loc: tuple[str | int, ...], text: str
):
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, _shout, **arguments)

    [problem] = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx) == ("invalid_place", loc, {"text": text})
    # the message says why the text is no place, though it is written as node.name
    assert (
        problem.msg
        == f"{text!r} isn't a place in the graph. Write it as 'node.name', {_THE_NAME_RULE}"
    )


def test_make_graph_takes_a_node_with_an_id_of_its_own_and_refuses_one_id_for_two_nodes():
    record = make_graph(
        ("first", _hello), ("second", _hello), edges=[("first.result", "second.name")]
    )
    assert list(record.nodes) == ["first", "second"] and record.nodes["second"].node == "hello"
    with pytest.raises(InvalidCallError) as refused:
        make_graph(("a", _hello), ("a", _hello), edges=[])
    assert [(p.type, p.loc, p.ctx["errors"][0]["type"]) for p in refused.value.problems] == [
        ("invalid_record", ("record", "nodes", "a"), "repeated_key")
    ]


def test_a_record_validator_refusal_of_what_make_graph_wrote_says_to_fix_make_graphs_arguments():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(
            _hello,
            _shout,
            edges=[("hello.result", "shout.text"), ("hello.result", "shout.text")],
            constants={"hello.name": "a"},
        )

    [problem] = refused.value.problems
    assert (
        problem.type,
        problem.loc,
        problem.ctx["errors"][0]["type"],
        problem.ctx["written_by"],
    ) == (
        "invalid_record",
        ("record", "nodes", "shout", "inputs", "text", "sources"),
        "repeated_source",
        "make_graph's arguments",
    )
    assert problem.msg.endswith("Fix make_graph's arguments.")


def test_make_graph_inputs_naming_one_input_leaves_the_other_automatic_graph_inputs_declared():
    record = make_graph(
        ("a", upper),
        ("b", upper),
        edges=[],
        inputs={"x": "a.text"},
        outputs={"a_result": "a.result", "b_result": "b.result"},
    )
    graph = Graph.compile(record, REGISTRY)

    assert {node_id: n.inputs for node_id, n in record.nodes.items()} == {
        "a": {"text": input_entry({"graph_input": "x"})},
        "b": {"text": input_entry({"graph_input": "text"})},
    }
    assert list(graph.inputs) == ["x", "text"] and graph.run({"x": "p", "text": "q"}).outputs == {
        "a_result": "P",
        "b_result": "Q",
    }


def test_make_graph_refuses_a_defaults_key_whose_name_is_also_given_in_inputs():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(
            _hello,
            _shout,
            edges=[],
            defaults={"shout.times": 2},
            inputs={"times": "hello.name"},
            outputs={"hello": "hello.result"},
        )

    assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [
        (
            "graph_inputs_share_a_name",
            ("inputs", "times"),
            {"name": "times", "inputs": ["shout.times", "hello.name"]},
        ),
    ]


def test_make_graph_refuses_an_output_with_a_field_as_the_record_does():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(_hello, outputs={"r": "hello.result.x"})

    assert [
        (p.type, p.loc, p.ctx["errors"][0]["type"], p.ctx["written_by"])
        for p in refused.value.problems
    ] == [
        (
            "invalid_record",
            ("record", "outputs", "r"),
            "output_has_field",
            "make_graph's arguments",
        ),
    ]


def test_make_graph_leaves_an_optional_input_with_no_entry_out_and_it_is_no_graph_input():
    record = make_graph(_shout)

    assert record.nodes["shout"].inputs == {"text": input_entry({"graph_input": "text"})}
    assert list(Graph.compile(record, PLAIN).inputs) == ["text"]


def test_make_graph_names_an_optional_input_in_inputs_and_the_graph_input_keeps_the_inputs_default():
    """Naming an optional input in ``inputs=`` renames it, and a caller may still leave it out.

    A graph input that feeds two inputs takes their default only when both declare the same one, since one graph
    input has one default. One that ``defaults=`` gives a default keeps that.
    """

    @node("repeat", title="Repeat", description="For a test.")
    def repeat(text: str, times: int = 2) -> str:
        return text * times

    record = make_graph(_shout, inputs={"n": "shout.times"})
    graph = Graph.compile(record, PLAIN)

    assert record.nodes["shout"].inputs["times"] == input_entry({"graph_input": "n"})
    assert record.inputs["n"] == GraphInputRecord(default=1)
    assert graph.run({"text": "a"}).outputs == {"result": "A!"}
    assert graph.run({"text": "a", "n": 3}).outputs == {"result": "A!!!"}
    given = make_graph(_shout, inputs={"n": "shout.times"}, defaults={"shout.times": 2})
    assert given.inputs["n"] == GraphInputRecord(default=2)

    def n_of(*nodes: Any) -> GraphInputRecord:
        both = make_graph(
            *nodes,
            edges=[],
            inputs={"n": ["a.times", "b.times"], "text": ["a.text", "b.text"]},
            outputs={"a": "a.result", "b": "b.result"},
        )
        assert [both.nodes[node_id].inputs["times"] for node_id in ("a", "b")] == [
            input_entry({"graph_input": "n"})
        ] * 2
        return both.inputs["n"]

    assert n_of(("a", _shout), ("b", _shout)) == GraphInputRecord(default=1)
    assert n_of(("a", _shout), ("b", repeat)) == GraphInputRecord()
    loud = subgraph("loud", title="Loud", description="For a test.")(record)
    assert make_graph(loud, inputs={"m": "loud.n"}).inputs["m"] == GraphInputRecord(default=1)


def test_make_graph_refuses_two_different_defaults_for_one_graph_input():
    with pytest.raises(InvalidCallError) as refused:
        make_graph(
            ("a", _shout),
            ("b", _shout),
            edges=[],
            inputs={"n": ["a.times", "b.times"], "text": ["a.text", "b.text"]},
            defaults={"a.times": 2, "b.times": 3},
            outputs={"a": "a.result", "b": "b.result"},
        )

    assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [
        (
            "graph_input_given_two_defaults",
            ("inputs", "n"),
            {"name": "n", "inputs": ["a.times", "b.times"]},
        )
    ]
    same = make_graph(
        ("a", _shout),
        ("b", _shout),
        edges=[],
        inputs={"n": ["a.times", "b.times"], "text": ["a.text", "b.text"]},
        defaults={"a.times": 2, "b.times": 2},
        outputs={"a": "a.result", "b": "b.result"},
    )
    assert same.inputs["n"] == GraphInputRecord(default=2)


def test_make_graph_makes_a_required_input_on_a_node_an_edge_leads_into_a_graph_input():
    record = make_graph(_hello, both, edges=[("hello.result", "both.yes")])
    graph = Graph.compile(record, registry_of(_hello, both))

    assert record.nodes["both"].inputs == {
        "yes": input_entry({"sources": (Source("hello", "result"),)}),
        "no": input_entry({"graph_input": "no"}),
    }
    assert list(graph.inputs) == ["name", "no"] and graph.run(
        {"name": "ida", "no": "!"}
    ).outputs == {"result": "hello ida!"}


_WRITTEN_AND_COMPILED_IN_ANOTHER_PROCESS = """
from typing import Literal

from conductor import Graph, InvalidCallError, NodeRegistry, make_graph, node

@node("number", title="Number", description="For a test.")
def number() -> int:
    return 3

@node("text", title="Text", description="For a test.")
def text() -> str:
    return "a"

@node("either", title="Either", description="For a test.")
def either[T](a: T, b: T, c: str, d: str, e: str = "-") -> T:
    return a

@node("other_way", title="Other way", description="For a test.")
def other_way() -> str | int:
    return 1

registry = NodeRegistry()
registry.add(number, text, either, other_way)
# a hint equal to int | str but written the other way, alive while the graph below compiles
Graph.compile(make_graph(other_way), registry)
record = make_graph(
    number,
    text,
    either,
    edges=[("number.result", "either.a"), ("text.result", "either.b")],
    constants={"either.d": "x"},
    defaults={"either.e": "y"},
    inputs={"c": "either.c"},
)
print(record.model_dump_json())
print(Graph.compile(record, registry).describe().nodes["either"].outputs[0].type)

@node("one", title="One", description="For a test.")
def one(value: Literal[1, True]) -> str:
    return "a"

@node("yes", title="Yes", description="For a test.")
def yes(value: Literal[True, 1]) -> str:
    return "a"

print([[choice.id for choice in d.describe().inputs[0].choices] for d in (one, yes)])
try:
    make_graph(("x", text), ("y", either), ("z", either), edges=[], defaults={"y.e": "1", "z.e": "2"})
except InvalidCallError as refused:
    print([problem.ctx for problem in refused.problems if problem.type == "graph_inputs_share_a_name"])
"""


def test_make_graph_writes_and_compiles_a_graph_the_same_in_every_process():
    """A record lists each node's inputs in the order the node declares them, whatever order its places are kept in.

    A type variable on two inputs binds the union in the order its inputs are declared, even while an equal union
    written the other way, like ``str | int`` for ``int | str``, is in use. A problem lists its places in the
    order they were given.

    ``Literal[1, True]`` and ``Literal[True, 1]`` are equal, since ``1 == True``, and each keeps its own choices.
    """
    finished = [
        subprocess.run(
            [sys.executable, "-c", _WRITTEN_AND_COMPILED_IN_ANOTHER_PROCESS],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        for seed in ("1", "2", "3")
    ]
    assert [process.returncode for process in finished] == [0, 0, 0], finished[0].stderr
    written = {process.stdout for process in finished}
    assert len(written) == 1
    record, union, choices, problem = written.pop().splitlines()
    assert list(json.loads(record)["nodes"]["either"]["inputs"]) == ["a", "b", "c", "d", "e"]
    assert union == "int | str"
    assert choices == "[[1, True], [True, 1]]"
    assert "'inputs': ['y.e', 'z.e']" in problem


def test_a_source_is_an_output_of_a_node_by_its_id_and_a_path_cannot_be_written_in_one():
    from pydantic import ValidationError

    assert str(Source("read", "result", "email")) == "read.result.email"
    with pytest.raises(ValidationError) as refused:
        Source("loud/shout", "result")
    assert [error["type"] for error in refused.value.errors()] == ["not_a_name"]
