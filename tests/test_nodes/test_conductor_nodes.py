"""The nodes of ``conductor_nodes`` in graphs: what each gives, and how each fails."""

from typing import Any

import conductor_nodes
import pytest
from conductor import make_graph
from conductor_nodes import decision, json_ops, regex_ops, text


def _run(node_id: str, **values: Any) -> dict[str, Any]:
    """One node of the registry in a graph, its inputs typed in, run to the end."""
    record = make_graph(conductor_nodes.registry[node_id], values={f"{node_id}.{name}": value for name, value in values.items()})
    run = conductor_nodes.registry.load(record).run()
    assert run.status == "complete", run.cause
    return run.results[node_id]


@pytest.mark.parametrize(("node_id", "values", "expected"), [
    ("text-uppercase", {"text": "hello"}, "HELLO"),
    ("text-length", {"text": "hello"}, 5),
    ("text-concat", {"a": "foo", "b": "bar", "separator": "-"}, "foo-bar"),
    ("text-contains", {"text": "Hello", "needle": "hello"}, False),
    ("text-join", {"parts": ["a", "b", "c"], "separator": "-"}, "a-b-c"),
    ("math-add", {"a": 2, "b": 3}, 5),
    ("math-divide", {"a": 10, "b": 4}, 2.5),
    ("math-round", {"value": 3.7}, 4),
    ("math-min", {"values": [3, 1, 2]}, 1),
    ("logic-not", {"value": True}, False),
    ("json-parse", {"text": '{"a": 1, "b": [2, 3]}'}, {"a": 1, "b": [2, 3]}),
    ("json-stringify", {"value": {"b": 2, "a": 1}, "sort_keys": True}, '{"a": 1, "b": 2}'),
    ("json-get", {"value": {"user": {"name": "Ada"}}, "path": "user.name"}, "Ada"),
    ("json-get", {"value": {"a": 1}, "path": "does.not.exist"}, None),
    ("regex-match", {"text": "Hello", "pattern": "hello", "ignore_case": True}, True),
    ("regex-replace", {"text": "a1b2c3", "pattern": r"\d", "replacement": "-"}, "a-b-c-"),
    ("regex-extract", {"text": "b ab", "pattern": r"(a)?b"}, ["", "a"]),
])
def test_a_node_gives_its_result(node_id: str, values: dict[str, Any], expected: Any) -> None:
    assert _run(node_id, **values) == {"result": expected}


def test_a_node_that_gives_many_values_makes_a_row_of_each_and_a_series_input_gathers_the_rows_under_its_own() -> None:
    graph = conductor_nodes.registry.load(make_graph(text.split, regex_ops.extract, text.join, edges=[
        ("text-split.result", "regex-extract.text"), ("regex-extract.result", "text-join.parts"),
    ], values={"regex-extract.pattern": r"\d+"}))

    assert graph.plan["regex-extract"].msg == "per row of text-split.result"
    assert graph.plan["text-join"].msg == "per row of text-split.result"
    assert graph.run(text="a1,b2 3,c").outputs == {"text-join.result": ["1", "2, 3", ""]}


def test_a_decision_sends_its_value_one_way_and_the_other_way_does_not_run() -> None:
    graph = conductor_nodes.registry.load(make_graph(decision.decision, text.uppercase, edges=[("decision.if_true", "text-uppercase.text")]))

    assert graph.exclusive_groups == {"decision": {"when": ["if_true", "if_false"]}}
    taken = graph.run(value="hi", when=True)
    assert taken.outputs == {"text-uppercase.result": "HI"} and taken.results["decision"] == {"if_true": "hi", "if_false": None}
    assert graph.run(value="hi", when=False).outputs == {"text-uppercase.result": None}


def test_a_decisions_outputs_take_the_type_that_arrives_on_its_value() -> None:
    graph = conductor_nodes.registry.load(make_graph(text.length, decision.decision, edges=[("text-length.result", "decision.value")]))

    outputs = graph.describe().nodes["decision"].outputs
    assert [(field.name, field.type) for field in outputs] == [("if_true", "int"), ("if_false", "int")]


@pytest.mark.parametrize(("node_id", "values", "code"), [
    ("math-divide", {"a": 1, "b": 0}, "division_by_zero"),
    ("math-max", {"values": []}, "no_numbers"),
    ("json-parse", {"text": "{not json"}, "not_json"),
    ("regex-match", {"text": "a", "pattern": "("}, "invalid_pattern"),
])
def test_a_failure_says_what_went_wrong_in_a_code_of_its_own(node_id: str, values: dict[str, Any], code: str) -> None:
    record = make_graph(conductor_nodes.registry[node_id], values={f"{node_id}.{name}": value for name, value in values.items()})
    run = conductor_nodes.registry.load(record).run()

    assert run.status == "failed" and run.cause is not None and run.cause.type == code


def test_a_pattern_that_takes_too_long_fails_its_node_instead_of_freezing_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regex_ops, "TIMEOUT", 0.01)
    record = make_graph(regex_ops.match, values={"regex-match.text": "a" * 50_000 + "b", "regex-match.pattern": r"(a|aa)+$"})

    run = conductor_nodes.registry.load(record).run()
    assert run.cause is not None and run.cause.type == "pattern_timeout"


def test_a_node_is_still_its_function() -> None:
    assert text.reverse("abc") == "cba"
    assert json_ops.get_path([1, 2, 3], "-1") == 3
