"""Tests of the nodes in ``conductor_nodes``, each run inside a graph.

They check what each node returns, and how it fails when it can't do its job.
"""

from typing import Any

import conductor_nodes
import polars as pl
import pytest
from conductor import Graph, InvalidCallError, NodeRegistry, Table, make_graph
from conductor_nodes import control, json_ops, logic, regex_ops, series, text
from pytest import MonkeyPatch


def _run(node_id: str, **values: Any) -> dict[str, Any]:
    """Runs one node of the registry on its own and returns its results.

    The node is put in a graph by itself, and each of ``values`` is given to it as a constant. The run must complete.
    """
    record = make_graph(
        conductor_nodes.registry[node_id],
        constants={f"{node_id}.{name}": value for name, value in values.items()},
    )
    run = Graph.compile(record, conductor_nodes.registry).run()
    assert run.status == "complete", run.cause
    return run.results[node_id]


@pytest.mark.parametrize(
    ("node_id", "values", "expected"),
    [
        ("text-uppercase", {"text": "hello"}, "HELLO"),
        ("text-length", {"text": "hello"}, 5),
        ("text-concat", {"a": "foo", "b": "bar", "separator": "-"}, "foo-bar"),
        ("text-contains", {"text": "Hello", "needle": "hello"}, False),
        ("text-join", {"parts": ["a", "b", "c"], "separator": "-"}, "a-b-c"),
        ("math-add", {"a": 2, "b": 3}, 5),
        ("math-divide", {"a": 10, "b": 4}, 2.5),
        ("math-round", {"value": 3.7}, 4),
        ("math-round", {"value": 2.5}, 3),
        ("math-round", {"value": -2.5}, -3),
        ("math-round", {"value": 0.125, "decimals": 2}, 0.13),
        ("math-min", {"values": [3, 1, 2]}, 1),
        ("logic-not", {"value": True}, False),
        ("logic-equals", {"value": "a", "other": "a"}, True),
        ("logic-equals", {"value": "a", "other": "A"}, False),
        ("logic-is-empty", {"value": None}, True),
        ("logic-is-empty", {"value": ""}, True),
        ("logic-is-empty", {"value": []}, True),
        ("logic-is-empty", {"value": {}}, True),
        ("logic-is-empty", {"value": " "}, False),
        ("logic-is-empty", {"value": 0}, False),
        ("logic-is-empty", {"value": False}, False),
        ("logic-is-empty", {"value": ["x"]}, False),
        ("json-parse", {"text": '{"a": 1, "b": [2, 3]}'}, {"a": 1, "b": [2, 3]}),
        ("json-stringify", {"value": {"b": 2, "a": 1}, "sort_keys": True}, '{"a": 1, "b": 2}'),
        ("json-get", {"value": {"user": {"name": "Ada"}}, "path": "user.name"}, "Ada"),
        ("json-get", {"value": {"a": 1}, "path": "does.not.exist"}, None),
        ("json-get", {"value": {"items": [1, 2, 3]}, "path": "items.-1"}, 3),
        ("json-get", {"value": {"items": [1, 2, 3]}, "path": "items.--1"}, None),
        ("json-get", {"value": {"items": [1, 2, 3]}, "path": "items.\u00b2"}, None),
        ("json-get", {"value": {" a": 1, "a": 2}, "path": " a"}, 1),
        ("json-get", {"value": {"user": {" name": "Ada"}}, "path": "user. name"}, "Ada"),
        ("json-get", {"value": {"items": [1, 2, 3]}, "path": "items. 1"}, None),
        ("regex-match", {"text": "Hello", "pattern": "hello", "ignore_case": True}, True),
        ("regex-replace", {"text": "a1b2c3", "pattern": r"\d", "replacement": "-"}, "a-b-c-"),
        ("regex-extract", {"text": "b ab", "pattern": r"(a)?b"}, ["", "a"]),
    ],
)
def test_a_node_gives_its_result(node_id: str, values: dict[str, Any], expected: Any) -> None:
    assert _run(node_id, **values) == {"result": expected}


def test_a_series_returned_inside_a_loop_starts_a_nested_loop_and_a_series_input_collects_only_its_own_items_series() -> (
    None
):
    graph = Graph.compile(
        make_graph(
            text.split,
            regex_ops.extract,
            text.join,
            edges=[
                ("text-split.result", "regex-extract.text"),
                ("regex-extract.result", "text-join.parts"),
            ],
            constants={"text-split.separator": ",", "regex-extract.pattern": r"\d+"},
        ),
        conductor_nodes.registry,
    )

    assert graph.describe().nodes["regex-extract"].per_item_of == "text-split.result"
    assert graph.describe().nodes["text-join"].per_item_of == "text-split.result"
    assert graph.run({"text": "a1,b2 3,c"}).outputs == {"result": ["1", "2, 3", ""]}


def test_a_decision_sends_its_value_down_one_branch_and_the_other_branch_is_skipped() -> None:
    graph = Graph.compile(
        make_graph(
            control.decision,
            text.uppercase,
            edges=[("control-decide.if_true", "text-uppercase.text")],
        ),
        conductor_nodes.registry,
    )

    assert {
        output.name: output.required for output in graph.describe().nodes["control-decide"].outputs
    } == {"if_true": False, "if_false": False}
    taken = graph.run({"value": "hi", "when": True})
    assert taken.outputs == {"result": "HI"} and taken.results["control-decide"] == {
        "if_true": "hi",
        "if_false": None,
    }
    assert graph.run({"value": "hi", "when": False}).outputs == {"result": None}


def test_a_decision_with_nothing_wired_to_when_is_a_problem_when_compiled() -> None:
    record = {
        "nodes": {
            "decide": {"node": "control-decide", "inputs": {"value": {"graph_input": "value"}}}
        }
    }

    assert [(p.type, p.loc) for p in Graph.compile(record, conductor_nodes.registry).problems] == [
        ("input_has_no_value", ("decide", "when"))
    ]


def test_a_test_wired_into_a_decision_sends_the_value_the_way_the_test_says() -> None:
    graph = Graph.compile(
        make_graph(
            text.length,
            logic.equals,
            control.decision,
            text.uppercase,
            edges=[
                ("text-length.result", "logic-equals.value"),
                ("logic-equals.result", "control-decide.when"),
                ("control-decide.if_true", "text-uppercase.text"),
            ],
            constants={"logic-equals.other": 2},
        ),
        conductor_nodes.registry,
    )

    assert graph.run({"text": "hi", "value": "pass"}).outputs == {"result": "PASS"}
    assert graph.run({"text": "hey", "value": "pass"}).outputs == {"result": None}


def test_equals_reads_the_other_value_by_the_type_wired_to_its_value() -> None:
    graph = Graph.compile(
        make_graph(text.length, logic.equals, edges=[("text-length.result", "logic-equals.value")]),
        conductor_nodes.registry,
    )

    assert [
        (each.field.name, each.field.type) for each in graph.describe().nodes["logic-equals"].inputs
    ] == [
        ("value", "int"),
        ("other", "int"),
    ]


def test_is_empty_is_true_for_nothing_found_and_for_a_table_of_no_rows() -> None:
    registry = NodeRegistry()
    registry.add_subregistry(conductor_nodes.registry)

    @registry.node("no-rows", title="No rows", description="Gives a table of no rows.")
    def no_rows() -> Table:
        return pl.DataFrame({"name": []}, schema={"name": pl.String})

    found = Graph.compile(
        make_graph(json_ops.get_path, logic.is_empty, constants={"json-get.path": "user.name"}),
        registry,
    )
    table = Graph.compile(make_graph(no_rows, logic.is_empty), registry)

    assert found.run({"value": {"user": {}}}).outputs == {"result": True}
    assert found.run({"value": {"user": {"name": "Ida"}}}).outputs == {"result": False}
    assert table.run().outputs == {"result": True}


def test_a_switch_sends_each_item_down_its_cases_branch_and_the_node_after_them_takes_the_one_taken() -> (
    None
):
    graph = Graph.compile(
        make_graph(
            series.split,
            control.switch,
            text.uppercase,
            series.collect,
            edges=[
                ("series-split.result.label", "control-switch.case"),
                ("series-split.result.text", "control-switch.value"),
                ("control-switch.invoice", "text-uppercase.text"),
                ("control-switch.receipt", "text-uppercase.text"),
                ("control-switch.other", "text-uppercase.text"),
                ("text-uppercase.result", "series-collect.values"),
            ],
            constants={"control-switch.cases": ["invoice", "receipt"]},
        ),
        conductor_nodes.registry,
    )
    documents = [
        {"label": "invoice", "text": "a"},
        {"label": "receipt", "text": "b"},
        {"label": "letter", "text": "c"},
    ]

    run = graph.run({"value": documents})

    assert [output.name for output in graph.describe().nodes["control-switch"].outputs] == [
        "invoice",
        "receipt",
        "other",
    ]
    assert run.outputs == {"result": ["A", "B", "C"]}
    assert run.results["control-switch"] == {
        "invoice": ["a", None, None],
        "receipt": [None, "b", None],
        "other": [None, None, "c"],
    }


@pytest.mark.parametrize(
    ("cases", "error"),
    [(["invoice", "other"], "case_named_other"), (["a", "b", "a"], "case_given_twice")],
)
def test_a_switch_whose_cases_name_other_or_one_case_twice_is_a_problem_when_compiled(
    cases: list[str], error: str
) -> None:
    graph = Graph.compile(
        make_graph(
            control.switch,
            constants={"control-switch.cases": cases},
            outputs={"a": "control-switch.a"},
        ),
        conductor_nodes.registry,
    )

    assert [(problem.type, problem.ctx["error"]) for problem in graph.problems] == [
        ("compute_hook_failed", error)
    ]


def test_default_gives_the_fallback_for_nothing_found_and_the_value_otherwise() -> None:
    graph = Graph.compile(
        make_graph(
            json_ops.get_path,
            control.default,
            constants={"json-get.path": "user.name", "control-default.fallback": "nobody"},
        ),
        conductor_nodes.registry,
    )

    assert graph.run({"value": {"user": {}}}).outputs == {"result": "nobody"}
    assert graph.run({"value": {"user": {"name": "Ida"}}}).outputs == {"result": "Ida"}


def test_a_decisions_branches_take_the_type_that_arrives_on_its_value() -> None:
    graph = Graph.compile(
        make_graph(
            text.length, control.decision, edges=[("text-length.result", "control-decide.value")]
        ),
        conductor_nodes.registry,
    )

    outputs = graph.describe().nodes["control-decide"].outputs
    assert [(field.name, field.type) for field in outputs] == [
        ("if_true", "int"),
        ("if_false", "int"),
    ]


@pytest.mark.parametrize(
    ("node_id", "values", "code"),
    [
        ("math-divide", {"a": 1, "b": 0}, "division_by_zero"),
        ("math-max", {"values": []}, "no_numbers"),
        ("json-parse", {"text": "{not json"}, "not_json"),
        ("json-parse", {"text": "1" * 5_000}, "not_json"),
        ("json-parse", {"text": "[NaN, 1]"}, "not_json"),
        ("json-parse", {"text": '{"a": Infinity}'}, "not_json"),
        ("json-parse", {"text": "-Infinity"}, "not_json"),
        ("json-parse", {"text": "1e999"}, "not_json"),
        ("json-parse", {"text": "-1e999"}, "not_json"),
        ("json-parse", {"text": "[1e400]"}, "not_json"),
        ("regex-match", {"text": "a", "pattern": "("}, "invalid_pattern"),
    ],
)
def test_a_failure_says_what_went_wrong_in_a_code_of_its_own(
    node_id: str, values: dict[str, Any], code: str
) -> None:
    record = make_graph(
        conductor_nodes.registry[node_id],
        constants={f"{node_id}.{name}": value for name, value in values.items()},
    )
    run = Graph.compile(record, conductor_nodes.registry).run()

    assert run.status == "failed" and run.cause is not None and run.cause.type == code


def test_a_replacement_that_names_a_group_the_pattern_lacks_is_its_own_error_not_the_patterns() -> (
    None
):
    record = make_graph(
        regex_ops.replace_all,
        constants={
            "regex-replace.text": "a",
            "regex-replace.pattern": "a",
            "regex-replace.replacement": r"\9",
        },
    )

    cause = Graph.compile(record, conductor_nodes.registry).run().cause
    assert cause is not None and cause.type == "invalid_replacement" and "pattern" not in cause.msg


def test_an_empty_separator_is_refused_at_the_field_before_the_node_runs() -> None:
    graph = Graph.compile(
        make_graph(text.split, defaults={"text-split.separator": "\n"}), conductor_nodes.registry
    )

    with pytest.raises(InvalidCallError) as refused:
        graph.run({"text": "a,b", "separator": ""})
    assert [(p.type, p.loc) for p in refused.value.problems] == [
        ("invalid_input", ("inputs", "separator"))
    ]


@pytest.mark.parametrize("definition", [text.replace, text.contains])
def test_an_empty_needle_is_refused_at_the_field_before_the_node_runs(definition: Any) -> None:
    graph = Graph.compile(
        make_graph(definition, constants={f"{definition.id}.text": "ab"}), conductor_nodes.registry
    )

    with pytest.raises(InvalidCallError) as refused:
        graph.run({"needle": ""})
    assert [(p.type, p.loc) for p in refused.value.problems] == [
        ("invalid_input", ("inputs", "needle"))
    ]


def test_a_pattern_that_takes_too_long_fails_its_node_instead_of_freezing_the_run(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(regex_ops, "TIMEOUT", 0.01)
    record = make_graph(
        regex_ops.match,
        constants={"regex-match.text": "a" * 50_000 + "b", "regex-match.pattern": r"(a|aa)+$"},
    )

    run = Graph.compile(record, conductor_nodes.registry).run()
    assert run.cause is not None and run.cause.type == "pattern_timeout"


def test_a_node_is_still_its_function() -> None:
    assert text.reverse("abc") == "cba"
    assert json_ops.get_path([1, 2, 3], "-1") == 3


def test_text_or_parsed_json_can_be_wired_into_a_json_value_and_a_json_null_is_written_as_null() -> (
    None
):
    for source, value in ((text.uppercase, "x"), (json_ops.parse, "null")):
        graph = Graph.compile(
            make_graph(
                source, json_ops.stringify, edges=[(f"{source.id}.result", "json-stringify.value")]
            ),
            conductor_nodes.registry,
        )
        assert graph.problems == []
        assert graph.run({"text": value}).outputs == {"result": '"X"' if value == "x" else "null"}


def test_a_number_that_isnt_finite_is_refused_by_write_json_since_json_has_no_way_to_write_it() -> (
    None
):
    graph = Graph.compile(
        make_graph(
            conductor_nodes.math.multiply,
            json_ops.stringify,
            edges=[("math-multiply.result", "json-stringify.value")],
        ),
        conductor_nodes.registry,
    )

    assert graph.run({"a": 2, "b": 3}).outputs == {"result": "6.0"}
    cause = graph.run({"a": 1e308, "b": 10}).cause
    assert cause is not None and (cause.type, cause.loc) == ("not_json", ("json-stringify",))
