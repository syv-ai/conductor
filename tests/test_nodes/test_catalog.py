"""What every node of ``conductor_nodes`` keeps, checked over the whole registry at once."""

import conductor_nodes
from conductor.testing import check_registry

#: Each node with the inputs of a call, checked by ``check_registry`` through the engine. Every node here only
#: computes, so every one has one.
EXAMPLES = {
    "text-uppercase": [{"text": "hi"}],
    "text-lowercase": [{"text": "HI"}],
    "text-trim": [{"text": " hi "}],
    "text-length": [{"text": "hi"}],
    "text-concat": [{"a": "a", "b": "b", "separator": "-"}],
    "text-replace": [{"text": "a b", "needle": "b", "replacement": "c"}],
    "text-contains": [{"text": "Hi", "needle": "hi", "case_sensitive": False}],
    "text-split": [{"text": "a\nb"}],
    "text-list": [{"items": ["a", "b"]}],
    "text-join": [{"parts": ["a", "b"]}],
    "text-reverse": [{"text": "ab"}],
    "math-add": [{"a": 1, "b": 2}],
    "math-subtract": [{"a": 1, "b": 2}],
    "math-multiply": [{"a": 2, "b": 3}],
    "math-divide": [{"a": 1, "b": 4}],
    "math-modulo": [{"a": 7, "b": 4}],
    "math-round": [{"value": 3.14159, "decimals": 2}],
    "math-min": [{"values": [3, 1, 2]}],
    "math-max": [{"values": [3, 1, 2]}],
    "math-abs": [{"value": -2}],
    "logic-if-empty": [{"text": ""}, {"text": "x"}],
    "logic-if-equals": [{"a": "A", "b": "a", "case_sensitive": False}],
    "logic-not": [{"value": True}],
    "json-parse": [{"text": '{"a": [1, 2]}'}],
    "json-stringify": [{"value": {"b": 1, "a": 2}, "sort_keys": True}],
    "json-get": [{"value": {"items": [{"id": "a"}]}, "path": "items.0.id"}],
    "regex-match": [{"text": "a1", "pattern": r"\d"}],
    "regex-replace": [{"text": "a1b2", "pattern": r"\d", "replacement": "-"}],
    "regex-extract": [{"text": "a1b22", "pattern": r"\d+"}],
    "decision": [{"value": "x", "when": False}],
    "split": [{"value": ["a", "b"]}],
    "stack": [{"tables": [[{"name": "Ann"}], [{"name": "Bo"}]]}],
    "table-collect": [{"records": [{"name": "Ann"}], "constant_columns": ["team"], "team": "red"}],
}


def test_the_whole_registry_passes_check_registry() -> None:
    assert check_registry(conductor_nodes.registry, examples=EXAMPLES) == []


def test_every_node_has_an_example_so_no_body_goes_untried() -> None:
    assert {definition.id for definition in conductor_nodes.registry} == set(EXAMPLES)


def test_every_node_says_what_it_does_and_where_it_is_listed() -> None:
    for definition in conductor_nodes.registry:
        assert definition.description, definition.id
        assert definition.category in {"control", "json", "logic", "math", "regex", "table", "text"}, definition.id


def test_each_module_offers_its_own_nodes_and_the_package_all_of_them() -> None:
    modules = (conductor_nodes.text, conductor_nodes.math, conductor_nodes.logic, conductor_nodes.json_ops, conductor_nodes.regex_ops, conductor_nodes.tables, conductor_nodes.control)
    assert sum(len(module.registry) for module in modules) == len(conductor_nodes.registry)
    for module in modules:
        (category,) = {d.category for d in module.registry}
        assert {d.id for d in conductor_nodes.registry.filter(lambda d, c=category: d.category == c)} == {d.id for d in module.registry}
