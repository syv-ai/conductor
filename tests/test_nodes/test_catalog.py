"""Checks that every node in ``conductor_nodes`` must pass, run over the whole registry at once."""

import conductor_nodes
from conductor import Graph, NodeRegistry, make_graph
from conductor.testing import check_registry

#: The package's modules, one per category, as its ``__all__`` lists them.
MODULES = [getattr(conductor_nodes, name) for name in conductor_nodes.__all__ if name != "registry"]
#: ``conductor_nodes.registry``, which has every node, and each module's own.
REGISTRIES = [conductor_nodes.registry, *(module.registry for module in MODULES)]

#: Sample inputs for each node, one dict per call. ``check_registry`` runs the node on each of them. The nodes here
#: only compute, with no outside service, so every one of them can have an example.
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
    "logic-not": [{"value": True}],
    "logic-equals": [{"value": "a", "other": "a"}, {"value": 1, "other": 2}],
    "logic-is-empty": [{"value": None}, {"value": []}, {"value": "x"}],
    "json-parse": [{"text": '{"a": [1, 2]}'}],
    "json-stringify": [{"value": {"b": 1, "a": 2}, "sort_keys": True}],
    "json-get": [{"value": {"items": [{"id": "a"}]}, "path": "items.0.id"}],
    "regex-match": [{"text": "a1", "pattern": r"\d"}],
    "regex-replace": [{"text": "a1b2", "pattern": r"\d", "replacement": "-"}],
    "regex-extract": [{"text": "a1b22", "pattern": r"\d+"}],
    "control-decide": [{"value": "x", "when": False}],
    "control-switch": [
        {"value": 1, "case": "b", "cases": ["a", "b"]},
        {"value": 1, "case": "z", "cases": ["a"]},
    ],
    "control-default": [{"value": None, "fallback": "x"}, {"value": "y", "fallback": "x"}],
    "series-split": [{"value": ["a", "b"]}],
    "series-collect": [{"values": ["a", "b"]}],
    "series-stack": [{"tables": [[{"name": "Ann"}], [{"name": "Bo"}]]}],
    "table-collect": [{"records": [{"name": "Ann"}], "constant_columns": ["team"], "team": "red"}],
}


def test_the_whole_registry_passes_check_registry() -> None:
    assert check_registry(conductor_nodes.registry, examples=EXAMPLES) == []


def test_every_node_has_an_example_so_no_body_goes_untried() -> None:
    assert {definition.id for definition in conductor_nodes.registry} == set(EXAMPLES)


def test_each_module_is_one_category_and_the_package_registry_has_the_nodes_of_every_module() -> (
    None
):
    assert sum(len(module.registry) for module in MODULES) == len(conductor_nodes.registry)
    for module in MODULES:
        (category,) = {d.category for d in module.registry}
        assert category, module.__name__
        assert {
            d.id for d in conductor_nodes.registry.filter(lambda d, c=category: d.category == c)
        } == {d.id for d in module.registry}


def test_every_version_of_every_node_costs_nothing_as_the_package_promises() -> None:
    for registry in REGISTRIES:
        for definition in registry:
            for version in definition.versions.values():
                assert version.policy.costs_nothing, (version.id, version.version)


def test_an_estimate_counts_the_calls_after_a_split_before_any_costly_call() -> None:
    registry = NodeRegistry()
    registry.add_subregistry(conductor_nodes.registry)

    @registry.node(
        "costly", title="Costly", description="Stands for a node whose call costs something."
    )
    def costly(item: str) -> str:
        return item

    graph = Graph.compile(
        make_graph(
            conductor_nodes.series.split, costly, edges=[("series-split.result", "costly.item")]
        ),
        registry,
    )
    estimated = graph.with_inputs({"value": ["a", "b", "c"]}).resume(only_costs_nothing=True)

    assert estimated.status == "ready"
    assert (
        estimated.progress.nodes["costly"].finished,
        estimated.progress.nodes["costly"].total,
    ) == (0, 3)
