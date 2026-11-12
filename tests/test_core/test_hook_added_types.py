"""``compute_outputs`` reads the types arriving on the inputs ``compute_added_inputs`` added, as ``**added: type``."""

from typing import Any

import pytest
from conductor import Graph, GraphRecord, NodeDeclarationError, NodeRegistry, Series, node


def _columns(names: list[str]) -> dict[str, Any]:
    return {name: Any for name in names}


def _typed_like_what_arrives(**added: type) -> dict[str, Any]:
    return dict(added)


@node("pass-on", title="Pass on", description="Gives each added input as an output.", compute_added_inputs=_columns, compute_outputs=_typed_like_what_arrives)
def pass_on(names: list[str], **values: Any) -> dict[str, Any]:
    return values


@node("number", title="Number", description="A number.")
def number() -> int:
    return 3


@node("words", title="Words", description="Some words.")
def words() -> Series[str]:
    return ["a", "b"]


REGISTRY = NodeRegistry()
for _definition in (pass_on, number, words):
    REGISTRY.add(_definition)


def _compiled(edges: list[list[str]], names: list[str], **constants: Any):
    nodes = [{"id": "n", "node": "number"}, {"id": "w", "node": "words"}, {"id": "p", "node": "pass-on", "constants": {"names": names, **constants}}]
    return Graph.compile(GraphRecord.model_validate({"nodes": nodes, "edges": edges}), REGISTRY)


def _output_types(graph) -> dict[str, str | None]:
    return {output.name: output.type for output in graph.describe().nodes["p"].outputs}


def test_added_receives_the_type_arriving_on_each_added_input_and_any_where_nothing_is_wired():
    graph = _compiled([["n.result", "p.count"]], ["count", "note"], note="x")
    assert graph.problems == [] and _output_types(graph) == {"count": "int", "note": "Any"}
    assert graph.run().results["p"] == {"count": 3, "note": "x"}


def test_added_receives_the_item_type_of_a_series_read_one_item_at_a_time():
    graph = _compiled([["w.result", "p.word"]], ["word"])
    assert graph.problems == [] and _output_types(graph) == {"word": "str"}
    assert graph.run().results["p"]["word"] == ["a", "b"]


def test_a_node_that_adds_no_inputs_has_no_added_types_to_read():
    def plain(text: str) -> dict[str, Any]:
        return {}

    with pytest.raises(NodeDeclarationError) as raised:
        node("plain", title="Plain", description="For a test.", compute_outputs=_typed_like_what_arrives)(plain)
    assert raised.value.code == "hook_parameter_unknown"


def test_var_keyword_on_a_hook_is_only_the_added_types():
    def values(**added: str) -> dict[str, Any]:
        return {}

    def adding(**added: type) -> dict[str, Any]:
        return {}

    def run(names: list[str], **more: Any) -> dict[str, Any]:
        return {}

    with pytest.raises(NodeDeclarationError) as untyped:
        node("a", title="A", description="For a test.", compute_added_inputs=_columns, compute_outputs=values)(run)
    with pytest.raises(NodeDeclarationError) as too_early:
        node("b", title="B", description="For a test.", compute_added_inputs=adding, compute_outputs=_typed_like_what_arrives)(run)
    assert (untyped.value.code, too_early.value.code) == ("hook_parameter_unknown", "arriving_on_compute_added_inputs")


def test_the_definition_says_what_each_hook_parameter_receives():
    def outputs(names: list[str], first: type, **added: type) -> dict[str, Any]:
        return {}

    @node("c", title="C", description="For a test.", compute_added_inputs=_columns, compute_outputs=outputs)
    def c(names: list[str], first: Any, **values: Any) -> dict[str, Any]:
        return {}

    assert c.hook_parameters == {"compute_added_inputs": {"names": "value"}, "compute_outputs": {"names": "value", "first": "arriving_type", "added": "added_arriving_types"}}
