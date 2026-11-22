"""``compute_outputs`` can ask for the types that arrive on the inputs ``compute_added_inputs`` added.

A node's ``compute_added_inputs`` hook may add inputs, such as one per column name. Its ``compute_outputs`` hook can
then take ``**added: type``. It gets the type on the edge into each added input, or ``Any`` where nothing is wired.
"""

from typing import Any

import pytest
from conductor import Graph, GraphRecord, NodeDeclarationError, NodeRegistry, Series, node


def _columns(names: list[str]) -> dict[str, Any]:
    return {name: Any for name in names}


def _typed_like_what_arrives(**added: type) -> dict[str, Any]:
    return dict(added)


@node(
    "pass-on",
    title="Pass on",
    description="Gives each added input as an output.",
    compute_added_inputs=_columns,
    compute_outputs=_typed_like_what_arrives,
)
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


def _compiled(source: tuple[str, str], target: str, names: list[str], **constants: Any):
    """Compiles a graph of the nodes ``n``, ``w`` and ``p``, with one edge into ``p``.

    Args:
        source: The ``(node, output)`` the edge comes from.
        target: The input of ``p`` the edge goes into.
        names: The inputs ``p`` adds.
        **constants: Constants for other inputs of ``p``.
    """
    inputs = {
        "names": {"constant": names},
        **{name: {"constant": value} for name, value in constants.items()},
        target: {"sources": [{"node": source[0], "output": source[1]}]},
    }
    nodes = {
        "n": {"node": "number"},
        "w": {"node": "words"},
        "p": {"node": "pass-on", "inputs": inputs},
    }
    return Graph.compile(GraphRecord.model_validate({"nodes": nodes}), REGISTRY)


def _output_types(graph) -> dict[str, str | None]:
    return {output.name: output.type for output in graph.describe().nodes["p"].outputs}


def test_added_receives_the_type_arriving_on_each_added_input_and_any_where_nothing_is_wired():
    graph = _compiled(("n", "result"), "count", ["count", "note"], note="x")
    assert graph.problems == [] and _output_types(graph) == {"count": "int", "note": "Any"}
    assert graph.run().results["p"] == {"count": 3, "note": "x"}


def test_added_receives_the_item_type_of_a_series_read_one_item_at_a_time():
    graph = _compiled(("w", "result"), "word", ["word"])
    assert graph.problems == [] and _output_types(graph) == {"word": "str"}
    assert graph.run().results["p"]["word"] == ["a", "b"]


def _values_as_text(**added: str) -> dict[str, Any]:
    return {}


def _adding_by_type(**added: type) -> dict[str, Any]:
    return {}


def _plain(text: str) -> dict[str, Any]:
    return {}


def _taking_more(names: list[str], **more: Any) -> dict[str, Any]:
    return {}


@pytest.mark.parametrize(
    ("compute_added_inputs", "compute_outputs", "function", "code"),
    [
        (None, _typed_like_what_arrives, _plain, "hook_parameter_var_keyword"),
        (_columns, _values_as_text, _taking_more, "hook_parameter_var_keyword"),
        (
            _adding_by_type,
            _typed_like_what_arrives,
            _taking_more,
            "arriving_on_compute_added_inputs",
        ),
    ],
    ids=["a-node-that-adds-no-inputs", "added-not-typed-as-type", "added-on-compute-added-inputs"],
)
def test_var_keyword_on_a_hook_is_only_the_added_types_on_compute_outputs(
    compute_added_inputs: Any, compute_outputs: Any, function: Any, code: str
) -> None:
    with pytest.raises(NodeDeclarationError) as raised:
        node(
            "a",
            title="A",
            description="For a test.",
            compute_added_inputs=compute_added_inputs,
            compute_outputs=compute_outputs,
        )(function)
    assert [problem.type for problem in raised.value.problems] == [code]


def test_each_hook_parameter_receives_a_constant_an_arriving_type_or_the_added_types():
    called: dict[str, dict[str, Any]] = {}

    def columns(names: list[str]) -> dict[str, Any]:
        called["compute_added_inputs"] = {"names": names}
        return {name: Any for name in names}

    def outputs(names: list[str], first: type, **added: type) -> dict[str, Any]:
        called["compute_outputs"] = {"names": names, "first": first, **added}
        return {}

    @node(
        "c",
        title="C",
        description="For a test.",
        compute_added_inputs=columns,
        compute_outputs=outputs,
    )
    def c(names: list[str], first: Any, **values: Any) -> dict[str, Any]:
        return {}

    registry = NodeRegistry()
    registry.add(c, number)
    nodes = {
        "n": {"node": "number"},
        "c": {
            "node": "c",
            "inputs": {
                "names": {"constant": ["a", "b"]},
                "first": {"sources": [{"node": "n", "output": "result"}]},
                "a": {"sources": [{"node": "n", "output": "result"}]},
                "b": {"constant": "x"},
            },
        },
    }
    graph = Graph.compile(GraphRecord.model_validate({"nodes": nodes}), registry)
    assert graph.problems == []
    assert called == {
        "compute_added_inputs": {"names": ["a", "b"]},
        "compute_outputs": {"names": ["a", "b"], "first": int, "a": int, "b": Any},
    }
