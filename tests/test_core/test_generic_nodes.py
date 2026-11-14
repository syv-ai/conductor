"""A generic node: a type variable in its signature is bound to the type that arrives on the edge.

Only an edge binds a variable, when the graph is compiled; in the compiled graph the node's inputs and outputs have
ordinary types.
"""

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict, TypeVar

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    NodeDeclarationError,
    Param,
    Series,
    Table,
    make_graph,
    node,
)

from tests.test_core.nodes import registry_of


class Row(TypedDict):
    name: str
    age: int


@dataclass(frozen=True)
class Decided[T]:
    if_true: Annotated[T, Param(exclusive_group="when")] = SKIPPED
    if_false: Annotated[T, Param(exclusive_group="when")] = SKIPPED


@node("decision", title="Decision", description="For a test.")
def decision[T](value: T, when: bool = True) -> Decided[T]:
    return Decided(if_true=value) if when else Decided(if_false=value)


@node("keep", title="Keep", description="For a test.")
def keep[T](value: T) -> T:
    return value


@node("either", title="Either", description="For a test.")
def either[T](a: T, b: T) -> T:
    return a


@node("maybe", title="Maybe", description="For a test.")
def maybe[T](value: T | None) -> T:
    return value  # type: ignore[return-value]


@node("split", title="Split", description="For a test.")
def split[T](value: Table[T] | list[T]) -> Series[T]:
    return value.to_dicts() if isinstance(value, pl.DataFrame) else value  # type: ignore[return-value]


@node("number", title="Number", description="For a test.")
def number() -> int:
    return 3


@node("text", title="Text", description="For a test.")
def text() -> str:
    return "a"


@node("returns-none", title="Returns none", description="For a test.")
def returns_none() -> str | None:
    return None


@node("texts", title="Texts", description="For a test.")
def texts() -> Series[str]:
    return ["a", "b"]


@node("rows", title="Rows", description="For a test.")
def rows() -> Table[Row]:
    return pl.DataFrame({"name": ["a", "b"], "age": [1, 2]})


@node("anything", title="Anything", description="For a test.")
def anything() -> Table:
    return pl.DataFrame({"name": ["a"]})


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


def _compiled(*definitions: Any, edges: list[list[str]], constants: dict[str, dict[str, Any]] | None = None) -> Any:
    """Each definition as a node named by its id, wired by ``edges``; ``constants`` are given per node id."""
    given = {f"{node_id}.{name}": value for node_id, named in (constants or {}).items() for name, value in named.items()}
    record = make_graph(*definitions, edges=[(source, target) for source, target in edges], constants=given)
    return Graph.compile(record, registry_of(*definitions))


def _types(graph: Any, node_id: str) -> dict[str, tuple[bool, str]]:
    described = graph.describe().nodes[node_id]
    fields = [*(node_input.field for node_input in described.inputs), *described.outputs]
    return {field.name: (field.is_series, field.type) for field in fields}


def test_a_type_variable_is_what_arrives_on_the_edge_in_every_input_and_in_the_return() -> None:
    graph = _compiled(text, decision, upper, edges=[["text.result", "decision.value"], ["decision.if_true", "upper.text"]])
    assert graph.problems == []
    assert _types(graph, "decision") == {"value": (False, "str"), "when": (False, "bool"), "if_true": (False, "str"), "if_false": (False, "str")}
    assert graph.run().outputs == {"result": "A"}


def test_only_an_edge_binds_a_type_variable_so_a_constant_and_a_graph_input_read_as_any() -> None:
    with_constant = _compiled(keep, edges=[], constants={"keep": {"value": 3}})
    assert _types(with_constant, "keep") == {"value": (False, "Any"), "result": (False, "Any")}
    assert with_constant.run().outputs == {"result": 3}
    assert Graph.compile(make_graph(keep), registry_of(keep)).run(value="x").outputs == {"result": "x"}


def test_an_output_is_checked_against_the_type_the_edge_bound() -> None:
    @node("liar", title="Liar", description="For a test.")
    def liar[T](value: T) -> T:
        return 5  # type: ignore[return-value]

    run = _compiled(text, liar, edges=[["text.result", "liar.value"]]).run()
    assert (run.status, run.cause.type) == ("failed", "invalid_output")


def test_the_choices_of_an_ask_are_checked_against_the_type_the_edge_bound() -> None:
    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick[T](value: T) -> T:
        raise Ask("Which?", choices=[1, 2])

    asked_for_numbers = _compiled(number, pick, edges=[["number.result", "pick.value"]]).run()
    asked_for_texts = _compiled(text, pick, edges=[["text.result", "pick.value"]]).run()
    assert asked_for_numbers.status == "waiting"
    assert (asked_for_texts.status, asked_for_texts.cause.type) == ("failed", "invalid_ask")


def test_a_series_wired_to_a_type_variable_is_looped_over_and_binds_the_item_type() -> None:
    graph = _compiled(texts, keep, edges=[["texts.result", "keep.value"]])
    assert _types(graph, "keep")["value"] == (False, "str")
    assert graph.describe().nodes["keep"].per_item_of == "texts.result" and graph.run().outputs == {"result": ["a", "b"]}


def test_in_a_union_the_option_shaped_like_the_arriving_type_binds() -> None:
    from_table = _compiled(rows, split, edges=[["rows.result", "split.value"]])
    assert _types(from_table, "split")["result"] == (True, "record") and from_table.run().outputs["result"] == [{"name": "a", "age": 1}, {"name": "b", "age": 2}]
    from_list = _compiled(keep, split, edges=[["keep.result", "split.value"]], constants={"keep": {"value": [1, 2]}})
    assert _types(from_list, "split")["result"] == (True, "Any")
    graph = _compiled(text, split, edges=[["text.result", "split.value"]])
    assert [(p.type, p.loc, p.ctx["got"], p.ctx["expected"]) for p in graph.problems] == [("type_mismatch", ("split", "value"), "str", "Table | list")]


def test_a_table_of_unknown_columns_binds_any() -> None:
    graph = _compiled(anything, split, upper, edges=[["anything.result", "split.value"], ["split.result.name", "upper.text"]])
    assert graph.problems == [] and _types(graph, "split")["result"] == (True, "Any")
    assert graph.run().outputs == {"result": ["A"]}


def test_a_variable_beside_none_binds_the_arriving_type_without_its_none() -> None:
    graph = _compiled(returns_none, maybe, edges=[["returns-none.result", "maybe.value"]])
    assert graph.problems == [] and _types(graph, "maybe")["result"] == (False, "str")


def test_a_bare_variable_binds_without_none_so_a_none_leaves_the_node_not_called() -> None:
    """An input gets ``None`` only if its hint says so: ``value: T`` wired to a ``str | None`` output is a ``str`` input."""
    calls: list[Any] = []

    @node("seen", title="Seen", description="For a test.")
    def seen[T](value: T) -> T:
        calls.append(value)
        return value

    graph = _compiled(returns_none, seen, edges=[["returns-none.result", "seen.value"]])
    assert graph.problems == [] and _types(graph, "seen") == {"value": (False, "str"), "result": (False, "str")}
    assert not graph.describe().nodes["seen"].inputs[0].field.json_schema.get("anyOf")
    run = graph.run()
    assert calls == [] and [(n.loc, n.input) for n in run.not_called] == [(("seen",), "value")]
    inside_a_list = _compiled(keep, seen, edges=[["keep.result", "seen.value"]], constants={"keep": {"value": 1}})
    assert _types(inside_a_list, "seen")["value"] == (False, "Any")  # Any takes None, and says so


def test_a_branch_node_not_called_for_a_none_leaves_both_branches_not_chosen() -> None:
    graph = _compiled(returns_none, decision, maybe, edges=[["returns-none.result", "decision.value"], ["decision.if_true", "maybe.value"]])
    assert graph.problems == []
    run = graph.run()
    assert run.results["maybe"] == {"result": None} and [n.loc for n in run.not_called] == [("decision",)]

    async def said() -> list[tuple[str, tuple[str, ...]]]:
        return [(event.type, event.loc) async for event in graph.with_inputs().stream() if event.type in ("node_complete", "node_skipped")]

    assert asyncio.run(said())[-2:] == [("node_complete", ("decision",)), ("node_skipped", ("maybe",))]  # skipped, where a None would leave it not called


def test_one_variable_on_two_inputs_binds_the_union_of_what_arrives() -> None:
    graph = _compiled(number, text, either, edges=[["number.result", "either.a"], ["text.result", "either.b"]])
    assert graph.problems == [] and _types(graph, "either")["result"] == (False, "int | str")


def test_what_arrives_outside_a_variables_bound_is_a_type_mismatch() -> None:
    @node("count", title="Count", description="For a test.")
    def count[N: (int | float)](value: N) -> N:
        return value

    assert _types(_compiled(count, edges=[]), "count")["value"] == (False, "int | float")
    assert [p.type for p in _compiled(text, count, edges=[["text.result", "count.value"]]).problems] == ["type_mismatch"]
    assert _types(_compiled(number, count, edges=[["number.result", "count.value"]]), "count")["result"] == (False, "int")


def test_a_variable_on_no_input_and_a_constrained_one_are_refused_where_the_node_is_declared() -> None:
    with pytest.raises(NodeDeclarationError) as only_in_return:

        @node("make", title="Make", description="For a test.")
        def make[T](text: str) -> list[T]:
            return []

    with pytest.raises(NodeDeclarationError) as constrained:

        @node("pick", title="Pick", description="For a test.")
        def pick[T: (int, str)](value: T) -> T:
            return value

    assert (only_in_return.value.problems[0].type, constrained.value.problems[0].type) == ("type_variable_only_in_return", "type_variable_constrained")


def test_a_generic_dataclass_of_outputs_returns_one_branch_at_run_time() -> None:
    graph = _compiled(text, decision, upper, edges=[["text.result", "decision.value"], ["decision.if_false", "upper.text"]])
    run = graph.run()
    assert run.status == "complete" and run.results["decision"] == {"if_true": "a", "if_false": None} and run.outputs == {"result": None}


def test_a_generic_definition_describes_any_and_its_compiled_node_the_bound_types() -> None:
    assert [(field.name, field.type) for field in decision.describe().inputs] == [("value", "Any"), ("when", "bool")]
    assert [(field.name, field.type) for field in decision.describe().outputs] == [("if_true", "Any"), ("if_false", "Any")]
    assert decision.describe().inputs[0].type_variables == ["T"]
    compiled_node = _compiled(text, decision, edges=[["text.result", "decision.value"]]).describe().nodes["decision"]
    assert [node_input.field.type_variables for node_input in compiled_node.inputs] == [[], []] and [output.type_variables for output in compiled_node.outputs] == [[], []]


def test_an_input_a_hook_adds_with_a_type_variable_is_bound_by_its_edge() -> None:
    variable = TypeVar("variable")

    @node("echo", title="Echo", description="For a test.", compute_added_inputs=lambda: {"value": variable})
    def echo(**added: Any) -> str:
        return repr(added["value"])

    graph = _compiled(text, echo, edges=[["text.result", "echo.value"]])
    assert [(node_input.field.type, node_input.field.type_variables) for node_input in graph.describe().nodes["echo"].inputs] == [("str", [])]
    assert graph.run().outputs == {"result": "'a'"}
