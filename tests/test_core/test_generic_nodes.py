"""A node for any type: a type variable in the signature is what arrives on the edge.

Only an edge binds a variable, when the graph loads; after that the placed node has ordinary fields.
"""

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import SKIPPED, NodeDeclarationError, Param, Series, Table, node

from tests.test_core.nodes import registry_of


class Row(TypedDict):
    name: str
    age: int


@dataclass(frozen=True)
class Decided[T]:
    if_true: Annotated[T, Param(exclusive_group="when")] = SKIPPED
    if_false: Annotated[T, Param(exclusive_group="when")] = SKIPPED


@node("decision", title="Decision", description="")
def decision[T](value: T, when: bool = True) -> Decided[T]:
    return Decided(if_true=value) if when else Decided(if_false=value)


@node("keep", title="Keep", description="")
def keep[T](value: T) -> T:
    return value


@node("either", title="Either", description="")
def either[T](a: T, b: T) -> T:
    return a


@node("maybe", title="Maybe", description="")
def maybe[T](value: T | None) -> T:
    return value  # type: ignore[return-value]


@node("split", title="Split", description="")
def split[T](value: Table[T] | list[T]) -> Series[T]:
    return value.to_dicts() if isinstance(value, pl.DataFrame) else value  # type: ignore[return-value]


@node("number", title="Number", description="")
def number() -> int:
    return 3


@node("text", title="Text", description="")
def text() -> str:
    return "a"


@node("blank", title="Blank", description="")
def blank() -> str | None:
    return None


@node("texts", title="Texts", description="")
def texts() -> Series[str]:
    return ["a", "b"]


@node("rows", title="Rows", description="")
def rows() -> Table[Row]:
    return pl.DataFrame({"name": ["a", "b"], "age": [1, 2]})


@node("anything", title="Anything", description="")
def anything() -> Table:
    return pl.DataFrame({"name": ["a"]})


@node("upper", title="Upper", description="")
def upper(text: str) -> str:
    return text.upper()


def _loaded(*definitions: Any, edges: list[list[str]], values: dict[str, dict[str, Any]] | None = None) -> Any:
    """Each definition as a node named by its id, wired by ``edges``; ``values`` are typed in by node id."""
    nodes = [{"id": d.id, "node": d.id, "values": (values or {}).get(d.id, {})} for d in definitions]
    return registry_of(*definitions).load({"nodes": nodes, "edges": edges})


def _types(graph: Any, node_id: str) -> dict[str, tuple[bool, str]]:
    described = graph.describe().nodes[node_id]
    fields = [*(placed.field for placed in described.inputs), *described.outputs]
    return {field.name: (field.many, field.type) for field in fields}


def test_a_type_variable_is_what_arrives_on_the_edge_in_every_input_and_in_the_return() -> None:
    graph = _loaded(text, decision, upper, edges=[["text.result", "decision.value"], ["decision.if_true", "upper.text"]])
    assert graph.problems == []
    assert _types(graph, "decision") == {"value": (False, "str"), "when": (False, "bool"), "if_true": (False, "str"), "if_false": (False, "str")}
    assert graph.run().outputs == {"upper.result": "A"}


def test_only_an_edge_binds_so_a_typed_in_value_and_an_open_input_read_as_any() -> None:
    typed = _loaded(keep, edges=[], values={"keep": {"value": 3}})
    assert _types(typed, "keep") == {"value": (False, "Any"), "result": (False, "Any")}
    assert typed.run().outputs == {"keep.result": 3}
    assert registry_of(keep).load({"nodes": [{"id": "keep", "node": "keep"}]}).run(value="x").outputs == {"keep.result": "x"}


def test_an_output_is_checked_against_the_type_the_edge_bound() -> None:
    @node("liar", title="Liar", description="")
    def liar[T](value: T) -> T:
        return 5  # type: ignore[return-value]

    run = _loaded(text, liar, edges=[["text.result", "liar.value"]]).run()
    assert (run.status, run.state.cause.type) == ("failed", "invalid_output")


def test_one_or_many_is_decided_first_so_a_series_at_a_variable_runs_per_item() -> None:
    graph = _loaded(texts, keep, edges=[["texts.result", "keep.value"]])
    assert _types(graph, "keep")["value"] == (False, "str")
    assert graph.plan["keep"].series == "texts.result" and graph.run().outputs == {"keep.result": ["a", "b"]}


def test_in_a_union_the_option_shaped_like_the_arriving_type_binds() -> None:
    from_table = _loaded(rows, split, edges=[["rows.result", "split.value"]])
    assert _types(from_table, "split")["result"] == (True, "record") and from_table.run().outputs["split.result"] == [{"name": "a", "age": 1}, {"name": "b", "age": 2}]
    from_list = _loaded(keep, split, edges=[["keep.result", "split.value"]], values={"keep": {"value": [1, 2]}})
    assert _types(from_list, "split")["result"] == (True, "Any")
    graph = _loaded(text, split, edges=[["text.result", "split.value"]])
    assert [(p.type, p.loc, p.ctx["got"], p.ctx["expected"]) for p in graph.problems] == [("type_mismatch", ("split", "value"), "str", "Table | list")]


def test_a_table_of_unknown_columns_binds_any() -> None:
    graph = _loaded(anything, split, upper, edges=[["anything.result", "split.value"], ["split.result.name", "upper.text"]])
    assert graph.problems == [] and _types(graph, "split")["result"] == (True, "Any")
    assert graph.run().outputs == {"upper.result": ["A"]}


def test_a_variable_beside_none_is_what_arrives_without_its_blank() -> None:
    graph = _loaded(blank, maybe, edges=[["blank.result", "maybe.value"]])
    assert graph.problems == [] and _types(graph, "maybe")["result"] == (False, "str")


def test_two_places_for_one_variable_join() -> None:
    graph = _loaded(number, text, either, edges=[["number.result", "either.a"], ["text.result", "either.b"]])
    assert graph.problems == [] and _types(graph, "either")["result"] == (False, "int | str")


def test_what_arrives_outside_a_variables_bound_is_a_type_mismatch() -> None:
    @node("count", title="Count", description="")
    def count[N: (int | float)](value: N) -> N:
        return value

    assert _types(_loaded(count, edges=[]), "count")["value"] == (False, "int | float")
    assert [p.type for p in _loaded(text, count, edges=[["text.result", "count.value"]]).problems] == ["type_mismatch"]
    assert _types(_loaded(number, count, edges=[["number.result", "count.value"]]), "count")["result"] == (False, "int")


def test_a_variable_on_no_input_and_a_constrained_one_are_refused_where_the_node_is_declared() -> None:
    with pytest.raises(NodeDeclarationError) as only_in_return:

        @node("make", title="Make", description="")
        def make[T](text: str) -> list[T]:
            return []

    with pytest.raises(NodeDeclarationError) as constrained:

        @node("pick", title="Pick", description="")
        def pick[T: (int, str)](value: T) -> T:
            return value

    assert (only_in_return.value.code, constrained.value.code) == ("type_variable_only_in_return", "type_variable_constrained")


def test_a_generic_dataclass_of_outputs_holds_one_alternative_at_run_time() -> None:
    graph = _loaded(text, decision, upper, edges=[["text.result", "decision.value"], ["decision.if_false", "upper.text"]])
    run = graph.run()
    assert run.status == "complete" and run.results["decision"] == {"if_true": "a", "if_false": None} and run.outputs == {"upper.result": None}


def test_a_generic_node_in_the_palette_shows_any() -> None:
    assert [(field.name, field.type) for field in decision.describe().inputs] == [("value", "Any"), ("when", "bool")]
    assert [(field.name, field.type) for field in decision.describe().outputs] == [("if_true", "Any"), ("if_false", "Any")]
    assert decision.describe().inputs[0].type_variables == ["T"]
    placed = _loaded(text, decision, edges=[["text.result", "decision.value"]]).describe().nodes["decision"]
    assert [placed_input.field.type_variables for placed_input in placed.inputs] == [[], []] and [output.type_variables for output in placed.outputs] == [[], []]
