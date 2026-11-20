"""Generic nodes, whose signature uses a type variable.

Take ``def keep[T](value: T) -> T``. When the graph is compiled, ``T`` becomes the type that arrives on the edge
into ``value``. Wire a ``str`` output into it, and the node's input and output are both ``str``. Only an edge sets
a type variable. Once the graph is compiled, the node's inputs and outputs have ordinary types.
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
    subgraph,
)

from tests.test_core.nodes import registry_of


class Row(TypedDict):
    name: str
    age: int


@dataclass(frozen=True)
class Decided[T]:
    if_true: T = SKIPPED
    if_false: T = SKIPPED


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


@node("fraction", title="Fraction", description="For a test.")
def fraction() -> float:
    return 0.5


@node("kind-of", title="Kind of", description="For a test.")
def kind_of[T](value: T, extra: T = 3) -> str:  # type: ignore[assignment]
    return type(extra).__name__


@node("kind-of-constant", title="Kind of constant", description="For a test.")
def kind_of_constant[T](
    value: T,
    extra: Annotated[T, Param(constant_only=True)] = 3,  # type: ignore[assignment]
) -> str:
    return type(extra).__name__


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


def _compiled(
    *definitions: Any, edges: list[list[str]], constants: dict[str, dict[str, Any]] | None = None
) -> Any:
    """Compiles a graph with one node per definition, named by the definition's id.

    Args:
        *definitions: The nodes of the graph.
        edges: The edges, each a ``[source, target]`` pair such as ``["keep.result", "upper.text"]``.
        constants: Constants by node id, then by input name.
    """
    given = {
        f"{node_id}.{name}": value
        for node_id, named in (constants or {}).items()
        for name, value in named.items()
    }
    record = make_graph(
        *definitions, edges=[(source, target) for source, target in edges], constants=given
    )
    return Graph.compile(record, registry_of(*definitions))


def _types(graph: Any, node_id: str) -> dict[str, tuple[bool, str]]:
    described = graph.describe().nodes[node_id]
    fields = [*(node_input.field for node_input in described.inputs), *described.outputs]
    return {field.name: (field.is_series, field.type) for field in fields}


def test_a_type_variable_is_what_arrives_on_the_edge_in_every_input_and_in_the_return() -> None:
    graph = _compiled(
        text,
        decision,
        upper,
        edges=[["text.result", "decision.value"], ["decision.if_true", "upper.text"]],
    )
    assert graph.problems == []
    assert _types(graph, "decision") == {
        "value": (False, "str"),
        "when": (False, "bool"),
        "if_true": (False, "str"),
        "if_false": (False, "str"),
    }
    assert graph.run().outputs == {"result": "A"}


def test_only_an_edge_binds_a_type_variable_so_a_constant_and_a_graph_input_read_as_any() -> None:
    with_constant = _compiled(keep, edges=[], constants={"keep": {"value": 3}})
    assert _types(with_constant, "keep") == {"value": (False, "Any"), "result": (False, "Any")}
    assert with_constant.run().outputs == {"result": 3}
    assert Graph.compile(make_graph(keep), registry_of(keep)).run({"value": "x"}).outputs == {
        "result": "x"
    }


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
    assert graph.describe().nodes["keep"].per_item_of == "texts.result" and graph.run().outputs == {
        "result": ["a", "b"]
    }


def test_in_a_union_the_option_shaped_like_the_arriving_type_binds() -> None:
    from_table = _compiled(rows, split, edges=[["rows.result", "split.value"]])
    assert _types(from_table, "split")["result"] == (True, "row") and from_table.run().outputs[
        "result"
    ] == [{"name": "a", "age": 1}, {"name": "b", "age": 2}]
    from_list = _compiled(
        keep, split, edges=[["keep.result", "split.value"]], constants={"keep": {"value": [1, 2]}}
    )
    assert _types(from_list, "split")["result"] == (True, "Any")
    graph = _compiled(text, split, edges=[["text.result", "split.value"]])
    assert [(p.type, p.loc, p.ctx["got"], p.ctx["expected"]) for p in graph.problems] == [
        ("type_mismatch", ("split", "value"), "str", "Table | list")
    ]


def test_a_table_of_unknown_columns_binds_any() -> None:
    graph = _compiled(
        anything,
        split,
        upper,
        edges=[["anything.result", "split.value"], ["split.result.name", "upper.text"]],
    )
    assert graph.problems == [] and _types(graph, "split")["result"] == (True, "Any")
    assert graph.run().outputs == {"result": ["A"]}


def test_a_variable_beside_none_binds_the_arriving_type_without_its_none() -> None:
    graph = _compiled(returns_none, maybe, edges=[["returns-none.result", "maybe.value"]])
    assert graph.problems == [] and _types(graph, "maybe")["result"] == (False, "str")


@node("give_some", title="Give some", description="For a test.")
def give_some() -> list[int | None]:
    return [1, None, 2]


@node("compact", title="Compact", description="For a test.")
def compact[T](values: list[T | None]) -> list[T]:
    return [value for value in values if value is not None]


@node("compact_none_first", title="Compact, None first", description="For a test.")
def compact_none_first[T](values: list[None | T]) -> list[T]:
    return [value for value in values if value is not None]


@node("total_of", title="Total of", description="For a test.")
def total_of(values: list[int]) -> int:
    return sum(values)


@pytest.mark.parametrize("dropping", [compact, compact_none_first], ids=["T-or-None", "None-or-T"])
def test_inside_a_value_the_none_of_a_union_takes_a_none_so_the_variable_binds_without_it(
    dropping: Any,
) -> None:
    """Fed ``list[int | None]``, ``list[T | None]`` binds ``T`` to ``int``, whichever member is written first."""
    graph = Graph.compile(
        make_graph(give_some, dropping, total_of), registry_of(give_some, dropping, total_of)
    )
    assert graph.problems == []
    assert graph.run().outputs == {"result": 3}


def test_a_bare_variable_binds_without_none_so_a_none_is_a_none_input() -> None:
    """A type variable on its own never takes ``None``.

    Wire a ``str | None`` output into ``value: T``, and the input is a ``str``. A run that gives it ``None`` leaves
    the node not called. An input takes ``None`` only when its hint says so.
    """
    calls: list[Any] = []

    @node("seen", title="Seen", description="For a test.")
    def seen[T](value: T) -> T:
        calls.append(value)
        return value

    graph = _compiled(returns_none, seen, edges=[["returns-none.result", "seen.value"]])
    assert graph.problems == [] and _types(graph, "seen") == {
        "value": (False, "str"),
        "result": (False, "str"),
    }
    assert not graph.describe().nodes["seen"].inputs[0].field.json_schema.get("anyOf")
    run = graph.run()
    assert calls == [] and [(n.loc, n.input) for n in run.none_input_calls] == [
        (("seen",), "value")
    ]
    inside_a_list = _compiled(
        keep, seen, edges=[["keep.result", "seen.value"]], constants={"keep": {"value": 1}}
    )
    assert _types(inside_a_list, "seen")["value"] == (False, "Any")  # Any allows None already


def test_a_branch_node_with_a_none_input_leaves_both_branches_skipped() -> None:
    graph = _compiled(
        returns_none,
        decision,
        maybe,
        edges=[["returns-none.result", "decision.value"], ["decision.if_true", "maybe.value"]],
    )
    assert graph.problems == []
    run = graph.run()
    assert run.results["maybe"] == {"result": None} and [n.loc for n in run.none_input_calls] == [
        ("decision",)
    ]

    async def said() -> list[tuple[tuple[str, ...], int, int]]:
        return [
            (event.loc, event.count.skipped, event.count.none_input)
            async for event in graph.with_inputs().stream()
            if event.type == "node_complete"
        ]

    assert asyncio.run(said())[-2:] == [
        (("decision",), 0, 1),
        (("maybe",), 1, 0),
    ]  # maybe is skipped, not a none input. Only a None input makes a none input call


def test_one_variable_on_two_inputs_binds_the_union_of_what_arrives() -> None:
    graph = _compiled(
        number, text, either, edges=[["number.result", "either.a"], ["text.result", "either.b"]]
    )
    assert graph.problems == [] and _types(graph, "either")["result"] == (False, "int | str")


def test_what_arrives_outside_a_variables_bound_is_a_type_mismatch() -> None:
    @node("count", title="Count", description="For a test.")
    def count[N: (int | float)](value: N) -> N:
        return value

    assert _types(_compiled(count, edges=[]), "count")["value"] == (False, "int | float")
    assert [
        p.type for p in _compiled(text, count, edges=[["text.result", "count.value"]]).problems
    ] == ["type_mismatch"]
    assert _types(_compiled(number, count, edges=[["number.result", "count.value"]]), "count")[
        "result"
    ] == (False, "int")


def test_a_variable_on_no_input_and_a_constrained_one_are_refused_where_the_node_is_declared() -> (
    None
):
    with pytest.raises(NodeDeclarationError) as only_in_return:

        @node("make", title="Make", description="For a test.")
        def make[T](text: str) -> list[T]:
            return []

    with pytest.raises(NodeDeclarationError) as constrained:

        @node("pick", title="Pick", description="For a test.")
        def pick[T: (int, str)](value: T) -> T:
            return value

    assert (only_in_return.value.problems[0].type, constrained.value.problems[0].type) == (
        "type_variable_only_in_return",
        "type_variable_constrained",
    )


def test_a_generic_dataclass_of_outputs_returns_one_branch_at_run_time() -> None:
    graph = _compiled(
        text,
        decision,
        upper,
        edges=[["text.result", "decision.value"], ["decision.if_false", "upper.text"]],
    )
    run = graph.run()
    assert (
        run.status == "complete"
        and run.results["decision"] == {"if_true": "a", "if_false": None}
        and run.outputs == {"result": None}
    )


def test_a_generic_dataclass_of_outputs_fills_in_its_arguments_and_reads_an_open_one_as_any() -> (
    None
):
    """``-> Decided[int]`` has ``int`` outputs, and a bare ``-> Decided`` has ``Any`` ones, as a bare ``TypedDict`` has.

    A bare one crashed ``@node`` with a ``ValueError`` before.
    """

    @node("filled", title="Filled", description="For a test.")
    def filled() -> Decided[int]:
        return Decided(if_true=1)

    @node("bare", title="Bare", description="For a test.")
    def bare() -> Decided:  # type: ignore[type-arg]
        return Decided(if_true=1)

    assert [(field.name, field.type) for field in filled.describe().outputs] == [
        ("if_true", "int"),
        ("if_false", "int"),
    ]
    assert [(field.name, field.type) for field in bare.describe().outputs] == [
        ("if_true", "Any"),
        ("if_false", "Any"),
    ]
    assert bare.describe().outputs[0].type_variables == []


def test_a_generic_definition_describes_any_and_its_compiled_node_the_bound_types() -> None:
    assert [(field.name, field.type) for field in decision.describe().inputs] == [
        ("value", "Any"),
        ("when", "bool"),
    ]
    assert [(field.name, field.type) for field in decision.describe().outputs] == [
        ("if_true", "Any"),
        ("if_false", "Any"),
    ]
    assert decision.describe().inputs[0].type_variables == ["T"]
    compiled_node = (
        _compiled(text, decision, edges=[["text.result", "decision.value"]])
        .describe()
        .nodes["decision"]
    )
    assert [node_input.field.type_variables for node_input in compiled_node.inputs] == [
        [],
        [],
    ] and [output.type_variables for output in compiled_node.outputs] == [[], []]


def test_an_input_a_hook_adds_with_a_type_variable_is_bound_by_its_edge() -> None:
    variable = TypeVar("variable")

    @node(
        "echo",
        title="Echo",
        description="For a test.",
        compute_added_inputs=lambda: {"value": variable},
    )
    def echo(**added: Any) -> str:
        return repr(added["value"])

    graph = _compiled(text, echo, edges=[["text.result", "echo.value"]])
    assert [
        (node_input.field.type, node_input.field.type_variables)
        for node_input in graph.describe().nodes["echo"].inputs
    ] == [("str", [])]
    assert graph.run().outputs == {"result": "'a'"}


@pytest.mark.parametrize("kind", [kind_of, kind_of_constant])
def test_a_constant_for_a_type_variable_is_checked_by_the_type_the_edge_bound(kind: Any) -> None:
    refused = _compiled(
        text, kind, edges=[["text.result", f"{kind.id}.value"]], constants={kind.id: {"extra": 5}}
    )
    assert [(problem.type, problem.loc) for problem in refused.problems] == [
        ("invalid_value", (kind.id, "extra"))
    ]
    taken = _compiled(
        text, kind, edges=[["text.result", f"{kind.id}.value"]], constants={kind.id: {"extra": "b"}}
    )
    assert taken.problems == []
    assert taken.run().outputs == {"result": "str"}


@pytest.mark.parametrize("kind", [kind_of, kind_of_constant])
def test_a_constant_for_a_type_variable_is_converted_by_the_type_the_edge_bound(kind: Any) -> None:
    graph = _compiled(
        fraction,
        kind,
        edges=[["fraction.result", f"{kind.id}.value"]],
        constants={kind.id: {"extra": 2}},
    )
    assert graph.problems == []
    assert graph.run().outputs == {"result": "float"}


@pytest.mark.parametrize("kind", [kind_of, kind_of_constant])
def test_a_constant_for_a_type_variable_inside_a_subgraph_is_checked_by_the_bound_type(
    kind: Any,
) -> None:
    inner = make_graph(
        text, kind, edges=[("text.result", f"{kind.id}.value")], constants={f"{kind.id}.extra": 5}
    )
    inside = subgraph("inside", title="Inside", description="For a test.")(inner)
    graph = Graph.compile(make_graph(inside), registry_of(text, kind, inside))
    assert [(problem.type, problem.loc) for problem in graph.problems] == [
        ("invalid_value", ("inside", kind.id, "extra"))
    ]


@pytest.mark.parametrize("constants", [{"extra": 2}, {}])
def test_compute_outputs_reads_a_constant_or_default_by_the_type_the_edge_bound(
    constants: dict[str, Any],
) -> None:
    @node(
        "typed-by-extra",
        title="Typed by extra",
        description="For a test.",
        compute_outputs=lambda extra: {"result": type(extra)},
    )
    def typed_by_extra[T](value: T, extra: T = 3) -> dict[str, Any]:  # type: ignore[assignment]
        return {"result": extra}

    record = make_graph(
        fraction,
        typed_by_extra,
        edges=[("fraction.result", "typed-by-extra.value")],
        constants={f"typed-by-extra.{name}": value for name, value in constants.items()},
        outputs={"result": "typed-by-extra.result"},
    )
    graph = Graph.compile(record, registry_of(fraction, typed_by_extra))
    assert graph.problems == []
    assert _types(graph, "typed-by-extra")["result"] == (False, "float")
    assert graph.run().outputs == {"result": float(constants.get("extra", 3))}
