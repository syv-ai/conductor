"""Compiling a graph record: what ``Graph.compile`` checks, what the graph offers a caller, and subgraphs.

A record that says something impossible compiles into a graph whose ``problems`` say what and where, never an
exception; a graph with problems refuses to run. A caller gives the graph inputs, and ``with_inputs`` checks them all
at once.
"""

import json
from dataclasses import dataclass
from io import StringIO
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    Choice,
    Deprecation,
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeError,
    NodeRecord,
    Param,
    Problem,
    Series,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from conductor.widgets import Dropdown, Textarea
from pydantic import ValidationError
from pydantic_core import core_schema
from pytest import MonkeyPatch

from tests.test_core.nodes import registry_of


class Person(TypedDict):
    name: str
    email: str


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    return text.upper()


@node("tail", title="Tail", description="For a test.")
def tail(text: str) -> str:
    return text + "."


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("both_ints", title="Both ints", description="For a test.")
def both_ints(a: int, b: int) -> str:
    return f"{a}{b}"


@node("fan", title="Fan", description="For a test.")
def fan() -> Series[int]:
    return [1, 2, 3]


@node("maybe", title="Maybe", description="For a test.")
def maybe(text: str | None) -> str:
    return text or "none"


@node("greet", title="Greet", description="For a test.")
def greet(text: str = "hello") -> str:
    return text


@node("people", title="People", description="For a test.")
def people(text: str) -> Table[Person]:
    return pl.read_csv(StringIO(text))


@node("styled", title="Styled", description="For a test.")
def styled(
    mode: Annotated[str, Param(title="Mode", choices=[Choice(id="a", title="A"), Choice(id="b", title="B")], widget=Dropdown())] = "a",
) -> str:
    return mode


@host_type("Name", title="Name")
class Name(str):
    """A subclass of ``str``: it fits any input that takes a ``str``."""


@node("name", title="Name of", description="For a test.")
def name_of(text: str) -> Name:
    return Name(text)


@host_type("Nonempty", title="Nonempty")
class Nonempty(str):
    def __new__(cls, value: str) -> "Nonempty":
        if not value:
            raise ValueError("empty")
        return super().__new__(cls, value)


@node("nonempty", title="Nonempty", description="For a test.")
def nonempty(text: Nonempty) -> str:
    return text


@dataclass(frozen=True)
class Two:
    zed: str
    alpha: str


@node("two", title="Two", description="For a test.")
def two(text: str) -> Two:
    return Two(f"zed {text}", f"alpha {text}")


def _columns(names: list[str]) -> dict[str, Any]:
    return {name: Annotated[str, Param(title=name.title())] for name in names}


@node("columns", title="Columns", description="For a test.", compute_outputs=_columns)
def columns(text: str, names: list[str] = []) -> dict[str, str]:  # noqa: B006
    return {name: text for name in names}


def _anything(n: int) -> dict[str, Any]:
    return {"items": Series[Any]}


@node("anything", title="Anything", description="For a test.", compute_outputs=_anything)
def anything(n: int = 2) -> dict[str, Any]:
    return {"items": ["x"] * n}


@node("height", title="Height", description="For a test.")
def height(rows: Table) -> int:
    return rows.height


def _refuses(mode: str) -> dict[str, Any]:
    raise ValueError("Not that mode.")


@node("refuses", title="Refuses", description="For a test.", compute_outputs=_refuses)
def refuses(mode: str = "ok") -> dict[str, Any]:
    return {}


enrich = subgraph("enrich", title="Enrich", description="For a test.")(make_graph(upper, tail))
inner = subgraph("inner", title="Inner", description="For a test.")(GraphRecord.model_validate({"nodes": {"up": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "up", "output": "result"}}}))
top = subgraph("top", title="Top", description="For a test.")(GraphRecord.model_validate({"nodes": {"inner": {"node": "inner", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "inner", "output": "result"}}}))
defaulted = subgraph("defaulted", title="Defaulted", description="For a test.")(GraphRecord.model_validate({"nodes": {"g": {"node": "greet", "inputs": {"text": {"graph_input": "text", "default": "inside"}}}}, "outputs": {"result": {"node": "g", "output": "result"}}}))
@node("each-person", title="Each person", description="For a test.")
def each_person(table: Table[Person]) -> Series[Person]:
    return table.to_dicts()


team = subgraph("team", title="Team", description="For a test.")(make_graph(people))
wrapped = subgraph("wrapped", title="Wrapped", description="For a test.", deprecation=Deprecation(header="Gone"))(make_graph(upper))

REGISTRY = registry_of(
    upper, tail, join, fan, both_ints, maybe, greet, people, each_person, styled, name_of, nonempty, two, columns, anything, height, refuses,
    enrich, inner, top, defaulted, team, wrapped,
)


def _problems(record: GraphRecord | dict[str, Any]) -> list[tuple[str, tuple[Any, ...]]]:
    return [(p.type, p.loc) for p in Graph.compile(GraphRecord.model_validate(record), REGISTRY).problems]


def _refused(graph: Any, *args: Any, **kwargs: Any) -> list[tuple[str, tuple[Any, ...]]]:
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(*args, **kwargs)
    return [(p.type, p.loc) for p in refused.value.problems]


# -- the record ------------------------------------------------------------------------------------


def test_a_node_record_is_at_version_1_unless_it_says_and_is_frozen():
    record = NodeRecord(node="upper")
    assert record.version == 1
    with pytest.raises(ValidationError):
        record.version = 2  # type: ignore[misc]


def test_a_node_record_without_the_node_it_uses_is_refused():
    with pytest.raises(ValidationError):
        NodeRecord.model_validate({"version": 1})


def test_a_graph_record_with_a_misspelled_key_is_refused_naming_it():
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": {"u": {"node": "upper", "valuse": {"text": "x"}}}}), REGISTRY)
    assert [(p.type, p.loc, p.ctx["error"]) for p in refused.value.problems] == [("invalid_record", ("record", "nodes", "u", "valuse"), "extra_forbidden")]


@pytest.mark.parametrize(("record", "loc", "error"), [
    ('{"nodes": [', ("record",), "json_invalid"),
    (b'{"nodes": \xff}', ("record",), "json_invalid"),
    ('{"nodes": ' + "1" * 5000 + "}", ("record",), "json_invalid"),
    (json.dumps({"nodes": {"u": {"type": "upper"}}}), ("record", "nodes", "u", "node"), "missing"),
    (json.dumps({"nodes": [{"id": "u", "node": "upper"}]}), ("record", "nodes"), "dict_type"),
    (json.dumps({"nodes": {"u": {"node": "upper", "inputs": {"text": {"constant": "a", "graph_input": "text"}}}}}), ("record", "nodes", "u", "inputs", "text"), "one_of_constant_graph_input_sources"),
    (json.dumps({"nodes": {"u": {"node": "upper"}}, "outputs": {"result": "u.result"}}), ("record", "outputs", "result"), "dataclass_type"),
])
def test_what_is_not_a_record_is_refused_as_invalid_record_never_a_validation_error(record, loc, error):
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(record, REGISTRY)
    problems = refused.value.problems
    assert all(p.type == "invalid_record" for p in problems) and (problems[0].loc, problems[0].ctx["error"]) == (loc, error)
    assert not isinstance(refused.value, ValidationError) and problems[0].msg.startswith("The record is not a graph record at")


def test_a_record_as_plain_data_compiles_as_its_json_does():
    as_a_dict = {"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}
    from_data = Graph.compile(as_a_dict, REGISTRY)  # type: ignore[arg-type]
    from_text = Graph.compile(json.dumps(as_a_dict), REGISTRY)
    assert from_data.record == from_text.record and from_data.problems == from_text.problems == []


def test_a_half_finished_graph_record_compiles_with_its_problems():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "x": {"node": "nothing"}, "y": {"node": "tail", "inputs": {"text": {"sources": [{"node": "ghost", "output": "result"}]}}}}, "outputs": {}}), REGISTRY)
    assert {p.type for p in graph.problems} == {"unknown_node", "edge_names_missing_node"}


def test_a_records_nodes_are_keyed_by_their_ids():
    record = GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "u", "output": "result"}}})
    assert list(record.nodes) == ["u"] and record.nodes["u"] == NodeRecord(node="upper", inputs={"text": InputEntry(graph_input="text")})


def test_what_a_host_saves_reads_back_as_it_was_written():
    saved = [
        NodeRecord(node="upper", version=2, inputs={"text": InputEntry(constant="x")}, display={"x": 3}),
        GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "t": {"node": "tail", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}),
        Problem(type="cycle", loc=("a",), msg="The node is part of a cycle, so the graph cannot run.", ctx={}),
        Deprecation(header="Gone", alternative="Use upper."),
        Choice(id="a", title="A"),
        Textarea(),
    ]
    for model in saved:
        assert type(model).model_validate_json(model.model_dump_json()) == model, model


def test_compiling_leaves_the_record_it_was_given_unchanged():
    record = GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "t": {"node": "tail", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "t", "output": "result"}}})
    before = record.model_dump_json()
    Graph.compile(record, REGISTRY)
    assert record.model_dump_json() == before


def test_compiling_one_record_twice_gives_the_same_graph():
    record = {"nodes": {"s": {"node": "fan"}, "n": {"node": "nonempty", "inputs": {"text": {"constant": ""}}}, "t": {"node": "tail", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"n_result": {"node": "n", "output": "result"}, "s_result": {"node": "s", "output": "result"}, "t_result": {"node": "t", "output": "result"}}}
    first, second = Graph.compile(GraphRecord.model_validate(record), REGISTRY), Graph.compile(GraphRecord.model_validate(record), REGISTRY)
    assert first.problems == second.problems and first.describe() == second.describe()
    good = {"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "t": {"node": "tail", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}
    first, second = Graph.compile(GraphRecord.model_validate(good), REGISTRY), Graph.compile(GraphRecord.model_validate(good), REGISTRY)
    assert first.describe() == second.describe() and list(first.inputs) == list(second.inputs) and first.outputs == second.outputs


# -- constants ------------------------------------------------------------------------------------


def test_a_constant_is_validated_once_when_the_graph_compiles_and_not_again_per_item():
    reads = []

    @host_type("Counted", title="Counted")
    class Counted(str):
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            def read(value: str) -> "Counted":
                reads.append(value)
                return cls(value)

            return core_schema.no_info_plain_validator_function(read)

    @node("use", title="Use", description="For a test.")
    def use(item: int, fixed: Counted) -> str:
        return f"{item}{fixed}"

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "fan"}, "u": {"node": "use", "inputs": {"fixed": {"constant": "x"}, "item": {"sources": [{"node": "f", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), registry_of(fan, use))
    assert graph.problems == [] and reads == ["x"]
    assert graph.run().outputs == {"result": ["1x", "2x", "3x"]}
    assert reads == ["x"]


def test_a_none_constant_is_refused_unless_the_input_takes_none():
    assert _problems({"nodes": {"u": {"node": "upper", "inputs": {"text": {"constant": None}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}) == [("invalid_value", ("u", "text"))]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"m": {"node": "maybe", "inputs": {"text": {"constant": None}}}}, "outputs": {"result": {"node": "m", "output": "result"}}}), REGISTRY)
    assert graph.problems == [] and graph.run().outputs == {"result": "none"}


def test_a_constant_its_types_own_validator_refuses_is_a_problem_on_its_input():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"n": {"node": "nonempty", "inputs": {"text": {"constant": ""}}}}, "outputs": {"result": {"node": "n", "output": "result"}}}), REGISTRY)
    [problem] = graph.problems
    assert (problem.type, problem.loc) == ("invalid_value", ("n", "text"))
    assert "empty" not in problem.msg  # the type's own error text is left out of the message: it may quote the value


def test_a_problem_of_a_refused_value_equals_what_its_own_json_reads_back():
    @node("total", title="Total", description="For a test.")
    def total(numbers: list[int]) -> int:
        return sum(numbers)

    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"t": {"node": "total", "inputs": {"numbers": {"constant": [1, "a"]}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}), registry_of(total)).problems

    assert (problem.type, problem.ctx["errors"][0]["loc"]) == ("invalid_value", [1])
    assert Problem.model_validate_json(problem.model_dump_json()) == problem


def test_a_single_value_as_the_constant_of_a_series_input_is_refused_not_wrapped_into_a_list():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"j": {"node": "join", "inputs": {"texts": {"constant": "a"}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}), REGISTRY).problems
    assert (problem.type, problem.loc) == ("invalid_value", ("j", "texts"))
    assert problem.ctx["reason"] == "Input should be a valid list"


# -- what the graph offers -------------------------------------------------------------------------


def test_a_graph_input_carries_its_title_type_and_widget_for_a_form():
    described = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "styled", "inputs": {"mode": {"graph_input": "mode"}}}}, "outputs": {"result": {"node": "s", "output": "result"}}}), REGISTRY).inputs["mode"]
    assert described.title == "Mode" and described.type == "str" and described.required is True
    assert described.choices == (Choice("a", "A"), Choice("b", "B"))


def test_graph_inputs_come_in_the_order_of_the_nodes_by_id_and_never_by_name():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "styled", "inputs": {"mode": {"graph_input": "mode"}}}, "u": {"node": "upper", "inputs": {"text": {"graph_input": "alpha"}}}, "f": {"node": "fan"}, "g": {"node": "greet", "inputs": {"text": {"graph_input": "zed"}}}}, "outputs": {"f_result": {"node": "f", "output": "result"}, "g_result": {"node": "g", "output": "result"}, "s_result": {"node": "s", "output": "result"}, "u_result": {"node": "u", "output": "result"}}}), REGISTRY)
    assert list(graph.inputs) == ["zed", "mode", "alpha"]


def test_the_graph_outputs_are_the_outputs_the_record_declares_and_no_other():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"w": {"node": "two", "inputs": {"text": {"graph_input": "text"}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "w", "output": "zed"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), REGISTRY)
    assert graph.outputs == ["result"] and graph.run(text="a").outputs == {"result": "ZED A"}


def test_an_output_a_hook_computed_is_a_graph_output_with_its_title():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"c": {"node": "columns", "inputs": {"names": {"constant": ["net", "vat"]}, "text": {"graph_input": "text"}}}}, "outputs": {"net": {"node": "c", "output": "net"}, "vat": {"node": "c", "output": "vat"}}}), REGISTRY)
    assert graph.problems == [] and graph.outputs == ["net", "vat"]
    assert [o.title for o in graph.describe().nodes["c"].outputs] == ["Net", "Vat"]
    assert graph.run(text="9").outputs == {"net": "9", "vat": "9"}


def test_a_graph_input_and_a_graph_output_are_each_named_by_the_record():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"g": {"node": "greet", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "g", "output": "result"}}}), REGISTRY)
    assert list(graph.inputs) == ["text"] and graph.run(text="hi").outputs == {"result": "hi"}
    record = {"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "t": {"node": "two", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"zed": {"node": "t", "output": "zed"}, "alpha": {"node": "t", "output": "alpha"}}}
    assert Graph.compile(GraphRecord.model_validate(record), REGISTRY).run(text="a").outputs == {"zed": "zed A", "alpha": "alpha A"}


def test_a_description_of_a_broken_graph_is_json_and_carries_its_problems():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"n": {"node": "nonempty", "inputs": {"text": {"constant": ""}}}, "u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"n_result": {"node": "n", "output": "result"}, "u_result": {"node": "u", "output": "result"}}}), REGISTRY)
    described = json.loads(graph.describe().model_dump_json())
    assert not graph.is_runnable
    assert [(p["type"], p["loc"]) for p in described["problems"]] == [("invalid_value", ["n", "text"])]
    assert list(described["inputs"]) == ["text"]


# -- the problems compiling reports ----------------------------------------------------------------


def test_nodes_reading_each_other_are_a_cycle_on_each_and_the_graph_does_not_run():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}, "b": {"node": "tail", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}}}}, "outputs": {}}), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert not graph.is_runnable
    assert [p.type for p in _refused_run(graph)] == ["cycle", "cycle"]


def _refused_run(graph: Any) -> list[Problem]:
    with pytest.raises(InvalidCallError) as refused:
        graph.run()
    return refused.value.problems


def test_an_unknown_node_is_one_problem_naming_the_nodes_there_are_and_adds_no_inputs_or_outputs():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"x": {"node": "uper"}, "t": {"node": "tail", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}), registry_of(upper, tail))
    [problem] = graph.problems
    assert (problem.type, problem.loc) == ("unknown_node", ("x",))
    assert problem.ctx == {"node": "uper", "choices": ["tail", "upper"]}
    assert list(graph.inputs) == ["text"] and graph.outputs == ["result"]


def test_a_version_the_node_does_not_have_is_a_problem_never_the_nearest_version():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "version": 2}}, "outputs": {}}), REGISTRY).problems
    assert (problem.type, problem.loc, problem.ctx) == ("unknown_version", ("u",), {"node": "upper", "version": 2, "newest": 1})


def test_a_hook_that_raises_is_a_problem_on_its_node_never_an_exception():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"r": {"node": "refuses"}}, "outputs": {}}), REGISTRY).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("r",), "ValueError")


def test_a_constant_its_type_refuses_never_reaches_a_hook_and_is_one_problem():
    # "refuses" reads its mode in a hook, which a list given as the constant of a str input would otherwise reach
    problems = Graph.compile(GraphRecord.model_validate({"nodes": {"r": {"node": "refuses", "inputs": {"mode": {"constant": ["a", "b"]}}}}, "outputs": {}}), REGISTRY).problems
    assert [(p.type, p.loc) for p in problems] == [("invalid_value", ("r", "mode"))]


def test_a_subtype_output_fits_an_input_of_its_parent_type_and_not_the_other_way():
    assert _problems({"nodes": {"n": {"node": "name", "inputs": {"text": {"graph_input": "text"}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "n", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}) == []
    assert _problems({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "n": {"node": "nonempty", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "n", "output": "result"}}}) == [
        ("type_mismatch", ("n", "text")),
    ]


def test_a_series_of_any_an_output_hook_declares_compiles_into_a_str_input_and_a_table_input():
    into_text = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "anything"}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "items"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), REGISTRY)
    assert into_text.problems == [] and into_text.run().outputs == {"result": ["X", "X"]}
    assert _problems({"nodes": {"a": {"node": "anything"}, "h": {"node": "height", "inputs": {"rows": {"sources": [{"node": "a", "output": "items"}]}}}}}) == []


def test_one_mistyped_edge_among_several_into_a_series_input_is_a_type_mismatch():
    record = {"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "f": {"node": "fan"}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "u", "output": "result"}, {"node": "f", "output": "result"}]}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}
    assert _problems(record) == [("type_mismatch", ("j", "texts"))]


def test_an_edge_from_another_nodes_input_is_not_a_source():
    problems = Graph.compile(GraphRecord.model_validate({"nodes": {"h": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "u": {"node": "tail", "inputs": {"text": {"sources": [{"node": "h", "output": "text"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), REGISTRY).problems
    assert [(p.type, p.loc, p.ctx["source"]) for p in problems] == [("unknown_output", ("u", "text"), "h.text")]


def test_an_edge_naming_a_field_a_record_lacks_lists_every_field_whatever_its_name():
    Hidden = TypedDict("Hidden", {".hidden": str, "b": str})

    @node("give_hidden", title="Give", description="For a test.")
    def give_hidden() -> Hidden:
        return {".hidden": "x", "b": "y"}

    record = {"nodes": {"g": {"node": "give_hidden"}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "g", "output": "result", "field": "zz"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}
    (found,) = Graph.compile(GraphRecord.model_validate(record), registry_of(give_hidden, upper)).problems
    assert (found.type, found.loc, found.ctx["outputs"]) == ("unknown_output", ("u", "text"), ["g.result..hidden", "g.result.b"])


def test_a_broken_edge_is_one_problem_not_repeated_on_the_nodes_downstream():
    record = {"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "nope"}]}}}, "c": {"node": "tail", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}
    assert _problems(record) == [("unknown_output", ("b", "text"))]


def test_an_edge_from_a_node_that_did_not_compile_is_not_reported_again():
    unknown = {"nodes": {"x": {"node": "gone"}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "x", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}
    assert _problems(unknown) == [("unknown_node", ("x",))]
    missing = {"nodes": {"u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "x", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}
    assert _problems(missing) == [("edge_names_missing_node", ("u", "text"))]


def test_a_node_id_or_a_source_that_is_no_name_is_refused_at_the_record_door():
    """``a@b`` and ``a/b`` can be no node's id, so a node under either and an edge from either are refused."""
    for bad in ("a@b", "a/b"):
        with pytest.raises(InvalidCallError) as refused:
            Graph.compile(json.dumps({"nodes": {bad: {"node": "upper"}, "u": {"node": "tail"}}}), REGISTRY)
        assert [(p.type, p.loc) for p in refused.value.problems] == [("invalid_record", ("record", "nodes", bad, "[key]"))]
        with pytest.raises(InvalidCallError) as refused:
            Graph.compile(json.dumps({"nodes": {"u": {"node": "tail", "inputs": {"text": {"sources": [{"node": bad, "output": "result"}]}}}}}), REGISTRY)
        problem = refused.value.problems[0]
        assert (problem.type, problem.loc[:5], problem.loc[-1]) == ("invalid_record", ("record", "nodes", "u", "inputs", "text"), "node")


# -- what a caller gives ---------------------------------------------------------------------------


def test_with_inputs_takes_a_mapping_by_name_beside_keywords_and_refuses_a_name_it_does_not_offer():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "a_text"}}}, "b": {"node": "tail", "inputs": {"text": {"graph_input": "b_text"}}}, "s": {"node": "styled", "inputs": {"mode": {"graph_input": "mode"}}}}, "outputs": {"a_result": {"node": "a", "output": "result"}, "b_result": {"node": "b", "output": "result"}, "s_result": {"node": "s", "output": "result"}}}), REGISTRY)
    assert graph.with_inputs({"a_text": "x", "b_text": "y"}, mode="b").resume().outputs == {"a_result": "X", "b_result": "y.", "s_result": "b"}
    assert _refused(graph, {"a_text": "x", "b_text": "y", "c_text": "z"}, mode="b") == [("not_offered", ("inputs", "c_text"))]


def test_two_inputs_naming_one_graph_input_each_get_the_callers_one_value():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "b": {"node": "tail", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"a_result": {"node": "a", "output": "result"}, "b_result": {"node": "b", "output": "result"}}}), REGISTRY)
    assert graph.problems == [] and list(graph.inputs) == ["text"]
    assert graph.run(text="x").outputs == {"a_result": "X", "b_result": "x."}


@pytest.mark.parametrize(("a", "b"), [
    ({"node": "upper", "inputs": {"text": {"graph_input": "x"}}}, {"node": "both_ints", "inputs": {"a": {"graph_input": "x"}, "b": {"constant": 1}}}),
    ({"node": "greet", "inputs": {"text": {"graph_input": "x", "default": "hi"}}}, {"node": "greet", "inputs": {"text": {"graph_input": "x", "default": "yo"}}}),
], ids=["types", "defaults"])
def test_two_inputs_naming_one_graph_input_with_different_types_or_defaults_is_graph_input_used_two_ways(a, b):
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": a, "b": b}}), REGISTRY)
    [problem] = graph.problems
    assert (problem.type, problem.loc[0], problem.ctx["graph_input"]) == ("graph_input_used_two_ways", "b", "x")
    assert not graph.is_runnable


def test_with_inputs_refuses_every_problem_at_once_listing_what_it_offers():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "styled", "inputs": {"mode": {"graph_input": "mode"}}}, "u": {"node": "upper", "inputs": {"text": {"constant": "fixed"}}}}, "outputs": {"s_result": {"node": "s", "output": "result"}, "u_result": {"node": "u", "output": "result"}}}), REGISTRY)
    assert list(graph.inputs) == ["mode"]
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(mode=5, text="a fixed input is not offered", typo="x")
    found = sorted((p.type, p.loc) for p in refused.value.problems)
    assert found == [("invalid_input", ("inputs", "mode")), ("not_offered", ("inputs", "text")), ("not_offered", ("inputs", "typo"))]
    assert [p.ctx["inputs"] for p in refused.value.problems if p.type == "not_offered"] == [["mode"], ["mode"]]


# -- subgraphs -------------------------------------------------------------------------------


def test_a_subgraph_node_offers_its_inner_graph_input_and_output_as_its_own():
    assert enrich.title == "Enrich"
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), REGISTRY)
    assert graph.problems == [] and list(graph.inputs) == ["text"] and graph.inputs["text"].title == "Text"
    run = graph.run(text="hi")
    assert graph.outputs == ["result"] and run.outputs == {"result": "HI."} and run.results["e"] == {"result": "HI."}


def test_a_constant_on_a_subgraph_node_reaches_the_inner_input_it_offers():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"constant": "hi"}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), REGISTRY)
    assert graph.problems == [] and graph.inputs == {} and graph.run().outputs == {"result": "HI."}


def test_a_subgraph_node_given_nothing_keeps_its_inner_defaults_and_runs_once():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "defaulted"}}, "outputs": {"result": {"node": "d", "output": "result"}}}), REGISTRY)
    assert graph.problems == [] and graph.inputs == {} and {k: v.per_item_of for k, v in graph.describe().nodes.items()} == {"d": None}
    assert graph.run().outputs == {"result": "inside"}


def test_a_graph_input_on_a_subgraph_node_is_required_unless_the_record_gives_it_a_default():
    """The subgraph's own default for the input plays no part: the graph input's default is the outer record's."""
    assert Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "defaulted", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "d", "output": "result"}}}), REGISTRY).inputs["text"].required is True
    assert Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "defaulted", "inputs": {"text": {"graph_input": "text", "default": "x"}}}}, "outputs": {"result": {"node": "d", "output": "result"}}}), REGISTRY).inputs["text"].required is False
    assert Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), REGISTRY).inputs["text"].required is True


def test_a_subgraph_inside_a_subgraph_is_one_node_from_outside():
    run = Graph.compile(GraphRecord.model_validate({"nodes": {"t": {"node": "top", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}), REGISTRY).run(text="hi")
    assert run.outputs == {"result": "HI"} and run.results == {"t": {"result": "HI"}}


def test_renaming_an_inner_node_keeps_the_subgraphs_input_and_output_names():
    renamed = subgraph("enrich", title="Enrich", description="For a test.")(GraphRecord.model_validate({
        "nodes": {"shout": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "end": {"node": "tail", "inputs": {"text": {"sources": [{"node": "shout", "output": "result"}]}}}},
        "outputs": {"result": {"node": "end", "output": "result"}},
    }))
    record = {"nodes": {"g": {"node": "greet", "inputs": {"text": {"graph_input": "text"}}}, "e": {"node": "enrich", "inputs": {"text": {"sources": [{"node": "g", "output": "result"}]}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}
    for definition in (enrich, renamed):
        graph = Graph.compile(GraphRecord.model_validate(record), registry_of(greet, upper, tail, definition))
        assert graph.problems == [] and graph.run(text="hi").outputs == {"result": "HI."}


def test_a_constant_for_an_input_the_subgraph_does_not_offer_is_unknown_input():
    named_like_an_output = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"constant": "a"}, "result": {"constant": "b"}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), REGISTRY).problems
    assert [(p.type, p.loc) for p in named_like_an_output] == [("unknown_input", ("e", "result"))]
    with pytest.raises(InvalidCallError) as by_inner_source:
        Graph.compile(json.dumps({"nodes": {"e": {"node": "enrich", "inputs": {"upper.text": {"constant": "a"}}}}}), REGISTRY)
    assert [(p.type, p.loc) for p in by_inner_source.value.problems] == [("invalid_record", ("record", "nodes", "e", "inputs", "upper.text", "[key]"))]


def test_a_table_returned_inside_a_subgraph_is_offered_whole_as_one_value():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"t": {"node": "team", "inputs": {"text": {"graph_input": "text"}}}, "p": {"node": "each-person", "inputs": {"table": {"sources": [{"node": "t", "output": "result"}]}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "p", "output": "result", "field": "email"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), REGISTRY)
    assert graph.problems == []
    assert graph.run(text="name,email\nIda,ida@a.dk\nBo,bo@b.dk\n").outputs == {"result": ["IDA@A.DK", "BO@B.DK"]}


def test_a_subgraph_offers_its_inner_inputs_and_carries_its_deprecation():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"w": {"node": "wrapped", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "w", "output": "result"}}}), REGISTRY)
    assert list(graph.inputs) == ["text"] and graph.run(text="a").outputs == {"result": "A"}
    assert REGISTRY.describe().nodes[[n.id for n in REGISTRY].index("wrapped")].deprecation.header == "Gone"


@node("pair", title="Pair", description="For a test.")
def pair(a: str, b: str) -> str:
    return a + b


def _hint_for(formula: str) -> dict[str, Any]:
    return {name: float for name in formula.split()}


@node("formula", title="Formula", description="For a test.", compute_added_inputs=_hint_for)
def formula(formula: str, **values: float) -> float:
    return sum(values.values())


gone_inside = subgraph("gone_inside", title="Gone inside", description="For a test.")(GraphRecord.model_validate({"nodes": {"x": {"node": "gone"}}, "outputs": {}}))
around = subgraph("around", title="Around", description="For a test.")(GraphRecord.model_validate({"nodes": {"b": {"node": "gone_inside"}}, "outputs": {}}))
invalid_inside = subgraph("invalid_inside", title="Invalid inside", description="For a test.")(GraphRecord.model_validate({"nodes": {"n": {"node": "nonempty", "inputs": {"text": {"constant": ""}}}}, "outputs": {"result": {"node": "n", "output": "result"}}}))
unfed_inside = subgraph("unfed_inside", title="Unfed inside", description="For a test.")(GraphRecord.model_validate({"nodes": {
    "upper": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
    "pair": {"node": "pair", "inputs": {"a": {"sources": [{"node": "upper", "output": "result"}]}}},
}, "outputs": {"result": {"node": "pair", "output": "result"}}}))
BROKEN = registry_of(upper, pair, nonempty, formula, gone_inside, around, invalid_inside, unfed_inside)


def test_a_hook_is_not_called_without_a_required_input_and_the_problem_says_it_needs_a_value():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "formula"}}, "outputs": {}}), BROKEN)
    assert [(p.type, p.loc) for p in graph.problems] == [("hook_needs_value", ("f", "formula"))]
    given = Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "formula", "inputs": {"formula": {"constant": "a b"}, "a": {"graph_input": "a"}, "b": {"graph_input": "b"}}}}, "outputs": {"result": {"node": "f", "output": "result"}}}), BROKEN)
    assert given.problems == [] and list(given.inputs) == ["a", "b"]


def test_an_input_a_hook_reads_is_a_constant_of_the_graph_never_a_graph_input():
    """The outputs a hook works out are the same for every run, so the input it reads comes from the graph record,
    never from the caller."""

    def outputs(mode: str) -> dict[str, Any]:
        return {"a": int} if mode == "a" else {"b": str}

    @node("by-mode", title="By mode", description="For a test.", compute_outputs=outputs)
    def by_mode(mode: str = "a") -> dict[str, Any]:
        return {"a": 1} if mode == "a" else {"b": "x"}

    registry = registry_of(by_mode)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"m": {"node": "by-mode"}}, "outputs": {"a": {"node": "m", "output": "a"}}}), registry)
    assert graph.problems == [] and graph.inputs == {} and graph.run().outputs == {"a": 1}
    with pytest.raises(InvalidCallError):
        graph.run(mode="b")

    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"m": {"node": "by-mode", "inputs": {"mode": {"graph_input": "mode", "default": "b"}}}}}), registry).problems
    assert (problem.type, problem.loc) == ("hook_needs_value", ("m", "mode"))
    assert Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "formula"}}, "outputs": {}}), BROKEN).problems[0].type == "hook_needs_value"


def test_a_problem_inside_a_subgraph_is_reported_at_its_node_path():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"g": {"node": "gone_inside"}}, "outputs": {}}), BROKEN)
    assert [(p.type, p.loc) for p in graph.problems] == [("unknown_node", ("g", "x"))]
    assert not graph.is_runnable and graph.inputs == {}


def test_a_problem_two_subgraphs_down_is_reported_at_its_full_node_path():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "around"}}, "outputs": {}}), BROKEN).problems
    assert (problem.type, problem.loc) == ("unknown_node", ("a", "b", "x"))


def test_an_invalid_constant_inside_a_subgraph_makes_the_outer_graph_not_runnable():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"i": {"node": "invalid_inside"}}, "outputs": {}}), BROKEN)
    assert [(p.type, p.loc) for p in graph.problems] == [("invalid_value", ("i", "n", "text"))]
    with pytest.raises(InvalidCallError):
        graph.run()


def test_a_subgraph_whose_inner_node_has_a_required_input_with_no_entry_is_input_has_no_value():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "unfed_inside", "inputs": {"text": {"constant": "x"}}}}, "outputs": {}}), BROKEN).problems
    assert (problem.type, problem.loc) == ("input_has_no_value", ("u", "pair", "b"))


def test_a_constant_on_a_broken_subgraph_node_adds_no_second_problem():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"g": {"node": "gone_inside", "inputs": {"text": {"constant": "x"}}}}, "outputs": {}}), BROKEN)
    assert [(p.type, p.loc) for p in graph.problems] == [("unknown_node", ("g", "x"))]


@node("both_halves", title="Both halves", description="For a test.")
def both_halves(a: str, b: str) -> str:
    return a + b


def _taken(formula: str) -> dict[str, Any]:
    return {"formula": float}


@node("taken", title="Taken", description="For a test.", compute_added_inputs=_taken)
def taken(formula: str = "", **values: float) -> float:
    return 0.0


def _refuses_bad(mode: str) -> dict[str, Any]:
    if mode == "bad":
        raise ValueError("Not that mode.")
    return {"result": str}


@node("refuses_bad", title="Refuses bad", description="For a test.", compute_outputs=_refuses_bad)
def refuses_bad(mode: str = "ok") -> dict[str, Any]:
    return {"result": mode}


halves = subgraph("halves", title="Halves", description="For a test.")(GraphRecord.model_validate({"nodes": {"x": {"node": "upper", "inputs": {"text": {"graph_input": "x_text"}}}, "y": {"node": "upper", "inputs": {"text": {"graph_input": "y_text"}}}, "b": {"node": "both_halves", "inputs": {"a": {"sources": [{"node": "x", "output": "result"}]}, "b": {"sources": [{"node": "y", "output": "result"}]}}}}, "outputs": {"result": {"node": "b", "output": "result"}}}))
picky = subgraph("picky", title="Picky", description="For a test.")(make_graph(refuses_bad, outputs={"result": "refuses_bad.result"}))
OUTSIDE = registry_of(upper, tail, refuses_bad, both_halves, taken, enrich, halves, picky)


def test_only_the_nodes_on_a_cycle_are_a_cycle_and_a_node_downstream_is_still_described():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}, "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}}}, "c": {"node": "tail", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert set(graph.describe().nodes) == {"a", "b", "c"}


def test_describe_lists_every_node_of_a_half_broken_graph_and_which_definitions_were_found():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "nope"}]}}}, "c": {"node": "tail", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}, "x": {"node": "gone"}}, "outputs": {"result": {"node": "c", "output": "result"}}}), REGISTRY)
    described_nodes = json.loads(graph.describe().model_dump_json())["nodes"]
    assert list(described_nodes) == ["a", "b", "c", "x"]
    assert {node_id: d["found"] for node_id, d in described_nodes.items()} == {"a": True, "b": True, "c": True, "x": False}


def _described(graph: Any, node_id: str) -> tuple[bool, str | None, dict[str, str], list[str], list[str]]:
    """A node as ``describe()`` gives it to an editor: whether its definition was found, which loop it is inside,
    where each input gets its value, its outputs, and its problems."""
    described = graph.describe().nodes[node_id]
    return (
        described.found, described.per_item_of, {i.field.name: i.source.mode for i in described.inputs},
        [o.name for o in described.outputs], [p.type for p in described.problems],
    )


def test_a_node_with_a_refused_edge_is_described_with_its_inputs_and_so_is_each_node_downstream():
    """The node with the problem keeps its inputs and outputs, the refused input marked missing; the nodes downstream
    are compiled as in a graph with no problem."""
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"f": {"node": "fan"}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "f", "output": "result"}]}}}, "t": {"node": "tail", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "t", "output": "result"}]}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("type_mismatch", ("u", "text"))]
    assert _described(graph, "u") == (True, None, {"text": "missing"}, ["result"], ["type_mismatch"])
    assert _described(graph, "t") == (True, None, {"text": "one_value"}, ["result"], [])
    assert _described(graph, "j") == (True, None, {"texts": "collected"}, ["result"], [])
    assert graph.describe().nodes["f"].found and not graph.is_runnable
    with pytest.raises(InvalidCallError) as refused:
        graph.run()
    assert [p.type for p in refused.value.problems] == ["type_mismatch"]


def test_the_nodes_on_a_cycle_and_a_node_downstream_are_described_with_their_inputs():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"c": {"node": "tail", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}, "a": {"node": "upper", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}, "b": {"node": "upper", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert _described(graph, "a") == (True, None, {"text": "missing"}, ["result"], ["cycle"])  # the edge that closes the cycle gives it no value
    assert _described(graph, "b") == (True, None, {"text": "one_value"}, ["result"], ["cycle"])
    assert _described(graph, "c") == (True, None, {"text": "one_value"}, ["result"], [])


def test_an_input_looping_over_a_series_not_nested_with_the_others_is_marked_missing_and_the_node_described():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"x": {"node": "fan"}, "y": {"node": "fan"}, "b": {"node": "both_ints", "inputs": {"a": {"sources": [{"node": "x", "output": "result"}]}, "b": {"sources": [{"node": "y", "output": "result"}]}}}, "t": {"node": "tail", "inputs": {"text": {"sources": [{"node": "b", "output": "result"}]}}}}, "outputs": {"result": {"node": "t", "output": "result"}}}), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("series_not_nested", ("b",))]
    assert _described(graph, "b") == (True, "x.result", {"a": "item", "b": "missing"}, ["result"], ["series_not_nested"])
    assert _described(graph, "t") == (True, "x.result", {"text": "item"}, ["result"], [])


def test_a_node_not_found_or_whose_hook_failed_has_no_inputs_or_outputs_and_the_nodes_downstream_keep_theirs():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"gone": {"node": "gone"}, "old": {"node": "upper", "version": 9}, "r": {"node": "refuses", "inputs": {"mode": {"constant": "bad"}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "gone", "output": "result"}]}}}, "t": {"node": "tail", "inputs": {"text": {"sources": [{"node": "old", "output": "result"}]}}}, "v": {"node": "upper", "inputs": {"text": {"sources": [{"node": "r", "output": "result"}]}}}}, "outputs": {"t_result": {"node": "t", "output": "result"}, "u_result": {"node": "u", "output": "result"}, "v_result": {"node": "v", "output": "result"}}}), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("unknown_node", ("gone",)), ("unknown_version", ("old",)), ("compute_hook_failed", ("r",))]
    for unknown in ("gone", "old", "r"):
        assert _described(graph, unknown)[:4] == (False, None, {}, [])
    for downstream in ("u", "t", "v"):
        assert _described(graph, downstream) == (True, None, {"text": "missing"}, ["result"], [])


def test_an_input_a_hook_adds_under_a_name_the_node_declares_is_the_hooks_failure():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "taken", "inputs": {"formula": {"constant": "x"}}}}, "outputs": {}}), OUTSIDE).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("d",), "added_input_taken")


def test_a_wrong_source_on_a_subgraph_node_is_unknown_output_or_unknown_input_on_it():
    def found(record: dict[str, Any]) -> list[tuple[str, tuple[Any, ...]]]:
        return [(p.type, p.loc) for p in Graph.compile(GraphRecord.model_validate(record), OUTSIDE).problems]

    def record(e_inputs: dict[str, Any], u_inputs: dict[str, Any]) -> dict[str, Any]:
        return {"nodes": {"e": {"node": "enrich", "inputs": e_inputs}, "u": {"node": "upper", "inputs": u_inputs}}}

    assert found(record({"text": {"constant": "a"}}, {"text": {"sources": [{"node": "e", "output": "nope"}]}})) == [("unknown_output", ("u", "text"))]
    assert found(record({"text": {"constant": "a"}}, {"text": {"sources": [{"node": "e", "output": "tail"}]}})) == [("unknown_output", ("u", "text"))]
    assert found(record({"text": {"constant": "a"}, "tail": {"sources": [{"node": "u", "output": "result"}]}}, {"text": {"constant": "a"}})) == [("unknown_input", ("e", "tail"))]


@pytest.mark.parametrize("ctx", [{"tables": [1, 2]}, {"source": "e/report.pdf", "node_id": "anything"}])
def test_an_inner_nodes_error_is_reported_at_its_node_path_with_the_ctx_its_author_wrote(ctx):
    @node("unreadable", title="Unreadable", description="For a test.")
    def unreadable(text: str) -> str:
        raise NodeError("The document could not be read.", code="unreadable", ctx=ctx)

    sub = subgraph("e", title="E", description="For a test.")(make_graph(unreadable))
    run = Graph.compile(make_graph(sub), registry_of(unreadable, sub)).run(text="hi")

    assert (run.status, run.cause.msg) == ("failed", "The document could not be read.")
    assert (run.cause.type, run.cause.loc, run.cause.ctx) == ("unreadable", ("e", "unreadable"), ctx)


def test_an_outer_edge_cannot_read_an_inner_node_by_its_node_path():
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": {"e": {"node": "enrich"}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "e/upper", "output": "result"}]}}}}}), OUTSIDE)
    [problem] = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("invalid_record", ("record", "nodes", "u", "inputs", "text", "sources", 0, "node"), "not_a_name")


def test_an_edge_from_a_missing_node_into_a_subgraph_node_is_a_problem_at_the_input_it_offers():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"sources": [{"node": "ghost", "output": "result"}]}}}}, "outputs": {"result": {"node": "e", "output": "result"}}}), OUTSIDE).problems
    assert (problem.type, problem.loc, problem.ctx["missing"]) == ("edge_names_missing_node", ("e", "text"), "ghost")
    assert (problem.ctx["source"], problem.ctx["target"]) == ("ghost.result", "e.text")


def test_an_inner_input_a_hook_reads_is_not_offered_so_a_constant_for_it_on_the_subgraph_node_is_unknown_input():
    assert Graph.compile(GraphRecord.model_validate({"nodes": {"p": {"node": "picky"}}, "outputs": {"result": {"node": "p", "output": "result"}}}), OUTSIDE).inputs == {}
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": {"p": {"node": "picky", "inputs": {"mode": {"constant": "bad"}}}}, "outputs": {"result": {"node": "p", "output": "result"}}}), OUTSIDE).problems
    assert (problem.type, problem.loc, problem.ctx["inputs"]) == ("unknown_input", ("p", "mode"), [])


def test_a_subgraph_nodes_inputs_are_given_by_the_graph_inputs_the_outer_record_names():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "inputs": {"text": {"graph_input": "e_text"}}}, "u": {"node": "upper", "inputs": {"text": {"graph_input": "u_text"}}}}, "outputs": {"e_result": {"node": "e", "output": "result"}, "u_result": {"node": "u", "output": "result"}}}), OUTSIDE)
    assert list(graph.inputs) == ["e_text", "u_text"]
    assert graph.run({"e_text": "hi", "u_text": "yo"}).outputs == {"e_result": "HI.", "u_result": "YO"}


def test_a_subgraphs_inputs_are_the_graph_inputs_its_record_names():
    assert [f.name for f in next(n for n in OUTSIDE.describe().nodes if n.id == "halves").inputs] == ["x_text", "y_text"]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "halves", "inputs": {"y_text": {"constant": "b"}, "x_text": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), OUTSIDE)
    assert graph.problems == [] and list(graph.inputs) == ["text"] and graph.run(text="a").outputs == {"result": "AB"}


def test_a_subgraph_whose_two_inner_inputs_name_one_graph_input_takes_one_value_that_feeds_both():
    both = subgraph("both", title="Both", description="For a test.")(GraphRecord.model_validate({
        "nodes": {
            "x": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
            "y": {"node": "tail", "inputs": {"text": {"graph_input": "text"}}},
            "b": {"node": "both_halves", "inputs": {"a": {"sources": [{"node": "x", "output": "result"}]}, "b": {"sources": [{"node": "y", "output": "result"}]}}},
        },
        "outputs": {"result": {"node": "b", "output": "result"}},
    }))
    registry = registry_of(upper, tail, both_halves, both)
    assert [f.name for f in next(n for n in registry.describe().nodes if n.id == "both").inputs] == ["text"]
    graph = Graph.compile(make_graph(upper, both, edges=[("upper.result", "both.text")], outputs={"result": "both.result"}), registry)
    assert graph.problems == [] and list(graph.inputs) == ["text"] and graph.run(text="a").outputs == {"result": "AA."}


def test_a_node_id_with_a_slash_beside_a_subgraph_node_is_refused_at_the_record_door():
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": {"e": {"node": "enrich"}, "e/upper": {"node": "upper"}}}), OUTSIDE)
    assert [(p.type, p.loc) for p in refused.value.problems] == [("invalid_record", ("record", "nodes", "e/upper", "[key]"))]


def test_a_set_given_for_a_series_input_is_refused_as_it_has_no_order():
    from conductor import NodeRegistry, make_graph, node

    @node("send", title="Send", description="For a test.")
    def send(contacts: Series[str]) -> str:
        return ",".join(contacts)

    registry = NodeRegistry()
    registry.add(send)
    graph = Graph.compile(make_graph(send), registry)
    with pytest.raises(InvalidCallError) as refused:
        graph.run(contacts={"b", "a"})
    assert "a set has no order" in refused.value.problems[0].msg
    assert graph.run(contacts=["b", "a"]).outputs == {"result": "b,a"}


def _nested_forty_deep() -> tuple[Any, str]:
    """A registry of forty subgraphs, each with the one before it as its only node, and the id of the outermost."""
    from conductor import NodeRegistry

    registry = NodeRegistry()

    @registry.node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    inner = "upper"
    for depth in range(40):
        registry.add(subgraph(f"nest-{depth}", title=f"Nest {depth}", description="For a test.")(GraphRecord.model_validate({"nodes": {"n": {"node": inner, "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "n", "output": "result"}}})))
        inner = f"nest-{depth}"
    return registry, inner


def test_subgraphs_nested_forty_deep_compile_in_under_two_seconds():
    """Each subgraph is compiled once per ``Graph.compile``, however deep it is nested, so the time grows in proportion
    to the depth."""
    import time

    registry, outermost = _nested_forty_deep()
    started = time.monotonic()
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"top": {"node": outermost, "inputs": {"text": {"graph_input": "text"}}}}}), registry)
    assert time.monotonic() - started < 2 and graph.problems == [] and list(graph.inputs) == ["text"]


def test_subgraphs_nested_forty_deep_run_as_the_one_function_node_inside_them():
    registry, outermost = _nested_forty_deep()
    record = GraphRecord.model_validate({"nodes": {"top": {"node": outermost, "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "top", "output": "result"}}})
    assert Graph.compile(record, registry).run(text="z").outputs == {"result": "Z"}


def test_describe_lists_each_nodes_problems_on_it():
    described = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "x": {"node": "uper"}}, "outputs": {"result": {"node": "u", "output": "result"}}}), REGISTRY).describe()
    assert [p.type for p in described.nodes["x"].problems] == ["unknown_node"] and described.nodes["u"].problems == []
    assert described.problems == described.nodes["x"].problems


def test_a_type_mismatch_into_a_series_says_the_series_it_reads():
    from conductor import NodeRegistry, Series, make_graph, node

    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str]:
        return text.split()

    @node("count", title="Count", description="For a test.")
    def count(flags: Series[bool]) -> int:
        return sum(flags)

    registry = NodeRegistry()
    registry.add(words, count)
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": {"w": {"node": "words", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "count", "inputs": {"flags": {"sources": [{"node": "w", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), registry).problems
    assert (problem.type, problem.ctx["got"], problem.ctx["expected"]) == ("type_mismatch", "Series[str]", "Series[bool]")
    assert [p.type for p in Graph.compile(make_graph(words, count), registry).problems] == ["type_mismatch"]


def test_a_constant_default_or_edge_on_a_subgraph_node_replaces_the_inner_default():
    from conductor import NodeRegistry, make_graph, node

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    @node("hello", title="Hello", description="For a test.")
    def hello() -> str:
        return "hi"

    registry = NodeRegistry()
    registry.add(upper, hello, subgraph("shout", title="Shout", description="For a test.")(make_graph(upper, defaults={"upper.text": "inner"})))
    shout = registry["shout"]
    assert Graph.compile(make_graph(shout), registry).run().outputs == {"result": "INNER"}
    with_constant = Graph.compile(make_graph(shout, constants={"shout.text": "outer"}), registry)
    assert with_constant.problems == [] and with_constant.run().outputs == {"result": "OUTER"}
    defaulted = Graph.compile(make_graph(shout, defaults={"shout.text": "outer"}), registry)
    assert defaulted.problems == [] and defaulted.run().outputs == {"result": "OUTER"}
    fed = Graph.compile(make_graph(hello, shout, edges=[("hello.result", "shout.text")]), registry)
    assert fed.problems == [] and fed.run().outputs == {"result": "HI"}


def test_a_bool_output_into_a_float_input_is_a_type_mismatch_and_a_union_is_named_by_its_members():
    from conductor import NodeRegistry, node

    @host_type("Text", title="Text")
    class Text(str):
        pass

    @node("flag", title="Flag", description="For a test.")
    def flag() -> bool:
        return True

    @node("maybe", title="Maybe", description="For a test.")
    def maybe() -> Text | None:
        return None

    @node("double", title="Double", description="For a test.")
    def double(x: float) -> float:
        return x * 2

    registry = NodeRegistry()
    registry.add(flag, maybe, double)
    for source, got in (("flag", "bool"), ("maybe", "Text | None")):
        (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": source}, "d": {"node": "double", "inputs": {"x": {"sources": [{"node": "s", "output": "result"}]}}}}}), registry).problems
        assert (problem.type, problem.ctx["got"]) == ("type_mismatch", got)


def test_a_graph_input_has_its_field_and_writes_as_json():
    import json

    from conductor import FieldDescription, NodeRegistry, make_graph, node

    @node("u", title="U", description="For a test.")
    def u(t: str, n: int = 1) -> str:
        return t

    registry = NodeRegistry()
    registry.add(u)
    graph = Graph.compile(make_graph(u, defaults={"u.n": 1}), registry)
    assert isinstance(graph.inputs["t"], FieldDescription) and graph.describe().inputs == graph.inputs
    assert json.loads(json.dumps({name: i.default for name, i in graph.inputs.items()})) == {"t": None, "n": 1}
    assert (graph.inputs["t"].name, graph.inputs["t"].required) == ("t", True)


def test_a_bool_as_a_constant_or_a_graph_input_is_refused_for_a_float_input():
    from conductor import NodeRegistry, node

    @node("double", title="Double", description="For a test.")
    def double(x: float) -> float:
        return x * 2

    registry = NodeRegistry()
    registry.add(double)
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "double", "inputs": {"x": {"constant": True}}}}, "outputs": {"result": {"node": "d", "output": "result"}}}), registry).problems
    assert (problem.type, problem.ctx["reason"]) == ("invalid_value", "a yes or no is not a number; give a number")
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "double", "inputs": {"x": {"graph_input": "x"}}}}, "outputs": {"result": {"node": "d", "output": "result"}}}), registry).run(x=True)
    assert refused.value.problems[0].type == "invalid_input"
    assert Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "double", "inputs": {"x": {"graph_input": "x"}}}}, "outputs": {"result": {"node": "d", "output": "result"}}}), registry).run(x=2).outputs == {"result": 4.0}


def test_a_table_wired_into_a_series_input_is_a_type_mismatch_naming_table():
    import polars as pl
    from conductor import NodeRegistry, Series, Table, node

    @node("sheet", title="Sheet", description="For a test.")
    def sheet() -> Table:
        return pl.DataFrame({"a": [1]})

    @node("count", title="Count", description="For a test.")
    def count(items: Series[int]) -> int:
        return len(items)

    registry = NodeRegistry()
    registry.add(sheet, count)
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "sheet"}, "c": {"node": "count", "inputs": {"items": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "c", "output": "result"}}}), registry).problems
    assert (problem.ctx["got"], problem.ctx["expected"]) == ("Table", "Series[int]")


def test_a_bool_is_refused_for_a_hosts_number_type_that_accepts_int():
    from conductor import Ask, NodeRegistry, make_graph

    @host_type("number", title="Number", accepts=[int])
    class Number(float):
        pass


    registry = NodeRegistry(types=[Number, bool])

    @registry.node("yes", title="Yes", description="For a test.")
    def yes(value: bool = True) -> bool:
        return value

    @registry.node("half", title="Half", description="For a test.")
    def half(value: Number) -> Number:
        return Number(value / 2)

    @registry.node("ask_number", title="Ask number", description="For a test.", asks=True)
    def ask_number() -> Number:
        raise Ask("How many?")

    accepted_as = {t.id: t.accepted_as for t in registry.describe().types}
    assert "number" not in accepted_as["bool"] and "number" in accepted_as["number"]
    mistyped = {"nodes": {"yes": {"node": "yes", "inputs": {"value": {"graph_input": "value"}}}, "half": {"node": "half", "inputs": {"value": {"sources": [{"node": "yes", "output": "result"}]}}}}, "outputs": {"result": {"node": "half", "output": "result"}}}
    assert [p.type for p in Graph.compile(GraphRecord.model_validate(mistyped), registry).problems] == ["type_mismatch"]
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": {"h": {"node": "half", "inputs": {"value": {"constant": True}}}}, "outputs": {"result": {"node": "h", "output": "result"}}}), registry).problems
    assert problem.type == "invalid_value"
    with pytest.raises(InvalidCallError):
        Graph.compile(make_graph(half), registry).run(value=True)
    waiting = Graph.compile(make_graph(ask_number), registry).run()
    with pytest.raises(InvalidCallError):
        waiting.answer(waiting.waiting[0], True)


def test_a_graph_is_compiled_once_and_freed_without_the_cycle_collector(monkeypatch: MonkeyPatch) -> None:
    import gc
    import weakref

    from conductor._internal.compiled.compile import _Compiler

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    compiles: list[object] = []
    original = _Compiler.compile

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        compiles.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_Compiler, "compile", counted)
    registry = registry_of(upper)
    gc.disable()
    try:
        graph = Graph.compile(GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text", "default": "a"}}}}, "outputs": {"result": {"node": "u", "output": "result"}}}), registry)
        graph_ref, count = weakref.ref(graph), len(compiles)
        del graph, compiles[:]
        assert count == 1
        assert graph_ref() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("ran", ["once", "per item", "again per item"])
def test_a_run_keeps_its_graph_and_both_are_freed_without_the_cycle_collector(ran: str) -> None:
    import gc
    import weakref

    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str]:
        return text.split()

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    @node("join", title="Join", description="For a test.")
    def join(texts: Series[str]) -> str:
        return "+".join(texts)

    registry = registry_of(words, upper, join)
    gc.collect()
    gc.disable()
    try:
        if ran == "once":
            graph = Graph.compile(make_graph(upper, join), registry)
            run = graph.run(text="a")
        else:
            graph = Graph.compile(make_graph(words, upper, join), registry)
            run = graph.run(text="a b") if ran == "per item" else graph.run(text="a b").again(text="c")
        assert run.graph is graph
        refs = [weakref.ref(graph), weakref.ref(run)]
        del graph, run
        assert [ref() for ref in refs] == [None, None]
    finally:
        gc.enable()


def test_a_typed_dict_output_is_a_record_and_an_edge_may_name_one_of_its_fields():
    class Contact(TypedDict):
        name: str

    @node("contact", title="Contact", description="For a test.")
    def contact() -> Contact:
        return {"name": "Bo"}

    described = contact.describe().outputs[0]
    assert (contact.outputs["result"].whole.id, described.type, list(described.json_schema["properties"])) == ("record", "record", ["name"])
    graph = Graph.compile(make_graph(contact, upper, edges=[("contact.result.name", "upper.text")]), registry_of(contact, upper))
    assert graph.problems == [] and graph.run().outputs == {"result": "BO"}


def _named_inputs(names: list[str] = []) -> dict[str, Any]:  # noqa: B006
    return {name: str for name in names}


@node("adds", title="Adds", description="For a test.", compute_added_inputs=_named_inputs)
def adds(names: list[str] = [], **values: Any) -> str:  # noqa: B006
    return ",".join(f"{k}={v}" for k, v in values.items())


def test_a_subgraph_offers_the_inputs_and_outputs_its_inner_hooks_add():
    added = GraphRecord.model_validate({"nodes": {"a": {"node": "adds", "inputs": {"names": {"constant": ["who"]}, "who": {"graph_input": "who"}}}}, "outputs": {"result": {"node": "a", "output": "result"}}})
    computed = GraphRecord.model_validate({"nodes": {"c": {"node": "columns", "inputs": {"names": {"constant": ["x", "y"]}, "text": {"graph_input": "text"}}}}, "outputs": {"x": {"node": "c", "output": "x"}, "y": {"node": "c", "output": "y"}}})
    registry = registry_of(adds, columns, subgraph("box_in", title="Box in", description="For a test.")(added), subgraph("box_out", title="Box out", description="For a test.")(computed))
    assert list(Graph.compile(added, registry).inputs) == ["who"]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "box_in", "inputs": {"who": {"graph_input": "who"}}}}, "outputs": {"result": {"node": "b", "output": "result"}}}), registry)
    assert graph.problems == [] and list(graph.inputs) == ["who"] and graph.run(who="x").outputs == {"result": "who=x"}
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"b": {"node": "box_out", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"x": {"node": "b", "output": "x"}, "y": {"node": "b", "output": "y"}}}), registry)
    assert graph.outputs == ["x", "y"] and graph.run(text="hi").outputs == {"x": "hi", "y": "hi"}
