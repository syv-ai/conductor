"""Loading a graph record: what ``Graph.compile`` checks, what the graph offers a caller, and subgraphs.

A record that says something impossible loads into a graph whose
``problems`` say what and where, never an exception; a graph with problems
refuses to run. A caller fills the open inputs of the first nodes, checked
by ``with_inputs`` all at once.
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
    return text or "blank"


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
    """A ``str`` that lands wherever a ``str`` is asked for."""


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
inner = subgraph("inner", title="Inner", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="up", node="upper")]))
top = subgraph("top", title="Top", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="inner", node="inner")]))
defaulted = subgraph("defaulted", title="Defaulted", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="g", node="greet", defaults={"text": "inside"})]))
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
    return [(p.type, p.loc) for p in refused.value.errors()]


# -- the record ------------------------------------------------------------------------------------


def test_a_node_record_is_at_version_1_unless_it_says_and_is_frozen():
    record = NodeRecord(id="u", node="upper")
    assert record.version == 1
    with pytest.raises(ValidationError):
        record.version = 2  # type: ignore[misc]


def test_a_node_record_without_the_node_it_uses_is_refused():
    with pytest.raises(ValidationError):
        NodeRecord.model_validate({"id": "u"})


def test_a_stored_graph_with_a_misspelled_key_is_refused_naming_it():
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(json.dumps({"nodes": [{"id": "u", "node": "upper", "valuse": {"text": "x"}}]}), REGISTRY)
    assert [(p.type, p.loc, p.ctx["error"]) for p in refused.value.errors()] == [("invalid_record", ("record", "nodes", 0, "valuse"), "extra_forbidden")]


@pytest.mark.parametrize(("record", "loc", "error"), [
    ('{"nodes": [', ("record",), "json_invalid"),
    (json.dumps({"nodes": [{"id": "u", "type": "upper"}]}), ("record", "nodes", 0, "node"), "missing"),
    (json.dumps({"nodes": [], "edges": [{"from": "a.result", "to": "b.text"}]}), ("record", "edges", 0), "tuple_type"),
    (json.dumps({"nodes": [], "edges": [["a", "b.text"]]}), ("record", "edges", 0, 0), "value_error"),
])
def test_what_is_not_a_record_is_refused_as_invalid_record_never_a_validation_error(record, loc, error):
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(record, REGISTRY)
    problems = refused.value.errors()
    assert all(p.type == "invalid_record" for p in problems) and (problems[0].loc, problems[0].ctx["error"]) == (loc, error)
    assert not isinstance(refused.value, ValidationError) and problems[0].msg.startswith("The record is not a graph record at")


def test_a_half_finished_drawing_with_a_records_shape_loads_with_its_problems():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper"}, {"id": "x", "node": "nothing"}], "edges": [["u.result", "y.text"]]}), REGISTRY)
    assert {p.type for p in graph.problems} == {"unknown_node", "edge_names_missing_node"}


def test_a_records_nodes_are_a_tuple():
    record = GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper"}]})
    assert isinstance(record.nodes, tuple)


def test_what_a_host_saves_reads_back_as_it_was_written():
    saved = [
        NodeRecord(id="u", node="upper", version=2, constants={"text": "x"}, display={"x": 3}),
        GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="t", node="tail")], edges=[("u.result", "t.text")]),
        Problem(type="cycle", loc=("a",), msg="The node is part of a cycle, so the graph cannot run.", ctx={}),
        Deprecation(header="Gone", alternative="Use upper."),
        Choice(id="a", title="A"),
        Textarea(),
    ]
    for model in saved:
        assert type(model).model_validate_json(model.model_dump_json()) == model, model


def test_loading_leaves_the_record_it_was_given_as_it_was():
    record = GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="t", node="tail", constants={"text": "x"})], edges=[("u.result", "t.text")])
    before = record.model_dump_json()
    Graph.compile(record, REGISTRY)
    assert record.model_dump_json() == before


def test_loading_one_record_twice_gives_the_same_graph():
    record = {"nodes": [{"id": "s", "node": "fan"}, {"id": "n", "node": "nonempty", "constants": {"text": ""}}, {"id": "t", "node": "tail"}]}
    first, second = Graph.compile(GraphRecord.model_validate(record), REGISTRY), Graph.compile(GraphRecord.model_validate(record), REGISTRY)
    assert first.problems == second.problems and first.describe() == second.describe()
    good = {"nodes": [{"id": "u", "node": "upper"}, {"id": "t", "node": "tail"}], "edges": [["u.result", "t.text"]]}
    first, second = Graph.compile(GraphRecord.model_validate(good), REGISTRY), Graph.compile(GraphRecord.model_validate(good), REGISTRY)
    assert first.runs_on == second.runs_on and list(first.inputs) == list(second.inputs) and first.outputs == second.outputs


# -- typed-in values ---------------------------------------------------------------------------


def test_a_typed_in_value_is_read_once_when_the_graph_loads_and_not_again_per_item():
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

    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "f", "node": "fan"}, {"id": "u", "node": "use", "constants": {"fixed": "x"}}], "edges": [["f.result", "u.item"]]}), registry_of(fan, use))
    assert graph.problems == [] and reads == ["x"]
    assert graph.run().outputs == {"u.result": ["1x", "2x", "3x"]}
    assert reads == ["x"]


def test_a_typed_in_blank_is_refused_unless_the_input_takes_a_blank():
    assert _problems({"nodes": [{"id": "u", "node": "upper", "constants": {"text": None}}]}) == [("invalid_value", ("u", "text"))]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "m", "node": "maybe", "constants": {"text": None}}]}), REGISTRY)
    assert graph.problems == [] and graph.run().outputs == {"m.result": "blank"}


def test_a_value_its_types_own_constructor_refuses_is_a_problem_where_it_was_typed():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "nonempty", "constants": {"text": ""}}]}), REGISTRY)
    [problem] = graph.problems
    assert (problem.type, problem.loc) == ("invalid_value", ("n", "text"))
    assert "empty" not in problem.msg  # the type's own words may quote the value


def test_a_problem_of_a_refused_value_equals_what_its_own_json_reads_back():
    @node("total", title="Total", description="For a test.")
    def total(numbers: list[int]) -> int:
        return sum(numbers)

    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "t", "node": "total", "constants": {"numbers": [1, "a"]}}]}), registry_of(total)).problems

    assert (problem.type, problem.ctx["errors"][0]["loc"]) == ("invalid_value", [1])
    assert Problem.model_validate_json(problem.model_dump_json()) == problem


def test_a_single_value_typed_into_a_series_input_is_refused_not_wrapped_into_a_list_of_one():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "j", "node": "join", "constants": {"texts": "a"}}]}), REGISTRY).problems
    assert (problem.type, problem.loc) == ("invalid_value", ("j", "texts"))
    assert problem.ctx["reason"] == "Input should be a valid list"


# -- what the graph offers -------------------------------------------------------------------------


def test_a_graph_input_carries_its_title_type_and_widget_for_a_form():
    described = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "styled"}]}), REGISTRY).inputs["mode"]
    assert described.field.title == "Mode" and described.field.type == "str" and described.field.required is False
    assert described.field.choices == [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]


def test_graph_inputs_come_in_the_order_of_the_nodes():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "styled"}, {"id": "u", "node": "upper"}, {"id": "f", "node": "fan"}, {"id": "g", "node": "greet"}]}), REGISTRY)
    assert list(graph.inputs) == ["mode", "u.text", "g.text"]


def test_a_node_one_of_whose_outputs_is_read_offers_none_of_its_outputs():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "w", "node": "two"}, {"id": "u", "node": "upper"}], "edges": [["w.zed", "u.text"]]}), REGISTRY)
    assert graph.outputs == ["u.result"]


def test_an_output_a_hook_computed_is_a_graph_output_with_its_title():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "c", "node": "columns", "constants": {"names": ["net", "vat"]}}]}), REGISTRY)
    assert graph.problems == [] and graph.outputs == ["c.net", "c.vat"]
    assert [o.title for o in graph.describe().nodes["c"].outputs] == ["Net", "Vat"]
    assert graph.run(text="9").outputs == {"c.net": "9", "c.vat": "9"}


def test_an_output_and_an_input_sharing_a_name_are_read_by_address():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "g", "node": "greet"}]}), REGISTRY)
    assert list(graph.inputs) == ["text"] and graph.run(text="hi").outputs == {"g.result": "hi"}
    record = {"nodes": [{"id": "u", "node": "upper"}, {"id": "t", "node": "two"}], "edges": [["u.result", "t.text"]]}
    assert Graph.compile(GraphRecord.model_validate(record), REGISTRY).run(text="a").outputs == {"t.zed": "zed A", "t.alpha": "alpha A"}


def test_a_description_of_a_broken_graph_is_json_and_carries_its_problems():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "n", "node": "nonempty", "constants": {"text": ""}}, {"id": "u", "node": "upper"}]}), REGISTRY)
    described = json.loads(graph.describe().model_dump_json())
    assert not graph.is_runnable
    assert [(p["type"], p["loc"]) for p in described["problems"]] == [("invalid_value", ["n", "text"])]
    assert list(described["inputs"]) == ["text"]


# -- what loading refuses --------------------------------------------------------------------------


def test_nodes_reading_each_other_are_a_cycle_on_each_and_the_graph_does_not_run():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "tail"}], "edges": [["a.result", "b.text"], ["b.result", "a.text"]]}), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert not graph.is_runnable
    assert [p.type for p in _refused_run(graph)] == ["cycle", "cycle"]


def _refused_run(graph: Any) -> list[Problem]:
    with pytest.raises(InvalidCallError) as refused:
        graph.run()
    return refused.value.errors()


def test_an_unknown_node_is_one_problem_naming_the_nodes_there_are_and_adds_no_inputs_or_outputs():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "x", "node": "uper"}, {"id": "t", "node": "tail"}]}), registry_of(upper, tail))
    [problem] = graph.problems
    assert (problem.type, problem.loc) == ("unknown_node", ("x",))
    assert problem.ctx == {"node": "uper", "choices": ["tail", "upper"]}
    assert list(graph.inputs) == ["text"] and graph.outputs == ["t.result"]


def test_a_version_the_node_does_not_have_is_a_problem_never_the_nearest_version():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper", "version": 2}]}), REGISTRY).problems
    assert (problem.type, problem.loc, problem.ctx) == ("unknown_version", ("u",), {"node": "upper", "version": 2, "newest": 1})


def test_a_hook_that_raises_is_a_problem_on_its_node_never_an_exception():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "r", "node": "refuses"}]}), REGISTRY).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("r",), "ValueError")


def test_a_value_its_type_refuses_never_reaches_a_hook_and_is_said_once():
    # "refuses" reads its mode in a hook; a list typed into the str would have reached it
    problems = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "r", "node": "refuses", "constants": {"mode": ["a", "b"]}}]}), REGISTRY).problems
    assert [(p.type, p.loc) for p in problems] == [("invalid_value", ("r", "mode"))]


def test_a_subtype_lands_where_its_parent_is_asked_for_and_not_the_other_way():
    assert _problems({"nodes": [{"id": "n", "node": "name"}, {"id": "u", "node": "upper"}], "edges": [["n.result", "u.text"]]}) == []
    assert _problems({"nodes": [{"id": "u", "node": "upper"}, {"id": "n", "node": "nonempty"}], "edges": [["u.result", "n.text"]]}) == [
        ("type_mismatch", ("n", "text")),
    ]


def test_a_list_of_anything_a_hook_computes_loads_into_a_typed_input_and_a_table_input_without_a_crash():
    into_text = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "anything"}, {"id": "u", "node": "upper"}], "edges": [["a.items", "u.text"]]}), REGISTRY)
    assert into_text.problems == [] and into_text.run().outputs == {"u.result": ["X", "X"]}
    assert _problems({"nodes": [{"id": "a", "node": "anything"}, {"id": "h", "node": "height"}], "edges": [["a.items", "h.rows"]]}) == []


def test_one_mistyped_edge_among_several_into_a_series_input_is_a_type_mismatch():
    record = {
        "nodes": [{"id": "u", "node": "upper"}, {"id": "f", "node": "fan"}, {"id": "j", "node": "join"}],
        "edges": [["u.result", "j.texts"], ["f.result", "j.texts"]],
    }
    assert _problems(record) == [("type_mismatch", ("j", "texts"))]


def test_an_edge_from_another_nodes_input_is_not_a_source():
    problems = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "h", "node": "upper"}, {"id": "u", "node": "tail"}], "edges": [["h.text", "u.text"]]}), REGISTRY).problems
    assert [(p.type, p.loc, p.ctx["source"]) for p in problems] == [("unknown_output", ("u", "text"), "h.text")]


def test_a_broken_edge_is_one_problem_and_no_echo_on_the_nodes_downstream():
    record = {"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "tail"}], "edges": [["a.nope", "b.text"], ["b.result", "c.text"]]}
    assert _problems(record) == [("unknown_output", ("b", "text"))]


def test_an_edge_from_a_node_that_did_not_load_is_not_reported_again():
    unknown = {"nodes": [{"id": "x", "node": "gone"}, {"id": "u", "node": "upper"}], "edges": [["x.result", "u.text"]]}
    assert _problems(unknown) == [("unknown_node", ("x",))]
    refused_id = {"nodes": [{"id": "a@b", "node": "upper"}, {"id": "u", "node": "tail"}], "edges": [["a@b.result", "u.text"]]}
    assert _problems(refused_id) == [("invalid_node_id", ("a@b",))]


def test_an_edge_from_a_slashed_name_under_a_plain_node_names_a_node_the_graph_lacks():
    problems = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper"}, {"id": "u", "node": "tail"}], "edges": [["a/b.result", "u.text"]]}), REGISTRY).problems
    assert [(p.type, p.loc, p.ctx["missing"]) for p in problems] == [("edge_names_missing_node", ("u", "text"), "a/b")]


# -- what a caller gives ---------------------------------------------------------------------------


def test_with_inputs_takes_an_address_mapping_beside_keywords_and_refuses_one_it_does_not_offer():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "tail"}, {"id": "s", "node": "styled"}]}), REGISTRY)
    assert graph.with_inputs({"a.text": "x", "b.text": "y"}, mode="b").resume().outputs == {"a.result": "X", "b.result": "y.", "s.result": "b"}
    assert _refused(graph, {"a.text": "x", "b.text": "y", "c.text": "z"}, mode="b") == [("not_offered", ("inputs", "c.text"))]


def test_a_bare_name_two_inputs_share_is_refused_naming_the_addresses():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "tail"}]}), REGISTRY)
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(text="x")
    not_offered = [p for p in refused.value.errors() if p.type == "not_offered"]
    assert [p.ctx for p in not_offered] == [{"name": "text", "inputs": ["a.text", "b.text"]}]


def test_with_inputs_refuses_every_fault_at_once_listing_what_it_offers():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "styled"}, {"id": "u", "node": "upper", "constants": {"text": "fixed"}}]}), REGISTRY)
    assert list(graph.inputs) == ["mode"]
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(mode=5, text="a fixed input is not offered", typo="x")
    found = sorted((p.type, p.loc) for p in refused.value.errors())
    assert found == [("invalid_input", ("inputs", "mode")), ("not_offered", ("inputs", "text")), ("not_offered", ("inputs", "typo"))]
    assert [p.ctx["inputs"] for p in refused.value.errors() if p.type == "not_offered"] == [["mode"], ["mode"]]


# -- subgraphs -------------------------------------------------------------------------------


def test_an_embedding_offers_its_inner_open_input_and_output_as_one_node():
    assert enrich.title == "Enrich"
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}]}), REGISTRY)
    assert graph.problems == [] and list(graph.inputs) == ["text"] and graph.inputs["text"].field.title == "Text"
    run = graph.run(text="hi")
    assert graph.outputs == ["e.tail.result"] and run.outputs == {"e.tail.result": "HI."} and run.results["e"] == {"tail.result": "HI."}


def test_a_value_typed_on_an_embedding_reaches_the_inner_input_it_offers():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich", "constants": {"text": "hi"}}]}), REGISTRY)
    assert graph.problems == [] and graph.inputs == {} and graph.run().outputs == {"e.tail.result": "HI."}


def test_an_unfed_embedding_keeps_its_inner_defaults_and_runs_once():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "defaulted"}]}), REGISTRY)
    assert {k: v.mode for k, v in graph.runs_on.items()} == {"d": "once"}
    assert graph.run().outputs == {"d.g.result": "inside"}


def test_an_inner_input_with_a_default_is_offered_as_not_required():
    assert Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "defaulted"}]}), REGISTRY).inputs["text"].field.required is False
    assert Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}]}), REGISTRY).inputs["text"].field.required is True


def test_a_graph_inside_a_graph_is_one_node_from_outside():
    run = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "t", "node": "top"}]}), REGISTRY).run(text="hi")
    assert run.outputs == {"t.inner.up.result": "HI"} and run.results == {"t": {"inner.up.result": "HI"}}


def test_renaming_a_node_inside_an_embedding_keeps_its_inputs_names_and_moves_its_outputs_address():
    renamed = subgraph("enrich", title="Enrich", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="shout", node="upper"), NodeRecord(id="end", node="tail")], edges=[("shout.result", "end.text")]))
    record = {"nodes": [{"id": "g", "node": "greet"}, {"id": "e", "node": "enrich"}], "edges": [["g.result", "e.text"]]}
    for embedded, output in ((enrich, "e.tail.result"), (renamed, "e.end.result")):
        graph = Graph.compile(GraphRecord.model_validate(record), registry_of(greet, upper, tail, embedded))
        assert graph.problems == [] and graph.run(text="hi").outputs == {output: "HI."}


def test_a_value_on_an_input_the_embedding_does_not_offer_is_an_unknown_input():
    fed_inside = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich", "constants": {"text": "a", "result": "b"}}]}), REGISTRY).problems
    assert [(p.type, p.loc) for p in fed_inside] == [("unknown_input", ("e", "result"))]
    by_inner_address = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich", "constants": {"upper.text": "a"}}]}), REGISTRY).problems
    assert [(p.type, p.loc, p.ctx["inputs"]) for p in by_inner_address] == [("unknown_input", ("e", "upper.text"), ["text"])]


def test_a_table_made_inside_an_embedding_is_offered_whole_as_the_one_value_it_is():
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "t", "node": "team"}, {"id": "p", "node": "each-person"}, {"id": "u", "node": "upper"}],
        "edges": [["t.people.result", "p.table"], ["p.result.email", "u.text"]],
    }), REGISTRY)
    assert graph.problems == []
    assert graph.run(text="name,email\nIda,ida@a.dk\nBo,bo@b.dk\n").outputs == {"u.result": ["IDA@A.DK", "BO@B.DK"]}


def test_a_node_built_from_a_graph_offers_its_inner_inputs_and_carries_its_deprecation():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "w", "node": "wrapped"}]}), REGISTRY)
    assert list(graph.inputs) == ["text"] and graph.run(text="a").outputs == {"w.upper.result": "A"}
    assert REGISTRY.describe().nodes[[n.id for n in REGISTRY].index("wrapped")].deprecation.header == "Gone"


@node("pair", title="Pair", description="For a test.")
def pair(a: str, b: str) -> str:
    return a + b


def _hint_for(formula: str) -> dict[str, Any]:
    return {name: float for name in formula.split()}


@node("formula", title="Formula", description="For a test.", compute_added_inputs=_hint_for)
def formula(formula: str, **values: float) -> float:
    return sum(values.values())


gone_inside = subgraph("gone_inside", title="Gone inside", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="x", node="gone")]))
around = subgraph("around", title="Around", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="b", node="gone_inside")]))
invalid_inside = subgraph("invalid_inside", title="Invalid inside", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="n", node="nonempty", constants={"text": ""})]))
unfed_inside = subgraph("unfed_inside", title="Unfed inside", description="For a test.")(make_graph(upper, pair, edges=[("upper.result", "pair.a")]))
BROKEN = registry_of(upper, pair, nonempty, formula, gone_inside, around, invalid_inside, unfed_inside)


def test_a_hook_is_not_handed_a_required_input_nobody_typed_in_and_the_node_says_why():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "f", "node": "formula"}]}), BROKEN)
    assert [(p.type, p.loc) for p in graph.problems] == [("hook_needs_value", ("f", "formula"))]
    typed = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "f", "node": "formula", "constants": {"formula": "a b"}}]}), BROKEN)
    assert typed.problems == [] and list(typed.inputs) == ["a", "b"]


def test_an_input_a_hook_reads_is_fixed_when_the_graph_loads_and_never_a_graph_input():
    """The fields a hook works out hold for every run, so what it read is the record's, never the caller's."""

    def outputs(mode: str) -> dict[str, Any]:
        return {"a": int} if mode == "a" else {"b": str}

    @node("by-mode", title="By mode", description="For a test.", compute_outputs=outputs)
    def by_mode(mode: str = "a") -> dict[str, Any]:
        return {"a": 1} if mode == "a" else {"b": "x"}

    registry = registry_of(by_mode)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "m", "node": "by-mode"}]}), registry)
    assert graph.problems == [] and graph.inputs == {} and graph.run().outputs == {"m.a": 1}
    with pytest.raises(InvalidCallError):
        graph.run(mode="b")

    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "m", "node": "by-mode", "defaults": {"mode": "b"}}]}), registry).problems
    assert (problem.type, problem.loc) == ("default_on_later_node", ("m", "mode"))
    assert Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "f", "node": "formula"}]}), BROKEN).problems[0].type == "hook_needs_value"


def test_a_broken_graphs_problems_are_its_own_where_it_is_embedded_with_the_embedding_in_front():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "g", "node": "gone_inside"}]}), BROKEN)
    assert [(p.type, p.loc) for p in graph.problems] == [("unknown_node", ("g", "x"))]
    assert not graph.is_runnable and graph.inputs == {}


def test_a_broken_graph_two_levels_down_breaks_the_graph_around_it_where_it_is_embedded():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "around"}]}), BROKEN).problems
    assert (problem.type, problem.loc) == ("unknown_node", ("a", "b", "x"))


def test_an_invalid_value_typed_inside_makes_the_embedding_not_runnable():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "i", "node": "invalid_inside"}]}), BROKEN)
    assert [(p.type, p.loc) for p in graph.problems] == [("invalid_value", ("i", "n", "text"))]
    with pytest.raises(InvalidCallError):
        graph.run()


def test_a_graph_with_a_required_input_on_a_later_node_nobody_feeds_cannot_be_embedded():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "u", "node": "unfed_inside"}]}), BROKEN).problems
    assert (problem.type, problem.loc) == ("unbound_on_later_node", ("u", "pair", "b"))


def test_a_value_typed_on_a_broken_embedding_adds_no_second_problem():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "g", "node": "gone_inside", "constants": {"text": "x"}}]}), BROKEN)
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


halves = subgraph("halves", title="Halves", description="For a test.")(GraphRecord(
    nodes=[NodeRecord(id="x", node="upper"), NodeRecord(id="y", node="upper"), NodeRecord(id="b", node="both_halves")],
    edges=[("x.result", "b.a"), ("y.result", "b.b")],
))
picky = subgraph("picky", title="Picky", description="For a test.")(make_graph(refuses_bad))
OUTSIDE = registry_of(upper, tail, refuses_bad, both_halves, taken, enrich, halves, picky)


def test_only_the_nodes_on_a_cycle_are_a_cycle_and_a_node_below_keeps_its_fields():
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "tail"}],
        "edges": [["a.result", "b.text"], ["b.result", "a.text"], ["b.result", "c.text"]],
    }), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert set(graph.describe().nodes) == {"a", "b", "c"}


def test_an_editor_reads_every_node_of_a_half_broken_graph_and_which_ones_loaded():
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "tail"}, {"id": "x", "node": "gone"}],
        "edges": [["a.nope", "b.text"], ["b.result", "c.text"]],
    }), REGISTRY)
    drawn = json.loads(graph.describe().model_dump_json())["nodes"]
    assert list(drawn) == ["a", "b", "c", "x"]
    assert {node_id: d["found"] for node_id, d in drawn.items()} == {"a": True, "b": True, "c": True, "x": False}


def _drawn(graph: Any, node_id: str) -> tuple[bool, str | None, dict[str, str], list[str], list[str]]:
    """A node as an editor draws it: whether its fields are known, where it runs, how each input is fed, its
    outputs, and its problems."""
    described = graph.describe().nodes[node_id]
    return (
        described.found, described.runs.msg if described.runs else None, {i.field.name: i.source.mode for i in described.inputs},
        [o.name for o in described.outputs], [p.type for p in described.problems],
    )


def test_a_node_with_a_refused_edge_is_drawn_with_its_fields_and_so_is_a_sound_node_that_reads_it():
    """The node its problem is on keeps its fields, the refused input marked; what reads it is planned as in a graph with no problem."""
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "f", "node": "fan"}, {"id": "u", "node": "upper"}, {"id": "t", "node": "tail"}, {"id": "j", "node": "join"}],
        "edges": [["f.result", "u.text"], ["u.result", "t.text"], ["t.result", "j.texts"]],
    }), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("type_mismatch", ("u", "text"))]
    assert _drawn(graph, "u") == (True, "once", {"text": "missing"}, ["result"], ["type_mismatch"])
    assert _drawn(graph, "t") == (True, "once", {"text": "one_value"}, ["result"], [])
    assert _drawn(graph, "j") == (True, "once", {"texts": "collected"}, ["result"], [])
    assert list(graph.runs_on) == ["f", "u", "t", "j"] and not graph.is_runnable
    with pytest.raises(InvalidCallError) as refused:
        graph.run()
    assert [p.type for p in refused.value.errors()] == ["type_mismatch"]


def test_the_nodes_on_a_cycle_and_a_node_below_it_are_drawn_with_their_fields():
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "c", "node": "tail"}, {"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
        "edges": [["a.result", "b.text"], ["b.result", "a.text"], ["b.result", "c.text"]],
    }), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert _drawn(graph, "a") == (True, "once", {"text": "missing"}, ["result"], ["cycle"])  # the edge that closes the cycle brings it nothing
    assert _drawn(graph, "b") == (True, "once", {"text": "one_value"}, ["result"], ["cycle"])
    assert _drawn(graph, "c") == (True, "once", {"text": "one_value"}, ["result"], [])


def test_an_input_whose_series_never_pairs_with_the_nodes_other_is_marked_and_the_node_is_drawn():
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "x", "node": "fan"}, {"id": "y", "node": "fan"}, {"id": "b", "node": "both_ints"}, {"id": "t", "node": "tail"}],
        "edges": [["x.result", "b.a"], ["y.result", "b.b"], ["b.result", "t.text"]],
    }), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("series_not_nested", ("b",))]
    assert _drawn(graph, "b") == (True, "per item of x.result", {"a": "item", "b": "missing"}, ["result"], ["series_not_nested"])
    assert _drawn(graph, "t") == (True, "per item of x.result", {"text": "item"}, ["result"], [])


def test_only_a_node_whose_definition_or_hook_gave_no_fields_is_drawn_without_them_and_its_reader_keeps_its_own():
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": [{"id": "gone", "node": "gone"}, {"id": "old", "node": "upper", "version": 9}, {"id": "r", "node": "refuses", "constants": {"mode": "bad"}},
                  {"id": "u", "node": "upper"}, {"id": "t", "node": "tail"}, {"id": "v", "node": "upper"}],
        "edges": [["gone.result", "u.text"], ["old.result", "t.text"], ["r.result", "v.text"]],
    }), REGISTRY)
    assert [(p.type, p.loc) for p in graph.problems] == [("unknown_node", ("gone",)), ("unknown_version", ("old",)), ("compute_hook_failed", ("r",))]
    for unknown in ("gone", "old", "r"):
        assert _drawn(graph, unknown)[:4] == (False, None, {}, [])
    for reader in ("u", "t", "v"):
        assert _drawn(graph, reader) == (True, "once", {"text": "missing"}, ["result"], [])


def test_an_input_a_hook_adds_under_a_name_the_node_declares_is_the_hooks_failure():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "taken", "constants": {"formula": "x"}}]}), OUTSIDE).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("d",), "added_input_taken")


def test_a_wrong_address_on_an_embedding_is_unknown_output_or_unknown_input_on_it():
    def found(record: dict[str, Any]) -> list[tuple[str, tuple[Any, ...]]]:
        return [(p.type, p.loc) for p in Graph.compile(GraphRecord.model_validate(record), OUTSIDE).problems]

    nodes = [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}]
    assert found({"nodes": nodes, "edges": [["e.nope", "u.text"]]}) == [("unknown_output", ("u", "text"))]
    assert found({"nodes": nodes, "edges": [["e.upper.result", "u.text"]]}) == [("unknown_output", ("u", "text"))]
    assert found({"nodes": nodes[::-1], "edges": [["u.result", "e.tail"]]}) == [("unknown_input", ("e", "tail"))]


@pytest.mark.parametrize("ctx", [{"tables": [1, 2]}, {"source": "e/report.pdf", "node_id": "anything"}])
def test_an_inner_nodes_own_error_is_said_where_it_is_inside_the_embedding_with_its_ctx_as_its_author_wrote_it(ctx):
    @node("unreadable", title="Unreadable", description="For a test.")
    def unreadable(text: str) -> str:
        raise NodeError("The document could not be read.", code="unreadable", ctx=ctx)

    embedding = subgraph("e", title="E", description="For a test.")(make_graph(unreadable))
    run = Graph.compile(make_graph(embedding), registry_of(unreadable, embedding)).run(text="hi")

    assert (run.status, run.state.cause.msg) == ("failed", "The document could not be read.")
    assert (run.state.cause.type, run.state.cause.loc, run.state.cause.ctx) == ("unreadable", ("e", "unreadable"), ctx)


def test_an_outer_edge_never_reads_inside_an_embedding_by_an_inner_id():
    problems = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}], "edges": [["e/upper.result", "u.text"]]}), OUTSIDE).problems
    assert [(p.type, p.loc, p.ctx["missing"]) for p in problems] == [("edge_names_missing_node", ("u", "text"), "e/upper")]


def test_an_edge_from_a_missing_node_into_an_embedding_is_its_problem_at_the_field_it_offers():
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}], "edges": [["ghost.result", "e.text"]]}), OUTSIDE).problems
    assert (problem.type, problem.loc, problem.ctx["missing"]) == ("edge_names_missing_node", ("e", "text"), "ghost")
    assert (problem.ctx["source"], problem.ctx["target"]) == ("ghost.result", "e.text")


def test_an_inner_input_a_hook_reads_is_set_inside_the_graph_so_a_value_typed_for_it_on_the_embedding_is_an_unknown_input():
    assert Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "p", "node": "picky"}]}), OUTSIDE).inputs == {}
    [problem] = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "p", "node": "picky", "constants": {"mode": "bad"}}]}), OUTSIDE).problems
    assert (problem.type, problem.loc, problem.ctx["inputs"]) == ("unknown_input", ("p", "mode"), [])


def test_an_embeddings_inputs_are_filled_by_its_own_address():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}]}), OUTSIDE)
    assert {name: i.address for name, i in graph.inputs.items()} == {"e.text": "e.text", "u.text": "u.text"}
    assert graph.run({"e.text": "hi", "u.text": "yo"}).outputs == {"e.tail.result": "HI.", "u.result": "YO"}


def test_two_inner_inputs_sharing_a_name_are_offered_by_the_inner_records_ids():
    assert [f.name for f in next(n for n in OUTSIDE.describe().nodes if n.id == "halves").inputs] == ["x.text", "y.text"]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "upper"}, {"id": "c", "node": "halves", "constants": {"y.text": "b"}}], "edges": [["s.result", "c.x.text"]]}), OUTSIDE)
    assert graph.problems == [] and list(graph.inputs) == ["text"]


def test_an_authored_id_with_a_slash_beside_an_embedding_is_refused_alone_and_stays_the_authors():
    beside = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich"}, {"id": "e/upper", "node": "upper"}]}), OUTSIDE).problems
    assert [(p.type, p.loc) for p in beside] == [("invalid_node_id", ("e/upper",))]
    broken = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "e", "node": "gone"}, {"id": "e/upper", "node": "upper"}]}), REGISTRY).problems
    assert sorted((p.type, p.loc) for p in broken) == [("invalid_node_id", ("e/upper",)), ("unknown_node", ("e",))]


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
    assert "a set has no order" in refused.value.errors()[0].msg
    assert graph.run(contacts=["b", "a"]).outputs == {"send.result": "b,a"}


def _nested_forty_deep() -> tuple[Any, str]:
    """A registry of forty graphs, each holding the one before it, and the id of the outermost."""
    from conductor import NodeRegistry

    registry = NodeRegistry()

    @registry.node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    inner = "upper"
    for depth in range(40):
        registry.add(subgraph(f"nest-{depth}", title=f"Nest {depth}", description="For a test.")(GraphRecord(nodes=[NodeRecord(id="n", node=inner)])))
        inner = f"nest-{depth}"
    return registry, inner


def test_graphs_embedded_forty_deep_load_at_once():
    """Each subgraph is loaded alone once per load, however deep it nests, so depth costs time in proportion."""
    import time

    registry, outermost = _nested_forty_deep()
    started = time.monotonic()
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "top", "node": outermost}]}), registry)
    assert time.monotonic() - started < 2 and graph.problems == [] and list(graph.inputs) == ["text"]


def test_graphs_embedded_forty_deep_run_as_the_one_node_inside_them():
    registry, outermost = _nested_forty_deep()
    assert Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "top", "node": outermost}]}), registry).run(text="z").outputs == {"top" + ".n" * 40 + ".result": "Z"}


def test_describe_lists_each_nodes_problems_on_it():
    described = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper"}, {"id": "x", "node": "uper"}]}), REGISTRY).describe()
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
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "w", "node": "words"}, {"id": "c", "node": "count"}], "edges": [["w.result", "c.flags"]]}), registry).problems
    assert (problem.type, problem.ctx["got"], problem.ctx["expected"]) == ("type_mismatch", "Series[str]", "Series[bool]")
    assert [p.type for p in Graph.compile(make_graph(words, count), registry).problems] == ["type_mismatch"]


def test_what_the_outside_gives_an_embedded_input_replaces_the_inner_default():
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
    assert Graph.compile(make_graph(shout), registry).run().outputs == {"shout.upper.result": "INNER"}
    typed = Graph.compile(make_graph(shout, constants={"shout.text": "outer"}), registry)
    assert typed.problems == [] and typed.run().outputs == {"shout.upper.result": "OUTER"}
    defaulted = Graph.compile(make_graph(shout, defaults={"shout.text": "outer"}), registry)
    assert defaulted.problems == [] and defaulted.run().outputs == {"shout.upper.result": "OUTER"}
    fed = Graph.compile(make_graph(hello, shout, edges=[("hello.result", "shout.text")]), registry)
    assert fed.problems == [] and fed.run().outputs == {"shout.upper.result": "HI"}


def test_a_yes_or_no_never_lands_on_a_number_and_a_union_is_said_by_its_members_names():
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
        (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": source}, {"id": "d", "node": "double"}], "edges": [["s.result", "d.x"]]}), registry).problems
        assert (problem.type, problem.ctx["got"]) == ("type_mismatch", got)


def test_a_graph_input_is_its_fields_record_and_writes_as_json():
    import json

    from conductor import GraphInput, NodeRegistry, make_graph, node

    @node("u", title="U", description="For a test.")
    def u(t: str, n: int = 1) -> str:
        return t

    registry = NodeRegistry()
    registry.add(u)
    graph = Graph.compile(make_graph(u), registry)
    assert isinstance(graph.inputs["t"], GraphInput) and graph.describe().inputs == graph.inputs
    assert json.loads(json.dumps({name: i.field.default for name, i in graph.inputs.items()})) == {"t": None, "n": 1}
    assert (graph.inputs["t"].field.name, graph.inputs["t"].address, graph.inputs["t"].field.required) == ("t", "u.t", True)


def test_a_yes_or_no_typed_in_or_given_never_lands_on_a_number():
    from conductor import NodeRegistry, node

    @node("double", title="Double", description="For a test.")
    def double(x: float) -> float:
        return x * 2

    registry = NodeRegistry()
    registry.add(double)
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "double", "constants": {"x": True}}]}), registry).problems
    assert (problem.type, problem.ctx["reason"]) == ("invalid_value", "a yes or no is not a number; give a number")
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "double"}]}), registry).run(x=True)
    assert refused.value.errors()[0].type == "invalid_input"
    assert Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "d", "node": "double"}]}), registry).run(x=2).outputs == {"d.result": 4.0}


def test_a_table_into_a_series_is_said_as_the_table_it_is():
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
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "sheet"}, {"id": "c", "node": "count"}], "edges": [["s.result", "c.items"]]}), registry).problems
    assert (problem.ctx["got"], problem.ctx["expected"]) == ("Table", "Series[int]")


def test_a_yes_or_no_never_lands_on_a_hosts_number_type_whose_own_rule_lets_an_int_in():
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

    @registry.node("ask_number", title="Ask number", description="For a test.")
    def ask_number() -> Number:
        raise Ask("How many?")

    accepted_as = {t.id: t.accepted_as for t in registry.describe().types}
    assert "number" not in accepted_as["bool"] and "number" in accepted_as["number"]
    mistyped = {"nodes": [{"id": "yes", "node": "yes"}, {"id": "half", "node": "half"}], "edges": [["yes.result", "half.value"]]}
    assert [p.type for p in Graph.compile(GraphRecord.model_validate(mistyped), registry).problems] == ["type_mismatch"]
    (problem,) = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "h", "node": "half", "constants": {"value": True}}]}), registry).problems
    assert problem.type == "invalid_value"
    with pytest.raises(InvalidCallError):
        Graph.compile(make_graph(half), registry).run(value=True)
    waiting = Graph.compile(make_graph(ask_number), registry).run()
    with pytest.raises(InvalidCallError):
        waiting.answer(waiting.waiting[0], True)


def test_a_loaded_graph_is_compiled_once_and_freed_without_the_cycle_collector(monkeypatch: MonkeyPatch) -> None:
    import gc
    import weakref

    from conductor._internal.compiled.compiler import _Compiler

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    loads: list[object] = []
    load = _Compiler.compile

    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        loads.append(self)
        return load(self, *args, **kwargs)

    monkeypatch.setattr(_Compiler, "compile", counted)
    registry = registry_of(upper)
    gc.disable()
    try:
        graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper", "defaults": {"text": "a"}}]}), registry)
        held, loaded = weakref.ref(graph), len(loads)
        del graph, loads[:]
        assert loaded == 1
        assert held() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("ran", ["once", "per item", "again per item"])
def test_a_run_holds_the_graph_that_was_loaded_and_both_are_freed_without_the_cycle_collector(ran: str) -> None:
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
        held = [weakref.ref(graph), weakref.ref(run)]
        del graph, run
        assert [ref() for ref in held] == [None, None]
    finally:
        gc.enable()


def test_a_typed_dict_on_a_field_is_a_record_with_its_fields_and_an_edge_may_name_one():
    class Contact(TypedDict):
        name: str

    @node("contact", title="Contact", description="For a test.")
    def contact() -> Contact:
        return {"name": "Bo"}

    described = contact.describe().outputs[0]
    assert (contact.outputs["result"].whole.id, described.type, list(described.json_schema["properties"])) == ("record", "record", ["name"])
    graph = Graph.compile(make_graph(contact, upper, edges=[("contact.result.name", "upper.text")]), registry_of(contact, upper))
    assert graph.problems == [] and graph.run().outputs == {"upper.result": "BO"}


def _named_inputs(names: list[str] = []) -> dict[str, Any]:  # noqa: B006
    return {name: str for name in names}


@node("adds", title="Adds", description="For a test.", compute_added_inputs=_named_inputs)
def adds(names: list[str] = [], **values: Any) -> str:  # noqa: B006
    return ",".join(f"{k}={v}" for k, v in values.items())


def test_an_embedding_offers_the_fields_its_graph_offers_alone_as_its_hooks_made_them():
    added = GraphRecord(nodes=[NodeRecord(id="a", node="adds", constants={"names": ["who"]})])
    made = GraphRecord(nodes=[NodeRecord(id="c", node="columns", constants={"names": ["x", "y"]})])
    registry = registry_of(adds, columns, subgraph("box_in", title="Box in", description="For a test.")(added), subgraph("box_out", title="Box out", description="For a test.")(made))
    assert list(Graph.compile(added, registry).inputs) == ["who"]
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "b", "node": "box_in"}]}), registry)
    assert graph.problems == [] and list(graph.inputs) == ["who"] and graph.run(who="x").outputs == {"b.a.result": "who=x"}
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "b", "node": "box_out"}]}), registry)
    assert graph.outputs == ["b.c.x", "b.c.y"] and graph.run(text="hi").outputs == {"b.c.x": "hi", "b.c.y": "hi"}
