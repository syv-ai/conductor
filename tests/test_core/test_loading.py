"""Loading a graph record: what ``registry.load`` checks, what the graph offers a caller, and embedded graphs.

A record that says something impossible loads into a graph whose
``problems`` say what and where, never an exception; a graph with problems
refuses to run. A caller fills the open inputs of the first nodes, checked
by ``with_inputs`` all at once.
"""

import io
import json
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    Deprecation,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Problem,
    Series,
    Table,
    make_graph,
    node,
)
from conductor.widgets import Choice, Dropdown, Textarea
from pydantic import ValidationError
from pydantic_core import core_schema

from tests.test_core.nodes import registry_of


class Person(TypedDict):
    name: str
    email: str


@node("upper")
def upper(text: str) -> str:
    return text.upper()


@node("tail")
def tail(text: str) -> str:
    return text + "."


@node("join")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@node("fan")
def fan() -> Series[int]:
    return [1, 2, 3]


@node("maybe")
def maybe(text: str | None) -> str:
    return text or "blank"


@node("greet")
def greet(text: str = "hello") -> str:
    return text


@node("people")
def people(text: str) -> Table[Person]:
    return pl.read_csv(io.StringIO(text))


@node("styled")
def styled(
    mode: Annotated[str, Param(title="Mode", widget=Dropdown(choices=[Choice(id="a", title="A"), Choice(id="b", title="B")]))] = "a",
) -> str:
    return mode


class Name(str):
    """A ``str`` that lands wherever a ``str`` is asked for."""


@node("name")
def name_of(text: str) -> Name:
    return Name(text)


class Nonempty(str):
    def __new__(cls, value: str) -> "Nonempty":
        if not value:
            raise ValueError("empty")
        return super().__new__(cls, value)


@node("nonempty")
def nonempty(text: Nonempty) -> str:
    return text


@dataclass(frozen=True)
class Two:
    zed: str
    alpha: str


@node("two")
def two(text: str) -> Two:
    return Two(f"zed {text}", f"alpha {text}")


def _columns(names: list[str]) -> dict[str, Any]:
    return {name: Annotated[str, Param(title=name.title())] for name in names}


@node("columns", compute_outputs=_columns)
def columns(text: str, names: list[str] = []) -> dict[str, str]:  # noqa: B006
    return {name: text for name in names}


def _anything(n: int) -> dict[str, Any]:
    return {"items": Series[Any]}


@node("anything", compute_outputs=_anything)
def anything(n: int = 2) -> dict[str, Any]:
    return {"items": ["x"] * n}


@node("height")
def height(rows: Table) -> int:
    return rows.height


def _refuses(mode: str) -> dict[str, Any]:
    raise ValueError("Not that mode.")


@node("refuses", compute_outputs=_refuses)
def refuses(mode: str = "ok") -> dict[str, Any]:
    return {}


enrich = node("enrich")(make_graph(upper, tail))
inner = node("inner")(GraphRecord(nodes=[NodeRecord(id="up", node="upper")]))
top = node("top")(GraphRecord(nodes=[NodeRecord(id="inner", node="inner")]))
defaulted = node("defaulted")(GraphRecord(nodes=[NodeRecord(id="g", node="greet", defaults={"text": "inside"})]))
team = node("team")(make_graph(people))
wrapped = node("wrapped", deprecation=Deprecation(header="Gone"))(make_graph(upper))

REGISTRY = registry_of(
    upper, tail, join, fan, maybe, greet, people, styled, name_of, nonempty, two, columns, anything, height, refuses,
    enrich, inner, top, defaulted, team, wrapped,
)


def _problems(record: GraphRecord | dict[str, Any]) -> list[tuple[str, tuple[Any, ...]]]:
    return [(p.type, p.loc) for p in REGISTRY.load(record).problems]


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
    with pytest.raises(ValidationError, match="valuse"):
        REGISTRY.load({"nodes": [{"id": "u", "node": "upper", "valuse": {"text": "x"}}]})


def test_a_records_nodes_are_a_tuple():
    record = GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper"}]})
    assert isinstance(record.nodes, tuple)


def test_what_a_host_saves_reads_back_as_it_was_written():
    saved = [
        NodeRecord(id="u", node="upper", version=2, values={"text": "x"}, display={"x": 3}),
        GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="t", node="tail")], edges=[("u.result", "t.text")]),
        Problem(type="cycle", loc=("a",), msg="The node is part of a cycle, so the graph cannot run.", ctx={}),
        Deprecation(header="Gone", alternative="Use upper."),
        Choice(id="a", title="A"),
        Textarea(),
    ]
    for model in saved:
        assert type(model).model_validate_json(model.model_dump_json()) == model, model


def test_loading_leaves_the_record_it_was_given_as_it_was():
    record = GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="t", node="tail", values={"text": "x"})], edges=[("u.result", "t.text")])
    before = record.model_dump_json()
    REGISTRY.load(record)
    assert record.model_dump_json() == before


def test_loading_one_record_twice_gives_the_same_graph():
    record = {"nodes": [{"id": "s", "node": "fan"}, {"id": "n", "node": "nonempty", "values": {"text": ""}}, {"id": "t", "node": "tail"}]}
    first, second = REGISTRY.load(record), REGISTRY.load(record)
    assert first.problems == second.problems and first.describe() == second.describe()
    good = {"nodes": [{"id": "u", "node": "upper"}, {"id": "t", "node": "tail"}], "edges": [["u.result", "t.text"]]}
    first, second = REGISTRY.load(good), REGISTRY.load(good)
    assert first.plan == second.plan and list(first.inputs) == list(second.inputs) and first.outputs == second.outputs


# -- typed-in values ---------------------------------------------------------------------------


def test_a_typed_in_value_is_read_once_when_the_graph_loads_and_not_again_per_row():
    reads = []

    class Counted(str):
        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            def read(value: str) -> "Counted":
                reads.append(value)
                return cls(value)

            return core_schema.no_info_plain_validator_function(read)

    @node("use")
    def use(row: int, fixed: Counted) -> str:
        return f"{row}{fixed}"

    graph = registry_of(fan, use).load({"nodes": [{"id": "f", "node": "fan"}, {"id": "u", "node": "use", "values": {"fixed": "x"}}], "edges": [["f.result", "u.row"]]})
    assert graph.problems == [] and reads == ["x"]
    assert graph.run().outputs == {"u.result": ["1x", "2x", "3x"]}
    assert reads == ["x"]


def test_a_typed_in_blank_is_refused_unless_the_input_takes_a_blank():
    assert _problems({"nodes": [{"id": "u", "node": "upper", "values": {"text": None}}]}) == [("invalid_value", ("u", "text"))]
    graph = REGISTRY.load({"nodes": [{"id": "m", "node": "maybe", "values": {"text": None}}]})
    assert graph.problems == [] and graph.run().outputs == {"m.result": "blank"}


def test_a_value_its_types_own_constructor_refuses_is_a_problem_where_it_was_typed():
    graph = REGISTRY.load({"nodes": [{"id": "n", "node": "nonempty", "values": {"text": ""}}]})
    [problem] = graph.problems
    assert (problem.type, problem.loc) == ("invalid_value", ("n", "text"))
    assert "empty" not in problem.msg  # the type's own words may quote the value


def test_a_single_value_typed_into_a_series_input_is_refused_not_wrapped_into_a_list_of_one():
    [problem] = REGISTRY.load({"nodes": [{"id": "j", "node": "join", "values": {"texts": "a"}}]}).problems
    assert (problem.type, problem.loc) == ("invalid_value", ("j", "texts"))
    assert problem.ctx["reason"] == "Input should be a valid list"


# -- what the graph offers -------------------------------------------------------------------------


def test_a_graph_input_carries_its_title_type_and_widget_for_a_form():
    described = REGISTRY.load({"nodes": [{"id": "s", "node": "styled"}]}).inputs["mode"].describe()
    assert described["title"] == "Mode" and described["type"] == "str" and described["required"] is False
    assert described["widget"]["choices"] == [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]


def test_graph_inputs_come_in_the_order_of_the_nodes():
    graph = REGISTRY.load({"nodes": [{"id": "s", "node": "styled"}, {"id": "u", "node": "upper"}, {"id": "f", "node": "fan"}, {"id": "g", "node": "greet"}]})
    assert list(graph.inputs) == ["mode", "u.text", "g.text"]


def test_a_node_one_of_whose_outputs_is_read_offers_none_of_its_outputs():
    graph = REGISTRY.load({"nodes": [{"id": "w", "node": "two"}, {"id": "u", "node": "upper"}], "edges": [["w.zed", "u.text"]]})
    assert graph.outputs == ["u.result"]


def test_an_output_a_hook_computed_is_a_graph_output_with_its_title():
    graph = REGISTRY.load({"nodes": [{"id": "c", "node": "columns", "values": {"names": ["net", "vat"]}}]})
    assert graph.problems == [] and graph.outputs == ["c.net", "c.vat"]
    assert [o["title"] for o in graph.describe()["nodes"]["c"]["outputs"]] == ["Net", "Vat"]
    assert graph.run(text="9").outputs == {"c.net": "9", "c.vat": "9"}


def test_an_output_and_an_input_sharing_a_name_are_read_by_address():
    graph = REGISTRY.load({"nodes": [{"id": "g", "node": "greet"}]})
    assert list(graph.inputs) == ["text"] and graph.run(text="hi").outputs == {"g.result": "hi"}
    record = {"nodes": [{"id": "u", "node": "upper"}, {"id": "t", "node": "two"}], "edges": [["u.result", "t.text"]]}
    assert REGISTRY.load(record).run(text="a").outputs == {"t.zed": "zed A", "t.alpha": "alpha A"}


def test_a_description_of_a_broken_graph_is_json_and_carries_its_problems():
    graph = REGISTRY.load({"nodes": [{"id": "n", "node": "nonempty", "values": {"text": ""}}, {"id": "u", "node": "upper"}]})
    described = json.loads(json.dumps(graph.describe(), allow_nan=False))
    assert not graph.is_runnable
    assert [(p["type"], p["loc"]) for p in described["problems"]] == [("invalid_value", ["n", "text"])]
    assert list(described["inputs"]) == ["text"]


# -- what loading refuses --------------------------------------------------------------------------


def test_nodes_reading_each_other_are_a_cycle_on_each_and_the_graph_does_not_run():
    graph = REGISTRY.load({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "tail"}], "edges": [["a.result", "b.text"], ["b.result", "a.text"]]})
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert not graph.is_runnable
    assert [p.type for p in _refused_run(graph)] == ["cycle", "cycle"]


def _refused_run(graph: Any) -> list[Problem]:
    with pytest.raises(InvalidCallError) as refused:
        graph.run()
    return refused.value.errors()


def test_an_unknown_node_is_one_problem_naming_the_nodes_there_are_and_adds_no_inputs_or_outputs():
    graph = registry_of(upper, tail).load({"nodes": [{"id": "x", "node": "uper"}, {"id": "t", "node": "tail"}]})
    [problem] = graph.problems
    assert (problem.type, problem.loc) == ("unknown_node", ("x",))
    assert problem.ctx == {"node": "uper", "choices": "tail, upper"}
    assert list(graph.inputs) == ["text"] and graph.outputs == ["t.result"]


def test_a_version_the_node_does_not_have_is_a_problem_never_the_nearest_version():
    [problem] = REGISTRY.load({"nodes": [{"id": "u", "node": "upper", "version": 2}]}).problems
    assert (problem.type, problem.loc, problem.ctx) == ("unknown_version", ("u",), {"node": "upper", "version": 2, "newest": 1})


def test_a_hook_that_raises_is_a_problem_on_its_node_never_an_exception():
    [problem] = REGISTRY.load({"nodes": [{"id": "r", "node": "refuses"}]}).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("r",), "ValueError")


def test_a_subtype_lands_where_its_parent_is_asked_for_and_not_the_other_way():
    assert _problems({"nodes": [{"id": "n", "node": "name"}, {"id": "u", "node": "upper"}], "edges": [["n.result", "u.text"]]}) == []
    assert _problems({"nodes": [{"id": "u", "node": "upper"}, {"id": "n", "node": "nonempty"}], "edges": [["u.result", "n.text"]]}) == [
        ("type_mismatch", ("n", "text")),
    ]


def test_a_list_of_anything_a_hook_computes_loads_into_a_typed_input_and_a_table_input_without_a_crash():
    into_text = REGISTRY.load({"nodes": [{"id": "a", "node": "anything"}, {"id": "u", "node": "upper"}], "edges": [["a.items", "u.text"]]})
    assert into_text.problems == [] and into_text.run().outputs == {"u.result": ["X", "X"]}
    assert _problems({"nodes": [{"id": "a", "node": "anything"}, {"id": "h", "node": "height"}], "edges": [["a.items", "h.rows"]]}) == []


def test_one_mistyped_edge_among_several_into_a_series_input_is_a_type_mismatch():
    record = {
        "nodes": [{"id": "u", "node": "upper"}, {"id": "f", "node": "fan"}, {"id": "j", "node": "join"}],
        "edges": [["u.result", "j.texts"], ["f.result", "j.texts"]],
    }
    assert _problems(record) == [("type_mismatch", ("j", "texts"))]


def test_an_edge_from_another_nodes_input_is_not_a_source():
    problems = REGISTRY.load({"nodes": [{"id": "h", "node": "upper"}, {"id": "u", "node": "tail"}], "edges": [["h.text", "u.text"]]}).problems
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
    problems = REGISTRY.load({"nodes": [{"id": "a", "node": "upper"}, {"id": "u", "node": "tail"}], "edges": [["a/b.result", "u.text"]]}).problems
    assert [(p.type, p.loc, p.ctx["node_id"]) for p in problems] == [("edge_names_missing_node", ("u", "text"), "a/b")]


# -- what a caller gives ---------------------------------------------------------------------------


def test_with_inputs_takes_an_address_mapping_beside_keywords_and_refuses_one_it_does_not_offer():
    graph = REGISTRY.load({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "tail"}, {"id": "s", "node": "styled"}]})
    assert graph.with_inputs({"a.text": "x", "b.text": "y"}, mode="b").resume().outputs == {"a.result": "X", "b.result": "y.", "s.result": "b"}
    assert _refused(graph, {"a.text": "x", "b.text": "y", "c.text": "z"}, mode="b") == [("not_offered", ("inputs", "c.text"))]


def test_a_bare_name_two_inputs_share_is_refused_naming_the_addresses():
    graph = REGISTRY.load({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "tail"}]})
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(text="x")
    not_offered = [p for p in refused.value.errors() if p.type == "not_offered"]
    assert [p.ctx for p in not_offered] == [{"name": "text", "inputs": "a.text, b.text"}]


def test_with_inputs_refuses_every_fault_at_once_listing_what_it_offers():
    graph = REGISTRY.load({"nodes": [{"id": "s", "node": "styled"}, {"id": "u", "node": "upper", "values": {"text": "fixed"}}]})
    assert list(graph.inputs) == ["mode"]
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(mode=5, text="a fixed input is not offered", typo="x")
    found = sorted((p.type, p.loc) for p in refused.value.errors())
    assert found == [("invalid_input", ("inputs", "mode")), ("not_offered", ("inputs", "text")), ("not_offered", ("inputs", "typo"))]
    assert {p.ctx["inputs"] for p in refused.value.errors() if p.type == "not_offered"} == {"mode"}


# -- embedded graphs -------------------------------------------------------------------------------


def test_an_embedding_offers_its_inner_open_input_and_output_as_one_node():
    assert enrich.title == "Enrich"
    graph = REGISTRY.load({"nodes": [{"id": "e", "node": "enrich"}]})
    assert graph.problems == [] and list(graph.inputs) == ["text"] and graph.inputs["text"].describe()["title"] == "Text"
    run = graph.run(text="hi")
    assert run.outputs == {"e.result": "HI."} and run.results["e"] == {"result": "HI."}


def test_an_embedding_is_wired_by_the_names_it_offers():
    record = {"nodes": [{"id": "g", "node": "greet"}, {"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}], "edges": [["g.result", "e.text"], ["e.result", "u.text"]]}
    graph = REGISTRY.load(record)
    assert graph.problems == [] and graph.run(text="hi").outputs == {"u.result": "HI."}


def test_a_value_typed_on_an_embedding_reaches_the_inner_input_it_offers():
    graph = REGISTRY.load({"nodes": [{"id": "e", "node": "enrich", "values": {"text": "hi"}}]})
    assert graph.problems == [] and graph.inputs == {} and graph.run().outputs == {"e.result": "HI."}


def test_an_unfed_embedding_keeps_its_inner_defaults_and_runs_once():
    graph = REGISTRY.load({"nodes": [{"id": "d", "node": "defaulted"}]})
    assert graph.plan == {"d": "once"}
    assert graph.run().outputs == {"d.result": "inside"}


def test_an_inner_input_with_a_default_is_offered_as_not_required():
    assert REGISTRY.load({"nodes": [{"id": "d", "node": "defaulted"}]}).inputs["text"].required is False
    assert REGISTRY.load({"nodes": [{"id": "e", "node": "enrich"}]}).inputs["text"].required is True


def test_a_graph_inside_a_graph_is_one_node_from_outside():
    run = REGISTRY.load({"nodes": [{"id": "t", "node": "top"}]}).run(text="hi")
    assert run.outputs == {"t.result": "HI"} and run.results["t"] == {"result": "HI"}


def test_renaming_a_node_inside_an_embedding_breaks_no_caller():
    renamed = node("enrich")(GraphRecord(nodes=[NodeRecord(id="shout", node="upper"), NodeRecord(id="end", node="tail")], edges=[("shout.result", "end.text")]))
    record = {"nodes": [{"id": "g", "node": "greet"}, {"id": "e", "node": "enrich"}], "edges": [["g.result", "e.text"]]}
    for embedded in (enrich, renamed):
        graph = registry_of(greet, upper, tail, embedded).load(record)
        assert graph.problems == [] and graph.run(text="hi").outputs[graph.outputs[0]] == "HI."


def test_a_value_on_an_input_the_embedding_does_not_offer_is_an_unknown_input():
    fed_inside = REGISTRY.load({"nodes": [{"id": "e", "node": "enrich", "values": {"text": "a", "result": "b"}}]}).problems
    assert [(p.type, p.loc) for p in fed_inside] == [("unknown_input", ("e", "result"))]
    by_inner_address = REGISTRY.load({"nodes": [{"id": "e", "node": "enrich", "values": {"upper.text": "a"}}]}).problems
    assert [(p.type, p.loc, p.ctx["inputs"]) for p in by_inner_address] == [("unknown_input", ("e", "upper.text"), "text")]


def test_a_table_made_inside_an_embedding_offers_its_columns():
    graph = REGISTRY.load({"nodes": [{"id": "t", "node": "team"}, {"id": "u", "node": "upper"}], "edges": [["t.result.email", "u.text"]]})
    assert graph.problems == []
    assert graph.run(text="name,email\nIda,ida@a.dk\nBo,bo@b.dk\n").outputs == {"u.result": ["IDA@A.DK", "BO@B.DK"]}


def test_a_node_built_from_a_graph_offers_its_inner_inputs_and_carries_its_deprecation():
    graph = REGISTRY.load({"nodes": [{"id": "w", "node": "wrapped"}]})
    assert list(graph.inputs) == ["text"] and graph.run(text="a").outputs == {"w.result": "A"}
    assert wrapped.describe()["deprecation"]["header"] == "Gone"
    assert REGISTRY.describe()["nodes"][[n.id for n in REGISTRY].index("wrapped")]["deprecation"]["header"] == "Gone"


@node("pair")
def pair(a: str, b: str) -> str:
    return a + b


def _hint_for(formula: str) -> dict[str, Any]:
    return {name: float for name in formula.split()}


@node("formula", compute_added_inputs=_hint_for)
def formula(formula: str, **values: float) -> float:
    return sum(values.values())


gone_inside = node("gone_inside")(GraphRecord(nodes=[NodeRecord(id="x", node="gone")]))
around = node("around")(GraphRecord(nodes=[NodeRecord(id="b", node="gone_inside")]))
invalid_inside = node("invalid_inside")(GraphRecord(nodes=[NodeRecord(id="n", node="nonempty", values={"text": ""})]))
unfed_inside = node("unfed_inside")(make_graph(upper, pair, edges=[("upper.result", "pair.a")]))
BROKEN = registry_of(upper, pair, nonempty, formula, gone_inside, around, invalid_inside, unfed_inside)


def test_a_hook_is_not_handed_a_required_input_nobody_typed_in_and_the_node_says_why():
    graph = BROKEN.load({"nodes": [{"id": "f", "node": "formula"}]})
    assert [(p.type, p.loc) for p in graph.problems] == [("hook_needs_value", ("f", "formula"))]
    typed = BROKEN.load({"nodes": [{"id": "f", "node": "formula", "values": {"formula": "a b"}}]})
    assert typed.problems == [] and list(typed.inputs) == ["a", "b"]


def test_a_broken_graph_is_one_problem_where_it_is_embedded_with_its_own_problems_inside():
    graph = BROKEN.load({"nodes": [{"id": "g", "node": "gone_inside"}]})
    [problem] = graph.problems
    assert (problem.type, problem.loc, problem.ctx["graph"], problem.ctx["count"]) == ("embedded_graph_broken", ("g",), "gone_inside", 1)
    assert [(p["type"], p["loc"]) for p in problem.ctx["problems"]] == [("unknown_node", ["x"])]
    assert not graph.is_runnable and graph.inputs == {}


def test_a_broken_graph_two_levels_down_breaks_the_graph_around_it_where_it_is_embedded():
    [problem] = BROKEN.load({"nodes": [{"id": "a", "node": "around"}]}).problems
    assert (problem.type, problem.loc) == ("embedded_graph_broken", ("a",))
    [inside] = problem.ctx["problems"]
    assert (inside["type"], inside["loc"], inside["ctx"]["problems"][0]["type"]) == ("embedded_graph_broken", ["b"], "unknown_node")


def test_an_invalid_value_typed_inside_makes_the_embedding_not_runnable():
    graph = BROKEN.load({"nodes": [{"id": "i", "node": "invalid_inside"}]})
    assert [(p.type, p.loc) for p in graph.problems] == [("embedded_graph_broken", ("i",))]
    assert graph.problems[0].ctx["problems"][0]["type"] == "invalid_value"
    with pytest.raises(InvalidCallError):
        graph.run()


def test_a_graph_with_a_required_input_on_a_later_node_nobody_feeds_cannot_be_embedded():
    [problem] = BROKEN.load({"nodes": [{"id": "u", "node": "unfed_inside"}]}).problems
    assert (problem.type, problem.loc) == ("embedded_graph_broken", ("u",))
    assert [(p["type"], p["loc"]) for p in problem.ctx["problems"]] == [("unbound_on_later_node", ["pair", "b"])]


def test_a_value_typed_on_a_broken_embedding_adds_no_second_problem():
    graph = BROKEN.load({"nodes": [{"id": "g", "node": "gone_inside", "values": {"text": "x"}}]})
    assert [(p.type, p.loc) for p in graph.problems] == [("embedded_graph_broken", ("g",))]
