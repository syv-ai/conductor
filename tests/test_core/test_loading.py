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
    Choice,
    Deprecation,
    GraphRecord,
    InvalidCallError,
    NodeError,
    NodeRecord,
    Param,
    Problem,
    Series,
    Table,
    make_graph,
    node,
)
from conductor.widgets import Dropdown, Textarea
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
    mode: Annotated[str, Param(title="Mode", choices=[Choice(id="a", title="A"), Choice(id="b", title="B")], widget=Dropdown())] = "a",
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
    with pytest.raises(InvalidCallError) as refused:
        REGISTRY.load({"nodes": [{"id": "u", "node": "upper", "valuse": {"text": "x"}}]})
    assert [(p.type, p.loc, p.ctx["error"]) for p in refused.value.errors()] == [("invalid_record", ("record", "nodes", 0, "valuse"), "extra_forbidden")]


@pytest.mark.parametrize(("record", "loc", "error"), [
    ('{"nodes": [', ("record",), "json_invalid"),
    ({"nodes": [{"id": "u", "type": "upper"}]}, ("record", "nodes", 0, "node"), "missing"),
    ({"nodes": [], "edges": [{"from": "a.result", "to": "b.text"}]}, ("record", "edges", 0), "tuple_type"),
    ({"nodes": [], "edges": [["a", "b.text"]]}, ("record", "edges", 0, 0), "value_error"),
])
def test_what_is_not_a_record_is_refused_as_invalid_record_never_a_validation_error(record, loc, error):
    with pytest.raises(InvalidCallError) as refused:
        REGISTRY.load(record)
    problems = refused.value.errors()
    assert all(p.type == "invalid_record" for p in problems) and (problems[0].loc, problems[0].ctx["error"]) == (loc, error)
    assert not isinstance(refused.value, ValidationError) and problems[0].msg.startswith("The record is not a graph record at")


def test_a_half_finished_drawing_with_a_records_shape_loads_with_its_problems():
    graph = REGISTRY.load({"nodes": [{"id": "u", "node": "upper"}, {"id": "x", "node": "nothing"}], "edges": [["u.result", "y.text"]]})
    assert {p.type for p in graph.problems} == {"unknown_node", "edge_names_missing_node"}


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
    described = REGISTRY.load({"nodes": [{"id": "s", "node": "styled"}]}).inputs["mode"]
    assert described.title == "Mode" and described.type == "str" and described.required is False
    assert described.choices == [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]


def test_graph_inputs_come_in_the_order_of_the_nodes():
    graph = REGISTRY.load({"nodes": [{"id": "s", "node": "styled"}, {"id": "u", "node": "upper"}, {"id": "f", "node": "fan"}, {"id": "g", "node": "greet"}]})
    assert list(graph.inputs) == ["mode", "u.text", "g.text"]


def test_a_node_one_of_whose_outputs_is_read_offers_none_of_its_outputs():
    graph = REGISTRY.load({"nodes": [{"id": "w", "node": "two"}, {"id": "u", "node": "upper"}], "edges": [["w.zed", "u.text"]]})
    assert graph.outputs == ["u.result"]


def test_an_output_a_hook_computed_is_a_graph_output_with_its_title():
    graph = REGISTRY.load({"nodes": [{"id": "c", "node": "columns", "values": {"names": ["net", "vat"]}}]})
    assert graph.problems == [] and graph.outputs == ["c.net", "c.vat"]
    assert [o.title for o in graph.describe().nodes["c"].outputs] == ["Net", "Vat"]
    assert graph.run(text="9").outputs == {"c.net": "9", "c.vat": "9"}


def test_an_output_and_an_input_sharing_a_name_are_read_by_address():
    graph = REGISTRY.load({"nodes": [{"id": "g", "node": "greet"}]})
    assert list(graph.inputs) == ["text"] and graph.run(text="hi").outputs == {"g.result": "hi"}
    record = {"nodes": [{"id": "u", "node": "upper"}, {"id": "t", "node": "two"}], "edges": [["u.result", "t.text"]]}
    assert REGISTRY.load(record).run(text="a").outputs == {"t.zed": "zed A", "t.alpha": "alpha A"}


def test_a_description_of_a_broken_graph_is_json_and_carries_its_problems():
    graph = REGISTRY.load({"nodes": [{"id": "n", "node": "nonempty", "values": {"text": ""}}, {"id": "u", "node": "upper"}]})
    described = json.loads(graph.describe().model_dump_json())
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
    assert problem.ctx == {"node": "uper", "choices": ["tail", "upper"]}
    assert list(graph.inputs) == ["text"] and graph.outputs == ["t.result"]


def test_a_version_the_node_does_not_have_is_a_problem_never_the_nearest_version():
    [problem] = REGISTRY.load({"nodes": [{"id": "u", "node": "upper", "version": 2}]}).problems
    assert (problem.type, problem.loc, problem.ctx) == ("unknown_version", ("u",), {"node": "upper", "version": 2, "newest": 1})


def test_a_hook_that_raises_is_a_problem_on_its_node_never_an_exception():
    [problem] = REGISTRY.load({"nodes": [{"id": "r", "node": "refuses"}]}).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("r",), "ValueError")


def test_a_value_its_type_refuses_never_reaches_a_hook_and_is_said_once():
    # "refuses" reads its mode in a hook; a list typed into the str would have reached it
    problems = REGISTRY.load({"nodes": [{"id": "r", "node": "refuses", "values": {"mode": ["a", "b"]}}]}).problems
    assert [(p.type, p.loc) for p in problems] == [("invalid_value", ("r", "mode"))]


def test_a_list_typed_into_an_input_a_hook_reads_is_refused_saying_why() -> None:
    def named(names: str) -> dict[str, Any]:
        return {name: bool for name in names.split(",")}

    @node("checks", compute_outputs=named)
    def checks(names: str) -> dict[str, Any]:
        return {}

    (problem,) = registry_of(checks).load({"nodes": [{"id": "c", "node": "checks", "values": {"names": ["a,b", "c"]}}]}).problems
    assert (problem.type, problem.loc) == ("invalid_value", ("c", "names"))
    assert problem.ctx["reason"] == "a hook works out this node's fields from one value; a list can't be typed in here"


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
    assert [p.ctx for p in not_offered] == [{"name": "text", "inputs": ["a.text", "b.text"]}]


def test_with_inputs_refuses_every_fault_at_once_listing_what_it_offers():
    graph = REGISTRY.load({"nodes": [{"id": "s", "node": "styled"}, {"id": "u", "node": "upper", "values": {"text": "fixed"}}]})
    assert list(graph.inputs) == ["mode"]
    with pytest.raises(InvalidCallError) as refused:
        graph.with_inputs(mode=5, text="a fixed input is not offered", typo="x")
    found = sorted((p.type, p.loc) for p in refused.value.errors())
    assert found == [("invalid_input", ("inputs", "mode")), ("not_offered", ("inputs", "text")), ("not_offered", ("inputs", "typo"))]
    assert [p.ctx["inputs"] for p in refused.value.errors() if p.type == "not_offered"] == [["mode"], ["mode"]]


# -- embedded graphs -------------------------------------------------------------------------------


def test_an_embedding_offers_its_inner_open_input_and_output_as_one_node():
    assert enrich.title == "Enrich"
    graph = REGISTRY.load({"nodes": [{"id": "e", "node": "enrich"}]})
    assert graph.problems == [] and list(graph.inputs) == ["text"] and graph.inputs["text"].title == "Text"
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
    assert {k: v.msg for k, v in graph.plan.items()} == {"d": "once"}
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
    assert [(p.type, p.loc, p.ctx["inputs"]) for p in by_inner_address] == [("unknown_input", ("e", "upper.text"), ["text"])]


def test_a_table_made_inside_an_embedding_offers_its_columns():
    graph = REGISTRY.load({"nodes": [{"id": "t", "node": "team"}, {"id": "u", "node": "upper"}], "edges": [["t.result.email", "u.text"]]})
    assert graph.problems == []
    assert graph.run(text="name,email\nIda,ida@a.dk\nBo,bo@b.dk\n").outputs == {"u.result": ["IDA@A.DK", "BO@B.DK"]}


def test_a_node_built_from_a_graph_offers_its_inner_inputs_and_carries_its_deprecation():
    graph = REGISTRY.load({"nodes": [{"id": "w", "node": "wrapped"}]})
    assert list(graph.inputs) == ["text"] and graph.run(text="a").outputs == {"w.result": "A"}
    assert wrapped.describe().deprecation.header == "Gone"
    assert REGISTRY.describe().nodes[[n.id for n in REGISTRY].index("wrapped")].deprecation.header == "Gone"


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


def test_an_input_a_hook_reads_is_fixed_when_the_graph_loads_and_never_a_graph_input():
    """The fields a hook works out hold for every run, so what it read is the record's, never the caller's."""

    def outputs(mode: str) -> dict[str, Any]:
        return {"a": int} if mode == "a" else {"b": str}

    @node("by-mode", compute_outputs=outputs)
    def by_mode(mode: str = "a") -> dict[str, Any]:
        return {"a": 1} if mode == "a" else {"b": "x"}

    registry = registry_of(by_mode)
    graph = registry.load({"nodes": [{"id": "m", "node": "by-mode"}]})
    assert graph.problems == [] and graph.inputs == {} and graph.run().outputs == {"m.a": 1}
    with pytest.raises(InvalidCallError):
        graph.run(mode="b")

    [problem] = registry.load({"nodes": [{"id": "m", "node": "by-mode", "defaults": {"mode": "b"}}]}).problems
    assert (problem.type, problem.loc) == ("default_on_later_node", ("m", "mode"))
    assert BROKEN.load({"nodes": [{"id": "f", "node": "formula"}]}).problems[0].type == "hook_needs_value"


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


@node("both_halves")
def both_halves(a: str, b: str) -> str:
    return a + b


def _taken(formula: str) -> dict[str, Any]:
    return {"formula": float}


@node("taken", compute_added_inputs=_taken)
def taken(formula: str = "", **values: float) -> float:
    return 0.0


def _refuses_bad(mode: str) -> dict[str, Any]:
    if mode == "bad":
        raise ValueError("Not that mode.")
    return {"result": str}


@node("refuses_bad", compute_outputs=_refuses_bad)
def refuses_bad(mode: str = "ok") -> dict[str, Any]:
    return {"result": mode}


halves = node("halves")(GraphRecord(
    nodes=[NodeRecord(id="x", node="upper"), NodeRecord(id="y", node="upper"), NodeRecord(id="b", node="both_halves")],
    edges=[("x.result", "b.a"), ("y.result", "b.b")],
))
picky = node("picky")(make_graph(refuses_bad))
OUTSIDE = registry_of(upper, tail, refuses_bad, both_halves, taken, enrich, halves, picky)


def test_only_the_nodes_on_a_cycle_are_a_cycle_and_a_node_below_keeps_its_fields():
    graph = REGISTRY.load({
        "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "tail"}],
        "edges": [["a.result", "b.text"], ["b.result", "a.text"], ["b.result", "c.text"]],
    })
    assert [(p.type, p.loc) for p in graph.problems] == [("cycle", ("a",)), ("cycle", ("b",))]
    assert set(graph.describe().nodes) == {"a", "b", "c"}


def test_an_editor_reads_every_node_of_a_half_broken_graph_and_which_ones_loaded():
    graph = REGISTRY.load({
        "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "tail"}, {"id": "x", "node": "gone"}],
        "edges": [["a.nope", "b.text"], ["b.result", "c.text"]],
    })
    drawn = json.loads(graph.describe().model_dump_json())["nodes"]
    assert list(drawn) == ["a", "b", "c", "x"]
    assert {node_id: d["loaded"] for node_id, d in drawn.items()} == {"a": True, "b": False, "c": False, "x": False}


def test_an_input_a_hook_adds_under_a_name_the_node_declares_is_the_hooks_failure():
    [problem] = OUTSIDE.load({"nodes": [{"id": "d", "node": "taken", "values": {"formula": "x"}}]}).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("d",), "added_input_taken")


def test_a_wrong_address_on_an_embedding_is_unknown_output_or_unknown_input_on_it():
    def found(record: dict[str, Any]) -> list[tuple[str, tuple[Any, ...]]]:
        return [(p.type, p.loc) for p in OUTSIDE.load(record).problems]

    nodes = [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}]
    assert found({"nodes": nodes, "edges": [["e.nope", "u.text"]]}) == [("unknown_output", ("u", "text"))]
    assert found({"nodes": nodes, "edges": [["e.upper.result", "u.text"]]}) == [("unknown_output", ("u", "text"))]
    assert found({"nodes": nodes[::-1], "edges": [["u.result", "e.tail"]]}) == [("unknown_input", ("e", "tail"))]


@pytest.mark.parametrize("ctx", [{"tables": [1, 2]}, {"source": "e/report.pdf", "node_id": "anything"}])
def test_an_inner_nodes_own_error_is_said_of_its_embedding_with_its_ctx_as_its_author_wrote_it(ctx):
    @node("unreadable")
    def unreadable(text: str) -> str:
        raise NodeError("The document could not be read.", code="unreadable", ctx=ctx)

    embedding = node("e")(make_graph(unreadable))
    run = registry_of(unreadable, embedding).load(make_graph(embedding)).run(text="hi")

    assert (run.cause.type, run.cause.loc, run.cause.ctx) == ("unreadable", ("e",), ctx)


def test_an_outer_edge_never_reads_inside_an_embedding_by_an_inner_id():
    problems = OUTSIDE.load({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}], "edges": [["e/upper.result", "u.text"]]}).problems
    assert [(p.type, p.loc, p.ctx["node_id"]) for p in problems] == [("edge_names_missing_node", ("u", "text"), "e/upper")]


def test_an_edge_from_a_missing_node_into_an_embedding_is_its_problem_at_the_field_it_offers():
    [problem] = OUTSIDE.load({"nodes": [{"id": "e", "node": "enrich"}], "edges": [["ghost.result", "e.text"]]}).problems
    assert (problem.type, problem.loc, problem.ctx["node_id"]) == ("edge_names_missing_node", ("e", "text"), "ghost")
    assert (problem.ctx["source"], problem.ctx["target"]) == ("ghost.result", "e.text")


def test_a_hook_inside_refusing_a_value_typed_on_the_embedding_is_the_embeddings_problem():
    [problem] = OUTSIDE.load({"nodes": [{"id": "p", "node": "picky", "values": {"mode": "bad"}}]}).problems
    assert (problem.type, problem.loc) == ("compute_hook_failed", ("p",))


def test_an_embeddings_inputs_are_filled_by_its_own_address():
    graph = OUTSIDE.load({"nodes": [{"id": "e", "node": "enrich"}, {"id": "u", "node": "upper"}]})
    assert {name: i.address for name, i in graph.inputs.items()} == {"e.text": "e.text", "u.text": "u.text"}
    assert graph.run({"e.text": "hi", "u.text": "yo"}).outputs == {"e.result": "HI.", "u.result": "YO"}


def test_two_inner_inputs_sharing_a_name_are_offered_by_the_inner_records_ids():
    assert [f.name for f in next(n for n in OUTSIDE.describe().nodes if n.id == "halves").inputs] == ["x/text", "y/text"]
    graph = OUTSIDE.load({"nodes": [{"id": "s", "node": "upper"}, {"id": "c", "node": "halves"}], "edges": [["s.result", "c.x/text"]]})
    assert graph.problems == []
    assert graph.run({"s.text": "a", "c.y/text": "b"}).outputs == {"c.result": "AB"}


def test_an_authored_id_with_a_slash_beside_an_embedding_is_refused_alone_and_stays_the_authors():
    beside = OUTSIDE.load({"nodes": [{"id": "e", "node": "enrich"}, {"id": "e/upper", "node": "upper"}]}).problems
    assert [(p.type, p.loc) for p in beside] == [("invalid_node_id", ("e/upper",))]
    broken = REGISTRY.load({"nodes": [{"id": "e", "node": "gone"}, {"id": "e/upper", "node": "upper"}]}).problems
    assert sorted((p.type, p.loc) for p in broken) == [("invalid_node_id", ("e/upper",)), ("unknown_node", ("e",))]


def test_a_set_given_for_an_input_of_one_value_is_refused_as_it_has_no_order():
    from conductor import NodeRegistry, make_graph, node

    @node("send")
    def send(contact: str) -> str:
        return contact

    registry = NodeRegistry()
    registry.add(send)
    graph = registry.load(make_graph(send))
    with pytest.raises(InvalidCallError) as refused:
        graph.run(contact={"b", "a"})
    assert "a set has no order" in refused.value.errors()[0].msg
    assert graph.run(contact=["b", "a"]).outputs == {"send.result": ["b", "a"]}


def test_graphs_embedded_forty_deep_load_at_once():
    """Each embedded graph is loaded alone once per load, however deep it nests, so depth costs time in proportion."""
    import time

    from conductor import NodeRegistry, node

    registry = NodeRegistry()

    @registry.node("upper")
    def upper(text: str) -> str:
        return text.upper()

    inner = "upper"
    for depth in range(40):
        registry.add(node(f"nest-{depth}")(GraphRecord(nodes=[NodeRecord(id="n", node=inner)])))
        inner = f"nest-{depth}"
    started = time.monotonic()
    graph = registry.load({"nodes": [{"id": "top", "node": inner}]})
    assert time.monotonic() - started < 2 and graph.run(text="z").outputs == {"top.result": "Z"}


def test_describe_lists_each_nodes_problems_on_it():
    described = REGISTRY.load({"nodes": [{"id": "u", "node": "upper"}, {"id": "x", "node": "uper"}]}).describe()
    assert [p.type for p in described.nodes["x"].problems] == ["unknown_node"] and described.nodes["u"].problems == []
    assert described.problems == described.nodes["x"].problems


def test_a_type_mismatch_into_a_series_says_the_series_it_reads():
    from conductor import NodeRegistry, Series, make_graph, node

    @node("words")
    def words(text: str) -> Series[str]:
        return text.split()

    @node("count")
    def count(flags: Series[bool]) -> int:
        return sum(flags)

    registry = NodeRegistry()
    registry.add(words, count)
    (problem,) = registry.load({"nodes": [{"id": "w", "node": "words"}, {"id": "c", "node": "count"}], "edges": [["w.result", "c.flags"]]}).problems
    assert (problem.type, problem.ctx["got"], problem.ctx["expected"]) == ("type_mismatch", "Series[str]", "Series[bool]")
    with pytest.raises(InvalidCallError) as refused:
        make_graph(words, count)
    assert [p.type for p in refused.value.errors()] == ["type_mismatch"]


def test_what_the_outside_gives_an_embedded_input_replaces_the_inner_default():
    from conductor import NodeRegistry, make_graph, node

    @node("upper")
    def upper(text: str) -> str:
        return text.upper()

    @node("hello")
    def hello() -> str:
        return "hi"

    registry = NodeRegistry()
    registry.add(upper, hello, node("shout")(make_graph(upper, defaults={"upper.text": "inner"})))
    shout = registry["shout"]
    assert registry.load(make_graph(shout)).run().outputs == {"shout.result": "INNER"}
    typed = registry.load(make_graph(shout, values={"shout.text": "outer"}))
    assert typed.problems == [] and typed.run().outputs == {"shout.result": "OUTER"}
    defaulted = registry.load(make_graph(shout, defaults={"shout.text": "outer"}))
    assert defaulted.problems == [] and defaulted.run().outputs == {"shout.result": "OUTER"}
    fed = registry.load(make_graph(hello, shout, edges=[("hello.result", "shout.text")]))
    assert fed.problems == [] and fed.run().outputs == {"shout.result": "HI"}


def test_a_yes_or_no_never_lands_on_a_number_and_a_union_is_said_by_its_members_names():
    from conductor import NodeRegistry, node

    class Text(str):
        pass

    @node("flag")
    def flag() -> bool:
        return True

    @node("maybe")
    def maybe() -> Text | None:
        return None

    @node("double")
    def double(x: float) -> float:
        return x * 2

    registry = NodeRegistry()
    registry.add(flag, maybe, double)
    for source, got in (("flag", "bool"), ("maybe", "Text | None")):
        (problem,) = registry.load({"nodes": [{"id": "s", "node": source}, {"id": "d", "node": "double"}], "edges": [["s.result", "d.x"]]}).problems
        assert (problem.type, problem.ctx["got"]) == ("type_mismatch", got)


def test_a_graph_input_is_its_fields_record_and_writes_as_json():
    import json

    from conductor import GraphInput, NodeRegistry, make_graph, node

    @node("u")
    def u(t: str, n: int = 1) -> str:
        return t

    registry = NodeRegistry()
    registry.add(u)
    graph = registry.load(make_graph(u))
    assert isinstance(graph.inputs["t"], GraphInput) and graph.describe().inputs == graph.inputs
    assert json.loads(json.dumps({name: i.default for name, i in graph.inputs.items()})) == {"t": None, "n": 1}
    assert (graph.inputs["t"].name, graph.inputs["t"].address, graph.inputs["t"].required) == ("t", "u.t", True)


def test_a_yes_or_no_typed_in_or_given_never_lands_on_a_number():
    from conductor import NodeRegistry, node

    @node("double")
    def double(x: float) -> float:
        return x * 2

    registry = NodeRegistry()
    registry.add(double)
    (problem,) = registry.load({"nodes": [{"id": "d", "node": "double", "values": {"x": True}}]}).problems
    assert (problem.type, problem.ctx["reason"]) == ("invalid_value", "a yes or no is not a number; give a number")
    with pytest.raises(InvalidCallError) as refused:
        registry.load({"nodes": [{"id": "d", "node": "double"}]}).run(x=True)
    assert refused.value.errors()[0].type == "invalid_input"
    assert registry.load({"nodes": [{"id": "d", "node": "double"}]}).run(x=2).outputs == {"d.result": 4.0}


def test_a_table_into_a_series_is_said_as_the_table_it_is():
    import polars as pl
    from conductor import NodeRegistry, Series, Table, node

    @node("sheet")
    def sheet() -> Table:
        return pl.DataFrame({"a": [1]})

    @node("count")
    def count(items: Series[int]) -> int:
        return len(items)

    registry = NodeRegistry()
    registry.add(sheet, count)
    (problem,) = registry.load({"nodes": [{"id": "s", "node": "sheet"}, {"id": "c", "node": "count"}], "edges": [["s.result", "c.items"]]}).problems
    assert (problem.ctx["got"], problem.ctx["expected"]) == ("Table", "Series[int]")


def test_a_yes_or_no_never_lands_on_a_hosts_number_type_whose_own_rule_lets_an_int_in():
    from conductor import Ask, NodeRegistry, make_graph

    class Number(float):
        id = "number"

        @classmethod
        def accepts(cls, source: type) -> bool:
            return issubclass(source, int)  # a host's rule that forgot bool

    registry = NodeRegistry(types=[Number, bool])

    @registry.node("yes")
    def yes(value: bool = True) -> bool:
        return value

    @registry.node("half")
    def half(value: Number) -> Number:
        return Number(value / 2)

    @registry.node("ask_number")
    def ask_number() -> Number:
        raise Ask("How many?")

    assert "number" not in registry.accepted_as(bool) and "number" in registry.accepted_as(int)
    mistyped = {"nodes": [{"id": "yes", "node": "yes"}, {"id": "half", "node": "half"}], "edges": [["yes.result", "half.value"]]}
    assert [p.type for p in registry.load(mistyped).problems] == ["type_mismatch"]
    (problem,) = registry.load({"nodes": [{"id": "h", "node": "half", "values": {"value": True}}]}).problems
    assert problem.type == "invalid_value"
    with pytest.raises(InvalidCallError):
        registry.load(make_graph(half)).run(value=True)
    waiting = registry.load(make_graph(ask_number)).run()
    with pytest.raises(InvalidCallError):
        waiting.answer(waiting.waiting[0], True)


@pytest.mark.parametrize("default", ["a", ["a", "b"]], ids=["one value", "a list"])
def test_a_loaded_graph_is_compiled_once_and_freed_without_the_cycle_collector(default: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import gc
    import weakref

    from conductor._internal import graph as graph_module

    @node("upper")
    def upper(text: str) -> str:
        return text.upper()

    loads: list[object] = []
    load = graph_module.Graph._load

    def counted(self: Any, *args: Any, **kwargs: Any) -> None:
        loads.append(self)
        load(self, *args, **kwargs)

    monkeypatch.setattr(graph_module.Graph, "_load", counted)
    registry = registry_of(upper)
    gc.disable()
    try:
        graph = registry.load({"nodes": [{"id": "u", "node": "upper", "defaults": {"text": default}}]})
        listed, compiled_twice = graph._listed, any(compiled is graph for compiled in graph._compiles.values())
        held, loaded = weakref.ref(graph), len(loads)
        del graph, loads[:]
        assert (loaded, compiled_twice, listed) == (1, False, frozenset() if default == "a" else frozenset({"u.text"}))
        assert held() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("ran", ["one value", "a list", "again with a list"])
def test_a_run_and_the_compile_it_ran_on_are_freed_without_the_cycle_collector(ran: str) -> None:
    import gc
    import weakref

    @node("upper")
    def upper(text: str) -> str:
        return text.upper()

    @node("join")
    def join(texts: Series[str]) -> str:
        return "+".join(texts)

    registry = registry_of(upper, join)
    gc.collect()
    gc.disable()
    try:
        graph = registry.load(make_graph(upper, join))
        if ran == "one value":
            run = graph.run(text="a")
        elif ran == "a list":
            run = graph.run(text=["a", "b"])
        else:
            run = graph.run(text=["a", "b"]).again(text=["c"])
        held = [weakref.ref(graph), weakref.ref(run.graph)]
        del graph, run
        assert [ref() for ref in held] == [None, None]
    finally:
        gc.enable()


class _Contact(TypedDict):
    name: str
    email: str


class _Named(TypedDict):
    name: str


class _Count(TypedDict):
    name: int


def test_a_typeddict_lands_on_one_whose_keys_it_holds_with_types_that_fit():
    """A TypedDict is read by its keys, as a type checker reads one."""

    @node("contact")
    def contact() -> _Contact:
        return {"name": "Bo", "email": "bo@example.com"}

    @node("greet")
    def greet(who: _Named) -> str:
        return who["name"]

    @node("count")
    def count(who: _Count) -> int:
        return who["name"]

    registry = registry_of(contact, greet, count)
    fits = registry.load({"nodes": [{"id": "c", "node": "contact"}, {"id": "g", "node": "greet"}], "edges": [["c.result", "g.who"]]})
    misfits = registry.load({"nodes": [{"id": "c", "node": "contact"}, {"id": "n", "node": "count"}], "edges": [["c.result", "n.who"]]})

    assert fits.problems == [] and fits.run().results["g"]["result"] == "Bo"
    assert [p.type for p in misfits.problems] == ["type_mismatch"]
    assert {n.id for n in registry.describe().nodes} == {"contact", "greet", "count"}


def _named_inputs(names: list[str] = []) -> dict[str, Any]:  # noqa: B006
    return {name: str for name in names}


@node("adds", compute_added_inputs=_named_inputs)
def adds(names: list[str] = [], **values: Any) -> str:  # noqa: B006
    return ",".join(f"{k}={v}" for k, v in values.items())


def test_an_embedding_offers_the_fields_its_graph_offers_alone_as_its_hooks_made_them():
    added = GraphRecord(nodes=[NodeRecord(id="a", node="adds", values={"names": ["who"]})])
    made = GraphRecord(nodes=[NodeRecord(id="c", node="columns", values={"names": ["x", "y"]})])
    registry = registry_of(adds, columns, node("box_in")(added), node("box_out")(made))
    assert list(registry.load(added).inputs) == ["who"]
    graph = registry.load({"nodes": [{"id": "b", "node": "box_in"}]})
    assert graph.problems == [] and list(graph.inputs) == ["who"] and graph.run(who="x").outputs == {"b.result": "who=x"}
    graph = registry.load({"nodes": [{"id": "b", "node": "box_out"}]})
    assert graph.outputs == ["b.x", "b.y"] and graph.run(text="hi").outputs == {"b.x": "hi", "b.y": "hi"}
