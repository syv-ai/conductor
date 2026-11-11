"""What a node author declares with ``@node``, and what a host reads back from it: fields, hooks, versions, widgets."""
import dataclasses
import datetime
import json
import logging
import typing
import warnings
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pydantic
import pytest
from conductor import (
    Choice,
    Deprecation,
    ExternalError,
    FromRun,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeRecord,
    NodeRegistry,
    Param,
    Policy,
    Series,
    Table,
    Upgrade,
    host_type,
    make_graph,
    node,
    stored_graph,
)
from conductor._internal.value_type import FieldValueError
from conductor.widgets import Dropdown, FileUpload, ListInput, Textarea, TextInput, Widget
from pydantic import BaseModel, ValidationError


def registry_of(*definitions):
    held = NodeRegistry()
    held.add(*definitions)
    return held


def load(*definitions, record=None, values=None):
    """The graph of ``record``, or of ``definitions`` chained, loaded against exactly ``definitions``."""
    return registry_of(*definitions).load(record if record is not None else make_graph(*definitions, values=values or {}))


def placed(node_id, kind, **record):
    return NodeRecord(id=node_id, node=kind, **record)


def codes(graph):
    return [p.type for p in graph.problems]


def declared(function, **options):
    """``function`` as a node ``x``; a declaration fault raises here."""
    title = function.__name__.replace("_", " ").strip().capitalize()
    return node("x", **{"title": title, "description": "For a test.", **options})(function)


def field_of(definition, name, side="inputs"):
    return next(f for f in getattr(definition.describe(), side) if f.name == name)


@host_type("Text", title="Text")
class Text(str):
    pass


@host_type("Whole", title="Whole")
class Whole(int):
    pass


class Invoice(TypedDict):
    name: str
    amount: int


# -- declaration faults ------------------------------------------------------------------------


def test_a_wrong_declaration_is_a_type_error_with_a_code_naming_the_node_and_function():
    def bad(text: str):
        return text

    with pytest.raises(NodeDeclarationError) as raised:
        declared(bad)
    assert isinstance(raised.value, TypeError)
    assert raised.value.code == "output_without_type"
    assert "@node('x') on bad" in str(raised.value)


def test_a_node_id_must_have_something_in_it_and_no_mark_an_address_or_waiting_id_uses():
    for wrong in ("", "a.b", "a/b", "a@b", "a~b"):
        with pytest.raises(NodeDeclarationError) as raised:
            node(wrong, title="Wrong", description="For a test.")
        assert raised.value.code == "invalid_node_id"

    def amount() -> int:
        return 1

    assert node("Beløb i kr", title="Amount", description="For a test.")(amount).id == "Beløb i kr"


def test_node_given_a_record_in_place_of_a_function_raises_pythons_own_error_that_it_is_not_callable():
    with pytest.raises(TypeError, match="is not a callable object"):
        node("enrich", title="Enrich", description="For a test.")(GraphRecord(nodes=[]))  # type: ignore[arg-type]



@pytest.mark.parametrize(("title", "description", "empty"), [
    ("", "Does something.", "title"),
    ("Something", "", "description"),
    ("   ", "Does something.", "title"),
    ("Something", "   ", "description"),
])
def test_an_empty_title_or_description_is_refused_where_a_node_or_a_stored_graph_is_declared(title, description, empty):
    for declaration in (
        lambda: node("x", title=title, description=description),
        lambda: stored_graph("x", title=title, description=description),
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            declaration()
        assert raised.value.code == "empty_title_or_description"
        assert f"('x') has an empty {empty}" in str(raised.value)


def test_a_node_id_in_a_record_with_a_tilde_is_invalid_node_id():
    registry = NodeRegistry()

    @registry.node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    loaded = registry.load(GraphRecord.model_validate({"nodes": [{"id": "a~b", "node": "upper"}]}))
    assert [(p.type, p.loc) for p in loaded.problems] == [("invalid_node_id", ("a~b",))]


def test_an_async_function_is_refused():
    async def fetch(url: str) -> str:
        return url

    with pytest.raises(NodeDeclarationError, match="@node\\('x'\\) on fetch") as raised:
        declared(fetch)
    assert raised.value.code == "async_run"


def test_a_default_its_type_cannot_read_fails_where_it_is_written():
    def f(x: int = "abc") -> int:
        return x

    with pytest.raises(NodeDeclarationError, match="'x'|x:") as raised:
        declared(f)
    assert raised.value.code == "default_invalid"


def test_a_default_outside_its_own_bounds_fails_where_it_is_written():
    def f(x: Annotated[int, Param(ge=0)] = -1) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.code == "default_invalid"


def test_a_series_default_is_held_to_the_limits_of_each_item():
    def f(xs: Annotated[Series[int], Param(ge=0)] = [1, -1]) -> int:  # noqa: B006
        return sum(xs)

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.code == "default_invalid"


def test_an_unknown_param_argument_is_pythons_own_type_error():
    with pytest.raises(TypeError, match="'titel'"):
        Param(titel="Times")  # type: ignore[call-arg]


def test_a_bound_written_wrong_is_refused_where_it_is_written():
    def on_text(x: Annotated[str, Param(ge=1)]) -> str:
        return x

    def on_table(x: Annotated[Table, Param(min_length=1)]) -> int:
        return 0

    def on_text_or_blank(x: Annotated[str | None, Param(ge=0)] = None) -> str:
        return ""

    def on_texts(x: Annotated[Series[str], Param(ge=0)]) -> str:
        return ""

    for wrong in (on_text, on_table, on_text_or_blank, on_texts):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(wrong)
        assert raised.value.code == "bound_invalid"


def test_a_dict_returned_is_one_output_and_only_a_loose_dataclass_is_refused():
    def summary(labels: Series[str]) -> dict[str, int]:
        return {label: labels.count(label) for label in labels}

    def anything(x: int) -> Any:
        return x

    for one in (summary, anything):
        assert list(declared(one).outputs) == ["result"], one
    assert load(declared(summary)).run(labels=["a", "b", "a"]).outputs == {"x.result": {"a": 2, "b": 1}}

    @dataclass
    class Loose:
        a: int

    def loose(x: int) -> Loose:
        return Loose(x)

    with pytest.raises(NodeDeclarationError) as raised:
        declared(loose)
    assert raised.value.code == "outputs_not_frozen"


def test_a_class_pydantic_cannot_check_is_refused_where_the_class_is_declared():
    with pytest.raises(NodeDeclarationError) as raised:

        @host_type("thing", title="Thing")
        class Thing:
            pass

    assert raised.value.code == "type_not_carried"


def test_a_series_of_tables_is_accepted_as_many_tables():
    def f(x: Series[Table[Invoice]]) -> int:
        return 1

    field = declared(f).inputs["x"]
    assert field.many is True and field.type.id == "Table[name: str, amount: int]"


def test_a_definition_is_frozen():
    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    with pytest.raises(dataclasses.FrozenInstanceError):
        upper.title = "Shout"  # type: ignore[misc]


# -- a node and what it declares -------------------------------------------------------------


def test_a_node_is_called_like_its_function_and_declares_from_its_signature():
    @node("split_lines", title="Split lines", description="Splits a text into lines.")
    def split_lines(text: str) -> Series[str]:
        """Splits a text into lines.

        One per line break."""
        return text.splitlines()

    assert split_lines("a\nb") == ["a", "b"]
    described = split_lines.describe()
    assert (described.title, described.description, described.category, described.version) == ("Split lines", "Splits a text into lines.", "", 1)
    assert [(f.name, f.required) for f in described.inputs] == [("text", True)]
    assert [f.name for f in described.outputs] == ["result"]


def test_describe_is_the_node_as_json_with_no_keys_beyond_its_own():
    @node("v", title="V1", description="For a test.")
    def v1(text: str) -> str:
        return text

    @node("v", title="V2", description="For a test.", replaces=v1, tags=("words",), policy=Policy(retries=2))
    def v2(text: Annotated[str, Param(widget=Textarea(rows=2))]) -> Text:
        return Text(text)

    described = json.loads(v2.describe().model_dump_json())
    assert set(described) == {
        "id", "version", "versions", "title", "description", "category", "tags", "extra", "docs", "deprecation", "kind",
        "policy", "inputs", "outputs", "takes_added_inputs", "computes_outputs",
    }
    assert (described["id"], described["version"], described["versions"], described["tags"]) == ("v", 2, [2, 1], ["words"])
    assert described["kind"] == "function" and described["policy"]["retries"] == 2
    assert described["inputs"][0]["widget"] == {"type": "textarea", "rows": 2}
    assert described["outputs"][0]["type"] == "Text"


def test_a_sink_node_declares_no_outputs_and_runs():
    seen = []

    @node("log", title="Log", description="For a test.")
    def log(text: str) -> None:
        seen.append(text)

    graph = load(log)
    assert log.outputs == {} and graph.problems == [] and graph.outputs == []
    assert graph.run(text="hi").status == "complete" and seen == ["hi"]


def test_each_of_several_outputs_carries_its_own_type_and_title():
    @dataclass(frozen=True)
    class Parts:
        name: str
        count: int | None
        case_sensitive: bool
        renamed: Annotated[float, Param(title="Score")]

    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Parts:
        return Parts(text, None, True, 1.0)

    outputs = {f.name: f for f in parts.describe().outputs}
    assert {n: f.type for n, f in outputs.items()} == {"name": "str", "count": "int", "case_sensitive": "bool", "renamed": "float"}
    assert {n: f.title for n, f in outputs.items()} == {"name": "Name", "count": "Count", "case_sensitive": "Case sensitive", "renamed": "Score"}
    # the field typed `| None` takes a blank, its sibling doesn't
    assert {"type": "null"} in outputs["count"].json_schema["anyOf"] and "anyOf" not in outputs["name"].json_schema
    assert load(parts).run(text="a").results["parts"] == {"name": "a", "count": None, "case_sensitive": True, "renamed": 1.0}


def test_an_untitled_field_reads_as_its_name_in_everyday_words():
    @node("f", title="F", description="For a test.")
    def f(case_sensitive: bool, texts: Series[str], titled: Annotated[str, Param(title="Given")]) -> str:
        return ""

    assert [i.title for i in f.describe().inputs] == ["Case sensitive", "Texts", "Given"]


# -- the type of a field -----------------------------------------------------------------------


def test_the_bare_type_is_what_a_field_is_typed_as_when_it_may_be_blank():
    @node("f", title="F", description="For a test.")
    def f(many: Series[Text | None], one: Text | None) -> Text | None:
        return None

    assert [(i.name, i.type, i.many) for i in f.describe().inputs] == [("many", "Text", True), ("one", "Text", False)]
    output = field_of(f, "result", "outputs")
    assert output.type == "Text" and {"type": "null"} in output.json_schema["anyOf"]
    assert load(f, record=GraphRecord(nodes=[placed("f", "f", values={"many": [None], "one": None})])).run().results["f"] == {"result": None}


def test_a_none_default_with_a_blank_hint_is_optional_and_arrives_as_none():
    @node("f", title="F", description="For a test.")
    def f(x: int | None = None, picked: Annotated[str | None, Param(widget=TextInput())] = None) -> str:
        return f"{x}-{picked}"

    assert [i.required for i in f.describe().inputs] == [False, False]
    assert load(f).run().results["f"]["result"] == "None-None"


def test_a_series_that_may_be_none_takes_none_given_typed_in_or_left_out_and_its_items_are_never_blank():
    @node("f", title="F", description="For a test.")
    def f(xs: Series[int] | None = None) -> str:
        return repr(xs)

    graph = load(f)
    assert graph.run().results["f"]["result"] == "None"
    assert graph.run(xs=None).results["f"]["result"] == "None"
    assert load(f, record=GraphRecord(nodes=[placed("f", "f", values={"xs": None})])).run().results["f"]["result"] == "None"
    assert graph.run(xs=[1, 2]).results["f"]["result"] == "[1, 2]"
    with pytest.raises(InvalidCallError):
        graph.run(xs=[1, None])


def test_a_series_that_may_not_be_none_refuses_none_and_a_default_of_none():
    @node("f", title="F", description="For a test.")
    def f(xs: Series[int]) -> str:
        return repr(xs)

    with pytest.raises(InvalidCallError) as refused:
        load(f).run(xs=None)
    assert [p.type for p in refused.value.errors()] == ["invalid_input"]
    assert codes(load(f, record=GraphRecord(nodes=[placed("f", "f", values={"xs": None})]))) == ["invalid_value"]

    def g(xs: Series[int] = None) -> int:  # type: ignore[assignment]
        return 0

    with pytest.raises(NodeDeclarationError) as raised:
        declared(g)
    assert raised.value.code == "default_invalid"


def test_a_blank_on_an_input_that_takes_one_is_held_to_no_bound():
    @node("f", title="F", description="For a test.")
    def f(x: Annotated[str | None, Param(max_length=2)] = "ab") -> str:
        return repr(x)

    graph = load(f)
    assert graph.run(x=None).results["f"]["result"] == "None"
    assert graph.run(x="ok").results["f"]["result"] == "'ok'"


def test_bounds_are_on_the_field_an_editor_reads():
    @node("f", title="F", description="For a test.")
    def f(n: Annotated[int, Param(ge=0, le=9)], s: Annotated[str, Param(max_length=3)] = "a") -> int:
        return n

    assert [i.json_schema for i in f.describe().inputs] == [{"type": "integer", "minimum": 0, "maximum": 9}, {"type": "string", "maxLength": 3}]


def test_a_value_typed_into_a_series_is_checked_item_by_item_at_load_and_arrives_as_a_list():
    @node("total", title="Total", description="For a test.")
    def total(xs: Annotated[Series[int], Param(ge=0)]) -> str:
        return f"{type(xs).__name__}:{sum(xs)}"

    def typed(values):
        return load(total, record=GraphRecord(nodes=[placed("t", "total", values={"xs": values})]))

    assert codes(typed([1, -1])) == ["invalid_value"]
    assert typed([1, 2]).run().results["t"]["result"] == "list:3"


def test_a_subclass_of_a_builtin_is_checked_as_the_builtin_then_made_one():
    @node("kind", title="Kind", description="For a test.")
    def kind(t: Text) -> str:
        return type(t).__name__

    assert load(kind).run(t="a").results["kind"]["result"] == "Text"


def test_a_subclass_lands_where_its_parent_is_asked_for_and_not_the_other_way():
    @node("make", title="Make", description="For a test.")
    def make(x: str) -> Text:
        return Text(x)

    @node("plain", title="Plain", description="For a test.")
    def plain(x: str) -> str:
        return x

    @node("on_str", title="On str", description="For a test.")
    def on_str(t: str) -> str:
        return type(t).__name__

    @node("on_text", title="On text", description="For a test.")
    def on_text(t: Text) -> str:
        return t

    assert load(make, on_str).run(x="a").results["on_str"]["result"] == "Text"
    mistyped = GraphRecord(nodes=[placed("plain", "plain"), placed("on_text", "on_text")], edges=[("plain.result", "on_text.t")])
    assert codes(load(plain, on_text, record=mistyped)) == ["type_mismatch"]


def test_each_plain_type_carries_its_own_value():
    @node("flags", title="Flags", description="For a test.")
    def flags(on: bool, day: datetime.date) -> bool:
        return on

    @node("flip", title="Flip", description="For a test.")
    def flip(on: bool) -> bool:
        return not on

    @node("when", title="When", description="For a test.")
    def when(day: datetime.date) -> datetime.date:
        return day + datetime.timedelta(days=1)

    graph = load(flags, flip, when, record=GraphRecord(
        nodes=[placed("f", "flags"), placed("n", "flip"), placed("w", "when")],
        edges=[("f.result", "n.on")],
    ))
    run = graph.run({"on": True, "f.day": datetime.date(2026, 1, 1), "w.day": datetime.date(2026, 1, 1)})
    assert run.results["n"]["result"] is False
    assert run.results["w"]["result"] == datetime.date(2026, 1, 2)


def test_a_pydantic_model_travels_on_an_edge_and_through_json():
    @host_type("Visit", title="Visit")
    class Visit(BaseModel):
        who: str
        when: datetime.date

    @node("make", title="Make", description="For a test.")
    def make(who: str) -> Visit:
        return Visit(who=who, when=datetime.date(2026, 1, 2))

    @node("show", title="Show", description="For a test.")
    def show(visit: Visit) -> str:
        return f"{visit.who} {visit.when.isoformat()}"

    graph = load(make, show)
    run = graph.run(who="Bo")
    back = graph.load_run(run.to_json())
    assert back.results == run.results and back.results["show"]["result"] == "Bo 2026-01-02"
    assert isinstance(back.results["make"]["result"], Visit)


def test_an_any_input_has_a_json_form_before_anything_is_wired():
    @node("f", title="F", description="For a test.")
    def f(x: Any) -> str:
        return ""

    assert json.loads(f.describe().model_dump_json(exclude_none=True))["inputs"][0] == {
        "name": "x", "title": "X", "many": False, "required": True, "type": "Any", "type_variables": [], "schema": {},
    }


def test_describe_writes_each_default_as_its_type_writes_it_as_json():
    @host_type("point", title="Point")
    @dataclass
    class Point:
        x: int

    @node("f", title="F", description="For a test.")
    def f(b: list[Point] = [Point(1)]) -> str:  # noqa: B006, B008
        return ""

    written = json.loads(f.describe().model_dump_json())
    assert [i["default"] for i in written["inputs"]] == [[{"x": 1}]]


def test_a_parameter_named_like_a_pydantic_model_attribute_checks_and_runs_without_a_warning():
    @node("f", title="F", description="For a test.")
    def f(schema: int, copy: int, json: int, model_config: int, json_: int) -> str:
        return f"{schema}{copy}{json}{model_config}{json_}"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run = load(f).run(schema=1, copy=2, json=3, model_config=4, json_=5)
    assert run.results["f"]["result"] == "12345"


# -- Series and Table --------------------------------------------------------------------------


def test_a_series_input_takes_many_values_of_its_element_type_and_a_list_is_one_value():
    @node("f", title="F", description="For a test.")
    def f(texts: Series[Text], numbers: list[float]) -> str:
        return ""

    assert [(i.many, i.type) for i in f.describe().inputs] == [(True, "Text"), (False, "list[float]")]


def test_a_table_declares_its_columns_and_their_types():
    @node("read", title="Read", description="For a test.")
    def read(text: str) -> Table[Invoice]:
        return pl.DataFrame({"name": [text], "amount": [1]})

    @node("bare", title="Bare", description="For a test.")
    def bare(text: str) -> Table:
        return pl.DataFrame()

    output = field_of(read, "result", "outputs")
    columns = {c["properties"]["name"]["const"]: c["properties"]["values"]["items"] for c in output.json_schema["properties"]["columns"]["prefixItems"]}
    assert not output.many and output.type == "Table[name: str, amount: int]" and columns == {"name": {"type": "string"}, "amount": {"type": "integer"}}
    any_columns = field_of(bare, "result", "outputs")
    assert (any_columns.many, any_columns.type) == (False, "Table") and "prefixItems" not in any_columns.json_schema["properties"]["columns"]


# -- Param and widgets -------------------------------------------------------------------------


def test_a_param_with_a_widget_describes_with_one_title_on_the_field():
    languages = (Choice(id="en", title="English"), Choice(id="da", title="Dansk"))

    @node("f", title="F", description="For a test.")
    def f(lang: Annotated[str, Param(title="Language", description="Which one", choices=languages, widget=Dropdown())] = "en") -> str:
        return lang

    described = f.describe().inputs[0]
    assert (described.title, described.description, described.type) == ("Language", "Which one", "str")
    assert described.widget == {"type": "dropdown"} and described.choices == [{"id": "en", "title": "English"}, {"id": "da", "title": "Dansk"}]


def test_a_param_with_a_title_alone_is_an_input_with_no_control():
    @node("f", title="F", description="For a test.")
    def f(x: Annotated[str, Param(title="Text")], y: str) -> str:
        return x

    assert [i.widget is not None for i in f.describe().inputs] == [False, False]


def test_a_widget_is_frozen_keyword_only_and_refuses_a_key_it_does_not_have():
    with pytest.raises(ValidationError):
        Textarea(row=5)
    with pytest.raises(TypeError):
        Textarea(5)  # type: ignore[misc]
    shown = TextInput()
    with pytest.raises(ValidationError):
        shown.type = "textarea"  # type: ignore[misc]


def test_a_widget_dumps_its_type_and_its_own_config_only():
    assert Dropdown().model_dump(mode="json") == {"type": "dropdown"}
    assert ListInput().model_dump(mode="json") == {"type": "list-input"}
    assert Textarea().type == "textarea"


def test_a_file_upload_offers_every_kind_as_an_empty_tuple_and_takes_no_none():
    assert FileUpload().accept == ()
    assert FileUpload().model_dump(mode="json")["accept"] == []
    with pytest.raises(ValidationError):
        FileUpload(accept=None)  # type: ignore[arg-type]


def test_a_host_declares_a_control_of_its_own_and_it_loads_runs_and_describes_whole():
    class EntityPicker(Widget):
        type: Literal["entity-picker"] = "entity-picker"
        entity_kind: str = ""

    @node("greet", title="Greet", description="For a test.")
    def greet(who: Annotated[str, Param(widget=EntityPicker(entity_kind="user"))]) -> str:
        return f"hi {who}"

    assert greet.describe().inputs[0].widget == {"type": "entity-picker", "entity_kind": "user"}
    graph = load(greet)
    assert graph.problems == [] and graph.run(who="Bo").results["greet"]["result"] == "hi Bo"


# -- versions, policy, deprecation -------------------------------------------------------------


def test_a_version_number_is_its_place_in_the_chain_and_each_keeps_its_own_settings():
    @node("v", title="V1", description="For a test.", policy=Policy(retries=1), deprecation=Deprecation(header="Use v3"))
    def v1(x: str) -> str:
        return "one"

    @node("v", title="V2", description="For a test.", replaces=v1, policy=Policy(retries=2))
    def v2(x: str) -> str:
        return "two"

    @node("v", title="V3", description="For a test.", replaces=v2, policy=Policy(concurrency=3))
    def v3(x: str) -> str:
        return "three"

    assert v3.versions == (3, 2, 1) and v3("a") == "three"
    assert [v3.version_of(n).policy for n in (1, 2, 3)] == [Policy(retries=1), Policy(retries=2), Policy(concurrency=3)]
    assert v3.version_of(1).deprecation == Deprecation(header="Use v3") and v3.deprecation is None
    assert v3.version_of(1).describe().deprecation.header == "Use v3" and v3.describe().deprecation is None
    with pytest.raises(KeyError, match="'v' has no version 4; it has 3, 2, 1"):
        v3.version_of(4)
    for saved, body in ((1, "one"), (3, "three")):
        graph = load(v3, record=GraphRecord(nodes=[placed("n", "v", version=saved)]))
        assert graph.run(x="a").results["n"]["result"] == body


def test_a_version_with_no_policy_gets_the_default_of_one_call_at_a_time():
    @node("f", title="F", description="For a test.")
    def f(x: str) -> str:
        return x

    assert f.policy == Policy() and f.policy.concurrency == 1


def test_a_deprecation_is_kept_as_written_and_is_never_a_problem():
    @node("old", title="Old", description="For a test.", deprecation=Deprecation())
    def old(x: str) -> str:
        return x

    @node("going", title="Going", description="For a test.", deprecation=Deprecation(header="Going", alternative="new"))
    def going(x: str) -> str:
        return x

    assert old.deprecation == Deprecation() and going.deprecation.alternative == "new"
    assert load(going).problems == []


def test_a_chain_written_wrong_is_refused_where_it_is_written():
    def f(x: int) -> int:
        return x

    def g(y: int) -> int:
        return y

    other = node("other", title="F", description="For a test.")(f)
    with pytest.raises(NodeDeclarationError) as raised:
        node("x", title="F", description="For a test.", replaces=other)(f)
    assert raised.value.code == "replaces_another_node"
    with pytest.raises(NodeDeclarationError) as raised:
        node("x", title="F", description="For a test.", upgrade=Upgrade())(f)
    assert raised.value.code == "upgrade_without_replaces"
    with pytest.raises(NodeDeclarationError) as raised:
        node("x", title="G", description="For a test.", replaces=node("x", title="F", description="For a test.")(f))(g)
    assert raised.value.code == "upgrade_missing"


def test_a_definition_may_hold_a_graph_at_a_version_of_its_own():
    @node("read", title="Read", description="For a test.")
    def read(text: str) -> str:
        return text

    inner = stored_graph("inner", title="Inner", description="For a test.", version=3)(make_graph(read))
    assert inner.version == 3 and inner.versions == (3,)
    assert next(n for n in registry_of(read, inner).describe().nodes if n.id == "inner").kind == "graph"
    graph = load(read, inner, record=GraphRecord(nodes=[placed("e", "inner", version=3)]))
    assert graph.problems == [] and list(graph.inputs) == ["text"]


# -- hooks -------------------------------------------------------------------------------------


def test_a_hook_parameter_must_name_an_input_by_name():
    def f(x: int) -> dict:
        return {}

    def names_nothing(nope: int):
        return {}

    def star(*args):
        return {}

    def double_star(**kwargs):
        return {}

    for hook in (names_nothing, star, double_star):
        with pytest.raises(NodeDeclarationError) as raised:
            declared(f, compute_outputs=hook)
        assert raised.value.code == "hook_parameter_unknown", hook


def test_added_inputs_need_kwargs_to_land_in():
    def f(x: int) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f, compute_added_inputs=lambda: {})
    assert raised.value.code == "compute_added_inputs_without_var_keyword"


def test_a_hook_gets_the_typed_in_value_or_else_the_declared_default():
    seen = []

    def outputs(n: int):
        seen.append(n)
        return {f"o{i}": int for i in range(n)}

    @node("fan", title="Fan", description="For a test.", compute_outputs=outputs)
    def fan(n: int = 2) -> dict:
        return {f"o{i}": i for i in range(n)}

    assert load(fan, record=GraphRecord(nodes=[placed("fan", "fan")])).describe().nodes["fan"].outputs[1].name == "o1"
    assert seen == [2]
    load(fan, record=GraphRecord(nodes=[placed("fan", "fan", values={"n": 3})]))
    assert seen == [2, 3]


def test_a_hook_is_not_called_when_a_value_it_needs_has_none():
    seen = []

    def outputs(n: int):
        seen.append(n)
        return {"result": int}

    @node("fan", title="Fan", description="For a test.", compute_outputs=outputs)
    def fan(n: int) -> dict:
        return {"result": n}

    graph = load(fan)
    assert seen == [] and codes(graph) == ["hook_needs_value"] and graph.problems[0].loc == ("fan", "n")


def test_an_added_inputs_hook_reads_only_what_the_author_typed_never_an_edge():
    seen = []

    def added(n: int = 2):
        seen.append(n)
        return {f"part{i}": str for i in range(n)}

    @node("number", title="Number", description="For a test.")
    def number(x: int) -> int:
        return x

    @node("joiner", title="Joiner", description="For a test.", compute_added_inputs=added)
    def joiner(n: int = 2, **parts: Any) -> str:
        return ",".join(parts.values())

    load(number, joiner, record=GraphRecord(nodes=[placed("u", "number"), placed("j", "joiner")], edges=[("u.result", "j.n")]))
    assert seen == [2]


def test_each_version_has_its_own_hooks():
    @node("h", title="H1", description="For a test.", compute_outputs=lambda: {"first": str})
    def h1(x: str) -> dict:
        return {"first": x}

    @node("h", title="H2", description="For a test.", replaces=h1, compute_outputs=lambda: {"second": str})
    def h2(x: str) -> dict:
        return {"second": x}

    for saved, name in ((1, "first"), (2, "second")):
        graph = load(h2, record=GraphRecord(nodes=[placed("n", "h", version=saved)]))
        assert graph.run(x="a").results["n"] == {name: "a"}


def test_a_field_built_by_a_hook_takes_a_blank_when_its_hint_says_so():
    @node("maybe", title="Maybe", description="For a test.", compute_outputs=lambda: {"result": int | None})
    def maybe(x: str) -> dict:
        return {"result": None}

    graph = load(maybe)
    output = graph.describe().nodes["maybe"].outputs[0]
    assert output.type == "int" and {"type": "null"} in output.json_schema["anyOf"]
    assert graph.run(x="a").results["maybe"]["result"] is None


def test_a_hook_reads_the_element_type_a_series_input_receives():
    def typed_by(items: type):
        return {"result": items}

    @node("lengths", title="Lengths", description="For a test.")
    def lengths(text: str) -> Series[int]:
        return [len(w) for w in text.split()]

    @node("first", title="First", description="For a test.", compute_outputs=typed_by)
    def first(items: Series[Any]) -> dict:
        return {"result": items[0]}

    @node("double", title="Double", description="For a test.")
    def double(n: int) -> int:
        return n * 2

    graph = load(lengths, first, double, record=GraphRecord(
        nodes=[placed("lengths", "lengths"), placed("first", "first"), placed("double", "double")],
        edges=[("lengths.result", "first.items"), ("first.result", "double.n")],
    ))
    assert graph.problems == [] and graph.describe().nodes["first"].outputs[0].type == "int"
    assert graph.runs_on["first"].msg == "once"
    assert graph.run(text="ab c").results["double"]["result"] == 4


def test_a_hook_reads_any_on_an_input_no_edge_feeds_even_with_a_value_typed_in():
    seen = []

    def outputs(x: type):
        seen.append(x)
        return {"result": int}

    @node("f", title="F", description="For a test.", compute_outputs=outputs)
    def f(x: Any) -> dict:
        return {"result": 1}

    load(f, record=GraphRecord(nodes=[placed("f", "f")]))
    load(f, record=GraphRecord(nodes=[placed("f", "f", values={"x": "typed"})]))
    assert seen == [Any, Any]


def test_a_node_whose_hook_gives_no_outputs_yet_leaves_the_graph_runnable():
    @node("empty", title="Empty", description="For a test.", compute_outputs=lambda: {})
    def empty(x: str) -> dict:
        return {}

    graph = load(empty)
    assert graph.problems == [] and graph.outputs == [] and graph.run(x="a").status == "complete"


def _refuses():
    raise ValueError("cannot say yet")


@node("echo", title="Echo", description="For a test.")
def echo(v: str) -> str:
    return v


def test_a_hook_that_raises_is_its_nodes_one_problem_never_an_exception():
    @node("outs", title="Outs", description="For a test.", compute_outputs=_refuses)
    def outs(x: str) -> dict:
        return {}

    @node("adds", title="Adds", description="For a test.", compute_added_inputs=_refuses)
    def adds(**kwargs: Any) -> str:
        return ""

    for kind, definition in (("outs", outs), ("adds", adds)):
        graph = load(definition, echo, record=GraphRecord(nodes=[placed("n", kind), placed("r", "echo")], edges=[("n.result", "r.v")]))
        assert [(p.type, p.loc) for p in graph.problems] == [("compute_hook_failed", ("n",))], kind  # the reader carries no echo
        assert graph.problems[0].ctx["error"] == "ValueError"


def test_a_hook_is_given_any_for_an_input_whose_edge_is_refused_as_for_one_nothing_is_wired_to():
    """No type arrives on a refused edge, so the hook answers as it does for a half-drawn graph and the node keeps its fields."""
    arrived: list[Any] = []

    def of_what_arrives(x: type) -> dict[str, Any]:
        arrived.append(x)
        return {"result": x}

    @node("outs", title="Outs", description="For a test.", compute_outputs=of_what_arrives)
    def outs(x: str) -> dict:
        return {}

    graph = load(echo, outs, record=GraphRecord(nodes=[placed("e", "echo"), placed("n", "outs")], edges=[("e.nothing", "n.x")]))
    assert codes(graph) == ["unknown_output"] and arrived == [Any]
    assert [o.type for o in graph.describe().nodes["n"].outputs] == ["Any"]


def test_a_series_of_tables_from_a_hook_loads_with_no_problems():
    def h(x: str) -> dict:
        return {}

    assert codes(load(node("h", title="H", description="For a test.", compute_outputs=lambda: {"result": Series[Table]})(h))) == []


def test_a_hint_from_a_hook_naming_a_class_that_is_not_declared_is_the_hooks_problem():
    class Sheet(Table):
        pass

    def h(x: str) -> dict:
        return {}

    def giving(hint):
        return lambda: {"result": hint}

    for hint in (Sheet, Series[Sheet]):
        assert codes(load(node("h", title="H", description="For a test.", compute_outputs=giving(hint))(h))) == ["compute_hook_failed"], hint


def _computing(hook: Any) -> Any:
    """A graph of one node, ``h``, whose outputs ``hook`` computes."""
    def h(x: str = "a") -> dict:
        return {}

    return load(node("h", title="H", description="For a test.", compute_outputs=hook)(h))


@pytest.mark.parametrize("returned", [None, ["result"], "result"])
def test_a_hook_that_returns_no_mapping_is_the_hooks_failure_saying_what_a_hook_returns(returned: Any):
    [problem] = _computing(lambda: returned).problems
    assert (problem.type, problem.loc, problem.ctx["error"]) == ("compute_hook_failed", ("h",), "not_a_mapping")
    assert f"a {type(returned).__name__}" in problem.msg and "by name" in problem.msg


def test_a_hook_that_names_a_field_with_no_text_is_refused_by_that_name():
    assert [(p.type, p.loc) for p in _computing(lambda: {1: int}).problems] == [("parameter_name_invalid", ("h", "1"))]


def test_a_bug_in_a_host_types_own_schema_hook_raises_as_itself_where_the_class_is_declared():
    """Only a class with no ``@host_type`` is a hook's failure; an ``AttributeError`` in a host's own code is the host's bug."""
    with pytest.raises(AttributeError, match="no_such_method"):

        @host_type("buggy", title="Buggy")
        class Buggy:

            @classmethod
            def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
                return handler.no_such_method(str)


def test_a_hooks_own_exception_is_logged_with_its_traceback_as_a_nodes_own_failure_is(caplog):
    def boom() -> dict[str, Any]:
        return {}["missing"]

    with caplog.at_level(logging.ERROR, logger="conductor"):
        [problem] = _computing(boom).problems
    assert (problem.type, problem.ctx["error"]) == ("compute_hook_failed", "KeyError")
    logged = caplog.records[0]  # once for each time the graph is loaded
    assert logged.exc_info is not None and logged.exc_info[0] is KeyError and "'h'" in logged.getMessage() and "compute_outputs" in logged.getMessage()


def test_a_hook_adding_an_input_whose_name_an_address_cannot_carry_is_refused():
    @node("adds", title="Adds", description="For a test.", compute_added_inputs=lambda: {"a.b": int})
    def adds(**kwargs: Any) -> int:
        return 1

    graph = load(adds)
    assert [(p.type, p.loc) for p in graph.problems] == [("parameter_name_invalid", ("adds", "a.b"))]


# -- what a signature says that conductor refuses or reads as written ------------------------


def test_a_none_default_on_an_input_that_takes_no_blank_is_refused():
    def f(x: int = None) -> int:  # type: ignore[assignment]
        return 1

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.code == "default_invalid"


def test_a_default_reaches_the_function_as_its_declared_type():
    @node("kind", title="Kind", description="For a test.")
    def kind(x: Whole = 3, t: Text = "a") -> str:  # type: ignore[assignment]
        return f"{type(x).__name__} {type(t).__name__}"

    assert load(kind).run().results["kind"]["result"] == "Whole Text"


def test_annotated_metadata_that_is_not_a_param_is_pydantics_so_a_note_is_ignored_and_a_field_limit_is_held():
    def noted(x: Annotated[int, "a note"]) -> int:
        return x

    def limited(x: Annotated[int, pydantic.Field(ge=0)]) -> str:
        return str(x)

    assert declared(noted).inputs["x"].type.bare is int
    graph = load(
        declared(limited),
        record=GraphRecord(nodes=[placed("l", "x", values={"x": -1})]),
    )
    assert codes(graph) == ["invalid_value"]


def test_params_stacked_on_an_alias_merge_as_pydantics_fields_do_the_later_one_winning():
    line = Annotated[str, Param(widget=TextInput(), title="Line")]

    def f(separator: Annotated[line, Param(min_length=1, title="Separator")] = ",") -> int:
        return 1

    field = declared(f).inputs["separator"]
    assert (field.description.title, field.param.widget, field.param.limits) == ("Separator", TextInput(), {"min_length": 1})
    with pytest.raises(FieldValueError):
        field.whole.read("")


def test_a_param_on_a_series_item_is_the_fields_own_so_an_int_lands_and_a_bool_does_not():
    count = Annotated[int, Param(ge=0, title="Count")]
    registry = NodeRegistry()

    @registry.node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str]:
        return text.split()

    @registry.node("length", title="Length", description="For a test.")
    def length(text: str) -> int:
        return len(text)

    @registry.node("total", title="Total", description="For a test.")
    def total(n: Series[count], m: Series[count | None] | None = None) -> int:
        return sum(n)

    for name, blank in (("n", False), ("m", True)):
        field = total.inputs[name]
        assert (field.type.bare, field.description.title, field.param.limits) == (int | None if blank else int, "Count", {"ge": 0})
        for wrong in ([-1], [True]):
            with pytest.raises(FieldValueError):
                field.whole.read(wrong)
    graph = registry.load(make_graph(words, length, total, edges=[("words.result", "length.text"), ("length.result", "total.n")]))
    assert graph.problems == []
    assert graph.run(text="ab c").outputs == {"total.result": 3}


def test_a_tables_column_annotated_with_a_param_declares_and_its_limits_hold_on_what_the_node_returns():
    class Row(TypedDict):
        n: Annotated[int, Param(ge=0)]

    @node("rows", title="Rows", description="For a test.")
    def rows(text: str) -> Table[Row]:
        return pl.DataFrame({"n": [-1]})

    assert registry_of(rows).load(GraphRecord(nodes=[placed("r", "rows")])).problems == []
    run = registry_of(rows).load(GraphRecord(nodes=[placed("r", "rows")])).run(text="x")
    assert (run.status, run.state.cause.type) == ("failed", "invalid_output")


def test_a_params_choices_limit_a_hosts_own_model_as_they_limit_a_nodes_field():
    class Settings(BaseModel):
        colour: Annotated[str, Param(choices=["red", Choice("blue", "Blue")])] = "red"

    assert Settings(colour="blue").colour == "blue"
    with pytest.raises(ValidationError, match="not_a_choice"):
        Settings(colour="green")


def test_a_value_an_edge_brings_is_held_to_the_inputs_choices_as_to_its_limits():
    registry = NodeRegistry()

    @registry.node("give", title="Give", description="For a test.")
    def give(text: str) -> str:
        return text

    @registry.node("pick", title="Pick", description="For a test.")
    def pick(colour: Annotated[str, Param(choices=["red", "blue"])]) -> str:
        return colour

    graph = registry.load(make_graph(give, pick))
    assert graph.run(text="red").outputs == {"pick.result": "red"}
    run = graph.run(text="green")
    assert (run.status, run.state.cause.type, run.state.cause.loc) == ("failed", "invalid_edge_value", ("pick",))


def test_a_params_choices_hold_on_a_field_of_a_dataclass_a_node_takes_as_its_limits_do():
    @host_type("Order", title="Order")
    @dataclass(frozen=True)
    class Order:
        colour: Annotated[str, Param(choices=["red", "blue"])]
        n: Annotated[int, Param(ge=0)]

    registry = NodeRegistry()

    @registry.node("take", title="Take", description="For a test.")
    def take(order: Order) -> str:
        return order.colour

    graph = registry.load(make_graph(take))
    assert graph.run(order={"colour": "red", "n": 1}).outputs == {"take.result": "red"}
    for wrong in ({"colour": "green", "n": 1}, {"colour": "red", "n": -1}):
        with pytest.raises(InvalidCallError) as raised:
            graph.run(order=wrong)
        assert [problem.type for problem in raised.value.errors()] == ["invalid_input"]


def test_a_series_of_series_does_not_exist():
    def f(x: Series[Series[int]]) -> int:
        return 1

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.code == "nested_series"


def test_a_bare_series_is_many_values_of_anything():
    @node("count", title="Count", description="For a test.")
    def count(xs: Series) -> int:
        return len(xs)

    assert [(i.many, i.type) for i in count.describe().inputs] == [(True, "Any")]
    assert load(count, record=GraphRecord(nodes=[placed("c", "count", values={"xs": [1, "a"]})])).run().results["c"]["result"] == 2


def test_a_series_that_may_be_none_is_still_many_values():
    @node("f", title="F", description="For a test.")
    def f(xs: Series[int] | None = None) -> str:
        return repr(xs)

    assert [(i.many, i.type) for i in f.describe().inputs] == [(True, "int")]


def test_a_limit_its_type_cant_hold_names_the_type_as_it_is_written():
    def f(x: Annotated[str | None, Param(ge=0)] = None) -> str:
        return ""

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert "['ge'] can't apply to str | None, as pydantic holds no such limit on it" in str(raised.value)


def test_a_wrongly_typed_limit_is_refused_as_a_bound():
    def f(x: Annotated[int, Param(ge="a")]) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        declared(f)
    assert raised.value.code == "bound_invalid"


def test_a_hook_reads_a_whole_table_arriving_with_its_columns():
    seen = []

    def outputs(rows: type):
        seen.append(rows)
        return {"result": str}

    @node("read", title="Read", description="For a test.")
    def read(text: str) -> Table[Invoice]:
        return pl.DataFrame({"name": [text], "amount": [1]})

    @node("look", title="Look", description="For a test.", compute_outputs=outputs)
    def look(rows: Table) -> dict:
        return {"result": ",".join(rows.columns)}

    graph = load(read, look)
    row_type = typing.get_args(seen[0])[0]  # Table[R] carries R, a TypedDict of the arriving columns
    assert graph.problems == [] and typing.get_type_hints(row_type) == {"name": str, "amount": int}
    assert graph.run(text="Bo").results["look"]["result"] == "name,amount"


def test_from_run_names_a_class() -> None:
    from conductor import FromRun

    with pytest.raises(NodeDeclarationError, match="FromRun names the class") as raised:

        @node("bare", title="Bare", description="For a test.")
        def bare(who: FromRun) -> str:  # type: ignore[type-arg]
            return "x"

    assert raised.value.code == "from_run_not_a_class"


@dataclass
class _Session:
    user: str


def _optional_session(session: FromRun[_Session] | None = None) -> str:
    return "x"


def _sessions(sessions: list[FromRun[_Session]]) -> str:
    return "x"


def _returns_a_session() -> FromRun[_Session]:
    return _Session("x")


@pytest.mark.parametrize("function", [_optional_session, _sessions, _returns_a_session])
def test_from_run_is_only_a_whole_parameter_so_a_caller_never_fills_it(function: Any) -> None:
    # Anywhere else the value would be an input a caller fills, and a node trusts what the host hands it.
    with pytest.raises(NodeDeclarationError, match="FromRun") as raised:
        node("who", title="Who", description="For a test.")(function)

    assert raised.value.code == "from_run_misplaced"


def test_a_bare_series_is_many_values_of_any_type() -> None:
    @node("count", title="Count", description="For a test.")
    def count(items: Series) -> int:  # type: ignore[type-arg]
        return len(items)

    assert count.inputs["items"].many and count.inputs["items"].type.hint is Any


def test_a_literal_inputs_choices_are_in_its_record_and_a_widget_draws_them():
    @node("priority", title="Priority", description="For a test.")
    def priority(level: Annotated[Literal["low", "high"], Param(widget=Dropdown())]) -> str:
        return level

    described = priority.describe().inputs[0]
    assert described.choices == [{"id": "low", "title": "low"}, {"id": "high", "title": "high"}]
    assert described.widget == {"type": "dropdown"}


def test_a_fields_choices_have_one_home_that_titles_a_literal_and_limits_any_other_type():
    from conductor import Ask, InvalidCallError, NodeRegistry, make_graph

    titled = (Choice("low", "Lav"), Choice("high", "Høj"))

    @node("priority", title="Priority", description="For a test.")
    def priority(level: Annotated[Literal["low", "high"], Param(choices=titled, widget=Dropdown())]) -> str:
        return level

    @node("size", title="Size", description="For a test.")
    def size(name: Annotated[str, Param(choices=["S", "M"])]) -> str:
        return name

    @node("rate", title="Rate", description="For a test.")
    def rate(text: str) -> Annotated[Literal["low", "high"], Param(choices=titled)]:
        raise Ask("How urgent?")

    assert priority.describe().inputs[0].choices == [{"id": "low", "title": "Lav"}, {"id": "high", "title": "Høj"}]
    assert size.describe().inputs[0].choices == [{"id": "S", "title": "S"}, {"id": "M", "title": "M"}]
    registry = NodeRegistry()
    registry.add(size, rate)
    with pytest.raises(InvalidCallError):
        registry.load(make_graph(size)).run(name="XL")
    assert registry.load(make_graph(size)).run(name="M").outputs == {"size.result": "M"}
    asked = registry.load(make_graph(rate)).run(text="x").waiting[0].questions[0]
    assert asked.choices == [{"id": "low", "title": "Lav"}, {"id": "high", "title": "Høj"}]
    for wrong in ((Choice("low", "Lav"),), ("low", "low", "high"), (1, 2)):
        with pytest.raises(NodeDeclarationError) as raised:

            @node("bad", title="Bad", description="For a test.")
            def bad(level: Annotated[Literal["low", "high"], Param(choices=wrong)]) -> str:
                return level

        assert raised.value.code == "choices_invalid"
        assert str(raised.value).endswith("; give each choice once, as a value of the field's type")


def test_an_input_a_hook_adds_may_have_a_default_and_a_hook_may_give_its_own_code():
    from conductor import NodeError

    def added(names: str) -> dict[str, Any]:
        if names == "boom":
            raise NodeError("No such column.", code="no_column", ctx={"names": names})
        return {name: (str, "x") for name in names.split(",")}

    @node("glue", title="Glue", description="For a test.", compute_added_inputs=added)
    def glue(text: str, names: str = "a,b", **columns: str) -> str:
        return text + "".join(columns.values())

    registry = NodeRegistry()
    registry.add(glue)
    graph = registry.load(GraphRecord(nodes=[NodeRecord(id="g", node="glue", values={"text": "t"})]))
    assert graph.problems == [] and graph.run().outputs == {"g.result": "txx"}
    (problem,) = registry.load(GraphRecord(nodes=[NodeRecord(id="g", node="glue", values={"text": "t", "names": "boom"})])).problems
    assert (problem.type, problem.ctx["error"], problem.ctx["error_ctx"], problem.msg) == (
        "compute_hook_failed", "no_column", {"names": "boom"}, "The node could not work out its fields from what it was given: No such column. Check what is typed into the node and what is wired to it.",
    )


@pytest.mark.parametrize("ctx", [{"reason": "r", "error": "e"}, {}])
def test_a_hooks_own_ctx_is_kept_whole_beside_the_reason_and_the_code(ctx: dict[str, Any]) -> None:
    from conductor import NodeError

    def added(names: str) -> dict[str, Any]:
        raise NodeError("No such column.", code="no_column", ctx=ctx)

    @node("glue", title="Glue", description="For a test.", compute_added_inputs=added)
    def glue(names: str, **columns: str) -> str:
        return names

    (problem,) = registry_of(glue).load(GraphRecord(nodes=[NodeRecord(id="g", node="glue", values={"names": "a"})])).problems
    assert problem.ctx == {"reason": "No such column.", "error": "no_column", **({"error_ctx": ctx} if ctx else {})}


@pytest.mark.parametrize(("raised", "code"), [(NodeError("No columns."), "failed"), (ExternalError("No columns."), "external_failed")])
def test_a_hooks_error_without_a_code_of_its_own_says_the_code_its_class_gives_every_error_without_one(raised: NodeError, code: str) -> None:
    def added(names: str) -> dict[str, Any]:
        raise raised

    @node("glue", title="Glue", description="For a test.", compute_added_inputs=added)
    def glue(names: str, **columns: str) -> str:
        return names

    (problem,) = registry_of(glue).load(GraphRecord(nodes=[NodeRecord(id="g", node="glue", values={"names": "a"})])).problems
    assert (raised.code, problem.ctx["error"]) == (code, code)


def test_a_configuration_the_graphs_author_fixes_is_a_value_and_one_left_open_is_the_callers():
    @node("ask-model", title="Ask model", description="For a test.")
    def ask_model(prompt: str, model: str = "small") -> str:
        return f"{model}:{prompt}"

    registry = NodeRegistry()
    registry.add(ask_model)
    fixed = registry.load(make_graph(ask_model, values={"ask-model.model": "big"}))
    assert list(fixed.inputs) == ["prompt"] and fixed.run(prompt="hi").outputs == {"ask-model.result": "big:hi"}
    open_to_caller = registry.load(make_graph(ask_model))
    assert list(open_to_caller.inputs) == ["prompt", "model"] and open_to_caller.run(prompt="hi").outputs == {"ask-model.result": "small:hi"}


@pytest.mark.parametrize(
    ("choices", "limits", "reason"),
    [
        ([True, 2], {}, "Input should be a valid integer"),
        (["1", 2], {}, "Input should be a valid integer"),
        ([-1, 1], {"ge": 0}, "Input should be greater than or equal to 0"),
    ],
    ids=["a bool", "a number as text", "below the field's limit"],
)
def test_a_declared_choice_is_read_strictly_and_held_to_the_fields_limits(choices: list[object], limits: dict[str, int], reason: str) -> None:
    with pytest.raises(NodeDeclarationError) as raised:

        @node("bad", title="Bad", description="For a test.")
        def bad(n: Annotated[int, Param(choices=choices, **limits)]) -> int:  # type: ignore[arg-type]
            return n

    assert raised.value.code == "choices_invalid"
    assert f"a choice is not a value of the field's type: {reason};" in str(raised.value)


def test_a_declared_choice_is_read_as_an_answer_is_so_an_enums_value_or_its_member_is_a_choice() -> None:
    import enum

    @host_type("color", title="Color")
    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    @node("by-value", title="By value", description="For a test.")
    def by_value(color: Annotated[Color, Param(choices=["red"])]) -> str:
        return color.value

    @node("by-member", title="By member", description="For a test.")
    def by_member(color: Annotated[Color, Param(choices=[Color.RED])]) -> str:
        return color.value

    @node("a-day", title="A day", description="For a test.")
    def a_day(day: Annotated[datetime.date, Param(choices=["2024-01-02"])]) -> str:
        return day.isoformat()

    assert [c["id"] for c in by_value.describe().inputs[0].choices or ()] == ["red"]
    assert [c["id"] for c in by_member.describe().inputs[0].choices or ()] == ["red"]
    assert a_day.describe().inputs[0].choices == [{"id": "2024-01-02", "title": "2024-01-02"}]


@pytest.mark.parametrize("given", [
    {"policy": Policy(concurrency=4)},
    {"compute_outputs": lambda: {"x": int}},
    {"compute_added_inputs": lambda: {"x": int}},
])
def test_a_stored_graph_takes_no_policy_or_hook_since_its_own_nodes_run_by_theirs(given: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        stored_graph("x", title="X", description="d", **given)


def test_a_limit_pydantic_holds_on_a_type_is_taken_and_one_it_cant_is_refused_in_a_model_a_node_takes_too():
    import dataclasses
    import datetime
    import decimal

    def held(
        price: Annotated[decimal.Decimal, Param(ge=0)],
        wait: Annotated[datetime.timedelta, Param(ge=datetime.timedelta(0))],
        tags: Annotated[list[str], Param(min_length=1)],
    ) -> int:
        return 0

    spec = declared(held).inputs
    with pytest.raises(FieldValueError, match="greater than or equal"):
        spec["price"].whole.read(decimal.Decimal("-1"))
    with pytest.raises(FieldValueError, match="at least 1"):
        spec["tags"].whole.read([])

    with pytest.raises(NodeDeclarationError) as raised:

        @host_type("Order", title="Order")
        @dataclasses.dataclass(frozen=True)
        class Order:
            code: Annotated[str, Param(ge=0)]

    assert raised.value.code == "bound_invalid"


def test_a_param_whose_choices_are_not_values_of_its_type_is_refused_at_node_in_a_model_a_node_takes_and_on_a_table():
    with pytest.raises(NodeDeclarationError) as raised:

        @host_type("Order", title="Order")
        @dataclasses.dataclass(frozen=True)
        class Order:
            n: Annotated[int, Param(choices=["one", "two"])]

    assert raised.value.code == "choices_invalid"

    def on_table(x: Annotated[Table, Param(choices=["a"])]) -> int:
        return 0

    with pytest.raises(NodeDeclarationError) as raised:
        declared(on_table)
    assert raised.value.code == "choices_invalid"


@pytest.mark.parametrize("default", [b"\xff\xfe", object()], ids=["bytes-not-utf-8", "any-object"])
def test_a_default_with_no_json_form_is_refused_where_the_node_is_declared(default: Any) -> None:
    with pytest.raises(NodeDeclarationError) as raised:

        @node("b", title="B", description="Takes a value.")
        def b(data: Any = default) -> int:
            return 1

    assert raised.value.code == "default_invalid"


def test_a_default_is_described_and_offered_by_a_loaded_graph_as_its_json():
    class Row(TypedDict):
        n: int

    @node("t", title="T", description="Counts rows.")
    def t(rows: Table[Row] = pl.DataFrame({"n": [1, 2]}), when: datetime.date = datetime.date(2026, 1, 2)) -> int:
        return len(rows)

    described = {field.name: field.default for field in t.describe().inputs}
    offered = {name: given.field.default for name, given in registry_of(t).load(make_graph(t)).inputs.items()}

    table = {"columns": [{"name": "n", "type": "int", "values": [1, 2]}]}
    assert described == {"rows": table, "when": "2026-01-02"}
    assert offered == {"rows": table, "when": "2026-01-02"}
