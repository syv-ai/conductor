"""The interface is derived from the signature, once, and not restated."""

from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor.dtype import DType, Single
from conductor.interface import Interface, model_of
from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import (
    AnyWidget,
    Choice,
    ConnectionList,
    DatePicker,
    Dropdown,
    List,
    SchemaBuilder,
    Textarea,
    Widget,
)
from pydantic import TypeAdapter


class Text(DType, str):
    """Conductor ships no domain types, so the test declares one."""

    id = "interface-test-text"
    title = "Text"


def sample(
    text: Annotated[Text, Textarea(title="Text", description="Free text")],
    language: Annotated[Text, Dropdown(title="Language", choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English")))] = Text("da"),
) -> Annotated[Text, Result(title="Result")]:
    return text


def test_one_pass_yields_inputs_outputs_and_the_validator():
    iface = Interface.of(sample)

    assert [i.name for i in iface.inputs] == ["text", "language"]
    assert [o.title for o in iface.outputs] == ["Result"]
    assert model_of(iface.inputs)(text="hi").language == "da"


def test_the_validator_coerces_into_the_dtype():
    """A str subclass validates into the subclass."""
    validated = model_of(Interface.of(sample).inputs)(text="hi")

    assert isinstance(validated.text, Text)


def test_there_is_one_validator_and_it_is_over_inputs():
    """`model_of` takes any tuple of inputs — a declaration or a placement's
    roster — so there is no second spelling on the record."""
    assert not hasattr(Interface, "model")


def test_an_input_keeps_its_widget_whole():
    """The widget is never destructured into an id plus a loose config dict."""
    language = Interface.of(sample).inputs[1]

    assert isinstance(language.widget, Dropdown)
    assert language.widget.choices == (Choice(id="da", title="Danish"), Choice(id="en", title="English"))
    assert not hasattr(language, "widget_config")


def test_presentation_is_lifted_onto_the_field():
    """An author writes `title` inside the widget annotation — one
    annotation object per field per side, as `Result` is on an output. The
    derivation lifts it onto the field, the same move it makes for
    `dtype`, and readers use the field."""
    language = Interface.of(sample).inputs[1]

    assert language.title == "Language"
    assert language.description is None
    assert language.show_handle is True


def test_an_input_carries_a_dtype_not_a_type_string():
    """Nothing downstream parses a string to learn what a value is."""
    text = Interface.of(sample).inputs[0]

    assert text.dtype is Text
    assert not hasattr(text, "type_str")
    assert not hasattr(text, "expects_list")
    assert not hasattr(text, "uses_connection_list")


def test_a_series_parameter_carries_its_element_type():
    def collects(
        sources: Annotated[Series[Text], ConnectionList(title="Sources")],
    ) -> Annotated[Text, Result(title="R")]:
        return Text("")

    dtype = Interface.of(collects).inputs[0].dtype

    assert dtype.describe() == {"id": "series", "of": {"id": "interface-test-text", "accepted_as": ["interface-test-text"]}}


def test_optional_is_derived_from_the_default():
    """`optional` separates "has a default" from "has none"."""
    text, language = Interface.of(sample).inputs

    assert (text.optional, text.default) == (False, None)
    assert (language.optional, language.default) == (True, "da")


def test_an_output_has_no_widget():
    """Nobody supplies a result, so there is no control to describe it with."""
    import conductor.widgets as widgets

    output = Interface.of(sample).outputs[0]

    assert output.title == "Result"
    assert output.dtype is Text
    assert not hasattr(output, "widget")
    assert not hasattr(widgets, "Output")


def test_several_outputs_are_the_fields_of_a_record():
    """A node that fans out returns a record; its fields are the outputs, by name."""

    @dataclass(frozen=True)
    class Answer:
        yes: Annotated[Text, Result(title="Yes")]
        no: Annotated[Text, Result(title="No")]

    def branches(value: Annotated[Text, Textarea(title="V")]) -> Answer:
        return Answer(yes=value, no=value)

    outputs = Interface.of(branches).outputs
    assert [(o.name, o.title) for o in outputs] == [("yes", "Yes"), ("no", "No")]
    assert Interface.of(branches).returns is Answer


def test_a_computed_roster_declares_no_outputs_and_returns_by_name():
    """The placement's roster says what the outputs are; run hands them back by name."""
    from collections.abc import Mapping
    from typing import Any

    def columns(spec: Annotated[Text, Textarea(title="Columns")]) -> Mapping[str, Any]:
        return {}

    interface = Interface.of(columns)
    assert interface.outputs == ()
    assert interface.returns is Mapping


def test_a_record_field_without_a_result_is_refused():
    @dataclass(frozen=True)
    class Bare:
        a: Text

    def node(value: Annotated[Text, Textarea(title="V")]) -> Bare:
        return Bare(a=value)

    with pytest.raises(TypeError, match="Bare.a"):
        Interface.of(node)


def test_each_of_several_outputs_carries_its_own_dtype():
    """`run` returns a tuple; its elements need not share a type."""

    class Num(DType, float):
        id = "interface-test-num-2"
        title = "Number"

    @dataclass(frozen=True)
    class Both:
        text: Annotated[Text, Result(title="T")]
        number: Annotated[Num, Result(title="N")]

    def pair(value: Annotated[Text, Textarea(title="V")]) -> Both:
        return Both(text=value, number=Num(1))

    assert [o.dtype for o in Interface.of(pair).outputs] == [Text, Num]


def test_a_param_without_a_widget_is_refused():
    """An input without a widget is an error, not a fallback."""

    def bare(x: Text = Text("")) -> Annotated[Text, Result(title="R")]:
        return x

    with pytest.raises(TypeError, match="widget"):
        Interface.of(bare)


def test_a_param_with_a_handle_must_declare_a_dtype():
    """A plain `str` is not a declaration where a cable can land: it cannot
    be wired (`accepts` has nothing to ask), rendered or checked."""

    def plain(x: Annotated[str, Textarea(title="X")] = "") -> Annotated[Text, Result(title="R")]:
        return Text(x)

    with pytest.raises(TypeError, match="handle"):
        Interface.of(plain)


def test_a_closed_param_may_declare_a_static_type():
    """A parameter no cable can reach may declare any type pydantic
    can validate — an authored schema, a set of branches. It is the
    `Input`'s `dtype`, read by `model_of` and by the compiler when it
    validates a typed-in value, and it has no JSON form."""
    from dataclasses import dataclass as dc

    @dc(frozen=True)
    class Schema:
        fields: tuple[str, ...] = ()

    def structured(
        schema: Annotated[Schema, SchemaBuilder(title="Schema", show_handle=False)] = Schema(),
    ) -> Annotated[Text, Result(title="R")]:
        return Text("")

    (inp,) = Interface.of(structured).inputs
    assert inp.dtype is Schema
    assert isinstance(model_of((inp,))(schema={"fields": ("a",)}).schema, Schema)
    assert TypeAdapter(type(inp)).dump_python(inp, mode="json")["dtype"] is None


def test_a_bare_dtype_input_is_refused():
    """An input declared as the base would accept anything, which is a
    declaration with no type in it. A node that means "whatever
    arrives" declares `Any`."""

    def vague(x: Annotated[DType, Textarea(title="X")]) -> Annotated[Text, Result(title="R")]:
        return Text("")

    with pytest.raises(TypeError, match="concrete"):
        Interface.of(vague)


# --- the unconstrained input -----------------------------------------------


def test_a_pass_through_declares_any():
    """An if/else node routes a value it never reads, so its type is
    whatever arrives. The parameter and the output both carry `Any` until
    the compiler types them from the wire — the output through the node's
    own `compute_outputs`."""
    from typing import Any

    def route(x: Annotated[Any, Textarea(title="X")]) -> Annotated[Any, Result(title="R")]:
        return x

    iface = Interface.of(route)

    assert iface.inputs[0].dtype is Any
    assert iface.outputs[0].dtype is Any
    assert iface.returns is Any


def test_an_any_roster_validates_a_call():
    """By the time a call reaches a node, the compiler has already said
    what arrives on an `Any` input — the validator passes the value through."""
    from typing import Any

    def route(x: Annotated[Any, Textarea(title="X")]) -> Annotated[Any, Result(title="R")]:
        return x

    model = model_of(Interface.of(route).inputs)
    assert model(x=Text("hi")).x == Text("hi")


# --- the open roster -------------------------------------------------------


def test_an_open_roster_is_single_on_the_keyword_parameter():
    """A node that takes whatever is wired to it: every wired name is an
    input, typed by its wire and received as one value. The signature
    declares no inputs for them — the compiler makes one per wire — and the
    interface says only that it is open, and how."""
    from collections.abc import Mapping
    from typing import Any

    def script(code: Annotated[Text, Textarea(title="Code")], **inputs: Single) -> Mapping[str, Any]:
        return {}

    iface = Interface.of(script)

    assert [i.name for i in iface.inputs] == ["code"]
    assert iface.open == "single"
    assert Interface.of(sample).open is None

    def columns(**columns: Series) -> Annotated[Text, Result(title="R")]:
        return Text("")

    assert Interface.of(columns).open == "series"  # every wire a reduction


def test_single_is_spelled_on_the_keyword_parameter_only():
    """`Single` is the open roster's shape and nothing else's: a
    named parameter declares a DType, or `Any` for whatever arrives."""

    def named(x: Annotated[Single, Textarea(title="X")]) -> Annotated[Text, Result(title="R")]:
        return Text("")

    with pytest.raises(TypeError, match="Single"):
        Interface.of(named)


def test_a_return_without_a_declaration_is_refused():
    def undeclared(x: Annotated[Text, Textarea(title="X")] = Text("")) -> Text:
        return x

    with pytest.raises(TypeError, match="Result"):
        Interface.of(undeclared)


def test_self_is_not_an_input():
    class Holder:
        def run(self, x: Annotated[Text, Textarea(title="X")] = Text("")) -> Annotated[Text, Result(title="R")]:
            return x

    assert [i.name for i in Interface.of(Holder.run).inputs] == ["x"]


def test_computed_inputs_arrive_as_keyword_arguments():
    """A node whose roster is computed takes the computed inputs as
    ``**values``. The signature declares nothing about them."""
    from conductor.metadata import Input

    def templated(template: Annotated[Text, Textarea(title="Template")] = Text(""), **values: Text) -> Annotated[Text, Result(title="R")]:
        return Text(template.format(**values))

    assert [i.name for i in Interface.of(templated).inputs] == ["template"]
    roster = (*Interface.of(templated).inputs, Input(name="name", dtype=Text, title="name", widget=Textarea(title="name")))
    validated = model_of(roster)(template="Hi {name}", name="Ida")
    assert isinstance(validated.name, Text)


# --- the record is the schema -----------------------------------------------


def test_the_input_record_dumps_as_the_wire():
    data = TypeAdapter(type(Interface.of(sample).inputs[1])).dump_python(
        Interface.of(sample).inputs[1], mode="json"
    )

    assert set(data) == {
        "name", "dtype", "title", "description",
        "widget", "show_handle", "default", "optional",
    }
    assert data["dtype"] == {"id": "interface-test-text", "accepted_as": ["interface-test-text"]}
    assert data["widget"]["kind"] == "dropdown"
    assert data["widget"]["choices"] == [{"id": "da", "title": "Danish", "element": None}, {"id": "en", "title": "English", "element": None}]


def test_the_wire_carries_presentation_on_the_field_not_inside_the_widget():
    """The contract test the field/widget split needs: the annotation object
    still holds what was lifted off it, so a consumer could read the wrong
    copy. It is not on the wire, and this is what says so."""
    from conductor.metadata import Input

    data = TypeAdapter(Input).dump_python(Interface.of(sample).inputs[1], mode="json")

    assert data["title"] == "Language"
    assert data["show_handle"] is True
    for lifted in ("title", "description", "show_handle"):
        assert lifted not in data["widget"]


def test_the_input_record_publishes_a_json_schema_per_widget():
    from conductor.metadata import Input

    schema = TypeAdapter(Input).json_schema(mode="serialization")
    widget = schema["properties"]["widget"]

    assert widget["discriminator"]["propertyName"] == "kind"
    assert "dropdown" in widget["discriminator"]["mapping"]
    assert "textarea" in widget["discriminator"]["mapping"]


def test_a_provided_parameter_is_a_need_not_an_input():
    """What the run supplies, by type: no widget, no handle, nothing in the
    roster. ``execute(provides={Who: …})`` hands it in."""
    from conductor.interface import Provided

    class Who:
        pass

    def greet(text: Annotated[Text, Textarea(title="T")], who: Annotated[Who, Provided()]) -> Annotated[Text, Result(title="R")]:
        return Text(f"{text} {who}")

    iface = Interface.of(greet)

    assert [i.name for i in iface.inputs] == ["text"]
    assert iface.needs == {"who": Who}
    assert "who" not in model_of(iface.inputs).model_fields


def test_a_name_on_both_sides_is_refused():
    """A ``Ref`` is ``(node_id, field)`` on either side, so a field name is
    unique within a node — an input and an output may not share one."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Pair:
        text: Annotated[Text, Result(title="Same name")]

    def echo(text: Annotated[Text, Textarea(title="T")] = Text("")) -> Pair:
        return Pair(text=text)

    with pytest.raises(TypeError, match="both sides"):
        Interface.of(echo)



# --- widgets are records ----------------------------------------------------


def test_a_widget_is_frozen_and_keyword_only():
    w = Textarea(title="Text", rows=6)

    with pytest.raises(Exception):
        w.rows = 2
    with pytest.raises(TypeError):
        Textarea("Text")


def test_a_widget_names_its_kind_once():
    """`kind` is the discriminator. There is no WidgetType enum and no
    widget_type property saying the same thing a second time."""
    assert Textarea(title="T").kind == "textarea"
    assert Dropdown(title="D").kind == "dropdown"
    assert not hasattr(Widget, "widget_type")
    assert not hasattr(Widget, "to_schema")


def test_a_widget_dumps_its_own_config_and_nothing_the_field_owns():
    data = TypeAdapter(AnyWidget).dump_python(
        Dropdown(title="Language", description="d", choices=(Choice(id="da", title="Danish"),)), mode="json"
    )

    assert data == {"kind": "dropdown", "choices": [{"id": "da", "title": "Danish", "element": None}]}


def test_a_widget_takes_a_cable_by_default():
    assert Textarea(title="Text").show_handle is True


def test_a_control_does_not_close_its_own_handle():
    """Rendering and wireability are different questions.

    A cable can legitimately deliver a dropdown's choice or a built schema,
    so the widget is the wrong place to decide. The node closes the input.
    """
    assert Dropdown(title="Choice", choices=(Choice(id="a", title="A"),)).show_handle is True
    assert DatePicker(title="Date").show_handle is True
    assert SchemaBuilder(title="Schema").show_handle is True


def test_no_widget_subclass_decides_wireability():
    """They all inherit True. A widget that changed engine control
    flow does not exist: a pause is a node returning Asks."""
    import conductor.widgets as w

    overriding = sorted(
        name
        for name in dir(w)
        if isinstance(getattr(w, name, None), type)
        and issubclass(getattr(w, name), w.Widget)
        and "show_handle" in getattr(w, name).__dataclass_fields__
        and getattr(w, name).__dataclass_fields__["show_handle"].default is False
    )
    assert overriding == []
    assert not hasattr(w, "HumanReview")


def test_an_input_closes_its_own_handle():
    assert Dropdown(title="Choice", choices=(Choice(id="a", title="A"),), show_handle=False).show_handle is False


def test_the_flags_that_were_not_about_editing_are_gone():
    for gone in ("disable_handle", "hidden_when", "advanced", "connection_input"):
        assert gone not in Widget.__dataclass_fields__, gone
    assert "variables" not in Textarea.__dataclass_fields__


def test_a_list_widget_declares_no_per_item_control():
    """The element type is the input's dtype, and the host derives the
    per-item control from it — a control inside a control would be the
    same answer declared twice."""
    data = TypeAdapter(AnyWidget).dump_python(
        List(title="Names", min_items=1), mode="json"
    )

    assert data == {"kind": "list", "min_items": 1, "max_items": None}
    assert "item_widget" not in List.__dataclass_fields__


def test_every_widget_is_in_the_union():
    from typing import get_args

    import conductor.widgets as w

    declared = {
        cls for cls in vars(w).values()
        if isinstance(cls, type) and issubclass(cls, w.Widget) and cls is not w.Widget
    }
    in_union = set(get_args(get_args(AnyWidget)[0]))

    assert declared == in_union


# --- there is no string type check ---------------------------------------------


def test_the_string_type_check_is_gone():
    """The one wiring question is `accepts`, and the compiler asks it."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("conductor.graph.type_check")
