"""The interface is derived from the signature, once, and not restated."""

from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor.dtype import DType, Single
from conductor.interface import Interface, model_of
from conductor.metadata import Param, Result
from conductor.series import Series
from conductor.widgets import (
    AnyWidget,
    Choice,
    DatePicker,
    Dropdown,
    ListWidget,
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
    text: Annotated[Text, Param(title="Text", description="Free text", widget=Textarea())],
    language: Annotated[Text, Param(title="Language", widget=Dropdown(choices=(Choice(id="da", title="Danish"), Choice(id="en", title="English"))))] = Text("da"),
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
    interface — so there is no second spelling on the record."""
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
        sources: Annotated[Series[Text], Param(title="Edges")],
    ) -> Annotated[Text, Result(title="R")]:
        return Text("")

    dtype = Interface.of(collects).inputs[0].dtype

    assert dtype.describe() == {"id": "series", "of": {"id": "interface-test-text"}}


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

    def branches(value: Annotated[Text, Param(title="V", widget=Textarea())]) -> Answer:
        return Answer(yes=value, no=value)

    outputs = Interface.of(branches).outputs
    assert [(o.name, o.title) for o in outputs] == [("yes", "Yes"), ("no", "No")]
    assert Interface.of(branches).returns is Answer


def test_a_computed_roster_declares_no_outputs_and_returns_by_name():
    """The placed node's interface says what the outputs are; run hands them back by name."""
    from collections.abc import Mapping
    from typing import Any

    def columns(spec: Annotated[Text, Param(title="Columns", widget=Textarea())]) -> Mapping[str, Any]:
        return {}

    interface = Interface.of(columns)
    assert interface.outputs == ()
    assert interface.returns is Mapping


def test_a_record_field_without_a_result_is_refused():
    @dataclass(frozen=True)
    class Bare:
        a: Text

    def node(value: Annotated[Text, Param(title="V", widget=Textarea())]) -> Bare:
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

    def pair(value: Annotated[Text, Param(title="V", widget=Textarea())]) -> Both:
        return Both(text=value, number=Num(1))

    assert [o.dtype for o in Interface.of(pair).outputs] == [Text, Num]


def test_a_param_with_a_handle_must_declare_a_dtype():
    """A plain `str` is not a declaration where an edge can land: it cannot
    be connected (`accepts` has nothing to ask), rendered or checked."""

    def plain(x: Annotated[str, Param(title="X", widget=Textarea())] = "") -> Annotated[Text, Result(title="R")]:
        return Text(x)

    with pytest.raises(TypeError, match="handle"):
        Interface.of(plain)


def test_a_closed_param_may_declare_a_static_type():
    """A parameter no edge can reach may declare any type pydantic
    can validate — an authored schema, a set of branches. It is the
    `Input`'s `dtype`, read by `model_of` and by the compiler when it
    validates a typed-in value, and it has no JSON form."""
    from dataclasses import dataclass as dc

    @dc(frozen=True)
    class Schema:
        fields: tuple[str, ...] = ()

    def structured(
        schema: Annotated[Schema, Param(title="Schema", show_handle=False, widget=SchemaBuilder())] = Schema(),
    ) -> Annotated[Text, Result(title="R")]:
        return Text("")

    (inp,) = Interface.of(structured).inputs
    assert inp.dtype is Schema
    # ``schema`` is a name ``BaseModel`` has, so the model keeps it under a
    # field of its own naming and takes it by alias.
    assert isinstance(dict(model_of((inp,))(schema={"fields": ("a",)}))["field_0"], Schema)
    assert TypeAdapter(type(inp)).dump_python(inp, mode="json")["dtype"] is None


def test_a_bare_dtype_input_is_refused():
    """An input declared as the base would accept anything, which is a
    declaration with no type in it. A node that means "whatever
    arrives" declares `Any`."""

    def vague(x: Annotated[DType, Param(title="X", widget=Textarea())]) -> Annotated[Text, Result(title="R")]:
        return Text("")

    with pytest.raises(TypeError, match="concrete"):
        Interface.of(vague)


# --- the unconstrained input -----------------------------------------------


def test_a_pass_through_declares_any():
    """An if/else node routes a value it never reads, so its type is
    whatever arrives. The parameter and the output both carry `Any` until
    the compiler types them from the edge — the output through the node's
    own `compute_outputs`."""
    from typing import Any

    def route(x: Annotated[Any, Param(title="X", widget=Textarea())]) -> Annotated[Any, Result(title="R")]:
        return x

    iface = Interface.of(route)

    assert iface.inputs[0].dtype is Any
    assert iface.outputs[0].dtype is Any
    assert iface.returns is Any


def test_an_any_roster_validates_a_call():
    """By the time a call reaches a node, the compiler has already said
    what arrives on an `Any` input — the validator passes the value through."""
    from typing import Any

    def route(x: Annotated[Any, Param(title="X", widget=Textarea())]) -> Annotated[Any, Result(title="R")]:
        return x

    model = model_of(Interface.of(route).inputs)
    assert model(x=Text("hi")).x == Text("hi")


# --- the open interface -------------------------------------------------------


def test_an_open_roster_is_single_on_the_keyword_parameter():
    """A node that takes whatever is connected to it: every connected name is an
    input, typed by its edge and received as one value. The signature
    declares no inputs for them — the compiler makes one per edge — and the
    interface says only that it is open, and how."""
    from collections.abc import Mapping
    from typing import Any

    def script(code: Annotated[Text, Param(title="Code", widget=Textarea())], **inputs: Single) -> Mapping[str, Any]:
        return {}

    iface = Interface.of(script)

    assert [i.name for i in iface.inputs] == ["code"]
    assert iface.open == "single"
    assert Interface.of(sample).open is None

    def columns(**columns: Series) -> Annotated[Text, Result(title="R")]:
        return Text("")

    assert Interface.of(columns).open == "series"  # every edge a reduction


def test_single_is_spelled_on_the_keyword_parameter_only():
    """`Single` is the open interface's shape and nothing else's: a
    named parameter declares a DType, or `Any` for whatever arrives."""

    def named(x: Annotated[Single, Param(title="X", widget=Textarea())]) -> Annotated[Text, Result(title="R")]:
        return Text("")

    with pytest.raises(TypeError, match="Single"):
        Interface.of(named)


def test_a_return_without_a_declaration_is_refused():
    def undeclared(x: Annotated[Text, Param(title="X", widget=Textarea())] = Text("")) -> Text:
        return x

    with pytest.raises(TypeError, match="Result"):
        Interface.of(undeclared)


def test_self_is_not_an_input():
    class Holder:
        def run(self, x: Annotated[Text, Param(title="X", widget=Textarea())] = Text("")) -> Annotated[Text, Result(title="R")]:
            return x

    assert [i.name for i in Interface.of(Holder.run).inputs] == ["x"]


def test_computed_inputs_arrive_as_keyword_arguments():
    """A node whose interface is computed takes the computed inputs as
    ``**values``. The signature declares nothing about them."""
    from conductor.metadata import Input

    def templated(template: Annotated[Text, Param(title="Template", widget=Textarea())] = Text(""), **values: Text) -> Annotated[Text, Result(title="R")]:
        return Text(template.format(**values))

    assert [i.name for i in Interface.of(templated).inputs] == ["template"]
    interface = (*Interface.of(templated).inputs, Input(name="name", dtype=Text, title="name", widget=Textarea()))
    validated = model_of(interface)(template="Hi {name}", name="Ida")
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
    assert data["dtype"] == {"id": "interface-test-text"}
    assert data["widget"]["kind"] == "dropdown"
    assert data["widget"]["choices"] == [{"id": "da", "title": "Danish", "element": None}, {"id": "en", "title": "English", "element": None}]


def test_the_wire_carries_presentation_on_the_field_not_inside_the_widget():
    """The contract test the field/widget split needs: what a person reads
    about the input is on the ``Input``, and the widget's dump is the
    control's configuration and nothing else."""
    from conductor.metadata import Input

    data = TypeAdapter(Input).dump_python(Interface.of(sample).inputs[1], mode="json")

    assert data["title"] == "Language"
    assert data["show_handle"] is True
    for lifted in ("title", "description", "show_handle"):
        assert lifted not in data["widget"]


def test_the_input_record_publishes_a_json_schema_per_widget():
    from conductor.metadata import Input

    schema = TypeAdapter(Input).json_schema(mode="serialization")
    (widget,) = [member for member in schema["properties"]["widget"]["anyOf"] if "discriminator" in member]

    assert widget["discriminator"]["propertyName"] == "kind"
    assert "dropdown" in widget["discriminator"]["mapping"]
    assert "textarea" in widget["discriminator"]["mapping"]


def test_a_from_run_parameter_is_a_need_not_an_input():
    """What the run supplies, by type: no widget, no handle, nothing in the
    interface. ``execute(from_run={Who: …})`` hands it in."""
    from conductor.interface import FromRun

    class Who:
        pass

    def greet(text: Annotated[Text, Param(title="T", widget=Textarea())], who: Annotated[Who, FromRun()]) -> Annotated[Text, Result(title="R")]:
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

    def echo(text: Annotated[Text, Param(title="T", widget=Textarea())] = Text("")) -> Pair:
        return Pair(text=text)

    with pytest.raises(TypeError, match="both sides"):
        Interface.of(echo)



# --- widgets are records ----------------------------------------------------


def test_a_widget_is_frozen_and_keyword_only():
    w = Textarea(rows=6)

    with pytest.raises(Exception):
        w.rows = 2
    with pytest.raises(TypeError):
        Textarea(6)


def test_a_widget_names_its_kind_once():
    """`kind` is the discriminator. There is no WidgetType enum and no
    widget_type property saying the same thing a second time."""
    assert Textarea().kind == "textarea"
    assert Dropdown().kind == "dropdown"
    assert not hasattr(Widget, "widget_type")
    assert not hasattr(Widget, "to_schema")


def test_a_widget_dumps_its_own_config_and_nothing_the_field_owns():
    data = TypeAdapter(AnyWidget).dump_python(Dropdown(choices=(Choice(id="da", title="Danish"),)), mode="json")

    assert data == {"kind": "dropdown", "choices": [{"id": "da", "title": "Danish", "element": None}]}


def test_a_widget_takes_a_cable_by_default():
    assert Param(title="Text", widget=Textarea()).show_handle is True


def test_a_control_does_not_close_its_own_handle():
    """Rendering and wireability are different questions.

    An edge can legitimately deliver a dropdown's choice or a built schema,
    so the widget is the wrong place to decide. The node closes the input.
    """
    assert Param(title="Choice", widget=Dropdown(choices=(Choice(id="a", title="A"),))).show_handle is True
    assert Param(title="Date", widget=DatePicker()).show_handle is True
    assert Param(title="Schema", widget=SchemaBuilder()).show_handle is True


def test_no_widget_decides_wireability():
    """Whether an edge can reach an input is the ``Param``'s answer; no
    control carries one. A widget that changed engine control does not
    exist: a pause is a node returning Asks."""
    import conductor.widgets as w

    carrying = sorted(
        name
        for name in dir(w)
        if isinstance(getattr(w, name, None), type)
        and issubclass(getattr(w, name), w.Widget)
        and {"show_handle", "title", "description"} & set(getattr(w, name).model_fields)
    )
    assert carrying == []
    assert not hasattr(w, "HumanReview")


def test_an_input_closes_its_own_handle():
    assert Param(title="Choice", show_handle=False, widget=Dropdown(choices=(Choice(id="a", title="A"),))).show_handle is False


def test_the_flags_that_were_not_about_editing_are_gone():
    for gone in ("disable_handle", "hidden_when", "advanced", "connection_input"):
        assert gone not in Widget.model_fields, gone
    assert "variables" not in Textarea.model_fields


def test_a_list_widget_declares_no_per_item_control():
    """The element type is the input's dtype, and the host derives the
    per-item control from it — a control inside a control would be the
    same answer declared twice."""
    data = TypeAdapter(AnyWidget).dump_python(ListWidget(min_items=1), mode="json")

    assert data == {"kind": "list", "min_items": 1, "max_items": None}
    assert "item_widget" not in ListWidget.model_fields


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
    """The one edge question is `accepts`, and the compiler asks it."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("conductor.graph.type_check")
