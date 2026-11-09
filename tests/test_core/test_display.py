"""How conductor's objects show themselves: their data, in constructor form.

The rule is pydantic's and scikit-learn's. An object prints as the call
that states what it is, leaving out what is at its default, and a
container prints its children's own reprs.
"""

from typing import Annotated

from conductor import (
    Deprecation,
    DType,
    Input,
    NodeRegistry,
    Param,
    Policy,
    Problem,
    Result,
    Upgrade,
    node,
)
from conductor._internal.index import Index
from conductor.widgets import Textarea, TextWidget


class Txt(DType, str):
    id = "display-test-text"
    display_title = "Text"


@node("greet", title="Greeting", description="Greets someone", category="text")
def greet(name: Annotated[Txt, Param(title="Name", widget=TextWidget())]) -> Annotated[Txt, Result(title="Greeting")]:
    return Txt(f"hello {name}")


def truncate_v1(text: Annotated[Txt, Param(title="Text", widget=Textarea())]) -> Annotated[Txt, Result(title="Short")]:
    return text

@node("truncate", title="Truncate", description="Cuts a text short", category="text", policy=Policy(retries=2), replaces=node("truncate")(truncate_v1), upgrade=Upgrade())
def truncate(text: Annotated[Txt, Param(title="Text", widget=Textarea())], limit: Annotated[Txt, Param(title="Limit", widget=TextWidget())] = Txt("10")
) -> Annotated[Txt, Result(title="Short")]:
    return text


def test_a_type_prints_as_its_name():
    assert repr(Txt) == "Txt"


def test_a_node_prints_what_it_declares():
    assert repr(greet) == "NodeDefinition(id='greet', title='Greeting', category='text', versions=(1,))"
    assert repr(truncate) == "NodeDefinition(id='truncate', title='Truncate', category='text', versions=(1, 2))"


def test_a_node_prints_its_tags_only_when_it_has_them():
    @node("old", title="Old", description="d", category="text", tags=("legacy",), deprecation=Deprecation(header="Gone"))
    def old(name: Annotated[Txt, Param(title="Name", widget=TextWidget())]) -> Annotated[Txt, Result(title="R")]:
        return name

    assert repr(old) == (
        "NodeDefinition(id='old', title='Old', category='text', tags=('legacy',), versions=(1,))"
    )


def test_a_registry_prints_its_nodes():
    registry = NodeRegistry()
    assert repr(registry) == "NodeRegistry(nodes=())"

    registry = registry.extended_with([greet])
    registry = registry.extended_with([truncate])
    assert repr(registry) == (
        "NodeRegistry(nodes=(\n"
        "    NodeDefinition(id='greet', title='Greeting', category='text', versions=(1,)),\n"
        "    NodeDefinition(id='truncate', title='Truncate', category='text', versions=(1, 2)),\n"
        "))"
    )


def test_a_record_prints_only_what_is_not_at_its_default():
    text = Input("text", Txt, title="Text", widget=Textarea())

    assert repr(text) == "Input(name='text', dtype=Txt, title='Text', widget=Textarea())"
    assert repr(Policy()) == "Policy()"
    assert repr(Policy(retries=2)) == "Policy(retries=2)"
    assert repr(Deprecation()) == "Deprecation()"
    assert repr(Problem(type="cycle", msg="m", loc=("a",))) == "Problem(type='cycle', loc=('a',), msg='m', fatal=True)"
    assert repr(Index("lines", parent=Index("docs"))) == "Index('lines', parent=Index('docs'))"


def test_a_record_prints_a_field_set_back_to_its_default_value_as_absent():
    """The repr is the data, not the history: a value equal to the default is not news."""
    assert repr(Policy(retries=0, concurrency=8)) == "Policy()"
