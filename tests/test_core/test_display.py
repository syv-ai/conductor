"""How conductor's objects show themselves: their data, in constructor form.

The rule is pydantic's and scikit-learn's. An object prints as the call
that states what it is, leaving out what is at its default, and a
container prints its children's own reprs.
"""

from typing import Annotated

from conductor import (
    Deprecation,
    NodeDefinition,
    NodeRegistry,
    Policy,
    Result,
    Series,
    deprecated,
    version,
)
from conductor.dtype import DType
from conductor.graph.problem import Problem
from conductor.metadata import Input
from conductor.series import Index
from conductor.widgets import Text as TextWidget
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "display-test-text"
    title = "Text"


class Greet(NodeDefinition):
    id = "greet"
    title = "Greeting"
    description = "Greets someone"
    category = "text"

    def run(self, name: Annotated[Txt, TextWidget(title="Name")]) -> Annotated[Txt, Result(title="Greeting")]:
        return Txt(f"hello {name}")


class Truncate(NodeDefinition):
    id = "truncate"
    title = "Truncate"
    description = "Cuts a text short"
    category = "text"

    @version(1)
    def run_v1(self, text: Annotated[Txt, Textarea(title="Text")]) -> Annotated[Txt, Result(title="Short")]:
        return text

    @version(2, policy=Policy(retries=2))
    def run(
        self, text: Annotated[Txt, Textarea(title="Text")], limit: Annotated[Txt, TextWidget(title="Limit")] = Txt("10")
    ) -> Annotated[Txt, Result(title="Short")]:
        return text


def test_a_type_prints_as_its_name():
    assert repr(Txt) == "Txt"
    assert repr(Series[Txt]) == "Series[Txt]"


def test_a_node_prints_what_it_declares():
    assert repr(Greet) == "Greet(id='greet', title='Greeting', category='text', versions=(1,))"
    assert repr(Truncate) == "Truncate(id='truncate', title='Truncate', category='text', versions=(1, 2))"


def test_a_node_prints_its_deprecation_and_tags_only_when_it_has_them():
    @deprecated(header="Gone")
    class Old(NodeDefinition):
        id = "old"
        title = "Old"
        description = "d"
        category = "text"
        tags = ("legacy",)

        def run(self, name: Annotated[Txt, TextWidget(title="Name")]) -> Annotated[Txt, Result(title="R")]:
            return name

    assert repr(Old) == (
        "Old(id='old', title='Old', category='text', tags=('legacy',), versions=(1,), "
        "deprecation=Deprecation(header='Gone'))"
    )


def test_a_class_that_declares_no_node_prints_as_a_class():
    class Shared(NodeDefinition):
        category = "io"

    assert repr(NodeDefinition) == "<class 'conductor.node.NodeDefinition'>"
    assert repr(Shared).startswith("<class '")


def test_a_registry_prints_its_nodes():
    registry = NodeRegistry()
    assert repr(registry) == "NodeRegistry(nodes=())"

    registry.register(Greet)
    registry.register(Truncate)
    assert repr(registry) == (
        "NodeRegistry(nodes=(\n"
        "    Greet(id='greet', title='Greeting', category='text', versions=(1,)),\n"
        "    Truncate(id='truncate', title='Truncate', category='text', versions=(1, 2)),\n"
        "))"
    )


def test_a_record_prints_only_what_is_not_at_its_default():
    text = Input(name="text", dtype=Txt, title="Text", widget=Textarea(title="Text"))

    assert repr(text) == "Input(name='text', dtype=Txt, title='Text', widget=Textarea(title='Text'))"
    assert repr(Policy()) == "Policy()"
    assert repr(Policy(retries=2)) == "Policy(retries=2)"
    assert repr(Deprecation()) == "Deprecation()"
    assert repr(Problem(code="cycle", message="m", fatal=True, node_id="a")) == "Problem(code='cycle', message='m', fatal=True, node_id='a')"
    assert repr(Index("lines", parent=Index("docs"))) == "Index(id='lines', parent=Index(id='docs'))"


def test_a_record_prints_a_field_set_back_to_its_default_value_as_absent():
    """The repr is the data, not the history: a value equal to the default is not news."""
    assert repr(Policy(retries=0, concurrency=8)) == "Policy()"

