"""An input has a ``Param``, as an output has a ``Result``.

What a person reads about an input — its title, its description, whether
an edge can reach it, and the control it is edited with — is written once,
on the ``Param`` inside ``Annotated``, and becomes the ``Input`` the node
carries. The widget is the control and nothing else: it has no title of
its own, and a bare widget in ``Annotated`` is refused when the class is
defined. ``Param(title=...)`` with no widget is an input only an edge can
fill. ``AskPerson()`` with no questions asks for the node's declared outputs,
and a per-row node's answer may be a plain list.
"""

from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    Input,
    NodeRegistry,
    Param,
    Result,
    Series,
    Static,
    node,
)
from conductor._internal.node_signature import read_signature
from conductor.widgets import NumberWidget, Textarea


class Txt(DType, str):
    id = "params-test-text"
    display_title = "Text"


class Num(DType, float):
    id = "params-test-number"
    display_title = "Number"


Out = Annotated[Txt, Result(title="Answer")]


# -- what the Param says, the Input carries once -----------------------------------------


def test_a_param_with_a_widget_describes_with_one_title():
    def run(a: Annotated[Num, Param(title="A", description="The first", widget=NumberWidget(step=2.0))]) -> Out: ...

    (a,) = read_signature(run).interface.inputs
    dumped = a.model_dump(mode="json")

    assert (a.title, a.description, a.connectable) == ("A", "The first", True)
    assert a.widget == NumberWidget(step=2.0)
    assert dumped["title"] == "A"
    assert dumped["widget"] == {"kind": "number", "step": 2.0, "integer_only": False}
    assert "title" not in dumped["widget"]


def test_a_bare_widget_in_annotated_is_refused_naming_param():
    def run(a: Annotated[Num, NumberWidget()]) -> Out: ...

    with pytest.raises(TypeError, match=r"Param\(title=.*widget=") as raised:
        read_signature(run).interface
    assert "'a'" in str(raised.value)


def test_a_param_with_a_title_alone_is_a_connection_only_input():
    param = Param(title="B")
    assert param.widget is None and param.connectable is True

    def run(b: Annotated[Series[Txt], Param(title="B")]) -> Out: ...

    (b,) = read_signature(run).interface.inputs
    assert b.widget is None and b.connectable is True and b.title == "B"


def test_a_result_on_a_parameter_and_a_param_on_the_return_are_refused():
    def swapped(a: Annotated[Num, Result(title="A")]) -> Out: ...

    with pytest.raises(TypeError, match="carries a Result"):
        read_signature(swapped).interface

    def returned(a: Annotated[Num, Param(title="A")]) -> Annotated[Txt, Param(title="X"), Result(title="Y")]: ...

    with pytest.raises(TypeError, match="carries a Param"):
        read_signature(returned).interface


def test_a_parameter_without_a_param_is_titled_by_its_name():
    def run(texts: Series[Txt]) -> Out: ...

    (texts,) = read_signature(run).interface.inputs
    assert (texts.title, texts.widget, texts.connectable) == ("Texts", None, True)


def test_connectable_false_closes_the_input_and_admits_a_static_type():
    def run(tags: Annotated[list[str], Param(title="Tags", connectable=False, widget=Textarea())] = []) -> Out: ...

    (tags,) = read_signature(run).interface.inputs
    assert tags.connectable is False and tags.dtype.wrapped_type == list[str]


def test_an_inputs_widget_reads_back_its_dump():
    """The widget carries nothing that is left out of its dump, so what an
    editor was sent reads back whole. The type does not: the class lives on
    the signature, and its description is not a type."""
    text = Input("text", Txt, title="Text", widget=Textarea(rows=2))
    dumped = text.model_dump(mode="json")

    assert (dumped["name"], dumped["title"], dumped["connectable"]) == ("text", "Text", True)
    assert Textarea.model_validate(dumped["widget"]) == Textarea(rows=2)
    assert dumped["dtype"] == Txt.describe()


# -- AskPerson() asks the declared outputs ---------------------------------------------------


@node("ask-each", title="Ask each", description="d", category="test")
def ask_each(proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | AskPerson:
    """AskPerson a person about every row, with no questions of its own."""
    return AskPerson()


@node("skip-short", title="Skip short", description="d", category="test")
def skip_short(text: Annotated[Txt, Param(title="Text")]) -> Out:
    """Passes a text on, skipping one shorter than two characters."""
    return SKIPPED if len(text) < 2 else text


@node("docs", title="Docs", description="d", category="test")
def docs(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Docs")]:
    return [Txt(part) for part in text.split(",")]


def _registry() -> NodeRegistry:
    reg = NodeRegistry([docs, skip_short, ask_each])
    return reg


def _asking_once() -> CompiledGraph:
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="ask", type="ask-each", version=1, bindings={"proposal": Static("hi")})]),
        _registry(),
    )
    assert compiled.is_runnable, compiled.problems
    return compiled


def _asking_per_row(text: str) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(text)}),
            GraphNode(id="ask", type="ask-each", version=1, bindings={"proposal": From("docs.result")}),
        ]),
        _registry(),
    )
    assert compiled.is_runnable, compiled.problems
    return compiled


