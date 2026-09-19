"""An input has a ``Param``, as an output has a ``Result``.

What a person reads about an input — its title, its description, whether
an edge can reach it, and the control it is edited with — is written once,
on the ``Param`` inside ``Annotated``, and becomes the ``Input`` the node
carries. The widget is the control and nothing else: it has no title of
its own, and a bare widget in ``Annotated`` is refused when the class is
defined. ``Param(title=...)`` with no widget is an input only an edge can
fill. ``Asks()`` with no questions asks for the node's declared outputs,
and a per-row node's answer may be a plain list.
"""

import asyncio
from typing import Annotated, Any

import pytest
from conductor import Asks, CompiledGraph, GraphNode, NodeRegistry
from conductor.dtype import DType
from conductor.execution.engine import execute
from conductor.graph.binding import Edges, Static
from conductor.graph.model import Graph
from conductor.interface import Interface
from conductor.metadata import Input, Param, Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import NumberWidget, Textarea


class Txt(DType, str):
    id = "params-test-text"
    title = "Text"


class Num(DType, float):
    id = "params-test-number"
    title = "Number"


Out = Annotated[Txt, Result(title="Answer")]


# -- what the Param says, the Input carries once -----------------------------------------


def test_a_param_with_a_widget_describes_with_one_title():
    def run(self, a: Annotated[Num, Param(title="A", description="The first", widget=NumberWidget(min_val=0))]) -> Out: ...

    (a,) = Interface.of(run).inputs
    dumped = a.model_dump(mode="json")

    assert (a.title, a.description, a.show_handle) == ("A", "The first", True)
    assert a.widget == NumberWidget(min_val=0)
    assert dumped["title"] == "A"
    assert dumped["widget"] == {"kind": "number", "min_val": 0.0, "max_val": None, "step": None, "integer_only": False}
    assert "title" not in dumped["widget"]


def test_a_bare_widget_in_annotated_is_refused_naming_param():
    def run(self, a: Annotated[Num, NumberWidget()]) -> Out: ...

    with pytest.raises(TypeError, match=r"Param\(title=.*widget=") as refused:
        Interface.of(run)
    assert "'a'" in str(refused.value)


def test_a_param_with_a_title_alone_is_a_connection_only_input():
    param = Param(title="B")
    assert param.widget is None and param.show_handle is True

    def run(self, b: Annotated[Series[Txt], Param(title="B")]) -> Out: ...

    (b,) = Interface.of(run).inputs
    assert b.widget is None and b.show_handle is True and b.title == "B"


def test_a_parameter_without_a_param_is_titled_by_its_name():
    def run(self, texts: Series[Txt]) -> Out: ...

    (texts,) = Interface.of(run).inputs
    assert (texts.title, texts.widget, texts.show_handle) == ("texts", None, True)


def test_show_handle_false_closes_the_input_and_admits_a_static_type():
    def run(self, tags: Annotated[list[str], Param(title="Tags", show_handle=False, widget=Textarea())] = []) -> Out: ...

    (tags,) = Interface.of(run).inputs
    assert tags.show_handle is False and tags.dtype == list[str]


def test_an_input_reads_back_its_dump():
    """The widget carries nothing that is left out of its dump, so what an
    editor was sent reads back whole; the type reads back as its
    description, since the class lives on the signature."""
    text = Input(name="text", dtype=Txt, title="Text", widget=Textarea(rows=2))
    dumped = text.model_dump(mode="json")

    back = Input.model_validate(dumped)

    assert (back.name, back.title, back.widget, back.show_handle) == ("text", "Text", Textarea(rows=2), True)
    assert back.dtype == Txt.describe()


# -- Asks() asks the declared outputs ---------------------------------------------------


class AskEach(NodeDefinition):
    """Asks a person about every row, with no questions of its own."""

    id = "ask-each"
    title = "Ask each"
    description = "d"
    category = "test"

    def run(self, proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | Asks:
        return Asks()


class Docs(NodeDefinition):
    id = "docs"
    title = "Docs"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Docs")]:
        return [Txt(part) for part in text.split(",")]


def _registry() -> NodeRegistry:
    reg = NodeRegistry()
    reg.register(Docs)
    reg.register(AskEach)
    return reg


def _leg(compiled: CompiledGraph, **kw: Any) -> list[dict]:
    async def run() -> list[dict]:
        return [event async for event in execute(compiled, **kw)]

    return asyncio.run(run())


def _asking_once() -> CompiledGraph:
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="ask", type="ask-each", version=1, bindings={"proposal": Static(value="hi")})]),
        _registry(),
    )
    assert compiled.is_runnable, compiled.problems
    return compiled


def _asking_per_row(text: str) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(value=text)}),
            GraphNode(id="ask", type="ask-each", version=1, bindings={"proposal": Edges(refs=(Ref("docs", "result"),))}),
        ]),
        _registry(),
    )
    assert compiled.is_runnable, compiled.problems
    return compiled


def test_asks_with_no_questions_asks_the_declared_outputs():
    ending = _leg(_asking_once())[-1]

    assert ending["type"] == "graph_pending"
    (pending,) = ending["pending"]
    (question,) = pending["questions"]
    assert question == Input(name=Ref("ask", "result"), dtype=Txt, title="Answer")
    assert question.widget is None


def test_a_list_answers_a_per_row_node():
    compiled = _asking_per_row("a,b,c")
    first = _leg(compiled)[-1]
    assert [w["row"] for w in first["pending"]] == [(0,), (1,), (2,)]

    second = _leg(compiled, record=first["record"], cache={"ask": {"result": ["x", "y", "z"]}})

    assert second[-1]["type"] == "graph_complete"
    assert list(second[-1]["results"]["ask"]["result"]) == ["x", "y", "z"]
    assert second[-1]["results"]["ask"]["result"] == Series(Index("docs"), [Txt("x"), Txt("y"), Txt("z")], rows=[(0,), (1,), (2,)])


def test_a_list_of_the_wrong_length_is_refused_naming_node_and_output():
    compiled = _asking_per_row("a,b,c")
    first = _leg(compiled)[-1]

    with pytest.raises(ValueError, match=r"'ask'.*'result'.*3 rows"):
        _leg(compiled, record=first["record"], cache={"ask": {"result": ["x", "y"]}})
