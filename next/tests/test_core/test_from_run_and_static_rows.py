"""What the run provides, and a static that iterates — two things a host's catalog needs of the engine."""

from typing import Annotated

from conductor import (
    DType,
    FromRun,
    NodeRegistry,
    Param,
    Result,
    Series,
    node,
)
from conductor._internal.node_signature import read_signature
from conductor.widgets import ListWidget, TextWidget


class Txt(DType, str):
    id = "txt"
    display_title = "Txt"


class Who:
    def __init__(self, name):
        self.name = name


@node("greet", title="Greet", description="d", category="test")
def greet(text: Annotated[Txt, Param(title="T", widget=TextWidget())], who: FromRun[Who]) -> Annotated[Txt, Result(title="R")]:
    return Txt(f"{text} {who.name}")


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="T", widget=ListWidget())] = Txt("")) -> Annotated[Txt, Result(title="R")]:
    return Txt(text.upper())


@node("joinall", title="Join", description="d", category="test")
def join_all(texts: Annotated[Series[Txt], Param(title="T")] = ()) -> Annotated[Txt, Result(title="R")]:
    return Txt("-".join(texts))


def _registry():
    r = NodeRegistry()
    for c in (greet, upper_node, join_all):
        r = r.extended_with([c])
    return r


def test_a_from_run_parameter_is_a_need_not_an_input():
    iface = read_signature(greet.versions[1].body).interface
    assert [i.name for i in iface.inputs] == ["text"]
    assert iface.needs == {Who}


