"""The engine runs rows."""

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, Any

from conductor import (
    SKIPPED,
    AskPerson,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    NodeError,
    NodeRegistry,
    Output,
    Param,
    Policy,
    Ref,
    Result,
    Series,
    Static,
    node,
)
from conductor.widgets import Textarea

from test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "rows-test-txt"
    display_title = "Tekst"


Out = Annotated[Txt, Result(title="Result")]


@dataclass(frozen=True)
class Documents:
    texts: Annotated[Series[Txt], Result(title="Texts")]
    names: Annotated[Series[Txt], Result(title="Navne")]


@dataclass(frozen=True)
class Length:
    long: Annotated[Txt, Result(title="If long")]
    short: Annotated[Txt, Result(title="If short")]


@node("docs", title="Docs", description="d", category="test")
def docs(text: Annotated[Txt, Param(title="Texts", widget=Textarea())] = Txt("")) -> Documents:
    parts = [Txt(p) for p in text.split(",")] if text else []
    return Documents(texts=parts, names=[Txt(f"doc{i}") for i in range(len(parts))])


@node("upper", title="Upper", description="d", category="test")
def upper_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return Txt(text.upper())


@node("pair", title="Pair", description="d", category="test")
def pair(a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""), b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt("")) -> Out:
    return Txt(f"{a}:{b}")


@node("lines", title="Lines", description="d", category="test")
def lines_node(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Lines")]:
    return [Txt(p) for p in text.split("/")]


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("+".join(texts))


@node("long-only", title="Long only", description="d", category="test")
def long_only(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Length:
    """A decision: keeps a text on `If long` when it has more than two characters."""
    return Length(long=text, short=SKIPPED) if len(text) > 2 else Length(long=SKIPPED, short=text)


@node("fails-on", title="Fails on", description="d", category="test")
def fails_on(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    if text == "boom":
        raise NodeError("kunne ikke", code="boom")
    return text


slow_state = SimpleNamespace(lock=threading.Lock(), active=0, peak=0, seen=[], calls=0)


@node("slow", title="Slow", description="d", category="test", policy=Policy(concurrency=1))
def slow(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    with slow_state.lock:
        slow_state.active += 1
        slow_state.peak = max(slow_state.peak, slow_state.active)
    time.sleep(0.02)
    with slow_state.lock:
        slow_state.active -= 1
        slow_state.seen.append(text)
    return text


wide_state = SimpleNamespace(lock=threading.Lock(), active=0, peak=0, seen=[], calls=0)


@node("wide", title="Wide", description="d", category="test")
def wide(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    with wide_state.lock:
        wide_state.active += 1
        wide_state.peak = max(wide_state.peak, wide_state.active)
    time.sleep(0.02)
    with wide_state.lock:
        wide_state.active -= 1
    return text


def columns_outputs(spec=""):
    spec = spec
    return tuple(Output(name, Txt, title=name.title()) for name in spec.split(",") if name)

@node("columns", title="Columns", description="d", category="test", compute_outputs=columns_outputs)
def columns(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""), spec: Annotated[Txt, Param(title="Kolonner", widget=Textarea())] = Txt("")) -> Mapping[str, Any]:
    """The unfold shape: the author names the columns; each is an output
    on the interface and nothing else. run hands them back by name."""
    return {name: Txt(f"{name}:{text}") for name in spec.split(",") if name}


def _registry():
    registry = NodeRegistry()
    for node_cls in (docs, upper_node, pair, lines_node, join_node, long_only, fails_on, slow, wide, columns):
        registry = registry.extended_with([node_cls])
    return registry


def _edge(*refs):
    return From(*(Ref(n, f) for n, f in refs))


def _docs(texts):
    return GraphNode(id="docs", type="docs", version=1, bindings={"text": Static(texts)})


def _run(nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), _registry())
    assert compiled.is_runnable, compiled.problems
    return compiled.results(compiled.run_sync())


# --- what no run reaches by another way ---------------------------------------------------


# --- a computed interface, by name -----------------------------------------------------


def test_a_computed_inputs_with_no_outputs_is_compiles_problem():
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="c", type="columns", version=1, bindings={"spec": Static("")})]), _registry())

    assert [(p.type, p.loc[0], p.fatal) for p in compiled.problems] == [("no_outputs", "c", False)]
    assert compiled.is_runnable


# --- iteration end to end -----------------------------------------------------------


# --- sparse series and skip ---------------------------------------------


# --- failure -------------------------------------------------------------------


# --- scheduling -----------------------------------------------------------------


# --- every ending is one shape ----------------------------------------------------


# --- a leg ends pending ---------------------------------------------------


@node("asks", title="Ask", description="d", category="test")
def ask_node(proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | AskPerson:
    """An asking node: packages its proposal into the question and
    returns AskPerson. Its output is what the person supplies."""
    return AskPerson(defaults={"result": proposal}, widgets={"result": Textarea()})


def test_a_run_that_may_ask_says_so_and_still_declares_its_outputs():
    """`-> Out | AskPerson` is the truthful annotation: the pause is read past, the
    record declares the one output the person's answer lands on."""
    interface = ask_node.versions[1].interface
    assert [(out.name, out.dtype) for out in interface.outputs] == [("result", Txt)]


def _registry_with_asks(*extra):
    registry = _registry()
    for node_cls in (ask_node, *extra):
        registry = registry.extended_with([node_cls])
    return registry


@node("asks-if-long", title="Ask if long", description="d", category="test")
def asks_if_long(proposal: Annotated[Txt, Param(title="Proposal", widget=Textarea())] = Txt("")) -> Out | AskPerson:
    """AskPerson about a text longer than two characters, and passes a short one through."""
    if len(proposal) <= 2:
        return proposal
    return AskPerson(defaults={"result": proposal}, widgets={"result": Textarea()})


def _asks_if_long(texts):
    return CompiledGraph.from_graph(
        Graph(nodes=[
            _docs(texts),
            GraphNode(id="ask", type="asks-if-long", version=1, bindings={"proposal": _edge(("docs", "texts"))}),
            GraphNode(id="up", type="upper", version=1, bindings={"text": _edge(("ask", "result"))}),
        ]),
        _registry_with_asks(asks_if_long),
    )


# --- an embedded graph runs as nodes of the one run --------------------------------


#: An embedded graph whose inner node pairs what enters with a list the author typed.
TypedInside = embedded_graph_node("typed-inside", (
    GraphNode(id="h", type="upper", version=1),
    GraphNode(id="t", type="pair", version=1, bindings={"a": _edge(("h", "result")), "b": Static(["p", "q"])}),
    GraphNode(id="j", type="join", version=1, bindings={"texts": _edge(("t", "result"))}),
), _registry(), "Indlejret")


