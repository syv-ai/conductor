"""A blank reaching a node: the hint says whether it is called."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Optional

import polars as pl
import pytest
from conductor import (
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    Input,
    NodeDeclarationError,
    NodeRegistry,
    Output,
    Param,
    Result,
    Series,
    Static,
    Table,
    node,
    table_with_columns,
)
from conductor._internal.node_signature import read_signature


class Txt(DType, str):
    id = "blanks-txt"


class Int(DType, int):
    id = "blanks-int"


CALLED: list[Any] = []


@node("rows")
def rows() -> Table:
    return (pl.DataFrame({"name": ["Ida", "Bo", "Nina"], "email": ["ida@x", None, "nina@x"]}))


@node("rows2", compute_outputs=lambda: [Output("result", table_with_columns({"name": Txt, "email": Txt}))])
def rows2() -> Mapping[str, Any]:
    return {"result": rows.versions[1].body()}


@node("plain")
def plain(email: Txt) -> Txt:
    CALLED.append(email)
    return Txt(email.upper())


@node("takes")
def takes(email: Txt | None) -> Txt:
    return Txt("none" if email is None else email)


@dataclass(frozen=True)
class Branched:
    known: Annotated[Txt, Result(exclusive_group="b")]
    unknown: Annotated[Txt, Result(exclusive_group="b")]


@node("branched")
def branched(email: Txt) -> Branched:
    return Branched(email, Txt(""))


@node("after-branch")
def after_branch(value: Txt) -> Txt:
    return value


@node("splits")
def splits(email: Txt) -> Series[Txt]:
    return [Txt(c) for c in email]


@node("count")
def count(values: Series[Any]) -> Int:
    return Int(len(values))


@node("join")
def join(values: Series[Txt]) -> Txt:
    return Txt(",".join(values))


@node("keep")
def keep(values: Series[Txt | None]) -> Int:
    return Int(sum(1 for value in values if value is None))


@node("zipped")
def zipped(a: Series[Txt], b: Series[Txt]) -> Txt:
    return Txt("|".join(f"{x}{y}" for x, y in zip(a, b, strict=True)))


@node("make-table")
def make_table(email: Txt) -> Table:
    return (pl.DataFrame({"x": [email]}))


def _registry(*more):
    return NodeRegistry([rows, rows2, plain, takes, branched, after_branch, splits, count, join, keep, zipped, make_table, *more])


def _results(*nodes, registry=None):
    compiled = CompiledGraph.from_graph(Graph(nodes=list(nodes)), registry or _registry())
    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert state.status == "complete", state.failure
    return compiled, state, compiled.results(state)


CSV = GraphNode(id="csv", type="rows2", version=1)


def _per_row(type_id, **bindings):
    return GraphNode(id="n", type=type_id, version=1, bindings=bindings or {"email": From("csv.result.email")})


# --- what can be written --------------------------------------------------------


@pytest.mark.parametrize("annotation, takes_blank", [
    (Txt, False),
    (Txt | None, True),
    (Optional[Txt], True),
    (Annotated[Txt | None, Param(title="X")], True),
    (Annotated[Txt, Param(title="X")] | None, True),
    (Any, True),
    (Series[Txt], False),
    (Series[Txt | None], True),
    (Series[Any], True),
])
def test_an_input_takes_a_blank_when_its_hint_says_so(annotation, takes_blank):
    def run(x): ...

    run.__annotations__ = {"x": annotation, "return": Txt}
    (input_,) = read_signature(run).interface.inputs
    assert input_.takes_blank is takes_blank


def test_the_bare_type_is_what_travels_on_an_edge():
    def run(x: Series[Txt | None], y: Txt | None) -> Txt | None: ...

    x, y = read_signature(run).interface.inputs
    assert x.dtype is Txt and x.many and y.dtype is Txt


def test_an_output_takes_a_blank_when_its_hint_says_so():
    def run(x: Txt) -> Annotated[Txt | None, Result(title="R")]: ...

    (out,) = read_signature(run).interface.outputs
    assert out.dtype is Txt and out.takes_blank


def test_a_record_field_takes_a_blank_when_its_hint_says_so():
    @dataclass(frozen=True)
    class Parts:
        a: Txt | None
        b: Txt

    def run(x: Txt) -> Parts: ...

    a, b = read_signature(run).interface.outputs
    assert a.takes_blank and not b.takes_blank


def test_a_field_built_by_a_hook_says_it_the_same_way():
    assert Input("x", Txt | None).takes_blank and Output("x", Txt | None).takes_blank
    assert not Input("x", Txt).takes_blank


def test_a_none_default_on_an_input_that_takes_no_blank_is_refused():
    def run(x: Txt = None) -> Txt: ...

    with pytest.raises(NodeDeclarationError) as raised:
        node("n")(run)
    assert raised.value.code == "none_default_needs_blank"


def test_a_none_default_with_a_blank_hint_is_fine():
    def run(x: Txt | None = None) -> Txt: ...

    (x,) = node("n")(run).versions[1].interface.inputs
    assert x.takes_blank and not x.required and x.default is None


# --- a node is not called on a blank unless it says so ---------------------------


def test_a_static_none_needs_the_hint():
    plain_static = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="plain", version=1, bindings={"email": Static(None)})]), _registry())
    takes_static = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="takes", version=1, bindings={"email": Static(None)})]), _registry())

    assert [p.type for p in plain_static.problems] == ["invalid_static"]
    assert takes_static.is_runnable


