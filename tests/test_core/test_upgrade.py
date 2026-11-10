"""Upgrading a node in a graph from the version it was saved at to a newer one.

A node's ``Upgrade`` steps are checked when the node is declared, and
``registry.upgrade`` applies them to one node of a graph record: its values and
defaults, its version, the edges into the inputs a step renames and every
edge that read an output it renamed. Loading never upgrades on its own; a graph pinned at an old
version runs that version.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    NodeRegistry,
    Param,
    Upgrade,
    node,
)
from conductor.widgets import Textarea

from tests.test_core.nodes import registry_of


class Txt(str):
    id = "upgrade-test-txt"
    display_title = "Text"


def _param(title: str):
    return Param(title=title, widget=Textarea())


@dataclass(frozen=True)
class Named:
    text: Annotated[Txt, Param(title="Text")]


def rename_upgrade_1(values):
    first, _, last = values.pop("name", "").partition(" ")
    return {**values, "first": first, "last": last}

def rename_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
    return name

def rename_v2(
    first: Annotated[Txt, _param("First")] = Txt(""),
    last: Annotated[Txt, _param("Last")] = Txt(""),
) -> Annotated[Txt, Param(title="R")]:
    return Txt(f"{first} {last}")

@node("rename", title="Rename", description="d", category="test", replaces=node("rename", replaces=node("rename")(rename_v1), upgrade=Upgrade(rewrite=rename_upgrade_1))(rename_v2), upgrade=Upgrade(renamed_outputs={"result": "text"}))
def rename(
    first: Annotated[Txt, _param("First")] = Txt(""),
    last: Annotated[Txt, _param("Last")] = Txt(""),
) -> Named:
    """Version 1 took ``name``; 2 splits it into ``first`` and ``last``; 3
    renames the output ``result`` to ``text``."""
    return Named(text=Txt(f"{first} {last}"))


@node("echo", title="Echo", description="d", category="test")
def echo(text: Annotated[Txt, _param("Text")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
    return text


def _registry() -> NodeRegistry:
    registry = registry_of(rename, echo)
    return registry


def _graph(version: int = 1) -> GraphRecord:
    return GraphRecord(nodes=[
        NodeRecord(id="r", node="rename", version=version, values={"name": "Ada Lovelace"} if version == 1 else {}),
        NodeRecord(id="e", node="echo"),
    ], edges=[("r.result" if version < 3 else "r.text", "e.text")])


# --- a field name gone from the next version needs an upgrade -------------------


def _older_x(x: Annotated[Txt, _param("X")]) -> Annotated[Txt, Param(title="R")]:
    return x


def _newer_y(y: Annotated[Txt, _param("Y")]) -> Annotated[Txt, Param(title="R")]:
    return y


def test_an_input_gone_from_the_next_version_needs_an_upgrade():
    with pytest.raises(NodeDeclarationError) as raised:
        node("two", replaces=node("two")(_older_x))(_newer_y)
    assert raised.value.code == "upgrade_missing" and "two" in str(raised.value)


def test_an_upgrade_must_say_where_a_gone_input_went():
    with pytest.raises(NodeDeclarationError, match="upgrade|Upgrade"):
        node("two", replaces=node("two")(_older_x), upgrade=Upgrade(renamed_outputs={}))(_newer_y)


def test_a_rename_or_a_rewrite_says_where_it_went():
    assert node("two", replaces=node("two")(_older_x), upgrade=Upgrade(renamed_inputs={"x": "y"}))(_newer_y).version == 2
    assert node("two", replaces=node("two")(_older_x), upgrade=Upgrade(rewrite=lambda v: {"y": v["x"]}))(_newer_y).version == 2


def test_a_version_keeping_its_fields_needs_no_upgrade():
    assert node("two", replaces=node("two")(_older_x))(_older_x).version == 2


# --- registry.upgraded -----------------------------------------------------------


def test_a_two_step_chain_rewrites_the_typed_in_values_and_the_version():
    upgraded = _registry().upgrade(_graph(), "r")

    graph_node = next(n for n in upgraded.nodes if n.id == "r")
    assert graph_node.version == 3
    assert graph_node.values == {"first": "Ada", "last": "Lovelace"}


def test_a_renamed_output_rewrites_the_readers_edges_and_the_graph_compiles_clean():
    registry = _registry()
    upgraded = registry.upgrade(_graph(), "r")

    assert upgraded.edges == (("r.text", "e.text"),)
    graph = registry.load(upgraded)
    assert graph.is_runnable and graph.problems == [], graph.problems


def test_to_stops_the_chain_early():
    upgraded = _registry().upgrade(_graph(), "r", to=2)

    assert next(n for n in upgraded.nodes if n.id == "r").version == 2
    assert upgraded.edges == (("r.result", "e.text"),)


def test_a_node_already_there_returns_the_graph_unchanged():
    graph = _graph(version=3)
    assert _registry().upgrade(graph, "r") is graph


def test_an_edge_into_an_input_the_step_renames_moves_with_it():
    graph = GraphRecord(nodes=[
        NodeRecord(id="src", node="echo"),
        NodeRecord(id="r", node="rename"),
    ], edges=[("src.result", "r.name")])

    def carry_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
        return name

    @node("carry", title="Carry", description="d", category="test", replaces=node("carry")(carry_v1), upgrade=Upgrade(renamed_inputs={"name": "full"}))
    def carry(full: Annotated[Txt, _param("Full")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
        return full

    registry = registry_of(echo, carry)
    graph = graph.model_copy(update={"nodes": (graph.nodes[0], NodeRecord(id="r", node="carry"))})

    upgraded = registry.upgrade(graph, "r")
    assert upgraded.edges == (("src.result", "r.full"),)


def test_an_edge_into_an_input_a_rewrite_leaves_gone_goes_with_it_and_the_graph_loads_clean():
    def summed_v1(a: int, b: int) -> int:
        return a + b

    @node("summed", replaces=node("summed")(summed_v1), upgrade=Upgrade(rewrite=lambda typed: {k: v for k, v in typed.items() if k != "b"}))
    def summed(a: int) -> int:
        return a

    @node("three")
    def three() -> int:
        return 3

    registry = registry_of(three, summed)
    graph = GraphRecord(nodes=[NodeRecord(id="t", node="three"), NodeRecord(id="s", node="summed", values={"a": 1})], edges=[("t.result", "s.b")])

    upgraded = registry.upgrade(graph, "s")
    assert (upgraded.edges, upgraded.nodes[1].values) == ((), {"a": 1})
    assert registry.load(upgraded).problems == []


def test_a_rewrite_sees_values_and_defaults_as_one_and_a_default_stays_one():
    def counted_v1(src: Annotated[Txt, _param("Source")] = Txt(""), n: int = 0) -> Annotated[Txt, Param(title="R")]:
        return src

    @node("counted", replaces=node("counted")(counted_v1), upgrade=Upgrade(rewrite=lambda typed: {**typed, "n": typed.get("n", 0) + 1, "added": 1}))
    def counted(src: Annotated[Txt, _param("Source")] = Txt(""), n: int = 0, added: int = 0) -> Annotated[Txt, Param(title="R")]:
        return src

    registry = registry_of(counted)
    with_both = registry.upgrade(GraphRecord(nodes=[NodeRecord(id="c", node="counted", values={"n": 2}, defaults={"src": "x"})]), "c").nodes[0]
    with_neither = registry.upgrade(GraphRecord(nodes=[NodeRecord(id="c", node="counted")]), "c").nodes[0]

    assert (with_both.values, with_both.defaults) == ({"n": 3, "added": 1}, {"src": "x"})
    assert (with_neither.values, with_neither.defaults) == ({"n": 1, "added": 1}, {})


def test_an_edge_reading_a_column_of_a_renamed_output_moves_with_it():
    registry = _registry()
    graph = GraphRecord(nodes=[NodeRecord(id="r", node="rename", version=2), NodeRecord(id="e", node="echo")], edges=[("r.result.text", "e.text")])

    assert registry.upgrade(graph, "r").edges == (("r.text.text", "e.text"),)


def test_a_downgrade_or_an_unknown_node_raises():
    registry = _registry()
    with pytest.raises(InvalidCallError, match="cannot move from version 3 to 1"):
        registry.upgrade(_graph(version=3), "r", to=1)
    with pytest.raises(InvalidCallError, match="nope"):
        registry.upgrade(_graph(), "nope")


def test_compile_leaves_a_pinned_version_alone():
    registry = _registry()
    graph = registry.load(_graph())

    assert graph.describe().nodes["r"].version == 1
    assert graph.is_runnable, graph.problems
    assert graph.run().outputs == {"e.result": "Ada Lovelace"}


# --- what moves with a renamed field ---------------------------------------------


def renames_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
    return name

@node("renames", title="Renames", description="d", category="test", replaces=node("renames")(renames_v1), upgrade=Upgrade(renamed_inputs={"name": "full"}, renamed_outputs={"result": "text"}))
def renames(full: Annotated[Txt, _param("Full")] = Txt("")) -> Named:
    """Version 2 renames the input ``name`` to ``full`` and the output ``result`` to ``text``."""
    return Named(text=full)


def test_a_renamed_input_takes_its_value_with_it():
    registry = registry_of(echo, renames)
    graph = GraphRecord(nodes=[
        NodeRecord(id="r", node="renames", values={"name": "Ada"}),
        NodeRecord(id="e", node="echo"),
    ], edges=[("r.result", "e.text")])

    upgraded = registry.upgrade(graph, "r")

    graph_node = upgraded.nodes[0]
    assert graph_node.values == {"full": "Ada"}
    assert upgraded.edges == (("r.text", "e.text"),)
    assert registry.load(upgraded).problems == []


def test_a_later_step_reusing_a_renamed_away_name_does_not_catch_an_older_edge():
    """Step 1 renames ``result`` to ``first``; step 2 brings back an output
    named ``result`` and renames it to ``second``. An edge saved at version 1
    read the output that is ``first`` now."""

    def reuse_outputs():
        return {}

    def reuse_v1() -> Annotated[Txt, Param(title="R")]:
        return Txt("")

    def reuse_v2() -> Mapping[str, Txt]:
        return {}

    @node("reuse", title="Reuse", description="d", category="test", compute_outputs=reuse_outputs, replaces=node("reuse", compute_outputs=reuse_outputs, replaces=node("reuse")(reuse_v1), upgrade=Upgrade(renamed_outputs={"result": "first"}))(reuse_v2), upgrade=Upgrade(renamed_outputs={"result": "second"}))
    def reuse() -> Mapping[str, Txt]:
        return {}

    registry = registry_of(echo, reuse)
    graph = GraphRecord(nodes=[
        NodeRecord(id="u", node="reuse"),
        NodeRecord(id="e", node="echo"),
    ], edges=[("u.result", "e.text")])

    upgraded = registry.upgrade(graph, "u")

    assert upgraded.edges == (("u.first", "e.text"),)


def test_a_definition_with_no_steps_only_has_its_version_set():
    """An embedded graph arrives by value, one version per saved flow, with
    no ``@upgrade`` to run: the node moves to the new version with its
    values as saved, and compile says what the new version makes of them."""

    def embedded(inner_id: str) -> GraphRecord:
        return GraphRecord(nodes=[NodeRecord(id=inner_id, node="echo")])

    first = node("embedded", title="Embedded", description="d")(embedded("inner"))
    registry = registry_of(echo, node("embedded", title="Embedded", description="d", replaces=first)(embedded("renamed")))
    graph = GraphRecord(nodes=[
        NodeRecord(id="f", node="embedded", values={"text": "x"}),
    ])

    upgraded = registry.upgrade(graph, "f")

    assert upgraded.nodes[0].version == 2
    assert upgraded.nodes[0].values == {"text": "x"}
    with pytest.raises(InvalidCallError, match="it has 1, 2"):
        registry.upgrade(graph, "f", to=5)


@dataclass(frozen=True)
class _X:
    x: str


@dataclass(frozen=True)
class _Y:
    y: str


def _older_out() -> _X:
    return _X("a")


def _newer_out() -> _Y:
    return _Y("b")


@pytest.mark.parametrize("upgrade", [Upgrade(renamed_inputs={}), Upgrade(rewrite=dict)], ids=["no-rename", "a-rewrite"])
def test_an_upgrade_that_leaves_an_output_gone_and_not_renamed_is_refused_where_it_is_declared(upgrade: Upgrade) -> None:
    """A rewrite reads what is typed into the node, never the edges out of it, so it can't say where an output went."""
    with pytest.raises(NodeDeclarationError, match="'x'") as raised:
        node("o", replaces=node("o")(_older_out), upgrade=upgrade)(_newer_out)
    assert raised.value.code == "upgrade_missing"


def test_an_output_an_upgrade_renamed_is_accounted_for():
    assert node("o", replaces=node("o")(_older_out), upgrade=Upgrade(renamed_outputs={"x": "y"}))(_newer_out).version == 2


def test_a_rewrite_is_written_as_json_and_one_with_no_json_form_is_invalid_value_at_the_node():
    import datetime

    import polars as pl

    @node("when")
    def when_v1(day: str) -> str:
        return day

    saved = GraphRecord.model_validate({"nodes": [{"id": "w", "node": "when", "values": {"day": "2026-01-02"}}]})
    for returned, written in ((datetime.date(2026, 1, 2), "2026-01-02"), (pl.DataFrame({"a": [1]}), None)):

        @node("when", replaces=when_v1, upgrade=Upgrade(rewrite=lambda typed, r=returned: {"day": r}))
        def when(day: datetime.date) -> str:
            return day.isoformat()

        registry = NodeRegistry()
        registry.add(when)
        if written is None:
            with pytest.raises(InvalidCallError) as refused:
                registry.upgrade(saved, "w")
            assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_value", ("w", "day"))]
        else:
            upgraded = registry.upgrade(saved, "w")
            assert upgraded.nodes[0].values == {"day": written}
            assert GraphRecord.model_validate_json(upgraded.model_dump_json()) == upgraded
