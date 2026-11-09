"""Upgrading a node in a graph from the version it was saved at to a newer one.

A node's ``@upgrade`` steps are checked when the class is defined, and
``registry.upgraded`` applies them to one node of a graph: its bindings,
its version, and every edge downstream that read an output the upgrade
renamed. Compile never upgrades on its own; a graph pinned at an old
version runs that version.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    CompiledGraph,
    DType,
    FieldContent,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeDeclarationError,
    NodeDefinition,
    NodeRegistry,
    OlderVersion,
    Param,
    Result,
    Static,
    Upgrade,
    node,
    upgrade_node,
)
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "upgrade-test-txt"
    display_title = "Text"


def _param(title: str):
    return Param(title=title, widget=Textarea())


@dataclass
class Named:
    text: Annotated[Txt, Result(title="Text")]


def rename_upgrade_1(values):
    first, _, last = values.pop("name", "").partition(" ")
    return {**values, "first": first, "last": last}

def rename_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
    return name

def rename_v2(
    first: Annotated[Txt, _param("First")] = Txt(""),
    last: Annotated[Txt, _param("Last")] = Txt(""),
) -> Annotated[Txt, Result(title="R")]:
    return Txt(f"{first} {last}")

@node("rename", title="Rename", description="d", category="test", older_versions=[OlderVersion(rename_v1, upgrade=Upgrade(rewrite=rename_upgrade_1)), OlderVersion(rename_v2, upgrade=Upgrade(renamed_outputs={"result": "text"}))])
def rename(
    first: Annotated[Txt, _param("First")] = Txt(""),
    last: Annotated[Txt, _param("Last")] = Txt(""),
) -> Named:
    """Version 1 took ``name``; 2 splits it into ``first`` and ``last``; 3
    renames the output ``result`` to ``text``."""
    return Named(text=Txt(f"{first} {last}"))


@node("echo", title="Echo", description="d", category="test")
def echo(text: Annotated[Txt, _param("Text")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
    return text


def _registry() -> NodeRegistry:
    registry = NodeRegistry([rename, echo])
    return registry


def _graph(version: int = 1) -> Graph:
    return Graph(nodes=[
        GraphNode(id="r", type="rename", version=version, bindings={"name": Static("Ada Lovelace")} if version == 1 else {}),
        GraphNode(id="e", type="echo", version=1, bindings={"text": From("r.result")}),
    ])


# --- a field name gone from the next version needs an upgrade -------------------


def _older_x(x: Annotated[Txt, _param("X")]) -> Annotated[Txt, Result(title="R")]:
    return x


def _newer_y(y: Annotated[Txt, _param("Y")]) -> Annotated[Txt, Result(title="R")]:
    return y


def test_an_input_gone_from_the_next_version_needs_an_upgrade():
    with pytest.raises(NodeDeclarationError) as raised:
        node("two", older_versions=[OlderVersion(_older_x)])(_newer_y)
    assert raised.value.code == "upgrade_missing" and raised.value.node_id == "two"


def test_an_upgrade_must_say_where_a_gone_input_went():
    with pytest.raises(NodeDeclarationError, match="upgrade|Upgrade"):
        node("two", older_versions=[OlderVersion(_older_x, upgrade=Upgrade(renamed_outputs={}))])(_newer_y)


def test_a_rename_or_a_rewrite_says_where_it_went():
    assert node("two", older_versions=[OlderVersion(_older_x, upgrade=Upgrade(renamed_inputs={"x": "y"}))])(_newer_y).current_version == 2
    assert node("two", older_versions=[OlderVersion(_older_x, upgrade=Upgrade(rewrite=lambda v: {"y": v["x"]}))])(_newer_y).current_version == 2


def test_a_version_keeping_its_fields_needs_no_upgrade():
    assert node("two", older_versions=[OlderVersion(_older_x)])(_older_x).current_version == 2


# --- registry.upgraded -----------------------------------------------------------


def test_a_two_step_chain_rewrites_the_typed_in_values_and_the_version():
    upgraded = upgrade_node(_graph(), "r", _registry())

    graph_node = next(n for n in upgraded.nodes if n.id == "r")
    assert graph_node.version == 3
    assert graph_node.bindings == {"first": Static("Ada"), "last": Static("Lovelace")}


def test_a_renamed_output_rewrites_the_readers_edges_and_the_graph_compiles_clean():
    registry = _registry()
    upgraded = upgrade_node(_graph(), "r", registry)

    reader = next(n for n in upgraded.nodes if n.id == "e")
    assert reader.bindings["text"] == From("r.text")
    compiled = CompiledGraph.from_graph(upgraded, registry)
    assert compiled.is_runnable and compiled.problems == (), compiled.problems


def test_to_stops_the_chain_early():
    upgraded = upgrade_node(_graph(), "r", _registry(), to=2)

    assert next(n for n in upgraded.nodes if n.id == "r").version == 2
    assert next(n for n in upgraded.nodes if n.id == "e").bindings["text"] == From("r.result")


def test_a_node_already_there_returns_the_graph_unchanged():
    graph = _graph(version=3)
    assert upgrade_node(graph, "r", _registry()) is graph


def test_an_edge_into_an_input_the_step_renames_moves_with_it():
    graph = Graph(nodes=[
        GraphNode(id="src", type="echo", version=1),
        GraphNode(id="r", type="rename", version=1, bindings={"name": From("src.result")}),
    ])

    def carry_upgrade_1(values):
        return {"full": values["name"]}

    def carry_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
        return name

    @node("carry", title="Carry", description="d", category="test", older_versions=[OlderVersion(carry_v1, upgrade=Upgrade(rewrite=carry_upgrade_1))])
    def carry(full: Annotated[Txt, _param("Full")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
        return full

    registry = NodeRegistry([echo, carry])
    graph = Graph(nodes=[graph.nodes[0], GraphNode(id="r", type="carry", version=1, bindings=graph.nodes[1].bindings)])

    upgraded = upgrade_node(graph, "r", registry)
    assert next(n for n in upgraded.nodes if n.id == "r").bindings == {"full": From("src.result")}


def test_a_downgrade_or_an_unknown_node_raises():
    registry = _registry()
    with pytest.raises(InvalidCallError, match="cannot move from version 3 to 1"):
        upgrade_node(_graph(version=3), "r", registry, to=1)
    with pytest.raises(InvalidCallError, match="nope"):
        upgrade_node(_graph(), "nope", registry)


def test_compile_leaves_a_pinned_version_alone():
    registry = _registry()
    compiled = CompiledGraph.from_graph(_graph(), registry)

    assert compiled.node("r").version == rename.versions[1]
    assert compiled.is_runnable, compiled.problems


# --- what moves with a renamed field ---------------------------------------------


def renames_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
    return name

@node("renames", title="Renames", description="d", category="test", older_versions=[OlderVersion(renames_v1, upgrade=Upgrade(renamed_inputs={"name": "full"}, renamed_outputs={"result": "text"}))])
def renames(full: Annotated[Txt, _param("Full")] = Txt("")) -> Named:
    """Version 2 renames the input ``name`` to ``full`` and the output ``result`` to ``text``."""
    return Named(text=full)


def test_a_renamed_input_takes_its_binding_lock_and_content_with_it():
    registry = NodeRegistry([echo, renames])
    graph = Graph(nodes=[
        GraphNode(
            id="r", type="renames", version=1, bindings={"name": Static("Ada")}, locked=("name",),
            fields={"name": FieldContent(title="Navn"), "result": FieldContent(title="Resultat")},
        ),
        GraphNode(id="e", type="echo", version=1, bindings={"text": From("r.result")}),
    ])

    upgraded = upgrade_node(graph, "r", registry)

    graph_node = upgraded.nodes[0]
    assert graph_node.bindings == {"full": Static("Ada")}
    assert graph_node.locked == ("full",)
    assert graph_node.fields == {"full": FieldContent(title="Navn"), "text": FieldContent(title="Resultat")}
    assert CompiledGraph.from_graph(upgraded, registry).problems == ()


def test_a_later_step_reusing_a_renamed_away_name_does_not_catch_an_older_edge():
    """Step 1 renames ``result`` to ``first``; step 2 brings back an output
    named ``result`` and renames it to ``second``. An edge saved at version 1
    read the output that is ``first`` now."""

    def reuse_outputs():
        return ()

    def reuse_v1() -> Annotated[Txt, Result(title="R")]:
        return Txt("")

    def reuse_v2() -> Mapping[str, Txt]:
        return {}

    @node("reuse", title="Reuse", description="d", category="test", compute_outputs=reuse_outputs, older_versions=[OlderVersion(reuse_v1, upgrade=Upgrade(renamed_outputs={"result": "first"})), OlderVersion(reuse_v2, compute_outputs=reuse_outputs, upgrade=Upgrade(renamed_outputs={"result": "second"}))])
    def reuse() -> Mapping[str, Txt]:
        return {}

    registry = NodeRegistry([echo, reuse])
    graph = Graph(nodes=[
        GraphNode(id="u", type="reuse", version=1),
        GraphNode(id="e", type="echo", version=1, bindings={"text": From("u.result")}),
    ])

    upgraded = upgrade_node(graph, "u", registry)

    assert upgraded.nodes[1].bindings["text"] == From("u.first")


def test_a_definition_with_no_steps_only_has_its_version_set():
    """An embedded graph arrives by value, one version per saved flow, with
    no ``@upgrade`` to run: the node moves to the new version with its
    bindings as saved, and compile says what the new version makes of them."""

    def embedded(inner_id: str) -> CompiledGraph:
        return CompiledGraph.from_graph(Graph(nodes=[GraphNode(id=inner_id, type="echo", version=1)]), _registry())

    registry = NodeRegistry([NodeDefinition.from_compiled_graphs("embedded", title="Embedded", description="d", versions={1: embedded("inner"), 2: embedded("renamed")})])
    graph = Graph(nodes=[GraphNode(id="f", type="embedded", version=1, bindings={"text": Static("x")})])

    upgraded = upgrade_node(graph, "f", registry)

    assert upgraded.nodes[0].version == 2
    assert upgraded.nodes[0].bindings == {"text": Static("x")}
    with pytest.raises(InvalidCallError, match="it has 1, 2"):
        upgrade_node(graph, "f", registry, to=5)
