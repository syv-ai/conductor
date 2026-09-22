"""Upgrading a node in a graph from the version it was saved at to a newer one.

A node's ``@upgrade`` steps are checked when the class is defined, and
``registry.upgraded`` applies them to one node of a graph: its bindings,
its version, and every edge downstream that read an output the upgrade
renamed. Compile never upgrades on its own; a graph pinned at an old
version runs that version.
"""

from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Param, Result
from conductor.node import NodeDefinition, upgrade, version
from conductor.ref import Ref
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "upgrade-test-txt"
    title = "Text"


def _param(title: str):
    return Param(title=title, widget=Textarea())


@dataclass
class Named:
    text: Annotated[Txt, Result(title="Text")]


class Rename(NodeDefinition):
    """Version 1 took ``name``; 2 splits it into ``first`` and ``last``; 3
    renames the output ``result`` to ``text``."""

    id = "rename"
    title = "Rename"
    description = "d"
    category = "test"

    @version(1)
    def run_v1(self, name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
        return name

    @version(2)
    def run_v2(
        self,
        first: Annotated[Txt, _param("First")] = Txt(""),
        last: Annotated[Txt, _param("Last")] = Txt(""),
    ) -> Annotated[Txt, Result(title="R")]:
        return Txt(f"{first} {last}")

    @version(3)
    def run(
        self,
        first: Annotated[Txt, _param("First")] = Txt(""),
        last: Annotated[Txt, _param("Last")] = Txt(""),
    ) -> Named:
        return Named(text=Txt(f"{first} {last}"))

    @upgrade(1, 2)
    def _split(values):
        first, _, last = values.pop("name", "").partition(" ")
        return {**values, "first": first, "last": last}

    @upgrade(2, 3, outputs={"result": "text"})
    def _rename_output(values):
        return values


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, _param("Text")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
        return text


def _registry() -> NodeRegistry:
    registry = NodeRegistry()
    registry.register(Rename)
    registry.register(Echo)
    return registry


def _graph(version: int = 1) -> Graph:
    return Graph(nodes=[
        GraphNode(id="r", type="rename", version=version, bindings={"name": Static("Ada Lovelace")} if version == 1 else {}),
        GraphNode(id="e", type="echo", version=1, bindings={"text": From(Ref("r", "result"))}),
    ])


# --- the steps, checked when the class is defined ------------------------------


def _two_versions(**upgrades):
    """A class body with versions 1 and 2 and the given upgrade functions."""
    body = {
        "id": "two",
        "title": "Two",
        "description": "d",
        "category": "test",
        "run_v1": version(1)(lambda self, x: x),
        "run": version(2)(lambda self, x: x),
        **upgrades,
    }
    for method in ("run_v1", "run"):
        body[method].__annotations__ = {"x": Annotated[Txt, _param("X")], "return": Annotated[Txt, Result(title="R")]}
    return type("Two", (NodeDefinition,), body)


def test_a_gap_in_the_upgrade_chain_is_refused_naming_it():
    with pytest.raises(TypeError, match=r"no @upgrade\(1, 2\)"):
        _two_versions()


def test_a_backwards_step_is_refused():
    with pytest.raises(TypeError, match=r"@upgrade\(2, 1\)"):
        _two_versions(up=upgrade(1, 2)(lambda v: v), down=upgrade(2, 1)(lambda v: v))


def test_a_step_that_skips_a_version_is_refused():
    with pytest.raises(TypeError, match=r"@upgrade\(1, 3\)"):
        _two_versions(up=upgrade(1, 2)(lambda v: v), skip=upgrade(1, 3)(lambda v: v))


def test_two_steps_for_one_pair_are_refused():
    with pytest.raises(TypeError, match=r"two methods declare @upgrade\(1, 2\)"):
        _two_versions(one=upgrade(1, 2)(lambda v: v), other=upgrade(1, 2)(lambda v: v))


def test_a_one_version_node_needs_no_upgrade():
    assert Echo.upgrades == {}


# --- registry.upgraded -----------------------------------------------------------


def test_a_two_step_chain_rewrites_the_typed_in_values_and_the_version():
    upgraded = _registry().upgraded(_graph(), "r")

    node = next(n for n in upgraded.nodes if n.id == "r")
    assert node.version == 3
    assert node.bindings == {"first": Static("Ada"), "last": Static("Lovelace")}


def test_a_renamed_output_rewrites_the_readers_edges_and_the_graph_compiles_clean():
    registry = _registry()
    upgraded = registry.upgraded(_graph(), "r")

    reader = next(n for n in upgraded.nodes if n.id == "e")
    assert reader.bindings["text"] == From(Ref("r", "text"))
    compiled = CompiledGraph.from_graph(upgraded, registry)
    assert compiled.is_runnable and compiled.problems == (), compiled.problems


def test_to_stops_the_chain_early():
    upgraded = _registry().upgraded(_graph(), "r", to=2)

    assert next(n for n in upgraded.nodes if n.id == "r").version == 2
    assert next(n for n in upgraded.nodes if n.id == "e").bindings["text"] == From(Ref("r", "result"))


def test_a_node_already_there_returns_the_graph_unchanged():
    graph = _graph(version=3)
    assert _registry().upgraded(graph, "r") is graph


def test_an_edge_into_an_input_the_step_renames_moves_with_it():
    graph = Graph(nodes=[
        GraphNode(id="src", type="echo", version=1),
        GraphNode(id="r", type="rename", version=1, bindings={"name": From(Ref("src", "result"))}),
    ])

    class Carry(NodeDefinition):
        id = "carry"
        title = "Carry"
        description = "d"
        category = "test"

        @version(1)
        def run_v1(self, name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
            return name

        @version(2)
        def run(self, full: Annotated[Txt, _param("Full")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
            return full

        @upgrade(1, 2)
        def _rename(values):
            return {"full": values["name"]}

    registry = NodeRegistry()
    registry.register(Echo)
    registry.register(Carry)
    graph = Graph(nodes=[graph.nodes[0], GraphNode(id="r", type="carry", version=1, bindings=graph.nodes[1].bindings)])

    upgraded = registry.upgraded(graph, "r")
    assert next(n for n in upgraded.nodes if n.id == "r").bindings == {"full": From(Ref("src", "result"))}


def test_a_downgrade_or_an_unknown_node_raises():
    registry = _registry()
    with pytest.raises(ValueError, match="no upgrade from version 3 to 1"):
        registry.upgraded(_graph(version=3), "r", to=1)
    with pytest.raises(KeyError, match="nope"):
        registry.upgraded(_graph(), "nope")


def test_compile_leaves_a_pinned_version_alone():
    registry = _registry()
    compiled = CompiledGraph.from_graph(_graph(), registry)

    assert compiled.node("r").version == Rename.versions[1]
    assert compiled.is_runnable, compiled.problems
