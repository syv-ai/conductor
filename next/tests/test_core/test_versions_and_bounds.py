"""A version chain written with ``replaces``, a version that is a graph, an input's bounds and a hook's error."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

import pytest
from conductor import (
    ComputeHookError,
    Graph,
    GraphNode,
    Input,
    NodeDeclarationError,
    NodeRegistry,
    Param,
    Policy,
    Series,
    Static,
    Upgrade,
    node,
    upgrade_node,
)
from conductor.widgets import Range, Widget


@node("split")
def split_v1(text: str, sep: str = ",") -> str:
    return text.split(sep)[0]


@node("split", policy=Policy(retries=2), replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
def split_v2(text: str, separator: str = ",") -> str:
    return text.split(separator)[-1]


def test_a_version_is_numbered_by_its_place_in_the_chain_and_keeps_its_own_settings():
    assert sorted(split_v2.versions) == [1, 2] and split_v2.current_version == 2
    assert split_v2.versions[1].policy == Policy() and split_v2.versions[2].policy == Policy(retries=2)
    assert split_v2("a,b") == "b" and split_v1("a,b") == "a"


def test_only_the_newest_goes_in_a_registry():
    with pytest.raises(NodeDeclarationError, match="declared twice"):
        NodeRegistry([split_v1, split_v2])


def test_the_upgrade_given_beside_replaces_moves_a_node_saved_at_the_older_version():
    graph = Graph(nodes=(GraphNode(id="s", type="split", version=1, bindings={"text": Static("a"), "sep": Static(";")}),))

    moved = upgrade_node(graph, "s", NodeRegistry([split_v2]))

    assert moved.nodes[0].version == 2 and set(moved.nodes[0].bindings) == {"text", "separator"}


@pytest.mark.parametrize(
    ("written", "code"),
    [
        (lambda: node("other", replaces=split_v1), "replaces_another_node"),
        (lambda: node("split", upgrade=Upgrade()), "upgrade_without_replaces"),
        (lambda: node("split", replaces=split_v1), "upgrade_missing"),
    ],
)
def test_a_chain_written_wrong_is_refused_where_it_is_written(written, code):
    def newer(text: str) -> str:
        return text

    with pytest.raises(NodeDeclarationError) as raised:
        written()(newer)
    assert raised.value.code == code and raised.value.node_id is not None


@node("temper")
def temper(temperature: Annotated[float, Param(ge=0, le=2, widget=Range())] = 1.0, name: Annotated[str | None, Param(max_length=3)] = None) -> float:
    return temperature


def _temper_graph(**statics) -> Graph:
    return Graph(nodes=(GraphNode(id="t", type="temper", version=1, bindings={name: Static(value) for name, value in statics.items()}),))


def test_bounds_are_on_the_input_an_editor_reads():
    assert temper.versions[1].interface.inputs[0].model_dump(mode="json")["bounds"] == {"ge": 0.0, "le": 2.0}


@pytest.mark.parametrize("written", [lambda: Param(gte=1), lambda: Input("a", str, max_length="x"), lambda: Input("a", Series[str], ge=0)])
def test_a_bound_written_wrong_is_refused_where_it_is_written(written):
    with pytest.raises(NodeDeclarationError) as raised:
        written()
    assert raised.value.code == "bound_invalid"


class _HostControl(Widget):
    kind: Literal["host-control"] = "host-control"


def _graph_with_hook(error: ComputeHookError) -> tuple[Graph, NodeRegistry]:
    def refuse():
        raise error

    @node("hooked", compute_outputs=refuse)
    def hooked() -> Mapping[str, Any]: ...

    return Graph(nodes=(GraphNode(id="h", type="hooked", version=1),)), NodeRegistry([hooked])


@node("first_two")
def first_two(items: Annotated[Series[str], Param(min_length=1, max_length=2)]) -> str:
    assert type(items) is list
    return ",".join(items)


